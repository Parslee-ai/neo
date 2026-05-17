"""Tests for CarAdapter and the car_inference runtime singleton.

These tests never import car_runtime — they exercise the adapter against a
fake runtime injected via the constructor or via ``car_inference.set_runtime``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from neo import car_inference
from neo.adapters import CarAdapter, create_adapter


class FakeRuntime:
    """Minimal CarRuntime stand-in for unit tests.

    Mirrors the real ``infer_tracked`` response schema validated against
    car-runtime 0.15.1: ``text``, ``model_used``, ``usage`` with
    ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` /
    ``context_window``, plus ``latency_ms``, ``time_to_first_token_ms``,
    ``trace_id``, ``tool_calls``.
    """

    def __init__(self, response: dict | str | None = None, raise_on: str | None = None):
        self.calls: list[tuple[str, dict]] = []
        self._response = response if response is not None else {
            "text": "ok",
            "model_used": "gpt-4.1-mini",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "context_window": 1_000_000,
            },
            "latency_ms": 250,
            "time_to_first_token_ms": 80,
            "trace_id": "t-fake-1",
            "tool_calls": [],
        }
        self._raise_on = raise_on

    def infer_tracked(self, prompt: str, **kwargs):
        self.calls.append((prompt, dict(kwargs)))
        if self._raise_on and self._raise_on in prompt:
            raise RuntimeError(f"FakeRuntime raised on: {self._raise_on}")
        if isinstance(self._response, str):
            return self._response
        return json.dumps(self._response)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test starts with no cached runtime."""
    car_inference.reset_for_testing()
    yield
    car_inference.reset_for_testing()


def test_messages_to_prompt_handles_chat_format():
    out = CarAdapter._messages_to_prompt([
        {"role": "system", "content": "You are Neo."},
        {"role": "user", "content": "ping"},
    ])
    assert "system: You are Neo." in out
    assert "user: ping" in out
    assert out.index("system") < out.index("user")


def test_messages_to_prompt_passes_through_string():
    assert CarAdapter._messages_to_prompt("already a prompt") == "already a prompt"


def test_generate_returns_text_field_from_infer_tracked():
    rt = FakeRuntime({"text": "hello from car", "model_used": "gpt-4.1-mini", "usage": {}})
    adapter = CarAdapter(runtime=rt)
    out = adapter.generate([{"role": "user", "content": "hi"}])
    assert out == "hello from car"
    assert len(rt.calls) == 1
    prompt, kwargs = rt.calls[0]
    # When a chat-style list is passed we send messages_json and use the
    # first user message as the (required-positional) prompt arg.
    assert prompt == "hi"
    assert kwargs["max_tokens"] == 4096
    assert "messages_json" in kwargs
    assert json.loads(kwargs["messages_json"]) == [{"role": "user", "content": "hi"}]
    assert "model" not in kwargs  # model=None means let router decide
    # Temperature is NOT passed to CAR — infer_tracked doesn't accept it.
    assert "temperature" not in kwargs
    # Default IntentHint should request the code task so the router
    # picks a code-capable model instead of the chat default.
    assert "intent_json" in kwargs
    assert json.loads(kwargs["intent_json"]) == {"task": "code"}


def test_default_intent_hint_is_code_task():
    """Neo's workload is code reasoning — the router should know."""
    rt = FakeRuntime()
    adapter = CarAdapter(runtime=rt)
    assert adapter.intent_hint == {"task": "code"}
    adapter.generate("anything", max_tokens=8)
    _, kwargs = rt.calls[0]
    assert json.loads(kwargs["intent_json"]) == {"task": "code"}


def test_explicit_intent_hint_overrides_default():
    """Caller-supplied intent always wins over the code-task default."""
    rt = FakeRuntime()
    adapter = CarAdapter(
        intent_hint={"task": "reasoning", "prefer_local": True},
        runtime=rt,
    )
    assert adapter.intent_hint == {"task": "reasoning", "prefer_local": True}
    adapter.generate("anything", max_tokens=8)
    _, kwargs = rt.calls[0]
    assert json.loads(kwargs["intent_json"]) == {
        "task": "reasoning",
        "prefer_local": True,
    }


def test_generate_string_prompt_skips_messages_json():
    rt = FakeRuntime()
    adapter = CarAdapter(runtime=rt)
    adapter.generate("just a string prompt", max_tokens=128)
    prompt, kwargs = rt.calls[0]
    assert prompt == "just a string prompt"
    assert "messages_json" not in kwargs
    # Default coding intent still applies on bare-string prompts.
    assert json.loads(kwargs["intent_json"]) == {"task": "code"}


def test_generate_pins_model_when_set():
    rt = FakeRuntime()
    adapter = CarAdapter(model="Qwen3-4B", runtime=rt)
    adapter.generate([{"role": "user", "content": "hi"}], max_tokens=512)
    _, kwargs = rt.calls[0]
    assert kwargs["model"] == "Qwen3-4B"
    assert kwargs["max_tokens"] == 512


def test_generate_passes_intent_hint_as_intent_json():
    rt = FakeRuntime()
    intent = {"task": "code", "prefer_local": True}
    adapter = CarAdapter(intent_hint=intent, runtime=rt)
    adapter.generate([{"role": "user", "content": "refactor this"}])
    _, kwargs = rt.calls[0]
    # CAR's API takes intent_json as a JSON string, not a dict
    assert "intent_json" in kwargs
    assert json.loads(kwargs["intent_json"]) == intent
    assert "intent_hint" not in kwargs


def test_metrics_capture_real_car_schema():
    """Metrics emit must use CAR's actual field names (model_used,
    prompt_tokens, completion_tokens, latency_ms, trace_id)."""
    recorded: list[tuple[str, dict]] = []
    rt = FakeRuntime()
    adapter = CarAdapter(runtime=rt)
    with patch("neo.memory.metrics.record", side_effect=lambda event, **f: recorded.append((event, f))):
        adapter.generate([{"role": "user", "content": "hi"}])
    assert len(recorded) == 1
    event, fields = recorded[0]
    assert event == "lm_call"
    assert fields["provider"] == "car"
    assert fields["model"] == "gpt-4.1-mini"      # from model_used
    assert fields["status"] == "success"          # explicit symmetry with error path
    assert fields["input_tokens"] == 100          # from prompt_tokens
    assert fields["output_tokens"] == 20          # from completion_tokens
    assert fields["context_window"] == 1_000_000
    assert fields["latency_ms"] == 250
    assert fields["time_to_first_token_ms"] == 80
    assert fields["trace_id"] == "t-fake-1"


def test_generate_tolerates_non_json_response():
    rt = FakeRuntime(response="raw string body, not JSON")
    adapter = CarAdapter(runtime=rt)
    out = adapter.generate([{"role": "user", "content": "?"}])
    assert out == "raw string body, not JSON"


def test_generate_returns_empty_string_when_text_missing():
    rt = FakeRuntime({"usage": {}})  # no "text" key
    adapter = CarAdapter(runtime=rt)
    assert adapter.generate([{"role": "user", "content": "?"}]) == ""


def test_name_includes_model_or_router_sentinel():
    assert CarAdapter(runtime=FakeRuntime()).name() == "car/router"
    assert CarAdapter(model="gpt-5", runtime=FakeRuntime()).name() == "car/gpt-5"


def test_create_adapter_factory_returns_car_adapter():
    fake = FakeRuntime()
    car_inference.set_runtime(fake)
    adapter = create_adapter("car")
    assert isinstance(adapter, CarAdapter)
    # api_key and base_url should be silently stripped — CarAdapter doesn't accept them
    adapter2 = create_adapter("car", model="qwen3-32b", api_key="ignored", base_url="ignored")
    assert isinstance(adapter2, CarAdapter)
    assert adapter2.model == "qwen3-32b"


def test_get_runtime_raises_clear_error_when_car_runtime_missing():
    with patch.object(car_inference, "is_available", return_value=False):
        # Force import path to fail by clearing the singleton and patching importlib
        car_inference.reset_for_testing()
        # importlib.util.find_spec inside is_available is no longer load-bearing;
        # the actual import in get_runtime is what raises.
        with patch("builtins.__import__", side_effect=ImportError("no car_runtime")):
            with pytest.raises(RuntimeError, match=r"neo-reasoner\[car\]"):
                car_inference.get_runtime()


def test_set_runtime_overrides_singleton():
    fake_a = FakeRuntime({"text": "A", "usage": {}})
    fake_b = FakeRuntime({"text": "B", "usage": {}})
    car_inference.set_runtime(fake_a)
    assert car_inference.get_runtime() is fake_a
    car_inference.set_runtime(fake_b)
    assert car_inference.get_runtime() is fake_b


def test_singleton_shared_across_adapter_instances():
    fake = FakeRuntime()
    car_inference.set_runtime(fake)
    a = CarAdapter()
    b = CarAdapter()
    assert a._rt is b._rt is fake


@pytest.mark.skipif(
    not car_inference.is_available(),
    reason=(
        "car_runtime not installed — car_host.run_server() early-returns at "
        "its top-level `import car_runtime as cr` before reaching the singleton "
        "acquisition path, so the identity assertion can't be exercised."
    ),
)
def test_car_host_uses_runtime_singleton_not_direct_constructor():
    """car_host.run_server must reuse the process-wide CarRuntime singleton
    so an inbound `neo serve` host and an outbound CarAdapter in the same
    process share state, policies, eventlog, and the auth handshake.

    Verified by identity: patch ``car_inference.get_runtime`` to return a
    sentinel, then invoke ``run_server``'s runtime-acquisition path with the
    rest of the function short-circuited via a stub that raises immediately
    after the runtime is bound. Assert the sentinel was returned.

    Behavior, not source shape — `inspect.getsource` would also accept
    `# get_runtime()` in a comment and break on whitespace edits.
    """
    sentinel = FakeRuntime()
    car_inference.set_runtime(sentinel)

    # Patch the schema-register call (the first thing run_server does with
    # the runtime) to record what it received and immediately abort, so
    # we can assert identity without booting the full A2A listener.
    captured: dict = {}

    class _Abort(Exception):
        pass

    def _capture_and_abort(self, *_args, **_kwargs):
        captured["rt"] = self
        raise _Abort

    with patch.object(FakeRuntime, "register_tool_schema", _capture_and_abort, create=True):
        # Mock the rest of the imports inside run_server so it gets to the
        # register_tool_schema call without needing a real daemon.
        with patch("neo.car_host.cr", create=True) as fake_cr:
            fake_cr.CarRuntime = FakeRuntime  # never used now — singleton path
            from neo import car_host
            try:
                car_host.run_server()
            except _Abort:
                pass
            except Exception:
                # Any earlier-stage failure (config load, etc.) is fine —
                # the assertion below catches whether we made it to the
                # runtime-binding step at all.
                pass

    assert captured.get("rt") is sentinel, (
        "car_host.run_server did not use the shared CarRuntime singleton — "
        "outbound CarAdapter and inbound `neo serve` are running on "
        "different runtime instances. Use car_inference.get_runtime()."
    )


def test_metrics_emit_is_best_effort():
    """Metrics failures must not break the adapter."""
    rt = FakeRuntime({"text": "x", "model_used": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    adapter = CarAdapter(runtime=rt)
    with patch("neo.memory.metrics.record", side_effect=RuntimeError("metrics down")):
        out = adapter.generate([{"role": "user", "content": "hi"}])
    assert out == "x"  # adapter still returned successfully


# ----------------------------------------------------------------------------
# Live integration test — runs only when car_runtime is importable AND the
# car-server daemon is reachable. Catches real schema drift between the
# adapter and the binding (signatures, kwarg names, response field names).
# ----------------------------------------------------------------------------

def test_error_path_emits_lm_call_metric_with_status_error():
    """When infer_tracked raises, an lm_call metric with status=error and
    error_type=<class name> must land in metrics.jsonl before the exception
    re-raises. Operators querying CAR failure rates need to see this."""
    class BrokenRuntime:
        def infer_tracked(self, prompt, **kwargs):
            raise RuntimeError("rpc infer: -32603 model not found: bogus")

    recorded: list[tuple[str, dict]] = []
    adapter = CarAdapter(model="bogus-model", runtime=BrokenRuntime())
    with patch("neo.memory.metrics.record", side_effect=lambda event, **f: recorded.append((event, f))):
        with pytest.raises(RuntimeError, match="model not found"):
            adapter.generate([{"role": "user", "content": "hi"}])

    assert len(recorded) == 1, "expected exactly one lm_call emit on the error path"
    event, fields = recorded[0]
    assert event == "lm_call"
    assert fields["provider"] == "car"
    assert fields["status"] == "error"
    assert fields["error_type"] == "RuntimeError"
    assert fields["model"] == "bogus-model"
    # Triage context — answer "which call shape blew up?" without re-running
    assert fields["input_shape"] == "messages_list"
    assert fields["max_tokens"] == 4096
    assert fields["intent_task"] == "code"  # default IntentHint


def test_error_path_metric_emit_failure_does_not_swallow_exception():
    """If the metrics emit itself raises, the original CAR exception must
    still propagate to the caller (metrics are never load-bearing)."""
    class BrokenRuntime:
        def infer_tracked(self, prompt, **kwargs):
            raise ConnectionError("daemon unreachable")

    adapter = CarAdapter(runtime=BrokenRuntime())
    with patch("neo.memory.metrics.record", side_effect=RuntimeError("metrics down")):
        with pytest.raises(ConnectionError, match="daemon unreachable"):
            adapter.generate([{"role": "user", "content": "hi"}])


def test_success_path_emits_status_success():
    """Schema-symmetric with error path: success rows explicitly carry
    status=success rather than relying on absence."""
    rt = FakeRuntime()
    adapter = CarAdapter(runtime=rt)
    recorded: list[tuple[str, dict]] = []
    with patch("neo.memory.metrics.record", side_effect=lambda event, **f: recorded.append((event, f))):
        adapter.generate([{"role": "user", "content": "hi"}])
    assert len(recorded) == 1
    _, fields = recorded[0]
    assert fields["status"] == "success", (
        "success-path emit must carry status=success explicitly — "
        "schema symmetry with error rows (status=error) avoids the NaN-bucket "
        "trap when consumers groupby(status).count()"
    )
    assert "error_type" not in fields


def test_keyboard_interrupt_does_not_emit_error_metric():
    """KeyboardInterrupt is the user choosing to leave, not a CAR failure.
    Locks the BaseException→Exception narrowing against regression."""
    class InterruptingRuntime:
        def infer_tracked(self, prompt, **kwargs):
            raise KeyboardInterrupt()

    recorded: list[tuple[str, dict]] = []
    adapter = CarAdapter(runtime=InterruptingRuntime())
    with patch("neo.memory.metrics.record", side_effect=lambda event, **f: recorded.append((event, f))):
        with pytest.raises(KeyboardInterrupt):
            adapter.generate([{"role": "user", "content": "hi"}])
    assert recorded == [], (
        "KeyboardInterrupt must NOT emit an error metric — that's the user "
        "ending the process, not a provider failure. capture_lm_call_failure "
        "catches Exception, not BaseException."
    )


def test_error_emit_happens_before_reraise_ordering():
    """The metrics row must be visible before the exception propagates —
    otherwise a caller that swallows the exception loses observability of
    a failure that actually happened."""
    events: list[str] = []

    class BrokenRuntime:
        def infer_tracked(self, prompt, **kwargs):
            events.append("infer_call")
            raise RuntimeError("boom")

    def fake_record(event_name, **fields):
        events.append(f"metric:{fields.get('status', '?')}")

    adapter = CarAdapter(runtime=BrokenRuntime())
    with patch("neo.memory.metrics.record", side_effect=fake_record):
        try:
            adapter.generate([{"role": "user", "content": "x"}])
        except RuntimeError:
            events.append("reraise_caught")

    assert events == ["infer_call", "metric:error", "reraise_caught"], (
        f"expected emit-before-reraise ordering; got {events}"
    )


def test_runtime_connection_error_propagates_uncaught(monkeypatch):
    """Daemon-down failures must surface, not be swallowed.

    If CAR's WebSocket connection drops or the daemon isn't running, the
    runtime raises a connection error. CarAdapter must NOT silently return
    empty text — operators need the failure visible so the engine can
    fall back or fail loudly.
    """
    class BrokenRuntime:
        def infer_tracked(self, prompt, **kwargs):
            raise ConnectionError("car-server unreachable at ws://127.0.0.1:9100")

    adapter = CarAdapter(runtime=BrokenRuntime())
    with pytest.raises(ConnectionError, match="car-server unreachable"):
        adapter.generate([{"role": "user", "content": "hi"}])


def test_runtime_runtime_error_propagates_uncaught(monkeypatch):
    """CAR's `rpc infer: -32xxx` errors (model not found, auth, backend
    rejection) come up as RuntimeError. They must surface to the caller."""
    class BrokenRuntime:
        def infer_tracked(self, prompt, **kwargs):
            raise RuntimeError("rpc infer: -32603 model not found: bogus-model")

    adapter = CarAdapter(model="bogus-model", runtime=BrokenRuntime())
    with pytest.raises(RuntimeError, match="model not found"):
        adapter.generate([{"role": "user", "content": "hi"}])


@pytest.mark.skipif(not car_inference.is_available(), reason="car_runtime not installed")
def test_live_bad_model_raises_actionable_error(monkeypatch):
    """Pinning a model CAR doesn't know about must raise a clear error
    (not silently succeed against a fallback)."""
    import os
    import pwd
    monkeypatch.setenv("HOME", pwd.getpwuid(os.getuid()).pw_dir)
    car_inference.reset_for_testing()

    adapter = CarAdapter(model="nonexistent-model-id-xyz")
    with pytest.raises(RuntimeError) as exc_info:
        adapter.generate([{"role": "user", "content": "hi"}], max_tokens=8)
    # The exact message comes from car-server, but it must mention the model
    # so operators can diagnose. We don't pin the wording — just require
    # the failed model id appears somewhere in the message.
    assert "nonexistent-model-id-xyz" in str(exc_info.value)


@pytest.mark.skipif(not car_inference.is_available(), reason="car_runtime not installed")
def test_live_oversized_prompt_does_not_crash_adapter(monkeypatch):
    """Adapter must not crash on a large prompt. CAR may compact, route to
    a high-context model, or return an error — any of those is fine as
    long as the adapter doesn't blow up before delivering the result."""
    import os
    import pwd
    monkeypatch.setenv("HOME", pwd.getpwuid(os.getuid()).pw_dir)
    car_inference.reset_for_testing()

    # ~64KB prompt — well below any reasonable model context limit but
    # large enough to exercise the JSON serialization + transport path.
    big_prompt = "Repeat after me: OK.\n\n" + ("def f(): pass\n" * 3000)
    adapter = CarAdapter()
    try:
        text = adapter.generate(
            [{"role": "user", "content": big_prompt}],
            max_tokens=8,
        )
        # If it succeeded, text is a string (possibly empty).
        assert isinstance(text, str)
    except RuntimeError as e:
        # If CAR rejected it (e.g., context overflow on the selected
        # backend), the error must be readable — not a crash.
        assert "rpc" in str(e).lower() or "context" in str(e).lower() or "token" in str(e).lower(), (
            f"oversized-prompt error should be a parseable CAR rpc error, got: {e}"
        )


@pytest.mark.skipif(not car_inference.is_available(), reason="car_runtime not installed")
def test_real_round_trip_against_live_daemon(monkeypatch):
    """End-to-end smoke against a real CarRuntime + live car-server.

    Skipped if car-runtime isn't installed; surfaces a clear failure (not a
    hang) if the daemon isn't reachable. Acts as the canary that catches the
    mock-vs-reality schema drift that bit us during 0.18.x integration
    (model_used vs model, prompt_tokens vs input_tokens, intent_json vs
    intent_hint).

    The repo's conftest patches ``$HOME`` to a tmp dir for filesystem
    isolation; CAR reads its per-launch auth token from
    ``~/Library/Application Support/ai.parslee.car/auth-token`` (macOS) or
    ``$XDG_RUNTIME_DIR/ai.parslee.car/auth-token`` (Linux). We restore the
    real home for the duration of this test via ``pwd.getpwuid`` so the
    auth handshake on first connection finds the token.
    """
    import os
    import pwd
    real_home = pwd.getpwuid(os.getuid()).pw_dir
    monkeypatch.setenv("HOME", real_home)

    car_inference.reset_for_testing()
    adapter = CarAdapter()  # router-picked model
    text = adapter.generate(
        [{"role": "user", "content": "Reply with exactly the word OK and nothing else."}],
        max_tokens=8,
    )
    # The router occasionally picks `apple/foundation:default` which has
    # been observed to return empty text with usage:null on trivial prompts.
    # That's a CAR-side quirk, not an adapter bug. Assert only the schema
    # contract: we got a string back without the adapter crashing on the
    # transport / JSON / metrics path.
    assert isinstance(text, str)
