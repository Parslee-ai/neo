"""Tests for OpenAIAdapter request shaping and transport resilience.

Every model goes through /v1/responses via the real ``openai`` SDK client; only the
network is replaced, with an ``httpx.MockTransport``. Mocking the SDK module
itself would test the mock — the point of these tests is that the SDK's own
retry and error typing actually apply to this path.
"""

import json
from unittest.mock import MagicMock

import httpx
import openai
import pytest

# Pre-warm the numpy-backed import chain that OpenAIAdapter.generate() pulls in
# lazily (via neo.memory.metrics).
import neo.memory.metrics  # noqa: E402,F401
from neo.adapters import OpenAIAdapter


def _responses_body(text="ok", status="completed"):
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "model": "gpt-5.5",
        "status": status,
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": status,
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _adapter(model, handler, base_url=None):
    """A real OpenAIAdapter whose SDK client talks to ``handler`` instead of
    the network. The constructor's own client is COPIED with only the transport
    swapped, so whatever the adapter configured (base_url, timeouts, retries)
    is what the request actually uses — building a fresh client here would
    test the SDK, not the adapter."""
    adapter = OpenAIAdapter(model=model, api_key="example-key", base_url=base_url)
    adapter.client = adapter.client.copy(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    return adapter


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    monkeypatch.setenv("NEO_METRICS", "off")
    # The SDK reads OPENAI_BASE_URL when base_url is None; a developer's
    # proxy setting must not move the URLs these tests assert.
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def _error(param, code, message):
    """A 400 in the shape the live API returns (probed against api.openai.com)."""
    return httpx.Response(400, json={"error": {
        "message": message, "type": "invalid_request_error",
        "param": param, "code": code}})


_UNSUPPORTED_TEMP = ("temperature", None,
                     "Unsupported parameter: 'temperature' is not supported with this model.")
_UNSUPPORTED_EFFORT = ("reasoning.effort", "unsupported_parameter",
                       "Unsupported parameter: 'reasoning.effort' is not supported with this model.")


def _bad_level(value="ultra"):
    return ("reasoning.effort", "invalid_value",
            f"Invalid value: '{value}'. Supported values are: 'none', 'minimal', "
            "'low', 'medium', 'high', 'xhigh', and 'max'.")


def _recording_handler(reject=None):
    """Records each request's path and JSON body. `reject(payload)` returns an
    `_error(...)` argument tuple to refuse the request, or None to accept."""
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append((request.url.path, payload))
        refusal = reject(payload) if reject else None
        if refusal:
            return _error(*refusal)
        return httpx.Response(200, json=_responses_body())

    handler.seen = seen
    return handler


def _learned():
    """What the compat store has persisted to disk, as a fresh process sees it."""
    from pathlib import Path
    return json.loads((Path.home() / ".neo" / "model_param_compat.json").read_text())


MSG = [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-5.6", "gpt-6.1-sol", "o3"])
def test_every_model_goes_through_responses_never_chat(model):
    """The endpoint is not chosen from the model name: every model hits
    /v1/responses and chat completions is never called."""
    handler = _recording_handler()
    assert _adapter(model, handler).generate(MSG, reasoning_effort="high") == "ok"
    assert [path for path, _ in handler.seen] == ["/v1/responses"]


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-5.6", "gpt-6.1-sol", "o3"])
def test_chat_completions_is_never_called(model):
    adapter = OpenAIAdapter(model=model, api_key="example-key")
    adapter.client = MagicMock()
    adapter.client.responses.create.return_value.to_dict.return_value = (
        _responses_body())
    adapter.generate(MSG, reasoning_effort="low")
    assert adapter.client.responses.create.call_count == 1
    adapter.client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("model", ["gpt-4o", "gpt-5.6", "gpt-6.1-sol", "o3"])
def test_the_requested_effort_is_sent_for_every_model(model):
    handler = _recording_handler()
    _adapter(model, handler).generate(MSG, reasoning_effort="xhigh")
    assert handler.seen[0][1]["reasoning"] == {"effort": "xhigh"}


def test_no_effort_requested_sends_no_reasoning_field():
    handler = _recording_handler()
    _adapter("gpt-6.1-sol", handler).generate(MSG, max_tokens=1234, stop=["</neo>"])
    payload = handler.seen[0][1]
    assert "reasoning" not in payload
    assert payload["max_output_tokens"] == 1234
    # The responses endpoint has no stop sequences; the argument is accepted
    # for the ABC and not sent.
    assert "stop" not in payload


def test_rejected_temperature_is_dropped_retried_once_and_remembered():
    """gpt-6.1-sol: 400 with param=temperature and code=None. The retry is paid
    once; the next call (a fresh adapter, as in a new process) sends it never."""
    handler = _recording_handler(
        lambda p: _UNSUPPORTED_TEMP if "temperature" in p else None)

    assert _adapter("gpt-6.1-sol", handler).generate(
        MSG, temperature=0.7, reasoning_effort="xhigh") == "ok"
    assert [("temperature" in p, p["reasoning"]) for _, p in handler.seen] == [
        (True, {"effort": "xhigh"}), (False, {"effort": "xhigh"})]
    assert _learned() == {"openai:gpt-6.1-sol": ["drop_temperature"]}

    handler.seen.clear()
    _adapter("gpt-6.1-sol", handler).generate(MSG, temperature=0.7, reasoning_effort="xhigh")
    assert len(handler.seen) == 1 and "temperature" not in handler.seen[0][1]


def test_rejected_effort_is_dropped_retried_once_and_remembered():
    """gpt-4o-mini: reasoning.effort unsupported; temperature is fine and stays."""
    handler = _recording_handler(
        lambda p: _UNSUPPORTED_EFFORT if "reasoning" in p else None)

    _adapter("gpt-4o-mini", handler).generate(MSG, temperature=0.7, reasoning_effort="high")
    assert len(handler.seen) == 2
    final = handler.seen[-1][1]
    assert "reasoning" not in final and final["temperature"] == 0.7
    assert _learned() == {"openai:gpt-4o-mini": ["drop_reasoning.effort"]}

    handler.seen.clear()
    _adapter("gpt-4o-mini", handler).generate(MSG, temperature=0.7, reasoning_effort="high")
    assert len(handler.seen) == 1 and "reasoning" not in handler.seen[0][1]


def test_a_rejected_level_is_lowered_to_the_highest_accepted_and_remembered():
    """The model supports effort up to `high`: xhigh is refused as invalid, so
    the level steps down once and effort is NOT dropped. The cap is remembered."""
    handler = _recording_handler(
        lambda p: _bad_level("xhigh") if p["reasoning"]["effort"] == "xhigh" else None)

    _adapter("gpt-5.6", handler).generate(MSG, reasoning_effort="xhigh")
    assert [p["reasoning"]["effort"] for _, p in handler.seen] == ["xhigh", "high"]
    assert _learned() == {"openai:gpt-5.6": ["max_reasoning.effort:high"]}

    handler.seen.clear()
    _adapter("gpt-5.6", handler).generate(MSG, reasoning_effort="xhigh")
    assert [p["reasoning"]["effort"] for _, p in handler.seen] == ["high"]
    # A level already under the cap is left alone.
    handler.seen.clear()
    _adapter("gpt-5.6", handler).generate(MSG, reasoning_effort="low")
    assert [p["reasoning"]["effort"] for _, p in handler.seen] == ["low"]


def test_level_steps_down_rung_by_rung_in_effort_levels_order():
    ok_levels = {"none", "low"}
    handler = _recording_handler(
        lambda p: None if p["reasoning"]["effort"] in ok_levels else _bad_level())
    _adapter("gpt-5.6", handler).generate(MSG, reasoning_effort="xhigh")
    assert [p["reasoning"]["effort"] for _, p in handler.seen] == [
        "xhigh", "high", "medium", "low"]


def test_effort_is_dropped_only_when_even_the_lowest_level_is_refused():
    handler = _recording_handler(
        lambda p: _bad_level() if "reasoning" in p else None)
    _adapter("gpt-5.6", handler).generate(MSG, reasoning_effort="medium")
    assert [p.get("reasoning", {}).get("effort") for _, p in handler.seen] == [
        "medium", "low", "none", None]
    assert "reasoning" not in handler.seen[-1][1]


def test_an_unrelated_400_reraises_and_learns_nothing():
    handler = _recording_handler(
        lambda p: ("input", "invalid_value", "Invalid value for 'input': bad shape"))
    with pytest.raises(openai.BadRequestError):
        _adapter("gpt-6.1-sol", handler).generate(MSG, temperature=0.7, reasoning_effort="low")
    assert len(handler.seen) == 1
    from pathlib import Path
    assert not (Path.home() / ".neo" / "model_param_compat.json").exists()


def test_auth_failure_reraises_untouched():
    def handler(request):
        return httpx.Response(401, json={"error": {
            "message": "Incorrect API key provided: temperature", "code": "invalid_api_key",
            "param": None, "type": "invalid_request_error"}})
    with pytest.raises(openai.AuthenticationError):
        _adapter("gpt-6.1-sol", handler).generate(MSG, temperature=0.7)


def test_an_unrelated_type_error_reraises():
    adapter = OpenAIAdapter(model="gpt-6.1-sol", api_key="example-key")
    adapter.client = MagicMock()
    adapter.client.responses.create.side_effect = TypeError("'NoneType' object is not subscriptable")
    with pytest.raises(TypeError, match="NoneType"):
        adapter.generate(MSG, temperature=0.7)
    assert adapter.client.responses.create.call_count == 1


def test_sdk_refusing_a_keyword_is_recovered_like_a_400():
    adapter = OpenAIAdapter(model="gpt-6.1-sol", api_key="example-key")
    adapter.client = MagicMock()
    ok = MagicMock()
    ok.to_dict.return_value = _responses_body()
    adapter.client.responses.create.side_effect = [
        TypeError("Responses.create() got an unexpected keyword argument 'temperature'"), ok]
    assert adapter.generate(MSG, temperature=0.7) == "ok"
    assert "temperature" not in adapter.client.responses.create.call_args.kwargs


def test_a_learnings_file_in_todays_format_still_applies():
    """Files written before the helper was generalised use the same shape and
    must keep working: drop_temperature applies to the responses payload, and
    the chat-only rename flag is simply inert here."""
    from pathlib import Path
    path = Path.home() / ".neo" / "model_param_compat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"openai:o3": ["drop_temperature", "rename_max_tokens"]}))

    handler = _recording_handler()
    _adapter("o3", handler).generate(MSG, temperature=0.7)
    assert len(handler.seen) == 1
    payload = handler.seen[0][1]
    assert "temperature" not in payload and payload["max_output_tokens"] == 4096


def test_a_successful_call_records_the_lm_call_metric(monkeypatch):
    from unittest.mock import patch
    handler = _recording_handler()
    with patch("neo.memory.metrics.record") as record:
        _adapter("gpt-6.1-sol", handler).generate(MSG)
    lm_calls = [c for c in record.call_args_list if c.args[0] == "lm_call"]
    assert len(lm_calls) == 1
    assert lm_calls[0].kwargs["input_tokens"] == 10
    assert lm_calls[0].kwargs["output_tokens"] == 2


def test_responses_path_survives_a_connection_reset():
    """One `Connection reset by peer` used to fail the whole neo run: the path
    was a bare httpx.post with no retry, and two live episodes ended as
    engine_error for exactly this. The SDK retries it."""
    attempts = []

    def handler(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            raise httpx.ReadError("[Errno 54] Connection reset by peer")
        return httpx.Response(200, json=_responses_body("recovered"))

    assert _adapter("gpt-5.6", handler).generate(
        [{"role": "user", "content": "hi"}]) == "recovered"
    assert attempts == ["/v1/responses", "/v1/responses"]


def test_responses_path_retries_a_503():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json=_responses_body())

    assert _adapter("gpt-5.6", handler).generate([{"role": "user", "content": "hi"}]) == "ok"
    assert len(attempts) == 2


def test_responses_path_does_not_retry_a_400_and_raises_typed_status():
    """A 400 is the caller's fault and will not improve on retry. It must also
    arrive TYPED — the raw post raised ValueError("API error 400: ...") and no
    caller could tell a rejected request from an outage."""
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(400, json={"error": {"message": "Unsupported parameter"}})

    with pytest.raises(openai.APIStatusError) as exc:
        _adapter("gpt-5.6", handler).generate([{"role": "user", "content": "hi"}])
    assert exc.value.status_code == 400
    assert len(attempts) == 1


def test_incomplete_response_still_raises():
    """A response cut off by max_output_tokens has no completed message; it
    must not be returned as an empty answer."""
    def handler(request):
        return httpx.Response(200, json=_responses_body(status="incomplete"))

    with pytest.raises(ValueError, match="No completed message") as exc:
        _adapter("gpt-5.6", handler).generate([{"role": "user", "content": "hi"}])
    # Diagnosable without echoing the model's partial output into logs.
    assert "incomplete" in str(exc.value)
    assert "output_text" not in str(exc.value)


def test_responses_base_url_follows_sdk_convention():
    """base_url ends in /v1, as the chat path always required; the raw post
    appended /v1 itself, so no single base_url served both paths."""
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_responses_body())

    _adapter("gpt-5.6", handler, base_url="https://proxy.example/v1").generate(
        [{"role": "user", "content": "hi"}])
    assert urls == ["https://proxy.example/v1/responses"]


def test_responses_call_keeps_the_short_connect_timeout():
    """A per-call `timeout=600.0` float raised the SDK's 5s connect timeout to
    600s, so a blackholed connect hung ten minutes per attempt, three times."""
    timeouts = []

    def handler(request):
        timeouts.append(request.extensions["timeout"])
        return httpx.Response(200, json=_responses_body())

    _adapter("gpt-5.6", handler).generate([{"role": "user", "content": "hi"}])
    assert timeouts[0]["connect"] <= 10, timeouts[0]
    assert timeouts[0]["read"] >= 600, timeouts[0]


def test_unknown_output_item_type_prints_no_warning(recwarn):
    """The API adds output item types over time. Serializing one must not emit
    a pydantic warning: stderr is the --json event stream."""
    body = _responses_body("still ok")
    body["output"].insert(0, {"type": "some_future_item", "id": "x", "payload": 1})

    def handler(request):
        return httpx.Response(200, json=body)

    assert _adapter("gpt-5.6", handler).generate(
        [{"role": "user", "content": "hi"}]) == "still ok"
    assert not [w for w in recwarn if "serializ" in str(w.message).lower()]


def test_sdk_timeout_is_classified_as_a_network_timeout():
    """The CLI's NetworkTimeout envelope checked only httpx timeouts, which the
    SDK wraps — so it never fired for an OpenAI or Azure call."""
    from neo.cli import _is_network_timeout

    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    assert _is_network_timeout(openai.APITimeoutError(request=request))
    assert _is_network_timeout(httpx.ReadTimeout("slow", request=request))
    assert _is_network_timeout(httpx.PoolTimeout("pool", request=request))
    assert not _is_network_timeout(
        openai.APIConnectionError(message="dns", request=request))
    assert not _is_network_timeout(ValueError("API error 400"))


def test_exhausted_retries_keep_the_status_code():
    """When every attempt returns 503 the caller must still get a TYPED error
    with the status: transcript mining decides retry-later vs give-up on it."""
    attempts = []

    def handler(request):
        attempts.append(1)
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    with pytest.raises(openai.APIStatusError) as exc:
        _adapter("gpt-5.6", handler).generate([{"role": "user", "content": "hi"}])
    assert exc.value.status_code == 503
    assert len(attempts) == 3  # one call + the SDK's two retries
