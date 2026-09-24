"""Tests for OpenAIAdapter request shaping and transport resilience.

The gpt-5*/codex path goes through the real ``openai`` SDK client; only the
network is replaced, with an ``httpx.MockTransport``. Mocking the SDK module
itself would test the mock — the point of these tests is that the SDK's own
retry and error typing actually apply to this path.
"""

import json

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


def test_gpt5_responses_payload_includes_output_controls_but_omits_temperature():
    """gpt-5*/o-series/codex on /v1/responses reject `temperature` with a 400
    ("Unsupported parameter"). Their reasoning behavior is steered by
    `reasoning.effort` instead. Adapter must omit temperature from the payload
    for these model families, even when the caller passes one."""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_responses_body())

    adapter = _adapter("gpt-5.5", handler)
    result = adapter.generate(
        [{"role": "user", "content": "hello"}],
        max_tokens=1234,
        temperature=0.2,             # caller passes one — must be ignored
        reasoning_effort="low",
    )

    assert result == "ok"
    payload = seen[0]
    assert payload["max_output_tokens"] == 1234
    assert payload["reasoning"] == {"effort": "low"}
    assert "temperature" not in payload, (
        "regression: temperature was sent to /v1/responses — "
        "gpt-5*/o-series/codex reject it with 400"
    )


def test_responses_payload_omits_temperature_for_codex_models():
    """Same constraint applies to codex models on /v1/responses."""
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_responses_body())

    _adapter("gpt-5.3-codex", handler).generate(
        [{"role": "user", "content": "hi"}], temperature=0.7)
    assert "temperature" not in seen[0]


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
