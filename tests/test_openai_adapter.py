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
    the network. Built with the adapter's own constructor, then given a client
    carrying the mock transport — the SDK code path is untouched."""
    adapter = OpenAIAdapter(model=model, api_key="test-key", base_url=base_url)
    kwargs = {"api_key": "test-key",
              "http_client": httpx.Client(transport=httpx.MockTransport(handler))}
    if base_url:
        kwargs["base_url"] = base_url
    adapter.client = openai.OpenAI(**kwargs)
    return adapter


@pytest.fixture(autouse=True)
def _no_metrics(monkeypatch):
    monkeypatch.setenv("NEO_METRICS", "off")


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

    with pytest.raises(ValueError, match="No completed message"):
        _adapter("gpt-5.6", handler).generate([{"role": "user", "content": "hi"}])


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
