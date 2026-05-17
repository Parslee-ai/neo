"""Tests for OpenAIAdapter request shaping."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_mock_response():
    return SimpleNamespace(
        status_code=200,
        json=lambda: {
            "output": [
                {
                    "type": "message",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2},
        },
    )


def test_gpt5_responses_payload_includes_output_controls_but_omits_temperature():
    """gpt-5*/o-series/codex on /v1/responses reject `temperature` with a 400
    ("Unsupported parameter"). Their reasoning behavior is steered by
    `reasoning.effort` instead. Adapter must omit temperature from the payload
    for these model families, even when the caller passes one."""
    mock_openai = MagicMock()
    mock_httpx = MagicMock()
    mock_httpx.post.return_value = _make_mock_response()

    with patch.dict(sys.modules, {"openai": mock_openai, "httpx": mock_httpx}):
        from neo.adapters import OpenAIAdapter

        adapter = OpenAIAdapter(model="gpt-5.5", api_key="test-key")
        result = adapter.generate(
            [{"role": "user", "content": "hello"}],
            max_tokens=1234,
            temperature=0.2,             # caller passes one — must be ignored
            reasoning_effort="low",
        )

    assert result == "ok"
    payload = mock_httpx.post.call_args.kwargs["json"]
    assert payload["max_output_tokens"] == 1234
    assert payload["reasoning"] == {"effort": "low"}
    assert "temperature" not in payload, (
        "regression: temperature was sent to /v1/responses — "
        "gpt-5*/o-series/codex reject it with 400"
    )


def test_responses_payload_omits_temperature_for_codex_models():
    """Same constraint applies to codex models on /v1/responses."""
    mock_openai = MagicMock()
    mock_httpx = MagicMock()
    mock_httpx.post.return_value = _make_mock_response()

    with patch.dict(sys.modules, {"openai": mock_openai, "httpx": mock_httpx}):
        from neo.adapters import OpenAIAdapter

        adapter = OpenAIAdapter(model="gpt-5.3-codex", api_key="test-key")
        adapter.generate([{"role": "user", "content": "hi"}], temperature=0.7)

    payload = mock_httpx.post.call_args.kwargs["json"]
    assert "temperature" not in payload
