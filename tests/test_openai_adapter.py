"""Tests for OpenAIAdapter request shaping."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_gpt5_responses_payload_includes_output_controls():
    mock_openai = MagicMock()
    mock_httpx = MagicMock()
    mock_httpx.post.return_value = SimpleNamespace(
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

    with patch.dict(sys.modules, {"openai": mock_openai, "httpx": mock_httpx}):
        from neo.adapters import OpenAIAdapter

        adapter = OpenAIAdapter(model="gpt-5.5", api_key="test-key")
        result = adapter.generate(
            [{"role": "user", "content": "hello"}],
            max_tokens=1234,
            temperature=0.2,
            reasoning_effort="low",
        )

    assert result == "ok"
    payload = mock_httpx.post.call_args.kwargs["json"]
    assert payload["max_output_tokens"] == 1234
    assert payload["temperature"] == 0.2
    assert payload["reasoning"] == {"effort": "low"}
