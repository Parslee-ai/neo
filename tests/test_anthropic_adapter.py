"""Tests for AnthropicAdapter request shaping.

Focus: newer Claude models (Opus 4.7+, Sonnet 5, Fable 5) reject the
`temperature` sampling parameter with a 400 ("temperature is deprecated for
this model"). The adapter must recover by dropping the param and retrying,
then remember the model so subsequent calls omit it up front.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Pre-warm the numpy-backed import chain that AnthropicAdapter.generate() pulls
# in lazily (via neo.memory.metrics). If these are first imported *inside* a
# patch.dict(sys.modules) window, patch.dict deletes them on teardown and the
# numpy C-extensions can't reload ("cannot load module more than once per
# process"). Importing them here keeps them out of the patched delta.
import neo.memory.metrics  # noqa: E402,F401
from neo import adapters  # noqa: E402,F401


class _BadRequestError(Exception):
    """Stand-in for anthropic.BadRequestError."""


@pytest.fixture
def _fake_anthropic():
    """Inject a fake `anthropic` module and reset the learned-model cache."""
    mock_anthropic = MagicMock()
    mock_anthropic.BadRequestError = _BadRequestError
    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        from neo.adapters import AnthropicAdapter

        AnthropicAdapter._models_without_temperature.clear()
        try:
            yield AnthropicAdapter
        finally:
            AnthropicAdapter._models_without_temperature.clear()


def _ok_response(text="ok"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=2,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def test_temperature_sent_for_models_that_accept_it(_fake_anthropic):
    adapter = _fake_anthropic(model="claude-sonnet-4-5-20250929", api_key="k")
    adapter.client = MagicMock()
    adapter.client.messages.create.return_value = _ok_response()

    adapter.generate([{"role": "user", "content": "hi"}], temperature=0.3)

    kwargs = adapter.client.messages.create.call_args.kwargs
    assert kwargs["temperature"] == 0.3


def test_drops_temperature_and_retries_on_400(_fake_anthropic):
    """A 400 naming `temperature` triggers a param-stripped retry that succeeds."""
    adapter = _fake_anthropic(model="claude-opus-4-8", api_key="k")
    adapter.client = MagicMock()
    adapter.client.messages.create.side_effect = [
        _BadRequestError("`temperature` is deprecated for this model."),
        _ok_response("recovered"),
    ]

    result = adapter.generate([{"role": "user", "content": "hi"}], temperature=0.7)

    assert result == "recovered"
    assert adapter.client.messages.create.call_count == 2
    # First call included temperature; retry omitted it.
    assert "temperature" in adapter.client.messages.create.call_args_list[0].kwargs
    assert "temperature" not in adapter.client.messages.create.call_args_list[1].kwargs
    # The model is remembered process-wide.
    assert "claude-opus-4-8" in adapter._models_without_temperature


def test_remembered_model_omits_temperature_up_front(_fake_anthropic):
    """After learning a model rejects temperature, later calls skip it — no
    wasted 400."""
    adapter = _fake_anthropic(model="claude-opus-4-8", api_key="k")
    adapter._models_without_temperature.add("claude-opus-4-8")
    adapter.client = MagicMock()
    adapter.client.messages.create.return_value = _ok_response()

    adapter.generate([{"role": "user", "content": "hi"}], temperature=0.7)

    assert adapter.client.messages.create.call_count == 1
    assert "temperature" not in adapter.client.messages.create.call_args.kwargs


def test_unrelated_400_reraises(_fake_anthropic):
    """A 400 that isn't about temperature propagates — no silent retry."""
    adapter = _fake_anthropic(model="claude-opus-4-8", api_key="k")
    adapter.client = MagicMock()
    adapter.client.messages.create.side_effect = _BadRequestError(
        "max_tokens: must be positive"
    )

    with pytest.raises(_BadRequestError):
        adapter.generate([{"role": "user", "content": "hi"}], temperature=0.7)

    assert adapter.client.messages.create.call_count == 1
    assert "claude-opus-4-8" not in adapter._models_without_temperature
