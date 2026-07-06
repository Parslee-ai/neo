"""Tests for reasoning-model parameter compatibility across the OpenAI-SDK
adapters (OpenAI chat path, Azure OpenAI, Local/OpenAI-compatible).

Newer reasoning models reject `temperature` and require `max_completion_tokens`
instead of `max_tokens`. `_chat_completion_resilient` learns each adaptation
from the API's own 400 and remembers it. On Azure the model is an arbitrary
deployment name, so this reactive approach is the only reliable one.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Pre-warm the numpy-backed import chain the OpenAI adapter pulls in lazily
# (via neo.memory.metrics), so patch.dict(sys.modules) doesn't delete numpy's
# C-extensions on teardown ("cannot load module more than once per process").
import neo.memory.metrics  # noqa: E402,F401


class _BadRequest(Exception):
    """Stand-in for openai.BadRequestError."""


_TEMP_ERR = _BadRequest(
    "Unsupported value: 'temperature' does not support 0.7 with this model. "
    "Only the default (1) value is supported."
)
_MAXTOK_ERR = _BadRequest(
    "Unsupported parameter: 'max_tokens' is not supported with this model. "
    "Use 'max_completion_tokens' instead."
)


def _ok(text="ok"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=None,
    )


@pytest.fixture
def _fake_openai():
    """Inject a fake `openai` module (with a real BadRequestError) and reset
    the process-wide learned-model caches."""
    mock_openai = MagicMock()
    mock_openai.BadRequestError = _BadRequest
    with patch.dict(sys.modules, {"openai": mock_openai}):
        from neo import adapters

        adapters._MODELS_REJECTING_TEMPERATURE.clear()
        adapters._MODELS_NEEDING_MAX_COMPLETION_TOKENS.clear()
        try:
            yield adapters
        finally:
            adapters._MODELS_REJECTING_TEMPERATURE.clear()
            adapters._MODELS_NEEDING_MAX_COMPLETION_TOKENS.clear()


def _client_with(side_effect):
    client = MagicMock()
    client.chat.completions.create.side_effect = side_effect
    return client


# --- the shared helper directly -------------------------------------------

def test_helper_passes_through_when_accepted(_fake_openai):
    client = _client_with([_ok("fine")])
    out = _fake_openai._chat_completion_resilient(
        client, {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}
    )
    assert out.choices[0].message.content == "fine"
    assert client.chat.completions.create.call_count == 1


def test_helper_drops_temperature_then_renames_max_tokens(_fake_openai):
    """A reasoning model that rejects both params: temperature is dropped, then
    max_tokens is renamed, then the call succeeds — three attempts total."""
    client = _client_with([_TEMP_ERR, _MAXTOK_ERR, _ok("recovered")])
    out = _fake_openai._chat_completion_resilient(
        client, {"model": "o3-mini", "temperature": 0.7, "max_tokens": 100}
    )
    assert out.choices[0].message.content == "recovered"
    calls = client.chat.completions.create.call_args_list
    assert len(calls) == 3
    # call 1: both present
    assert "temperature" in calls[0].kwargs and "max_tokens" in calls[0].kwargs
    # call 2: temperature gone, max_tokens still there
    assert "temperature" not in calls[1].kwargs and "max_tokens" in calls[1].kwargs
    # call 3: renamed to max_completion_tokens, no max_tokens, no temperature
    assert "max_tokens" not in calls[2].kwargs
    assert calls[2].kwargs["max_completion_tokens"] == 100
    assert "temperature" not in calls[2].kwargs
    # both adaptations remembered
    assert "o3-mini" in _fake_openai._MODELS_REJECTING_TEMPERATURE
    assert "o3-mini" in _fake_openai._MODELS_NEEDING_MAX_COMPLETION_TOKENS


def test_helper_applies_learned_adaptations_up_front(_fake_openai):
    """Once learned, a model skips both bad params on the first attempt."""
    _fake_openai._MODELS_REJECTING_TEMPERATURE.add("o3-mini")
    _fake_openai._MODELS_NEEDING_MAX_COMPLETION_TOKENS.add("o3-mini")
    client = _client_with([_ok()])

    _fake_openai._chat_completion_resilient(
        client, {"model": "o3-mini", "temperature": 0.7, "max_tokens": 100}
    )

    assert client.chat.completions.create.call_count == 1
    kw = client.chat.completions.create.call_args.kwargs
    assert "temperature" not in kw
    assert "max_tokens" not in kw
    assert kw["max_completion_tokens"] == 100


def test_helper_reraises_unrelated_400(_fake_openai):
    client = _client_with([_BadRequest("messages: must be a non-empty array")])
    with pytest.raises(_BadRequest):
        _fake_openai._chat_completion_resilient(
            client, {"model": "gpt-4", "temperature": 0.7, "max_tokens": 100}
        )
    assert client.chat.completions.create.call_count == 1
    assert not _fake_openai._MODELS_REJECTING_TEMPERATURE


def test_helper_renames_on_unsupported_wording_without_replacement_named(_fake_openai):
    """Robust to wording that doesn't name the replacement param — an
    `unsupported`/`not supported` signal on max_tokens is enough."""
    err = _BadRequest("Unsupported parameter: 'max_tokens' is not supported with this model.")
    client = _client_with([err, _ok("renamed")])
    out = _fake_openai._chat_completion_resilient(
        client, {"model": "some-reasoner", "max_tokens": 100}
    )
    assert out.choices[0].message.content == "renamed"
    assert client.chat.completions.create.call_args_list[-1].kwargs["max_completion_tokens"] == 100
    assert "some-reasoner" in _fake_openai._MODELS_NEEDING_MAX_COMPLETION_TOKENS


def test_helper_does_not_rename_on_plain_max_tokens_value_error(_fake_openai):
    """A value error on max_tokens (no unsupported signal) must re-raise, not
    trigger a spurious rename."""
    client = _client_with([_BadRequest("max_tokens must be at least 1")])
    with pytest.raises(_BadRequest):
        _fake_openai._chat_completion_resilient(
            client, {"model": "gpt-4", "max_tokens": 0}
        )
    assert client.chat.completions.create.call_count == 1
    assert not _fake_openai._MODELS_NEEDING_MAX_COMPLETION_TOKENS


def test_helper_does_not_mutate_callers_kwargs(_fake_openai):
    """The helper copies kwargs — a caller passing a shared/cached dict is safe."""
    caller_kwargs = {"model": "o3-mini", "temperature": 0.7, "max_tokens": 100}
    snapshot = dict(caller_kwargs)
    client = _client_with([_TEMP_ERR, _MAXTOK_ERR, _ok()])
    _fake_openai._chat_completion_resilient(client, caller_kwargs)
    assert caller_kwargs == snapshot  # untouched despite two adaptations


# --- through the adapters ---------------------------------------------------

def test_local_adapter_recovers_from_temperature_rejection(_fake_openai):
    adapter = _fake_openai.LocalAdapter(model="grok-4-reasoning")
    adapter.client = _client_with([_TEMP_ERR, _ok("hi")])

    out = adapter.generate([{"role": "user", "content": "yo"}], temperature=0.5)

    assert out == "hi"
    assert adapter.client.chat.completions.create.call_count == 2
    assert "grok-4-reasoning" in _fake_openai._MODELS_REJECTING_TEMPERATURE


def test_azure_adapter_recovers_from_both_rejections(_fake_openai):
    adapter = _fake_openai.AzureOpenAIAdapter(
        model="my-o3-deploy", api_key="k", endpoint="https://x.openai.azure.com"
    )
    adapter.client = _client_with([_TEMP_ERR, _MAXTOK_ERR, _ok("azure-ok")])

    out = adapter.generate([{"role": "user", "content": "yo"}], temperature=0.5)

    assert out == "azure-ok"
    calls = adapter.client.chat.completions.create.call_args_list
    assert len(calls) == 3
    assert calls[-1].kwargs["max_completion_tokens"] == 4096
    assert "temperature" not in calls[-1].kwargs
    assert "my-o3-deploy" in _fake_openai._MODELS_NEEDING_MAX_COMPLETION_TOKENS


def test_openai_chat_path_recovers_for_o_series(_fake_openai):
    """o-series slips past the gpt-5/codex responses-endpoint routing and lands
    on the chat path, where the helper recovers it."""
    adapter = _fake_openai.OpenAIAdapter(model="o3-mini", api_key="k")
    adapter.client = _client_with([_TEMP_ERR, _ok("chat-ok")])

    out = adapter.generate([{"role": "user", "content": "yo"}], temperature=0.9)

    assert out == "chat-ok"
    assert "o3-mini" in _fake_openai._MODELS_REJECTING_TEMPERATURE
