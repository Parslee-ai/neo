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
    """Inject a fake `anthropic` module and reset the persistent param-compat
    store's in-memory cache (the autouse `isolate_neo_home` fixture keeps the
    on-disk state in a fresh tmp dir)."""
    mock_anthropic = MagicMock()
    mock_anthropic.BadRequestError = _BadRequestError
    with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
        from neo import adapters

        def _reset():
            adapters._PARAM_COMPAT._path = None
            adapters._PARAM_COMPAT._data = {}

        _reset()
        try:
            yield adapters.AnthropicAdapter
        finally:
            _reset()


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
    # The model is remembered in the persistent store.
    assert adapters._PARAM_COMPAT.has("anthropic", "claude-opus-4-8", "drop_temperature")


def test_remembered_model_omits_temperature_up_front(_fake_anthropic):
    """After learning a model rejects temperature, later calls skip it — no
    wasted 400."""
    adapter = _fake_anthropic(model="claude-opus-4-8", api_key="k")
    adapters._PARAM_COMPAT.learn("anthropic", "claude-opus-4-8", "drop_temperature")
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
    assert not adapters._PARAM_COMPAT.has("anthropic", "claude-opus-4-8", "drop_temperature")


# ------------------------------------------------- the SDK dropping the keyword


class TestSdkKeywordRemoval:
    """A provider can retire a sampling parameter in two ways, and only one of
    them is an HTTP error.

    `anthropic` 1.0.0 removed `temperature` from `Messages.create()` outright,
    so the call raises a **client-side TypeError** before any request is made.
    The `BadRequestError` recovery above never saw it, and every Anthropic call
    failed with `ProcessingError: Messages.create() got an unexpected keyword
    argument 'temperature'`. `pyproject.toml` declared `anthropic>=0.21.0` with
    no upper bound, so CI resolved into the major bump on its own.

    It was caught by the release gate — one real LLM round trip per language —
    and by nothing else in 2769 tests, because every test here mocks the SDK
    and a mock accepts any keyword.
    """

    def test_recognises_the_sdk_refusing_a_keyword(self):
        exc = TypeError("Messages.create() got an unexpected keyword argument 'temperature'")
        assert adapters._sdk_rejects_keyword(exc, "temperature")

    @pytest.mark.parametrize("message", [
        "unsupported operand type(s) for +: 'int' and 'str'",
        "Messages.create() got an unexpected keyword argument 'top_p'",
        "Messages.create() missing 1 required positional argument: 'model'",
        "'NoneType' object is not subscriptable",
    ])
    def test_unrelated_type_errors_are_not_swallowed(self, message):
        """This sits on the inference path. Treating an unrelated TypeError as
        a parameter rejection would silently degrade every call."""
        assert not adapters._sdk_rejects_keyword(TypeError(message), "temperature")

    def test_drops_temperature_and_retries_on_sdk_type_error(self, _fake_anthropic):
        """One refusal, one retry without the parameter, a successful answer —
        the behaviour the release gate demanded."""
        adapter = _fake_anthropic(model="claude-sonnet-4-5-20250929", api_key="k")
        adapter.client = MagicMock()
        adapter.client.messages.create.side_effect = [
            TypeError("Messages.create() got an unexpected keyword argument 'temperature'"),
            _ok_response(),
        ]

        assert adapter.generate([{"role": "user", "content": "hi"}], temperature=0.7) == "ok"

        assert adapter.client.messages.create.call_count == 2
        assert "temperature" in adapter.client.messages.create.call_args_list[0].kwargs
        assert "temperature" not in adapter.client.messages.create.call_args_list[1].kwargs

    def test_the_rejection_is_remembered_so_the_next_call_omits_it(self, _fake_anthropic):
        """Otherwise every call pays a wasted round of argument binding."""
        adapter = _fake_anthropic(model="claude-sonnet-4-5-20250929", api_key="k")
        adapter.client = MagicMock()
        adapter.client.messages.create.side_effect = [
            TypeError("Messages.create() got an unexpected keyword argument 'temperature'"),
            _ok_response(),
            _ok_response(),
        ]
        adapter.generate([{"role": "user", "content": "hi"}], temperature=0.7)
        adapter.generate([{"role": "user", "content": "again"}], temperature=0.7)

        assert "temperature" not in adapter.client.messages.create.call_args_list[-1].kwargs

    def test_an_unrelated_type_error_still_propagates(self, _fake_anthropic):
        adapter = _fake_anthropic(model="claude-sonnet-4-5-20250929", api_key="k")
        adapter.client = MagicMock()
        adapter.client.messages.create.side_effect = TypeError(
            "'NoneType' object is not subscriptable"
        )
        with pytest.raises(TypeError, match="NoneType"):
            adapter.generate([{"role": "user", "content": "hi"}], temperature=0.7)
