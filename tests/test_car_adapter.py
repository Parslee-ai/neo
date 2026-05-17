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
    """Minimal CarRuntime stand-in for unit tests."""

    def __init__(self, response: dict | str | None = None, raise_on: str | None = None):
        self.calls: list[tuple[str, dict]] = []
        self._response = response if response is not None else {
            "text": "ok",
            "model": "qwen3-32b",
            "usage": {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 30},
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
    rt = FakeRuntime({"text": "hello from car", "model": "gpt-5", "usage": {}})
    adapter = CarAdapter(runtime=rt)
    out = adapter.generate([{"role": "user", "content": "hi"}])
    assert out == "hello from car"
    assert len(rt.calls) == 1
    prompt, kwargs = rt.calls[0]
    assert "user: hi" in prompt
    assert kwargs["max_tokens"] == 4096
    assert "model" not in kwargs  # model=None means let router decide


def test_generate_pins_model_when_set():
    rt = FakeRuntime()
    adapter = CarAdapter(model="qwen3-32b", runtime=rt)
    adapter.generate([{"role": "user", "content": "hi"}], max_tokens=512, temperature=0.2)
    _, kwargs = rt.calls[0]
    assert kwargs["model"] == "qwen3-32b"
    assert kwargs["max_tokens"] == 512
    assert kwargs["temperature"] == 0.2


def test_generate_passes_intent_hint():
    rt = FakeRuntime()
    intent = {"task": "code", "prefer_local": True}
    adapter = CarAdapter(intent_hint=intent, runtime=rt)
    adapter.generate([{"role": "user", "content": "refactor this"}])
    _, kwargs = rt.calls[0]
    assert kwargs["intent_hint"] == intent


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


def test_metrics_emit_is_best_effort():
    """Metrics failures must not break the adapter."""
    rt = FakeRuntime({"text": "x", "model": "m", "usage": {"input_tokens": 1, "output_tokens": 1}})
    adapter = CarAdapter(runtime=rt)
    with patch("neo.memory.metrics.record", side_effect=RuntimeError("metrics down")):
        out = adapter.generate([{"role": "user", "content": "hi"}])
    assert out == "x"  # adapter still returned successfully
