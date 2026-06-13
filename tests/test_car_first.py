"""Tests for CAR-first inference resolution (resolve_adapter + AutoAdapter)."""

import neo.adapters as A
import neo.car_inference as ci
from neo.adapters import AutoAdapter, resolve_adapter
from neo.config import NeoConfig


class _FakeAdapter:
    def __init__(self, label):
        self.label = label
        self.calls = 0
        self.fail = False

    def generate(self, messages, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("car down")
        return self.label

    def name(self):
        return self.label


def _stub_create(monkeypatch):
    """Make create_adapter return a labeled fake per provider."""
    monkeypatch.setattr(A, "create_adapter",
                        lambda provider, model=None, **kw: _FakeAdapter(f"{provider}"))


# --- config ---------------------------------------------------------------

def test_inference_mode_default_is_static():
    # Default stays static (gpt-5.5) until a CAR release verifies router quality.
    assert NeoConfig().inference_mode == "static"


def test_inference_mode_env_override(monkeypatch):
    monkeypatch.setenv("NEO_INFERENCE_MODE", "static")
    assert NeoConfig.from_env().inference_mode == "static"


# --- resolve_adapter ------------------------------------------------------

def test_static_mode_never_uses_car(monkeypatch):
    _stub_create(monkeypatch)
    # Even if CAR is fully available, static mode must not route through it.
    monkeypatch.setattr(ci, "is_available", lambda: True)
    import neo.a2ui as ui
    monkeypatch.setattr(ui, "is_daemon_reachable", lambda timeout=0.5: True)
    a = resolve_adapter(NeoConfig(provider="openai", inference_mode="static"))
    assert a.name() == "openai" and not isinstance(a, AutoAdapter)


def test_auto_falls_back_when_car_not_installed(monkeypatch):
    _stub_create(monkeypatch)
    monkeypatch.setattr(ci, "is_available", lambda: False)
    a = resolve_adapter(NeoConfig(provider="openai", inference_mode="auto"))
    assert a.name() == "openai" and not isinstance(a, AutoAdapter)


def test_auto_falls_back_when_daemon_unreachable(monkeypatch):
    _stub_create(monkeypatch)
    monkeypatch.setattr(ci, "is_available", lambda: True)
    import neo.a2ui as ui
    monkeypatch.setattr(ui, "is_daemon_reachable", lambda timeout=0.5: False)
    a = resolve_adapter(NeoConfig(provider="openai", inference_mode="auto"))
    assert a.name() == "openai" and not isinstance(a, AutoAdapter)


def test_auto_uses_car_when_available(monkeypatch):
    _stub_create(monkeypatch)
    monkeypatch.setattr(ci, "is_available", lambda: True)
    import neo.a2ui as ui
    monkeypatch.setattr(ui, "is_daemon_reachable", lambda timeout=0.5: True)
    a = resolve_adapter(NeoConfig(provider="openai", inference_mode="auto"))
    assert isinstance(a, AutoAdapter)
    assert "car" in a.name()  # car-first; fallback is lazy ("auto(car -> static)")


def test_auto_does_not_build_static_when_car_available(monkeypatch):
    # CAR-only scenario: static provider would raise (no key) — must NOT block CAR.
    monkeypatch.setattr(ci, "is_available", lambda: True)
    import neo.a2ui as ui
    monkeypatch.setattr(ui, "is_daemon_reachable", lambda timeout=0.5: True)

    def _create(provider, model=None, **kw):
        if provider == "car":
            return _FakeAdapter("car")
        raise ValueError("OpenAI API key required")  # static is unconfigured

    monkeypatch.setattr(A, "create_adapter", _create)
    a = resolve_adapter(NeoConfig(provider="openai", inference_mode="auto"))
    assert isinstance(a, AutoAdapter) and a.generate([]) == "car"  # no static build needed


# --- AutoAdapter runtime fallback + circuit breaker -----------------------

def test_autoadapter_prefers_car():
    car, fb = _FakeAdapter("car"), _FakeAdapter("fb")
    assert AutoAdapter(car, lambda: fb).generate([]) == "car"
    assert car.calls == 1 and fb.calls == 0


def test_autoadapter_falls_back_and_circuit_breaks():
    car, fb = _FakeAdapter("car"), _FakeAdapter("fb")
    a = AutoAdapter(car, lambda: fb)  # default cooldown -> breaker stays open
    assert a.generate([]) == "car"        # 1: car ok
    car.fail = True
    assert a.generate([]) == "fb"         # 2: car fails -> fallback, breaker opens
    car.fail = False
    assert a.generate([]) == "fb"         # 3: breaker open -> straight to fallback
    assert car.calls == 2                 # car not retried while breaker open
    assert fb.calls == 2


def test_autoadapter_half_opens_and_recovers():
    car, fb = _FakeAdapter("car"), _FakeAdapter("fb")
    a = AutoAdapter(car, lambda: fb, retry_cooldown=0)  # immediate half-open
    assert a.generate([]) == "car"
    car.fail = True
    assert a.generate([]) == "fb"         # fails -> fallback
    car.fail = False
    assert a.generate([]) == "car"        # cooldown 0 -> half-open -> CAR retried, recovers


def test_autoadapter_fallback_built_lazily():
    car = _FakeAdapter("car")
    built = []
    a = AutoAdapter(car, lambda: built.append(1) or _FakeAdapter("fb"))
    a.generate([])                        # CAR ok -> fallback never constructed
    assert built == []
