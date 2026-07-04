"""Integration tests for the engine's reasoning-tier wiring (gate + branch)."""

import json

from neo.engine import NeoEngine
from neo.reasoning_effort import MemorySignal
from neo.reasoning_mode import ReasoningMode


class FakeLM:
    def __init__(self, responder):
        self._r = responder
        self.model = "fake"
        self.provider = "fake"

    def generate(self, messages, **kw):
        return self._r(messages)

    def name(self):
        return "fake-lm"


def _engine(lm, config=None):
    return NeoEngine(lm_adapter=lm, enable_persistent_memory=False, config=config)


class _NI:
    prompt = "do a novel thing"


def test_no_car_forces_fast(monkeypatch):
    e = _engine(FakeLM(lambda m: "{}"))
    monkeypatch.setattr(e, "_car_route_capability", lambda p: (False, 0, None))
    monkeypatch.setattr(e, "_compute_memory_signal", lambda c: MemorySignal(0, 0.0))
    d, route = e._decide_reasoning_mode({"prompt": "x"}, "hard", _NI())
    assert d.mode is ReasoningMode.FAST
    assert route is None


def test_car_with_pool_and_novelty_deliberates(monkeypatch):
    e = _engine(FakeLM(lambda m: "{}"))
    monkeypatch.setattr(e, "_car_route_capability", lambda p: (True, 3, lambda a, b: "{}"))
    monkeypatch.setattr(e, "_compute_memory_signal", lambda c: MemorySignal(0, 0.0))
    d, route = e._decide_reasoning_mode({"prompt": "x"}, "medium", _NI())
    assert d.mode is ReasoningMode.MULTI_AGENT


def test_config_fast_beats_car(monkeypatch):
    from neo.config import NeoConfig
    cfg = NeoConfig()
    cfg.reasoning_mode = "fast"
    e = _engine(FakeLM(lambda m: "{}"), config=cfg)
    monkeypatch.setattr(e, "_car_route_capability", lambda p: (True, 5, lambda a, b: "{}"))
    monkeypatch.setattr(e, "_compute_memory_signal", lambda c: MemorySignal(0, 0.0))
    d, _ = e._decide_reasoning_mode({"prompt": "x"}, "hard", _NI())
    assert d.mode is ReasoningMode.FAST


_PLAN = json.dumps({"approach": "a", "steps": [
    {"description": "x", "rationale": "r", "risk": "low"}]})
_CODE = json.dumps({"code": "def f(): return 1", "explanation": "e",
                    "edge_cases": [], "confidence": 0.9, "file_path": "f.py"})


def test_deliberate_runs_panel_over_fallback_lm():
    def responder(messages):
        s = messages[0]["content"]
        if "solution PLAN" in s:
            return _PLAN
        if "best plan" in s:
            return json.dumps({"winner": 0, "confidence": 0.8})
        if "Implement the given PLAN" in s:
            return _CODE
        if "adversarial code reviewer" in s:
            return json.dumps({"verdict": "ok", "severity": "none", "issues": []})
        if "fixing your code" in s:
            return _CODE
        return "{}"

    e = _engine(FakeLM(responder))
    plan, sims, code, result = e._deliberate({"prompt": "solve it"}, route_fn=None)
    assert result is not None and result.confidence > 0.0
    assert code and code[0].code_block == "def f(): return 1"
    assert result.provenance == "multi-agent"


def test_deliberate_failure_returns_none_for_fallback():
    # planner never emits JSON -> no plans -> confidence 0 -> caller falls back
    e = _engine(FakeLM(lambda m: "no json"))
    plan, sims, code, result = e._deliberate({"prompt": "x"}, route_fn=None)
    assert result is not None and result.confidence == 0.0
