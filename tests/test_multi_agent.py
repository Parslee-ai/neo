"""Tests for the multi-agent deliberation orchestrator (with fake adapters)."""

import json

from neo.multi_agent import MultiAgentReasoner, _extract_json


class FakeAdapter:
    """Returns a canned response per role; records calls."""

    def __init__(self, label, script):
        self.label = label
        self._script = script          # callable(messages, call_index) -> str
        self.calls = 0

    def generate(self, messages, max_tokens=2048, temperature=0.7):
        out = self._script(messages, self.calls)
        self.calls += 1
        return out

    def name(self):
        return self.label


def _factory(scripts):
    """Build a role_adapter returning a distinct FakeAdapter per role."""
    adapters = {role: FakeAdapter(f"model-{role}", scr) for role, scr in scripts.items()}
    return lambda role: adapters[role], adapters


_PLAN = json.dumps({"approach": "a", "steps": [
    {"description": "do x", "rationale": "because", "risk": "low"}], "key_risks": []})
_CODE = json.dumps({"code": "def f(): return 1", "explanation": "impl",
                    "edge_cases": ["empty input"], "confidence": 0.9, "file_path": "f.py"})


def test_full_panel_flow_produces_engine_tuple():
    factory, adapters = _factory({
        "planner": lambda m, i: _PLAN,
        "judge": lambda m, i: json.dumps({"winner": 0, "confidence": 0.8}),
        "coder": lambda m, i: _CODE,
        "critic": lambda m, i: json.dumps({"issues": [], "verdict": "ok", "severity": "none"}),
    })
    r = MultiAgentReasoner(factory, k_plans=3).deliberate("solve it", context="ctx")
    plan, sims, code = r.as_engine_tuple()
    assert plan and code and sims
    assert code[0].code_block == "def f(): return 1"
    assert r.provenance == "multi-agent"
    assert adapters["planner"].calls == 3          # k candidate plans
    assert r.meta["n_plans"] == 3
    assert 0.0 < r.confidence <= 1.0


def test_distinct_models_are_tracked():
    factory, _ = _factory({
        "planner": lambda m, i: _PLAN,
        "judge": lambda m, i: json.dumps({"winner": 0, "confidence": 0.9}),
        "coder": lambda m, i: _CODE,
        "critic": lambda m, i: json.dumps({"verdict": "ok"}),
    })
    r = MultiAgentReasoner(factory, k_plans=2).deliberate("t")
    # critic and coder resolved to different model labels
    assert r.models_used["coder"] != r.models_used["critic"]
    assert r.meta["distinct_models"] >= 3


def test_repair_loop_runs_when_critic_says_revise():
    # critic: first call "revise", then "ok"
    def critic_script(m, i):
        return json.dumps({"issues": ["off-by-one"], "verdict": "revise", "severity": "major"}) if i == 0 \
            else json.dumps({"issues": [], "verdict": "ok", "severity": "none"})

    factory, adapters = _factory({
        "planner": lambda m, i: _PLAN,
        "judge": lambda m, i: json.dumps({"winner": 0, "confidence": 0.7}),
        "coder": lambda m, i: _CODE,
        "critic": critic_script,
    })
    r = MultiAgentReasoner(factory, k_plans=1, max_repair_rounds=1).deliberate("t")
    assert r.rounds == 1
    assert adapters["critic"].calls == 2           # initial + post-repair
    assert adapters["coder"].calls == 2            # initial code + 1 repair
    assert r.meta["critic_verdict"] == "ok"


def test_unrepaired_major_issue_lowers_confidence():
    factory, _ = _factory({
        "planner": lambda m, i: _PLAN,
        "judge": lambda m, i: json.dumps({"winner": 0, "confidence": 0.9}),
        "coder": lambda m, i: _CODE,
        "critic": lambda m, i: json.dumps({"issues": ["wrong"], "verdict": "revise", "severity": "major"}),
    })
    r = MultiAgentReasoner(factory, k_plans=1, max_repair_rounds=0).deliberate("t")
    assert r.meta["critic_verdict"] == "revise"
    assert r.confidence < 0.5                       # penalized for unresolved major issue


def test_low_plan_consensus_lowers_confidence():
    # two plans, judge barely prefers one (0.5) -> lower confidence than a clear win
    factory, _ = _factory({
        "planner": lambda m, i: _PLAN,
        "judge": lambda m, i: json.dumps({"winner": 0, "confidence": 0.5}),
        "coder": lambda m, i: _CODE,
        "critic": lambda m, i: json.dumps({"verdict": "ok", "severity": "none"}),
    })
    low = MultiAgentReasoner(factory, k_plans=2).deliberate("t")

    factory2, _ = _factory({
        "planner": lambda m, i: _PLAN,
        "judge": lambda m, i: json.dumps({"winner": 0, "confidence": 1.0}),
        "coder": lambda m, i: _CODE,
        "critic": lambda m, i: json.dumps({"verdict": "ok", "severity": "none"}),
    })
    high = MultiAgentReasoner(factory2, k_plans=2).deliberate("t")
    assert low.confidence < high.confidence


def test_no_parseable_plans_signals_failure():
    factory, _ = _factory({
        "planner": lambda m, i: "sorry, no JSON here",
        "judge": lambda m, i: "{}",
        "coder": lambda m, i: "{}",
        "critic": lambda m, i: "{}",
    })
    r = MultiAgentReasoner(factory, k_plans=2).deliberate("t")
    assert r.confidence == 0.0 and not r.code_suggestions
    assert "error" in r.meta


def test_extract_json_tolerates_fences_and_prose():
    assert _extract_json('```json\n{"a": 1}\n```')["a"] == 1
    assert _extract_json('here you go: {"b": 2} thanks')["b"] == 2
    assert _extract_json("no json at all") is None
