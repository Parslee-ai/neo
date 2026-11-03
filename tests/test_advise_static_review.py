"""Burndown#95: a completed static review in advise mode is an answer.

Two real advise-mode static reviews (R13's ACCEPT with no patch, R08's
CHANGES_NEEDED with findings) named the file they reviewed with an empty diff.
Neo reported both as ``no_verifiable_change`` and told the operator to treat the
run as no answer, beside their own substantive verdicts. In advise mode no patch
is asked for, so a finding filed against the file it read is an analysis; the
#199 empty patch outside advise mode stays the floor.
"""

import json
from pathlib import Path

from neo.engine import NeoEngine
from neo.models import (
    CONFIDENCE_BASIS_ANALYSIS_ONLY,
    CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE,
    CodeSuggestion,
    NeoInput,
    TaskType,
    suggestion_is_located_review_note,
)
from neo.operating_mode import OperatingMode
from neo.schemas import SCHEMA_VERSION
from neo.subcommands import _interpret_confidence

FIXTURES = Path(__file__).parent / "fixtures" / "burndown95"


class FakeLM:
    provider = "fake"

    def __init__(self, response):
        self._response = response

    def generate(self, messages, **kwargs):
        return self._response

    def name(self):
        return "fake-lm"


def _block(kind, payload):
    return f"<<<NEO:SCHEMA=v3:KIND={kind}>>>\n{json.dumps(payload)}\n<<<END:{kind}>>>"


def _replay(fixture, mode):
    """Run Neo on the model output a real review produced, in the given mode."""
    rec = json.loads((FIXTURES / fixture).read_text())
    plan = [dict(step, id=step.get("id") or f"ps_{i}", schema_version=SCHEMA_VERSION)
            for i, step in enumerate(rec["plan"], 1)]
    sims = [dict(t, n=i, schema_version=SCHEMA_VERSION)
            for i, t in enumerate(rec["simulation_traces"], 1)]
    code = [dict(c, code_block=c.get("code_block", ""), schema_version=SCHEMA_VERSION)
            for c in rec["code_suggestions"]]
    response = "\n".join([_block("plan", plan), _block("simulation", sims), _block("code", code)])
    engine = NeoEngine(lm_adapter=FakeLM(response), enable_persistent_memory=False)
    return engine.process(NeoInput(
        prompt="static review of the supplied delta; no patch",
        task_type=TaskType.EXPLANATION,
        operating_mode=mode,
    ))


def _interpret(output):
    return _interpret_confidence(output.confidence, output.next_questions, output.plan,
                                 output.code_suggestions, output.confidence_basis)


def test_both_recorded_shapes_reproduce_the_contradiction_on_the_old_reading():
    """The fixtures are the reported runs: both were filed as no verifiable change."""
    for name in ("r13_static_accept.json", "r08_static_changes_needed.json"):
        rec = json.loads((FIXTURES / name).read_text())
        assert rec["confidence_basis"] == CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE
        assert all(not c["unified_diff"] for c in rec["code_suggestions"])


def test_a_clean_static_accept_is_a_completed_analysis_without_a_fabricated_patch():
    out = _replay("r13_static_accept.json", OperatingMode.ADVISE)
    assert out.confidence is None                      # no invented score
    assert out.confidence_basis == CONFIDENCE_BASIS_ANALYSIS_ONLY
    assert _interpret(out)["action"] == "ANALYSIS_ONLY"
    assert all(not c.unified_diff for c in out.code_suggestions)   # nothing fabricated
    said = repr(out.orchestrator)
    assert "treat this as no answer" not in said and "no verifiable" not in said.lower()


def test_a_static_review_with_findings_keeps_them():
    out = _replay("r08_static_changes_needed.json", OperatingMode.ADVISE)
    assert out.confidence_basis == CONFIDENCE_BASIS_ANALYSIS_ONLY
    assert len(out.next_questions) == 2                # the two simulation findings
    assert "Revision required" in out.code_suggestions[0].description
    assert any(t.issues_found for t in out.simulation_traces) or out.next_questions


def test_outside_advise_the_same_shape_is_still_the_floor():
    """#199 is unchanged where a patch could have been asked for."""
    out = _replay("r13_static_accept.json", OperatingMode.LEARN)
    assert out.confidence_basis == CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE
    assert _interpret(out)["action"] == "NO_VERIFIABLE_CHANGE"


def test_an_empty_suggestion_with_no_finding_is_no_answer_even_in_advise():
    blank = CodeSuggestion(file_path="src/a.py", unified_diff="", description="  ",
                           confidence=0.9, tradeoffs=[])
    assert suggestion_is_located_review_note(blank) is False
    real = CodeSuggestion(file_path="src/a.py", unified_diff="--- a\n+++ b\n@@\n+x\n",
                          description="fix", confidence=0.9, tradeoffs=[])
    assert suggestion_is_located_review_note(real) is False
