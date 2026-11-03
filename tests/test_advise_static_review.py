"""Burndown#95: an advise-mode static review states its verdict, and Neo reports it.

Two real advise-mode static reviews (R13's ACCEPT with no patch, R08's
CHANGES_NEEDED with findings) named the file they reviewed with an empty diff.
Neo reported both as ``no_verifiable_change`` and told the operator to treat the
run as no answer, beside their own substantive verdicts. Nothing in the prose can
tell a verdict from "I could not see enough code", so advise mode now asks for an
explicit verdict block: only ACCEPT or CHANGES_NEEDED makes such a run a completed
analysis; no block, a malformed one or NO_VERDICT keeps the #199 floor.
"""

import json
from pathlib import Path

from neo.engine import NeoEngine
from neo.models import (
    CONFIDENCE_BASIS_ANALYSIS_ONLY,
    CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE,
    NeoInput,
    TaskType,
)
from neo.operating_mode import OperatingMode
from neo.schemas import SCHEMA_VERSION
from neo.subcommands import _interpret_confidence

FIXTURES = Path(__file__).parent / "fixtures" / "burndown95"


class FakeLM:
    provider = "fake"

    def __init__(self, response):
        self._response = response
        self.seen = []

    def generate(self, messages, **kwargs):
        self.seen.append(messages)
        return self._response

    def name(self):
        return "fake-lm"


def _block(kind, payload):
    return f"<<<NEO:SCHEMA=v3:KIND={kind}>>>\n{json.dumps(payload)}\n<<<END:{kind}>>>"


def _replay(fixture, mode, verdict=None, raw_verdict=None, description=None):
    """Run Neo on the model output a real review produced, in the given mode."""
    rec = json.loads((FIXTURES / fixture).read_text())
    plan = [dict(step, id=step.get("id") or f"ps_{i}", schema_version=SCHEMA_VERSION)
            for i, step in enumerate(rec["plan"], 1)]
    sims = [dict(t, n=i, schema_version=SCHEMA_VERSION)
            for i, t in enumerate(rec["simulation_traces"], 1)]
    code = [dict(c, code_block=c.get("code_block", ""), schema_version=SCHEMA_VERSION)
            for c in rec["code_suggestions"]]
    if description is not None:
        code = [dict(c, description=description) for c in code]
    blocks = [_block("plan", plan), _block("simulation", sims), _block("code", code)]
    if verdict is not None:
        blocks.append(_block("verdict", {"verdict": verdict, "reason": "replayed"}))
    if raw_verdict is not None:
        blocks.append(f"<<<NEO:SCHEMA=v3:KIND=verdict>>>\n{raw_verdict}\n<<<END:verdict>>>")
    lm = FakeLM("\n".join(blocks))
    engine = NeoEngine(lm_adapter=lm, enable_persistent_memory=False)
    out = engine.process(NeoInput(
        prompt="static review of the supplied delta; no patch",
        task_type=TaskType.EXPLANATION,
        operating_mode=mode,
    ))
    return out, lm


def _interpret(output):
    return _interpret_confidence(output.confidence, output.next_questions, output.plan,
                                 output.code_suggestions, output.confidence_basis)


def test_the_recorded_shapes_reproduce_the_contradiction():
    """The fixtures are the reported runs: both were filed as no verifiable change."""
    for name in ("r13_static_accept.json", "r08_static_changes_needed.json"):
        rec = json.loads((FIXTURES / name).read_text())
        assert rec["confidence_basis"] == CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE
        assert all(not c["unified_diff"] for c in rec["code_suggestions"])


def test_advise_mode_asks_for_a_verdict_and_other_modes_do_not():
    _, lm = _replay("r13_static_accept.json", OperatingMode.ADVISE, verdict="ACCEPT")
    assert any("KIND=verdict" in m["content"] for call in lm.seen for m in call)
    _, lm = _replay("r13_static_accept.json", OperatingMode.LEARN)
    assert not any("KIND=verdict" in m["content"] for call in lm.seen for m in call)


def test_a_stated_accept_is_a_completed_analysis_without_a_fabricated_patch():
    out, _ = _replay("r13_static_accept.json", OperatingMode.ADVISE, verdict="ACCEPT")
    assert out.confidence is None                      # no invented score
    assert out.confidence_basis == CONFIDENCE_BASIS_ANALYSIS_ONLY
    assert _interpret(out)["action"] == "ANALYSIS_ONLY"
    assert out.metadata["review_verdict"] == "ACCEPT"
    assert all(not c.unified_diff for c in out.code_suggestions)   # nothing fabricated
    said = repr(out.orchestrator)
    assert "treat this as no answer" not in said
    assert "I'd change" not in said                    # a finding is not a change


def test_a_stated_changes_needed_keeps_its_findings():
    out, _ = _replay("r08_static_changes_needed.json", OperatingMode.ADVISE,
                     verdict="CHANGES_NEEDED")
    assert out.confidence_basis == CONFIDENCE_BASIS_ANALYSIS_ONLY
    assert out.metadata["review_verdict"] == "CHANGES_NEEDED"
    assert len(out.next_questions) == 2                # the two simulation findings
    assert "Revision required" in out.code_suggestions[0].description


def test_without_a_verdict_the_same_shape_is_still_no_answer():
    """Prose is never read: a run that states no verdict proves nothing."""
    out, _ = _replay("r13_static_accept.json", OperatingMode.ADVISE)
    assert out.confidence_basis == CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE
    assert "review_verdict" not in out.metadata


def test_no_verdict_keeps_a_located_refusal_as_no_answer():
    out, _ = _replay("r13_static_accept.json", OperatingMode.ADVISE, verdict="NO_VERDICT",
                     description="I could not see enough code to review this file.")
    assert out.confidence_basis == CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE
    assert _interpret(out)["action"] == "NO_VERIFIABLE_CHANGE"
    assert out.metadata["review_verdict"] == "NO_VERDICT"


def test_a_malformed_verdict_block_counts_as_none():
    for raw in ('{"verdict": "accept"}', "ACCEPT", '{"verdict": "LGTM"}', "{"):
        out, _ = _replay("r13_static_accept.json", OperatingMode.ADVISE, raw_verdict=raw)
        assert out.confidence_basis == CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE, raw
        assert "review_verdict" not in out.metadata, raw


def test_outside_advise_a_verdict_changes_nothing():
    """#199 is unchanged where a patch could have been asked for."""
    out, _ = _replay("r13_static_accept.json", OperatingMode.LEARN, verdict="ACCEPT")
    assert out.confidence_basis == CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE
    assert _interpret(out)["action"] == "NO_VERIFIABLE_CHANGE"
