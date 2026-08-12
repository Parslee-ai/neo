"""#199: confidence 0.5 was a "no suggestions" sentinel sitting on the scale.

The reported inversion, observed 2026-08-11 in one session on one subject:

  * a planner-stage run with `unified_diff: ""` that explicitly declined to name
    findings scored its own self-reported **0.96** and rendered as
    READY_TO_IMPLEMENT;
  * the run that actually answered three sub-questions produced no
    `CodeSuggestion` at all, hit the `return 0.5` sentinel, and rendered as
    "Proceed with caution - some uncertainties remain".

The empty patch outranked the answer. The tests below reproduce that pair
end-to-end and assert the ordering is now the right way round, plus the two
mechanisms that get it there: the sentinel comes off the numeric scale, and a
suggestion naming a real file with no diff and no code block is excluded from
the average and from the early-exit gate.
"""

import json

from neo.engine import NeoEngine
from neo.models import (
    CONFIDENCE_BASIS_ANALYSIS_ONLY,
    CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE,
    CONFIDENCE_BASIS_SUGGESTIONS,
    CodeSuggestion,
    NeoInput,
    TaskType,
    confidence_action_rank,
    suggestion_is_scoreable,
)
from neo.schemas import SCHEMA_VERSION
from neo.subcommands import _interpret_confidence


class FakeLM:
    """Returns one canned structured response regardless of the prompt."""

    model = "fake"
    provider = "fake"

    def __init__(self, response):
        self._response = response

    def generate(self, messages, **kwargs):
        return self._response

    def name(self):
        return "fake-lm"


def _block(kind, payload):
    return f"<<<NEO:SCHEMA=v3:KIND={kind}>>>\n{json.dumps(payload)}\n<<<END:{kind}>>>"


def _response(code):
    """A well-formed run whose only variable is what it proposes."""
    plan = [{
        "id": "ps_1",
        "description": "Compare the refresh path against the equivalence rules",
        "rationale": "the audit asks whether the two paths agree",
        "dependencies": [],
        "risk": "low",
        "schema_version": SCHEMA_VERSION,
    }]
    sim = [{
        "n": 1,
        "input_data": "a stale entry",
        "expected_output": "refreshed once",
        "reasoning_steps": ["the refresh runs before the comparison"],
        "issues_found": [],
        "schema_version": SCHEMA_VERSION,
    }]
    return "\n".join([
        _block("plan", plan), _block("simulation", sim), _block("code", code),
    ])


# The run that declined: a real file path, an empty diff, and a high
# self-reported number attached to a patch it did not write.
_EMPTY_PATCH = [{
    "file_path": "src/Odi/RefreshService.cs",
    "unified_diff": "",
    "code_block": "",
    "description": (
        "No speculative patch is provided at planner stage because the supplied "
        "source excerpts omit portions needed to answer the property audit."
    ),
    "confidence": 0.96,
    "tradeoffs": [],
    "schema_version": SCHEMA_VERSION,
}]

# The run that answered: an analysis, no code change proposed.
_NO_SUGGESTIONS: list = []

# A genuine mid-confidence code change, for the "these must be distinguishable"
# assertion the issue asks for.
_MID_CONFIDENCE_PATCH = [{
    "file_path": "src/Odi/RefreshService.cs",
    "unified_diff": "--- a\n+++ b\n@@\n+    if (entry is null) return;\n",
    "code_block": "if (entry is null) return;",
    "description": "guard the null entry",
    "confidence": 0.55,
    "tradeoffs": [],
    "schema_version": SCHEMA_VERSION,
}]


def _run(code):
    engine = NeoEngine(
        lm_adapter=FakeLM(_response(code)), enable_persistent_memory=False
    )
    return engine.process(NeoInput(
        prompt="audit the refresh path for equivalence with the cached path",
        task_type=TaskType.EXPLANATION,
    ))


def _interpret(output):
    return _interpret_confidence(
        output.confidence,
        output.next_questions,
        output.plan,
        output.code_suggestions,
        output.confidence_basis,
    )


# ------------------------------------------------ the reported inversion


def test_the_correct_analysis_now_outranks_the_empty_patch():
    """The exact #199 scenario, both runs, ordered."""
    empty_patch = _run(_EMPTY_PATCH)
    analysis = _run(_NO_SUGGESTIONS)

    empty_action = _interpret(empty_patch)["action"]
    analysis_action = _interpret(analysis)["action"]

    assert confidence_action_rank(analysis_action) > confidence_action_rank(
        empty_action
    ), (analysis_action, empty_action)
    # And specifically: the empty patch no longer reads as ready to implement.
    assert empty_action == "NO_VERIFIABLE_CHANGE"
    assert analysis_action == "ANALYSIS_ONLY"


def test_the_empty_patch_no_longer_carries_its_self_reported_number():
    """0.96 was the model's opinion of a patch that does not exist."""
    output = _run(_EMPTY_PATCH)

    assert output.confidence is None
    assert output.confidence_basis == CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE
    assert output.metadata.get("early_exit") is not True


def test_the_analysis_run_reports_no_score_rather_than_the_sentinel():
    output = _run(_NO_SUGGESTIONS)

    assert output.confidence is None
    assert output.confidence_basis == CONFIDENCE_BASIS_ANALYSIS_ONLY


def test_an_analysis_run_and_a_mid_confidence_run_are_distinguishable():
    """The issue's third acceptance criterion.

    Both used to land in the 0.4-0.7 band and read as "some uncertainties
    remain" — one because it measured 0.55, the other because it measured
    nothing.
    """
    analysis = _run(_NO_SUGGESTIONS)
    mid = _run(_MID_CONFIDENCE_PATCH)

    assert analysis.confidence is None
    assert mid.confidence is not None
    assert analysis.confidence_basis != mid.confidence_basis
    assert mid.confidence_basis == CONFIDENCE_BASIS_SUGGESTIONS

    assert _interpret(analysis)["action"] == "ANALYSIS_ONLY"
    assert _interpret(mid)["action"] == "PROCEED_WITH_CAUTION"
    assert "uncertainties remain" not in _interpret(analysis)["message"]


def test_the_empty_patch_run_says_so_in_its_cautions():
    """A host is told never to drop a caution; this is one it must not."""
    output = _run(_EMPTY_PATCH)
    joined = " ".join(output.orchestrator.cautions)
    assert "no diff and no code" in joined


def test_no_number_is_invented_in_the_orchestrator_summary():
    for code in (_EMPTY_PATCH, _NO_SUGGESTIONS):
        summary = _run(code).orchestrator.summary
        assert "No confidence score" in summary
        assert "0.96" not in summary
        assert "0.50" not in summary


# ------------------------------------------------------- the mechanisms


def _suggestion(path, diff="", code="", confidence=0.9):
    return CodeSuggestion(
        file_path=path,
        unified_diff=diff,
        description="",
        confidence=confidence,
        code_block=code,
    )


def _engine():
    return NeoEngine(lm_adapter=FakeLM(""), enable_persistent_memory=False)


def test_a_real_path_with_no_change_is_not_scoreable():
    assert suggestion_is_scoreable(_suggestion("src/a.py")) is False
    assert suggestion_is_scoreable(_suggestion("src/a.py", diff="--- a\n+++ b\n"))
    assert suggestion_is_scoreable(_suggestion("src/a.py", code="return 1"))


def test_the_schema_review_markers_are_not_scoreable():
    """`/` and `N/A` with an empty diff are the schema's analysis placeholder."""
    for path in ("/", "N/A", "n/a", ""):
        assert suggestion_is_scoreable(_suggestion(path)) is False


def test_a_review_marker_run_reports_analysis_only_not_no_verifiable_change():
    """Declining correctly is not the same failure as claiming a file."""
    confidence, basis = _engine()._calculate_confidence(
        [], [], [_suggestion("/", confidence=0.96)], []
    )
    assert confidence is None
    assert basis == CONFIDENCE_BASIS_ANALYSIS_ONLY


def test_an_empty_change_is_dropped_from_the_average_not_averaged_in():
    """Mixed run: the real patch decides the number on its own."""
    confidence, basis = _engine()._calculate_confidence(
        [],
        [],
        [
            _suggestion("src/a.py", diff="--- a\n+++ b\n", confidence=0.60),
            _suggestion("src/b.py", confidence=0.96),
        ],
        [],
    )
    assert basis == CONFIDENCE_BASIS_SUGGESTIONS
    assert confidence == 0.60


def test_no_suggestions_at_all_no_longer_returns_the_sentinel():
    confidence, basis = _engine()._calculate_confidence([], [], [], [])
    assert confidence is None
    assert basis == CONFIDENCE_BASIS_ANALYSIS_ONLY


# ---------------------------------------------------- the reported ladder


def test_no_verifiable_change_is_the_floor_of_the_trust_ladder():
    """The one relation the fix depends on."""
    floor = confidence_action_rank("NO_VERIFIABLE_CHANGE")
    for action in (
        "GATHER_MORE_DATA",
        "PROCEED_WITH_CAUTION",
        "ANALYSIS_ONLY",
        "READY_TO_IMPLEMENT",
    ):
        assert confidence_action_rank(action) > floor


def test_an_unrecognized_action_ranks_below_every_known_one():
    """A verdict this build cannot place is not evidence anything is safe."""
    assert confidence_action_rank("SHIP_IT") < confidence_action_rank(
        "NO_VERIFIABLE_CHANGE"
    )


def test_the_interpretation_names_the_basis_and_the_null_band():
    interpretation = _interpret_confidence(
        None, [], [], [], CONFIDENCE_BASIS_ANALYSIS_ONLY
    )
    assert interpretation["confidence_basis"] == CONFIDENCE_BASIS_ANALYSIS_ONLY
    assert "null" in interpretation["confidence_scale"]
    assert "Absence of a score is not a low score" in interpretation["note"]


def test_the_numeric_bands_are_unchanged_for_scored_runs():
    """The fix takes the sentinel off the scale; it does not retune the scale."""
    assert _interpret_confidence(0.85, [], [], [])["action"] == "READY_TO_IMPLEMENT"
    assert _interpret_confidence(0.55, [], [], [])["action"] == "PROCEED_WITH_CAUTION"
    assert _interpret_confidence(0.20, [], [], [])["action"] == "GATHER_MORE_DATA"


# --------------------------------------------------------- wire contract


def test_a_null_confidence_serializes_on_both_transports():
    """`--json` consumers and the CAR artifact see the same absent value."""
    from neo.car_tool_schema import neo_output_to_dict

    output = _run(_NO_SUGGESTIONS)
    payload = neo_output_to_dict(output)

    assert payload["confidence"] is None
    assert payload["confidence_basis"] == CONFIDENCE_BASIS_ANALYSIS_ONLY
    assert json.loads(json.dumps(payload))["confidence"] is None
