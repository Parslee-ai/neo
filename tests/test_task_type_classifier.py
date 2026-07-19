"""Tests for prompt->TaskType classification (models.classify_task_type).

The CLI previously hardcoded TaskType.FEATURE for every plain-text prompt, which
maps to the non-promotable 'decision' candidate kind — so the interactive
evidence-learning loop could never mint a durable fact. classify_task_type routes
genuinely algorithmic/bugfix work to the promotable 'pattern' kind while keeping
feature/refactor/explanation on the intentionally non-promotable kinds.

The classifier is deliberately fail-safe: it only lands on a promotable kind when
a promotable signal scores STRICTLY highest — ties and no-signal both resolve to
a non-promotable kind, so incidental keywords never leak feature work into
durable-pattern eligibility.
"""

import pytest

from neo.models import TaskType, classify_task_type

# Mirror of engine.py kind_map: which classifications can auto-promote (pattern).
_KIND_MAP = {
    "algorithm": "pattern",
    "refactor": "architecture",
    "bugfix": "pattern",
    "feature": "decision",
    "explanation": "review",
}
_PROMOTABLE = {"pattern"}


@pytest.mark.parametrize("prompt,expected", [
    # Promotable intents -> pattern (a clear, dominant promotable signal).
    ("In dedupe.py the dedupe function is O(n^2). Rewrite dedupe to be O(n) using a set.",
     TaskType.ALGORITHM),
    ("optimize the query for performance", TaskType.ALGORITHM),
    ("reduce the complexity of this loop", TaskType.ALGORITHM),
    ("Fix the null-pointer crash in the request handler", TaskType.BUGFIX),
    ("the parser throws an exception on empty input", TaskType.BUGFIX),
    ("the test fails with a traceback", TaskType.BUGFIX),
    # Non-promotable intents.
    ("Explain what the observer daemon does", TaskType.EXPLANATION),
    ("why does this function return None sometimes?", TaskType.EXPLANATION),
    ("Add a new endpoint for fetching user profiles", TaskType.FEATURE),
    ("implement dark mode support", TaskType.FEATURE),
    ("refactor the auth module and rename the class", TaskType.REFACTOR),
    ("clean up and simplify this code", TaskType.REFACTOR),
])
def test_classifies_expected_type(prompt, expected):
    assert classify_task_type(prompt) == expected


def test_no_signal_falls_back_to_feature():
    """An unclassifiable prompt stays on the conservative (non-promotable) default
    rather than being forced into a promotable kind."""
    assert classify_task_type("help me with this code") == TaskType.FEATURE
    assert classify_task_type("do the thing") == TaskType.FEATURE
    assert classify_task_type("") == TaskType.FEATURE
    assert classify_task_type(None) == TaskType.FEATURE  # non-str guard


def test_promotable_intents_map_to_pattern_kind():
    """Clear algorithmic/bugfix prompts reach the promotable 'pattern' kind;
    feature/refactor/explanation do not."""
    for prompt in ("optimize this loop", "fix the crash"):
        assert _KIND_MAP[classify_task_type(prompt).value] in _PROMOTABLE
    for prompt in ("add a new feature", "refactor and rename this",
                   "explain this module"):
        assert _KIND_MAP[classify_task_type(prompt).value] not in _PROMOTABLE


def test_strictly_higher_score_wins():
    """A promotable type wins only when it out-SCORES feature, not merely ties."""
    # optimiz + faster (algorithm 2) > add (feature 1) -> ALGORITHM.
    assert classify_task_type("add an optimized faster path") == TaskType.ALGORITHM
    # fix + bug (bugfix 2) > create (feature 1) -> BUGFIX.
    assert classify_task_type("create a fix for the bug") == TaskType.BUGFIX


def test_ties_fail_safe_to_non_promotable():
    """A prompt that ties one promotable signal against a feature/refactor verb
    resolves to the NON-promotable kind — the fail-safe tie-break. These are the
    cases the reviewers flagged as over-promotion leaks."""
    # add (feature 1) tie faster (algorithm 1) -> FEATURE, not ALGORITHM.
    assert classify_task_type("implement a faster caching layer") == TaskType.FEATURE
    # add (feature 1) tie error (bugfix 1) -> FEATURE, not BUGFIX.
    assert classify_task_type("add error handling to the parser") == TaskType.FEATURE
    # clean up (refactor 1) tie failing (bugfix 1) -> REFACTOR, not BUGFIX.
    assert classify_task_type("clean up the failing tests") == TaskType.REFACTOR
    for prompt in ("implement a faster caching layer",
                   "add error handling to the parser",
                   "clean up the failing tests"):
        assert _KIND_MAP[classify_task_type(prompt).value] not in _PROMOTABLE


def test_explanation_ties_stay_non_promotable():
    """An explanation that ties a promotable signal (the most common dev-question
    shape) must resolve to EXPLANATION, not leak into a promotable kind — prose
    Q&A is not a durable code lesson. EXPLANATION sits ahead of BUGFIX/ALGORITHM
    in the tie-break precisely for this."""
    # Explanation verb TIES a single promotable signal -> stays EXPLANATION.
    assert classify_task_type("explain how optimize works") == TaskType.EXPLANATION
    assert classify_task_type("walk me through the optimized loop") == TaskType.EXPLANATION
    assert classify_task_type("why does the optimized query fail") == TaskType.EXPLANATION
    for prompt in ("explain how optimize works",
                   "walk me through the optimized loop",
                   "understand why it is faster"):
        assert _KIND_MAP[classify_task_type(prompt).value] not in _PROMOTABLE
    # A STRICTLY-higher promotable score still wins over a lone explanation verb
    # (this is intended: two promotable signals outweigh one explanation verb).
    assert classify_task_type("explain how to fix the crash") == TaskType.BUGFIX


@pytest.mark.parametrize("prompt", [
    "add a call to info(user) in the handler",   # info( must NOT trip Big-O
    "wire foo() and undo() into the new flow",   # foo(/undo(
    "the todo() helper needs a new owner",       # todo(
])
def test_lowercase_paren_does_not_trip_algorithm(prompt):
    """Big-O detection is case-sensitive: a lowercase ``o(`` inside ordinary call
    references (info(, foo(, undo(, todo()) must not flag ALGORITHM and leak these
    feature/refactor prompts into a promotable kind."""
    result = classify_task_type(prompt)
    assert result != TaskType.ALGORITHM
    assert _KIND_MAP[result.value] not in _PROMOTABLE


def test_uppercase_big_o_still_detected():
    """Genuine Big-O notation (uppercase) still signals ALGORITHM."""
    assert classify_task_type("reduce this from O(n^2) to O(n)") == TaskType.ALGORITHM


def test_error_trace_biases_bugfix():
    """A supplied error_trace is strong (but not absolute) evidence of a bugfix:
    it tips signal-less and lightly-feature prompts to BUGFIX, but a strongly
    dominant different intent still overrides."""
    trace = "Traceback (most recent call last):\n  ...\nValueError: boom"
    # Signal-less prompt + trace -> BUGFIX (not the FEATURE fallback).
    assert classify_task_type("please review this output", trace) == TaskType.BUGFIX
    # A realistic single-signal feature prompt + trace -> BUGFIX (trace tips it).
    assert classify_task_type("add a user settings page", trace) == TaskType.BUGFIX
    # Empty prompt + trace still reads as a debugging task.
    assert classify_task_type("", trace) == TaskType.BUGFIX
    # A genuine bugfix prompt is only reinforced.
    assert classify_task_type("fix the crash", trace) == TaskType.BUGFIX


def test_error_trace_not_absolute():
    """The trace bias is overridable: a strongly-dominant non-bugfix intent wins,
    so error_trace doesn't blindly force BUGFIX regardless of the prompt."""
    trace = "Traceback: ..."
    # FEATURE scores 3 (add/new/feature) > BUGFIX 2 (trace boost) -> FEATURE.
    assert classify_task_type("add a new feature", trace) == TaskType.FEATURE


def test_no_trace_is_unchanged():
    """Passing no error_trace (the plain-text CLI path) leaves classification
    identical to the single-arg form."""
    for prompt in ("optimize this loop", "add a settings page", "explain the flow",
                   "help me here"):
        assert classify_task_type(prompt) == classify_task_type(prompt, None)


def test_bugfix_symptoms_derived_from_shared_vocabulary():
    """The BUGFIX failure symptoms are the shared FAILURE_SIGNAL_KEYWORDS
    (execution_context) — proving a single source of truth, so adding a symptom
    word there propagates to this classifier and can't drift from _infer_intent."""
    from neo.execution_context import FAILURE_SIGNAL_KEYWORDS
    for kw in FAILURE_SIGNAL_KEYWORDS:
        # Each shared symptom word, alone, classifies as BUGFIX here.
        assert classify_task_type(f"the {kw} happened") == TaskType.BUGFIX


def test_deterministic():
    """Same prompt always classifies the same way (no randomness)."""
    prompt = "optimize the O(n^2) scan to O(n)"
    assert len({classify_task_type(prompt) for _ in range(5)}) == 1
