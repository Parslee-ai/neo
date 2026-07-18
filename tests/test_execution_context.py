"""Goal-aware execution envelope, loop assessment, and safety tests."""

import json

from neo.execution_context import (
    AttemptContext,
    CallerRole,
    GoalSpec,
    IntentSpec,
    OutcomeContext,
    ProgressSignal,
    SuccessCriterion,
    TrajectoryContext,
    assess_loop,
    execution_fields_from_dict,
    resolve_execution_context,
)
from neo.memory.store import FactStore
from neo.models import CodeSuggestion, NeoInput


def test_explicit_goal_intent_and_criteria_remain_authoritative():
    context = resolve_execution_context(NeoInput(
        prompt="Investigate the remaining failures",
        goal=GoalSpec("All auth tests pass"),
        intent=IntentSpec("diagnose_failed_attempt", "Explain the stalled fix"),
        constraints=["Do not weaken tests"],
        success_criteria=[SuccessCriterion(
            type="command", command="pytest tests/auth", expected_exit_code=0,
        )],
        role=CallerRole.DIAGNOSTICIAN,
    ))

    assert context.goal.origin == "explicit"
    assert context.goal.confidence == 1.0
    assert context.intent.origin == "explicit"
    assert context.success_criteria[0].command == "pytest tests/auth"
    assert "Do not weaken tests" in context.retrieval_query()


def test_missing_goal_and_intent_are_provisional_and_expose_unknowns():
    context = resolve_execution_context(NeoInput(
        prompt="Tests still failing in auth/session_test.py",
    ))

    assert context.goal.origin == "inferred"
    assert 0.0 < context.goal.confidence < 1.0
    assert context.intent.origin == "inferred"
    assert context.unknowns
    assert "inferred" in context.prompt_section()


def test_inferred_confidences_are_honest_coarse_bands():
    """Inferred (non-explicit) confidences must come from a small honest tier
    set, not fabricated precision like 0.78/0.64 — a keyword heuristic can't
    produce a calibrated probability."""
    from neo.execution_context import (
        _CONF_KEYWORD, _CONF_NONE, _CONF_ROLE_DERIVED,
    )
    bands = {_CONF_ROLE_DERIVED, _CONF_KEYWORD, _CONF_NONE}
    for prompt in ("tests still failing", "got a crash", "refactor the parser"):
        ctx = resolve_execution_context(NeoInput(prompt=prompt))
        if ctx.goal.origin == "inferred":
            assert ctx.goal.confidence in bands
        if ctx.intent.origin == "inferred":
            assert ctx.intent.confidence in bands
    # A verifier role yields the deterministic role-derived tier, not a guess.
    verifier = resolve_execution_context(
        NeoInput(prompt="check this", role=CallerRole.VERIFIER)
    )
    if verifier.intent.origin == "inferred":
        assert verifier.intent.confidence == _CONF_ROLE_DERIVED


def test_identical_symptom_produces_goal_conditioned_retrieval_queries():
    symptom = "Error: database write timeout"
    restore = resolve_execution_context(NeoInput(
        prompt=symptom,
        goal=GoalSpec("Restore service immediately"),
    ))
    redesign = resolve_execution_context(NeoInput(
        prompt=symptom,
        goal=GoalSpec("Eliminate the scalability bottleneck"),
    ))

    assert restore.retrieval_query() != redesign.retrieval_query()
    assert "Restore service immediately" in restore.retrieval_query()
    assert "Eliminate the scalability bottleneck" in redesign.retrieval_query()


def test_wire_parser_derives_progress_direction_and_bounds_trajectory():
    fields = execution_fields_from_dict({
        "progress": {"metric": "failing_tests", "before": 11, "after": 3},
        "trajectory": {
            "iteration": 4,
            "max_iterations": 10,
            "attempts": [{"summary": "one"}, "invalid"],
        },
        "role": "critic",
    })

    assert fields["progress"].direction == "improved"
    assert fields["trajectory"].attempts == [{"summary": "one"}]
    assert fields["role"] is CallerRole.CRITIC


def test_loop_assessment_uses_observed_progress_not_model_confidence():
    improving = resolve_execution_context(NeoInput(
        prompt="Continue fixing auth",
        goal=GoalSpec("All auth tests pass"),
        success_criteria=[SuccessCriterion("command", "pytest tests/auth", 0)],
        attempt=AttemptContext("Changed expiry handling"),
        outcome=OutcomeContext("failed", summary="3 tests remain"),
        progress=ProgressSignal("failing_tests", 11, 3, "improved"),
    ))
    goal, strategy = assess_loop(improving)
    assert goal.status == "in_progress"
    assert strategy.decision == "continue"

    stalled = resolve_execution_context(NeoInput(
        prompt="Reassess auth",
        goal=GoalSpec("All auth tests pass"),
        attempt=AttemptContext("Changed expiry handling"),
        outcome=OutcomeContext("failed", summary="No change"),
        progress=ProgressSignal("failing_tests", 3, 3, "unchanged"),
    ))
    _, strategy = assess_loop(stalled)
    assert strategy.decision == "change_strategy"


def test_success_requires_explicit_criterion_and_iteration_limit_blocks():
    verified = resolve_execution_context(NeoInput(
        prompt="Assess checkout",
        goal=GoalSpec("Checkout tests pass"),
        success_criteria=[SuccessCriterion("command", "pytest checkout", 0)],
        outcome=OutcomeContext("passed", summary="exit 0"),
    ))
    goal, strategy = assess_loop(verified)
    assert goal.status == "satisfied"
    assert strategy.decision == "stop_success"

    exhausted = resolve_execution_context(NeoInput(
        prompt="Try again",
        trajectory=TrajectoryContext(iteration=10, max_iterations=10),
    ))
    goal, strategy = assess_loop(exhausted)
    assert goal.status == "blocked"
    assert strategy.decision == "stop_blocked"


def test_advisory_role_suppresses_unrequested_implementation():
    from neo.engine import NeoEngine

    engine = object.__new__(NeoEngine)
    engine.resolved_execution_context = resolve_execution_context(NeoInput(
        prompt="Critique this attempt",
        role=CallerRole.CRITIC,
        requested_output="next_action",
    ))
    suggestion = CodeSuggestion("src/a.py", "+x", "replace implementation", 0.9)

    assert engine._apply_role_boundary([suggestion]) == []
    engine.resolved_execution_context.requested_output = "patch"
    assert engine._apply_role_boundary([suggestion]) == [suggestion]


def test_attempt_outcome_fact_omits_inferred_goal_and_preserves_observation(tmp_path):
    store = FactStore(
        codebase_root=str(tmp_path),
        eager_init=False,
        facts_dir=tmp_path / "facts",
        episodes_dir=tmp_path / "episodes",
        emit_metrics=False,
    )
    context = resolve_execution_context(NeoInput(
        prompt="Why is this timeout happening?",
        attempt=AttemptContext("Added a retry"),
        outcome=OutcomeContext(
            "failed", summary="Duplicates appeared", side_effects=["duplicate writes"],
        ),
    )).to_dict()

    fact = store.persist_attempt_outcome(
        execution_context=context,
        learning_episode_id="episode-1",
        repository_revision="abc123",
    )

    assert fact is not None
    payload = json.loads(fact.body)
    assert payload["goal"] == ""
    assert payload["action"] == "Added a retry"
    assert payload["outcome"]["side_effects"] == ["duplicate writes"]
    assert "observed-outcome" in fact.tags
    assert fact.metadata.confidence == 0.7

    repeated = store.persist_attempt_outcome(
        execution_context=context,
        learning_episode_id="episode-2",
        repository_revision="def456",
    )
    assert repeated.id != fact.id
    assert fact.is_valid is True
    assert repeated.is_valid is True

    context["outcome"]["status"] = "unverified"
    unverified = store.persist_attempt_outcome(
        execution_context=context,
        learning_episode_id="episode-3",
        repository_revision="ghi789",
    )
    assert unverified.metadata.confidence == 0.3
