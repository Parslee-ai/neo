"""Fail-closed completion proofs for the goal-aware execution envelope."""

from neo.execution_context import (
    OutcomeContext,
    ValidationGate,
    ValidationObservation,
    TrajectoryContext,
    assess_loop,
    assess_validation,
    execution_fields_from_dict,
    resolve_execution_context,
)
from neo.models import NeoInput


def test_every_required_gate_needs_its_own_compatible_observation():
    context = resolve_execution_context(NeoInput(
        prompt="prove both boundaries",
        validation_gates=[
            ValidationGate("off", "disabled mode"),
            ValidationGate("on", "enabled mode"),
        ],
        validation_observations=[
            ValidationObservation("off-ok", "off", "passed"),
            ValidationObservation("on-ok", "on", "passed"),
        ],
        outcome=OutcomeContext("success"),
    ))

    assessment = assess_validation(context)
    goal, strategy = assess_loop(context)

    assert assessment.required == assessment.passed == 2
    assert goal.status == "satisfied"
    assert strategy.decision == "stop_success"


def test_unknown_observation_does_not_satisfy_a_gate():
    context = resolve_execution_context(NeoInput(
        prompt="prove the boundary",
        validation_gates=[ValidationGate("real", "real boundary")],
        validation_observations=[
            ValidationObservation("wrong", "other", "passed"),
        ],
    ))

    assessment = assess_validation(context)
    assert assessment.passed == 0
    assert assessment.pending == 1
    assert assessment.unknown_observation_gate_ids == ["other"]


def test_duplicate_gate_ids_fail_closed_instead_of_collapsing_obligations():
    fields = execution_fields_from_dict({
        "validation_gates": [
            {"gate_id": "suite", "description": "disabled suite"},
            {"gate_id": "suite", "description": "enabled suite"},
        ],
        "validation_observations": [
            {"observation_id": "one", "gate_id": "suite", "status": "passed"},
        ],
    })
    context = resolve_execution_context(NeoInput(prompt="prove both", **fields))

    assessment = assess_validation(context)
    assert assessment.required == 2
    assert assessment.passed == 1
    assert assessment.pending == 1
    assert assessment.blocking_gate_ids[0].startswith("invalid-duplicate-suite")
    assert assess_loop(context)[1].decision != "stop_success"


def test_duplicate_observation_ids_are_unavailable_not_ambiguous_success():
    fields = execution_fields_from_dict({
        "validation_gates": [{"gate_id": "suite", "description": "suite"}],
        "validation_observations": [
            {"observation_id": "same", "gate_id": "suite", "status": "passed"},
            {"observation_id": "same", "gate_id": "suite", "status": "failed"},
        ],
    })
    context = resolve_execution_context(NeoInput(prompt="prove suite", **fields))

    assessment = assess_validation(context)
    assert assessment.passed == 0
    assert assessment.unavailable == 1
    assert assess_loop(context)[1].decision == "stop_blocked"


def test_expected_value_and_exit_code_are_both_enforced():
    context = resolve_execution_context(NeoInput(
        prompt="validate exact result",
        validation_gates=[ValidationGate(
            "exact", "exact result", expected_exit_code=0, expected_value=42,
        )],
        validation_observations=[ValidationObservation(
            "wrong", "exact", "passed", actual_exit_code=0, actual_value=41,
        )],
    ))

    assessment = assess_validation(context)
    assert assessment.failed == 1
    assert assess_loop(context)[1].decision == "change_strategy"


def test_invalid_status_is_normalized_to_unavailable():
    fields = execution_fields_from_dict({
        "validation_gates": [{"gate_id": "gate", "description": "gate"}],
        "validation_observations": [{
            "observation_id": "obs", "gate_id": "gate", "status": "probably",
        }],
    })
    context = resolve_execution_context(NeoInput(prompt="check", **fields))
    assessment = assess_validation(context)

    assert context.validation_observations[0].status == "unavailable"
    assert assessment.unavailable == 1
    assert assess_loop(context)[0].status == "unverifiable"


def test_iteration_exhaustion_blocks_even_when_the_last_gate_failed():
    context = resolve_execution_context(NeoInput(
        prompt="final attempt",
        validation_gates=[ValidationGate("suite", "suite")],
        validation_observations=[ValidationObservation(
            "suite-failed", "suite", "failed",
        )],
        trajectory=TrajectoryContext(iteration=3, max_iterations=3),
    ))

    goal, strategy = assess_loop(context)
    assert goal.status == "blocked"
    assert strategy.decision == "stop_blocked"
