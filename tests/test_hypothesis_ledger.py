"""Evidence-backed, episode-local hypothesis state."""

from neo.execution_context import execution_fields_from_dict, resolve_execution_context
from neo.memory.episodes import (
    EPISODE_SCHEMA_VERSION,
    HypothesisEvidence,
    LearningEpisode,
    ValidationGateEvidence,
    VerificationEvidence,
)
from neo.models import NeoInput


def _context(hypothesis, observation_status="passed"):
    fields = execution_fields_from_dict({
        "validation_gates": [{"gate_id": "probe", "description": "causal probe"}],
        "validation_observations": [{
            "observation_id": "probe-result",
            "gate_id": "probe",
            "status": observation_status,
        }],
        "hypotheses": [hypothesis],
    })
    return resolve_execution_context(NeoInput(prompt="diagnose", **fields))


def test_confirmed_hypothesis_requires_falsifier_and_passing_evidence():
    missing_falsifier = _context({
        "hypothesis_id": "h1",
        "statement": "duplicate injection causes the failure",
        "status": "confirmed",
        "supporting_observation_ids": ["probe-result"],
        "public_claim_safe": True,
    })
    failed_probe = _context({
        "hypothesis_id": "h1",
        "statement": "duplicate injection causes the failure",
        "status": "confirmed",
        "falsifying_test": "remove the second injection",
        "supporting_observation_ids": ["probe-result"],
        "public_claim_safe": True,
    }, observation_status="failed")

    assert missing_falsifier.hypotheses[0].status == "supported"
    assert missing_falsifier.hypotheses[0].public_claim_safe is False
    assert failed_probe.hypotheses[0].status == "candidate"
    assert failed_probe.hypotheses[0].public_claim_safe is False


def test_confirmed_hypothesis_can_be_public_safe_with_compatible_evidence():
    context = _context({
        "hypothesis_id": "h1",
        "statement": "duplicate injection causes the failure",
        "status": "confirmed",
        "falsifying_test": "remove the second injection",
        "supporting_observation_ids": ["probe-result"],
        "public_claim_safe": True,
    })

    assert context.hypotheses[0].status == "confirmed"
    assert context.hypotheses[0].public_claim_safe is True


def test_rejection_requires_a_failed_contradicting_observation():
    unsupported = _context({
        "hypothesis_id": "h1",
        "statement": "foreign facts cause the failure",
        "status": "rejected",
        "contradicting_observation_ids": ["probe-result"],
    })
    supported = _context({
        "hypothesis_id": "h1",
        "statement": "foreign facts cause the failure",
        "status": "rejected",
        "contradicting_observation_ids": ["probe-result"],
    }, observation_status="failed")

    assert unsupported.hypotheses[0].status == "candidate"
    assert supported.hypotheses[0].status == "rejected"


def test_rejected_hypothesis_cannot_reopen_without_new_evidence():
    context = resolve_execution_context(NeoInput(
        prompt="diagnose again",
        **execution_fields_from_dict({
            "hypotheses": [{
                "hypothesis_id": "h1",
                "statement": "foreign facts are causal",
                "prior_status": "rejected",
                "status": "candidate",
            }],
        }),
    ))

    assert context.hypotheses[0].status == "rejected"


def test_v4_episode_round_trip_preserves_proof_and_task_provenance():
    episode = LearningEpisode(
        session_id="session-1",
        goal_id="goal-1",
        task_id="task-2",
        parent_task_id="task-1",
        trace_id="trace-1",
        discovery_source="runtime_observation",
        repositories_touched=["neo", "car"],
        artifact_refs=["hashed-artifact"],
        validation_gates=[ValidationGateEvidence(
            gate_id="probe", description="causal probe",
        )],
        verification=[VerificationEvidence(
            verification_id="obs", kind="test", status="passed", gate_id="probe",
        )],
        hypotheses=[HypothesisEvidence(
            hypothesis_id="h1", statement="one cause", status="confirmed",
            falsifying_test="one probe", supporting_observation_ids=["obs"],
        )],
    )

    restored = LearningEpisode.from_dict(episode.to_dict())
    assert restored.schema_version == EPISODE_SCHEMA_VERSION == 4
    assert restored.parent_task_id == "task-1"
    assert restored.repositories_touched == ["neo", "car"]
    assert restored.validation_gates[0].gate_id == "probe"
    assert restored.verification[0].gate_id == "probe"
    assert restored.hypotheses[0].status == "confirmed"
