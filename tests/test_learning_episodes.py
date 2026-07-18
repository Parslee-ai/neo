"""Evidence-ledger tests for versioned learning episodes."""

from __future__ import annotations

import json

from neo.engine import NeoEngine
from neo.memory.episodes import (
    EPISODE_SCHEMA_VERSION,
    LearningEpisode,
    LearningEpisodeStore,
)
from neo.memory.models import ContextResult, Fact
from neo.models import ContextFile, NeoInput


class _CombinedLM:
    provider = "test-provider"
    model = "test-model"

    def name(self):
        return "test-provider/test-model"

    def generate(self, messages, **kwargs):
        return """<<<NEO:SCHEMA=v3:KIND=plan>>>
[{"id":"ps_1","description":"change it","rationale":"requested","dependencies":[],"schema_version":"3"}]
<<<END:plan>>>
<<<NEO:SCHEMA=v3:KIND=simulation>>>
[{"n":1,"input_data":"x","expected_output":"y","reasoning_steps":["**NO_MODIFY**"],"issues_found":[],"schema_version":"3"}]
<<<END:simulation>>>
<<<NEO:SCHEMA=v3:KIND=code>>>
[{"file_path":"src/example.py","unified_diff":"+value = 1","code_block":"value = 1","description":"set value","confidence":0.8,"tradeoffs":[],"schema_version":"3"}]
<<<END:code>>>"""


def test_partial_legacy_record_loads_conservatively(tmp_path):
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    (store.path / "old.json").write_text(json.dumps({
        "episode_id": "old",
        "objective": "legacy task",
    }))

    episode = store.load("old")

    assert episode is not None
    assert episode.schema_version == EPISODE_SCHEMA_VERSION
    assert episode.final_outcome == "pending"
    assert episode.verification == []
    assert episode.operating_mode == "learn"
    assert episode.authority == {}


def test_legacy_mutation_without_state_snapshots_loads_conservatively(tmp_path):
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    (store.path / "old-mutation.json").write_text(json.dumps({
        "episode_id": "old-mutation",
        "memory_mutations": [{
            "mutation_id": "mutation-1",
            "operation": "legacy_fact_write",
            "fact_id": "fact-1",
        }],
    }))

    episode = store.load("old-mutation")

    assert episode is not None
    assert episode.memory_mutations[0].before_state == {}
    assert episode.memory_mutations[0].after_state == {}


def test_malformed_record_is_preserved_and_skipped(tmp_path):
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    target = store.path / "broken.json"
    target.write_text("{not-json")

    assert store.load("broken") is None
    assert not target.exists()
    assert list(store.path.glob("broken.json.corrupt-*"))


def test_store_is_bounded(tmp_path, monkeypatch):
    from neo.memory import episodes as episode_module

    monkeypatch.setattr(episode_module, "MAX_EPISODES_PER_PROJECT", 2)
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    for i in range(3):
        store.save(LearningEpisode(episode_id=f"ep-{i}", started_at=float(i)))

    assert len(list(store.path.glob("*.json"))) == 2


def test_engine_persists_trace_without_raw_source_or_verification_claim(tmp_path, monkeypatch):
    engine = NeoEngine(
        lm_adapter=_CombinedLM(),
        enable_persistent_memory=False,
        codebase_root=str(tmp_path),
    )
    monkeypatch.setattr(engine, "_car_route_capability", lambda prompt: (False, 0, None))
    monkeypatch.setattr(engine, "_run_static_checks", lambda suggestions, constraints=None: [])
    source = "SECRET_SOURCE_CONTENT = 'local-only'"

    output = engine.process(NeoInput(
        prompt="set the example value",
        context_files=[ContextFile(path="src/example.py", content=source)],
        working_directory=str(tmp_path),
    ))

    episode_id = output.metadata["learning_episode_id"]
    episode = engine.episode_store.load(episode_id)
    assert episode is not None
    assert episode.session_id and episode.task_id
    assert episode.provider == "test-provider"
    assert episode.model == "test-model"
    assert episode.reasoning_mode == "fast"
    assert episode.suggestions[0].suggestion_id
    assert episode.suggestions[0].code_sha256
    assert episode.verification[0].status == "skipped"
    assert episode.final_outcome == "suggested_pending_downstream_outcome"

    persisted = (engine.episode_store.path / f"{episode_id}.json").read_text()
    assert source not in persisted
    assert "value = 1" not in persisted


def test_future_schema_fails_safely(tmp_path):
    store = LearningEpisodeStore("project", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    (store.path / "future.json").write_text(json.dumps({
        "episode_id": "future",
        "schema_version": EPISODE_SCHEMA_VERSION + 1,
    }))

    assert store.load("future") is None


def test_legacy_static_check_kind_is_normalized():
    episode = LearningEpisode.from_dict({
        "verification": [{
            "verification_id": "v1",
            "kind": "static_check",
            "status": "passed",
        }],
    })

    assert episode.verification[0].kind == "neo_static"

    downstream = LearningEpisode.from_dict({
        "schema_version": 1,
        "verification": [{
            "verification_id": "v2",
            "kind": "downstream_outcome",
            "status": "failed",
        }],
    })
    assert downstream.verification[0].kind == "user_modification"


def test_verification_aggregate_is_fail_closed():
    from neo.memory.episodes import VerificationEvidence, aggregate_verification_status

    evidence = [
        VerificationEvidence("v1", "lint", "passed"),
        VerificationEvidence("v2", "test", "unavailable"),
    ]

    assert aggregate_verification_status(evidence) == "unavailable"
    evidence.append(VerificationEvidence("v3", "compile", "failed"))
    assert aggregate_verification_status(evidence) == "failed"


def test_retrieved_fact_score_and_context_inclusion_are_traced():
    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    fact = Fact(id="fact-1", subject="Convention", body="Use typed IDs")
    context = ContextResult(valid_facts=[fact], retrieval_scores={fact.id: 0.875})

    engine._capture_retrieval_context(context, included=False)
    engine._capture_retrieval_context(context, included=True)

    evidence = engine.current_learning_episode.retrieved_facts
    assert len(evidence) == 1
    assert evidence[0].fact_id == "fact-1"
    assert evidence[0].score == 0.875
    assert evidence[0].included_in_context is True


def test_only_explicit_fact_citations_are_marked_used():
    from neo.models import CodeSuggestion, PlanStep, SimulationTrace

    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    facts = [
        Fact(id="fact-used", subject="Convention", body="Use typed IDs"),
        Fact(id="fact-only-retrieved", subject="Other", body="Use UTC"),
    ]
    context = ContextResult(
        valid_facts=facts,
        retrieval_scores={fact.id: 0.8 for fact in facts},
    )
    engine._capture_retrieval_context(context, included=True)

    engine._capture_detectable_fact_use(
        [PlanStep(description="Apply convention", rationale="Use [fact:fact-used]")],
        [SimulationTrace("input", "output", [])],
        [CodeSuggestion("src/a.py", "", "Change IDs", 0.8)],
    )

    by_id = {item.fact_id: item for item in engine.current_learning_episode.retrieved_facts}
    assert by_id["fact-used"].used_in_reasoning is True
    assert by_id["fact-only-retrieved"].used_in_reasoning is False


def test_failed_verification_is_associated_only_with_used_retrieval(tmp_path):
    from neo.memory.episodes import RetrievedFactEvidence
    from neo.models import StaticCheckResult

    engine = NeoEngine(
        lm_adapter=_CombinedLM(),
        enable_persistent_memory=False,
        codebase_root=str(tmp_path),
    )
    episode = LearningEpisode(project_id=engine.episode_store.project_id)
    episode.retrieved_facts = [
        RetrievedFactEvidence("used", included_in_context=True, used_in_reasoning=True),
        RetrievedFactEvidence("unused", included_in_context=True, used_in_reasoning=False),
    ]
    engine.current_learning_episode = episode

    engine._complete_learning_episode(
        code_suggestions=[],
        static_checks=[StaticCheckResult(
            tool_name="pytest",
            diagnostics=[{"severity": "error"}],
            summary="failed",
            kind="test",
            status="failed",
        )],
        reasoning_fact=None,
        simulation_facts=[],
        metadata={},
    )

    assert episode.retrieved_facts[0].outcome_association == "failure"
    assert episode.retrieved_facts[1].outcome_association == ""


def test_objective_credentials_are_redacted(tmp_path, monkeypatch):
    engine = NeoEngine(
        lm_adapter=_CombinedLM(),
        enable_persistent_memory=False,
        codebase_root=str(tmp_path),
    )
    monkeypatch.setattr(engine, "_car_route_capability", lambda prompt: (False, 0, None))
    monkeypatch.setattr(engine, "_run_static_checks", lambda suggestions, constraints=None: [])
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"

    output = engine.process(NeoInput(prompt=f"Use token {secret}"))

    episode = engine.episode_store.load(output.metadata["learning_episode_id"])
    assert episode is not None
    assert secret not in episode.objective
    assert "[REDACTED]" in episode.objective
