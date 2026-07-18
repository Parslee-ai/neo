"""Deterministic fact-provenance explanation tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from neo.memory.episodes import (
    LearningEpisode,
    LearningEpisodeStore,
    MemoryMutationEvidence,
    RetrievedFactEvidence,
    VerificationEvidence,
)
from neo.memory.explain import (
    FactLookupError,
    LearningEpisodeCatalog,
    explain_fact,
    resolve_fact,
)
from neo.memory.models import Fact, FactKind, FactMetadata, FactScope


def _fact(**overrides):
    values = {
        "id": "fact-verified-1234",
        "subject": "validate input",
        "body": "Validate input before processing.",
        "kind": FactKind.PATTERN,
        "scope": FactScope.PROJECT,
        "project_id": "project-1",
        "metadata": FactMetadata(
            confidence=0.4,
            success_count=2,
            provenance="observed",
            effectiveness_c=1.0,
            effectiveness_n=2,
        ),
        "supporting_episode_ids": ["support-1"],
        "contradicting_episode_ids": ["conflict-1"],
        "source_candidate_id": "candidate-1",
    }
    values.update(overrides)
    return Fact(**values)


def test_explanation_joins_support_conflict_retrieval_and_mutation(tmp_path):
    store = LearningEpisodeStore("project-1", base_dir=tmp_path)
    support = LearningEpisode(
        episode_id="support-1",
        session_id="session-1",
        task_id="task-1",
        project_id="project-1",
        objective="fix validation",
        repository_revision="good-revision",
        final_outcome="accepted",
    )
    support.verification.append(VerificationEvidence(
        verification_id="verify-1",
        kind="test",
        status="passed",
        tool_name="pytest",
        summary="focused tests passed",
    ))
    support.memory_mutations.append(MemoryMutationEvidence(
        mutation_id="mutation-1",
        operation="promote_repeated_episode_candidate",
        fact_id="fact-verified-1234",
        before_state={},
        after_state={"confidence": 0.6, "is_valid": True},
    ))
    store.save(support)

    conflict = LearningEpisode(
        episode_id="conflict-1",
        project_id="project-1",
        objective="retest validation",
        repository_revision="bad-revision",
        final_outcome="regression",
    )
    conflict.verification.append(VerificationEvidence(
        verification_id="verify-2",
        kind="later_regression",
        status="failed",
        tool_name="regression_reporter",
        summary="regression reproduced",
    ))
    conflict.memory_mutations.append(MemoryMutationEvidence(
        mutation_id="mutation-2",
        operation="demote_regressed_fact",
        fact_id="fact-verified-1234",
        before_state={"confidence": 0.6, "is_valid": True},
        after_state={"confidence": 0.4, "is_valid": True},
    ))
    store.save(conflict)

    retrieval = LearningEpisode(
        episode_id="retrieval-1",
        project_id="project-1",
        objective="add another endpoint",
        final_outcome="accepted",
    )
    retrieval.retrieved_facts.append(RetrievedFactEvidence(
        fact_id="fact-verified-1234",
        score=0.8125,
        included_in_context=True,
        used_in_reasoning=True,
        outcome_association="accepted",
    ))
    store.save(retrieval)

    explanation = explain_fact([_fact()], "fact-ver", episode_store=store)

    assert explanation["model_calls"] == 0
    assert explanation["fact"]["source_candidate_id"] == "candidate-1"
    assert explanation["supporting_evidence"][0]["episode_id"] == "support-1"
    assert explanation["supporting_evidence"][0]["verification"][0]["status"] == "passed"
    assert explanation["contradicting_evidence"][0]["final_outcome"] == "regression"
    assert explanation["retrieval_history"][0]["score"] == pytest.approx(0.8125)
    assert explanation["retrieval_history"][0]["used_in_reasoning"] is True
    assert [item["operation"] for item in explanation["mutation_history"]] == [
        "promote_repeated_episode_candidate",
        "demote_regressed_fact",
    ]
    assert explanation["mutation_history"][1]["before_state"]["confidence"] == 0.6
    assert explanation["mutation_history"][1]["after_state"]["confidence"] == 0.4


def test_explanation_includes_tombstone_and_supersession_chain():
    oldest = _fact(id="oldest", subject="oldest", supporting_episode_ids=[],
                   contradicting_episode_ids=[], superseded_by="middle", is_valid=False)
    middle = _fact(id="middle", subject="middle", supporting_episode_ids=[],
                   contradicting_episode_ids=[], supersedes="oldest",
                   superseded_by="newest", is_valid=False)
    newest = _fact(id="newest", subject="newest", supporting_episode_ids=[],
                   contradicting_episode_ids=[], supersedes="middle")

    explanation = explain_fact([oldest, middle, newest], "middle")

    assert explanation["fact"]["is_valid"] is False
    assert [item["id"] for item in explanation["supersession"]["previous"]] == ["oldest"]
    assert [item["id"] for item in explanation["supersession"]["replacements"]] == ["newest"]


def test_explanation_preserves_missing_evidence_reference_when_record_is_malformed(tmp_path):
    store = LearningEpisodeStore("project-1", base_dir=tmp_path)
    store.path.mkdir(parents=True)
    (store.path / "missing-episode.json").write_text("{not-json")
    fact = _fact(
        supporting_episode_ids=["missing-episode"],
        contradicting_episode_ids=[],
    )

    explanation = explain_fact([fact], fact.id, episode_store=store)

    assert explanation["supporting_evidence"] == [{
        "episode_id": "missing-episode",
        "relationship": "supports",
        "missing": True,
    }]
    assert list(store.path.glob("missing-episode.json.corrupt-*"))


def test_explanation_is_deterministic_for_identical_state(tmp_path):
    store = LearningEpisodeStore("project-1", base_dir=tmp_path)
    store.save(LearningEpisode(episode_id="support-1", project_id="project-1"))
    fact = _fact(contradicting_episode_ids=[])

    first = json.dumps(explain_fact([fact], fact.id, episode_store=store), sort_keys=True)
    second = json.dumps(explain_fact([fact], fact.id, episode_store=store), sort_keys=True)

    assert first == second


def test_global_fact_explanation_aggregates_support_from_multiple_projects(tmp_path):
    first = LearningEpisodeStore("project-a", base_dir=tmp_path)
    second = LearningEpisodeStore("project-b", base_dir=tmp_path)
    first.save(LearningEpisode(episode_id="support-a", project_id="project-a"))
    second.save(LearningEpisode(episode_id="support-b", project_id="project-b"))
    fact = _fact(
        scope=FactScope.GLOBAL,
        project_id="",
        supporting_episode_ids=["support-a", "support-b"],
        contradicting_episode_ids=[],
    )

    explanation = explain_fact(
        [fact],
        fact.id,
        episode_store=LearningEpisodeCatalog(tmp_path),
    )

    assert {
        item["episode_id"] for item in explanation["supporting_evidence"]
    } == {"support-a", "support-b"}
    assert not any(item.get("missing") for item in explanation["supporting_evidence"])


def test_fact_prefix_resolution_rejects_missing_and_ambiguous_ids():
    facts = [_fact(id="abc-one"), _fact(id="abc-two")]
    with pytest.raises(FactLookupError, match="ambiguous"):
        resolve_fact(facts, "abc")
    with pytest.raises(FactLookupError, match="not found"):
        resolve_fact(facts, "missing")


def test_memory_explain_cli_parser_is_wired(monkeypatch):
    from neo.cli import parse_args

    monkeypatch.setattr(sys, "argv", ["neo", "memory", "explain", "abc123", "--json"])
    args = parse_args()

    assert args.command == "memory"
    assert args.memory_action == "explain"
    assert args.fact_id == "abc123"
    assert args.json is True


def test_memory_explain_cli_emits_json_without_model_or_embedding(tmp_path):
    from neo.memory.scope import _compute_project_id

    repository = Path(__file__).resolve().parents[1]
    project_id = _compute_project_id(str(repository))
    fake_home = tmp_path / "cli-home"
    facts_dir = fake_home / ".neo" / "facts"
    facts_dir.mkdir(parents=True)
    fact = _fact(project_id=project_id, supporting_episode_ids=[],
                 contradicting_episode_ids=[])
    (facts_dir / f"facts_project_{project_id}.json").write_text(json.dumps({
        "version": "2.0",
        "facts": [fact.to_dict()],
    }))
    env = {
        **os.environ,
        "HOME": str(fake_home),
        "PYTHONPATH": str(repository / "src"),
        "NEO_SKIP_UPDATE_CHECK": "1",
        "NEO_OBSERVER_AUTOSTART": "0",
    }
    for key in ("NEO_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        env.pop(key, None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "neo",
            "memory",
            "explain",
            "fact-ver",
            "--json",
            "--cwd",
            str(repository),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["fact"]["id"] == fact.id
    assert payload["model_calls"] == 0
    assert "fastembed" not in result.stderr.lower()
