"""Tests for CLI subcommand handlers."""

from unittest.mock import patch


def test_show_version_does_not_eager_initialize_fact_store(capsys):
    """Version display should read stored facts without startup ingestion."""
    from neo.config import NeoConfig
    from neo.memory.models import Fact, FactMetadata, FactKind, FactScope
    from neo.subcommands import show_version

    calls = {}

    class FakeFactStore:
        def __init__(self, codebase_root=None, config=None, eager_init=True):
            calls["eager_init"] = eager_init
            self.entries = [
                Fact(
                    subject="Stored pattern",
                    body="Loaded from disk.",
                    kind=FactKind.PATTERN,
                    scope=FactScope.PROJECT,
                    metadata=FactMetadata(confidence=0.8),
                )
            ]

        def memory_level(self):
            return 0.1

        def find_contributable(self):
            return []

    with patch.object(NeoConfig, "load", return_value=NeoConfig()), \
         patch("neo.memory.store.FactStore", FakeFactStore), \
         patch("neo.car_discovery.discover_car", side_effect=RuntimeError("skip car")):
        show_version("/tmp/project")

    assert calls["eager_init"] is False
    assert "neo " in capsys.readouterr().out


def test_citation_stats_aggregates_per_signal(capsys):
    """citation-stats sums per-signal counts and ignores other event types."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from neo.subcommands import _handle_citation_stats

    metrics = Path.home() / ".neo" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"ts": 1000, "event": "citation_survival", "retrieved": 10, "included": 8,
         "used": 2, "by_marker": 0, "by_self_report": 2, "by_overlap": 1,
         "by_overlap_only": 0, "model": "gpt-5.5"},
        {"ts": 2000, "event": "citation_survival", "retrieved": 5, "included": 5,
         "used": 1, "by_marker": 0, "by_self_report": 0, "by_overlap": 1,
         "by_overlap_only": 1, "model": "gpt-5.5"},
        {"ts": 3000, "event": "lm_call", "model": "other"},  # must be ignored
        '["citation_survival"]',  # valid JSON, non-object — must not crash
    ]
    metrics.write_text(
        "\n".join(e if isinstance(e, str) else json.dumps(e) for e in events) + "\n")

    _handle_citation_stats(SimpleNamespace(json=True, since=None))
    out = json.loads(capsys.readouterr().out)
    assert out["requests"] == 2
    assert out["included"] == 13
    assert out["used"] == 3
    assert out["by_self_report"] == 2
    assert out["by_overlap"] == 2
    assert out["by_overlap_only"] == 1  # the decision number
    assert out["by_marker"] == 0
    assert out["by_model"]["gpt-5.5"]["requests"] == 2


def test_citation_stats_since_filters_old_events(capsys):
    """--since excludes events older than the window."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from neo.subcommands import _handle_citation_stats

    metrics = Path.home() / ".neo" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps({
        "ts": 1000, "event": "citation_survival", "retrieved": 3, "included": 3,
        "used": 1, "by_marker": 0, "by_self_report": 1, "by_overlap": 0, "model": "m",
    }) + "\n")

    _handle_citation_stats(SimpleNamespace(json=True, since="1d"))  # ts=1000 is ancient
    out = json.loads(capsys.readouterr().out)
    assert out["requests"] == 0


def test_learning_stats_aggregates_ledger(capsys):
    """learning-stats sums promotions/rollbacks and candidate statuses from the
    episode ledger, and reports the loop as ACTIVE when facts move."""
    import json
    from types import SimpleNamespace
    from neo.memory.episodes import (
        LearningEpisode, LearningEpisodeStore, MemoryCandidateEvidence,
        MemoryMutationEvidence,
    )
    from neo.subcommands import _handle_learning_stats

    es = LearningEpisodeStore("proj")  # base_dir defaults to ~/.neo/episodes (fake home)
    ep1 = LearningEpisode(episode_id="e1", started_at=1000.0,
                          final_outcome="suggested_pending_downstream_outcome")
    ep1.memory_candidates.append(MemoryCandidateEvidence(
        candidate_id="c1", suggestion_id="s1", subject="x", body="y",
        kind="pattern", status="durable"))
    ep1.memory_mutations.append(MemoryMutationEvidence(
        mutation_id="m1", operation="promote_repeated_episode_candidate", fact_id="f1"))
    es.save(ep1)
    ep2 = LearningEpisode(episode_id="e2", started_at=2000.0, final_outcome="modified")
    ep2.memory_candidates.append(MemoryCandidateEvidence(
        candidate_id="c2", suggestion_id="s2", subject="x", body="y",
        kind="pattern", status="contradicted"))
    ep2.memory_mutations.append(MemoryMutationEvidence(
        mutation_id="m2", operation="rollback_contradicted_fact", fact_id="f1"))
    es.save(ep2)

    _handle_learning_stats(SimpleNamespace(json=True, since=None))
    out = json.loads(capsys.readouterr().out)
    assert out["episodes"] == 2
    assert out["promotions"] == 1 and out["promotions_project"] == 1
    assert out["rollbacks"] == 1
    assert out["candidate_status"]["durable"] == 1
    assert out["candidate_status"]["contradicted"] == 1
    assert out["interactive_loop_active"] is True


def test_learning_stats_cited_fact_credit_counts_as_active(capsys):
    """A ledger whose only mutation is credit_used_retrieved_fact is genuine
    fact-level learning and must NOT read as IDLE (the missed-op regression)."""
    import json
    from types import SimpleNamespace
    from neo.memory.episodes import (
        LearningEpisode, LearningEpisodeStore, MemoryMutationEvidence,
    )
    from neo.subcommands import _handle_learning_stats

    es = LearningEpisodeStore("proj")
    ep = LearningEpisode(episode_id="e1", started_at=1000.0, final_outcome="accepted")
    ep.memory_mutations.append(MemoryMutationEvidence(
        mutation_id="m1", operation="credit_used_retrieved_fact", fact_id="f1"))
    es.save(ep)

    _handle_learning_stats(SimpleNamespace(json=True, since=None))
    out = json.loads(capsys.readouterr().out)
    assert out["cited_fact_credits"] == 1
    assert out["reinforcements"] == 1
    assert out["interactive_loop_active"] is True


def test_learning_stats_idle_when_no_promotions(capsys):
    """Episodes recorded but no fact-level mutations -> loop reported IDLE."""
    import json
    from types import SimpleNamespace
    from neo.memory.episodes import (
        LearningEpisode, LearningEpisodeStore, MemoryCandidateEvidence,
    )
    from neo.subcommands import _handle_learning_stats

    es = LearningEpisodeStore("proj")
    ep = LearningEpisode(episode_id="e1", started_at=1000.0,
                         final_outcome="suggested_pending_downstream_outcome")
    ep.memory_candidates.append(MemoryCandidateEvidence(
        candidate_id="c1", suggestion_id="s1", subject="x", body="y",
        kind="pattern", status="observed_unverified"))
    es.save(ep)

    _handle_learning_stats(SimpleNamespace(json=True, since=None))
    out = json.loads(capsys.readouterr().out)
    assert out["episodes"] == 1
    assert out["promotions"] == 0 and out["rollbacks"] == 0
    assert out["interactive_loop_active"] is False
