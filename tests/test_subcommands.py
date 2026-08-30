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

        @staticmethod
        def is_contribution_candidate(fact):
            return True

    with patch.object(NeoConfig, "load", return_value=NeoConfig()), \
         patch("neo.memory.store.FactStore", FakeFactStore), \
         patch("neo.car_discovery.discover_car", side_effect=RuntimeError("skip car")):
        show_version("/tmp/project")

    assert calls["eager_init"] is False
    assert "neo " in capsys.readouterr().out


def test_contribution_gap_names_only_the_gate_that_binds():
    """A fact already at full confidence must not be told to raise confidence.

    The status banner used to print a fixed "need 0.8 confidence + 3
    successes" for every stalled fact, which points the operator at a
    threshold that is already cleared — the same defect this repo bans for
    file-selection caps ("never blame a cap for an absence it did not
    cause").
    """
    from neo.memory.models import Fact, FactMetadata, FactKind, FactScope
    from neo.subcommands import _describe_contribution_gap

    def fact(confidence, successes):
        return Fact(
            subject="s", body="b", kind=FactKind.PATTERN, scope=FactScope.PROJECT,
            metadata=FactMetadata(confidence=confidence, success_count=successes),
        )

    successes_only = _describe_contribution_gap([fact(1.0, 2), fact(1.0, 1)])
    assert "confidence" not in successes_only
    assert "2 short on successes (best 2 of 3)" == successes_only

    confidence_only = _describe_contribution_gap([fact(0.7, 5)])
    assert "success" not in confidence_only
    assert "1 short on confidence (best 0.70 of 0.8)" == confidence_only

    both = _describe_contribution_gap([fact(0.7, 5), fact(1.0, 1)])
    assert "short on confidence" in both and "short on successes" in both

    # Never emits a dangling clause for a fact that clears both gates.
    assert _describe_contribution_gap([fact(1.0, 9)]) == "no gate short"


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


def _give_project_a_fact_store(project_id: str) -> None:
    """Create `facts_project_<id>.json` for a ledger project.

    learning-stats counts a project only when its fact store exists, because a
    promotion WRITES to that file and a project id with no such file cannot
    have received one. Production always satisfies this: `FactStore.initialize`
    creates the file on a brand-new project, before any episode can be
    recorded. A test that saves episodes without it is modelling a state that
    only hand-run drill state reaches.
    """
    from neo.memory.store import FACTS_DIR
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    (FACTS_DIR / f"facts_project_{project_id}.json").write_text(
        '{"facts": []}', encoding="utf-8"
    )


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

    _give_project_a_fact_store("proj")
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

    _give_project_a_fact_store("proj")
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

    _give_project_a_fact_store("proj")
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


def test_learning_stats_excludes_projects_with_no_fact_store(capsys):
    """A project id with no `facts_project_<id>.json` is excluded from every
    count, and the exclusion is REPORTED rather than silently applied.

    Hand-run drills leave exactly this shape. A live ledger held 50
    `testproj1234` episodes whose 19 `durable` candidates were the entirety of
    a reported "21 promoted -> loop ACTIVE" against zero real promotions.
    """
    import json
    from types import SimpleNamespace
    from neo.memory.episodes import (
        LearningEpisode, LearningEpisodeStore, MemoryCandidateEvidence,
        MemoryMutationEvidence,
    )
    from neo.subcommands import _handle_learning_stats

    # A drill project: episodes and a promotion, but no fact store behind it.
    drill = LearningEpisodeStore("testproj1234")
    ep = LearningEpisode(episode_id="d1", started_at=1000.0, final_outcome="accepted")
    ep.memory_candidates.append(MemoryCandidateEvidence(
        candidate_id="c1", suggestion_id="s1", subject="x", body="y",
        kind="pattern", status="durable"))
    ep.memory_mutations.append(MemoryMutationEvidence(
        mutation_id="m1", operation="promote_repeated_episode_candidate",
        fact_id="f1"))
    drill.save(ep)

    _handle_learning_stats(SimpleNamespace(json=True, since=None))
    out = json.loads(capsys.readouterr().out)
    assert out["episodes"] == 0
    assert out["promotions"] == 0
    assert out["candidate_status"] == {}
    assert out["interactive_loop_active"] is False
    assert out["excluded_projects_without_fact_store"] == [
        {"project_id": "testproj1234", "episodes": 1}
    ]


def test_learning_stats_reports_exclusion_when_ledger_is_only_drill_state(capsys):
    """The empty-ledger branch must still print the exclusion, or a ledger
    holding nothing BUT drill state reads as 'neo has never run' instead of
    'what is here is junk'."""
    from types import SimpleNamespace
    from neo.memory.episodes import LearningEpisode, LearningEpisodeStore
    from neo.subcommands import _handle_learning_stats

    LearningEpisodeStore("testproj1234").save(
        LearningEpisode(episode_id="d1", started_at=1000.0, final_outcome="accepted"))

    _handle_learning_stats(SimpleNamespace(json=False, since=None))
    printed = capsys.readouterr().out
    assert "no learning episodes recorded yet" in printed
    assert "testproj1234" in printed
    assert "no fact store" in printed


def test_learning_stats_keeps_the_unscoped_sentinel(capsys):
    """`unscoped` is the id `LearningEpisodeStore` assigns when a run resolves
    no project, so its episodes are real runs (engine errors land there) and
    must survive the fact-store filter that removes drill projects."""
    import json
    from types import SimpleNamespace
    from neo.memory.episodes import LearningEpisode, LearningEpisodeStore
    from neo.subcommands import _handle_learning_stats

    LearningEpisodeStore("").save(  # -> "unscoped"
        LearningEpisode(episode_id="u1", started_at=1000.0,
                        final_outcome="engine_error"))

    _handle_learning_stats(SimpleNamespace(json=True, since=None))
    out = json.loads(capsys.readouterr().out)
    assert out["episodes"] == 1
    assert out["final_outcomes"]["engine_error"] == 1
    assert out["excluded_projects_without_fact_store"] == []


def test_learning_stats_buckets_suggestions_four_ways(tmp_path, monkeypatch):
    """Collapsing these makes the number unactionable: a prompt that legitimately
    had no edit target is a usage property, a real code suggestion we failed to
    attribute is a bug, and a vanished root is neither."""
    import json as _json

    from neo import subcommands

    sessions = tmp_path / ".neo" / "sessions"
    sessions.mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "real.py").write_text("x = 1")

    (sessions / "session_a.json").write_text(_json.dumps({
        "codebase_root": str(repo),
        "suggestions": [
            {"file_path": "real.py", "suggested_code": "x = 2"},              # verifiable
            {"file_path": "/REVIEW_ONLY/no_code_change.md", "suggested_code": "z"},  # advisory
            {"file_path": "/planning/critique.md", "suggested_code": "z"},    # advisory
            {"file_path": "NO_MODIFY", "suggested_code": "z"},                # advisory (no suffix)
            {"file_path": "real.py"},                                         # advisory (no code)
            {"file_path": "gone/Service.cs", "suggested_code": "z"},          # unattributable
        ],
    }))
    monkeypatch.setattr(subcommands.Path, "home", staticmethod(lambda: tmp_path))
    assert subcommands._suggestion_verifiability() == {
        "verifiable": 1, "advisory": 4, "unattributable": 1, "root_unavailable": 0,
    }


def test_classify_does_not_widen_the_production_predicate(tmp_path):
    """The advisory rule is reporting-only. A real, unresolved SOURCE path must
    stay 'unattributable' — otherwise a genuine attribution bug hides as usage."""
    from neo import subcommands
    assert subcommands._classify_suggestion(
        "src/Thing.cs", True, str(tmp_path)) == "unattributable"
    assert subcommands._classify_suggestion(
        "notes/design.md", True, str(tmp_path)) == "advisory"


def test_learning_stats_verifiability_handles_missing_sessions(tmp_path, monkeypatch):
    from neo import subcommands
    monkeypatch.setattr(subcommands.Path, "home", staticmethod(lambda: tmp_path))
    assert subcommands._suggestion_verifiability() == {
        "verifiable": 0, "advisory": 0, "unattributable": 0, "root_unavailable": 0,
    }


def test_vanished_root_is_unmeasurable_not_unattributable(tmp_path):
    """Once a test needs the LIVE filesystem, a root that no longer exists makes
    it meaningless. Measured: 8 of 10 "unattributable" suggestions were this — 7
    from deleted Claude Code agent worktrees — burying the signal the bucket
    exists to carry."""
    from neo import subcommands
    gone = str(tmp_path / "deleted-worktree")
    assert subcommands._classify_suggestion(
        "api/Svc.cs", True, gone) == "root_unavailable"
    assert subcommands._classify_suggestion(
        "/api/Svc.cs", True, gone) == "root_unavailable"
    # No root recorded at all is equally unmeasurable, not advisory.
    assert subcommands._classify_suggestion(
        "api/Svc.cs", True, "") == "root_unavailable"
    # A root that DOES exist still classifies normally.
    assert subcommands._classify_suggestion(
        "api/Svc.cs", True, str(tmp_path)) == "unattributable"


def test_a_dead_root_makes_every_filesystem_answer_unavailable(tmp_path):
    """Only the empty-input test is decidable without the disk. The prose-suffix,
    sentinel and docs/ branches all stat, and under a dead root each degrades to
    "does not exist" and would silently return advisory — under-counting the
    integrity signal (measured 8 reported against 14 real) and inflating
    `measurable` by the difference."""
    from neo import subcommands
    gone = str(tmp_path / "deleted-worktree")
    # Decidable from the arguments alone: still advisory.
    assert subcommands._classify_suggestion("", False, gone) == "advisory"
    # Everything else is unknowable once the tree is gone. We cannot tell an
    # edit to an existing README from an invented one.
    assert subcommands._classify_suggestion("plan.md", True, gone) == "root_unavailable"
    assert subcommands._classify_suggestion("NO_MODIFY", True, gone) == "root_unavailable"
    assert subcommands._classify_suggestion("docs/x.py", True, gone) == "root_unavailable"
    assert subcommands._classify_suggestion("README.md", True, gone) == "root_unavailable"


def test_invented_non_code_path_is_advisory(tmp_path):
    """An unresolved path that could not be code is an advisory answer's invented
    target, not an edit we failed to find."""
    from neo import subcommands
    assert subcommands._classify_suggestion(
        "/review/wave2_residual_bugs.json", True, str(tmp_path)) == "advisory"
    # Same meaning without the leading slash -> same bucket. An earlier version
    # keyed on the slash and split these on nothing but LM formatting.
    assert subcommands._classify_suggestion(
        "review/wave2_residual_bugs.json", True, str(tmp_path)) == "advisory"


def test_unresolved_code_paths_stay_a_bug_signal(tmp_path):
    """The cases a leading-slash rule wrongly buried. Both are code, both are
    invisible to suggestion_is_verifiable, and both are exactly what the
    unattributable bucket exists to surface."""
    from neo import subcommands
    # Model emitted an unexpanded placeholder segment.
    assert subcommands._classify_suggestion(
        "/dotnet/src/<entitlements>/Store.cs", True, str(tmp_path)) == "unattributable"
    # Genuine new module in a directory that does not exist yet.
    assert subcommands._classify_suggestion(
        "/src/newpkg/handler.py", True, str(tmp_path)) == "unattributable"
    assert subcommands._classify_suggestion(
        "src/newpkg/handler.py", True, str(tmp_path)) == "unattributable"


def test_invented_review_prose_is_not_counted_verifiable(tmp_path):
    """A bare-slash doc at repo root normalizes to a name whose parent IS the
    root, so `suggestion_is_verifiable`'s plausible-new-file rule granted it
    "verifiable". Measured: 11 of 21 supposedly verifiable suggestions were
    review prose that does not exist, making the headline number wrong."""
    from neo import subcommands
    assert subcommands._classify_suggestion(
        "/ARCHITECTURAL_REVIEW.md", True, str(tmp_path)) == "advisory"
    assert subcommands._classify_suggestion(
        "REVIEW_TASK_1.md", True, str(tmp_path)) == "advisory"
    # But EDITING a doc that really exists is still a real edit target.
    (tmp_path / "README.md").write_text("hi")
    assert subcommands._classify_suggestion(
        "README.md", True, str(tmp_path)) == "verifiable"


def test_real_new_file_in_existing_dir_stays_verifiable(tmp_path):
    """Proposing a new file inside a directory that exists is a real suggestion
    and shows up in git log once committed."""
    from neo import subcommands
    (tmp_path / "config").mkdir()
    assert subcommands._classify_suggestion(
        "config/settings.json", True, str(tmp_path)) == "verifiable"
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    assert subcommands._classify_suggestion(
        "/src/app.py", True, str(tmp_path)) == "verifiable"


def _write_session(name: str, root: str, paths: list[str]) -> None:
    import json as _json
    from pathlib import Path as _Path
    sessions = _Path.home() / ".neo" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    (sessions / f"session_{name}.json").write_text(_json.dumps({
        "codebase_root": root,
        "suggestions": [{"file_path": p, "suggested_code": "x"} for p in paths],
    }))


def _record_idle_episode() -> None:
    """The bucket report is only reached once at least one COUNTED episode
    exists — and an episode is only counted when its project has a fact store,
    which production guarantees and a bare ledger write does not."""
    from neo.memory.episodes import LearningEpisode, LearningEpisodeStore
    _give_project_a_fact_store("proj")
    LearningEpisodeStore("proj").save(LearningEpisode(
        episode_id="e1", started_at=1000.0,
        final_outcome="suggested_pending_downstream_outcome"))


def test_one_dead_root_does_not_suppress_the_starved_verdict(capsys):
    """The regression both reviewers caught. Comparing the unmeasurable count
    against a content bucket made `root_unavailable > verifiable` true at 1 > 0,
    so a single stale worktree hid STARVED and claimed "most recorded
    suggestions" off a count of one."""
    from pathlib import Path as _Path
    from types import SimpleNamespace

    from neo.subcommands import _handle_learning_stats

    _record_idle_episode()
    live = _Path.home() / "repo"
    live.mkdir(parents=True, exist_ok=True)
    _write_session("live", str(live), ["plan.md", "notes.md", "summary.md"])
    _write_session("dead", str(_Path.home() / "gone-worktree"), ["api/Svc.cs"])

    _handle_learning_stats(SimpleNamespace(json=False, since=None))
    out = capsys.readouterr().out
    assert "STARVED" in out
    assert "UNMEASURABLE" not in out
    # And the integrity note still qualifies the reading.
    assert "1 of 4 recorded suggestion(s) could not be classified" in out


def test_unmeasurable_requires_every_suggestion_to_be_unclassifiable(capsys):
    from pathlib import Path as _Path
    from types import SimpleNamespace

    from neo.subcommands import _handle_learning_stats

    _record_idle_episode()
    # Includes a doc path on purpose: it previously slipped past the root check
    # into `advisory`, leaving measurable == 1 and silently defeating this
    # verdict on a corpus that was 100% dead-root.
    _write_session("dead", str(_Path.home() / "gone-worktree"),
                   ["api/Svc.cs", "b.py", "plan.md"])

    _handle_learning_stats(SimpleNamespace(json=False, since=None))
    out = capsys.readouterr().out
    assert "UNMEASURABLE" in out
    assert "STARVED" not in out


def test_real_extensionless_files_are_not_called_advisory(tmp_path):
    """Makefile/Dockerfile are genuine edit targets — only ALL-CAPS sentinels
    ("NO_MODIFY") get the advisory bucket."""
    from neo import subcommands
    (tmp_path / "Dockerfile").write_text("FROM scratch")
    assert subcommands._classify_suggestion(
        "Dockerfile", True, str(tmp_path)) == "verifiable"
    assert subcommands._classify_suggestion(
        "NO_MODIFY", True, str(tmp_path)) == "advisory"
    # An existing ALL-CAPS file is a real target, not a sentinel.
    (tmp_path / "LICENSE").write_text("MIT")
    assert subcommands._classify_suggestion(
        "LICENSE", True, str(tmp_path)) == "verifiable"
