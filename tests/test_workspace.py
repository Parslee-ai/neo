"""Read-only host inspection and correlated recorder contracts."""

import json
from pathlib import Path

from neo import hook
from neo.memory.models import Fact, FactKind, FactScope
from neo.memory.scope import detect_org_and_project
from neo.workspace import workspace_explain, workspace_status


def test_inspector_is_project_local_and_does_not_mutate(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "work"
    root.mkdir()
    _, project = detect_org_and_project(str(root))
    facts_dir = tmp_path / ".neo" / "facts"
    facts_dir.mkdir(parents=True)
    fact = Fact(id="local1", subject="Empty input", body="Handle it", kind=FactKind.PATTERN,
                scope=FactScope.PROJECT, project_id=project, tags=["probation"])
    path = facts_dir / f"facts_project_{project}.json"
    path.write_text(json.dumps({"facts": [fact.to_dict()]}))
    (facts_dir / "facts_global.json").write_text('{"facts": [{"id": "BODY_MUST_NOT_PERSIST"}]}')
    corrupt = tmp_path / ".neo" / "episodes" / project / "broken.json"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_text("{broken")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    status = workspace_status(str(root))
    assert status["fact_count"] == status["probation_count"] == 1
    explanation = workspace_explain(str(root), "local1")
    assert explanation["fact"]["body"] == "Handle it"
    assert explanation["model_calls"] == 0
    assert explanation["cautions"]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_mod_and_legacy_record_one_edit_but_distinct_calls_survive(tmp_path):
    path = tmp_path / "host_events.jsonl"
    enriched = {"session_id": "s1", "tool_use_id": "call1", "turn_id": "t1", "event": "edit_applied", "file_path": "/work/a.py"}
    hook.append(enriched, path)
    hook.append({"session_id": "s1", "tool_use_id": "call1", "file_path": "/work/a.py"}, path)
    hook.append({**enriched, "tool_use_id": "call2"}, path)
    assert [json.loads(line)["tool_use_id"] for line in path.read_text().splitlines()] == ["call1", "call2"]


def test_recorder_retains_identity_and_drops_contents(tmp_path, monkeypatch):
    monkeypatch.setattr(hook, "_head", lambda cwd: "head")
    payload = {"tool_name": "NotebookEdit", "tool_input": {"notebook_path": "notes.ipynb", "new_source": "BODY_MUST_NOT_PERSIST"}, "cwd": str(tmp_path), "session_id": "s", "tool_use_id": "call", "turn_id": "turn", "agent_id": "agent", "invocation_id": "inv", "source": "claude-mod"}
    record = hook.build_record(payload)
    assert record["file_path"] == str(tmp_path / "notes.ipynb")
    assert record["turn_id"] == "turn"
    assert "BODY_MUST_NOT_PERSIST" not in json.dumps(record)
    life = hook.build_lifecycle_record({**payload, "event": "turn_completed", "answer": "BODY_MUST_NOT_PERSIST"})
    assert "file_path" not in life and "BODY_MUST_NOT_PERSIST" not in json.dumps(life)


def test_busy_ledger_lock_is_bounded_and_fail_open(tmp_path):
    import time

    import pytest

    fcntl = pytest.importorskip("fcntl")

    path = tmp_path / "host_events.jsonl"
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        started = time.monotonic()
        hook.append({"event": "turn_started"}, path)
        assert time.monotonic() - started < 1
        assert not path.exists()


def test_lifecycle_events_cannot_establish_edits_and_identity_is_diagnostic(tmp_path, monkeypatch):
    from neo.memory.outcomes import OutcomeTracker

    path = tmp_path / "host_events.jsonl"
    monkeypatch.setattr(hook, "HOOK_LEDGER", path)
    record = {"ts": 10, "file_path": str(tmp_path / "a.py"), "tool_use_id": "tool1", "turn_id": "turn1", "recent_invocation_id": "neo1"}
    path.write_text(json.dumps({**record, "event": "turn_completed"}) + "\n")
    tracker = OutcomeTracker(codebase_root=str(tmp_path), project_id="test")
    assert tracker._load_host_edit_events() == []
    path.write_text(json.dumps({**record, "event": "edit_applied"}) + "\n")
    assert tracker._load_host_edit_events() == [(10, "a.py", True)]
    assert tracker._host_observations[0][2]["relation"] == "observed_edit_not_causal_acceptance"
    assert tracker._host_observations[0][2]["turn_id"] == "turn1"


def test_json_context_gathering_requires_explicit_opt_in(tmp_path):
    import subprocess
    import sys

    (tmp_path / "app.py").write_text("def marker_function():\n    return 'SOURCE_MARKER_719'\n")
    for enabled in (False, True):
        payload = {"prompt": "Explain app.py marker_function", "working_directory": str(tmp_path), "auto_context": enabled}
        result = subprocess.run(
            [sys.executable, "-m", "neo", "--stdin-json", "--dry-run", "--no-memory"],
            input=json.dumps(payload), capture_output=True, text=True, timeout=240,
        )
        assert result.returncode == 0, result.stderr
        assert ("SOURCE_MARKER_719" in result.stderr) is enabled


def test_retrieval_evidence_preserves_scope_and_hard_citation():
    from types import SimpleNamespace

    from neo.memory.episodes import LearningEpisode, RetrievedFactEvidence
    from neo.workspace import memory_evidence_for

    fact = Fact(id="f", subject="Known rule", scope=FactScope.GLOBAL)
    engine = SimpleNamespace(
        fact_store=SimpleNamespace(entries=[fact]),
        current_learning_episode=LearningEpisode(retrieved_facts=[
            RetrievedFactEvidence(fact_id="f", included_in_context=False,
                                  used_in_reasoning=True, hard_cited=False),
        ]),
    )
    evidence = memory_evidence_for(engine)[0]
    assert evidence["scope"] == "global"
    assert evidence["used_in_reasoning"] is True
    assert evidence["hard_cited"] is False
    assert evidence["included_in_context"] is False


def test_host_identity_persists_as_diagnostic_without_promoting(tmp_path):
    from unittest.mock import patch

    from neo.memory.episodes import LearningEpisode, LearningEpisodeStore
    from neo.memory.outcomes import Outcome, OutcomeType
    from neo.memory.store import FactStore

    with patch.object(FactStore, "initialize"):
        store = FactStore(codebase_root=str(tmp_path))
    episodes = LearningEpisodeStore(store.project_id, base_dir=store._episodes_dir)
    episode = LearningEpisode(episode_id="advisory", final_outcome="pending")
    episodes.save(episode)
    observation = {"turn_id": "t", "tool_use_id": "call", "relation": "observed_edit_not_causal_acceptance"}
    outcome = Outcome(OutcomeType.UNVERIFIED, "a.py", learning_episode_id="advisory", host_observations=[observation])
    store._record_attributed_episode_outcome(outcome)
    saved = episodes.load("advisory")
    assert saved.outcome_details["host_observations"] == [observation]
    assert saved.final_outcome == "unverified"
    assert all(item.status == "skipped" for item in saved.verification)
    assert store.entries == []



def _write_facts(tmp_path, root, facts):
    _, project = detect_org_and_project(str(root))
    facts_dir = tmp_path / ".neo" / "facts"
    facts_dir.mkdir(parents=True, exist_ok=True)
    rows = [Fact(id=i, subject=subj, body=body, kind=FactKind.REVIEW,
                 scope=FactScope.PROJECT, project_id=project).to_dict()
            for i, subj, body in facts]
    (facts_dir / f"facts_project_{project}.json").write_text(json.dumps({"facts": rows}))


def test_status_names_pattern_facts_about_files_the_checkout_lacks(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "work"
    (root / "src").mkdir(parents=True)
    (root / "src" / "live.py").write_text("x = 1\n")
    _write_facts(tmp_path, root, [
        ("gone", "pattern: src/car/ir/verify.py", ""),
        ("live", "pattern: src/live.py", ""),
        # History records stay true after the files go: never checked.
        ("hist", "history:de3e40ae Remove old module", "Files: old/a.rs\n"),
        ("prose", "pattern: I/O", ""),
        ("clipped", "pattern: mean/va", ""),
        ("decision", "Sibling project relationships", ""),
    ])
    status = workspace_status(str(root))
    missing = {f["id"]: f["missing_file"] for f in status["facts"]}
    assert missing == {"gone": "src/car/ir/verify.py", "live": "", "hist": "",
                       "prose": "", "clipped": "", "decision": ""}
    assert status["missing_file_count"] == 1
    assert status["model_calls"] == 0


def test_paths_are_relative_to_the_repository_top_not_the_session_dir(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "work"
    (root / "src").mkdir(parents=True)
    (root / "src" / "live.py").write_text("x = 1\n")
    (root / "sub").mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _write_facts(tmp_path, root, [("live", "pattern: src/live.py", "")])
    # Same project id from a subdirectory; the path must still resolve.
    monkeypatch.setattr("neo.workspace.project_snapshot",
                        lambda r, _orig=__import__("neo.workspace", fromlist=["x"]).project_snapshot: _orig(str(root)))
    status = workspace_status(str(root / "sub"))
    assert status["facts"][0]["missing_file"] == ""


def test_paths_outside_the_checkout_and_unstatable_paths_are_never_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / "work"
    root.mkdir()
    _write_facts(tmp_path, root, [
        ("abs", "pattern: /nonexistent/x.py", ""),
        ("up", "pattern: ../outside/x.py", ""),
        ("win", "pattern: C:/x/y.py", ""),
        ("url", "pattern: https://x.io/a.html", ""),
        ("long", "pattern: src/" + "a" * 300 + ".py", ""),
    ])
    status = workspace_status(str(root))
    flagged = {f["id"]: f["missing_file"] for f in status["facts"]}
    assert flagged["abs"] == flagged["up"] == flagged["win"] == flagged["url"] == ""
    assert status["missing_file_count"] in (0, 1)  # the 300-char name may or may not stat


def test_recall_delivery_records_ids_only():
    life = hook.build_lifecycle_record({
        "event": "recall_delivered", "session_id": "s", "cwd": "/w",
        "fact_ids": ["a", "", 7, "b" * 100] + [f"x{i}" for i in range(20)],
        "text": "BODY_MUST_NOT_PERSIST",
    })
    assert life["fact_ids"][:2] == ["a", "b" * 64]
    assert len(life["fact_ids"]) == hook.MAX_RECALL_IDS
    assert "BODY_MUST_NOT_PERSIST" not in json.dumps(life)
    assert hook.build_lifecycle_record({"event": "recall_delivered", "session_id": "s"}) is None
    assert hook.build_lifecycle_record({"event": "recall_delivered", "fact_ids": ["a"]}) is None


def test_auto_runs_record_kind_and_outcome():
    life = hook.build_lifecycle_record({
        "event": "auto_finished", "kind": "review", "status": "prose",
        "session_id": "s", "invocation_id": "i", "cwd": "/w", "diff": "BODY_MUST_NOT_PERSIST",
    })
    assert life["kind"] == "review" and life["status"] == "prose"
    assert "BODY_MUST_NOT_PERSIST" not in json.dumps(life)


def test_a_failed_auto_run_records_why_and_how_long():
    life = hook.build_lifecycle_record({
        "event": "auto_finished", "kind": "review", "status": "failed",
        "reason": "timed out", "elapsed_ms": 90012.6, "session_id": "s",
    })
    assert life["reason"] == "timed out" and life["elapsed_ms"] == 90012
    for bad in ("90", -1, True, None):
        life = hook.build_lifecycle_record({"event": "auto_finished", "elapsed_ms": bad})
        assert "elapsed_ms" not in life
