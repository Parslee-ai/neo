"""`collect_outcomes` reads the host's own record of what it edited.

The git-based detector answers "did anything in this repo change since T".
That misses a whole class of real acceptance, most sharply the one this file
leads with: `git diff --name-only HEAD` does not list UNTRACKED files, so a
suggestion to create a NEW file — which `suggestion_is_verifiable` explicitly
admits as legitimate — is invisible until someone commits it. The
`neo hook record` PostToolUse hook records the edit when it happens; this is
the consumer for that ledger.

Every test here runs against a DIRTY tree. Pristine-tree tests are what let the
per-session retention bug ship: a spotless working tree is the one state neo is
never actually invoked in.
"""

import json
import subprocess
import time
from pathlib import Path

import pytest

from neo.memory.outcomes import OutcomeTracker, OutcomeType

TICK = 2


def _git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True,
                   capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "foo.py").write_text("def f():\n    return 1\n")
    _git(root, "init", "-q", ".")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "T")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    # Ordinary working state: something unrelated is dirty, always.
    (root / "src" / "unrelated.py").write_text("noise = 1\n")
    time.sleep(TICK)
    return root


class _Suggestion:
    def __init__(self, file_path):
        self.file_path = file_path
        self.unified_diff = "--- /dev/null\n+++ b/src/new_mod.py\n@@\n+VALUE = 1\n"
        self.description = "add the module"
        self.confidence = 0.9
        self.suggestion_id = "sug-1"
        self.code_block = "VALUE = 1"


def _tracker(repo, project_id=None):
    if project_id is None:
        project_id = f"ledger-{repo.parent.name}-{repo.name}"
    return OutcomeTracker(codebase_root=str(repo), project_id=project_id)


def _ledger_path():
    from neo.hook import HOOK_LEDGER
    return HOOK_LEDGER


def _record(file_path, ts, *, cwd="/somewhere/else", head="deadbeef", ledger=None):
    """Append one hook-shaped record."""
    target = Path(ledger) if ledger else _ledger_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": ts, "tool": "Write", "file_path": str(file_path),
            "cwd": cwd, "head": head,
        }) + "\n")


def test_a_new_untracked_file_is_invisible_to_git_but_caught_by_the_ledger(repo):
    """The headline case, and the reason this consumer exists.

    Neo suggests creating `src/new_mod.py`. The user creates it and has not
    committed. `git diff --name-only HEAD` lists tracked modifications only, so
    the acceptance is undetectable by git while the ledger has it outright.
    """
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/new_mod.py")], "add a module",
                         {"src/new_mod.py": "fact-1"})
    time.sleep(TICK)

    # The user applies it: a NEW file, left untracked.
    (repo / "src" / "new_mod.py").write_text("VALUE = 1\n")

    # Establish the premise rather than assuming it: git genuinely cannot see it.
    assert "src/new_mod.py" not in tracker._get_working_tree_changes(), (
        "premise broken: git diff HEAD now reports untracked files"
    )

    _record(repo / "src" / "new_mod.py", time.time())

    outcomes, _ = tracker.detect_outcomes()
    assert any(
        o.outcome_type is OutcomeType.ACCEPTED and o.file_path == "src/new_mod.py"
        for o in outcomes
    ), "host-recorded edit did not produce an acceptance"


def test_attribution_is_by_path_not_by_the_recorded_cwd_or_head(repo):
    """`cwd`/`head` name where the HOST was launched, not the edited file's repo.

    Measured directly: an edit to a scratch repository, made from a Claude Code
    session rooted in the neo checkout, recorded neo's OWN head. Keying
    attribution on either field would drop the record; only the path can say
    which project an edit belongs to. The record below carries a deliberately
    foreign cwd and head.
    """
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/new_mod.py")], "add a module", {})
    time.sleep(TICK)
    (repo / "src" / "new_mod.py").write_text("VALUE = 1\n")
    _record(repo / "src" / "new_mod.py", time.time(),
            cwd="/Users/somebody/a/completely/different/repo", head="0" * 40)

    outcomes, _ = tracker.detect_outcomes()
    assert any(o.file_path == "src/new_mod.py" for o in outcomes)


def test_edits_outside_the_project_root_are_ignored(repo, tmp_path):
    """One ledger serves every project, so a foreign path must not attribute."""
    other = tmp_path / "other_repo"
    (other / "src").mkdir(parents=True)
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/new_mod.py")], "add a module", {})
    time.sleep(TICK)
    _record(other / "src" / "new_mod.py", time.time())

    events = tracker._load_host_edit_events()
    assert events == [], f"a foreign path attributed to this project: {events}"


def test_edits_older_than_the_suggestion_do_not_count(repo):
    """A file edited BEFORE the suggestion is not evidence the suggestion was
    applied — otherwise every prior edit to a suggested path is an acceptance."""
    tracker = _tracker(repo)
    _record(repo / "src" / "new_mod.py", time.time() - 3600)  # an hour before
    tracker.save_session([_Suggestion("src/new_mod.py")], "add a module", {})
    time.sleep(TICK)

    outcomes, _ = tracker.detect_outcomes()
    assert not any(
        o.outcome_type is OutcomeType.ACCEPTED and o.file_path == "src/new_mod.py"
        for o in outcomes
    )


def test_the_rotated_generation_is_read(repo):
    """Rotation moves the RECENT records into `.1` and leaves the active file
    nearly empty, so reading only the active file loses exactly the window this
    consumer exists to protect."""
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/new_mod.py")], "add a module", {})
    time.sleep(TICK)
    (repo / "src" / "new_mod.py").write_text("VALUE = 1\n")

    rotated = _ledger_path().with_name(_ledger_path().name + ".1")
    _record(repo / "src" / "new_mod.py", time.time(), ledger=rotated)

    outcomes, _ = tracker.detect_outcomes()
    assert any(o.file_path == "src/new_mod.py" for o in outcomes), (
        "records in the rotated generation were not read"
    )


def test_a_malformed_ledger_is_survivable(repo):
    """A torn final write is the expected corruption for an append-only file.
    Skip the bad line, keep the good ones, never raise."""
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/new_mod.py")], "add a module", {})
    time.sleep(TICK)
    (repo / "src" / "new_mod.py").write_text("VALUE = 1\n")

    ledger = _ledger_path()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write("not json at all\n")
        fh.write('{"ts": "not-a-number", "file_path": "/x"}\n')
        fh.write('{"no_file_path": true, "ts": 1}\n')
    _record(repo / "src" / "new_mod.py", time.time())
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write('{"ts": 1, "file_path": "/y", ')  # torn final write

    outcomes, _ = tracker.detect_outcomes()
    assert any(o.file_path == "src/new_mod.py" for o in outcomes)


def test_a_missing_ledger_is_not_an_error(repo):
    """The hook is opt-out and may never have run."""
    ledger = _ledger_path()
    if ledger.exists():
        ledger.unlink()
    tracker = _tracker(repo)
    assert tracker._load_host_edit_events() == []


def test_the_ledger_is_read_once_per_call_not_once_per_session(repo, monkeypatch):
    """Hot-path invariant, and a mistake this module has already made once.

    `_get_working_tree_changes` had to be hoisted out of the per-session loop
    after re-forking it measured 0.88s at 40 pending sessions. The ledger read
    is the same shape: retention means many pending sessions, and a
    multi-megabyte re-read per session would put a linear cost on every request.
    """
    tracker = _tracker(repo)
    for i in range(4):
        tracker.save_session([_Suggestion(f"src/mod_{i}.py")], f"add {i}", {})
        time.sleep(0.05)

    calls = []
    real = OutcomeTracker._load_host_edit_events
    monkeypatch.setattr(
        OutcomeTracker, "_load_host_edit_events",
        lambda self: (calls.append(1), real(self))[1],
    )
    tracker.detect_outcomes()
    assert len(calls) == 1, f"ledger read {len(calls)}x for 4 sessions"
