"""Acceptance made in ANOTHER checkout of the repository still counts (#254).

A suggestion is often applied somewhere other than where Neo was asked: a CAR
Lattice node answers from the main checkout while the asking agent works in a
linked worktree on its own branch. Detection used to read only the main
checkout's HEAD and working tree, so that acceptance was invisible until merged;
and promotion's distinct-revision gate compared HEAD at ask time, which on an
idle main checkout is one value forever. It now compares the revision each
change was APPLIED ON TOP OF — and the straddle and fan-out cases below are why
that, not the landing commit, is the thing to compare.

Every test runs against real git repositories with real linked worktrees, and
the main checkout is left DIRTY with something unrelated, as it is in use.
"""

import subprocess
import time

import pytest

from neo.memory.outcomes import OutcomeTracker, OutcomeType

TICK = 2
SUGGESTED_DIFF = "--- a/src/foo.py\n+++ b/src/foo.py\n@@\n-    return 1\n+    return 2\n"
SECOND_DIFF = "--- a/src/foo.py\n+++ b/src/foo.py\n@@\n-    return 2\n+    return 3\n"


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "foo.py").write_text("def f():\n    return 1\n")
    (root / "src" / "bar.py").write_text("def g():\n    return 1\n")
    _git(root, "init", "-q", "-b", "main", ".")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "T")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    (root / "notes.txt").write_text("unrelated\n")
    # `git log --since` is second-granular: a suggestion recorded in the same
    # second as `init` would see that commit as a change made after it.
    time.sleep(TICK)
    return root


class _Suggestion:
    def __init__(self, file_path="src/foo.py", diff=SUGGESTED_DIFF):
        self.file_path = file_path
        self.unified_diff = diff
        self.description = "return 2"
        self.confidence = 0.9
        self.suggestion_id = "sug-1"
        self.code_block = ""


def _suggest(repo, request, diff=SUGGESTED_DIFF):
    tracker = OutcomeTracker(codebase_root=str(repo), project_id=f"wt-{request.node.name}")
    tracker.save_session([_Suggestion(diff=diff)], "make f return 2", {})
    time.sleep(TICK)
    return tracker


def _worktree(repo, path, branch):
    _git(repo, "worktree", "add", "-q", "-b", branch, str(path))
    return path


def _apply(checkout, value=2):
    (checkout / "src" / "foo.py").write_text(f"def f():\n    return {value}\n")


def _accepted(outcomes):
    return [o for o in outcomes if o.outcome_type == OutcomeType.ACCEPTED]


def test_a_commit_on_a_linked_worktree_branch_is_an_acceptance(repo, tmp_path, request):
    tracker = _suggest(repo, request)
    wt = _worktree(repo, tmp_path / "agent-wt", "agent")
    base = _git(wt, "rev-parse", "HEAD")
    _apply(wt)
    _git(wt, "commit", "-qam", "apply")

    outcomes, _ = tracker.detect_outcomes()

    accepted = _accepted(outcomes)
    assert [o.file_path for o in accepted] == ["src/foo.py"], outcomes
    assert accepted[0].applied_on_revision == base


def test_an_uncommitted_edit_in_a_linked_worktree_is_an_acceptance(repo, tmp_path, request):
    tracker = _suggest(repo, request)
    wt = _worktree(repo, tmp_path / "agent-wt", "agent")
    _apply(wt)

    outcomes, _ = tracker.detect_outcomes()

    accepted = _accepted(outcomes)
    assert [o.file_path for o in accepted] == ["src/foo.py"], outcomes
    # Uncommitted: applied on top of HEAD of the checkout holding it.
    assert accepted[0].applied_on_revision == _git(wt, "rev-parse", "HEAD")


def test_work_in_progress_from_before_the_suggestion_does_not_resolve_it(
        repo, tmp_path, request):
    """An agent keeps its work uncommitted, often in the very file it asks
    about. That edit predates the suggestion and is not a response to it;
    resolving on it would record MODIFIED and drop the real acceptance."""
    wt = _worktree(repo, tmp_path / "agent-wt", "agent")
    (wt / "src" / "foo.py").write_text("def f():\n    return 1  # wip\n")
    time.sleep(TICK)
    tracker = _suggest(repo, request)

    outcomes, _ = tracker.detect_outcomes()

    assert [o for o in outcomes if o.file_path == "src/foo.py"] == [], outcomes
    _apply(wt)  # now the suggestion is applied
    outcomes, _ = tracker.detect_outcomes()
    assert [o.file_path for o in _accepted(outcomes)] == ["src/foo.py"], outcomes


def test_another_checkouts_work_is_not_read_as_independent_change(repo, tmp_path, request):
    """The wider evidence resolves Neo's own suggestions and nothing else.
    Otherwise every agent worktree's edits become INDEPENDENT candidates."""
    tracker = _suggest(repo, request)
    wt = _worktree(repo, tmp_path / "agent-wt", "agent")
    (wt / "src" / "bar.py").write_text("def g():\n    return 99\n")
    _git(wt, "commit", "-qam", "unrelated work")
    (wt / "src" / "baz.py").write_text("x = 1\n")  # dirty, untracked
    (repo / "src" / "foo.py").write_text("def f():\n    return 2\n")  # the acceptance

    diffed = []
    real_diff = tracker._get_file_diff_since

    def spy(path, ts, **kw):
        diffed.append(path)
        return real_diff(path, ts, **kw)

    tracker._get_file_diff_since = spy
    outcomes, _ = tracker.detect_outcomes()

    assert not [o for o in outcomes if o.outcome_type == OutcomeType.INDEPENDENT
                and o.file_path in {"src/bar.py", "src/baz.py"}], outcomes
    # Not even examined: each examined path costs git forks, and an agent
    # worktree can hold hundreds of changed files.
    assert set(diffed) == {"src/foo.py"}, diffed
    assert [o.file_path for o in _accepted(outcomes)] == ["src/foo.py"]


def test_a_fetched_remote_branch_is_not_an_acceptance(repo, request):
    """`--branches`, never `--all`: a teammate's fetched commit is not this
    user applying a suggestion."""
    tracker = _suggest(repo, request)
    _git(repo, "checkout", "-q", "-b", "teammate")
    _apply(repo)
    _git(repo, "commit", "-qam", "teammate's change")
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "update-ref", "refs/remotes/origin/teammate", sha)
    _git(repo, "branch", "-q", "-D", "teammate")

    outcomes, _ = tracker.detect_outcomes()

    assert _accepted(outcomes) == [], outcomes


def test_a_main_checkout_commit_is_applied_on_its_parent(repo, request):
    """The single-player path keeps its old answer: applied right away in the
    checkout Neo ran in, the base IS HEAD when the advice was asked for."""
    tracker = _suggest(repo, request)
    asked_at = _git(repo, "rev-parse", "HEAD")
    _apply(repo)
    _git(repo, "commit", "-qm", "apply", "--", "src/foo.py")

    outcomes, _ = tracker.detect_outcomes()

    accepted = _accepted(outcomes)
    assert accepted and accepted[0].applied_on_revision == asked_at


def test_one_sitting_straddling_a_commit_reads_as_one_base(repo, request):
    """Linus's sequence: ask at H and apply; a later run sees it DIRTY; ask
    again and commit as C; the next run sees it COMMITTED. Landing revisions
    (H, C) differ and would promote one operator's one sitting. Both were
    applied on H."""
    first = _suggest(repo, request)
    _apply(repo)
    seen_dirty = _accepted(first.detect_outcomes()[0])

    second = OutcomeTracker(codebase_root=str(repo),
                            project_id=f"wt2-{request.node.name}")
    second.save_session([_Suggestion()], "make f return 2", {})
    time.sleep(TICK)
    _git(repo, "commit", "-qm", "apply", "--", "src/foo.py")
    seen_committed = _accepted(second.detect_outcomes()[0])

    assert seen_dirty and seen_committed
    assert seen_dirty[0].applied_on_revision == seen_committed[0].applied_on_revision


def test_parallel_worktrees_committing_one_fix_read_as_one_base(repo, tmp_path, request):
    """Best-of-N: two agents branched from one base land the same fix as two
    commits. That is one lesson applied twice at once, not recurrence."""
    a = _suggest(repo, request)
    b = OutcomeTracker(codebase_root=str(repo), project_id=f"wt2-{request.node.name}")
    b.save_session([_Suggestion()], "make f return 2", {})
    time.sleep(TICK)
    for name in ("a", "b"):
        wt = _worktree(repo, tmp_path / f"wt-{name}", f"agent-{name}")
        _apply(wt)
        _git(wt, "commit", "-qam", f"apply {name}")

    seen_a = _accepted(a.detect_outcomes()[0])
    seen_b = _accepted(b.detect_outcomes()[0])

    assert seen_a and seen_b
    assert seen_a[0].applied_on_revision == seen_b[0].applied_on_revision


def test_a_lesson_recurring_after_the_repo_moved_on_reads_as_two_bases(repo, request):
    """What promotion claims: the lesson came up again after the first
    application had already landed."""
    first = _suggest(repo, request)
    _apply(repo)
    _git(repo, "commit", "-qm", "first", "--", "src/foo.py")
    seen_first = _accepted(first.detect_outcomes()[0])
    time.sleep(TICK)  # `--since` is second-granular; keep "first" out of it

    second = OutcomeTracker(codebase_root=str(repo),
                            project_id=f"wt2-{request.node.name}")
    second.save_session([_Suggestion(diff=SECOND_DIFF)], "make f return 3", {})
    time.sleep(TICK)
    _apply(repo, 3)
    _git(repo, "commit", "-qm", "second", "--", "src/foo.py")
    seen_second = _accepted(second.detect_outcomes()[0])

    assert seen_first and seen_second
    assert seen_first[0].applied_on_revision != seen_second[0].applied_on_revision


def test_ledger_edits_in_a_nested_worktree_attribute_to_the_repo_path(repo, request):
    """A worktree under `.claude/worktrees/x/` sits INSIDE the main checkout.
    Relative to the main checkout its file is `.claude/worktrees/x/src/foo.py`,
    which matches no suggestion; relative to its own checkout it is
    `src/foo.py`."""
    import json

    from neo.hook import HOOK_LEDGER

    tracker = _suggest(repo, request)
    wt = _worktree(repo, repo / ".claude" / "worktrees" / "x", "agent")
    HOOK_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with HOOK_LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.time(), "tool": "Edit",
                             "file_path": str(wt / "src" / "foo.py")}) + "\n")
        fh.write(json.dumps({"ts": time.time(), "tool": "Edit",
                             "file_path": str(repo / "src" / "bar.py")}) + "\n")

    tracker._checkout_roots = tracker._list_checkout_roots()
    events = {(path, own) for _, path, own in tracker._load_host_edit_events()}

    assert events == {("src/foo.py", False), ("src/bar.py", True)}
