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

Time is a fake clock, not `sleep`. `git log --since` is second-granular, so
anything that must fall BEFORE a suggestion has to land in an earlier second;
sleeping two seconds per ordering cost this file 76s and pushed CI's test jobs
into their 10-minute timeout. Every event — session save, commit, file write —
instead takes its own timestamp off one monotonic clock, ten seconds apart and
in the past, so ordering is exact and costs nothing.
"""

import os
import subprocess
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from neo.memory import outcomes as outcomes_mod
from neo.memory.outcomes import OutcomeTracker, OutcomeType
SUGGESTED_DIFF = "--- a/src/foo.py\n+++ b/src/foo.py\n@@\n-    return 1\n+    return 2\n"
SECOND_DIFF = "--- a/src/foo.py\n+++ b/src/foo.py\n@@\n-    return 2\n+    return 3\n"


class _Clock:
    """Monotonic fake time, ten seconds per event, ending well before now."""

    def __init__(self):
        self.now = time.time() - 3600

    def tick(self) -> float:
        self.now += 10
        return self.now


_clock = _Clock()


@pytest.fixture(autouse=True)
def _fresh_clock():
    global _clock
    _clock = _Clock()


def _git(root, *args):
    env = None
    if args and args[0] == "commit":
        stamp = f"@{int(_clock.tick())} +0000"
        env = {**os.environ, "GIT_AUTHOR_DATE": stamp, "GIT_COMMITTER_DATE": stamp}
    return subprocess.run(["git", *args], cwd=root, check=True, env=env,
                          capture_output=True, text=True).stdout.strip()


def _write(path, text):
    path.write_text(text)
    stamp = _clock.tick()
    os.utime(path, (stamp, stamp))


@contextmanager
def _at_clock():
    """Stamp a session save with the fake clock (it reads `time.time()`)."""
    stamp = _clock.tick()

    class _T:
        @staticmethod
        def time():
            return stamp

    with patch.object(outcomes_mod, "time", _T):
        yield


def _save(tracker, suggestions, prompt):
    with _at_clock():
        tracker.save_session(suggestions, prompt, {})


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    _write(root / "src" / "foo.py", "def f():\n    return 1\n")
    _write(root / "src" / "bar.py", "def g():\n    return 1\n")
    _git(root, "init", "-q", "-b", "main", ".")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "T")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")
    (root / "notes.txt").write_text("unrelated\n")
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
    _save(tracker, [_Suggestion(diff=diff)], "make f return 2")
    return tracker


def _worktree(repo, path, branch):
    _git(repo, "worktree", "add", "-q", "-b", branch, str(path))
    return path


def _apply(checkout, value=2):
    _write(checkout / "src" / "foo.py", f"def f():\n    return {value}\n")


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
    _write(wt / "src" / "foo.py", "def f():\n    return 1  # wip\n")
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
    _write(wt / "src" / "bar.py", "def g():\n    return 99\n")
    _git(wt, "commit", "-qam", "unrelated work")
    _write(wt / "src" / "baz.py", "x = 1\n")  # dirty, untracked
    _write(repo / "src" / "foo.py", "def f():\n    return 2\n")  # the acceptance

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
    _save(second, [_Suggestion()], "make f return 2")
    _git(repo, "commit", "-qm", "apply", "--", "src/foo.py")
    seen_committed = _accepted(second.detect_outcomes()[0])

    assert seen_dirty and seen_committed
    assert seen_dirty[0].applied_on_revision == seen_committed[0].applied_on_revision


def test_parallel_worktrees_committing_one_fix_read_as_one_base(repo, tmp_path, request):
    """Best-of-N: two agents branched from one base land the same fix as two
    commits. That is one lesson applied twice at once, not recurrence."""
    a = _suggest(repo, request)
    b = OutcomeTracker(codebase_root=str(repo), project_id=f"wt2-{request.node.name}")
    _save(b, [_Suggestion()], "make f return 2")
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
    application had already landed. Pins the REVISIONS only: the second fix is
    a different diff, so whether the two correlate as one lesson is the
    signature's business (test_fact_store), not this test's."""
    first = _suggest(repo, request)
    _apply(repo)
    _git(repo, "commit", "-qm", "first", "--", "src/foo.py")
    seen_first = _accepted(first.detect_outcomes()[0])

    second = OutcomeTracker(codebase_root=str(repo),
                            project_id=f"wt2-{request.node.name}")
    _save(second, [_Suggestion(diff=SECOND_DIFF)], "make f return 3")
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


def test_an_unrelated_commit_to_the_same_file_does_not_supply_the_base(
        repo, tmp_path, request):
    """Touching the path is not carrying the fix. A sibling agent's unrelated
    edit to the same file, on a branch with history of its own, lands between
    the suggestion and the fix; taking ITS parent split one sitting into two
    bases. The base must come from the commit whose patch carries the
    suggestion."""
    sib = _worktree(repo, tmp_path / "sibling", "sibling")
    _write(sib / "src" / "bar.py", "def g():\n    return 7\n")
    _git(sib, "commit", "-qam", "sibling's own history")
    tracker = _suggest(repo, request)
    asked_at = _git(repo, "rev-parse", "HEAD")
    _write(sib / "src" / "foo.py", "# sibling note\ndef f():\n    return 1\n")
    _git(sib, "commit", "-qam", "unrelated edit to foo")
    _apply(repo)
    _git(repo, "commit", "-qm", "apply", "--", "src/foo.py")

    outcomes, _ = tracker.detect_outcomes()

    accepted = _accepted(outcomes)
    assert accepted and accepted[0].applied_on_revision == asked_at


def test_one_lesson_committed_file_by_file_collapses_to_one_base(repo, request):
    """Two questions asked at H, the lesson applied to foo.py (C1) and then
    to bar.py on top of it (C2). Bases H and C1 differ; the carrier links them
    back into one sitting."""
    from neo.memory.store import FactStore

    first = _suggest(repo, request)
    second = OutcomeTracker(codebase_root=str(repo),
                            project_id=f"wt2-{request.node.name}")
    bar = _Suggestion("src/bar.py", diff=SUGGESTED_DIFF.replace("foo", "bar"))
    _save(second, [bar], "make g return 2")
    _apply(repo)
    _git(repo, "commit", "-qm", "foo", "--", "src/foo.py")
    _write(repo / "src" / "bar.py", "def g():\n    return 2\n")
    _git(repo, "commit", "-qm", "bar", "--", "src/bar.py")

    a = _accepted(first.detect_outcomes()[0])
    b = _accepted(second.detect_outcomes()[0])

    assert a and b
    assert a[0].applied_on_revision != b[0].applied_on_revision
    bases = FactStore._sitting_bases({
        o: (x.applied_on_revision, x.carrier_revision, x.applied_on_parent)
        for o, x in (("a", a[0]), ("b", b[0]))
    })
    assert bases["a"] == bases["b"]


def test_dirty_first_application_then_commit_collapses_to_one_base(repo, request):
    """The usual editor flow: A edited at H and seen DIRTY by a run (no
    carrier), committed as C1, then B applied on top of C1. C1's parent is
    A's base, which is what folds them into one sitting."""
    from neo.memory.store import FactStore

    first = _suggest(repo, request)
    _apply(repo)
    a = _accepted(first.detect_outcomes()[0])
    _git(repo, "commit", "-qm", "foo", "--", "src/foo.py")

    second = OutcomeTracker(codebase_root=str(repo),
                            project_id=f"wt2-{request.node.name}")
    bar = _Suggestion("src/bar.py", diff=SUGGESTED_DIFF.replace("foo", "bar"))
    _save(second, [bar], "make g return 2")
    _write(repo / "src" / "bar.py", "def g():\n    return 2\n")
    _git(repo, "commit", "-qm", "bar", "--", "src/bar.py")
    b = _accepted(second.detect_outcomes()[0])

    assert a and b and a[0].carrier_revision == ""
    assert a[0].applied_on_revision != b[0].applied_on_revision
    bases = FactStore._sitting_bases({
        o: (x.applied_on_revision, x.carrier_revision, x.applied_on_parent)
        for o, x in (("a", a[0]), ("b", b[0]))
    })
    assert bases["a"] == bases["b"]


def test_dirty_files_across_many_worktrees_keep_the_first_listed_checkout(
    repo, tmp_path, request,
):
    """Worktrees are queried in parallel (one `git diff` each in turn cost 12.4 s
    at 101 worktrees); the first checkout `git worktree list` names still wins
    a path dirty in several, whichever query finishes first."""
    trees = [_worktree(repo, tmp_path / f"wt{i}", f"agent-{i}") for i in range(12)]
    for tree in trees[3:]:
        _apply(tree)
    _write(trees[5] / "src" / "bar.py", "def g():\n    return 5\n")

    tracker = _suggest(repo, request)
    tracker._checkout_roots = tracker._list_checkout_roots()
    listed = tracker._checkout_roots[1:]

    changed = tracker._get_linked_worktree_changes({"src/foo.py", "src/bar.py"})

    dirty = {str(tree) for tree in trees[3:]}
    assert changed == {
        "src/foo.py": next(r for r in listed if r in dirty),
        "src/bar.py": next(r for r in listed if r.endswith("wt5")),
    }


def test_only_suggested_paths_are_asked_about_in_other_checkouts(repo, tmp_path, request):
    """A whole-tree `git diff` per checkout took 12.4 s at 101 worktrees; the
    answer only ever resolves suggested paths, so only those are asked about.
    A path that cannot be repository-relative must not sink the query."""
    tree = _worktree(repo, tmp_path / "wt", "agent-1")
    _apply(tree)
    _write(tree / "src" / "bar.py", "def g():\n    return 9\n")

    tracker = _suggest(repo, request)
    tracker._checkout_roots = tracker._list_checkout_roots()

    assert tracker._get_linked_worktree_changes({"src/foo.py"}) == {"src/foo.py": str(tree)}
    assert tracker._get_linked_worktree_changes(
        {"src/foo.py", "/elsewhere/x.py", "../outside.py", ""}
    ) == {"src/foo.py": str(tree)}


def test_a_global_pathspec_setting_does_not_silence_other_checkouts(
    repo, tmp_path, request, monkeypatch,
):
    """git refuses `--literal-pathspecs` alongside GIT_GLOB/NOGLOB/ICASE_PATHSPECS,
    which would empty the answer without a word."""
    tree = _worktree(repo, tmp_path / "wt", "agent-1")
    _apply(tree)
    tracker = _suggest(repo, request)
    tracker._checkout_roots = tracker._list_checkout_roots()
    for name in ("GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS"):
        monkeypatch.setenv(name, "1")
    assert tracker._get_linked_worktree_changes({"src/foo.py"}) == {"src/foo.py": str(tree)}


def test_many_suggested_paths_are_asked_in_chunks(repo, tmp_path, request, monkeypatch):
    from neo.memory import outcomes

    tree = _worktree(repo, tmp_path / "wt", "agent-1")
    _apply(tree)
    tracker = _suggest(repo, request)
    tracker._checkout_roots = tracker._list_checkout_roots()
    monkeypatch.setattr(outcomes, "PATHSPEC_CHUNK", 3)
    calls = []
    real = outcomes.OutcomeTracker._git_dirty
    monkeypatch.setattr(outcomes.OutcomeTracker, "_git_dirty",
                        staticmethod(lambda root, argv, env=None: calls.append(argv) or real(root, argv, env)))

    suggested = {f"src/m{i}.py" for i in range(7)} | {"src/foo.py"}
    assert tracker._get_linked_worktree_changes(suggested) == {"src/foo.py": str(tree)}
    assert [len(argv) - argv.index("--") - 1 for argv in calls] == [3, 3, 2]


def test_a_directory_suggestion_does_not_widen_the_query(repo, tmp_path, request, monkeypatch):
    _apply(_worktree(repo, tmp_path / "wt", "agent-1"))
    tracker = _suggest(repo, request)
    tracker._checkout_roots = tracker._list_checkout_roots()
    asked = []
    monkeypatch.setattr(tracker, "_get_working_tree_changes",
                        lambda root=None, paths=None: asked.append(paths) or set())
    tracker._get_linked_worktree_changes({".", "src/", "src/foo.py"})
    assert asked == [["src/foo.py"]]


def test_no_outstanding_suggestion_asks_no_other_checkout(repo, tmp_path, request, monkeypatch):
    _apply(_worktree(repo, tmp_path / "wt", "agent-1"))
    tracker = _suggest(repo, request)
    tracker._checkout_roots = tracker._list_checkout_roots()
    asked = []
    monkeypatch.setattr(tracker, "_get_working_tree_changes",
                        lambda root=None, paths=None: asked.append(root) or set())

    assert tracker._get_linked_worktree_changes(set()) == {}
    assert tracker._get_linked_worktree_changes({"/abs/only.py"}) == {}
    assert asked == []


def test_a_suggested_file_renamed_away_in_another_checkout_does_not_live_there(
    repo, tmp_path, request,
):
    """Asked about by its old name, a renamed file reads as deleted; it must not
    become that checkout's dirty file, or `_applied_on_revision` takes that
    checkout's HEAD as the base of an acceptance it never held."""
    tree = _worktree(repo, tmp_path / "wt", "agent-1")
    _git(tree, "mv", "src/foo.py", "src/renamed.py")
    (tree / "src" / "bar.py").unlink()
    tracker = _suggest(repo, request)
    tracker._checkout_roots = tracker._list_checkout_roots()

    assert tracker._get_linked_worktree_changes({"src/foo.py", "src/bar.py"}) == {}


NON_ASCII = "src/café.py"
NON_ASCII_DIFF = f"--- a/{NON_ASCII}\n+++ b/{NON_ASCII}\n@@\n-    return 1\n+    return 2\n"


@pytest.mark.parametrize("where", ["here", "linked worktree"])
@pytest.mark.parametrize("committed", [False, True])
def test_a_non_ascii_path_is_matched_as_written(repo, tmp_path, request, where, committed):
    """git C-quotes a non-ASCII path by default (`"src/caf\\303\\251.py"`),
    so an edit to it matched no suggestion and was never seen."""
    _write(repo / NON_ASCII, "def f():\n    return 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add café")
    tracker = OutcomeTracker(codebase_root=str(repo), project_id=f"wt-{request.node.name}")
    suggestion = _Suggestion(file_path=NON_ASCII, diff=NON_ASCII_DIFF)
    _save(tracker, [suggestion], "make f return 2")
    checkout = repo if where == "here" else _worktree(repo, tmp_path / "agent-wt", "agent")

    _write(checkout / NON_ASCII, "def f():\n    return 2\n")
    if committed:
        _git(checkout, "commit", "-qam", "apply")

    outcomes, _ = tracker.detect_outcomes()

    assert [o.file_path for o in _accepted(outcomes)] == [NON_ASCII], outcomes


def test_commit_history_lists_non_ascii_paths_as_written(repo, request):
    _write(repo / NON_ASCII, "x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add café")
    tracker = OutcomeTracker(codebase_root=str(repo), project_id=f"wt-{request.node.name}")

    assert NON_ASCII in tracker._get_commits_since(None, 10)[0]["files"]


@pytest.mark.skipif(os.name == "nt", reason="Windows forbids `\"` in file names")
def test_working_tree_paths_survive_quotes_and_edge_spaces(repo):
    """Unquoted line output still escapes `"` and a strip eats edge spaces;
    only NUL-separated output names these files as they are."""
    names = ['src/q"x.py', "src/ lead.py"]
    for name in names:
        _write(repo / name, "x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "odd names")
    for name in names:
        _write(repo / name, "x = 2\n")
    tracker = OutcomeTracker(codebase_root=str(repo), project_id="odd-names")

    assert set(names) <= tracker._get_working_tree_changes(), names
    assert tracker._get_working_tree_changes(paths=names) == set(names)


@pytest.mark.skipif(os.name == "nt", reason="Windows forbids these characters in file names")
def test_committed_paths_survive_quotes_tabs_and_newlines(repo, request):
    """`git log` line output C-quotes these, and a tab in a path could pass for
    the `hash\\ttime\\tsubject` header; `-z` hands back each name as it is."""
    names = ['src/q"x.py', "src/tab\tx.py", "src/nl\nx.py", "src/ lead.py", NON_ASCII]
    for name in names:
        _write(repo / name, "x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "odd\tsubject")
    tracker = OutcomeTracker(codebase_root=str(repo), project_id=f"wt-{request.node.name}")

    newest = tracker._get_commits_since(None, 10)[0]
    assert set(names) <= set(newest["files"]), newest
    assert newest["message"] == "odd\tsubject"
    assert len(tracker._get_commits_since(None, 10)) == int(_git(repo, "rev-list", "--count", "HEAD"))
    assert set(names) <= tracker._get_changed_files_since(0, working_tree=set())
    assert set(names) <= tracker._get_changed_elsewhere_since(0)


def _commit_spelled(repo, path, text):
    """Commit ``path`` byte-for-byte and check it out. Through stdin, not argv
    or the filesystem: macOS's `core.precomposeUnicode` folds those to NFC, and
    a tree entry stored NFD (committed on Linux) must stay NFD here too."""
    sha = subprocess.run(["git", "hash-object", "-w", "--stdin"], cwd=repo, input=text,
                         check=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "update-index", "--add", "--index-info"], cwd=repo, check=True,
                   input=f"100644 {sha}\t{path}\n", text=True, encoding="utf-8")
    _git(repo, "commit", "-qm", "add café")
    _git(repo, "reset", "-q", "--hard")
    on_disk = next(p for p in os.listdir(repo / "src") if p.startswith("caf"))
    return repo / "src" / on_disk


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("where", ["here", "linked worktree"])
@pytest.mark.parametrize("stored, suggested", [("NFD", "NFC"), ("NFC", "NFD")])
def test_a_suggestion_matches_its_file_in_either_unicode_form(
    repo, tmp_path, request, stored, suggested, where, committed,
):
    """`é` is one code point (NFC) or `e` + a combining accent (NFD). A model
    may write either, whichever form git holds the name in."""
    import unicodedata

    held = unicodedata.normalize(stored, NON_ASCII)
    said = unicodedata.normalize(suggested, NON_ASCII)
    _commit_spelled(repo, held, "def f():\n    return 1\n")
    tracker = OutcomeTracker(codebase_root=str(repo), project_id=f"wt-{request.node.name}")
    diff = f"--- a/{said}\n+++ b/{said}\n@@\n-    return 1\n+    return 2\n"
    _save(tracker, [_Suggestion(file_path=said, diff=diff)], "make f return 2")
    checkout = repo if where == "here" else _worktree(repo, tmp_path / "agent-wt", "agent")
    target = next(p for p in os.listdir(checkout / "src") if p.startswith("caf"))
    base = _git(checkout, "rev-parse", "HEAD")

    _write(checkout / "src" / target, "def f():\n    return 2\n")
    if committed:
        _git(checkout, "commit", "-qam", "apply")

    outcomes, _ = tracker.detect_outcomes()

    accepted = _accepted(outcomes)
    assert [o.file_path for o in accepted] == [held], outcomes
    assert accepted[0].applied_on_revision == base


@pytest.mark.parametrize("stored, spoken", [("NFD", "NFC"), ("NFC", "NFD")])
def test_a_host_edit_in_the_other_unicode_form_takes_gits_spelling(
    repo, request, stored, spoken,
):
    """The host ledger and the model spell the name one way, git the other.
    git must only be asked about its own spelling, and the file is ours on
    every later round, never INDEPENDENT."""
    import json
    import unicodedata

    from neo.hook import HOOK_LEDGER

    held = unicodedata.normalize(stored, NON_ASCII)
    said = unicodedata.normalize(spoken, NON_ASCII)
    path = _commit_spelled(repo, held, "def f():\n    return 1\n")
    tracker = OutcomeTracker(codebase_root=str(repo), project_id=f"wt-{request.node.name}")
    diff = f"--- a/{said}\n+++ b/{said}\n@@\n-    return 1\n+    return 2\n"
    # A second, untouched suggestion keeps the session retained for round two.
    _save(tracker, [_Suggestion(file_path=said, diff=diff), _Suggestion("src/bar.py")],
          "make f return 2")

    _write(path, "def f():\n    return 2\n")
    HOOK_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with HOOK_LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": _clock.tick(), "tool": "Edit",
                             "file_path": str(repo / said)}) + "\n")

    first, _ = tracker.detect_outcomes()
    second, _ = tracker.detect_outcomes()

    assert [(o.outcome_type, o.file_path) for o in first] == [(OutcomeType.ACCEPTED, held)]
    assert "return 2" in first[0].diff_summary
    assert not [o for o in second if unicodedata.normalize("NFC", o.file_path)
                == unicodedata.normalize("NFC", NON_ASCII)], second


def test_commit_headers_are_not_mistaken_for_paths(repo, request):
    """A path may start with the header mark; empty and merge commits list no
    files, and the commit after either still parses."""
    _write(repo / "\x01a.py", "x = 1\n")
    _write(repo / "\x01b.py", "x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "marks")
    _git(repo, "commit", "-q", "--allow-empty", "--allow-empty-message", "-m", "")
    _git(repo, "checkout", "-q", "-b", "side")
    _write(repo / "src" / "bar.py", "def g():\n    return 2\n")
    _git(repo, "commit", "-qam", "side work")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "-m", "merge side", "side")
    tracker = OutcomeTracker(codebase_root=str(repo), project_id=f"wt-{request.node.name}")

    commits = tracker._get_commits_since(None, 10)
    by_message = {c["message"]: c for c in commits}

    assert len(commits) == int(_git(repo, "rev-list", "--count", "HEAD"))
    assert by_message["merge side"]["files"] == []
    assert by_message[""]["files"] == []
    assert by_message["side work"]["files"] == ["src/bar.py"]
    assert {"\x01a.py", "\x01b.py"} <= set(by_message["marks"]["files"])


# ---------------------------------------------------------------------------
# Suggestions that NAME a file in another checkout.
#
# Every test above suggests `src/foo.py` relative to the main checkout and
# applies it elsewhere. Claude Code's Stop review does the opposite: the agent
# works in `.claude/worktrees/agent-*/`, so the suggestion names the file by its
# ABSOLUTE path in that worktree, while Neo runs rooted at the main checkout.
# Resolved against the main checkout that read `.claude/worktrees/x/src/foo.py`,
# which git never prints; 57 of 60 such suggestions on a live install were
# committed to main and never matched. One test per checkout shape seen there.
# ---------------------------------------------------------------------------


def _tracker(root, request):
    return OutcomeTracker(codebase_root=str(root), project_id=f"wt-{request.node.name}")


def _outcomes_for(tracker, path):
    outcomes, _ = tracker.detect_outcomes()
    return [(o.outcome_type, o.file_path) for o in outcomes if o.file_path == path]


def test_a_suggestion_in_a_removed_nested_worktree_is_accepted_once_merged(repo, request):
    """The Stop-review shape end to end: the worktree is gone by the time the
    next Neo run looks, so the path must have been resolved when saved."""
    wt = _worktree(repo, repo / ".claude" / "worktrees" / "x", "agent")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path=str(wt / "src" / "foo.py"))], "make f return 2")
    _apply(wt)
    _git(wt, "commit", "-qam", "apply")
    _git(repo, "worktree", "remove", str(wt))
    _git(repo, "merge", "-q", "--ff-only", "agent")

    assert _outcomes_for(tracker, "src/foo.py") == [(OutcomeType.ACCEPTED, "src/foo.py")]


def test_a_suggestion_in_a_live_nested_worktree_is_accepted_uncommitted(repo, request):
    wt = _worktree(repo, repo / ".claude" / "worktrees" / "x", "agent")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path=str(wt / "src" / "foo.py"))], "make f return 2")
    _apply(wt)

    assert _outcomes_for(tracker, "src/foo.py") == [(OutcomeType.ACCEPTED, "src/foo.py")]


def test_a_suggestion_in_a_sibling_worktree_is_accepted_on_its_branch(repo, tmp_path, request):
    wt = _worktree(repo, tmp_path / "sibling", "agent")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path=str(wt / "src" / "foo.py"))], "make f return 2")
    _apply(wt)
    _git(wt, "commit", "-qam", "apply")

    assert _outcomes_for(tracker, "src/foo.py") == [(OutcomeType.ACCEPTED, "src/foo.py")]


def test_a_suggestion_in_a_clone_of_the_same_remote_is_accepted_once_fetched(
    repo, tmp_path, request
):
    """A separate clone shares no git dir, only the project's identity (its
    remote). Its work counts once it reaches a local branch here."""
    remote = f"https://example.invalid/o/{request.node.name}.git"
    _git(repo, "remote", "add", "origin", remote)
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True,
                   capture_output=True)
    _git(clone, "remote", "set-url", "origin", remote)
    _git(clone, "config", "user.email", "t@t.t")
    _git(clone, "config", "user.name", "T")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path=str(clone / "src" / "foo.py"))], "make f return 2")
    _apply(clone)
    _git(clone, "commit", "-qam", "apply")
    _git(repo, "fetch", "-q", str(clone), "main:from-clone")

    assert _outcomes_for(tracker, "src/foo.py") == [(OutcomeType.ACCEPTED, "src/foo.py")]


def _assert_closed_once(tracker, path):
    """Unverifiable: one UNVERIFIED, nothing retained, nothing re-emitted."""
    first, _ = tracker.detect_outcomes()
    assert [(o.outcome_type, o.file_path) for o in first] == [(OutcomeType.UNVERIFIED, path)]
    assert tracker._last_pending == []
    second, _ = tracker.detect_outcomes()
    assert second == []


def test_a_suggestion_in_another_repository_is_closed_not_left_pending(
    repo, tmp_path, request
):
    other = tmp_path / "other"
    (other / "src").mkdir(parents=True)
    _write(other / "src" / "foo.py", "def f():\n    return 1\n")
    _git(other, "init", "-q", "-b", "main", ".")
    _git(other, "remote", "add", "origin", f"https://example.invalid/other/{request.node.name}.git")
    tracker = _tracker(repo, request)
    path = str(other / "src" / "foo.py")
    _save(tracker, [_Suggestion(file_path=path)], "make f return 2")

    _assert_closed_once(tracker, path)


def test_a_scratchpad_suggestion_is_closed_and_never_verifiable(repo, tmp_path, request):
    scratch = tmp_path / "scratchpad"
    scratch.mkdir()
    _write(scratch / "probe.sh", "echo 1\n")
    path = str(scratch / "probe.sh")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path=path)], "fix the probe")

    assert not outcomes_mod.suggestion_is_verifiable(path, True, str(repo))
    _assert_closed_once(tracker, path)


def test_a_session_rooted_in_a_subdirectory_matches_git_paths(repo, request):
    """Asked from `src/`, the model names `foo.py` (relative to what it was
    shown) or the absolute path; git prints `src/foo.py` either way."""
    tracker = _tracker(repo / "src", request)
    _save(tracker, [_Suggestion(file_path="foo.py")], "make f return 2")
    _apply(repo)
    _git(repo, "commit", "-qam", "apply")

    assert _outcomes_for(tracker, "src/foo.py") == [(OutcomeType.ACCEPTED, "src/foo.py")]
    assert outcomes_mod.normalize_suggestion_path(str(repo / "src" / "foo.py"),
                                                  str(repo / "src")) == "src/foo.py"


def test_the_resolver_names_each_shape_the_way_git_prints_it(repo, tmp_path):
    nested = _worktree(repo, repo / ".claude" / "worktrees" / "x", "nested")
    sibling = _worktree(repo, tmp_path / "sibling", "sibling")
    resolve = outcomes_mod.resolve_suggestion_target

    assert resolve(str(nested / "src" / "foo.py"), str(repo)) == (str(nested), "src/foo.py")
    assert resolve(str(sibling / "src" / "new.py"), str(repo)) == (str(sibling), "src/new.py")
    assert resolve("src/foo.py", str(repo)) == (str(repo), "src/foo.py")
    assert resolve("/src/foo.py", str(repo)) == (str(repo), "src/foo.py")
    assert resolve("src/../src/foo.py", str(repo))[1] == "src/foo.py"
    assert resolve(str(tmp_path / "elsewhere.py"), str(repo)) is None
    assert resolve(str(repo), str(repo)) is None
    assert outcomes_mod.suggestion_is_verifiable(str(nested / "src" / "foo.py"), True, str(repo))


def test_the_checkout_holding_the_file_supplies_its_flagged_regions(repo, tmp_path, request):
    """ADDRESSED compares against the code the review saw, which for a sibling
    worktree is only in that worktree."""
    wt = _worktree(repo, tmp_path / "sibling", "agent")
    _write(wt / "src" / "foo.py", "def f():\n    x = 0\n    return 1\n    # end\n")
    diff = ("--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1,4 +1,4 @@\n def f():\n"
            "     x = 0\n-    return 1\n+    return 2\n     # end\n")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path=str(wt / "src" / "foo.py"), diff=diff)], "fix")

    record = tracker._load_unprocessed_sessions()[0].suggestions[0]
    assert (record["checkout"], record["repo_path"]) == (str(wt), "src/foo.py")
    assert record["flagged_regions"]


def test_a_record_saved_before_repo_path_is_rescanned_once(repo, request):
    """An old record resolved the worktree path wrongly on every run while its
    watermark moved past the commit that applied it. It is looked for once more
    from the suggestion itself, and only for Neo's own suggested paths:
    unrelated work in that window is not re-minted as INDEPENDENT."""
    import json

    wt = _worktree(repo, repo / ".claude" / "worktrees" / "x", "agent")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path=str(wt / "src" / "foo.py"))], "make f return 2")
    _apply(wt)
    _git(wt, "commit", "-qam", "apply")
    _git(repo, "merge", "-q", "--ff-only", "agent")
    _write(repo / "src" / "bar.py", "def g():\n" + "    return 2\n" * 40)
    _git(repo, "commit", "-qam", "unrelated")
    log = tracker._session_log_path
    record = json.loads(log.read_text().strip())
    for sugg in record["suggestions"]:
        sugg.pop("repo_path")
        sugg.pop("checkout")
    record["scanned_through"] = _clock.tick()
    log.write_text(json.dumps(record) + "\n")

    outcomes, _ = tracker.detect_outcomes()

    assert [(o.outcome_type, o.file_path) for o in outcomes] == [
        (OutcomeType.ACCEPTED, "src/foo.py")
    ]


def test_a_suggestion_nothing_happened_to_is_closed_at_expiry(repo, request, monkeypatch):
    monkeypatch.setattr(OutcomeTracker, "PENDING_SESSION_TTL_SECONDS", 60)
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion()], "make f return 2")

    _assert_closed_once(tracker, "src/foo.py")


def test_a_relative_root_still_finds_its_checkout(repo, monkeypatch):
    monkeypatch.chdir(repo / "src")

    assert outcomes_mod.checkout_top(outcomes_mod.Path(".")) == repo
    assert OutcomeTracker(codebase_root=".").codebase_root == str(repo)


def test_a_relative_path_into_another_repository_is_closed_not_pending(repo, request):
    """`vendor/lib/x.py` inside a nested repository with its own identity is
    not a name this repository's git prints, so it must not wait for one."""
    nested = repo / "vendor" / "lib"
    nested.mkdir(parents=True)
    _write(nested / "x.py", "x = 1\n")
    _git(nested, "init", "-q", "-b", "main", ".")
    _git(nested, "remote", "add", "origin", f"https://example.invalid/v/{request.node.name}.git")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path="vendor/lib/x.py")], "fix x")

    record = tracker._load_unprocessed_sessions()[0].suggestions[0]
    assert outcomes_mod.Path(record["repo_path"]).is_absolute()
    _assert_closed_once(tracker, "vendor/lib/x.py")


def test_a_directory_alias_resolves_to_the_path_git_prints(repo):
    (repo / "alias").symlink_to(repo / "src", target_is_directory=True)

    assert outcomes_mod.resolve_suggestion_target("alias/foo.py", str(repo))[1] == "src/foo.py"


def test_a_tracked_symlink_keeps_its_own_name(repo, request):
    (repo / "link.py").symlink_to(repo / "src" / "foo.py")
    _git(repo, "add", "link.py")
    _git(repo, "commit", "-qm", "link")

    assert outcomes_mod.resolve_suggestion_target("link.py", str(repo))[1] == "link.py"


def test_a_case_misspelled_suggestion_is_saved_as_git_spells_it(repo, request):
    if not (repo / "SRC" / "FOO.PY").exists():
        pytest.skip("case-sensitive filesystem: the misspelling names no file")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path=str(repo / "Src" / "Foo.py"))], "make f return 2")
    _apply(repo)
    _git(repo, "commit", "-qam", "apply")

    assert tracker._load_unprocessed_sessions()[0].suggestions[0]["repo_path"] == "src/foo.py"
    assert _outcomes_for(tracker, "src/foo.py") == [(OutcomeType.ACCEPTED, "src/foo.py")]


def test_an_accepted_worktree_suggestion_is_never_reminted_as_independent(repo, request):
    """(Linus a) The second run after an acceptance whose worktree is gone:
    `resolved_paths` must hold git's name, or the file comes back INDEPENDENT."""
    wt = _worktree(repo, repo / ".claude" / "worktrees" / "x", "agent")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path=str(wt / "src" / "foo.py")),
                    _Suggestion(file_path=str(wt / "src" / "bar.py"))], "two fixes")
    _apply(wt)
    _git(wt, "commit", "-qam", "apply")
    _git(repo, "worktree", "remove", str(wt))
    _git(repo, "merge", "-q", "--ff-only", "agent")
    first, _ = tracker.detect_outcomes()
    assert [(o.outcome_type, o.file_path) for o in first] == [(OutcomeType.ACCEPTED, "src/foo.py")]
    _write(repo / "src" / "foo.py", "def f():\n" + "    return 9\n" * 40)

    second, _ = tracker.detect_outcomes()

    assert [o for o in second if o.file_path == "src/foo.py"] == []


def test_a_legacy_record_from_a_removed_worktree_root_still_resolves(repo, request):
    """(Linus b) Saved by a run INSIDE the worktree, with a relative path; the
    worktree is gone when detection reads it. Walking through the removed
    directory into the main checkout named `.claude/worktrees/x/src/foo.py`."""
    import json

    wt = _worktree(repo, repo / ".claude" / "worktrees" / "x", "agent")
    tracker_in_wt = _tracker(wt, request)
    _save(tracker_in_wt, [_Suggestion()], "make f return 2")
    record = json.loads(tracker_in_wt._session_log_path.read_text().strip())
    for sugg in record["suggestions"]:
        sugg.pop("repo_path")
        sugg.pop("checkout")
    record["codebase_root"] = str(wt)
    _apply(wt)
    _git(wt, "commit", "-qam", "apply")
    # --force: the run inside it left its own (untracked) cache there.
    _git(repo, "worktree", "remove", "--force", str(wt))
    _git(repo, "merge", "-q", "--ff-only", "agent")
    tracker = _tracker(repo, request)
    tracker._session_log_path.write_text(json.dumps(record) + "\n")

    outcomes, _ = tracker.detect_outcomes()

    assert [(o.outcome_type, o.file_path) for o in outcomes] == [
        (OutcomeType.ACCEPTED, "src/foo.py")
    ]


def test_a_subdirectory_link_is_keyed_by_the_name_git_prints(repo, request):
    """(Linus c, tracker half) The link is keyed by what the model wrote."""
    tracker = _tracker(repo / "src", request)
    with _at_clock():
        tracker.save_session([_Suggestion(file_path="foo.py")], "fix", {"foo.py": "fact-1"})
    _apply(repo)
    _git(repo, "commit", "-qam", "apply")

    outcomes, fact_ids = tracker.detect_outcomes()

    assert [(o.outcome_type, o.file_path) for o in outcomes] == [(OutcomeType.ACCEPTED, "src/foo.py")]
    assert fact_ids["src/foo.py"] == "fact-1"


def test_a_same_named_top_level_file_does_not_resolve_a_subdirectory_suggestion(repo, request):
    """(Linus d) Scope `src/`, suggestion `foo.py`: an edit to the unrelated
    top-level `foo.py` is not this suggestion being acted on."""
    _write(repo / "foo.py", "y = 1\n")
    _git(repo, "add", "foo.py")
    _git(repo, "commit", "-qm", "top-level foo")
    tracker = _tracker(repo / "src", request)
    _save(tracker, [_Suggestion(file_path="foo.py")], "fix")
    _write(repo / "foo.py", "y = 2\n")
    _git(repo, "commit", "-qam", "edit top-level foo")

    outcomes, _ = tracker.detect_outcomes()

    assert [o for o in outcomes if o.outcome_type != OutcomeType.INDEPENDENT] == []
    assert [s["repo_path"] for p in tracker._last_pending for s in p.suggestions] == ["src/foo.py"]


def test_an_ambient_git_dir_does_not_make_a_foreign_repository_ours(repo, tmp_path, monkeypatch):
    """(Linus e) Hooks export GIT_DIR; every checkout then reports the same
    common dir unless the resolver strips it."""
    other = tmp_path / "other"
    (other / "src").mkdir(parents=True)
    _write(other / "src" / "x.py", "x = 1\n")
    _git(other, "init", "-q", "-b", "main", ".")
    monkeypatch.setenv("GIT_DIR", str(repo / ".git"))

    assert outcomes_mod.resolve_suggestion_target(str(other / "src" / "x.py"), str(repo)) is None


def test_two_sessions_naming_one_raw_path_keep_both_acceptances(repo, request):
    """Monorepo: from the top `src/foo.py` is one file; from `backend/` the
    same spelling is `backend/src/foo.py`. Deduplication must key each outcome
    by its own git name, or one acceptance silently replaces the other."""
    (repo / "backend" / "src").mkdir(parents=True)
    _write(repo / "backend" / "src" / "foo.py", "def f():\n    return 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "backend")
    top = _tracker(repo, request)
    _save(top, [_Suggestion(file_path="src/foo.py")], "fix top")
    _save(_tracker(repo / "backend", request), [_Suggestion(file_path="src/foo.py")], "fix backend")
    _apply(repo)
    _write(repo / "backend" / "src" / "foo.py", "def f():\n    return 2\n")
    _git(repo, "commit", "-qam", "apply both")

    outcomes, _ = top.detect_outcomes()

    assert sorted((o.outcome_type, o.file_path) for o in outcomes) == [
        (OutcomeType.ACCEPTED, "backend/src/foo.py"),
        (OutcomeType.ACCEPTED, "src/foo.py"),
    ]


def test_retained_worktree_resolutions_cost_no_git_fork_per_request(repo, request, monkeypatch):
    """Resolved worktree paths are kept with their git names; re-resolving the
    raw absolute ones forked git twice per path on every request."""
    _write(repo / "other.py", "o = 1\n")
    _git(repo, "add", "other.py")
    _git(repo, "commit", "-qm", "other")
    wt = _worktree(repo, repo / ".claude" / "worktrees" / "x", "agent")
    _write(wt / "src" / "baz.py", "z = 1\n")
    tracker = _tracker(repo, request)
    _save(tracker, [_Suggestion(file_path=str(wt / "src" / name))
                    for name in ("foo.py", "bar.py", "baz.py")], "three fixes")
    _apply(wt)
    _write(wt / "src" / "bar.py", "def g():\n    return 2\n")
    tracker.detect_outcomes()
    assert len(tracker._last_pending) == 1  # baz.py still outstanding
    _write(repo / "other.py", "o = 2\n")  # tracked: the run reaches matching
    real_run = outcomes_mod.subprocess.run
    forks = []

    def counting_run(argv, *args, **kwargs):
        if "--git-common-dir" in argv:
            forks.append(argv)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(outcomes_mod.subprocess, "run", counting_run)
    tracker.detect_outcomes()

    assert forks == []
