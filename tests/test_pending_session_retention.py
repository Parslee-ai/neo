"""Pending suggestions must survive a neo invocation that resolves nothing.

`collect_outcomes` used to delete the entire session log whenever any session
existed, even when nothing had changed and no outcome was produced. So any neo
run between "neo suggests X" and "user applies X" silently destroyed the
pending suggestion, and the acceptance could never be attributed.

The multi-session read in `collect_outcomes` was added to prevent exactly that
loss; the unconditional clear defeated it one level up. Measured over 30 days
of real traffic: 108 episodes, 58 stuck at
`suggested_pending_downstream_outcome`, and **zero** `accepted` outcomes ever —
so the promote path, which requires two git-verified acceptances, could never
fire regardless of how correct its own gates were.
"""

import json
import subprocess
import time
from pathlib import Path

import pytest

from neo.memory.outcomes import OutcomeTracker, OutcomeType

# `git log --since` is second-granular, so a commit made in the same wall-clock
# second as a session timestamp is reported as "changed since" it. Separate the
# steps or the test measures clock rounding rather than retention.
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
    time.sleep(TICK)
    return root


class _Suggestion:
    def __init__(self, file_path):
        self.file_path = file_path
        self.unified_diff = (
            "--- a/src/foo.py\n+++ b/src/foo.py\n@@\n"
            "-    return 1\n+    if True:\n+        return 1\n"
        )
        self.description = "guard the return"
        self.confidence = 0.9
        self.suggestion_id = "sug-1"
        self.code_block = ""


def _tracker(repo, project_id=None):
    """One tracker per test, scoped to a unique project_id.

    conftest now redirects `outcomes.SESSIONS_DIR` to a per-test fake home, but
    that is one directory shared by every test in the run. A fixed project_id
    would still let one test read another's leftover session log — which is
    exactly what happened before this was scoped: these tests passed in
    isolation and failed in the full suite, because a stale pending session
    carried a timestamp old enough to match the next test's repo-init commit.
    """
    if project_id is None:
        # tmp_path is unique per test, so its leaf name is a safe scope key.
        project_id = f"pending-{repo.parent.name}-{repo.name}"
    return OutcomeTracker(codebase_root=str(repo), project_id=project_id)


def _apply_the_suggestion(repo):
    (repo / "src" / "foo.py").write_text("def f():\n    if True:\n        return 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "applied")
    time.sleep(TICK)


def test_acceptance_survives_a_dirty_working_tree(repo):
    """The case the first fix missed, and the only case that actually occurs.

    `_get_changed_files_since` reports every file changed anywhere in the repo,
    not files related to the suggestion. The first version asked "is
    changed_files empty?" — true only in a repo with no commits and a spotless
    tree since the suggestion. One unrelated dirty file dropped the session and
    lost the acceptance exactly as before the fix. Every other test in this file
    runs against a pristine tree, which is the one state neo is never invoked in.
    """
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/foo.py")], "fix foo",
                         {"src/foo.py": "fact-1"})
    time.sleep(TICK)

    # Ordinary working state: something unrelated is dirty.
    (repo / "src" / "unrelated.py").write_text("x = 2\n")

    outcomes, _ = tracker.detect_outcomes()
    assert not any(o.outcome_type is OutcomeType.ACCEPTED for o in outcomes)
    assert Path(tracker._session_log_path).exists(), (
        "an unrelated dirty file destroyed the pending suggestion"
    )

    _apply_the_suggestion(repo)

    outcomes, fact_ids = tracker.detect_outcomes()
    assert any(o.outcome_type is OutcomeType.ACCEPTED for o in outcomes)
    assert fact_ids.get("src/foo.py") == "fact-1"


def test_a_resolved_suggestion_is_dropped_from_a_partly_pending_session(repo):
    """A session suggesting two files, one of which lands, must retain only the
    other — or the landed one re-fires its outcome on every later run."""
    tracker = _tracker(repo)
    tracker.save_session(
        [_Suggestion("src/foo.py"), _Suggestion("src/bar.py")], "fix both", {},
    )
    time.sleep(TICK)

    _apply_the_suggestion(repo)  # touches src/foo.py only

    tracker.detect_outcomes()

    log = Path(tracker._session_log_path)
    assert log.exists()
    retained = json.loads(log.read_text().strip())
    paths = [s["file_path"] for s in retained["suggestions"]]
    assert paths == ["src/bar.py"], f"expected only the unresolved path, got {paths}"


def test_a_review_only_path_does_not_re_emit_forever(repo):
    """Mixed session: the docs path produced its weak UNVERIFIED on the first
    pass and must not produce it again on every subsequent invocation."""
    tracker = _tracker(repo)
    tracker.save_session(
        [_Suggestion("src/foo.py"), _Suggestion("docs/guide.md")], "both", {},
    )
    time.sleep(TICK)

    first, _ = tracker.detect_outcomes()
    assert any(o.outcome_type is OutcomeType.UNVERIFIED for o in first)

    for _ in range(3):
        later, _ = tracker.detect_outcomes()
        assert not any(
            o.outcome_type is OutcomeType.UNVERIFIED and o.file_path.endswith(".md")
            for o in later
        ), "the review-only outcome re-fired"


def test_acceptance_survives_intervening_runs(repo):
    """The regression itself: neo runs twice before the user acts, and the
    acceptance must still be detected afterwards."""
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/foo.py")], "fix foo",
                         {"src/foo.py": "fact-1"})
    time.sleep(TICK)

    for _ in range(2):
        outcomes, _ = tracker.detect_outcomes()
        assert outcomes == [], "nothing has changed yet"

    _apply_the_suggestion(repo)

    outcomes, fact_ids = tracker.detect_outcomes()
    assert any(o.outcome_type is OutcomeType.ACCEPTED for o in outcomes)
    assert fact_ids.get("src/foo.py") == "fact-1", "the fact link must survive too"


def test_pending_session_is_retained_not_deleted(repo):
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/foo.py")], "fix foo", {})
    time.sleep(TICK)
    log = Path(tracker._session_log_path)

    tracker.detect_outcomes()

    assert log.exists(), "the pending session was deleted"
    assert len(log.read_text().strip().splitlines()) == 1


def test_a_pending_session_is_not_duplicated_by_repeated_runs(repo):
    """Retention must rewrite the log, not append to it — otherwise one
    suggestion accrues a fresh copy per invocation and its fact gets bumped
    once per copy when it finally lands."""
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/foo.py")], "fix foo", {})
    time.sleep(TICK)
    log = Path(tracker._session_log_path)

    for _ in range(3):
        tracker.detect_outcomes()

    assert len(log.read_text().strip().splitlines()) == 1


def test_the_log_is_cleared_once_the_session_resolves(repo):
    """Retention must not become a leak: a processed session goes away."""
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/foo.py")], "fix foo", {})
    time.sleep(TICK)
    _apply_the_suggestion(repo)

    tracker.detect_outcomes()

    assert not Path(tracker._session_log_path).exists()


def test_review_only_sessions_are_not_retained(repo):
    """Their weak UNVERIFIED outcome fires on the first pass; keeping them
    would replay the same signal on every later invocation."""
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("/review/commit-abc.md")], "review this", {})
    time.sleep(TICK)

    outcomes, _ = tracker.detect_outcomes()

    assert any(o.outcome_type is OutcomeType.UNVERIFIED for o in outcomes)
    assert not Path(tracker._session_log_path).exists()


def test_sessions_with_no_suggestions_are_not_retained(repo):
    """An advise-only run has nothing to wait for."""
    tracker = _tracker(repo)
    tracker.save_session([], "just explain this", {})
    time.sleep(TICK)

    tracker.detect_outcomes()

    assert not Path(tracker._session_log_path).exists()


def test_an_accepted_suggestion_never_comes_back_as_independent(repo):
    """The regression the reduction introduced.

    Resolved suggestions are dropped from the retained record so they cannot
    re-match. But `_match_to_suggestions` builds its "ours" set from that same
    list, and the scan anchor stayed pinned to the original session — so on the
    next run git still reported the file changed, it was no longer recognized
    as suggested, and it landed in the independent branch. neo then recorded
    "user changed a file neo didn't suggest" about a file neo suggested and the
    user demonstrably applied, on every invocation until the 14-day TTL.

    Only a SECOND detect_outcomes() call exposes it; every earlier retention
    test stopped after one round.
    """
    tracker = _tracker(repo)
    (repo / "src" / "bar.py").write_text("def g():\n    return 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add bar")
    time.sleep(TICK)

    tracker.save_session(
        [_Suggestion("src/foo.py"), _Suggestion("src/bar.py")], "fix both", {},
    )
    time.sleep(TICK)
    _apply_the_suggestion(repo)  # applies src/foo.py only

    first, _ = tracker.detect_outcomes()
    assert any(
        o.outcome_type is OutcomeType.ACCEPTED and o.file_path == "src/foo.py"
        for o in first
    )

    for round_number in (2, 3):
        later, _ = tracker.detect_outcomes()
        offenders = [
            o.file_path for o in later
            if o.outcome_type is OutcomeType.INDEPENDENT
            and o.file_path.endswith("foo.py")
        ]
        assert not offenders, (
            f"round {round_number}: accepted suggestion re-reported as "
            f"independent: {offenders}"
        )


def test_the_scan_anchor_advances_so_work_does_not_repeat(repo):
    """The other half of the same root cause.

    A retained record kept querying from the original suggestion time, so every
    later invocation re-scanned the whole window and re-forked git per changed
    file — measured at seconds per invocation, persisting for the full TTL,
    where before the (buggy) clearing made it decay to zero.
    """
    tracker = _tracker(repo)
    tracker.save_session(
        [_Suggestion("src/foo.py"), _Suggestion("src/bar.py")], "fix both", {},
    )
    time.sleep(TICK)
    _apply_the_suggestion(repo)

    tracker.detect_outcomes()

    retained = json.loads(Path(tracker._session_log_path).read_text().strip())
    assert retained["scanned_through"] > retained["timestamp"], (
        "the scan anchor never advanced; the window re-scans forever"
    )
    assert "src/foo.py" in retained["resolved_paths"]


def test_a_peer_process_save_is_not_erased_by_our_rewrite(repo):
    """The lost update.

    `collect_outcomes` reads the log, then does git and LM work, then rewrites.
    A second neo process saving a session inside that window used to be erased
    without trace — the rewrite wrote back a view of the file predating their
    save. Locking only the write did not fix it; the rewrite has to re-read
    under the lock and preserve what it never processed.

    Fixed by merge-on-write rather than a wider lock: holding the lock across
    the reasoning work would let a slow run block another process's save.
    """
    ours = _tracker(repo, project_id="peer-test-shared")
    peer = _tracker(repo, project_id="peer-test-shared")

    ours.save_session([_Suggestion("src/foo.py")], "ours", {})
    time.sleep(1)

    # Our read-modify-write begins.
    sessions = ours._load_unprocessed_sessions()
    ours._last_pending = list(sessions)
    ours._last_loaded_keys = {ours._session_key(s) for s in sessions}

    # A peer saves while we are still working.
    peer.save_session([_Suggestion("src/peer.py")], "theirs", {})

    # We complete our rewrite.
    ours.consume_sessions_keeping_pending()

    log = Path(ours._session_log_path)
    survivors = sorted(
        json.loads(line)["suggestions"][0]["file_path"]
        for line in log.read_text().strip().splitlines()
    )
    assert "src/peer.py" in survivors, "the peer's session was erased"
    assert "src/foo.py" in survivors, "our own pending session was dropped"
    assert len(survivors) == len(set(survivors)), "merge duplicated a session"


def test_the_merge_does_not_resurrect_processed_sessions(repo):
    """The other direction: a session we DID process must not come back just
    because it is still in the file when we re-read."""
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("docs/guide.md")], "review only", {})
    time.sleep(TICK)

    tracker.detect_outcomes()  # review-only resolves and is not retained

    log = Path(tracker._session_log_path)
    assert not log.exists(), "a processed session was written back by the merge"


def test_a_weak_outcome_cannot_overwrite_a_verified_one():
    """`final_outcome` must report the strongest evidence in the batch.

    One invocation commonly suggests a code edit *and* a review/docs path, and
    `_dedup_outcomes` keys by path, so both survive. Assigning unconditionally
    let dict order decide: the weak UNVERIFIED from the docs path landed after
    the git-verified ACCEPTED, and the episode reported "unverified" for a run
    whose code suggestion was demonstrably applied. Review paths are neo's most
    common suggestion shape, so that under-reported acceptance in the very
    ledger `neo memory learning-stats` reads.
    """
    from neo.memory.store import FactStore

    rank = FactStore._outcome_rank
    assert rank(OutcomeType.ACCEPTED) > rank(OutcomeType.UNVERIFIED)
    assert rank(OutcomeType.MODIFIED) > rank(OutcomeType.INDEPENDENT)
    assert rank(OutcomeType.INDEPENDENT) > rank(OutcomeType.UNVERIFIED)
    # An unknown/absent prior outcome must lose to anything real.
    assert rank(None) == 0
    assert rank(OutcomeType.UNVERIFIED) > rank(None)


def test_session_expiry_is_exercised_directly(repo):
    """`_session_expired` in isolation.

    The end-to-end expiry test below reaches the same postcondition through the
    git path (ageing the record makes the fixture's own init commit look
    "changed since"), so it passed while `_session_expired` was never called
    once. Assert the predicate itself, or the TTL is untested.
    """
    from neo.memory.outcomes import SessionRecord

    tracker = _tracker(repo)
    fresh = SessionRecord(timestamp=time.time())
    stale = SessionRecord(
        timestamp=time.time() - tracker.PENDING_SESSION_TTL_SECONDS - 60
    )

    assert not tracker._session_expired(fresh)
    assert tracker._session_expired(stale)


def test_abandoned_suggestions_expire(repo):
    """Retention is bounded: a suggestion nobody acted on for two weeks is
    abandoned, not pending, and must not grow the log forever."""
    tracker = _tracker(repo)
    tracker.save_session([_Suggestion("src/foo.py")], "fix foo", {})
    log = Path(tracker._session_log_path)

    # Age the record past the TTL.
    record = json.loads(log.read_text().strip())
    record["timestamp"] -= tracker.PENDING_SESSION_TTL_SECONDS + 60
    log.write_text(json.dumps(record) + "\n")

    tracker.detect_outcomes()

    assert not log.exists()


def test_retention_survives_many_pending_sessions(repo):
    """Several distinct suggestions can be outstanding at once; each must be
    kept until its own file moves."""
    tracker = _tracker(repo)
    (repo / "src" / "bar.py").write_text("def g():\n    return 2\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "add bar")
    time.sleep(TICK)

    tracker.save_session([_Suggestion("src/foo.py")], "fix foo", {})
    time.sleep(1)
    tracker.save_session([_Suggestion("src/bar.py")], "fix bar", {})
    time.sleep(TICK)

    tracker.detect_outcomes()

    log = Path(tracker._session_log_path)
    assert log.exists()
    assert len(log.read_text().strip().splitlines()) == 2
