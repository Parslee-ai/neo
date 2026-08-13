"""Tests for the persistent eligibility walk.

The cache exists to make the walk cheap. It must not make it different, and
the failure it could produce is the quietest one in the system: a file that
still exists but stops being offered, or a file that was deleted and keeps
being offered, with no error anywhere and a plausible answer on screen.

Four properties carry it, each a distinct way it could go wrong:

1. **Parity.** `TestParity` runs the cached walk and the uncached walk over
   the same tree, before and after every mutation shape that matters — add,
   delete, rename, gitignore edit, a nested checkout appearing — and asserts
   the two lists are equal. This is the test that would catch a wrong answer
   regardless of which mechanism produced it.
2. **The stamps stay live.** `TestStampsAreNeverCached` pins the one
   cross-cache invariant: a reused directory listing must still yield the
   file's CURRENT size and mtime, because the content index next door uses
   them as its own freshness stamp. Remembering them would make an edited
   file look unedited — this cache's staleness leaking into that one's.
3. **Reporting.** `TestReport` asserts the mode a call reports is the work it
   actually did. A cache that reports `warm` while re-listing everything, or
   `cold` while a corrupt file is being discarded on every call, hides the
   only signal an operator has.
4. **Degradation.** `TestDegradation` corrupts, truncates, mangles and
   write-protects the cache. Each must warn and return the right files.
"""

import json
import os
import time

import pytest

from neo import eligibility
from neo.eligibility import RACY_WINDOW_NS, WalkPolicy
from neo.index import walk_cache
from neo.index.walk_cache import cache_path, cached_walk

REPO = {
    "src/pkg/alpha.py": "def alpha():\n    return 1\n",
    "src/pkg/beta.py": "def beta():\n    return 2\n",
    "src/main.py": "from pkg import alpha\n",
    "docs/guide.md": "# Guide\n",
    "README.md": "# Project\n",
    ".gitignore": "build/\n*.log\n",
    "build/generated.py": "GENERATED = True\n",
    "debug.log": "noise\n",
}


def _repo(tmp_path, files=REPO):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


def _age(root, seconds: float = 10.0) -> None:
    """Move every directory's mtime back, out of the racy window.

    A tree a test has just built is entirely inside `RACY_WINDOW_NS`, so a
    cache of it is deliberately not trusted (see `eligibility.RACY_WINDOW_NS`)
    and every call would report `incremental`. Ageing the directories is how a
    test asks about steady state rather than about the guard — the guard has
    its own tests below.
    """
    when = time.time() - seconds
    for dirpath, dirnames, _ in os.walk(str(root)):
        os.utime(dirpath, (when, when))
        del dirnames  # walked for its directories only


def _settle(root) -> None:
    """Bring the cache to the state a repository is in between two calls.

    Walk once so `.neo/` exists — writing the cache CREATES that directory,
    which moves the repository root's own mtime, so a tree that is otherwise
    unchanged still re-lists its root on the call after its very first. Then
    age, so no directory sits inside `RACY_WINDOW_NS`, then walk again to
    record the aged stamps. Ageing before the final walk rather than after is
    the load-bearing order: the cache stores the mtime it saw, so ageing a
    tree the cache was already written against makes every directory look
    changed.
    """
    cached_walk(str(root), quiet=True)
    _age(root)
    cached_walk(str(root), quiet=True)


def _names(result) -> list[str]:
    return [entry.rel_path for entry in result.paths]


def _uncached(root, policy=None) -> list[str]:
    return _names(eligibility.walk(str(root), policy))


def _cached(root, policy=None) -> list[str]:
    return _names(cached_walk(str(root), policy, quiet=True))


class TestParity:
    """The cached walk answers exactly what the uncached walk answers."""

    def test_a_cold_walk_matches_the_uncached_walk(self, tmp_path):
        root = _repo(tmp_path)
        assert _cached(root) == _uncached(root)

    def test_a_warm_walk_matches_the_uncached_walk(self, tmp_path):
        root = _repo(tmp_path)
        _settle(root)
        assert _cached(root) == _uncached(root)

    @pytest.mark.parametrize("mutation", ["add", "delete", "rename", "new_directory"])
    def test_the_cache_survives_a_mutation(self, tmp_path, mutation):
        root = _repo(tmp_path)
        _settle(root)

        if mutation == "add":
            (root / "src/pkg/gamma.py").write_text("def gamma():\n    return 3\n")
        elif mutation == "delete":
            (root / "src/pkg/beta.py").unlink()
        elif mutation == "rename":
            (root / "src/pkg/beta.py").rename(root / "src/pkg/renamed.py")
        else:
            (root / "src/deep/nested").mkdir(parents=True)
            (root / "src/deep/nested/leaf.py").write_text("LEAF = 1\n")

        assert _cached(root) == _uncached(root)

    def test_a_gitignore_edit_takes_effect_on_the_next_call(self, tmp_path):
        """The case a directory mtime cannot see.

        Editing `.gitignore` changes a file's CONTENT, so no directory's mtime
        moves and every cached verdict looks current — while every one of them
        may now be wrong. The signature hashes the effective pattern list for
        exactly this.
        """
        root = _repo(tmp_path)
        _settle(root)
        assert "docs/guide.md" in _cached(root)

        (root / ".gitignore").write_text("build/\n*.log\ndocs/\n")

        assert "docs/guide.md" not in _cached(root)
        assert _cached(root) == _uncached(root)

    def test_a_nested_checkout_appearing_is_excluded(self, tmp_path):
        """`.worktrees` is in the shared default list; the cache must not widen it."""
        root = _repo(tmp_path)
        _settle(root)

        nested = root / ".worktrees" / "copy" / "src" / "pkg"
        nested.mkdir(parents=True)
        (nested / "alpha.py").write_text("def alpha():\n    return 1\n")

        found = _cached(root)
        assert not any(name.startswith(".worktrees/") for name in found)
        assert found == _uncached(root)

    def test_the_exclusion_counts_survive_a_warm_call(self, tmp_path):
        """`--dry-run`'s G1 accounting must not depend on cache state."""
        root = _repo(tmp_path)
        _settle(root)
        fresh = eligibility.walk(str(root))
        warm = cached_walk(str(root), quiet=True)

        assert warm.rescanned_dirs == 0
        assert (warm.excluded_dirs, warm.excluded_files) == (
            fresh.excluded_dirs,
            fresh.excluded_files,
        )
        assert warm.excluded_dirs > 0 and warm.excluded_files > 0

    def test_a_policy_filter_still_applies_to_a_warm_walk(self, tmp_path):
        """The cache stores ignore verdicts; `exts` is a consumer policy on top."""
        root = _repo(tmp_path)
        _settle(root)

        policy = WalkPolicy(exts=eligibility.normalize_exts(["py"]))
        assert _cached(root, policy) == _uncached(root, policy)
        assert all(name.endswith(".py") for name in _cached(root, policy))

    def test_extra_ignores_are_never_served_from_a_cache(self, tmp_path):
        """A caller's patterns are appended AFTER the repo's, and last match wins.

        So a `!negation` in `extra_ignores` can re-include a path the stored
        verdict excluded. The stored verdict is not a stale answer to this
        question, it is an answer to a different one.
        """
        root = _repo(tmp_path)
        _settle(root)

        policy = WalkPolicy(extra_ignores=("docs/",))
        result = cached_walk(str(root), policy, quiet=True)
        assert "docs/guide.md" not in _names(result)
        assert _names(result) == _uncached(root, policy)
        assert walk_cache.last_report().mode == "bypassed"

    def test_a_bypassed_call_does_not_overwrite_the_cache(self, tmp_path):
        root = _repo(tmp_path)
        _settle(root)
        before = cache_path(root).read_text()

        cached_walk(str(root), WalkPolicy(extra_ignores=("docs/",)), quiet=True)

        assert cache_path(root).read_text() == before

    def test_a_symlinked_directory_is_listed_but_never_descended(self, tmp_path):
        """The rule `os.walk(followlinks=False)` enforced, restated by hand.

        This traversal recurses itself, so `followlinks=False` protects only
        one directory read at a time. A link pointing at an ancestor is an
        infinite descent and a link pointing at `/` walks the machine — this
        test hangs rather than fails if the guard goes, which is the honest
        shape of that defect.
        """
        root = _repo(tmp_path)
        (root / "src" / "mirror").symlink_to(root / "docs", target_is_directory=True)
        (root / "src" / "loop").symlink_to(root, target_is_directory=True)

        found = _cached(root)

        # The finite case fails fast and pins the same rule; the loop is what
        # makes the consequence of losing it unbounded.
        assert "docs/guide.md" in found
        assert "src/mirror/guide.md" not in found
        assert not any(name.startswith("src/loop/") for name in found)

    def test_a_symlink_is_still_a_policy_choice_on_a_warm_walk(self, tmp_path):
        root = _repo(tmp_path)
        (root / "src/link.py").symlink_to(root / "src/main.py")
        _settle(root)

        assert "src/link.py" not in _cached(root)
        permissive = WalkPolicy(skip_symlinks=False)
        assert "src/link.py" in _cached(root, permissive)
        assert _cached(root, permissive) == _uncached(root, permissive)


class TestStampsAreNeverCached:
    """A reused listing must still report the file's CURRENT size and mtime.

    This is the invariant that keeps one cache's staleness out of another's:
    the content index decides what to re-tokenize from `EligiblePath.size` and
    `EligiblePath.mtime_ns`, so a remembered stamp would make an edited file
    look unedited and leave the index answering with text the file no longer
    holds. Editing a file does not move its directory's mtime, so the listing
    IS reused here — which is exactly the dangerous case.
    """

    def test_an_edit_is_visible_in_the_stamp_of_a_reused_listing(self, tmp_path):
        root = _repo(tmp_path)
        _settle(root)

        target = root / "src/pkg/alpha.py"
        before = {e.rel_path: e for e in cached_walk(str(root), quiet=True).paths}
        assert walk_cache.last_report().mode == "warm"

        target.write_text("def alpha():\n    return 1  # edited, and longer now\n")

        result = cached_walk(str(root), quiet=True)
        after = {e.rel_path: e for e in result.paths}
        # The directory was NOT re-listed -- an edit does not move a directory's
        # mtime -- and the stamp moved anyway.
        assert result.rescanned_dirs == 0
        assert after["src/pkg/alpha.py"].size != before["src/pkg/alpha.py"].size
        assert (
            after["src/pkg/alpha.py"].mtime_ns
            != before["src/pkg/alpha.py"].mtime_ns
        )
        assert after["src/pkg/alpha.py"].mtime_ns == target.stat().st_mtime_ns

    def test_a_size_ceiling_is_applied_to_the_current_size(self, tmp_path):
        """A file that grew past the ceiling drops out without a re-listing."""
        root = _repo(tmp_path)
        policy = WalkPolicy(max_file_bytes=64)
        _settle(root)
        assert "src/pkg/alpha.py" in _cached(root, policy)

        (root / "src/pkg/alpha.py").write_text("x = 1\n" * 100)

        assert "src/pkg/alpha.py" not in _cached(root, policy)


class TestReport:
    """The mode reported is the work that was done."""

    def test_a_first_call_is_cold(self, tmp_path):
        root = _repo(tmp_path)
        cached_walk(str(root), quiet=True)
        report = walk_cache.last_report()
        assert report.mode == "cold"
        assert report.rescanned == report.directories > 0
        assert report.reused == 0
        assert report.files == len(_uncached(root))

    def test_an_unchanged_repository_is_warm(self, tmp_path):
        root = _repo(tmp_path)
        _settle(root)
        cached_walk(str(root), quiet=True)

        report = walk_cache.last_report()
        assert report.mode == "warm"
        assert report.rescanned == 0
        assert report.reused == report.directories

    def test_one_changed_directory_is_incremental(self, tmp_path):
        root = _repo(tmp_path)
        _settle(root)
        (root / "src/pkg/gamma.py").write_text("GAMMA = 3\n")

        cached_walk(str(root), quiet=True)
        report = walk_cache.last_report()
        assert report.mode == "incremental"
        assert report.rescanned == 1
        assert report.reused == report.directories - 1

    def test_a_discarded_cache_reports_rebuilt_not_cold(self, tmp_path):
        """Distinct on purpose: only one of them means something went wrong."""
        root = _repo(tmp_path)
        cached_walk(str(root), quiet=True)
        cache_path(root).write_text("{not json")

        cached_walk(str(root), quiet=True)
        assert walk_cache.last_report().mode == "rebuilt"

    def test_the_summary_never_claims_work_it_did_not_do(self, tmp_path):
        root = _repo(tmp_path)
        _settle(root)
        cached_walk(str(root), quiet=True)

        summary = walk_cache.last_report().describe()
        assert "read warm" in summary
        assert "re-listed" not in summary

    def test_a_warm_call_does_not_rewrite_the_cache(self, tmp_path):
        """A steady-state call must not open a write.

        The content index next door learned this the expensive way: rewriting
        an unchanged signature on every warm call put two ordinary overlapping
        invocations into lock contention on the one path that should be free.
        """
        root = _repo(tmp_path)
        _settle(root)
        stamp = cache_path(root).stat().st_mtime_ns

        time.sleep(0.01)
        cached_walk(str(root), quiet=True)

        assert cache_path(root).stat().st_mtime_ns == stamp


class TestRacyDirectories:
    """A directory modified in the same timestamp tick it was read is suspect.

    Every test here pre-creates `.neo/` before the tree, and that is not
    incidental. Writing the cache CREATES that directory, which moves the
    repository ROOT's mtime, so the call after a first-ever call re-lists the
    root no matter what this guard does — and a test that only asserted
    "incremental" would pass with the guard deleted. Mutation-verified: with
    `is_current` reduced to an mtime comparison, `rescanned == directories`
    below fails and `rescanned >= 1` would not have.
    """

    def _tree(self, tmp_path):
        (tmp_path / ".neo").mkdir(exist_ok=True)
        return _repo(tmp_path)

    def test_a_freshly_built_tree_is_not_trusted(self, tmp_path):
        """Timestamp granularity is not always finer than the interval.

        HFS+ stamps whole seconds. A directory listed at time E whose mtime is
        M can be modified afterwards and land in the same bucket whenever
        `E - M` is under one tick, so a cache of a tree built moments ago is
        not trustworthy no matter how equal the two mtimes look.
        """
        root = self._tree(tmp_path)
        cached_walk(str(root), quiet=True)

        cached_walk(str(root), quiet=True)

        report = walk_cache.last_report()
        assert report.mode == "incremental"
        # EVERY directory, not merely one: nothing but the guard re-lists a
        # tree that has not been touched since it was read.
        assert report.rescanned == report.directories > 1
        assert report.reused == 0

    def test_a_settled_tree_is_trusted(self, tmp_path):
        root = self._tree(tmp_path)
        cached_walk(str(root), quiet=True)
        _age(root, seconds=RACY_WINDOW_NS / 1e9 + 1)
        cached_walk(str(root), quiet=True)

        cached_walk(str(root), quiet=True)

        report = walk_cache.last_report()
        assert report.mode == "warm"
        assert report.reused == report.directories

    def test_the_window_is_what_decides(self, tmp_path):
        """The same tree, the same call, with only the timestamps moved.

        Ageing by less than the window keeps every directory suspect; ageing
        past it makes every one of them reusable. If both halves ever agree,
        the two tests above have stopped measuring the guard.
        """
        root = self._tree(tmp_path)
        cached_walk(str(root), quiet=True)
        _age(root, seconds=RACY_WINDOW_NS / 1e9 / 2)
        cached_walk(str(root), quiet=True)
        cached_walk(str(root), quiet=True)
        assert walk_cache.last_report().reused == 0

        _age(root, seconds=RACY_WINDOW_NS / 1e9 + 1)
        cached_walk(str(root), quiet=True)
        cached_walk(str(root), quiet=True)
        report = walk_cache.last_report()
        assert report.reused == report.directories


class TestForgedTimestamps:
    """mtime is a value a tool can restore; the cache must not trust it alone.

    `touch -r`, `tar -x`, `rsync -a` and every snapshot restore write a
    directory's mtime back to a recorded value. A restore that adds or removes
    a file can therefore land on exactly the mtime the cache holds, and the
    walk would report `warm` while answering with a file list from before the
    restore. The inode change time moves on any metadata change and no API
    restores it — and it arrives in the same `stat`.
    """

    def test_a_restored_mtime_does_not_hide_an_added_file(self, tmp_path):
        root = _repo(tmp_path)
        _settle(root)
        target = root / "src" / "pkg"
        saved = target.stat()

        (target / "gamma.py").write_text("GAMMA = 3\n")
        os.utime(target, ns=(saved.st_atime_ns, saved.st_mtime_ns))
        assert target.stat().st_mtime_ns == saved.st_mtime_ns

        found = _cached(root)
        assert "src/pkg/gamma.py" in found
        assert found == _uncached(root)

    def test_a_restored_mtime_does_not_hide_a_deleted_file(self, tmp_path):
        root = _repo(tmp_path)
        _settle(root)
        target = root / "src" / "pkg"
        saved = target.stat()

        (target / "beta.py").unlink()
        os.utime(target, ns=(saved.st_atime_ns, saved.st_mtime_ns))

        assert "src/pkg/beta.py" not in _cached(root)


class TestTheWalkGuardsItselfToo:
    """`eligibility.walk` re-states the `extra_ignores` rule its caller enforces.

    `cached_walk` never hands listings to a policy carrying `extra_ignores`,
    so this guard is unreachable through the front door — which is exactly why
    it needs its own test: a direct `walk(root, policy, listings)` caller is
    the case it exists for, and without this the guard is one of the few edits
    the suite would not notice.
    """

    def test_a_negation_in_extra_ignores_is_not_answered_from_a_listing(self, tmp_path):
        root = _repo(tmp_path)
        # `docs/` is excluded by the repo's own rules in this fixture's second
        # state, so the listing that records that verdict is the wrong answer
        # to a call whose own patterns re-include it.
        (root / ".gitignore").write_text("build/\n*.log\ndocs/\n")
        # Out of the racy window BEFORE the listings are taken, or the guard
        # under test never gets asked: a freshly-written tree is re-listed on
        # the next call whatever `extra_ignores` says, and the assertion below
        # would pass for that reason instead of this one.
        _age(root)
        first = eligibility.walk(str(root))
        assert "docs/guide.md" not in _names(first)

        policy = WalkPolicy(extra_ignores=("!docs/", "!docs/**"))
        with_cache = eligibility.walk(str(root), policy, first.listings)
        without = eligibility.walk(str(root), policy)

        assert _names(with_cache) == _names(without)
        assert "docs/guide.md" in _names(with_cache)


class TestDegradation:
    """Every failure costs a warning and a full walk. None costs an answer."""

    @pytest.mark.parametrize("damage", [
        "{not json at all",
        "",
        "[]",
        '{"signature": {}, "directories": {}}',
        '{"signature": null, "directories": null}',
    ])
    def test_a_damaged_cache_still_returns_the_right_files(self, tmp_path, damage):
        root = _repo(tmp_path)
        expected = _cached(root)
        cache_path(root).write_text(damage)

        assert _cached(root) == expected

    def test_a_malformed_entry_discards_the_whole_cache(self, tmp_path):
        """Half a cache would be half a repository, silently."""
        root = _repo(tmp_path)
        expected = _cached(root)
        raw = json.loads(cache_path(root).read_text())
        raw["directories"]["src"] = {"mtime_ns": "not a number"}
        cache_path(root).write_text(json.dumps(raw))

        assert _cached(root) == expected
        assert walk_cache.last_report().mode == "rebuilt"

    def test_a_truncated_cache_is_discarded(self, tmp_path):
        root = _repo(tmp_path)
        expected = _cached(root)
        text = cache_path(root).read_text()
        cache_path(root).write_text(text[: len(text) // 2])

        assert _cached(root) == expected

    def test_a_signature_from_another_build_is_discarded(self, tmp_path):
        root = _repo(tmp_path)
        expected = _cached(root)
        raw = json.loads(cache_path(root).read_text())
        raw["signature"]["matcher_version"] = "999"
        cache_path(root).write_text(json.dumps(raw))

        assert _cached(root) == expected
        assert walk_cache.last_report().mode == "rebuilt"

    def test_an_unwritable_cache_costs_a_warning_not_a_run(self, tmp_path):
        root = _repo(tmp_path)
        neo_dir = root / ".neo"
        neo_dir.mkdir(exist_ok=True)
        neo_dir.chmod(0o500)
        try:
            result = cached_walk(str(root), quiet=True)
        finally:
            neo_dir.chmod(0o700)

        assert _names(result) == _uncached(root)
        report = walk_cache.last_report()
        assert report.mode == "cold"
        assert report.warning is not None

    def test_a_repository_that_cannot_be_walked_yields_nothing(self, tmp_path):
        """No cache, no crash: a missing root is an empty repository.

        And no directory either. `_save` creates `.neo/`, so writing a cache of
        nothing turned a mistyped `--cwd` into a silent `mkdir -p` of a path
        the operator never meant to exist — and pointed at an unmounted
        mountpoint it would create a local directory shadowing the mount.
        """
        missing = tmp_path / "does-not-exist"
        result = cached_walk(str(missing), quiet=True)

        assert result.paths == []
        assert not missing.exists()

    def test_an_empty_cache_is_discarded_rather_than_believed(self, tmp_path):
        """A cache of no directories is damage, not a cache of an empty repo.

        Every walk that reaches a readable root records at least the root, so
        an empty map cannot be a true answer — and believing one made every
        later call in that repository report `incremental` forever, which is
        the signal `rebuilt` exists to preserve.
        """
        root = _repo(tmp_path)
        expected = _cached(root)
        raw = json.loads(cache_path(root).read_text())
        raw["directories"] = {}
        cache_path(root).write_text(json.dumps(raw))

        assert _cached(root) == expected
        assert walk_cache.last_report().mode == "rebuilt"

    def test_an_abandoned_temp_file_is_swept(self, tmp_path):
        root = _repo(tmp_path)
        _cached(root)
        stranded = root / ".neo" / (walk_cache.CACHE_FILENAME + ".abcd.tmp")
        stranded.write_text("{}")
        old = time.time() - walk_cache._TEMP_FILE_TTL_SECONDS - 60
        os.utime(stranded, (old, old))

        (root / "src/pkg/gamma.py").write_text("GAMMA = 3\n")
        _cached(root)

        assert not stranded.exists()

    def test_a_recent_temp_file_is_left_alone(self, tmp_path):
        """A slow concurrent write must never be the thing being deleted."""
        root = _repo(tmp_path)
        _cached(root)
        peer = root / ".neo" / (walk_cache.CACHE_FILENAME + ".peer.tmp")
        peer.write_text("{}")

        (root / "src/pkg/gamma.py").write_text("GAMMA = 3\n")
        _cached(root)

        assert peer.exists()
