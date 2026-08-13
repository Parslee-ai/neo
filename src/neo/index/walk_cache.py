"""The eligibility walk, persisted beside the content index.

#208 gave the repository one walker; #209 stopped the content index rebuilding
itself on every call. What was left was the walk itself, and by the end of
#209 it was the largest single item in a warm Neo invocation: **4.64 s to
re-derive that 9,348 of m365dotnet's files are eligible**, on every call, from
a repository that had not changed (#210).

**The cost is the pattern matching, not the filesystem.** Measured on
m365dotnet before this module existed:

    raw directory listing, pruned     0.11 s   951 directories
    should_ignore over the entries    6.7  s   11,219 calls
    stat over the admitted files      0.10 s   9,378 files

A file's ignore verdict is a function of its NAME and the repository's pattern
set. Neither changes when a file is edited. So the verdicts are what gets
stored, per directory, and a directory's mtime is what says whether they still
hold — on every POSIX filesystem that mtime moves when an entry is created,
deleted or renamed, and does not move when a file's content changes. Which is
exactly the distinction: an edit can change what a file SAYS, never whether it
is eligible.

**Sizes and mtimes are never remembered.** They are read fresh from the `stat`
the walk owes its callers anyway (0.10 s), because the content index next door
uses them as its own freshness stamp. Serving it a remembered mtime would make
an edited file look unedited — one cache's staleness leaking into another's,
which is the worst failure this whole goal could produce.

**JSON, not SQLite, and the reason is the access shape.** The content index is
SQLite because a query touches ten terms of a few hundred thousand and a
parse-whole format would spend the warm budget deserializing postings nothing
asked for. This cache is the opposite: every directory is validated on every
call, so the whole file is read every time — the case JSON is for, and the
case the semantic catalog next door already uses it for. On m365dotnet the
file is ~340 KB and parses in ~10 ms.

**A gitignore edit invalidates by content, not by timestamp.** The signature
holds a hash of the effective pattern list — the shared defaults plus the
repo's own `.gitignore` and `.ignore` — so editing any of them, or upgrading
Neo to a build with a different default list or a different matcher, discards
the cache and walks. That is a full walk on the next call, which is correct
and rare; the alternative, re-deriving which directories a pattern edit could
have affected, is a great deal of machinery guarding against a 5-second cost
paid when someone edits a `.gitignore`.

**Every degradation is loud, and none is fatal.** A corrupt or unreadable
cache is discarded with a warning and the walk runs in full; a cache that
cannot be written costs a warning and nothing else. There is no path here that
can return a wrong file list — the worst outcome available is the walk Neo did
before this module existed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from neo import progress
from neo.eligibility import (
    DirectoryListing,
    WalkPolicy,
    WalkResult,
    load_ignore_patterns,
    walk,
)

logger = logging.getLogger(__name__)

#: Bumped when the on-disk LAYOUT changes. An older cache is discarded rather
#: than migrated: it is derived from the working tree, so rebuilding is always
#: available and always correct, and migration code for a cache is liability
#: with no upside.
SCHEMA_VERSION = 1

#: Bumped when the same path and the same patterns would produce a DIFFERENT
#: verdict than they did before — a change to `should_ignore`, to
#: `compile_glob`, or to the layering in `load_ignore_patterns`. The pattern
#: hash below cannot see those: the patterns are identical, the matcher is not,
#: and every stored verdict is wrong in a way no per-directory stamp can
#: detect.
MATCHER_VERSION = 1

#: Beside `index.json` and `content_index.sqlite3`, in the repository's own
#: `.neo/` — which is in the walker's default ignore list, so Neo never walks
#: its own cache.
CACHE_FILENAME = "walk_cache.json"

@dataclass
class WalkReport:
    """What the walk cost this call, and why.

    The modes are the content index's, meaning the same things, so the two
    lines an operator reads under `--dry-run` can be compared:

    - ``cold``        — no cache existed. First call in this repository.
    - ``rebuilt``     — a cache existed and was unusable: corrupt, or written
                        under a different schema, matcher or ignore-pattern
                        set. A `.gitignore` edit lands here.
    - ``incremental`` — a cache was reused and some directories were re-listed.
    - ``warm``        — a cache was reused and nothing had to be re-listed.
    - ``bypassed``    — the caller passed `--exclude`-style extra ignore
                        patterns, which change what a verdict MEANS, so no
                        cache was read or written.

    ``rebuilt`` is kept distinct from ``cold`` for the reason it is next door:
    both walk everything, but only one of them means something went wrong, and
    a cache thrown away on every call must not read as an ordinary first run.
    """

    mode: str
    directories: int = 0
    reused: int = 0
    rescanned: int = 0
    files: int = 0
    elapsed_ms: float = 0.0
    warning: Optional[str] = None

    def describe(self) -> str:
        """One line for stderr. Never claims work it did not do."""
        seconds = self.elapsed_ms / 1000.0
        if self.mode in ("cold", "rebuilt"):
            why = (
                "first run for this repository"
                if self.mode == "cold"
                else "previous cache discarded"
            )
            head = (
                f"Eligibility walk: full walk of {self.directories} directories "
                f"({why}), {self.files} files eligible, in {seconds:.1f}s"
            )
        elif self.mode == "incremental":
            head = (
                f"Eligibility walk: re-listed {self.rescanned} of "
                f"{self.directories} directories, {self.files} files eligible, "
                f"in {seconds:.1f}s"
            )
        elif self.mode == "warm":
            head = (
                f"Eligibility walk: read warm, {self.directories} directories "
                f"unchanged, {self.files} files eligible ({seconds:.1f}s)"
            )
        else:
            head = (
                f"Eligibility walk: full walk of {self.directories} directories "
                f"(not cacheable for this call), {self.files} files eligible, "
                f"in {seconds:.1f}s"
            )
        return head if not self.warning else f"{head} - {self.warning}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "directories": self.directories,
            "reused": self.reused,
            "rescanned": self.rescanned,
            "files": self.files,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "warning": self.warning,
            "summary": self.describe(),
        }


#: The most recent cached walk in this process, for reporting surfaces that run
#: after the gather has returned — `--dry-run`'s JSON report in particular,
#: where `--quiet` is implied and the stderr note the operator would otherwise
#: read is suppressed. Process-global for the same reason `progress` is: it is
#: a report about the run, read once, several layers up from where it is made.
_LAST_REPORT: Optional[WalkReport] = None


def last_report() -> Optional[WalkReport]:
    """The `WalkReport` from this process's most recent `cached_walk`, if any."""
    return _LAST_REPORT


def cache_path(repo_root: str) -> Path:
    """Where this repository's cache lives."""
    return Path(repo_root).resolve() / ".neo" / CACHE_FILENAME


def signature(repo_root: str) -> dict[str, str]:
    """What this build of Neo, in this repository, would write.

    Any mismatch discards the cache. `patterns` is hashed rather than stored
    whole because the point is equality, not readability, and a large repo's
    `.gitignore` would otherwise dominate the file.
    """
    patterns = "\n".join(load_ignore_patterns(repo_root))
    return {
        "schema_version": str(SCHEMA_VERSION),
        "matcher_version": str(MATCHER_VERSION),
        "patterns_sha256": hashlib.sha256(patterns.encode("utf-8")).hexdigest(),
    }


def _load(path: Path, expected: dict[str, str]) -> tuple[
    Optional[dict[str, DirectoryListing]], Optional[str]
]:
    """`(listings, why_not)` — the stored listings, or why there are none.

    `why_not` is `None` when the file simply does not exist (an ordinary first
    run) and a sentence when something was there and could not be used. The
    caller turns that difference into `cold` versus `rebuilt`, which is the
    difference between "this is the first call" and "this repository is
    throwing its cache away on every call".
    """
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as exc:
        return None, f"cache unreadable ({exc})"

    if not isinstance(raw, dict):
        return None, "cache is not an object"
    if raw.get("signature") != expected:
        return None, "ignore rules or Neo version changed"

    directories = raw.get("directories")
    if not isinstance(directories, dict):
        return None, "cache holds no directory listings"

    listings: dict[str, DirectoryListing] = {}
    try:
        for rel_dir, entry in directories.items():
            listings[rel_dir] = DirectoryListing(
                mtime_ns=int(entry["mtime_ns"]),
                scanned_at_ns=int(entry["scanned_at_ns"]),
                subdirs=tuple(entry["subdirs"]),
                files=tuple(entry["files"]),
                symlinks=tuple(entry.get("symlinks", ())),
                excluded_dirs=int(entry.get("excluded_dirs", 0)),
                excluded_files=int(entry.get("excluded_files", 0)),
            )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        # A malformed entry is malformed data, not a missing feature: half a
        # cache would give half a repository, silently.
        return None, f"cache entry malformed ({exc})"
    return listings, None


def _save(path: Path, sig: dict[str, str], listings: dict[str, DirectoryListing]) -> Optional[str]:
    """Write the cache atomically. Returns a warning string on failure, else None.

    Temp file plus `os.replace`, in the destination directory so the replace is
    atomic, so a reader can never see a half-written cache and a crash mid-write
    leaves the previous one intact. Two Neo processes racing here is fine: both
    write the same derived facts and the loser's work is simply superseded.
    """
    payload = {
        "signature": sig,
        "directories": {
            rel_dir: {
                "mtime_ns": listing.mtime_ns,
                "scanned_at_ns": listing.scanned_at_ns,
                "subdirs": list(listing.subdirs),
                "files": list(listing.files),
                "symlinks": list(listing.symlinks),
                "excluded_dirs": listing.excluded_dirs,
                "excluded_files": listing.excluded_files,
            }
            for rel_dir, listing in listings.items()
        },
    }
    tmp_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=CACHE_FILENAME + ".", suffix=".tmp"
        )
        tmp_path = Path(tmp_name)
        with os.fdopen(handle, "w") as out:
            json.dump(payload, out)
        # `mkstemp` creates 0600, which the cache must not keep: a checkout
        # shared by two accounts would leave the second unable to read it, and
        # an unreadable cache is a full walk on every call, forever, with only
        # a warning to say so. Match the store next door, which SQLite creates
        # world-readable.
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
        tmp_path = None
    except (OSError, TypeError, ValueError) as exc:
        return f"cache not written ({exc})"
    finally:
        if tmp_path is not None:
            # `os.replace` never ran, so the temp file is ours to remove. A
            # SIGKILL between `mkstemp` and `replace` runs no handler and does
            # strand one — hence the sweep below, which is the same lesson
            # `FactStore._reap_stale_temp_files` was written for.
            try:
                tmp_path.unlink()
            except OSError:
                pass
    _reap_stale_temp_files(path.parent)
    return None


#: How long an abandoned temp file is left alone before it is swept. Long
#: enough that a slow concurrent write is never the thing being deleted.
_TEMP_FILE_TTL_SECONDS = 24 * 3600


def _reap_stale_temp_files(directory: Path) -> None:
    """Remove this module's own abandoned temp files older than the TTL."""
    cutoff = time.time() - _TEMP_FILE_TTL_SECONDS
    try:
        stale = list(directory.glob(CACHE_FILENAME + ".*.tmp"))
    except OSError:
        return
    for leftover in stale:
        try:
            if leftover.stat().st_mtime < cutoff:
                leftover.unlink()
        except OSError:
            continue


def cached_walk(
    repo_root: str,
    policy: Optional[WalkPolicy] = None,
    *,
    quiet: bool = False,
) -> WalkResult:
    """`eligibility.walk`, accelerated by and refreshed into the on-disk cache.

    Always returns the same `WalkResult` an uncached walk would: the cache can
    make a call cheaper, never different. Every failure mode — no cache, a
    corrupt one, one written under different rules, an unwritable `.neo/` —
    degrades to the full walk and says so.
    """
    started = time.time()
    policy = policy or WalkPolicy()

    if policy.extra_ignores:
        # See the matching guard in `eligibility.walk`: caller-supplied
        # patterns are appended after the repository's own, and gitignore is
        # last-match-wins, so a stored verdict is not an answer to this call's
        # question. Nothing in Neo takes this path today — `--exclude` is
        # applied after the walk by `context_gatherer.filter_candidates`, so
        # the corpus stays a property of the repository — and it exists so that
        # a future caller which does gets a correct answer rather than a fast
        # wrong one.
        result = walk(repo_root, policy)
        return _finish(result, "bypassed", started, quiet, None)

    path = cache_path(repo_root)
    sig = signature(repo_root)
    listings, why_not = _load(path, sig)
    if why_not:
        logger.warning("Eligibility walk cache at %s discarded: %s", path, why_not)

    if listings is None:
        # BEFORE the walk, not after. A first call on a large repository
        # spends seconds here and silence for seconds is indistinguishable
        # from a hang — the same reason the content index announces its cold
        # build. `neo --index` warms this cache too, but it is optional: any
        # invocation pays the one-time cost and every later one is warm.
        if not quiet:
            progress.note(
                "Eligibility walk: no cache for this repository "
                f"({why_not or 'first run'}); walking the tree once"
            )

    result = walk(repo_root, policy, listings)

    warning = None
    if listings is None or result.rescanned_dirs:
        warning = _save(path, sig, result.listings or {})

    if listings is None:
        mode = "rebuilt" if why_not else "cold"
    elif result.rescanned_dirs:
        mode = "incremental"
    else:
        mode = "warm"
    return _finish(result, mode, started, quiet, warning)


def _finish(
    result: WalkResult,
    mode: str,
    started: float,
    quiet: bool,
    warning: Optional[str],
) -> WalkResult:
    global _LAST_REPORT
    report = WalkReport(
        mode=mode,
        directories=result.reused_dirs + result.rescanned_dirs,
        reused=result.reused_dirs,
        rescanned=result.rescanned_dirs,
        files=len(result.paths),
        elapsed_ms=(time.time() - started) * 1000.0,
        warning=warning,
    )
    _LAST_REPORT = report
    if not quiet:
        progress.note(report.describe())
    return result
