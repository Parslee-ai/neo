"""The one answer to "which indexed files went stale since we last looked?".

Both on-disk indexes — the semantic catalog (`index.project_index`) and the
keyword content index (`index.content_index`) — persist a per-file stamp and
compare it against the filesystem on every open. That comparison is the same
question twice, so it is written once here.

This module owns the COMPARISON only, never the storage. The two indexes keep
their own files (`.neo/index.json`, `.neo/content_index.sqlite3`) because they
hold different artifacts on different write schedules; what they must not do
is disagree about what "changed" means.

Two stamp strengths, and the difference is the whole performance story:

- **`content_hash`** is authoritative. Two files with the same bytes are the
  same document no matter what their metadata says.
- **`size` + `mtime_ns`** is the cheap pre-filter. The walk already stats every
  file, so this costs nothing extra, whereas hashing means reading every byte
  of the repository on every invocation — precisely the per-call cost this
  index exists to remove.

The pre-filter is used to decide *what to hash*, never to decide that a file
changed. A file whose size and mtime both match is treated as unchanged; a
file where either differs is hashed, and only a hash mismatch counts as a real
change. So a `touch` with no edit updates the stamp and re-tokenizes nothing,
and an edit that preserves size is caught because mtime moved. The residual
blind spot is an edit that preserves size AND restores mtime — which requires
deliberate effort — and `force_rehash` exists for callers who cannot accept
even that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional

#: A hasher takes an absolute path and returns a hex digest, or `""` when the
#: file cannot be read. `eligibility.file_content_hash` is the implementation;
#: it is injected rather than imported so a caller can supply a cheaper one and
#: so tests can count calls.
Hasher = Callable[[str], str]


#: `size` / `mtime_ns` value meaning "this store never recorded one". It is
#: NEGATIVE rather than zero because zero is a legal size and a legal mtime,
#: and it is checked explicitly rather than left to compare unequal against
#: real values: an earlier cut stamped both the store AND the candidate with
#: the sentinel, so `-1 == -1` took the cheap path and reported every changed
#: file as unchanged. A sentinel that can be mistaken for data is not one.
UNRECORDED = -1


@dataclass(frozen=True)
class FileStamp:
    """What was recorded about one file the last time it was indexed.

    `size` or `mtime_ns` may be `UNRECORDED`, which disables the cheap
    pre-filter for that file and forces a hash — the honest behaviour for a
    store that persists hashes alone.
    """

    size: int
    mtime_ns: int
    content_hash: str

    @property
    def has_cheap_stamp(self) -> bool:
        return self.size >= 0 and self.mtime_ns >= 0


@dataclass(frozen=True)
class Candidate:
    """One file the walk currently admits, with its cheap stamp already read.

    `abs_path` is what a hasher is handed; `rel_path` is the identity a stamp
    is keyed on, so the same repository indexed through two different absolute
    prefixes (a worktree, a container mount) still matches its own history.
    """

    abs_path: str
    rel_path: str
    size: int
    mtime_ns: int


@dataclass
class Changes:
    """The verdict: what to index, what to drop, what to leave alone.

    `touched` is deliberately separate from `changed`. Both need a stamp
    rewrite, but only `changed` needs the file re-read and re-tokenized, and
    conflating them turns `touch *` into a full rebuild. The two are reported
    separately for the same reason a truncation is reported: a caller that
    prints "re-indexed 400 files" when it re-tokenized none is lying about its
    own work.
    """

    added: list[Candidate] = field(default_factory=list)
    changed: list[Candidate] = field(default_factory=list)
    touched: list[Candidate] = field(default_factory=list)
    unchanged: list[Candidate] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)

    @property
    def needs_indexing(self) -> list[Candidate]:
        """Files whose CONTENT must be read again."""
        return self.added + self.changed

    @property
    def is_clean(self) -> bool:
        """True when nothing at all has to be written."""
        return not (self.added or self.changed or self.touched or self.removed)


def detect_changes(
    stored: Mapping[str, FileStamp],
    current: Iterable[Candidate],
    hasher: Hasher,
    *,
    force_rehash: bool = False,
) -> Changes:
    """Compare the recorded stamps against what is on disk now.

    **A file that cannot be hashed stays in the result**, carrying the empty
    hash the hasher returned. An earlier cut dropped it — which reads as the
    careful choice and is not, because the caller's `removed` set is
    "everything the walk no longer admits", so dropping a candidate the walk
    DID admit deletes it from the index. Measured: `chmod 000` on a file made
    it vanish from the corpus permanently, while the per-call index this
    replaces still ranked it on its path tokens (its content simply read as
    empty). Permission was withdrawn from the CONTENT; the name is still a
    real name in the repository, and losing it is a silent retrieval
    regression.

    The empty hash is safe as a comparison value precisely because stamps are
    keyed by path: `"" == ""` is only ever asked of one file against its own
    previous state, where it correctly means "still unreadable, nothing to
    re-tokenize". It never compares two different files.

    A file that vanished between the walk and here takes the same path and
    self-heals: it is re-indexed to its path tokens now, and the next walk
    does not list it, so it lands in `removed` then.
    """
    changes = Changes()
    seen: set[str] = set()

    for candidate in current:
        seen.add(candidate.rel_path)
        stamp = stored.get(candidate.rel_path)

        if (
            stamp is not None
            and not force_rehash
            and stamp.has_cheap_stamp
            and stamp.size == candidate.size
            and stamp.mtime_ns == candidate.mtime_ns
        ):
            changes.unchanged.append(candidate)
            changes.hashes[candidate.rel_path] = stamp.content_hash
            continue

        digest = hasher(candidate.abs_path)
        changes.hashes[candidate.rel_path] = digest
        if stamp is None:
            changes.added.append(candidate)
        elif stamp.content_hash != digest:
            changes.changed.append(candidate)
        else:
            changes.touched.append(candidate)

    changes.removed = sorted(set(stored) - seen)
    return changes


def stale_paths(
    stored: Mapping[str, FileStamp],
    current: Iterable[Candidate],
    hasher: Hasher,
) -> tuple[list[str], list[str]]:
    """`(changed_or_added, removed)` as plain paths, for callers wanting names.

    `ProjectIndex.check_staleness` speaks this shape; the content index wants
    the richer `Changes`. One comparison, two projections.
    """
    changes = detect_changes(stored, current, hasher)
    return (
        [c.rel_path for c in changes.needs_indexing],
        list(changes.removed),
    )


def stamps_from_hashes(
    hashes: Mapping[str, str],
    sizes: Optional[Mapping[str, int]] = None,
) -> dict[str, FileStamp]:
    """Adapt a bare `rel_path -> hash` map into stamps.

    The semantic catalog persists hashes only, so its stamps carry no cheap
    pre-filter and every file it tracks gets hashed. That is the honest
    translation: claiming a size and mtime it never recorded would make a
    changed file look unchanged.
    """
    sizes = sizes or {}
    return {
        path: FileStamp(
            size=sizes.get(path, UNRECORDED),
            mtime_ns=UNRECORDED,
            content_hash=digest,
        )
        for path, digest in hashes.items()
    }
