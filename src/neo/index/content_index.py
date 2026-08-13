"""The keyword content index, persisted beside the semantic catalog.

`neo.file_retrieval` introduced BM25 over file CONTENT and took file selection
from MRR 0.05 to 0.61 on the largest flagship. It also read and tokenized
every eligible file on every invocation, and on m365dotnet (4,272 C# files)
that is the whole cost of a Neo call: a median of ~60 s wall and ~1.8 GB peak
RSS to assemble context, before a single token reaches the model (#195).

Nothing about that work is per-call. The corpus changes when the repository
changes, which between two invocations is usually not at all. So it moves to
disk, next to the semantic catalog in the repository's own `.neo/`, and every
invocation pays only for the files that actually moved.

**SQLite, not JSON, and the reason is the metric.** The catalog next door is
JSON because it is read whole — every chunk's embedding participates in every
query. A keyword index is the opposite: a prompt touches the postings of ten
or twenty terms out of a few hundred thousand, so a format that must be parsed
whole to answer anything would spend the entire warm budget deserializing
postings no query asked for. Measured on m365dotnet the corpus is ~2.5 M
postings; a term-indexed table answers from a handful of B-tree probes and
keeps resident memory in the tens of megabytes, where holding the same
postings as Python lists of ints would cost several hundred. The store still
holds INDEX ARTIFACTS ONLY — postings, lengths, stamps and hashes, never a
file body. Delivery reads whole files fresh from disk, as it always did.

**Ranking parity is a hard requirement, not an aspiration.** The scorer here
reimplements nothing: `K1`, `B` and the IDF formula are imported from
`neo.memory.bm25`, and the document is built by the same expression the
per-call index used — `code_tokens(rel_path) * PATH_TOKEN_WEIGHT +
code_tokens(content)`. `tests/test_content_index.py` scores the same corpus
both ways and asserts the two agree to floating-point tolerance, because a
persistence change that quietly re-ranked would be indistinguishable from a
retrieval regression.

**A filtered call gets filtered statistics.** N, document frequency and
average document length are computed over the files that call admitted, not
over the whole indexed repository — because the per-call index computed them
over exactly that set, and BM25's score depends on all three. Repo-global
statistics were the first cut and they are the more defensible IR design in
the abstract; they are also a re-ranking. Measured: unflagged runs matched
`main` exactly while `--exts py` changed all 25 selected lines. Parity with
the ranker being replaced is the requirement here, so the subset is the
corpus and `scores(prompt, candidates)` takes both from `candidates`.

**Freshness is inline and cheap.** `refresh()` takes the eligibility walk's
own output (there is no second walk, and no file-walking or exclusion logic
lives in this module), compares stamps via `index.freshness`, and re-tokenizes
only what changed. A file the walker no longer admits — newly gitignored, or
deleted — has its postings dropped in the same pass, so a stale index can
never answer with a file the walker excludes.

**Every degradation is loud and none is fatal.** A missing store cold-builds
and says so before it starts, because a silent multi-minute stall on a large
repository is indistinguishable from a hang. A corrupt store is discarded and
rebuilt with a warning. A store that cannot be written — a read-only checkout,
a peer process holding the write lock — still serves scores for this call from
memory and warns; it does not raise, and it does not silently return an empty
ranking, which would look exactly like "no file matches your prompt".
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from neo import progress
from neo.eligibility import file_content_hash
from neo.file_retrieval import (
    MAX_INDEXED_CHARS,
    PATH_TOKEN_WEIGHT,
    _read,
    code_tokens,
)
from neo.index.freshness import Candidate, FileStamp, detect_changes
from neo.memory.bm25 import B, K1

logger = logging.getLogger(__name__)

#: Bumped when the on-disk LAYOUT changes. An older store is discarded rather
#: than migrated: it is a derived cache of the working tree, so rebuilding is
#: always available and always correct, and migration code for a cache is
#: liability with no upside.
SCHEMA_VERSION = 3

#: Bumped when the same file would produce DIFFERENT TOKENS than it did
#: before. That covers `code_tokens` itself, `PATH_TOKEN_WEIGHT`, the
#: `MAX_INDEXED_CHARS` read limit and the binary-file rule in `_read` — change
#: any of them and every stored posting is wrong in a way no per-file hash can
#: detect, because the files did not change, the tokenizer did. The stored
#: value records the read limit and path weight alongside the version so a
#: mismatch is visible in the store rather than only in a constant.
TOKENIZER_VERSION = 1

#: Files between progress notes during a cold build. A build of a large
#: repository takes minutes; silence for minutes is the failure mode this
#: exists to prevent, and one line per few hundred files is enough to show
#: motion without flooding stderr.
_PROGRESS_EVERY = 250

#: How long to wait for a peer process's write transaction before giving up
#: and serving this call from memory. Two Neo invocations in the same
#: repository is an ordinary situation (an editor plugin and a shell), and the
#: loser of that race must degrade, not fail.
_BUSY_TIMEOUT_MS = 5_000

#: The store lives inside the repository's own `.neo/`, beside `index.json`.
INDEX_FILENAME = "content_index.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id           INTEGER PRIMARY KEY,
    rel_path     TEXT NOT NULL UNIQUE,
    size         INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    doc_len      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS terms (
    id   INTEGER PRIMARY KEY,
    term TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS postings (
    term_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    tf      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS postings_term ON postings(term_id);
CREATE INDEX IF NOT EXISTS postings_file ON postings(file_id);
"""


@dataclass
class IndexReport:
    """What the index did on this invocation, in the words the operator needs.

    `mode` is the headline and the four values are not interchangeable:

    - ``cold``        — no usable store existed; the whole repository was read.
    - ``rebuilt``     — a store existed but was unusable (corrupt, or written
                        by a different schema/tokenizer) and was discarded.
    - ``incremental`` — a store was reused and some files were re-read.
    - ``warm``        — a store was reused and nothing had to be re-read.
    - ``memory``      — the store could not be used at all this call; the index
                        was built in memory and not persisted.

    ``rebuilt`` is kept distinct from ``cold`` on purpose. Both read every
    file, but only one of them means something went wrong, and reporting a
    discarded corrupt store as an ordinary first run hides a store that is
    being thrown away on every single call.
    """

    mode: str
    total_files: int = 0
    indexed: int = 0
    added: int = 0
    changed: int = 0
    touched: int = 0
    removed: int = 0
    elapsed_ms: float = 0.0
    warning: Optional[str] = None

    def describe(self) -> str:
        """One line for stderr. Never claims work it did not do."""
        seconds = self.elapsed_ms / 1000.0
        if self.mode in ("cold", "rebuilt"):
            why = (
                "first run for this repository"
                if self.mode == "cold"
                else "previous store discarded"
            )
            head = (
                f"Content index: cold build of {self.total_files} files "
                f"({why}) in {seconds:.1f}s"
            )
        elif self.mode == "incremental":
            parts = []
            if self.added:
                parts.append(f"{self.added} added")
            if self.changed:
                parts.append(f"{self.changed} changed")
            if self.removed:
                parts.append(f"{self.removed} removed")
            if self.touched:
                parts.append(f"{self.touched} touched (content identical)")
            head = (
                f"Content index: incrementally updated {self.indexed} of "
                f"{self.total_files} files ({', '.join(parts)}) in {seconds:.1f}s"
            )
        elif self.mode == "warm":
            head = (
                f"Content index: read warm, {self.total_files} files "
                f"unchanged ({seconds:.1f}s)"
            )
        else:
            head = (
                f"Content index: built in memory for this call only, "
                f"{self.total_files} files in {seconds:.1f}s"
            )
        return head if not self.warning else f"{head} - {self.warning}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "total_files": self.total_files,
            "indexed": self.indexed,
            "added": self.added,
            "changed": self.changed,
            "touched": self.touched,
            "removed": self.removed,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "warning": self.warning,
            "summary": self.describe(),
        }


#: The most recent refresh in this process, for reporting surfaces that run
#: after the gather has returned — `--dry-run`'s JSON report in particular,
#: where `--quiet` is implied and the stderr note the operator would otherwise
#: read is suppressed. Process-global for the same reason `progress` is: it is
#: a report about the run, read once, five layers up from where it is
#: produced.
_LAST_REPORT: Optional[IndexReport] = None


def last_report() -> Optional[IndexReport]:
    """The `IndexReport` from this process's most recent `refresh`, if any."""
    return _LAST_REPORT


def _document(rel_path: str, abs_path: str) -> list[str]:
    """The token list for one file — the ONE definition of a document.

    Identical by construction to what `file_retrieval.FileIndex` built per
    call. A file that reads as binary or cannot be decoded still contributes
    its path tokens, exactly as before: it is a real name in the repository
    even when its bytes are not text.
    """
    return code_tokens(rel_path) * PATH_TOKEN_WEIGHT + code_tokens(_read(abs_path))


class ContentIndex:
    """Persistent BM25 over repository file content.

    Construct, `refresh()` with the eligibility walk's output, then `scores()`
    as many times as needed. `close()` when done; the object is also a context
    manager.
    """

    def __init__(self, repo_root: str, db_path: Optional[str] = None):
        self.repo_root = Path(repo_root).resolve()
        self.neo_dir = self.repo_root / ".neo"
        self.db_path = Path(db_path) if db_path else self.neo_dir / INDEX_FILENAME
        self._conn: Optional[sqlite3.Connection] = None
        self._persistent = True
        #: rel_path -> (file_id, doc_len), loaded once per refresh. A few
        #: thousand small tuples; the postings themselves stay on disk.
        self._files: dict[str, tuple[int, int]] = {}
        self._n = 0
        self._avgdl = 0.0
        self._vocab: dict[str, int] = {}
        #: Only populated on the fallback path; see `_build_in_memory`.
        self._memory_index: Optional[Any] = None
        #: Set by `_connect` when it threw a corrupt store away, so the report
        #: says `rebuilt` rather than `cold`. Both read every file; only one of
        #: them means something went wrong, and a store being discarded on
        #: every single call must not read as an ordinary first run.
        self._rebuilt_from_corruption = False

    # ---------------------------------------------------------------- store

    def _connect(self, *, discard_corrupt: bool = True) -> Optional[sqlite3.Connection]:
        """Open (creating if needed) the on-disk store, or None if impossible.

        Returning None rather than raising is the contract: every caller of
        this module has a working fallback (build in memory, score, do not
        persist), and a read-only checkout or a full disk must cost a warning,
        not a run.

        A CORRUPT store is not that case and must not be allowed to reach the
        fallback, because the fallback is permanent — a store that fails to
        open is never repaired, so every future invocation in that repository
        would silently pay the full per-call rebuild this module exists to
        remove. Corruption is detected here (SQLite raises while opening, not
        while querying), the file is deleted, and the open is retried once.

        The two are told apart by exception CLASS, which for `sqlite3` is
        precise enough to rely on: a malformed or non-database file raises
        `DatabaseError` itself, while a locked, read-only or full store raises
        the `OperationalError` subclass. Anything unrecognized keeps the
        conservative behaviour and degrades rather than deleting a file it
        does not understand.
        """
        if self._conn is not None:
            return self._conn
        try:
            self.neo_dir.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=_BUSY_TIMEOUT_MS / 1000)
            conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
            # WAL lets a reader run while a peer writes, which is the common
            # two-process case. It is a no-op on filesystems that refuse it.
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.Error:
                pass
            conn.executescript(_SCHEMA)
            self._conn = conn
            return conn
        except sqlite3.DatabaseError as exc:
            if discard_corrupt and not isinstance(exc, sqlite3.OperationalError):
                logger.warning(
                    "Content index at %s is corrupt (%s); discarding and rebuilding",
                    self.db_path,
                    exc,
                )
                self._discard_store()
                self._rebuilt_from_corruption = True
                return self._connect(discard_corrupt=False)
            logger.warning("Content index unavailable at %s: %s", self.db_path, exc)
            return None
        except (sqlite3.Error, OSError) as exc:
            logger.warning("Content index unavailable at %s: %s", self.db_path, exc)
            return None

    def _discard_store(self) -> None:
        """Delete the store on disk so the next open starts clean."""
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
        for suffix in ("", "-wal", "-shm"):
            try:
                os.unlink(str(self.db_path) + suffix)
            except OSError:
                pass

    def _stored_signature(self, conn: sqlite3.Connection) -> dict[str, str]:
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
        return {key: value for key, value in rows}

    def _signature(self) -> dict[str, str]:
        """What this build of Neo would write. Any mismatch invalidates."""
        return {
            "schema_version": str(SCHEMA_VERSION),
            "tokenizer_version": str(TOKENIZER_VERSION),
            "path_token_weight": str(PATH_TOKEN_WEIGHT),
            "max_indexed_chars": str(MAX_INDEXED_CHARS),
        }

    def _load_stamps(self, conn: sqlite3.Connection) -> dict[str, FileStamp]:
        rows = conn.execute(
            "SELECT rel_path, size, mtime_ns, content_hash FROM files"
        ).fetchall()
        return {
            rel_path: FileStamp(size=size, mtime_ns=mtime_ns, content_hash=digest)
            for rel_path, size, mtime_ns, digest in rows
        }

    # --------------------------------------------------------------- refresh

    def refresh(self, paths: Sequence[Any], *, quiet: bool = False) -> IndexReport:
        """Bring the store level with `paths` and load the query-side state.

        `paths` is the eligibility walk's own output — `EligiblePath` records,
        or anything carrying `path` / `rel_path` / `size` / `mtime_ns`. This
        module never walks; it is handed what the one walker admitted, which
        is what makes "a file the walker excludes cannot remain queryable"
        true by construction rather than by a second exclusion list.
        """
        global _LAST_REPORT
        started = time.time()
        candidates = _as_candidates(paths)

        conn = self._connect()
        if conn is None:
            report = self._build_in_memory(
                candidates, started, "content index could not be opened"
            )
            _LAST_REPORT = report
            if not quiet:
                progress.note(report.describe())
            return report

        forced = "rebuilt" if self._rebuilt_from_corruption else None
        # Consumed, not carried: leaving it set made a REUSED instance report
        # `rebuilt` and wipe the whole repository on its second refresh, long
        # after the corruption that justified the first one.
        self._rebuilt_from_corruption = False
        try:
            report = self._refresh_persistent(
                conn, candidates, started, quiet=quiet, forced_cold=forced
            )
        except sqlite3.OperationalError as exc:
            # Locked by a peer, read-only filesystem, disk full. Serve this
            # call from memory rather than failing it — and, above all, DO NOT
            # fall through to the corruption handler below.
            #
            # `OperationalError` is a SUBCLASS of `DatabaseError`, so an
            # `except DatabaseError` written first catches lock contention and
            # deletes the store, which is the worst available response: it
            # throws away a valid index (109 MB and 122 s of rebuild on
            # m365dotnet) AND destroys the peer, whose transaction then commits
            # into an unlinked inode and vanishes without an error. Two Neo
            # invocations in one repository is an ordinary situation — an
            # editor plugin and a shell — not an exotic one. This clause is
            # ordered FIRST because Python matches except clauses in order and
            # subclass-before-superclass is the only ordering that works;
            # `test_a_peer_holding_the_write_lock_does_not_delete_the_store`
            # fails if it is ever moved or merged.
            logger.warning("Content index not writable (%s); using memory", exc)
            report = self._build_in_memory(
                candidates, started, f"store busy or not writable ({exc})"
            )
        except sqlite3.DatabaseError as exc:
            # Genuine corruption: a malformed image, never a lock. Not worth
            # diagnosing — the store is derived from the working tree, so
            # throwing it away loses nothing but the time to rebuild.
            logger.warning("Content index unusable (%s); rebuilding", exc)
            self._discard_store()
            conn = self._connect()
            if conn is None:
                report = self._build_in_memory(
                    candidates, started, "content index could not be rebuilt"
                )
            else:
                try:
                    report = self._refresh_persistent(
                        conn, candidates, started, quiet=quiet, forced_cold="rebuilt"
                    )
                except sqlite3.Error as second:
                    report = self._build_in_memory(
                        candidates, started, f"content index write failed ({second})"
                    )
        except sqlite3.Error as exc:
            logger.warning("Content index unavailable (%s); using memory", exc)
            report = self._build_in_memory(
                candidates, started, f"store unavailable ({exc})"
            )

        _LAST_REPORT = report
        if not quiet:
            progress.note(report.describe())
        return report

    def _refresh_persistent(
        self,
        conn: sqlite3.Connection,
        candidates: list[Candidate],
        started: float,
        *,
        quiet: bool = False,
        forced_cold: Optional[str] = None,
    ) -> IndexReport:
        signature = self._signature()
        stored = self._stored_signature(conn)
        mode = forced_cold
        if mode is None and stored and stored != signature:
            # A tokenizer or layout change makes every posting wrong while
            # every file hash still matches, so per-file freshness cannot see
            # it. Wipe rather than reconcile.
            logger.info(
                "Content index signature changed (%s -> %s); rebuilding",
                stored,
                signature,
            )
            mode = "rebuilt"

        wipe = mode is not None
        if wipe:
            stamps: dict[str, FileStamp] = {}
        else:
            stamps = self._load_stamps(conn)
            if not stamps:
                mode = "cold"

        # BEFORE `detect_changes`, which sha256s the whole repository on a cold
        # build. Announcing after it meant the promised "no silent stall" still
        # opened with a full-repo byte read in silence — small on a toy repo,
        # a real wait on m365dotnet. The count is the candidate count rather
        # than the work count for the same reason: the work count is not known
        # until the hashing this line exists to announce has finished.
        if mode in ("cold", "rebuilt") and candidates and not quiet:
            progress.note(
                f"Content index: no usable index for this repository - "
                f"building one over {len(candidates)} files. This runs once; "
                f"later calls update only what changed."
            )

        changes = detect_changes(stamps, candidates, file_content_hash)
        work = changes.needs_indexing

        if mode is None:
            mode = "incremental" if not changes.is_clean else "warm"

        # A warm call must not open a WRITE transaction. Rewriting the
        # (unchanged) signature unconditionally made every single invocation a
        # writer, so two ordinary overlapping calls contended for the store on
        # the steady-state path rather than only during a rebuild.
        if wipe or work or changes.touched or changes.removed or stored != signature:
            with conn:  # one transaction; SQLite gives us the atomicity
                if wipe:
                    conn.execute("DELETE FROM postings")
                    conn.execute("DELETE FROM files")
                    conn.execute("DELETE FROM terms")
                # AFTER the wipe, never before. Loading the vocabulary first
                # and then emptying `terms` left `_index_file` believing every
                # term already existed, so it skipped the inserts and wrote
                # postings pointing at deleted term ids — a rebuilt index that
                # answered nothing.
                self._vocab = {
                    term: term_id
                    for term_id, term in conn.execute("SELECT id, term FROM terms")
                }
                for rel_path in changes.removed:
                    self._delete_file(conn, rel_path)
                for index, candidate in enumerate(work, 1):
                    self._index_file(
                        conn, candidate, changes.hashes.get(candidate.rel_path, "")
                    )
                    if (not quiet and mode in ("cold", "rebuilt")
                            and index % _PROGRESS_EVERY == 0):
                        progress.note(
                            f"Content index: {index}/{len(work)} files tokenized"
                        )
                for candidate in changes.touched:
                    # Content identical, metadata moved. Stamp only —
                    # re-reading would be work with a known-empty result.
                    conn.execute(
                        "UPDATE files SET size = ?, mtime_ns = ? WHERE rel_path = ?",
                        (candidate.size, candidate.mtime_ns, candidate.rel_path),
                    )
                conn.executemany(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    sorted(signature.items()),
                )

        self._load_query_state(conn)
        self._persistent = True
        return IndexReport(
            mode=mode,
            total_files=self._n,
            indexed=len(work),
            added=len(changes.added),
            changed=len(changes.changed),
            touched=len(changes.touched),
            removed=len(changes.removed),
            elapsed_ms=(time.time() - started) * 1000.0,
        )

    def _delete_file(self, conn: sqlite3.Connection, rel_path: str) -> None:
        row = conn.execute(
            "SELECT id FROM files WHERE rel_path = ?", (rel_path,)
        ).fetchone()
        if row is None:
            return
        conn.execute("DELETE FROM postings WHERE file_id = ?", (row[0],))
        conn.execute("DELETE FROM files WHERE id = ?", (row[0],))

    def _index_file(
        self, conn: sqlite3.Connection, candidate: Candidate, digest: str
    ) -> None:
        tokens = _document(candidate.rel_path, candidate.abs_path)
        counts = Counter(tokens)

        row = conn.execute(
            "SELECT id FROM files WHERE rel_path = ?", (candidate.rel_path,)
        ).fetchone()
        if row is None:
            cursor = conn.execute(
                "INSERT INTO files(rel_path, size, mtime_ns, content_hash, doc_len)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    candidate.rel_path,
                    candidate.size,
                    candidate.mtime_ns,
                    digest,
                    len(tokens),
                ),
            )
            file_id = cursor.lastrowid
        else:
            file_id = row[0]
            conn.execute(
                "UPDATE files SET size = ?, mtime_ns = ?, content_hash = ?,"
                " doc_len = ? WHERE id = ?",
                (candidate.size, candidate.mtime_ns, digest, len(tokens), file_id),
            )
            conn.execute("DELETE FROM postings WHERE file_id = ?", (file_id,))

        new_terms = [(term,) for term in counts if term not in self._vocab]
        if new_terms:
            conn.executemany("INSERT OR IGNORE INTO terms(term) VALUES (?)", new_terms)
            placeholders = ",".join("?" * len(new_terms))
            for term_id, term in conn.execute(
                f"SELECT id, term FROM terms WHERE term IN ({placeholders})",
                [t[0] for t in new_terms],
            ):
                self._vocab[term] = term_id

        conn.executemany(
            "INSERT INTO postings(term_id, file_id, tf) VALUES (?, ?, ?)",
            [(self._vocab[term], file_id, tf) for term, tf in counts.items()],
        )

    def _load_query_state(self, conn: sqlite3.Connection) -> None:
        self._files = {
            rel_path: (file_id, doc_len)
            for file_id, rel_path, doc_len in conn.execute(
                "SELECT id, rel_path, doc_len FROM files"
            )
        }
        self._n = len(self._files)
        total_len = sum(doc_len for _fid, doc_len in self._files.values())
        self._avgdl = (total_len / self._n) if self._n else 0.0

    # -------------------------------------------------------------- fallback

    def _build_in_memory(
        self, candidates: list[Candidate], started: float, warning: str
    ) -> IndexReport:
        """Last resort: the pre-persistence behaviour, for this call only.

        Slow — it is exactly the per-call cost this module removes — but a
        slow correct ranking beats an empty one, which the caller cannot
        distinguish from "your prompt matches nothing".
        """
        from neo.file_retrieval import FileIndex

        self._memory_index = FileIndex(
            [(c.abs_path, c.rel_path, c.size) for c in candidates]
        )
        self._persistent = False
        self._files = {}
        return IndexReport(
            mode="memory",
            total_files=len(candidates),
            indexed=len(candidates),
            added=len(candidates),
            elapsed_ms=(time.time() - started) * 1000.0,
            warning=warning,
        )

    # ----------------------------------------------------------------- query

    def scores(
        self, prompt: str, candidates: Optional[Iterable[str]] = None
    ) -> dict[str, float]:
        """BM25 score per `rel_path` for `prompt`; unmatched files are absent.

        `candidates` restricts the corpus to the subset this call's `--exts` /
        `--include` / `--exclude` admitted — the ANSWER *and* the statistics.
        Restricting only the answer is the obvious shortcut and it silently
        re-ranks: N, document frequency and average document length all come
        out of the corpus, so leaving them repo-global scores a filtered call
        differently from the per-call index that preceded this one. Measured
        before it was fixed: unflagged runs matched `main` exactly while
        `--exts py` changed all 25 selected lines. Parity is the requirement,
        so the subset is a corpus.
        """
        if not self._persistent:
            raw = self._memory_index.scores(prompt)
            if candidates is None:
                return raw
            allowed = set(candidates)
            return {p: s for p, s in raw.items() if p in allowed}

        conn = self._conn
        if conn is None or not self._n:
            return {}
        terms = code_tokens(prompt)
        if not terms:
            return {}

        allowed_ids: Optional[set[int]] = None
        if candidates is not None:
            allowed_ids = {
                self._files[p][0] for p in candidates if p in self._files
            }
            if not allowed_ids:
                return {}

        # Query-term MULTIPLICITY is preserved, because the in-memory scorer
        # this replaces iterated the term LIST: a prompt naming `getUserById`
        # emits `get`, `user`, `by`, `id` once each plus the whole identifier,
        # and a prompt saying "user" twice weighs `user` twice. Collapsing to
        # a set here would be a silent ranking change, so the count is carried
        # as a multiplier instead of the term being scored twice.
        return self._score_terms(conn, Counter(terms), allowed_ids)

    def _score_terms(
        self,
        conn: sqlite3.Connection,
        repeats: "Counter[str]",
        allowed_ids: Optional[set[int]],
    ) -> dict[str, float]:
        """Accumulate the BM25 sum over the query's terms.

        The length normalization belongs inside the per-(file, term) term,
        not outside it: BM25's denominator contains `tf`, so a version that
        summed `idf * tf` first and normalized once per file would be a
        different formula wearing the same name.
        """
        by_id: dict[int, float] = {}
        doc_lens = {fid: dl for fid, dl in self._files.values()}

        # The corpus is the subset when there is one, statistics included.
        if allowed_ids is None:
            n, avgdl = self._n, self._avgdl
        else:
            n = len(allowed_ids)
            avgdl = (
                sum(doc_lens.get(fid, 0) for fid in allowed_ids) / n if n else 0.0
            )

        # Term ids are resolved per query rather than from a preloaded
        # vocabulary. Loading every term to answer for ten of them read the
        # whole `terms` table — a few hundred thousand rows on a large repo —
        # on the warm path this module exists to make cheap.
        wanted = list(repeats)
        placeholders = ",".join("?" * len(wanted))
        term_ids = dict(
            conn.execute(
                f"SELECT term, id FROM terms WHERE term IN ({placeholders})", wanted
            )
        )

        for term, occurrences in repeats.items():
            term_id = term_ids.get(term)
            if term_id is None:
                continue
            rows = conn.execute(
                "SELECT file_id, tf FROM postings WHERE term_id = ?", (term_id,)
            ).fetchall()
            if allowed_ids is not None:
                rows = [r for r in rows if r[0] in allowed_ids]
            if not rows:
                continue
            # Document frequency counts documents IN THIS CORPUS, so it is
            # taken after the subset filter, never before.
            idf = _idf(n, len(rows))
            for file_id, tf in rows:
                dl = doc_lens.get(file_id, 0)
                if dl == 0:
                    continue
                denom_norm = K1 * (1.0 - B + B * dl / avgdl) if avgdl > 0 else K1
                contribution = idf * (tf * (K1 + 1.0)) / (tf + denom_norm)
                by_id[file_id] = by_id.get(file_id, 0.0) + contribution * occurrences

        id_to_path = {fid: path for path, (fid, _dl) in self._files.items()}
        return {
            id_to_path[fid]: score
            for fid, score in by_id.items()
            if score > 0.0 and fid in id_to_path
        }

    # ------------------------------------------------------------- lifecycle

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None

    def __enter__(self) -> "ContentIndex":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _idf(n: int, df: int) -> float:
    """Lucene-style smoothed IDF — the same expression `memory.bm25` uses."""
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def _as_candidates(paths: Sequence[Any]) -> list[Candidate]:
    """Accept `EligiblePath` records or plain tuples, emit `Candidate`s.

    `mtime_ns` is read from the walk when it carries one and stat-ed here when
    it does not, so a caller holding older `(abs, rel, size)` tuples still
    gets correct freshness rather than a silently degraded one.
    """
    out: list[Candidate] = []
    for entry in paths:
        if isinstance(entry, tuple):
            abs_path, rel_path, size = entry[0], entry[1], entry[2]
            mtime_ns = 0
        else:
            abs_path = entry.path
            rel_path = entry.rel_path
            size = entry.size
            mtime_ns = getattr(entry, "mtime_ns", 0)
        if not mtime_ns:
            try:
                mtime_ns = os.stat(abs_path).st_mtime_ns
            except OSError:
                continue
        out.append(Candidate(abs_path, rel_path, size, mtime_ns))
    return out
