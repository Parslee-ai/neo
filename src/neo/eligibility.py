#!/usr/bin/env python3
"""The one answer to "which paths in this repository exist and are eligible?".

Every subsystem that needs a list of source files reads it from here:
prompt assembly (`context_gatherer`), the semantic index
(`index.project_index`) and the architecture scan (`architecture_metrics`).
There is exactly one walk, one ignore-pattern list, one gitignore matcher and
one content-dedup primitive, and `tests/test_eligibility_single_source.py`
fails if a second one appears anywhere under `src/`.

That guard is not decoration. The gatherer and the index each carried their
own copy of this logic for months and the copies drifted apart in ways that
were invisible from the outside: the index indexed 83 Python files of a stale
worktree in a repository of 4,272 C# files and exited 0 (#159), and the
gatherer selected gitignored build output (#186). Each was fixed on its own
side, and each fix left the other copy wrong — #186's PR message documents the
exclusion list that had silently diverged. Unification is what makes a third
drift structurally impossible rather than merely unlikely.

Layering, because "eligible" mixes two different kinds of rule and only one of
them is git's:

- **Existence and ignore rules** — the walk, `.gitignore`/`.ignore` semantics,
  and the defaults that apply when a repo's ignore files say nothing. This
  layer is validated against `git check-ignore` itself; see
  `tests/test_eligibility_differential.py`.
- **Consumer policy** — extension filters, a per-file size ceiling, symlink
  rejection, byte-identical dedup. These are `WalkPolicy` knobs, not gitignore,
  and they differ per consumer on purpose. They are named separately so that a
  file missing from context can be attributed to the rule that dropped it
  rather than blamed on the repo's `.gitignore`.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# Patterns that apply even when a repo's `.gitignore` says nothing, because
# the worst offenders are routinely untracked-but-not-ignored.
#
# A nested checkout or an agent worktree is a SECOND COPY of a tree, so it does
# not merely add noise — it competes with the originals for the same context
# slots and wins as often as it loses. Measured on one live repo, a single
# prompt came back holding `LedgerActiveStateMonotonicityTests.cs` six times
# and `AUTHENTICATION_ARCHITECTURE.md` six times, one per agent worktree, at
# identical scores: 12 of 16 selected files were duplicates of two.
#
# Only the worktree LAYOUTS are excluded, never the agent directories that
# contain them — see the comment on those entries. An earlier cut excluded
# `.claude`/`.codex`/`.car` outright and justified it by noting that
# `agent_context.discover` globs independently and so still delivers
# CLAUDE.md. That was true and beside the point: `discover` handles markdown
# only, so the skill *source* under those directories had no other route and
# simply vanished.
#
# Ambiguous names — `bin`, `build`, `out`, `target`, `vendor` — are deliberately
# absent except where they are unambiguous by convention (`dist`, `build` for
# Python packaging are here because this list predates the ruling and removing
# them is a selection change, not a unification). Each of the omitted names is
# real source somewhere (`src/bin/main.rs`, vendored trees), and the asymmetry
# is one-sided: over-excluding hides code permanently, over-including only
# spends slots. A repo that generates into them gitignores them, and that is
# the layer that should decide.
DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    '*.pyc', '__pycache__', '.git', '.svn', '.hg',
    'node_modules', '.env', '*.key', '*.pem', '*.secret',
    '.neo', 'venv', 'env', '.venv', 'dist', 'build',
    '*.egg-info', '.tox', '.coverage', 'htmlcov',
    # Agent worktrees — second copies of a tree, which do not merely add
    # noise but compete with the originals for the same context slots.
    #
    # Named by LAYOUT, not by either bare component, because both of the
    # obvious shortcuts hide committed source:
    #   `worktrees`  alone hides `src/worktrees/manager.ts` — a worktree
    #                MANAGER keeps its source in a directory named for
    #                the thing it manages.
    #   `.claude`    alone hides 60 tracked skill implementations across
    #   `.codex`     two local repos, e.g.
    #   `.car`       `.claude/skills/deploy-app/scripts/deploy_verify.py`.
    # Both are the same error: excluding a container for what sometimes
    # sits inside it. `.worktrees` stays a bare component — that dotted
    # name is unambiguously machine-generated.
    '.worktrees', '**/.claude/worktrees', '**/.codex/worktrees', '**/.car/worktrees',
    # Build output and dependency trees the list above missed.
    'obj', 'bower_components', 'site-packages', 'Pods', 'Carthage',
    '.next', '.nuxt', '.svelte-kit', '.output',
    # Tool caches.
    '.mypy_cache', '.pytest_cache', '.ruff_cache', '.nox', '.eggs',
    # Editor / IDE.
    '.idea', '.vscode', '.vs',
)

#: Ignore files read at the repository root. Nested ignore files are NOT read
#: — the known, accepted limit of this implementation, measured as the only
#: source of under-exclusion against `git check-ignore`.
IGNORE_FILENAMES: tuple[str, ...] = ('.gitignore', '.ignore')


def load_ignore_patterns(root: str) -> list[str]:
    """Load the effective ignore patterns for `root`.

    The shared defaults first, then the repo's own root `.gitignore` and
    `.ignore`, in that order — so a repo's `!negation` can re-include
    something a default excluded, which is gitignore's last-match-wins rule.
    """
    patterns = list(DEFAULT_IGNORE_PATTERNS)

    for ignore_file in IGNORE_FILENAMES:
        ignore_path = Path(root) / ignore_file
        if ignore_path.exists():
            with open(ignore_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)

    return patterns


@functools.lru_cache(maxsize=2048)
def compile_glob(pattern: str) -> "re.Pattern":
    """Compile a gitignore glob where `*` does NOT cross a path separator.

    `fnmatch` is component-blind: its `*` spans `/`, so `/*.png` — meaning
    "a PNG at the repository root" — matched
    `docs/audits/.../img/concept-5-b.png` six directories down. Measured
    against `git check-ignore` over 7,534 on-disk paths, that was the last
    class of over-exclusion left once anchoring and negation were right.

    `*` matches within one component, `**` spans components, `?` is one
    non-separator character. Character classes are translated rather than
    passed through — see `_class_body` for why `[!a-z]` and `[]]` cannot
    survive a copy. A pattern that will not compile degrades to a literal
    match rather than raising.
    """
    out: list[str] = []
    i = 0
    while i < len(pattern):
        char = pattern[i]
        if char == '*':
            if pattern[i:i + 3] == '**/':
                # `**/foo` matches `foo` at any depth INCLUDING the root, so
                # the separator has to be part of the optional group. Emitting
                # `.*` + a literal `/` would require at least one leading
                # component and miss the root case.
                out.append('(?:.*/)?')
                i += 3
                continue
            if pattern[i:i + 2] == '**':
                # `**` spans components only when it is a whole component —
                # delimited by separators or the ends of the pattern. Glued to
                # ordinary characters (`a**b`) git treats it as a plain star,
                # so it must stay inside one component.
                before_ok = i == 0 or pattern[i - 1] == '/'
                after_ok = i + 2 == len(pattern) or pattern[i + 2] == '/'
                if before_ok and after_ok:
                    out.append('.*')
                    i += 2
                    continue
            out.append('[^/]*')
        elif char == '?':
            out.append('[^/]')
        elif char == '[':
            body, end = _class_body(pattern, i)
            if body is None:
                out.append(re.escape(char))
            else:
                out.append(body)
                i = end
                continue
        else:
            out.append(re.escape(char))
        i += 1

    try:
        return re.compile('(?s:' + ''.join(out) + r')\Z')
    except re.error:
        # A pattern that will not compile must not take the process with it.
        # `should_ignore` runs on every directory and every file of every
        # walk, so one unparseable `.gitignore` line would turn any neo
        # invocation into a traceback. `fnmatch`, which this replaced, could
        # not crash — it degraded to a non-match. Match the literal instead.
        return re.compile(re.escape(pattern) + r'\Z')


def _class_body(pattern: str, start: int) -> tuple:
    """Translate a glob character class at `start` into a regex class.

    Returns `(regex_text, index_after)`, or `(None, start)` when the class is
    unterminated and the `[` should be treated as a literal.

    Two things `fnmatch` gets right that a raw pass-through does not:

    - Glob negates with `[!…]`, regex with `[^…]`. Passing `[!a-z]` through
      unchanged yields a class containing a literal `!` plus `a-z` — the
      exact inverse of the intended set, silently.
    - A `]` in FIRST position is a literal member, not the terminator, which
      is the POSIX spelling for "a class containing `]`". Scanning for the
      first `]` cuts `[]]` into an empty class and leaves the rest of the
      regex unbalanced.
    """
    i = start + 1
    negated = i < len(pattern) and pattern[i] in '!^'
    if negated:
        i += 1
    # A leading `]` is a member, so it cannot end the class.
    if i < len(pattern) and pattern[i] == ']':
        i += 1
    close = pattern.find(']', i)
    if close == -1:
        return None, start
    members = pattern[start + 1 + (1 if negated else 0):close]
    # Escape a backslash so the class cannot terminate early or introduce an
    # escape the author did not write. `[` is escaped for a different reason:
    # it is already a literal inside a regex class, but Python emits
    # `FutureWarning: Possible nested set` for `[[`, and a `.gitignore`
    # containing `[[]` would print that on every neo invocation. Escaping is
    # semantically identical and silent.
    #
    # The ORDER is load-bearing and must not be swapped: doubling runs first,
    # so `[` -> `\[` inserts a backslash the doubling pass can no longer eat.
    # Reversed, `[\[]` would double the backslash this line just added and
    # re-open the nested set it exists to close.
    members = members.replace('\\', '\\\\').replace('[', r'\[')
    body = f"[{'^' if negated else ''}{members}]"
    # A wildcard class must not consume `/` no matter how it was written:
    # `a[/]b` matching `a/b` would let one component's rule reach across a
    # separator, which is the same defect as `*` crossing one.
    return f"(?:(?!/){body})", close + 1


def should_ignore(rel_path: str, patterns: list[str], is_dir: bool = False) -> bool:
    """Check if `rel_path` matches any gitignore-style pattern.

    Two properties decide every case, and the old branch structure conflated
    them, so they are now read off the pattern once and applied uniformly:

    - **anchored** — a leading `/`, or a `/` anywhere inside the pattern.
      Both mean "match from the repository root", not at arbitrary depth.
    - **directory-only** — a trailing `/`. Matches a directory and everything
      beneath it, never a file of that name.

    The previous shape tested `pattern.endswith('/')` first and returned from
    that branch, so an UNANCHORED directory rule never reached the
    match-at-any-depth logic and was silently treated as root-anchored.
    `build/` did not match `src/build`, `node_modules/` did not match
    `pkg/node_modules/x.js`. That is the form 80% of real directory rules in
    this workspace take — the anchored form the earlier fix addressed is 19%,
    and nearly all of those live in a single repo.

    Note this tests ONLY the path it is handed: a file under an ignored
    directory does not match on its own name. `walk` prunes those directories
    before descending, which is git's own rule and the reason a re-including
    `!` beneath an excluded directory does not fire.
    """
    # Separator-agnostic: callers may build candidates with `os.path.join`,
    # which emits backslashes on Windows. Splitting on '/' alone silently
    # stopped pruning every nested `node_modules` there.
    parts = [p for p in re.split(r'[\\/]', rel_path) if p]
    if not parts:
        return False
    norm = '/'.join(parts)

    # LAST match wins, which is gitignore's rule and the reason negation has
    # to be evaluated rather than skipped. Returning on the first match makes
    # `!` unreachable by construction. A real repo here pairs
    # `.claude/*` with `!.claude/skills/`, and skipping the second hid seven
    # git-tracked files that `git check-ignore` reports as NOT ignored.
    ignored = False

    for raw in patterns:
        negated = raw.startswith('!')
        pattern = raw[1:] if negated else raw

        anchored = pattern.startswith('/')
        if anchored:
            pattern = pattern[1:]
        dir_only = pattern.endswith('/')
        pattern = pattern.strip('/')
        if not pattern:
            continue
        # A slash inside the pattern anchors it to the root too, which is
        # gitignore's rule and not an extra of ours.
        anchored = anchored or '/' in pattern

        if anchored:
            matched = (
                (compile_glob(pattern).match(norm) and (is_dir or not dir_only))
                # Everything beneath a matched directory.
                or compile_glob(pattern + '/**').match(norm)
            )
        else:
            # Unanchored: the pattern matches a COMPONENT at any depth. A
            # non-final component is necessarily a directory, so everything
            # below it goes; a final component only goes if the rule permits
            # files.
            # Same compiler as the anchored branch. Using `fnmatch` here
            # meant one glob dialect for `docs/[!_]*.md` and another for
            # `[!_]*.md` — identical syntax, opposite meaning, selected by
            # whether the pattern happened to contain a slash.
            matched = any(
                compile_glob(pattern).match(part)
                and (index < len(parts) - 1 or is_dir or not dir_only)
                for index, part in enumerate(parts)
            )

        if matched:
            ignored = not negated

    return ignored


def file_content_hash(path) -> str:
    """SHA-256 of a file's bytes, or `""` when it cannot be read.

    The dedup primitive. A worktree, a vendored copy or a generated barrel
    file makes the same bytes reachable by several paths; a consumer that
    budgets slots must not spend two on one piece of content. Returning `""`
    rather than raising keeps a vanished-since-the-walk file from aborting a
    build — the caller treats a falsy hash as "skip, do not count".

    The catch is broad because the code it replaces was. Narrowing it to
    `OSError` would let a `ValueError` — an embedded NUL in a path, say —
    escape and abort an entire index build over one unreadable file, which is
    the failure this fallback exists to prevent. One bad file costs one file,
    not the run.
    """
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return ""


@dataclass(frozen=True)
class EligiblePath:
    """One file the walk admitted.

    `rel_path` is repository-relative and always POSIX-separated, so a
    downstream comparison against a `.gitignore` pattern, a git path or
    another `rel_path` needs no per-platform normalization.

    `mtime_ns` rides along because the walk already stats every file it
    admits, so carrying the modification time costs one attribute read rather
    than a second stat. It is what lets the persistent content index decide
    which files to hash without reading the repository's every byte on every
    invocation. Defaulted, so a caller constructing one by hand — a test, a
    synthetic candidate — is not forced to invent a timestamp.
    """
    path: str
    rel_path: str
    size: int
    mtime_ns: int = 0


@dataclass(frozen=True)
class WalkPolicy:
    """Consumer-specific admission rules, layered on top of ignore rules.

    Everything here is a *policy* choice rather than a gitignore verdict, and
    the two are kept apart on purpose — a file dropped by `max_file_bytes` was
    not "gitignored", and telling an operator otherwise sends them to the
    wrong file to fix it.

    - `extra_ignores` — additional gitignore-syntax patterns from the caller
      (neo's own `--exclude`), appended after the repo's own so they win.
    - `match_globs` — gitignore-syntax path globs a file must match to be
      admitted at all; `None` admits every extension. The index passes its
      per-language patterns here. `**/*.py` matches a root-level `a.py`,
      matching `pathlib.Path.glob` and git, and unlike `fnmatch`.
    - `exts` — extensions without the leading dot, compared case-insensitively.
    - `max_file_bytes` — per-file ceiling; `None` means no ceiling.
    - `skip_symlinks` — reject symlinks without dereferencing them. A symlink
      is not read, so nothing outside the repository can be pulled in through
      one, and a broken link cannot raise.
    """
    extra_ignores: tuple[str, ...] = ()
    match_globs: Optional[tuple[str, ...]] = None
    exts: Optional[frozenset[str]] = None
    max_file_bytes: Optional[int] = None
    skip_symlinks: bool = True


#: How close a directory's modification time may sit to the moment its entries
#: were read before that reading stops being trustworthy.
#:
#: Filesystem timestamp granularity is not always finer than the interval
#: between two events. HFS+ stamps whole seconds; a network mount can be
#: coarser still. So a directory read at time E, whose recorded mtime is M,
#: can be modified afterwards and land in the SAME timestamp bucket whenever
#: `E - M` is under one granularity tick — and the next walk then compares two
#: equal mtimes and concludes, wrongly, that nothing was added or deleted.
#:
#: One second is the coarsest granularity worth defending against, and the
#: cost of the defence is that a directory touched in the second before it was
#: read is re-listed on the next call rather than trusted. Git carries the same
#: guard for the same reason, under the name "racily clean".
RACY_WINDOW_NS = 1_000_000_000


@dataclass(frozen=True)
class DirectoryListing:
    """The ignore-rule verdicts for the entries of ONE directory.

    This is what makes the walk cacheable. Deciding whether a path is ignored
    means matching it against every pattern in the repository's effective
    ignore set, and on a large repository that pattern matching — not the
    filesystem — is the entire cost of the walk: measured on m365dotnet, the
    full walk takes 6.85 s, the same traversal with the per-file ignore test
    removed takes 0.80 s, and stat-ing all 9,378 admitted files takes 0.10 s.
    Caching the syscalls would save almost nothing; caching the verdicts saves
    almost everything.

    A verdict is a function of the entry's NAME and the pattern set, so it
    stays valid for exactly as long as the directory's entry names do. On
    every POSIX filesystem a directory's mtime moves when an entry is created,
    removed or renamed inside it — and does NOT move when the CONTENT of a
    file inside it changes, which is precisely the distinction wanted here: an
    edit cannot change who is eligible, only what they say.

    Four fields exist to keep that promise honest:

    - `ctime_ns` is stored beside `mtime_ns` because **mtime alone is
      forgeable**. `touch -r`, `tar -x`, `rsync -a` and every snapshot restore
      set a directory's mtime back to a recorded value, so a restore that adds
      or removes a file can land on exactly the mtime this cache holds and be
      reported `warm`. Reproduced with two lines of `os.utime`. The inode
      change time moves on any metadata change and no API restores it, and it
      arrives in the same `stat` — so the defence costs nothing. On Windows
      `st_ctime` is a CREATION time instead, hence constant, which makes this
      comparison a no-op there rather than a false invalidation.
    - `scanned_at_ns` pairs with `RACY_WINDOW_NS` above.
    - `symlinks` is kept beside `files` rather than dropped, because
      `WalkPolicy.skip_symlinks` is a per-consumer choice and a cache that
      recorded only one consumer's answer would silently give the other the
      wrong one.
    - `excluded_dirs` / `excluded_files` are counts of what THIS directory's
      rules rejected, so a reused listing reports the same exclusion totals a
      fresh walk would. A cache that quietly reported zero exclusions would
      make `--dry-run`'s G1 accounting depend on whether the last call
      happened to be warm.

    What is deliberately NOT here: sizes, mtimes and content hashes of the
    files. Those are read fresh on every walk, from the `stat` the walk owes
    its callers anyway, because the persistent content index next door uses
    them as its own freshness stamp — serving it a remembered mtime would make
    an edited file look unedited and turn one cache's staleness into another's.
    """

    mtime_ns: int
    scanned_at_ns: int
    subdirs: tuple[str, ...]
    files: tuple[str, ...]
    symlinks: tuple[str, ...] = ()
    excluded_dirs: int = 0
    excluded_files: int = 0
    ctime_ns: int = 0

    def is_current(self, mtime_ns: int, ctime_ns: int) -> bool:
        """True when this listing may be reused for a directory now at these stamps."""
        return (
            self.mtime_ns == mtime_ns
            and self.ctime_ns == ctime_ns
            and self.mtime_ns <= self.scanned_at_ns - RACY_WINDOW_NS
        )


@dataclass
class WalkResult:
    """The admitted paths plus an honest account of what was left out.

    `excluded_dirs` counts pruned SUBTREES, not the files inside them: the
    walk does not descend into an excluded directory, so it genuinely does
    not know how many files are down there and must not invent a number. That
    is a deliberate trade — the previous index-side count enumerated every
    file under `.worktrees/` in order to report "200 paths excluded", which
    cost a full walk of the very trees the exclusion exists to avoid.

    `listings` is the walk's own record of what it decided, ready to be
    persisted by `index.walk_cache` and handed back on the next call.
    `reused_dirs` / `rescanned_dirs` are how much of it survived, which is the
    only honest basis for reporting a cache as warm.
    """
    paths: list[EligiblePath]
    excluded_dirs: int = 0
    excluded_files: int = 0
    listings: Optional[dict[str, DirectoryListing]] = None
    reused_dirs: int = 0
    rescanned_dirs: int = 0

    @property
    def excluded(self) -> int:
        """Total excluded paths the walk actually saw."""
        return self.excluded_dirs + self.excluded_files


def _read_directory(
    abs_dir: str,
    rel_dir: str,
    mtime_ns: int,
    ctime_ns: int,
    patterns: list[str],
) -> Optional[DirectoryListing]:
    """List one directory and apply the ignore rules to its entries.

    `next(os.walk(...))` rather than `os.listdir`, and that is not a stylistic
    preference: `os.walk` is a generator that reads one directory per step, so
    taking only its first item costs exactly one directory read and nothing
    below it. It also keeps every directory read in this package behind a
    single primitive, which `tests/test_eligibility_single_source.py` enforces
    — a second traversal spelling is how both historical copies of this logic
    began.

    A directory that cannot be read yields nothing and returns `None`, which
    is the behaviour `os.walk` had here before: one unreadable directory costs
    that directory, not the walk.
    """
    walked = next(os.walk(abs_dir, followlinks=False), None)
    if walked is None:
        return None
    _, dirnames, filenames = walked

    prefix = '' if not rel_dir else rel_dir + '/'
    subdirs: list[str] = []
    files: list[str] = []
    symlinks: list[str] = []
    excluded_dirs = 0
    excluded_files = 0

    for name in dirnames:
        if should_ignore(prefix + name, patterns, is_dir=True):
            excluded_dirs += 1
        elif os.path.islink(os.path.join(abs_dir, name)):
            # `os.walk(followlinks=False)` LISTS a symlinked directory and
            # refuses to descend into it. This traversal recurses by hand, so
            # the refusal has to be restated here or `followlinks=False`
            # protects only the one directory read at a time — a link to an
            # ancestor becomes an infinite descent, and a link to `/` walks the
            # machine. Not counted as an exclusion: no ignore rule rejected it,
            # and `skip_symlinks` does not gate this. That flag is about
            # whether a symlinked FILE is delivered; refusing to traverse a
            # symlinked DIRECTORY is what the walk has always done, for every
            # caller.
            logger.debug("Not descending into symlinked directory: %s", prefix + name)
        else:
            subdirs.append(name)

    for name in filenames:
        if should_ignore(prefix + name, patterns):
            excluded_files += 1
            continue
        # Before anything that stats the target. `os.path.getsize`,
        # `Path.is_file` and `Path.resolve` all dereference, and the point of
        # rejecting a symlink is to not touch what it points at. Recorded
        # rather than dropped so that `skip_symlinks=False` still gets a true
        # answer out of a reused listing.
        if os.path.islink(os.path.join(abs_dir, name)):
            symlinks.append(name)
        else:
            files.append(name)

    # AFTER the entries are in hand, so the stamp bounds the read it describes
    # rather than starting before it. See `RACY_WINDOW_NS`.
    return DirectoryListing(
        mtime_ns=mtime_ns,
        ctime_ns=ctime_ns,
        scanned_at_ns=time.time_ns(),
        subdirs=tuple(subdirs),
        files=tuple(files),
        symlinks=tuple(symlinks),
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
    )


def walk(
    root: str,
    policy: Optional[WalkPolicy] = None,
    listings: Optional[dict[str, DirectoryListing]] = None,
) -> WalkResult:
    """The one filesystem walk. Yields every eligible file under `root`.

    Ignored directories are PRUNED rather than filtered per-file, which is
    both git's semantics and the difference between a warm call and a walk of
    `node_modules`.

    `listings` is the previous walk's `WalkResult.listings`, if a caller kept
    one. A directory whose mtime still matches its listing is not read again
    and its entries are not re-tested against the pattern set; its files are
    still `stat`-ed, because that is where their size and mtime come from and
    those must never be remembered (see `DirectoryListing`). Passing `None` —
    the default, and what every caller that does not want a cache does — walks
    exactly as before.

    Results are sorted by `rel_path`. Directory-entry order is whatever the
    filesystem hands back, so without this a tie anywhere downstream — two
    files at the same relevance score, two chunks at the same rank — resolves
    by inode order and the same repository can select differently on two
    machines.
    """
    policy = policy or WalkPolicy()
    patterns = load_ignore_patterns(root)
    patterns.extend(policy.extra_ignores)
    globs = (
        [compile_glob(g) for g in policy.match_globs]
        if policy.match_globs is not None
        else None
    )
    if policy.extra_ignores:
        # A listing records verdicts reached under the repository's OWN ignore
        # rules. `extra_ignores` is appended after those, and gitignore is
        # last-match-wins, so a caller's `!negation` can re-include something
        # the recorded verdict excluded. Reusing a listing here would not be a
        # stale answer, it would be the wrong question — so the cache is not
        # consulted, and `index.walk_cache` does not persist one either.
        listings = None

    paths: list[EligiblePath] = []
    excluded_dirs = 0
    excluded_files = 0
    fresh: dict[str, DirectoryListing] = {}
    reused = 0
    rescanned = 0

    # `(rel_dir, abs_dir)` in step, rather than joining `root` to a POSIX
    # `rel_dir` at use time: `rel_path` is separator-normalized by contract and
    # `path` must stay native, and deriving one from the other produces the
    # mixed `C:\repo\src/neo` spelling on Windows.
    pending = [('', root)]
    while pending:
        rel_dir, abs_dir = pending.pop()
        try:
            # One `stat`, two stamps. See `DirectoryListing.ctime_ns` for why
            # the second one is not optional.
            info = os.stat(abs_dir)
        except OSError:
            # Removed between being listed by its parent and being reached.
            continue

        cached = listings.get(rel_dir) if listings is not None else None
        if cached is not None and cached.is_current(info.st_mtime_ns, info.st_ctime_ns):
            listing = cached
            reused += 1
        else:
            listing = _read_directory(
                abs_dir, rel_dir, info.st_mtime_ns, info.st_ctime_ns, patterns
            )
            if listing is None:
                continue
            rescanned += 1

        fresh[rel_dir] = listing
        excluded_dirs += listing.excluded_dirs
        excluded_files += listing.excluded_files

        prefix = '' if not rel_dir else rel_dir + '/'
        for name in listing.subdirs:
            pending.append((prefix + name, os.path.join(abs_dir, name)))

        names = listing.files
        if not policy.skip_symlinks:
            names = names + listing.symlinks
        elif listing.symlinks:
            # Named, because the index used to warn here and an operator
            # asking "why is this file not indexed?" got an answer. A policy
            # rejection is not an ignore-rule verdict, so it is not counted
            # into `excluded_*` — but it must not be silent.
            for name in listing.symlinks:
                logger.debug("Skipping symlink: %s", prefix + name)

        for name in names:
            rel_path = prefix + name
            if globs is not None and not any(g.match(rel_path) for g in globs):
                continue
            if policy.exts is not None:
                ext = os.path.splitext(name)[1].lstrip('.').lower()
                if ext not in policy.exts:
                    continue
            abs_path = os.path.join(abs_dir, name)
            try:
                # One `stat`, two facts. `os.path.getsize` is a `stat` that
                # throws the rest of the struct away, and the modification
                # time it discards is what an incremental index needs.
                info = os.stat(abs_path)
            except OSError:
                # Vanished between the walk and the stat, or unreadable.
                continue
            size = info.st_size
            if policy.max_file_bytes is not None and size > policy.max_file_bytes:
                continue

            paths.append(EligiblePath(abs_path, rel_path, size, info.st_mtime_ns))

    paths.sort(key=lambda entry: entry.rel_path)
    return WalkResult(
        paths,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
        listings=fresh,
        reused_dirs=reused,
        rescanned_dirs=rescanned,
    )


def normalize_exts(exts: Optional[Iterable[str]]) -> Optional[frozenset[str]]:
    """Build the `WalkPolicy.exts` set from either `py` or `.py` spellings.

    `None` — and ONLY `None` — means "no extension filter". Every other input
    produces a set, including an empty one, because the two are opposite
    instructions and collapsing them inverts the flag: an earlier cut returned
    `None` for a set that normalized to empty, so `exts=[]` went from admitting
    NOTHING (the pre-unification behaviour, where `ext not in []` is always
    true) to admitting the entire repository. A narrowing argument must never
    widen when it is handed nothing.

    For the same reason the empty string is KEPT rather than filtered out: `""`
    is the extension of `Makefile`, so it is a real selector, and dropping it
    silently turned `exts=["py", ""]` into `{"py"}`.
    """
    if exts is None:
        return None
    return frozenset(ext.lstrip('.').lower() for ext in exts)


def walk_paths(
    root: str,
    *,
    extra_ignores: Sequence[str] = (),
    match_globs: Optional[Sequence[str]] = None,
    exts: Optional[Iterable[str]] = None,
    max_file_bytes: Optional[int] = None,
    skip_symlinks: bool = True,
) -> WalkResult:
    """Keyword-argument front door to `walk`, for callers not holding a policy."""
    return walk(
        root,
        WalkPolicy(
            extra_ignores=tuple(extra_ignores),
            match_globs=tuple(match_globs) if match_globs is not None else None,
            exts=normalize_exts(exts),
            max_file_bytes=max_file_bytes,
            skip_symlinks=skip_symlinks,
        ),
    )
