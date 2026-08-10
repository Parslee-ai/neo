#!/usr/bin/env python3
"""
Context gathering for Neo - discovers and scores relevant files from working directory.

Approximates Claude Code/Codex ergonomics with:
- .gitignore-aware file discovery
- Git-based prioritization
- Keyword-based relevance scoring
- Smart chunking for large files
- Budget enforcement
"""

import fnmatch
import functools
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from neo import progress

# Constants
MIN_SCORE_THRESHOLD = 0.2  # Filter files with very low relevance (was 0.3, reduced for broad prompts)
MAX_CHUNKS_PER_FILE = 2    # Cap chunks per file so one large file doesn't dominate the budget
MAX_CHUNK_CENTERS = 20     # Best-scoring lines considered as window centers before merging
MAX_MERGED_WINDOW_LINES = 200   # Ceiling on a merged window so one file can't eat the budget
DISCRIMINATIVE_MAX_LINE_FRACTION = 0.25  # A token on >25% of a file's lines is noise in it
TEST_PENALTY = 0.4         # Multiplier; a test file retains 60% of its score


# Test-file conventions across the languages neo indexes. Anchored on
# component and word boundaries, because the first version was
# `rel_path.startswith("test")` -- an unanchored prefix with no separator,
# which is the same defect class as `"spec" in prompt` matching "in-spec-tor".
# Measured against real conventions, that version was wrong on 9 of 11 cases:
# it missed Go, .NET, JS/TS, Jest, RSpec and JUnit entirely, and claimed
# `testdata/`, `testing/` and `testbed/` -- ordinary source directories -- as
# tests.
#
# It looked correct only because the corpus it was measured on is a single
# Python repo, where it matched 99 of 295 files with no false positives.
_TEST_DIR_NAMES = frozenset({"test", "tests", "spec", "specs", "__tests__"})
_TEST_BASENAME_RE = re.compile(
    r"""
      ^test_                 # Python:  test_foo.py
    | _test\.                # Go:      foo_test.go
    | _spec\.                # RSpec:   user_spec.rb
    | \.test\.               # Jest:    foo.test.ts
    | \.spec\.               # Angular: foo.spec.ts
    | ^Test[A-Z0-9]          # JUnit:   TestFoo.java
    | Tests?\.               # .NET:    FooTests.cs / FooTest.cs
    """,
    re.VERBOSE,
)


def is_test_path(rel_path: str) -> bool:
    """Whether `rel_path` looks like a test file, in any language neo indexes.

    Matched on whole path COMPONENTS and on basename boundaries, never on a
    bare prefix. `testdata/`, `testing/` and `testbed/` are ordinary source
    directories and must survive; `Foo.Tests/` is not.
    """
    parts = [part for part in re.split(r"[\\/]", rel_path) if part]
    if not parts:
        return False

    for component in parts[:-1]:
        lowered = component.lower()
        if lowered in _TEST_DIR_NAMES or lowered.endswith(".tests"):
            return True

    return bool(_TEST_BASENAME_RE.search(parts[-1]))


# Word-boundary anchored, which the substring version this replaces was not.
# `"spec" in prompt` fires on "inspector", "specific", "respective" and
# "aspect"; measured, "the A2UI inspector shows a stale fact count" was
# classified as a prompt about testing and every test file kept full score.
# That was survivable while the flag only softened a cosine boost, and stops
# being survivable now that it gates the keyword path too.
#
# `\btest` is a prefix rather than a whole word so "testing" and "tests"
# match, while "latest" does not -- there is no boundary before its `test`.
_TEST_PROMPT_RE = re.compile(r"\btest|\bpytest\b|\bspecs?\b", re.IGNORECASE)


def prompt_targets_tests(prompt: str) -> bool:
    """Whether the prompt is itself about testing, in which case test files
    are the answer rather than noise and must not be demoted."""
    return bool(_TEST_PROMPT_RE.search(prompt))


@dataclass
class ContextFile:
    """A file selected for context."""
    path: str
    rel_path: str
    language: Optional[str] = None
    bytes: int = 0
    start: Optional[int] = None
    end: Optional[int] = None
    content: Optional[str] = None
    score: float = 0.0


@dataclass
class GatherConfig:
    """Configuration for context gathering."""
    root: str
    prompt: str
    exts: Optional[list[str]] = None
    includes: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)
    max_bytes: int = 100_000
    max_files: int = 30
    diff_since: Optional[str] = None
    use_git: bool = True


def load_gitignore_patterns(root: str) -> list[str]:
    """Load patterns from .gitignore and .ignore files.

    The defaults below apply even when a repo's `.gitignore` says nothing,
    because the worst offenders are routinely untracked-but-not-ignored.
    A nested checkout or an agent worktree is a SECOND COPY of a tree, so it
    does not merely add noise — it competes with the originals for the same
    context slots and wins as often as it loses. Measured on one live repo,
    a single prompt came back holding `LedgerActiveStateMonotonicityTests.cs`
    six times and `AUTHENTICATION_ARCHITECTURE.md` six times, one per agent
    worktree, at identical scores: 12 of 16 selected files were duplicates
    of two.

    Only the worktree LAYOUTS are excluded, never the agent directories that
    contain them — see the comment on those entries. An earlier cut excluded
    `.claude`/`.codex`/`.car` outright and justified it by noting that
    `agent_context.discover` globs independently and so still delivers
    CLAUDE.md. That was true and beside the point: `discover` handles
    markdown only, so the skill *source* under those directories had no
    other route and simply vanished.
    """
    patterns = []

    # Default ignore patterns
    patterns.extend([
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
    ])

    for ignore_file in ['.gitignore', '.ignore']:
        ignore_path = Path(root) / ignore_file
        if ignore_path.exists():
            with open(ignore_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)

    return patterns


@functools.lru_cache(maxsize=2048)
def _path_glob(pattern: str) -> "re.Pattern":
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
    """
    # Separator-agnostic: `iter_paths` builds candidates with `os.path.join`,
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
                (_path_glob(pattern).match(norm) and (is_dir or not dir_only))
                # Everything beneath a matched directory.
                or _path_glob(pattern + '/**').match(norm)
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
                _path_glob(pattern).match(part)
                and (index < len(parts) - 1 or is_dir or not dir_only)
                for index, part in enumerate(parts)
            )

        if matched:
            ignored = not negated

    return ignored


def iter_paths(root: str, includes: list[str], excludes: list[str], exts: Optional[list[str]]) -> list[tuple[str, str, int]]:
    """Walk directory respecting .gitignore patterns."""
    patterns = load_gitignore_patterns(root)
    patterns.extend(excludes)

    results = []

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)

        # Prune ignored directories
        dirnames[:] = [
            d for d in dirnames
            if not should_ignore(os.path.join(rel_dir, d) if rel_dir != '.' else d, patterns, is_dir=True)
        ]

        for filename in filenames:
            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root)

            if should_ignore(rel_path, patterns):
                continue

            # Apply includes filter if specified
            if includes and not any(fnmatch.fnmatch(rel_path, g) for g in includes):
                continue

            # Apply extension filter if specified
            if exts:
                ext = os.path.splitext(filename)[1].lstrip('.')
                if ext not in exts:
                    continue

            # Skip very large files
            try:
                size = os.path.getsize(abs_path)
                if size > 512_000:  # 512 KB hard limit per file
                    continue
                results.append((abs_path, rel_path, size))
            except OSError:
                continue

    return results


def get_git_recent_files(root: str, diff_since: Optional[str] = None) -> set[str]:
    """Get recently modified files from git."""
    recent = set()

    try:
        # Check if we're in a git repo
        subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            cwd=root,
            capture_output=True,
            check=True
        )

        # Get unstaged and staged files
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            cwd=root,
            capture_output=True,
            text=True
        )
        for line in result.stdout.splitlines():
            if len(line) > 3:
                recent.add(line[3:].strip())

        # Get files changed since ref/duration
        if diff_since:
            result = subprocess.run(
                ['git', 'diff', '--name-only', diff_since],
                cwd=root,
                capture_output=True,
                text=True
            )
            recent.update(result.stdout.splitlines())
        else:
            # Get last 50 commits
            result = subprocess.run(
                ['git', 'log', '-n', '50', '--name-only', '--pretty=format:'],
                cwd=root,
                capture_output=True,
                text=True
            )
            recent.update(line for line in result.stdout.splitlines() if line.strip())

    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    return recent


def extract_prompt_tokens(prompt: str) -> set[str]:
    """Extract identifiers and keywords from prompt."""
    tokens = set()

    # Extract CamelCase and snake_case identifiers
    identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b|[A-Z][a-z]+(?:[A-Z][a-z]+)*', prompt)
    tokens.update(t.lower() for t in identifiers)

    # Extract quoted strings
    quoted = re.findall(r'["\']([^"\']+)["\']', prompt)
    tokens.update(q.lower() for q in quoted)

    # Extract simple words
    words = re.findall(r'\b\w{3,}\b', prompt.lower())
    tokens.update(words)

    return tokens


# Enough to clear any organic score (filename overlap caps at +1.8, the two
# re-rank boosts at +1.0 and +1.2). Naming a path is the least ambiguous signal
# a prompt can carry, so it outranks every heuristic rather than competing with
# them.
EXPLICIT_PATH_BOOST = 10.0

# Loose: real filtering is "does this match a file we actually found", which no
# amount of prose punctuation can fake.
_PATH_LIKE = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z][A-Za-z0-9]{0,5}')


def extract_explicit_paths(prompt: str) -> set[str]:
    """Path-like tokens in the prompt, normalized to forward slashes.

    These are candidates only — `matches_explicit_path` decides whether one
    names a file that exists, so "e.g." and "0.42.0" cost nothing.

    Leading "./" is stripped as a PREFIX, never with `str.strip("./")`: that
    strips a character *set* from both ends, so an absolute
    "/Users/me/repo/src/app.py" lost its leading slash and silently stopped
    being recognizable as absolute.
    """
    found = set()
    for raw in _PATH_LIKE.findall(prompt):
        token = raw.replace("\\", "/")
        while token.startswith("./"):
            token = token[2:]
        if token:
            found.add(token.lower())
    return found


def matches_explicit_path(rel_path: str, explicit: set[str]) -> bool:
    """Does `rel_path` name one of the paths the prompt spelled out?

    Containment is tested in BOTH directions on a path boundary:

    * the prompt gives a repo-relative or bare name ("subcommands.py") and the
      candidate is longer ("src/neo/subcommands.py"), and
    * the prompt gives an ABSOLUTE path ("/Users/me/repo/src/neo/subcommands.py")
      and the candidate is the shorter repo-relative form.

    The second direction was missing, which killed the highest-value input:
    tracebacks, IDE "copy path", and neo's own suggestion output all emit
    absolute paths.

    Matching is anchored on "/" so a bare "subcommands.py" hits
    "src/neo/subcommands.py" but NOT "tests/test_subcommands.py" — matching the
    neighbour would re-bury the named file under its own test, which is the
    failure this exists to fix.
    """
    if not rel_path or not explicit:
        return False
    candidate = rel_path.replace("\\", "/").lower()
    for token in explicit:
        if not token:
            continue
        if (candidate == token
                or candidate.endswith("/" + token)
                or token.endswith("/" + candidate)):
            return True
    return False


def calculate_adaptive_limit(prompt: str, default_max: int = 30) -> int:
    """
    Calculate adaptive file limit based on prompt specificity.

    Vague prompts (few specific tokens) -> more files for broad overview (15-25)
    Specific prompts (many tokens/technical terms) -> targeted files (20-30)

    Args:
        prompt: User's query
        default_max: Maximum files to return

    Returns:
        Adaptive limit between 15 and default_max
    """
    tokens = extract_prompt_tokens(prompt)

    # Count technical terms (CamelCase, snake_case, paths)
    technical_terms = sum(1 for t in tokens
                         if '_' in t or any(c.isupper() for c in t) or '/' in t or '.' in t)

    # Count words longer than 6 chars (usually more specific)
    long_words = sum(1 for t in tokens if len(t) > 6)

    # Specificity score with adjusted weights
    # Base token count contributes more, technical terms have high weight
    specificity = (len(tokens) * 0.8) + (technical_terms * 3.0) + (long_words * 1.2)

    # Map to range 15-default_max with adjusted thresholds
    # Broad prompts now get MORE files to provide overview context
    if specificity < 2:
        return 15  # Very vague: "review this" - need broad context
    elif specificity < 5:
        return 20  # Somewhat vague: "review this codebase" - need overview
    elif specificity < 10:
        return 25  # Moderate: "review the semantic search implementation"
    else:
        return default_max  # Specific: "review ProjectIndex.retrieve() and gather_context_semantic()"


def infer_language(path: str) -> Optional[str]:
    """Infer programming language from file extension."""
    ext_map = {
        'py': 'python', 'js': 'javascript', 'ts': 'typescript',
        'jsx': 'javascript', 'tsx': 'typescript', 'java': 'java',
        'c': 'c', 'cpp': 'cpp', 'cc': 'cpp', 'h': 'c', 'hpp': 'cpp',
        'go': 'go', 'rs': 'rust', 'rb': 'ruby', 'php': 'php',
        'cs': 'csharp', 'swift': 'swift', 'kt': 'kotlin',
        'html': 'html', 'css': 'css', 'scss': 'scss', 'json': 'json',
        'yaml': 'yaml', 'yml': 'yaml', 'toml': 'toml', 'xml': 'xml',
        'md': 'markdown', 'sql': 'sql', 'sh': 'shell', 'bash': 'shell',
    }
    ext = os.path.splitext(path)[1].lstrip('.').lower()
    return ext_map.get(ext)


def score_candidate(rel_path: str, size: int, prompt_tokens: set[str],
                    git_recent: set[str], entry_points: set[str],
                    *, demote_tests: bool) -> float:
    """Score a candidate file for relevance.

    `demote_tests` applies the same `TEST_PENALTY` the ProjectIndex boost has
    always applied to its cosine, and it is a consistency fix rather than a
    new policy: the two scoring paths disagreed about test files, and only
    one of them was ever corrected. A test file is a strict SUPERSET of its
    subject's filename tokens -- `test_car_adapter.py` matches {adapter, car}
    where `adapters.py` matches {adapter} -- so on the keyword path it can
    only ever outrank the code it tests, and then the implementation takes a
    size penalty on top.

    Measured over 12 self-chosen code prompts on THIS repo: the first source
    file ranked 4.50 on average and 5.58 of the top 10 slots went to tests and
    docs; with the penalty, 1.75 and 4.41. Six doc-seeking prompts went
    2.16 -> 3.00 docs in the top 10, because demoting tests leaves room rather
    than competing with documentation. Treat those numbers as DIRECTIONAL: the
    prompts are self-chosen and unlabelled, the corpus is a single Python
    repo, and a mean over an unbounded rank is dominated by its worst case.
    The harness is not in the tree, so nobody -- including the author -- can
    re-run them as written.

    Keyword-only and REQUIRED. It was briefly optional-defaulting-to-False so
    that existing pure-scoring tests would not need editing -- which is a
    production default shaped around test convenience, and the default was the
    buggy behaviour. The call site reads `not prompt_targets_tests(...)`, so a
    forgotten argument would have failed open toward exactly the defect this
    fixes. One production caller; no reason for a default at all.

    The penalty RE-RANKS and must not EVICT. Scaling bonuses can push a file
    under `MIN_SCORE_THRESHOLD`, which drops it from context entirely: a test
    file with one token hit at depth 1 goes 0.55 -> 0.19 against a 0.2 floor.
    Both metrics used to justify this change -- rank of the first source file,
    count of tests and docs in the top 10 -- improve monotonically when a file
    DISAPPEARS, so neither could have detected that cost. A file that would
    have been admitted stays admitted, ranked below everything else.
    """
    score = 0.0
    name_lower = rel_path.lower()
    basename = os.path.basename(rel_path).lower()

    # Documentation/architecture bonus (for broad prompts)
    doc_patterns = ['readme', 'architecture', 'design', 'claude.md', 'contributing', 'docs/']
    if any(pat in name_lower for pat in doc_patterns):
        score += 0.8  # Strong boost for documentation

    # Penalize archive/old documentation
    if 'archive' in name_lower or 'old' in name_lower or 'deprecated' in name_lower:
        score -= 0.5

    # Boost main implementation files for broad queries. Match against
    # the filename stem (no extension) so the bonus fires on `main.py`,
    # `main.go`, `Main.java` (basename was lowercased above) but NOT
    # on `library.py`, `accessibility.tsx`, or `reindex.py` — substring
    # matching here would catch huge swaths of any real codebase.
    #
    # Note: there's intentional overlap with `entry_points` below. A file
    # whose stem == "main" is both "THE main file" (this +0.4) and "looks
    # like an entry point" (entry_points adds +0.2). Files that merely
    # *start* with "main" (e.g. main_v2.py) only get the entry_points
    # bonus — the stacking distinguishes "canonical" from "adjacent."
    # The `main_impl_stems` whitelist that used to live here is gone. It gave
    # +0.4 to seven hardcoded stems AND exempted them from the size penalty,
    # and it existed only because that penalty was killing large central
    # files: it rescued `engine.py` (-0.13) and left `store.py` (-1.62), a 12x
    # disparity between two files of near-identical size decided by whether
    # someone had thought of the name. With the penalty gone the whitelist has
    # nothing to patch, and content BM25 identifies a central file by what is
    # in it rather than by a list of names someone maintained.

    # Content relevance: the dominant term, and the only one that has read the
    # file. Normalized to [0, 3] so it outweighs every tie-breaker combined
    # (0.8 docs + 0.3 git + 0.2 entry = 1.3) while staying far below
    # EXPLICIT_PATH_BOOST, which encodes an explicit user instruction.
    score += 3.0 * content_relevance

    # Filename overlap, kept as a weak tie-breaker rather than a primary
    # signal. It is a substring test against the whole path, so short prompt
    # tokens match by coincidence -- `a` is inside 49 of 85 basenames here.
    # That noise was survivable at 0.6 per hit only because nothing better
    # existed; at 0.15 it can separate two files whose content ranks equally
    # and little else.
    hits = sum(1 for token in prompt_tokens if token in name_lower)
    score += 0.15 * min(hits, 3)

    # Git recency bonus
    if rel_path in git_recent:
        score += 0.3

    # Entry point bonus
    if any(basename.startswith(ep) for ep in entry_points):
        score += 0.2

    # Demote tests, unless the prompt is about testing. Same constant and
    # same predicate as the ProjectIndex boost path -- see `is_test_path`.
    # The multiplier lands here, on the accumulated BONUSES and before the
    # additive penalties below, so it does not shrink those penalties too.
    # `forfeited` records what it removed; the floor is applied at the end,
    # because everything after this line is subtractive and a floor applied
    # here would be eaten by the depth penalty (measured: 0.24 -> 0.19,
    # straight back under the threshold).
    forfeited = 0.0
    if demote_tests and is_test_path(rel_path):
        forfeited = score * (1.0 - TEST_PENALTY)
        score *= TEST_PENALTY

    # Penalize by depth
    depth = rel_path.count(os.sep)
    score -= 0.05 * depth

    # No size penalty. See the docstring: it was the dominant term and it was
    # anti-correlated with relevance. BugLocator's rVSM (ICSE 2012) ranks
    # larger files HIGHER for this exact task; BM25's `b` handles the real
    # concern with bounded, corpus-derived length normalization.

    # Re-rank, never evict. `score + forfeited` reconstructs the undemoted
    # final score exactly, because every step after the multiplier is
    # additive. A test file that would have been admitted without the penalty
    # stays admitted, ranked beneath everything above the threshold; one that
    # would have been dropped anyway is unaffected.
    if forfeited and score + forfeited >= MIN_SCORE_THRESHOLD:
        score = max(score, MIN_SCORE_THRESHOLD)

    return max(0.0, score)


def select_chunks(content: str, prompt_tokens: set[str], max_chunk_bytes: int = 12_000) -> list[tuple[str, int, int]]:
    """Select relevant chunks from large file content."""
    lines = content.splitlines()

    if len(content) <= max_chunk_bytes:
        return [(content, 1, len(lines))]

    # Rank lines by the total length of the DISCRIMINATIVE tokens they carry.
    #
    # This used to take the first five matching lines in FILE ORDER. Matching is
    # a substring test, and `extract_prompt_tokens` emits every 3+ character word
    # — "the", "in", "is", "py", "src" — so "in" matches `int`, `using`, `point`
    # and virtually every line qualified. "First five matches" therefore meant
    # "lines 1-5" for any prompt against any large file, and with a 40-line
    # window that is the module docstring and imports. Asked to fix a named
    # function in an 86KB file, neo received that file's import block twice and
    # answered that the function body was not provided — so it emitted no patch,
    # and a suggestion with no diff text can never be verified or learned from.
    #
    # Two properties matter, and BOTH are load-bearing — the merge below does
    # not rescue a bad ranking, it only hides it on the cases that happen to
    # tie:
    #
    # 1. "Discriminative" is measured PER FILE, by document frequency. A length
    #    cutoff was tried and is inverted on real prompts: English stopwords are
    #    long and identifiers are short, so `len >= 4` keeps "does" and "here"
    #    while discarding "db", "fs", "os" and "api" — frequently the entire
    #    subject of the prompt. A token matching most lines of THIS file is
    #    noise regardless of its length.
    # 2. Lines are weighted by matched token LENGTH, not match count. Counting
    #    made a specific prompt degenerate: `_classify_suggestion` and
    #    `subcommands` both scored 1, so the module docstring tied with the
    #    function body and won on the file-order tie-break — still returning
    #    lines 1-43 as the first chunk. Length weighting prefers the line
    #    carrying the longer, more specific token.
    lowered_lines = [line.lower() for line in lines]
    line_count = len(lines) or 1
    frequency = {
        token: sum(1 for line in lowered_lines if token in line)
        for token in prompt_tokens
    }
    present = {token for token, hits in frequency.items() if hits}
    discriminative = {
        token for token in present
        if frequency[token] <= DISCRIMINATIVE_MAX_LINE_FRACTION * line_count
    } or present

    scored_lines = []
    for i, line in enumerate(lowered_lines):
        weight = sum(len(token) for token in discriminative if token in line)
        if weight:
            scored_lines.append((weight, i))

    if not scored_lines:
        # No matches, return header + first N lines
        header_size = min(200, len(lines))
        chunk = '\n'.join(lines[:header_size])
        return [(chunk, 1, header_size)]

    scored_lines.sort(key=lambda pair: (-pair[0], pair[1]))
    window_size = 40
    best_by_index = {index: weight for weight, index in scored_lines[:MAX_CHUNK_CENTERS]}

    # Merge overlapping windows instead of emitting near-duplicates. Two centers
    # a couple of lines apart previously produced two chunks differing by two
    # lines, each consuming a slot of the per-file cap.
    #
    # Merging is bounded. Unbounded, ~20 centers spaced just under 2*window_size
    # chain into one window measured at 8.3x `max_chunk_bytes` — and because the
    # caller admits a chunk all-or-nothing, an oversized one that no longer fits
    # the global budget is DROPPED, so the explicitly-named file contributes
    # nothing. That is the original bug returning through a different door.
    merged: list[list[int]] = []
    for index in sorted(best_by_index):
        start = max(0, index - window_size)
        end = min(len(lines), index + window_size)
        if (merged and start <= merged[-1][1]
                and end - merged[-1][0] <= MAX_MERGED_WINDOW_LINES):
            merged[-1][1] = max(merged[-1][1], end)
            if best_by_index[index] > merged[-1][2]:
                merged[-1][2] = best_by_index[index]
                merged[-1][3] = index
        else:
            # Refusing a merge must not emit an overlapping range: clamp the new
            # window to start where the previous one ended.
            start = max(start, merged[-1][1]) if merged else start
            if start < end:
                merged.append([start, end, best_by_index[index], index])

    # Strongest window first, so the per-file cap keeps the most relevant region
    # rather than whichever happens to appear earliest in the file.
    merged.sort(key=lambda window: window[2], reverse=True)

    chunks = []
    for start, end, _weight, center in merged[:MAX_CHUNKS_PER_FILE]:
        start, end = _fit_to_budget(lines, start, end, center, max_chunk_bytes)
        chunks.append(('\n'.join(lines[start:end]), start + 1, end))
        if sum(len(c[0]) for c in chunks) >= max_chunk_bytes:
            break

    return chunks


def _fit_to_budget(
    lines: list[str], start: int, end: int, center: int, max_bytes: int
) -> tuple[int, int]:
    """Shrink [start, end) to fit `max_bytes`, growing outward from `center`.

    Truncating from the front would be simpler and wrong: the center is the line
    that earned the window, so it is the one line that must survive.
    """
    if len('\n'.join(lines[start:end])) <= max_bytes:
        return start, end
    low = high = min(max(center, start), end - 1)
    size = len(lines[low])
    while size < max_bytes and (low > start or high < end - 1):
        if low > start:
            low -= 1
            size += len(lines[low]) + 1
        if high < end - 1 and size < max_bytes:
            high += 1
            size += len(lines[high]) + 1
    return low, high + 1


def _project_index_boost(root: str, prompt: str, k: int) -> dict[str, float]:
    """If a ProjectIndex exists for ``root``, return per-file relevance boosts.

    The index stores semantic embeddings of code chunks (functions, classes)
    extracted by tree-sitter. Calling ``retrieve(prompt, k)`` returns the
    top-k most semantically-relevant chunks; we project those back to file
    paths and give each file a boost proportional to its best chunk's
    similarity. Files surfaced by multiple chunk hits accumulate.

    Falls through silently when the index doesn't exist or fails to load —
    gather_context still works on the existing filename heuristics.
    """
    try:
        from neo.index.project_index import ProjectIndex

        index = ProjectIndex(root)
        if not index.chunks:
            # First-run hint: surface the most impactful smart-gather upgrade
            # the user can opt into. Cheap, silent on subsequent runs (we only
            # print when the snapshot file is genuinely missing, not just empty).
            if not index.snapshot_path.exists():
                progress.note("Tip: run 'neo --index' to enable semantic file selection")
            return {}
        chunks = index.retrieve(prompt, k=k)

        # Test files often contain the prompt's literal keywords (because
        # they assert against named behaviors), so the FAISS index ranks
        # them above the source file the prompt is actually about.
        # Demote test-file hits unless the prompt is itself about testing.
        prompt_is_test = prompt_targets_tests(prompt)

        boost: dict[str, float] = {}
        for chunk in chunks:
            # Normalize path the same way the rest of the gatherer does.
            rel = os.path.relpath(chunk.file_path, root)
            sim = float(getattr(chunk, "similarity", 0.0))
            sim = max(0.0, sim)
            if is_test_path(rel) and not prompt_is_test:
                sim *= TEST_PENALTY
            # 1.0 cosine = +1.0 boost (dominant signal); test demotion above.
            prev = boost.get(rel, 0.0)
            boost[rel] = max(prev, sim)
        if boost:
            progress.note(f"ProjectIndex boost: {len(boost)} files matched semantically")
        return boost
    except Exception:  # missing index, faiss unavailable, etc.
        # Quiet — index is opt-in, must-not-break path.
        return {}


def _history_boost(root: str, prompt: str, k: int = 10) -> dict[str, float]:
    """If a FactStore exists for the user, boost files that past similar
    Neo runs touched.

    The feedback loop B3 set up: every Neo run persists each simulation
    as an EPISODE fact tagged ``file:<rel_path>`` for each file the run's
    code_suggestions touched. Here we query the store for facts similar
    to the current prompt, scrape ``file:*`` tags off the EPISODE hits,
    and produce a per-file boost weighted by:

      boost = min(0.5, hits_count * 0.15)

    Capped at +0.5 so a hot-history file can't drown out fresh semantic
    signals — past behavior is signal, not destiny. Returns {} when no
    fact store exists or no episodes match. The actual past behavior is
    "the user re-ran a similar prompt before" → it's worth telling the
    gatherer "these files were probably relevant last time."
    """
    try:
        from neo.memory.store import FactStore  # heavy import — defer
        store = FactStore(codebase_root=root, eager_init=False)
        if not store._facts:
            return {}
        hits = store.retrieve_relevant(prompt, k=k)
        counts: dict[str, int] = {}
        for fact in hits:
            for tag in fact.tags or []:
                if isinstance(tag, str) and tag.startswith("file:"):
                    path = tag[len("file:"):]
                    counts[path] = counts.get(path, 0) + 1
        if not counts:
            return {}
        boost = {p: min(0.5, n * 0.15) for p, n in counts.items()}
        progress.note(f"EPISODE-history boost: {len(boost)} files seen in past similar runs")
        return boost
    except Exception:
        # FactStore missing, fact_store init crashed, etc. — never break gather.
        return {}


def _symbol_score(
    abs_path: str,
    prompt_tokens: set[str],
    parser_cache: dict,
) -> float:
    """Tree-sitter-extracted symbol overlap with the prompt.

    Stronger signal than filename substring match: catches files whose
    contents define the function or class the user is asking about even
    when the filename is generic (``utils.py``, ``helpers.py``).

    Returns at most +1.2 (3 symbol hits × 0.4). Failures (unsupported
    language, parse error, OSError) return 0 — falls through to the
    filename score.
    """
    try:
        # Lazy-init the parser exactly once per gather call.
        if "parser" not in parser_cache:
            from neo.index.language_parser import TreeSitterParser
            parser_cache["parser"] = TreeSitterParser()
        parser = parser_cache["parser"]

        path = Path(abs_path)
        if not parser.supports_extension(path.suffix.lower()):
            return 0.0

        # Read with a hard byte cap so giant files don't dominate gather latency.
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read(50_000)
        chunks = parser.parse_file(path, content)
        if not chunks:
            return 0.0

        # Collect all symbols across all chunks (function/class names + imports).
        symbols: set[str] = set()
        for c in chunks:
            for s in c.symbols or []:
                symbols.add(s.lower())
            for imp in c.imports or []:
                symbols.add(imp.lower())

        # Substring match: catches "simulator" matching "_simulation_consensus",
        # "decode" matching "_decode_response", etc. Exact match is too strict
        # for free-form prompts vs snake_case symbol names. Filter very short
        # tokens (≤3 chars) to avoid spurious hits like "the" matching "thread".
        hits = 0
        for t in prompt_tokens:
            if len(t) <= 3:
                continue
            if any(t in s or s in t for s in symbols):
                hits += 1
        return 0.4 * min(hits, 3)
    except Exception:
        return 0.0


def gather_context(config: GatherConfig) -> list[ContextFile]:
    """Main context gathering pipeline."""
    root = config.root
    prompt_tokens = extract_prompt_tokens(config.prompt)

    # Calculate adaptive file limit based on prompt specificity
    adaptive_limit = calculate_adaptive_limit(config.prompt, config.max_files)
    progress.note(f"Adaptive limit: {adaptive_limit} files (based on prompt specificity)")

    # ProjectIndex semantic boost (uses pre-built tree-sitter + FAISS index
    # if present in .neo/). Computed once up front — boosts apply to
    # candidates' final scores below.
    pi_boost = _project_index_boost(root, config.prompt, k=adaptive_limit * 2)

    # EPISODE-history boost from FactStore (the W2.B3 feedback loop):
    # past Neo runs persist file paths as ``file:*`` tags on EPISODE
    # facts. Similar past prompts → similar files were touched → boost
    # those files for the current run.
    hist_boost = _history_boost(root, config.prompt, k=10)

    parser_cache: dict = {}

    # Discover candidates
    candidates = iter_paths(root, config.includes, config.excludes, config.exts)

    # Get git context if enabled
    git_recent = set()
    if config.use_git:
        git_recent = get_git_recent_files(root, config.diff_since)

    # Entry point filenames to boost
    entry_points = {'main', 'app', 'server', 'index', 'login', 'auth', '__init__'}

    # A path the prompt named outright must be in the bundle. Without this it
    # competed on generic filename overlap and lost: asked to fix a function in
    # `src/neo/subcommands.py`, neo ranked that file 163rd of 296 — below its own
    # test file — because filename hits cap at 3 and an 86KB file takes a large
    # size penalty. It never reached the context, so the model correctly refused
    # to write a patch for code it had not seen, and the suggestion was recorded
    # with no diff text and could never be verified or learned from.
    explicit_paths = extract_explicit_paths(config.prompt)

    # Computed once here. The ProjectIndex boost path calls the same pure
    # function on the same argument rather than receiving this value, so the
    # two agree by construction -- but the earlier comment claimed they
    # "shared" a decision that was never passed anywhere, and in a file whose
    # thesis is that two copies drift, an unearned claim of sharing is the
    # rot starting.
    demote_tests = not prompt_targets_tests(config.prompt)

    # Score all candidates
    scored = []
    explicit_hits = 0
    for abs_path, rel_path, size in candidates:
        score = score_candidate(
            rel_path, size, prompt_tokens, git_recent, entry_points,
            demote_tests=demote_tests,
        )
        if matches_explicit_path(rel_path, explicit_paths):
            score += EXPLICIT_PATH_BOOST
            explicit_hits += 1
        if score > 0:
            scored.append((abs_path, rel_path, size, score))
    if explicit_hits:
        progress.note(
            f"Prompt names {explicit_hits} file(s) explicitly - pinned to context")
    elif explicit_paths:
        # A named path that matched nothing is the single highest-value
        # diagnostic here, and staying silent makes it indistinguishable from
        # "no path mentioned". Causes: a typo, a path outside the scan root, or
        # one filtered out by --exclude/--exts.
        progress.note(
            "Warning: prompt names a path but no scanned file matched "
            f"({', '.join(sorted(explicit_paths)[:3])}) - check spelling, "
            "--exclude and --exts")

    # Sort by score descending
    scored.sort(key=lambda x: x[3], reverse=True)

    # Filter by minimum score threshold
    scored_before_filter = len(scored)
    scored_filtered = [(a, r, s, sc) for (a, r, s, sc) in scored if sc >= MIN_SCORE_THRESHOLD]

    # For very broad prompts (<= 5 tokens), boost architectural/entry point files
    if len(prompt_tokens) <= 5:
        arch_patterns = ['README', 'main', 'app', '__init__', 'index', 'setup', 'config']
        arch_files = [(a, r, s, sc) for (a, r, s, sc) in scored
                      if any(pat.lower() in r.lower() for pat in arch_patterns)]

        # Ensure we include at least 5 architectural files
        if arch_files:
            scored_filtered.extend(arch_files[:5])
            # Remove duplicates while preserving order
            seen = set()
            scored_filtered = [x for x in scored_filtered if not (x[1] in seen or seen.add(x[1]))]
            progress.note(f"Broad prompt detected: including {len(arch_files[:5])} architectural files")

    # If no files pass threshold, keep top 10 anyway to avoid empty results
    if not scored_filtered and scored_before_filter > 0:
        progress.note(f"Warning: All files scored below {MIN_SCORE_THRESHOLD}, using top 10")
        scored = scored[:10]
    else:
        filtered_count = scored_before_filter - len(scored_filtered)
        if filtered_count > 0:
            progress.note(f"Filtered {filtered_count} low-relevance files (score < {MIN_SCORE_THRESHOLD})")
        scored = scored_filtered

    # Re-rank pass: union in ProjectIndex semantic hits, then layer
    # tree-sitter symbol relevance on the top filename-scored candidates.
    # The two boosts are additive on top of the existing score.
    #
    # ProjectIndex boost dominates when present (+1.0 for a perfect
    # cosine hit) because it's the strongest signal: it knows the file
    # actually contains code semantically related to the prompt. Tree-
    # sitter symbol overlap (+1.2 max) is a fallback when no index
    # exists, or a tiebreaker when both signals fire.
    scored_by_path = {r: (a, s, sc) for (a, r, s, sc) in scored}

    # Union in any ProjectIndex or history hits not already in the
    # filename-scored set. These deserve a chance even when filename
    # matching missed them.
    for boost_map in (pi_boost, hist_boost):
        for rel_path in boost_map:
            if rel_path not in scored_by_path:
                abs_path = os.path.join(root, rel_path)
                try:
                    size = os.path.getsize(abs_path)
                except OSError:
                    continue
                scored_by_path[rel_path] = (abs_path, size, 0.0)

    # Symbol pass: cap at 3x adaptive_limit candidates by current score
    # to keep parse-overhead bounded. Tree-sitter parsing is ~5-20ms per
    # small file; 75 files * 10ms = ~1s worst case.
    symbol_pass_limit = max(50, adaptive_limit * 3)
    top_for_symbols = sorted(
        scored_by_path.items(),
        key=lambda kv: (
            kv[1][2]
            + pi_boost.get(kv[0], 0.0)
            + hist_boost.get(kv[0], 0.0)
        ),
        reverse=True,
    )[:symbol_pass_limit]

    enriched: list[tuple[str, str, int, float]] = []
    symbol_hit_count = 0
    for rel_path, (abs_path, size, base_score) in top_for_symbols:
        pi = pi_boost.get(rel_path, 0.0)
        hist = hist_boost.get(rel_path, 0.0)
        sym = _symbol_score(abs_path, prompt_tokens, parser_cache)
        if sym > 0:
            symbol_hit_count += 1
        final_score = base_score + pi + hist + sym
        enriched.append((abs_path, rel_path, size, final_score))
    if symbol_hit_count:
        progress.note(f"Symbol-relevance boost applied to {symbol_hit_count} files")

    enriched.sort(key=lambda x: x[3], reverse=True)
    scored = enriched

    # Budget: greedily fill up to max_bytes and adaptive max_files
    selected = []
    total_bytes = 0
    large_files_warned = []

    for abs_path, rel_path, size, score in scored:
        if len(selected) >= adaptive_limit:
            break
        if total_bytes >= config.max_bytes:
            break

        try:
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lang = infer_language(abs_path)

            # For large files, select chunks
            if len(content) > 15_000:
                # Warn about god objects
                size_kb = len(content) / 1024
                if size_kb > 50:
                    if rel_path not in large_files_warned:
                        progress.note(f"Warning: {rel_path} is {size_kb:.0f}KB - consider refactoring into smaller modules")
                        large_files_warned.append(rel_path)

                chunks = select_chunks(content, prompt_tokens)
                # Cap chunks per file. Without this a 90KB engine.py with
                # 5 keyword windows consumes 5 slots of the adaptive_limit
                # budget, starving other distinct files. 2 chunks keeps
                # the highest-relevance windows for a large file without
                # letting it dominate the prompt context.
                chunks = chunks[:MAX_CHUNKS_PER_FILE]
                for chunk_content, start, end in chunks:
                    # Prepend warning for large files
                    if size_kb > 50:
                        warning_header = f"# WARNING: This file is {size_kb:.0f}KB - consider refactoring into smaller modules\n\n"
                        chunk_content = warning_header + chunk_content

                    chunk_bytes = len(chunk_content.encode('utf-8'))
                    if total_bytes + chunk_bytes > config.max_bytes:
                        break

                    selected.append(ContextFile(
                        path=abs_path,
                        rel_path=rel_path,
                        language=lang,
                        bytes=chunk_bytes,
                        start=start,
                        end=end,
                        content=chunk_content,
                        score=score
                    ))
                    total_bytes += chunk_bytes
                    # Recheck the file-cap mid-chunk-loop so a single
                    # large file can't push selected past adaptive_limit.
                    if len(selected) >= adaptive_limit:
                        break
            else:
                content_bytes = len(content.encode('utf-8'))
                if total_bytes + content_bytes > config.max_bytes:
                    continue

                selected.append(ContextFile(
                    path=abs_path,
                    rel_path=rel_path,
                    language=lang,
                    bytes=content_bytes,
                    content=content,
                    score=score
                ))
                total_bytes += content_bytes

        except (OSError, UnicodeDecodeError):
            continue

    return selected


def mmr_pack_chunks(chunks: list, max_bytes: int, max_files: int, lambda_param: float = 0.7) -> list:
    """
    Pack chunks using Maximal Marginal Relevance for file diversity.

    MMR balances relevance (similarity score) and diversity (different files).
    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity

    Args:
        chunks: List of CodeChunk objects with similarity scores
        max_bytes: Maximum total bytes
        max_files: Maximum number of files
        lambda_param: Balance between relevance (1.0) and diversity (0.0)

    Returns:
        List of selected chunks meeting budget constraints
    """
    if not chunks:
        return []

    selected = []
    selected_files = set()
    total_bytes = 0
    remaining = list(chunks)

    # First chunk: highest similarity
    first = remaining.pop(0)
    selected.append(first)
    selected_files.add(first.file_path)
    total_bytes += len(first.content.encode('utf-8'))

    # Iteratively select chunks with MMR
    while remaining and len(selected_files) < max_files and total_bytes < max_bytes:
        best_score = -1
        best_idx = -1

        for i, chunk in enumerate(remaining):
            chunk_bytes = len(chunk.content.encode('utf-8'))
            if total_bytes + chunk_bytes > max_bytes:
                continue

            # Relevance: similarity to query
            relevance = chunk.similarity or 0.0

            # Diversity: bonus for new files
            diversity = 1.0 if chunk.file_path not in selected_files else 0.0

            # MMR score
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx == -1:
            break

        # Select best chunk
        chunk = remaining.pop(best_idx)
        selected.append(chunk)
        selected_files.add(chunk.file_path)
        total_bytes += len(chunk.content.encode('utf-8'))

    return selected


def gather_context_semantic(config: GatherConfig) -> list[ContextFile]:
    """
    Gather context using semantic search via ProjectIndex.

    Falls back to keyword search if no index exists.

    Args:
        config: GatherConfig with prompt, root, and budget constraints

    Returns:
        List of ContextFile objects
    """
    root = config.root
    index_path = Path(root) / ".neo" / "index.json"

    # Check if index exists
    if not index_path.exists():
        progress.note(f"No semantic index found at {index_path}")
        progress.note("Falling back to keyword search. Run 'neo index' to build semantic index.")
        return gather_context(config)

    # Load ProjectIndex
    try:
        from neo.index.project_index import ProjectIndex

        start_time = time.time()
        index = ProjectIndex(root)

        # Retrieve top 100 chunks
        chunks = index.retrieve(config.prompt, k=100)

        if not chunks:
            progress.note("No chunks found in semantic index")
            progress.note("Falling back to keyword search")
            return gather_context(config)

        # Pack chunks using MMR for diversity
        selected_chunks = mmr_pack_chunks(chunks, config.max_bytes, config.max_files)

        # Convert to ContextFile format
        context_files = []
        for chunk in selected_chunks:
            abs_path = Path(root) / chunk.file_path
            chunk_bytes = len(chunk.content.encode('utf-8'))

            context_files.append(ContextFile(
                path=str(abs_path),
                rel_path=chunk.file_path,
                language=infer_language(chunk.file_path),
                bytes=chunk_bytes,
                start=chunk.start_line,
                end=chunk.end_line,
                content=chunk.content,
                score=chunk.similarity or 0.0
            ))

        elapsed = time.time() - start_time

        # Log metrics
        log_context_metrics(
            method="semantic",
            elapsed_ms=elapsed * 1000,
            chunks_retrieved=len(chunks),
            chunks_selected=len(selected_chunks),
            files_selected=len(set(cf.rel_path for cf in context_files)),
            total_bytes=sum(cf.bytes for cf in context_files),
            root=root
        )

        progress.note(f"Semantic search: {len(selected_chunks)} chunks from {len(set(cf.rel_path for cf in context_files))} files in {elapsed*1000:.0f}ms")

        return context_files

    except ImportError as e:
        progress.note(f"Failed to load ProjectIndex: {e}")
        progress.note("Falling back to keyword search")
        return gather_context(config)
    except Exception as e:
        progress.note(f"Semantic search error: {e}")
        progress.note("Falling back to keyword search")
        return gather_context(config)


def log_context_metrics(method: str, elapsed_ms: float, chunks_retrieved: int,
                        chunks_selected: int, files_selected: int, total_bytes: int,
                        root: str):
    """
    Log context gathering metrics to .neo/context_metrics.jsonl

    Args:
        method: "semantic" or "keyword"
        elapsed_ms: Time taken in milliseconds
        chunks_retrieved: Total chunks retrieved (before packing)
        chunks_selected: Chunks selected (after packing)
        files_selected: Number of unique files
        total_bytes: Total bytes in selected context
        root: Repository root
    """
    try:
        metrics_path = Path(root) / ".neo" / "context_metrics.jsonl"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        metric = {
            "timestamp": time.time(),
            "method": method,
            "elapsed_ms": round(elapsed_ms, 2),
            "chunks_retrieved": chunks_retrieved,
            "chunks_selected": chunks_selected,
            "files_selected": files_selected,
            "total_bytes": total_bytes
        }

        with open(metrics_path, 'a') as f:
            f.write(json.dumps(metric) + '\n')
    except Exception as e:
        # Don't fail on metrics logging errors
        progress.note(f"Warning: Failed to log metrics: {e}")
