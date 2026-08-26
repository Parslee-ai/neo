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
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from neo import eligibility, progress
from neo.text_budget import MARKER_TEMPLATE, apportion

# Constants
MIN_SCORE_THRESHOLD = 0.2  # Filter files with very low relevance (was 0.3, reduced for broad prompts)

# 0.2 is far too low to separate signal from noise, and the reason is NOT that
# scores scale with prompt length — an earlier version of this comment claimed
# that and it is false. `content_relevance` arrives already normalized to
# [0, 1] by the per-query maximum (file_retrieval.normalize), so
# the weighted content term is capped at CONTENT_WEIGHT for a three-word prompt and for a
# three-hundred-word one alike.
#
# What actually clears 0.2 without any content signal is the filename
# tie-breaker, `0.15 * min(hits, 3)` (capped at 0.45), plus the +0.8
# documentation bonus below. On a long prompt nearly every path in a large repo
# contains three prompt tokens as substrings, so nearly every file scores
# 0.2-0.45 on filename alone: measured, 224 of 14,428 filtered, 99.2% "passed",
# which is not a filter.
#
# A file therefore also has to reach a fraction of the BEST ORGANIC score for
# this prompt. Because the content term is max-normalized, that best is pinned
# near CONTENT_WEIGHT (3.0) whenever anything matches, which puts the effective
# floor around 0.5 — deliberately just above the 0.45 tie-breaker ceiling. That
# coupling is the whole mechanism: if CONTENT_WEIGHT or the 0.45 cap moves,
# RELATIVE_SCORE_FLOOR has to be re-derived, and this comment is the only place
# that says so.
#
# KNOWN LIMIT: anchored on max(), the least robust statistic available. One
# outlier organic hit sets the bar for the whole scan. A p90 of the organic
# distribution would be robust to that and costs one sorted().
# Measured against the best ORGANIC score — a file that merely got
# EXPLICIT_PATH_BOOST does not set the bar for everything else. That boost is
# tuned to exceed any organic score by construction (see EXPLICIT_PATH_BOOST),
# so including it put the floor above the entire scan and left prompts that
# named a file with nothing but that file.
RELATIVE_SCORE_FLOOR = 0.15  # keep files scoring >= 15% of the best organic hit

# A prompt this short cannot express a specific target, so architectural and
# documentation files are the best available answer. Above it, the prompt names
# what it wants and documentation has to earn its place on content like
# anything else. Same threshold the architectural-file top-up uses below, named
# once so the two cannot drift apart.
BROAD_PROMPT_TOKENS = 5
MAX_CHUNKS_PER_FILE = 2    # Cap chunks per file so one large file doesn't dominate the budget
MAX_CHUNK_CENTERS = 20     # Best-scoring lines considered as window centers before merging
MAX_MERGED_WINDOW_LINES = 200   # Ceiling on a merged window so one file can't eat the budget
DISCRIMINATIVE_MAX_LINE_FRACTION = 0.25  # A token on >25% of a file's lines is noise in it
TEST_PENALTY = 0.4         # Multiplier; a test file retains 60% of its score
CONTENT_WEIGHT = 3.0       # Weight on normalized content BM25; see score_candidate


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
    #: Selected because `--include` named it, not because it ranked. A pinned
    #: file bypasses `MIN_SCORE_THRESHOLD`, the rank cut, the `--max-files`
    #: cap and `MAX_CHUNKS_PER_FILE`, and the prompt renderer does not apply
    #: its per-file character cap to it. See `pin_included_files`.
    pinned: bool = False
    #: True when the `--max-bytes` ceiling forced a cut in this file — pinned
    #: or scanned. The content then carries `text_budget`'s marker; this field
    #: is what lets the CLI report the cut without re-parsing the text. It
    #: covers scan files too since delivery became whole-file: an over-budget
    #: file arrives as one marked excerpt rather than as unmarked windows,
    #: so the cut has a place to be reported from.
    truncated: bool = False


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
    #: `--semantic`. A HINT that biases stage 4 of the front door, not a fork
    #: to a second pipeline. The semantic catalog is consulted on every
    #: invocation when it exists; this raises how deep it is read and how much
    #: its verdict weighs. See `gather_context`.
    semantic: bool = False


# --- Stage 4: the semantic supplement ---------------------------------------
#
# The embedding catalog re-ranks and supplements the keyword result; it is
# never a separate lane the caller has to choose. These two constants are the
# whole of what `--semantic` does.
#
# `SEMANTIC_WEIGHT` is the multiplier on a chunk's cosine when the flag is
# absent — 1.0, the weight the boost has always carried, so an unflagged run
# ranks exactly as it did. `SEMANTIC_HINT_WEIGHT` is CONTENT_WEIGHT, which is
# the point: with the hint the embedding verdict weighs the same as the BM25
# verdict, so a concept-shaped query can be answered by meaning rather than by
# shared vocabulary. Below CONTENT_WEIGHT the flag would be decorative; above
# it, it would be the old fork wearing a smaller word.
SEMANTIC_WEIGHT = 1.0
SEMANTIC_HINT_WEIGHT = CONTENT_WEIGHT

#: Multiplier on how many catalog chunks stage 4 reads under the hint. Depth
#: and weight are separate knobs because they fix different misses: weight
#: moves a file the catalog already surfaced, depth is what surfaces a file
#: the keyword stage never scored at all.
SEMANTIC_HINT_DEPTH = 3

#: Largest share of `--max-bytes` the pin block may take while the scan still
#: has candidates of its own. Ruling 1 is "the named files AND keep scanning";
#: an unbounded pin block satisfies the first clause by deleting the second.
#: See the call site in `gather_context` for the measurement that forced it.
PIN_BUDGET_SHARE = 0.5

#: Floor on a scan file's share of `--max-bytes`. Below this a file arrives as
#: little more than its own truncation marker, which costs a delivery slot and
#: answers nothing — so the file count is reduced to what the ceiling can
#: actually seat, and the reduction is REPORTED rather than absorbed. Only
#: binds on a hand-set, very small `--max-bytes`: at the 100,000 default it
#: permits 195 files against an adaptive limit of at most 30.
MIN_FILE_SHARE_BYTES = 512


#: Per-file ceiling for prompt assembly. A file larger than this is not
#: evidence a prompt can use — the largest hand-written file in this repo is
#: 177 KB — and reading it costs the whole budget. This is a GATHERER budget,
#: not an eligibility rule, which is why it is passed to the shared walk as
#: policy rather than living inside it.
MAX_FILE_BYTES = 512_000


def base_paths(root: str) -> list[eligibility.EligiblePath]:
    """The repository's indexable corpus: the shared walk, no per-call flags.

    Split out from `iter_paths` because the persistent content index and one
    invocation's candidate list are answers to different questions. The index
    is a property of the REPOSITORY — it must survive a call that passed
    `--exts py` without being pruned to Python and rebuilt on the next call
    that did not. A call's candidates are a property of THAT CALL.

    So the walk happens once, unfiltered by anything the caller typed, and
    `--exts` / `--exclude` / `--include` are applied afterwards to the result.
    That also keeps the two in lockstep by construction: a candidate is always
    a subset of what was indexed, so a per-call flag can only ever narrow the
    answer, never smuggle in a file the walker did not admit.

    What stays here is the gatherer's own policy: the 512 KB ceiling. The
    walk, the ignore rules and the gitignore matcher live in
    `neo.eligibility`, shared with the project index — there is no second copy
    here to drift.

    The walk itself runs through `index.walk_cache`, which remembers each
    directory's ignore verdicts and re-lists only the directories whose
    contents moved. That is a cost change and never a result change: the cache
    is keyed on a hash of the effective ignore patterns and on each directory's
    mtime, and every failure mode degrades to the full walk.
    """
    from neo.index.walk_cache import cached_walk

    policy = eligibility.WalkPolicy(max_file_bytes=MAX_FILE_BYTES)
    return cached_walk(root, policy).paths


def filter_candidates(
    paths: list[eligibility.EligiblePath],
    includes: list[str],
    excludes: list[str],
    exts: Optional[list[str]],
) -> list[eligibility.EligiblePath]:
    """Narrow the corpus to what this invocation's flags admit.

    `--exclude` is evaluated with `eligibility.should_ignore`, the same
    gitignore dialect the walk applies to its own patterns, so moving the
    evaluation out of the walk did not change what a pattern means. `--include`
    keeps its `fnmatch` dialect, which is a user-facing CLI contract rather
    than a gitignore rule and was already applied after eligibility.

    Two things this cannot reproduce, both cost rather than correctness, and
    stated in full because the first draft of this note understated the second:

    - The walk PRUNES an excluded directory instead of testing every file
      beneath it, so a `--exclude` naming a directory no longer saves the cost
      of descending into it. It still excludes every file under it —
      `should_ignore` matches a non-final path component — and the directories
      that make pruning matter (`node_modules`, `.worktrees`, build output)
      are in the shared default list, which the walk still prunes.
    - A file this function drops is still INDEXED: the content index is
      refreshed from the unfiltered walk, so an `--exts py` run reads,
      tokenizes and stores the whole repository. That is the deliberate trade —
      the index is a property of the repository, and pruning it to one call's
      flags would make the next call rebuild — but it means `--exclude` does
      not stop a file being read, only being ranked and delivered.
    """
    ext_set = eligibility.normalize_exts(exts) if exts else None
    out: list[eligibility.EligiblePath] = []
    for entry in paths:
        if excludes and eligibility.should_ignore(entry.rel_path, list(excludes)):
            continue
        if ext_set is not None:
            ext = os.path.splitext(entry.rel_path)[1].lstrip('.').lower()
            if ext not in ext_set:
                continue
        if includes and not any(
            fnmatch.fnmatch(entry.rel_path, g) for g in includes
        ):
            continue
        out.append(entry)
    return out


def iter_paths(root: str, includes: list[str], excludes: list[str], exts: Optional[list[str]]) -> list[tuple[str, str, int]]:
    """`(abs_path, rel_path, size)` for everything this call admits."""
    return [
        (entry.path, entry.rel_path, entry.size)
        for entry in filter_candidates(base_paths(root), includes, excludes, exts)
    ]


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


# Enough to clear any organic score: content caps at +3.0 (CONTENT_WEIGHT),
# filename overlap at +0.45 (0.15 x min(hits, 3)), the two re-rank boosts at
# +1.0 and +1.2. Naming a path is the least ambiguous signal a prompt can
# carry, so it outranks every heuristic rather than competing with them.
# The +1.8 this comment used to claim for filename overlap was the pre-BM25
# weight; it dropped to a tie-breaker when content became the dominant term.
EXPLICIT_PATH_BOOST = 10.0

# Loose: real filtering is "does this match a file we actually found", which no
# amount of prose punctuation can fake.
# The extension bound is 10 characters, not 6. At {0,5} the pattern MATCHED a
# long extension but TRUNCATED it: "schema.graphql" was extracted as
# "schema.graphq" and "notes.markdown" as "notes.markdo". A truncated token
# matches no real file, so the named file silently lost EXPLICIT_PATH_BOOST,
# and it is not a known extension either, so the not-found warning stayed
# quiet too — the file the user explicitly named just quietly did not arrive.
# 10 covers dockerfile/properties/markdown/graphql.
_PATH_LIKE = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_./\\-]*\.[A-Za-z][A-Za-z0-9]{0,9}')


# Extensions that make a bare dotted token a plausible FILE reference. A token
# with a "/" needs no such evidence; one without it is only a path if it ends
# in something a file actually ends in.
_SOURCE_EXTENSIONS = frozenset("""
    py pyi ipynb js jsx mjs cjs ts tsx vue svelte rb rs go java kt kts scala
    c h cc cpp cxx hpp hh cs fs fsx swift m mm php pl pm r jl lua dart ex exs
    erl hrl clj cljs sh bash zsh fish ps1 psm1 sql graphql proto
    json yaml yml toml ini cfg conf env lock
    md markdown rst txt csv tsv xml html htm css scss sass less
    tf tfvars dockerfile mk cmake gradle sbt bzl bazel
""".split())


def _looks_like_a_real_path(token: str) -> bool:
    """Is this token actually naming a file, or is it just prose?

    `_PATH_LIKE` is deliberately loose, and its comment says that is safe
    because "real filtering is 'does this match a file we actually found'".
    That holds for the EXPLICIT_PATH_BOOST consumer, where a token matching
    nothing costs nothing. It is false for the not-found WARNING, which fires
    precisely ON the no-match case: there, every prose token the regex
    over-captures becomes a bogus "check spelling, --exclude and .gitignore"
    for something the user never claimed was a path.

    Observed: "ASP.NET Core 10" in a prompt produced
        Warning: prompt names a path but no scanned file matched (asp.net)
    Same shape as "Node.js", "e.g.", "0.42.0", "vs.".

    A separator is proof enough. Without one, require a real file extension.
    """
    if "/" in token:
        return True
    _, _, ext = token.rpartition(".")
    # Lowered here rather than relying on extract_explicit_paths having done
    # it: this function is public enough to be called directly, and the
    # coupling was stated nowhere. Unlowered, "README.MD" was not warnable.
    return ext.lower() in _SOURCE_EXTENSIONS


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
        return default_max  # Specific: "review ProjectIndex.retrieve() and _project_index_boost()"


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
                    *, demote_tests: bool, content_relevance: float = 0.0) -> float:
    """
    Two signals, in order of weight.

    `content_relevance` is a normalized BM25 score over the file's CONTENT
    (see `neo.file_retrieval`) and is the dominant term. Selection used to
    score a path string and a byte count and never open the file; the
    dominant term was then `score -= 0.01 * size_kb`, uncapped, which made a
    file with one keyword hit unrankable above 60 KB. Measured, the 162 KB
    `src/neo/memory/store.py` scored 0.000 and ranked 200th of 284 for a
    prompt about the fact store. That penalty is gone: BugLocator's rVSM
    (ICSE 2012) ranks larger files HIGHER for this task, and BM25's `b`
    handles the real concern with bounded, corpus-derived normalization.
    `size` stays in the signature but is no longer scored.

    `demote_tests` applies the same `TEST_PENALTY` the ProjectIndex boost
    always applied to its cosine — a consistency fix, since the two scoring
    paths disagreed about test files and only one was ever corrected. It
    matters MORE under content BM25 than it did before, because a test file
    contains its subject's identifiers plus test scaffolding and so is often
    the better lexical match. The A/B that established this was run on a
    first-pass harness whose absolute numbers are superseded; the DIRECTION
    held on every repo and is what the parameter rests on. For current
    end-to-end figures see `neo.file_retrieval` — one table, one harness
    generation, deliberately not restated here.

    Keyword-only and REQUIRED. It was briefly optional-defaulting-to-False so
    existing pure-scoring tests would not need editing — a production default
    shaped around test convenience, where the default was the buggy
    behaviour and the call site reads `not prompt_targets_tests(...)`, so a
    forgotten argument failed OPEN toward the defect being fixed.

    The penalty RE-RANKS and must not EVICT. Scaling can push a file under
    `MIN_SCORE_THRESHOLD`, dropping it from context entirely, and rank-based
    metrics improve when that happens — so a file that would have been
    admitted stays admitted, ranked below everything else.
    """
    score = 0.0
    name_lower = rel_path.lower()
    basename = os.path.basename(rel_path).lower()

    # Documentation/architecture bonus (for broad prompts)
    #
    # Gated on the prompt actually BEING broad, or on the file carrying some
    # content signal. The comment always said "for broad prompts"; the code
    # applied it to every prompt, unconditionally, with no reference to whether
    # the file matched anything. At +0.8 that is four times MIN_SCORE_THRESHOLD
    # and above the relative floor, so any path containing "design", "docs/" or
    # "readme" was admitted on its filename alone — which is why a review
    # prompt asking about controllers and entities came back holding design
    # documents and historical audits, with the implementation files it named
    # crowded out.
    #
    # Scaled rather than dropped for specific prompts: a design doc that DOES
    # match the query is still worth surfacing, just not ahead of the code.
    doc_patterns = ['readme', 'architecture', 'design', 'claude.md', 'contributing', 'docs/']
    if any(pat in name_lower for pat in doc_patterns):
        if len(prompt_tokens) <= BROAD_PROMPT_TOKENS:
            score += 0.8  # Strong boost for documentation on a broad prompt
        else:
            score += 0.8 * content_relevance

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

    # Content relevance: the dominant term, and the only one that has read
    # the file. Scaled by CONTENT_WEIGHT so it outweighs every tie-breaker
    # combined (0.8 docs + 0.3 git + 0.2 entry + 0.45 filename = 1.75) while
    # staying far below EXPLICIT_PATH_BOOST, which encodes a user instruction.
    #
    # This statement appeared TWICE for three commits — the cherry-pick that
    # merged #188's scorer with this one kept both blocks, so the effective
    # weight was 6.0 while every docstring, comment and measurement table
    # said 3.0. Named constant now, so the value has exactly one home.
    score += CONTENT_WEIGHT * content_relevance

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


def _project_index_boost(
    root: str,
    prompt: str,
    k: int,
    *,
    weight: float = SEMANTIC_WEIGHT,
    hinted: bool = False,
) -> dict[str, float]:
    """Stage 4 of the front door: the semantic supplement, when it exists.

    The catalog stores embeddings of code chunks (functions, classes) extracted
    by tree-sitter. ``retrieve(prompt, k)`` returns the top-k most
    semantically-relevant chunks; those project back to file paths and each
    file takes a boost proportional to its best chunk's similarity, scaled by
    `weight`.

    This is a RE-RANK AND SUPPLEMENT, never a lane: the caller unions in files
    the keyword stage never scored and adds the boost to those it did, so the
    embedding verdict can move a file up or surface one from nothing, and can
    never remove a file the keyword stage found. That asymmetry is deliberate —
    a catalog is a snapshot taken at `--index` time, and letting a stale
    snapshot suppress a live BM25 hit would make freshness a correctness
    problem rather than a quality one.

    `hinted` says `--semantic` was passed, and changes exactly two things: the
    weight the caller supplies, and what is said when there is no catalog to
    read. Without the flag a missing catalog is a one-line tip; with it, the
    caller asked for this stage by name and gets told plainly that it did not
    run, because silence there is indistinguishable from "the catalog was
    consulted and had nothing to say".

    Falls through silently on any other failure — the front door still ranks on
    the pinned paths and the keyword index.
    """
    try:
        from neo.index.project_index import ProjectIndex

        index = ProjectIndex(root)
        if not index.chunks:
            # First-run hint: surface the most impactful smart-gather upgrade
            # the user can opt into. Cheap, silent on subsequent runs (we only
            # print when the snapshot file is genuinely missing, not just empty).
            if hinted:
                progress.note(
                    "--semantic asked for the embedding catalog and there is "
                    f"none at {index.snapshot_path} - ranking on the keyword "
                    "index alone; run 'neo --index' to build it")
            elif not index.snapshot_path.exists():
                # Phrased as an addition, never as something being switched
                # on: stages 1-3 already ran and this run's files were already
                # selected. The verb "enable" belongs to the retired lane, in
                # which the catalog WAS the selector, so its absence meant
                # selection fell back to something lesser. It no longer does.
                progress.note(
                    "Tip: 'neo --index' builds the embedding catalog; stage 4 "
                    "then re-ranks every run, with no flag")
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
            # A 1.0 cosine is worth `weight`; test demotion applied above.
            prev = boost.get(rel, 0.0)
            boost[rel] = max(prev, sim * weight)
        if boost:
            progress.note(
                f"Semantic re-rank: {len(boost)} file(s) matched the catalog "
                f"(weight {weight:g}{', --semantic' if hinted else ''})")
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


# Bytes held back from a pinned file's byte allowance so the truncation marker
# fits inside the ceiling rather than pushing the payload over it. The marker
# is `text_budget.MARKER_TEMPLATE` with two integers in it — 55 characters at
# realistic sizes, all ASCII. The reserve is checked against the marker that
# actually gets emitted, so a template change cannot silently break the bound.
_PIN_MARKER_RESERVE = 96


def resolve_explicit_paths(
    candidates: list[tuple[str, str, int]], explicit: set[str]
) -> list[tuple[str, str, int]]:
    """Stage 1 of the front door: the files the prompt spelled out, as pins.

    `EXPLICIT_PATH_BOOST` made a named path rank first and stopped there, which
    left the guarantee resting on arithmetic. +10.0 clears every organic signal
    combined, so the file did reach the top of the ranking — and then met the
    same delivery machinery as everything below it: it could be windowed into
    a fragment, and on a prompt naming more files than the adaptive limit
    admits, the last-named ones fell off the end. "Ranked first" and "present"
    are different claims, and only the second is what a caller who typed a path
    is asking for.

    So a named path is now pinned, on the same terms `--include` is: delivered
    whole, ahead of the scan, bypassing `MIN_SCORE_THRESHOLD`, the rank cut and
    the file cap, with `--max-bytes` the only thing that may still cut it and
    that cut marked and reported. The boost stays where it is, because pinning
    is not retrospective: a file can be pinned only if the walk found it, and
    the boost is what the RANKING does with a named path in every other
    respect.

    Matching is `matches_explicit_path`'s and nothing else — anchored on a path
    boundary, tested in both directions so an absolute path from a traceback
    hits the repo-relative candidate. `extract_explicit_paths` is deliberately
    loose (any dotted token), and the narrowing is precisely that a candidate
    the walk admitted has to match it: "e.g." names no file, so it costs
    nothing. Ordered by path, so the byte ceiling is spent deterministically.
    """
    if not explicit:
        return []
    return sorted(
        (c for c in candidates if matches_explicit_path(c[1], explicit)),
        key=lambda c: c[1],
    )


def resolve_includes(
    candidates: list[tuple[str, str, int]],
    includes: list[str],
    root: Optional[str] = None,
    excludes: Optional[list[str]] = None,
) -> tuple[list[tuple[str, str, int]], list[str], list[str]]:
    """Split `--include` globs into the files they matched and the ones they didn't.

    `--include` used to narrow the walk, which is why an included file still
    had to win on score: it entered the same ranking as everything else and
    could be evicted or windowed like any candidate (#198). Standing ruling 1
    of the unified-store plan makes it an assertion instead — the named files
    are guaranteed, AND the ordinary scan still runs for extra context. So the
    scan is no longer narrowed, and matching happens here, against what the
    walk already found.

    Matching against the scanned candidates rather than re-walking is
    deliberate: a second walk with different eligibility rules would both
    double the walk cost and put a second answer to "which files exist" in the
    tree, which is what the unified-store plan exists to delete.

    **The exact-path rescue.** Two of the walk's rules are scan-cost guards
    rather than relevance judgements — the 512 KB per-file skip and the
    `--exts` filter — and they are the same class of rule as
    `MIN_SCORE_THRESHOLD` and the chunk cap, which a pin already bypasses.
    Leaving them in force gives the guarantee unstated cliffs: #198's own
    reported case was a 449 KB file, 63 KB under the size one. So a pattern
    that matched no candidate is re-tested as an exact path with one `stat`,
    and admitted when it names a real file that the walker's own ignore rules
    do not exclude. **`--include` therefore overrides `--exts`**, deliberately:
    `--exts` narrows the search and `--include` asserts the inputs, so when
    they disagree the explicit instruction wins.

    `.gitignore` and `--exclude` are NOT overridden — G1-inv is a hard
    invariant and trading a reported absence for an unreported presence is the
    worse failure — and they are enforced by calling `eligibility.should_ignore`
    and `eligibility.load_ignore_patterns`, the same shared module the walk
    itself uses (#208), rather than restating them. Eligibility keeps exactly
    one definition; the rescue only chooses which of the gatherer's own budget
    rules to skip.

    A pattern that is not an exact path (a glob, or a typo) gets no rescue:
    expanding a glob means walking, which is the cost this avoids. The test is
    "does this name a file", not "does it contain `*`" — a real file called
    `a[1].py` is an exact path, and classifying it as a glob on its punctuation
    would tell the operator something false about their own filename.

    **The honest limit, stated because it is a silent one.** The rescue is
    attempted only when a pattern matched NOTHING, so a glob that matched some
    files never reaches it — and a glob whose pattern also covers a file over
    the walker's 512 KB ceiling delivers the smaller files and drops the large
    one with no signal at all. `--include '*.py'` beside a 700 KB `huge.py`
    pins the small ones and never mentions it, where `--include huge.py` pins
    it whole. Closing that needs the walk to report which paths its size
    ceiling skipped, which is `neo.eligibility`'s to give and this goal is
    scoped out of. The guarantee is therefore over EXACT PATHS; a glob is a
    search, and naming the file is what asserts it.

    Order follows the pattern order the operator gave, then path order within
    a pattern, because that is the order the byte ceiling is spent in.
    """
    matched: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    unmatched: list[str] = []
    refused: list[str] = []
    ignore_patterns: Optional[list[str]] = None

    for pattern in includes:
        hits = sorted(
            (c for c in candidates if fnmatch.fnmatch(c[1], pattern)),
            key=lambda c: c[1],
        )
        if not hits and root is not None:
            if ignore_patterns is None:
                ignore_patterns = (
                    eligibility.load_ignore_patterns(root) + list(excludes or [])
                )
            rescued, reason = _rescue_exact_include(root, pattern, ignore_patterns)
            if rescued:
                hits = [rescued]
            elif reason:
                # A refusal we can NAME never joins the generic bucket. The
                # 64 MB ceiling and the symlink rule both used to surface as
                # "matched no file - check spelling", which is three wrong
                # causes and a fourth that told the operator an exact path IS
                # rescued past a size limit — the opposite of what happened.
                refused.append(f"{pattern} ({reason})")
                continue
        if not hits:
            unmatched.append(pattern)
            continue
        for hit in hits:
            if hit[1] not in seen:
                seen.add(hit[1])
                matched.append(hit)

    return matched, unmatched, refused


# Ceiling on a rescued file's size. `iter_paths` bounds every candidate at
# 512 KB, so the pin read used to be bounded for free; the rescue lifts that
# bound deliberately and something has to replace it, or `--include` on a
# multi-gigabyte artefact reads the whole thing into memory (twice — `read()`
# then `encode()`) before cutting it to `--max-bytes`. 64 MB is 128x the
# walker's limit and ~140x #198's 449 KB case, so it is out of the way of
# source files while still being a bound. Hitting it is REPORTED, by name and
# with the file's size — it surfaced for one round as a generic "matched no
# file - check spelling", which is the wrong-cause failure this goal deletes.
_PIN_RESCUE_MAX_BYTES = 64 * 1024 * 1024


def _rescue_exact_include(
    root: str, rel_path: str, ignore_patterns: list[str]
) -> tuple[Optional[tuple[str, str, int]], Optional[str]]:
    """Admit an exactly-named file the walk skipped, if it is really eligible.

    Returns ``(entry, reason)``. `entry` is the same `(abs, rel, size)` shape
    `iter_paths` produces. `reason` is set ONLY when this function refused a
    path it could identify — a symlink, an escape, an ignore rule, the size
    ceiling — and is None when the pattern simply names nothing. The caller
    reports the two differently, because "no such file, check your spelling"
    is actively misleading advice for a file that is right there and was
    refused for a stated reason.

    The returned `rel_path` is **normalized**, not the operator's spelling.
    That is not tidiness: `pinned_rels` is keyed on it and the scan loop skips
    pinned paths by membership, so returning `./app.py` for a file the walk
    calls `app.py` sends one file to the model twice — a G1-inv duplicate
    created by the fix for a G2-inv absence. Reproduced with
    `--include ./app.py`, which is what shell completion and `find .` emit.
    An ABSOLUTE path inside the root normalizes the same way, so `--include`
    accepts the form a traceback or an IDE "copy path" produces, which is
    already true of a path named in the prompt (`matches_explicit_path`).

    Ancestor directories are tested separately, twice over, because
    `should_ignore` only tests the path handed to it and `os.walk` never
    descends a symlinked directory. The shared walk gets both for free by
    pruning as it descends; a check that only tested the leaf admitted a file
    inside an ignored directory, and a symlink check that only tested the leaf
    admitted `linked/secret.py` through `repo/linked -> /outside` — a read of
    a file outside the repository, through a flag, past both guards written to
    prevent exactly that.

    Symlinks are refused because `eligibility.WalkPolicy.skip_symlinks`
    defaults to True and the shared walk therefore refuses them. Parity with
    the walk is the invariant: a rescue that admits what the walk skips puts a
    second, looser answer to "what is inside the root" in a codebase that just
    finished collapsing those into one (#208).
    """
    if os.path.isabs(rel_path):
        candidate = os.path.normpath(rel_path)
    else:
        candidate = os.path.join(root, rel_path.lstrip("/"))
    if not os.path.isfile(candidate):
        return None, None

    normalized = os.path.relpath(candidate, root)
    if normalized.startswith(".."):
        return None, "outside the scan root"

    # Leaf first, then every ancestor. `os.walk` does not follow a symlinked
    # directory, so a file reachable only through one is not in the walk's
    # world at all — and, unlike the lexical `..` test above, a symlinked
    # ancestor can leave the repository without the path ever saying so.
    parts = normalized.split(os.sep)
    if os.path.islink(candidate):
        return None, "a symlink, which the shared walk skips"
    for depth in range(1, len(parts)):
        ancestor = os.sep.join(parts[:depth])
        if os.path.islink(os.path.join(root, ancestor)):
            return None, f"reached through the symlinked directory {ancestor}/"
        if eligibility.should_ignore(ancestor, ignore_patterns, is_dir=True):
            return None, f"inside the ignored directory {ancestor}/"
    if eligibility.should_ignore(normalized, ignore_patterns):
        return None, "excluded by .gitignore or --exclude"

    try:
        size = os.path.getsize(candidate)
    except OSError:
        return None, "could not be stat'd"
    if size > _PIN_RESCUE_MAX_BYTES:
        return None, (
            f"{size / 1024 / 1024:.1f}MB, over the "
            f"{_PIN_RESCUE_MAX_BYTES // 1024 // 1024}MB --include ceiling"
        )
    return (candidate, normalized, size), None


def _cut_to_bytes(
    content: str, budget: int, *, whole: Optional[str] = None
) -> tuple[str, bool]:
    """Cut `content` to at most `budget` UTF-8 bytes, marking the cut.

    Returns ``(text, was_cut)``. Unlike `text_budget.truncate_marked`, which
    bounds the CONTENT and appends its marker on top, this aims to land the
    whole return value inside the budget: the budget here is `--max-bytes`,
    which the operator set, so overshooting it quietly would make the number a
    suggestion. The marker text is `truncate_marked`'s, so a pinned file's cut
    reads the same as every other marked cut in a Neo prompt.

    **`whole` is what the marker's numbers are measured against**, and it is
    the guard against this repo's documented nested-cut footgun. `content` is
    normally the entire file and `whole` is left None. When the caller has
    ALREADY reduced the file — the budgeted excerpt in `gather_context`, which
    hands over one region of an over-budget file — the marker must report the
    loss against the file, not against the region. Passing the region alone
    made a 4 KB excerpt of an 86 KB file announce that it had dropped nothing:
    a marker whose presence is the only signal a cut happened, stating the
    smaller of two losses as the whole of it. Supplying `whole` also means the
    marker is emitted even when the excerpt fits the budget, because the cut
    that matters already happened upstream.

    **One exception, and it is the caller's to report.** A budget below the
    marker's own ~51 bytes yields the marker alone and therefore exceeds the
    budget. The alternatives are worse in both directions: returning "" is a
    section indistinguishable from an empty file, and a truncated marker is a
    truncation notice that lies about its own numbers. `pin_included_files`
    detects the overshoot and hands the caller the real total, so what comes
    out of this function is never the last word on the ceiling.
    """
    source = whole if whole is not None else content
    raw = content.encode("utf-8")
    if whole is None and len(raw) <= budget:
        return content, False

    keep = max(budget - _PIN_MARKER_RESERVE, 0)
    head = raw[:keep].decode("utf-8", errors="ignore") if len(raw) > keep else content
    marker = MARKER_TEMPLATE.format(
        dropped=len(source) - len(head), total=len(source)
    )
    # The reserve is an estimate of the marker's size; this is the check that
    # it was big enough. Trimming the head is the safe direction — the marker
    # is the part that says what happened.
    while len((head + marker).encode("utf-8")) > budget and head:
        head = head[:-1]
        marker = MARKER_TEMPLATE.format(
            dropped=len(source) - len(head), total=len(source)
        )
    return head + marker, True


@dataclass
class PinReport:
    """What pinning actually did, in the terms the caller has to report.

    Every field exists because a warning built from a cheaper predicate said
    something untrue. `whole` and `truncated_rels` are separate because a run
    that cut a file must not be described as delivering it whole; `overshoot`
    is separate from `truncated_rels` because a cut that honours the ceiling
    and a ceiling that could not be honoured are different facts with
    different remedies.
    """
    files: list[ContextFile] = field(default_factory=list)
    truncated_rels: list[str] = field(default_factory=list)
    #: Named files that could not be opened. Nothing can be shown for them, so
    #: they are the one case with no entry in the bundle — which is exactly why
    #: they need their own field. Counting them as "unmatched" would report a
    #: cause that is not theirs, and leaving them out entirely made
    #: `--include locked.py` print nothing at all: a named file, silently gone,
    #: under a guarantee whose whole content is that this does not happen.
    unreadable_rels: list[str] = field(default_factory=list)
    #: Total bytes delivered, which is what to compare against `--max-bytes`.
    total_bytes: int = 0
    #: True when the ceiling could not seat one marker per named file, so the
    #: pinned block runs OVER `--max-bytes`. Never silent.
    overshoot: bool = False
    #: rel_path -> which stage asked for it: "prompt" (stage 1) or "--include"
    #: (stage 2). Two stages now feed one pin block, and a warning that names
    #: the wrong one sends the operator to a flag they did not pass. Carried
    #: rather than recomputed so the report cannot disagree with the bundle.
    origins: dict = field(default_factory=dict)

    @property
    def whole(self) -> int:
        return len(self.files) - len(self.truncated_rels)

    def count_from(self, origin: str) -> int:
        return sum(1 for f in self.files if self.origins.get(f.rel_path) == origin)


def pin_included_files(
    matched: list[tuple[str, str, int]],
    max_bytes: int,
    origins: Optional[dict] = None,
) -> PinReport:
    """Deliver every `--include` match whole, or cut with an explicit marker.

    These files are not candidates. They bypass `MIN_SCORE_THRESHOLD`, the
    rank ordering, the `--max-files` cap and `MAX_CHUNKS_PER_FILE`, because
    each of those is a judgement about relevance and the operator has already
    made that judgement. The one thing that can still cut a pinned file is
    `--max-bytes`, and that cut is marked in the content and reported.

    When the pins together exceed the ceiling the budget is apportioned
    max-min fair (`text_budget.apportion`) rather than spent first-come. The
    difference matters: filling greedily in order gives the last-named file
    zero bytes and no marker, which is exactly the silent drop #198 is about.
    Fair shares mean every named file arrives, small ones arrive whole, and
    only the large ones pay.

    **`--max-bytes` bounds pinned CONTENT, not the pinned block.** Below about
    one marker per named file — 200 files against a 5,000-byte ceiling — every
    share is smaller than the notice that has to sit in it, and the block runs
    over. Dropping named files to make the number come out is the one move
    that is not available, so the overshoot is reported instead of hidden:
    `PinReport.overshoot` says it happened and `total_bytes` says by how much.
    Reporting it is what keeps `--max-bytes` an honest number rather than a
    claim the code quietly breaks.
    """
    origins = dict(origins or {})
    if not matched:
        return PinReport(origins=origins)

    contents: dict[str, str] = {}
    sizes: dict[str, int] = {}
    order: list[tuple[str, str]] = []
    unreadable: list[str] = []
    for abs_path, rel_path, _size in matched:
        try:
            with open(abs_path, "r", encoding="utf-8", errors="ignore") as fh:
                content = fh.read()
        except OSError:
            # The one named file with no entry in the bundle: there is nothing
            # to show and nothing to mark. It is carried out on the report and
            # named in the warning, because an absence the operator asked for
            # and is not told about is the defect, not the unreadable file.
            unreadable.append(rel_path)
            continue
        contents[rel_path] = content
        sizes[rel_path] = len(content.encode("utf-8"))
        order.append((abs_path, rel_path))

    if not order:
        return PinReport(unreadable_rels=unreadable, origins=origins)

    total = sum(sizes.values())
    if total <= max_bytes:
        allowances = dict(sizes)
    elif max_bytes >= len(sizes):
        allowances = apportion(sizes, max_bytes)
    else:
        # `--max-bytes` smaller than the number of pinned files. Nonsense
        # config, but the guarantee still holds: each file arrives carrying a
        # marker that says none of it fit.
        allowances = {rel: 0 for rel in sizes}

    report = PinReport(unreadable_rels=unreadable, origins=origins)
    for abs_path, rel_path in order:
        text, was_cut = _cut_to_bytes(contents[rel_path], allowances.get(rel_path, 0))
        if was_cut:
            report.truncated_rels.append(rel_path)
        emitted = len(text.encode("utf-8"))
        report.total_bytes += emitted
        report.files.append(ContextFile(
            path=abs_path,
            rel_path=rel_path,
            language=infer_language(abs_path),
            bytes=emitted,
            content=text,
            # Above EXPLICIT_PATH_BOOST so the dry-run rank listing shows pins
            # at the top, where they are in the bundle. It is a display order,
            # not an input to selection — pins never enter the ranking.
            score=EXPLICIT_PATH_BOOST + 1.0,
            pinned=True,
            truncated=was_cut,
        ))

    report.overshoot = report.total_bytes > max_bytes
    return report


def _pin_notes(
    report: PinReport,
    max_bytes: int,
    unmatched: list[str],
    refused: Optional[list[str]] = None,
) -> None:
    """Say what pinning did, in terms that match what happened.

    One function for the whole pin block. Two stages feed it — paths the
    prompt named (stage 1) and `--include` patterns (stage 2) — and the source
    split is stated rather than collapsed, because the remedy differs: a
    surprise pin from a path mentioned in passing is fixed by rewording the
    prompt, and one from a glob is fixed by narrowing the flag. A single
    "--include pins 4 files" line for a run that passed no `--include` at all
    names a flag the operator never typed, which is this repo's own rule about
    never blaming the wrong knob.
    """
    if report.files:
        if report.truncated_rels:
            shape = (
                f"{report.whole} whole, {len(report.truncated_rels)} cut and marked"
            )
        else:
            shape = "whole"
        from_prompt = report.count_from("prompt")
        from_include = report.count_from("--include")
        split = ", ".join(
            part for part in (
                f"{from_prompt} named in the prompt" if from_prompt else "",
                f"{from_include} from --include" if from_include else "",
            ) if part
        )
        progress.note(
            f"Pinned {len(report.files)} file(s) {shape} "
            f"({report.total_bytes:,} bytes), ahead of the scan"
            + (f" - {split}" if split else ""))
    if report.truncated_rels:
        progress.note(
            f"Warning: --max-bytes={max_bytes:,} forced a cut in "
            f"{len(report.truncated_rels)} pinned file(s) "
            f"({', '.join(report.truncated_rels[:3])}) - marked inline")
    if report.overshoot:
        progress.note(
            f"Warning: --max-bytes={max_bytes:,} cannot seat "
            f"{len(report.files)} pinned file(s) even as truncation markers; "
            f"the pinned block is {report.total_bytes:,} bytes. Named files "
            "are never dropped to meet the ceiling - narrow --include or "
            "raise --max-bytes")
    if report.unreadable_rels:
        progress.note(
            f"Warning: {len(report.unreadable_rels)} --include file(s) could "
            f"not be read ({', '.join(report.unreadable_rels[:3])}) - they are "
            "NOT in the context and nothing can be shown for them")
    if refused:
        # A refusal we can name gets named. Folding these into "matched no
        # file - check spelling" told the operator to check the spelling of a
        # path that was correct, and pointed at .gitignore for a file nothing
        # ignores. Wrong cause, confidently stated, on the one surface whose
        # job is to say what happened.
        progress.note(
            f"Warning: {len(refused)} --include path(s) NOT admitted: "
            f"{'; '.join(refused[:3])}")
    if unmatched:
        # An include that matched nothing is the highest-value diagnostic on
        # this path, for the same reason a prompt-named path that matched
        # nothing is: silence makes it indistinguishable from not asking. The
        # glob clause is not padding — only an exact path is rescued past the
        # walker's size limit, so it is a real and distinct cause.
        progress.note(
            "Warning: --include pattern(s) matched no file "
            f"({', '.join(unmatched[:3])}) - check spelling, --exclude and "
            ".gitignore; a glob, unlike an exact path, is not rescued past "
            "the walker's 512KB per-file limit")


def _budget_notes(
    config: GatherConfig,
    adaptive_limit: int,
    *,
    dropped_by_file_cap: int,
    dropped_by_byte_cap: int,
) -> None:
    """Name each cap that reduced the selection, charged for what IT removed.

    G3-inv: no silent caps. The scan used to stop mid-loop and say nothing —
    the operator saw twelve files and no indication that a thirteenth had lost
    to a ceiling rather than to the ranking. Those are different facts with
    different remedies and they looked identical.

    **Both caps can bind at once and then both are named, with their own
    counts.** The first version took one boolean per cap, derived the file
    cap's from the full candidate list before the byte cap had cut anything,
    and reported the file cap alone whenever both were true. Reproduced: 41
    candidates, 3 delivered, "38 lower-ranked file(s) not delivered: the file
    budget was reached" — where the file cap accounted for 16 and the byte cap
    for 22, and raising `--max-files` moved nothing. A count attributed to the
    wrong knob is worse than no count, because the operator acts on it.

    The two are charged in application order: the file cap sees every
    candidate, the byte cap sees only what survived it. So the numbers sum to
    the real loss and neither is double-counted.
    """
    if dropped_by_file_cap:
        # The limit that bound is the ADAPTIVE one, and printing it as
        # `--max-files=25` names a number the operator never typed — worse,
        # below specificity 10 `calculate_adaptive_limit` returns a fixed
        # 15/20/25 regardless of `--max-files`, so the knob it names cannot
        # move the limit it reports. Both numbers, and the relationship.
        knob = (
            f"adaptive limit {adaptive_limit}, from --max-files="
            f"{config.max_files}"
            if adaptive_limit != config.max_files
            else f"--max-files={config.max_files}"
        )
        progress.note(
            f"{dropped_by_file_cap} lower-ranked file(s) not delivered: the "
            f"file budget ({knob}) was reached")
    if dropped_by_byte_cap:
        progress.note(
            f"{dropped_by_byte_cap} lower-ranked file(s) not delivered: "
            f"--max-bytes={config.max_bytes:,} cannot fund them above "
            f"{MIN_FILE_SHARE_BYTES} bytes each")


def gather_context(config: GatherConfig) -> list[ContextFile]:
    """The retrieval front door: one pipeline, four stages, in priority order.

    Every invocation goes through here. There is no second lane and no flag
    that routes around it — `--semantic` is a hint that biases stage 4, not a
    fork to a different pipeline, because a retrieval strategy the caller has
    to choose in advance is a question only the code can answer.

    1. **Paths the prompt named** — pinned (`resolve_explicit_paths`).
    2. **`--include` patterns** — pinned, per standing ruling 1, and the scan
       keeps running alongside (`resolve_includes`).
    3. **Keyword** — BM25 over file CONTENT, from the persistent index.
    4. **Semantic** — the embedding catalog, as a re-rank and supplement of
       stage 3 whenever it exists, never a substitute for it.

    Stages 1 and 2 assert presence; 3 and 4 rank what is left. That ordering is
    the whole design: an assertion cannot lose to a score, so a file the caller
    named is delivered before the ranking spends a byte, and the ranking then
    fills what remains. Everything below stage 2 is a judgement about
    relevance, and the caller has already made that judgement for the pins.

    **Delivery is one entry per file, read whole from disk.** A file used to
    be able to arrive as two windows, which cost two of the file cap's slots,
    made `len(gathered)` disagree with the number of files (#197), and let one
    large file crowd out two small ones. Chunking survives as a RANKING
    internal only: it now decides which region of an over-budget file survives,
    never how many entries a file contributes. `--max-bytes` is apportioned
    max-min fair (`text_budget.apportion`) across the selection instead of
    being spent greedily in rank order, so a file's size no longer decides how
    many OTHER files reach the model — small files arrive whole, large ones pay,
    and the count of distinct files delivered stops depending on their sizes.
    Every cut that budget forces is marked in the content and reported.
    """
    root = config.root
    prompt_tokens = extract_prompt_tokens(config.prompt)

    # Calculate adaptive file limit based on prompt specificity
    adaptive_limit = calculate_adaptive_limit(config.prompt, config.max_files)
    progress.note(f"Adaptive limit: {adaptive_limit} files (based on prompt specificity)")

    # Stage 4, computed once up front — the boosts apply to candidates' final
    # scores below and union in files the keyword stage never scored. Depth and
    # weight both rise under `--semantic`; with the flag absent this is the
    # same +1.0-per-perfect-cosine supplement it has always been, so an
    # unflagged run ranks exactly as it did.
    pi_boost = _project_index_boost(
        root,
        config.prompt,
        k=adaptive_limit * 2 * (SEMANTIC_HINT_DEPTH if config.semantic else 1),
        weight=SEMANTIC_HINT_WEIGHT if config.semantic else SEMANTIC_WEIGHT,
        hinted=config.semantic,
    )

    # EPISODE-history boost from FactStore (the W2.B3 feedback loop):
    # past Neo runs persist file paths as ``file:*`` tags on EPISODE
    # facts. Similar past prompts → similar files were touched → boost
    # those files for the current run.
    hist_boost = _history_boost(root, config.prompt, k=10)

    parser_cache: dict = {}

    # Discover candidates. ONE walk: the corpus the persistent content index
    # maintains, then this call's flags applied to it — except `--include`,
    # which is deliberately NOT among them. Under standing ruling 1 it asserts
    # inputs rather than narrowing the search, so the scan keeps running over
    # everything else eligible and the named files are pinned separately.
    eligible = base_paths(root)
    admitted = filter_candidates(eligible, [], config.excludes, config.exts)
    candidates = [(e.path, e.rel_path, e.size) for e in admitted]

    # The pin pool is the same corpus minus `--exclude`, but WITHOUT `--exts`.
    # `--exts` narrows the search and `--include` asserts the inputs, so an
    # exactly-named file overrides it; `.gitignore` (applied by the walk) and
    # `--exclude` are not overridden, because G1-inv is a hard invariant and
    # trading a reported absence for an unreported presence is the worse
    # failure. Since #209 split the unfiltered corpus out, that distinction is
    # structural here rather than something the rescue has to reconstruct.
    pin_pool = [
        (e.path, e.rel_path, e.size)
        for e in filter_candidates(eligible, [], config.excludes, None)
    ]

    # Stages 1 and 2, in that order, sharing one pin block and therefore one
    # byte apportionment. The order is the priority order and it is spent
    # order: `pin_included_files` funds max-min fair, so the two stages cannot
    # starve each other, but a duplicate must resolve to the earlier stage or
    # the same file would be delivered twice.
    explicit_paths = extract_explicit_paths(config.prompt)
    explicit_matches = resolve_explicit_paths(pin_pool, explicit_paths)
    include_matches, include_misses, include_refused = resolve_includes(
        pin_pool, config.includes, root, config.excludes
    )

    pin_entries: list[tuple[str, str, int]] = []
    pin_origins: dict[str, str] = {}
    for entry, origin in (
        *((e, "prompt") for e in explicit_matches),
        *((e, "--include") for e in include_matches),
    ):
        if entry[1] not in pin_origins:
            pin_origins[entry[1]] = origin
            pin_entries.append(entry)

    # **The pin block cannot spend the whole ceiling while anything else is
    # eligible.** Standing ruling 1 has two clauses — the named files, AND the
    # scan keeps running — and funding pins to the last byte honours the first
    # by deleting the second. Measured on the M2 battery: the prompt naming
    # `src/Parslee.M365.Api/Program.cs` (442,867 bytes) pinned it, spent 299,959
    # of the 300,000-byte default, left 41 bytes, and delivered **one file**
    # where main delivered 22. The ruling permits a marked cut ("whole, or with
    # an explicit marker"); it does not permit the scan contributing nothing
    # because one named file was large.
    #
    # Half, and symmetric, because neither side has a claim to more of the
    # other's budget than that: "the files you named" and "everything else
    # useful" are the two things the caller asked for. It binds only when pins
    # would otherwise take more than half, which on every prompt in the M2
    # battery except P1 is never — pins are typically a few KB against 300,000.
    # With nothing else eligible the reserve would fund nothing, so the pins
    # take the whole ceiling and arrive whole.
    scan_pool_exists = any(rel not in pin_origins for _a, rel, _s in candidates)
    pin_budget = (
        max(int(config.max_bytes * PIN_BUDGET_SHARE), 1)
        if scan_pool_exists and pin_entries
        else config.max_bytes
    )
    pin_report = pin_included_files(pin_entries, pin_budget, pin_origins)
    pinned = pin_report.files
    pinned_rels = {f.rel_path for f in pinned}
    _pin_notes(pin_report, pin_budget, include_misses, include_refused)
    if pin_budget != config.max_bytes and pin_report.truncated_rels:
        progress.note(
            f"The pinned block is held to {pin_budget:,} bytes "
            f"({PIN_BUDGET_SHARE:.0%} of --max-bytes={config.max_bytes:,}) so "
            "the scan still has a budget to run in; raise --max-bytes to give "
            "both more room")

    warnable_paths = {t for t in explicit_paths if _looks_like_a_real_path(t)}
    if warnable_paths and not explicit_matches:
        # A named path that matched nothing is the single highest-value
        # diagnostic here, and staying silent makes it indistinguishable from
        # "no path mentioned". Causes: a typo, a path outside the scan root, or
        # one excluded by .gitignore/--exclude. It is NOT `--exts`: the pin
        # pool is deliberately ext-unfiltered, so a named `.txt` under
        # `--exts py` is pinned rather than lost.
        progress.note(
            "Warning: prompt names a path but no scanned file matched "
            f"({', '.join(sorted(warnable_paths)[:3])}) - check spelling, "
            "--exclude and .gitignore")

    # Get git context if enabled
    git_recent = set()
    if config.use_git:
        git_recent = get_git_recent_files(root, config.diff_since)

    # Entry point filenames to boost
    entry_points = {'main', 'app', 'server', 'index', 'login', 'auth', '__init__'}

    # Computed once here. The ProjectIndex boost path calls the same pure
    # function on the same argument rather than receiving this value, so the
    # two agree by construction -- but the earlier comment claimed they
    # "shared" a decision that was never passed anywhere, and in a file whose
    # thesis is that two copies drift, an unearned claim of sharing is the
    # rot starting.
    demote_tests = not prompt_targets_tests(config.prompt)

    # Content relevance. Selection used to happen on the path string and the
    # byte count alone, with content first opened afterwards to chunk whatever
    # had already been chosen; #194 made it BM25 over file content, and this
    # is where that stopped being re-derived from scratch on every call. The
    # index lives in the repository's `.neo/`, is brought level with the walk
    # above, and re-reads only what moved.
    from neo.file_retrieval import normalize
    from neo.index.content_index import ContentIndex

    relevance: dict[str, float] = {}
    if eligible:
        with ContentIndex(root) as content_index:
            content_index.refresh(eligible)
            relevance = normalize(
                content_index.scores(
                    config.prompt, [entry.rel_path for entry in admitted]
                )
            )
    if relevance:
        progress.note(f"Content relevance: {len(relevance)} files match the prompt")

    # Score all candidates. `EXPLICIT_PATH_BOOST` stays on the RANKING even
    # though stage 1 now pins the same files: a named path can only be pinned
    # if the walk found it, and the boost is what the ranking does with one
    # that it did not — a `--exts`-narrowed candidate list, a path spelled in a
    # way `matches_explicit_path` reads but the pin pool never held. Deleting
    # it would make the two disagree in exactly the cases where the pin is
    # unavailable, which is when the boost is the only thing left.
    scored = []
    # Tracked so the relative floor below can be measured against the ORGANIC
    # score distribution. EXPLICIT_PATH_BOOST is deliberately larger than any
    # organic score can reach, so letting a prompt-named file set the bar would
    # put the floor above everything the scan found on its own.
    top_organic_score = 0.0
    for abs_path, rel_path, size in candidates:
        score = score_candidate(
            rel_path, size, prompt_tokens, git_recent, entry_points,
            demote_tests=demote_tests,
            content_relevance=relevance.get(rel_path, 0.0),
        )
        if matches_explicit_path(rel_path, explicit_paths):
            score += EXPLICIT_PATH_BOOST
        else:
            top_organic_score = max(top_organic_score, score)
        if score > 0:
            scored.append((abs_path, rel_path, size, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[3], reverse=True)

    # Filter by minimum score threshold — absolute AND relative-to-best.
    # See RELATIVE_SCORE_FLOOR: the absolute floor alone does not discriminate
    # on long prompts, because the score scales with the number of query terms.
    scored_before_filter = len(scored)
    effective_floor = max(
        MIN_SCORE_THRESHOLD, top_organic_score * RELATIVE_SCORE_FLOOR,
    )
    scored_filtered = [(a, r, s, sc) for (a, r, s, sc) in scored if sc >= effective_floor]

    # For very broad prompts (<= 5 tokens), boost architectural/entry point files
    if len(prompt_tokens) <= BROAD_PROMPT_TOKENS:
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
        progress.note(f"Warning: All files scored below {effective_floor:.2f}, using top 10")
        scored = scored[:10]
    else:
        filtered_count = scored_before_filter - len(scored_filtered)
        if filtered_count > 0:
            progress.note(
                f"Filtered {filtered_count} low-relevance files "
                f"(score < {effective_floor:.2f})"
            )
        scored = scored_filtered

    # Stage 4 lands here: the semantic hits are unioned in, then tree-sitter
    # symbol relevance is layered on the top candidates. Both are additive on
    # top of the keyword score — a re-rank and a supplement, never a
    # replacement, so the catalog can lift a file or surface one from nothing
    # and can never remove one the keyword stage found.
    #
    # Weight is `SEMANTIC_WEIGHT` per perfect cosine, or `SEMANTIC_HINT_WEIGHT`
    # under `--semantic`, where it equals `CONTENT_WEIGHT` and the two verdicts
    # carry the same authority. Tree-sitter symbol overlap (+1.2 max) is a
    # fallback when no catalog exists, and a tiebreaker when both signals fire.
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

    # ------------------------------------------------------------------
    # Delivery: one entry per file, whole from disk, budget apportioned.
    #
    # The scan used to fill greedily in rank order and stop at the first file
    # that would not fit, windowing anything over 15,000 characters into up to
    # two entries on the way. Both halves of that are gone.
    #
    # Greedy order made a file's SIZE decide how many OTHER files reached the
    # model: one 90 KB module ahead of the queue could spend most of a 100,000
    # byte ceiling and leave nothing for the rest of the ranking. Max-min fair
    # apportionment funds every file in the selection instead — small files in
    # full, and the surplus they do not use redistributed to the ones that can
    # — so the count of distinct files delivered stops depending on their
    # sizes, which is the property `--max-files` is supposed to control.
    #
    # Windowing produced two entries for one file, and that is what made
    # `len(gathered)` and "files" two different numbers (#197). It cost two
    # slots of the file cap for one file's worth of coverage. Chunk selection
    # survives, but as a RANKING internal: it now chooses WHICH region of an
    # over-budget file is delivered, and never how many entries a file gets.
    # A file that fits its share arrives whole — the common case, and the
    # contract; one that does not arrives as a single excerpt centred on the
    # region that earned it its rank, marked with the loss measured against
    # the whole file.
    selected = list(pinned)
    pinned_bytes = sum(f.bytes for f in pinned)
    scan_entries_before = len(selected)

    # The pinned skip runs BEFORE the cap arithmetic, so a cap can only be
    # recorded as binding when it turned away a file the scan could actually
    # have added. With the order reversed, a pin appearing in `scored` tripped
    # the file cap and the run reported "pinned files filled the file budget"
    # for a repo where every file was pinned and no `--max-files` value would
    # have changed anything.
    eligible_scan = [c for c in scored if c[1] not in pinned_rels]
    scan_slots = max(adaptive_limit - len(selected), 0)
    after_file_cap = eligible_scan[:scan_slots]

    # The byte ceiling reduces the file COUNT rather than truncating the tail
    # of the ranking mid-loop, so what it costs is one number the operator can
    # act on. `MIN_FILE_SHARE_BYTES` is the floor below which a share buys
    # nothing but the file's own truncation marker.
    remaining_bytes = max(config.max_bytes - pinned_bytes, 0)
    affordable = remaining_bytes // MIN_FILE_SHARE_BYTES
    scan_candidates = after_file_cap[:affordable]

    # **Each cap is charged for what IT removed, and the two are counted
    # separately.** The first version set `file_cap_rejected` from the full
    # candidate list and then let the byte cap cut further, so whenever both
    # bound the file cap was named and the byte cap was never mentioned —
    # reproduced live at `--max-files=30`, where the message read "pinned files
    # filled the file budget (--max-files=30)" for a run with 29 of 30 slots
    # free and the BYTE ceiling holding the count at one. Raising the knob the
    # run named provably changed nothing, which is this repo's own rule about
    # never blaming a cap for an absence it did not cause, broken inside the
    # goal whose subject is selection truthfulness.
    dropped_by_file_cap = len(eligible_scan) - len(after_file_cap)
    dropped_by_byte_cap = len(after_file_cap) - len(scan_candidates)

    # Apportion on the walk's own size, not on a pre-pass that reads every
    # candidate: the walk already stat'd them, and reading files the ceiling
    # cannot fund is the cost this stage exists to avoid. A zero-byte file is
    # floored at 1 because `apportion` drops non-positive sections.
    sizes = {rel: max(size, 1) for _abs, rel, size, _sc in scan_candidates}
    allowances = apportion(sizes, remaining_bytes) if sizes else {}

    total_bytes = pinned_bytes
    for abs_path, rel_path, size, score in scan_candidates:
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue

        share = allowances.get(rel_path, 0)
        start = end = None
        if len(content.encode('utf-8')) <= share:
            text, was_cut = content, False
        else:
            # One region, chosen by the same ranking that chose the file, and
            # fitted to the share. `select_chunks` is asked for its windows and
            # only the strongest is taken — the per-file chunk cap no longer
            # reaches delivery, so which of its entries is "second" no longer
            # means anything.
            region, start, end = select_chunks(
                content, prompt_tokens, max_chunk_bytes=max(share, 1)
            )[0]
            # `whole=content` so the marker reports the loss against the FILE.
            # Measured against the region it would announce that a 4 KB
            # excerpt of an 86 KB file had dropped nothing.
            text, was_cut = _cut_to_bytes(region, share, whole=content)

        emitted = len(text.encode('utf-8'))
        total_bytes += emitted
        selected.append(ContextFile(
            path=abs_path,
            rel_path=rel_path,
            language=infer_language(abs_path),
            bytes=emitted,
            start=start,
            end=end,
            content=text,
            score=score,
            truncated=was_cut,
        ))

    scan_delivered_nothing = pinned and len(selected) == scan_entries_before
    if not scan_delivered_nothing:
        # Mutually exclusive with the block below, which is the same fact told
        # from the pins' side. Emitting both restates one cap twice in two
        # wordings, and a reader who has to reconcile two sentences about one
        # number learns to skim past the surface whose whole job is to be
        # believed.
        _budget_notes(
            config,
            adaptive_limit,
            dropped_by_file_cap=dropped_by_file_cap,
            dropped_by_byte_cap=dropped_by_byte_cap,
        )

    if scan_delivered_nothing:
        # Ruling 1 is "the named files AND keep scanning", so a scan that added
        # nothing is worth saying. What it is NOT worth is guessing why.
        #
        # A cap only gets named when it actually bound. The first version keyed
        # solely on "the scan added nothing" and printed both knobs, so a
        # three-file repo with 135 bytes pinned against a 300,000-byte ceiling
        # was told its budget was used up — sending the operator to two
        # settings that would change nothing. That is this repo's own rule
        # about never blaming a cap for an absence it did not cause, broken
        # inside the goal whose subject is selection truthfulness.
        # Which cap to name is decided by which one REMOVED candidates, in the
        # same accounting `_budget_notes` uses — and the BYTE cap is tested
        # first here, because it is applied second and is therefore the one
        # holding the margin when both bound. The reverse order named
        # `--max-files=30` on a run with 29 slots free.
        if dropped_by_byte_cap:
            progress.note(
                f"Warning: pinned files filled the byte budget "
                f"(--max-bytes={config.max_bytes:,}); the scan contributed "
                "nothing")
        elif dropped_by_file_cap:
            # The limit that bound is the ADAPTIVE one, and printing it as
            # `--max-files=25` names a number the operator never typed — worse,
            # below specificity 10 `calculate_adaptive_limit` returns a fixed
            # 15/20/25 regardless of `--max-files`, so the knob it names cannot
            # move the limit it reports. Both numbers, and the relationship.
            knob = (
                f"adaptive limit {adaptive_limit}, from --max-files="
                f"{config.max_files}"
                if adaptive_limit != config.max_files
                else f"--max-files={config.max_files}"
            )
            progress.note(
                f"Warning: pinned files filled the file budget ({knob}); "
                "the scan contributed nothing")
        else:
            # No cap bound. The scan simply found nothing else eligible, which
            # no setting changes — so it gets a bare statement and no remedy.
            progress.note(
                "The scan found nothing to add beside the pinned file(s)")

    return selected
