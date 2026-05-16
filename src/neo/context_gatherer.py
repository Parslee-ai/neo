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
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Constants
MIN_SCORE_THRESHOLD = 0.2  # Filter files with very low relevance (was 0.3, reduced for broad prompts)
MAX_CHUNKS_PER_FILE = 2    # Cap chunks per file so one large file doesn't dominate the budget


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
    """Load patterns from .gitignore and .ignore files."""
    patterns = []

    # Default ignore patterns
    patterns.extend([
        '*.pyc', '__pycache__', '.git', '.svn', '.hg',
        'node_modules', '.env', '*.key', '*.pem', '*.secret',
        '.neo', 'venv', 'env', '.venv', 'dist', 'build',
        '*.egg-info', '.tox', '.coverage', 'htmlcov',
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


def should_ignore(rel_path: str, patterns: list[str], is_dir: bool = False) -> bool:
    """Check if path matches any ignore pattern."""
    path_with_slash = rel_path + '/' if is_dir else rel_path

    for pattern in patterns:
        # Handle directory-specific patterns
        if pattern.endswith('/'):
            if is_dir and fnmatch.fnmatch(path_with_slash, pattern):
                return True
        # Handle negation patterns
        elif pattern.startswith('!'):
            continue
        # Standard glob matching
        elif fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(path_with_slash, pattern):
            return True
        # Match pattern anywhere in path
        elif '/' not in pattern and fnmatch.fnmatch(os.path.basename(rel_path), pattern):
            return True

    return False


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
                    git_recent: set[str], entry_points: set[str]) -> float:
    """Score a candidate file for relevance."""
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
    main_impl_stems = {"core", "engine", "main", "index", "app", "server", "lib"}
    stem = os.path.splitext(basename)[0]
    is_main_impl = stem in main_impl_stems
    if is_main_impl:
        score += 0.4

    # Keyword overlap in filename
    hits = sum(1 for token in prompt_tokens if token in name_lower)
    score += 0.6 * min(hits, 3)

    # Git recency bonus
    if rel_path in git_recent:
        score += 0.3

    # Entry point bonus
    if any(basename.startswith(ep) for ep in entry_points):
        score += 0.2

    # Penalize by depth
    depth = rel_path.count(os.sep)
    score -= 0.05 * depth

    # Penalize by size (god objects are code smell). main_impl files
    # (engine.py, server.py, app.py, etc.) get a much gentler penalty
    # — they're large *because* they're central, and the old 0.002
    # multiplier was pushing THE relevant file (93KB engine.py) below
    # threshold on prompts about it. New main_impl penalty is 0.001
    # per KB-over-50, so 93KB loses 0.043 instead of 0.086.
    size_kb = size / 1024
    if size_kb > 10 and not is_main_impl:
        # Penalty for large files: 10KB = -0.1, 50KB = -0.5, 100KB = -1.0
        score -= 0.01 * size_kb
    elif size_kb > 50 and is_main_impl:
        # Lighter penalty for main implementation files: 50KB = 0,
        # 100KB = -0.05, 500KB = -0.45.
        score -= 0.001 * (size_kb - 50)

    return max(0.0, score)


def select_chunks(content: str, prompt_tokens: set[str], max_chunk_bytes: int = 12_000) -> list[tuple[str, int, int]]:
    """Select relevant chunks from large file content."""
    lines = content.splitlines()

    if len(content) <= max_chunk_bytes:
        return [(content, 1, len(lines))]

    # Find lines with keyword matches
    matching_idxs = [
        i for i, line in enumerate(lines)
        if any(token in line.lower() for token in prompt_tokens)
    ]

    if not matching_idxs:
        # No matches, return header + first N lines
        header_size = min(200, len(lines))
        chunk = '\n'.join(lines[:header_size])
        return [(chunk, 1, header_size)]

    # Build windows around matches
    chunks = []
    window_size = 40

    for idx in matching_idxs[:5]:  # Limit to 5 windows
        start = max(0, idx - window_size)
        end = min(len(lines), idx + window_size)
        chunk = '\n'.join(lines[start:end])
        chunks.append((chunk, start + 1, end))

        if sum(len(c[0]) for c in chunks) >= max_chunk_bytes:
            break

    return chunks


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
            return {}
        chunks = index.retrieve(prompt, k=k)

        # Test files often contain the prompt's literal keywords (because
        # they assert against named behaviors), so the FAISS index ranks
        # them above the source file the prompt is actually about.
        # Demote test-file hits unless the prompt is itself about testing.
        prompt_lower = prompt.lower()
        prompt_is_test = any(
            t in prompt_lower for t in ("test", "pytest", "unit test", "spec")
        )
        TEST_PENALTY = 0.4  # multiplier; tests retain 60% of their cosine

        boost: dict[str, float] = {}
        for chunk in chunks:
            # Normalize path the same way the rest of the gatherer does.
            rel = os.path.relpath(chunk.file_path, root)
            sim = float(getattr(chunk, "similarity", 0.0))
            sim = max(0.0, sim)
            is_test = (
                rel.startswith("test")
                or rel.startswith("tests" + os.sep)
                or "/tests/" in rel
                or os.path.basename(rel).startswith("test_")
            )
            if is_test and not prompt_is_test:
                sim *= TEST_PENALTY
            # 1.0 cosine = +1.0 boost (dominant signal); test demotion above.
            prev = boost.get(rel, 0.0)
            boost[rel] = max(prev, sim)
        if boost:
            print(
                f"[Neo] ProjectIndex boost: {len(boost)} files matched semantically",
                file=sys.stderr,
            )
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
        print(
            f"[Neo] EPISODE-history boost: {len(boost)} files seen in past similar runs",
            file=sys.stderr,
        )
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
    print(f"[Neo] Adaptive limit: {adaptive_limit} files (based on prompt specificity)", file=sys.stderr)

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

    # Score all candidates
    scored = []
    for abs_path, rel_path, size in candidates:
        score = score_candidate(rel_path, size, prompt_tokens, git_recent, entry_points)
        if score > 0:
            scored.append((abs_path, rel_path, size, score))

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
            print(f"[Neo] Broad prompt detected: including {len(arch_files[:5])} architectural files", file=sys.stderr)

    # If no files pass threshold, keep top 10 anyway to avoid empty results
    if not scored_filtered and scored_before_filter > 0:
        print(f"[Neo] Warning: All files scored below {MIN_SCORE_THRESHOLD}, using top 10", file=sys.stderr)
        scored = scored[:10]
    else:
        filtered_count = scored_before_filter - len(scored_filtered)
        if filtered_count > 0:
            print(f"[Neo] Filtered {filtered_count} low-relevance files (score < {MIN_SCORE_THRESHOLD})", file=sys.stderr)
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
        print(
            f"[Neo] Symbol-relevance boost applied to {symbol_hit_count} files",
            file=sys.stderr,
        )

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
                        print(f"[Neo] Warning: {rel_path} is {size_kb:.0f}KB - consider refactoring into smaller modules", file=sys.stderr)
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
        print(f"[Neo] No semantic index found at {index_path}", file=sys.stderr)
        print("[Neo] Falling back to keyword search. Run 'neo index' to build semantic index.", file=sys.stderr)
        return gather_context(config)

    # Load ProjectIndex
    try:
        from neo.index.project_index import ProjectIndex

        start_time = time.time()
        index = ProjectIndex(root)

        # Retrieve top 100 chunks
        chunks = index.retrieve(config.prompt, k=100)

        if not chunks:
            print("[Neo] No chunks found in semantic index", file=sys.stderr)
            print("[Neo] Falling back to keyword search", file=sys.stderr)
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

        print(f"[Neo] Semantic search: {len(selected_chunks)} chunks from {len(set(cf.rel_path for cf in context_files))} files in {elapsed*1000:.0f}ms", file=sys.stderr)

        return context_files

    except ImportError as e:
        print(f"[Neo] Failed to load ProjectIndex: {e}", file=sys.stderr)
        print("[Neo] Falling back to keyword search", file=sys.stderr)
        return gather_context(config)
    except Exception as e:
        print(f"[Neo] Semantic search error: {e}", file=sys.stderr)
        print("[Neo] Falling back to keyword search", file=sys.stderr)
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
        print(f"[Neo] Warning: Failed to log metrics: {e}", file=sys.stderr)
