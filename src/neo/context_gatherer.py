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


def iter_paths(root: str, includes: list[str], excludes: list[str], exts: Optional[list[str]]) -> list[tuple[str, str]]:
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

    # Boost main implementation files for broad queries
    main_impl_patterns = ['neo.py', 'persistent', 'context_gatherer', 'structured_parser', 'schemas.py']
    if any(pat in basename for pat in main_impl_patterns):
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

    # Penalize by size (god objects are code smell), but not for main implementation
    is_main_impl = any(pat in basename for pat in main_impl_patterns)
    size_kb = size / 1024
    if size_kb > 10 and not is_main_impl:
        # Penalty for large files: 10KB = -0.1, 50KB = -0.5, 100KB = -1.0
        score -= 0.01 * size_kb
    elif size_kb > 50 and is_main_impl:
        # Lighter penalty for main implementation files: 50KB = -0.1, 100KB = -0.2
        score -= 0.002 * (size_kb - 50)

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


def gather_context(config: GatherConfig) -> list[ContextFile]:
    """Main context gathering pipeline."""
    root = config.root
    prompt_tokens = extract_prompt_tokens(config.prompt)

    # Calculate adaptive file limit based on prompt specificity
    adaptive_limit = calculate_adaptive_limit(config.prompt, config.max_files)
    print(f"[Neo] Adaptive limit: {adaptive_limit} files (based on prompt specificity)", file=sys.stderr)

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
        print(f"[Neo] Falling back to keyword search. Run 'neo index' to build semantic index.", file=sys.stderr)
        return gather_context(config)

    # Load ProjectIndex
    try:
        # Import here to avoid circular dependency
        # Add src/index to path if needed
        src_index_path = Path(__file__).parent / "src" / "index"
        if src_index_path.exists():
            sys.path.insert(0, str(src_index_path.parent))

        from index.project_index import ProjectIndex

        start_time = time.time()
        index = ProjectIndex(root)

        # Retrieve top 100 chunks
        chunks = index.retrieve(config.prompt, k=100)

        if not chunks:
            print(f"[Neo] No chunks found in semantic index", file=sys.stderr)
            print(f"[Neo] Falling back to keyword search", file=sys.stderr)
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
        print(f"[Neo] Falling back to keyword search", file=sys.stderr)
        return gather_context(config)
    except Exception as e:
        print(f"[Neo] Semantic search error: {e}", file=sys.stderr)
        print(f"[Neo] Falling back to keyword search", file=sys.stderr)
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
