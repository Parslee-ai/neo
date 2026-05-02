"""Lightweight Python-only architectural metrics for outcome modulation.

A session that accepted a suggestion *and* introduced an import cycle is a
weaker positive signal than one that accepted *and* didn't degrade structure.
We capture three cheap, defensible metrics at session-save time and diff
them at outcome-detection time to modulate confidence adjustments.

Scope is narrow on purpose:
- Only Python files (.py). Cross-language is out of scope for V1.
- Three metrics: import-cycle count, god-file count, max nesting depth.
- All computation must be cheap (sub-second on neo-sized repos) and
  failure-tolerant — a metrics error must never break the main path.

This is the native, self-contained version of sentrux's session-delta
idea: same shape, narrower set, no external dependency.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directories we never descend into. Common venv / build / cache layouts.
_IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn",
    ".venv", "venv", "env", ".env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    "node_modules",
    "build", "dist", "site-packages",
    ".neo",  # neo's own state dir if scanned
})

# A file is a "god file" if either threshold is breached. Picked so common
# entry-point modules don't get flagged but obvious dumping-grounds do.
_GOD_FILE_LOC_THRESHOLD = 800
_GOD_FILE_FUNC_THRESHOLD = 30

# Depth-delta jitter: small day-to-day noise shouldn't read as regression.
_DEPTH_NOISE_BAND = 1


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchSnapshot:
    """Cheap structural fingerprint of a Python codebase root.

    All fields are zeros when computation fails or no files were found, so a
    snapshot is always comparable to another snapshot.
    """
    cycle_count: int = 0
    god_file_count: int = 0
    max_nesting_depth: int = 0
    files_scanned: int = 0  # sanity field; not used in delta

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ArchSnapshot":
        if not data:
            return cls()
        return cls(
            cycle_count=int(data.get("cycle_count", 0)),
            god_file_count=int(data.get("god_file_count", 0)),
            max_nesting_depth=int(data.get("max_nesting_depth", 0)),
            files_scanned=int(data.get("files_scanned", 0)),
        )


Severity = Literal["improvement", "neutral", "regression"]


@dataclass(frozen=True)
class ArchDelta:
    """Difference between two snapshots (after - before).

    Positive deltas mean *worse* (more cycles, more god files, deeper
    nesting). `severity()` collapses the three into one label callers can
    branch on.
    """
    cycles_delta: int
    god_files_delta: int
    max_depth_delta: int

    def severity(self) -> Severity:
        # Any new cycle or god file is meaningful. Depth-deltas need to
        # exceed the noise band — minor refactors can wiggle nesting by 1.
        if (
            self.cycles_delta > 0
            or self.god_files_delta > 0
            or self.max_depth_delta > _DEPTH_NOISE_BAND
        ):
            return "regression"
        if (
            self.cycles_delta < 0
            or self.god_files_delta < 0
            or self.max_depth_delta < -_DEPTH_NOISE_BAND
        ):
            return "improvement"
        return "neutral"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute(root: Optional[Path | str]) -> ArchSnapshot:
    """Compute a snapshot of the Python code under `root`.

    Never raises — on any failure, returns the zero snapshot. Callers should
    treat zero as "we have no signal" rather than "everything is great."
    """
    if root is None:
        return ArchSnapshot()
    try:
        root_path = Path(root).expanduser()
    except (OSError, ValueError):
        return ArchSnapshot()
    if not root_path.is_dir():
        return ArchSnapshot()

    try:
        py_files = list(_iter_python_files(root_path))
    except OSError as exc:
        logger.debug("architecture_metrics: walk failed under %s: %s", root_path, exc)
        return ArchSnapshot()

    if not py_files:
        return ArchSnapshot()

    # Per-file analysis (tolerates per-file failures via _analyze_file).
    god_count = 0
    max_depth = 0
    module_imports: dict[str, set[str]] = {}

    for path in py_files:
        result = _analyze_file(path, root_path)
        if result is None:
            continue
        is_god, depth, module_name, imports = result
        if is_god:
            god_count += 1
        if depth > max_depth:
            max_depth = depth
        module_imports[module_name] = imports

    cycle_count = _count_cycles(module_imports)

    return ArchSnapshot(
        cycle_count=cycle_count,
        god_file_count=god_count,
        max_nesting_depth=max_depth,
        files_scanned=len(py_files),
    )


def compare(before: ArchSnapshot, after: ArchSnapshot) -> ArchDelta:
    """Return after - before as an ArchDelta."""
    return ArchDelta(
        cycles_delta=after.cycle_count - before.cycle_count,
        god_files_delta=after.god_file_count - before.god_file_count,
        max_depth_delta=after.max_nesting_depth - before.max_nesting_depth,
    )


# ---------------------------------------------------------------------------
# Internals: file iteration
# ---------------------------------------------------------------------------

def _iter_python_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in _safe_walk(root):
        # Mutate dirnames in-place so os.walk skips ignored directories.
        dirnames[:] = [d for d in dirnames if d not in _IGNORE_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def _safe_walk(root: Path):
    import os
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        yield dirpath, dirnames, filenames


# ---------------------------------------------------------------------------
# Internals: per-file analysis
# ---------------------------------------------------------------------------

def _analyze_file(
    path: Path, root: Path
) -> Optional[tuple[bool, int, str, set[str]]]:
    """Return (is_god, max_depth_in_file, module_name, imports) or None.

    None signals "skip this file" — unreadable, oversized binary, parse error.
    """
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # Skip pathologically large generated files quickly — they aren't useful
    # signal and can hurt scan latency.
    if len(source) > 1_000_000:
        return None

    loc = source.count("\n") + 1

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    func_count = sum(
        1 for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    is_god = loc > _GOD_FILE_LOC_THRESHOLD or func_count > _GOD_FILE_FUNC_THRESHOLD

    max_depth = _max_function_nesting(tree)
    module_name = _module_name(path, root)
    imports = _module_imports(tree, module_name)

    return is_god, max_depth, module_name, imports


def _max_function_nesting(tree: ast.AST) -> int:
    """Max nesting depth of control-flow inside any function in the tree.

    Each function is measured independently — when we encounter a nested
    function/class definition, we do NOT recurse through it (those will be
    visited as their own top-level entries from the outer loop). Otherwise a
    nested helper's `if` would inflate its enclosing function's depth.
    """
    nesting_kinds: tuple[type, ...] = (
        ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith,
        ast.Try,
    )
    if hasattr(ast, "TryStar"):
        nesting_kinds = nesting_kinds + (ast.TryStar,)
    if hasattr(ast, "Match"):
        nesting_kinds = nesting_kinds + (ast.Match,)

    boundary_kinds: tuple[type, ...] = (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
    )

    overall = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            overall = max(overall, _walk_depth(node, 0, nesting_kinds, boundary_kinds))
    return overall


def _walk_depth(
    node: ast.AST,
    depth: int,
    kinds: tuple[type, ...],
    boundaries: tuple[type, ...],
) -> int:
    deepest = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, boundaries) and child is not node:
            # Don't dive into nested defs/classes — they're measured separately.
            continue
        if isinstance(child, kinds):
            deepest = max(deepest, _walk_depth(child, depth + 1, kinds, boundaries))
        else:
            deepest = max(deepest, _walk_depth(child, depth, kinds, boundaries))
    return deepest


# ---------------------------------------------------------------------------
# Internals: module names & imports
# ---------------------------------------------------------------------------

def _module_name(path: Path, root: Path) -> str:
    """Convert a file path under `root` to a dotted module name.

    Handles __init__.py (collapses to package name). Result is best-effort —
    we use it only to key the cycle graph.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else path.stem


def _module_imports(tree: ast.AST, module_name: str) -> set[str]:
    """Extract every imported dotted name from `tree`.

    External-package imports are kept here intentionally; the cycle detector
    naturally drops any edge that points to a module outside the scanned
    graph (`if nxt not in graph: continue`). This avoids a fragile heuristic
    for "is this import internal" — flat-layout repos that don't have a
    common package prefix would otherwise miss real cycles.
    """
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if not node.module:
                continue
            out.add(node.module)
            # `from pkg import sub` could mean either an attribute of pkg or
            # the submodule pkg.sub. We can't disambiguate without resolving
            # bindings, so emit both edges; the cycle detector will only
            # follow the one that actually exists in the scanned graph.
            for alias in node.names:
                if alias.name and alias.name != "*":
                    out.add(f"{node.module}.{alias.name}")
    out.discard(module_name)  # never self-loop
    return out


# ---------------------------------------------------------------------------
# Internals: cycle detection (Tarjan SCC)
# ---------------------------------------------------------------------------

def _count_cycles(graph: dict[str, set[str]]) -> int:
    """Count strongly-connected components of size >= 2 in `graph`.

    Self-loops are excluded (we never store them — see _module_imports). Uses
    Tarjan's algorithm with an explicit work stack to avoid Python recursion
    limits on large repos.
    """
    if not graph:
        return 0

    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = [0]
    cycle_count = 0

    def visit(start: str) -> None:
        nonlocal cycle_count
        # Explicit DFS to handle deep import chains without recursion.
        work: list[tuple[str, Iterable[str]]] = [(start, iter(graph.get(start, ())))]
        index[start] = lowlink[start] = counter[0]
        counter[0] += 1
        stack.append(start)
        on_stack.add(start)

        while work:
            node, neighbors = work[-1]
            try:
                nxt = next(neighbors)
            except StopIteration:
                # Finished node — pop SCC root if applicable.
                if lowlink[node] == index[node]:
                    component_size = 0
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        component_size += 1
                        if w == node:
                            break
                    if component_size >= 2:
                        cycle_count += 1
                work.pop()
                if work:
                    parent = work[-1][0]
                    lowlink[parent] = min(lowlink[parent], lowlink[node])
                continue

            if nxt not in index:
                if nxt not in graph:
                    # Imported module not in the local graph (e.g. neighbor
                    # whose source we couldn't parse). Skip — can't form a cycle.
                    continue
                index[nxt] = lowlink[nxt] = counter[0]
                counter[0] += 1
                stack.append(nxt)
                on_stack.add(nxt)
                work.append((nxt, iter(graph.get(nxt, ()))))
            elif nxt in on_stack:
                lowlink[node] = min(lowlink[node], index[nxt])

    for node in graph:
        if node not in index:
            visit(node)

    return cycle_count
