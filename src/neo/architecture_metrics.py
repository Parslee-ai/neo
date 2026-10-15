"""Lightweight architectural metrics for outcome modulation.

A session that accepted a suggestion *and* introduced an import cycle is a
weaker positive signal than one that accepted *and* didn't degrade structure.
We capture three cheap, defensible metrics at session-save time and diff
them at outcome-detection time to modulate confidence adjustments.

Metrics and language coverage:
- **God-file count** — multi-language (Python via ast; JS/TS/Java/C#/Go/
  Rust/C/C++ via tree-sitter when installed). LOC + function-count
  thresholds work uniformly across languages.
- **Import-cycle count** — Python-only. Cross-language import semantics
  (relative paths vs. dotted modules vs. package paths) don't reconcile
  into one graph without per-language module-name schemes; the cost
  isn't worth a noise-prone signal.
- **Max nesting depth** — Python-only. Per-language control-flow node
  maps would need ongoing maintenance for a metric that's already
  noise-banded.

All computation must be cheap and failure-tolerant — a metrics error must
never break the main path. "Cheap" holds through a per-file memo in the
repository's own `.neo/` (`MEMO_FILENAME`): a learn-mode run snapshots twice,
and parsing every source file each time cost 61 s of a 111 s request in a
21k-file repository (1.7 s on neo itself), so only files whose stat changed
are parsed again.

This is the native, self-contained version of sentrux's session-delta
idea: same shape, narrower set, no external dependency.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

from neo.eligibility import WalkPolicy, normalize_exts
from neo.index.language_parser import QUERIES, TreeSitterParser
from neo.languages import EXTENSION_TO_LANGUAGE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# A file is a "god file" if either threshold is breached. Picked so common
# entry-point modules don't get flagged but obvious dumping-grounds do.
_GOD_FILE_LOC_THRESHOLD = 800
_GOD_FILE_FUNC_THRESHOLD = 30

# Depth-delta jitter: small day-to-day noise shouldn't read as regression.
_DEPTH_NOISE_BAND = 1

#: Per-file results, beside the walk cache in the repository's own `.neo/`
#: (which the walker never descends into).
MEMO_FILENAME = "arch_cache.json"
#: Bumped whenever a per-file analyzer or a threshold above changes what it
#: returns for the same file: a stored result is then wrong in a way no stat
#: can detect, so the whole memo is dropped. The interpreter and parser
#: versions join it in `_memo_signature`, because the memo lives in the
#: repository and a PyPI install and a dev checkout share it.
ANALYZER_VERSION = 1


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArchSnapshot:
    """Cheap structural fingerprint of a codebase root.

    All fields are zeros when computation fails or no files were found, so a
    snapshot is always comparable to another snapshot.

    `python_files_scanned` is tracked separately because cycle_count and
    max_nesting_depth are Python-only metrics — a JS-only session leaves
    both at 0, indistinguishable from "Python was measured and is flat."
    Consumers (and `ArchDelta.severity`) use this to avoid reading
    no-coverage zeros as real signal.
    """
    cycle_count: int = 0
    god_file_count: int = 0
    max_nesting_depth: int = 0
    files_scanned: int = 0          # all source files walked
    python_files_scanned: int = 0   # subset that was eligible for cycle/depth

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ArchSnapshot":
        if not data:
            return cls()
        files = int(data.get("files_scanned", 0))
        # Legacy snapshots predate the python_files_scanned split; they were
        # Python-only by construction, so default to files_scanned when the
        # key is missing.
        python_files = int(data.get("python_files_scanned", files))
        return cls(
            cycle_count=int(data.get("cycle_count", 0)),
            god_file_count=int(data.get("god_file_count", 0)),
            max_nesting_depth=int(data.get("max_nesting_depth", 0)),
            files_scanned=files,
            python_files_scanned=python_files,
        )


Severity = Literal["improvement", "neutral", "regression"]


@dataclass(frozen=True)
class ArchDelta:
    """Difference between two snapshots (after - before).

    Positive deltas mean *worse* (more cycles, more god files, deeper
    nesting). `severity()` collapses the three into one label callers can
    branch on.

    `python_coverage` is true iff both snapshots actually measured Python
    files. When false, cycle/depth deltas are no-coverage zeros and
    `severity()` ignores those channels — only god_files_delta can drive
    the verdict. Defaults to True so manually-constructed test deltas
    behave like the legacy single-language regime.
    """
    cycles_delta: int
    god_files_delta: int
    max_depth_delta: int
    python_coverage: bool = True

    def severity(self) -> Severity:
        # God-file channel is language-agnostic and always trusted.
        if self.god_files_delta > 0:
            return "regression"
        if self.god_files_delta < 0:
            return "improvement"

        # Cycle and depth channels are Python-only. Without Python coverage
        # in both snapshots, their zero values are no-signal, not "flat."
        if not self.python_coverage:
            return "neutral"

        if (
            self.cycles_delta > 0
            or self.max_depth_delta > _DEPTH_NOISE_BAND
        ):
            return "regression"
        if (
            self.cycles_delta < 0
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
        source_files = list(_iter_source_files(root_path))
    except OSError as exc:
        logger.debug("architecture_metrics: walk failed under %s: %s", root_path, exc)
        return ArchSnapshot()

    if not source_files:
        return ArchSnapshot()

    # Per-file analysis (tolerates per-file failures via _analyze_file).
    god_count = 0
    max_depth = 0
    module_imports: dict[str, set[str]] = {}
    python_count = 0
    ts_parser: Optional[TreeSitterParser] = None
    ts_parser_built = False
    memo = _load_memo(root_path)
    fresh: dict[str, list] = {}

    for path in source_files:
        key, stamp = _memo_key(path, root_path)
        stored = memo.get(key)
        if stamp is not None and stored is not None and stored[0] == stamp:
            result = _result_from_memo(stored[1])
        elif path.suffix == ".py":
            result = _analyze_python_file(path, root_path)
        else:
            if not ts_parser_built:
                ts_parser, ts_parser_built = _lazy_ts_parser(), True
            result = _analyze_tree_sitter_file(path, root_path, ts_parser)
        # A file that yielded nothing is not memoized: the cause may be the
        # environment (no parser, an unreadable file, a parse exception), not
        # the file, and freezing it would misreport the file until it changed.
        # Measured: 3 of 15,183 files in a large repository.
        if stamp is not None and result is not None:
            fresh[key] = [stamp, _result_to_memo(result)]
        if result is None:
            continue
        if path.suffix == ".py":
            python_count += 1
        is_god, depth, module_name, imports = result
        if is_god:
            god_count += 1
        if depth > max_depth:
            max_depth = depth
        module_imports[module_name] = imports

    cycle_count = _count_cycles(module_imports)
    if fresh != memo:
        _save_memo(root_path, fresh)

    return ArchSnapshot(
        cycle_count=cycle_count,
        god_file_count=god_count,
        max_nesting_depth=max_depth,
        files_scanned=len(source_files),
        python_files_scanned=python_count,
    )


# ---------------------------------------------------------------------------
# Internals: per-file memo
# ---------------------------------------------------------------------------

def _memo_path(root: Path) -> Path:
    return root / ".neo" / MEMO_FILENAME


def _memo_signature() -> str:
    """What a stored result depends on besides the file itself."""
    from importlib import metadata

    from neo import __version__

    # Neo's own version too: the analyzers read queries and language maps from
    # other modules that can change without anyone bumping ANALYZER_VERSION.
    parts = [str(ANALYZER_VERSION), __version__, "py%d.%d" % sys.version_info[:2]]
    for package in ("tree-sitter", "tree-sitter-language-pack"):
        try:
            parts.append(f"{package}={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            parts.append(f"{package}=none")
    return " ".join(parts)


def _memo_key(path: Path, root: Path) -> tuple[str, Optional[list[int]]]:
    """(path relative to `root`, its stat stamp), the stamp None when unstattable.

    Size, mtime and ctime, so an edit that restores the old mtime still reads
    as changed.
    """
    try:
        key = str(path.relative_to(root))
    except ValueError:
        key = str(path)
    try:
        st = path.stat()
    except OSError:
        return key, None
    return key, [st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def _result_to_memo(result: Optional[tuple[bool, int, str, set[str]]]) -> Optional[list]:
    if result is None:
        return None
    is_god, depth, module_name, imports = result
    return [is_god, depth, module_name, sorted(imports)]


def _result_from_memo(stored: Optional[list]) -> Optional[tuple[bool, int, str, set[str]]]:
    if stored is None:
        return None
    is_god, depth, module_name, imports = stored
    return bool(is_god), int(depth), str(module_name), set(imports)


def _load_memo(root: Path) -> dict[str, list]:
    """The stored per-file results, or `{}` when absent, stale or damaged.

    Any entry that is not well formed discards the whole memo: it costs one
    full parse, where trusting half a memo would misreport the repository.
    """
    try:
        raw = json.loads(_memo_path(root).read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict) or raw.get("signature") != _memo_signature():
        return {}
    files = raw.get("files")
    if not isinstance(files, dict):
        return {}
    try:
        for stamp, stored in files.values():
            if len(stamp) != 3 or not all(isinstance(n, int) for n in stamp):
                return {}
            if stored is None:
                return {}
            _result_from_memo(stored)
    except (TypeError, ValueError):
        return {}
    return files


def _save_memo(root: Path, files: dict[str, list]) -> None:
    """Write the memo atomically (temp file plus `os.replace`); never raises.

    Two processes racing here can write results from different moments, and the
    last writer wins. That is safe because each entry carries the stamp taken
    BEFORE its file was parsed: a result older than the file holds an older
    stamp, so the next run parses that file again.
    """
    path = _memo_path(root)
    tmp_name = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=MEMO_FILENAME + ".", suffix=".tmp"
        )
        with os.fdopen(handle, "w") as out:
            json.dump({"signature": _memo_signature(), "files": files}, out)
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
        tmp_name = None
    except (OSError, TypeError, ValueError) as exc:
        logger.debug("architecture_metrics: memo not written under %s: %s", root, exc)
    finally:
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _lazy_ts_parser() -> Optional[TreeSitterParser]:
    """Construct a TreeSitterParser once per compute(), or None if its
    initialization fails (unlikely now that tree-sitter is a hard
    dependency, but harmless to keep the safety net).
    """
    try:
        return TreeSitterParser()
    except Exception as exc:  # noqa: BLE001
        logger.debug("architecture_metrics: TreeSitterParser init failed: %s", exc)
        return None


def compare(before: ArchSnapshot, after: ArchSnapshot) -> ArchDelta:
    """Return after - before as an ArchDelta.

    Sets python_coverage=True only when both snapshots actually measured
    Python files. Without that, cycle and depth zeros are no-coverage,
    not "flat," and severity() will ignore them.
    """
    return ArchDelta(
        cycles_delta=after.cycle_count - before.cycle_count,
        god_files_delta=after.god_file_count - before.god_file_count,
        max_depth_delta=after.max_nesting_depth - before.max_nesting_depth,
        python_coverage=before.python_files_scanned > 0 and after.python_files_scanned > 0,
    )


# ---------------------------------------------------------------------------
# Internals: file iteration
# ---------------------------------------------------------------------------

def _iter_source_files(root: Path) -> Iterable[Path]:
    """Yield source files across all supported languages.

    .py is always yielded (Python ast can analyze it standalone). Other
    supported extensions are only yielded when tree-sitter is available;
    otherwise yielding them would just waste a per-file open() that
    can't produce any signal.
    """
    # Only walk extensions whose languages have tree-sitter queries —
    # without QUERIES, parse_file returns [] and we'd produce no
    # function-count signal, just inflate files_scanned with no benefit.
    extras = {ext for ext, lang in EXTENSION_TO_LANGUAGE.items() if lang in QUERIES}
    extras.discard(".py")  # Python is handled by the ast path regardless

    # The shared walk, so this scan cannot disagree with prompt assembly and
    # the index about what counts as source. It replaced a third hand-written
    # ignore list; every name that list held is in the shared defaults, which
    # additionally honour the repo's own `.gitignore` — so generated code a
    # repo has declared generated stops being counted as a god file.
    #
    # Extension matching is exact and case-insensitive: `name.endswith(".c")`
    # would also match `foo.bc`, which is not a C file.
    wanted = {".py", *extras}
    # Through the cache, because this runs on the outcome-detection path of
    # every real invocation, not only on `--index`. The cache stores ignore
    # verdicts, which are policy-independent, so this share the one the
    # gatherer built and pays no walk of its own on a warm repository.
    from neo.index.walk_cache import cached_walk

    policy = WalkPolicy(exts=normalize_exts(wanted))
    # `record=False` / `quiet=True`: this is a secondary, extension-filtered
    # walk running after the answer is assembled. It must not narrate over the
    # result, and it must not become the walk `--dry-run` reports — that one is
    # the corpus walk, and this one's file count is a subset of it.
    for entry in cached_walk(str(root), policy, quiet=True, record=False).paths:
        yield Path(entry.path)


# ---------------------------------------------------------------------------
# Internals: per-file analysis
# ---------------------------------------------------------------------------

def _analyze_python_file(
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


# Tree-sitter chunk types that count as "a function" for god-file purposes.
# Mirrors the QUERIES dict in neo.index.language_parser. Known gap: the
# TypeScript / JavaScript queries don't extract `method_definition` nodes
# inside `class_body`, so a 50-method TS class will report `func_count==0`.
# In practice that file still trips the LOC threshold long before it
# reaches 30 methods, so the gap rarely changes the outcome — but it does
# mean god-file detection is LOC-only for class-heavy JS/TS files.
_FUNCTION_CHUNK_TYPES = frozenset({"function", "method"})


def _analyze_tree_sitter_file(
    path: Path,
    root: Path,
    parser: Optional[TreeSitterParser],
) -> Optional[tuple[bool, int, str, set[str]]]:
    """Tree-sitter equivalent of _analyze_python_file for non-Python source.

    Only the god-file signal is computed here — nesting depth and imports
    are returned as zero/empty so the cycle and depth metrics stay
    Python-only (see module docstring for rationale).
    """
    if parser is None:
        return None

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(source) > 1_000_000:
        return None

    loc = source.count("\n") + 1

    try:
        chunks = parser.parse_file(path, source)
    except Exception as exc:  # noqa: BLE001
        logger.debug("architecture_metrics: tree-sitter parse failed for %s: %s", path, exc)
        return None

    func_count = sum(1 for c in chunks if c.chunk_type in _FUNCTION_CHUNK_TYPES)
    is_god = loc > _GOD_FILE_LOC_THRESHOLD or func_count > _GOD_FILE_FUNC_THRESHOLD

    # Path-based module name keeps this from colliding with Python's
    # dotted scheme — different separator guarantees no overlap.
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    module_name = str(rel.with_suffix("")).replace("\\", "/")

    return is_god, 0, module_name, set()


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
