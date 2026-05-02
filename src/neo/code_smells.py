"""Code-smell scanner for the relevance-ranked file set.

Surfaced to the LM during context assembly so the reasoning step sees
"known issues in nearby code" alongside repository content. Detectors are
intentionally high-precision — false positives turn into prompt bloat that
makes the model worse, not better.

Scope:
- TODO/FIXME/HACK/XXX markers in any text file
- Python-specific stubs: `pass`-only / `...`-only / NotImplementedError
- Python bare `except:` (catches everything, including KeyboardInterrupt)
- Python swallowed errors: `except ...:` whose body is exactly pass / ...
- Hardcoded credentials matching well-known prefixes (OpenAI, AWS, GitHub)
"""

import ast
import re
from dataclasses import dataclass
from typing import Iterable, Optional, Union

from neo.models import ContextFile


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodeSmell:
    file_path: str
    line: int
    kind: str       # "todo" | "stub" | "bare_except" | "swallowed_except" | "secret"
    severity: str   # "info" | "warn" | "high"
    message: str
    snippet: str    # offending line, truncated for prompt safety


# Per-file finding cap. Keeps a single noisy file from drowning the prompt.
_PER_FILE_CAP = 8


def scan_files(files: Iterable[ContextFile]) -> list[CodeSmell]:
    """Scan a sequence of context files and return all findings.

    Each file is capped at `_PER_FILE_CAP` findings so a single
    pathological file can't dominate the prompt.
    """
    out: list[CodeSmell] = []
    for f in files:
        if not f.content:
            continue
        findings = _scan_one(f.path, f.content)
        out.extend(findings[:_PER_FILE_CAP])
    return out


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

# Markers we look for in comments. Word-boundary anchored so we don't match
# inside identifiers like `update_todo_list`.
_MARKER_PATTERN = re.compile(
    r"(?:#|//|/\*|\*|<!--)\s*(TODO|FIXME|HACK|XXX)\b[:\s]?(.*)",
    re.IGNORECASE,
)
_MARKER_SEVERITY = {"TODO": "info", "FIXME": "info", "HACK": "warn", "XXX": "warn"}

# Conservative secret patterns: only well-known prefixed shapes. Generic
# high-entropy detection is intentionally out of scope (too noisy).
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI key", re.compile(r"sk-[A-Za-z0-9_\-]{20,}")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bghp_[A-Za-z0-9]{20,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
]


def _scan_one(path: str, content: str) -> list[CodeSmell]:
    findings: list[CodeSmell] = []

    findings.extend(_scan_markers(path, content))
    findings.extend(_scan_secrets(path, content))

    if path.endswith(".py"):
        findings.extend(_scan_python(path, content))

    return findings


def _scan_markers(path: str, content: str) -> list[CodeSmell]:
    out: list[CodeSmell] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        m = _MARKER_PATTERN.search(line)
        if not m:
            continue
        marker = m.group(1).upper()
        note = (m.group(2) or "").strip().rstrip("*/-> ").strip()
        out.append(CodeSmell(
            file_path=path,
            line=line_no,
            kind="todo",
            severity=_MARKER_SEVERITY[marker],
            message=f"{marker}{(': ' + note) if note else ''}",
            snippet=_truncate(line),
        ))
    return out


def _scan_secrets(path: str, content: str) -> list[CodeSmell]:
    out: list[CodeSmell] = []
    for line_no, line in enumerate(content.splitlines(), start=1):
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(line):
                out.append(CodeSmell(
                    file_path=path,
                    line=line_no,
                    kind="secret",
                    severity="high",
                    message=f"hardcoded credential pattern ({label})",
                    snippet=_truncate(line),
                ))
                break  # one finding per line is enough
    return out


def _scan_python(path: str, content: str) -> list[CodeSmell]:
    """AST-based Python detectors. Falls through silently on parse error."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    lines = content.splitlines()
    out: list[CodeSmell] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stub_kind = _detect_stub(node)
            if stub_kind:
                out.append(CodeSmell(
                    file_path=path,
                    line=node.lineno,
                    kind="stub",
                    severity="warn",
                    message=f"function `{node.name}` is a stub ({stub_kind})",
                    snippet=_truncate(_line_at(lines, node.lineno)),
                ))
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                out.append(CodeSmell(
                    file_path=path,
                    line=node.lineno,
                    kind="bare_except",
                    severity="warn",
                    message="bare `except:` swallows KeyboardInterrupt and SystemExit",
                    snippet=_truncate(_line_at(lines, node.lineno)),
                ))
            if _body_is_silent(node.body):
                out.append(CodeSmell(
                    file_path=path,
                    line=node.lineno,
                    kind="swallowed_except",
                    severity="warn",
                    message="exception caught and silently dropped",
                    snippet=_truncate(_line_at(lines, node.lineno)),
                ))

    return out


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _detect_stub(func: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[str]:
    """Return a short label for the stub kind, or None if the function
    has real content. Docstrings don't count as content.
    """
    body = list(func.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # strip docstring

    if not body:
        # Empty after docstring strip — unreachable in valid Python, but safe to skip.
        return None

    if len(body) == 1:
        only = body[0]
        if isinstance(only, ast.Pass):
            return "pass-only body"
        if (
            isinstance(only, ast.Expr)
            and isinstance(only.value, ast.Constant)
            and only.value.value is Ellipsis
        ):
            return "ellipsis-only body"
        if isinstance(only, ast.Raise) and _raises_not_implemented(only):
            return "raises NotImplementedError"
    return None


def _raises_not_implemented(node: ast.Raise) -> bool:
    exc = node.exc
    if exc is None:
        return False
    if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
        return True
    if (
        isinstance(exc, ast.Call)
        and isinstance(exc.func, ast.Name)
        and exc.func.id == "NotImplementedError"
    ):
        return True
    return False


def _body_is_silent(body: list[ast.stmt]) -> bool:
    """True iff the except-handler body is exactly `pass` or `...`."""
    if len(body) != 1:
        return False
    only = body[0]
    if isinstance(only, ast.Pass):
        return True
    if (
        isinstance(only, ast.Expr)
        and isinstance(only.value, ast.Constant)
        and only.value.value is Ellipsis
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _line_at(lines: list[str], line_no: int) -> str:
    idx = line_no - 1
    if 0 <= idx < len(lines):
        return lines[idx]
    return ""


def _truncate(text: str, limit: int = 160) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

def format_for_prompt(smells: list[CodeSmell], *, max_findings: int = 20) -> str:
    """Render findings as a compact prompt section. Empty string when nothing
    was found, so the caller can concatenate without conditional logic.
    """
    if not smells:
        return ""

    # Sort high → warn → info, then by file/line for stable output.
    severity_rank = {"high": 0, "warn": 1, "info": 2}
    ordered = sorted(
        smells,
        key=lambda s: (severity_rank.get(s.severity, 9), s.file_path, s.line),
    )[:max_findings]

    lines = ["", "KNOWN ISSUES IN NEARBY CODE (consider whether these affect the task):"]
    for s in ordered:
        lines.append(f"- {s.file_path}:{s.line} [{s.kind}/{s.severity}] {s.message}")
    if len(smells) > max_findings:
        lines.append(f"- (+{len(smells) - max_findings} more findings suppressed)")
    return "\n".join(lines)
