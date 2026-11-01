"""
Static analysis tool integrations for Neo.

All tools run in read-only/check-only mode.

Adding a new language analyzer:
1. Implement `run_<tool>_check(suggestion) -> StaticCheckResult`.
2. Add the tool's CLI name to `_KNOWN_TOOLS` (drives PATH detection).
3. Register a `_LanguageChecker` in `_LANGUAGE_CHECKERS` for the
   extensions it applies to.
4. Add an `enable_<tool>` kwarg to `run_static_checks` and pass it
   through via `_enable_map`.
"""

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from neo.models import CodeSuggestion, StaticCheckResult


# ============================================================================
# Python - Ruff
# ============================================================================

# Every checker below shells out to an external tool. None of these calls was
# bounded, so a wedged `pyright` or an `eslint` waiting on a lockfile could
# hang a neo run indefinitely — in the one phase whose whole job is to make the
# output trustworthy. Bounded here so the checks are cheap enough to run
# UNCONDITIONALLY (see EngineCore's static-check phase); a checker that times
# out reports that plainly rather than silently contributing no diagnostics.
STATIC_CHECK_TIMEOUT_SECONDS = 30

# Bounding each SUBPROCESS is not bounding the PHASE, and conflating the two is
# how removing the old time gate could have replaced "never verifies" with "can
# run for half an hour". Each (suggestion, checker) pair costs up to two bounded
# subprocesses — `patch` inside apply_diff_to_content, then the tool — and
# nothing caps how many suggestions arrive. 10 suggestions x 3 checkers x 60s is
# 30 minutes in a foreground phase.
#
# So the phase carries its own monotonic deadline. Checks that do not start
# before it expires are reported as `skipped` with a reason, never as passed and
# never as silently absent.
STATIC_CHECK_PHASE_BUDGET_SECONDS = 180

# Under `--fast`: the Claude Code mod runs its automatic reviews with it and
# gives the whole run 90 s, of which the model call alone takes 11-27 s. A
# 180 s phase can only end one way there. Checks that do not fit are reported
# as skipped, which reads as unverified, never as clean.
FAST_STATIC_CHECK_PHASE_BUDGET_SECONDS = 20


# ============================================================================
# What a checker is handed
# ============================================================================

@dataclass
class PreparedChange:
    """One suggestion as the file it would produce, prepared once.

    ``original`` is the file before the change (None for a new file): only
    the diagnostics the change INTRODUCED count, and counting the whole file
    blocked promotion for every suggestion against a file with one old lint
    error. ``target`` and ``root`` are where the file really lives, so a
    checker can read the project's own configuration.
    """
    file_path: str
    content: str
    timeout: float = STATIC_CHECK_TIMEOUT_SECONDS
    original: Optional[str] = None
    target: Optional[Path] = None
    root: Optional[str] = None


def _resolve_target(file_path: str, codebase_root: Optional[str]) -> Path:
    """The file a suggestion names, the way outcome detection resolves it.
    Read relative to the process's cwd it named a different file, or none."""
    if codebase_root:
        from neo.memory.outcomes import suggested_file

        located = suggested_file(file_path, codebase_root)
        if located is not None:
            return located
    return Path(file_path)


_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _new_file_content(unified_diff: str) -> Optional[str]:
    """The file a creation diff writes, or None if it is not one, or is not
    a well-formed one.

    Only consulted when the target does not exist (on a file that does,
    `-0,0` is an insertion at the top). Validated here because `patch` will
    not apply a creation diff to an empty file: exactly one file section,
    every hunk `-0,0 +1,N` with exactly N added lines and nothing else. The
    added lines of every section used to be gathered without a check, so a
    diff `patch` would reject, or another file's lines, read as this file.
    """
    lines = unified_diff.splitlines()
    if sum(1 for line in lines if line.startswith("+++ ")) > 1:
        return None
    added: list[str] = []
    expected = 0
    in_hunk = False
    for line in lines:
        header = _HUNK_HEADER.match(line)
        if header:
            if expected or header.group(1) != "0" or (header.group(2) or "1") != "0":
                return None
            expected = int(header.group(4) or "1")
            in_hunk = True
            continue
        if not in_hunk or line.startswith(("--- ", "+++ ")):
            continue
        if not line.startswith("+") or not expected:
            return None
        added.append(line[1:])
        expected -= 1
    if expected or not added:
        return None
    return "\n".join(added) + "\n"


def prepare_change(
    suggestion: CodeSuggestion,
    codebase_root: Optional[str] = None,
    timeout: float = STATIC_CHECK_TIMEOUT_SECONDS,
) -> tuple[Optional[PreparedChange], str]:
    """The file ``suggestion`` would produce, or ``(None, why it can't be
    checked)``.

    Never a guess. A diff `patch` rejects, a target that cannot be read, a
    header-only or blank diff and a code block with no diff are all reported
    as not checked. Each used to reach a checker as SOME file (the unmodified
    original, a reconstruction from the diff's lines) that linted clean and
    read as "passed" for a patch that was never applied.

    A diff already in the file (it applies in REVERSE, not forwards) is
    checked as the change it is: the file as it stands is the after, the
    reverse-applied file the before. `patch --batch` used to reverse it
    silently and the BEFORE was checked as the change, which is what VERIFY
    is asked about a change just made.
    """
    from neo.models import diff_changes_lines

    diff = suggestion.unified_diff or ""
    if not diff_changes_lines(diff):
        if (suggestion.code_block or "").strip():
            return None, "a code block with no diff cannot be placed in its file"
        return None, "no change to check"
    target = _resolve_target(suggestion.file_path, codebase_root)
    if not target.exists():
        content = _new_file_content(diff)
        if content is None:
            return None, f"{suggestion.file_path} does not exist to apply the diff to"
        return PreparedChange(suggestion.file_path, content, timeout,
                              target=target, root=codebase_root), ""
    try:
        current = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, f"cannot read {suggestion.file_path} to apply the diff"
    started = time.monotonic()
    after, why = apply_diff_to_content(diff, current, timeout=timeout)
    if after is not None:
        before = current
    else:
        # Already applied only if the diff ADDS lines: those lines being in
        # the file is the evidence. A pure deletion of a line that was never
        # there also applies in reverse (it only needs its context), and was
        # read as a change already made and checked clean.
        left = timeout - (time.monotonic() - started)
        adds = any(line.startswith("+") and not line.startswith("+++")
                   for line in diff.splitlines())
        if left <= 0 or not adds:
            return None, why
        before, _ = apply_diff_to_content(diff, current, timeout=left, reverse=True)
        if before is None:
            return None, why
        after = current
    if after == before:
        return None, "the diff changes nothing in the file"
    return PreparedChange(suggestion.file_path, after, timeout, original=before,
                          target=target, root=codebase_root), ""


def apply_diff_to_content(
    unified_diff: str, original_content: str,
    timeout: float = STATIC_CHECK_TIMEOUT_SECONDS, reverse: bool = False,
) -> tuple[Optional[str], str]:
    """``(patched content, "")``, or ``(None, why)`` when `patch` did not
    apply the diff exactly.

    - The exit status is read. It never was: a diff that did not apply left
      the original untouched, the checkers linted THAT, and it read as passed.
    - `--forward`: with `--batch` alone, a diff that looks reversed is
      applied in reverse without a word.
    - `--fuzz=0`: fuzz lets a hunk land where its context only half matches,
      and the file checked is then not the file proposed. A diff with wrong
      context is reported as not applied; line OFFSETS are still accepted.
    - A private directory, so a reject file or backup cannot outlive it.
    - stdin=DEVNULL: `patch` prompts on a malformed diff and otherwise
      inherits neo's own stdin, the one a host may be feeding the prompt on.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="neo-check-") as work:
            target = Path(work) / "target"
            change = Path(work) / "change.diff"
            target.write_text(original_content, encoding="utf-8")
            change.write_text(unified_diff, encoding="utf-8")
            result = subprocess.run(
                ["patch", "--batch", "--forward", "--fuzz=0", "-p0",
                 *(["--reverse"] if reverse else []), str(target), str(change)],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                cwd=work,
                timeout=timeout,
            )
            if result.returncode != 0:
                return None, "the diff did not apply to the file"
            return target.read_text(encoding="utf-8", errors="replace"), ""
    except subprocess.TimeoutExpired:
        # TimeoutExpired is a SubprocessError, not an OSError: unnamed, it
        # escaped into the calling checker and read as that checker timing out.
        return None, f"applying the diff timed out after {timeout:g}s"
    except FileNotFoundError:
        return None, "`patch` is not installed, so the diff could not be applied"
    except OSError as exc:
        return None, f"the diff could not be applied: {exc}"


def _run_tool(
    tool: str, argv: list[str], prepared: PreparedChange, suffix: str,
    parse: Callable[[subprocess.CompletedProcess], list[dict]],
    install_hint: str,
    stdin_argv: Optional[list[str]] = None,
) -> StaticCheckResult:
    """Run ``tool`` on ``prepared.content`` and parse what it reports.

    ``stdin_argv`` (ruff) feeds the content on stdin under the file's real
    path from the project's own directory, so the project's configuration
    (ignores, per-file ignores, selected rules) applies; otherwise the content
    is written to a private file and ``argv + [file]`` runs on it.

    Exit status 0 or 1 is a verdict (1: issues found). Anything else is the
    tool failing, which is reported as NOT run: read as "no output, no
    issues" it said passed. A timeout is the same, set explicitly, because
    the status derived from no diagnostics is "passed".
    """
    diagnostics: list[dict] = []
    status = ""
    try:
        if stdin_argv is not None and prepared.target is not None and prepared.root:
            result = subprocess.run(
                [*stdin_argv, str(prepared.target)], input=prepared.content,
                capture_output=True, text=True, timeout=prepared.timeout,
                cwd=prepared.root,
            )
        else:
            with tempfile.TemporaryDirectory(prefix="neo-check-") as work:
                path = Path(work) / f"suggested{suffix}"
                path.write_text(prepared.content, encoding="utf-8")
                # From the project's directory, so a config file there
                # (pyrightconfig, mypy.ini, pyproject) applies.
                result = subprocess.run(
                    [*argv, str(path)], capture_output=True, text=True,
                    timeout=prepared.timeout, cwd=prepared.root or work,
                )
        if result.returncode == 1 and not result.stdout.strip():
            # "Issues found" with no report at all is not "no issues". (Not
            # "no diagnostics": the environment-only ones are filtered out.)
            result = subprocess.CompletedProcess(result.args, 99, result.stdout, result.stderr)
        diagnostics = parse(result) if result.returncode in (0, 1) else []
        if result.returncode not in (0, 1):
            detail = (result.stderr or result.stdout or "").strip().splitlines()[:1]
            return StaticCheckResult(
                tool_name=tool, diagnostics=[], status="unavailable",
                summary=(f"{tool} failed (exit {result.returncode})"
                         + (f": {detail[0][:200]}" if detail else "")
                         + "; treat this check as NOT run"),
            )
        summary = f"Found {len(diagnostics)} issue(s)" if diagnostics else "No issues found"
    except subprocess.TimeoutExpired:
        summary = (f"{tool} timed out after {prepared.timeout:g}s - "
                   "no diagnostics; treat this check as NOT run")
        status = "unavailable"
    except FileNotFoundError:
        summary = f"{tool} not found - install with: {install_hint}"
    except Exception as exc:  # a checker's own failure is reported, not raised
        summary = f"{tool} check failed: {exc}"
    return StaticCheckResult(tool_name=tool, diagnostics=diagnostics,
                             summary=summary, status=status)


def _ruff_severity(code: Optional[str]) -> str:
    """Error only for what is broken, not for style.

    The file does not parse (`invalid-syntax`, or a finding with no code),
    E9xx, or a name that does not exist (F821-F823). It was "any E": in
    ruff's default set those are style rules (E4/E7: `l = x`, a lambda
    assignment, a bare except), and a `failed` check vetoes promotion for the
    whole episode, while an undefined name was only a warning.
    """
    if not code or code == "invalid-syntax" or code.startswith("E9"):
        return "error"
    if code in ("F821", "F822", "F823"):
        return "error"
    return "warning"


def _parse_ruff(result: subprocess.CompletedProcess) -> list[dict]:
    return [{
        "line": item.get("location", {}).get("row"),
        "column": item.get("location", {}).get("column"),
        "code": item.get("code"),
        "message": item.get("message"),
        "severity": _ruff_severity(item.get("code")),
    } for item in (json.loads(result.stdout) if result.stdout.strip() else [])]


# Diagnostics that exist only because the checker reads the change from a
# private copy outside the project: imports of the project's own modules and
# packages installed elsewhere. They say nothing about the change. (A truly
# misspelled NEW import is hidden with them; that is the price.)
_PYRIGHT_ENVIRONMENT_RULES = frozenset({"reportMissingImports", "reportMissingModuleSource"})
_MYPY_ENVIRONMENT_CODES = ("[import-not-found]", "[import-untyped]",
                           "Cannot find implementation or library stub")


def _parse_pyright(result: subprocess.CompletedProcess) -> list[dict]:
    out = json.loads(result.stdout) if result.stdout.strip() else {}
    return [{
        # pyright counts lines from 0; every other checker from 1.
        "line": (diag.get("range", {}).get("start", {}).get("line") or 0) + 1,
        "column": diag.get("range", {}).get("start", {}).get("character"),
        "rule": diag.get("rule"),
        "message": diag.get("message"),
        "severity": diag.get("severity", "error"),
    } for diag in out.get("generalDiagnostics", [])
        if diag.get("rule") not in _PYRIGHT_ENVIRONMENT_RULES]


_MYPY_LINE = re.compile(r"^.*?:(\d+):(?:(\d+):)? (error|warning|note): (.*)$")


def _parse_mypy(result: subprocess.CompletedProcess) -> list[dict]:
    """`file:line:col: severity: message [code]`. Split on ":" without the
    column, the message landed in `severity` and no mypy finding was ever an
    error."""
    diagnostics = []
    for line in result.stdout.splitlines():
        match = _MYPY_LINE.match(line)
        if not match or any(code in line for code in _MYPY_ENVIRONMENT_CODES):
            continue
        if match.group(3) == "note":
            continue
        diagnostics.append({
            "line": int(match.group(1)),
            "column": int(match.group(2)) if match.group(2) else None,
            "severity": match.group(3),
            "message": match.group(4).strip(),
        })
    return diagnostics


def _parse_eslint(result: subprocess.CompletedProcess) -> list[dict]:
    return [{
        "line": msg.get("line"),
        "column": msg.get("column"),
        "rule": msg.get("ruleId"),
        "message": msg.get("message"),
        "severity": "error" if msg.get("severity") == 2 else "warning",
    } for file_result in (json.loads(result.stdout) if result.stdout.strip() else [])
        for msg in file_result.get("messages", [])]


def run_ruff_check(prepared: PreparedChange) -> StaticCheckResult:
    """ruff, check-only, under the project's own configuration."""
    return _run_tool(
        "ruff", ["ruff", "check", "--output-format=json"], prepared, ".py",
        _parse_ruff, "pip install ruff",
        # --no-force-exclude: with the project's config, a file it excludes
        # (generated code, migrations) was skipped and printed `[]`: passed.
        # The question is whether the change breaks the file, not whether CI
        # lints it.
        stdin_argv=["ruff", "check", "--output-format=json", "--no-force-exclude",
                    "--stdin-filename"],
    )


def run_pyright_check(prepared: PreparedChange) -> StaticCheckResult:
    """pyright on the change's file."""
    return _run_tool("pyright", ["pyright", "--outputjson"], prepared, ".py",
                     _parse_pyright, "npm install -g pyright")


def run_mypy_check(prepared: PreparedChange) -> StaticCheckResult:
    """mypy on the change's file."""
    return _run_tool(
        "mypy",
        ["mypy", "--no-error-summary", "--show-error-codes", "--show-column-numbers"],
        prepared, ".py", _parse_mypy, "pip install mypy",
    )


def run_eslint_check(prepared: PreparedChange) -> StaticCheckResult:
    """eslint, no fixes, on the change's file."""
    # No `--no-eslintrc`: eslint 9 rejects it, and it threw away the
    # project's rules. With no config to find, eslint fails, which reads as
    # not run.
    return _run_tool(
        "eslint", ["eslint", "--format=json"], prepared,
        Path(prepared.file_path).suffix or ".js", _parse_eslint, "npm install -g eslint",
        stdin_argv=["eslint", "--format=json", "--stdin", "--stdin-filename"],
    )


# ============================================================================
# Language Checker Registry
# ============================================================================

@dataclass(frozen=True)
class _LanguageChecker:
    """Pairs a CLI tool with the extensions it analyzes.

    `extensions` is matched against `Path.suffix` (leading dot, lower).
    """
    tool_name: str
    run: Callable[[PreparedChange], StaticCheckResult]
    extensions: frozenset[str]


_LANGUAGE_CHECKERS: tuple[_LanguageChecker, ...] = (
    _LanguageChecker("ruff", run_ruff_check, frozenset({".py"})),
    _LanguageChecker("pyright", run_pyright_check, frozenset({".py"})),
    _LanguageChecker("mypy", run_mypy_check, frozenset({".py"})),
    _LanguageChecker(
        "eslint", run_eslint_check,
        frozenset({".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx"}),
    ),
)

# Drives `detect_available_tools()` — derived from the registry so adding a
# checker doesn't require keeping a second list in sync.
_KNOWN_TOOLS: frozenset[str] = frozenset(c.tool_name for c in _LANGUAGE_CHECKERS)

_TOOL_KINDS = {
    "ruff": "lint",
    "eslint": "lint",
    "pyright": "type_check",
    "mypy": "type_check",
}


# ============================================================================
# Tool Detection
# ============================================================================

def detect_available_tools() -> set[str]:
    """Detect which static analysis tools are on $PATH."""
    import shutil

    return {tool for tool in _KNOWN_TOOLS if shutil.which(tool)}


# ============================================================================
# Main Checker
# ============================================================================

def _diagnostic_key(item: dict, lines: list[str]) -> tuple:
    """What a diagnostic says and the code it is about, without where.

    The same finding moves when the change adds or removes lines above it, so
    the line NUMBER cannot be part of it, nor any number in its message
    ("already defined on line 12"). The flagged line's TEXT is: without it,
    fixing an error in one function and introducing the same error in
    another cancelled out. (Moving an IDENTICAL broken line still cancels:
    the finding is the same one, relocated.)
    """
    line = item.get("line")
    text = lines[line - 1] if isinstance(line, int) and 0 < line <= len(lines) else ""
    # Without its trailing comment: editing only the comment on a line that
    # already had an error read as introducing it. (Approximate: a `#` or
    # `//` inside a string ends the text early, which only widens a match.)
    text = re.split(r"\s(?:#|//)", " " + text, maxsplit=1)[0].strip()
    message = re.sub(r"\d+", "N", str(item.get("message") or ""))
    return (item.get("code") or item.get("rule"), message, text)


def _judge_against_the_original(
    result: StaticCheckResult, prepared: PreparedChange,
    checker: "_LanguageChecker",
) -> None:
    """Keep only the diagnostics the change introduced.

    The checker reads the original too; a finding counts only beyond what it
    already reported there. That catches what a change breaks on a line it
    did not touch (an unclosed bracket, a removed name still used below).
    When the original cannot be checked (timeout, no budget left, the tool
    failing), nothing can be said about which findings are new: the result is
    reported as not run. It used to keep only findings on the changed lines,
    so a syntax error reported past the change was dropped, called "older",
    and the check read as passed.
    """
    if prepared.original is None or not result.diagnostics or result.status:
        return
    from collections import Counter

    baseline = checker.run(PreparedChange(
        prepared.file_path, prepared.original, prepared.timeout,
        target=prepared.target, root=prepared.root,
    )) if prepared.timeout > 0 else None
    if baseline is None or baseline.status or "not found" in baseline.summary.lower() \
            or "failed:" in baseline.summary.lower():
        result.status = "unavailable"
        result.summary = (
            f"Found {len(result.diagnostics)} issue(s), but the file before the "
            "change could not be checked to tell which the change introduced; "
            "treat this check as NOT run"
        )
        return
    before_lines = prepared.original.splitlines()
    after_lines = prepared.content.splitlines()
    already = Counter(_diagnostic_key(item, before_lines) for item in baseline.diagnostics)
    kept = []
    for item in result.diagnostics:
        key = _diagnostic_key(item, after_lines)
        if already[key]:
            already[key] -= 1
        else:
            kept.append(item)
    older = len(result.diagnostics) - len(kept)
    result.diagnostics = kept
    if older:
        result.summary = (
            (f"Found {len(kept)} issue(s) in the change" if kept
             else "No issues found in the change")
            + f"; {older} issue(s) the file already had not counted"
        )


def run_static_checks(
    suggestions: list[CodeSuggestion],
    enable_ruff: bool = True,
    enable_pyright: bool = True,
    enable_mypy: bool = False,
    enable_eslint: bool = True,
    codebase_root: Optional[str] = None,
    phase_budget: float = STATIC_CHECK_PHASE_BUDGET_SECONDS,
) -> list[StaticCheckResult]:
    """
    Run static analysis tools on code suggestions.

    Args:
        suggestions: List of code suggestions to check
        enable_ruff: Run ruff on Python files
        enable_pyright: Run pyright on Python files
        enable_mypy: Run mypy on Python files
        enable_eslint: Run eslint on JS/TS files
        codebase_root: Where relative suggestion paths live
        phase_budget: Seconds the whole phase may take; each tool gets at
            most what is left of it

    Returns:
        List of static check results

    Each change is prepared ONCE (`prepare_change`) and every checker reads
    that copy. Only diagnostics the change introduced count
    (`_judge_against_the_original`). Statuses:

    - `skipped`: the CHANGE could not be checked (it did not apply, no
      checker exists for its language, the phase budget ran out). Every
      suggestion yields at least one result, so one clean check can never
      vouch for a change nothing looked at.
    - `unavailable`: a TOOL could not give a verdict (not installed, timed
      out, failed).

    Note: enabling both pyright AND mypy runs both. They flag different
    things, so co-enabling is rarely redundant — the previous fallback
    semantics (only-mypy-if-pyright-disabled) coupled the two without a
    real reason.
    """
    enable_map = {
        "ruff": enable_ruff,
        "pyright": enable_pyright,
        "mypy": enable_mypy,
        "eslint": enable_eslint,
    }
    available_tools = detect_available_tools()
    results: list[StaticCheckResult] = []
    deadline = time.monotonic() + phase_budget

    def not_run(tool: str, summary: str, status: str) -> StaticCheckResult:
        return StaticCheckResult(tool_name=tool, diagnostics=[], summary=summary,
                                 kind=_TOOL_KINDS.get(tool, "lint"), status=status,
                                 file_path=suggestion.file_path)

    def budget_spent(tool: str) -> StaticCheckResult:
        return not_run(tool, f"{tool} not run - the static-check phase budget of "
                             f"{phase_budget:g}s expired", "skipped")

    for suggestion in suggestions:
        suffix = Path(suggestion.file_path).suffix.lower()
        claimed = [c for c in _LANGUAGE_CHECKERS if suffix in c.extensions]
        checkers = [c for c in claimed if enable_map.get(c.tool_name, False)]
        if not checkers:
            results.append(not_run(
                "static-checks",
                f"{suggestion.file_path} not checked - "
                + ("every checker for it is disabled" if claimed
                   else f"no checker for {suffix or 'this file type'}"),
                "skipped",
            ))
            continue
        prepared: Optional[PreparedChange] = None
        why = ""
        for checker in checkers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Reported, not dropped. A check that never started is not a
                # check that passed.
                results.append(budget_spent(checker.tool_name))
                continue
            if checker.tool_name not in available_tools:
                results.append(not_run(checker.tool_name,
                                       f"{checker.tool_name} is unavailable", "unavailable"))
                continue
            if prepared is None and not why:
                prepared, why = prepare_change(
                    suggestion, codebase_root,
                    timeout=min(STATIC_CHECK_TIMEOUT_SECONDS, remaining),
                )
            if prepared is None:
                results.append(not_run(
                    checker.tool_name, f"{checker.tool_name} not run - {why}", "skipped",
                ))
                continue
            prepared.timeout = min(STATIC_CHECK_TIMEOUT_SECONDS, deadline - time.monotonic())
            if prepared.timeout <= 0:
                results.append(budget_spent(checker.tool_name))
                continue
            result = checker.run(prepared)
            result.file_path = suggestion.file_path
            if not result.kind:
                result.kind = _TOOL_KINDS[checker.tool_name]
            prepared.timeout = min(STATIC_CHECK_TIMEOUT_SECONDS, deadline - time.monotonic())
            _judge_against_the_original(result, prepared, checker)
            if not result.status:
                severities = {
                    str(item.get("severity", "")).lower()
                    for item in result.diagnostics
                }
                if "error" in severities:
                    result.status = "failed"
                elif result.diagnostics:
                    result.status = "warning"
                elif "not found" in result.summary.lower() or "failed:" in result.summary.lower():
                    result.status = "unavailable"
                else:
                    result.status = "passed"
            results.append(result)

    return results
