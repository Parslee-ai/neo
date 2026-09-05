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
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


def run_ruff_check(suggestion: CodeSuggestion) -> StaticCheckResult:
    """
    Run ruff in check-only mode on the suggested code.

    Creates a temporary file with the suggested changes and runs ruff on it.
    """
    diagnostics = []
    # "" means "let run_static_checks derive it from the diagnostics". Only the
    # timeout path sets it explicitly; see that handler for why.
    status = ""

    try:
        # Apply diff to get the new content
        new_content = apply_diff_to_content(
            suggestion.unified_diff,
            get_original_content(suggestion.file_path),
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
        ) as f:
            f.write(new_content)
            temp_path = f.name

        # Run ruff check
        result = subprocess.run(
            ['ruff', 'check', '--output-format=json', temp_path],
            capture_output=True,
            text=True,
            timeout=STATIC_CHECK_TIMEOUT_SECONDS,
        )

        # Parse JSON output
        if result.stdout:
            ruff_output = json.loads(result.stdout)
            for item in ruff_output:
                diagnostics.append({
                    'line': item.get('location', {}).get('row'),
                    'column': item.get('location', {}).get('column'),
                    'code': item.get('code'),
                    'message': item.get('message'),
                    'severity': 'error' if item.get('code', '').startswith('E') else 'warning',
                })

        # Clean up
        Path(temp_path).unlink()

        summary = f"Found {len(diagnostics)} issue(s)" if diagnostics else "No issues found"

    except subprocess.TimeoutExpired:
        summary = (
            f"ruff timed out after {STATIC_CHECK_TIMEOUT_SECONDS}s - "
            "no diagnostics; treat this check as NOT run"
        )
        # Set EXPLICITLY. run_static_checks derives an empty status from the
        # diagnostics, and a timeout produces none — no error severities, no
        # "not found", no "failed:" — so the derivation lands on "passed" and
        # a wedged checker reads as CLEAN. That is what licenses the early
        # exit and suppresses the unverified caution. The prose said "treat
        # this check as NOT run" while the field the engine actually reads
        # said the opposite.
        status = "unavailable"
    except FileNotFoundError:
        summary = "ruff not found - install with: pip install ruff"
    except Exception as e:
        summary = f"ruff check failed: {e}"

    return StaticCheckResult(
        tool_name="ruff",
        diagnostics=diagnostics,
        summary=summary,
        status=status,
    )


# ============================================================================
# Python - Pyright
# ============================================================================

def run_pyright_check(suggestion: CodeSuggestion) -> StaticCheckResult:
    """
    Run pyright in check-only mode on the suggested code.
    """
    diagnostics = []
    # "" means "let run_static_checks derive it from the diagnostics". Only the
    # timeout path sets it explicitly; see that handler for why.
    status = ""

    try:
        # Apply diff to get the new content
        new_content = apply_diff_to_content(
            suggestion.unified_diff,
            get_original_content(suggestion.file_path),
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
        ) as f:
            f.write(new_content)
            temp_path = f.name

        # Run pyright with JSON output
        result = subprocess.run(
            ['pyright', '--outputjson', temp_path],
            capture_output=True,
            text=True,
            timeout=STATIC_CHECK_TIMEOUT_SECONDS,
        )

        # Parse JSON output
        if result.stdout:
            pyright_output = json.loads(result.stdout)
            for diag in pyright_output.get('generalDiagnostics', []):
                diagnostics.append({
                    'line': diag.get('range', {}).get('start', {}).get('line'),
                    'column': diag.get('range', {}).get('start', {}).get('character'),
                    'message': diag.get('message'),
                    'severity': diag.get('severity', 'error'),
                })

        # Clean up
        Path(temp_path).unlink()

        summary = f"Found {len(diagnostics)} issue(s)" if diagnostics else "No issues found"

    except subprocess.TimeoutExpired:
        summary = (
            f"pyright timed out after {STATIC_CHECK_TIMEOUT_SECONDS}s - "
            "no diagnostics; treat this check as NOT run"
        )
        # Set EXPLICITLY. run_static_checks derives an empty status from the
        # diagnostics, and a timeout produces none — no error severities, no
        # "not found", no "failed:" — so the derivation lands on "passed" and
        # a wedged checker reads as CLEAN. That is what licenses the early
        # exit and suppresses the unverified caution. The prose said "treat
        # this check as NOT run" while the field the engine actually reads
        # said the opposite.
        status = "unavailable"
    except FileNotFoundError:
        summary = "pyright not found - install with: npm install -g pyright"
    except Exception as e:
        summary = f"pyright check failed: {e}"

    return StaticCheckResult(
        tool_name="pyright",
        diagnostics=diagnostics,
        summary=summary,
        status=status,
    )


# ============================================================================
# Python - MyPy
# ============================================================================

def run_mypy_check(suggestion: CodeSuggestion) -> StaticCheckResult:
    """
    Run mypy in check-only mode on the suggested code.
    """
    diagnostics = []
    # "" means "let run_static_checks derive it from the diagnostics". Only the
    # timeout path sets it explicitly; see that handler for why.
    status = ""

    try:
        # Apply diff to get the new content
        new_content = apply_diff_to_content(
            suggestion.unified_diff,
            get_original_content(suggestion.file_path),
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
        ) as f:
            f.write(new_content)
            temp_path = f.name

        # Run mypy
        result = subprocess.run(
            ['mypy', '--no-error-summary', temp_path],
            capture_output=True,
            text=True,
            timeout=STATIC_CHECK_TIMEOUT_SECONDS,
        )

        # Parse output (line format: file:line:col: severity: message)
        for line in result.stdout.splitlines():
            match = line.split(':', 4)
            if len(match) >= 4:
                diagnostics.append({
                    'line': int(match[1]) if match[1].isdigit() else None,
                    'column': int(match[2]) if match[2].isdigit() else None,
                    'severity': match[3].strip().lower(),
                    'message': match[4].strip() if len(match) > 4 else '',
                })

        # Clean up
        Path(temp_path).unlink()

        summary = f"Found {len(diagnostics)} issue(s)" if diagnostics else "No issues found"

    except subprocess.TimeoutExpired:
        summary = (
            f"mypy timed out after {STATIC_CHECK_TIMEOUT_SECONDS}s - "
            "no diagnostics; treat this check as NOT run"
        )
        # Set EXPLICITLY. run_static_checks derives an empty status from the
        # diagnostics, and a timeout produces none — no error severities, no
        # "not found", no "failed:" — so the derivation lands on "passed" and
        # a wedged checker reads as CLEAN. That is what licenses the early
        # exit and suppresses the unverified caution. The prose said "treat
        # this check as NOT run" while the field the engine actually reads
        # said the opposite.
        status = "unavailable"
    except FileNotFoundError:
        summary = "mypy not found - install with: pip install mypy"
    except Exception as e:
        summary = f"mypy check failed: {e}"

    return StaticCheckResult(
        tool_name="mypy",
        diagnostics=diagnostics,
        summary=summary,
        status=status,
    )


# ============================================================================
# JavaScript/TypeScript - ESLint
# ============================================================================

def run_eslint_check(suggestion: CodeSuggestion) -> StaticCheckResult:
    """
    Run eslint in no-fix mode on the suggested code.
    """
    diagnostics = []
    # "" means "let run_static_checks derive it from the diagnostics". Only the
    # timeout path sets it explicitly; see that handler for why.
    status = ""

    try:
        # Apply diff to get the new content
        new_content = apply_diff_to_content(
            suggestion.unified_diff,
            get_original_content(suggestion.file_path),
        )

        # Determine file extension
        suffix = Path(suggestion.file_path).suffix or '.js'

        # Write to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=suffix,
            delete=False,
        ) as f:
            f.write(new_content)
            temp_path = f.name

        # Run eslint with JSON output
        result = subprocess.run(
            ['eslint', '--format=json', '--no-eslintrc', temp_path],
            capture_output=True,
            text=True,
            timeout=STATIC_CHECK_TIMEOUT_SECONDS,
        )

        # Parse JSON output
        if result.stdout:
            eslint_output = json.loads(result.stdout)
            for file_result in eslint_output:
                for msg in file_result.get('messages', []):
                    diagnostics.append({
                        'line': msg.get('line'),
                        'column': msg.get('column'),
                        'rule': msg.get('ruleId'),
                        'message': msg.get('message'),
                        'severity': 'error' if msg.get('severity') == 2 else 'warning',
                    })

        # Clean up
        Path(temp_path).unlink()

        summary = f"Found {len(diagnostics)} issue(s)" if diagnostics else "No issues found"

    except subprocess.TimeoutExpired:
        summary = (
            f"eslint timed out after {STATIC_CHECK_TIMEOUT_SECONDS}s - "
            "no diagnostics; treat this check as NOT run"
        )
        # Set EXPLICITLY. run_static_checks derives an empty status from the
        # diagnostics, and a timeout produces none — no error severities, no
        # "not found", no "failed:" — so the derivation lands on "passed" and
        # a wedged checker reads as CLEAN. That is what licenses the early
        # exit and suppresses the unverified caution. The prose said "treat
        # this check as NOT run" while the field the engine actually reads
        # said the opposite.
        status = "unavailable"
    except FileNotFoundError:
        summary = "eslint not found - install with: npm install -g eslint"
    except Exception as e:
        summary = f"eslint check failed: {e}"

    return StaticCheckResult(
        tool_name="eslint",
        diagnostics=diagnostics,
        summary=summary,
        status=status,
    )


# ============================================================================
# Helper Functions
# ============================================================================

def get_original_content(file_path: str) -> str:
    """Get original file content if it exists."""
    try:
        return Path(file_path).read_text()
    except (FileNotFoundError, IsADirectoryError, OSError):  # File doesn't exist or can't be read
        return ""


def apply_diff_to_content(unified_diff: str, original_content: str) -> str:
    """
    Apply unified diff to original content to get new content.

    For simplicity, if this is a new file or we can't parse the diff,
    just return the diff content with +/- markers stripped.
    """
    if not unified_diff:
        return original_content

    # Simple heuristic: if original is empty, extract all + lines
    if not original_content:
        lines = []
        for line in unified_diff.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                lines.append(line[1:])
        return '\n'.join(lines)

    # For actual diff application, use patch (would need patch command)
    # For now, try simple extraction of new content
    orig_path = diff_path = None
    try:
        import tempfile
        import subprocess

        # Write original to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(original_content)
            orig_path = f.name

        # Write diff to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write(unified_diff)
            diff_path = f.name

        # Apply patch.
        #
        # --batch and stdin=DEVNULL, because `patch` PROMPTS on a malformed
        # diff ("File to patch:", "Assume -R? [n]") and otherwise inherits
        # neo's own stdin — the stdin a host may be feeding the prompt on. The
        # timeout bounds the hang; only these two stop it eating input that
        # was not meant for it.
        subprocess.run(
            ['patch', '--batch', orig_path, diff_path],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=STATIC_CHECK_TIMEOUT_SECONDS,
        )

        # Read patched content
        patched_content = Path(orig_path).read_text()

        return patched_content
    # TimeoutExpired is a SubprocessError, NOT an OSError and not a
    # CalledProcessError, so without naming it here it escaped this function
    # entirely — straight past the +-lines fallback below, into the CALLING
    # checker's own `except subprocess.TimeoutExpired`, where it was reported
    # as "ruff timed out" on a run where ruff had never been invoked.
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError,
            FileNotFoundError, OSError):  # patch command failed or file issues
        # Fallback: reconstruct from the diff's + and context lines.
        #
        # The header guard used to be one-sided: "+++ b/file" was excluded from
        # the ADDED branch but the context branch does not start with "-",
        # "@@" or "---", so it fell through and was appended VERBATIM. The
        # reconstructed file then carried a literal "+++ b/file" line, which
        # the checkers dutifully linted and reported as a real diagnostic
        # against the user's code.
        lines = []
        for line in unified_diff.splitlines():
            if line.startswith('+++') or line.startswith('---') or line.startswith('@@'):
                continue
            if line.startswith('+'):
                lines.append(line[1:])
            elif not line.startswith('-'):
                # Context lines carry a single leading space in unified diff
                # format. Appending them verbatim shifted EVERY context line
                # one column right, which in Python is not cosmetic: the
                # reconstructed file is mis-indented and the checkers report
                # indentation errors against code the user never wrote.
                lines.append(line[1:] if line.startswith(' ') else line)
        return '\n'.join(lines) if lines else original_content
    finally:
        # In a finally: the old cleanup sat after the `return`, so every path
        # that raised — which is now every path that times out — leaked both
        # temp files.
        for path in (orig_path, diff_path):
            if path:
                try:
                    Path(path).unlink()
                except OSError:
                    pass


# ============================================================================
# Language Checker Registry
# ============================================================================

@dataclass(frozen=True)
class _LanguageChecker:
    """Pairs a CLI tool with the extensions it analyzes.

    `extensions` is matched against `Path.suffix` (leading dot, lower).
    """
    tool_name: str
    run: Callable[[CodeSuggestion], StaticCheckResult]
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

def run_static_checks(
    suggestions: list[CodeSuggestion],
    enable_ruff: bool = True,
    enable_pyright: bool = True,
    enable_mypy: bool = False,
    enable_eslint: bool = True,
) -> list[StaticCheckResult]:
    """
    Run static analysis tools on code suggestions.

    Args:
        suggestions: List of code suggestions to check
        enable_ruff: Run ruff on Python files
        enable_pyright: Run pyright on Python files
        enable_mypy: Run mypy on Python files
        enable_eslint: Run eslint on JS/TS files

    Returns:
        List of static check results

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
    deadline = time.monotonic() + STATIC_CHECK_PHASE_BUDGET_SECONDS

    for suggestion in suggestions:
        suffix = Path(suggestion.file_path).suffix.lower()
        for checker in _LANGUAGE_CHECKERS:
            if suffix not in checker.extensions:
                continue
            if not enable_map.get(checker.tool_name, False):
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Reported, not dropped. A check that never started is not a
                # check that passed, and the phase has to be able to say which
                # of its work it did not reach.
                results.append(StaticCheckResult(
                    tool_name=checker.tool_name,
                    diagnostics=[],
                    summary=(
                        f"{checker.tool_name} not run - the static-check phase "
                        f"budget of {STATIC_CHECK_PHASE_BUDGET_SECONDS}s expired"
                    ),
                    kind=_TOOL_KINDS[checker.tool_name],
                    status="skipped",
                ))
                continue
            if checker.tool_name not in available_tools:
                results.append(StaticCheckResult(
                    tool_name=checker.tool_name,
                    diagnostics=[],
                    summary=f"{checker.tool_name} is unavailable",
                    kind=_TOOL_KINDS[checker.tool_name],
                    status="unavailable",
                ))
                continue
            result = checker.run(suggestion)
            if not result.kind:
                result.kind = _TOOL_KINDS[checker.tool_name]
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
