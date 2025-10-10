"""
Static analysis tool integrations for Neo.

All tools run in read-only/check-only mode.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from neo import CodeSuggestion, StaticCheckResult


# ============================================================================
# Python - Ruff
# ============================================================================

def run_ruff_check(suggestion: CodeSuggestion) -> StaticCheckResult:
    """
    Run ruff in check-only mode on the suggested code.

    Creates a temporary file with the suggested changes and runs ruff on it.
    """
    diagnostics = []

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

    except FileNotFoundError:
        summary = "ruff not found - install with: pip install ruff"
    except Exception as e:
        summary = f"ruff check failed: {e}"

    return StaticCheckResult(
        tool_name="ruff",
        diagnostics=diagnostics,
        summary=summary,
    )


# ============================================================================
# Python - Pyright
# ============================================================================

def run_pyright_check(suggestion: CodeSuggestion) -> StaticCheckResult:
    """
    Run pyright in check-only mode on the suggested code.
    """
    diagnostics = []

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

    except FileNotFoundError:
        summary = "pyright not found - install with: npm install -g pyright"
    except Exception as e:
        summary = f"pyright check failed: {e}"

    return StaticCheckResult(
        tool_name="pyright",
        diagnostics=diagnostics,
        summary=summary,
    )


# ============================================================================
# Python - MyPy
# ============================================================================

def run_mypy_check(suggestion: CodeSuggestion) -> StaticCheckResult:
    """
    Run mypy in check-only mode on the suggested code.
    """
    diagnostics = []

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

    except FileNotFoundError:
        summary = "mypy not found - install with: pip install mypy"
    except Exception as e:
        summary = f"mypy check failed: {e}"

    return StaticCheckResult(
        tool_name="mypy",
        diagnostics=diagnostics,
        summary=summary,
    )


# ============================================================================
# JavaScript/TypeScript - ESLint
# ============================================================================

def run_eslint_check(suggestion: CodeSuggestion) -> StaticCheckResult:
    """
    Run eslint in no-fix mode on the suggested code.
    """
    diagnostics = []

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

    except FileNotFoundError:
        summary = "eslint not found - install with: npm install -g eslint"
    except Exception as e:
        summary = f"eslint check failed: {e}"

    return StaticCheckResult(
        tool_name="eslint",
        diagnostics=diagnostics,
        summary=summary,
    )


# ============================================================================
# Helper Functions
# ============================================================================

def get_original_content(file_path: str) -> str:
    """Get original file content if it exists."""
    try:
        return Path(file_path).read_text()
    except:
        # File doesn't exist yet (new file)
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

        # Apply patch
        result = subprocess.run(
            ['patch', orig_path, diff_path],
            capture_output=True,
        )

        # Read patched content
        patched_content = Path(orig_path).read_text()

        # Clean up
        Path(orig_path).unlink()
        Path(diff_path).unlink()

        return patched_content
    except:
        # Fallback: extract lines starting with +
        lines = []
        for line in unified_diff.splitlines():
            if line.startswith('+') and not line.startswith('+++'):
                lines.append(line[1:])
            elif not line.startswith('-') and not line.startswith('@@') and not line.startswith('---'):
                # Keep context lines
                lines.append(line)
        return '\n'.join(lines) if lines else original_content


# ============================================================================
# Tool Detection
# ============================================================================

def detect_available_tools() -> set[str]:
    """Detect which static analysis tools are available."""
    import shutil

    tools = set()
    for tool in ["ruff", "pyright", "mypy", "eslint"]:
        if shutil.which(tool):
            tools.add(tool)
    return tools


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
    """
    results = []
    available_tools = detect_available_tools()

    for suggestion in suggestions:
        file_path = Path(suggestion.file_path)

        # Python checks
        if file_path.suffix == ".py":
            if enable_ruff and "ruff" in available_tools:
                results.append(run_ruff_check(suggestion))
            if enable_pyright and "pyright" in available_tools:
                results.append(run_pyright_check(suggestion))
            elif enable_mypy and "mypy" in available_tools:
                results.append(run_mypy_check(suggestion))

        # JavaScript/TypeScript checks
        elif file_path.suffix in {".js", ".ts", ".jsx", ".tsx"}:
            if enable_eslint and "eslint" in available_tools:
                results.append(run_eslint_check(suggestion))

    return results