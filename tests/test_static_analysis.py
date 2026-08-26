"""Tests for neo.static_analysis — language dispatch registry."""

import pytest

from neo.models import CodeSuggestion
from neo.static_analysis import (
    _KNOWN_TOOLS,
    _LANGUAGE_CHECKERS,
    run_static_checks,
)


def _suggestion(file_path: str) -> CodeSuggestion:
    """Build a minimal CodeSuggestion. Contents don't matter for dispatch."""
    return CodeSuggestion(
        file_path=file_path,
        unified_diff="",
        code_block="",
        description="",
        confidence=0.0,
    )


class TestRegistryShape:
    def test_known_tools_derived_from_registry(self):
        # _KNOWN_TOOLS must match what the registry actually uses; drift
        # here means detect_available_tools() can't see new checkers.
        assert _KNOWN_TOOLS == frozenset(c.tool_name for c in _LANGUAGE_CHECKERS)

    def test_every_checker_has_extensions(self):
        for c in _LANGUAGE_CHECKERS:
            assert c.extensions, f"{c.tool_name} has no extensions"
            for ext in c.extensions:
                assert ext.startswith("."), f"{c.tool_name} ext {ext!r} missing leading dot"
                assert ext == ext.lower(), f"{c.tool_name} ext {ext!r} not lowercase"


def _stub_registry(monkeypatch, run_overrides: dict):
    """Rebuild _LANGUAGE_CHECKERS with caller-supplied run functions.

    Explicit dict (rather than `getattr` fallback) so a typo or rename
    fails loudly instead of silently routing to the real subprocess.
    """
    from neo import static_analysis as sa
    new_registry = tuple(
        sa._LanguageChecker(c.tool_name, run_overrides[c.tool_name], c.extensions)
        for c in sa._LANGUAGE_CHECKERS
    )
    monkeypatch.setattr("neo.static_analysis._LANGUAGE_CHECKERS", new_registry)


class TestDispatch:
    def test_missing_enabled_tool_is_explicitly_unavailable(self, monkeypatch):
        monkeypatch.setattr("neo.static_analysis.detect_available_tools", set)

        results = run_static_checks(
            [_suggestion("foo.py")],
            enable_ruff=True,
            enable_pyright=False,
            enable_mypy=False,
            enable_eslint=False,
        )

        assert len(results) == 1
        assert results[0].tool_name == "ruff"
        assert results[0].kind == "lint"
        assert results[0].status == "unavailable"

    def test_unsupported_extension_skipped(self, monkeypatch):
        # Files with extensions no checker claims (e.g. .rs today) produce
        # no results, even with every checker enabled and a stub installed.
        from neo.models import StaticCheckResult

        def _boom(_s):
            raise AssertionError("checker should not be invoked for .rs")

        _stub_registry(monkeypatch, {name: _boom for name in _KNOWN_TOOLS})
        monkeypatch.setattr(
            "neo.static_analysis.detect_available_tools", lambda: set(_KNOWN_TOOLS)
        )
        assert run_static_checks(
            [_suggestion("foo.rs")],
            enable_ruff=True,
            enable_pyright=True,
            enable_mypy=True,
            enable_eslint=True,
        ) == []
        _ = StaticCheckResult  # imported only to keep the symbol referenced

    def test_pyright_and_mypy_both_run_when_both_enabled(self, monkeypatch):
        # Pre-refactor, mypy was skipped whenever pyright was enabled. The
        # new dispatcher runs both — they catch different things.
        from neo.models import StaticCheckResult

        calls: list[str] = []

        def fake(tool: str):
            def _run(_s):
                calls.append(tool)
                return StaticCheckResult(tool_name=tool, diagnostics=[], summary="ok")
            return _run

        _stub_registry(monkeypatch, {name: fake(name) for name in _KNOWN_TOOLS})
        monkeypatch.setattr(
            "neo.static_analysis.detect_available_tools",
            lambda: {"ruff", "pyright", "mypy"},
        )

        run_static_checks(
            [_suggestion("foo.py")],
            enable_ruff=True,
            enable_pyright=True,
            enable_mypy=True,
        )
        assert set(calls) == {"ruff", "pyright", "mypy"}

    def test_eslint_routes_all_js_ts_variants(self, monkeypatch):
        from neo.models import StaticCheckResult

        called_with: list[str] = []

        def fake_eslint(s):
            called_with.append(s.file_path)
            return StaticCheckResult(tool_name="eslint", diagnostics=[], summary="ok")

        def _unused(_s):
            raise AssertionError("non-eslint checker invoked on .js/.ts file")

        _stub_registry(
            monkeypatch,
            {name: (fake_eslint if name == "eslint" else _unused) for name in _KNOWN_TOOLS},
        )
        monkeypatch.setattr(
            "neo.static_analysis.detect_available_tools", lambda: {"eslint"}
        )

        run_static_checks(
            [_suggestion(p) for p in ["a.js", "b.jsx", "c.ts", "d.tsx", "e.mjs", "f.cjs"]],
            enable_ruff=False,
            enable_pyright=False,
            enable_eslint=True,
        )
        assert len(called_with) == 6


class TestCheckersAreBounded:
    """Every checker shells out to an external tool, and none of these calls
    was bounded. A wedged pyright or an eslint waiting on a lockfile could hang
    a neo run indefinitely — in the one phase whose whole job is to make the
    output trustworthy. Being bounded is also what makes it safe to run these
    unconditionally instead of dropping them when inference runs long.
    """

    @pytest.mark.parametrize("checker_name", ["ruff", "pyright", "mypy", "eslint"])
    def test_checker_passes_a_timeout(self, checker_name, monkeypatch):
        from neo import static_analysis as sa

        seen = {}

        def fake_run(cmd, **kwargs):
            seen["timeout"] = kwargs.get("timeout")
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(sa.subprocess, "run", fake_run)
        checker = getattr(sa, f"run_{checker_name}_check")
        checker(_suggestion(f"x.{'ts' if checker_name == 'eslint' else 'py'}"))

        assert seen.get("timeout") == sa.STATIC_CHECK_TIMEOUT_SECONDS, (
            f"{checker_name} ran unbounded"
        )

    @pytest.mark.parametrize("checker_name", ["ruff", "pyright", "mypy", "eslint"])
    def test_timeout_is_reported_as_not_run(self, checker_name, monkeypatch):
        """A timed-out checker contributed no diagnostics. That must not read
        as "checked and clean" — it is "not checked"."""
        import subprocess as sp
        from neo import static_analysis as sa

        def fake_run(cmd, **kwargs):
            raise sp.TimeoutExpired(cmd, sa.STATIC_CHECK_TIMEOUT_SECONDS)

        monkeypatch.setattr(sa.subprocess, "run", fake_run)
        checker = getattr(sa, f"run_{checker_name}_check")
        result = checker(_suggestion(f"x.{'ts' if checker_name == 'eslint' else 'py'}"))

        assert "timed out" in result.summary.lower()
        assert "not run" in result.summary.lower()
        assert not result.diagnostics
