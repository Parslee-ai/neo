"""Tests for neo.static_analysis — language dispatch registry."""

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
