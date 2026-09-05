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
        # STATUS is what the engine reads — _static_check_status returns it
        # verbatim and _checks_that_evaluated gates the "unverified" caution
        # and the early exit on it. Asserting only on the prose let a version
        # ship where the summary said "treat this check as NOT run" and the
        # status said "passed".
        assert result.status == "unavailable", (
            f"a wedged checker reports {result.status!r} — reads as clean"
        )

    @pytest.mark.parametrize("checker_name", ["ruff", "pyright", "mypy", "eslint"])
    def test_timeout_does_not_derive_passed_end_to_end(self, checker_name, monkeypatch):
        """Through run_static_checks, which is where the status is derived.

        A timeout yields no diagnostics, no "not found" and no "failed:", so
        the derivation at the end of run_static_checks landed on "passed".
        """
        import subprocess as sp
        from neo import static_analysis as sa

        def fake_run(cmd, **kwargs):
            raise sp.TimeoutExpired(cmd, sa.STATIC_CHECK_TIMEOUT_SECONDS)

        monkeypatch.setattr(sa.subprocess, "run", fake_run)
        ext = "ts" if checker_name == "eslint" else "py"
        sug = CodeSuggestion(
            file_path=f"x.{ext}",
            unified_diff="--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b\n",
            code_block="", description="", confidence=0.0,
        )
        results = sa.run_static_checks(
            [sug],
            enable_ruff=checker_name == "ruff",
            enable_pyright=checker_name == "pyright",
            enable_mypy=checker_name == "mypy",
            enable_eslint=checker_name == "eslint",
        )
        for r in results:
            assert r.status != "passed", (
                f"{r.tool_name} timed out and was recorded as passed"
            )


class TestDiffApplicationIsBounded:
    """apply_diff_to_content shells out to `patch` and sits under every
    checker."""

    def test_timeout_is_handled_not_escaped(self, monkeypatch):
        """TimeoutExpired is a SubprocessError, NOT an OSError, so it escaped
        this function's except tuple entirely — past the fallback, into the
        calling checker's handler, where it was reported as that checker
        timing out on a run where the checker never ran."""
        import subprocess as sp
        from neo import static_analysis as sa

        monkeypatch.setattr(
            sa.subprocess, "run",
            lambda cmd, **kw: (_ for _ in ()).throw(sp.TimeoutExpired(cmd, 30)),
        )
        out = sa.apply_diff_to_content(
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n", "old\n"
        )
        assert out == "new"

    def test_temp_files_are_not_leaked_on_failure(self, monkeypatch):
        """Cleanup sat after the `return`, so every raising path leaked both
        temp files — which is now every path that times out."""
        import glob
        import os
        import subprocess as sp
        import tempfile
        from neo import static_analysis as sa

        pattern = os.path.join(tempfile.gettempdir(), "tmp*")
        before = set(glob.glob(pattern))
        monkeypatch.setattr(
            sa.subprocess, "run",
            lambda cmd, **kw: (_ for _ in ()).throw(sp.TimeoutExpired(cmd, 30)),
        )
        sa.apply_diff_to_content(
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n", "old\n"
        )
        assert not (set(glob.glob(pattern)) - before)

    def test_patch_cannot_consume_neos_stdin(self, monkeypatch):
        """`patch` prompts on a malformed diff and otherwise inherits neo's
        own stdin — the stdin a host may be feeding the prompt on."""
        import subprocess as sp
        from neo import static_analysis as sa

        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            seen["stdin"] = kwargs.get("stdin")
            raise sp.TimeoutExpired(cmd, 30)

        monkeypatch.setattr(sa.subprocess, "run", fake_run)
        sa.apply_diff_to_content("--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b\n", "a\n")

        assert seen["stdin"] == sp.DEVNULL
        assert "--batch" in seen["cmd"]

    def test_fallback_does_not_leak_diff_headers_into_content(self, monkeypatch):
        """The header guard was one-sided: "+++ b/file" was excluded from the
        added-lines branch but fell through the context branch and was
        appended verbatim, so the checkers linted a literal "+++ b/file" line
        and reported it against the user's code."""
        import subprocess as sp
        from neo import static_analysis as sa

        monkeypatch.setattr(
            sa.subprocess, "run",
            lambda cmd, **kw: (_ for _ in ()).throw(sp.TimeoutExpired(cmd, 30)),
        )
        out = sa.apply_diff_to_content(
            "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n ctx\n-old\n+new\n", "ctx\nold\n"
        )
        assert "+++" not in out and "---" not in out and "@@" not in out
        assert out.splitlines() == ["ctx", "new"]

    def test_fallback_preserves_context_indentation(self, monkeypatch):
        """Context lines carry a leading space in unified diff format.
        Appending them verbatim shifted every one a column right — in Python
        that is a syntax change, and the checkers reported indentation errors
        against code the user never wrote."""
        import subprocess as sp
        from neo import static_analysis as sa

        monkeypatch.setattr(
            sa.subprocess, "run",
            lambda cmd, **kw: (_ for _ in ()).throw(sp.TimeoutExpired(cmd, 30)),
        )
        diff = (
            "--- a/x.py\n+++ b/x.py\n@@ -1,3 +1,3 @@\n"
            " def f():\n"
            "-    return 1\n"
            "+    return 2\n"
        )
        out = sa.apply_diff_to_content(diff, "def f():\n    return 1\n")

        assert out.splitlines() == ["def f():", "    return 2"], out.splitlines()
        # The reconstructed file must actually parse; that is the whole point
        # of handing it to a checker.
        import ast
        ast.parse(out)
