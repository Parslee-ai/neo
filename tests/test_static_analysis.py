"""Tests for neo.static_analysis — language dispatch registry."""

import pytest

from neo.models import CodeSuggestion
from neo.static_analysis import (
    _KNOWN_TOOLS,
    _LANGUAGE_CHECKERS,
    run_static_checks,
)


def _suggestion(file_path: str) -> CodeSuggestion:
    """A suggestion creating ``file_path``: prepares without `patch` or disk,
    so dispatch tests reach the (stubbed) checkers."""
    return CodeSuggestion(
        file_path=file_path,
        unified_diff=f"--- /dev/null\n+++ b/{file_path}\n@@ -0,0 +1 @@\n+x = 1\n",
        code_block="",
        description="",
        confidence=0.0,
    )


def _prepared(file_path: str):
    from neo.static_analysis import PreparedChange

    return PreparedChange(file_path=file_path, content="x = 1\n")


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
        results = run_static_checks(
            [_suggestion("foo.rs")],
            enable_ruff=True,
            enable_pyright=True,
            enable_mypy=True,
            enable_eslint=True,
        )
        # Reported, not dropped: a clean .py check beside an unchecked .rs
        # change read as the whole answer verified.
        assert [(r.status, "no checker" in r.summary) for r in results] == [("skipped", True)]
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
        checker(_prepared(f"x.{'ts' if checker_name == 'eslint' else 'py'}"))

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
        result = checker(_prepared(f"x.{'ts' if checker_name == 'eslint' else 'py'}"))

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
        monkeypatch.setattr(sa, "detect_available_tools", lambda: set(sa._KNOWN_TOOLS))
        ext = "ts" if checker_name == "eslint" else "py"
        sug = _suggestion(f"x.{ext}")
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
    checker. It answers ``(content, "")`` or ``(None, why)``, never a guess."""

    def test_timeout_is_handled_not_escaped(self, monkeypatch):
        """TimeoutExpired is a SubprocessError, NOT an OSError: unnamed, it
        escaped into the calling checker and read as that checker timing out."""
        import subprocess as sp
        from neo import static_analysis as sa

        monkeypatch.setattr(
            sa.subprocess, "run",
            lambda cmd, **kw: (_ for _ in ()).throw(sp.TimeoutExpired(cmd, 30)),
        )
        content, why = sa.apply_diff_to_content(
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n", "old\n"
        )
        assert content is None and "timed out" in why

    def test_work_files_are_not_leaked(self, monkeypatch):
        """Only neo's own work directories are counted: globbing the shared
        temp dir raced every other process writing `tmp*` there."""
        import glob
        import os
        import subprocess as sp
        import tempfile
        from neo import static_analysis as sa

        pattern = os.path.join(tempfile.gettempdir(), "neo-check-*")
        before = set(glob.glob(pattern))
        monkeypatch.setattr(
            sa.subprocess, "run",
            lambda cmd, **kw: (_ for _ in ()).throw(sp.TimeoutExpired(cmd, 30)),
        )
        sa.apply_diff_to_content("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-old\n+new\n", "old\n")
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

    @pytest.mark.skipif(not __import__("shutil").which("patch"), reason="needs patch")
    def test_a_diff_that_does_not_apply_is_not_applied(self):
        """`patch` exits 1 and leaves the file alone. Its status was never
        read, so the ORIGINAL was linted and read as passed; when patch failed
        outright, a reconstruction from the diff's own lines was linted
        instead. Neither is the change."""
        import glob
        import os
        import tempfile
        from neo import static_analysis as sa

        before = set(glob.glob(os.path.join(tempfile.gettempdir(), "neo-check-*")))
        content, why = sa.apply_diff_to_content(
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-not there\n+broken(\n", "x = 1\n"
        )
        assert content is None and "did not apply" in why
        # The reject file lived beside the work file and leaked on every miss.
        assert not (set(glob.glob(os.path.join(tempfile.gettempdir(), "neo-check-*"))) - before)

    @pytest.mark.skipif(not __import__("shutil").which("patch"), reason="needs patch")
    def test_a_diff_that_applies_is_applied(self):
        from neo import static_analysis as sa

        content, why = sa.apply_diff_to_content(
            "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n",
            "def f():\n    return 1\n",
        )
        assert (content, why) == ("def f():\n    return 2\n", "")


_RUFF = __import__("shutil").which("ruff") and __import__("shutil").which("patch")


@pytest.mark.skipif(not _RUFF, reason="needs ruff and patch")
class TestTheCheckerJudgesTheChange:
    """Against real ruff and patch: what Linus measured passing that should
    not, and what failed that should not."""

    FILE = ("def f():\n    try:\n        return 1\n    except:\n        return 0\n\n\n"
            "def g():\n    return 1\n")

    def _check(self, tmp_path, diff="", code=""):
        (tmp_path / "c.py").write_text(self.FILE)
        return run_static_checks(
            [CodeSuggestion(file_path="c.py", unified_diff=diff, code_block=code,
                            description="", confidence=0.9)],
            enable_pyright=False, enable_mypy=False, enable_eslint=False,
            codebase_root=str(tmp_path),
        )[0]

    def _change_g(self, line):
        return f"--- a/c.py\n+++ b/c.py\n@@ -8,2 +8,2 @@\n def g():\n-    return 1\n+    {line}\n"

    def test_an_old_error_elsewhere_does_not_fail_a_clean_change(self, tmp_path):
        """The bare `except:` predates the change. Counting it failed every
        suggestion against this file, and a failed check vetoes promotion."""
        result = self._check(tmp_path, self._change_g("return 2"))
        assert result.status == "passed", result
        assert "1 issue(s) the file already had not counted" in result.summary

    def test_a_syntax_error_the_change_causes_fails_where_it_is_reported(self, tmp_path):
        """Reported past the last changed line, and ruff's `invalid-syntax`
        does not start with E: both used to let it through."""
        result = self._check(tmp_path, self._change_g("return ("))
        assert result.status == "failed", result

    def test_a_diff_that_does_not_apply_is_not_checked(self, tmp_path):
        result = self._check(
            tmp_path, "--- a/c.py\n+++ b/c.py\n@@ -1 +1 @@\n-def nothere():\n+def broken(:\n")
        assert result.status == "skipped" and "did not apply" in result.summary

    def test_a_header_only_diff_is_not_checked(self, tmp_path):
        result = self._check(tmp_path, "--- a/c.py\n+++ b/c.py\n")
        assert result.status == "skipped" and "no change" in result.summary

    def test_a_code_block_without_a_diff_is_not_checked(self, tmp_path):
        result = self._check(tmp_path, code="def broken(:")
        assert result.status == "skipped" and "code block" in result.summary

    def test_a_relative_path_is_read_from_the_codebase_not_the_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir("/")
        assert self._check(tmp_path, self._change_g("return 2")).status == "passed"

    def test_the_phase_budget_bounds_the_checks(self, tmp_path):
        (tmp_path / "c.py").write_text(self.FILE)
        result = run_static_checks(
            [CodeSuggestion(file_path="c.py", unified_diff=self._change_g("return 2"),
                            code_block="", description="", confidence=0.9)],
            enable_pyright=False, enable_mypy=False, enable_eslint=False,
            codebase_root=str(tmp_path), phase_budget=0,
        )[0]
        assert result.status == "skipped" and "budget" in result.summary

    def test_a_diff_already_in_the_file_is_checked_as_that_change(self, tmp_path):
        """`patch --batch` reversed an already-applied diff silently and the
        BEFORE was checked as the change: VERIFY on a change just made."""
        diff = self._change_g("return (")
        broken = self.FILE.rsplit("    return 1\n", 1)[0] + "    return (\n"
        (tmp_path / "c.py").write_text(broken)
        result = run_static_checks(
            [CodeSuggestion(file_path="c.py", unified_diff=diff, code_block="",
                            description="", confidence=0.9)],
            enable_pyright=False, enable_mypy=False, enable_eslint=False,
            codebase_root=str(tmp_path),
        )[0]
        assert result.status == "failed", result

    def test_a_blank_line_is_not_a_change(self, tmp_path):
        """`"" in "+-"` is True: a blank line passed for a change, and an
        empty new file linted clean."""
        from neo.models import diff_changes_lines

        assert not diff_changes_lines("--- a/x.py\n+++ b/x.py\n\n")
        assert not diff_changes_lines("See above.\n\nNothing to change.")
        result = run_static_checks(
            [CodeSuggestion(file_path="new.py", unified_diff="--- /dev/null\n+++ b/new.py\n\n",
                            code_block="", description="", confidence=0.9)],
            enable_pyright=False, enable_mypy=False, enable_eslint=False,
            codebase_root=str(tmp_path),
        )[0]
        assert result.status == "skipped" and "no change" in result.summary

    def test_a_new_file_header_on_a_file_that_exists_goes_through_patch(self, tmp_path):
        """The diff's header was believed over the disk: `-0,0` (an insert
        at the top) linted only the added lines as the whole file."""
        result = self._check(tmp_path, "--- a/c.py\n+++ b/c.py\n@@ -0,0 +1 @@\n+import os\n")
        # The whole file was checked: the insert's unused import is found,
        # and the bare except the file already had is not blamed on it.
        assert [d["code"] for d in result.diagnostics] == ["F401"], result
        assert "1 issue(s) the file already had" in result.summary

    def test_a_missing_baseline_never_reads_as_passed(self, tmp_path, monkeypatch):
        """The original could not be checked, so which findings are new is
        unknown: a syntax error reported past the change was dropped as
        "older" and the check passed."""
        from neo import static_analysis as sa

        real = sa.run_ruff_check
        calls = []

        def ruff_once_then_fail(prepared):
            calls.append(prepared)
            if len(calls) == 1:
                return real(prepared)
            return sa.StaticCheckResult(tool_name="ruff", diagnostics=[], summary="x",
                                        status="unavailable")

        monkeypatch.setattr(sa, "_LANGUAGE_CHECKERS", tuple(
            sa._LanguageChecker(c.tool_name, ruff_once_then_fail if c.tool_name == "ruff"
                                else c.run, c.extensions) for c in sa._LANGUAGE_CHECKERS))
        result = self._check(tmp_path, self._change_g("return ("))
        assert result.status == "unavailable" and "NOT run" in result.summary

    def test_style_lint_the_change_adds_is_a_warning_not_a_failure(self, tmp_path):
        """E7 is style in ruff's default set; `failed` vetoes promotion for
        the whole episode."""
        assert self._check(tmp_path, self._change_g("return [l for l in (1,)]")).status == "warning"

    def test_an_undefined_name_the_change_adds_fails(self, tmp_path):
        assert self._check(tmp_path, self._change_g("return nowhere")).status == "failed"

    def test_the_projects_ruff_config_applies(self, tmp_path):
        (tmp_path / "ruff.toml").write_text('[lint]\nignore = ["E741"]\n')
        assert self._check(tmp_path, self._change_g("return [l for l in (1,)]")).status == "passed"

    def test_a_crashed_tool_is_not_a_clean_one(self, tmp_path, monkeypatch):
        """Exit 2 with empty stdout parsed as "no issues"."""
        import subprocess as sp
        from neo import static_analysis as sa

        real = sp.run

        def crash_ruff(cmd, **kw):
            if cmd[0] == "ruff":
                return sp.CompletedProcess(cmd, 2, stdout="", stderr="error: bad config")
            return real(cmd, **kw)

        monkeypatch.setattr(sa.subprocess, "run", crash_ruff)
        result = self._check(tmp_path, self._change_g("return 2"))
        assert result.status == "unavailable" and "exit 2" in result.summary


def test_mypy_output_is_parsed_with_its_severity():
    """Split on ":" without the column, the message landed in `severity` and
    no mypy finding was ever an error."""
    import subprocess as sp
    from neo.static_analysis import _parse_mypy

    out = sp.CompletedProcess([], 1, stdout=(
        "/t/x.py:3:5: error: Incompatible types in assignment  [assignment]\n"
        "/t/x.py:4: note: See docs\n"
        "/t/x.py:1:1: error: Cannot find implementation or library stub  [import-not-found]\n"))
    assert _parse_mypy(out) == [{"line": 3, "column": 5, "severity": "error",
                                 "message": "Incompatible types in assignment  [assignment]"}]


def test_fast_runs_get_the_short_phase_budget(monkeypatch):
    from neo.config import NeoConfig
    from neo.engine import NeoEngine
    from neo import static_analysis as sa

    seen = {}
    monkeypatch.setattr(sa, "run_static_checks", lambda s, **kw: seen.update(kw) or [])
    config = NeoConfig()
    config.reasoning_mode = "fast"
    engine = NeoEngine(lm_adapter=None, enable_persistent_memory=False, config=config)
    engine._run_static_checks([], [])
    assert seen["phase_budget"] == sa.FAST_STATIC_CHECK_PHASE_BUDGET_SECONDS


@pytest.mark.skipif(not _RUFF, reason="needs ruff and patch")
class TestReviewRoundTwo:
    """Neo's second review, each reproduced with real ruff first."""

    def _run(self, tmp_path, before, diff, path="c.py"):
        (tmp_path / path).write_text(before)
        return run_static_checks(
            [CodeSuggestion(file_path=path, unified_diff=diff, code_block="",
                            description="", confidence=0.9)],
            enable_pyright=False, enable_mypy=False, enable_eslint=False,
            codebase_root=str(tmp_path),
        )

    def test_editing_only_the_comment_on_a_broken_line_does_not_fail(self, tmp_path):
        result = self._run(
            tmp_path, "def f():\n    return missing  # old\n",
            "--- a/c.py\n+++ b/c.py\n@@ -1,2 +1,2 @@\n def f():\n"
            "-    return missing  # old\n+    return missing  # new\n",
        )[0]
        assert result.status == "passed", result

    def test_a_malformed_creation_diff_is_not_checked(self, tmp_path):
        """Hunk says two lines, carries one: `patch` would reject it."""
        result = run_static_checks(
            [CodeSuggestion(file_path="new.py", code_block="", description="", confidence=0.9,
                            unified_diff="--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n")],
            enable_pyright=False, enable_mypy=False, enable_eslint=False,
            codebase_root=str(tmp_path),
        )[0]
        assert result.status == "skipped", result

    def test_a_well_formed_creation_diff_is_checked(self, tmp_path):
        result = run_static_checks(
            [CodeSuggestion(file_path="new.py", code_block="", description="", confidence=0.9,
                            unified_diff="--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n"
                                         "+x = 1\n+y = nowhere\n")],
            enable_pyright=False, enable_mypy=False, enable_eslint=False,
            codebase_root=str(tmp_path),
        )[0]
        assert result.status == "failed" and result.file_path == "new.py", result

    def test_another_files_section_is_not_read_as_this_one(self):
        from neo.static_analysis import _new_file_content

        two = ("--- /dev/null\n+++ b/a.py\n@@ -0,0 +1 @@\n+a = 1\n"
               "--- /dev/null\n+++ b/b.py\n@@ -0,0 +1 @@\n+b = 1\n")
        assert _new_file_content(two) is None

    def test_exit_one_with_no_report_is_not_clean(self, tmp_path, monkeypatch):
        import subprocess as sp
        from neo import static_analysis as sa

        real = sp.run

        def silent_failure(cmd, **kw):
            if cmd[0] == "ruff":
                return sp.CompletedProcess(cmd, 1, stdout="", stderr="")
            return real(cmd, **kw)

        monkeypatch.setattr(sa.subprocess, "run", silent_failure)
        result = self._run(tmp_path, "x = 1\n",
                           "--- a/c.py\n+++ b/c.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n")[0]
        assert result.status == "unavailable", result


def test_a_change_no_check_reached_is_unchecked_even_beside_a_clean_one():
    """A passed .py check and an unavailable eslint on a .ts change read as
    "all clean": coverage was counted by check, not by change."""
    from neo.engine import NeoEngine
    from neo.models import StaticCheckResult

    checks = [
        StaticCheckResult("ruff", [], "ok", status="passed", file_path="a.py"),
        StaticCheckResult("pyright", [], "x", status="unavailable", file_path="a.py"),
        StaticCheckResult("eslint", [], "x", status="unavailable", file_path="b.ts"),
        StaticCheckResult("constraints", [], "x", status="skipped"),
    ]
    assert NeoEngine._unchecked_changes(checks) == ["b.ts"]


@pytest.mark.skipif(not _RUFF, reason="needs ruff and patch")
class TestReviewRoundThree:
    """Linus's third review, reproduced with real ruff and patch."""

    def _run(self, tmp_path, before, diff, path="c.py"):
        (tmp_path / path).write_text(before)
        return run_static_checks(
            [CodeSuggestion(file_path=path, unified_diff=diff, code_block="",
                            description="", confidence=0.9)],
            enable_pyright=False, enable_mypy=False, enable_eslint=False,
            codebase_root=str(tmp_path),
        )[0]

    def test_deleting_a_line_that_is_not_there_is_not_already_applied(self, tmp_path):
        """It fails forwards and applies in reverse (only its context must
        match): it read as a change already made, and passed."""
        result = self._run(
            tmp_path, "def f(x):\n    return x\n",
            "--- a/c.py\n+++ b/c.py\n@@ -1,3 +1,2 @@\n def f(x):\n"
            "-    print('debug', undefined_dbg)\n     return x\n",
        )
        assert result.status == "skipped" and "did not apply" in result.summary, result

    def test_a_file_the_project_excludes_is_still_checked(self, tmp_path):
        """With force-exclude, ruff skipped it and printed `[]`: passed."""
        (tmp_path / "ruff.toml").write_text('force-exclude = true\nextend-exclude = ["c.py"]\n')
        result = self._run(
            tmp_path, "def f(x):\n    return x\n",
            "--- a/c.py\n+++ b/c.py\n@@ -1,2 +1,2 @@\n def f(x):\n-    return x\n+    return (x\n",
        )
        assert result.status == "failed", result

    def test_a_multi_file_creation_diff_does_not_splice_files(self, tmp_path):
        """The other file's `    return 2` was spliced into new.py: invalid
        syntax, `failed`, and a correct suggestion rejected by verification."""
        diff = ("--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x = 1\n"
                "--- a/d.py\n+++ b/d.py\n@@ -1,2 +1,2 @@\n def g():\n-    return 1\n+    return 2\n")
        result = run_static_checks(
            [CodeSuggestion(file_path="new.py", unified_diff=diff, code_block="",
                            description="", confidence=0.9)],
            enable_pyright=False, enable_mypy=False, enable_eslint=False,
            codebase_root=str(tmp_path),
        )[0]
        assert result.status == "skipped", result


def test_double_sign_lines_inside_a_hunk_are_changes():
    """`+++i;` and `-- comment` are code, not headers, once a hunk opens."""
    from neo.models import diff_changes_lines

    assert diff_changes_lines("--- a/x.js\n+++ b/x.js\n@@ -1 +1 @@\n i;\n+++i;\n")
    assert diff_changes_lines("--- a/q.sql\n+++ b/q.sql\n@@ -1,2 +1 @@\n select 1;\n--- old\n")
    assert not diff_changes_lines("--- a/x.js\n+++ b/x.js\n")
