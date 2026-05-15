"""Tests for neo.architecture_metrics — structural snapshots."""

from pathlib import Path

from neo.architecture_metrics import (
    ArchSnapshot,
    compare,
    compute,
)


def _write(root: Path, rel: str, content: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    def test_flat_layout_two_module_cycle(self, tmp_path: Path):
        _write(tmp_path, "a.py", "import b\n")
        _write(tmp_path, "b.py", "import a\n")
        snap = compute(tmp_path)
        assert snap.cycle_count == 1

    def test_no_cycle(self, tmp_path: Path):
        _write(tmp_path, "a.py", "import b\n")
        _write(tmp_path, "b.py", "import c\n")
        _write(tmp_path, "c.py", "x = 1\n")
        assert compute(tmp_path).cycle_count == 0

    def test_three_node_cycle(self, tmp_path: Path):
        _write(tmp_path, "a.py", "import b\n")
        _write(tmp_path, "b.py", "import c\n")
        _write(tmp_path, "c.py", "import a\n")
        assert compute(tmp_path).cycle_count == 1

    def test_external_import_does_not_count(self, tmp_path: Path):
        # `os` and friends aren't in our scanned graph — no cycle.
        _write(tmp_path, "a.py", "import os\nimport sys\n")
        assert compute(tmp_path).cycle_count == 0

    def test_package_layout_cycle(self, tmp_path: Path):
        # pkg/x.py imports pkg.y; pkg/y.py imports pkg.x.
        _write(tmp_path, "pkg/__init__.py", "")
        _write(tmp_path, "pkg/x.py", "from pkg import y\n")
        _write(tmp_path, "pkg/y.py", "from pkg import x\n")
        assert compute(tmp_path).cycle_count == 1


# ---------------------------------------------------------------------------
# God file detection
# ---------------------------------------------------------------------------

class TestGodFiles:
    def test_short_file_not_god(self, tmp_path: Path):
        _write(tmp_path, "a.py", "x = 1\n")
        assert compute(tmp_path).god_file_count == 0

    def test_long_file_is_god(self, tmp_path: Path):
        # > _GOD_FILE_LOC_THRESHOLD (800) lines triggers it.
        _write(tmp_path, "a.py", "x = 1\n" * 1000)
        assert compute(tmp_path).god_file_count == 1

    def test_many_functions_is_god(self, tmp_path: Path):
        # > _GOD_FILE_FUNC_THRESHOLD (30) functions also triggers it.
        funcs = "\n".join(f"def f{i}(): pass" for i in range(40))
        _write(tmp_path, "a.py", funcs)
        assert compute(tmp_path).god_file_count == 1


# ---------------------------------------------------------------------------
# Nesting depth
# ---------------------------------------------------------------------------

class TestNestingDepth:
    def test_flat_function(self, tmp_path: Path):
        _write(tmp_path, "a.py", "def f():\n    return 1\n")
        assert compute(tmp_path).max_nesting_depth == 0

    def test_nested_ifs(self, tmp_path: Path):
        src = (
            "def f():\n"
            "    if 1:\n"
            "        if 2:\n"
            "            if 3:\n"
            "                pass\n"
        )
        _write(tmp_path, "a.py", src)
        assert compute(tmp_path).max_nesting_depth == 3

    def test_nested_function_does_not_inflate_outer(self, tmp_path: Path):
        # outer's own depth is 1 (one if). The nested helper function has
        # its own depth-3 nesting, but that should NOT propagate up to the
        # outer function's measurement — they're measured independently.
        src = (
            "def outer():\n"
            "    def helper():\n"
            "        if 1:\n"
            "            if 2:\n"
            "                if 3:\n"
            "                    pass\n"
            "    if 1:\n"
            "        pass\n"
        )
        _write(tmp_path, "a.py", src)
        # Result is helper's depth (3), not outer's-plus-helper's (4).
        assert compute(tmp_path).max_nesting_depth == 3

    def test_for_while_with_try_count_as_nesting(self, tmp_path: Path):
        src = (
            "def f():\n"
            "    for x in []:\n"
            "        while True:\n"
            "            with open('x') as h:\n"
            "                try:\n"
            "                    pass\n"
            "                except Exception:\n"
            "                    pass\n"
        )
        _write(tmp_path, "a.py", src)
        assert compute(tmp_path).max_nesting_depth == 4


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

class TestSnapshotIO:
    def test_to_from_dict_roundtrip(self):
        snap = ArchSnapshot(
            cycle_count=2,
            god_file_count=3,
            max_nesting_depth=5,
            files_scanned=42,
            python_files_scanned=40,
        )
        assert ArchSnapshot.from_dict(snap.to_dict()) == snap

    def test_from_dict_handles_missing_keys(self):
        snap = ArchSnapshot.from_dict({"cycle_count": 1})
        assert snap.cycle_count == 1
        assert snap.god_file_count == 0
        assert snap.max_nesting_depth == 0
        assert snap.files_scanned == 0

    def test_from_dict_handles_none(self):
        assert ArchSnapshot.from_dict(None) == ArchSnapshot()

    def test_legacy_dict_without_python_files_scanned_falls_back(self):
        # Snapshots predating the python_files_scanned split were
        # Python-only by construction. Loading one should set the new
        # field to match files_scanned so coverage gating still trusts
        # the cycle/depth signal.
        legacy = {
            "cycle_count": 1,
            "god_file_count": 0,
            "max_nesting_depth": 3,
            "files_scanned": 5,
        }
        snap = ArchSnapshot.from_dict(legacy)
        assert snap.python_files_scanned == 5


# ---------------------------------------------------------------------------
# Delta + severity
# ---------------------------------------------------------------------------

class TestDeltaSeverity:
    # All snapshots in this class declare python_files_scanned == files_scanned
    # so the python_coverage gate is on — these tests exercise the legacy
    # Python-only semantics.

    def test_no_change_is_neutral(self):
        before = after = ArchSnapshot(1, 1, 3, 10, 10)
        assert compare(before, after).severity() == "neutral"

    def test_new_cycle_is_regression(self):
        before = ArchSnapshot(0, 0, 3, 10, 10)
        after = ArchSnapshot(1, 0, 3, 10, 10)
        assert compare(before, after).severity() == "regression"

    def test_new_god_file_is_regression(self):
        before = ArchSnapshot(0, 0, 3, 10, 10)
        after = ArchSnapshot(0, 1, 3, 10, 10)
        assert compare(before, after).severity() == "regression"

    def test_depth_jitter_within_band_is_neutral(self):
        # depth_delta == 1 is within the noise band; not a regression.
        before = ArchSnapshot(0, 0, 3, 10, 10)
        after = ArchSnapshot(0, 0, 4, 10, 10)
        assert compare(before, after).severity() == "neutral"

    def test_depth_increase_above_band_is_regression(self):
        before = ArchSnapshot(0, 0, 3, 10, 10)
        after = ArchSnapshot(0, 0, 5, 10, 10)
        assert compare(before, after).severity() == "regression"

    def test_cycle_removed_is_improvement(self):
        before = ArchSnapshot(2, 0, 3, 10, 10)
        after = ArchSnapshot(1, 0, 3, 10, 10)
        assert compare(before, after).severity() == "improvement"

    def test_god_file_removed_is_improvement(self):
        before = ArchSnapshot(0, 2, 3, 10, 10)
        after = ArchSnapshot(0, 1, 3, 10, 10)
        assert compare(before, after).severity() == "improvement"


# ---------------------------------------------------------------------------
# Coverage-gated severity (Python-only metric channels)
# ---------------------------------------------------------------------------

class TestCoverageGatedSeverity:
    def test_no_python_coverage_suppresses_cycle_signal(self):
        # JS-only sessions can't produce a real cycle signal. If both
        # snapshots have python_files_scanned=0, the cycle delta is no
        # signal — severity should ignore it.
        before = ArchSnapshot(0, 0, 0, 5, 0)
        after = ArchSnapshot(1, 0, 0, 5, 0)  # synthetic — real JS-only can't produce this
        assert compare(before, after).severity() == "neutral"

    def test_no_python_coverage_suppresses_depth_signal(self):
        before = ArchSnapshot(0, 0, 0, 5, 0)
        after = ArchSnapshot(0, 0, 5, 5, 0)
        assert compare(before, after).severity() == "neutral"

    def test_no_python_coverage_still_trusts_god_files(self):
        # God-file detection is language-agnostic — a new JS god file
        # IS a regression even without any Python in either snapshot.
        before = ArchSnapshot(0, 0, 0, 5, 0)
        after = ArchSnapshot(0, 1, 0, 5, 0)
        assert compare(before, after).severity() == "regression"

    def test_partial_coverage_loss_suppresses_python_channels(self):
        # Python files existed before but were removed mid-session. The
        # "improvement" reading on depth would be a false positive.
        before = ArchSnapshot(0, 0, 5, 10, 10)
        after = ArchSnapshot(0, 0, 0, 8, 0)
        assert compare(before, after).severity() == "neutral"

    def test_manually_constructed_delta_defaults_to_coverage(self):
        # Constructing an ArchDelta directly (legacy tests, callers
        # outside compare()) must keep working — default coverage=True.
        from neo.architecture_metrics import ArchDelta
        d = ArchDelta(cycles_delta=1, god_files_delta=0, max_depth_delta=0)
        assert d.severity() == "regression"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_none_root_returns_zero_snapshot(self):
        assert compute(None) == ArchSnapshot()

    def test_missing_root_returns_zero_snapshot(self, tmp_path: Path):
        assert compute(tmp_path / "nope") == ArchSnapshot()

    def test_empty_root_returns_zero_snapshot(self, tmp_path: Path):
        assert compute(tmp_path) == ArchSnapshot()

    def test_syntax_error_file_skipped_silently(self, tmp_path: Path):
        _write(tmp_path, "broken.py", "def bad(:\n")
        _write(tmp_path, "ok.py", "x = 1\n")
        snap = compute(tmp_path)
        # Only the parseable file counts toward files_scanned.
        assert snap.files_scanned == 2  # both walked, broken one is parsed-and-skipped
        # No metrics breakage.
        assert snap.cycle_count == 0
        assert snap.god_file_count == 0

    def test_ignored_directories_are_skipped(self, tmp_path: Path):
        _write(tmp_path, ".venv/lib/big.py", "x = 1\n" * 2000)
        _write(tmp_path, "real.py", "x = 1\n")
        snap = compute(tmp_path)
        # The .venv god file shouldn't count.
        assert snap.god_file_count == 0
        assert snap.files_scanned == 1


# ---------------------------------------------------------------------------
# Multi-language god-file detection (tree-sitter)
# ---------------------------------------------------------------------------

class TestMultiLanguageGodFiles:
    def test_long_javascript_file_is_god(self, tmp_path: Path):
        _write(tmp_path, "a.js", "const x = 1;\n" * 1000)
        snap = compute(tmp_path)
        assert snap.god_file_count == 1
        assert snap.files_scanned == 1

    def test_many_typescript_functions_is_god(self, tmp_path: Path):
        funcs = "\n".join(f"function f{i}(): void {{}}" for i in range(40))
        _write(tmp_path, "a.ts", funcs)
        snap = compute(tmp_path)
        assert snap.god_file_count == 1

    def test_short_java_file_not_god(self, tmp_path: Path):
        _write(tmp_path, "A.java", "class A { void m() {} }\n")
        assert compute(tmp_path).god_file_count == 0

    def test_long_go_file_is_god(self, tmp_path: Path):
        _write(tmp_path, "main.go", "package main\n" + "var x = 1\n" * 900)
        assert compute(tmp_path).god_file_count == 1

    def test_mixed_language_files_all_scanned(self, tmp_path: Path):
        _write(tmp_path, "a.py", "x = 1\n")
        _write(tmp_path, "a.js", "const x = 1;\n")
        _write(tmp_path, "A.java", "class A {}\n")
        snap = compute(tmp_path)
        assert snap.files_scanned == 3

    def test_non_python_does_not_contribute_to_cycles(self, tmp_path: Path):
        # JS imports each other but cycle detection is Python-only by
        # design — we don't try to reconcile JS module paths with
        # Python's dotted scheme.
        _write(tmp_path, "a.js", "import './b';\n")
        _write(tmp_path, "b.js", "import './a';\n")
        assert compute(tmp_path).cycle_count == 0

    def test_non_python_does_not_contribute_to_depth(self, tmp_path: Path):
        # Deeply nested JS — depth metric is Python-only and stays 0.
        src = (
            "function f() {\n"
            "  if (a) { if (b) { if (c) { if (d) { x(); } } } }\n"
            "}\n"
        )
        _write(tmp_path, "a.js", src)
        assert compute(tmp_path).max_nesting_depth == 0

    # --- Ruby / Kotlin / Swift / PHP now have query coverage too ---

    def test_long_ruby_file_is_god(self, tmp_path: Path):
        _write(tmp_path, "a.rb", "x = 1\n" * 1000)
        assert compute(tmp_path).god_file_count == 1

    def test_many_ruby_methods_is_god(self, tmp_path: Path):
        methods = "\n".join(f"def m{i}; end" for i in range(40))
        _write(tmp_path, "a.rb", f"class Big\n{methods}\nend\n")
        assert compute(tmp_path).god_file_count == 1

    def test_long_kotlin_file_is_god(self, tmp_path: Path):
        _write(tmp_path, "a.kt", "val x = 1\n" * 1000)
        assert compute(tmp_path).god_file_count == 1

    def test_long_swift_file_is_god(self, tmp_path: Path):
        _write(tmp_path, "a.swift", "let x = 1\n" * 1000)
        assert compute(tmp_path).god_file_count == 1

    def test_long_php_file_is_god(self, tmp_path: Path):
        _write(tmp_path, "a.php", "<?php\n" + "$x = 1;\n" * 1000)
        assert compute(tmp_path).god_file_count == 1
