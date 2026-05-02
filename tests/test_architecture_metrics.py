"""Tests for neo.architecture_metrics — Python-only structural snapshots."""

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
        snap = ArchSnapshot(cycle_count=2, god_file_count=3, max_nesting_depth=5, files_scanned=42)
        assert ArchSnapshot.from_dict(snap.to_dict()) == snap

    def test_from_dict_handles_missing_keys(self):
        snap = ArchSnapshot.from_dict({"cycle_count": 1})
        assert snap.cycle_count == 1
        assert snap.god_file_count == 0
        assert snap.max_nesting_depth == 0
        assert snap.files_scanned == 0

    def test_from_dict_handles_none(self):
        assert ArchSnapshot.from_dict(None) == ArchSnapshot()


# ---------------------------------------------------------------------------
# Delta + severity
# ---------------------------------------------------------------------------

class TestDeltaSeverity:
    def test_no_change_is_neutral(self):
        before = after = ArchSnapshot(1, 1, 3, 10)
        assert compare(before, after).severity() == "neutral"

    def test_new_cycle_is_regression(self):
        before = ArchSnapshot(0, 0, 3, 10)
        after = ArchSnapshot(1, 0, 3, 10)
        assert compare(before, after).severity() == "regression"

    def test_new_god_file_is_regression(self):
        before = ArchSnapshot(0, 0, 3, 10)
        after = ArchSnapshot(0, 1, 3, 10)
        assert compare(before, after).severity() == "regression"

    def test_depth_jitter_within_band_is_neutral(self):
        # depth_delta == 1 is within the noise band; not a regression.
        before = ArchSnapshot(0, 0, 3, 10)
        after = ArchSnapshot(0, 0, 4, 10)
        assert compare(before, after).severity() == "neutral"

    def test_depth_increase_above_band_is_regression(self):
        before = ArchSnapshot(0, 0, 3, 10)
        after = ArchSnapshot(0, 0, 5, 10)
        assert compare(before, after).severity() == "regression"

    def test_cycle_removed_is_improvement(self):
        before = ArchSnapshot(2, 0, 3, 10)
        after = ArchSnapshot(1, 0, 3, 10)
        assert compare(before, after).severity() == "improvement"

    def test_god_file_removed_is_improvement(self):
        before = ArchSnapshot(0, 2, 3, 10)
        after = ArchSnapshot(0, 1, 3, 10)
        assert compare(before, after).severity() == "improvement"


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
