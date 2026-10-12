"""`neo --update`: reachable on its own, sees new files, reports what it did,
and never loses catalog data doing it (#217).

Every refresh here runs on a catalog RELOADED from disk, as the CLI does: a
`ProjectIndex` that built the catalog still holds a live parser, and a refresh
on that object hid a refresh that wiped every changed file's chunks."""

import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from neo.cli import _format_refresh_report
from neo.index.project_index import IndexSnapshot, ProjectIndex

REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _build(root: Path, **kw) -> None:
    ProjectIndex(str(root)).build_index(file_patterns=["**/*.py"], **kw)


def _refresh(root: Path, **kw):
    index = ProjectIndex(str(root))
    return index.refresh_changed_files(**kw), index


def _chunks(root: Path) -> dict:
    out: dict = {}
    for c in ProjectIndex(str(root)).chunks:
        out.setdefault(c.file_path, []).append(c.content)
    return out


def test_bare_update_refreshes_and_keeps_every_chunk(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _build(tmp_path)
    _write(tmp_path, "a.py", "def a():\n    return 2\n")
    _write(tmp_path, "b.py", "def b():\n    return 2\n")
    result = subprocess.run(
        [sys.executable, "-m", "neo.cli", "--update"], cwd=str(tmp_path),
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": REPO_SRC, "HOME": str(tmp_path / "home")},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("[Neo] Updated index: 1 changed, 1 new files, 2 chunks embedded")
    chunks = _chunks(tmp_path)
    assert set(chunks) == {"a.py", "b.py"}
    assert "return 2" in chunks["a.py"][0]


def test_a_changed_file_is_re_extracted_from_a_reloaded_catalog(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _build(tmp_path)
    _write(tmp_path, "a.py", "def a():\n    return 2\n\ndef c():\n    return 3\n")
    result, _ = _refresh(tmp_path)
    assert (result["refreshed"], result["chunks_embedded"]) == (1, 2)
    assert len(_chunks(tmp_path)["a.py"]) == 2


def test_a_new_file_is_picked_up_once(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _build(tmp_path)
    _write(tmp_path, "b.py", "def b():\n    return 2\n")
    assert ProjectIndex(str(tmp_path)).check_staleness()[2] == ["b.py"]

    result, _ = _refresh(tmp_path)
    assert (result["added"], result["refreshed"], result["bound"]) == (1, 0, None)
    assert set(_chunks(tmp_path)) == {"a.py", "b.py"}
    assert ProjectIndex(str(tmp_path)).check_staleness()[2] == []


def test_a_file_the_cap_passed_over_is_not_new_and_new_files_respect_the_cap(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _write(tmp_path, "b.py", "def b():\n    return 2\n")
    _build(tmp_path, max_files=1)
    assert ProjectIndex(str(tmp_path)).check_staleness()[2] == []

    _write(tmp_path, "c.py", "def c():\n    return 3\n")
    result, index = _refresh(tmp_path)
    assert (result["added"], result["skipped_cap"]) == (0, 1)
    assert any("--max-files=1 limit" in line for line in _format_refresh_report(result, index))
    assert _refresh(tmp_path)[0]["skipped_cap"] == 0  # recorded as seen, not retried


def test_a_new_copy_of_catalogued_content_is_skipped(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _build(tmp_path)
    _write(tmp_path, "copy.py", "def a():\n    return 1\n")
    result, _ = _refresh(tmp_path)
    assert (result["added"], result["skipped_duplicate"]) == (0, 1)
    assert set(_chunks(tmp_path)) == {"a.py"}


def test_a_deletion_alone_is_saved_and_a_recreated_file_is_new(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _write(tmp_path, "b.py", "def b():\n    return 2\n")
    _build(tmp_path)
    (tmp_path / "b.py").unlink()
    assert _refresh(tmp_path)[0]["removed"] == 1
    assert set(_chunks(tmp_path)) == {"a.py"}
    assert "b.py" not in ProjectIndex(str(tmp_path)).snapshot.file_hashes

    _write(tmp_path, "b.py", "def b():\n    return 3\n")
    assert _refresh(tmp_path)[0]["added"] == 1


def test_a_file_newly_ignored_leaves_the_catalog(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _write(tmp_path, "b.py", "def b():\n    return 2\n")
    _build(tmp_path)
    _write(tmp_path, ".gitignore", "b.py\n")
    assert _refresh(tmp_path)[0]["removed"] == 1
    assert set(_chunks(tmp_path)) == {"a.py"}


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads unreadable files")
def test_an_unreadable_file_keeps_its_entries_and_is_reported_not_removed(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _write(tmp_path, "b.py", "def b():\n    return 2\n")
    _build(tmp_path)
    (tmp_path / "b.py").chmod(0)
    try:
        for _ in range(2):
            result, index = _refresh(tmp_path)
            assert result["removed"] == 0 and result["failed"] == ["b.py"]
            assert set(_chunks(tmp_path)) == {"a.py", "b.py"}
        assert any("could not read or parse 1 file (b.py)" in line
                   for line in _format_refresh_report(result, index))
    finally:
        (tmp_path / "b.py").chmod(0o644)


def test_edges_follow_the_file(tmp_path):
    _write(tmp_path, "a.py", "import os\n\ndef a():\n    return os.getcwd()\n")
    _build(tmp_path)
    before = [e for e in ProjectIndex(str(tmp_path)).edges if e["source_file"] == "a.py"]
    assert any("os" in e["target_symbol"] for e in before), before

    _write(tmp_path, "a.py", "import sys\n\ndef a():\n    return sys.argv\n")
    _refresh(tmp_path)
    after = [e for e in ProjectIndex(str(tmp_path)).edges if e["source_file"] == "a.py"]
    assert not any("os" in e["target_symbol"] for e in after), after
    assert any("sys" in e["target_symbol"] for e in after), after


def test_a_budget_bound_refresh_says_so_and_always_progresses(tmp_path):
    for name in ("a", "b", "c"):
        _write(tmp_path, f"{name}.py", f"def {name}():\n    return 1\n")
    _build(tmp_path)
    for name in ("a", "b", "c"):
        _write(tmp_path, f"{name}.py", f"def {name}():\n    return 2\n")

    result, index = _refresh(tmp_path, max_chunks=1)
    assert result["bound"] == "chunks" and result["refreshed"] == 1
    assert result["remaining"] == ["b.py", "c.py"]
    assert any("Stopped at the 100-chunk budget with 2 stale files left" in line
               for line in _format_refresh_report(result, index))

    result, _ = _refresh(tmp_path, budget_ms=0)
    assert result["bound"] == "time" and result["refreshed"] == 1

    result, index = _refresh(tmp_path)
    lines = _format_refresh_report(result, index)
    assert lines == [f"[Neo] Updated index: 1 changed file, 1 chunk embedded; now "
                     f"{len(index.chunks)} chunks, {len(index.edges)} edges"]


def test_older_neo_can_still_read_the_snapshot(tmp_path):
    """Older neo builds `IndexSnapshot(**json)` and discards the catalog on an
    unknown key, so new state lives beside index.json, never in it."""
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _build(tmp_path)
    keys = set(json.loads((tmp_path / ".neo" / "index.json").read_text()))
    assert keys <= {f.name for f in fields(IndexSnapshot)}
    assert (tmp_path / ".neo" / "walked.json").exists()


def test_a_newer_snapshot_key_is_ignored_not_fatal(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _build(tmp_path)
    snapshot = tmp_path / ".neo" / "index.json"
    data = json.loads(snapshot.read_text())
    data["from_the_future"] = 1
    snapshot.write_text(json.dumps(data))
    assert set(_chunks(tmp_path)) == {"a.py"}


def test_a_catalog_from_an_older_version_says_it_cannot_see_new_files(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _build(tmp_path)
    (tmp_path / ".neo" / "walked.json").unlink()
    _write(tmp_path, "b.py", "def b():\n    return 2\n")

    result, index = _refresh(tmp_path)
    assert result["added"] == 0 and result["tracks_new_files"] is False
    assert any("predates new-file tracking" in line
               for line in _format_refresh_report(result, index))
