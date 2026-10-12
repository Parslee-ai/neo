"""`neo --update`: reachable on its own, sees new files, reports what it did (#217)."""

import json
import os
import subprocess
import sys
from pathlib import Path

from neo.cli import _format_refresh_report
from neo.index.project_index import ProjectIndex

REPO_SRC = str(Path(__file__).resolve().parents[1] / "src")


def _write(root: Path, rel: str, body: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _built(root: Path, **kw) -> ProjectIndex:
    index = ProjectIndex(str(root))
    index.build_index(file_patterns=["**/*.py"], **kw)
    return index


def test_bare_update_refreshes_instead_of_reasoning(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _built(tmp_path)
    _write(tmp_path, "b.py", "def b():\n    return 2\n")
    result = subprocess.run(
        [sys.executable, "-m", "neo.cli", "--update"], cwd=str(tmp_path),
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": REPO_SRC, "HOME": str(tmp_path / "home")},
    )
    assert result.returncode == 0, result.stderr
    assert "[Neo] Updated index: 1 new file" in result.stdout
    assert "b.py" in ProjectIndex(str(tmp_path)).snapshot.file_hashes


def test_a_new_file_is_picked_up_once(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    index = _built(tmp_path)
    _write(tmp_path, "b.py", "def b():\n    return 2\n")
    assert index.check_staleness()[2] == ["b.py"]

    result = index.refresh_changed_files()
    assert (result["added"], result["refreshed"], result["bound"]) == (1, 0, None)
    assert {c.file_path for c in index.chunks} == {"a.py", "b.py"}
    assert index.refresh_changed_files()["added"] == 0
    assert ProjectIndex(str(tmp_path)).check_staleness()[2] == []


def test_a_file_the_cap_passed_over_is_not_new(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _write(tmp_path, "b.py", "def b():\n    return 2\n")
    index = _built(tmp_path, max_files=1)
    assert len(index.snapshot.file_hashes) == 1
    assert index.check_staleness()[2] == []


def test_a_deletion_alone_is_saved_and_a_recreated_file_is_new(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _write(tmp_path, "b.py", "def b():\n    return 2\n")
    index = _built(tmp_path)
    (tmp_path / "b.py").unlink()
    assert index.refresh_changed_files()["removed"] == 1

    reloaded = ProjectIndex(str(tmp_path))
    assert "b.py" not in reloaded.snapshot.file_hashes
    assert {c.file_path for c in reloaded.chunks} == {"a.py"}

    _write(tmp_path, "b.py", "def b():\n    return 3\n")
    assert reloaded.refresh_changed_files()["added"] == 1


def test_edges_follow_the_file(tmp_path):
    _write(tmp_path, "a.py", "import os\n\ndef a():\n    return os.getcwd()\n")
    index = _built(tmp_path)
    before = [e for e in index.edges if e["source_file"] == "a.py"]
    assert any("os" in e["target_symbol"] for e in before), before

    _write(tmp_path, "a.py", "import sys\n\ndef a():\n    return sys.argv\n")
    index.refresh_changed_files()
    after = [e for e in ProjectIndex(str(tmp_path)).edges if e["source_file"] == "a.py"]
    assert not any("os" in e["target_symbol"] for e in after), after
    assert any("sys" in e["target_symbol"] for e in after), after


def test_a_budget_bound_refresh_says_so(tmp_path):
    for name in ("a", "b", "c"):
        _write(tmp_path, f"{name}.py", f"def {name}():\n    return 1\n")
    index = _built(tmp_path)
    for name in ("a", "b", "c"):
        _write(tmp_path, f"{name}.py", f"def {name}():\n    return 2\n")

    result = index.refresh_changed_files(max_chunks=1)
    assert result["bound"] == "chunks" and result["refreshed"] == 1
    assert result["remaining"] == ["b.py", "c.py"]
    lines = _format_refresh_report(result, index)
    assert any("Stopped at the 100-chunk budget with 2 stale files left" in line
               for line in lines), lines

    complete = _format_refresh_report(index.refresh_changed_files(), index)
    assert len(complete) == 1 and complete[0].startswith("[Neo] Updated index: 2 changed files")


def test_a_catalog_from_an_older_version_says_it_cannot_see_new_files(tmp_path):
    _write(tmp_path, "a.py", "def a():\n    return 1\n")
    _built(tmp_path)
    snapshot = tmp_path / ".neo" / "index.json"
    data = json.loads(snapshot.read_text())
    del data["file_patterns"], data["walked_paths"]
    snapshot.write_text(json.dumps(data))
    _write(tmp_path, "b.py", "def b():\n    return 2\n")

    index = ProjectIndex(str(tmp_path))
    result = index.refresh_changed_files()
    assert result["added"] == 0 and result["tracks_new_files"] is False
    assert any("predates new-file tracking" in line
               for line in _format_refresh_report(result, index))
