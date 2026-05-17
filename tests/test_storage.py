"""Tests for neo.storage file persistence."""

import json

import pytest

from neo.storage import FileStorage


def test_file_storage_save_is_atomic_and_loads_roundtrip(tmp_path):
    storage = FileStorage(base_path=str(tmp_path))

    storage.save_entries("global", [{"pattern": "x", "confidence": 0.5}])

    path = tmp_path / "global_memory.json"
    data = json.loads(path.read_text())
    assert data["version"] == "1.0"
    assert data["entries"] == [{"pattern": "x", "confidence": 0.5}]
    assert storage.load_entries("global") == [{"pattern": "x", "confidence": 0.5}]
    assert list(tmp_path.glob("*.tmp")) == []


def test_file_storage_backs_up_corrupt_json_before_raising(tmp_path):
    storage = FileStorage(base_path=str(tmp_path))
    path = tmp_path / "global_memory.json"
    path.write_text('{"entries": [')

    with pytest.raises(json.JSONDecodeError):
        storage.load_entries("global")

    backups = list(tmp_path.glob("global_memory.json.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == '{"entries": ['
