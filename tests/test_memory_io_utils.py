"""Tests for memory I/O helpers."""

import json

from neo.memory.io_utils import atomic_write_json


def test_atomic_write_json_roundtrip_and_cleans_temp_files(tmp_path):
    path = tmp_path / "state.json"

    atomic_write_json(path, {"items": [1, 2]}, indent=2)

    assert json.loads(path.read_text()) == {"items": [1, 2]}
    assert list(tmp_path.glob("*.tmp")) == []
