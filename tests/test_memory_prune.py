import json
import time
from unittest.mock import MagicMock, patch

from neo.subcommands import _compact_fact_file


def test_compact_fact_file_drops_old_invalid_facts(tmp_path):
    now = time.time()
    path = tmp_path / "facts_project_test.json"
    path.write_text(json.dumps({
        "version": "2.0",
        "facts": [
            {
                "id": "old-invalid",
                "is_valid": False,
                "metadata": {"last_accessed": now - 60 * 86400},
            },
            {
                "id": "recent-invalid",
                "is_valid": False,
                "metadata": {"last_accessed": now},
            },
            {
                "id": "valid",
                "is_valid": True,
                "metadata": {"last_accessed": now - 60 * 86400},
            },
        ],
    }))

    stats = _compact_fact_file(path, max_invalid_age_days=30)

    assert stats["removed"] == 1
    data = json.loads(path.read_text())
    assert [f["id"] for f in data["facts"]] == ["recent-invalid", "valid"]


def test_compact_fact_file_dry_run_does_not_write(tmp_path):
    path = tmp_path / "facts_project_test.json"
    original = {
        "version": "2.0",
        "facts": [
            {
                "id": "old-invalid",
                "is_valid": False,
                "metadata": {"last_accessed": time.time() - 60 * 86400},
            },
        ],
    }
    path.write_text(json.dumps(original))

    stats = _compact_fact_file(path, max_invalid_age_days=30, dry_run=True)

    assert stats["removed"] == 1
    assert json.loads(path.read_text()) == original


def test_compact_fact_file_strips_retained_tombstone_embeddings(tmp_path):
    now = time.time()
    path = tmp_path / "facts_project_test.json"
    path.write_text(json.dumps({
        "version": "2.0",
        "facts": [
            {
                # Invalid but too young to purge — keep the row, drop the vector.
                "id": "recent-invalid",
                "is_valid": False,
                "embedding": [0.1] * 768,
                "metadata": {"last_accessed": now},
            },
            {
                # Valid — embedding must survive (still retrievable).
                "id": "valid",
                "is_valid": True,
                "embedding": [0.2] * 768,
                "metadata": {"last_accessed": now},
            },
        ],
    }))

    stats = _compact_fact_file(path, max_invalid_age_days=30)

    assert stats["removed"] == 0
    assert stats["stripped"] == 1
    facts = {f["id"]: f for f in json.loads(path.read_text())["facts"]}
    assert "embedding" not in facts["recent-invalid"]
    assert facts["valid"]["embedding"] == [0.2] * 768


def test_compact_fact_file_write_takes_scope_lock(tmp_path):
    """A real (non-dry-run) compaction must serialize under the scope-file lock
    so it can't clobber a concurrent FactStore.save()."""
    now = time.time()
    path = tmp_path / "facts_project_test.json"
    path.write_text(json.dumps({
        "version": "2.0",
        "facts": [{
            "id": "recent-invalid", "is_valid": False,
            "embedding": [0.1] * 768, "metadata": {"last_accessed": now},
        }],
    }))

    entered = MagicMock()
    with patch("neo.memory.store.scope_file_lock") as mock_lock:
        mock_lock.return_value.__enter__ = lambda *_: entered()
        mock_lock.return_value.__exit__ = lambda *a: False
        _compact_fact_file(path, max_invalid_age_days=30)

    mock_lock.assert_called_once_with(path)
    entered.assert_called_once()


def test_compact_fact_file_dry_run_skips_scope_lock(tmp_path):
    """Dry-run never writes, so it must not contend for the lock."""
    path = tmp_path / "facts_project_test.json"
    path.write_text(json.dumps({
        "version": "2.0",
        "facts": [{
            "id": "recent-invalid", "is_valid": False,
            "embedding": [0.1] * 768, "metadata": {"last_accessed": time.time()},
        }],
    }))

    with patch("neo.memory.store.scope_file_lock") as mock_lock:
        _compact_fact_file(path, max_invalid_age_days=30, dry_run=True)

    mock_lock.assert_not_called()


def test_compact_fact_file_dry_run_does_not_strip(tmp_path):
    now = time.time()
    path = tmp_path / "facts_project_test.json"
    original = {
        "version": "2.0",
        "facts": [
            {
                "id": "recent-invalid",
                "is_valid": False,
                "embedding": [0.1] * 768,
                "metadata": {"last_accessed": now},
            },
        ],
    }
    path.write_text(json.dumps(original))

    stats = _compact_fact_file(path, max_invalid_age_days=30, dry_run=True)

    assert stats["stripped"] == 1
    assert json.loads(path.read_text()) == original
