import json
import time

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
