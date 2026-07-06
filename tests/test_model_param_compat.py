"""Tests for `_ModelParamCompat` — the persistent store that records which
parameter adaptations each model needs (issue #133).

The autouse `isolate_neo_home` fixture (tests/conftest.py) redirects
`Path.home()` to a per-test tmp dir, so these exercise the real on-disk path
without touching the user's ~/.neo.
"""

import json
from pathlib import Path

import pytest

from neo.adapters import _ModelParamCompat


def _store_file() -> Path:
    return Path.home() / ".neo" / "model_param_compat.json"


def test_learn_persists_to_disk():
    store = _ModelParamCompat()
    store.learn("openai", "o3-mini", "drop_temperature")

    path = _store_file()
    assert path.exists()
    data = json.loads(path.read_text())
    assert data == {"openai:o3-mini": ["drop_temperature"]}


def test_learning_survives_a_fresh_store_instance():
    """The whole point of #133: a new process (a new store instance loading
    from disk) skips the bad param without re-paying the 400 penalty."""
    _ModelParamCompat().learn("openai", "o3-mini", "drop_temperature")

    fresh = _ModelParamCompat()  # simulates a new CLI invocation
    assert fresh.has("openai", "o3-mini", "drop_temperature")
    assert not fresh.has("openai", "o3-mini", "rename_max_tokens")


def test_has_false_for_unknown():
    store = _ModelParamCompat()
    assert not store.has("openai", "gpt-4", "drop_temperature")


def test_provider_scoping_avoids_collisions():
    store = _ModelParamCompat()
    store.learn("azure", "gpt-4", "drop_temperature")
    assert store.has("azure", "gpt-4", "drop_temperature")
    assert not store.has("openai", "gpt-4", "drop_temperature")
    assert not store.has("local", "gpt-4", "drop_temperature")


def test_multiple_adaptations_accumulate():
    store = _ModelParamCompat()
    store.learn("openai", "o3-mini", "drop_temperature")
    store.learn("openai", "o3-mini", "rename_max_tokens")

    data = json.loads(_store_file().read_text())
    assert sorted(data["openai:o3-mini"]) == ["drop_temperature", "rename_max_tokens"]


def test_merge_on_write_unions_concurrent_learnings():
    """Two independent store instances (stand-in for two neo processes) each
    learn something; neither clobbers the other's on-disk entry."""
    a = _ModelParamCompat()
    b = _ModelParamCompat()
    a.learn("openai", "model-a", "drop_temperature")
    b.learn("azure", "model-b", "rename_max_tokens")  # b loaded before a wrote

    disk = json.loads(_store_file().read_text())
    assert disk["openai:model-a"] == ["drop_temperature"]
    assert disk["azure:model-b"] == ["rename_max_tokens"]


def test_reloads_when_home_changes(monkeypatch, tmp_path):
    """The in-memory cache reloads when the resolved path changes, so an
    isolated home never sees another home's state."""
    home1 = tmp_path / "h1"
    home1.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home1))
    store = _ModelParamCompat()
    store.learn("openai", "o3-mini", "drop_temperature")
    assert store.has("openai", "o3-mini", "drop_temperature")

    home2 = tmp_path / "h2"
    home2.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home2))
    assert not store.has("openai", "o3-mini", "drop_temperature")  # fresh home


def test_corrupt_file_is_tolerated():
    path = _store_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not valid json")

    store = _ModelParamCompat()
    assert not store.has("openai", "o3-mini", "drop_temperature")  # no raise
    # and it recovers by overwriting on the next learn
    store.learn("openai", "o3-mini", "drop_temperature")
    assert json.loads(path.read_text()) == {"openai:o3-mini": ["drop_temperature"]}


@pytest.mark.parametrize("bad", [
    "null",                        # valid JSON, not a dict
    "[]",                          # a list
    "42",                          # a scalar
    '{"openai:gpt": "notalist"}',  # value isn't a list
    '{"openai:gpt": [[1, 2]]}',    # unhashable flag element
    '{"openai:gpt": [1, 2]}',      # non-str flag elements
])
def test_wrong_shape_json_is_tolerated(bad):
    """Valid JSON of the wrong shape must NOT crash inference — it degrades to
    empty (regression: `_read`'s structural coercion once ran outside the
    try/except and raised AttributeError/TypeError on these inputs)."""
    path = _store_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(bad)

    store = _ModelParamCompat()
    assert not store.has("openai", "gpt", "drop_temperature")  # no raise
    # recovers cleanly on the next learn
    store.learn("openai", "gpt", "drop_temperature")
    assert store.has("openai", "gpt", "drop_temperature")


def test_home_resolution_failure_is_tolerated(monkeypatch):
    """If Path.home() itself raises, has()/learn() degrade silently rather than
    breaking the inference path."""
    def _boom():
        raise RuntimeError("no home")

    monkeypatch.setattr(Path, "home", staticmethod(_boom))
    store = _ModelParamCompat()
    assert not store.has("openai", "o3-mini", "drop_temperature")  # no raise
    store.learn("openai", "o3-mini", "drop_temperature")  # no raise


def test_persistence_failure_is_best_effort(monkeypatch):
    """If the file can't be written, learning still holds in memory and never
    raises — inference must not break on a persistence failure."""
    store = _ModelParamCompat()

    def _boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("neo.adapters.tempfile.mkstemp", _boom)
    store.learn("openai", "o3-mini", "drop_temperature")  # must not raise
    assert store.has("openai", "o3-mini", "drop_temperature")  # in-memory holds
