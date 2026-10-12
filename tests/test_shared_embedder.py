"""One embedding model per process (#211): it is ~1.3 GB resident, and one
reasoning run used to build it for the gatherer and again for the engine."""

import pytest

import neo.memory.store as store


@pytest.fixture
def counted(monkeypatch):
    built = []
    monkeypatch.setattr(store, "FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(store, "_EMBEDDERS", {})
    monkeypatch.setattr(store, "_build_embedder",
                        lambda name, cache_dir, prefix: built.append(name) or object())
    return built


def test_every_caller_shares_one_model(counted, tmp_path):
    a = store.build_resilient_embedder(cache_dir=tmp_path, log_prefix="FactStore")
    b = store.build_resilient_embedder(cache_dir=tmp_path, log_prefix="ProjectIndex")
    assert a is b and len(counted) == 1


def test_a_different_model_or_cache_is_its_own_model(counted, tmp_path):
    store.build_resilient_embedder(cache_dir=tmp_path)
    store.build_resilient_embedder(model_name="other/model", cache_dir=tmp_path)
    store.build_resilient_embedder(cache_dir=tmp_path / "elsewhere")
    assert len(counted) == 3


def test_a_failed_build_is_not_retried_by_every_caller(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(store, "FASTEMBED_AVAILABLE", True)
    monkeypatch.setattr(store, "_EMBEDDERS", {})
    monkeypatch.setattr(store, "_build_embedder",
                        lambda *a: calls.append(1) and None)
    assert store.build_resilient_embedder(cache_dir=tmp_path) is None
    assert store.build_resilient_embedder(cache_dir=tmp_path) is None
    assert len(calls) == 1
