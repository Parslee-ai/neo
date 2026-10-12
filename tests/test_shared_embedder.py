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


def test_the_model_lock_serves_callers_in_arrival_order():
    """A thread that releases and immediately re-acquires must not jump a
    waiter: on Linux `threading.Lock` let one run's batch hold the model end
    to end while another run's query timed out."""
    import threading
    import time

    lock = store._FairLock()
    order = []
    started = threading.Event()

    def greedy():
        for i in range(20):
            with lock:
                if i == 0:
                    started.set()
                order.append("a")
                time.sleep(0.01)

    def waiter():
        started.wait()
        with lock:
            order.append("b")

    a, b = threading.Thread(target=greedy), threading.Thread(target=waiter)
    a.start()
    b.start()
    a.join()
    b.join()
    assert order.index("b") <= 2
