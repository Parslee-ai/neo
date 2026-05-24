"""Tests for the async synthesis observer (neo.memory.observer).

Lifecycle smoke tests use a stubbed daemon process so we don't actually
fork a Python interpreter under pytest. The Observer class itself is
exercised directly with monkeypatched sleeps.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def project_id(monkeypatch):
    """Pin the project_id so tests don't depend on the host's git remote."""
    fixed = "test1234abcdef00"
    monkeypatch.setattr(
        "neo.memory.observer.detect_org_and_project",
        lambda _root: ("testorg", fixed),
    )
    return fixed


def test_pid_helpers(project_id, _isolated_home):
    from neo.memory.observer import _pid_file, _read_pid, _sessions_dir

    assert _read_pid(project_id) is None
    _sessions_dir(project_id).mkdir(parents=True, exist_ok=True)
    _pid_file(project_id).write_text("12345")
    assert _read_pid(project_id) == 12345


def test_pid_alive_for_current_process():
    from neo.memory.observer import _pid_alive
    assert _pid_alive(os.getpid()) is True


def test_pid_alive_for_dead_pid():
    from neo.memory.observer import _pid_alive
    # PID 1 is always init/launchd — alive. Use an arbitrarily large PID
    # that is overwhelmingly likely to be dead.
    assert _pid_alive(2**31 - 1) is False


def test_observer_config_defaults(monkeypatch):
    monkeypatch.delenv("NEO_OBSERVER_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("NEO_OBSERVER_COOLDOWN", raising=False)
    monkeypatch.delenv("NEO_OBSERVER_IDLE_SECONDS", raising=False)
    from neo.memory.observer import ObserverConfig
    cfg = ObserverConfig.from_env()
    assert cfg.interval_seconds == 300.0
    assert cfg.cooldown_seconds == 60.0
    assert cfg.idle_seconds == 1800.0


def test_observer_config_env_overrides(monkeypatch):
    monkeypatch.setenv("NEO_OBSERVER_INTERVAL_SECONDS", "10")
    monkeypatch.setenv("NEO_OBSERVER_COOLDOWN", "2")
    monkeypatch.setenv("NEO_OBSERVER_IDLE_SECONDS", "60")
    from neo.memory.observer import ObserverConfig
    cfg = ObserverConfig.from_env()
    assert cfg.interval_seconds == 10.0
    assert cfg.cooldown_seconds == 2.0
    assert cfg.idle_seconds == 60.0


def test_observer_config_rejects_garbage(monkeypatch):
    monkeypatch.setenv("NEO_OBSERVER_INTERVAL_SECONDS", "not-a-number")
    monkeypatch.setenv("NEO_OBSERVER_COOLDOWN", "-5")  # non-positive
    from neo.memory.observer import ObserverConfig
    cfg = ObserverConfig.from_env()
    assert cfg.interval_seconds == 300.0  # falls back to default
    assert cfg.cooldown_seconds == 60.0


class TestObserverStatus:
    """status() before any observer is running."""

    def test_status_when_no_observer(self, project_id):
        from neo.memory.observer import observer_status
        result = observer_status("/some/path")
        assert result["status"] == "not_running"
        assert result["pid"] is None

    def test_status_when_pid_file_stale(self, project_id, _isolated_home):
        from neo.memory.observer import _pid_file, _sessions_dir, observer_status

        _sessions_dir(project_id).mkdir(parents=True, exist_ok=True)
        _pid_file(project_id).write_text(str(2**31 - 1))  # dead PID

        result = observer_status("/some/path")
        assert result["status"] == "stale"


class TestObserverStopAndKick:
    def test_stop_when_not_running(self, project_id):
        from neo.memory.observer import stop_observer
        result = stop_observer("/some/path")
        assert result["status"] == "not_running"

    def test_kick_when_not_running(self, project_id):
        from neo.memory.observer import kick_observer
        result = kick_observer("/some/path")
        assert result["status"] == "not_running"

    def test_stop_clears_stale_pid_file(self, project_id, _isolated_home):
        from neo.memory.observer import _pid_file, _sessions_dir, stop_observer

        _sessions_dir(project_id).mkdir(parents=True, exist_ok=True)
        _pid_file(project_id).write_text(str(2**31 - 1))

        result = stop_observer("/some/path")
        assert result["status"] == "not_running"
        assert not _pid_file(project_id).exists()


class TestObserverCycleUnit:
    """Drive Observer._cycle directly — no daemon, no signals."""

    def test_cycle_calls_synthesize_reviews(self, project_id, _isolated_home, monkeypatch):
        from neo.memory.observer import Observer

        observer = Observer(codebase_root="/some/path")

        call_count = {"n": 0}

        class _FakeStore:
            def __init__(self, **kwargs):
                pass

            def initialize(self):
                pass

            def synthesize_reviews(self):
                call_count["n"] += 1
                return 3

        monkeypatch.setattr("neo.memory.store.FactStore", _FakeStore)
        observer._cycle()

        assert call_count["n"] == 1
        assert observer._last_analysis_epoch > 0
        # last_analysis file written
        from neo.memory.observer import _last_analysis_file
        assert _last_analysis_file(project_id).exists()

    def test_cycle_swallows_errors(self, project_id, _isolated_home, monkeypatch):
        from neo.memory.observer import Observer

        observer = Observer(codebase_root="/some/path")

        class _BrokenStore:
            def __init__(self, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr("neo.memory.store.FactStore", _BrokenStore)
        # Should not raise
        observer._cycle()


class TestCooldownAndIdle:
    def test_cooldown_blocks_until_elapsed(self, project_id, _isolated_home):
        from neo.memory.observer import Observer, ObserverConfig

        observer = Observer(
            codebase_root="/some/path",
            config=ObserverConfig(cooldown_seconds=10.0),
        )
        observer._last_analysis_epoch = time.time()
        assert observer._cooldown_ok() is False

        observer._last_analysis_epoch = time.time() - 11
        assert observer._cooldown_ok() is True

    def test_idle_returns_false_when_no_metrics_file(self, project_id, _isolated_home):
        from neo.memory.observer import Observer
        observer = Observer(codebase_root="/some/path")
        assert observer._idle_too_long() is False

    def test_idle_returns_true_when_metrics_file_stale(
        self, project_id, _isolated_home,
    ):
        from neo.memory.observer import Observer, ObserverConfig, _metrics_file

        observer = Observer(
            codebase_root="/some/path",
            config=ObserverConfig(idle_seconds=10.0),
        )
        # Backdate the observer's _start_time too, since _idle_too_long uses
        # max(mtime, start_time) so a newly-started observer gets a grace
        # window even against a stale file.
        observer._start_time = time.time() - 100

        mf = _metrics_file()
        mf.parent.mkdir(parents=True, exist_ok=True)
        mf.write_text("{}\n")
        old = time.time() - 100
        os.utime(mf, (old, old))

        assert observer._idle_too_long() is True

    def test_idle_returns_false_during_startup_grace_window(
        self, project_id, _isolated_home,
    ):
        """A freshly-started observer must not self-exit on a stale metrics file."""
        from neo.memory.observer import Observer, ObserverConfig, _metrics_file

        observer = Observer(
            codebase_root="/some/path",
            config=ObserverConfig(idle_seconds=10.0),
        )

        mf = _metrics_file()
        mf.parent.mkdir(parents=True, exist_ok=True)
        mf.write_text("{}\n")
        old = time.time() - 9999
        os.utime(mf, (old, old))

        # _start_time is now() so the reference is now → not idle yet
        assert observer._idle_too_long() is False


class TestStartObserverGuards:
    def test_start_returns_error_without_project_id(self, monkeypatch):
        monkeypatch.setattr(
            "neo.memory.observer.detect_org_and_project",
            lambda _root: ("unknown", ""),
        )
        from neo.memory.observer import start_observer
        result = start_observer("/nowhere")
        assert result["status"] == "error"

    def test_start_returns_already_running_when_pid_alive(
        self, project_id, _isolated_home,
    ):
        from neo.memory.observer import (
            _pid_file,
            _sessions_dir,
            start_observer,
        )

        _sessions_dir(project_id).mkdir(parents=True, exist_ok=True)
        # Use OUR own PID — guaranteed alive
        _pid_file(project_id).write_text(str(os.getpid()))

        result = start_observer("/some/path")
        assert result["status"] == "already_running"
        assert result["pid"] == os.getpid()
