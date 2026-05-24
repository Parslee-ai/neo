"""Tests for the CAR-supervised async synthesis observer.

Lifecycle tests stub a fake ``car_runtime`` module with just the
``agents_*`` functions we depend on, so the tests don't require
car-server to be running or the car-runtime wheel to be installed.

The Observer class itself (the daemon body) is exercised directly.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_project_id(monkeypatch):
    """Pin the project_id so tests don't depend on the host's git remote."""
    fixed = "test1234abcdef00"
    monkeypatch.setattr(
        "neo.memory.observer.detect_org_and_project",
        lambda _root: ("testorg", fixed),
    )
    return fixed


@pytest.fixture
def fake_car(monkeypatch):
    """Inject a fake car_runtime module with stubbable agents_* methods."""
    car = types.SimpleNamespace(
        agents_upsert=MagicMock(return_value="{}"),
        agents_start=MagicMock(),
        agents_stop=MagicMock(),
        agents_restart=MagicMock(),
        agents_list=MagicMock(return_value="[]"),
        # Sentinel attr that _require_car_runtime checks for to gate
        # the version requirement.
        __spec__=None,
    )
    monkeypatch.setitem(sys.modules, "car_runtime", car)
    return car


class TestRequireCarRuntime:
    def test_missing_module_raises_actionable_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "car_runtime", None)
        from neo.memory.observer import _require_car_runtime
        with pytest.raises(RuntimeError, match="car-runtime"):
            _require_car_runtime()

    def test_too_old_module_raises_actionable_error(self, monkeypatch):
        # No agents_upsert attr → pre-0.16 install.
        old = types.SimpleNamespace(infer=lambda *a, **kw: None)
        monkeypatch.setitem(sys.modules, "car_runtime", old)
        from neo.memory.observer import _require_car_runtime
        with pytest.raises(RuntimeError, match="agents_\\*"):
            _require_car_runtime()


class TestSpecBuilding:
    def test_spec_uses_current_python(self, fake_project_id):
        from neo.memory.observer import _build_spec
        spec = _build_spec(fake_project_id, "/repo/root")
        assert spec["command"] == sys.executable
        assert spec["args"] == ["-m", "neo.memory.observer", "--daemon",
                                "--cwd", "/repo/root"]
        assert spec["cwd"] == "/repo/root"
        assert spec["restart"] == "on_failure"
        assert spec["auto_start"] is True

    def test_spec_id_is_filename_safe(self, fake_project_id):
        from neo.memory.observer import _agent_id, _build_spec
        spec = _build_spec(fake_project_id, "/r")
        # Filename-safe: alphanumerics + hyphens only
        assert spec["id"] == _agent_id(fake_project_id)
        for ch in spec["id"]:
            assert ch.isalnum() or ch in "-_"

    def test_spec_forwards_only_neo_env(self, fake_project_id, monkeypatch):
        monkeypatch.setenv("NEO_OBSERVER_INTERVAL_SECONDS", "30")
        monkeypatch.setenv("NEO_PROFILE", "minimal")
        monkeypatch.setenv("USER_SECRET", "hunter2")
        from neo.memory.observer import _build_spec
        spec = _build_spec(fake_project_id, "/r")
        assert spec["env"]["NEO_OBSERVER_INTERVAL_SECONDS"] == "30"
        assert spec["env"]["NEO_PROFILE"] == "minimal"
        assert "USER_SECRET" not in spec["env"]


class TestObserverConfig:
    def test_defaults(self, monkeypatch):
        for k in ("NEO_OBSERVER_INTERVAL_SECONDS", "NEO_OBSERVER_COOLDOWN"):
            monkeypatch.delenv(k, raising=False)
        from neo.memory.observer import ObserverConfig
        cfg = ObserverConfig.from_env()
        assert cfg.interval_seconds == 300.0
        assert cfg.cooldown_seconds == 60.0

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("NEO_OBSERVER_INTERVAL_SECONDS", "15")
        monkeypatch.setenv("NEO_OBSERVER_COOLDOWN", "3")
        from neo.memory.observer import ObserverConfig
        cfg = ObserverConfig.from_env()
        assert cfg.interval_seconds == 15.0
        assert cfg.cooldown_seconds == 3.0

    def test_rejects_garbage(self, monkeypatch):
        monkeypatch.setenv("NEO_OBSERVER_INTERVAL_SECONDS", "junk")
        monkeypatch.setenv("NEO_OBSERVER_COOLDOWN", "-5")
        from neo.memory.observer import ObserverConfig
        cfg = ObserverConfig.from_env()
        assert cfg.interval_seconds == 300.0
        assert cfg.cooldown_seconds == 60.0


class TestStartObserver:
    def test_start_with_no_project_id_returns_error(self, fake_car, monkeypatch):
        monkeypatch.setattr(
            "neo.memory.observer.detect_org_and_project",
            lambda _root: ("unknown", ""),
        )
        from neo.memory.observer import start_observer
        result = start_observer("/nowhere")
        assert result["status"] == "error"
        # Did NOT touch CAR if project_id resolution failed
        fake_car.agents_upsert.assert_not_called()

    def test_start_calls_upsert_then_start(self, fake_project_id, fake_car):
        fake_car.agents_list.return_value = "[]"
        fake_car.agents_start.return_value = json.dumps({"id": "x", "pid": 9001})

        from neo.memory.observer import start_observer
        result = start_observer("/some/path")

        assert result["status"] == "started"
        assert result["pid"] == 9001
        fake_car.agents_upsert.assert_called_once()
        fake_car.agents_start.assert_called_once()
        # The spec passed to agents_upsert must be valid JSON with our fields
        spec_arg = fake_car.agents_upsert.call_args[0][0]
        spec = json.loads(spec_arg)
        assert spec["id"].startswith("neo-observer-")
        assert spec["restart"] == "on_failure"

    def test_start_when_already_running(self, fake_project_id, fake_car):
        from neo.memory.observer import _agent_id
        aid = _agent_id(fake_project_id)
        fake_car.agents_list.return_value = json.dumps(
            [{"id": aid, "pid": 1234, "running": True}]
        )

        from neo.memory.observer import start_observer
        result = start_observer("/some/path")
        assert result["status"] == "already_running"
        assert result["pid"] == 1234
        # Spec gets upserted (idempotent); but start is NOT called when already running
        fake_car.agents_start.assert_not_called()


class TestStopObserver:
    def test_stop_when_not_registered(self, fake_project_id, fake_car):
        fake_car.agents_list.return_value = "[]"
        from neo.memory.observer import stop_observer
        result = stop_observer("/some/path")
        assert result["status"] == "not_running"
        fake_car.agents_stop.assert_not_called()

    def test_stop_when_running(self, fake_project_id, fake_car):
        from neo.memory.observer import _agent_id
        aid = _agent_id(fake_project_id)
        fake_car.agents_list.return_value = json.dumps(
            [{"id": aid, "pid": 9001, "running": True}]
        )
        fake_car.agents_stop.return_value = json.dumps({"id": aid, "pid": 9001})

        from neo.memory.observer import stop_observer
        result = stop_observer("/some/path")
        assert result["status"] == "stopped"
        fake_car.agents_stop.assert_called_once_with(aid)


class TestKickObserver:
    def test_kick_when_not_registered(self, fake_project_id, fake_car):
        fake_car.agents_list.return_value = "[]"
        from neo.memory.observer import kick_observer
        result = kick_observer("/some/path")
        assert result["status"] == "not_running"
        fake_car.agents_restart.assert_not_called()

    def test_kick_maps_to_restart(self, fake_project_id, fake_car):
        from neo.memory.observer import _agent_id
        aid = _agent_id(fake_project_id)
        fake_car.agents_list.return_value = json.dumps(
            [{"id": aid, "pid": 1, "running": True}]
        )
        fake_car.agents_restart.return_value = json.dumps({"id": aid, "pid": 2})

        from neo.memory.observer import kick_observer
        result = kick_observer("/some/path")
        assert result["status"] == "kicked"
        assert result["pid"] == 2
        fake_car.agents_restart.assert_called_once_with(aid)


class TestObserverStatus:
    def test_status_when_not_registered(self, fake_project_id, fake_car):
        fake_car.agents_list.return_value = "[]"
        from neo.memory.observer import observer_status
        result = observer_status("/some/path")
        assert result["status"] == "not_running"

    def test_status_when_running(self, fake_project_id, fake_car):
        from neo.memory.observer import _agent_id
        aid = _agent_id(fake_project_id)
        fake_car.agents_list.return_value = json.dumps([{
            "id": aid, "pid": 12345, "running": True, "restart_count": 2,
        }])
        from neo.memory.observer import observer_status
        result = observer_status("/some/path")
        assert result["status"] == "running"
        assert result["pid"] == 12345
        assert result["restart_count"] == 2

    def test_status_when_registered_but_stopped(self, fake_project_id, fake_car):
        from neo.memory.observer import _agent_id
        aid = _agent_id(fake_project_id)
        fake_car.agents_list.return_value = json.dumps(
            [{"id": aid, "pid": 0, "running": False, "restart_count": 0}]
        )
        from neo.memory.observer import observer_status
        result = observer_status("/some/path")
        assert result["status"] == "stopped"


class TestObserverCycleUnit:
    """Drive Observer._cycle directly — no daemon, no signals."""

    def test_cycle_calls_synthesize_reviews(self, fake_project_id, monkeypatch):
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

    def test_cycle_swallows_errors(self, fake_project_id, monkeypatch):
        from neo.memory.observer import Observer
        observer = Observer(codebase_root="/some/path")

        class _BrokenStore:
            def __init__(self, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr("neo.memory.store.FactStore", _BrokenStore)
        observer._cycle()  # must not raise


class TestCooldown:
    def test_cooldown_blocks_until_elapsed(self, fake_project_id):
        from neo.memory.observer import Observer, ObserverConfig
        observer = Observer(
            codebase_root="/some/path",
            config=ObserverConfig(cooldown_seconds=10.0),
        )
        observer._last_analysis_epoch = time.time()
        assert observer._cooldown_ok() is False

        observer._last_analysis_epoch = time.time() - 11
        assert observer._cooldown_ok() is True

    def test_first_run_passes_cooldown(self, fake_project_id):
        """``_last_analysis_epoch`` starts at 0, so the first call should
        always be allowed."""
        from neo.memory.observer import Observer
        observer = Observer(codebase_root="/some/path")
        assert observer._cooldown_ok() is True
