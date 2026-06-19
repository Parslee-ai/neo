"""Tests for the CAR-supervised async synthesis observer.

Lifecycle tests stub a fake ``car_runtime`` module with just the
``agents_*`` functions we depend on, so the tests don't require
car-server to be running or the car-runtime wheel to be installed.

The Observer class itself (the daemon body) is exercised directly.
"""

from __future__ import annotations

import json
import sys
import time
import types
from unittest.mock import MagicMock

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


class TestCarVersionFloor:
    def test_parse_version(self):
        from neo.memory.observer import _parse_version
        assert _parse_version("0.27.0") == (0, 27, 0)
        assert _parse_version("0.18.0rc1") == (0, 18, 0)
        assert _parse_version("0.16.1") == (0, 16, 1)
        assert _parse_version(None) is None
        assert _parse_version("garbage") is None

    def test_old_version_rejected(self, fake_car, monkeypatch):
        # agents_upsert present (0.16.x/0.17.0) but below the 0.18.0 floor.
        from neo.memory import observer as obs
        monkeypatch.setattr(obs, "_installed_car_version", lambda: "0.17.0")
        with pytest.raises(RuntimeError, match="too old"):
            obs._require_car_runtime()

    def test_current_version_accepted(self, fake_car, monkeypatch):
        from neo.memory import observer as obs
        monkeypatch.setattr(obs, "_installed_car_version", lambda: "0.27.0")
        assert obs._require_car_runtime() is fake_car

    def test_unknown_version_allowed(self, fake_car, monkeypatch):
        # Lenient about unknown (vendored build), strict about known-too-old.
        from neo.memory import observer as obs
        monkeypatch.setattr(obs, "_installed_car_version", lambda: None)
        assert obs._require_car_runtime() is fake_car


class _FakeProc:
    """Minimal psutil.Process stand-in for orphan-detection tests."""

    def __init__(self, pid, ppid, cmdline, parent_alive=True):
        self.pid = pid
        self.info = {"pid": pid, "ppid": ppid, "cmdline": cmdline}
        self._parent_alive = parent_alive

    def parent(self):
        return object() if self._parent_alive else None


def _install_fake_psutil(monkeypatch, procs):
    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    fake = types.SimpleNamespace(
        process_iter=lambda attrs=None: list(procs),
        NoSuchProcess=NoSuchProcess,
        AccessDenied=AccessDenied,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake)
    return fake


def _obs_cmd(root):
    return ["/usr/bin/python", "-m", "neo.memory.observer", "--daemon", "--cwd", root]


class TestOrphanDetectionPsutil:
    """Cross-platform path (psutil): POSIX ppid==1 AND Windows parent-gone."""

    def test_posix_orphan_ppid1(self, monkeypatch, tmp_path):
        root = str(tmp_path)
        _install_fake_psutil(monkeypatch, [_FakeProc(100, 1, _obs_cmd(root), parent_alive=True)])
        from neo.memory.observer import _find_orphan_observers
        assert _find_orphan_observers(root) == [100]

    def test_windows_orphan_parent_gone(self, monkeypatch, tmp_path):
        # No ppid==1 reparenting on Windows; the launching car-server is just gone.
        root = str(tmp_path)
        _install_fake_psutil(monkeypatch, [_FakeProc(200, 4321, _obs_cmd(root), parent_alive=False)])
        from neo.memory.observer import _find_orphan_observers
        assert _find_orphan_observers(root) == [200]

    def test_supervised_not_flagged(self, monkeypatch, tmp_path):
        root = str(tmp_path)
        _install_fake_psutil(monkeypatch, [_FakeProc(300, 20656, _obs_cmd(root), parent_alive=True)])
        from neo.memory.observer import _find_orphan_observers
        assert _find_orphan_observers(root) == []

    def test_ignores_other_project_and_nonobserver(self, monkeypatch, tmp_path):
        root = str(tmp_path)
        procs = [
            _FakeProc(1, 1, _obs_cmd("/some/other/project"), parent_alive=False),
            _FakeProc(2, 1, ["/usr/bin/python", "-m", "http.server"], parent_alive=False),
        ]
        _install_fake_psutil(monkeypatch, procs)
        from neo.memory.observer import _find_orphan_observers
        assert _find_orphan_observers(root) == []


class TestOrphanDetectionPs:
    """POSIX `ps` fallback path (psutil absent)."""

    def _fake_ps(self, monkeypatch, text):
        import neo.memory.observer as obs
        monkeypatch.setitem(sys.modules, "psutil", None)  # force ImportError -> ps fallback
        monkeypatch.setattr(
            obs.subprocess, "run",
            lambda *a, **k: types.SimpleNamespace(stdout=text),
        )

    def test_finds_launchd_orphan_for_this_root(self, monkeypatch, tmp_path):
        root = str(tmp_path)
        text = (
            f"100     1 /usr/bin/python -m neo.memory.observer --daemon --cwd {root}\n"       # orphan
            f"200 20656 /usr/bin/python -m neo.memory.observer --daemon --cwd {root}\n"       # supervised
            f"300     1 /usr/bin/python -m neo.memory.observer --daemon --cwd /other/proj\n"  # other project
            f"400     1 /usr/bin/python -m http.server\n"                                     # not observer
        )
        self._fake_ps(monkeypatch, text)
        from neo.memory.observer import _find_orphan_observers
        assert _find_orphan_observers(root) == [100]

    def test_no_orphan_when_only_supervised(self, monkeypatch, tmp_path):
        root = str(tmp_path)
        self._fake_ps(monkeypatch, f"200 20656 python -m neo.memory.observer --daemon --cwd {root}\n")
        from neo.memory.observer import _find_orphan_observers
        assert _find_orphan_observers(root) == []

    def test_ps_failure_is_safe(self, monkeypatch, tmp_path):
        import neo.memory.observer as obs
        monkeypatch.setitem(sys.modules, "psutil", None)

        def boom(*a, **k):
            raise OSError("no ps here")

        monkeypatch.setattr(obs.subprocess, "run", boom)
        assert obs._find_orphan_observers(str(tmp_path)) == []


class TestOrphanStatus:
    def test_status_includes_orphans(self, monkeypatch):
        import neo.memory.observer as obs
        monkeypatch.setattr(obs, "_resolve_project_id", lambda root: "abc123def456")
        monkeypatch.setattr(obs, "_require_car_runtime", lambda: object())
        monkeypatch.setattr(
            obs, "_find_managed_agent",
            lambda car, aid: {"status": "running", "pid": 555, "restart_count": 0},
        )
        monkeypatch.setattr(obs, "_find_orphan_observers", lambda root: [777])
        st = obs.observer_status("/repo")
        assert st["status"] == "running"
        assert st["orphans"] == [777]


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
        # Ingest bounds are committed constants, not env-toggleable.
        assert cfg.ingest_budget == 8
        assert cfg.ingest_deadline_seconds == 120.0

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

    def test_zero_budget_is_defensive_noop(self):
        # Not a feature toggle (budget is a committed constant) — just a guard
        # so a 0 can't waste an adapter build.
        from neo.memory.observer import Observer, ObserverConfig
        obs = Observer(codebase_root="/tmp/x", config=ObserverConfig(ingest_budget=0))
        assert obs._ingest_transcripts(store=None) == 0

    def test_ingest_transcripts_swallows_errors(self, monkeypatch):
        # Force an error inside the ingest path; the cycle must not crash.
        from neo.memory.observer import Observer, ObserverConfig
        monkeypatch.setattr("neo.config.NeoConfig.load",
                            staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))))
        obs = Observer(codebase_root="/tmp/x", config=ObserverConfig(ingest_budget=5))
        assert obs._ingest_transcripts(store=None) == 0


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
            [{"id": aid, "pid": 1234, "status": "running"}]
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
            [{"id": aid, "pid": 9001, "status": "running"}]
        )
        fake_car.agents_stop.return_value = json.dumps({"id": aid, "pid": 9001})

        from neo.memory.observer import stop_observer
        result = stop_observer("/some/path")
        assert result["status"] == "stopped"
        fake_car.agents_stop.assert_called_once_with(aid)

    def test_stop_when_already_stopped(self, fake_project_id, fake_car):
        """Spec exists but `status` says stopped — should NOT call agents_stop."""
        from neo.memory.observer import _agent_id
        aid = _agent_id(fake_project_id)
        fake_car.agents_list.return_value = json.dumps(
            [{"id": aid, "pid": None, "status": "stopped"}]
        )
        from neo.memory.observer import stop_observer
        result = stop_observer("/some/path")
        assert result["status"] == "not_running"
        fake_car.agents_stop.assert_not_called()


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
            [{"id": aid, "pid": 1, "status": "running"}]
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
            "id": aid, "pid": 12345, "status": "running", "restart_count": 2,
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
            [{"id": aid, "pid": None, "status": "stopped", "restart_count": 0,
              "last_exit_code": 0}]
        )
        from neo.memory.observer import observer_status
        result = observer_status("/some/path")
        # We now surface CAR's raw status verbatim (running|stopped|backoff|...)
        assert result["status"] == "stopped"
        assert result["pid"] is None

    def test_status_when_in_backoff(self, fake_project_id, fake_car):
        """`backoff` is a valid CAR status — restart-loop diagnosis."""
        from neo.memory.observer import _agent_id
        aid = _agent_id(fake_project_id)
        fake_car.agents_list.return_value = json.dumps(
            [{"id": aid, "pid": None, "status": "backoff", "restart_count": 7,
              "last_exit_code": 1}]
        )
        from neo.memory.observer import observer_status
        result = observer_status("/some/path")
        assert result["status"] == "backoff"
        assert result["restart_count"] == 7


class TestObserverCycleUnit:
    """Drive Observer._cycle directly — no daemon, no signals."""

    def test_cycle_calls_synthesize_reviews(self, fake_project_id, monkeypatch):
        from neo.memory.observer import Observer, ObserverConfig
        # ingest_budget=0 so this isolates synthesis (otherwise the real ingest
        # path runs and only passes because it swallows its own failure).
        observer = Observer(codebase_root="/some/path",
                            config=ObserverConfig(ingest_budget=0))

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
        from neo.memory.observer import Observer, ObserverConfig
        observer = Observer(codebase_root="/some/path",
                            config=ObserverConfig(ingest_budget=0))

        class _BrokenStore:
            def __init__(self, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr("neo.memory.store.FactStore", _BrokenStore)
        observer._cycle()  # must not raise

    def test_cycle_ingests_and_records(self, fake_project_id, monkeypatch, tmp_path):
        """Positive end-to-end: ingest runs inside _cycle and admits a fact."""
        import neo.memory.transcript as tr
        from neo.memory.observer import Observer, ObserverConfig

        monkeypatch.setattr(tr, "SESSIONS_DIR", tmp_path / "sessions")
        admitted = []

        class _FakeStore:
            project_id = "testproj"

            def __init__(self, **kwargs):
                pass

            def initialize(self):
                pass

            def synthesize_reviews(self):
                return 0

            def add_fact(self, **kw):
                admitted.append(kw)
                return object()

        class _Cfg:
            provider, model, api_key, base_url = "openai", "m", "k", None

        class _Adapter:
            def generate(self, messages, **kw):
                p = messages[0]["content"]
                if '"lessons"' in p:
                    return ('{"lessons":[{"kind":"pattern","subject":"s","body":"b",'
                            '"domain":"testing","evidence_span":"hello world"}]}')
                return '{"keep": true}'

        ep = tr.Episode(session_id="s", anchor_uuid="u1", last_uuid="u1", timestamp="t",
                        ask="why", assistant_text=["hello world"], tools=[])

        monkeypatch.setattr("neo.memory.store.FactStore", _FakeStore)
        monkeypatch.setattr("neo.config.NeoConfig.load", staticmethod(lambda *a, **k: _Cfg()))
        monkeypatch.setattr("neo.adapters.create_adapter", lambda *a, **k: _Adapter())
        monkeypatch.setattr(tr, "collect_episodes", lambda root: [ep])

        observer = Observer(codebase_root="/p", config=ObserverConfig(ingest_budget=5))
        observer._cycle()

        assert len(admitted) == 1                       # a lesson was admitted
        assert admitted[0]["domain"] == "testing"
        assert observer._last_ingest_error is None
        assert "1 mined" in observer._recent_cycles[-1]["text"]


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
