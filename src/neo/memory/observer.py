"""Async out-of-band synthesis observer, supervised by car-server.

A per-project background process that runs REVIEW→PATTERN/FAILURE
synthesis on a wall-clock cadence. Lifecycle is owned by CAR's
agent supervisor:

  - ``agents_upsert`` registers the spec under ``~/.car/agents.json``
  - ``agents_start`` spawns the child
    (``python -m neo.memory.observer --daemon --cwd <root>``)
  - Supervisor handles restart-on-failure, clean SIGTERM shutdown, log
    redirection to ``~/.car/logs/<id>.{stdout,stderr}.log``, and auto-
    start at daemon boot when ``auto_start=True``.

Hard dependency on ``car-runtime>=0.18.0`` (the ``agents_*`` lifecycle
API landed in 0.16.1, but earlier bindings grab the supervisor manifest
lock in-process and collide with a running car-server — see
Parslee-ai/car-releases#54). The car-server daemon must be running; the
CAR bindings will auto-spawn it via ``CAR_AUTOSTART`` if reachable
on the default ``ws://127.0.0.1:9100``.

Tunables (env, read by the daemon child):

    NEO_OBSERVER_INTERVAL_SECONDS  — wake cadence (default 300)
    NEO_OBSERVER_COOLDOWN          — min seconds between analyses
                                     (default 60). With CAR-managed
                                     lifecycle there's no SIGUSR1
                                     kick; the cooldown still bounds
                                     the rate at which restarts re-
                                     trigger synthesis.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

from neo.memory.scope import detect_org_and_project

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CAR runtime adapter (lazy-imported so non-observer neo commands work
# without the car-runtime dependency installed)
# ---------------------------------------------------------------------------


_CAR_REQUIRED_MSG = (
    "Observer requires car-runtime >= 0.18.0 (where the agents_* "
    "calls route to the running car-server daemon instead of "
    "colliding with its supervisor lock — see Parslee-ai/car-releases#54). "
    "Install: `pip install neo[car]`, then ensure car-server is running "
    "(CarHost.app or `python -m car_runtime.server`)."
)


def _require_car_runtime():
    """Import car_runtime and verify it has the agent-supervisor API.

    Raises ``RuntimeError`` with an actionable message when missing or
    too old, instead of a cryptic ImportError/AttributeError at the
    call site.
    """
    try:
        import car_runtime
    except ImportError as e:
        raise RuntimeError(_CAR_REQUIRED_MSG) from e

    if not hasattr(car_runtime, "agents_upsert"):
        raise RuntimeError(_CAR_REQUIRED_MSG)

    return car_runtime


def _resolve_project_id(codebase_root: Optional[str]) -> str:
    _, project_id = detect_org_and_project(codebase_root or os.getcwd())
    return project_id


def _agent_id(project_id: str) -> str:
    """Filename-safe ID for the supervisor. Short prefix keeps logs
    grep-able when multiple projects are active."""
    return f"neo-observer-{project_id[:12]}"


def _build_spec(project_id: str, codebase_root: str) -> dict:
    """Spec for ``agents_upsert``. ``command`` must be an absolute
    interpreter path per the CAR contract; ``args`` invokes our daemon
    module.

    Picks ``sys.executable`` so the supervisor uses the *same* Python
    that neo itself is running on — important when the user installed
    neo in a virtualenv.
    """
    return {
        "id": _agent_id(project_id),
        "name": f"Neo observer ({project_id[:8]})",
        "command": sys.executable,
        "args": ["-m", "neo.memory.observer", "--daemon", "--cwd", codebase_root],
        "cwd": codebase_root,
        "env": {
            # Forward only the knobs the daemon body honors. Don't dump
            # the parent's full env into the manifest — that bloats
            # ~/.car/agents.json and exposes secrets to other readers.
            k: os.environ[k]
            for k in (
                "NEO_OBSERVER_INTERVAL_SECONDS",
                "NEO_OBSERVER_COOLDOWN",
                "NEO_PROFILE",
                "NEO_METRICS",
            )
            if k in os.environ
        },
        "restart": "on_failure",
        "max_restarts": 10,
        "backoff_secs": 5,
        "auto_start": True,
    }


# ---------------------------------------------------------------------------
# Observer body — runs in the supervised child process
# ---------------------------------------------------------------------------


@dataclass
class ObserverConfig:
    """Tunables for one observer run. Resolved from env in ``from_env``."""

    interval_seconds: float = 300.0
    cooldown_seconds: float = 60.0
    # Committed safety bounds for the per-cycle transcript-mining pass — not
    # runtime toggles. ``ingest_budget`` caps episodes per cycle (the backlog
    # drains across cycles); ``ingest_deadline_seconds`` caps wall time so a
    # hung provider can't park the cycle and stonewall SIGTERM. Kept well under
    # the interval so a slow pass doesn't swallow the wake cadence.
    ingest_budget: int = 8
    ingest_deadline_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "ObserverConfig":
        def _read(name: str, default: float) -> float:
            raw = os.getenv(name, "").strip()
            if not raw:
                return default
            try:
                v = float(raw)
                return v if v > 0 else default
            except ValueError:
                return default

        return cls(
            interval_seconds=_read("NEO_OBSERVER_INTERVAL_SECONDS", 300.0),
            cooldown_seconds=_read("NEO_OBSERVER_COOLDOWN", 60.0),
        )


class Observer:
    """Synthesis loop. One instance per supervised child process.

    Process lifecycle (spawn, restart, stop) is owned by CAR's
    supervisor. This class only owns the in-process loop: wake on
    interval, run synthesis, sleep. Receives SIGTERM from the
    supervisor on stop and exits cleanly.
    """

    def __init__(self, codebase_root: str, config: Optional[ObserverConfig] = None):
        self.codebase_root = codebase_root
        self.config = config or ObserverConfig.from_env()
        self.project_id = _resolve_project_id(codebase_root)
        if not self.project_id:
            raise RuntimeError(
                "Cannot run observer without a resolvable project_id "
                "(no codebase_root and no git repo in cwd)"
            )
        self._stop = False
        self._last_analysis_epoch = 0.0
        self._cycles_total = 0
        # Recent cycles for the A2UI Observer tab. List of
        # ``{"timestamp": float, "text": "..."}`` capped client-side at
        # RECENT_CYCLES_MAX.
        self._recent_cycles: list[dict] = []
        # Latest FactStore snapshot — used to update the A2UI Memory tab
        # without paying a second load.
        self._last_store_snapshot: Optional[dict] = None
        # Name of the last transcript-ingest exception (or None), surfaced in
        # the cycle record so a failing LM key isn't invisible.
        self._last_ingest_error: Optional[str] = None
        # Surface manager (async). Created in run() since it needs an
        # event loop.
        self._surface = None

    def run(self) -> None:
        """Main entry — switches to async so we can run a WS client for
        A2UI alongside the synthesis loop. ``_cycle`` stays sync
        (heavy I/O) and runs in the default executor."""
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            # asyncio.run already handles SIGINT cleanup; this just keeps
            # the supervisor's exit-code expectations sane.
            pass

    async def _run_async(self) -> None:
        signal.signal(signal.SIGTERM, self._handle_stop)
        signal.signal(signal.SIGINT, self._handle_stop)

        print(
            f"neo observer started pid={os.getpid()} project={self.project_id[:8]}",
            flush=True,
        )

        # Best-effort A2UI surface. Daemon may not be up; if not, we
        # skip silently and the rest of the loop runs unchanged.
        await self._init_surface()

        # `get_running_loop()` (not `get_event_loop()`) is the
        # canonical API inside a coroutine — the latter is deprecated
        # since Python 3.10 and on 3.14 tracebacks have surfaced
        # `NameError: name 'asyncio' is not defined` here on supervisor
        # restarts. Switching avoids both the deprecation warning and
        # the observed crash.
        loop = asyncio.get_running_loop()
        while not self._stop:
            if self._cooldown_ok():
                # _cycle is synchronous and does blocking I/O. Run in
                # the executor so the WS recv loop keeps draining
                # action notifications during a long synthesis pass.
                await loop.run_in_executor(None, self._cycle)
                await self._push_surface_after_cycle()

            # Sleep in 1-second slices so SIGTERM is responsive without
            # leaving the supervisor waiting through a full interval.
            slept = 0.0
            while slept < self.config.interval_seconds and not self._stop:
                await asyncio.sleep(1.0)
                slept += 1.0

        await self._teardown_surface()
        print(f"neo observer stopped pid={os.getpid()}", flush=True)

    def _handle_stop(self, _signum: int, _frame) -> None:
        self._stop = True

    def _cooldown_ok(self) -> bool:
        elapsed = time.time() - self._last_analysis_epoch
        return elapsed >= self.config.cooldown_seconds

    def _cycle(self) -> None:
        """One synthesis pass. Errors are caught and logged — never
        propagate, so a bad cycle doesn't cause the supervisor to
        restart-loop us into a backoff hell.

        Also stashes a FactStore snapshot on ``self`` so the post-cycle
        A2UI push can update the Memory tab without re-loading.
        """
        try:
            # Lazy import — FactStore pulls fastembed/numpy/etc.; defer
            # so a malformed --cwd reports cleanly before paying that
            # import cost.
            from neo.memory.store import FactStore

            t0 = time.time()
            store = FactStore(codebase_root=self.codebase_root, eager_init=False)
            store.initialize()
            count = store.synthesize_reviews()
            # Transcript mining MUST run AFTER synthesis: synthesis is then
            # already durable, so an ingest/LM failure (isolated in its own
            # guard) can never abort it. Both passes share this executor-run
            # call (run_in_executor keeps the WS loop draining); tick is logged
            # so we can tell if it stalls and needs splitting into a separate
            # cycle. Do not reorder.
            ingested = self._ingest_transcripts(store)
            tick = time.time() - t0
            mined = (f"{ingested} mined" if not self._last_ingest_error
                     else f"ingest ERROR {self._last_ingest_error}")
            self._last_analysis_epoch = time.time()
            self._cycles_total += 1
            self._last_cycle_count = count
            self._last_cycle_error = None
            self._recent_cycles.append({
                "timestamp": self._last_analysis_epoch,
                "text": (
                    f"{time.strftime('%H:%M:%S', time.localtime(self._last_analysis_epoch))} · "
                    f"{count} synthesized, {mined}"
                ),
            })
            # Stash for the post-cycle Memory tab update. Keep the ref
            # around for ~1 cycle; FactStore is GC'd otherwise.
            self._last_store_snapshot = self._build_store_snapshot(store)
            print(
                f"neo observer cycle ok: {count} synthesized, {mined} ({tick:.1f}s)",
                flush=True,
            )
        except Exception as e:
            self._last_cycle_count = None
            self._last_cycle_error = type(e).__name__
            self._recent_cycles.append({
                "timestamp": time.time(),
                "text": f"{time.strftime('%H:%M:%S')} · ERROR {type(e).__name__}",
            })
            print(
                f"neo observer cycle error: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )

    def _ingest_transcripts(self, store) -> int:
        """Mine Claude Code transcripts for lessons, bounded by the per-cycle
        episode budget AND a wall-clock deadline. Isolated in its own
        try/except so an LM or transcript failure never aborts the synthesis
        cycle (which has already completed). Returns facts admitted; records a
        failure marker in ``self._last_ingest_error`` so a silently-failing LM
        key is visible in the cycle record, not just stderr.
        """
        self._last_ingest_error = None
        if self.config.ingest_budget <= 0:
            return 0
        try:
            from neo.adapters import resolve_adapter
            from neo.config import NeoConfig
            from neo.memory.transcript import TranscriptIngester

            cfg = NeoConfig.load()
            adapter = resolve_adapter(cfg)
            ingester = TranscriptIngester(
                store=store, lm_adapter=adapter, codebase_root=self.codebase_root
            )
            stats = ingester.ingest(
                max_episodes=self.config.ingest_budget,
                max_seconds=self.config.ingest_deadline_seconds,
                should_stop=lambda: self._stop,  # honor SIGTERM between episodes
            )
            return int(stats.get("facts_admitted", 0))
        except Exception as e:
            self._last_ingest_error = type(e).__name__
            print(
                f"neo observer transcript ingest error: {type(e).__name__}: {e}",
                file=sys.stderr,
                flush=True,
            )
            return 0

    def _build_store_snapshot(self, store) -> Optional[dict]:
        """Extract memory-tab + header-stage state from a FactStore.

        Wrapped so an unexpected schema change can't blow up the
        cycle's main path. Returns a {"memory": ..., "header": ...}
        dict that the post-cycle push splits into the two updateDataModel
        envelopes.
        """
        try:
            from neo.a2ui import memory_state_snapshot, version_state_snapshot
            repo_display = getattr(self._surface, "repo_display", "this project") \
                if self._surface else "this project"
            return {
                "memory": memory_state_snapshot(store, repo_display),
                "header": version_state_snapshot(store),
            }
        except Exception as e:  # noqa: BLE001
            logger.debug("memory snapshot extraction failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # A2UI surface integration
    # ------------------------------------------------------------------ #

    async def _init_surface(self) -> None:
        """Try to connect to car-server and register the surface.

        On failure (daemon down, websockets pkg missing) we silently
        stay disconnected — observer keeps doing its job. This is
        observability, not load-bearing.
        """
        try:
            from neo.a2ui import SurfaceManager
        except ImportError as e:
            logger.debug("a2ui module unavailable: %s", e)
            return

        self._surface = SurfaceManager(
            self.project_id,
            codebase_root=self.codebase_root,
            action_handlers={
                "kick": self._on_kick_action,
                "stop": self._on_stop_action,
            },
        )
        try:
            if await self._surface.connect():
                await self._surface.ensure_surface()
        except Exception as e:  # noqa: BLE001
            logger.debug("a2ui surface init failed: %s", e)
            self._surface = None

    async def _teardown_surface(self) -> None:
        if self._surface:
            try:
                await self._surface.close()
            except Exception:  # noqa: BLE001
                pass

    async def _push_surface_after_cycle(self) -> None:
        if not self._surface:
            return
        try:
            from neo.a2ui import observer_state_snapshot
            last_count = getattr(self, "_last_cycle_count", None)
            last_error = getattr(self, "_last_cycle_error", None)
            obs = observer_state_snapshot(
                interval_seconds=self.config.interval_seconds,
                last_cycle_epoch=self._last_analysis_epoch or None,
                last_cycle_count=last_count,
                last_cycle_error=last_error,
                cycles_total=self._cycles_total,
            )
            await self._surface.push_observer_state(obs)
            if self._last_store_snapshot is not None:
                # snapshot is {"memory": ..., "header": ...} — push the
                # memory subtree and the header's dynamic fields.
                snap = self._last_store_snapshot
                if "memory" in snap:
                    await self._surface.push_memory_state(snap["memory"])
                if "header" in snap:
                    await self._surface.push_header_state(snap["header"])
        except Exception as e:  # noqa: BLE001
            logger.debug("a2ui push after cycle failed: %s", e)

    async def _on_kick_action(self, _action: dict) -> None:
        """A2UI `kick` button → restart self via CAR's supervisor.

        ``kick_observer`` calls ``agents_restart``, which SIGTERMs the
        current child and brings up a new one — exactly the behavior a
        renderer-driven kick should produce.
        """
        try:
            kick_observer(self.codebase_root)
        except Exception as e:  # noqa: BLE001
            logger.warning("a2ui kick action failed: %s", e)

    async def _on_stop_action(self, _action: dict) -> None:
        """A2UI `stop` button → graceful SIGTERM via the supervisor."""
        try:
            stop_observer(self.codebase_root)
        except Exception as e:  # noqa: BLE001
            logger.warning("a2ui stop action failed: %s", e)


# ---------------------------------------------------------------------------
# Public lifecycle API — delegates to CAR's supervisor
# ---------------------------------------------------------------------------


def _find_managed_agent(car, agent_id: str) -> Optional[dict]:
    """Return the ManagedAgent dict for ``agent_id`` or None."""
    try:
        managed_json = car.agents_list()
        managed = json.loads(managed_json)
    except (json.JSONDecodeError, Exception) as e:
        logger.debug("agents_list failed: %s", e)
        return None
    for m in managed:
        if m.get("id") == agent_id:
            return m
    return None


def start_observer(codebase_root: Optional[str] = None) -> dict:
    """Register the spec and start the supervised child.

    Returns: ``{"status": "started"|"already_running"|"error", ...}``.
    """
    project_id = _resolve_project_id(codebase_root)
    if not project_id:
        return {"status": "error", "project_id": "",
                "message": "No project_id (run from a git repo or pass --cwd)"}

    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "project_id": project_id, "message": str(e)}

    aid = _agent_id(project_id)
    spec = _build_spec(project_id, codebase_root or os.getcwd())

    try:
        car.agents_upsert(json.dumps(spec))
    except Exception as e:
        return {"status": "error", "project_id": project_id,
                "message": f"agents_upsert failed: {e}"}

    existing = _find_managed_agent(car, aid)
    if existing and existing.get("status") == "running":
        return {"status": "already_running", "project_id": project_id,
                "agent_id": aid, "pid": existing.get("pid"),
                "message": f"observer pid {existing.get('pid')}"}

    try:
        managed_json = car.agents_start(aid)
        managed = json.loads(managed_json)
    except Exception as e:
        return {"status": "error", "project_id": project_id,
                "message": f"agents_start failed: {e}"}

    return {
        "status": "started",
        "project_id": project_id,
        "agent_id": aid,
        "pid": managed.get("pid"),
        "message": (
            f"observer pid {managed.get('pid')}, "
            f"log ~/.car/logs/{aid}.stdout.log"
        ),
    }


def stop_observer(codebase_root: Optional[str] = None) -> dict:
    project_id = _resolve_project_id(codebase_root)
    if not project_id:
        return {"status": "error", "message": "No project_id"}

    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "project_id": project_id, "message": str(e)}

    aid = _agent_id(project_id)
    existing = _find_managed_agent(car, aid)
    if not existing:
        return {"status": "not_running", "project_id": project_id, "agent_id": aid,
                "message": "no managed agent registered"}
    if existing.get("status") != "running":
        return {"status": "not_running", "project_id": project_id, "agent_id": aid,
                "pid": existing.get("pid"),
                "message": f"agent {existing.get('status', 'unknown')}"}

    try:
        managed_json = car.agents_stop(aid)
        managed = json.loads(managed_json)
    except Exception as e:
        return {"status": "error", "project_id": project_id,
                "message": f"agents_stop failed: {e}"}

    return {"status": "stopped", "project_id": project_id, "agent_id": aid,
            "pid": managed.get("pid"),
            "message": f"SIGTERM sent to pid {managed.get('pid')}"}


def kick_observer(codebase_root: Optional[str] = None) -> dict:
    """Force an early cycle by restarting the child.

    CAR's supervisor has no SIGUSR1 / signal-passthrough primitive, so
    kick maps to ``agents_restart`` — stop, then start. The new
    process runs its first cycle immediately (modulo cooldown, which
    is per-process and so resets on restart).
    """
    project_id = _resolve_project_id(codebase_root)
    if not project_id:
        return {"status": "error", "message": "No project_id"}

    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "project_id": project_id, "message": str(e)}

    aid = _agent_id(project_id)
    existing = _find_managed_agent(car, aid)
    if not existing:
        return {"status": "not_running", "project_id": project_id, "agent_id": aid,
                "message": "no managed agent registered"}

    try:
        managed_json = car.agents_restart(aid)
        managed = json.loads(managed_json)
    except Exception as e:
        return {"status": "error", "project_id": project_id,
                "message": f"agents_restart failed: {e}"}

    return {"status": "kicked", "project_id": project_id, "agent_id": aid,
            "pid": managed.get("pid"),
            "message": f"restarted as pid {managed.get('pid')}"}


def observer_status(codebase_root: Optional[str] = None) -> dict:
    project_id = _resolve_project_id(codebase_root)
    if not project_id:
        return {"status": "error", "message": "No project_id"}

    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "project_id": project_id, "message": str(e)}

    aid = _agent_id(project_id)
    existing = _find_managed_agent(car, aid)
    if not existing:
        return {"status": "not_running", "project_id": project_id, "agent_id": aid,
                "message": "no managed agent registered"}

    # CAR's `status` field is one of: stopped | starting | running |
    # backoff | errored — we surface it directly so operators can tell
    # restart-loop ("backoff") apart from a clean stop.
    car_status = existing.get("status", "unknown")
    is_running = car_status == "running"
    return {
        "status": car_status,
        "project_id": project_id,
        "agent_id": aid,
        "pid": existing.get("pid") if is_running else None,
        "restart_count": existing.get("restart_count", 0),
        "last_exit_code": existing.get("last_exit_code"),
        "log_file": f"~/.car/logs/{aid}.stdout.log",
        "message": (
            f"observer pid {existing.get('pid')} "
            f"(restarts={existing.get('restart_count', 0)})"
            if is_running
            else f"agent {car_status} (last pid {existing.get('pid')}, "
                 f"last exit {existing.get('last_exit_code')})"
        ),
    }


# ---------------------------------------------------------------------------
# Daemon entrypoint — `python -m neo.memory.observer --daemon --cwd <path>`
# Invoked by CAR's supervisor via the spec from ``_build_spec``.
# ---------------------------------------------------------------------------


def _daemon_main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="neo.memory.observer")
    p.add_argument("--daemon", action="store_true", required=True,
                   help="(internal) marks this as the daemon entrypoint")
    p.add_argument("--cwd", required=True, help="codebase root for project resolution")
    args = p.parse_args(argv)

    try:
        observer = Observer(codebase_root=args.cwd)
    except RuntimeError as e:
        print(f"observer init failed: {e}", file=sys.stderr)
        return 1

    observer.run()
    return 0


if __name__ == "__main__":
    sys.exit(_daemon_main(sys.argv[1:]))
