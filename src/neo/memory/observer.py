"""Async out-of-band synthesis observer, supervised by car-server.

A **single global** background process (CAR agent ``neo-observer``,
``GLOBAL_AGENT_ID``) that runs REVIEW→PATTERN/FAILURE synthesis on a
wall-clock cadence, **sweeping every discovered project each cycle** —
round-robin/budgeted (``max_projects_per_cycle``, watermark-gated so
unchanged projects do near-zero work). The earlier model spawned one
per-project agent (``neo-observer-<id12>``, ``--cwd <root>``); those
legacy agents are stopped and ``agents_remove``d on bootstrap. Lifecycle
is owned by CAR's agent supervisor:

  - ``agents_upsert`` registers the spec under ``~/.car/agents.json``
  - ``agents_start`` spawns the child
    (``python -m neo.memory.observer --daemon --all``; the legacy
    ``--cwd <root>`` form runs a single project and is kept only for
    migration/back-compat)
  - Supervisor handles restart-on-failure, clean SIGTERM shutdown, log
    redirection to ``~/.car/logs/neo-observer.{stdout,stderr}.log``, and
    auto-start at daemon boot when ``auto_start=True``.

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
import importlib.metadata
import json
import logging
import os
import re
import signal
import subprocess
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


_CAR_MIN_VERSION = (0, 18, 0)


def _parse_version(v: Optional[str]) -> Optional[tuple]:
    """Best-effort parse of a version string to a numeric tuple, or None."""
    nums = re.findall(r"\d+", v or "")
    return tuple(int(x) for x in nums[:3]) if nums else None


def _installed_car_version() -> Optional[str]:
    try:
        return importlib.metadata.version("car-runtime")
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_car_runtime():
    """Import car_runtime and verify it satisfies the observer's floor.

    Two gates: the ``agents_*`` supervisor API must be present, AND the
    installed version must be >= 0.18.0. 0.16.x/0.17.0 expose ``agents_upsert``
    but carry the supervisor footguns (orphaned child / restart-storm / stale
    exit code) that 0.18.0 fixed, so the attribute check alone is not enough —
    we must also enforce the version. Raises ``RuntimeError`` with an actionable
    message instead of silently running on an under-spec binding. An
    unparseable / missing version is allowed through (lenient about unknown,
    strict about known-too-old) so a vendored build is not falsely rejected.
    """
    try:
        import car_runtime
    except ImportError as e:
        raise RuntimeError(_CAR_REQUIRED_MSG) from e

    if not hasattr(car_runtime, "agents_upsert"):
        raise RuntimeError(_CAR_REQUIRED_MSG)

    version = _installed_car_version()
    parsed = _parse_version(version)
    if parsed is not None and parsed < _CAR_MIN_VERSION:
        raise RuntimeError(f"car-runtime {version} is too old. " + _CAR_REQUIRED_MSG)

    return car_runtime


def _resolve_project_id(codebase_root: Optional[str]) -> str:
    _, project_id = detect_org_and_project(codebase_root or os.getcwd())
    return project_id


# A single, global observer supervises ALL projects (one process, one CAR
# agent) rather than one per project. Legacy per-project agents use the
# ``neo-observer-<id12>`` form and are migrated away on bootstrap.
GLOBAL_AGENT_ID = "neo-observer"
_LEGACY_AGENT_RE = re.compile(r"^neo-observer-[0-9a-f]{6,}$")


def _agent_id(project_id: str) -> str:
    """Filename-safe ID for the supervisor. Short prefix keeps logs
    grep-able when multiple projects are active."""
    return f"neo-observer-{project_id[:12]}"


def _discover_project_roots() -> list[str]:
    """All project roots neo has seen, for the global observer's sweep.

    Source of truth is Claude Code's per-project transcript dirs
    (``~/.claude/projects/<encoded>``); the encoded name decodes back to an
    absolute root. Best-effort — returns ``[]`` if the dir is absent/unreadable.
    """
    from neo.memory.transcript import CLAUDE_PROJECTS_DIR
    from neo.prompt.scanner import _decode_project_path

    roots: set[str] = set()
    try:
        entries = list(CLAUDE_PROJECTS_DIR.iterdir())
    except OSError:
        return []
    for d in entries:
        if not d.is_dir():
            continue
        root = _decode_project_path(d.name)
        if root:
            roots.add(root)
    return sorted(roots)


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
    # Global-mode sweep: cap projects visited per cycle. The sweep round-robins
    # across cycles, so all projects get covered over a few cycles without one
    # cycle paying to load every project's FactStore. Projects with no new
    # transcripts since their watermark do near-zero work (ingest finds nothing).
    max_projects_per_cycle: int = 25

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

    def __init__(self, codebase_root: Optional[str] = None,
                 config: Optional[ObserverConfig] = None, global_mode: bool = False):
        self.global_mode = global_mode
        self.codebase_root = codebase_root
        self.config = config or ObserverConfig.from_env()
        if global_mode:
            # Sweeps all discovered projects; no single project_id.
            self.project_id = ""
        else:
            self.project_id = _resolve_project_id(codebase_root)
            if not self.project_id:
                raise RuntimeError(
                    "Cannot run observer without a resolvable project_id "
                    "(no codebase_root and no git repo in cwd)"
                )
        self._stop = False
        self._last_analysis_epoch = 0.0
        self._cycles_total = 0
        self._sweep_offset = 0  # round-robin cursor for global-mode sweeps
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

        scope = "all projects" if self.global_mode else f"project={self.project_id[:8]}"
        print(f"neo observer started pid={os.getpid()} {scope}", flush=True)

        # Best-effort A2UI surface (per-project inspector). The surface is keyed
        # to one project_id, so it only applies in per-project mode; the global
        # sweep skips it.
        if not self.global_mode:
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
        """One pass. Dispatches to the global sweep or the per-project cycle."""
        if self.global_mode:
            self._cycle_global()
        else:
            self._cycle_one()

    def _run_project(self, root: str) -> tuple[int, int, object]:
        """Synthesize + transcript-mine one project. Returns (synthesized,
        mined, store). Transcript mining MUST run AFTER synthesis so a durable
        synthesis can't be aborted by an ingest/LM failure (isolated in its own
        guard). Do not reorder.
        """
        from neo.memory.store import FactStore

        store = FactStore(codebase_root=root, eager_init=False)
        store.initialize()
        count = store.synthesize_reviews()
        mined = self._ingest_transcripts(store, root)
        return count, mined, store

    def _cycle_one(self) -> None:
        """Per-project pass (also feeds the A2UI inspector). Errors are caught
        and logged — never propagate, so a bad cycle can't drive the supervisor
        into backoff hell.
        """
        try:
            t0 = time.time()
            count, ingested, store = self._run_project(self.codebase_root)
            try:
                store.reconcile_cross_project_promotions()
            except Exception as e:
                logger.debug("cross-project reconcile failed: %s", e)
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

    def _cycle_global(self) -> None:
        """Global sweep: round-robin a budgeted batch of all discovered projects,
        synthesizing + mining each. Per-project errors are isolated so one bad
        project never aborts the sweep.
        """
        t0 = time.time()
        roots = _discover_project_roots()
        if not roots:
            self._last_analysis_epoch = time.time()
            self._cycles_total += 1
            print("neo observer cycle ok: 0 projects discovered", flush=True)
            return

        budget = max(1, int(self.config.max_projects_per_cycle))
        start = self._sweep_offset % len(roots)
        batch = (roots[start:] + roots[:start])[:budget]
        self._sweep_offset = (start + len(batch)) % len(roots)

        n = len(batch)
        print(
            f"neo observer sweep start: {n} of {len(roots)} project(s) this cycle",
            flush=True,
        )

        total_synth = total_mined = errors = covered = 0
        last_store = None
        for i, root in enumerate(batch, 1):
            if self._stop:
                break
            label = os.path.basename(root.rstrip("/")) or root
            p0 = time.time()
            try:
                count, mined, store = self._run_project(root)
                last_store = store
                total_synth += count
                total_mined += mined
                covered += 1
                # Per-project heartbeat so a long first sweep is legible (it logs
                # only at batch end otherwise — invisible during a multi-minute
                # backlog catch-up).
                print(
                    f"neo observer sweep [{i}/{n}] {label}: "
                    f"{count} synthesized, {mined} mined ({time.time() - p0:.1f}s)",
                    flush=True,
                )
            except Exception as e:
                errors += 1
                print(
                    f"neo observer sweep [{i}/{n}] {label}: ERROR "
                    f"{type(e).__name__}: {e}",
                    file=sys.stderr, flush=True,
                )

        # Cross-project global-promotion reconcile: once per cycle, off every
        # request path. Catches asymmetric evidence splits the request-path gate
        # can't trigger (a minority project supplying the completing episode).
        # Reuses a swept store — global scope is shared across all projects.
        reconciled = 0
        if last_store is not None and not self._stop:
            try:
                reconciled = last_store.reconcile_cross_project_promotions()
            except Exception as e:
                print(
                    f"neo observer reconcile ERROR {type(e).__name__}: {e}",
                    file=sys.stderr, flush=True,
                )

        tick = time.time() - t0
        self._last_analysis_epoch = time.time()
        self._cycles_total += 1
        self._last_cycle_count = total_synth
        self._last_cycle_error = None
        summary = (
            f"swept {covered}/{len(roots)} projects: {total_synth} synthesized, "
            f"{total_mined} mined"
            + (f", {reconciled} global-promoted" if reconciled else "")
            + (f", {errors} errors" if errors else "")
        )
        self._recent_cycles.append({
            "timestamp": self._last_analysis_epoch,
            "text": f"{time.strftime('%H:%M:%S', time.localtime(self._last_analysis_epoch))} · {summary}",
        })
        print(f"neo observer cycle ok: {summary} ({tick:.1f}s)", flush=True)

    def _ingest_transcripts(self, store, root: str) -> int:
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
                store=store, lm_adapter=adapter, codebase_root=root
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


def _build_global_spec() -> dict:
    """Spec for the single global observer that sweeps all projects."""
    return {
        "id": GLOBAL_AGENT_ID,
        "name": "Neo observer (all projects)",
        "command": sys.executable,
        "args": ["-m", "neo.memory.observer", "--daemon", "--all"],
        "cwd": os.path.expanduser("~"),
        "env": {
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


def _migrate_legacy_per_project_agents(car) -> list[str]:
    """Stop + remove legacy per-project observer agents (``neo-observer-<id>``).

    The single global observer replaces them. Best-effort; returns removed ids.
    """
    removed: list[str] = []
    try:
        managed = json.loads(car.agents_list())
    except Exception as e:
        logger.debug("agents_list failed during migration: %s", e)
        return removed
    for m in managed:
        aid = m.get("id", "")
        if aid == GLOBAL_AGENT_ID or not _LEGACY_AGENT_RE.match(aid):
            continue
        try:
            car.agents_stop(aid)
        except Exception:
            pass
        try:
            car.agents_remove(aid)
            removed.append(aid)
        except Exception as e:
            logger.debug("agents_remove(%s) failed: %s", aid, e)
    return removed


def start_observer(codebase_root: Optional[str] = None) -> dict:
    """Register + start the single global observer (migrating legacy agents).

    ``codebase_root`` is accepted for CLI compatibility but ignored — the
    observer is global, not per-project.
    """
    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}

    reaped = _reap_orphan_observers()
    migrated = _migrate_legacy_per_project_agents(car)
    try:
        car.agents_upsert(json.dumps(_build_global_spec()))
    except Exception as e:
        return {"status": "error", "message": f"agents_upsert failed: {e}"}

    existing = _find_managed_agent(car, GLOBAL_AGENT_ID)
    if existing and existing.get("status") == "running":
        return {"status": "already_running", "agent_id": GLOBAL_AGENT_ID,
                "pid": existing.get("pid"), "migrated": migrated,
                "reaped": reaped,
                "message": f"observer pid {existing.get('pid')}"}

    try:
        managed = json.loads(car.agents_start(GLOBAL_AGENT_ID))
    except Exception as e:
        return {"status": "error", "message": f"agents_start failed: {e}"}

    return {
        "status": "started",
        "agent_id": GLOBAL_AGENT_ID,
        "pid": managed.get("pid"),
        "migrated": migrated,
        "reaped": reaped,
        "message": f"observer pid {managed.get('pid')}, log ~/.car/logs/{GLOBAL_AGENT_ID}.stdout.log",
    }


def stop_observer(codebase_root: Optional[str] = None) -> dict:
    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}

    # "Stop" must halt ALL synthesis, including any unsupervised straggler —
    # an orphan isn't CAR-managed, so agents_stop alone wouldn't touch it.
    reaped = _reap_orphan_observers()

    existing = _find_managed_agent(car, GLOBAL_AGENT_ID)
    if not existing:
        return {"status": "not_running", "agent_id": GLOBAL_AGENT_ID,
                "reaped": reaped, "message": "no managed agent registered"}
    if existing.get("status") != "running":
        return {"status": "not_running", "agent_id": GLOBAL_AGENT_ID,
                "pid": existing.get("pid"), "reaped": reaped,
                "message": f"agent {existing.get('status', 'unknown')}"}

    try:
        managed = json.loads(car.agents_stop(GLOBAL_AGENT_ID))
    except Exception as e:
        return {"status": "error", "message": f"agents_stop failed: {e}"}

    return {"status": "stopped", "agent_id": GLOBAL_AGENT_ID,
            "pid": managed.get("pid"), "reaped": reaped,
            "message": f"SIGTERM sent to pid {managed.get('pid')}"}


def kick_observer(codebase_root: Optional[str] = None) -> dict:
    """Force an early sweep by restarting the global observer.

    CAR's supervisor has no SIGUSR1 / signal-passthrough primitive, so kick maps
    to ``agents_restart``. The new process runs its first sweep immediately.
    """
    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}

    existing = _find_managed_agent(car, GLOBAL_AGENT_ID)
    if not existing:
        return {"status": "not_running", "agent_id": GLOBAL_AGENT_ID,
                "message": "no managed agent registered"}

    try:
        managed = json.loads(car.agents_restart(GLOBAL_AGENT_ID))
    except Exception as e:
        return {"status": "error", "message": f"agents_restart failed: {e}"}

    return {"status": "kicked", "agent_id": GLOBAL_AGENT_ID,
            "pid": managed.get("pid"),
            "message": f"restarted as pid {managed.get('pid')}"}


def _cmd_is_our_observer(cmd: str) -> bool:
    """True if a command line is a neo observer daemon (global or legacy)."""
    return "neo.memory.observer" in cmd and "--daemon" in cmd


def _find_orphan_observers(codebase_root: Optional[str] = None) -> list[int]:
    """Orphaned observer daemons for this project — cross-platform.

    An orphan is an observer process the CAR supervisor no longer parents — left
    behind when a prior car-server died without reaping its child (the pre-0.18.0
    footgun). The supervised observer is always parented by a live car-server, so
    it is never matched. "No live parent" is the portable signal:

    - POSIX: a dead parent reparents the child to init/launchd (``ppid == 1``).
    - Windows: there is no reparenting; the parent pid simply no longer maps to a
      live process, which ``psutil`` reports as ``parent() is None``.

    Prefers ``psutil`` (works on macOS/Linux/Windows); falls back to ``ps`` on
    POSIX when ``psutil`` is absent. With the single global observer there is at
    most one supervised daemon, so any neo observer daemon with no live parent —
    a stale per-project legacy daemon, or one stranded by a dead car-server — is
    an orphan. Best-effort: returns ``[]`` on any failure so it never breaks
    ``status``.
    """
    try:
        import psutil
    except ImportError:
        return _find_orphan_observers_ps()

    orphans: list[int] = []
    for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if not _cmd_is_our_observer(cmd):
                continue
            try:
                parent_alive = proc.parent() is not None
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                parent_alive = False
            # POSIX orphan: reparented to init (ppid 1). Windows orphan: the
            # launching car-server is gone, so psutil finds no live parent.
            if proc.info.get("ppid") == 1 or not parent_alive:
                orphans.append(int(proc.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(orphans)


def _find_orphan_observers_ps() -> list[int]:
    """POSIX ``ps`` fallback for orphan detection (used when psutil is absent)."""
    try:
        out = subprocess.run(
            ["ps", "-ax", "-o", "pid=,ppid=,command="],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []

    orphans: list[int] = []
    for line in out.splitlines():
        fields = line.split(None, 2)
        if len(fields) < 3:
            continue
        pid_s, ppid_s, cmd = fields
        if ppid_s != "1":  # supervised processes are parented by car-server, not init
            continue
        if not _cmd_is_our_observer(cmd):
            continue
        try:
            orphans.append(int(pid_s))
        except ValueError:
            continue
    return sorted(orphans)


# SIGKILL where available; on Windows there is no SIGKILL and os.kill maps any
# signal to TerminateProcess, so SIGTERM already is the hard kill.
_SIGKILL = getattr(signal, "SIGKILL", signal.SIGTERM)


def _pid_cmdline(pid: int) -> Optional[str]:
    """Best-effort current command line for ``pid``, or None if unreadable.

    Used to re-verify a process's identity immediately before we signal it, so
    a recycled pid (the orphan exited and the OS reassigned its number between
    detection and the kill) isn't mistaken for the observer.
    """
    try:
        import psutil
        try:
            return " ".join(psutil.Process(pid).cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    except ImportError:
        pass
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def _reap_orphan_observers(codebase_root: Optional[str] = None,
                           force: bool = False) -> list[int]:
    """Signal any orphaned observer daemons, returning the pids signalled.

    In global mode exactly one observer is legitimate — the one CAR supervises
    (it always has a live parent, so ``_find_orphan_observers`` never returns
    it). Any neo observer daemon with no live parent is therefore redundant: a
    straggler left by a car-server that died without reaping its child. Left
    alone it keeps running synthesis cycles, doubling LM spend and exercising
    the (documented-lossy) concurrent confidence-merge path on the shared fact
    files, so we clear it before standing up a fresh supervised observer.

    ``force`` escalates SIGTERM to SIGKILL — used by the daemon when a straggler
    is mid-cycle and won't honor SIGTERM before our lock-acquire window closes.
    A hard kill is safe for the fact files: ``FactStore._save_file`` writes a
    temp file and ``os.replace``s it (atomic), so an interrupted save can only
    leave a stray ``.tmp``, never a torn fact file.

    The caller is never itself a match: a ``neo`` CLI process doesn't carry the
    ``neo.memory.observer --daemon`` command line, and a freshly-spawned
    supervised daemon still has its live car-server parent. We re-read each
    pid's command line right before signalling to defend against pid reuse.
    Best-effort — a pid that already exited (``ProcessLookupError``) or that we
    can't signal (permissions) is skipped, so this never raises into the
    lifecycle path.
    """
    sig = _SIGKILL if force else signal.SIGTERM
    reaped: list[int] = []
    for pid in _find_orphan_observers(codebase_root):
        # Identity recheck: detection snapshotted this pid earlier; re-confirm
        # it's still our observer so a recycled pid isn't signalled by mistake.
        cmd = _pid_cmdline(pid)
        if cmd is None or not _cmd_is_our_observer(cmd):
            continue
        try:
            os.kill(pid, sig)
            reaped.append(pid)
        except ProcessLookupError:
            continue  # already gone between recheck and signal
        except OSError as e:
            logger.debug("could not reap orphan observer pid %s: %s", pid, e)
    if reaped:
        verb = "killed" if force else "reaped"
        logger.info("%s %d orphaned observer(s): %s", verb, len(reaped), reaped)
    return reaped


def observer_status(codebase_root: Optional[str] = None) -> dict:
    orphans = _find_orphan_observers()

    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "message": str(e), "orphans": orphans}

    existing = _find_managed_agent(car, GLOBAL_AGENT_ID)
    if not existing:
        return {"status": "not_running", "agent_id": GLOBAL_AGENT_ID,
                "message": "no managed agent registered", "orphans": orphans}

    # CAR's `status` field is one of: stopped | starting | running |
    # backoff | errored — we surface it directly so operators can tell
    # restart-loop ("backoff") apart from a clean stop.
    car_status = existing.get("status", "unknown")
    is_running = car_status == "running"
    return {
        "status": car_status,
        "agent_id": GLOBAL_AGENT_ID,
        "pid": existing.get("pid") if is_running else None,
        "restart_count": existing.get("restart_count", 0),
        "last_exit_code": existing.get("last_exit_code"),
        "log_file": f"~/.car/logs/{GLOBAL_AGENT_ID}.stdout.log",
        "orphans": orphans,
        "message": (
            f"observer pid {existing.get('pid')} "
            f"(restarts={existing.get('restart_count', 0)})"
            if is_running
            else f"agent {car_status} (last pid {existing.get('pid')}, "
                 f"last exit {existing.get('last_exit_code')})"
        ),
    }


# ---------------------------------------------------------------------------
# Auto-bootstrap — neo registers the single global observer when CAR is present,
# so users never opt in per project. Opt out with NEO_OBSERVER_AUTOSTART=0.
# ---------------------------------------------------------------------------

_CAR_HINT_FLAG = os.path.expanduser("~/.neo/.car_observer_hint_shown")


def _car_server_reachable(host: str = "127.0.0.1", port: int = 9100,
                          timeout: float = 0.3) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _maybe_print_car_hint() -> None:
    """One-time quiet hint that CAR enables continuous background observation."""
    try:
        if os.path.exists(_CAR_HINT_FLAG):
            return
        os.makedirs(os.path.dirname(_CAR_HINT_FLAG), exist_ok=True)
        with open(_CAR_HINT_FLAG, "w", encoding="utf-8") as f:
            f.write("shown\n")
    except OSError:
        return  # can't track -> stay silent rather than nag every run
    print(
        "[Neo] tip: install the `car` extra and run car-server to enable "
        "continuous background memory observation across your projects.",
        file=sys.stderr,
    )


def maybe_autostart_observer() -> None:
    """Register + start the single global observer when CAR is present.

    Called once per neo CLI run. No-op (with a one-time hint) when CAR is absent;
    opt out with ``NEO_OBSERVER_AUTOSTART=0``. Never raises — must not break a
    neo command for any reason.
    """
    try:
        if os.getenv("NEO_OBSERVER_AUTOSTART", "").strip() == "0":
            return
        if not _car_server_reachable():
            _maybe_print_car_hint()
            return
        try:
            car = _require_car_runtime()
        except RuntimeError:
            return
        # Reap before the early return: a straggler can coexist with a live
        # managed agent (a dead car-server orphaned its child, then a new one
        # registered a fresh observer), so clear it on every run, not just when
        # we're about to register.
        _reap_orphan_observers()
        if _find_managed_agent(car, GLOBAL_AGENT_ID) is not None:
            return  # already registered; the supervisor owns its lifecycle
        _migrate_legacy_per_project_agents(car)
        car.agents_upsert(json.dumps(_build_global_spec()))
        car.agents_start(GLOBAL_AGENT_ID)
        logger.debug("auto-started global observer")
    except Exception as e:  # never break the CLI
        logger.debug("observer autostart skipped: %s", e)


# ---------------------------------------------------------------------------
# Single-instance lock — belt-and-suspenders against doubled synthesis.
# Reaping (above) resolves *which* observer survives; this lock guarantees that
# even in the handoff window two observers never write the shared fact files at
# once. Cross-platform: ``fcntl`` on POSIX, ``msvcrt`` on Windows.
# ---------------------------------------------------------------------------

# A reaped straggler can stay mid-cycle (synthesis/ingest aren't interruptible)
# well past its 1s sleep slice, so retry the lock briefly. If it still holds
# after a SIGTERM grace, escalate to SIGKILL so the kernel drops its flock
# inside the window — otherwise we'd concede and exit, leaving the unsupervised
# straggler as the sole observer.
_LOCK_ACQUIRE_ATTEMPTS = 10
_LOCK_ESCALATE_AFTER = 3  # attempts of SIGTERM grace before SIGKILL
_LOCK_PATH = os.path.expanduser("~/.neo/observer.lock")


class _SingleInstanceLock:
    """Advisory exclusive file lock held for the observer's lifetime.

    The kernel releases it when the holder exits or the fd closes, so a crashed
    observer never wedges the lock (unlike a pidfile). ``acquire`` is
    non-blocking: it returns ``False`` when another live observer holds the
    lock rather than waiting.
    """

    def __init__(self, path: str):
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            fh = open(self.path, "a+")
        except OSError as e:
            # If we can't even open the lock file, don't block the observer —
            # fall back to reap-only protection. Warn (not debug): we're
            # silently dropping the second guarantee, which an operator should
            # be able to see.
            logger.warning(
                "single-instance lock unavailable (%s); running without it", e)
            return True
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False  # another live observer holds it
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass  # process exit releases it regardless
        finally:
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Daemon entrypoint — `python -m neo.memory.observer --daemon --all`
# (legacy `--cwd <path>` runs a single project). Invoked by CAR's supervisor.
# ---------------------------------------------------------------------------


def _daemon_main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="neo.memory.observer")
    p.add_argument("--daemon", action="store_true", required=True,
                   help="(internal) marks this as the daemon entrypoint")
    p.add_argument("--all", action="store_true",
                   help="sweep all discovered projects (global observer)")
    p.add_argument("--cwd", help="single-project mode: codebase root")
    args = p.parse_args(argv)

    if not args.all and not args.cwd:
        print("observer requires --all or --cwd", file=sys.stderr)
        return 2

    # Clear any straggler from a dead car-server, then claim the lock so we
    # never run synthesis concurrently with another observer. SIGTERM first;
    # the reap frees a straggler's lock once it finishes its current cycle.
    orphans = _reap_orphan_observers()
    lock = _SingleInstanceLock(_LOCK_PATH)
    acquired = False
    for attempt in range(_LOCK_ACQUIRE_ATTEMPTS):
        if lock.acquire():
            acquired = True
            break
        # If a straggler is still holding the lock after the SIGTERM grace,
        # escalate to SIGKILL so the kernel releases its flock immediately
        # (safe: saves are atomic). Guaranteeing the lock frees here is what
        # keeps exit-0 reserved for a genuine second live observer.
        if attempt == _LOCK_ESCALATE_AFTER and orphans:
            _reap_orphan_observers(force=True)
        if attempt < _LOCK_ACQUIRE_ATTEMPTS - 1:
            time.sleep(1.0)
    if not acquired:
        # Another live observer owns synthesis. Exit cleanly (0) so CAR treats
        # this as a benign no-op, not a failure that triggers backoff.
        print("another neo observer holds the single-instance lock; exiting",
              file=sys.stderr)
        return 0

    try:
        try:
            observer = (Observer(global_mode=True) if args.all
                        else Observer(codebase_root=args.cwd))
        except RuntimeError as e:
            print(f"observer init failed: {e}", file=sys.stderr)
            return 1

        observer.run()
        return 0
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(_daemon_main(sys.argv[1:]))
