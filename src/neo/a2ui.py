"""A2UI surface for neo's memory + observer state.

Exposes a single per-project A2UI surface on the running car-server
daemon so any conformant renderer (CarHost.app, future webviews, etc.)
can inspect what neo's memory store + async observer are doing in
real time.

Architecture
------------
- The Python ``car_runtime.a2ui_*`` helpers are *in-process only* (the
  wheel says so explicitly). To reach the daemon's a2ui store — the
  one renderers actually subscribe to — we speak JSON-RPC over the
  daemon's WebSocket directly.
- One surface per project: ``neo-<project_id8>``. Owned by whichever
  long-lived process pushes first; today that's the observer.
- The surface has two tabs (Observer / Memory) populated by data-model
  updates the renderer auto-binds to via JSON pointers.
- Inbound ``a2ui.event`` notifications get dispatched to action
  handlers — Kick / Stop fire the same CLI lifecycle functions a user
  would invoke from a shell.

Auth
----
``car-server`` writes an auth token to a per-platform well-known path
(see ``_read_auth_token``); we read it and send ``session.auth`` as
the first frame. If the file is missing (``--no-auth`` daemon), we
skip the handshake — matches FFI bindings' "auto-detect" behavior.

Failure mode
------------
All public methods on ``SurfaceManager`` swallow connection / push
errors. A2UI is observability; it must never bring down the observer
cycle or a serve session. Errors land at debug level so operators
can find them when looking, without flooding the log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 9100
DAEMON_URL = os.environ.get("CAR_DAEMON_URL", f"ws://{DAEMON_HOST}:{DAEMON_PORT}/")
SURFACE_PREFIX = "neo-"
# Cap recent-cycles list so the data model doesn't grow without bound across
# a long observer lifetime. 20 is small enough to render cleanly in a List
# component and large enough to spot a pattern.
RECENT_CYCLES_MAX = 20


def surface_id_for(project_id: str) -> str:
    """Stable, filename-safe surface id derived from the project hash."""
    return f"{SURFACE_PREFIX}{project_id[:12]}"


def is_daemon_reachable(timeout: float = 0.5) -> bool:
    """Cheap TCP probe — true if the daemon port is accepting connections.

    Run this before attempting a full WS handshake so the "daemon
    isn't up" path stays sub-millisecond.
    """
    try:
        with socket.create_connection((DAEMON_HOST, DAEMON_PORT), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _read_auth_token() -> Optional[str]:
    """Read the daemon's auth token from the well-known per-platform path.

    Returns None when the file is missing (``car-server --no-auth``) or
    unreadable — caller skips the handshake in that case, matching the
    FFI bindings.
    """
    home = Path.home()
    candidates = [
        # macOS
        home / "Library" / "Application Support" / "ai.parslee.car" / "auth-token",
        # Linux
        Path(os.environ.get("XDG_RUNTIME_DIR", "")) / "ai.parslee.car" / "auth-token"
        if os.environ.get("XDG_RUNTIME_DIR")
        else home / ".config" / "ai.parslee.car" / "auth-token",
    ]
    for p in candidates:
        if p.exists():
            try:
                token = p.read_text().strip()
                return token or None
            except OSError:
                continue
    return None


# ---------------------------------------------------------------------------
# Minimal async JSON-RPC over WebSocket client
# ---------------------------------------------------------------------------


class DaemonClient:
    """Single-purpose JSON-RPC WS client for the a2ui surface.

    Maintains one connection, auto-auths on connect, multiplexes calls
    by integer id, and dispatches inbound notifications to registered
    handlers. Not general-purpose — designed for this module's needs.
    """

    def __init__(self, url: str = DAEMON_URL):
        self.url = url
        self._ws = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._notif_handlers: dict[str, Callable[[dict], Awaitable[None]]] = {}
        self._recv_task: Optional[asyncio.Task] = None
        self._closed = False

    async def connect(self) -> None:
        """Open the WS, run the auth handshake, start the recv loop.

        Raises on connection failure — callers wrap this so the
        "daemon not up" path doesn't propagate up.
        """
        import websockets  # lazy import — not all neo installs have it

        self._ws = await websockets.connect(self.url)
        self._recv_task = asyncio.create_task(self._recv_loop())

        token = _read_auth_token()
        if token:
            # session.auth must be the first frame on every connection
            # when the daemon has auth enabled (the default since 2026-05).
            await self._call("session.auth", {"token": token})

    async def close(self) -> None:
        self._closed = True
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001 — best-effort
                pass

    def on_notification(
        self, method: str, handler: Callable[[dict], Awaitable[None]]
    ) -> None:
        """Register an async handler for an incoming JSON-RPC notification.

        Single-subscriber-per-method by design; matches the daemon's
        `register_notification_handler` contract.
        """
        self._notif_handlers[method] = handler

    async def _call(self, method: str, params: dict) -> Any:
        if not self._ws:
            raise RuntimeError("daemon client not connected")
        self._next_id += 1
        req_id = self._next_id
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[req_id] = fut

        msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
        await self._ws.send(json.dumps(msg))
        response = await fut

        if "error" in response:
            raise RuntimeError(
                f"jsonrpc {method} failed: "
                f"{response['error'].get('message', response['error'])}"
            )
        return response.get("result")

    async def _recv_loop(self) -> None:
        """Read frames forever; demux replies to futures, route notifications."""
        try:
            assert self._ws is not None
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("ignoring non-json frame")
                    continue

                if "id" in msg and msg["id"] in self._pending:
                    self._pending.pop(msg["id"]).set_result(msg)
                elif "method" in msg:  # notification
                    handler = self._notif_handlers.get(msg["method"])
                    if handler:
                        try:
                            await handler(msg.get("params", {}))
                        except Exception as e:  # noqa: BLE001
                            logger.warning(
                                "notification handler for %s raised: %s",
                                msg["method"], e,
                            )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if not self._closed:
                logger.debug("a2ui recv loop exiting: %s", e)
        finally:
            # Resolve any in-flight calls so awaiters don't hang.
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("daemon connection closed"))
            self._pending.clear()

    # Convenience wrappers around the a2ui.* surface --------------------------

    async def a2ui_apply(self, envelope: dict) -> dict:
        """Send a single A2UI envelope. The envelope IS the params shape —
        the daemon doesn't expect a wrapper ({"envelope": ...} silently
        creates an empty surface)."""
        return await self._call("a2ui.apply", envelope)

    async def a2ui_get(self, surface_id: str) -> Optional[dict]:
        return await self._call("a2ui.get", {"surfaceId": surface_id})


# ---------------------------------------------------------------------------
# Surface schema builders
# ---------------------------------------------------------------------------


def _initial_surface_envelope(surface_id: str, project_id: str) -> dict:
    """The ``createSurface`` envelope — layout + initial data model.

    Component tree uses only types in ``BASIC_CATALOG_V0_9`` so any
    conformant renderer (CarHost.app, future webview) can draw it.
    All mutable values are JSON-pointer-bound against ``dataModel``;
    later updates call ``updateDataModel`` instead of re-emitting
    components.
    """
    return {
        "createSurface": {
            "surfaceId": surface_id,
            "components": [
                {"id": "root", "component": "Column",
                 "children": ["header", "tabs"]},
                {"id": "header", "component": "Card", "children": ["title"]},
                {"id": "title", "component": "Text",
                 "text": {"path": "/header/title"}, "variant": "title"},
                {"id": "tabs", "component": "Tabs", "children": [
                    {"id": "tab-observer", "label": "Observer",
                     "child": "observer-card"},
                    {"id": "tab-memory", "label": "Memory",
                     "child": "memory-card"},
                ]},
                # --- Observer tab --------------------------------------------
                {"id": "observer-card", "component": "Card", "children": [
                    "obs-status", "obs-pid", "obs-last", "obs-recent",
                    "obs-actions",
                ]},
                {"id": "obs-status", "component": "Badge",
                 "label": {"path": "/observer/status"}},
                {"id": "obs-pid", "component": "Text",
                 "text": {"path": "/observer/header_text"}, "variant": "subtitle"},
                {"id": "obs-last", "component": "Text",
                 "text": {"path": "/observer/last_cycle_text"}},
                {"id": "obs-recent", "component": "List",
                 "forEach": {"path": "/observer/recent_cycles"},
                 "itemTemplate": {"component": "Text",
                                  "text": {"path": "/text"}}},
                {"id": "obs-actions", "component": "Row",
                 "children": ["btn-kick", "btn-stop"]},
                {"id": "btn-kick", "component": "Button",
                 "label": "Kick", "action": "kick"},
                {"id": "btn-stop", "component": "Button",
                 "label": "Stop", "action": "stop"},
                # --- Memory tab ----------------------------------------------
                {"id": "memory-card", "component": "Card", "children": [
                    "mem-total", "mem-by-kind", "mem-by-scope", "mem-probation",
                ]},
                {"id": "mem-total", "component": "Text",
                 "text": {"path": "/memory/total_text"}, "variant": "subtitle"},
                {"id": "mem-by-kind", "component": "Text",
                 "text": {"path": "/memory/by_kind_text"}},
                {"id": "mem-by-scope", "component": "Text",
                 "text": {"path": "/memory/by_scope_text"}},
                {"id": "mem-probation", "component": "Text",
                 "text": {"path": "/memory/probation_text"}},
            ],
            "dataModel": {
                "header": {"title": f"Neo · project {project_id[:8]}"},
                "observer": {
                    "status": "starting",
                    "header_text": "—",
                    "last_cycle_text": "no cycles yet",
                    "recent_cycles": [],
                },
                "memory": {
                    "total_text": "loading…",
                    "by_kind_text": "—",
                    "by_scope_text": "—",
                    "probation_text": "—",
                },
            },
        }
    }


def _update_path_envelope(surface_id: str, path: str, value: Any) -> dict:
    """``updateDataModel`` for a single JSON-pointer path."""
    return {
        "updateDataModel": {
            "surfaceId": surface_id,
            "path": path,
            "value": value,
        }
    }


# ---------------------------------------------------------------------------
# SurfaceManager — public API used by Observer (and eventually serve)
# ---------------------------------------------------------------------------


class SurfaceManager:
    """Lifecycle wrapper combining client + surface state.

    Usage::

        mgr = SurfaceManager(project_id, action_handlers={
            "kick": on_kick, "stop": on_stop,
        })
        if await mgr.connect():
            await mgr.ensure_surface()
            await mgr.push_observer_state({...})
            ...
            await mgr.close()

    All methods are best-effort: ``connect`` returns False when the
    daemon's unreachable; push methods no-op when not connected.
    """

    def __init__(
        self,
        project_id: str,
        action_handlers: Optional[dict[str, Callable[[dict], Awaitable[None]]]] = None,
    ):
        self.project_id = project_id
        self.surface_id = surface_id_for(project_id)
        self.client = DaemonClient()
        self._connected = False
        self._action_handlers = action_handlers or {}

    async def connect(self) -> bool:
        """Attempt to connect + auth. Returns False if the daemon is down
        or the handshake fails — callers degrade silently."""
        if not is_daemon_reachable():
            return False
        try:
            await self.client.connect()
            self.client.on_notification("a2ui.event", self._on_a2ui_event)
            self._connected = True
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug("a2ui connect failed: %s", e)
            try:
                await self.client.close()
            except Exception:  # noqa: BLE001
                pass
            return False

    async def close(self) -> None:
        if self._connected:
            try:
                await self.client.close()
            finally:
                self._connected = False

    async def ensure_surface(self) -> None:
        """Create the surface if it doesn't already exist on the daemon.

        Safe to call repeatedly — first call wins. If two processes race,
        the second's ``createSurface`` will error and we treat it as
        "already created" (no rethrow).
        """
        if not self._connected:
            return
        try:
            existing = await self.client.a2ui_get(self.surface_id)
            if existing:
                return
        except Exception as e:  # noqa: BLE001
            logger.debug("a2ui_get probe failed: %s", e)

        try:
            await self.client.a2ui_apply(
                _initial_surface_envelope(self.surface_id, self.project_id)
            )
        except Exception as e:  # noqa: BLE001
            # Race: another process beat us to it. Fine.
            logger.debug("createSurface skipped (likely already exists): %s", e)

    async def push_observer_state(self, state: dict) -> None:
        await self._push("/observer", state)

    async def push_memory_state(self, state: dict) -> None:
        await self._push("/memory", state)

    async def _push(self, path: str, value: Any) -> None:
        if not self._connected:
            return
        try:
            await self.client.a2ui_apply(
                _update_path_envelope(self.surface_id, path, value)
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("a2ui push %s failed: %s", path, e)

    async def _on_a2ui_event(self, params: dict) -> None:
        """Dispatch inbound action events to registered handlers."""
        # The daemon's a2ui.event wraps an action under "action".
        # Surface filter: only fire when the event belongs to OUR surface
        # (others may belong to peer agents on the same daemon).
        if params.get("surfaceId") and params["surfaceId"] != self.surface_id:
            return
        action = params.get("action") or {}
        name = action.get("name")
        if not name:
            return
        handler = self._action_handlers.get(name)
        if not handler:
            logger.debug("no handler for a2ui action '%s'", name)
            return
        try:
            await handler(action)
        except Exception as e:  # noqa: BLE001
            logger.warning("a2ui action handler '%s' raised: %s", name, e)


# ---------------------------------------------------------------------------
# State extractors — pure functions over runtime objects
# ---------------------------------------------------------------------------


def observer_state_snapshot(
    *,
    pid: int,
    project_id: str,
    last_cycle_epoch: Optional[float],
    last_cycle_count: Optional[int],
    cycles_total: int,
    recent_cycles: list[dict],
) -> dict:
    """Pack observer-tab fields into the JSON shape the surface expects."""
    import time as _t

    if last_cycle_epoch:
        age = max(0, int(_t.time() - last_cycle_epoch))
        last_text = f"last cycle {_humanize_seconds(age)} ago · {last_cycle_count} synthesized"
    else:
        last_text = "no cycles yet"

    return {
        "status": "running",
        "header_text": f"pid {pid} · project {project_id[:8]} · {cycles_total} cycles",
        "last_cycle_text": last_text,
        "recent_cycles": recent_cycles[-RECENT_CYCLES_MAX:],
    }


def memory_state_snapshot(store: Any) -> dict:
    """Pack memory-tab fields from a FactStore. Counts valid facts only."""
    from collections import Counter

    facts = getattr(store, "_facts", []) or []
    valid = [f for f in facts if getattr(f, "is_valid", False)]

    total = len(valid)
    by_kind = Counter(getattr(f.kind, "value", str(f.kind)) for f in valid)
    by_scope = Counter(getattr(f.scope, "value", str(f.scope)) for f in valid)

    probation_count = sum(
        1 for f in valid if "probation" in getattr(f, "tags", []) or []
    )

    return {
        "total_text": f"{total} valid facts",
        "by_kind_text": "by kind: " + ", ".join(
            f"{k}={v}" for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1])
        ) if by_kind else "by kind: —",
        "by_scope_text": "by scope: " + ", ".join(
            f"{s}={v}" for s, v in sorted(by_scope.items(), key=lambda kv: -kv[1])
        ) if by_scope else "by scope: —",
        "probation_text": f"probation: {probation_count} fact(s)",
    }


def _humanize_seconds(s: int) -> str:
    """`90 → "1m30s"`, `3700 → "1h1m"`. Capped at hours."""
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60}s"
    return f"{s // 3600}h{(s % 3600) // 60}m"
