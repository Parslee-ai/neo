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


def _components_tree() -> list[dict]:
    """Component layout for the surface.

    Carved out separately from the ``createSurface`` envelope because
    A2UI v0.9's wire schema (``car-a2ui/src/lib.rs:120``) splits surface
    creation from component population:

      - ``createSurface`` carries *only* ``surfaceId`` + catalog metadata.
      - ``updateComponents`` carries the component list. We emit it
        right after ``createSurface``.
      - ``updateDataModel`` (no ``path``, ``value=<full tree>``) seeds
        the data model. Subsequent state pushes target specific paths.

    All component names below are from ``BASIC_CATALOG_V0_9``. Layout
    optimized for a human reader: real repo name in the header, plain-
    English status line ("Neo says…"), action buttons labeled by what
    they DO rather than supervisor primitives.
    """
    return [
        {"id": "root", "component": "Column",
         "children": ["header-card", "tabs"]},

        # --- Header card ---------------------------------------------
        # Mirrors `neo --version`: personality quote, then the repo
        # title, then the version + learning-stage metadata. The
        # Observer tab below owns the operational "what's it doing
        # right now" detail — header stays summary-only.
        {"id": "header-card", "component": "Card",
         "children": ["hdr-quote", "hdr-title", "hdr-version", "hdr-stage"]},
        {"id": "hdr-quote", "component": "Text",
         "text": {"path": "/header/quote"}, "variant": "body"},
        {"id": "hdr-title", "component": "Text",
         "text": {"path": "/header/title"}, "variant": "title"},
        {"id": "hdr-version", "component": "Text",
         "text": {"path": "/header/version"}, "variant": "caption"},
        {"id": "hdr-stage", "component": "Text",
         "text": {"path": "/header/stage"}, "variant": "subtitle"},

        # --- Tabs ----------------------------------------------------
        # Tabs expects two parallel arrays per the renderer contract
        # (A2uiRenderer.swift:595): `tabs` carries `{id, label}` per
        # tab; `children` carries the matching content component ids
        # in the same order.
        {"id": "tabs", "component": "Tabs",
         "tabs": [
             {"id": "tab-observer", "label": "Observer"},
             {"id": "tab-memory", "label": "Memory"},
         ],
         "children": ["observer-card", "memory-card"]},

        # --- Observer tab --------------------------------------------
        {"id": "observer-card", "component": "Card", "children": [
            "obs-status", "obs-headline", "obs-last", "obs-cadence",
            "obs-actions",
        ]},
        # Badge: per the renderer contract (A2uiRenderer.swift:675)
        # the visible value lives on `text`, not `label`. `tone` is a
        # literal (renderer reads `.asString` directly, no path
        # resolution), so it can't follow status changes — we use
        # neutral here because the badge text already says what's
        # happening; the color would be redundant or misleading on
        # error/backoff transitions we can't currently tone-bind.
        {"id": "obs-status", "component": "Badge",
         "text": {"path": "/observer/status_label"}, "tone": "success"},
        {"id": "obs-headline", "component": "Text",
         "text": {"path": "/observer/headline"}, "variant": "subtitle"},
        {"id": "obs-last", "component": "Text",
         "text": {"path": "/observer/last_check"}},
        {"id": "obs-cadence", "component": "Text",
         "text": {"path": "/observer/cadence"}, "variant": "caption"},
        # Buttons labeled by user intent, not supervisor primitive.
        # `Run now` maps to a kick (`agents_restart`); `Pause` maps to
        # a stop (`agents_stop`). The action name on the wire matches
        # the original primitive so handlers don't need to rename.
        {"id": "obs-actions", "component": "Row",
         "children": ["btn-run-now", "btn-pause"]},
        {"id": "btn-run-now", "component": "Button",
         "label": "Run now", "action": "kick"},
        {"id": "btn-pause", "component": "Button",
         "label": "Pause", "action": "stop"},

        # --- Memory tab ----------------------------------------------
        {"id": "memory-card", "component": "Card", "children": [
            "mem-headline", "mem-patterns", "mem-reviews",
            "mem-constraints", "mem-scope", "mem-probation",
        ]},
        {"id": "mem-headline", "component": "Text",
         "text": {"path": "/memory/headline"}, "variant": "subtitle"},
        {"id": "mem-patterns", "component": "Text",
         "text": {"path": "/memory/patterns"}},
        {"id": "mem-reviews", "component": "Text",
         "text": {"path": "/memory/reviews"}},
        {"id": "mem-constraints", "component": "Text",
         "text": {"path": "/memory/constraints"}},
        {"id": "mem-scope", "component": "Text",
         "text": {"path": "/memory/scope"}, "variant": "caption"},
        {"id": "mem-probation", "component": "Text",
         "text": {"path": "/memory/probation"}, "variant": "caption"},
    ]


def _repo_display_name(codebase_root: Optional[str]) -> str:
    """Human-readable repo identifier for the header. Falls back to
    the codebase root basename if no git remote is detected."""
    from neo.memory.scope import (
        _detect_org,
        _get_git_remote_url,
        _normalize_remote_url,
    )

    if not codebase_root:
        return "this project"

    remote = _get_git_remote_url(codebase_root)
    if remote:
        normalized = _normalize_remote_url(remote)
        if normalized:
            # normalized is "host/org/repo[/...]" — take last two
            # segments so e.g. "github.com/Parslee-ai/neo" → "Parslee-ai/neo"
            parts = normalized.split("/")
            if len(parts) >= 3:
                return f"{parts[-2]}/{parts[-1]}"

    org = _detect_org(codebase_root)
    from pathlib import Path
    basename = Path(codebase_root).resolve().name or "this project"
    if org and org != "unknown":
        return f"{org}/{basename}"
    return basename


def _neo_version() -> str:
    """Current neo-reasoner version string. Reads `neo.__version__`
    lazily so this module stays import-light."""
    try:
        from neo import __version__
        return __version__
    except ImportError:
        return "?"


def _initial_data_model(repo_display: str) -> dict:
    """Seed data model. JSON-pointer paths under here are what
    component bindings resolve against; ``updateDataModel`` calls
    replace subtrees on every cycle."""
    return {
        "header": {
            "quote": "Just opened my eyes.",
            "title": repo_display,
            "version": f"neo {_neo_version()}",
            "stage": "Loading…",
        },
        "observer": {
            "status_label": "Starting",
            "headline": "Looking for patterns in your project memory",
            "last_check": "—",
            "cadence": "—",
        },
        "memory": {
            "headline": f"Memory for {repo_display}",
            "patterns": "—",
            "reviews": "—",
            "constraints": "—",
            "scope": "—",
            "probation": "—",
        },
    }


def _initial_envelopes(surface_id: str, repo_display: str) -> list[dict]:
    """The three envelopes ``ensure_surface`` sends in order.

    Split because A2UI v0.9's wire schema requires it — see
    ``_components_tree`` for the long story.
    """
    return [
        {"createSurface": {"surfaceId": surface_id}},
        {"updateComponents": {
            "surfaceId": surface_id,
            "components": _components_tree(),
        }},
        {"updateDataModel": {
            "surfaceId": surface_id,
            "value": _initial_data_model(repo_display),
        }},
    ]


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
        codebase_root: Optional[str] = None,
        action_handlers: Optional[dict[str, Callable[[dict], Awaitable[None]]]] = None,
    ):
        self.project_id = project_id
        self.codebase_root = codebase_root
        self.surface_id = surface_id_for(project_id)
        self.repo_display = _repo_display_name(codebase_root)
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
        """Initialize the surface: createSurface → updateComponents →
        updateDataModel. Three envelopes, in order, per the A2UI v0.9
        wire schema (see ``_components_tree`` docstring).

        Always re-emits the component tree even when the surface
        already exists. ``updateComponents`` replaces by id, so it's
        idempotent — and it heals surfaces that were left empty by the
        pre-fix code (one ``createSurface``-only envelope, components
        silently dropped). The initial data model is only seeded when
        the surface is brand new, so we don't wipe live observer/memory
        state on reconnect.
        """
        if not self._connected:
            return
        try:
            existing = await self.client.a2ui_get(self.surface_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("a2ui_get probe failed: %s", e)
            existing = None

        envelopes = _initial_envelopes(self.surface_id, self.repo_display)
        for env in envelopes:
            # Skip createSurface when the surface already exists (the
            # daemon would refuse it). Skip the initial-dataModel seed
            # when reconnecting to an existing surface — caller has
            # live state we mustn't clobber.
            if existing and "createSurface" in env:
                continue
            if existing and "updateDataModel" in env:
                continue
            try:
                await self.client.a2ui_apply(env)
            except Exception as e:  # noqa: BLE001
                kind = next(iter(env.keys()), "?")
                logger.debug("a2ui %s failed: %s", kind, e)

    async def push_observer_state(self, state: dict) -> None:
        await self._push("/observer", state)

    async def push_memory_state(self, state: dict) -> None:
        await self._push("/memory", state)

    async def push_header_state(self, state: dict) -> None:
        """Update the header card's dynamic fields (quote, stage line).

        Static fields (title, version) are seeded by ``ensure_surface``
        and unchanged across cycles; this push only carries the parts
        that move with each FactStore snapshot.
        """
        # Push paths individually so we don't clobber the static
        # title/version bindings the renderer already cached.
        for key in ("quote", "stage"):
            if key in state:
                await self._push(f"/header/{key}", state[key])

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


def _fact_count_label(kind: str, n: int, descriptions: dict[str, str]) -> str:
    """`"50 patterns — stable techniques and conventions"` etc.
    Returns plain "0 patterns" when count is 0 so the row is still
    visible (signals "neo hasn't distilled any patterns yet"), and
    drops the em-dash trailer when no description is provided."""
    label = f"{n} {kind}"
    desc = descriptions.get(kind, "")
    return f"{label} — {desc}" if desc else label


def observer_state_snapshot(
    *,
    interval_seconds: float,
    last_cycle_epoch: Optional[float],
    last_cycle_count: Optional[int],
    last_cycle_error: Optional[str] = None,
    cycles_total: int,
) -> dict:
    """Pack observer-tab fields into the surface's data-model shape.

    All text is user-facing plain English. `interval_seconds` is used
    to render cadence; `last_cycle_*` describe the most recent
    synthesis pass.
    """
    import time as _t

    if last_cycle_epoch:
        ts = _t.strftime("%H:%M:%S", _t.localtime(last_cycle_epoch))
        if last_cycle_error:
            last_text = f"Last check at {ts} hit {last_cycle_error}"
        elif last_cycle_count and last_cycle_count > 0:
            noun = "pattern" if last_cycle_count == 1 else "patterns"
            last_text = f"Last check at {ts} distilled {last_cycle_count} new {noun}"
        else:
            last_text = f"Last check at {ts} · no new patterns"
    else:
        last_text = "No checks yet"

    interval_minutes = max(1, int(interval_seconds / 60))
    plural = "minute" if interval_minutes == 1 else "minutes"
    cadence = (
        f"Checks every {interval_minutes} {plural} · "
        f"{cycles_total} {'check' if cycles_total == 1 else 'checks'} this session"
    )

    return {
        "status_label": "Recovering" if last_cycle_error else "Active",
        "headline": "Looking for patterns in your project memory",
        "last_check": last_text,
        "cadence": cadence,
    }


# Plain-English descriptions for each fact-kind. Keep these short —
# they're inline help text, not documentation.
_KIND_DESCRIPTIONS = {
    "patterns": "stable techniques and conventions",
    "reviews": "recent observations being distilled into patterns",
    "constraints": "hard rules from project docs (CLAUDE.md, AGENTS.md)",
    "failures": "approaches that didn't work and why",
    "decisions": "feature and design decisions",
    "architecture": "architectural choices about this codebase",
    "episodes": "specific events with full context",
    "known unknowns": "explicit gaps in knowledge",
}


def memory_state_snapshot(store: Any, repo_display: str) -> dict:
    """Pack memory-tab fields from a FactStore. Counts valid facts
    only; presents kind/scope breakdowns in user-readable English.

    ``repo_display`` is passed in (not derived) so this stays pure —
    the SurfaceManager owns repo identity.
    """
    from collections import Counter

    facts = getattr(store, "_facts", []) or []
    valid = [f for f in facts if getattr(f, "is_valid", False)]

    total = len(valid)
    by_kind_raw = Counter(getattr(f.kind, "value", str(f.kind)) for f in valid)
    by_scope = Counter(getattr(f.scope, "value", str(f.scope)) for f in valid)
    probation_count = sum(
        1 for f in valid if "probation" in (getattr(f, "tags", []) or [])
    )

    # Pluralize fact-kind names into the labels users see. ``review``
    # → ``reviews``, ``known_unknown`` → ``known unknowns``, etc. The
    # internal enum stays as-is; this transform is presentation-only.
    def _pluralize_kind(k: str) -> str:
        k = k.replace("_", " ")
        # Lazy English pluralization — covers the FactKind set.
        if k.endswith("y"):
            return k[:-1] + "ies"
        if k.endswith("s"):
            return k
        return k + "s"

    by_kind = {_pluralize_kind(k): n for k, n in by_kind_raw.items()}

    # Render the three most common kinds inline (patterns / reviews /
    # constraints are the ones with the most actionable interpretation).
    # Anything else gets folded into a tail.
    primary_kinds = ["patterns", "reviews", "constraints"]
    primary_lines = {
        kind: _fact_count_label(kind, by_kind.get(kind, 0), _KIND_DESCRIPTIONS)
        for kind in primary_kinds
    }
    other = {k: v for k, v in by_kind.items() if k not in primary_kinds}

    proj_count = by_scope.get("project", 0)
    global_count = by_scope.get("global", 0)
    org_count = by_scope.get("org", 0)
    scope_parts = [f"{proj_count} specific to this project"]
    if global_count:
        scope_parts.append(f"{global_count} from your global knowledge")
    if org_count:
        scope_parts.append(f"{org_count} from your organization")
    if other:
        # Tail of less-common kinds — single line so the layout doesn't
        # explode when synthesis surfaces more variants.
        other_str = ", ".join(f"{n} {k}" for k, n in sorted(other.items(), key=lambda kv: -kv[1]))
        scope_parts.append(f"plus {other_str}")

    fact_noun = "fact" if total == 1 else "facts"
    return {
        "headline": f"{total} {fact_noun} about {repo_display}",
        "patterns": primary_lines["patterns"],
        "reviews": primary_lines["reviews"],
        "constraints": primary_lines["constraints"],
        "scope": " · ".join(scope_parts),
        "probation": (
            f"{probation_count} {'fact' if probation_count == 1 else 'facts'} "
            "still being validated (auto-promote on reuse)"
            if probation_count
            else "All facts validated — no probation"
        ),
    }


# Stage thresholds — must stay in lockstep with `subcommands.show_version`.
# Both renderers compute the same stage so the inspector header and the
# CLI version banner say the same thing about Neo's progress.
_STAGE_TABLE = [
    (0.2, 1, "Sleeper"),
    (0.4, 2, "Glitch"),
    (0.6, 3, "Unplugged"),
    (0.8, 4, "Training"),
    (1.01, 5, "The One"),  # 1.01 so the 1.0 boundary lands inside The One
]


def _stage_for_memory_level(level: float) -> tuple[int, str]:
    """Map a memory-level [0, 1] to (stage_number, stage_name)."""
    for upper, num, name in _STAGE_TABLE:
        if level < upper:
            return num, name
    return 5, "The One"


def _stage_quote(stage_num: int) -> str:
    """Stage-appropriate quote from the bundled beat deck. Falls back
    to a sensible default when the deck file or PyYAML is unavailable.
    """
    fallback = "What is real? How do you define 'real'?"
    try:
        import yaml
        from pathlib import Path

        beat_deck_path = (
            Path(__file__).parent / "config" / "beats" / "neo_matrix.yaml"
        )
        if not beat_deck_path.exists():
            return fallback
        with open(beat_deck_path) as f:
            beat_deck = yaml.safe_load(f) or {}
        stage_expr = (beat_deck.get("base_expressions") or {}).get(stage_num, {})
        return stage_expr.get("internal", fallback)
    except Exception:  # noqa: BLE001 — observability path; never error
        return fallback


def version_state_snapshot(store: Any) -> dict:
    """Header fields that mirror ``neo --version`` output — quote,
    version, stage line.

    The CLI's version banner reads:
        "<quote>"
        neo <version>
        Stage: <stage> | Memory: <pct>%
        <bar>
        <N> patterns | <C> avg confidence

    The inspector header flattens that into three Texts (quote · title
    · version · stage) — same data, no progress bar (A2UI v0.9's basic
    catalog has no Progress component).
    """
    facts = getattr(store, "_facts", []) or []
    valid = [f for f in facts if getattr(f, "is_valid", False)]
    total = len(valid)

    confs = [
        getattr(f, "metadata").confidence
        for f in valid
        if hasattr(f, "metadata")
    ]
    avg_conf = (sum(confs) / len(confs)) if confs else 0.0

    try:
        level = float(store.memory_level())
    except Exception:  # noqa: BLE001
        level = 0.0
    stage_num, stage_name = _stage_for_memory_level(level)

    fact_noun = "pattern" if total == 1 else "patterns"
    return {
        "quote": f'"{_stage_quote(stage_num)}"',
        "stage": (
            f"Stage: {stage_name} · Memory {level:.1%} · "
            f"{total} {fact_noun} · {avg_conf:.2f} avg confidence"
        ),
    }
