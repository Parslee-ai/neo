"""Tests for neo.a2ui — A2UI surface client + state extractors.

The async WS client is exercised against a fake DaemonClient that
just records calls instead of opening a real socket. Pure-function
helpers (envelope builders, state extractors) get direct unit tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Surface id + reachability helpers
# ---------------------------------------------------------------------------


class TestSurfaceId:
    def test_uses_first_12_hex_chars(self):
        from neo.a2ui import surface_id_for
        assert surface_id_for("fcbc43ed0a20b8b8") == "neo-fcbc43ed0a20"

    def test_short_project_id(self):
        from neo.a2ui import surface_id_for
        # Caller can pass an already-short id; we just prefix.
        assert surface_id_for("abc") == "neo-abc"


class TestDaemonReachable:
    def test_returns_false_when_port_closed(self):
        """Probe a TCP port we *know* nothing listens on."""
        from neo.a2ui import is_daemon_reachable
        # Patch the DAEMON_HOST/PORT used by the probe. Targeting
        # localhost:1 is reliably unbound on dev machines.
        from neo import a2ui
        port = a2ui.DAEMON_PORT
        host = a2ui.DAEMON_HOST
        try:
            a2ui.DAEMON_HOST = "127.0.0.1"
            a2ui.DAEMON_PORT = 1  # well-known unbound
            assert is_daemon_reachable(timeout=0.1) is False
        finally:
            a2ui.DAEMON_HOST = host
            a2ui.DAEMON_PORT = port


# ---------------------------------------------------------------------------
# Envelope builders — pure functions
# ---------------------------------------------------------------------------


class TestInitialEnvelopes:
    """A2UI v0.9 splits createSurface from components/dataModel — the
    wire schema rejects them as fields on createSurface. ensure_surface
    sends three envelopes in order; this group tests each.
    """

    def test_emits_three_envelopes_in_correct_order(self):
        from neo.a2ui import _initial_envelopes
        envs = _initial_envelopes("neo-test12345678", "test12345678abcd")
        assert len(envs) == 3
        kinds = [next(iter(e.keys())) for e in envs]
        assert kinds == ["createSurface", "updateComponents", "updateDataModel"]

    def test_createSurface_carries_only_surfaceId(self):
        """Regression: serde silently drops unknown fields, so any
        ``components``/``dataModel`` on createSurface would create an
        empty surface that the renderer can't draw."""
        from neo.a2ui import _initial_envelopes
        envs = _initial_envelopes("neo-x", "x")
        cs = envs[0]["createSurface"]
        assert cs == {"surfaceId": "neo-x"}, (
            f"createSurface must only carry surfaceId; got {cs}"
        )

    def test_updateComponents_has_full_component_tree(self):
        from neo.a2ui import _initial_envelopes
        envs = _initial_envelopes("neo-x", "x")
        uc = envs[1]["updateComponents"]
        assert uc["surfaceId"] == "neo-x"
        comp_ids = [c["id"] for c in uc["components"]]
        assert "root" in comp_ids
        assert "obs-status" in comp_ids
        assert "mem-total" in comp_ids
        assert "btn-kick" in comp_ids

    def test_updateDataModel_seeds_full_value_tree(self):
        from neo.a2ui import _initial_envelopes
        envs = _initial_envelopes("neo-fcbc43ed0a20", "fcbc43ed0a20b8b8")
        udm = envs[2]["updateDataModel"]
        assert udm["surfaceId"] == "neo-fcbc43ed0a20"
        # No "path" — whole-value replace, per the Rust UpdateDataModel
        # struct (`path: Option<String>`).
        assert "path" not in udm
        dm = udm["value"]
        assert "observer" in dm
        assert "memory" in dm
        assert "fcbc43ed" in dm["header"]["title"]

    def test_uses_only_basic_catalog_components(self):
        """Component types must be in BASIC_CATALOG_V0_9. Otherwise
        non-CarHost renderers can't draw the surface."""
        from neo.a2ui import _components_tree
        allowed = {
            "Column", "Row", "Card", "Divider", "Spacer", "Tabs", "Modal", "List",
            "Text", "Image", "File", "Badge",
            "Button", "TextField", "CheckBox", "ChoicePicker", "Select", "Slider",
            "FilePicker",
        }
        for c in _components_tree():
            if "component" in c:  # tabs children dicts have no 'component'
                assert c["component"] in allowed, f"unknown component: {c['component']}"


class TestUpdatePathEnvelope:
    def test_envelope_shape(self):
        from neo.a2ui import _update_path_envelope
        env = _update_path_envelope("neo-x", "/observer/status", "running")
        assert env == {
            "updateDataModel": {
                "surfaceId": "neo-x",
                "path": "/observer/status",
                "value": "running",
            }
        }


# ---------------------------------------------------------------------------
# State extractors
# ---------------------------------------------------------------------------


class TestObserverStateSnapshot:
    def test_no_cycles_yet(self):
        from neo.a2ui import observer_state_snapshot
        s = observer_state_snapshot(
            pid=123, project_id="abcdef1234567890",
            last_cycle_epoch=None, last_cycle_count=None,
            cycles_total=0, recent_cycles=[],
        )
        assert s["status"] == "running"
        assert s["last_cycle_text"] == "no cycles yet"
        assert "pid 123" in s["header_text"]
        assert s["recent_cycles"] == []

    def test_includes_recent_cycle_summary(self):
        import time
        from neo.a2ui import observer_state_snapshot
        now = time.time() - 65  # 1m5s ago
        s = observer_state_snapshot(
            pid=123, project_id="abcdef1234567890",
            last_cycle_epoch=now, last_cycle_count=3,
            cycles_total=5, recent_cycles=[{"text": "x"}],
        )
        assert "3 synthesized" in s["last_cycle_text"]
        assert "5 cycles" in s["header_text"]

    def test_recent_cycles_capped(self):
        """RECENT_CYCLES_MAX = 20; longer histories should be trimmed."""
        from neo.a2ui import RECENT_CYCLES_MAX, observer_state_snapshot
        history = [{"text": f"cycle-{i}"} for i in range(50)]
        s = observer_state_snapshot(
            pid=1, project_id="x", last_cycle_epoch=None,
            last_cycle_count=None, cycles_total=50, recent_cycles=history,
        )
        assert len(s["recent_cycles"]) == RECENT_CYCLES_MAX
        # Tail kept, not head — most recent at the end.
        assert s["recent_cycles"][-1]["text"] == "cycle-49"


class TestMemoryStateSnapshot:
    def test_empty_store(self):
        from neo.a2ui import memory_state_snapshot
        fake = MagicMock(_facts=[])
        s = memory_state_snapshot(fake)
        assert "0 valid facts" in s["total_text"]
        assert s["by_kind_text"] == "by kind: —"
        assert s["by_scope_text"] == "by scope: —"
        assert "0 fact" in s["probation_text"]

    def test_counts_valid_only(self):
        from neo.a2ui import memory_state_snapshot
        facts = []
        for i in range(3):
            f = MagicMock()
            f.is_valid = True
            f.kind = MagicMock(value="pattern")
            f.scope = MagicMock(value="project")
            f.tags = []
            facts.append(f)
        invalid = MagicMock()
        invalid.is_valid = False
        invalid.kind = MagicMock(value="pattern")
        invalid.scope = MagicMock(value="project")
        invalid.tags = []
        facts.append(invalid)

        fake = MagicMock(_facts=facts)
        s = memory_state_snapshot(fake)
        assert "3 valid facts" in s["total_text"]
        assert "pattern=3" in s["by_kind_text"]
        assert "project=3" in s["by_scope_text"]

    def test_probation_tag_counted(self):
        from neo.a2ui import memory_state_snapshot
        f1 = MagicMock(is_valid=True, kind=MagicMock(value="review"),
                       scope=MagicMock(value="project"), tags=["probation"])
        f2 = MagicMock(is_valid=True, kind=MagicMock(value="pattern"),
                       scope=MagicMock(value="project"), tags=[])
        s = memory_state_snapshot(MagicMock(_facts=[f1, f2]))
        assert "1 fact" in s["probation_text"]


class TestHumanizeSeconds:
    def test_seconds(self):
        from neo.a2ui import _humanize_seconds
        assert _humanize_seconds(45) == "45s"

    def test_minutes(self):
        from neo.a2ui import _humanize_seconds
        assert _humanize_seconds(90) == "1m30s"

    def test_hours(self):
        from neo.a2ui import _humanize_seconds
        assert _humanize_seconds(3700) == "1h1m"


# ---------------------------------------------------------------------------
# SurfaceManager — async lifecycle with a stubbed DaemonClient
# ---------------------------------------------------------------------------


class FakeDaemonClient:
    """In-memory DaemonClient stand-in. Records every WS call."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.connected = False
        self.notif_handler = None
        self.get_returns: dict = None  # a2ui_get response

    async def connect(self):
        self.connected = True

    async def close(self):
        self.connected = False

    def on_notification(self, method, handler):
        self.notif_handler = (method, handler)

    async def a2ui_apply(self, envelope):
        self.calls.append(("a2ui.apply", envelope))
        return {"surfaceId": envelope.get("createSurface", {}).get("surfaceId", "")}

    async def a2ui_get(self, surface_id):
        self.calls.append(("a2ui.get", {"surfaceId": surface_id}))
        return self.get_returns


@pytest.fixture
def fake_reachable(monkeypatch):
    monkeypatch.setattr("neo.a2ui.is_daemon_reachable", lambda timeout=0.5: True)


@pytest.fixture
def fake_unreachable(monkeypatch):
    monkeypatch.setattr("neo.a2ui.is_daemon_reachable", lambda timeout=0.5: False)


@pytest.fixture
def stub_client(monkeypatch):
    """Replace DaemonClient in the SurfaceManager with our fake."""
    fake = FakeDaemonClient()
    # Patch the class so SurfaceManager() picks the fake instance up.
    monkeypatch.setattr("neo.a2ui.DaemonClient", lambda: fake)
    return fake


class TestSurfaceManagerConnect:
    @pytest.mark.asyncio
    async def test_connect_returns_false_when_daemon_unreachable(
        self, fake_unreachable, stub_client,
    ):
        from neo.a2ui import SurfaceManager
        mgr = SurfaceManager("abc12345")
        assert (await mgr.connect()) is False
        # Did NOT open a connection
        assert stub_client.connected is False

    @pytest.mark.asyncio
    async def test_connect_succeeds_when_reachable(self, fake_reachable, stub_client):
        from neo.a2ui import SurfaceManager
        mgr = SurfaceManager("abc12345")
        assert (await mgr.connect()) is True
        assert stub_client.connected is True
        # Notification handler registered for a2ui.event
        assert stub_client.notif_handler is not None
        assert stub_client.notif_handler[0] == "a2ui.event"


class TestSurfaceManagerEnsureSurface:
    @pytest.mark.asyncio
    async def test_creates_emits_three_envelopes_in_order(
        self, fake_reachable, stub_client,
    ):
        """Fresh surface: createSurface → updateComponents → updateDataModel.
        Order matters — components reference dataModel paths that must
        exist (or be created) by the time the renderer evaluates them."""
        from neo.a2ui import SurfaceManager
        stub_client.get_returns = None  # surface doesn't exist

        mgr = SurfaceManager("abc12345")
        await mgr.connect()
        await mgr.ensure_surface()

        apply_envelopes = [c[1] for c in stub_client.calls if c[0] == "a2ui.apply"]
        envelope_kinds = [next(iter(e.keys())) for e in apply_envelopes]
        assert envelope_kinds == ["createSurface", "updateComponents", "updateDataModel"]

    @pytest.mark.asyncio
    async def test_existing_surface_repairs_components(
        self, fake_reachable, stub_client,
    ):
        """Existing surface (e.g. left empty by pre-fix code): re-emit
        updateComponents to heal the tree, but skip createSurface and
        skip the initial dataModel seed so live state isn't wiped."""
        from neo.a2ui import SurfaceManager
        stub_client.get_returns = {"surfaceId": "neo-abc12345"}

        mgr = SurfaceManager("abc12345")
        await mgr.connect()
        await mgr.ensure_surface()

        apply_envelopes = [c[1] for c in stub_client.calls if c[0] == "a2ui.apply"]
        kinds = [next(iter(e.keys())) for e in apply_envelopes]
        assert "createSurface" not in kinds
        assert "updateDataModel" not in kinds
        assert kinds == ["updateComponents"]


class TestSurfaceManagerPush:
    @pytest.mark.asyncio
    async def test_push_observer_state_emits_updateDataModel(
        self, fake_reachable, stub_client,
    ):
        from neo.a2ui import SurfaceManager
        mgr = SurfaceManager("abc12345")
        await mgr.connect()
        await mgr.push_observer_state({"status": "running"})

        apply_calls = [c for c in stub_client.calls if c[0] == "a2ui.apply"]
        envelope = apply_calls[-1][1]
        assert "updateDataModel" in envelope
        assert envelope["updateDataModel"]["path"] == "/observer"
        assert envelope["updateDataModel"]["value"] == {"status": "running"}

    @pytest.mark.asyncio
    async def test_push_memory_state_emits_updateDataModel(
        self, fake_reachable, stub_client,
    ):
        from neo.a2ui import SurfaceManager
        mgr = SurfaceManager("abc12345")
        await mgr.connect()
        await mgr.push_memory_state({"total_text": "100 valid facts"})

        envelope = stub_client.calls[-1][1]
        assert envelope["updateDataModel"]["path"] == "/memory"

    @pytest.mark.asyncio
    async def test_push_when_disconnected_is_silent_noop(
        self, fake_unreachable, stub_client,
    ):
        from neo.a2ui import SurfaceManager
        mgr = SurfaceManager("abc12345")
        await mgr.connect()  # returns False
        await mgr.push_observer_state({"status": "running"})  # must not raise
        # No calls reached the (stub) client.
        assert all(c[0] != "a2ui.apply" for c in stub_client.calls)


class TestWireShape:
    """Regression tests against the actual JSON-RPC params dict sent to
    the daemon — not the envelope shape, which is identical but the
    bug class is "wrapping the envelope under a key the daemon ignores."
    """

    @pytest.mark.asyncio
    async def test_a2ui_apply_passes_envelope_as_params_directly(self):
        """The daemon expects ``params == envelope``, NOT
        ``params == {"envelope": envelope}``. Wrapping silently creates
        an empty surface — caught by the live test on 2026-05-24."""
        from neo.a2ui import DaemonClient

        client = DaemonClient()
        recorded: dict = {}

        async def fake_call(method, params):
            recorded["method"] = method
            recorded["params"] = params
            return {"surfaceId": "neo-x"}

        client._call = fake_call

        envelope = {"createSurface": {"surfaceId": "neo-x",
                                      "components": [], "dataModel": {}}}
        await client.a2ui_apply(envelope)

        assert recorded["method"] == "a2ui.apply"
        assert recorded["params"] == envelope
        # The bug shape — params being wrapped — would put "envelope" at
        # the top level of params, not "createSurface".
        assert "createSurface" in recorded["params"]
        assert "envelope" not in recorded["params"]


class TestActionDispatch:
    @pytest.mark.asyncio
    async def test_action_dispatches_to_registered_handler(
        self, fake_reachable, stub_client,
    ):
        from neo.a2ui import SurfaceManager
        fired: list[dict] = []

        async def on_kick(action):
            fired.append(action)

        mgr = SurfaceManager("abc12345", action_handlers={"kick": on_kick})
        await mgr.connect()

        # Simulate the daemon firing an a2ui.event notification
        await mgr._on_a2ui_event({
            "surfaceId": "neo-abc12345",
            "action": {"name": "kick", "context": {}},
        })
        assert len(fired) == 1
        assert fired[0]["name"] == "kick"

    @pytest.mark.asyncio
    async def test_action_for_other_surface_ignored(self, fake_reachable, stub_client):
        from neo.a2ui import SurfaceManager
        fired: list[dict] = []

        async def on_kick(action):
            fired.append(action)

        mgr = SurfaceManager("abc12345", action_handlers={"kick": on_kick})
        await mgr.connect()

        # Different surfaceId
        await mgr._on_a2ui_event({
            "surfaceId": "some-other-surface",
            "action": {"name": "kick"},
        })
        assert fired == []

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_propagate(self, fake_reachable, stub_client):
        from neo.a2ui import SurfaceManager

        async def boom(_action):
            raise RuntimeError("explode")

        mgr = SurfaceManager("abc12345", action_handlers={"kick": boom})
        await mgr.connect()
        # Must not raise — handler errors are logged + swallowed.
        await mgr._on_a2ui_event({
            "surfaceId": "neo-abc12345",
            "action": {"name": "kick"},
        })
