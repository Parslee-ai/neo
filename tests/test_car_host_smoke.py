"""Live smoke test for neo.car_host's tool-dispatch round-trip.

Skipped automatically when the local car-server daemon (default
ws://127.0.0.1:9100) is not reachable, so the suite stays
greenable without external infra.

Confirms what the unit tests can't: the full path daemon ->
register_tool_handler -> Python callback -> daemon -> caller
actually works end-to-end with the v0.8.0+ car-runtime FFI that
ships the Parslee-ai/car-releases#38 fix.

Run manually with the daemon up:
    car-server &
    pytest tests/test_car_host_smoke.py -v
"""

from __future__ import annotations

import json
import os
import socket
import threading
from pathlib import Path

import pytest


def _daemon_reachable(host: str = "127.0.0.1", port: int = 9100) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


# Capture the real HOME at module import — before conftest.py's
# isolate_neo_home autouse fixture rewrites it on every test.
# The daemon auth-token (`~/Library/Application Support/ai.parslee.car/
# auth-token` on macOS) lives under the real HOME; if the FFI tries to
# read it under a tmpdir HOME, the handshake is skipped and the daemon
# rejects every call with -32001 auth required.
_REAL_HOME = os.environ["HOME"]


@pytest.fixture(autouse=True)
def use_real_home(monkeypatch):
    """Override conftest's isolate_neo_home for this file.

    The smoke test needs to reach the daemon, which requires reading
    its auth-token from $HOME. The neo-memory isolation isn't relevant
    here — we don't touch ~/.neo at all.
    """
    monkeypatch.setenv("HOME", _REAL_HOME)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: Path(_REAL_HOME)))
    yield


pytestmark = pytest.mark.skipif(
    not _daemon_reachable(),
    reason="car-server daemon not running on ws://127.0.0.1:9100",
)

car_runtime = pytest.importorskip(
    "car_runtime",
    reason="car-runtime not installed (install with the [car] extra)",
)


def _has_submit_proposal() -> bool:
    return hasattr(car_runtime.CarRuntime, "submit_proposal") and hasattr(
        car_runtime, "register_tool_handler"
    )


@pytest.mark.skipif(
    not _has_submit_proposal(),
    reason=(
        "car-runtime missing register_tool_handler / submit_proposal. "
        "Requires Parslee-ai/car-releases#38 (car-runtime >= 0.8.0)."
    ),
)
def test_neo_process_tool_round_trips_through_daemon():
    """End-to-end: daemon calls our Python handler for neo.process."""
    from neo.car_tool_schema import TOOL_NAME, tool_schema_json

    rt = car_runtime.CarRuntime()
    rt.register_tool_schema(tool_schema_json())

    fired = threading.Event()
    received_prompt: list[str] = []

    def handler(call_json: str) -> str:
        fired.set()
        call = json.loads(call_json)
        received_prompt.append((call.get("params") or {}).get("prompt", ""))
        # Return a minimal valid NeoOutput-shaped dict; the unit tests
        # in test_car_tool_schema cover the real serializer.
        return json.dumps({
            "plan": [],
            "simulation_traces": [],
            "code_suggestions": [],
            "static_checks": [],
            "next_questions": [],
            "confidence": 1.0,
            "notes": "smoke",
            "metadata": {"smoke": True},
        })

    car_runtime.register_tool_handler(handler)

    try:
        proposal = json.dumps({
            "actions": [
                {
                    "id": "a1",
                    "type": "tool_call",
                    "tool": TOOL_NAME,
                    "parameters": {"prompt": "hello from smoke"},
                }
            ]
        })

        check = json.loads(rt.verify_proposal(proposal))
        assert check.get("valid"), f"verify_proposal failed: {check}"

        raw = rt.submit_proposal(proposal)
        result = json.loads(raw)

        assert fired.is_set(), "Python handler was never invoked"
        assert received_prompt == ["hello from smoke"]
        # Daemon wraps our return value as an ActionResult inside results[].
        assert "results" in result, f"unexpected daemon response shape: {result}"
        assert result["results"][0]["action_id"] == "a1"
        assert result["results"][0]["output"]["notes"] == "smoke"
    finally:
        try:
            car_runtime.unregister_tool_handler()
        except Exception:
            pass
