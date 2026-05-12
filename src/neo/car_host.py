"""Host Neo as a CAR-backed Agent2Agent v1.0 endpoint.

Boots a single `CarRuntime` against the local car-server daemon,
registers Neo as the `neo.process` tool, installs the Python
`tools.execute` handler (Parslee-ai/car-releases#38), and binds
the A2A HTTP listener. Blocks until SIGINT/SIGTERM.

Requirements:
- The `car-server` daemon must be running locally (default
  ws://127.0.0.1:9100). Start with `python -m car_runtime.server`
  or `car-server`.
- `car-runtime >= 0.8.0` (provides the Python `register_tool_handler`).

Invoked from the `neo serve` CLI subcommand.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
from typing import TYPE_CHECKING, Optional

from neo.car_tool_schema import (
    TOOL_NAME,
    dict_to_neo_input,
    neo_output_to_dict,
    tool_schema_json,
)

if TYPE_CHECKING:
    from neo.engine import NeoEngine

logger = logging.getLogger(__name__)

# Process-wide shutdown gate. Signal handler flips it; the main thread
# polls so signal delivery doesn't have to interrupt a long wait.
_SHUTDOWN = threading.Event()


def run_server(
    bind: str = "127.0.0.1:9101",
    public_url: Optional[str] = None,
    agent_name: str = "neo",
    agent_description: str = (
        "Read-only code-reasoning helper (MapCoder/CodeSim style). "
        "Returns a plan, simulation traces, and unified-diff code "
        "suggestions for the prompt + provided context."
    ),
    organization: str = "Parslee AI",
    organization_url: str = "https://github.com/Parslee-ai/neo",
) -> int:
    """Bind the A2A listener and serve until killed. Returns exit code."""
    try:
        import car_runtime as cr
    except ImportError:
        print(
            "car-runtime not installed. Install with:\n"
            "    pip install 'neo-reasoner[car]'\n"
            "And start the daemon before launching `neo serve`:\n"
            "    python -m car_runtime.server &",
            file=sys.stderr,
        )
        return 1

    from neo.config import NeoConfig

    config = NeoConfig.load()

    # Single CarRuntime for the server's lifetime — the daemon scopes
    # session state to the WS connection, so one runtime keeps the
    # tool registry, policy set, and (eventual) memgine partition
    # consistent across A2A calls.
    rt = cr.CarRuntime()

    # Declare the tool to the daemon. car-a2a's auto-generated Agent
    # Card lifts the schema into the skill list — peers discover Neo's
    # input shape from /.well-known/agent-card.json with no glue.
    try:
        rt.register_tool_schema(tool_schema_json())
    except Exception as e:
        print(f"register_tool_schema failed: {e}", file=sys.stderr)
        print(
            "Is the car-server daemon running? "
            "Start it with `python -m car_runtime.server &`.",
            file=sys.stderr,
        )
        return 1

    # Per-codebase NeoEngine cache. FactStore is per-codebase
    # (org_id/project_id derived from working_directory + git remote)
    # so we need a fresh engine for each distinct cwd. Keyed by the
    # resolved working_directory string; lock-guarded for the rare
    # concurrent first-call case (CAR's drain task is single-threaded
    # today but that's an implementation detail).
    engines: dict[str, "NeoEngine"] = {}
    engines_lock = threading.Lock()

    def _get_or_create_engine(working_dir: str) -> "NeoEngine":
        with engines_lock:
            engine = engines.get(working_dir)
            if engine is not None:
                return engine
            from neo.adapters import create_adapter
            from neo.engine import NeoEngine

            adapter = create_adapter(
                provider=config.provider,
                model=config.model,
                api_key=config.api_key,
            )
            engine = NeoEngine(
                lm_adapter=adapter,
                codebase_root=working_dir,
                config=config,
            )
            engines[working_dir] = engine
            return engine

    def _handle_call(call_json: str) -> str:
        """`tools.execute` callback. JSON in, JSON out — every error path
        also returns valid JSON so the daemon response stays well-formed."""
        try:
            call = json.loads(call_json)
        except json.JSONDecodeError as e:
            logger.exception("malformed call_json from daemon")
            return json.dumps({
                "error": "MalformedCall",
                "message": f"daemon delivered non-JSON call payload: {e}",
            })

        tool = call.get("tool")
        if tool != TOOL_NAME:
            return json.dumps({
                "error": "UnknownTool",
                "message": f"this handler only serves {TOOL_NAME!r}, got {tool!r}",
            })

        params = call.get("params") or {}
        if not isinstance(params, dict):
            return json.dumps({
                "error": "BadParams",
                "message": f"params must be an object, got {type(params).__name__}",
            })

        try:
            neo_input = dict_to_neo_input(params)
        except Exception as e:
            logger.exception("failed to build NeoInput from params")
            return json.dumps({
                "error": "BadParams",
                "message": str(e),
                "error_type": type(e).__name__,
            })

        if not neo_input.prompt:
            return json.dumps({
                "error": "BadParams",
                "message": "prompt is required and must be a non-empty string",
            })

        working_dir = neo_input.working_directory or os.getcwd()

        try:
            engine = _get_or_create_engine(working_dir)
            output = engine.process(neo_input)
        except Exception as e:
            logger.exception("NeoEngine.process raised")
            return json.dumps({
                "error": "ProcessingError",
                "message": str(e),
                "error_type": type(e).__name__,
            })

        try:
            return json.dumps(neo_output_to_dict(output))
        except (TypeError, ValueError) as e:
            logger.exception("serialization of NeoOutput failed")
            return json.dumps({
                "error": "SerializationError",
                "message": str(e),
            })

    cr.register_tool_handler(_handle_call)

    params: dict[str, object] = {
        "bind": bind,
        "agent_name": agent_name,
        "agent_description": agent_description,
        "organization": organization,
        "organization_url": organization_url,
        # Tell car-server to use THIS session's runtime for the A2A
        # dispatcher. Without it, start_a2a spins up an isolated
        # Runtime that only has register_agent_basics, and our
        # neo.process tool + register_tool_handler callback are
        # invisible to A2A peers. Requires car-server >= the commit
        # that landed Parslee-ai/car#?? (share_session_runtime).
        "share_session_runtime": True,
    }
    if public_url:
        params["public_url"] = public_url

    try:
        result_raw = cr.start_a2a_server(rt, json.dumps(params))
    except Exception as e:
        print(f"start_a2a_server failed: {e}", file=sys.stderr)
        try:
            cr.unregister_tool_handler()
        except Exception:
            pass
        return 1

    try:
        result = json.loads(result_raw)
        bound = result.get("bound", bind)
    except (json.JSONDecodeError, AttributeError):
        bound = bind

    print(f"Neo CAR-native tool '{TOOL_NAME}' registered.", file=sys.stderr)
    print(f"  Daemon: {os.environ.get('CAR_DAEMON_URL', 'ws://127.0.0.1:9100')}", file=sys.stderr)
    print("  Callers using car-runtime can submit proposals against this tool.", file=sys.stderr)
    print(f"Neo A2A endpoint bound at http://{bound}", file=sys.stderr)
    print(
        f"  Agent Card: http://{bound}/.well-known/agent-card.json",
        file=sys.stderr,
    )
    print(
        f"  A2A peers can call {TOOL_NAME} via message/send; "
        "the daemon routes tool dispatch back to this process's "
        "registered handler.",
        file=sys.stderr,
    )
    print("Ctrl-C to stop.", file=sys.stderr)

    def _request_shutdown(signum, _frame):
        logger.info("received signal %d, shutting down", signum)
        _SHUTDOWN.set()

    signal.signal(signal.SIGINT, _request_shutdown)
    signal.signal(signal.SIGTERM, _request_shutdown)

    try:
        # Poll loop so Python signal handlers wake the main thread on
        # platforms where signal.signal alone doesn't interrupt a wait.
        while not _SHUTDOWN.is_set():
            _SHUTDOWN.wait(timeout=1.0)
    finally:
        for action_name, action in (
            ("stop_a2a_server", lambda: cr.stop_a2a_server(rt)),
            ("unregister_tool_handler", cr.unregister_tool_handler),
        ):
            try:
                action()
            except Exception:
                logger.exception("error during shutdown step: %s", action_name)

    return 0
