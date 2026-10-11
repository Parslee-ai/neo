"""Run against an owned fixture daemon and its actual candidate SDK, no inference.

Invoke with PYTHONDONTWRITEBYTECODE=1, an isolated HOME, and PYTHONPATH=src.
Config is mode 0600: {url, owned_home, expected_sdk_sha256,
parents: [{agent_id, token, work_context}, ...]}.
Both originals must use neo.lattice.SCOPED_PARENT_METHODS in their approved native spec.
The fixture owner retires the original parents after this driver closes sockets.
"""
import argparse
import asyncio
import hashlib
import importlib
import json
import logging
import os
from pathlib import Path
import stat
import traceback
from types import SimpleNamespace

from neo.adapters import CarAdapter
from neo.lattice import Answerer, _default_client, _original_receipt
from neo.scoped_consumer import ScopedConsumer


_REFUSALS = frozenset({
    "-32001 consumer revoked; request a new original-context grant",
    "-32001 consumer revoked",
    "consumer refused, expired, revoked, or daemon restarted; request a new original-context grant",
    "-32603 consumer expired, revoked, or daemon restarted; request a new original-context grant",
    "-32603 consumer revoked",
})


def require_sdk_refusal(error, config):
    # Diagnostics classify only exact source-backed literals. This does not
    # change the refusal assertion below or treat transport loss as revocation.
    source_literals = {
        "-32001 consumer revoked; request a new original-context grant": "local_grant_revoked_rpc",
        "-32001 consumer expired, revoked, or daemon restarted; request a new original-context grant": "local_grant_missing_rpc",
        "-32001 consumer revoked; request a new original-context grant; owned inference cleanup remains unconfirmed": "local_grant_revoked_cleanup_unconfirmed_rpc",
        "rpc channel closed for models.route": "transport_channel_closed",
        "daemon closed during scoped consumer authentication": "transport_auth_closed",
    }
    message = error.args[0] if len(error.args) == 1 and type(error.args[0]) is str else None
    config["_sdk_refusal_diagnostic"] = {
        "exception_type": type(error).__name__,
        "argument_count": len(error.args),
        "known_source_literal": source_literals.get(message, "existing_refusal_literal" if message in _REFUSALS else "unclassified"),
        "private_error_sha256": hashlib.sha256(str(error).encode()).hexdigest(),
    }
    # route_model exposes RuntimeError; outages are not revocation evidence.
    assert type(error) is RuntimeError and len(error.args) == 1
    assert type(error.args[0]) is str and error.args[0] in _REFUSALS


async def run(config):
    parents = config["parents"]
    assert len(parents) == 2
    assert parents[0]["work_context"] != parents[1]["work_context"]
    clients, scopes, grants = [], [], []
    observed = config["_observed_original_bindings"] = []
    controls = config["_completed_controls"] = []
    try:
        for parent in parents:
            config["_diagnostic_stage"] = "original-authentication"
            client = _default_client(config["url"])
            clients.append(client)
            auth = await client.connect({"agent_id": parent["agent_id"], "token": parent["token"]})
            task, methods = _original_receipt(auth, parent["agent_id"], parent["work_context"])
            if "expected_task_incarnation" in parent:
                assert task["task_incarnation"] == parent["expected_task_incarnation"]
            # Token-free private evidence permits exact cleanup after native
            # bootstrap's null resource is adopted by original authentication.
            observed.append({"agent_id": parent["agent_id"],
                             "work_context": parent["work_context"],
                             "task_incarnation": task["task_incarnation"],
                             "method_allowlist": list(methods),
                             "resource_key": task["resource_key"]})
            config["_diagnostic_stage"] = "consumer-issuance"
            grant = await client.call("session.consumer.issue", {
                "methods": ["infer", "models.route"], "ttl_secs": 900,
            })
            grants.append(grant)
            config["_diagnostic_stage"] = "sdk-child-construction"
            scope = await asyncio.to_thread(
                ScopedConsumer.from_issued, config["url"], json.dumps(grant),
                parent["agent_id"], task["resource_key"], parent["work_context"],
                Path.home() / "scoped-sdk-smoke",
            )
            scopes.append(scope)
            config["_diagnostic_stage"] = "active-private-state-admission"
            await asyncio.to_thread(scope.prepare_state_root)
        assert scopes[0].state_root != scopes[1].state_root
        controls.extend(["actual_original_receipts", "actual_sdk_child_admission", "private_partitions"])
        loop = asyncio.get_running_loop()
        config["_diagnostic_stage"] = "active-direct-disclosure"
        adapter = CarAdapter(scoped_consumer=scopes[0])

        adapter._generate = lambda *args: "fixture-private-result"
        assert await asyncio.to_thread(adapter.generate, []) == "fixture-private-result"
        config["_diagnostic_stage"] = "active-private-memory"
        answerer = Answerer(str(Path.home()), scoped_consumer=scopes[0])
        reads = []
        def retrieve(*args, **kwargs):
            reads.append(True)
            return []
        answerer._store = SimpleNamespace(retrieve_relevant=retrieve)
        baseline = await asyncio.to_thread(answerer.memory, "fixture-private-query")
        assert reads == [True] and "fixture-private-query" in baseline
        reads.clear()
        controls.append("active_private_read_and_disclosure_succeed")
        generated = []

        def revoke_during_result(*args):
            # Use the original A socket; do not obtain ambient operator authority.
            revoked = asyncio.run_coroutine_threadsafe(clients[0].call(
                "session.consumer.revoke", {"consumer_id": grants[0]["consumer_id"]}
            ), loop).result(timeout=10)
            assert revoked.get("revoked") is True
            generated.append(True)
            return "fixture-private-result"

        config["_diagnostic_stage"] = "revoked-direct-disclosure"
        adapter._generate = revoke_during_result
        try:
            await asyncio.to_thread(adapter.generate, [])
        except Exception as error:
            assert generated, "Revocation must succeed and private work finish before disclosure check"
            require_sdk_refusal(error, config)
            controls.append("actual_revocation_blocks_direct_disclosure")
        else:
            raise AssertionError("Revoked private result escaped")
        config["_diagnostic_stage"] = "revoked-private-memory"
        try:
            await asyncio.to_thread(answerer.memory, "fixture-private-query")
        except Exception as error:
            require_sdk_refusal(error, config)
            assert not reads
            controls.append("actual_revocation_blocks_cached_private_read")
        else:
            raise AssertionError("Revoked private read was admitted")
        config["_diagnostic_stage"] = "other-organization-routing"
        await asyncio.to_thread(scopes[1].authorize)
        controls.append("other_org_actual_routing_continues")
        config["_diagnostic_stage"] = "retained-original-sockets"
        for client in clients:
            assert client.connected
            decision = await client.call("models.route", {
                "prompt": "Validate retained original coordination socket",
                "intent": {"task": "code"},
            })
            assert isinstance(decision, dict)
        controls.append("both_original_sockets_retained")
        return controls
    finally:
        for scope in scopes:
            await asyncio.to_thread(scope.runtime.close)
        for client in clients:
            await client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
    info = args.fixture_config.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077
    config = json.loads(args.fixture_config.read_text())
    # Never print config, issued credentials, exception payloads or private text.
    logging.disable(logging.CRITICAL)
    try:
        assert Path.home().resolve() == Path(config["owned_home"]).resolve()
        home_info = Path.home().lstat()
        assert stat.S_ISDIR(home_info.st_mode) and home_info.st_uid == os.getuid() and not home_info.st_mode & 0o077
        sdk = importlib.import_module("car_runtime.car_runtime")
        sdk_sha = hashlib.sha256(Path(sdk.__file__).read_bytes()).hexdigest()
        assert sdk_sha == config["expected_sdk_sha256"]
        controls = asyncio.run(run(config))
    except Exception as error:
        frame = traceback.extract_tb(error.__traceback__)[-1]
        args.receipt.write_text(json.dumps({"status": "FAILED", "provider_calls": 0,
            "original_bindings": config.get("_observed_original_bindings", []),
            "controls_completed": config.get("_completed_controls", []),
            "sdk_refusal_diagnostic": config.get("_sdk_refusal_diagnostic"),
            "stage": config.get("_diagnostic_stage", "entry-validation"),
            "exception_type": type(error).__name__,
            "source_location": {"file": Path(frame.filename).name,
                                "function": frame.name, "line": frame.lineno},
            "private_error_sha256": hashlib.sha256(str(error).encode()).hexdigest()}) + "\n")
        raise SystemExit(1) from None
    args.receipt.write_text(json.dumps({
        "status": "PASSED", "controls": controls, "provider_calls": 0,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sdk_sha256": sdk_sha,
        "original_bindings": config["_observed_original_bindings"],
        "synthetic_fixture_only": True, "real_user_acceptance": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
