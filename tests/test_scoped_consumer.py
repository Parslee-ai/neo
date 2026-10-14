"""Original-context integration controls; mocks make no provider calls."""

from dataclasses import FrozenInstanceError
import json
import threading
from unittest.mock import patch
from types import SimpleNamespace

import pytest

from neo.adapters import CarAdapter, resolve_adapter
from neo.engine import NeoEngine
from neo.scoped_consumer import ScopedConsumer


class Runtime:
    def __init__(self):
        self.calls = []
        self.refused = False

    def route_model(self, prompt, intent_json):
        self.calls.append((prompt, intent_json))
        if self.refused:
            raise PermissionError("original consumer revoked")
        return json.dumps({"candidates": []})


def scope(tmp_path, org="a", runtime=None):
    runtime = runtime or Runtime()
    context = {"api_base": "https://api.example", "account_id": "alice", "organization_id": org}
    calls = []

    def factory(*args):
        calls.append(args)
        return runtime

    binding = ScopedConsumer.from_issued(
        "ws://127.0.0.1:9100",
        "fixture-issued-receipt",
        "original-parent",
        "context-private:work-" + "a" * 64,
        context,
        tmp_path,
        factory=factory,
    )
    return binding, context, calls


def test_factory_receives_original_identity_and_captured_context(tmp_path):
    binding, context, calls = scope(tmp_path)
    context["organization_id"] = "selected-b"
    assert json.loads(binding.context_json)["organization_id"] == "a"
    assert calls[0][2:4] == (binding.parent_agent_id, binding.resource_key)
    assert calls[0][4] == binding.context_json
    with pytest.raises(FrozenInstanceError):
        binding.resource_key = "other"


def test_two_orgs_same_repository_have_private_banks_and_revocation_isolated(tmp_path):
    a, _, _ = scope(tmp_path, "a")
    b, _, _ = scope(tmp_path, "b")
    assert a.prepare_state_root() != b.prepare_state_root()
    assert a.state_root.stat().st_mode & 0o077 == 0
    a.runtime.refused = True
    with pytest.raises(PermissionError):
        a.authorize()
    b.authorize()


def test_scoped_adapter_never_resolves_personal_provider(tmp_path):
    binding, _, _ = scope(tmp_path)
    with (
        patch("neo.adapters.create_adapter", side_effect=AssertionError("personal provider")),
        patch("neo.car_inference.get_runtime", side_effect=AssertionError("global runtime")),
    ):
        adapter = resolve_adapter(object(), scoped_consumer=binding)
        assert isinstance(adapter, CarAdapter)
        assert adapter._rt is binding.runtime
        binding.runtime.refused = True
        with pytest.raises(PermissionError):
            resolve_adapter(object(), scoped_consumer=binding)


def test_revoked_scope_blocks_process_before_any_private_read(tmp_path):
    binding, _, _ = scope(tmp_path)
    engine = object.__new__(NeoEngine)
    engine.scoped_consumer = binding
    engine._process_lock = threading.Lock()
    engine._process_guarded = lambda _: pytest.fail("private read after refusal")
    binding.runtime.refused = True
    with pytest.raises(PermissionError):
        engine.process(object())
    assert not engine._process_lock.locked()


def test_routing_uses_same_facade_and_refusal_is_terminal(tmp_path):
    binding, _, _ = scope(tmp_path)
    engine = object.__new__(NeoEngine)
    engine.scoped_consumer = binding
    with patch("neo.car_inference.get_runtime", side_effect=AssertionError("global runtime")):
        available, count, route = engine._car_route_capability("code")
        assert available and count == 0 and callable(route)
        binding.runtime.refused = True
        with pytest.raises(PermissionError):
            engine._car_route_capability("code")


def test_role_factory_carries_binding_and_does_not_swallow_refusal(tmp_path):
    binding, _, _ = scope(tmp_path)
    engine = object.__new__(NeoEngine)
    engine.scoped_consumer = binding
    engine.lm = CarAdapter(runtime=binding.runtime)
    with (
        patch("neo.panel.plan_role_models", return_value={"planner": "catalog/code"}),
        patch(
            "neo.panel.build_role_factory", side_effect=lambda models, factory, fallback: factory
        ),
    ):
        factory = engine._build_car_role_factory(lambda *_: None, "code")
        assert factory("catalog/code")._rt is binding.runtime
    with patch("neo.panel.plan_role_models", side_effect=PermissionError("revoked")):
        with pytest.raises(PermissionError):
            engine._build_car_role_factory(lambda *_: None, "code")


def test_metrics_partition_is_restored_after_failure(tmp_path):
    from neo.memory.metrics import _metrics_path, scoped_metrics

    before = _metrics_path()
    with pytest.raises(RuntimeError):
        with scoped_metrics(tmp_path / "original" / "metrics.jsonl"):
            assert _metrics_path() == tmp_path / "original" / "metrics.jsonl"
            raise RuntimeError("canceled")
    assert _metrics_path() == before


def test_engine_partitions_fact_and_episode_store(tmp_path):
    binding, _, _ = scope(tmp_path)
    with (
        patch(
            "neo.memory.store.FactStore", return_value=SimpleNamespace(project_id="repo")
        ) as facts,
        patch("neo.memory.episodes.LearningEpisodeStore") as episodes,
    ):
        NeoEngine(
            resolve_adapter(object(), scoped_consumer=binding),
            codebase_root=str(tmp_path),
            scoped_consumer=binding,
        )
    assert facts.call_args.kwargs["facts_dir"] == binding.state_root / "facts"
    assert facts.call_args.kwargs["episodes_dir"] == binding.state_root / "episodes"
    assert facts.call_args.kwargs["emit_metrics"] is False
    assert facts.call_args.kwargs["read_only"] is True
    assert episodes.call_args.kwargs["base_dir"] == binding.state_root / "episodes"


def test_revoked_answerer_does_not_disclose_cached_memory(tmp_path):
    from neo.lattice import Answerer

    binding, _, _ = scope(tmp_path)
    answerer = Answerer(str(tmp_path), scoped_consumer=binding)
    answerer._store = SimpleNamespace(
        retrieve_relevant=lambda *a, **kw: pytest.fail("private read")
    )
    binding.runtime.refused = True
    with pytest.raises(PermissionError):
        answerer.memory("private")


@pytest.mark.asyncio
async def test_node_issues_child_on_original_socket_and_pins_reconnect(tmp_path):
    from neo.lattice import LatticeNode, NodeConfig, SCOPED_PARENT_METHODS
    import time

    binding, context, _ = scope(tmp_path)
    auth = {
        "ok": True,
        "auth_enabled": True,
        "agent_id": "original-parent",
        "work_context": context,
        "task_binding": {
            "version": 1,
            "task_incarnation": "01234567-89ab-4cde-8fab-0123456789ab",
            "resource_key": binding.resource_key,
        },
        "method_allowlist": list(SCOPED_PARENT_METHODS),
    }
    calls = []

    class Client:
        connected = True

        def on_request(self, *args):
            pass

        async def connect(self, params):
            assert params == {"agent_id": "original-parent", "token": "fixture-parent-token"}
            return auth

        async def call(self, method, params, timeout=None):
            calls.append((method, params))
            now = int(time.time() * 1000)
            return (
                {
                    "token": "fixture-child-token",
                    "consumer_id": "fixture-child",
                    "issued_at_ms": now,
                    "expires_at_ms": now + 900000,
                }
                if method == "session.consumer.issue"
                else {}
            )

    config = NodeConfig(
        root=str(tmp_path),
        project="repo",
        agent_id="original-parent",
        token="fixture-parent-token",
        url="ws://127.0.0.1:9100",
        work_context_json=json.dumps(context),
    )
    node = LatticeNode(config, client_factory=lambda _: Client())
    try:
        with patch.object(ScopedConsumer, "from_issued", return_value=binding) as factory:
            await node.connect_once()
            assert calls[0] == (
                "session.consumer.issue",
                {"methods": ["infer", "models.route"], "ttl_secs": 900},
            )
            assert factory.call_args.args[2:5] == ("original-parent", binding.resource_key, context)
            assert node.answerer.scoped_consumer is binding
            auth["task_binding"] = {**auth["task_binding"], "task_incarnation": "replacement-task"}
            count = len(calls)
            with pytest.raises(PermissionError):
                await node.connect_once()
            assert len(calls) == count
            auth["work_context"] = {**context, "organization_id": "b"}
            config.work_context_json = json.dumps(auth["work_context"])
            with pytest.raises(PermissionError):
                await node.connect_once()
            assert len(calls) == count
    finally:
        node._executor.shutdown(wait=False, cancel_futures=True)


def test_bound_consumer_cannot_autojoin_personal_node(monkeypatch):
    from neo.lattice import maybe_autojoin

    monkeypatch.setenv("CAR_WORK_CONTEXT", "{}")
    with patch(
        "neo.memory.observer._require_car_runtime", side_effect=AssertionError("ambient runtime")
    ):
        assert maybe_autojoin("/fixture/repo") is None


@pytest.mark.asyncio
async def test_bound_auth_without_original_launch_context_is_rejected():
    from neo.lattice import LatticeNode, NodeConfig

    class Client:
        def on_request(self, *args):
            pass

        async def connect(self, *args):
            return {"ok": True, "work_context": {"organization_id": "a"}}

        async def call(self, *args, **kwargs):
            pytest.fail("must refuse before join or issuance")

    node = LatticeNode(
        NodeConfig(
            root="/fixture/repo",
            project="fixture",
            agent_id="original",
            token="fixture",
            url="ws://fixture",
        ),
        client_factory=lambda _: Client(),
    )
    with pytest.raises(PermissionError, match="original context"):
        await node.connect_once()


@pytest.mark.parametrize("entry", ["adapter", "engine"])
def test_revocation_during_work_refuses_direct_result_disclosure(tmp_path, entry):
    binding, _, _ = scope(tmp_path)

    def private_work(*args):
        binding.runtime.refused = True
        return "private result must not escape"

    if entry == "adapter":
        consumer = CarAdapter(runtime=binding.runtime, scoped_consumer=binding)
        consumer._generate = private_work

        def invoke():
            return consumer.generate([])
    else:
        consumer = object.__new__(NeoEngine)
        consumer.scoped_consumer = binding
        consumer._process_lock = threading.Lock()
        consumer._process_guarded = private_work

        def invoke():
            from neo.models import NeoInput
            from neo.operating_mode import OperatingMode

            return consumer.process(NeoInput(prompt="fixture", operating_mode=OperatingMode.ADVISE))

    with pytest.raises(PermissionError):
        invoke()
    if entry == "engine":
        assert not consumer._process_lock.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "revoke_mode", ["revoke", "rpc_failure", "false_receipt", "original_disconnect"]
)
async def test_sdk_smoke_only_counts_confirmed_revocation(tmp_path, monkeypatch, revoke_mode):
    import importlib.util
    from neo.lattice import SCOPED_PARENT_METHODS
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "scoped_sdk_smoke", Path(__file__).parents[1] / "scripts/scoped_sdk_smoke.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setenv("HOME", str(tmp_path))

    class SdkRuntime(Runtime):
        def route_model(self, *args):
            if self.refused:
                raise RuntimeError(
                    "consumer refused, expired, revoked, or daemon restarted; request a new original-context grant"
                )
            return super().route_model(*args)

    bindings = [scope(tmp_path, "a", SdkRuntime())[0], scope(tmp_path, "b", SdkRuntime())[0]]
    parents = [
        {"agent_id": str(i), "token": "fixture-token", "work_context": json.loads(b.context_json)}
        for i, b in enumerate(bindings)
    ]

    class Client:
        connected = True

        async def connect(self, params):
            self.index = int(params["agent_id"])
            return {
                "ok": True,
                "auth_enabled": True,
                **parents[self.index],
                "task_binding": {
                    "version": 1,
                    "task_incarnation": "01234567-89ab-4cde-8fab-0123456789ab",
                    "resource_key": bindings[self.index].resource_key,
                },
                "method_allowlist": list(SCOPED_PARENT_METHODS),
            }

        async def call(self, method, params):
            if method == "session.consumer.issue":
                return {"consumer_id": str(self.index)}
            if method == "models.route":
                if revoke_mode == "original_disconnect":
                    raise ConnectionError("original socket cannot answer")
                return {"candidates": []}
            assert method == "session.consumer.revoke"
            if revoke_mode == "rpc_failure":
                raise ConnectionError("fixture RPC failed")
            if revoke_mode in {"revoke", "original_disconnect"}:
                bindings[0].runtime.refused = True
            return {"revoked": revoke_mode in {"revoke", "original_disconnect"}}

        async def close(self):
            self.connected = False

    for b in bindings:
        b.runtime.close = lambda: None
    monkeypatch.setattr(driver, "_default_client", lambda _: Client())
    monkeypatch.setattr(driver.ScopedConsumer, "from_issued", lambda *args: bindings[int(args[2])])
    config = {"url": "ws://fixture", "parents": parents}
    if revoke_mode == "revoke":
        controls = await driver.run(config)
        assert "actual_revocation_blocks_direct_disclosure" in controls
        assert "other_org_actual_routing_continues" in controls
    elif revoke_mode == "original_disconnect":
        with pytest.raises(ConnectionError, match="original socket"):
            await driver.run(config)
    else:
        with pytest.raises(AssertionError, match="Revocation must succeed"):
            await driver.run(config)


@pytest.mark.parametrize("has_facts", [False, True])
def test_revocation_during_direct_memory_read_blocks_disclosure(tmp_path, has_facts):
    from neo.lattice import Answerer
    from neo.memory.models import FactScope

    binding, _, _ = scope(tmp_path)
    answerer = Answerer(str(tmp_path), scoped_consumer=binding)
    retrieved = []

    def retrieve(*args, **kwargs):
        retrieved.append(True)
        binding.runtime.refused = True
        return (
            [
                SimpleNamespace(
                    id="private",
                    scope=FactScope.PROJECT,
                    subject="fixture-private-subject",
                    kind="lesson",
                    metadata=SimpleNamespace(confidence=None),
                    tags=[],
                    body="fixture-private-body",
                )
            ]
            if has_facts
            else []
        )

    answerer._store = SimpleNamespace(retrieve_relevant=retrieve)
    with pytest.raises(PermissionError):
        answerer.memory("fixture-private-query")
    assert retrieved == [True]


@pytest.mark.parametrize(
    "error",
    [ConnectionError("outage"), AssertionError("unrelated"), RuntimeError("unrelated SDK failure")],
)
def test_actual_sdk_smoke_refuses_unrelated_errors(error):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "scoped_sdk_smoke", Path(__file__).parents[1] / "scripts/scoped_sdk_smoke.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    with pytest.raises(AssertionError):
        driver.require_sdk_refusal(error, {})


def test_direct_scoped_adapter_binds_omitted_runtime_and_rejects_mismatch(tmp_path):
    binding, _, _ = scope(tmp_path)
    with patch("neo.car_inference.get_runtime", side_effect=AssertionError("global runtime")):
        adapter = CarAdapter(scoped_consumer=binding)
        assert adapter._rt is binding.runtime
        with pytest.raises(ValueError, match="admitted consumer runtime"):
            CarAdapter(runtime=Runtime(), scoped_consumer=binding)


@pytest.mark.parametrize("mode", ["learn", "agent", "patch", "verify"])
def test_scoped_engine_refuses_non_advisory_work_before_side_effects(tmp_path, mode):
    from neo.models import NeoInput
    from neo.operating_mode import OperatingMode, ModeValidationError

    binding, _, _ = scope(tmp_path)
    engine = object.__new__(NeoEngine)
    engine.scoped_consumer = binding
    engine._process_lock = threading.Lock()
    engine._process_guarded = lambda _: pytest.fail("execution, learning or private read")
    with pytest.raises(ModeValidationError, match="ADVISE"):
        engine.process(NeoInput(prompt="fixture", operating_mode=OperatingMode(mode)))
    assert not engine._process_lock.locked()


def test_scoped_bugfix_answer_is_advisory_while_personal_learning_remains(tmp_path):
    from neo.lattice import Answerer, Task
    from neo.models import TaskType
    from neo.operating_mode import OperatingMode

    binding, _, _ = scope(tmp_path)
    observed = []
    engine = SimpleNamespace(process=lambda request: observed.append(request) or object())
    with (
        patch("neo.models.classify_task_type", return_value=TaskType.BUGFIX),
        patch("neo.lattice.render_output", return_value="fixture answer"),
    ):
        for consumer in [binding, None]:
            answerer = Answerer(str(tmp_path), scoped_consumer=consumer)
            answerer._engine = engine
            assert answerer.reason(Task("reason", prompt="fix a defect")) == "fixture answer"
    assert [r.operating_mode for r in observed] == [OperatingMode.ADVISE, OperatingMode.LEARN]


def test_scoped_native_spec_has_explicit_original_parent_profile(tmp_path):
    from neo.lattice import build_spec, SCOPED_PARENT_METHODS

    binding, context, _ = scope(tmp_path)
    with patch("neo.lattice.agent_id_for", return_value="neo-fixture"):
        scoped = build_spec(str(tmp_path), "repo", context)
        personal = build_spec(str(tmp_path), "repo")
    context["organization_id"] = "changed"
    assert scoped["work_context"] == json.loads(binding.context_json)
    assert scoped["method_allowlist"] == list(SCOPED_PARENT_METHODS)
    assert "session.consumer.issue" in scoped["method_allowlist"]
    assert "session.consumer.revoke" in scoped["method_allowlist"]
    assert "method_allowlist" not in personal
    assert scoped["id"] != personal["id"]


@pytest.mark.parametrize(
    "defect",
    [
        "nil_task",
        "uppercase_task",
        "extra_task_field",
        "boolean_version",
        "resource_case",
        "missing_methods",
        "extra_method",
        "blank_method",
    ],
)
def test_primary_receipt_refuses_noncanonical_or_unapproved_authority(tmp_path, defect):
    from neo.lattice import _original_receipt, SCOPED_PARENT_METHODS

    binding, context, _ = scope(tmp_path)
    task = {
        "version": 1,
        "task_incarnation": "01234567-89ab-4cde-8fab-0123456789ab",
        "resource_key": binding.resource_key,
    }
    auth = {
        "ok": True,
        "auth_enabled": True,
        "agent_id": "original",
        "work_context": context,
        "task_binding": task,
        "method_allowlist": list(SCOPED_PARENT_METHODS),
    }
    if defect == "nil_task":
        task["task_incarnation"] = "00000000-0000-0000-0000-000000000000"
    elif defect == "uppercase_task":
        task["task_incarnation"] = task["task_incarnation"].upper()
    elif defect == "extra_task_field":
        task["other"] = "authority"
    elif defect == "boolean_version":
        task["version"] = True
    elif defect == "resource_case":
        task["resource_key"] = task["resource_key"].upper()
    elif defect == "missing_methods":
        auth.pop("method_allowlist")
    elif defect == "extra_method":
        auth["method_allowlist"].append("host.restart")
    elif defect == "blank_method":
        auth["method_allowlist"].append(" ")
    with pytest.raises(PermissionError):
        _original_receipt(auth, "original", context)


def lifecycle_node(tmp_path, monkeypatch):
    import time
    from neo.lattice import LatticeNode, NodeConfig, SCOPED_PARENT_METHODS

    context = {"api_base": "https://fixture.invalid", "account_id": "alice", "organization_id": "a"}
    auth = {
        "ok": True,
        "auth_enabled": True,
        "agent_id": "original",
        "work_context": context,
        "task_binding": {
            "version": 1,
            "task_incarnation": "01234567-89ab-4cde-8fab-0123456789ab",
            "resource_key": "context-private:work-" + "a" * 64,
        },
        "method_allowlist": list(SCOPED_PARENT_METHODS),
    }
    calls, runtimes, clients = [], [], []

    class Child(Runtime):
        def __init__(self):
            super().__init__()
            self.closes = 0

        def close(self):
            self.closes += 1

    class Client:
        connected = True
        retire_ok = True

        def __init__(self):
            self.handlers = {}

        def on_request(self, method, handler):
            self.handlers[method] = handler

        async def connect(self, params):
            assert params == {"agent_id": "original", "token": "fixture-parent-token"}
            return auth

        async def call(self, method, params, timeout=None):
            calls.append((self, method, params))
            if method == "session.consumer.issue":
                now = int(time.time() * 1000)
                return {
                    "consumer_id": f"child-{len(runtimes)}",
                    "issued_at_ms": now,
                    "expires_at_ms": now + 900000,
                    "token": "mock-child-token",
                }
            if method == "session.consumer.revoke":
                return {"revoked": self.retire_ok}
            return {}

        async def close(self):
            self.connected = False

    def client_factory(url):
        assert url == "ws://fixture"
        client = Client()
        clients.append(client)
        return client

    original_factory = ScopedConsumer.from_issued

    def factory(*args):
        runtime = Child()
        runtimes.append(runtime)
        return original_factory(*args, factory=lambda *a: runtime)

    monkeypatch.setattr(ScopedConsumer, "from_issued", factory)
    node = LatticeNode(
        NodeConfig(
            root=str(tmp_path),
            project="repo",
            agent_id="original",
            token="fixture-parent-token",
            url="ws://fixture",
            work_context_json=json.dumps(context),
        ),
        client_factory=client_factory,
    )
    return node, auth, calls, runtimes, clients


@pytest.mark.asyncio
async def test_child_renewal_preserves_original_inbox_context_and_private_bank(
    tmp_path, monkeypatch
):
    import asyncio

    node, auth, calls, runtimes, clients = lifecycle_node(tmp_path, monkeypatch)
    try:
        await node.connect_once()
        first = node._scoped_consumer
        node.queue = asyncio.Queue()
        queue = node.queue
        client = node.client
        node._consumer_renew_at = 0
        await node.renew_consumer_if_due()
        assert node.client is client and client.connected
        assert node.queue is queue and len(clients) == 1
        assert client.handlers["agent.peer_message"] == node.on_peer_message
        assert node._scoped_consumer.state_root == first.state_root
        assert node._scoped_consumer.context_json == first.context_json
        assert runtimes[0].closes == 1 and runtimes[1].closes == 0
        assert [m for _, m, _ in calls].count("session.consumer.issue") == 2
        assert [p for _, m, p in calls if m == "session.consumer.revoke"] == [
            {"consumer_id": "child-0"}
        ]
        assert all(c is client for c, _, _ in calls)
        await node._drop_consumer(client)
        assert runtimes[1].closes == 1 and node._scoped_consumer is None
        assert [p for _, m, p in calls if m == "session.consumer.revoke"][-1] == {
            "consumer_id": "child-1"
        }
    finally:
        node._executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_renewal_waits_for_work_using_the_current_child(tmp_path, monkeypatch):
    import asyncio

    node, _, calls, runtimes, _ = lifecycle_node(tmp_path, monkeypatch)
    try:
        await node.connect_once()
        node._consumer_renew_at = 0
        async with node._consumer_lock:
            renewal = asyncio.create_task(node.renew_consumer_if_due())
            await asyncio.sleep(0)
            assert not renewal.done() and len(runtimes) == 1
        await renewal
        assert len(runtimes) == 2 and runtimes[0].closes == 1
    finally:
        await node._drop_consumer(node.client)
        node._executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_revoked_child_cannot_trigger_reissue_or_reconnect_fallback(tmp_path, monkeypatch):
    node, _, calls, runtimes, _ = lifecycle_node(tmp_path, monkeypatch)
    try:
        await node.connect_once()
        node._consumer_renew_at = 0
        runtimes[0].refused = True
        with pytest.raises(PermissionError):
            await node.renew_consumer_if_due()
        with pytest.raises(PermissionError):
            await node.connect_once()
        assert len(runtimes) == 1
        assert [m for _, m, _ in calls].count("session.consumer.issue") == 1
    finally:
        await node._drop_consumer(node.client)
        node._executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_original_reconnect_rejects_retargeting_and_accepts_equivalent_scope(
    tmp_path, monkeypatch
):
    node, auth, calls, runtimes, clients = lifecycle_node(tmp_path, monkeypatch)
    try:
        await node.connect_once()
        first_bank = node._scoped_consumer.state_root
        await node._drop_consumer(node.client)
        await node.client.close()
        auth["method_allowlist"] = list(reversed(auth["method_allowlist"])) + [
            auth["method_allowlist"][0]
        ]
        node.config.agent_id = "replacement"
        node.config.token = "replacement"
        node.config.url = "ws://other"
        await node.connect_once()
        assert node._scoped_consumer.state_root == first_bank and len(runtimes) == 2
        auth["task_binding"]["task_incarnation"] = "11234567-89ab-4cde-8fab-0123456789ab"
        count = len(calls)
        with pytest.raises(PermissionError, match="changed on reconnect"):
            await node.connect_once()
        assert len(calls) == count
    finally:
        await node._drop_consumer(node.client)
        node._executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_unconfirmed_old_child_retirement_does_not_publish_replacement(tmp_path, monkeypatch):
    node, _, calls, runtimes, _ = lifecycle_node(tmp_path, monkeypatch)
    try:
        await node.connect_once()
        old = node._scoped_consumer
        node.client.retire_ok = False
        node._consumer_renew_at = 0
        with pytest.raises(PermissionError, match="retirement unconfirmed"):
            await node.renew_consumer_if_due()
        assert node._scoped_consumer is old
        assert runtimes[0].closes == 0 and runtimes[1].closes == 1
        assert [p for _, m, p in calls if m == "session.consumer.revoke"] == [
            {"consumer_id": "child-0"},
            {"consumer_id": "child-1"},
        ]
    finally:
        await node._drop_consumer(node.client)
        node._executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_cancelled_sdk_attach_closes_and_retires_unpublished_child(tmp_path, monkeypatch):
    import asyncio

    node, _, calls, runtimes, _ = lifecycle_node(tmp_path, monkeypatch)
    original_factory = ScopedConsumer.from_issued
    started, release = threading.Event(), threading.Event()

    def blocked_factory(*args):
        started.set()
        assert release.wait(timeout=5)
        return original_factory(*args)

    monkeypatch.setattr(ScopedConsumer, "from_issued", blocked_factory)
    attach = asyncio.create_task(node.connect_once())
    try:
        assert await asyncio.to_thread(started.wait, 5)
        attach.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await attach
        assert node._scoped_consumer is None and runtimes[0].closes == 1
        assert [p for _, m, p in calls if m == "session.consumer.revoke"] == [
            {"consumer_id": "child-0"}
        ]
    finally:
        release.set()
        await asyncio.gather(attach, return_exceptions=True)
        node._executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_failed_sdk_attach_retires_grant_without_publishing_binding(tmp_path, monkeypatch):
    node, _, calls, runtimes, _ = lifecycle_node(tmp_path, monkeypatch)

    def refused_factory(*args):
        raise PermissionError("mock SDK original context refusal")

    monkeypatch.setattr(ScopedConsumer, "from_issued", refused_factory)
    try:
        with pytest.raises(PermissionError):
            await node.connect_once()
        assert node._scoped_consumer is None and runtimes == []
        assert [p for _, m, p in calls if m == "session.consumer.revoke"] == [
            {"consumer_id": "child-0"}
        ]
    finally:
        node._executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_disconnected_parent_cannot_renew_and_original_recovery_retains_bank(
    tmp_path, monkeypatch
):
    node, _, calls, runtimes, _ = lifecycle_node(tmp_path, monkeypatch)
    try:
        await node.connect_once()
        bank = node._scoped_consumer.state_root
        node.client.connected = False
        node._consumer_renew_at = 0
        with pytest.raises(ConnectionError, match="Original Neo coordination"):
            await node.renew_consumer_if_due()
        assert len(runtimes) == 1 and not node._consumer_refused
        await node._drop_consumer(node.client)
        await node.connect_once()
        assert len(runtimes) == 2 and node._scoped_consumer.state_root == bank
    finally:
        await node._drop_consumer(node.client)
        node._executor.shutdown(wait=False, cancel_futures=True)


def test_admitted_adapter_inference_uses_the_same_scoped_runtime(tmp_path):
    runtime = Runtime()
    inference = []

    def infer(request):
        inference.append(json.loads(request))
        return json.dumps({"text": "fixture scoped reply"})

    runtime.infer_tracked_with_request = infer
    binding, _, _ = scope(tmp_path, runtime=runtime)
    with patch("neo.car_inference.get_runtime", side_effect=AssertionError("personal authority")):
        assert (
            CarAdapter(scoped_consumer=binding).generate("fixture scoped prompt")
            == "fixture scoped reply"
        )
    assert inference[0]["prompt"] == "fixture scoped prompt"
    assert len(runtime.calls) == 2


def test_scoped_advisory_pipeline_never_executes_or_promotes_learning(tmp_path):
    from tests.test_operating_modes import _ModeLM, _Executor
    from neo.models import NeoInput
    from neo.operating_mode import OperatingMode

    model = _ModeLM()
    runtime = Runtime()
    runtime.infer_tracked_with_request = lambda request: json.dumps({"text": model.generate([])})
    binding, _, _ = scope(tmp_path, runtime=runtime)
    executor = _Executor()
    engine = NeoEngine(
        CarAdapter(scoped_consumer=binding),
        enable_persistent_memory=False,
        codebase_root=str(tmp_path),
        scoped_consumer=binding,
        execution_adapter=executor,
    )
    engine._car_route_capability = lambda prompt: (False, 0, None)
    engine._run_static_checks = lambda suggestions, constraints=None: []
    engine._store_reasoning = lambda *args: pytest.fail("learning promotion")
    output = engine.process(NeoInput(prompt="advise me", operating_mode=OperatingMode.ADVISE))
    episode = engine.episode_store.load(output.metadata["learning_episode_id"])
    assert output.metadata["learning_enabled_for_request"] is False
    assert episode.final_outcome == "advised_no_learning" and episode.memory_candidates == []
    assert model.calls > 0 and executor.calls == 0
    assert engine.episode_store.path.parent == binding.state_root / "episodes"


@pytest.mark.asyncio
async def test_malformed_child_lifetime_refuses_factory_and_retires_exact_grant(
    tmp_path, monkeypatch
):
    node, _, calls, runtimes, clients = lifecycle_node(tmp_path, monkeypatch)
    original_factory = node._client_factory

    def factory(url):
        client = original_factory(url)
        call = client.call

        async def invalid(method, params, timeout=None):
            result = await call(method, params, timeout)
            if method == "session.consumer.issue":
                result["expires_at_ms"] = True
            return result

        client.call = invalid
        return client

    node._client_factory = factory
    try:
        with pytest.raises(PermissionError, match="lifetime receipt"):
            await node.connect_once()
        assert runtimes == [] and node._scoped_consumer is None
        assert [p for _, m, p in calls if m == "session.consumer.revoke"] == [
            {"consumer_id": "child-0"}
        ]
    finally:
        node._executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [ConnectionError("mock outage"), RuntimeError("rpc channel closed for models.route")]
)
async def test_child_transport_loss_recovers_only_through_original_parent(
    tmp_path, monkeypatch, error
):
    node, _, calls, runtimes, _ = lifecycle_node(tmp_path, monkeypatch)
    try:
        await node.connect_once()
        bank = node._scoped_consumer.state_root

        def unavailable(*args):
            raise error

        runtimes[0].route_model = unavailable
        node._consumer_renew_at = 0
        with pytest.raises(ConnectionError):
            await node.renew_consumer_if_due()
        assert not node._consumer_refused and len(runtimes) == 1
        await node._drop_consumer(node.client)
        await node.client.close()
        await node.connect_once()
        assert node._scoped_consumer.state_root == bank and len(runtimes) == 2
    finally:
        await node._drop_consumer(node.client)
        node._executor.shutdown(wait=False, cancel_futures=True)


@pytest.mark.asyncio
async def test_running_node_stops_on_refusal_without_issuing_another_child(tmp_path, monkeypatch):
    import asyncio
    from neo import lattice

    node, _, calls, runtimes, clients = lifecycle_node(tmp_path, monkeypatch)
    monkeypatch.setattr(lattice, "POLL_SECS", 0.001)
    running = asyncio.create_task(node.run())
    try:
        await asyncio.wait_for(node._connected.wait(), timeout=5)
        runtimes[0].refused = True
        node._consumer_renew_at = 0
        await asyncio.wait_for(running, timeout=5)
        assert node._consumer_refused and node._stop.is_set()
        assert len(clients) == 1 and not clients[0].connected
        assert len(runtimes) == 1 and runtimes[0].closes == 1
        assert node._scoped_consumer is None
        assert [m for _, m, _ in calls].count("session.consumer.issue") == 1
    finally:
        node._stop.set()
        await asyncio.gather(running, return_exceptions=True)
