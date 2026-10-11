"""Pure bridge controls: mock transport only; no installed/native acceptance."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from typing import Optional
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("neo")
package.__path__ = [str(ROOT / "src/neo")]
sys.modules["neo"] = package
lattice = types.ModuleType("neo.lattice")
sys.modules["neo.lattice"] = lattice
package.lattice = lattice
for name in ("lattice_enrollment", "lattice_native_owner"):
    spec = importlib.util.spec_from_file_location("neo." + name, ROOT / "src/neo" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    setattr(package, name, module)
enrollment, bridge = package.lattice_enrollment, package.lattice_native_owner


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        lattice._state_dir = lambda: self.root / "state"
        lattice.agent_id_for = lambda root: "neo-" + hashlib.sha256(root.encode()).hexdigest()[:16]
        lattice.daemon_url = lambda: os.environ["CAR_DAEMON_URL"]
        lattice.repository_root = lambda root: None
        lattice.touch_seen = lambda *args: None
        lattice.derive_project_name = lambda root: "project"
        lattice.build_spec = lambda root, project, context: {"id": "neo-test", "name": "Neo",
            "command": "/unused", "args": ["-P", "-m", "neo.lattice", "run", "--cwd", root],
            "work_context": context, "method_allowlist": ["session.consumer.issue"], "env": {"SECRET": "must-drop"}}
        self.context = {"api_base": "https://issuer.example", "account_id": "alice", "organization_id": "a"}
        self.config = {"guardian_python": str(self.root / "guardian-python"),
            "guardian_scripts": str(self.root / "scripts"), "child_python": str(self.root / "child-python"),
            "child_sdk": str(self.root / "sdk.so"), "child_package": str(self.root / "neo"),
            "ownership_root": str(self.root / "owners"), "pins": {}}
        (self.root / "scripts").mkdir()
        (self.root / "neo").mkdir()
        for name in [self.config["guardian_python"], self.config["child_python"], self.config["child_sdk"],
                     str(self.root / "scripts/supervised-owner.py"), str(self.root / "neo/__init__.py")]:
            p = Path(name)
            p.write_text("mock artifact")
            p.chmod(0o700)
            self.config["pins"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
        self.calls = []
        self.failure = None
        self.response = None
        self.inspect_response = None
        self.transport = patch.object(bridge, "_run", self.run_transport)
        self.transport.start()
        self.addCleanup(self.transport.stop)
        self.env = patch.dict(os.environ, {"CAR_DAEMON_URL": "ws://127.0.0.1:12345"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def run_transport(self, config, action, spec_path, spec):
        self.calls.append(action)
        assert spec["auto_start"] is False and spec["env"] == {} and "token" not in spec
        if action == "inspect":
            if callable(self.inspect_response):
                return self.inspect_response(spec)
            return self.inspect_response or {"version": 2, "state": "absent", "agent_id": spec["id"], "daemon_instance_id": "boot"}
        if self.failure:
            raise RuntimeError(self.failure)
        if self.response is not None:
            return self.response
        receipt = {"version": 2, "agent_id": spec["id"], "state": "present", "work_context": spec["work_context"],
            "daemon_instance_id": "boot", "task_incarnation": "12345678-1234-4234-8234-123456789abc",
            "intent_revision": "1", "spec_digest": "d" * 64, "resource_key": "context-private:work-" + "e" * 64,
            "authority_active": action != "leave", "renewal_allowed": action != "leave", "project_ids": []}
        if action == "leave":
            receipt.update(ended=True, cleanup_confirmed=True)
        return {"phase": "stopped" if action == "leave" else "running", "lifecycle": receipt}

    def call(self, action="join", config=None):
        return bridge.operation(action, str(self.root / "repo"), "project", self.context, None, config or self.config)

    def record(self):
        return next(p for p in (self.root / "state/enrollments").glob("*.json") if "-ended-" not in p.name)

    def prepare_rejoin(self):
        self.call()
        self.call("leave")
        original = json.loads(self.record().read_text())
        self.inspect_response = lambda spec: (original["receipt"] if spec["id"] == original["spec"]["id"]
            else {"version": 2, "state": "absent", "agent_id": spec["id"], "daemon_instance_id": "boot"})
        self.calls.clear()
        return original

    def test_confirmed_leave_rejoin_rechecks_original_and_archives(self):
        original = self.prepare_rejoin()
        result = self.call()
        self.assertEqual(result["status"], "running")
        self.assertNotEqual(result["agent_id"], original["spec"]["id"])
        self.assertEqual(self.calls, ["inspect", "inspect", "join"])
        archives = list((self.root / "state/enrollments").glob("*-ended-*.json"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(json.loads(archives[0].read_text()), original)

    def test_rejoin_refuses_changed_original_authority_or_identity(self):
        original = self.prepare_rejoin()
        for changes in ({"authority_active": True}, {"renewal_allowed": True},
                        {"work_revocation_pending": True}, {"task_incarnation": "replaced"},
                        {"work_context": dict(self.context, organization_id="b")}, {"state": "absent"},
                        {"daemon_instance_id": "new-boot"}, {"project_ids": ["unapproved"]},
                        {"resource_key": "context-private:work-" + "f" * 64}):
            self.inspect_response = dict(original["receipt"], **changes)
            self.calls.clear()
            self.assertEqual(self.call()["status"], "error")
            self.assertEqual(self.calls, ["inspect"])
            self.assertEqual(json.loads(self.record().read_text()), original)
            self.assertEqual(list((self.root / "state/enrollments").glob("*-ended-*.json")), [])

    def test_rejoin_lost_read_or_no_owned_does_not_replace_original(self):
        original = self.prepare_rejoin()
        def lost(spec):
            raise RuntimeError("original observation unavailable")
        for response in (lost, {"no_owned_authority": True}):
            self.inspect_response = response
            self.calls.clear()
            self.assertEqual(self.call()["status"], "error")
            self.assertEqual(self.calls, ["inspect"])
            self.assertEqual(json.loads(self.record().read_text()), original)
            self.assertEqual(list((self.root / "state/enrollments").glob("*-ended-*.json")), [])

    def test_join_and_confirmed_leave_retain_original(self):
        joined = self.call()
        self.assertEqual(joined["status"], "running")
        self.assertEqual(self.calls, ["inspect", "join"])
        left = self.call("leave")
        self.assertEqual(left["status"], "left")
        self.assertEqual(joined["task_incarnation"], left["task_incarnation"])
        self.assertTrue(json.loads(self.record().read_text())["left"])

    def test_returned_wrong_context_preserves_pending_admission(self):
        self.response = {"phase": "running", "lifecycle": {"version": 2, "work_context": dict(self.context, organization_id="b")}}
        self.assertEqual(self.call()["status"], "error")
        self.assertEqual(json.loads(self.record().read_text())["phase"], "pending_create")
        self.calls.clear()
        self.assertEqual(self.call()["status"], "error")
        self.assertEqual(self.calls, [])

    def test_changed_configuration_does_not_call_owner(self):
        self.call()
        changed = dict(self.config, ownership_root=str(self.root / "other-owners"))
        self.calls.clear()
        self.assertEqual(self.call(config=changed)["status"], "error")
        self.assertEqual(self.calls, [])

    def test_no_owned_authority_is_not_successful_leave(self):
        self.call()
        self.response = {"no_owned_authority": True}
        self.assertEqual(self.call("leave")["status"], "error")
        record = json.loads(self.record().read_text())
        self.assertEqual(record["phase"], "pending_cleanup")
        self.assertFalse(record.get("left", False))

    def test_failed_stop_preserves_cleanup_and_refuses_new_work(self):
        self.call()
        self.failure = "stop unavailable"
        self.assertEqual(self.call("leave")["status"], "error")
        self.failure = None
        self.calls.clear()
        self.assertEqual(self.call()["status"], "error")
        self.assertEqual(self.calls, [])

    def test_legacy_record_never_uses_native_cleanup(self):
        self.call()
        p = self.record()
        record = json.loads(p.read_text())
        record.pop("backend")
        p.write_text(json.dumps(record))
        self.calls.clear()
        self.assertEqual(self.call("leave")["status"], "error")
        self.assertEqual(self.calls, [])

    def test_replacement_task_is_refused(self):
        self.call()
        self.response = self.run_transport(self.config, "status", None, json.loads(self.record().read_text())["spec"])
        self.response["lifecycle"]["task_incarnation"] = "replaced"
        self.assertEqual(self.call("status")["status"], "error")

    def test_existing_original_refuses_initial_launch(self):
        self.inspect_response = {"version": 2, "state": "present", "agent_id": "other", "daemon_instance_id": "boot"}
        self.assertEqual(self.call()["status"], "error")
        self.assertEqual(self.calls, ["inspect"])
        self.assertEqual(json.loads(self.record().read_text())["phase"], "pending_create")

    def test_pending_cleanup_status_is_not_confirmed_running(self):
        self.call()
        self.failure = "stop unavailable"
        self.call("leave")
        self.failure = None
        status = self.call("status")
        self.assertEqual(status["status"], "cleanup_pending")
        self.assertTrue(status["cleanup_pending"])
        self.assertEqual(json.loads(self.record().read_text())["phase"], "pending_cleanup")

    def test_saved_native_owner_routes_without_supplied_config(self):
        joined = self.call()
        with patch.object(enrollment, "_host_call", side_effect=AssertionError("legacy fallback")):
            status = enrollment.operation("status", str(self.root / "repo"), None, self.context)
            self.assertEqual(status["task_incarnation"], joined["task_incarnation"])
            left = enrollment.operation("leave", str(self.root / "repo"), None, self.context)
            self.assertEqual(left["status"], "left")

    def test_saved_pending_cleanup_routes_and_refuses_join(self):
        self.call()
        self.failure = "stop unavailable"
        self.call("leave")
        self.failure = None
        with patch.object(enrollment, "_host_call", side_effect=AssertionError("legacy fallback")):
            status = enrollment.operation("status", str(self.root / "repo"), None, self.context)
            self.assertEqual(status["status"], "cleanup_pending")
            self.calls.clear()
            joined = enrollment.operation("join", str(self.root / "repo"), None, self.context)
            self.assertEqual(joined["status"], "error")
            self.assertEqual(self.calls, [])

    def test_inactive_pending_admission_cannot_enable_join(self):
        self.failure = "admission unavailable"
        self.call()
        record = json.loads(self.record().read_text())
        self.failure = None
        self.response = self.run_transport(self.config, "leave", None, record["spec"])
        self.assertEqual(self.call("status")["status"], "error")
        self.assertEqual(json.loads(self.record().read_text())["phase"], "pending_create")
        self.calls.clear()
        self.assertEqual(self.call()["status"], "error")
        self.assertEqual(self.calls, [])

    def test_owner_directory_symlink_refuses_before_native_effect(self):
        target = self.root / "target"
        target.mkdir(mode=0o700)
        Path(self.config["ownership_root"]).symlink_to(target, target_is_directory=True)
        self.assertEqual(self.call()["status"], "error")
        self.assertEqual(self.calls, [])

    def test_public_api_requires_context_and_routes_owner(self):
        source = ast.parse((ROOT / "src/neo/lattice.py").read_text())
        functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"join", "leave", "status"}]
        namespace = {"Optional": Optional}
        exec(compile(ast.Module(body=functions, type_ignores=[]), "lattice-api", "exec"), namespace)
        with patch.object(enrollment, "operation", return_value={"status": "observed"}) as operation:
            for name in ("join", "leave", "status"):
                self.assertEqual(namespace[name]("/repo", native_owner=self.config)["status"], "error")
                operation.assert_not_called()
                self.assertEqual(namespace[name]("/repo", work_context=self.context,
                    native_owner=self.config)["status"], "observed")
                self.assertEqual(operation.call_args.kwargs["native_owner"], self.config)
                operation.reset_mock()


if __name__ == "__main__":
    unittest.main()
