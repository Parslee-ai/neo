import copy
import json
from pathlib import Path

import pytest

from neo import lattice
from neo import lattice_enrollment as enrollment

A = {"api_base": "https://issuer.example", "account_id": "member", "organization_id": "parslee"}
B = dict(A, organization_id="flyexclusive")
PROJECT = "repo:" + "a" * 64


class Host:
    def __init__(self):
        self.boot = "boot-1"
        self.parents = {}
        self.managed = {}
        self.calls = []
        self.fail = None
        self.unconfirmed = False
        self.lost_create = False

    def __call__(self, method, params):
        self.calls.append((method, copy.deepcopy(params)))
        if self.fail == method:
            raise PermissionError("original membership or device enrollment refused")
        ident = params.get("agent_id", params.get("id"))
        if method == "agents.parent_lifecycle":
            result = copy.deepcopy(self.parents.get(ident, {"version": 2, "state": "absent", "agent_id": ident}))
            result["daemon_instance_id"] = self.boot
            return result
        if method == "agents.parent_create":
            assert params["expected_daemon_instance_id"] == self.boot
            spec = copy.deepcopy(params["spec"])
            assert spec["method_allowlist"] == list(lattice.SCOPED_PARENT_METHODS)
            ident = spec["id"]
            self.managed[ident] = dict(spec, status="stopped")
            self.parents[ident] = {"version": 2, "agent_id": ident, "state": "present", "created": True,
                "daemon_instance_id": self.boot, "work_context": spec["work_context"],
                "task_incarnation": "task-" + ident, "intent_revision": "1", "spec_digest": "d" * 64,
                "resource_key": None, "authority_active": True, "renewal_allowed": False,
                "project_ids": params["project_ids"]}
            if self.lost_create:
                raise ConnectionError("admission response lost")
            return copy.deepcopy(self.parents[ident])
        if method == "agents.list":
            rows = list(copy.deepcopy(self.managed).values())
            for row in rows:
                row.pop("work_context", None)
            return rows
        if method == "agents.start":
            assert "id" in params and "agent_id" not in params
            self.managed[ident]["status"] = "running"
            self.parents[ident]["resource_key"] = "context-private:work-" + "e" * 64
            self.parents[ident]["renewal_allowed"] = True
            return copy.deepcopy(self.managed[ident])
        if method == "agents.parent_end":
            assert params["token"] == self.managed[ident]["token"]
            p = self.parents[ident]
            p.update(ended=True, cleanup_confirmed=not self.unconfirmed,
                     authority_active=False, renewal_allowed=False,
                     intent_revision=str(int(p["intent_revision"]) + 1))
            self.managed[ident]["status"] = "stopped"
            return copy.deepcopy(p)
        if method == "agents.parent_renew":
            p = self.parents[ident]
            for field in ("task_incarnation", "intent_revision", "spec_digest", "work_context", "resource_key"):
                assert params["expected_" + field] == p[field]
            assert "project_ids" not in params  # CAR preserves original approved scope on renewal.
            assert params["spec"]["token"] != self.managed[ident]["token"]
            self.managed[ident] = dict(copy.deepcopy(params["spec"]), status="stopped")
            p.update(renewed=True, authority_active=True, daemon_instance_id=self.boot,
                     intent_revision=str(int(p["intent_revision"]) + 1))
            return copy.deepcopy(p)
        raise AssertionError(method)


@pytest.fixture
def host(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("CAR_AGENT_ID", "CAR_AGENT_TOKEN", "CAR_WORK_CONTEXT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(lattice, "repository_root", lambda root: root)
    monkeypatch.setattr(lattice, "agent_id_for", lambda root: "neo-lattice-" + "a" * 12)
    result = Host()
    monkeypatch.setattr(enrollment, "_host_call", lambda: result)
    return result


def join(context=A, **kwargs):
    return lattice.join("/repo", kwargs.pop("project", "repo"), work_context=context, **kwargs)


def test_two_orgs_share_repo_without_retargeting_or_restarting(host, monkeypatch):
    first = join(project_ids=[PROJECT])
    second = join(B, project_ids=[PROJECT])
    assert first["status"] == second["status"] == "running"
    assert first["agent_id"] != second["agent_id"]
    monkeypatch.setenv("VIEWED_ORG", "flyexclusive")
    original = join()
    assert original["task_incarnation"] == first["task_incarnation"]
    assert original["work_context"] == A
    assert [m for m, _ in host.calls].count("agents.start") == 2
    assert [m for m, _ in host.calls].count("agents.parent_create") == 2
    files = list((Path.home() / ".neo/lattice/enrollments").glob("*.json"))
    assert len(files) == 2 and all(p.stat().st_mode & 0o077 == 0 for p in files)
    assert "token" not in json.dumps(first)


def test_leave_one_org_confirms_original_and_keeps_other_running(host):
    first, second = join(), join(B)
    left = lattice.leave("/repo", work_context=A)
    assert left["status"] == "left"
    assert left["task_incarnation"] == first["task_incarnation"]
    assert lattice.status("/repo", work_context=B)["status"] == "running"
    assert host.managed[second["agent_id"]]["status"] == "running"
    assert not lattice._left_marker("/repo").exists()
    replacement = join()
    assert replacement["status"] == "running"
    assert replacement["task_incarnation"] != first["task_incarnation"]
    assert list((Path.home() / ".neo/lattice/enrollments").glob("*-ended-*.json"))


def test_unconfirmed_leave_does_not_report_success_and_retry_can_finish(host):
    join()
    host.unconfirmed = True
    assert lattice.leave("/repo", work_context=A)["status"] == "error"
    host.unconfirmed = False
    assert lattice.leave("/repo", work_context=A)["status"] == "left"


def test_daemon_restart_renews_original_task_and_private_resource(host):
    first = join(project_ids=[PROJECT])
    ident = first["agent_id"]
    original = copy.deepcopy(host.parents[ident])
    host.boot = "boot-2"
    host.parents[ident]["authority_active"] = False
    host.managed[ident]["status"] = "stopped"
    renewed = join(project_ids=[PROJECT])
    assert renewed["status"] == "running"
    assert renewed["task_incarnation"] == first["task_incarnation"]
    assert host.parents[ident]["resource_key"] == original["resource_key"]
    assert host.parents[ident]["project_ids"] == [PROJECT]
    assert [m for m, _ in host.calls].count("agents.parent_create") == 1


@pytest.mark.parametrize("field,value", [("task_incarnation", "other-task"), ("work_context", B),
    ("spec_digest", "f" * 64), ("resource_key", "context-private:work-" + "f" * 64)])
def test_replaced_original_refuses_recovery(host, field, value):
    first = join()
    host.parents[first["agent_id"]][field] = value
    host.calls.clear()
    assert join()["status"] == "error"
    assert all(m not in {"agents.parent_create", "agents.parent_renew", "agents.start"} for m, _ in host.calls)


def test_membership_device_refusal_no_personal_fallback(host):
    host.fail = "agents.parent_create"
    assert join()["status"] == "error"
    assert all(m not in {"agents.upsert", "agents.start"} for m, _ in host.calls)
    host.calls.clear()
    assert join()["status"] == "error"
    assert not host.calls  # pending admission is never blindly retried.


def test_response_lost_after_create_does_not_adopt_or_duplicate_original(host):
    host.lost_create = True
    assert join()["status"] == "error"
    assert len(host.parents) == 1
    assert join()["status"] == "error"
    assert [m for m, _ in host.calls].count("agents.parent_create") == 1
    assert all(m != "agents.start" for m, _ in host.calls)


def test_changed_project_approval_refuses_existing_task(host):
    join(project_ids=[PROJECT])
    assert join(project_ids=[])["status"] == "error"
    assert join(A, project="different")["status"] == "error"


def test_revocation_blocks_join_without_disrupting_second_org(host):
    first, second = join(), join(B)
    host.parents[first["agent_id"]].update(authority_active=False, renewal_allowed=False)
    assert join()["status"] == "error"
    assert lattice.status("/repo", work_context=B)["status"] == "running"
    assert host.managed[second["agent_id"]]["status"] == "running"


@pytest.mark.parametrize("boundary", [
    {"authority_active": False},
    {"resource_key": "context-private:work-" + "e" * 64},
    {"work_revocation_pending": True},
])
def test_nonrenewable_admission_refuses_inactive_bound_or_pending_work(host, monkeypatch, boundary):
    def call(method, params):
        value = host(method, params)
        if method == "agents.parent_create":
            value.update(boundary)
        return value
    monkeypatch.setattr(enrollment, "_host_call", lambda: call)
    assert join()["status"] == "error"
    assert all(method != "agents.start" for method, _ in host.calls)


def test_bound_agent_cannot_use_operator_enrollment(host, monkeypatch):
    monkeypatch.setenv("CAR_WORK_CONTEXT", json.dumps(A))
    assert join()["status"] == "error"
    assert host.calls == []


def test_personal_autojoin_reaper_does_not_touch_org_nodes(host):
    first = join()
    org = host.managed[first["agent_id"]]
    org["command"] = "/missing"
    lattice._reap_stale_nodes(object(), [org], keep="personal", now=1e20)
    assert host.managed[first["agent_id"]]["status"] == "running"


def test_cli_explicit_context_and_project_input(host, tmp_path, capsys):
    context_file = tmp_path / "owner.json"
    context_file.write_text(json.dumps(A))
    args = ["join", "--cwd", "/repo", "--project", "repo", "--work-context-file", str(context_file), "--project-id", PROJECT]
    assert lattice.cli_main(args) == 0
    admitted = json.loads(capsys.readouterr().out)
    assert admitted["work_context"] == A
    assert admitted["project_ids"] == [PROJECT]
    assert lattice.cli_main(["status", "--cwd", "/repo", "--work-context-file", str(context_file)]) == 0
    assert json.loads(capsys.readouterr().out)["task_incarnation"] == admitted["task_incarnation"]
    assert lattice.cli_main(["leave", "--cwd", "/repo", "--work-context-file", str(context_file)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "left"


@pytest.mark.parametrize("context", [dict(A, organization_id=None), dict(A, extra=True), dict(A, api_base="https://user:secret@issuer.example")])
def test_bad_context_is_refused_before_host(host, context):
    assert join(context)["status"] == "error"
    assert not host.calls


def test_scoped_status_unavailable_host_is_error_no_personal_lookup(host, monkeypatch):
    join()
    def unavailable():
        raise RuntimeError("installed CAR SDK lacks authenticated HostClient")
    monkeypatch.setattr(enrollment, "_host_call", unavailable)
    result = lattice.status("/repo", work_context=A)
    assert result["status"] == "error"
    assert "HostClient" in result["message"]


def test_host_client_uses_ordinary_list_without_sending_host_token(monkeypatch):
    import sys
    from types import SimpleNamespace
    from neo.memory import observer
    calls = []
    class Runtime:
        def agents_list(self, params_json):
            assert json.loads(params_json) == {}
            calls.append("ordinary-list")
            return "[]"
    class HostClient:
        def __init__(self, runtime):
            pass
        def daemon_call_host_management(self, method, params):
            assert method != "agents.list"
            calls.append(method)
            return '{}'
    monkeypatch.setattr(observer, "_require_car_runtime", lambda: SimpleNamespace(CarRuntime=Runtime))
    monkeypatch.setitem(sys.modules, "car_runtime", SimpleNamespace(HostClient=HostClient))
    call = enrollment._host_call()
    assert call("agents.list", {}) == []
    call("agents.parent_lifecycle", {"agent_id": "node"})
    assert calls == ["ordinary-list", "agents.parent_lifecycle"]


def test_start_failure_recovers_original_without_duplicate_admission(host):
    host.fail = "agents.start"
    first = join()
    assert first["status"] == "error"
    host.fail = None
    assert join()["status"] == "running"
    assert [m for m, _ in host.calls].count("agents.parent_create") == 1


def test_renewal_lost_response_preserves_pending_new_token_without_retry(host):
    first = join()
    host.parents[first["agent_id"]]["authority_active"] = False
    host.fail = "agents.parent_renew"
    assert join()["status"] == "error"
    host.calls.clear()
    assert join()["status"] == "error"
    assert not host.calls


def test_missing_context_file_cannot_fall_back_to_personal(host, tmp_path, capsys):
    assert lattice.cli_main(["join", "--cwd", "/repo", "--work-context-file", str(tmp_path / "missing")]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"
    assert not host.calls


def test_personal_reaper_refuses_org_id_even_if_listing_omits_ownership(host):
    first = join()
    unknown = dict(host.managed[first["agent_id"]], command="/missing")
    unknown.pop("work_context")
    assert lattice._reap_stale_nodes(object(), [unknown], keep="personal", now=1e20) == []


def test_invalid_project_ids_refuse_before_admission(host):
    assert join(project_ids=["repo:invalid"])["status"] == "error"
    assert all(method != "agents.parent_create" for method, _ in host.calls)


def test_lost_create_can_be_explicitly_retired_before_new_admission(host):
    host.lost_create = True
    assert join()["status"] == "error"
    original = next(iter(host.parents))
    assert lattice.leave("/repo", work_context=A)["status"] == "left"
    host.lost_create = False
    assert join()["status"] == "running"
    assert len(host.parents) == 2
    assert host.parents[original]["authority_active"] is False


def test_absent_pending_create_cannot_be_declared_canceled(host):
    host.fail = "agents.parent_create"
    assert join()["status"] == "error"
    host.fail = None
    assert lattice.leave("/repo", work_context=A)["status"] == "error"
    assert join()["status"] == "error"
    assert not host.parents


def test_endpoint_change_refuses_original_retarget(host, monkeypatch):
    join()
    host.calls.clear()
    monkeypatch.setenv("CAR_DAEMON_URL", "ws://127.0.0.1:9292")
    assert join()["status"] == "error"
    assert lattice.leave("/repo", work_context=A)["status"] == "error"
    assert not host.calls


def test_scoped_live_pid_manifest_skew_refuses_duplicate_start(host, monkeypatch):
    first = join()
    host.managed[first["agent_id"]]["status"] = "stopped"
    monkeypatch.setattr(lattice, "_live_node_pid", lambda ident: 12345)
    host.calls.clear()
    assert join()["status"] == "error"
    assert all(method != "agents.start" for method, _ in host.calls)


def test_two_org_admissions_can_overlap_without_global_org_lock(host, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    admitted = Barrier(2)
    def call(method, params):
        if method == "agents.parent_create":
            admitted.wait(timeout=3)
        return host(method, params)
    monkeypatch.setattr(enrollment, "_host_call", lambda: call)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(join, A), pool.submit(join, B)
        first, second = a.result(timeout=5), b.result(timeout=5)
    assert first["status"] == second["status"] == "running"
    assert first["work_context"] == A and second["work_context"] == B
    assert first["task_incarnation"] != second["task_incarnation"]


def test_concurrent_same_org_join_admits_only_one_original(host):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = pool.submit(join), pool.submit(join)
        first, second = a.result(timeout=5), b.result(timeout=5)
    assert first["status"] == second["status"] == "running"
    assert first["task_incarnation"] == second["task_incarnation"]
    assert [method for method, _ in host.calls].count("agents.parent_create") == 1


def test_status_reports_revoked_authority_even_if_process_is_still_running(host):
    first = join()
    host.parents[first["agent_id"]].update(authority_active=False, renewal_allowed=False)
    result = lattice.status("/repo", work_context=A)
    assert result["status"] == "revoked"
    assert result["process_status"] == "running"
    assert result["renewal_allowed"] is False
