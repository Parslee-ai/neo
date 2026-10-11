"""Explicit host admission for organization-owned Neo Lattice nodes.

Files retain the original task receipt and token; neither the selected org nor
repository access authorizes admission. CAR checks original membership/device
standing. A lost mutation receipt remains pending and cannot silently mint work.
"""
from __future__ import annotations

from contextlib import contextmanager
try:
    import fcntl
except ImportError:  # Personal workflows remain available on platforms without POSIX locks.
    fcntl = None
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import tempfile
from urllib.parse import urlsplit


def work_context(value: dict) -> dict:
    if not isinstance(value, dict) or set(value) != {"api_base", "account_id", "organization_id"}:
        raise ValueError("Organization work context requires api_base, account_id and organization_id")
    if any(not isinstance(v, str) or not v or v != v.strip() for v in value.values()):
        raise ValueError("Organization work context fields must be nonempty strings")
    url = urlsplit(value["api_base"])
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("Work context requires an HTTP(S) issuer without embedded credentials")
    return dict(value)


def projects(values) -> list[str]:
    result = list(values or [])
    if len(result) > 32 or len(set(result)) != len(result) or any(
            not isinstance(v, str) or re.fullmatch(r"repo:[0-9a-f]{64}", v) is None for v in result):
        raise ValueError("Approve at most 32 unique canonical repo:<64 lowercase hex> project IDs")
    return result


def _key(root: str, context: dict) -> str:
    from neo.lattice import agent_id_for
    digest = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    return agent_id_for(root) + "-" + digest


@contextmanager
def _record_lock(key: str):
    from neo.lattice import _state_dir
    if fcntl is None:
        raise RuntimeError("Organization enrollment requires supported native receipt locking on this platform")
    directory = _state_dir() / "enrollments"
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    if directory.is_symlink() or directory.stat().st_uid != os.getuid():
        raise PermissionError("Neo enrollment directory must be owned by this user")
    directory.chmod(0o700)
    fd = os.open(directory / (key + ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        os.fchmod(fd, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield directory / (key + ".json")
    finally:
        os.close(fd)


def _load(path: Path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(fd) as handle:
        if os.fstat(handle.fileno()).st_mode & 0o077:
            raise PermissionError("Neo enrollment receipt must be private (0600)")
        return json.load(handle)


def _save(path: Path, value: dict):
    fd, temporary = tempfile.mkstemp(prefix=".enrollment-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _host_call():
    from neo.memory.observer import _require_car_runtime
    from car_runtime import HostClient
    runtime = _require_car_runtime().CarRuntime()
    host = HostClient(runtime)

    def call(method, params):
        # agents.list is an ordinary read, not a host-token RPC in CAR's
        # closed management method inventory. Mutations always use HostClient.
        raw = runtime.agents_list(json.dumps(params)) if method == "agents.list" else host.daemon_call_host_management(method, json.dumps(params))
        value = json.loads(raw)
        if not isinstance(value, (dict, list)):
            raise RuntimeError("CAR returned no original parent receipt")
        return value
    return call


def _receipt(value: dict, record: dict, *, original=None):
    if (value.get("version") != 2 or value.get("agent_id") != record["spec"]["id"]
            or value.get("work_context") != record["work_context"]
            or not isinstance(value.get("daemon_instance_id"), str) or not value["daemon_instance_id"]
            or not isinstance(value.get("task_incarnation"), str) or not value["task_incarnation"]
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("spec_digest"))) is None
            or not isinstance(value.get("intent_revision"), str) or not value["intent_revision"].isdigit()
            or value.get("project_ids", []) != record["project_ids"]
            or type(value.get("authority_active")) is not bool
            or type(value.get("renewal_allowed")) is not bool):
        raise PermissionError("CAR original Neo parent receipt is incomplete or changed ownership")
    resource = value.get("resource_key")
    if resource is not None and re.fullmatch(r"context-private:work-[0-9a-f]{64}", str(resource)) is None:
        raise PermissionError("CAR original Neo private resource is invalid")
    if original is not None:
        for field in ("task_incarnation", "spec_digest", "work_context"):
            if value[field] != original[field]:
                raise PermissionError("CAR original Neo task was replaced; refusing recovery")
        if original.get("resource_key") is not None and resource != original["resource_key"]:
            raise PermissionError("CAR original Neo private resource changed")
    return value


def _view(record: dict, receipt: dict, state=None):
    return {"status": state or receipt["state"], "agent_id": record["spec"]["id"],
            "work_context": record["work_context"], "task_incarnation": receipt["task_incarnation"],
            "root": record["root"], "project": record["project"],
            "project_ids": record["project_ids"], "authority_active": receipt["authority_active"],
            "renewal_allowed": receipt["renewal_allowed"],
            "cleanup_pending": receipt.get("work_revocation_pending") is True}


def _managed(call, record):
    observed = call("agents.parent_lifecycle", {"agent_id": record["spec"]["id"]})
    _receipt(observed, record, original=record["receipt"])
    rows = call("agents.list", {})
    if not isinstance(rows, list):
        raise PermissionError("CAR did not confirm original Neo process state")
    selected = [row for row in rows if row.get("id") == record["spec"]["id"]]
    if len(selected) != 1 or ("work_context" in selected[0]
            and selected[0]["work_context"] != record["work_context"]):
        raise PermissionError("Original Neo supervised process is absent or changed owner")
    row = selected[0]
    from neo.lattice import _skew_message
    skew = _skew_message(record["spec"]["id"], row)
    if skew:
        raise PermissionError(skew)
    if sorted(row.get("method_allowlist") or []) != sorted(record["spec"]["method_allowlist"]):
        raise PermissionError("Original Neo permission profile changed")
    return row


def _cancel_pending(call, path, record):
    observed = call("agents.parent_lifecycle", {"agent_id": record["spec"]["id"]})
    # An absent snapshot cannot prove a timed-out create will not commit later.
    # Keep the receipt until the server can confirm this exact original's end.
    _receipt(observed, record, original=record.get("receipt"))
    candidates = [record.get("pending_spec"), record["spec"]]
    for spec in candidates:
        if spec is None:
            continue
        ended = call("agents.parent_end", {"agent_id": spec["id"],
                     "expected_daemon_instance_id": observed["daemon_instance_id"], "token": spec["token"]})
        if ended.get("no_owned_authority") is True:
            continue
        _receipt(ended, record, original=record.get("receipt"))
        if ended.get("ended") is not True or ended.get("cleanup_confirmed") is not True:
            raise PermissionError("Pending original Neo termination remains unconfirmed")
        record.update(phase="confirmed", receipt=ended, spec=spec, left=True)
        record.pop("pending_spec", None)
        _save(path, record)
        return _view(record, ended, "left")
    raise PermissionError("No matching original Neo authority was retired; pending admission requires reconciliation")


def operation(action: str, root: str, project: str | None, context: dict, project_ids=None, *, native_owner=None) -> dict:
    """Operate on one explicitly owned original; never use personal fallback."""
    if native_owner is not None:
        from neo.lattice_native_owner import operation as native_operation
        return native_operation(action, root, project, context, project_ids, native_owner)
    from neo.lattice import build_spec, daemon_url, derive_project_name, repository_root, touch_seen
    ownership = None
    try:
        context = work_context(context)
        ownership = context
        if any(os.environ.get(k) for k in ("CAR_AGENT_ID", "CAR_AGENT_TOKEN", "CAR_WORK_CONTEXT")):
            raise PermissionError("Organization onboarding requires the operator's authenticated host, not a bound agent")
        root = repository_root(root) or os.path.abspath(root)
        key = _key(root, context)
        # Select the saved backend under the original context key, then release
        # the lock: the native operation rechecks the record under its own lock.
        with _record_lock(key) as path:
            original = _load(path)
            saved_owner = original.get("native_owner") if original is not None and original.get("backend") == "native_guardian" else None
            native_record = original is not None and original.get("backend") == "native_guardian"
        if native_record:
            from neo.lattice_native_owner import operation as native_operation
            return native_operation(action, root, project, context, project_ids, saved_owner)
        with _record_lock(key) as path:
            record = _load(path)
            if record is not None:
                if record.get("backend") == "native_guardian":
                    raise PermissionError("Original backend changed; retry through its saved native owner")
                if (record.get("work_context") != context or record.get("root") != root
                        or record.get("daemon_url") != daemon_url()):
                    raise PermissionError("Local original Neo receipt belongs to a different workspace or owner")
                if record.get("phase") != "confirmed":
                    if action == "leave":
                        return _cancel_pending(_host_call(), path, record)
                    raise PermissionError("Original Neo admission is unconfirmed; use scoped leave to retire the observed original task before retrying")
            call = _host_call()
            if record is not None and action == "join" and record.get("left") is True:
                ended = call("agents.parent_lifecycle", {"agent_id": record["spec"]["id"]})
                _receipt(ended, record, original=record["receipt"])
                if ended["authority_active"] or ended["renewal_allowed"] or ended.get("work_revocation_pending") is True:
                    raise PermissionError("Original Neo termination is not confirmed; cannot admit replacement work")
                archive = path.with_name(path.stem + "-ended-" + hashlib.sha256(ended["task_incarnation"].encode()).hexdigest()[:16] + ".json")
                _save(archive, record)
                record = None
            if record is None:
                if action != "join":
                    return {"status": "not_joined", "work_context": context, "root": root}
                approved = projects(project_ids)
                project = project or derive_project_name(root)
                spec = build_spec(root, project, context)
                spec["id"] += "-" + secrets.token_hex(6)
                spec["token"] = secrets.token_hex(32)
                # Create only records admission. Launch follows receipt persistence.
                spec["auto_start"] = True
                record = {"version": 1, "phase": "pending_create", "root": root, "project": project,
                          "work_context": context, "project_ids": approved, "spec": spec, "daemon_url": daemon_url()}
                observed = call("agents.parent_lifecycle", {"agent_id": spec["id"]})
                if observed.get("state") != "absent" or not observed.get("daemon_instance_id"):
                    raise PermissionError("Fresh original Neo parent identity collided")
                _save(path, record)
                receipt = call("agents.parent_create", {"expected_daemon_instance_id": observed["daemon_instance_id"],
                                                      "spec": spec, "project_ids": approved})
                _receipt(receipt, record)
                if receipt.get("created") is not True or receipt.get("authority_active") is not True:
                    raise PermissionError("CAR did not confirm original Neo admission")
                record.update(phase="confirmed", receipt=receipt)
                _save(path, record)
            else:
                if action == "join" and ((project is not None and project != record["project"])
                        or (project_ids is not None and projects(project_ids) != record["project_ids"])):
                    raise PermissionError("Original Neo project scope is immutable; leave before admitting different work")
                receipt = call("agents.parent_lifecycle", {"agent_id": record["spec"]["id"]})
                _receipt(receipt, record, original=record["receipt"])
                if action == "join" and receipt["intent_revision"] != record["receipt"]["intent_revision"]:
                    raise PermissionError("Original Neo intent changed; reconcile the original task before acting")
                record["receipt"] = receipt
                _save(path, record)
            if action == "status":
                managed = _managed(call, record)
                state = managed["status"]
                if record.get("left") is True:
                    state = "left"
                elif receipt.get("work_revocation_pending") is True:
                    state = "cleanup_pending"
                elif not receipt["authority_active"]:
                    state = "inactive" if receipt["renewal_allowed"] else "revoked"
                result = _view(record, receipt, state)
                result["process_status"] = managed["status"]
                return result
            if action == "leave":
                ended = call("agents.parent_end", {"agent_id": record["spec"]["id"],
                             "expected_daemon_instance_id": receipt["daemon_instance_id"],
                             "token": record["spec"]["token"]})
                _receipt(ended, record, original=receipt)
                if ended.get("ended") is not True or ended.get("cleanup_confirmed") is not True:
                    raise PermissionError("CAR has not confirmed original Neo termination; cleanup remains pending")
                record.update(receipt=ended, left=True)
                _save(path, record)
                return _view(record, ended, "left")
            if not receipt["renewal_allowed"] and (receipt["authority_active"] is not True
                    or receipt.get("resource_key") is not None
                    or receipt.get("work_revocation_pending") is True):
                raise PermissionError("Original Neo task ended or was revoked; explicit new enrollment is required")
            if receipt["authority_active"] is not True:
                if receipt.get("resource_key") is None:
                    raise PermissionError("Original Neo has no bound private resource; cannot renew an unconfirmed task")
                fresh_spec = dict(record["spec"], token=secrets.token_hex(32))
                fence = {"agent_id": fresh_spec["id"], "expected_daemon_instance_id": receipt["daemon_instance_id"],
                         **{"expected_" + field: receipt[field] for field in
                            ("task_incarnation", "intent_revision", "spec_digest", "work_context", "resource_key")},
                         "spec": fresh_spec}
                record.update(phase="pending_renewal", pending_spec=fresh_spec)
                _save(path, record)
                renewed = call("agents.parent_renew", fence)
                _receipt(renewed, record, original=receipt)
                if renewed.get("renewed") is not True or renewed["authority_active"] is not True:
                    raise PermissionError("CAR did not confirm original Neo renewal")
                record.update(phase="confirmed", spec=fresh_spec, receipt=renewed)
                record.pop("pending_spec", None)
                _save(path, record)
                receipt = renewed
            managed = _managed(call, record)
            if managed["status"] in {"running", "starting", "backoff"}:
                return _view(record, receipt, managed["status"])
            started = call("agents.start", {"id": record["spec"]["id"]})
            if started.get("id") != record["spec"]["id"] or started.get("status") not in {"running", "starting"}:
                raise PermissionError("CAR did not confirm the admitted original Neo launch")
            after = call("agents.parent_lifecycle", {"agent_id": record["spec"]["id"]})
            _receipt(after, record, original=receipt)
            if after["intent_revision"] != receipt["intent_revision"] or after["authority_active"] is not True:
                raise PermissionError("Original Neo launch intent ended before confirmation")
            record["receipt"] = after
            _save(path, record)
            touch_seen(record["spec"]["id"])
            return _view(record, receipt, started["status"])
    except Exception as error:
        return {"status": "error", "work_context": ownership, "message": str(error)}
