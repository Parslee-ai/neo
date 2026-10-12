"""Explicit organization enrollment through CAR's installed generic guardian."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import BinaryIO


# Complete installed consumer and guardian closures can exceed 8,192 pins.
MAX_OWNER_PINS = 32768

# Installed guardian inventories can contain thousands of full path/hash pins.
MAX_GUARDIAN_REQUEST_BYTES = 8 * 1024 * 1024


def _read_worker_request(stream: BinaryIO):
    payload = stream.read(MAX_GUARDIAN_REQUEST_BYTES + 1)
    if len(payload) > MAX_GUARDIAN_REQUEST_BYTES:
        raise ValueError("Native owner request exceeds its 8 MiB bound")
    return json.loads(payload.decode("utf-8"))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _config(value):
    required = {"guardian_python", "guardian_scripts", "child_python", "child_sdk", "child_package", "ownership_root", "pins"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Complete explicit native owner configuration required")
    value = copy.deepcopy(value)
    for field in required - {"pins"}:
        if not isinstance(value[field], str) or not Path(value[field]).is_absolute():
            raise ValueError("Absolute native owner paths required")
    pins = value["pins"]
    if not isinstance(pins, dict) or not 4 <= len(pins) <= MAX_OWNER_PINS:
        raise ValueError("Pinned guardian and child package identity required")
    needed = {value["guardian_python"], value["child_python"], value["child_sdk"],
              str(Path(value["guardian_scripts"]) / "supervised-owner.py")}
    package = Path(value["child_package"])
    sources = list(package.rglob("*.py"))
    if not sources or package.is_symlink() or (package / ".env").exists():
        raise ValueError("Pinned isolated Neo package without environment fallback required")
    needed.update(str(path) for path in sources)
    if any(path.is_symlink() for path in sources):
        raise ValueError("Isolated Neo source must not use ambient source links")
    if not needed <= pins.keys():
        raise ValueError("Guardian interpreter, entrypoint and child SDK pins required")
    for name, digest in pins.items():
        if not Path(name).is_absolute() or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid native owner artifact pin")
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("Native owner artifact changed: " + name)
    if not os.access(value["guardian_python"], os.X_OK) or not os.access(value["child_python"], os.X_OK):
        raise ValueError("Executable native owner interpreters required")
    return value


def _run(config, action, spec_path, spec):
    # The helper runs under the inventoried guardian interpreter, never the Neo SDK.
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL",
           "CAR_HOME", "CAR_KEYCHAIN_PATH", "CAR_KEYCHAIN_PROOF_ROOT", "CAR_DAEMON_URL") if k in os.environ}
    request = {"config": config, "action": action, "spec_path": str(spec_path), "agent_id": spec["id"]}
    payload = _canonical(request)
    if len(payload.encode("utf-8")) > MAX_GUARDIAN_REQUEST_BYTES:
        raise ValueError("Native owner request exceeds its 8 MiB bound")
    result = subprocess.run([config["guardian_python"], "-I", "-B", str(Path(__file__).resolve()),
                             "--guardian-worker"], input=payload, text=True,
                            capture_output=True, env=env)
    if result.returncode != 0:
        fd = _directory(config["ownership_root"])
        name = "operation-" + secrets.token_hex(8) + ".diagnostic.json"
        try:
            output = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            with os.fdopen(output, "w") as stream:
                json.dump({"raw_exit": result.returncode, "stdout": result.stdout[:32768],
                           "stderr": result.stderr[:32768]}, stream)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
        raise RuntimeError("Native original owner operation refused; private diagnostic: " +
                           str(Path(config["ownership_root"]) / name))
    if len(result.stdout.encode()) > 1024 * 1024:
        raise RuntimeError("Native owner result exceeds its bound")
    return json.loads(result.stdout)


def _directory(path):
    """Open/create the private owner root without following any path component."""
    directory = Path(path)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in directory.parts[1:]:
            try:
                following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
                following = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = following
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("Private original ownership root required")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _worker(request):
    config = _config(request["config"])
    sys.path.insert(0, config["guardian_scripts"])
    from supervised_launcher import dependencies, guardian
    dependencies.verify()
    if request["action"] == "inspect":
        from car_runtime import CarRuntime, HostClient
        runtime = CarRuntime()
        try:
            return json.loads(HostClient(runtime).daemon_call_host_management(
                "agents.parent_lifecycle", json.dumps({"agent_id": request["agent_id"]})))
        finally:
            runtime.close()
    args = argparse.Namespace(approved_spec=Path(request["spec_path"]), assistant=None,
                              resume_id=None, mcp_binary=None,
                              renderer=Path(config["guardian_scripts"]) / "supervised-mcp-config.py")
    directory = Path(config["ownership_root"])
    binding, identity, contract = guardian.launch_binding(args, os.environ, directory)
    if request["action"] == "join":
        guardian.ensure_persistent_original_owner(args, directory, binding, identity, contract)
    operation = "stop" if request["action"] == "leave" else "status"
    channel, status = guardian.connect(directory, binding, identity, contract, operation)
    channel.close()
    return status


def _receipt(value, record, original=None):
    from neo.lattice_enrollment import _receipt as check_receipt
    check_receipt(value, record, original=original)
    if not value.get("task_incarnation") or not value.get("daemon_instance_id"):
        raise PermissionError("Original native task and daemon identity required")
    return value


def _stopped(status, record):
    value = status.get("lifecycle")
    if not isinstance(value, dict):
        raise PermissionError("Original native termination receipt required")
    _receipt(value, record, record.get("receipt"))
    if (status.get("phase") not in {"stopped", "failed"} or value.get("ended") is not True
            or value.get("cleanup_confirmed") is not True or value["authority_active"]
            or value["renewal_allowed"] or value.get("work_revocation_pending") is True):
        raise PermissionError("Original native termination and cleanup remain unconfirmed")
    return value


def operation(action, root, project, context, project_ids, native_owner):
    from neo import lattice
    from neo import lattice_enrollment as enrollment
    ownership = context
    try:
        context = enrollment.work_context(context)
        if any(os.environ.get(k) for k in ("CAR_AGENT_ID", "CAR_AGENT_TOKEN", "CAR_WORK_CONTEXT")):
            raise PermissionError("Native enrollment requires the authenticated operator")
        config = _config(native_owner)
        root = lattice.repository_root(root) or os.path.abspath(root)
        with enrollment._record_lock(enrollment._key(root, context)) as path:
            record = enrollment._load(path)
            new = record is None
            if record is not None:
                if record.get("backend") != "native_guardian":
                    # Never pass a legacy managed record to native cohort cleanup.
                    raise PermissionError("Legacy managed enrollment requires proven original termination and explicit migration")
                if (record["work_context"] != context or record["root"] != root
                        or record["daemon_url"] != lattice.daemon_url() or record["native_owner"] != config):
                    raise PermissionError("Original native owner configuration or work context changed")
                if action == "join" and record["phase"] != "confirmed":
                    raise PermissionError("Original admission is pending; reconcile status or leave before retrying")
                if ((project is not None and project != record["project"])
                        or (project_ids is not None and enrollment.projects(project_ids) != record["project_ids"])):
                    raise PermissionError("Original approved project scope is immutable")
                if action == "join" and record.get("left"):
                    _stopped({"phase": "stopped", "lifecycle": record["receipt"]}, record)
                    original = _run(config, "inspect", Path(config["ownership_root"]) /
                                    (record["spec"]["id"] + ".approved.json"), record["spec"])
                    _receipt(original, record, record["receipt"])
                    if (original.get("daemon_instance_id") != record["receipt"]["daemon_instance_id"]
                            or original.get("state") != "present" or original["authority_active"]
                            or original["renewal_allowed"] or original.get("work_revocation_pending") is True):
                        raise PermissionError("Original native task termination no longer holds")
                    archive = path.with_name(path.stem + "-ended-" +
                              hashlib.sha256(original["task_incarnation"].encode()).hexdigest()[:16] + ".json")
                    enrollment._save(archive, record)
                    project = record["project"]
                    project_ids = record["project_ids"]
                    record = None
                    new = True
            if record is None and action != "join":
                return {"status": "not_joined", "work_context": context, "root": root}
            if record is None:
                approved = enrollment.projects(project_ids)
                project = project or lattice.derive_project_name(root)
                spec = lattice.build_spec(root, project, context)
                spec.update(id=spec["id"] + "-" + secrets.token_hex(6), command=config["child_python"],
                            auto_start=False, env={})
                spec.pop("token", None)
                spec["work_project_ids"] = approved
                record = {"version": 1, "backend": "native_guardian", "phase": "pending_create",
                          "root": root, "project": project, "work_context": context,
                          "project_ids": approved, "spec": spec, "native_owner": config,
                          "daemon_url": lattice.daemon_url()}
                # Persist the exact original identity before any launch effect.
                enrollment._save(path, record)
            if record.get("left"):
                return dict(enrollment._view(record, record["receipt"], "left"), process_status="stopped")
            spec = record["spec"]
            directory = Path(config["ownership_root"])
            directory_fd = _directory(directory)
            spec_path = directory / (spec["id"] + ".approved.json")
            encoded = _canonical(spec) + "\n"
            try:
                try:
                    fd = os.open(spec_path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory_fd)
                except FileExistsError:
                    fd = os.open(spec_path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
                    with os.fdopen(fd) as stream:
                        info = os.fstat(stream.fileno())
                        if info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_nlink != 1 or stream.read() != encoded:
                            raise PermissionError("Original approved specification changed")
                else:
                    with os.fdopen(fd, "w") as stream:
                        stream.write(encoded)
                        stream.flush()
                        os.fsync(stream.fileno())
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            if new:
                initial = _run(config, "inspect", spec_path, spec)
                if (initial.get("version") != 2 or initial.get("state") != "absent"
                        or initial.get("agent_id") != spec["id"] or not initial.get("daemon_instance_id")):
                    raise PermissionError("Fresh native original identity is not absent; admission remains pending")
            if action == "leave":
                record["phase"] = "pending_cleanup"
                enrollment._save(path, record)
            status = _run(config, action, spec_path, spec)
            if action == "leave":
                observed = _stopped(status, record)
                record.update(receipt=observed, phase="confirmed", left=True)
                state = "left"
            elif action == "status" and record["phase"] == "pending_cleanup" and status.get("phase") in {"stopped", "failed"}:
                observed = _stopped(status, record)
                record.update(receipt=observed, phase="confirmed", left=True)
                state = "left"
            else:
                observed = status.get("lifecycle")
                if not isinstance(observed, dict):
                    raise PermissionError("Authenticated original guardian receipt required")
                _receipt(observed, record, record.get("receipt"))
                state = status.get("phase")
                if state not in {"starting", "running", "recovering", "stopped", "failed"}:
                    raise PermissionError("Unknown original guardian process state")
                if action == "join" and (not observed["authority_active"]
                        or observed.get("work_revocation_pending") is True):
                    raise PermissionError("Original guardian has no active work authority")
                if record["phase"] == "pending_cleanup":
                    state = "cleanup_pending"
                record.update(receipt=observed)
                if record["phase"] == "pending_create":
                    if (state not in {"starting", "running", "recovering"}
                            or observed["authority_active"] is not True
                            or observed.get("work_revocation_pending") is True):
                        raise PermissionError("Original native admission remains unconfirmed; reconcile leave")
                    record["phase"] = "confirmed"
            enrollment._save(path, record)
            return dict(enrollment._view(record, observed, state), process_status=status["phase"],
                        cleanup_pending=record["phase"] == "pending_cleanup"
                        or observed.get("work_revocation_pending") is True)
    except Exception as error:
        return {"status": "error", "work_context": ownership, "message": str(error)}


if __name__ == "__main__":
    if sys.argv[1:] != ["--guardian-worker"]:
        raise SystemExit("Internal guardian bridge invocation required")
    request = _read_worker_request(sys.stdin.buffer)
    print(_canonical(_worker(request)))
