"""Async out-of-band synthesis observer.

A long-lived background process that runs REVIEW→PATTERN/FAILURE synthesis
on a wall-clock cadence, decoupled from the request path. Modeled after
ECC's continuous-learning-v2 observer (skills/continuous-learning-v2/
agents/observer-loop.sh).

The observer is *additive* to the inline triple-trigger gate in
``FactStore.synthesize_reviews``: both paths call the same entry point, the
gate inside ``synthesize_reviews`` is idempotent (watermark-protected), and
running the observer just makes synthesis fire more often than the inline
gate alone would. Operators who want only out-of-band synthesis can lift
``synthesize_reviews``'s gate in a follow-up.

Lifecycle:
    neo memory observer start    # spawn, write PID, write log
    neo memory observer status   # PID liveness + last-analysis epoch
    neo memory observer kick     # SIGUSR1 — force an early cycle (cooldown applies)
    neo memory observer stop     # SIGTERM

Per-project state lives under ``~/.neo/sessions/<project_id>/``:
    .observer.pid          — PID of the running observer
    .observer.log          — appended log lines
    .observer.last_analysis — epoch of last successful synthesis call

Tunables (env):
    NEO_OBSERVER_INTERVAL_SECONDS  — wake cadence (default 300)
    NEO_OBSERVER_COOLDOWN          — min seconds between analyses (default 60)
    NEO_OBSERVER_IDLE_SECONDS      — exit if no metrics activity for this long (default 1800)
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from neo.memory.scope import detect_org_and_project

logger = logging.getLogger(__name__)


def _sessions_dir(project_id: str) -> Path:
    """Project-scoped state directory. Created on demand by start_observer."""
    return Path.home() / ".neo" / "sessions" / project_id


def _pid_file(project_id: str) -> Path:
    return _sessions_dir(project_id) / ".observer.pid"


def _log_file(project_id: str) -> Path:
    return _sessions_dir(project_id) / ".observer.log"


def _last_analysis_file(project_id: str) -> Path:
    return _sessions_dir(project_id) / ".observer.last_analysis"


def _metrics_file() -> Path:
    return Path.home() / ".neo" / "metrics.jsonl"


def _read_pid(project_id: str) -> Optional[int]:
    pf = _pid_file(project_id)
    if not pf.exists():
        return None
    try:
        return int(pf.read_text().strip())
    except (ValueError, OSError):
        return None


def _pid_alive(pid: int) -> bool:
    """``kill -0`` semantics: signal 0 raises if PID is dead."""
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _resolve_project_id(codebase_root: Optional[str]) -> str:
    _, project_id = detect_org_and_project(codebase_root or os.getcwd())
    return project_id


@dataclass
class ObserverConfig:
    """Tunables for one observer run. Resolved from env in ``from_env``."""

    interval_seconds: float = 300.0
    cooldown_seconds: float = 60.0
    idle_seconds: float = 1800.0

    @classmethod
    def from_env(cls) -> "ObserverConfig":
        def _read(name: str, default: float) -> float:
            raw = os.getenv(name, "").strip()
            if not raw:
                return default
            try:
                v = float(raw)
                return v if v > 0 else default
            except ValueError:
                return default

        return cls(
            interval_seconds=_read("NEO_OBSERVER_INTERVAL_SECONDS", 300.0),
            cooldown_seconds=_read("NEO_OBSERVER_COOLDOWN", 60.0),
            idle_seconds=_read("NEO_OBSERVER_IDLE_SECONDS", 1800.0),
        )


class Observer:
    """Per-project synthesis loop. One instance per running process."""

    def __init__(self, codebase_root: str, config: Optional[ObserverConfig] = None):
        self.codebase_root = codebase_root
        self.config = config or ObserverConfig.from_env()
        self.project_id = _resolve_project_id(codebase_root)
        if not self.project_id:
            raise RuntimeError(
                "Cannot run observer without a resolvable project_id "
                "(no codebase_root and no git repo in cwd)"
            )
        self._sessions = _sessions_dir(self.project_id)
        self._sessions.mkdir(parents=True, exist_ok=True)
        self._stop = False
        self._kick = False
        self._last_analysis_epoch = 0.0
        self._start_time = time.time()

    def run(self) -> None:
        """Main loop. Returns when SIGTERM/SIGINT received or idle-timeout hits."""
        signal.signal(signal.SIGTERM, self._handle_stop)
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGUSR1, self._handle_kick)

        self._log(f"observer started pid={os.getpid()} project={self.project_id[:8]}")
        _pid_file(self.project_id).write_text(str(os.getpid()))

        try:
            while not self._stop:
                if self._idle_too_long():
                    self._log("idle timeout — exiting")
                    break

                if self._cooldown_ok():
                    self._cycle()

                # Sleep in 1-second slices so SIGUSR1 / SIGTERM are responsive.
                slept = 0.0
                while slept < self.config.interval_seconds and not self._stop:
                    if self._kick:
                        self._kick = False
                        self._log("kick received — running cycle")
                        if self._cooldown_ok():
                            self._cycle()
                        else:
                            self._log("kick suppressed by cooldown")
                    time.sleep(1.0)
                    slept += 1.0
        finally:
            try:
                _pid_file(self.project_id).unlink(missing_ok=True)
            except OSError:
                pass
            self._log("observer stopped")

    def _handle_stop(self, _signum: int, _frame) -> None:
        self._stop = True

    def _handle_kick(self, _signum: int, _frame) -> None:
        self._kick = True

    def _cooldown_ok(self) -> bool:
        elapsed = time.time() - self._last_analysis_epoch
        return elapsed >= self.config.cooldown_seconds

    def _idle_too_long(self) -> bool:
        """Exit if metrics.jsonl hasn't been touched in ``idle_seconds``.

        Treats a missing metrics file as "not idle" — neo just hasn't run
        anything yet on this machine, but might in a moment. The reference
        time is ``max(mtime, _start_time)``: an observer launched against a
        stale metrics file gets a full ``idle_seconds`` grace window to see
        activity before exiting, rather than self-exiting on iteration 1.
        """
        mf = _metrics_file()
        if not mf.exists():
            return False
        try:
            mtime = mf.stat().st_mtime
        except OSError:
            return False
        reference = max(mtime, self._start_time)
        return (time.time() - reference) > self.config.idle_seconds

    def _cycle(self) -> None:
        """One synthesis pass. Errors are caught and logged — never propagate."""
        try:
            # Lazy import — `Observer` is a daemon entrypoint, and we want
            # the heavy FactStore import to happen inside the child process
            # so `start_observer` returns fast.
            from neo.memory.store import FactStore

            store = FactStore(codebase_root=self.codebase_root, eager_init=False)
            store.initialize()
            count = store.synthesize_reviews()
            self._last_analysis_epoch = time.time()
            _last_analysis_file(self.project_id).write_text(str(self._last_analysis_epoch))
            self._log(f"cycle ok: {count} synthesized facts")
        except Exception as e:  # never let one bad cycle kill the loop
            self._log(f"cycle error: {type(e).__name__}: {e}")

    def _log(self, message: str) -> None:
        try:
            with _log_file(self.project_id).open("a") as f:
                f.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}\n")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Public lifecycle API (called from subcommands.py)
# ---------------------------------------------------------------------------


def start_observer(codebase_root: Optional[str] = None) -> dict:
    """Spawn a detached observer process for the resolved project.

    Returns a status dict: ``{"status": "started"|"already_running"|"error",
    "pid": int|None, "project_id": str, "message": str}``. The parent process
    returns immediately — the daemon does its own logging.
    """
    project_id = _resolve_project_id(codebase_root)
    if not project_id:
        return {"status": "error", "pid": None, "project_id": "",
                "message": "No project_id (run from a git repo or pass --cwd)"}

    existing = _read_pid(project_id)
    if existing and _pid_alive(existing):
        return {"status": "already_running", "pid": existing,
                "project_id": project_id, "message": f"observer pid {existing}"}

    if existing:
        _pid_file(project_id).unlink(missing_ok=True)

    _sessions_dir(project_id).mkdir(parents=True, exist_ok=True)
    log_path = _log_file(project_id)
    log_path.touch(exist_ok=True)

    # Re-exec ourselves into the hidden daemon entrypoint. `start_new_session`
    # detaches the child from the parent's controlling terminal so closing the
    # shell doesn't kill it.
    env = os.environ.copy()
    cmd = [sys.executable, "-m", "neo.memory.observer", "--daemon",
           "--cwd", codebase_root or os.getcwd()]
    try:
        with log_path.open("a") as log_fp:
            proc = subprocess.Popen(  # noqa: S603 — argv is constructed, not shell
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_fp,
                stderr=log_fp,
                start_new_session=True,
                env=env,
                close_fds=True,
            )
    except OSError as e:
        return {"status": "error", "pid": None, "project_id": project_id,
                "message": f"spawn failed: {e}"}

    # Give the child a moment to write its PID file. Bounded poll, not sleep,
    # so we fail fast if the child died on start.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if _read_pid(project_id):
            break
        if proc.poll() is not None:
            return {"status": "error", "pid": None, "project_id": project_id,
                    "message": f"daemon exited immediately (code {proc.returncode})"}
        time.sleep(0.05)

    pid = _read_pid(project_id)
    if not pid:
        return {"status": "error", "pid": None, "project_id": project_id,
                "message": "daemon did not write PID file within 3s"}

    return {"status": "started", "pid": pid, "project_id": project_id,
            "message": f"observer pid {pid}, log {log_path}"}


def stop_observer(codebase_root: Optional[str] = None) -> dict:
    project_id = _resolve_project_id(codebase_root)
    if not project_id:
        return {"status": "error", "message": "No project_id"}

    pid = _read_pid(project_id)
    if not pid:
        return {"status": "not_running", "pid": None, "project_id": project_id,
                "message": "no PID file"}
    if not _pid_alive(pid):
        _pid_file(project_id).unlink(missing_ok=True)
        return {"status": "not_running", "pid": pid, "project_id": project_id,
                "message": "stale PID file removed"}

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        return {"status": "error", "pid": pid, "project_id": project_id,
                "message": f"SIGTERM failed: {e}"}
    return {"status": "stopped", "pid": pid, "project_id": project_id,
            "message": f"SIGTERM sent to pid {pid}"}


def kick_observer(codebase_root: Optional[str] = None) -> dict:
    """Send SIGUSR1 to force an early cycle (still subject to cooldown)."""
    project_id = _resolve_project_id(codebase_root)
    if not project_id:
        return {"status": "error", "message": "No project_id"}

    pid = _read_pid(project_id)
    if not pid or not _pid_alive(pid):
        return {"status": "not_running", "pid": None, "project_id": project_id,
                "message": "no running observer"}

    try:
        os.kill(pid, signal.SIGUSR1)
    except OSError as e:
        return {"status": "error", "pid": pid, "project_id": project_id,
                "message": f"SIGUSR1 failed: {e}"}
    return {"status": "kicked", "pid": pid, "project_id": project_id,
            "message": f"SIGUSR1 sent to pid {pid}"}


def observer_status(codebase_root: Optional[str] = None) -> dict:
    project_id = _resolve_project_id(codebase_root)
    if not project_id:
        return {"status": "error", "message": "No project_id"}

    pid = _read_pid(project_id)
    if not pid:
        return {"status": "not_running", "pid": None, "project_id": project_id,
                "message": "no PID file"}
    if not _pid_alive(pid):
        return {"status": "stale", "pid": pid, "project_id": project_id,
                "message": "PID file references dead process"}

    last_analysis_path = _last_analysis_file(project_id)
    last_analysis: Optional[float] = None
    if last_analysis_path.exists():
        try:
            last_analysis = float(last_analysis_path.read_text().strip())
        except (ValueError, OSError):
            pass

    return {
        "status": "running",
        "pid": pid,
        "project_id": project_id,
        "log_file": str(_log_file(project_id)),
        "last_analysis_epoch": last_analysis,
        "message": f"observer pid {pid}",
    }


# ---------------------------------------------------------------------------
# Daemon entrypoint — `python -m neo.memory.observer --daemon --cwd <path>`
# ---------------------------------------------------------------------------


def _daemon_main(argv: list[str]) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="neo.memory.observer")
    p.add_argument("--daemon", action="store_true", required=True,
                   help="(internal) marks this as the daemon entrypoint")
    p.add_argument("--cwd", required=True, help="codebase root for project resolution")
    args = p.parse_args(argv)

    try:
        observer = Observer(codebase_root=args.cwd)
    except RuntimeError as e:
        print(f"observer init failed: {e}", file=sys.stderr)
        return 1

    observer.run()
    return 0


if __name__ == "__main__":
    sys.exit(_daemon_main(sys.argv[1:]))
