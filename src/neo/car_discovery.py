"""Local CAR installation discovery.

Neo can use CAR through several surfaces depending on what is installed:
Python bindings (`car_runtime` / `car_native`), the native `car` CLI, and the
`car-server` daemon. Discovery stays side-effect free so it is safe to run from
`neo --version` and status commands.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_DAEMON_HOST = "127.0.0.1"
DEFAULT_DAEMON_PORT = 9100


@dataclass
class CarInstallInfo:
    """Detected CAR surfaces on the local machine."""

    cli_path: Optional[str] = None
    cli_version: Optional[str] = None
    server_path: Optional[str] = None
    python_binding: Optional[str] = None
    daemon_running: bool = False
    daemon_url: str = f"ws://{DEFAULT_DAEMON_HOST}:{DEFAULT_DAEMON_PORT}"

    @property
    def available(self) -> bool:
        return bool(self.cli_path or self.server_path or self.python_binding or self.daemon_running)

    @property
    def has_python_runtime(self) -> bool:
        return self.python_binding is not None

    def summary(self) -> str:
        """One-line status for version output."""
        if not self.available:
            return "not found"
        parts: list[str] = []
        if self.cli_path:
            version = f" {self.cli_version}" if self.cli_version else ""
            parts.append(f"cli{version} at {self.cli_path}")
        if self.server_path:
            parts.append(f"server at {self.server_path}")
        if self.python_binding:
            parts.append(f"python binding {self.python_binding}")
        parts.append("daemon running" if self.daemon_running else "daemon not detected")
        return " | ".join(parts)


def _candidate_executable(name: str) -> Optional[str]:
    """Find an executable in PATH or common local CAR install directories."""
    found = shutil.which(name)
    if found:
        return found

    candidates = [
        Path.home() / ".car" / "bin" / name,
        Path("/opt/homebrew/bin") / name,
        Path("/usr/local/bin") / name,
        Path.home() / "git" / "car" / "target" / "debug" / name,
        Path.home() / "git" / "car" / "target" / "release" / name,
    ]
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def _version_for(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return (result.stdout or result.stderr).strip() or None


def _python_binding() -> Optional[str]:
    for module in ("car_runtime", "car_native"):
        if importlib.util.find_spec(module) is not None:
            return module
    return None


def _daemon_running(host: str = DEFAULT_DAEMON_HOST, port: int = DEFAULT_DAEMON_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def discover_car() -> CarInstallInfo:
    """Return local CAR availability without starting or installing anything."""
    cli_path = _candidate_executable("car")
    server_path = _candidate_executable("car-server")
    return CarInstallInfo(
        cli_path=cli_path,
        cli_version=_version_for(cli_path),
        server_path=server_path,
        python_binding=_python_binding(),
        daemon_running=_daemon_running(),
    )
