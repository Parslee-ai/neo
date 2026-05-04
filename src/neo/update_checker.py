"""
Auto-update checker for Neo.

Checks PyPI for newer versions using stale-while-revalidate semantics:
the cached answer is returned immediately (zero added latency on the
hot path), and if the cache is older than UPDATE_CHECK_INTERVAL a
background thread refreshes the cache so the *next* invocation has
fresh data. Users on auto-update see new releases within ~1 hour
of publication, instead of up to 24h with the old fixed interval.

The auto-installer routes upgrades through whichever package manager
actually owns this neo install (pipx / pip-venv / brew / system pip).
Forcing pip into a foreign install (e.g. Homebrew Python's
externally-managed site-packages) used to write a duplicate copy that
importlib.metadata then resolved to the *old* version on next start —
producing an infinite "upgrading from X to Y" loop. See
_detect_install_method below.
"""

import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

# Configuration
UPDATE_CHECK_INTERVAL = 3600  # 1 hour in seconds. Was 24h, but with SWR
# the per-invocation cost of a stale check is zero (background refresh),
# so we can afford to be much more responsive to fresh releases.

PYPI_PACKAGE_NAME = "neo-reasoner"
PYPI_API_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE_NAME}/json"
REQUEST_TIMEOUT = 3  # seconds

# Install-method tags returned by _detect_install_method().
INSTALL_PIPX = "pipx"
INSTALL_PIP_VENV = "pip-venv"
INSTALL_BREW = "brew"
INSTALL_EXTERNAL = "external"  # PEP-668 / system Python; pip-upgrade is unsafe

_BREW_PREFIXES = ("/opt/homebrew/", "/usr/local/Cellar/", "/home/linuxbrew/")


def _brew_owns(formula: str) -> bool:
    """Return True if Homebrew has a formula with this name installed."""
    try:
        result = subprocess.run(
            ["brew", "list", "--formula", formula],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def _detect_install_method() -> str:
    """Identify how the running neo was installed.

    Returns one of INSTALL_PIPX / INSTALL_PIP_VENV / INSTALL_BREW /
    INSTALL_EXTERNAL. The caller uses this to pick the correct upgrade
    command — using pip on a brew install (or vice versa) leaves stale
    metadata behind and breaks future auto-updates.
    """
    try:
        import neo  # local import to avoid circular concerns
        pkg_path = str(Path(neo.__file__).resolve())
    except Exception:
        return INSTALL_EXTERNAL

    # pipx puts each app in its own venv under ~/.local/pipx/venvs/<app>/.
    # Match POSIX and Windows separators.
    if "/pipx/venvs/" in pkg_path or "\\pipx\\venvs\\" in pkg_path:
        return INSTALL_PIPX

    # An isolated, non-pipx venv: sys.prefix diverges from base_prefix and
    # pip operates safely on its own site-packages.
    if sys.prefix != sys.base_prefix:
        return INSTALL_PIP_VENV

    # Under a Homebrew prefix? Could still be "pip --break-system-packages
    # into brew's Python", which is NOT a brew formula install — confirm
    # with `brew list`.
    if pkg_path.startswith(_BREW_PREFIXES) and _brew_owns(PYPI_PACKAGE_NAME):
        return INSTALL_BREW

    return INSTALL_EXTERNAL


def _pip_install_cmd(package: str, quiet: bool = False) -> list[str]:
    """Build a pip install command for the *current* interpreter.

    Used only on the INSTALL_PIP_VENV path. Other install methods
    (pipx, brew, external) route through their own helpers.
    """
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", package]
    if quiet:
        cmd.append("--quiet")
    return cmd


def _get_cache_file() -> Path:
    """Get the path to the update check cache file."""
    cache_dir = Path.home() / ".neo"
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir / "update_check.json"


def _read_cache() -> Optional[dict]:
    """Read the update check cache if it exists and is valid."""
    cache_file = _get_cache_file()
    if not cache_file.exists():
        return None

    try:
        cache_data = json.loads(cache_file.read_text())
        return cache_data
    except (json.JSONDecodeError, OSError) as e:
        logger.debug(f"Failed to read update cache: {e}")
        return None


def _write_cache(current_version: str, latest_version: str) -> None:
    """Write update check results to cache atomically.

    Uses write-temp-then-rename so a background refresh thread and a
    foreground reader can't observe a partially-written file.
    """
    cache_file = _get_cache_file()
    cache_data = {
        "last_check": time.time(),
        "current_version": current_version,
        "latest_version": latest_version,
        "new_version": latest_version if latest_version != current_version else None
    }

    try:
        tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
        tmp.write_text(json.dumps(cache_data, indent=2))
        tmp.replace(cache_file)  # atomic on POSIX and Windows ≥ 3.3
    except OSError as e:
        logger.debug(f"Failed to write update cache: {e}")


def _get_current_version() -> str:
    """Get the currently installed version of neo-reasoner."""
    try:
        return importlib.metadata.version(PYPI_PACKAGE_NAME)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _fetch_latest_version_from_pypi() -> Optional[str]:
    """Fetch the latest version from PyPI API."""
    try:
        request = Request(
            PYPI_API_URL,
            headers={"User-Agent": f"{PYPI_PACKAGE_NAME}/{_get_current_version()}"}
        )

        with urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data["info"]["version"]
    except (URLError, json.JSONDecodeError, KeyError, OSError) as e:
        logger.debug(f"Failed to fetch latest version from PyPI: {e}")
        return None


def _should_check_for_updates() -> bool:
    """Determine if we should check for updates based on cache."""
    cache = _read_cache()
    if cache is None:
        return True

    last_check = cache.get("last_check", 0)
    time_since_check = time.time() - last_check

    return time_since_check >= UPDATE_CHECK_INTERVAL


def _compare_versions(current: str, latest: str) -> bool:
    """
    Compare version strings to determine if latest > current.

    Returns True if latest is newer than current.
    Simple lexicographic comparison for now - works for semantic versioning.
    """
    if current == "unknown":
        return False

    try:
        from packaging import version
        return version.parse(latest) > version.parse(current)
    except (ImportError, version.InvalidVersion):
        # Fallback: parse as tuples for proper comparison
        def parse_version(v):
            return tuple(int(x) for x in v.split('.') if x.isdigit())
        try:
            return parse_version(latest) > parse_version(current)
        except (ValueError, TypeError):
            return False  # Can't compare, assume no update


def _refresh_cache_sync(current_version: str) -> None:
    """Fetch the latest version from PyPI and write it to the cache.

    Exposed (with leading underscore) so tests can exercise the refresh
    logic without dealing with thread timing. Production code uses the
    background wrapper below.
    """
    try:
        latest = _fetch_latest_version_from_pypi()
        if latest:
            _write_cache(current_version, latest)
    except Exception as e:
        logger.debug(f"Update cache refresh failed: {e}")


def _refresh_in_background(current_version: str) -> threading.Thread:
    """Kick off an async cache refresh.

    Returns the thread so callers (tests, mostly) can `.join()` it. The
    thread is daemonic — it will be killed if the main process exits
    before it completes, which is acceptable for `neo --version` style
    quick exits. Long-running neo invocations easily outlive the refresh.
    """
    thread = threading.Thread(
        target=_refresh_cache_sync,
        args=(current_version,),
        daemon=True,
        name="neo-update-check",
    )
    thread.start()
    return thread


def check_for_updates(suppress_output: bool = False, auto_install: bool = False) -> Optional[str]:
    """
    Check PyPI for a newer version of neo-reasoner.

    Uses stale-while-revalidate semantics:
      - Fresh cache (age < UPDATE_CHECK_INTERVAL): return cached answer.
      - Stale cache: return cached answer immediately AND kick off a
        background refresh whose result will be available next call.
      - Cache miss: synchronous fetch (one-time tax for fresh installs).

    Args:
        suppress_output: If True, don't print update notifications
        auto_install: If True, automatically install updates when found

    Returns:
        The new version string if an update is available, None otherwise
    """
    if os.getenv("NEO_SKIP_UPDATE_CHECK"):
        return None

    try:
        current_version = _get_current_version()
        if current_version == "unknown":
            logger.debug("Could not determine current version")
            return None

        cache = _read_cache()

        # Cache miss: do a synchronous fetch so first-ever run isn't blind.
        if cache is None:
            latest_version = _fetch_latest_version_from_pypi()
            if latest_version is None:
                return None
            _write_cache(current_version, latest_version)
            cache = _read_cache()
            if cache is None:
                return None
        else:
            # Cache hit. If stale, start a background refresh so the
            # *next* invocation sees fresh data. This call still returns
            # the cached answer immediately — that's the SWR pattern.
            cache_age = time.time() - cache.get("last_check", 0)
            if cache_age >= UPDATE_CHECK_INTERVAL:
                _refresh_in_background(current_version)

        # Act on whatever's in the cache (just-fetched, fresh, or stale).
        new_version = cache.get("new_version")
        if new_version and _compare_versions(current_version, new_version):
            if auto_install:
                perform_auto_install(new_version)
            elif not suppress_output:
                _print_update_notification(current_version, new_version)
            return new_version

        return None

    except Exception as e:
        # Silent failure - don't disrupt user workflow
        logger.debug(f"Update check failed: {e}")
        return None


def _print_update_notification(current: str, latest: str) -> None:
    """Print a user-friendly update notification, tailored to install method."""
    method = _detect_install_method()
    print(f"\n⚡ Neo update available: {current} → {latest}", file=sys.stderr)
    if method == INSTALL_PIPX:
        print(f"   Run: pipx upgrade {PYPI_PACKAGE_NAME}", file=sys.stderr)
        print("   Or:  neo update\n", file=sys.stderr)
    elif method == INSTALL_PIP_VENV:
        print(f"   Run: pip install --upgrade {PYPI_PACKAGE_NAME}", file=sys.stderr)
        print("   Or:  neo update\n", file=sys.stderr)
    elif method == INSTALL_BREW:
        print(f"   Run: brew upgrade {PYPI_PACKAGE_NAME}", file=sys.stderr)
        print("   (Homebrew formula may lag PyPI by 24-48h.)\n", file=sys.stderr)
    else:  # INSTALL_EXTERNAL
        print(
            "   Your install is in a system / externally-managed Python.\n"
            f"   Switch to pipx for safe auto-updates:\n"
            f"     pipx install {PYPI_PACKAGE_NAME}\n",
            file=sys.stderr,
        )


def _notified_marker_path() -> Path:
    """Where we record the last version we told the user about."""
    return _get_cache_file().parent / "install_notified.json"


def _already_notified(version: str) -> bool:
    try:
        data = json.loads(_notified_marker_path().read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return data.get("notified_version") == version


def _mark_notified(version: str) -> None:
    try:
        _notified_marker_path().write_text(json.dumps({"notified_version": version}))
    except OSError as e:
        logger.debug(f"Failed to write notified marker: {e}")


def _append_log(message: str) -> None:
    """Append a line to ~/.neo/auto_update.log."""
    log_file = _get_cache_file().parent / "auto_update.log"
    try:
        with open(log_file, "a") as f:
            f.write(message)
    except OSError as e:
        logger.debug(f"Failed to write auto-update log: {e}")


def _run_pipx_upgrade(quiet: bool = True) -> tuple[bool, str]:
    """Run `pipx upgrade neo-reasoner`. Returns (success, stderr)."""
    try:
        result = subprocess.run(
            ["pipx", "upgrade", PYPI_PACKAGE_NAME],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0, result.stderr
    except FileNotFoundError:
        return False, "pipx not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "pipx upgrade timed out after 120s"


def _run_pip_upgrade(quiet: bool = True) -> tuple[bool, str]:
    """Run `pip install --upgrade neo-reasoner` in the current venv."""
    try:
        result = subprocess.run(
            _pip_install_cmd(PYPI_PACKAGE_NAME, quiet=quiet),
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0, result.stderr
    except subprocess.TimeoutExpired:
        return False, "pip upgrade timed out after 120s"


def perform_auto_install(new_version: str) -> bool:
    """
    Perform automatic background update installation.

    Dispatches to the appropriate package manager based on how neo was
    installed. For brew/external installs, prints guidance instead of
    pip-overriding (the latter creates duplicate metadata that breaks
    future detection — see auto_update.log loop bug, fixed here).

    Args:
        new_version: The version to install

    Returns:
        True if the upgrade actually happened. False if the upgrade
        failed, OR if the install method requires manual user action
        (brew/external) — in those cases we print guidance and return
        False so the caller knows neo did not self-update.
    """
    current_version = _get_current_version()
    if current_version == new_version:
        return True

    method = _detect_install_method()

    # Methods that require user action — print guidance, throttled per version.
    if method in (INSTALL_BREW, INSTALL_EXTERNAL):
        if not _already_notified(new_version):
            _print_update_notification(current_version, new_version)
            _mark_notified(new_version)
        _append_log(
            f"\n[{_now_iso()}] Update available {current_version} -> {new_version}"
            f"; install method '{method}' requires manual upgrade — skipping pip.\n"
        )
        return False

    # Auto-installable methods.
    timestamp = _now_iso()
    _append_log(
        f"\n[{timestamp}] Auto-updating from {current_version} to {new_version} via {method}\n"
    )
    print(
        f"\n⚡ Auto-installing neo update: {current_version} → {new_version} (via {method})",
        file=sys.stderr,
    )
    print("   This happens in the background. Please wait...\n", file=sys.stderr)

    try:
        if method == INSTALL_PIPX:
            ok, err = _run_pipx_upgrade(quiet=True)
        else:  # INSTALL_PIP_VENV
            ok, err = _run_pip_upgrade(quiet=True)
    except Exception as e:
        logger.debug(f"Auto-update failed: {e}")
        _append_log(f"   ✗ Failed: {e}\n")
        return False

    if ok:
        _append_log(f"   ✓ Success! Updated to {new_version}\n")
        print(f"✓ Auto-update completed: {new_version}", file=sys.stderr)
        print("   Restart neo to use the new version.\n", file=sys.stderr)
        # Clear the version-check cache so next run re-verifies.
        cache_file = _get_cache_file()
        if cache_file.exists():
            cache_file.unlink()
        return True

    _append_log(f"   ✗ Failed: {err}\n")
    logger.debug(f"Auto-update failed: {err}")
    return False


def _now_iso() -> str:
    import datetime
    return datetime.datetime.now().isoformat()


def perform_update() -> bool:
    """
    Perform a self-update, dispatching to the correct package manager
    for the current install method. Used by `neo update`.
    """
    current_version = _get_current_version()

    print("Checking for updates...")
    new_version = check_for_updates(suppress_output=True)

    if not new_version:
        print(f"✓ Neo is already up to date (version {current_version})")
        return True

    method = _detect_install_method()

    if method in (INSTALL_BREW, INSTALL_EXTERNAL):
        # Manual action required — print guidance and bail without trying pip.
        _print_update_notification(current_version, new_version)
        return False

    print(f"Updating {PYPI_PACKAGE_NAME} from {current_version} to {new_version} (via {method})...")

    try:
        if method == INSTALL_PIPX:
            ok, err = _run_pipx_upgrade(quiet=False)
        else:  # INSTALL_PIP_VENV
            ok, err = _run_pip_upgrade(quiet=False)
    except Exception as e:
        print(f"✗ Unexpected error during update: {e}", file=sys.stderr)
        return False

    if ok:
        print(f"✓ Successfully updated to version {new_version}")
        print("\nPlease restart neo for changes to take effect.")
        cache_file = _get_cache_file()
        if cache_file.exists():
            cache_file.unlink()
        return True

    print(f"✗ Update failed: {err}", file=sys.stderr)
    return False
