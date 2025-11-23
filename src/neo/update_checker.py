"""
Auto-update checker for Neo.

Periodically checks PyPI for newer versions and notifies users.
Uses local caching to avoid excessive API calls.
"""

import importlib.metadata
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

# Configuration
UPDATE_CHECK_INTERVAL = 86400  # 24 hours in seconds
PYPI_PACKAGE_NAME = "neo-reasoner"
PYPI_API_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE_NAME}/json"
REQUEST_TIMEOUT = 3  # seconds


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
    """Write update check results to cache."""
    cache_file = _get_cache_file()
    cache_data = {
        "last_check": time.time(),
        "current_version": current_version,
        "latest_version": latest_version,
        "new_version": latest_version if latest_version != current_version else None
    }

    try:
        cache_file.write_text(json.dumps(cache_data, indent=2))
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
    except ImportError:
        # Fallback to simple string comparison if packaging not available
        return latest != current


def check_for_updates(suppress_output: bool = False, auto_install: bool = False) -> Optional[str]:
    """
    Check PyPI for a newer version of neo-reasoner.

    Args:
        suppress_output: If True, don't print update notifications
        auto_install: If True, automatically install updates when found

    Returns:
        The new version string if an update is available, None otherwise
    """
    # Respect environment variable to skip update checks
    if os.getenv("NEO_SKIP_UPDATE_CHECK"):
        return None

    try:
        current_version = _get_current_version()

        if current_version == "unknown":
            logger.debug("Could not determine current version")
            return None

        # Check cache first
        if not _should_check_for_updates():
            cache = _read_cache()
            if cache:
                new_version = cache.get("new_version")
                if new_version:
                    # Auto-install if enabled (even from cache)
                    if auto_install:
                        perform_auto_install(new_version)
                    elif not suppress_output:
                        _print_update_notification(current_version, new_version)
                return new_version

        # Fetch latest version from PyPI
        latest_version = _fetch_latest_version_from_pypi()

        if latest_version is None:
            logger.debug("Could not fetch latest version from PyPI")
            return None

        # Update cache
        _write_cache(current_version, latest_version)

        # Check if update is available
        if _compare_versions(current_version, latest_version):
            # Auto-install if enabled
            if auto_install:
                perform_auto_install(latest_version)
            elif not suppress_output:
                _print_update_notification(current_version, latest_version)
            return latest_version

        return None

    except Exception as e:
        # Silent failure - don't disrupt user workflow
        logger.debug(f"Update check failed: {e}")
        return None


def _print_update_notification(current: str, latest: str) -> None:
    """Print a user-friendly update notification."""
    print(f"\n⚡ Neo update available: {current} → {latest}", file=sys.stderr)
    print(f"   Run: pip install --upgrade {PYPI_PACKAGE_NAME}", file=sys.stderr)
    print(f"   Or:  neo update\n", file=sys.stderr)


def perform_auto_install(new_version: str) -> bool:
    """
    Perform automatic background update installation.

    This is more aggressive than perform_update() - it installs without prompting.
    Should only be called when auto_install_updates is enabled in config.

    Args:
        new_version: The version to install

    Returns:
        True if update was successful, False otherwise
    """
    import subprocess

    current_version = _get_current_version()

    try:
        # Write update log
        log_file = _get_cache_file().parent / "auto_update.log"
        with open(log_file, 'a') as f:
            import datetime
            timestamp = datetime.datetime.now().isoformat()
            f.write(f"\n[{timestamp}] Auto-updating from {current_version} to {new_version}\n")

        # Notify user that auto-update is happening
        print(f"\n⚡ Auto-installing neo update: {current_version} → {new_version}", file=sys.stderr)
        print(f"   This happens in the background. Please wait...\n", file=sys.stderr)

        # Use pip to upgrade
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", PYPI_PACKAGE_NAME, "--quiet"],
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        if result.returncode == 0:
            # Log success
            with open(log_file, 'a') as f:
                f.write(f"   ✓ Success! Updated to {new_version}\n")

            print(f"✓ Auto-update completed: {new_version}", file=sys.stderr)
            print(f"   Restart neo to use the new version.\n", file=sys.stderr)

            # Clear cache so next check will re-verify
            cache_file = _get_cache_file()
            if cache_file.exists():
                cache_file.unlink()

            return True
        else:
            # Log failure
            with open(log_file, 'a') as f:
                f.write(f"   ✗ Failed: {result.stderr}\n")

            logger.debug(f"Auto-update failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.debug("Auto-update timed out after 120 seconds")
        return False
    except Exception as e:
        logger.debug(f"Auto-update failed: {e}")
        return False


def perform_update() -> bool:
    """
    Perform a self-update using pip.

    Returns:
        True if update was successful, False otherwise
    """
    import subprocess

    current_version = _get_current_version()

    # Check if update is available
    print("Checking for updates...")
    new_version = check_for_updates(suppress_output=True)

    if not new_version:
        print(f"✓ Neo is already up to date (version {current_version})")
        return True

    print(f"Updating {PYPI_PACKAGE_NAME} from {current_version} to {new_version}...")

    try:
        # Use pip to upgrade
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", PYPI_PACKAGE_NAME],
            capture_output=True,
            text=True,
            check=True
        )

        print(f"✓ Successfully updated to version {new_version}")
        print("\nPlease restart neo for changes to take effect.")

        # Clear cache so next check will re-verify
        cache_file = _get_cache_file()
        if cache_file.exists():
            cache_file.unlink()

        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Update failed: {e.stderr}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"✗ Unexpected error during update: {e}", file=sys.stderr)
        return False
