#!/usr/bin/env python3
"""
Comprehensive tests for the update_checker module.

Tests cover:
1. Version comparison edge cases (_compare_versions function)
2. Cache expiry logic and _should_check_for_updates() behavior
3. Auto-install idempotency
4. check_for_updates() behavior with cache hit/miss and auto_install
5. Timeout handling
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure neo is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neo import update_checker
from neo.update_checker import (
    UPDATE_CHECK_INTERVAL,
    _compare_versions,
    _get_cache_file,
    _get_current_version,
    _read_cache,
    _should_check_for_updates,
    _write_cache,
    check_for_updates,
    perform_auto_install,
    perform_update,
)


class TestCompareVersions:
    """Test _compare_versions function with various edge cases."""

    def test_normal_version_comparison_newer(self):
        """Test that a newer version returns True."""
        assert _compare_versions("0.9.0", "0.10.0") is True
        assert _compare_versions("1.0.0", "1.0.1") is True
        assert _compare_versions("1.0.0", "2.0.0") is True

    def test_normal_version_comparison_older(self):
        """Test that an older version returns False."""
        assert _compare_versions("0.10.0", "0.9.0") is False
        assert _compare_versions("1.0.1", "1.0.0") is False
        assert _compare_versions("2.0.0", "1.0.0") is False

    def test_same_versions_returns_false(self):
        """Test that identical versions return False (no update needed)."""
        assert _compare_versions("0.9.0", "0.9.0") is False
        assert _compare_versions("1.0.0", "1.0.0") is False
        assert _compare_versions("10.20.30", "10.20.30") is False

    def test_unknown_current_version_returns_false(self):
        """Test that 'unknown' current version returns False."""
        assert _compare_versions("unknown", "1.0.0") is False
        assert _compare_versions("unknown", "0.0.1") is False

    def test_multi_digit_version_parts(self):
        """Test versions with multi-digit parts (e.g., 0.10.0 > 0.9.0)."""
        assert _compare_versions("0.9.0", "0.10.0") is True
        assert _compare_versions("0.9.9", "0.10.0") is True
        assert _compare_versions("0.99.0", "0.100.0") is True

    def test_invalid_version_strings(self):
        """Test behavior with invalid version strings."""
        # These should not crash, just return False when comparison fails
        result = _compare_versions("abc", "1.0.0")
        assert isinstance(result, bool)

    def test_version_with_alpha_beta_rc(self):
        """Test pre-release versions (alpha, beta, rc) if packaging supports them."""
        try:
            from packaging import version  # noqa: F401
            # With packaging module, pre-release versions are handled
            assert _compare_versions("1.0.0a1", "1.0.0") is True
            assert _compare_versions("1.0.0b1", "1.0.0") is True
            assert _compare_versions("1.0.0rc1", "1.0.0") is True
            assert _compare_versions("1.0.0", "1.0.0a1") is False
        except ImportError:
            pytest.skip("packaging module not available for pre-release version tests")


class TestCacheExpiry:
    """Test cache expiry logic and _should_check_for_updates() behavior."""

    def test_should_check_when_no_cache_exists(self):
        """Test that check is needed when cache file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                update_checker,
                "_get_cache_file",
                return_value=Path(tmpdir) / "nonexistent.json",
            ):
                assert _should_check_for_updates() is True

    def test_should_not_check_when_cache_is_fresh(self):
        """Test that check is skipped when cache is within 24 hours."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "update_check.json"
            cache_data = {
                "last_check": time.time(),
                "current_version": "0.9.0",
                "latest_version": "0.9.0",
                "new_version": None,
            }
            cache_file.write_text(json.dumps(cache_data))

            with patch.object(
                update_checker, "_get_cache_file", return_value=cache_file
            ):
                assert _should_check_for_updates() is False

    def test_should_check_when_cache_expired(self):
        """Test that check is needed when cache is older than 24 hours."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "update_check.json"
            old_timestamp = time.time() - UPDATE_CHECK_INTERVAL - 1
            cache_data = {
                "last_check": old_timestamp,
                "current_version": "0.9.0",
                "latest_version": "0.9.0",
                "new_version": None,
            }
            cache_file.write_text(json.dumps(cache_data))

            with patch.object(
                update_checker, "_get_cache_file", return_value=cache_file
            ):
                assert _should_check_for_updates() is True

    def test_24_hour_boundary_exactly(self):
        """Test behavior at exactly 24 hours (boundary condition)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "update_check.json"
            boundary_timestamp = time.time() - UPDATE_CHECK_INTERVAL
            cache_data = {
                "last_check": boundary_timestamp,
                "current_version": "0.9.0",
                "latest_version": "0.9.0",
                "new_version": None,
            }
            cache_file.write_text(json.dumps(cache_data))

            with patch.object(
                update_checker, "_get_cache_file", return_value=cache_file
            ):
                result = _should_check_for_updates()
                assert result is True

    def test_should_check_with_corrupted_cache(self):
        """Test that check is needed when cache is corrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "update_check.json"
            cache_file.write_text("not valid json{{{")

            with patch.object(
                update_checker, "_get_cache_file", return_value=cache_file
            ):
                assert _should_check_for_updates() is True


class TestAutoInstallIdempotency:
    """Test perform_auto_install skips if already on target version."""

    def test_skip_install_when_already_on_target_version(self):
        """Test that auto_install is skipped when current version matches target."""
        with patch.object(
            update_checker, "_get_current_version", return_value="1.0.0"
        ):
            result = perform_auto_install("1.0.0")
            assert result is True

    def test_attempts_install_when_versions_differ(self):
        """Test that auto_install runs pip when versions differ."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "update_check.json"

            with patch.object(
                update_checker, "_get_current_version", return_value="0.9.0"
            ):
                with patch.object(
                    update_checker, "_get_cache_file", return_value=cache_file
                ):
                    with patch("subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(returncode=0)
                        result = perform_auto_install("1.0.0")
                        assert mock_run.called
                        assert result is True


class TestCheckForUpdates:
    """Test check_for_updates() behavior in various scenarios."""

    def test_skip_check_when_env_var_set(self):
        """Test that update check is skipped when NEO_SKIP_UPDATE_CHECK is set."""
        with patch.dict("os.environ", {"NEO_SKIP_UPDATE_CHECK": "1"}):
            result = check_for_updates()
            assert result is None

    def test_returns_none_when_current_version_unknown(self):
        """Test that None is returned when current version cannot be determined."""
        import os
        old_val = os.environ.pop("NEO_SKIP_UPDATE_CHECK", None)
        try:
            with patch.object(update_checker, "_get_current_version", return_value="unknown"):
                result = check_for_updates()
                assert result is None
        finally:
            if old_val is not None:
                os.environ["NEO_SKIP_UPDATE_CHECK"] = old_val

    def test_cache_hit_returns_cached_new_version(self):
        """Test that cached new_version is returned on cache hit."""
        import os
        old_val = os.environ.pop("NEO_SKIP_UPDATE_CHECK", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_file = Path(tmpdir) / "update_check.json"
                cache_data = {
                    "last_check": time.time(),
                    "current_version": "0.9.0",
                    "latest_version": "1.0.0",
                    "new_version": "1.0.0",
                }
                cache_file.write_text(json.dumps(cache_data))

                with patch.object(
                    update_checker, "_get_cache_file", return_value=cache_file
                ):
                    with patch.object(
                        update_checker, "_get_current_version", return_value="0.9.0"
                    ):
                        result = check_for_updates(suppress_output=True)
                        assert result == "1.0.0"
        finally:
            if old_val is not None:
                os.environ["NEO_SKIP_UPDATE_CHECK"] = old_val

    def test_cache_miss_fetches_from_pypi(self):
        """Test that PyPI is queried when cache is expired."""
        import os
        old_val = os.environ.pop("NEO_SKIP_UPDATE_CHECK", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_file = Path(tmpdir) / "update_check.json"

                with patch.object(
                    update_checker, "_get_cache_file", return_value=cache_file
                ):
                    with patch.object(
                        update_checker, "_get_current_version", return_value="0.9.0"
                    ):
                        with patch.object(
                            update_checker,
                            "_fetch_latest_version_from_pypi",
                            return_value="1.0.0",
                        ):
                            result = check_for_updates(suppress_output=True)
                            assert result == "1.0.0"
                            assert cache_file.exists()
        finally:
            if old_val is not None:
                os.environ["NEO_SKIP_UPDATE_CHECK"] = old_val

    def test_no_update_available(self):
        """Test behavior when no update is available (versions equal)."""
        import os
        old_val = os.environ.pop("NEO_SKIP_UPDATE_CHECK", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_file = Path(tmpdir) / "update_check.json"

                with patch.object(
                    update_checker, "_get_cache_file", return_value=cache_file
                ):
                    with patch.object(
                        update_checker, "_get_current_version", return_value="1.0.0"
                    ):
                        with patch.object(
                            update_checker,
                            "_fetch_latest_version_from_pypi",
                            return_value="1.0.0",
                        ):
                            result = check_for_updates(suppress_output=True)
                            assert result is None
        finally:
            if old_val is not None:
                os.environ["NEO_SKIP_UPDATE_CHECK"] = old_val

    def test_pypi_fetch_failure_returns_none(self):
        """Test that None is returned when PyPI fetch fails."""
        import os
        old_val = os.environ.pop("NEO_SKIP_UPDATE_CHECK", None)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                cache_file = Path(tmpdir) / "update_check.json"

                with patch.object(
                    update_checker, "_get_cache_file", return_value=cache_file
                ):
                    with patch.object(
                        update_checker, "_get_current_version", return_value="0.9.0"
                    ):
                        with patch.object(
                            update_checker,
                            "_fetch_latest_version_from_pypi",
                            return_value=None,
                        ):
                            result = check_for_updates(suppress_output=True)
                            assert result is None
        finally:
            if old_val is not None:
                os.environ["NEO_SKIP_UPDATE_CHECK"] = old_val


class TestTimeoutHandling:
    """Test timeout handling in update operations."""

    def test_auto_install_handles_timeout(self):
        """Test that perform_auto_install handles subprocess timeout."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "update_check.json"

            with patch.object(
                update_checker, "_get_current_version", return_value="0.9.0"
            ):
                with patch.object(
                    update_checker, "_get_cache_file", return_value=cache_file
                ):
                    with patch(
                        "subprocess.run",
                        side_effect=subprocess.TimeoutExpired("pip", 120),
                    ):
                        result = perform_auto_install("1.0.0")
                        assert result is False

    def test_pypi_fetch_respects_timeout(self):
        """Test that PyPI fetch handles timeout errors."""
        from urllib.error import URLError

        with patch.object(update_checker, "urlopen") as mock_urlopen:
            mock_urlopen.side_effect = URLError("timeout")

            from neo.update_checker import _fetch_latest_version_from_pypi
            result = _fetch_latest_version_from_pypi()
            assert result is None


class TestCacheOperations:
    """Test cache read/write operations."""

    def test_write_cache_creates_cache_file(self):
        """Test that _write_cache creates cache file correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "update_check.json"

            with patch.object(
                update_checker, "_get_cache_file", return_value=cache_file
            ):
                _write_cache("0.9.0", "1.0.0")

                assert cache_file.exists()
                data = json.loads(cache_file.read_text())
                assert data["current_version"] == "0.9.0"
                assert data["latest_version"] == "1.0.0"
                assert data["new_version"] == "1.0.0"
                assert "last_check" in data

    def test_write_cache_no_new_version_when_same(self):
        """Test that new_version is None when versions are the same."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "update_check.json"

            with patch.object(
                update_checker, "_get_cache_file", return_value=cache_file
            ):
                _write_cache("1.0.0", "1.0.0")

                data = json.loads(cache_file.read_text())
                assert data["new_version"] is None

    def test_read_cache_returns_none_for_nonexistent(self):
        """Test that _read_cache returns None for non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "nonexistent.json"

            with patch.object(
                update_checker, "_get_cache_file", return_value=cache_file
            ):
                result = _read_cache()
                assert result is None

    def test_read_cache_handles_corrupted_json(self):
        """Test that _read_cache handles corrupted JSON gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "update_check.json"
            cache_file.write_text("not valid json")

            with patch.object(
                update_checker, "_get_cache_file", return_value=cache_file
            ):
                result = _read_cache()
                assert result is None


class TestPerformUpdate:
    """Test perform_update() function."""

    def test_perform_update_when_already_up_to_date(self):
        """Test perform_update when no update is available."""
        with patch.object(
            update_checker, "_get_current_version", return_value="1.0.0"
        ):
            with patch.object(
                update_checker, "check_for_updates", return_value=None
            ):
                result = perform_update()
                assert result is True

    def test_perform_update_success(self):
        """Test successful update via perform_update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "update_check.json"
            cache_file.write_text("{}")

            with patch.object(
                update_checker, "_get_cache_file", return_value=cache_file
            ):
                with patch.object(
                    update_checker, "_get_current_version", return_value="0.9.0"
                ):
                    with patch.object(
                        update_checker, "check_for_updates", return_value="1.0.0"
                    ):
                        with patch("subprocess.run") as mock_run:
                            mock_run.return_value = MagicMock(returncode=0)
                            result = perform_update()
                            assert result is True
                            assert mock_run.called
                            assert not cache_file.exists()

    def test_perform_update_pip_failure(self):
        """Test perform_update when pip fails."""
        import subprocess

        with patch.object(
            update_checker, "_get_current_version", return_value="0.9.0"
        ):
            with patch.object(
                update_checker, "check_for_updates", return_value="1.0.0"
            ):
                with patch(
                    "subprocess.run",
                    side_effect=subprocess.CalledProcessError(
                        1, "pip", stderr="pip error"
                    ),
                ):
                    result = perform_update()
                    assert result is False

    def test_perform_update_timeout(self):
        """Test perform_update handles timeout."""
        import subprocess

        with patch.object(
            update_checker, "_get_current_version", return_value="0.9.0"
        ):
            with patch.object(
                update_checker, "check_for_updates", return_value="1.0.0"
            ):
                with patch(
                    "subprocess.run",
                    side_effect=subprocess.TimeoutExpired("pip", 120),
                ):
                    result = perform_update()
                    assert result is False


class TestGetCurrentVersion:
    """Test _get_current_version() function."""

    def test_returns_version_when_installed(self):
        """Test that version is returned when package is installed."""
        with patch("importlib.metadata.version", return_value="1.2.3"):
            result = _get_current_version()
            assert result == "1.2.3"

    def test_returns_unknown_when_not_installed(self):
        """Test that 'unknown' is returned when package is not installed."""
        import importlib.metadata

        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("neo-reasoner"),
        ):
            result = _get_current_version()
            assert result == "unknown"


class TestGetCacheFile:
    """Test _get_cache_file() function."""

    def test_returns_correct_path(self):
        """Test that correct cache file path is returned."""
        cache_file = _get_cache_file()
        assert cache_file.name == "update_check.json"
        assert cache_file.parent.name == ".neo"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
