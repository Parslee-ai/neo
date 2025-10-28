#!/usr/bin/env python3
"""
Integration test for CLI global flags.

Tests that global flags (--version, --config, etc.) work correctly
without requiring subcommand-specific attributes.

This test validates the fix for issue #34 where global flags were
checked after the construct subcommand check, causing AttributeError
when running commands like `neo --version`.
"""

import subprocess
import sys
import pytest


class TestCLIGlobalFlags:
    """Test CLI global flags work correctly at entry point."""

    def test_version_flag_works(self):
        """
        Test that `neo --version` returns exit code 0 and shows version info.

        This test validates the fix for issue #34. Before the fix, running
        `neo --version` would crash with AttributeError because the construct
        parser doesn't have a 'version' attribute.

        Expected behavior:
        - Command exits with code 0
        - Output contains version information
        - No AttributeError occurs
        """
        result = subprocess.run(
            [sys.executable, "-m", "neo", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should exit successfully
        assert result.returncode == 0, f"Expected exit code 0, got {result.returncode}. stderr: {result.stderr}"

        # Should output version info (check for common version indicators)
        output = result.stdout.lower() + result.stderr.lower()
        assert any(indicator in output for indicator in ["version", "v0.", "neo"]), \
            f"Expected version info in output, got: stdout={result.stdout}, stderr={result.stderr}"

        # Should not contain error messages
        assert "error" not in output, f"Unexpected error in output: {result.stdout + result.stderr}"
        assert "traceback" not in output, f"Unexpected traceback in output: {result.stdout + result.stderr}"

    def test_construct_list_still_works(self):
        """
        Test that construct subcommand still works after global flag reordering.

        This test ensures we didn't break the construct subcommand functionality
        when moving global flag checks before construct check.

        Expected behavior:
        - Command exits with code 0
        - No errors occur
        """
        result = subprocess.run(
            [sys.executable, "-m", "neo", "construct", "list"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should exit successfully
        assert result.returncode == 0, f"Expected exit code 0, got {result.returncode}. stderr: {result.stderr}"

        # Should not contain error messages
        output = result.stdout.lower() + result.stderr.lower()
        assert "attributeerror" not in output, f"Unexpected AttributeError in output: {result.stdout + result.stderr}"
        assert "traceback" not in output, f"Unexpected traceback in output: {result.stdout + result.stderr}"

    def test_config_flag_works(self):
        """
        Test that `neo --config list` works correctly.

        Expected behavior:
        - Command exits with code 0
        - Shows configuration information
        - No AttributeError occurs
        """
        result = subprocess.run(
            [sys.executable, "-m", "neo", "--config", "list"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Should exit successfully (even if no config exists, it should handle gracefully)
        # Exit code might be 0 or 1 depending on config state, but should not crash
        assert result.returncode in [0, 1], f"Expected exit code 0 or 1, got {result.returncode}"

        # Should not contain AttributeError
        output = result.stdout.lower() + result.stderr.lower()
        assert "attributeerror" not in output, f"Unexpected AttributeError in output: {result.stdout + result.stderr}"

    def test_global_flags_work_with_construct_subcommand(self):
        """
        Verify global flags work when used with construct subcommand.

        This is the originally broken scenario from issue #34 - running
        `neo construct --version` would fail because global flags were
        duplicated instead of using the parent parser pattern.

        Expected behavior:
        - Command exits with code 0
        - Shows version information
        - No errors occur
        """
        result = subprocess.run(
            [sys.executable, "-m", "neo", "construct", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )

        # Should exit successfully
        assert result.returncode == 0, f"Expected exit code 0, got {result.returncode}. stderr: {result.stderr}"

        # Should output version info (neo X.Y.Z format or contains "version" keyword)
        output = result.stdout + result.stderr
        output_lower = output.lower()
        assert "neo 0." in output_lower or "version" in output_lower, \
            f"Expected version info in output, got: stdout={result.stdout}, stderr={result.stderr}"

        # Should not contain error messages
        assert "attributeerror" not in output_lower, f"Unexpected AttributeError in output: {output}"
        assert "traceback" not in output_lower, f"Unexpected traceback in output: {output}"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
