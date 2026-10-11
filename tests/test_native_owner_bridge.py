"""Run the native-owner controls in an isolated import context."""
from pathlib import Path
import subprocess
import sys


def test_native_owner_controls():
    # The control runner substitutes a minimal Neo package and mocked guardian
    # transport. A separate interpreter keeps those substitutions out of pytest.
    runner = Path(__file__).resolve().parents[1] / "tools/test_native_owner_bridge.py"
    result = subprocess.run(
        [sys.executable, "-B", str(runner)], capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
