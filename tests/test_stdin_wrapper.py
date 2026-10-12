"""The generated stdin/stdout wrapper calls the function by name, not by evaluating code."""

import subprocess
import sys

import neo.stdin_wrapper as stdin_wrapper


def test_fallback_wrapper_calls_the_function_by_name(tmp_path, monkeypatch):
    # Force the fallback: pattern inference found nothing.
    monkeypatch.setattr(stdin_wrapper, "generate_stdin_wrapper", lambda *args: None)
    code = "def double(x):\n    return int(x) * 2\n"

    wrapper = stdin_wrapper.wrap_function_for_stdin(code, "", "")

    assert "globals()[func_name](line)" in wrapper
    script = tmp_path / "wrapped.py"
    script.write_text(wrapper, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        input="3\n5\n",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "6\n10\n"
