"""ClaudeCodeAdapter runs `claude -p` as plain, isolated inference (#260)."""

import json
import subprocess

import pytest

from neo.adapters import ClaudeCodeAdapter


class FakeProcess:
    def __init__(self, stdout_lines, returncode=0, stderr="", hang=False):
        self._stdout = "\n".join(json.dumps(x) for x in stdout_lines)
        self.returncode = returncode
        self._stderr = stderr
        self._hang = hang
        self.input = None
        self.killed = False

    def communicate(self, input=None, timeout=None):
        if self._hang and not self.killed:
            raise subprocess.TimeoutExpired("claude", timeout)
        self.input = input
        return self._stdout, self._stderr

    def kill(self):
        self.killed = True


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "2.1.296", ""),
    )
    return ClaudeCodeAdapter(model="haiku", timeout=5)


def run_with(monkeypatch, adapter, proc, **kwargs):
    seen = {}

    def fake_popen(args, **kw):
        seen["args"] = args
        seen["env"] = kw.get("env")
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    out = adapter.generate([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], **kwargs)
    return out, seen


def assistant(text, stop=None):  # the real CLI sends stop_reason null on normal chunks
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}], "stop_reason": stop}}


def test_runs_isolated_from_the_users_setup(monkeypatch, adapter):
    proc = FakeProcess([assistant("ok"), {"type": "result", "subtype": "success", "is_error": False}])
    out, seen = run_with(monkeypatch, adapter, proc)
    assert out == "ok"
    args = seen["args"]
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in args
    assert "--no-session-persistence" in args
    assert "--bare" not in args  # it would skip the subscription keychain read
    assert "ANTHROPIC_API_KEY" not in seen["env"]


def test_a_result_error_on_stdout_is_reported(monkeypatch, adapter):
    proc = FakeProcess(
        [{"type": "result", "subtype": "error_max_turns", "is_error": True,
          "errors": ["Reached maximum number of turns (1)"]}],
        returncode=1, stderr="",
    )
    with pytest.raises(RuntimeError) as err:
        run_with(monkeypatch, adapter, proc)
    assert "error_max_turns" in str(err.value)
    assert "Reached maximum number of turns" in str(err.value)


def test_is_error_fails_even_with_exit_zero(monkeypatch, adapter):
    proc = FakeProcess([{"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "boom"}])
    with pytest.raises(RuntimeError, match="error_during_execution"):
        run_with(monkeypatch, adapter, proc)


def test_the_timeout_covers_the_whole_call_and_kills_the_child(monkeypatch, adapter):
    proc = FakeProcess([], hang=True)
    with pytest.raises(RuntimeError, match="did not finish within 5s"):
        run_with(monkeypatch, adapter, proc)
    assert proc.killed


def test_stop_sequences_cut_the_response(monkeypatch, adapter):
    proc = FakeProcess([assistant("plan here</neo> trailing junk"), {"type": "result", "subtype": "success", "is_error": False}])
    out, _ = run_with(monkeypatch, adapter, proc, stop=["</neo>", "```"])
    assert out == "plan here"


def test_messages_go_to_stdin_and_the_system_prompt_to_a_file(monkeypatch, adapter):
    proc = FakeProcess([assistant("x"), {"type": "result", "subtype": "success", "is_error": False}])
    _, seen = run_with(monkeypatch, adapter, proc)
    assert json.loads(proc.input) == [{"role": "user", "content": "u"}]
    assert "--system-prompt-file" in seen["args"]


def test_an_unknown_model_fails_with_the_cli_s_own_words(monkeypatch, adapter):
    """Shape recorded from claude 2.1.296 with --model no-such-model-xyz:
    exit 1, subtype "success" yet is_error true, text not starting "API Error"."""
    proc = FakeProcess(
        [assistant("There's an issue with the selected model (no-such-model-xyz).", "stop_sequence"),
         {"type": "result", "subtype": "success", "is_error": True,
          "result": "There's an issue with the selected model (no-such-model-xyz)."}],
        returncode=1,
        stderr="[claude-code:unrecognized_model] no-such-model-xyz",
    )
    with pytest.raises(RuntimeError) as err:
        run_with(monkeypatch, adapter, proc)
    message = str(err.value)
    assert message.startswith("Claude Code failed (exit 1): ")
    assert "issue with the selected model" in message
    assert "unrecognized_model" in message
    assert "success" not in message


def test_output_without_a_result_line_is_an_error(monkeypatch, adapter):
    proc = FakeProcess([assistant("partial ans")])  # cut short, exit 0
    with pytest.raises(RuntimeError, match="no result line"):
        run_with(monkeypatch, adapter, proc)


def test_an_old_cli_says_to_upgrade(monkeypatch, adapter):
    proc = FakeProcess([], returncode=1, stderr="error: unknown option '--setting-sources'")
    with pytest.raises(RuntimeError, match="upgrade it"):
        run_with(monkeypatch, adapter, proc)
