"""Tests for the host-hook recorder (`neo hook record`).

Two of these pin properties the module *claims* rather than behaviour it
returns, and they are the reason the file exists:

- `test_hook_stays_off_the_slow_path` — the recorder fires on every edit, and
  the thing that would ruin it is a future edit to `cli.main` that moves store
  construction above the dispatch. That is invisible to every behavioural test.
- `TestNeverFails` — a `PostToolUse` hook that exits non-zero reports an error
  against a tool call that already succeeded. CAR's hook documents fail-open and
  still wedged this repository's session, because the hole was in argument
  parsing rather than in the code that had the guard.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from neo import hook


def _payload(**overrides):
    payload = {
        "session_id": "abc123",
        "tool_name": "Edit",
        "tool_input": {"file_path": "src/neo/engine.py", "old_string": "a", "new_string": "b"},
        "cwd": "/tmp/project",
    }
    payload.update(overrides)
    return payload


# ------------------------------------------------------------ what is recorded


class TestBuildRecord:
    def test_records_the_identifying_fields(self, monkeypatch):
        monkeypatch.setattr(hook, "_head", lambda cwd: "deadbeef")
        record = hook.build_record(_payload(), now=1000.0)
        assert record == {
            "ts": 1000.0,
            "tool": "Edit",
            "file_path": "src/neo/engine.py",
            "cwd": "/tmp/project",
            "head": "deadbeef",
            "session_id": "abc123",
        }

    def test_file_contents_never_reach_the_ledger(self, monkeypatch):
        """`tool_input` carries the text being written. Recording a path is
        telemetry; recording the new file body is exfiltration."""
        monkeypatch.setattr(hook, "_head", lambda cwd: "")
        payload = _payload()
        payload["tool_input"]["new_string"] = "SECRET_TOKEN=hunter2"
        record = hook.build_record(payload, now=0.0)
        assert "hunter2" not in json.dumps(record)
        assert set(record) <= {"ts", "tool", "file_path", "cwd", "head", "session_id"}

    def test_the_payload_cwd_wins_over_the_process_cwd(self, monkeypatch):
        """A hook runs wherever the host launched it, which is not necessarily
        the project the edit belongs to."""
        monkeypatch.setattr(hook, "_head", lambda cwd: cwd)
        record = hook.build_record(_payload(cwd="/elsewhere"), now=0.0)
        assert record["cwd"] == "/elsewhere"
        assert record["head"] == "/elsewhere"

    def test_falls_back_to_process_cwd_when_the_payload_omits_it(self, monkeypatch):
        monkeypatch.setattr(hook, "_head", lambda cwd: "")
        monkeypatch.setattr(hook.os, "getcwd", lambda: "/fallback")
        record = hook.build_record(_payload(cwd=""), now=0.0)
        assert record["cwd"] == "/fallback"

    @pytest.mark.parametrize("tool", ["Bash", "Read", "Grep", "WebFetch", ""])
    def test_untracked_tools_record_nothing(self, tool):
        """`Bash` especially: a shell command that writes a file names no path
        we could attribute, and admitting it records every command the host
        runs."""
        assert hook.build_record(_payload(tool_name=tool)) is None

    @pytest.mark.parametrize("tool_input", [None, {}, {"file_path": ""}, {"file_path": 5}, "str"])
    def test_a_payload_naming_no_file_records_nothing(self, tool_input):
        assert hook.build_record(_payload(tool_input=tool_input)) is None

    def test_session_id_is_omitted_rather_than_invented(self):
        """The field is documented but was not confirmed against a real payload.
        Nothing depends on it, so its absence must not fabricate a value."""
        record = hook.build_record(_payload(session_id=None), now=0.0)
        assert "session_id" not in record


class TestHead:
    def test_blank_outside_a_repository(self, tmp_path):
        assert hook._head(str(tmp_path)) == ""

    def test_reads_head_inside_one(self, tmp_path):
        env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        def run(*argv):
            return subprocess.run(argv, cwd=tmp_path, capture_output=True,
                                  env={**dict(hook.os.environ), **env})

        run("git", "init", "-q")
        (tmp_path / "f.txt").write_text("x")
        run("git", "add", "f.txt")
        run("git", "commit", "-qm", "c")
        assert len(hook._head(str(tmp_path))) == 40

    def test_a_blank_revision_is_never_fabricated(self, tmp_path, monkeypatch):
        """Promotion requires two acceptances spanning two distinct revisions.
        Inventing a revision here would manufacture the independence that gate
        exists to establish."""
        monkeypatch.setattr(hook.subprocess, "run", lambda *a, **k: 1 / 0)
        assert hook._head(str(tmp_path)) == ""


# ------------------------------------------------------------ the ledger


class TestLedger:
    def test_appends_one_line_per_call(self, tmp_path):
        ledger = tmp_path / "nested" / "host_events.jsonl"
        hook.append({"a": 1}, ledger)
        hook.append({"a": 2}, ledger)
        lines = ledger.read_text().splitlines()
        assert [json.loads(line)["a"] for line in lines] == [1, 2]

    def test_rotates_one_generation_when_large(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hook, "MAX_LEDGER_BYTES", 10)
        ledger = tmp_path / "host_events.jsonl"
        hook.append({"first": "x" * 50}, ledger)
        hook.append({"second": 1}, ledger)
        assert "second" in ledger.read_text()
        assert "first" in (tmp_path / "host_events.jsonl.1").read_text()


# ------------------------------------------------------------ it never fails


class TestNeverFails:
    """Every one of these would, uncaught, surface as an error against a tool
    call that already succeeded."""

    @pytest.mark.parametrize("stdin", ["", "not json", "[]", "null", '{"tool_name": 5}'])
    def test_malformed_stdin_still_exits_zero(self, stdin, monkeypatch, tmp_path):
        monkeypatch.setattr(hook, "HOOK_LEDGER", tmp_path / "l.jsonl")
        monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(stdin))
        assert hook.run_hook(["record"]) == 0

    @pytest.mark.parametrize("argv", [[], ["nonsense"], ["record", "extra"]])
    def test_an_unknown_action_exits_zero(self, argv, monkeypatch, tmp_path):
        """The exact hole in CAR's hook: it fails open on an unreadable payload,
        but an unrecognized subcommand died in argument parsing before any of
        that ran, and wedged every tool call in the session."""
        monkeypatch.setattr(hook, "HOOK_LEDGER", tmp_path / "l.jsonl")
        monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps(_payload())))
        assert hook.run_hook(argv) == 0

    def test_an_unwritable_ledger_still_exits_zero(self, monkeypatch):
        monkeypatch.setattr(hook, "HOOK_LEDGER", Path("/proc/nope/l.jsonl"))
        monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps(_payload())))
        assert hook.run_hook(["record"]) == 0

    def test_opting_out_records_nothing_and_exits_zero(self, monkeypatch, tmp_path):
        ledger = tmp_path / "l.jsonl"
        monkeypatch.setattr(hook, "HOOK_LEDGER", ledger)
        monkeypatch.setenv("NEO_HOOKS", "0")
        monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps(_payload())))
        assert hook.run_hook(["record"]) == 0
        assert not ledger.exists()

    def test_a_recorded_edit_reaches_the_ledger(self, monkeypatch, tmp_path):
        """The positive case, so the tests above cannot pass by recording
        nothing at all."""
        ledger = tmp_path / "l.jsonl"
        monkeypatch.setattr(hook, "HOOK_LEDGER", ledger)
        monkeypatch.setattr(hook, "_head", lambda cwd: "cafe")
        monkeypatch.setattr(sys, "stdin", __import__("io").StringIO(json.dumps(_payload())))
        assert hook.run_hook(["record"]) == 0
        assert json.loads(ledger.read_text())["file_path"] == "src/neo/engine.py"


# ------------------------------------------------------------ it stays cheap


def test_hook_stays_off_the_slow_path():
    """`cli.main` must dispatch to the hook before it builds anything.

    This is the invariant no behavioural test can see. The recorder fires on
    every edit; `neo --version` costs 0.36s against `import neo.cli`'s 0.04s,
    and the whole difference is FactStore construction. A future edit that moves
    store setup, the update check or the observer autostart above this dispatch
    would put that cost on every keystroke-level tool call, silently.
    """
    source = Path(hook.__file__).with_name("cli.py").read_text()
    body = source.split("def main():", 1)[1]
    dispatch = body.index('sys.argv[1] == "hook"')
    for expensive in ("FactStore(", "check_for_updates", "maybe_autostart_observer", "parse_args()"):
        assert expensive not in body[:dispatch], (
            f"{expensive} now runs before the hook dispatch in cli.main; the "
            "recorder fires on every edit and must stay ahead of it."
        )


def test_the_ledger_is_registered_for_home_isolation():
    """`HOOK_LEDGER` captures `Path.home()` at import. Without a conftest entry
    the test suite writes into the developer's real `~/.neo`."""
    from tests.conftest import HOME_PATH_CONSTANTS

    assert ("neo.hook", "HOOK_LEDGER", ".neo/sessions/host_events.jsonl") in HOME_PATH_CONSTANTS
