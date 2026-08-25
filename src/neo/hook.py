"""Record host tool-call events, so acceptance can be observed instead of inferred.

`neo hook record` reads a Claude Code hook payload on stdin and appends one line
to the host-event ledger. It exists because Neo's acceptance detection is
inferred from git on the *next* invocation, which cannot see an edit that was
never followed by another Neo run, and which records the revision the episode
BEGAN at rather than the one the fix landed in. Design note and the measured
case for it: `docs/solutions/host-hooks-for-outcome-detection.md`.

Three rules govern everything in this module.

**It never fails.** `run_hook` returns 0 on every path, by construction — not by
catching the errors we thought of. That means `BaseException`, and it means the
failure-reporting path is guarded too. Two things remain outside its reach and
are stated rather than papered over: a signal that terminates the process
(SIGKILL), and a library calling `os._exit`. Neither is reachable from this
module's own code. A `PostToolUse` hook that exits non-zero
reports an error against a tool call that already succeeded. CAR's equivalent
hook documents exactly this intent and still has a hole: an unrecognized
subcommand dies in argument parsing before any fail-open code runs, which wedged
every `Bash`/`Write`/`Edit` call in the session this module was written in. So
an unknown action here returns 0 rather than raising, and the outer `try` covers
the whole body.

**It is cheap.** It fires on every edit. Measured: `import neo.cli` is 0.04 s,
`neo --version` is 0.36 s, and the difference is `FactStore` construction.
`cli.main` dispatches here before argument parsing, the update check, the
observer autostart and any store setup, and `test_hook_stays_off_the_slow_path`
pins that ordering.

**It records paths, never contents.** `tool_input` carries the text being
written. Only the identifying fields are read — the same rule CAR's hook applies
for the weaker case of a decision that is never written to disk.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Import-time home capture: `tests/conftest.py::HOME_PATH_CONSTANTS` must carry
# this or `test_home_isolation.py` fails, which is the intended outcome.
HOOK_LEDGER = Path.home() / ".neo" / "sessions" / "host_events.jsonl"

# One generation, rotated at 8 MB — smaller than `metrics.jsonl`'s 32 MB because
# nothing summarises this file yet, so an operator may have to read it by hand.
MAX_LEDGER_BYTES = 8 * 1024 * 1024

# Tools whose payload names a file a suggestion could have been applied to.
# `Bash` is deliberately absent: a shell command that happens to write a file
# names no path we could attribute, and admitting it would record every command
# the host runs.
RECORDED_TOOLS = frozenset({"Edit", "Write", "MultiEdit", "NotebookEdit"})

_GIT_TIMEOUT_SECONDS = 2.0


def _debug(message: str) -> None:
    """Failures are silent by default and findable on request.

    A hook writing to stderr on every edit would be noise in the host's UI, but
    a hook that fails silently forever is undiagnosable.

    Swallows everything, including `BaseException`. This runs *inside* the
    handler that exists to keep the hook from failing, so a closed or full
    stderr here would defeat the whole guarantee by way of the code reporting
    it. Neo found this one in review; no test covered it.
    """
    if os.environ.get("NEO_HOOK_DEBUG") != "1":
        return
    try:
        print(f"[neo hook] {message}", file=sys.stderr)
    except BaseException:
        pass


def _debug_failure(exc: BaseException) -> None:
    """Report a swallowed failure, totally.

    Formatting an exception can itself raise — a custom `__str__` is under no
    obligation to succeed — so the interpolation is guarded too, not just the
    write.
    """
    try:
        _debug(f"{type(exc).__name__}: {exc}")
    except BaseException:
        pass


def _head(cwd: str) -> str:
    """HEAD at the moment of the edit — the field this module exists to capture.

    Returns "" when there is no git, no repo, or no commit yet. Blank is the
    honest answer and the promotion path already fails closed on it; inventing a
    revision would manufacture the independence that `repository_revision` is
    supposed to establish.
    """
    try:
        result = subprocess.run(
            ["git", "-C", cwd or ".", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # git missing, timeout, permission
        _debug(f"rev-parse failed: {exc}")
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def build_record(payload: dict, *, now: Optional[float] = None) -> Optional[dict]:
    """Reduce a hook payload to the fields the outcome path needs.

    Returns None when the payload names no tool we record or no file path. That
    is not an error — most tool calls have nothing to say here.
    """
    tool = payload.get("tool_name") or ""
    if tool not in RECORDED_TOOLS:
        return None

    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    file_path = tool_input.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return None

    # The payload names the directory the HOST is working in. Prefer it over
    # this process's own: a hook runs wherever the host launched it, which is
    # not necessarily the project the edit belongs to.
    cwd = payload.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        cwd = os.getcwd()

    record = {
        "ts": time.time() if now is None else now,
        "tool": tool,
        "file_path": file_path,
        "cwd": cwd,
        "head": _head(cwd),
    }

    # Recorded when offered, depended on by nothing. The field is documented but
    # was not confirmed against a real payload when this was written.
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id:
        record["session_id"] = session_id
    return record


def _rotate_if_large(ledger: Path) -> None:
    try:
        if ledger.stat().st_size < MAX_LEDGER_BYTES:
            return
    except OSError:
        return  # no file yet, or unreadable — either way, nothing to rotate
    try:
        os.replace(ledger, ledger.with_name(ledger.name + ".1"))
    except OSError as exc:
        _debug(f"rotation failed: {exc}")


def append(record: dict, ledger: Optional[Path] = None) -> None:
    """Append one JSON line. Callers treat failure as nothing-happened."""
    target = HOOK_LEDGER if ledger is None else ledger
    target.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_large(target)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def run_hook(argv: list) -> int:
    """Entry point for `neo hook <action>`. Returns 0 on every path."""
    try:
        if os.environ.get("NEO_HOOKS") == "0":
            return 0
        action = argv[0] if argv else ""
        if action != "record":
            # Not worth failing a tool call over, and a future host may send an
            # action this build predates.
            _debug(f"unknown action {action!r}")
            return 0
        payload = json.loads(sys.stdin.read())
        if not isinstance(payload, dict):
            _debug("payload was not a JSON object")
            return 0
        record = build_record(payload)
        if record is None:
            return 0
        append(record)
    except BaseException as exc:
        # `BaseException`, not `Exception`. `KeyboardInterrupt`, `SystemExit`
        # and `GeneratorExit` bypass an `Exception` handler, and the docstring
        # above claims this returns 0 on every path BY CONSTRUCTION — a
        # stronger claim than `except Exception` delivers. Neo caught the gap;
        # every test in `TestNeverFails` raised an `Exception` subclass, so
        # none of them could have.
        #
        # Swallowing `KeyboardInterrupt` is normally wrong. It is right here:
        # this process lives ~60ms, does one append, and is spawned by the host
        # rather than a terminal. The tool call it reports on has ALREADY
        # succeeded, so exiting non-zero would annotate a successful edit with a
        # failure — worse than losing one telemetry line.
        _debug_failure(exc)
    return 0
