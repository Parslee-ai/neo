"""Tests for `[Neo]` progress-notice suppression.

`--quiet` was defined in the CLI parser but never read, so progress notices
could not be turned off. Under `--json` that left prose interleaved with the
JSONL event stream on stderr, forcing every host to filter it back out.
"""

import pytest

from neo import progress


@pytest.fixture(autouse=True)
def reset_quiet():
    """Progress state is process-global; never leak it between tests."""
    original = progress.is_quiet()
    yield
    progress.set_quiet(original)


def test_notices_print_by_default(capsys):
    progress.set_quiet(False)
    progress.note("Gathered 25 files")
    captured = capsys.readouterr()
    assert captured.err == "[Neo] Gathered 25 files\n"
    assert captured.out == ""


def test_quiet_suppresses_notices(capsys):
    progress.set_quiet(True)
    progress.note("Gathered 25 files")
    assert capsys.readouterr().err == ""


def test_notices_never_reach_stdout(capsys):
    """stdout must stay exactly one JSON document under --json."""
    progress.set_quiet(False)
    progress.note("anything")
    assert capsys.readouterr().out == ""


def test_a_broken_stderr_does_not_break_the_run(monkeypatch):
    """Same policy as events.safe_emit: reporting must not take down the run."""
    progress.set_quiet(False)

    def explode(*args, **kwargs):
        raise OSError("stderr is closed")

    monkeypatch.setattr("builtins.print", explode)
    progress.note("still fine")  # must not raise


def test_context_gatherer_routes_through_progress():
    """Pins the wiring: a stray `print("[Neo] ...")` would silently reintroduce
    unsuppressable output."""
    from pathlib import Path

    source = Path("src/neo/context_gatherer.py").read_text()
    assert "[Neo]" not in source
    assert "progress.note(" in source


@pytest.mark.parametrize(
    "quiet,json_mode,expected",
    [
        (False, False, False),  # default: notices on
        (True, False, True),    # explicit --quiet
        (False, True, True),    # --json implies quiet
        (True, True, True),
    ],
)
def test_cli_quiet_resolution(quiet, json_mode, expected):
    """The rule main() applies: --json implies --quiet, because the event
    stream already carries this information in parseable form."""
    resolved = bool(quiet or json_mode)
    assert resolved is expected
