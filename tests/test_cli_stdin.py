"""Tests for CLI stdin handling — must never block forever on a non-tty
stdin that the peer holds open without data or EOF (the multi-hour-hang bug)."""

import os
import sys
import time
import types

import neo.cli as cli


def _args(**kw):
    """Minimal args namespace for detect_input_mode / read_prompt_from_argv_or_stdin."""
    base = {"stdin_json": False, "stdin_text": False, "prompt": None}
    base.update(kw)
    return types.SimpleNamespace(**base)


class _BlockingStdin:
    """A real, open pipe with no data and the write end held open: read()
    would block forever, select() reports it as not-ready. Models the
    background-job / daemon stdin that caused the hang."""

    def __init__(self):
        self._r, self._w = os.pipe()           # write end stays open -> never EOF
        self._f = os.fdopen(self._r, "r")

    def fileno(self):
        return self._f.fileno()

    def read(self, *a):                         # would block; the guard must avoid calling it
        return self._f.read(*a)

    def isatty(self):
        return False

    def close(self):
        self._f.close()
        os.close(self._w)


def test_guarded_read_does_not_hang_on_open_stdin(monkeypatch):
    monkeypatch.setenv("NEO_STDIN_TIMEOUT_SECONDS", "0.3")
    blocking = _BlockingStdin()
    monkeypatch.setattr(sys, "stdin", blocking)
    try:
        start = time.monotonic()
        out = cli._read_stdin_guarded()
        elapsed = time.monotonic() - start
        assert out == ""                        # treated as empty, not hung
        assert elapsed < 5.0                     # returned promptly via the deadline
    finally:
        blocking.close()


def test_guarded_read_returns_piped_data(monkeypatch):
    r, w = os.pipe()
    os.write(w, b"hello from a real pipe")
    os.close(w)                                  # EOF available immediately
    monkeypatch.setattr(sys, "stdin", os.fdopen(r, "r"))
    assert cli._read_stdin_guarded() == "hello from a real pipe"


def test_argv_prompt_never_reads_stdin(monkeypatch):
    # `neo "query"` with a blocking non-tty stdin must NOT touch stdin.
    blocking = _BlockingStdin()
    monkeypatch.setattr(sys, "stdin", blocking)
    try:
        start = time.monotonic()
        assert cli.detect_input_mode(_args(prompt="a real prompt")) == "text"
        assert cli.read_prompt_from_argv_or_stdin(_args(prompt="a real prompt")) == "a real prompt"
        assert time.monotonic() - start < 2.0    # no blocking read happened
    finally:
        blocking.close()


def test_guarded_read_tolerates_stringio(monkeypatch):
    # select() can't poll a StringIO (no fileno) — must fall back, not raise.
    import io
    monkeypatch.setattr(sys, "stdin", io.StringIO("inline text"))
    assert cli._read_stdin_guarded() == "inline text"
