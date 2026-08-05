"""Human-readable progress notices on stderr.

These are the `[Neo] …` lines a person watching a terminal reads while context
is gathered. They are distinct from two neighbouring channels and should not be
confused with either:

- `neo.events` — machine-readable JSONL for a host, emitted on the same stream.
- `logging` (`--verbose` / `--debug`) — diagnostics, off by default.

Progress notices are on by default and suppressed by `--quiet`, and `--json`
implies quiet: under `--json` the event stream carries the same information in
a parseable form, and mixing prose into it forces every host to filter.

The suppression flag is process-global rather than threaded through call
arguments. It is a display setting fixed once from argv and read by leaf
functions five layers below the CLI (`gather_context` → `_project_index_boost`
→ …); passing it down would put a presentation parameter into the signature of
every scoring helper it crosses.
"""

import sys

_quiet = False


def set_quiet(quiet: bool) -> None:
    """Enable or disable progress notices for the rest of the process."""
    global _quiet
    _quiet = quiet


def is_quiet() -> bool:
    return _quiet


def note(message: str) -> None:
    """Print one `[Neo] …` progress line to stderr, unless suppressed.

    Best-effort: a closed or unwritable stderr must not take down a run that
    was otherwise fine, for the same reason `events.safe_emit` swallows sink
    failures.
    """
    if _quiet:
        return
    try:
        print(f"[Neo] {message}", file=sys.stderr)
    except Exception:
        pass
