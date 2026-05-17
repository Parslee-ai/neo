"""
Memory-operation metrics recorder.

Append-only JSONL log of retrieval and write events, used to track Layer 2
(quality) and Layer 3 (efficiency) signals from the memory-system survey
(paper 2603.07670 §7). No analysis here — just structured emit; downstream
tools (or future Layer-2 evaluators) consume the log.

Disabled by setting NEO_METRICS=off (or 0/false/no). Writes are best-effort:
any I/O failure is logged at debug-level and swallowed so retrieval is never
blocked by metrics emission.

## lm_call event convention

Every adapter that emits ``lm_call`` MUST include a ``status`` field:

- ``status="success"`` on the success path (with token counts, latency, etc.)
- ``status="error"`` on the failure path (with ``error_type=<exception class
  name>`` and whatever request-side context the adapter has — model, input
  shape, max_tokens, intent_hint, etc.). Token counts are absent (we never
  made the call).

Querying for ``status="error"`` gives failure rates per provider/model.
Older rows predating this convention may have ``status`` absent — consumers
should treat that as ``success`` for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_DISABLED_VALUES = frozenset({"off", "0", "false", "no"})


def _enabled() -> bool:
    value = os.getenv("NEO_METRICS", "on").strip().lower()
    return value not in _DISABLED_VALUES


def _metrics_path() -> Path:
    """Resolve ~/.neo/metrics.jsonl at call time so per-test HOME stubs apply.

    Computing this at module-import time would pin the path before the test
    conftest's monkeypatched Path.home() fixture fires, polluting the real
    user's home with test events.
    """
    return Path.home() / ".neo" / "metrics.jsonl"


def record(event: str, **fields: Any) -> None:
    """Append a structured event to metrics.jsonl.

    Required field ``event`` names the metric (e.g. "retrieve", "add_fact").
    Every record also gets a millisecond-precision ``ts``. All other fields
    pass through unmodified — callers decide what's useful per event type.
    """
    if not _enabled():
        return
    try:
        path = _metrics_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = {"ts": time.time(), "event": event, **fields}
        with path.open("a") as f:
            f.write(json.dumps(line, default=str) + "\n")
    except Exception as e:  # never let a metrics write break retrieval
        logger.debug("metrics write failed: %s", e)


def time_block() -> "TimedBlock":
    """Context manager that returns elapsed ms via ``.elapsed_ms``."""
    return TimedBlock()


class TimedBlock:
    """Trivial elapsed-ms timer for use with a context manager."""

    __slots__ = ("_t0", "elapsed_ms")

    def __init__(self) -> None:
        self._t0 = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "TimedBlock":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000.0


@contextmanager
def capture_lm_call_failure(provider: str, model: str, **context: Any) -> Iterator[None]:
    """Wrap an adapter's outbound inference call to record errors as a
    ``status="error"`` ``lm_call`` row before the exception re-raises.

    Catches ``Exception`` (not ``BaseException``): KeyboardInterrupt and
    SystemExit are the user choosing to leave, not a CAR/OpenAI/Anthropic
    failure worth recording. ``context`` is passed through unmodified — call
    sites add provider-specific triage fields (input shape, intent_hint,
    max_tokens, etc.) so operators can answer "which call shape blew up?"
    without re-running.

    Metrics emit failures are swallowed; the original exception always
    propagates with its traceback intact.
    """
    try:
        yield
    except Exception as exc:
        try:
            record(
                "lm_call",
                provider=provider,
                model=model,
                status="error",
                error_type=type(exc).__name__,
                **context,
            )
        except Exception:
            pass  # Metrics are never load-bearing.
        raise
