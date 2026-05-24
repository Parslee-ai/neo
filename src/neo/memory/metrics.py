"""
Memory-operation metrics recorder.

Append-only JSONL log of retrieval and write events, used to track Layer 2
(quality) and Layer 3 (efficiency) signals from the memory-system survey
(paper 2603.07670 §7). No analysis here — just structured emit; downstream
tools (or future Layer-2 evaluators) consume the log.

Gated by ``NEO_PROFILE``:

  - ``off``      — emit nothing (kill switch)
  - ``minimal``  — emit only high-signal events (currently ``lm_call``);
                   right for production deployments that only care about
                   model-call observability.
  - ``standard`` — emit everything (default)
  - ``strict``   — emit everything plus reserved verbose events
                   (currently identical to ``standard``).

``NEO_METRICS=off`` (or ``0``/``false``/``no``) remains a hard kill-switch
that overrides ``NEO_PROFILE``, so existing kill-the-emit scripts keep
working. Writes are best-effort: any I/O failure is logged at debug-level
and swallowed so retrieval is never blocked by metrics emission.

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

PROFILE_OFF = "off"
PROFILE_MINIMAL = "minimal"
PROFILE_STANDARD = "standard"
PROFILE_STRICT = "strict"
VALID_PROFILES = frozenset({PROFILE_OFF, PROFILE_MINIMAL, PROFILE_STANDARD, PROFILE_STRICT})

# Events that emit under the "minimal" profile. Everything else requires
# "standard" or higher. Keep this short — the point of "minimal" is to give
# operators an LM-call audit trail without the per-retrieval volume.
_MINIMAL_EVENTS = frozenset({"lm_call"})


def _profile() -> str:
    """Resolve the active profile.

    ``NEO_METRICS=off`` is a hard kill-switch and overrides ``NEO_PROFILE``;
    otherwise ``NEO_PROFILE`` decides (default ``standard``). Unknown profile
    values fall back to ``standard`` after a debug-level log.
    """
    metrics_value = os.getenv("NEO_METRICS", "").strip().lower()
    if metrics_value in _DISABLED_VALUES:
        return PROFILE_OFF

    profile = os.getenv("NEO_PROFILE", PROFILE_STANDARD).strip().lower()
    if profile not in VALID_PROFILES:
        logger.debug("Unknown NEO_PROFILE=%r, falling back to %s", profile, PROFILE_STANDARD)
        return PROFILE_STANDARD
    return profile


def _should_emit(event: str) -> bool:
    """Whether ``event`` passes the current profile filter."""
    profile = _profile()
    if profile == PROFILE_OFF:
        return False
    if profile == PROFILE_MINIMAL:
        return event in _MINIMAL_EVENTS
    return True  # standard, strict


def _enabled() -> bool:
    """Legacy entry point — true unless profile == off.

    Kept so callers that just want to know "are we emitting anything?"
    (without naming a specific event) still work after the profile migration.
    """
    return _profile() != PROFILE_OFF


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
    if not _should_emit(event):
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
