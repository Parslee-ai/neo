"""
Asynchronous structured-output overseer (paper 2504.15228 §A.2).

A background thread that periodically calls a caller-supplied check
function and records its structured decision to metrics. The check
function returns an ``OverseerCheck`` dataclass with five fields:

  making_progress     bool  — best-guess whether work is advancing
  is_looping          bool  — agent stuck repeating itself
  needs_notification  Optional[str] — short steering note to inject
  force_cancel_agent  bool  — request an in-flight task be cancelled
  next_check_delay    float — seconds until the next watchdog cycle

The overseer's *control flow* is fully deterministic: thread lifecycle,
metric emission, delay between checks, stop signal. Only the contents
of each ``OverseerCheck`` (which the check function produces) reflect a
judgement — typically an LLM call in the caller's context. Keeping the
two separated means the watchdog itself can be unit-tested without
any LLM in scope.

Usage:

    def check(state):
        # Caller decides what's happening; may call an LLM.
        return OverseerCheck(making_progress=True, next_check_delay=30.0)

    overseer = StructuredOverseer(check_fn=check, default_delay=30.0)
    overseer.start()
    ... do work ...
    overseer.stop()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OverseerCheck:
    """Structured output schema from one watchdog tick."""
    making_progress: bool = True
    is_looping: bool = False
    needs_notification: Optional[str] = None
    force_cancel_agent: bool = False
    next_check_delay: float = 30.0


CheckFn = Callable[[], OverseerCheck]


class StructuredOverseer:
    """Daemon-thread watchdog with deterministic lifecycle.

    The thread runs ``check_fn`` every ``next_check_delay`` seconds
    (initialized from ``default_delay``). Each result is emitted to the
    metrics jsonl. ``stop()`` is safe to call repeatedly. If
    ``check_fn`` raises, the exception is debug-logged and the thread
    falls back to ``default_delay`` for the next tick.
    """

    __slots__ = (
        "_check_fn", "_default_delay", "_stop_event", "_thread", "_min_delay",
    )

    def __init__(
        self,
        check_fn: CheckFn,
        *,
        default_delay: float = 30.0,
        min_delay: float = 1.0,
    ) -> None:
        self._check_fn = check_fn
        self._default_delay = float(default_delay)
        self._min_delay = float(min_delay)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Spawn the daemon thread. Idempotent — second start is a no-op."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="neo-overseer",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the thread to exit and join. Safe to call twice."""
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)
        self._thread = None

    def _run(self) -> None:
        from neo.memory.metrics import record as metrics_record

        delay = self._default_delay
        while not self._stop_event.is_set():
            # wait() returns True if the event was set during the wait,
            # which is our stop signal — drop out immediately.
            if self._stop_event.wait(timeout=max(self._min_delay, delay)):
                return

            try:
                check = self._check_fn()
            except Exception as e:  # pragma: no cover — defensive
                logger.debug("overseer check raised: %s", e)
                delay = self._default_delay
                continue

            try:
                metrics_record(
                    "overseer_tick",
                    making_progress=check.making_progress,
                    is_looping=check.is_looping,
                    needs_notification=check.needs_notification,
                    force_cancel_agent=check.force_cancel_agent,
                    next_check_delay=check.next_check_delay,
                )
            except Exception:  # never let metric IO kill the watchdog
                pass

            delay = check.next_check_delay if check.next_check_delay > 0 else self._default_delay
