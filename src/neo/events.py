"""
Lifecycle events emitted by NeoEngine during a single run.

Neo's reasoning is otherwise a black box: `engine.py` makes no print calls, so
a host (Claude Code, an MCP server, an IDE) sees nothing between invocation and
the final NeoOutput. This module is the neutral seam that lets a host observe
progress without the engine knowing which host it is talking to.

Design rule: events carry *facts about Neo's process* — phase names, counts,
findings. They never carry host-specific presentation wording. Deciding how and
when a fact becomes user-visible conversation belongs to the host, not here.

Sinks are best-effort observers. `safe_emit` swallows sink failures so a broken
or full output stream can never take down a reasoning run.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


class NeoEventType(str, Enum):
    """Kinds of lifecycle event a run can emit.

    Inherits from `str` so `json.dumps` serializes members directly and
    comparisons against raw strings (in tests, in host adapters) hold.
    """

    STARTED = "started"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    MEMORY_FOUND = "memory_found"
    HYPOTHESIS_FORMED = "hypothesis_formed"
    HYPOTHESIS_REJECTED = "hypothesis_rejected"
    RISK_FOUND = "risk_found"
    PERSONALITY_BEAT = "personality_beat"
    COMPLETED = "completed"
    FAILED = "failed"


# Canonical phase names. Hosts key progress display off these, so they are a
# compatibility surface: add freely, rename never.
#
# CONTEXT covers file gathering only — it is deliberately NOT called
# "retrieval", because fact retrieval happens later, while REASONING is open.
# A phase whose name overstates its span forces every emitter inside it to
# either lie or work around the name.
PHASE_CONTEXT = "context"
PHASE_REASONING = "reasoning"
PHASE_STATIC_CHECKS = "static_checks"
# Not a phase — never opened or closed, only used to label the personality
# beat emitted while the output is assembled. Hosts will never see
# `phase_started: finalize`.
PHASE_FINALIZE = "finalize"


@dataclass
class NeoEvent:
    """One observation about an in-flight run.

    `message` is a short human-readable statement of fact. `data` carries the
    structured detail a host may want to render itself rather than parse back
    out of prose.
    """

    type: NeoEventType
    phase: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializable form. Empty fields are dropped to keep lines short."""
        payload: dict[str, Any] = {"type": self.type.value}
        if self.phase:
            payload["phase"] = self.phase
        if self.message:
            payload["message"] = self.message
        if self.data:
            payload["data"] = self.data
        return payload


class NeoEventSink(Protocol):
    """Receives events as they happen.

    Implementations must be cheap and must not raise; callers route through
    `safe_emit`, which enforces the second half of that contract anyway.
    """

    def emit(self, event: NeoEvent) -> None: ...


class NullSink:
    """Discards everything. The default, so an unobserved run costs nothing."""

    def emit(self, event: NeoEvent) -> None:
        return None


class RecordingSink:
    """Keeps events in memory. For tests and for post-hoc inspection."""

    def __init__(self) -> None:
        self.events: list[NeoEvent] = []

    def emit(self, event: NeoEvent) -> None:
        self.events.append(event)

    def of_type(self, event_type: NeoEventType) -> list[NeoEvent]:
        return [e for e in self.events if e.type is event_type]


# Messages are raw LM prose (a plan step, a simulation issue, a linter
# summary) and so have no natural bound. One multi-KB line is hostile to a
# host reading line-by-line, and the rest of this engine already truncates
# for logs. Truncation is marked, never silent.
MAX_MESSAGE_CHARS = 500


class JsonlSink:
    """Writes one JSON object per line to a text stream.

    Flushed per event so a streaming host sees progress as it happens. The
    CLI's own plugin host cannot read incrementally — a `Bash` call returns
    everything at once — but an MCP server or IDE can, and at single-digit
    events per run the flush costs nothing either way.

    Not thread-safe: interleaved writes from two threads could split a line.
    Only the main thread emits today.
    """

    def __init__(self, stream) -> None:
        self.stream = stream

    def emit(self, event: NeoEvent) -> None:
        payload = event.to_dict()
        message = payload.get("message", "")
        if len(message) > MAX_MESSAGE_CHARS:
            payload["message"] = message[: MAX_MESSAGE_CHARS - 1].rstrip() + "…"
            payload["truncated"] = True
        self.stream.write(json.dumps(payload) + "\n")
        self.stream.flush()


_sink_failure_reported = False


def safe_emit(sink: Optional[NeoEventSink], event: NeoEvent) -> None:
    """Emit without letting an observer break the thing being observed.

    A closed pipe, a full disk, or a buggy third-party sink must degrade to
    "no progress reporting", never to a failed reasoning run.

    The first failure is logged at WARNING and the rest at DEBUG: a broken sink
    should be findable at default log levels without one bad stream turning
    into a per-event flood.
    """
    global _sink_failure_reported
    if sink is None:
        return
    try:
        sink.emit(event)
    except Exception as exc:
        if not _sink_failure_reported:
            _sink_failure_reported = True
            logger.warning(
                "event sink failed (%s: %s); progress reporting is degraded "
                "for the rest of this process", type(exc).__name__, exc
            )
        else:
            logger.debug("event sink failed: %s", exc)
