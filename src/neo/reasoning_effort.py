"""Memory-driven reasoning effort selection for OpenAI gpt-5* models.

The OpenAI /v1/responses endpoint accepts a `reasoning.effort` parameter that
controls how many "thinking" tokens the model spends before answering. Effort
levels (for gpt-5.5): none < low < medium < high < xhigh.

Neo's memory system already tracks per-pattern confidence, so we can use that
signal to spend reasoning tokens where they actually help: novel queries get
more thinking; familiar queries that pattern-match cleanly get less.
"""

from dataclasses import dataclass
from typing import Optional


# Ordered low-to-high. Index in this tuple == relative effort.
EFFORT_LEVELS: tuple[str, ...] = ("none", "low", "medium", "high", "xhigh")
DEFAULT_EFFORT = "medium"  # Matches the gpt-5.5 API default when field omitted.

# Thresholds for the memory-driven heuristic.
# Tuned to align with the existing 0.8-confidence / 3-success contribution
# threshold used elsewhere in neo (see FactStore promotion logic).
_HIGH_CONFIDENCE = 0.8
_LOW_CONFIDENCE = 0.5
_MIN_PATTERNS_FOR_LOW = 3


@dataclass
class MemorySignal:
    """Summary of the memory hit for the current query.

    `pattern_count` is the number of relevant retrieved facts/patterns.
    `avg_confidence` is the mean confidence over those patterns (0.0 if none).

    Confidence already encodes the project's outcome-detection feedback loop,
    so a low-confidence retrieval implicitly means "we've been wrong here
    before" — no separate failure signal is needed.
    """

    pattern_count: int = 0
    avg_confidence: float = 0.0


def effort_from_memory(signal: MemorySignal) -> str:
    """Map a memory signal to a reasoning-effort level.

    | Signal                                         | Effort |
    |------------------------------------------------|--------|
    | ≥3 patterns, avg confidence ≥ 0.8              | low    |
    | Some patterns, avg confidence in [0.5, 0.8)    | medium |
    | No patterns OR avg confidence < 0.5            | high   |

    The "no patterns" case maps to high (not xhigh) because pessimizing the
    cold-start case would tax greenfield use. xhigh is reserved for explicit
    escalation (e.g. self-correction after a failed call).
    """
    if signal.pattern_count == 0:
        return "high"
    if signal.avg_confidence < _LOW_CONFIDENCE:
        return "high"
    if (
        signal.pattern_count >= _MIN_PATTERNS_FOR_LOW
        and signal.avg_confidence >= _HIGH_CONFIDENCE
    ):
        return "low"
    return "medium"


def escalate(effort: str) -> str:
    """Bump effort one level (capped at xhigh).

    Used by repair/self-correction call sites where the prior LLM call already
    produced an unusable result — we know cheap-thinking failed, so spend more.
    """
    try:
        idx = EFFORT_LEVELS.index(effort)
    except ValueError:
        return DEFAULT_EFFORT
    return EFFORT_LEVELS[min(idx + 1, len(EFFORT_LEVELS) - 1)]


def apply_cap(effort: str, cap: Optional[str]) -> str:
    """Clamp `effort` to be no higher than `cap`. None means no cap."""
    if cap is None:
        return effort
    if effort not in EFFORT_LEVELS or cap not in EFFORT_LEVELS:
        return effort
    return effort if EFFORT_LEVELS.index(effort) <= EFFORT_LEVELS.index(cap) else cap


def validate_effort(value: Optional[str]) -> Optional[str]:
    """Return `value` if it's a valid effort or None; raise ValueError otherwise.

    Used at config-load time so a bad value fails fast rather than burning an
    API round-trip with `unsupported_value`.
    """
    if value is None:
        return None
    if value not in EFFORT_LEVELS:
        raise ValueError(
            f"Invalid reasoning effort: {value!r}. "
            f"Must be one of {EFFORT_LEVELS}."
        )
    return value


def signal_from_facts(facts: list) -> MemorySignal:
    """Build a MemorySignal from FactStore-style Fact objects.

    Each fact is expected to have `metadata.confidence`. Facts without
    metadata contribute 0.0 confidence (they exist but aren't trusted).
    """
    if not facts:
        return MemorySignal()

    confidences: list[float] = []
    for f in facts:
        meta = getattr(f, "metadata", None)
        if meta is None:
            confidences.append(0.0)
            continue
        confidences.append(float(getattr(meta, "confidence", 0.0)))

    avg = sum(confidences) / len(confidences) if confidences else 0.0
    return MemorySignal(pattern_count=len(facts), avg_confidence=avg)


def signal_from_legacy_entries(entries: list) -> MemorySignal:
    """Build a MemorySignal from PersistentReasoningMemory entries.

    Each entry is expected to have a `confidence` attribute (already adjusted
    by the success/failure feedback loop).
    """
    if not entries:
        return MemorySignal()
    confidences = [float(getattr(e, "confidence", 0.0)) for e in entries]
    avg = sum(confidences) / len(confidences)
    return MemorySignal(pattern_count=len(entries), avg_confidence=avg)
