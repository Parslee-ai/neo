"""Reasoning-mode gate: decide FAST (single call + memory) vs MULTI_AGENT
(CAR-orchestrated deliberation) for a given query.

Promotes the existing novelty signal (``reasoning_effort.effort_from_memory``)
from a token-budget dimmer to a mode switch, per
``docs/solutions/tiered-reasoning-multi-agent.md``.

Key rule: multi-agent is offered ONLY when CAR is reachable AND there are at
least ``min_models`` genuinely-capable, *distinct* models available — because
the value of a panel is model diversity, and a single-model panel just pays
latency to re-confirm one model's blind spots. Below that bar we degrade to the
fast path (or a high-effort single pass for explicit ``--deep``).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from neo.reasoning_effort import MemorySignal, effort_from_memory

__all__ = ["ReasoningMode", "ModeDecision", "decide_mode", "DEFAULT_MIN_MODELS"]

#: Effort levels that mean "novel / low-confidence" — the deliberation-worthy end
#: of ``effort_from_memory``. (``low``/``medium`` mean memory has this covered.)
_DELIBERATE_EFFORTS = frozenset({"high", "xhigh"})

#: A panel needs at least this many distinct capable models to be worth running.
DEFAULT_MIN_MODELS = 2


class ReasoningMode(str, Enum):
    FAST = "fast"
    MULTI_AGENT = "multi_agent"


class ModeDecision:
    """The chosen mode plus a human-readable reason (logged + surfaced in
    metadata, so the routing is always explainable)."""

    __slots__ = ("mode", "reason", "effort")

    def __init__(self, mode: ReasoningMode, reason: str, effort: Optional[str] = None):
        self.mode = mode
        self.reason = reason
        self.effort = effort

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ModeDecision(mode={self.mode.value!r}, reason={self.reason!r})"


def decide_mode(
    signal: MemorySignal,
    *,
    difficulty: str = "medium",
    car_available: bool = False,
    capable_model_count: int = 0,
    explicit: Optional[str] = None,
    min_models: int = DEFAULT_MIN_MODELS,
) -> ModeDecision:
    """Decide FAST vs MULTI_AGENT.

    Args:
        signal: novelty signal (pattern_count / avg_confidence) from retrieval.
        difficulty: engine difficulty estimate ("easy"|"medium"|"hard").
        car_available: is CAR reachable (same check ``resolve_adapter`` makes).
        capable_model_count: distinct capable models CAR could route to (from
            ``route_model``'s ``candidates``). 0 when unknown/no CAR.
        explicit: user override — "fast" | "deep" | None.
        min_models: distinct-model floor for a worthwhile panel.

    Returns:
        ModeDecision(mode, reason, effort).
    """
    effort = effort_from_memory(signal, difficulty=difficulty)
    diversity_ok = car_available and capable_model_count >= min_models
    override = (explicit or "").strip().lower()

    if override in ("fast", "single"):
        return ModeDecision(ReasoningMode.FAST, "explicit --fast", effort)

    if override in ("deep", "deliberate", "multi", "multi_agent", "multi-agent"):
        if diversity_ok:
            return ModeDecision(
                ReasoningMode.MULTI_AGENT,
                f"explicit --deep ({capable_model_count} capable models)",
                effort,
            )
        # Degrade, don't error: --deep without the model pool becomes a
        # max-effort single pass.
        return ModeDecision(
            ReasoningMode.FAST,
            (
                f"--deep requested but multi-agent needs CAR + >={min_models} capable "
                f"models (car={car_available}, models={capable_model_count}); "
                "running a high-effort single pass instead"
            ),
            "xhigh",
        )

    # Auto: novelty gates the mode, diversity gates whether we *can* deliberate.
    novel = effort in _DELIBERATE_EFFORTS
    if novel and diversity_ok:
        return ModeDecision(
            ReasoningMode.MULTI_AGENT,
            (
                f"novel (effort={effort}, patterns={signal.pattern_count}, "
                f"conf={signal.avg_confidence:.2f}) with {capable_model_count} capable models"
            ),
            effort,
        )
    if novel:
        return ModeDecision(
            ReasoningMode.FAST,
            (
                f"novel (effort={effort}) but no diverse panel available "
                f"(car={car_available}, models={capable_model_count})"
            ),
            effort,
        )
    return ModeDecision(
        ReasoningMode.FAST,
        (
            f"familiar (effort={effort}, patterns={signal.pattern_count}, "
            f"conf={signal.avg_confidence:.2f}) — memory covers this"
        ),
        effort,
    )
