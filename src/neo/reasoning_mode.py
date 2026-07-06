"""Reasoning-mode gate: decide FAST (single call + memory) vs MULTI_AGENT
(CAR-orchestrated deliberation) for a given query.

Promotes the existing novelty signal (``reasoning_effort.effort_from_memory``)
from a token-budget dimmer to a mode switch, per
``docs/solutions/tiered-reasoning-multi-agent.md``.

Key rule: multi-agent is offered when CAR is reachable AND there is at least
``min_models`` capable model available (default 1). A controlled A/B/A
(``tools/ab_controlled.py``) found the panel's gain is the *orchestration
structure* (plan-vote → adversarial critique → repair), worth +1.12/10, and that
it holds **fully same-model** — an adversarial critic in a fresh context catches
errors via reframing, not different weights; a distinct frontier critic added
~0. So one capable model suffices. (The original ``≥2 distinct models`` bar
assumed diversity was the value; the experiment overturned that.) Below the bar
we degrade to the fast path (or a high-effort single pass for explicit
``--deep``). See ``docs/solutions/tiered-reasoning-multi-agent.md``.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from neo.reasoning_effort import MemorySignal, effort_from_memory

__all__ = ["ReasoningMode", "ModeDecision", "decide_mode", "DEFAULT_MIN_MODELS"]

#: Effort levels that mean "novel / low-confidence" — the deliberation-worthy end
#: of ``effort_from_memory``. (``low``/``medium`` mean memory has this covered.)
_DELIBERATE_EFFORTS = frozenset({"high", "xhigh"})

#: A panel needs at least this many capable models to be worth running. A
#: controlled A/B/A showed the orchestration structure — not model diversity —
#: is the win and it holds same-model, so 1 capable model suffices. (Kept as a
#: tunable so a caller can still require a diverse pool for workloads where
#: diversity might pay off — harder/ambiguous tasks not yet measured.)
DEFAULT_MIN_MODELS = 1


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
        min_models: capable-model floor for a worthwhile panel (default 1).

    Returns:
        ModeDecision(mode, reason, effort).
    """
    effort = effort_from_memory(signal, difficulty=difficulty)
    can_deliberate = car_available and capable_model_count >= min_models
    override = (explicit or "").strip().lower()

    if override in ("fast", "single"):
        return ModeDecision(ReasoningMode.FAST, "explicit --fast", effort)

    if override in ("deep", "deliberate", "multi", "multi_agent", "multi-agent"):
        if can_deliberate:
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

    # Auto: novelty gates the mode; the model-pool floor gates whether we *can*.
    novel = effort in _DELIBERATE_EFFORTS
    if novel and can_deliberate:
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
                f"novel (effort={effort}) but no capable panel available "
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
