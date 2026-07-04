"""Model-diversity planning for the multi-agent panel.

The panel's value is diverse models disagreeing (a critic that shares the
coder's weights shares its blind spots). CAR's router optimizes *per-request
fitness* and won't diversify on its own, so neo engineers distinctness via
``route_model``'s ``candidates`` (see the pool) + ``exclude_models`` (force a
different model — the field CAR added for "adversarial-reviewer separation",
car#358). See ``docs/solutions/tiered-reasoning-multi-agent.md`` §4.

The routing-plan is a cheap up-front pass (``route_model`` is decision-only, no
inference), so the whole role→model assignment is computed before any agent
runs. The pure logic here is unit-tested with a fake ``route_fn``; the CAR
wiring is a thin wrapper.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: Per-role intent hints. Planner/critic want reasoning quality; coder wants a
#: code model; judge is a cheap classification. The critic is threaded with the
#: coder's model excluded so it's a *different* model wherever the pool allows.
ROLE_INTENTS: dict[str, dict] = {
    "planner": {"task": "reasoning", "prefer_quality": True},
    "coder": {"task": "code", "prefer_quality": True},
    "critic": {"task": "reasoning", "prefer_quality": True},
    "judge": {"task": "classify"},
}

# route_model reports the winner under one of these keys depending on version.
_SELECTED_KEYS = ("selected", "chosen", "model", "model_id")


def _selected_model(decision: dict) -> Optional[str]:
    for k in _SELECTED_KEYS:
        v = decision.get(k)
        if isinstance(v, str) and v:
            return v
    # Fall back to the selected candidate in the ranking.
    for c in decision.get("candidates", []) or []:
        if isinstance(c, dict) and c.get("selected") and c.get("model_id"):
            return c["model_id"]
    cands = decision.get("candidates") or []
    if cands and isinstance(cands[0], dict):
        return cands[0].get("model_id")
    return None


def _family(model_id: str) -> str:
    """Coarse model-*family* key for diversity counting. Providers expose
    multiple catalog IDs backed by the *same* model — e.g. ``parslee/advisor``
    and ``parslee/reasoning`` are both Azure gpt-5.5 (assistant personas), so
    they must count as ONE, not two, or the ``>=2 distinct models`` gate is
    fooled into deliberating with no real diversity. We key on the provider
    prefix before the first ``/`` (``parslee/*`` → ``parslee``, ``mlx/*`` →
    ``mlx``). This is conservative — it may merge genuinely-distinct models of
    one provider — which is the safe direction (bias toward the fast path).
    """
    return model_id.split("/", 1)[0] if "/" in model_id else model_id


def capable_model_count(route_fn: Callable[[str, str], str], probe: str = "code task") -> int:
    """Distinct capable model *families* CAR could route a code task to (the
    panel's real diversity ceiling — see ``_family``). Reads ``route_model``'s
    advisory ``candidates`` ranking. Returns 0 on any failure (→ gate falls back
    to the fast path)."""
    try:
        decision = json.loads(route_fn(probe, json.dumps(ROLE_INTENTS["coder"])) or "{}")
    except (ValueError, TypeError):
        return 0
    cands = decision.get("candidates") or []
    families = {_family(c["model_id"]) for c in cands
                if isinstance(c, dict) and c.get("model_id")}
    if families:
        return len(families)
    # Cold-start / explicit-model paths return no ranking but did pick one.
    return 1 if _selected_model(decision) else 0


def plan_role_models(
    route_fn: Callable[[str, str], str], prompt: str, *, roles=("planner", "coder", "critic", "judge")
) -> dict[str, str]:
    """Assign a concrete model per role, forcing the critic ≠ coder where the
    pool allows (exclude_models threading). Pure w.r.t. ``route_fn``.

    ``route_fn(prompt, intent_json) -> decision_json``.
    """
    assigned: dict[str, str] = {}
    for role in roles:
        intent = dict(ROLE_INTENTS.get(role, {"task": "code"}))
        # The critic must not be the coder's model (diversity is the point).
        if role == "critic" and assigned.get("coder"):
            intent["exclude_models"] = [assigned["coder"]]
        try:
            decision = json.loads(route_fn(prompt, json.dumps(intent)) or "{}")
        except (ValueError, TypeError):
            decision = {}
        model = _selected_model(decision)
        if model:
            assigned[role] = model
    return assigned


def build_role_factory(
    role_models: dict[str, str],
    adapter_for_model: Callable[[str], object],
    fallback_adapter: object,
) -> Callable[[str], object]:
    """Return ``role_adapter(role) -> LMAdapter``: a distinct pinned adapter per
    role from the routing plan, falling back to ``fallback_adapter`` for roles
    with no assignment. Adapters are built once and cached."""
    cache: dict[str, object] = {}

    def factory(role: str):
        model = role_models.get(role)
        if not model:
            return fallback_adapter
        if model not in cache:
            try:
                cache[model] = adapter_for_model(model)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("adapter_for_model(%s) failed: %s; using fallback", model, e)
                cache[model] = fallback_adapter
        return cache[model]

    return factory
