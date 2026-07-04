"""Multi-agent deliberation tier.

A real (non-simulated) reasoning panel: generate *k* candidate plans with
diverse models, judge/rank them, code the winner, have an adversarial critic
(a *distinct* model) review it, and repair on failure — a genuine
plan → vote → code → critique → repair loop.

Design: ``docs/solutions/tiered-reasoning-multi-agent.md``. This module owns only
the *orchestration*; model diversity is injected via ``role_adapter`` (the
caller maps each role to a distinct CAR-routed model). It is deliberately
self-contained (no engine internals) so it's unit-testable with fake adapters
and callable from ``NeoEngine`` for the deliberation path.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from neo.models import CodeSuggestion, PlanStep, SimulationTrace

logger = logging.getLogger(__name__)

#: Roles the orchestrator asks the ``role_adapter`` factory for. The factory is
#: responsible for handing back *distinct* models where possible (critic ≠ coder).
ROLES = ("planner", "judge", "coder", "critic")


@dataclass
class DeliberationResult:
    """Output of a panel run, shaped to slot into the engine's tuple."""

    plan: list[PlanStep]
    simulation_traces: list[SimulationTrace]
    code_suggestions: list[CodeSuggestion]
    confidence: float
    consensus: float                       # panel agreement signal [0,1]
    rounds: int                            # repair rounds run
    provenance: str = "multi-agent"
    models_used: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def as_engine_tuple(self):
        """(plan, simulation_traces, code_suggestions) — matches ``_process_combined``."""
        return self.plan, self.simulation_traces, self.code_suggestions


def _extract_json(text: str) -> Optional[dict]:
    """Leniently pull the first JSON object out of an LLM response (tolerates
    ```json fences and prose around it)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if 0 <= start < end else None
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo


class MultiAgentReasoner:
    """Orchestrates a plan-vote → code → critique → repair panel.

    ``role_adapter(role)`` returns an object with
    ``generate(messages, max_tokens=..., temperature=...) -> str`` (neo's
    ``LMAdapter``). It may return a *different* model per role — that diversity is
    the whole point and is the caller's responsibility (see
    ``engine``'s CAR-routed factory).
    """

    def __init__(
        self,
        role_adapter: Callable[[str], object],
        *,
        k_plans: int = 3,
        max_repair_rounds: int = 1,
        max_tokens: int = 8000,  # reasoning models (gpt-5*) spend output on hidden
                                 # reasoning; give the visible answer real headroom
    ):
        self._role_adapter = role_adapter
        self.k_plans = max(1, k_plans)
        self.max_repair_rounds = max(0, max_repair_rounds)
        self.max_tokens = max_tokens
        self._models_used: dict[str, str] = {}

    # -- LLM plumbing ------------------------------------------------------

    def _call(self, role: str, system: str, user: str, *, temperature: float = 0.7) -> str:
        adapter = self._role_adapter(role)
        name = getattr(adapter, "name", lambda: role)
        try:
            self._models_used[role] = name() if callable(name) else str(name)
        except Exception:  # pragma: no cover - name() is best-effort
            self._models_used[role] = role
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return adapter.generate(messages, max_tokens=self.max_tokens, temperature=temperature) or ""

    # -- phases ------------------------------------------------------------

    def _generate_plans(self, prompt: str, context: str) -> list[dict]:
        system = (
            "You are a senior engineer producing a concise solution PLAN (not code). "
            "Respond ONLY with JSON: "
            '{"approach": str, "steps": [{"description": str, "rationale": str, '
            '"risk": "low"|"medium"|"high"}], "key_risks": [str]}'
        )
        plans: list[dict] = []
        for i in range(self.k_plans):
            # Vary temperature across candidates to spread the search even when
            # the pool can't give fully distinct models.
            temp = 0.4 + 0.25 * i
            raw = self._call("planner", system, self._task_block(prompt, context), temperature=temp)
            obj = _extract_json(raw)
            if obj and obj.get("steps"):
                plans.append(obj)
        return plans

    def _judge_plans(self, prompt: str, plans: list[dict]) -> tuple[int, float]:
        """Return (winning index, consensus in [0,1]). Consensus is how clearly
        one plan beat the field — a low value means the panel disagreed, which
        we surface as lower final confidence."""
        if len(plans) == 1:
            return 0, 0.5
        listing = "\n\n".join(
            f"PLAN {i}: approach={p.get('approach','')!r}\nsteps="
            + "; ".join(s.get("description", "") for s in p.get("steps", []))
            for i, p in enumerate(plans)
        )
        system = (
            "You are a critical reviewer choosing the best plan. Respond ONLY with JSON: "
            '{"winner": int, "confidence": 0.0-1.0, "why": str}. '
            "confidence = how clearly the winner beats the others (0.5 = a coin flip)."
        )
        raw = self._call("judge", system, f"Task:\n{prompt}\n\nCandidates:\n{listing}", temperature=0.2)
        obj = _extract_json(raw) or {}
        idx = obj.get("winner", 0)
        if not isinstance(idx, int) or not (0 <= idx < len(plans)):
            idx = 0
        return idx, _clamp(obj.get("confidence", 0.5))

    def _generate_code(self, prompt: str, context: str, plan: dict) -> dict:
        system = (
            "You are a senior engineer. Implement the given PLAN. Respond ONLY with JSON: "
            '{"code": str, "explanation": str, "edge_cases": [str], "confidence": 0.0-1.0, '
            '"file_path": str}'
        )
        user = f"{self._task_block(prompt, context)}\n\nPLAN:\n{json.dumps(plan)}"
        return _extract_json(self._call("coder", system, user, temperature=0.3)) or {}

    def _critique(self, prompt: str, code: dict) -> dict:
        """Adversarial review by a (ideally distinct) model — asked to find the flaw."""
        system = (
            "You are an adversarial code reviewer. Your job is to FIND THE FLAW. "
            "Scrutinize correctness, edge cases, and whether it actually solves the task. "
            "Respond ONLY with JSON: "
            '{"issues": [str], "verdict": "ok"|"revise", "severity": "none"|"minor"|"major"}. '
            "Default to verdict=revise if anything is wrong or unverified."
        )
        user = (
            f"Task:\n{prompt}\n\nProposed solution:\n{code.get('code','')}\n\n"
            f"Author's edge cases: {code.get('edge_cases', [])}"
        )
        obj = _extract_json(self._call("critic", system, user, temperature=0.2)) or {}
        obj.setdefault("issues", [])
        obj.setdefault("verdict", "ok")
        obj.setdefault("severity", "none")
        return obj

    def _repair(self, prompt: str, code: dict, critique: dict) -> dict:
        system = (
            "You are a senior engineer fixing your code given a reviewer's issues. "
            "Respond ONLY with JSON: "
            '{"code": str, "explanation": str, "edge_cases": [str], "confidence": 0.0-1.0, '
            '"file_path": str}'
        )
        user = (
            f"Task:\n{prompt}\n\nYour previous solution:\n{code.get('code','')}\n\n"
            f"Reviewer issues (fix ALL):\n- " + "\n- ".join(critique.get("issues", []))
        )
        return _extract_json(self._call("coder", system, user, temperature=0.3)) or code

    @staticmethod
    def _task_block(prompt: str, context: str) -> str:
        ctx = f"\n\nContext:\n{context}" if context else ""
        return f"Task:\n{prompt}{ctx}"

    # -- orchestration -----------------------------------------------------

    def deliberate(self, prompt: str, context: str = "") -> DeliberationResult:
        self._models_used = {}

        plans = self._generate_plans(prompt, context)
        if not plans:
            # Panel produced nothing parseable — signal failure to the caller,
            # which falls back to the fast path.
            return DeliberationResult(
                plan=[], simulation_traces=[], code_suggestions=[],
                confidence=0.0, consensus=0.0, rounds=0,
                meta={"error": "no parseable plans", "k_requested": self.k_plans},
            )

        winner_idx, consensus = self._judge_plans(prompt, plans)
        chosen = plans[winner_idx]

        code = self._generate_code(prompt, context, chosen)
        critique = self._critique(prompt, code)

        rounds = 0
        while critique.get("verdict") == "revise" and rounds < self.max_repair_rounds:
            code = self._repair(prompt, code, critique)
            critique = self._critique(prompt, code)
            rounds += 1

        return self._assemble(chosen, code, critique, consensus, rounds, len(plans))

    def _assemble(
        self, plan: dict, code: dict, critique: dict, consensus: float, rounds: int, n_plans: int
    ) -> DeliberationResult:
        steps = [
            PlanStep(
                description=s.get("description", ""),
                rationale=s.get("rationale", ""),
                risk=s.get("risk", "low") if s.get("risk") in ("low", "medium", "high") else "low",
                confidence=consensus,
            )
            for s in plan.get("steps", [])
        ]

        coder_conf = _clamp(code.get("confidence", 0.5))
        # Confidence is a *consensus* of independent signals, not one model's
        # self-report: plan agreement × coder confidence, penalized if the
        # adversarial critic still isn't satisfied after the repair budget.
        verdict_ok = critique.get("verdict") == "ok"
        severity = critique.get("severity", "none")
        critic_factor = {"none": 1.0, "minor": 0.85, "major": 0.55}.get(severity, 0.7)
        if not verdict_ok:
            critic_factor = min(critic_factor, 0.5)
        confidence = _clamp(coder_conf * (0.5 + 0.5 * consensus) * critic_factor)

        suggestions = []
        if code.get("code"):
            suggestions.append(
                CodeSuggestion(
                    file_path=code.get("file_path", "") or "suggestion",
                    unified_diff="",
                    description=code.get("explanation", ""),
                    confidence=confidence,
                    code_block=code.get("code", ""),
                    tradeoffs=list(critique.get("issues", []))[:5],
                )
            )

        # The "simulation trace" is now grounded in the adversarial review, not
        # the model narrating its own dry-run.
        traces = [
            SimulationTrace(
                input_data="adversarial review",
                expected_output=critique.get("verdict", "ok"),
                reasoning_steps=[s.get("description", "") for s in plan.get("steps", [])],
                issues_found=list(critique.get("issues", [])),
            )
        ]

        return DeliberationResult(
            plan=steps,
            simulation_traces=traces,
            code_suggestions=suggestions,
            confidence=confidence,
            consensus=consensus,
            rounds=rounds,
            models_used=dict(self._models_used),
            meta={
                "n_plans": n_plans,
                "critic_verdict": critique.get("verdict"),
                "critic_severity": severity,
                "distinct_models": len(set(self._models_used.values())),
            },
        )
