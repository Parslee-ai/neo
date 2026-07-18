"""Goal-aware execution envelopes for Neo's role inside external agent loops.

The resolver is deterministic and local. Inferred values are explicitly marked
as provisional and are never authoritative enough to become durable policy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class CallerRole(str, Enum):
    PLANNER = "planner"
    DIAGNOSTICIAN = "diagnostician"
    CRITIC = "critic"
    VERIFIER = "verifier"
    STRATEGY_SELECTOR = "strategy-selector"
    MEMORY_RETRIEVER = "memory-retriever"
    POSTMORTEM_ANALYZER = "postmortem-analyzer"


class StrategyDecision(str, Enum):
    CONTINUE = "continue"
    CHANGE_STRATEGY = "change_strategy"
    STOP_SUCCESS = "stop_success"
    STOP_BLOCKED = "stop_blocked"


class GoalStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SATISFIED = "satisfied"
    BLOCKED = "blocked"
    UNVERIFIABLE = "unverifiable"


@dataclass
class SuccessCriterion:
    """Caller-supplied evidence that defines goal completion."""

    type: str
    command: str = ""
    expected_exit_code: Optional[int] = None
    description: str = ""
    expected_value: Any = None


@dataclass
class GoalSpec:
    """Desired final state, separate from the current invocation task."""

    description: str
    success_criteria: list[SuccessCriterion] = field(default_factory=list)


@dataclass
class IntentSpec:
    """Why Neo was invoked at this point in the larger trajectory."""

    type: str
    description: str = ""


@dataclass
class AttemptContext:
    """Action already taken or currently under consideration."""

    summary: str
    action_id: str = ""
    state_fingerprint: str = ""


@dataclass
class OutcomeContext:
    """Observed evidence after an attempt; never model self-confidence."""

    status: str
    goal_progress: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)
    new_errors: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    summary: str = ""
    lesson: str = ""
    disposition: str = ""


@dataclass
class ProgressSignal:
    """Explicit before/after progress measurement."""

    metric: str
    before: Any = None
    after: Any = None
    direction: str = "unknown"


@dataclass
class TrajectoryContext:
    """Bounded loop position plus prior attempts supplied by the orchestrator."""

    iteration: int = 0
    max_iterations: Optional[int] = None
    attempts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DerivedValue:
    """Explicit or inferred field with provenance and bounded confidence."""

    value: str
    origin: str  # explicit | inferred
    confidence: float


@dataclass
class ResolvedExecutionContext:
    """Normalized request frame consumed by retrieval and reasoning."""

    task: str
    goal: DerivedValue
    intent: DerivedValue
    constraints: list[str]
    success_criteria: list[SuccessCriterion]
    attempt: Optional[AttemptContext]
    outcome: Optional[OutcomeContext]
    progress: Optional[ProgressSignal]
    trajectory: TrajectoryContext
    role: CallerRole
    requested_output: str
    current_state: dict[str, Any]
    unknowns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def retrieval_query(self) -> str:
        """Stable goal-conditioned semantic query without raw trajectory dumps."""
        parts = [
            f"task: {self.task}",
            f"goal: {self.goal.value}",
            f"intent: {self.intent.value}",
            f"role: {self.role.value}",
        ]
        if self.constraints:
            parts.append("constraints: " + "; ".join(self.constraints[:8]))
        if self.success_criteria:
            parts.append(
                "success criteria: " + "; ".join(
                    item.description or item.command or item.type
                    for item in self.success_criteria[:8]
                )
            )
        if self.attempt:
            parts.append(f"attempt: {self.attempt.summary}")
        if self.outcome:
            parts.append(f"outcome: {self.outcome.status} {self.outcome.summary}")
        if self.progress:
            parts.append(
                f"progress: {self.progress.metric} {self.progress.before!r} -> "
                f"{self.progress.after!r} ({self.progress.direction})"
            )
        if self.trajectory.iteration:
            parts.append(f"iteration: {self.trajectory.iteration}")
        if self.current_state:
            parts.append(
                "current state: "
                + _bounded_json(self.current_state, 1500)
            )
        for prior in self.trajectory.attempts[-3:]:
            parts.append("prior attempt: " + _bounded_json(prior, 500))
        return "\n".join(parts)

    def prompt_section(self) -> str:
        """Bounded role contract and execution frame for provider prompts."""
        lines = [
            "## Execution Envelope",
            f"Goal ({self.goal.origin}, confidence={self.goal.confidence:.2f}): "
            f"{self.goal.value}",
            f"Intent ({self.intent.origin}, confidence={self.intent.confidence:.2f}): "
            f"{self.intent.value}",
            f"Caller role: {self.role.value}",
            f"Requested output: {self.requested_output}",
        ]
        if self.constraints:
            lines.append("Constraints: " + "; ".join(self.constraints[:12]))
        if self.success_criteria:
            criteria = [
                item.description or item.command or item.type
                for item in self.success_criteria[:8]
            ]
            lines.append("Success criteria: " + "; ".join(criteria))
        if self.attempt:
            lines.append(f"Current attempt: {self.attempt.summary}")
        if self.outcome:
            lines.append(f"Observed outcome: {self.outcome.status} — {self.outcome.summary}")
        if self.progress:
            lines.append(
                f"Progress: {self.progress.metric} {self.progress.before!r} -> "
                f"{self.progress.after!r} ({self.progress.direction})"
            )
        if self.trajectory.iteration or self.trajectory.max_iterations is not None:
            lines.append(
                f"Trajectory: iteration {self.trajectory.iteration} of "
                f"{self.trajectory.max_iterations if self.trajectory.max_iterations is not None else 'unbounded'}"
            )
        if self.trajectory.attempts:
            lines.append(
                "Recent attempts: "
                + _bounded_json(self.trajectory.attempts[-3:], 1500)
            )
        if self.current_state:
            lines.append(
                "Current state: "
                + _bounded_json(self.current_state, 1500)
            )
        lines.append(
            "Stay within the caller role. Do not invent success criteria or claim the "
            "larger goal is complete without matching observed evidence."
        )
        return "\n".join(lines)


@dataclass
class GoalAssessment:
    status: str
    progress: str
    evidence: str


@dataclass
class StrategyAssessment:
    decision: str
    reason: str


# Honest coarse confidence bands for DERIVED (non-explicit) values. A keyword
# heuristic cannot produce a calibrated probability, so we avoid false precision
# like 0.78 vs 0.64 and map to a few defensible tiers. Explicit, caller-supplied
# values use 1.0 directly (see resolve_execution_context). These are provisional
# and never become durable policy.
_CONF_ROLE_DERIVED = 0.9  # deterministic from an explicit caller role
_CONF_KEYWORD = 0.5       # a lexical signal matched (test/error/regression/…)
_CONF_NONE = 0.3          # no signal — restating the task verbatim


def _infer_goal(task: str, error_trace: Optional[str]) -> tuple[str, float, list[str]]:
    text = f"{task}\n{error_trace or ''}".lower()
    unknowns: list[str] = []
    if "test" in text and any(token in text for token in ("fail", "error", "still")):
        unknowns.append("The exact command and exit condition that define success")
        return "Restore the affected test suite to a passing state", _CONF_KEYWORD, unknowns
    if any(token in text for token in ("error", "exception", "crash", "timeout")):
        unknowns.append("Whether symptom mitigation or root-cause elimination is preferred")
        return "Resolve the reported failure without introducing regressions", _CONF_KEYWORD, unknowns
    unknowns.append("The larger final state beyond the current task")
    return task, _CONF_NONE, unknowns


def _infer_intent(task: str, error_trace: Optional[str], role: CallerRole) -> tuple[str, float]:
    text = f"{task}\n{error_trace or ''}".lower()
    if role is CallerRole.VERIFIER:
        return "Verify the supplied attempt against the stated success criteria", _CONF_ROLE_DERIVED
    if role is CallerRole.CRITIC:
        return "Critique the current attempt and identify the highest-value correction", _CONF_ROLE_DERIVED
    if any(token in text for token in ("still fail", "did not", "regression", "timeout")):
        return "Diagnose why the current attempt did not produce sufficient progress", _CONF_KEYWORD
    if any(token in text for token in ("error", "fail", "exception", "crash")):
        return "Diagnose the reported failure and recommend the next action", _CONF_KEYWORD
    return "Produce the requested reasoning artifact", _CONF_NONE


def resolve_execution_context(neo_input) -> ResolvedExecutionContext:
    """Resolve explicit envelope fields and conservative provisional defaults."""
    role = neo_input.role
    if neo_input.goal is not None and neo_input.goal.description.strip():
        goal = DerivedValue(neo_input.goal.description.strip(), "explicit", 1.0)
        criteria = list(neo_input.goal.success_criteria)
        unknowns: list[str] = []
    else:
        value, confidence, unknowns = _infer_goal(neo_input.prompt, neo_input.error_trace)
        goal = DerivedValue(value, "inferred", confidence)
        criteria = []
    if neo_input.success_criteria:
        criteria = list(neo_input.success_criteria)

    if neo_input.intent is not None and (
        neo_input.intent.type.strip() or neo_input.intent.description.strip()
    ):
        intent_value = neo_input.intent.description or neo_input.intent.type
        intent = DerivedValue(intent_value.strip(), "explicit", 1.0)
    else:
        value, confidence = _infer_intent(neo_input.prompt, neo_input.error_trace, role)
        intent = DerivedValue(value, "inferred", confidence)

    if not criteria:
        unknowns.append("No explicit success criterion was supplied")

    return ResolvedExecutionContext(
        task=neo_input.prompt,
        goal=goal,
        intent=intent,
        constraints=list(neo_input.constraints),
        success_criteria=criteria,
        attempt=neo_input.attempt,
        outcome=neo_input.outcome,
        progress=neo_input.progress,
        trajectory=neo_input.trajectory,
        role=role,
        requested_output=neo_input.requested_output,
        current_state=dict(neo_input.current_state),
        unknowns=list(dict.fromkeys(unknowns)),
    )


def assess_loop(context: ResolvedExecutionContext) -> tuple[GoalAssessment, StrategyAssessment]:
    """Deterministically assess loop state from observed evidence, never confidence."""
    outcome_status = (context.outcome.status.lower() if context.outcome else "")
    direction = (context.progress.direction.lower() if context.progress else "unknown")
    exhausted = bool(
        context.trajectory.max_iterations is not None
        and context.trajectory.iteration >= context.trajectory.max_iterations
    )

    if outcome_status in {"passed", "succeeded", "success"} and context.success_criteria:
        goal_status = GoalStatus.SATISFIED
        decision = StrategyDecision.STOP_SUCCESS
        reason = "Observed outcome reports success against explicit completion criteria"
    elif exhausted:
        goal_status = GoalStatus.BLOCKED
        decision = StrategyDecision.STOP_BLOCKED
        reason = "The caller-provided iteration limit has been reached"
    elif not context.success_criteria and outcome_status in {"passed", "succeeded", "success"}:
        goal_status = GoalStatus.UNVERIFIABLE
        decision = StrategyDecision.STOP_BLOCKED
        reason = "Success was reported but no explicit criterion makes it verifiable"
    elif direction in {"regressed", "unchanged", "no_progress", "worse"}:
        goal_status = GoalStatus.IN_PROGRESS
        decision = StrategyDecision.CHANGE_STRATEGY
        reason = "Observed progress does not support continuing the current strategy"
    elif direction in {"improved", "better"}:
        goal_status = GoalStatus.IN_PROGRESS
        decision = StrategyDecision.CONTINUE
        reason = "Observed progress supports continuing the current strategy"
    elif outcome_status in {"failed", "failure", "regressed"} and context.attempt:
        goal_status = GoalStatus.IN_PROGRESS
        decision = StrategyDecision.CHANGE_STRATEGY
        reason = "The observed attempt failed and no improving progress signal supports it"
    else:
        goal_status = GoalStatus.IN_PROGRESS
        decision = StrategyDecision.CONTINUE
        reason = "Observed evidence does not justify stopping or abandoning the strategy"

    evidence = "No explicit progress evidence supplied"
    if context.progress:
        evidence = (
            f"{context.progress.metric}: {context.progress.before!r} -> "
            f"{context.progress.after!r} ({context.progress.direction})"
        )
    elif context.outcome:
        evidence = context.outcome.summary or f"Outcome status: {context.outcome.status}"

    return (
        GoalAssessment(goal_status.value, direction, evidence),
        StrategyAssessment(decision.value, reason),
    )


def execution_fields_from_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Parse the optional wire envelope with conservative forward compatibility."""
    raw_goal = data.get("goal")
    goal = None
    if isinstance(raw_goal, str):
        goal = GoalSpec(raw_goal)
    elif isinstance(raw_goal, dict) and isinstance(raw_goal.get("description"), str):
        goal = GoalSpec(
            raw_goal["description"],
            [_criterion(item) for item in raw_goal.get("success_criteria", [])
             if isinstance(item, dict)],
        )

    raw_intent = data.get("intent")
    intent = None
    if isinstance(raw_intent, str):
        intent = IntentSpec(raw_intent, raw_intent)
    elif isinstance(raw_intent, dict):
        intent = IntentSpec(
            str(raw_intent.get("type", "")),
            str(raw_intent.get("description", "")),
        )

    raw_attempt = data.get("attempt")
    attempt = None
    if isinstance(raw_attempt, str):
        attempt = AttemptContext(raw_attempt)
    elif isinstance(raw_attempt, dict) and raw_attempt.get("summary") is not None:
        attempt = AttemptContext(
            str(raw_attempt.get("summary", "")),
            str(raw_attempt.get("action_id", "")),
            str(raw_attempt.get("state_fingerprint", "")),
        )

    raw_outcome = data.get("outcome")
    outcome = None
    if isinstance(raw_outcome, dict) and raw_outcome.get("status") is not None:
        progress_value = raw_outcome.get("goal_progress")
        outcome = OutcomeContext(
            status=str(raw_outcome.get("status", "")),
            goal_progress=(
                float(progress_value)
                if isinstance(progress_value, (int, float)) else None
            ),
            metrics=dict(raw_outcome.get("metrics", {}))
            if isinstance(raw_outcome.get("metrics"), dict) else {},
            new_errors=_strings(raw_outcome.get("new_errors")),
            side_effects=_strings(raw_outcome.get("side_effects")),
            summary=str(raw_outcome.get("summary", "")),
            lesson=str(raw_outcome.get("lesson", "")),
            disposition=str(raw_outcome.get("disposition", "")),
        )

    raw_progress = data.get("progress")
    progress = None
    if isinstance(raw_progress, dict) and raw_progress.get("metric") is not None:
        direction = str(raw_progress.get("direction", "unknown"))
        before = raw_progress.get("before")
        after = raw_progress.get("after")
        metric = str(raw_progress.get("metric", ""))
        if direction == "unknown" and isinstance(before, (int, float)) and isinstance(after, (int, float)):
            lower_is_better = any(
                token in metric.lower() for token in ("fail", "error", "defect", "latency")
            )
            if before == after:
                direction = "unchanged"
            elif (after < before) == lower_is_better:
                direction = "improved"
            else:
                direction = "regressed"
        progress = ProgressSignal(
            metric=metric,
            before=before,
            after=after,
            direction=direction,
        )

    raw_trajectory = data.get("trajectory")
    trajectory = TrajectoryContext()
    if isinstance(raw_trajectory, dict):
        max_iterations = raw_trajectory.get("max_iterations")
        attempts = raw_trajectory.get("attempts", [])
        trajectory = TrajectoryContext(
            iteration=max(0, int(raw_trajectory.get("iteration", 0) or 0)),
            max_iterations=(
                max(0, int(max_iterations))
                if isinstance(max_iterations, int) else None
            ),
            attempts=[dict(item) for item in attempts[:50] if isinstance(item, dict)]
            if isinstance(attempts, list) else [],
        )

    try:
        role = CallerRole(str(data.get("role", CallerRole.PLANNER.value)))
    except ValueError:
        role = CallerRole.PLANNER

    return {
        "goal": goal,
        "intent": intent,
        "constraints": _strings(data.get("constraints")),
        "success_criteria": [
            _criterion(item) for item in data.get("success_criteria", [])
            if isinstance(item, dict)
        ] if isinstance(data.get("success_criteria", []), list) else [],
        "attempt": attempt,
        "outcome": outcome,
        "progress": progress,
        "trajectory": trajectory,
        "current_state": {
            str(key)[:100]: value
            for key, value in list(data.get("current_state", {}).items())[:50]
        } if isinstance(data.get("current_state"), dict) else {},
        "role": role,
        "requested_output": str(data.get("requested_output", "next_action")),
    }


def _strings(value: Any) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _criterion(item: dict[str, Any]) -> SuccessCriterion:
    expected_exit = item.get("expected_exit_code")
    return SuccessCriterion(
        type=str(item.get("type", "state")),
        command=str(item.get("command", "")),
        expected_exit_code=(expected_exit if isinstance(expected_exit, int) else None),
        description=str(item.get("description", "")),
        expected_value=item.get("expected_value"),
    )


def _bounded_json(value: Any, max_chars: int) -> str:
    """Serialize caller state without walking or copying an unbounded payload."""
    def bound(item: Any, depth: int = 0) -> Any:
        if depth > 4:
            return "[TRUNCATED]"
        if isinstance(item, str):
            return item[:500]
        if isinstance(item, dict):
            return {
                str(key)[:100]: bound(child, depth + 1)
                for key, child in list(item.items())[:30]
            }
        if isinstance(item, list):
            return [bound(child, depth + 1) for child in item[:30]]
        if isinstance(item, (int, float, bool)) or item is None:
            return item
        return str(item)[:500]

    return json.dumps(bound(value), sort_keys=True, ensure_ascii=False)[:max_chars]
