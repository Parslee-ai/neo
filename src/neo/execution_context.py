"""Goal-aware execution envelopes for Neo's role inside external agent loops.

The resolver is deterministic and local. Inferred values are explicitly marked
as provisional and are never authoritative enough to become durable policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Optional

from neo.text_budget import shown_of


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


# Caps on what `prompt_section` shows. Every one is paired with a `shown_of`
# annotation at its use site -- an unmarked cut in prompt-bound text is the
# defect #178 existed to remove, and this module was missed by that sweep.
#
# The numbers are unchanged from the bare slices they replaced: this change is
# about making the loss VISIBLE, not about tuning how much survives. Retuning
# them is a separate decision that wants evidence about real list lengths.
_MAX_CONSTRAINTS = 12
_MAX_SUCCESS_CRITERIA = 8
_MAX_RECENT_ATTEMPTS = 3
# The proof-aware execution sections (#200) landed while this branch was open
# and arrived as bare `[:12]`/`[:12]`/`[:5]` slices — the same defect, three
# more times, in the one function whose docstring promises every cut is
# marked. Named and marked here rather than left for a later sweep, because
# an unmarked cut beneath that docstring makes the docstring the lie.
_MAX_VALIDATION_GATES = 12
_MAX_VALIDATION_OBSERVATIONS = 12
_MAX_HYPOTHESES = 5


@dataclass
class SuccessCriterion:
    """Caller-supplied evidence that defines goal completion."""

    type: str
    command: str = ""
    expected_exit_code: Optional[int] = None
    description: str = ""
    expected_value: Any = None


VALIDATION_STATUSES = frozenset({
    "passed", "failed", "warning", "unavailable", "skipped", "pending", "waived",
})
HYPOTHESIS_STATUSES = frozenset({
    "candidate", "supported", "confirmed", "rejected", "contradicted",
})


@dataclass
class ValidationGate:
    """One explicit proof obligation for completion."""

    gate_id: str
    description: str
    kind: str = "state"
    boundary: str = ""
    required: bool = True
    expected_exit_code: Optional[int] = None
    expected_value: Any = None
    state_fingerprint: str = ""
    repository_revision: str = ""
    allow_waiver: bool = False
    source: str = "explicit"


@dataclass
class ValidationObservation:
    """Caller-observed evidence linked to exactly one validation gate."""

    observation_id: str
    gate_id: str
    status: str = "unavailable"
    summary: str = ""
    actual_exit_code: Optional[int] = None
    actual_value: Any = None
    tool_name: str = ""
    observed_at: Optional[float] = None
    state_fingerprint: str = ""
    repository_revision: str = ""
    evidence_sha256: str = ""
    source: str = "caller"
    waiver_reason: str = ""


@dataclass
class HypothesisRecord:
    """A falsifiable, episode-local causal claim; never durable truth itself."""

    hypothesis_id: str
    statement: str
    status: str = "candidate"
    prior_status: str = ""
    competing_explanations: list[str] = field(default_factory=list)
    falsifying_test: str = ""
    supporting_observation_ids: list[str] = field(default_factory=list)
    contradicting_observation_ids: list[str] = field(default_factory=list)
    source: str = "caller"
    public_claim_safe: bool = False


@dataclass
class ExecutionIdentity:
    """Stable caller-controlled identity across goals, tasks, and repositories."""

    session_id: str = ""
    goal_id: str = ""
    task_id: str = ""
    parent_task_id: str = ""
    trace_id: str = ""
    discovery_source: str = ""
    blocking_goal_reason: str = ""
    repositories_touched: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)


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
    validation_gates: list[ValidationGate]
    validation_observations: list[ValidationObservation]
    hypotheses: list[HypothesisRecord]
    execution_identity: ExecutionIdentity
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
        """Stable goal-conditioned semantic query without raw trajectory dumps.

        The slices below are deliberately NOT marked, unlike the visually
        identical ones in `prompt_section`. This string is embedded and
        compared by cosine similarity; it is never read as instructions. A
        `[showing 8 of 13]` annotation here would be tokens in the query
        vector, moving every retrieval slightly toward facts about
        truncation. Marking is for text a reader could be misled by, and
        nothing reads this.
        """
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
        if self.validation_gates:
            parts.append(
                "validation gates: " + "; ".join(
                    f"{item.gate_id}={item.description}"
                    for item in self.validation_gates[:12]
                )
            )
        for item in self.validation_observations[:12]:
            parts.append(
                f"validation observation: {item.gate_id}={item.status} {item.summary}"
            )
        for item in self.hypotheses[:5]:
            parts.append(
                f"hypothesis: {item.hypothesis_id}={item.status} {item.statement}"
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
        """Bounded role contract and execution frame for provider prompts.

        Every cut here is MARKED. This text reaches the model through seven
        prompt builders in `engine.py`, and the list cuts below used to be
        bare slices: `constraints[:12]` and `success_criteria[:8]` dropped
        their tail silently. Under a prompt that says "satisfy these
        constraints", twelve of thirteen constraints read as thirteen — the
        model cannot ask what it was not told exists, and neither can the
        operator reading `--dry-run`.

        This module was missed by the sweep that introduced `text_budget`
        (#178), which found nineteen such cuts across nine prompt builders in
        six modules. It was missed for the reason that sweep itself wrote
        down: it looked for slices in the known prompt builders, and this one
        is a `prompt_section()` on a dataclass, reached only indirectly
        through `_retrieve_context`. A sweep that looks where the last bug was
        finds the last bug.
        """
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
            lines.append(
                f"Constraints{shown_of(self.constraints, _MAX_CONSTRAINTS)}: "
                + "; ".join(self.constraints[:_MAX_CONSTRAINTS])
            )
        if self.success_criteria:
            criteria = [
                item.description or item.command or item.type
                for item in self.success_criteria[:_MAX_SUCCESS_CRITERIA]
            ]
            lines.append(
                f"Success criteria"
                f"{shown_of(self.success_criteria, _MAX_SUCCESS_CRITERIA)}: "
                + "; ".join(criteria)
            )
        if self.validation_gates:
            lines.append(
                f"Validation gates"
                f"{shown_of(self.validation_gates, _MAX_VALIDATION_GATES)}: "
                + "; ".join(
                    f"{item.gate_id} ({'required' if item.required else 'optional'}): "
                    f"{item.description}"
                    for item in self.validation_gates[:_MAX_VALIDATION_GATES]
                )
            )
        if self.validation_observations:
            lines.append(
                f"Validation evidence"
                f"{shown_of(self.validation_observations, _MAX_VALIDATION_OBSERVATIONS)}: "
                + "; ".join(
                    f"{item.gate_id}={item.status}"
                    for item in self.validation_observations[:_MAX_VALIDATION_OBSERVATIONS]
                )
            )
        if self.hypotheses:
            lines.append(
                f"Hypotheses{shown_of(self.hypotheses, _MAX_HYPOTHESES)}: "
                + "; ".join(
                    f"{item.hypothesis_id}={item.status}: {item.statement}"
                    for item in self.hypotheses[:_MAX_HYPOTHESES]
                )
            )
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
            # A TAIL slice, deliberately, and marked for the same reason the
            # head cuts are: "Recent attempts" names the intent but not the
            # loss, so four attempts shown as three read as a complete history
            # of a loop that has run four times. `shown_of` counts, which is
            # all the reader needs to know something is missing.
            lines.append(
                f"Recent attempts"
                f"{shown_of(self.trajectory.attempts, _MAX_RECENT_ATTEMPTS)}: "
                + _bounded_json(
                    self.trajectory.attempts[-_MAX_RECENT_ATTEMPTS:], 1500
                )
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


@dataclass
class ValidationAssessment:
    required: int
    passed: int
    failed: int
    pending: int
    unavailable: int
    waived: int
    blocking_gate_ids: list[str] = field(default_factory=list)
    stale_gate_ids: list[str] = field(default_factory=list)
    unknown_observation_gate_ids: list[str] = field(default_factory=list)


# Honest coarse confidence bands for DERIVED (non-explicit) values. A keyword
# heuristic cannot produce a calibrated probability, so we avoid false precision
# like 0.78 vs 0.64 and map to a few defensible tiers. Explicit, caller-supplied
# values use 1.0 directly (see resolve_execution_context). These are provisional
# and never become durable policy.
_CONF_ROLE_DERIVED = 0.9  # deterministic from an explicit caller role
_CONF_KEYWORD = 0.5       # a lexical signal matched (test/error/regression/…)
_CONF_NONE = 0.3          # no signal — restating the task verbatim

# Core failure-symptom lexicon shared by TWO of three consumers: _infer_intent
# (below) and models.classify_task_type's BUGFIX symptom patterns. Adding a word
# here moves both as one, so they can't drift on "what signals a failure/bug".
# NOT fully authoritative: _infer_goal keeps its own timeout-inclusive set on
# purpose (see the marker there). If a THIRD module outside execution_context /
# models reaches for these words, extract this to a neutral `neo.lexicon` then —
# a 4-tuple doesn't earn its own module yet.
FAILURE_SIGNAL_KEYWORDS = ("error", "fail", "exception", "crash")


def _infer_goal(task: str, error_trace: Optional[str]) -> tuple[str, float, list[str]]:
    text = f"{task}\n{error_trace or ''}".lower()
    unknowns: list[str] = []
    if "test" in text and any(token in text for token in ("fail", "error", "still")):
        unknowns.append("The exact command and exit condition that define success")
        return "Restore the affected test suite to a passing state", _CONF_KEYWORD, unknowns
    # Deliberately NOT FAILURE_SIGNAL_KEYWORDS: goal-framing drops "fail" (handled
    # by the test-state branch above) and adds "timeout" as a goal symptom. This
    # third failure vocabulary diverges on purpose — do not fold it into the shared
    # constant.
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
    if any(token in text for token in FAILURE_SIGNAL_KEYWORDS):
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

    gates = list(neo_input.validation_gates)
    if not gates and criteria:
        gates = [
            ValidationGate(
                gate_id=f"legacy-{index}",
                description=item.description or item.command or item.type,
                kind=item.type,
                expected_exit_code=item.expected_exit_code,
                expected_value=item.expected_value,
                source="legacy_success_criterion",
            )
            for index, item in enumerate(criteria, 1)
        ]
    gates = _deduplicate_gates(gates)
    observations = _deduplicate_observations(neo_input.validation_observations)
    hypotheses = _normalize_hypotheses(neo_input.hypotheses, observations, gates)

    if neo_input.intent is not None and (
        neo_input.intent.type.strip() or neo_input.intent.description.strip()
    ):
        intent_value = neo_input.intent.description or neo_input.intent.type
        intent = DerivedValue(intent_value.strip(), "explicit", 1.0)
    else:
        value, confidence = _infer_intent(neo_input.prompt, neo_input.error_trace, role)
        intent = DerivedValue(value, "inferred", confidence)

    if not gates:
        unknowns.append("No explicit validation gate was supplied")

    return ResolvedExecutionContext(
        task=neo_input.prompt,
        goal=goal,
        intent=intent,
        constraints=list(neo_input.constraints),
        success_criteria=criteria,
        validation_gates=gates,
        validation_observations=observations,
        hypotheses=hypotheses,
        execution_identity=neo_input.execution_identity,
        attempt=neo_input.attempt,
        outcome=neo_input.outcome,
        progress=neo_input.progress,
        trajectory=neo_input.trajectory,
        role=role,
        requested_output=neo_input.requested_output,
        current_state=dict(neo_input.current_state),
        unknowns=list(dict.fromkeys(unknowns)),
    )


def assess_validation(context: ResolvedExecutionContext) -> ValidationAssessment:
    """Join declared gates to observations without trusting aggregate success prose."""
    gates = {item.gate_id: item for item in context.validation_gates}
    latest: dict[str, ValidationObservation] = {}
    unknown: list[str] = []
    for observation in context.validation_observations:
        if observation.gate_id not in gates:
            unknown.append(observation.gate_id)
            continue
        current = latest.get(observation.gate_id)
        current_time = current.observed_at if current and current.observed_at is not None else -1.0
        observed_time = observation.observed_at if observation.observed_at is not None else -1.0
        if current is None or observed_time >= current_time:
            latest[observation.gate_id] = observation

    passed = failed = pending = unavailable = waived = 0
    blocking: list[str] = []
    stale: list[str] = []
    required = [item for item in context.validation_gates if item.required]
    for gate in required:
        observation = latest.get(gate.gate_id)
        if observation is None or observation.status == "pending":
            pending += 1
            blocking.append(gate.gate_id)
            continue
        compatible, is_stale = _observation_matches_gate(gate, observation)
        if is_stale:
            pending += 1
            blocking.append(gate.gate_id)
            stale.append(gate.gate_id)
        elif observation.status == "passed" and compatible:
            passed += 1
        elif (
            observation.status == "waived"
            and gate.allow_waiver
            and bool(observation.waiver_reason.strip())
        ):
            waived += 1
        elif observation.status == "failed" or (
            observation.status == "passed" and not compatible
        ):
            failed += 1
            blocking.append(gate.gate_id)
        else:
            unavailable += 1
            blocking.append(gate.gate_id)

    return ValidationAssessment(
        required=len(required),
        passed=passed,
        failed=failed,
        pending=pending,
        unavailable=unavailable,
        waived=waived,
        blocking_gate_ids=list(dict.fromkeys(blocking)),
        stale_gate_ids=list(dict.fromkeys(stale)),
        unknown_observation_gate_ids=list(dict.fromkeys(unknown)),
    )


def assess_loop(context: ResolvedExecutionContext) -> tuple[GoalAssessment, StrategyAssessment]:
    """Deterministically assess loop state from observed evidence, never confidence."""
    outcome_status = (context.outcome.status.lower() if context.outcome else "")
    direction = (context.progress.direction.lower() if context.progress else "unknown")
    exhausted = bool(
        context.trajectory.max_iterations is not None
        and context.trajectory.iteration >= context.trajectory.max_iterations
    )
    validation = assess_validation(context)
    complete = bool(
        validation.required
        and validation.passed + validation.waived == validation.required
    )

    if complete:
        goal_status = GoalStatus.SATISFIED
        decision = StrategyDecision.STOP_SUCCESS
        reason = "Every required validation gate has compatible observed evidence"
    elif exhausted:
        goal_status = GoalStatus.BLOCKED
        decision = StrategyDecision.STOP_BLOCKED
        reason = "The caller-provided iteration limit has been reached"
    elif validation.failed:
        goal_status = GoalStatus.IN_PROGRESS
        decision = StrategyDecision.CHANGE_STRATEGY
        reason = "One or more required validation gates failed or had incompatible evidence"
    elif not context.validation_gates and outcome_status in {"passed", "succeeded", "success"}:
        goal_status = GoalStatus.UNVERIFIABLE
        decision = StrategyDecision.STOP_BLOCKED
        reason = "Success was reported but no explicit validation gate makes it verifiable"
    elif validation.unavailable:
        goal_status = GoalStatus.UNVERIFIABLE
        decision = StrategyDecision.STOP_BLOCKED
        reason = "Required validation evidence is unavailable, skipped, warning, or invalid"
    elif validation.pending and outcome_status in {"passed", "succeeded", "success"}:
        goal_status = GoalStatus.UNVERIFIABLE
        decision = StrategyDecision.CONTINUE
        reason = "Aggregate success was reported but required validation gates remain pending"
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
    if context.validation_gates:
        evidence = (
            f"validation gates: {validation.passed} passed, {validation.failed} failed, "
            f"{validation.pending} pending, {validation.unavailable} unavailable, "
            f"{validation.waived} waived"
        )
    elif context.progress:
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


def _observation_matches_gate(
    gate: ValidationGate,
    observation: ValidationObservation,
) -> tuple[bool, bool]:
    stale = bool(
        (gate.repository_revision and gate.repository_revision != observation.repository_revision)
        or (gate.state_fingerprint and gate.state_fingerprint != observation.state_fingerprint)
    )
    if stale:
        return False, True
    if gate.expected_exit_code is not None:
        if observation.actual_exit_code != gate.expected_exit_code:
            return False, False
    if gate.expected_value is not None and observation.actual_value != gate.expected_value:
        return False, False
    return True, False


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

    gates = [
        _validation_gate(item, index)
        for index, item in enumerate(data.get("validation_gates", [])[:50], 1)
        if isinstance(item, dict)
    ] if isinstance(data.get("validation_gates", []), list) else []
    observations = [
        _validation_observation(item, index)
        for index, item in enumerate(data.get("validation_observations", [])[:100], 1)
        if isinstance(item, dict)
    ] if isinstance(data.get("validation_observations", []), list) else []
    hypotheses = [
        _hypothesis(item, index)
        for index, item in enumerate(data.get("hypotheses", [])[:20], 1)
        if isinstance(item, dict)
    ] if isinstance(data.get("hypotheses", []), list) else []

    return {
        "goal": goal,
        "intent": intent,
        "constraints": _strings(data.get("constraints")),
        "success_criteria": [
            _criterion(item) for item in data.get("success_criteria", [])
            if isinstance(item, dict)
        ] if isinstance(data.get("success_criteria", []), list) else [],
        "validation_gates": gates,
        "validation_observations": observations,
        "hypotheses": hypotheses,
        "execution_identity": _execution_identity(data.get("execution_identity")),
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
        expected_value=_bounded_value(item.get("expected_value")),
    )


def _bounded_text(value: Any, maximum: int = 500) -> str:
    return str(value or "")[:maximum]


def _identifier(value: Any, fallback: str) -> str:
    raw = _bounded_text(value, 128).strip()
    normalized = "".join(ch for ch in raw if ch.isalnum() or ch in "-_.:")
    return normalized or fallback


def _validation_gate(item: dict[str, Any], index: int) -> ValidationGate:
    expected_exit = item.get("expected_exit_code")
    return ValidationGate(
        gate_id=_identifier(item.get("gate_id"), f"gate-{index}"),
        description=_bounded_text(item.get("description") or item.get("command") or "validation gate"),
        kind=_bounded_text(item.get("kind") or item.get("type") or "state", 64),
        boundary=_bounded_text(item.get("boundary"), 100),
        required=bool(item.get("required", True)),
        expected_exit_code=expected_exit if isinstance(expected_exit, int) else None,
        expected_value=_bounded_value(item.get("expected_value")),
        state_fingerprint=_bounded_text(item.get("state_fingerprint"), 128),
        repository_revision=_bounded_text(item.get("repository_revision"), 128),
        allow_waiver=bool(item.get("allow_waiver", False)),
        source=_bounded_text(item.get("source") or "explicit", 64),
    )


def _validation_observation(item: dict[str, Any], index: int) -> ValidationObservation:
    status = _bounded_text(item.get("status") or "unavailable", 32).lower()
    if status not in VALIDATION_STATUSES:
        status = "unavailable"
    actual_exit = item.get("actual_exit_code")
    observed_at = item.get("observed_at")
    return ValidationObservation(
        observation_id=_identifier(item.get("observation_id"), f"observation-{index}"),
        gate_id=_identifier(item.get("gate_id"), f"unknown-{index}"),
        status=status,
        summary=_bounded_text(item.get("summary"), 1000),
        actual_exit_code=actual_exit if isinstance(actual_exit, int) else None,
        actual_value=_bounded_value(item.get("actual_value")),
        tool_name=_bounded_text(item.get("tool_name"), 100),
        observed_at=float(observed_at) if isinstance(observed_at, (int, float)) else None,
        state_fingerprint=_bounded_text(item.get("state_fingerprint"), 128),
        repository_revision=_bounded_text(item.get("repository_revision"), 128),
        evidence_sha256=_bounded_text(item.get("evidence_sha256"), 128),
        source=_bounded_text(item.get("source") or "caller", 64),
        waiver_reason=_bounded_text(item.get("waiver_reason"), 500),
    )


def _hypothesis(item: dict[str, Any], index: int) -> HypothesisRecord:
    status = _bounded_text(item.get("status") or "candidate", 32).lower()
    if status not in HYPOTHESIS_STATUSES:
        status = "candidate"
    return HypothesisRecord(
        hypothesis_id=_identifier(item.get("hypothesis_id"), f"hypothesis-{index}"),
        statement=_bounded_text(item.get("statement"), 1000),
        status=status,
        prior_status=_bounded_text(item.get("prior_status"), 32).lower(),
        competing_explanations=[_bounded_text(v) for v in _strings(item.get("competing_explanations"))[:10]],
        falsifying_test=_bounded_text(item.get("falsifying_test"), 1000),
        supporting_observation_ids=[_identifier(v, "") for v in _strings(item.get("supporting_observation_ids"))[:20] if _identifier(v, "")],
        contradicting_observation_ids=[_identifier(v, "") for v in _strings(item.get("contradicting_observation_ids"))[:20] if _identifier(v, "")],
        source=_bounded_text(item.get("source") or "caller", 64),
        public_claim_safe=bool(item.get("public_claim_safe", False)),
    )


def _execution_identity(value: Any) -> ExecutionIdentity:
    if not isinstance(value, dict):
        return ExecutionIdentity()
    return ExecutionIdentity(
        session_id=_identifier(value.get("session_id"), "") if value.get("session_id") else "",
        goal_id=_identifier(value.get("goal_id"), "") if value.get("goal_id") else "",
        task_id=_identifier(value.get("task_id"), "") if value.get("task_id") else "",
        parent_task_id=_identifier(value.get("parent_task_id"), "") if value.get("parent_task_id") else "",
        trace_id=_identifier(value.get("trace_id"), "") if value.get("trace_id") else "",
        discovery_source=_bounded_text(value.get("discovery_source"), 100),
        blocking_goal_reason=_bounded_text(value.get("blocking_goal_reason"), 500),
        repositories_touched=[_bounded_text(v, 200) for v in _strings(value.get("repositories_touched"))[:20]],
        artifact_refs=[_bounded_text(v, 200) for v in _strings(value.get("artifact_refs"))[:20]],
    )


def _bounded_value(value: Any) -> Any:
    """Keep scalar comparisons exact and large structured evidence hash-only."""
    if isinstance(value, str):
        if len(value) <= 2000:
            return value
        return {
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "size": len(value),
        }
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    serialized = json.dumps(value, sort_keys=True, default=str)
    if len(serialized) <= 2000:
        try:
            return json.loads(serialized)
        except json.JSONDecodeError:
            return serialized
    return {
        "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "size": len(serialized),
    }


def _deduplicate_gates(items: list[ValidationGate]) -> list[ValidationGate]:
    result: list[ValidationGate] = []
    seen: set[str] = set()
    duplicate_counts: dict[str, int] = {}
    for item in items[:50]:
        if item.gate_id in seen:
            duplicate_counts[item.gate_id] = duplicate_counts.get(item.gate_id, 0) + 1
            item = replace(
                item,
                gate_id=(
                    f"invalid-duplicate-{item.gate_id}-{duplicate_counts[item.gate_id]}"
                )[:128],
                source="duplicate_gate_id",
            )
        seen.add(item.gate_id)
        result.append(item)
    return result


def _deduplicate_observations(
    items: list[ValidationObservation],
) -> list[ValidationObservation]:
    result: list[ValidationObservation] = []
    counts: dict[str, int] = {}
    for item in items[:100]:
        counts[item.observation_id] = counts.get(item.observation_id, 0) + 1
    duplicate_index: dict[str, int] = {}
    for item in items[:100]:
        if counts[item.observation_id] > 1:
            duplicate_index[item.observation_id] = (
                duplicate_index.get(item.observation_id, 0) + 1
            )
            item = replace(
                item,
                observation_id=(
                    f"invalid-duplicate-{item.observation_id}-"
                    f"{duplicate_index[item.observation_id]}"
                )[:128],
                status="unavailable",
                source="duplicate_observation_id",
            )
        result.append(item)
    return result


def _normalize_hypotheses(
    items: list[HypothesisRecord],
    observations: list[ValidationObservation],
    gates: list[ValidationGate],
) -> list[HypothesisRecord]:
    by_id = {item.observation_id: item for item in observations}
    gates_by_id = {item.gate_id: item for item in gates}
    result: list[HypothesisRecord] = []
    seen: set[str] = set()
    for original in items[:20]:
        item = replace(
            original,
            competing_explanations=list(original.competing_explanations),
            supporting_observation_ids=list(original.supporting_observation_ids),
            contradicting_observation_ids=list(original.contradicting_observation_ids),
        )
        if item.hypothesis_id in seen or not item.statement.strip():
            continue
        seen.add(item.hypothesis_id)
        allowed_transitions = {
            "": HYPOTHESIS_STATUSES,
            "candidate": frozenset({"candidate", "supported", "rejected"}),
            "supported": frozenset({"supported", "confirmed", "rejected"}),
            "confirmed": frozenset({"confirmed", "contradicted"}),
            "rejected": frozenset({"rejected", "candidate"}),
            "contradicted": frozenset({"contradicted", "candidate"}),
        }
        if (
            item.prior_status in HYPOTHESIS_STATUSES
            and item.status not in allowed_transitions[item.prior_status]
        ):
            item.status = item.prior_status
            item.public_claim_safe = False
        supporting = [
            by_id[item_id]
            for item_id in item.supporting_observation_ids
            if item_id in by_id
            and by_id[item_id].status == "passed"
            and by_id[item_id].gate_id in gates_by_id
            and _observation_matches_gate(
                gates_by_id[by_id[item_id].gate_id], by_id[item_id]
            ) == (True, False)
        ]
        contradicting = [
            by_id[item_id]
            for item_id in item.contradicting_observation_ids
            if item_id in by_id and by_id[item_id].status == "failed"
        ]
        if (
            item.prior_status in {"rejected", "contradicted"}
            and item.status == "candidate"
            and not supporting
            and not contradicting
        ):
            item.status = item.prior_status
            item.public_claim_safe = False
        if contradicting and item.status not in {"rejected", "contradicted"}:
            item.status = "contradicted" if item.status == "confirmed" else "rejected"
        elif (
            item.status in {"rejected", "contradicted"}
            and not contradicting
            and item.prior_status not in {"rejected", "contradicted"}
        ):
            item.status = "supported" if supporting else "candidate"
            item.public_claim_safe = False
        elif item.status == "confirmed" and (
            not item.falsifying_test.strip()
            or not any(observation.status == "passed" for observation in supporting)
        ):
            item.status = "supported" if supporting else "candidate"
            item.public_claim_safe = False
        elif item.status == "supported" and not supporting:
            item.status = "candidate"
            item.public_claim_safe = False
        if item.status != "confirmed":
            item.public_claim_safe = False
        result.append(item)
    return result


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
