"""
Neo data models and abstract interfaces.

Contains core data structures (TaskType, ContextFile, NeoInput, etc.)
and the LMAdapter abstract base class.

Split from cli.py for modularity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import re
import uuid
from typing import Any, Literal, Optional, TypedDict

from neo.execution_context import (
    FAILURE_SIGNAL_KEYWORDS,
    AttemptContext,
    CallerRole,
    GoalAssessment,
    GoalSpec,
    IntentSpec,
    OutcomeContext,
    ProgressSignal,
    StrategyAssessment,
    SuccessCriterion,
    TrajectoryContext,
)
from neo.operating_mode import AuthorityPolicy, OperatingMode


# ============================================================================
# Core Data Structures
# ============================================================================

class TaskType(Enum):
    """Type of task being requested."""
    ALGORITHM = "algorithm"
    REFACTOR = "refactor"
    BUGFIX = "bugfix"
    FEATURE = "feature"
    EXPLANATION = "explanation"


# Failure-symptom sub-patterns for BUGFIX, DERIVED from the shared
# FAILURE_SIGNAL_KEYWORDS (execution_context) so this classifier and
# _infer_intent can't drift on the failure vocabulary. Word-boundary-anchored
# with a generic inflection tail (errors/failing/crashed/exceptions), which is
# stricter than execution_context's substring match — intentional here.
_FAILURE_SYMPTOM_PATTERNS = tuple(
    rf"\b{kw}(?:s|es|ed|ing|ure)?\b" for kw in FAILURE_SIGNAL_KEYWORDS
)

# Keyword signals per task type, checked case-insensitively. Kept OUTSIDE the
# Enum body because a plain container assigned there would be coerced into an
# enum member. Ordered most- to least-specific for readability only; scoring
# (not order) decides the winner in classify_task_type.
_TASK_TYPE_SIGNALS: list[tuple[TaskType, tuple[str, ...]]] = [
    (TaskType.EXPLANATION, (
        r"\bexplain\b", r"\bdescribe\b", r"\bsummar", r"\bwalk me through\b",
        r"\bwhat (?:does|is|are|happens)\b", r"\bhow (?:does|do|is|can)\b",
        r"\bwhy (?:does|is|do|are)\b", r"\bunderstand\b",
    )),
    (TaskType.BUGFIX, (
        # Fix-action / bug-noun vocabulary local to task classification, PLUS the
        # shared failure symptoms (_FAILURE_SYMPTOM_PATTERNS). Deliberately excludes
        # ambient descriptors (wrong/incorrect/broken) that show up in feature/
        # design prose as often as in bug reports.
        r"\bfix(?:es|ed|ing)?\b", r"\bbugs?\b", r"\btraceback\b",
        r"\bregression\b", r"\bnot working\b", r"\bdoesn'?t work\b", r"\bthrows?\b",
    ) + _FAILURE_SYMPTOM_PATTERNS),
    (TaskType.ALGORITHM, (
        # (?-i:O) keeps Big-O case-sensitive; under the file-wide IGNORECASE a
        # bare ``O\(`` would match any lowercase ``o(`` — i.e. every ``info(``,
        # ``foo(``, ``undo(`` call reference — and falsely flag ALGORITHM.
        r"\boptimiz", r"\bperformance\b", r"\bfaster\b", r"\bspeed ?up\b",
        r"\befficient\b", r"\bcomplexity\b", r"(?-i:O)\(", r"\balgorithms?\b",
        r"\bdata structure\b", r"\bmemoiz",
    )),
    (TaskType.REFACTOR, (
        # No \bdedup: "dedupe"/"deduplicate" is far more often an operation/
        # function name (algorithmic) than a refactor intent; "consolidate"
        # already covers "remove duplication". Matching it tied genuine ALGORITHM
        # prompts (e.g. "optimize the dedupe function") into non-promotable REFACTOR.
        r"\brefactor", r"\bclean ?up\b", r"\brestructure\b", r"\brename\b",
        r"\bextract\b", r"\bsimplif(?:y|ies|ied)\b", r"\breorganiz",
        r"\bconsolidat", r"\btidy\b",
    )),
    (TaskType.FEATURE, (
        r"\badd\b", r"\bimplement\b", r"\bcreate\b", r"\bbuild\b",
        r"\bsupport\b", r"\bnew\b", r"\bfeature\b",
    )),
]
_TASK_TYPE_COMPILED = [
    (tt, tuple(re.compile(p, re.IGNORECASE) for p in pats))
    for tt, pats in _TASK_TYPE_SIGNALS
]
# Score-tie break, ordered to FAIL SAFE: ALL THREE non-promotable kinds
# (feature→decision, refactor→architecture, explanation→review in the engine
# kind_map) come before the two promotable ones (bugfix→pattern, algorithm→
# pattern). So a genuinely ambiguous prompt that ties a non-promotable signal
# against a promotable one resolves to the conservative non-promotable kind
# rather than leaking into durable-pattern eligibility. EXPLANATION MUST stay
# ahead of BUGFIX/ALGORITHM: "explain how optimize works" ties explanation vs
# algorithm, and prose Q&A must not mint a durable pattern. A type only wins
# outright by scoring STRICTLY higher; ties never promote.
_TASK_TYPE_PRIORITY = (
    TaskType.FEATURE, TaskType.REFACTOR, TaskType.EXPLANATION,
    TaskType.BUGFIX, TaskType.ALGORITHM,
)

# How many keyword signals a supplied error_trace is worth toward BUGFIX. Two:
# strong enough to beat a lone feature/refactor verb, weak enough that a
# 3-signal dominant intent still overrides. A named calibration, not a literal.
_FAILURE_TRACE_WEIGHT = 2


def classify_task_type(prompt: str, error_trace: Optional[str] = None) -> TaskType:
    """Infer a task type from a free-text prompt via deterministic keyword
    scoring (no LLM): the type whose distinct signal patterns match most wins; a
    tie breaks by ``_TASK_TYPE_PRIORITY`` (which fails SAFE toward the
    non-promotable kinds); a prompt with NO signal falls back to FEATURE.

    ``error_trace`` (present on the JSON/structured input path) is strong — but
    not absolute — evidence of a bugfix: a supplied traceback weights BUGFIX like
    two keyword signals, consistent with the engine's ``error_trace + BUGFIX``
    handling. It wins ties and near-ties but a clearly dominant different intent
    (e.g. a prompt that scores ALGORITHM 3) can still override, and an error_trace
    with a signal-less prompt classifies BUGFIX rather than the FEATURE fallback.

    Replaces the CLI's previously hardcoded ``TaskType.FEATURE``. That default
    routed every interactive prompt to the non-promotable ``decision`` candidate
    kind (engine ``kind_map``), so the interactive evidence-learning loop could
    never mint a durable fact. Honest classification lets a genuinely
    algorithmic/bugfix request reach the promotable ``pattern`` kind, while
    feature/refactor/explanation stay on the intentionally non-promotable kinds —
    conservatism is preserved at the kind level and, for ties, in the tie-break
    direction, not forced by a blanket default.
    """
    has_trace = isinstance(error_trace, str) and bool(error_trace.strip())
    if not isinstance(prompt, str) or not prompt:
        return TaskType.BUGFIX if has_trace else TaskType.FEATURE
    scores = {
        tt: sum(1 for pat in patterns if pat.search(prompt))
        for tt, patterns in _TASK_TYPE_COMPILED
    }
    if has_trace:
        scores[TaskType.BUGFIX] = scores.get(TaskType.BUGFIX, 0) + _FAILURE_TRACE_WEIGHT
    best = max(scores.values())
    if best == 0:
        return TaskType.FEATURE
    for task_type in _TASK_TYPE_PRIORITY:
        if scores.get(task_type, 0) == best:
            return task_type
    return TaskType.FEATURE


@dataclass
class ContextFile:
    """A file provided in the context bundle."""
    path: str
    content: str
    line_range: Optional[tuple[int, int]] = None


@dataclass
class NeoInput:
    """Input payload from the CLI tool."""
    prompt: str
    task_type: Optional[TaskType] = None
    context_files: list[ContextFile] = field(default_factory=list)
    error_trace: Optional[str] = None
    recent_commands: list[str] = field(default_factory=list)
    safe_read_paths: list[str] = field(default_factory=list)
    working_directory: Optional[str] = None
    operating_mode: OperatingMode = OperatingMode.LEARN
    authority: Optional[AuthorityPolicy] = None
    proposed_changes: list["ProposedChange"] = field(default_factory=list)
    goal: Optional[GoalSpec] = None
    intent: Optional[IntentSpec] = None
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[SuccessCriterion] = field(default_factory=list)
    attempt: Optional[AttemptContext] = None
    outcome: Optional[OutcomeContext] = None
    progress: Optional[ProgressSignal] = None
    trajectory: TrajectoryContext = field(default_factory=TrajectoryContext)
    current_state: dict[str, Any] = field(default_factory=dict)
    role: CallerRole = CallerRole.PLANNER
    requested_output: str = "next_action"


@dataclass
class ProposedChange:
    """Caller-supplied change for deterministic VERIFY mode."""

    file_path: str
    description: str = "caller-provided change"
    unified_diff: str = ""
    code_block: str = ""


@dataclass
class AppliedAction:
    """Host-executor evidence for one explicitly authorized action."""

    suggestion_id: str
    file_path: str
    status: Literal["applied", "skipped", "failed"]
    summary: str = ""
    repository_revision: str = ""


@dataclass
class PlanStep:
    """A single step in the execution plan."""
    description: str
    rationale: str
    dependencies: list[int] = field(default_factory=list)
    preconditions: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    exit_criteria: list[str] = field(default_factory=list)
    risk: Literal["low", "medium", "high"] = "low"
    retrieval_keys: list[str] = field(default_factory=list)
    failure_signatures: list[str] = field(default_factory=list)
    verifier_checks: list[str] = field(default_factory=list)
    expanded: bool = False  # Track if this step has been expanded from seed
    # MapCoder-style per-step confidence (paper 2405.11403): scaled to [0, 1].
    # Default 1.0 so legacy single-plan paths behave as before. Future
    # multi-plan generation can sort by this descending and try plans in
    # confidence order with a fallback loop.
    confidence: float = 1.0

    @property
    def aggregate_confidence(self) -> float:
        """Compose self.confidence with risk to a single [0, 1] number.

        Cheap heuristic: low/medium/high risk multiplies the planner-
        emitted confidence by 1.0 / 0.8 / 0.5 respectively. Used as the
        plan-level signal for early-exit decisions until the engine grows
        a real multi-plan branch.
        """
        risk_multiplier = {"low": 1.0, "medium": 0.8, "high": 0.5}.get(self.risk, 1.0)
        return max(0.0, min(1.0, self.confidence * risk_multiplier))


@dataclass
class SimulationTrace:
    """Trace of a simulation run."""
    input_data: str
    expected_output: str
    reasoning_steps: list[str]
    issues_found: list[str] = field(default_factory=list)


@dataclass
class CodeSuggestion:
    """A suggested code change."""
    file_path: str
    unified_diff: str
    description: str
    confidence: float  # 0.0 to 1.0
    tradeoffs: list[str] = field(default_factory=list)
    code_block: str = ""  # Optional: executable Python code (preferred over diff extraction)
    patch_content: str = ""
    apply_command: str = ""
    rollback_command: str = ""
    test_command: str = ""
    dependencies: list[str] = field(default_factory=list)
    estimated_risk: Literal["", "low", "medium", "high"] = ""
    blast_radius: float = 0.0  # 0.0-100.0 percentage
    suggestion_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class StaticCheckResult:
    """Results from static analysis tools."""
    tool_name: str
    diagnostics: list[dict[str, Any]]
    summary: str
    kind: str = ""
    status: str = ""


@dataclass
class NeoOutput:
    """Output payload back to the CLI tool."""
    plan: list[PlanStep]
    simulation_traces: list[SimulationTrace]
    code_suggestions: list[CodeSuggestion]
    static_checks: list[StaticCheckResult]
    next_questions: list[str]
    confidence: float
    notes: str
    metadata: dict[str, Any] = field(default_factory=dict)
    goal_assessment: Optional[GoalAssessment] = None
    strategy_assessment: Optional[StrategyAssessment] = None
    recommended_next_action: dict[str, Any] = field(default_factory=dict)


class RegenerateStats(TypedDict):
    """Statistics from embedding regeneration operation."""
    total: int
    success: int
    failed: int
    success_rate: float
    model: str
    duration: float


# ============================================================================
# LM Adapter Interface
# ============================================================================

class LMAdapter(ABC):
    """Abstract interface for language model providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        stop: Optional[list[str]] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Generate a response from the model.

        `reasoning_effort` (one of "none", "low", "medium", "high", "xhigh")
        controls thinking budget on OpenAI gpt-5* models. None means the
        provider's default. Adapters that don't support reasoning effort
        accept and ignore the parameter.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of this adapter."""
        pass
