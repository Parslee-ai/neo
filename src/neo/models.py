"""
Neo data models and abstract interfaces.

Contains core data structures (TaskType, ContextFile, NeoInput, etc.)
and the LMAdapter abstract base class.

Split from cli.py for modularity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import uuid
from typing import Any, Literal, Optional, TypedDict

from neo.execution_context import (
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
