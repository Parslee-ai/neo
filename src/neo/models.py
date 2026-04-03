"""
Neo data models and abstract interfaces.

Contains core data structures (TaskType, ContextFile, NeoInput, etc.)
and the LMAdapter abstract base class.

Split from cli.py for modularity.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, TypedDict


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


@dataclass
class StaticCheckResult:
    """Results from static analysis tools."""
    tool_name: str
    diagnostics: list[dict[str, Any]]
    summary: str


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
    ) -> str:
        """Generate a response from the model."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Return the name of this adapter."""
        pass
