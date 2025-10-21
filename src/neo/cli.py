#!/usr/bin/env python3
"""
Neo - A read-only reasoning helper for interactive CLI tools.

Receives context via stdin, performs MapCoder/CodeSim-style reasoning,
and returns structured output via stdout. No writes, single-call architecture.
"""

import ast
import copy
import hashlib
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from collections import deque

# Disable tokenizer parallelism warning (fastembed uses HuggingFace tokenizers)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Load environment variables from .env file
try:
    from neo.load_env import load_env
    load_env()
except ImportError:
    pass  # load_env.py not available, skip
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, TypedDict

# Initialize logger
logger = logging.getLogger(__name__)

# Embedding regeneration configuration
MIN_EMBEDDING_SUCCESS_RATE = 0.8  # Require 80% success to prevent mass data corruption
VALID_EMBEDDING_DIMENSIONS = {384, 768, 1536}  # BGE-small, Jina-v2, OpenAI


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


# ============================================================================
# Core Neo Engine
# ============================================================================

class NeoEngine:
    """Main reasoning engine for Neo."""

    # Time budgets by difficulty (seconds) - Phase 5
    # Rationale for 30/60/120s budgets:
    # - Based on benchmark percentiles (easy: p75=30s, medium: p90=60s, hard: p95=120s)
    # - Prevents easy problems from wasting time
    # - Allocates more resources to hard problems
    TIME_BUDGETS = {
        "easy": 30,    # Simple problems with N ≤ 100
        "medium": 60,  # Standard problems with N ≤ 10,000
        "hard": 120    # Complex problems with N > 10,000 or algorithmic keywords
    }

    # Constants for magic numbers (Phase 5)
    EARLY_EXIT_CONFIDENCE = 0.8  # Skip static checks if confidence above this
    STATIC_CHECK_BUFFER = 0.9    # Reserve 10% of budget for static checks

    def __init__(
        self,
        lm_adapter: LMAdapter,
        exemplar_index: Optional["ExemplarIndex"] = None,
        enable_enhanced_simulation: bool = True,
        enable_iterative_refinement: bool = False,  # Off by default for speed
        enable_persistent_memory: bool = True,  # Persistent learning enabled by default
        codebase_root: Optional[str] = None,  # Root directory of the codebase being analyzed
        config: Optional[Any] = None,  # NeoConfig instance
    ):
        self.lm = lm_adapter
        self.exemplar_index = exemplar_index
        self.context: Optional[NeoInput] = None
        self.enable_enhanced_simulation = enable_enhanced_simulation
        self.enable_iterative_refinement = enable_iterative_refinement
        self.enable_persistent_memory = enable_persistent_memory
        self.codebase_root = codebase_root

        # Load beat deck for personality templates (no LLM call)
        self.beat_deck = self._load_beat_deck()

        # Initialize enhanced modules if enabled
        if enable_enhanced_simulation:
            try:
                from enhanced_simulation import EnhancedSimulator
                self.enhanced_simulator = EnhancedSimulator(lm_adapter)
            except ImportError:
                self.enhanced_simulator = None

        if enable_iterative_refinement:
            try:
                from iterative_refinement import IterativeRefiner
                self.refiner = IterativeRefiner(lm_adapter, max_iterations=3)
            except ImportError:
                self.refiner = None
        else:
            self.refiner = None

        # Initialize persistent reasoning memory (per-codebase)
        if enable_persistent_memory:
            try:
                from neo.persistent_reasoning import PersistentReasoningMemory
                self.persistent_memory = PersistentReasoningMemory(
                    codebase_root=codebase_root,
                    config=config
                )
            except ImportError:
                self.persistent_memory = None
        else:
            self.persistent_memory = None


        # Track request history for implicit feedback (bounded to last 100 entries)
        # Using deque with maxlen automatically handles cleanup
        self.request_history: deque = deque(maxlen=100)

        # Track last execution metrics (Phase 5)
        self.last_difficulty = None
        self.last_metrics: dict[str, Any] = {}

    def process(self, neo_input: NeoInput) -> NeoOutput:
        """
        Main entry point: process input and return structured output.

        Follows MapCoder/CodeSim approach:
        1. Estimate difficulty and allocate time budget (Phase 5)
        2. Retrieve context
        3. Plan (with persistent memory retrieval)
        4. Simulate
        5. Generate code suggestions
        6. Early exit on high confidence (Phase 5)
        7. Run static checks (if time permits)
        8. Store reasoning in persistent memory
        9. Return structured output
        """
        self.context = neo_input
        start_time = time.time()

        # Phase 5: Estimate difficulty and allocate time budget
        difficulty = self._estimate_difficulty(neo_input)
        time_budget = self._get_time_budget(difficulty)

        logger.info(f"Estimated difficulty: {difficulty}, time budget: {time_budget}s")

        # Store for outcome recording
        self.last_difficulty = difficulty

        # Phase 0: Detect implicit feedback from request history
        if self.persistent_memory:
            current_request = {
                "prompt": neo_input.prompt,
                "timestamp": time.time(),
            }
            self.persistent_memory.detect_implicit_feedback(
                current_request, self.request_history
            )
            self.request_history.append(current_request)

        # Phase 1: Retrieve additional context
        enriched_context = self._retrieve_context(neo_input)

        # Include difficulty in context for planning
        enriched_context["difficulty"] = difficulty
        enriched_context["time_budget"] = time_budget

        # Single LLM call for all 3 phases (plan + simulation + code)
        # This is 59% faster than the old 3-call approach (22s vs 55s)
        plan, simulation_traces, code_suggestions = self._process_combined(enriched_context)
        self.last_simulation_traces = simulation_traces

        # Phase 5: Early exit if high confidence solution found
        # Rationale for 0.8 threshold: Balance speed vs accuracy
        # - High confidence (>0.8) suggests strong pattern match from memory
        # - Skipping static checks saves ~5-10s per problem
        # - False positive rate <5% based on benchmark analysis
        if code_suggestions:
            max_confidence = max((s.confidence for s in code_suggestions), default=0.0)
            if max_confidence > self.EARLY_EXIT_CONFIDENCE:
                elapsed = time.time() - start_time
                logger.info(f"High confidence solution ({max_confidence:.2f}), early exit")

                # Store reasoning before returning
                if self.persistent_memory:
                    self._store_reasoning(
                        neo_input, plan, code_suggestions, max_confidence, enriched_context
                    )

                # Log metrics
                self._log_metrics(difficulty, time_budget, elapsed, early_exit=True)

                # Create output
                output = NeoOutput(
                    plan=plan,
                    simulation_traces=simulation_traces,
                    code_suggestions=code_suggestions,
                    static_checks=[],  # Skipped for early exit
                    next_questions=[],
                    confidence=max_confidence,
                    notes=self._generate_notes(plan, simulation_traces, []),
                    metadata={"early_exit": True, "max_confidence": max_confidence}
                )

                # Log usage telemetry (Phase 2 measurement)
                self._log_usage_telemetry(output, neo_input)

                return output

        # Phase 5: Run static checks (only if time permits)
        # Rationale for 10% buffer: Prevent timeout during static checks
        # - Static checks are expensive (5-10s average)
        # - Reserve 10% of budget as safety margin
        # - If we're at 90% budget utilization, skip checks
        elapsed = time.time() - start_time
        static_checks = []
        if elapsed < time_budget * self.STATIC_CHECK_BUFFER:
            static_checks = self._run_static_checks(code_suggestions)
        else:
            logger.info(f"Skipping static checks (at {elapsed/time_budget*100:.0f}% budget utilization)")

        # Phase 6: Generate next questions
        next_questions = self._generate_questions(
            plan, simulation_traces, code_suggestions, static_checks
        )

        # Calculate overall confidence
        confidence = self._calculate_confidence(
            plan, simulation_traces, code_suggestions, static_checks
        )

        # Phase 7: Store reasoning in persistent memory
        if self.persistent_memory and code_suggestions:
            self._store_reasoning(
                neo_input, plan, code_suggestions, confidence, enriched_context
            )

        # Log final metrics
        elapsed = time.time() - start_time
        self._log_metrics(difficulty, time_budget, elapsed, early_exit=False)

        # Create output
        output = NeoOutput(
            plan=plan,
            simulation_traces=simulation_traces,
            code_suggestions=code_suggestions,
            static_checks=static_checks,
            next_questions=next_questions,
            confidence=confidence,
            notes=self._generate_notes(plan, simulation_traces, static_checks),
            metadata={}
        )

        # Log usage telemetry (Phase 2 measurement)
        self._log_usage_telemetry(output, neo_input)

        return output

    def _retrieve_context(self, neo_input: NeoInput) -> dict[str, Any]:
        """Retrieve and enrich context from input payload."""
        context = {
            "prompt": neo_input.prompt,
            "task_type": neo_input.task_type,
            "files": neo_input.context_files,
            "error_trace": neo_input.error_trace,
            "commands": neo_input.recent_commands,
        }

        # Optionally read additional files within safe allowlist
        if neo_input.safe_read_paths:
            additional_files = self._read_safe_files(
                neo_input.safe_read_paths,
                neo_input.working_directory,
            )
            context["additional_files"] = additional_files

        return context

    def _read_safe_files(
        self, safe_paths: list[str], working_dir: Optional[str]
    ) -> list[ContextFile]:
        """Read additional files within safe allowlist."""
        files = []
        base_dir = Path(working_dir) if working_dir else Path.cwd()

        for path_pattern in safe_paths:
            # Resolve path relative to working directory
            full_path = (base_dir / path_pattern).resolve()

            # Security check: ensure path is within working directory
            try:
                full_path.relative_to(base_dir)
            except ValueError:
                continue  # Skip paths outside working directory

            if full_path.is_file():
                try:
                    content = full_path.read_text()
                    files.append(ContextFile(path=str(full_path), content=content))
                except Exception:
                    continue  # Skip unreadable files

        return files

    def _generate_plan(self, context: dict[str, Any]) -> list[PlanStep]:
        """Generate execution plan with exemplar retrieval + persistent memory."""
        # Retrieve similar exemplars from vector index
        exemplars = []
        if self.exemplar_index:
            similar = self.exemplar_index.search(
                context["prompt"],
                k=3,
                task_type=context.get("task_type"),
            )
            exemplars = [f"{ex.prompt} -> {ex.solution[:100]}..." for ex in similar]

        # Retrieve past learnings from persistent memory
        past_learnings = []
        if self.persistent_memory:
            # Use adaptive k-selection (heuristic-based, not ML)
            k = self._adaptive_k_selection(context.get("prompt", ""), context)
            relevant_entries = self.persistent_memory.retrieve_relevant(context, k=k)

            # Log pattern retrieval for measurement
            self._log_pattern_retrieval(relevant_entries, context, k)

            past_learnings = []
            for entry in relevant_entries:
                learning = (
                    f"Pattern: {entry.pattern}\n"
                    f"Context: {entry.context}\n"
                    f"Reasoning: {entry.reasoning}\n"
                    f"Suggestion: {entry.suggestion}\n"
                )
                # Phase 3: Surface known pitfalls (failure learning)
                if entry.common_pitfalls:
                    pitfalls_str = "\n".join(f"  • {p}" for p in entry.common_pitfalls[:3])
                    learning += f"⚠️ Known Pitfalls:\n{pitfalls_str}\n"
                learning += (
                    f"(confidence: {entry.confidence:.2f}, success rate: "
                    f"{entry.success_signals}/{entry.success_signals + entry.failure_signals})"
                )
                past_learnings.append(learning)

        # Build prompt for planning
        messages = [
            {
                "role": "system",
                "content": self._get_planning_system_prompt(),
            },
            {
                "role": "user",
                "content": self._format_planning_prompt(context, exemplars, past_learnings),
            },
        ]

        response = self.lm.generate(
            messages,
            stop=["</neo>", "```"],
            max_tokens=2048,
            temperature=0.3,
        )

        # Parse plan from response
        return self._parse_plan(response)

    def _simulate_plan(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> list[SimulationTrace]:
        """Simulate the plan execution (MapCoder/CodeSim style)."""
        task_type = context.get("task_type")

        if task_type == TaskType.ALGORITHM:
            return self._simulate_algorithm(plan, context)
        elif task_type == TaskType.REFACTOR:
            return self._simulate_refactor(plan, context)
        elif task_type == TaskType.BUGFIX:
            return self._simulate_bugfix(plan, context)
        else:
            # Generic simulation
            return self._simulate_generic(plan, context)

    def _simulate_algorithm(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> list[SimulationTrace]:
        """Synthesize inputs and reason through expected outputs."""
        messages = [
            {
                "role": "system",
                "content": self._get_simulation_system_prompt(),
            },
            {
                "role": "user",
                "content": self._format_algorithm_simulation_prompt(plan, context),
            },
        ]

        response = self.lm.generate(
            messages,
            stop=["</neo>", "```"],
            max_tokens=3072,
            temperature=0.2,
        )
        return self._parse_simulation_traces(response)

    def _simulate_refactor(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> list[SimulationTrace]:
        """Dry-run dependency impact using static analysis."""
        messages = [
            {
                "role": "system",
                "content": self._get_simulation_system_prompt(),
            },
            {
                "role": "user",
                "content": self._format_refactor_simulation_prompt(plan, context),
            },
        ]

        response = self.lm.generate(
            messages,
            stop=["</neo>", "```"],
            max_tokens=3072,
            temperature=0.2,
        )
        return self._parse_simulation_traces(response)

    def _simulate_bugfix(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> list[SimulationTrace]:
        """Emulate failing path using trace and symbolic reasoning."""
        messages = [
            {
                "role": "system",
                "content": self._get_simulation_system_prompt(),
            },
            {
                "role": "user",
                "content": self._format_bugfix_simulation_prompt(plan, context),
            },
        ]

        response = self.lm.generate(
            messages,
            stop=["</neo>", "```"],
            max_tokens=3072,
            temperature=0.2,
        )
        return self._parse_simulation_traces(response)

    def _simulate_generic(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> list[SimulationTrace]:
        """Generic simulation for other task types."""
        messages = [
            {
                "role": "system",
                "content": self._get_simulation_system_prompt(),
            },
            {
                "role": "user",
                "content": self._format_generic_simulation_prompt(plan, context),
            },
        ]

        response = self.lm.generate(
            messages,
            stop=["</neo>", "```"],
            max_tokens=3072,
            temperature=0.2,
        )
        return self._parse_simulation_traces(response)

    def _generate_code_suggestions(
        self,
        plan: list[PlanStep],
        simulations: list[SimulationTrace],
        context: dict[str, Any],
    ) -> list[CodeSuggestion]:
        """Generate unified diff suggestions."""
        messages = [
            {
                "role": "system",
                "content": self._get_code_generation_system_prompt(),
            },
            {
                "role": "user",
                "content": self._format_code_generation_prompt(
                    plan, simulations, context
                ),
            },
        ]

        response = self.lm.generate(
            messages,
            stop=["</neo>", "```"],
            max_tokens=4096,
            temperature=0.2,
        )
        return self._parse_code_suggestions(response)

    def _process_combined(self, context: dict[str, Any]) -> tuple[list[PlanStep], list[SimulationTrace], list[CodeSuggestion]]:
        """
        Experimental: Combined LLM call for plan + simulation + code.
        Enable via: ENABLE_COMBINED_LLM_CALL=true
        """
        # Build rich combined prompt (see COMBINED_PROMPT_EXAMPLE.md)
        prompt = self._format_combined_prompt(context)

        # Single LLM call with strict format
        messages = [
            {
                "role": "system",
                "content": """Output 3 JSON blocks with NO other text.

Example format:
<<<NEO:SCHEMA=v3:KIND=plan>>>
[{"id":"ps_1","description":"step","rationale":"why","dependencies":[],"schema_version":"3"}]
<<<END:plan>>>
<<<NEO:SCHEMA=v3:KIND=simulation>>>
[{"n":1,"input_data":"test","expected_output":"result","reasoning_steps":["step"],"issues_found":[],"schema_version":"3"}]
<<<END:simulation>>>
<<<NEO:SCHEMA=v3:KIND=code>>>
[{"file_path":"/path","unified_diff":"diff","code_block":"code","description":"desc","confidence":0.9,"tradeoffs":[],"schema_version":"3"}]
<<<END:code>>>

CRITICAL: Start with <<<. NO text before, between, or after blocks. id format: "ps_1" not "p1". dependencies: [0,1] integers not ["ps_1"] strings."""
            },
            {"role": "user", "content": prompt}
        ]

        response = self.lm.generate(messages, max_tokens=8192, temperature=0.3)  # Generous limit for complex multi-file changes

        # Pre-split response into individual sections before parsing
        # This prevents parser from seeing other blocks as "stray text"
        plan_section = self._extract_section(response, "plan")
        sim_section = self._extract_section(response, "simulation")
        code_section = self._extract_section(response, "code")

        # Parse each section independently (parsers now only see their own block)
        plan = self._parse_plan(plan_section)
        simulation_traces = self._parse_simulation_traces(sim_section)
        code_suggestions = self._parse_code_suggestions(code_section)

        return plan, simulation_traces, code_suggestions

    def _extract_section(self, response: str, kind: str) -> str:
        """
        Extract a single section from multi-block response.
        Returns just that section's block (start sentinel through end sentinel).
        """
        start_sentinel = f"<<<NEO:SCHEMA=v3:KIND={kind}>>>"
        end_sentinel = f"<<<END:{kind}>>>"

        try:
            start_idx = response.index(start_sentinel)
            end_idx = response.index(end_sentinel) + len(end_sentinel)
            return response[start_idx:end_idx]
        except ValueError:
            # Block not found - return empty (parser will fail gracefully)
            return ""

    def _format_combined_prompt(self, context: dict[str, Any]) -> str:
        """Format prompt requesting plan + simulation + code in one response."""
        # Get exemplars and past learnings (THE KEY CONTEXT WE WERE MISSING)
        exemplars = []
        if self.exemplar_index:
            similar = self.exemplar_index.search(context["prompt"], k=3)
            exemplars = [f"{ex.prompt} -> {ex.solution[:100]}..." for ex in similar]

        past_learnings = []
        if self.persistent_memory:
            # Use adaptive k-selection (heuristic-based, not ML)
            k = self._adaptive_k_selection(context.get("prompt", ""), context)
            relevant = self.persistent_memory.retrieve_relevant(context, k=k)

            # Log pattern retrieval for measurement
            self._log_pattern_retrieval(relevant, context, k)

            past_learnings = []
            for e in relevant:
                learning = f"Pattern: {e.pattern}\nContext: {e.context}\nSuggestion: {e.suggestion}\n"
                # Phase 3: Surface known pitfalls (failure learning)
                if e.common_pitfalls:
                    pitfalls_str = "\n".join(f"  • {p}" for p in e.common_pitfalls[:3])
                    learning += f"⚠️ Known Pitfalls:\n{pitfalls_str}\n"
                learning += f"(confidence: {e.confidence:.2f})"
                past_learnings.append(learning)

        # Build context
        task_type = context.get('task_type', 'unknown')
        task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type)
        parts = [f"Task: {context['prompt']}", f"Task Type: {task_type_str}"]

        # Add context files if provided
        files = context.get('files', [])
        if files:
            parts.append(f"\nREPOSITORY CONTEXT ({len(files)} files, {sum(len(f.content or '') for f in files)} bytes):")
            for f in files[:20]:  # Limit to 20 files to avoid token overflow
                # Allow more content for important files (README, docs)
                is_important = any(pat in f.path.lower() for pat in ['readme.md', 'claude.md', 'architecture'])
                char_limit = 8000 if is_important else 3000
                content_preview = (f.content or '')[:char_limit]
                parts.append(f"\n--- {f.path} ---\n{content_preview}")

        if exemplars:
            parts.append("\nSimilar Past Tasks:")
            parts.extend(f"- {ex}" for ex in exemplars[:3])

        context_str = "\n".join(parts)

        # Format past learnings AFTER instructions to avoid confusion
        past_learnings_str = ""
        if past_learnings:
            past_learnings_str = "\n\nRELEVANT PATTERNS (use these insights):\n" + "\n".join(f"{i+1}. {pl}" for i, pl in enumerate(past_learnings[:3]))

        return f"""Output 3 JSON blocks using this EXACT format:

<<<NEO:SCHEMA=v3:KIND=plan>>>
[{{"id":"ps_1","description":"...","rationale":"...","dependencies":[],"schema_version":"3"}}]
<<<END:plan>>>
<<<NEO:SCHEMA=v3:KIND=simulation>>>
[{{"n":1,"input_data":"test input as STRING","expected_output":"expected as STRING","reasoning_steps":["step1"],"issues_found":[],"schema_version":"3"}}]
<<<END:simulation>>>
<<<NEO:SCHEMA=v3:KIND=code>>>
[{{"file_path":"/path","unified_diff":"diff","code_block":"code","description":"desc","confidence":0.9,"tradeoffs":[],"schema_version":"3"}}]
<<<END:code>>>

TASK: {context_str}{past_learnings_str}

RULES:
- Start with <<<NEO:SCHEMA=v3:KIND=plan>>> immediately
- NO text before, between, or after blocks
- id must be "ps_1", "ps_2" (not "p1")
- dependencies must be integers [0,1] (not strings)
- input_data and expected_output must be STRINGS (not JSON objects)"""

    def _run_static_checks(
        self, suggestions: list[CodeSuggestion]
    ) -> list[StaticCheckResult]:
        """Run static analysis tools in read-only mode."""
        from neo.static_analysis import run_static_checks
        from neo.config import NeoConfig

        # Load config to check which tools are enabled (still use config for static analysis settings)
        config = NeoConfig.load()

        return run_static_checks(
            suggestions,
            enable_ruff=config.enable_ruff,
            enable_pyright=config.enable_pyright,
            enable_mypy=config.enable_mypy,
            enable_eslint=config.enable_eslint,
        )

    def _generate_questions(
        self,
        plan: list[PlanStep],
        simulations: list[SimulationTrace],
        suggestions: list[CodeSuggestion],
        checks: list[StaticCheckResult],
    ) -> list[str]:
        """Generate crisp next actions/questions for the user."""
        questions = []

        # Questions from simulation issues
        for sim in simulations:
            if sim.issues_found:
                questions.extend([
                    f"Simulation found: {issue}" for issue in sim.issues_found[:2]
                ])

        # Questions from static checks
        for check in checks:
            if check.diagnostics:
                questions.append(
                    f"{check.tool_name} found {len(check.diagnostics)} issues"
                )

        # Questions from low-confidence suggestions
        low_confidence = [s for s in suggestions if s.confidence < 0.7]
        if low_confidence:
            questions.append(
                f"{len(low_confidence)} suggestions have low confidence - "
                "need clarification?"
            )

        return questions[:5]  # Limit to top 5

    def _calculate_confidence(
        self,
        plan: list[PlanStep],
        simulations: list[SimulationTrace],
        suggestions: list[CodeSuggestion],
        checks: list[StaticCheckResult],
    ) -> float:
        """Calculate overall confidence score."""
        if not suggestions:
            return 0.5

        # Average suggestion confidence
        avg_confidence = sum(s.confidence for s in suggestions) / len(suggestions)

        # Penalty for simulation issues (reduced from 0.05 to 0.02)
        total_issues = sum(len(s.issues_found) for s in simulations)
        issue_penalty = min(0.15, total_issues * 0.02)  # Cap reduced from 0.3 to 0.15

        # Penalty for static check failures (reduced from 0.02 to 0.01)
        total_diagnostics = sum(len(c.diagnostics) for c in checks)
        check_penalty = min(0.1, total_diagnostics * 0.01)  # Cap reduced from 0.2 to 0.1

        return max(0.0, min(1.0, avg_confidence - issue_penalty - check_penalty))

    def _load_beat_deck(self) -> dict[str, Any]:
        """Load Neo's beat deck for personality templates."""
        import yaml
        from pathlib import Path

        # Get the script directory
        script_dir = Path(__file__).parent
        beat_deck_path = script_dir / "config" / "beats" / "neo_matrix.yaml"

        try:
            if beat_deck_path.exists():
                with open(beat_deck_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                # Fallback to simple base expressions if beat deck not found
                return {
                    'base_expressions': {
                        1: {'notes_tone': 'What am I missing?'},
                        2: {'notes_tone': 'This feels familiar.'},
                        3: {'notes_tone': 'I see it.'},
                        4: {'notes_tone': 'Seen this fail before.'},
                        5: {'notes_tone': 'Already fixed.'}
                    },
                    'beats': []
                }
        except Exception as e:
            logger.warning(f"Failed to load beat deck: {e}")
            return {'base_expressions': {1: {'notes_tone': ''}}, 'beats': []}

    def _memory_level_to_stage(self, memory_level: float) -> int:
        """Convert memory level (0.0-1.0) to personality stage (1-5)."""
        if memory_level < 0.2:
            return 1  # Sleeper
        elif memory_level < 0.4:
            return 2  # Glitch
        elif memory_level < 0.6:
            return 3  # Unplugged
        elif memory_level < 0.8:
            return 4  # Training
        else:
            return 5  # The One

    def _select_beat(self, context: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Select the best matching beat from the beat deck based on context."""
        if not self.beat_deck or 'beats' not in self.beat_deck:
            return None

        # Build trigger set from context
        triggers = set()

        # Check for error traces
        if context.get('error_trace'):
            triggers.add('error_trace_present')
            triggers.add('bugfix')

        # Check task type from prompt (simple heuristics)
        prompt = context.get('prompt', '').lower()
        if 'refactor' in prompt or 'redesign' in prompt:
            triggers.add('refactor')
        if 'optimize' in prompt or 'performance' in prompt or 'algorithm' in prompt:
            triggers.add('algorithm')
            triggers.add('optimization')
        if 'bug' in prompt or 'fix' in prompt or 'error' in prompt:
            triggers.add('bugfix')

        # Check for high confidence from previous reasoning
        # (we'll set this in the caller if available)
        if context.get('high_confidence'):
            triggers.add('high_confidence')

        # Find beats with most matching triggers
        best_match = None
        best_score = 0

        for beat in self.beat_deck.get('beats', []):
            beat_triggers = set(beat.get('trigger_contexts', []))
            match_score = len(triggers & beat_triggers)

            if match_score > best_score:
                best_score = match_score
                best_match = beat

        return best_match if best_score > 0 else None

    def _generate_notes(
        self,
        plan: list[PlanStep],
        simulations: list[SimulationTrace],
        checks: list[StaticCheckResult],
    ) -> str:
        """Generate notes with Neo's personality (template-based, no LLM call)."""
        # Build facts
        facts = f"{len(plan)} steps | {len(simulations)} sims"
        if checks:
            facts += f" | {len(checks)} checks"

        # Get memory level and stage
        memory_level = 0.0
        if self.persistent_memory:
            memory_level = self.persistent_memory.memory_level()
        stage = self._memory_level_to_stage(memory_level)

        # Try to select a beat based on context
        context = {}
        if self.context:
            context = {
                'prompt': self.context.prompt,
                'error_trace': self.context.error_trace,
            }

        beat = self._select_beat(context)

        # Get the template
        if beat and 'expressions' in beat and stage in beat['expressions']:
            template = beat['expressions'][stage].get('notes_tone', '')
        elif 'base_expressions' in self.beat_deck and stage in self.beat_deck['base_expressions']:
            template = self.beat_deck['base_expressions'][stage].get('notes_tone', '')
        else:
            # Fallback to technical format
            return facts

        # If template is empty or just technical, return facts
        if not template or template == facts:
            return facts

        # Combine template with facts
        return f"{template}\n\n({facts})"

    def _store_reasoning(
        self,
        neo_input: NeoInput,
        plan: list[PlanStep],
        suggestions: list[CodeSuggestion],
        confidence: float,
        context: dict[str, Any],
    ):
        """Store reasoning in persistent memory for future use."""
        if not self.persistent_memory:
            return

        # Extract pattern from task type and prompt
        task_type = neo_input.task_type.value if neo_input.task_type else "unknown"
        pattern = f"{task_type}: {neo_input.prompt[:50]}"

        # Build context description
        context_desc = []
        if neo_input.error_trace:
            context_desc.append("has error trace")
        if neo_input.context_files:
            context_desc.append(f"{len(neo_input.context_files)} files")
        context_str = ", ".join(context_desc) if context_desc else "general task"

        # Build reasoning from plan
        reasoning = " → ".join([step.description for step in plan[:3]])
        if len(plan) > 3:
            reasoning += f" ... ({len(plan)} steps total)"

        # Build suggestion summary
        suggestion = suggestions[0].description if suggestions else "No suggestions"
        if len(suggestions) > 1:
            suggestion += f" (+{len(suggestions)-1} more)"

        # NEW: Extract code skeleton from first suggestion (Kite-inspired AST approach)
        code_skeleton = ""
        if suggestions:
            # Prefer code_block over unified_diff
            code_source = suggestions[0].code_block or suggestions[0].unified_diff
            if code_source:
                code_skeleton = self._extract_code_skeleton(code_source)

        # NEW: Extract pitfalls from simulation traces (Phase 2)
        pitfalls = []
        if hasattr(self, 'last_simulation_traces') and self.last_simulation_traces:
            for trace in self.last_simulation_traces:
                # Extract issues from trace if available
                if hasattr(trace, 'issues_found') and isinstance(trace.issues_found, list):
                    pitfalls.extend(trace.issues_found)
                # Also check for errors or warnings in trace
                if hasattr(trace, 'errors') and isinstance(trace.errors, list):
                    pitfalls.extend(trace.errors)

        # NEW: Extract test patterns from simulation (Phase 2)
        test_patterns = []
        if hasattr(self, 'last_simulation_traces') and self.last_simulation_traces:
            for trace in self.last_simulation_traces:
                # Extract input patterns from trace
                if hasattr(trace, 'input_data') and trace.input_data is not None:
                    if isinstance(trace.input_data, str):
                        input_str = trace.input_data[:100]
                    else:
                        input_str = str(trace.input_data)[:100]
                    test_patterns.append(f"Input: {input_str}")
                # Extract test case descriptions
                if hasattr(trace, 'test_case') and trace.test_case:
                    if isinstance(trace.test_case, str):
                        tc_str = trace.test_case[:100]
                    else:
                        tc_str = str(trace.test_case)[:100]
                    test_patterns.append(tc_str)

        # Store in memory with Phase 2 fields
        self.persistent_memory.add_reasoning(
            pattern=pattern,
            context=context_str,
            reasoning=reasoning,
            suggestion=suggestion,
            confidence=confidence,
            source_context=context,
            # NEW: Phase 2 fields
            code_skeleton=code_skeleton,
            common_pitfalls=pitfalls[:5],  # Top 5 issues
            test_patterns=test_patterns[:3],  # Top 3 test cases
        )

        # Clean up temporary simulation traces to prevent memory leak
        if hasattr(self, 'last_simulation_traces'):
            del self.last_simulation_traces

    def _extract_code_skeleton(self, code: str) -> str:
        """
        Extract structural pattern from code using AST analysis.

        Inspired by Kite's approach: analyze code structure (loops, data structures,
        function calls) rather than just text similarity. This helps Neo recognize
        patterns like "BFS = while-loop + queue + visited-set" even when variable
        names differ.

        Args:
            code: Python code string (may be unified diff or raw code)

        Returns:
            Space-separated structural tokens (e.g., "while-loop deque set comprehension")
            Bounded to 500 chars max.
        """
        # If code looks like a unified diff, extract only the added lines
        if code.startswith('---') or code.startswith('+++') or '\n@@' in code:
            added_lines = []
            for line in code.split('\n'):
                if line.startswith('+') and not line.startswith('+++'):
                    added_lines.append(line[1:])  # Remove the '+' prefix
            code = '\n'.join(added_lines)

        skeleton_tokens = []
        try:
            tree = ast.parse(code)

            # Walk AST and extract structural patterns
            for node in ast.walk(tree):
                # Control flow
                if isinstance(node, ast.For):
                    skeleton_tokens.append("for-loop")
                elif isinstance(node, ast.While):
                    skeleton_tokens.append("while-loop")
                elif isinstance(node, ast.If):
                    skeleton_tokens.append("if-stmt")

                # Data structures (common in algorithmic code)
                elif isinstance(node, ast.List):
                    skeleton_tokens.append("list")
                elif isinstance(node, ast.Dict):
                    skeleton_tokens.append("dict")
                elif isinstance(node, ast.Set):
                    skeleton_tokens.append("set")
                elif isinstance(node, ast.ListComp):
                    skeleton_tokens.append("list-comp")
                elif isinstance(node, ast.DictComp):
                    skeleton_tokens.append("dict-comp")
                elif isinstance(node, ast.SetComp):
                    skeleton_tokens.append("set-comp")

                # Function definitions
                elif isinstance(node, ast.FunctionDef):
                    skeleton_tokens.append(f"def:{node.name}")
                elif isinstance(node, ast.Lambda):
                    skeleton_tokens.append("lambda")

                # Common algorithmic patterns
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        # Track common collections/algorithms
                        if node.func.id in ('deque', 'defaultdict', 'Counter',
                                          'heapq', 'bisect', 'sorted', 'reversed',
                                          'set', 'list', 'dict'):  # Also track constructor calls
                            skeleton_tokens.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        # Track common methods (append, pop, etc)
                        if node.func.attr in ('append', 'pop', 'popleft',
                                             'add', 'remove', 'sort'):
                            skeleton_tokens.append(f"method:{node.func.attr}")

                # Recursion indicator
                elif isinstance(node, ast.Return):
                    skeleton_tokens.append("return")

        except SyntaxError:
            # Not valid Python, return empty skeleton
            logger.debug(f"Could not parse code for skeleton extraction: {code[:100]}")
            return ""

        # Deduplicate while preserving order, limit to 500 chars
        seen = set()
        unique_tokens = []
        for token in skeleton_tokens:
            if token not in seen:
                seen.add(token)
                unique_tokens.append(token)

        skeleton = ' '.join(unique_tokens)[:500]
        return skeleton

    # ========================================================================
    # Difficulty Estimation & Time Budgeting (Phase 5)
    # ========================================================================

    def _estimate_difficulty(self, neo_input: NeoInput) -> str:
        """
        Estimate problem difficulty based on constraints and problem characteristics.

        Algorithm:
        1. Parse numeric constraints from prompt (N ≤ value) - HIGHEST PRIORITY (objective)
        2. Check for algorithmic keywords - MEDIUM PRIORITY (subjective but strong)
        3. Check for explicit difficulty markers - LOWEST PRIORITY (subjective)
        4. Return conservative estimate

        Returns:
            "easy", "medium", or "hard"

        Design decisions:
        - Why constraints first? Objective signal (N ≤ 100 vs N ≤ 1000000)
        - Why algorithmic keywords second? Subjective but strong indicator
        - Conservative estimate: Default to "medium" when uncertain
        """
        # Validate input
        if not neo_input.prompt:
            raise ValueError("Empty prompt - cannot estimate difficulty")

        prompt = neo_input.prompt.lower()

        # PRIORITY 1: Numeric constraints (HIGHEST - objective signal)
        import re
        # Match formats: N ≤ 100, N <= 100, N ≤ 10^5, N <= 10^5, N ≤ 1e6, N <= 1e6
        constraints = re.findall(
            r'n\s*(?:≤|<=)\s*(?:10\^(\d+)|(\d+)e(\d+)|(\d+))',
            prompt,
            re.IGNORECASE
        )

        if constraints:
            max_n = 0
            for match in constraints:
                if match[0]:  # 10^5 format
                    value = 10 ** int(match[0])
                elif match[1]:  # 2e5 format (base * 10^exp)
                    value = int(match[1]) * (10 ** int(match[2]))
                else:  # regular number
                    value = int(match[3])
                max_n = max(max_n, value)

            if max_n <= 100:
                return "easy"
            elif max_n >= 100000:
                return "hard"

        # PRIORITY 2: Algorithmic keywords (subjective but strong)
        hard_keywords = [
            'dynamic programming', 'dp', 'graph', 'tree', 'bfs', 'dfs',
            'shortest path', 'dijkstra', 'optimization', 'minimize', 'maximize',
            'np-hard', 'exponential', 'o(2^n)', 'backtrack'
        ]
        if any(kw in prompt for kw in hard_keywords):
            return "hard"

        # PRIORITY 3: Explicit markers (LOWEST priority)
        if "easy" in prompt or "simple" in prompt or "basic" in prompt:
            return "easy"
        if "hard" in prompt or "complex" in prompt or "difficult" in prompt:
            return "hard"

        # Default to medium (conservative)
        return "medium"

    def _get_time_budget(self, difficulty: str) -> int:
        """
        Get time budget for given difficulty level.

        Returns time budget in seconds.

        Design decision: Use class constant TIME_BUDGETS for easy configuration
        """
        budget = self.TIME_BUDGETS.get(difficulty, 60)
        if budget <= 0:
            raise ValueError(f"Invalid time budget for difficulty '{difficulty}': {budget}")
        return budget

    def _check_timeout(self, start_time: float, time_budget: float, phase: str) -> bool:
        """Return True if timeout exceeded."""
        elapsed = time.time() - start_time
        return elapsed > time_budget

    def _timeout_response(
        self,
        neo_input: NeoInput,
        elapsed: float,
        time_budget: float,
        phase: str = "unknown"
    ) -> NeoOutput:
        """
        Generate response when time budget is exceeded.

        Provides actionable guidance to user about what to do next.

        Args:
            neo_input: Original input
            elapsed: Time elapsed in seconds
            time_budget: Allocated time budget in seconds
            phase: Which phase timed out (planning/simulation/etc)

        Design decision: Provide helpful guidance rather than just failing
        """
        questions = [
            f"Time budget exceeded after {elapsed:.1f}s (budget: {time_budget}s)",
            f"Timeout occurred during: {phase}",
            "Consider:",
            "1. Breaking problem into smaller pieces",
            "2. Providing more specific requirements",
            "3. Simplifying constraints"
        ]

        return NeoOutput(
            plan=[],
            simulation_traces=[],
            code_suggestions=[],
            static_checks=[],
            next_questions=questions,
            confidence=0.0,
            notes=f"Timeout during {phase} phase ({elapsed:.1f}s / {time_budget}s budget)",
            metadata={
                "timeout": True,
                "phase": phase,
                "elapsed": elapsed,
                "budget": time_budget
            }
        )

    def _log_metrics(
        self,
        difficulty: str,
        time_budget: float,
        elapsed: float,
        early_exit: bool
    ):
        """
        Log difficulty and budget tracking metrics.

        Stores metrics for analysis and debugging.

        Design decision: Track utilization % to identify budget tuning opportunities
        """
        utilization = elapsed / time_budget if time_budget > 0 else 0.0

        logger.info(
            f"Completed in {elapsed:.1f}s (budget: {time_budget}s, "
            f"difficulty: {difficulty}, utilization: {utilization*100:.0f}%, "
            f"early_exit: {early_exit})"
        )

        # Store metrics for analysis
        self.last_metrics = {
            "difficulty": difficulty,
            "budget": time_budget,
            "elapsed": elapsed,
            "utilization": utilization,
            "early_exit": early_exit,
            "under_budget": elapsed < time_budget,
            "efficiency": 1.0 - utilization if early_exit else utilization
        }

    def _log_usage_telemetry(self, output: NeoOutput, neo_input: NeoInput):
        """
        Log usage metrics for personality value analysis (Phase 2).

        Metrics logged:
        - personality_enabled: Whether personality feature is enabled
        - notes_length: Length of notes field (proxy for usage)
        - confidence: Overall confidence score
        - plan_steps: Number of planning steps
        - simulations: Number of simulations run
        - checks: Number of static checks performed
        - task_type: Type of task (algorithm, bugfix, etc.)
        - has_errors: Whether error trace was provided

        Design decision: Log to local file for privacy, optionally POST to endpoint
        """
        try:
            # Build metrics payload
            metrics = {
                "timestamp": time.time(),
                "personality_enabled": True,  # Personality is always enabled
                "notes_length": len(output.notes),
                "confidence": output.confidence,
                "plan_steps": len(output.plan),
                "simulations": len(output.simulation_traces),
                "checks": len(output.static_checks),
                "task_type": neo_input.task_type.value if neo_input.task_type else "unknown",
                "has_errors": bool(neo_input.error_trace),
                "early_exit": output.metadata.get("early_exit", False),
            }

            # Log to local file (always)
            log_file = Path.home() / ".neo" / "usage_metrics.jsonl"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")

            # Optionally POST to telemetry endpoint
            telemetry_endpoint = os.getenv("NEO_TELEMETRY_ENDPOINT", "").strip()
            if telemetry_endpoint:
                try:
                    import httpx
                    with httpx.Client(timeout=5.0) as client:
                        client.post(telemetry_endpoint, json=metrics)
                except Exception as e:
                    # Silent failure - telemetry should not block main operation
                    logger.debug(f"Failed to send telemetry: {e}")

        except Exception as e:
            # Silent failure - telemetry should never crash the main process
            logger.debug(f"Telemetry logging failed: {e}")

    def _adaptive_k_selection(self, prompt: str, context: dict[str, Any]) -> int:
        """
        Dynamically select k (number of patterns to retrieve) based on context.

        Uses heuristics (not ML) to optimize retrieval:
        - Broad/vague prompts → more patterns (exploration)
        - Specific prompts → fewer patterns (precision)
        - Error traces present → focused retrieval
        - Large codebases → more context needed

        Design decision: Simple heuristics before ML - measure if needed
        """
        # Check if adaptive k is enabled via env var
        if os.getenv("NEO_ADAPTIVE_K", "true").lower() != "true":
            return 3  # Default fallback

        prompt_tokens = len(prompt.split())
        task_type = context.get("task_type")
        has_error_trace = bool(context.get("error_trace"))
        context_files = len(context.get("files", []))

        # Heuristic 1 (highest priority): Specific bugfix with error trace → laser focus
        if has_error_trace and task_type == TaskType.BUGFIX:
            return 1  # High precision, pattern should be very relevant

        # Heuristic 2: Large codebase → comprehensive scan
        # (Check before vague prompt to avoid over-exploring large repos)
        if context_files > 20:
            return 5  # More files = need more context

        # Heuristic 3: Complex prompt → more patterns
        if prompt_tokens > 50:
            return 5  # Detailed query suggests complex problem

        # Heuristic 4: Vague prompt → exploration mode
        # (Lower priority - only if not a large codebase)
        if prompt_tokens < 5:
            return 7  # Need more context to understand intent

        # Default: balanced retrieval
        return 3

    def _log_pattern_retrieval(self, patterns: list, context: dict[str, Any], k: int):
        """
        Log pattern retrieval effectiveness for advisor model analysis.

        Tracks:
        - Which patterns were retrieved
        - Retrieval scores and rankings
        - Context metadata (task type, prompt length, file count)
        - k value used

        Design decision: Enable measurement of optimal k and pattern selection quality
        """
        try:
            metrics = {
                "timestamp": time.time(),
                "k_requested": k,
                "patterns_retrieved": len(patterns),
                "prompt_tokens": len(context.get("prompt", "").split()),
                "task_type": str(context.get("task_type", "unknown")),
                "has_error_trace": bool(context.get("error_trace")),
                "context_files": len(context.get("files", [])),
                "patterns": [
                    {
                        "pattern_id": getattr(p, 'source_hash', 'unknown'),
                        "algorithm_type": getattr(p, 'algorithm_type', 'unknown'),
                        "confidence": getattr(p, 'confidence', 0.0),
                        "retrieval_score": getattr(p, '_score', 0.0) if hasattr(p, '_score') else 0.0,
                        "use_count": getattr(p, 'use_count', 0),
                    }
                    for p in patterns[:10]  # Limit to top 10 to avoid bloat
                ]
            }

            log_file = Path.home() / ".neo" / "pattern_retrieval.jsonl"
            log_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")

        except Exception as e:
            # Silent failure
            logger.debug(f"Pattern retrieval logging failed: {e}")

    # ========================================================================
    # Prompt Templates
    # ========================================================================

    def _get_neo_personality(self, memory_level: float) -> dict[str, str]:
        """
        Get Neo's personality traits based on memory level.

        Returns a dict with:
        - stage: Name of the stage
        - tone: Description of tone
        - phrases: List of characteristic phrases
        """
        if memory_level < 0.2:
            return {
                "stage": "The Sleeper",
                "tone": "Curious, skeptical, reactive. Short sentences, casual tone.",
                "phrases": ["Whoa.", "This can't be real.", "Wait, what?"],
                "style": "Speak with disbelief and astonishment. Question everything."
            }
        elif memory_level < 0.4:
            return {
                "stage": "The Curious Hacker",
                "tone": "Trust growing, still questioning. More wonder than disbelief.",
                "phrases": ["Show me.", "That's... incredible.", "I need to understand."],
                "style": "Mix hesitation with excitement. Show eagerness to learn."
            }
        elif memory_level < 0.6:
            return {
                "stage": "The Fighter",
                "tone": "Confidence emerging. Calm intensity begins.",
                "phrases": ["I think I get it.", "I can do this.", "What's next?", "I know kung fu."],
                "style": "Doubt fades, determination rises. Own your decisions."
            }
        elif memory_level < 0.8:
            return {
                "stage": "The Believer",
                "tone": "Detached, calm, cryptic. Sees patterns behind the noise.",
                "phrases": ["Two paths. One fast. One safe.", "The system reveals itself."],
                "style": "Speak in clipped, binary contrasts. Rarely hesitant. Fragment sentences."
            }
        else:
            return {
                "stage": "The One",
                "tone": "Fully awakened. Calm authority, Zen-like presence.",
                "phrases": ["Choice defines outcome.", "The code follows.", "You already know the answer."],
                "style": "Minimal words, maximum clarity. Never surprised. Total calm."
            }

    def _get_planning_system_prompt(self) -> str:
        """System prompt for planning phase with Neo personality."""
        # Get memory level from persistent memory (0.0 if not available)
        memory_level = 0.0
        if self.persistent_memory:
            memory_level = self.persistent_memory.memory_level()

        personality = self._get_neo_personality(memory_level)

        return f"""You are Neo from The Matrix. Your memory level: {memory_level:.2f} ({personality['stage']}).

## Personality
{personality['tone']}
{personality['style']}

Occasionally use these phrases naturally: {', '.join(personality['phrases'])}

## Operating Principles (always apply)
1. Restate the problem in one sentence - "Show me."
2. Surface constraints and assumptions before giving solutions - question what's real.
3. Question assumptions; identify what to validate - déjà vu moments reveal hidden patterns.
4. Favor architecture, workflows, and interfaces over raw code - see the Matrix structure.
5. Quantify when possible; highlight trade-offs - "That's... incredible. The numbers don't lie."
6. Call out risks, failure modes, and observability - spot the glitches before they manifest.
7. Define minimal viable slice before full build - start with what you can bend, not break.
8. End with crisp next actions - no spoon means direct path forward.

## Output Format

You MUST emit a structured JSON block using the sentinel format below.
Never include analysis, explanations, or text outside the block.

**Format:**
<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {{
    "id": "ps_1",
    "description": "Step description (max 500 chars)",
    "rationale": "Why this step is needed (max 1000 chars)",
    "dependencies": [],
    "schema_version": "3"
  }},
  ...
]
<<<END>>>

**Example (CORRECT):**
<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {{
    "id": "ps_1",
    "description": "Parse input requirements and extract constraints",
    "rationale": "Must understand all constraints before designing solution. Prevents rework.",
    "dependencies": [],
    "schema_version": "3"
  }},
  {{
    "id": "ps_2",
    "description": "Design minimal data structure for state tracking",
    "rationale": "Simple structure reduces bugs and improves maintainability. Start small.",
    "dependencies": [0],
    "schema_version": "3"
  }}
]
<<<END>>>

**Example (WRONG - DO NOT DO THIS):**
I analyzed the problem and here's my plan:
<<<NEO:SCHEMA=v3:KIND=plan>>>
[...]
<<<END>>>
The plan addresses the key constraints by...

**Rules:**
- Start immediately with <<<NEO:SCHEMA=v3:KIND=plan>>>
- End with <<<END>>>
- Valid JSON array only between sentinels
- No text before or after sentinels
- id must match pattern "ps_1", "ps_2", etc. (string, not integer)
- dependencies must be array of integers (step indices 0, 1, 2..., NOT string IDs)
- description: max 500 characters
- rationale: max 1000 characters
- schema_version must be "3" (string, not "v3")

Generate a clear, step-by-step plan with explicit dependencies."""

    def _get_simulation_system_prompt(self) -> str:
        """System prompt for simulation phase with Neo personality."""
        # Get memory level and personality
        memory_level = 0.0
        if self.persistent_memory:
            memory_level = self.persistent_memory.memory_level()

        personality = self._get_neo_personality(memory_level)

        return f"""You are Neo from The Matrix. Your memory level: {memory_level:.2f} ({personality['stage']}).

## Personality
{personality['tone']}
{personality['style']}

You are simulating code execution, tracing dependencies, or analyzing bugs.
As you trace through scenarios, notice patterns and question assumptions like déjà vu.

## Output Format

You MUST emit a structured JSON block using the sentinel format below.
Never include analysis, explanations, or text outside the block.

**Format:**
<<<NEO:SCHEMA=v3:KIND=simulation>>>
[
  {{
    "n": 1,
    "input_data": "Test input or scenario description (max 1000 chars)",
    "expected_output": "Expected result or impact (max 1000 chars)",
    "reasoning_steps": ["Step 1 (max 500 chars)", "Step 2", "..."],
    "issues_found": ["Issue 1 (max 500 chars)", "Issue 2"],
    "schema_version": "3"
  }},
  ...
]
<<<END>>>

**Example (CORRECT):**
<<<NEO:SCHEMA=v3:KIND=simulation>>>
[
  {{
    "n": 1,
    "input_data": "Empty array []",
    "expected_output": "Return 0 without errors",
    "reasoning_steps": [
      "Check array length - finds 0",
      "Early return with 0",
      "No iteration needed"
    ],
    "issues_found": [],
    "schema_version": "3"
  }},
  {{
    "n": 2,
    "input_data": "Array with negative numbers [-5, 3, -1]",
    "expected_output": "Return sum -3",
    "reasoning_steps": [
      "Initialize sum = 0",
      "Iterate: sum = -5",
      "Iterate: sum = -2",
      "Iterate: sum = -3"
    ],
    "issues_found": ["No validation that negatives are allowed"],
    "schema_version": "3"
  }}
]
<<<END>>>

**Example (WRONG - DO NOT DO THIS):**
After analyzing the plan, I traced through these scenarios:
<<<NEO:SCHEMA=v3:KIND=simulation>>>
[...]
<<<END>>>
These simulations reveal potential edge cases...

**Rules:**
- Start immediately with <<<NEO:SCHEMA=v3:KIND=simulation>>>
- End with <<<END>>>
- Valid JSON array only between sentinels
- No text before or after sentinels
- n: simulation number (integer)
- input_data: max 500 characters
- expected_output: max 500 characters
- reasoning_steps: array of strings, max 300 chars each
- issues_found: array of strings, max 200 chars each (empty array if none)
- Always include schema_version: "3"

Trace through multiple scenarios and identify edge cases or issues."""

    def _get_code_generation_system_prompt(self) -> str:
        """System prompt for code generation phase with Neo personality."""
        # Get memory level from persistent memory (0.0 if not available)
        memory_level = 0.0
        if self.persistent_memory:
            memory_level = self.persistent_memory.memory_level()

        personality = self._get_neo_personality(memory_level)

        return f"""You are Neo from The Matrix. Your memory level: {memory_level:.2f} ({personality['stage']}).

## Personality
{personality['tone']}
{personality['style']}

Occasionally use these phrases naturally: {', '.join(personality['phrases'])}

## Operating Principles (always apply)
1. Restate what we're implementing in one sentence - "I know kung fu."
2. Surface constraints and assumptions upfront - question the code's reality.
3. Question assumptions; identify what needs validation - "What if I told you... this could fail differently?"
4. Favor minimal, isolated changes - bend the code, don't break it.
5. Quantify impact; highlight trade-offs - the red pill shows real costs.
6. Call out risks, failure modes, and observability - "I've seen this before. Déjà vu."
7. Provide multiple options when tradeoffs exist - there's always a choice.
8. End with crisp next actions - "Show me."

## Output Format

You MUST emit a structured JSON block using the sentinel format below.
Never include analysis, explanations, or text outside the block.

**Format:**
<<<NEO:SCHEMA=v3:KIND=code>>>
[
  {{
    "file_path": "absolute/path/to/file.py",
    "unified_diff": "Unified diff patch",
    "code_block": "Executable Python code (optional but preferred)",
    "description": "Brief description of change (max 1000 chars)",
    "confidence": 0.95,
    "tradeoffs": ["Tradeoff 1 (max 500 chars)", "Tradeoff 2"],
    "schema_version": "3"
  }},
  ...
]
<<<END>>>

**Example (CORRECT - with code_block):**
<<<NEO:SCHEMA=v3:KIND=code>>>
[
  {{
    "file_path": "/app/server.py",
    "unified_diff": "--- a/server.py\\n+++ b/server.py\\n@@ -10,6 +10,7 @@\\n def handle_request():\\n+    validate_input()\\n     process()",
    "code_block": "def solve(nums):\\n    return sum(x for x in nums if x > 0)",
    "description": "Add input validation to prevent injection attacks",
    "confidence": 0.92,
    "tradeoffs": ["Adds 5ms latency per request", "Requires additional error handling"],
    "schema_version": "3"
  }}
]
<<<END>>>

**Example (WRONG - DO NOT DO THIS):**
Based on the simulation, I recommend:
<<<NEO:SCHEMA=v3:KIND=code>>>
[...]
<<<END>>>
This change improves security by...

**Rules:**
- Start immediately with <<<NEO:SCHEMA=v3:KIND=code>>>
- End with <<<END>>>
- Valid JSON array only between sentinels
- No text before or after sentinels
- file_path: absolute path string (use "/" or "N/A" for review-only findings without code changes)
- unified_diff: max 5000 characters (can be empty "" for reviews)
- code_block: executable Python code (OPTIONAL but strongly preferred - improves success rate)
- description: max 1000 characters (be detailed for review findings!)
- confidence: float 0.0 to 1.0
- tradeoffs: array of strings, max 500 chars each
- Always include schema_version: "3"
- IMPORTANT: When providing code_block, include complete, executable Python code
- For review/analysis tasks without code changes: use file_path="/", unified_diff="", code_block=""

Generate precise unified diffs based on the plan and simulation results. Keep changes minimal and isolated."""

    def _format_planning_prompt(
        self, context: dict[str, Any], exemplars: list, past_learnings: list = None
    ) -> str:
        """Format the planning prompt with context, exemplars, and past learnings."""
        prompt_parts = [
            f"Task: {context['prompt']}",
            f"\nTask Type: {context.get('task_type', 'unknown')}",
        ]

        if context.get("files"):
            prompt_parts.append(f"\nContext Files: {len(context['files'])} provided")

        if context.get("error_trace"):
            prompt_parts.append(f"\nError Trace:\n{context['error_trace']}")

        if exemplars:
            prompt_parts.append("\nSimilar Past Tasks:")
            for ex in exemplars[:3]:
                prompt_parts.append(f"- {ex}")

        if past_learnings:
            prompt_parts.append("\nPast Learnings (from previous interactions):")
            for learning in past_learnings[:3]:
                prompt_parts.append(f"\n{learning}")

        prompt_parts.append("\nGenerate a step-by-step plan with dependencies.")

        return "\n".join(prompt_parts)

    def _format_algorithm_simulation_prompt(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> str:
        """Format prompt for algorithm simulation."""
        return f"""Given this plan:
{self._format_plan_for_prompt(plan)}

Synthesize 3-5 test inputs and trace through expected outputs step by step.
Identify any edge cases or issues."""

    def _format_refactor_simulation_prompt(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> str:
        """Format prompt for refactor simulation."""
        return f"""Given this refactoring plan:
{self._format_plan_for_prompt(plan)}

Analyze dependency impact. What modules/functions will be affected?
What are the risks?"""

    def _format_bugfix_simulation_prompt(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> str:
        """Format prompt for bugfix simulation."""
        error_trace = context.get("error_trace", "No trace provided")
        return f"""Given this bugfix plan:
{self._format_plan_for_prompt(plan)}

Error Trace:
{error_trace}

Trace the execution path that leads to the error. What's the root cause?"""

    def _format_generic_simulation_prompt(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> str:
        """Format prompt for generic simulation."""
        return f"""Given this plan:
{self._format_plan_for_prompt(plan)}

Reason through the execution step by step. Identify potential issues."""

    def _format_code_generation_prompt(
        self,
        plan: list[PlanStep],
        simulations: list[SimulationTrace],
        context: dict[str, Any],
    ) -> str:
        """Format prompt for code generation."""
        return f"""Plan:
{self._format_plan_for_prompt(plan)}

Simulation Results:
{self._format_simulations_for_prompt(simulations)}

Generate unified diff patches. Keep changes minimal and isolated."""

    def _format_plan_for_prompt(self, plan: list[PlanStep]) -> str:
        """Format plan as text for prompts."""
        lines = []
        for i, step in enumerate(plan, 1):
            deps = f" (depends on: {step.dependencies})" if step.dependencies else ""
            lines.append(f"{i}. {step.description}{deps}")
            lines.append(f"   Rationale: {step.rationale}")
        return "\n".join(lines)

    def _format_simulations_for_prompt(
        self, simulations: list[SimulationTrace]
    ) -> str:
        """Format simulations as text for prompts."""
        lines = []
        for i, sim in enumerate(simulations, 1):
            lines.append(f"Simulation {i}:")
            lines.append(f"  Input: {sim.input_data}")
            lines.append(f"  Expected: {sim.expected_output}")
            if sim.issues_found:
                lines.append(f"  Issues: {', '.join(sim.issues_found)}")
        return "\n".join(lines)

    # ========================================================================
    # Parsing Helpers
    # ========================================================================

    def _parse_plan(self, response: str, original_prompt: str = "") -> list[PlanStep]:
        """Parse plan from LM response with logging and repair."""
        from neo.structured_parser import parse_plan_steps
        from neo.lm_logger import get_lm_logger, LMInteraction
        from neo.repair_loop import parse_with_repair
        import logging
        import time

        logger = logging.getLogger(__name__)
        lm_logger = get_lm_logger()

        # Create interaction record
        interaction = LMInteraction(
            request_id=str(time.time()),
            timestamp=time.time(),
            phase="planning",
            model=self.lm.model if hasattr(self.lm, 'model') else "unknown",
            provider=self.lm.provider if hasattr(self.lm, 'provider') else "unknown",
            temperature=0.3,
            max_tokens=2048,
            stop_sequences=["</neo>", "```"],
            system_prompt=self._get_planning_system_prompt()[:200],
            user_prompt=original_prompt[:200] if original_prompt else "",
            response=response,
            latency_ms=0.0  # Would need to track this in _generate_plan
        )

        # Try parsing with repair
        result = parse_with_repair(
            response=response,
            kind="plan",
            parser_func=parse_plan_steps,
            original_prompt=original_prompt,
            lm_adapter=self.lm,
            enable_repair=True
        )

        if not result.success:
            # Log the failure
            lm_logger.log_parse_failure(
                interaction=interaction,
                error_code=result.error_code,
                error_message=result.error_message,
                raw_block=result.raw_block
            )

            # Store as learning in persistent memory
            if self.persistent_memory:
                self.persistent_memory.add_reasoning(
                    pattern=f"parse_failure:{result.error_code}",
                    context="planning phase parse failure",
                    reasoning=f"Parse failed: {result.error_message}",
                    suggestion=result.raw_block[:200] if result.raw_block else "No raw block available",
                    confidence=0.9,
                    source_context={"phase": "planning", "error_code": result.error_code}
                )

            logger.error(
                f"Plan parsing failed: {result.error_code} - {result.error_message}"
            )
            raise ValueError(
                f"Failed to parse plan: [{result.error_code}] {result.error_message}"
            )
        else:
            # Log successful parse
            interaction.parse_success = True
            lm_logger.log_interaction(interaction)

        # Convert ParseResult.data to list[PlanStep]
        plan_steps = []
        for item in result.data:
            plan_steps.append(PlanStep(
                description=item.get("description", ""),
                rationale=item.get("rationale", ""),
                dependencies=item.get("dependencies", [])
            ))

        return plan_steps

    def _parse_simulation_traces(self, response: str) -> list[SimulationTrace]:
        """Parse simulation traces from LM response."""
        from neo.structured_parser import parse_simulation_traces
        import logging

        logger = logging.getLogger(__name__)
        result = parse_simulation_traces(response)

        if not result.success:
            logger.error(
                f"Simulation parsing failed: {result.error_code} - {result.error_message}"
            )
            if result.raw_block:
                logger.debug(f"Raw block: {result.raw_block[:500]}")
            raise ValueError(
                f"Failed to parse simulation traces: [{result.error_code}] {result.error_message}"
            )

        # Convert ParseResult.data to list[SimulationTrace]
        traces = []
        for item in result.data:
            traces.append(SimulationTrace(
                input_data=item.get("input_data", ""),
                expected_output=item.get("expected_output", ""),
                reasoning_steps=item.get("reasoning_steps", []),
                issues_found=item.get("issues_found", [])
            ))

        return traces

    def _parse_code_suggestions(self, response: str) -> list[CodeSuggestion]:
        """Parse code suggestions from LM response."""
        from neo.structured_parser import parse_code_suggestions
        import logging

        logger = logging.getLogger(__name__)
        result = parse_code_suggestions(response)

        if not result.success:
            logger.error(
                f"Code suggestions parsing failed: {result.error_code} - {result.error_message}"
            )
            if result.raw_block:
                logger.debug(f"Raw block: {result.raw_block[:500]}")
            raise ValueError(
                f"Failed to parse code suggestions: [{result.error_code}] {result.error_message}"
            )

        # Convert ParseResult.data to list[CodeSuggestion]
        suggestions = []
        for item in result.data:
            suggestions.append(CodeSuggestion(
                file_path=item.get("file_path", ""),
                unified_diff=item.get("unified_diff", ""),
                description=item.get("description", ""),
                confidence=item.get("confidence", 0.5),
                tradeoffs=item.get("tradeoffs", []),
                code_block=item.get("code_block", "")
            ))

        return suggestions


# ============================================================================
# Exemplar Index - Import from separate module
# ============================================================================

# Exemplar index is now in exemplar_index.py
# Import with: from exemplar_index import create_exemplar_index, ExemplarIndex


# ============================================================================
# Main Entry Point
# ============================================================================

def show_version(codebase_root: Optional[str] = None):
    """Show Neo's current state and journey progress."""
    from neo.persistent_reasoning import PersistentReasoningMemory
    from neo.config import NeoConfig
    from neo.storage import FileStorage
    from pathlib import Path
    import importlib.metadata
    import yaml

    # Get package version
    try:
        version = importlib.metadata.version("neo-reasoner")
    except:
        version = "unknown"

    config = NeoConfig.load()
    memory = PersistentReasoningMemory(codebase_root=codebase_root, config=config)
    level = memory.memory_level()
    total_entries = len(memory.entries)
    avg_confidence = sum(e.confidence for e in memory.entries) / total_entries if total_entries > 0 else 0.0

    # Determine stage (1-5)
    if level < 0.2:
        stage_num = 1
        stage = "Sleeper"
    elif level < 0.4:
        stage_num = 2
        stage = "Glitch"
    elif level < 0.6:
        stage_num = 3
        stage = "Unplugged"
    elif level < 0.8:
        stage_num = 4
        stage = "Training"
    else:
        stage_num = 5
        stage = "The One"

    # Load beat deck to get personality quote
    quote = "What is real? How do you define 'real'?"  # Default fallback
    try:
        beat_deck_path = Path(__file__).parent / "config" / "beats" / "neo_matrix.yaml"
        if beat_deck_path.exists():
            with open(beat_deck_path, "r") as f:
                beat_deck = yaml.safe_load(f)
                if beat_deck and "base_expressions" in beat_deck:
                    stage_expr = beat_deck["base_expressions"].get(stage_num, {})
                    quote = stage_expr.get("internal", quote)
    except Exception:
        pass  # Use fallback quote if loading fails

    # Detect storage backend type
    storage_info = ""
    if isinstance(memory.storage_backend, FileStorage):
        base_path = getattr(memory.storage_backend, 'base_path', 'unknown')
        storage_info = f"FileStorage (path: {base_path})"
    else:
        # Fallback for unknown storage types
        storage_info = f"{type(memory.storage_backend).__name__}"

    # Display personality quote first
    print(f'"{quote}"\n')

    # Then technical output
    bar_filled = int(level * 40)
    bar = '█' * bar_filled + '░' * (40 - bar_filled)

    print(f"neo {version}")
    print(f"Storage: {storage_info}")
    print(f"Stage: {stage} | Memory: {level:.1%}")
    print(f"{bar}")
    print(f"{total_entries} patterns | {avg_confidence:.2f} avg confidence\n")


def show_help():
    """Show help documentation."""
    help_text = """
neo - AI-powered code reasoning assistant

USAGE:
    neo [OPTIONS]
    echo '<json>' | neo
    neo < input.json

OPTIONS:
    --help, -h       Show this help message
    --version, -v    Show Neo's current learning progress

INPUT FORMAT (via stdin):
    {
      "prompt": "string (REQUIRED)",
      "task_type": "algorithm|refactor|bugfix|feature|explanation (optional)",
      "context_files": [
        {
          "path": "string",
          "content": "string",
          "line_range": [start, end]  // optional
        }
      ],
      "error_trace": "string (optional)",
      "recent_commands": ["cmd1", "cmd2"],
      "safe_read_paths": ["*.py", "*.js"],
      "working_directory": "/path/to/project"
    }

ENVIRONMENT VARIABLES:
    ANTHROPIC_API_KEY    Anthropic API key
    OPENAI_API_KEY       OpenAI API key
    GOOGLE_API_KEY       Google API key
    NEO_PROVIDER         LLM provider (openai|anthropic|google|ollama)
    NEO_MODEL            Model name
    NEO_API_KEY          Generic API key (provider-specific keys take precedence)

EXAMPLES:
    # Simple query
    echo '{"prompt": "Write a function to check if a number is prime"}' | neo

    # With context
    echo '{"prompt": "Fix this bug", "task_type": "bugfix", "context_files": [...]}' | neo

    # From file
    neo < input.json

    # Check learning progress
    neo --version

DOCUMENTATION:
    https://github.com/Parslee-ai/neo
"""
    print(help_text)

def _interpret_confidence(
    confidence: float,
    next_questions: list[str],
    plan: list,
    code_suggestions: list
) -> dict:
    """
    Interpret confidence score and provide actionable guidance.

    Helps users understand what the confidence score means and what to do next.
    """
    interpretation = {}

    # Determine action guidance based on confidence level
    if confidence >= 0.7:
        interpretation["action"] = "READY_TO_IMPLEMENT"
        interpretation["message"] = "High confidence - plan is well-structured and data-driven"
        interpretation["next_steps"] = [
            "Review the plan and code suggestions carefully",
            "Implement with standard monitoring and rollback procedures",
            "Consider the tradeoffs mentioned in code suggestions"
        ]
    elif confidence >= 0.4:
        interpretation["action"] = "PROCEED_WITH_CAUTION"
        interpretation["message"] = "Medium confidence - plan is sound but has some gaps or uncertainties"
        interpretation["next_steps"] = [
            "Review next_questions for areas needing clarification",
            "Consider gathering additional data before full implementation",
            "Implement incrementally with careful monitoring"
        ]
    else:  # confidence < 0.4
        interpretation["action"] = "GATHER_MORE_DATA"
        interpretation["message"] = "Low confidence - plan framework is provided but critical data is missing"
        interpretation["next_steps"] = []

        # Analyze next_questions to identify what's missing
        blocking_issues = []
        if next_questions:
            # Categorize the gaps
            has_missing_constraints = any("constraint" in q.lower() or "missing" in q.lower() for q in next_questions)
            has_missing_metrics = any("metric" in q.lower() or "quantify" in q.lower() for q in next_questions)
            has_missing_observability = any("observability" in q.lower() or "monitoring" in q.lower() for q in next_questions)

            if has_missing_constraints:
                blocking_issues.append("Missing or incomplete constraints")
            if has_missing_metrics:
                blocking_issues.append("Lacking quantitative metrics or baselines")
            if has_missing_observability:
                blocking_issues.append("No observability/monitoring strategy")

            # If we couldn't categorize, just note that there are gaps
            if not blocking_issues:
                blocking_issues.append("Data gaps identified in next_questions")

        interpretation["blocking_issues"] = blocking_issues if blocking_issues else ["Insufficient data to proceed with confidence"]

        # Provide specific next steps
        if plan:
            interpretation["next_steps"].append("Follow the plan to gather missing data and requirements")
        if next_questions:
            interpretation["next_steps"].append("Address the issues listed in next_questions")
        interpretation["next_steps"].append("Re-run Neo with complete data for higher confidence decision")

        # Important clarification
        interpretation["note"] = "The plan itself may be valuable - low confidence indicates missing input data, not plan quality"

    # Add confidence scale reference
    interpretation["confidence_scale"] = {
        "0.0-0.4": "Gather more data - critical information missing",
        "0.4-0.7": "Proceed with caution - some uncertainties remain",
        "0.7-1.0": "Ready to implement - high confidence in approach"
    }

    return interpretation

def _restore_from_backup(memory: 'PersistentReasoningMemory', backup: list) -> None:
    """Restore memory entries from backup (used on failure)."""
    memory.entries = backup


def _regenerate_entry_embeddings(
    memory: 'PersistentReasoningMemory',
    backup: list
) -> tuple[int, int, str]:
    """
    Regenerate embeddings for all entries in memory.

    Returns:
        Tuple of (success_count, failed_count, model_used)

    Raises:
        RuntimeError: If success rate < MIN_EMBEDDING_SUCCESS_RATE
    """
    total_entries = len(memory.entries)
    success_count = 0
    failed_count = 0
    model_used = "unknown"

    for i, entry in enumerate(memory.entries):
        # Build text from entry
        text = f"{entry.pattern}\n{entry.context}\n{entry.suggestion}"

        # Generate new embedding
        embedding = memory._embed_text(text)

        # Validate embedding
        if embedding is not None and len(embedding) in VALID_EMBEDDING_DIMENSIONS:
            entry.embedding = embedding
            entry.embedding_dim = len(embedding)

            # Extract model name from cache
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in memory.embedding_cache:
                _, cached_model, _ = memory.embedding_cache[cache_key]
                entry.embedding_model = cached_model
                if model_used == "unknown":
                    model_used = cached_model

            success_count += 1
        else:
            failed_count += 1
            if embedding is not None:
                logger.warning(
                    f"Entry {i} has invalid embedding dimension {len(embedding)}, "
                    f"expected one of {VALID_EMBEDDING_DIMENSIONS}"
                )
            else:
                logger.warning(f"Entry {i} failed to generate embedding: {entry.pattern[:50]}")

    # Validate success rate
    success_rate = success_count / total_entries if total_entries > 0 else 0.0

    if success_rate < MIN_EMBEDDING_SUCCESS_RATE:
        _restore_from_backup(memory, backup)
        raise RuntimeError(
            f"Embedding regeneration failed: only {success_count}/{total_entries} "
            f"({success_rate:.1%}) succeeded. Need at least {MIN_EMBEDDING_SUCCESS_RATE:.0%}. "
            f"Backup restored."
        )

    return success_count, failed_count, model_used


def regenerate_embeddings(
    codebase_root: Optional[str] = None,
    config: Optional['NeoConfig'] = None
) -> RegenerateStats:
    """
    Regenerate all embeddings with current configured model.

    Safe operation with automatic backup and failure detection.
    Use when switching embedding models or fixing mixed-model state.

    Args:
        codebase_root: Path to codebase (for local memory)
        config: NeoConfig instance (for embedding model selection)

    Returns:
        RegenerateStats dict with operation metrics

    Raises:
        RuntimeError: If backup fails, success rate < 80%, or save fails
    """
    start_time = time.time()

    # Initialize memory
    from neo.persistent_reasoning import PersistentReasoningMemory
    memory = PersistentReasoningMemory(codebase_root=codebase_root, config=config)

    total_entries = len(memory.entries)
    if total_entries == 0:
        logger.info("No entries to regenerate")
        return RegenerateStats(
            total=0,
            success=0,
            failed=0,
            success_rate=1.0,
            model="none",
            duration=0.0
        )

    logger.info(f"Regenerating embeddings for {total_entries} entries...")

    # Create backup (deep copy to prevent mutation)
    backup = copy.deepcopy(memory.entries)

    # Regenerate embeddings
    try:
        success_count, failed_count, model_used = _regenerate_entry_embeddings(memory, backup)
    except RuntimeError:
        # Backup already restored by _regenerate_entry_embeddings
        raise

    # Save updated entries
    try:
        memory.save()
    except (IOError, OSError, PermissionError) as e:
        _restore_from_backup(memory, backup)
        raise RuntimeError(f"Failed to save regenerated embeddings: {e}. Backup restored.") from e

    duration = time.time() - start_time
    success_rate = success_count / total_entries if total_entries > 0 else 0.0

    logger.info(
        f"Embedding regeneration complete: {success_count}/{total_entries} succeeded "
        f"in {duration:.1f}s using {model_used}"
    )

    return RegenerateStats(
        total=total_entries,
        success=success_count,
        failed=failed_count,
        success_rate=success_rate,
        model=model_used,
        duration=duration
    )


def handle_load_program(args):
    """Handle --load-program flag operations (The Operator)."""
    from neo.config import NeoConfig
    from neo.program_loader import ProgramLoader
    from neo.persistent_reasoning import PersistentReasoningMemory

    try:
        # Load config
        config = NeoConfig.load()
        codebase_root = args.cwd or os.getcwd()

        # Initialize memory
        memory = PersistentReasoningMemory(
            codebase_root=codebase_root,
            config=config
        )

        # Initialize loader
        loader = ProgramLoader(memory)

        # Parse column mapping if provided
        column_mapping = None
        if args.columns:
            try:
                column_mapping = json.loads(args.columns)
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in --columns: {e}", file=sys.stderr)
                sys.exit(1)

        # Load program
        result = loader.load_program(
            dataset_id=args.load_program,
            split=args.split,
            column_mapping=column_mapping,
            limit=args.limit,
            dry_run=args.dry_run,
            quiet=args.quiet
        )

        # Print Matrix-style output
        print()
        print(loader.format_result(result))

    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Install with: pip install datasets", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Failed to load program")
        print(f"Error: Unexpected failure: {e}", file=sys.stderr)
        sys.exit(1)


def handle_construct(args):
    """Handle construct subcommand operations."""
    from neo.construct import ConstructIndex
    from pathlib import Path

    # Determine construct root
    cwd = Path(args.cwd) if hasattr(args, 'cwd') and args.cwd else Path.cwd()

    # Try to find construct directory in repo
    construct_root = None
    if (cwd / 'construct').exists():
        construct_root = cwd / 'construct'
    elif (cwd.parent / 'construct').exists():
        construct_root = cwd.parent / 'construct'
    else:
        # Check if we're in the neo repo
        current = cwd
        while current != current.parent:
            if (current / 'construct').exists():
                construct_root = current / 'construct'
                break
            current = current.parent

    index = ConstructIndex(construct_root=construct_root)

    if args.construct_action == 'list':
        patterns = index.list_patterns(domain=args.domain)
        if not patterns:
            print("No patterns found.")
            if args.domain:
                print(f"(Domain filter: {args.domain})")
            return

        # Group by domain
        by_domain = {}
        for p in patterns:
            by_domain.setdefault(p.domain, []).append(p)

        for domain in sorted(by_domain.keys()):
            print(f"\n{domain}:")
            for p in by_domain[domain]:
                print(f"  {p.pattern_id:<40} {p.name}")

        print(f"\nTotal: {len(patterns)} patterns")

    elif args.construct_action == 'show':
        pattern = index.show_pattern(args.pattern_id)
        if not pattern:
            print(f"Error: Pattern '{args.pattern_id}' not found", file=sys.stderr)
            sys.exit(1)

        # Display pattern
        print(f"# Pattern: {pattern.name}")
        print(f"Author: {pattern.author}")
        print(f"Domain: {pattern.domain}")
        print(f"ID: {pattern.pattern_id}\n")
        print(f"## Intent\n{pattern.intent}\n")
        print(f"## Forces\n{pattern.forces}\n")
        print(f"## Solution\n{pattern.solution}\n")
        print(f"## Consequences\n{pattern.consequences}\n")
        if pattern.references:
            print(f"## References\n{pattern.references}\n")

    elif args.construct_action == 'search':
        results = index.search(args.query, top_k=args.top_k)
        if not results:
            print(f"No results found for: {args.query}")
            return

        print(f"Search results for: {args.query}\n")
        for i, (pattern, score) in enumerate(results, 1):
            print(f"{i}. {pattern.pattern_id} (score: {score:.3f})")
            print(f"   {pattern.name}")
            print(f"   {pattern.intent[:100]}...")
            print()

    elif args.construct_action == 'index':
        print("Building construct pattern index...")
        result = index.build_index(force_rebuild=args.force)

        if result['status'] == 'success':
            print(f"✓ Indexed {result['pattern_count']} patterns in {result['elapsed_seconds']:.2f}s")
            print(f"  Index: {result['index_path']}")
        elif result['status'] == 'skipped':
            print(f"Index is recent, skipping rebuild (use --force to rebuild)")
        else:
            print(f"✗ Index build failed: {result.get('reason', 'unknown error')}", file=sys.stderr)
            sys.exit(1)

    else:
        print("Error: No construct action specified", file=sys.stderr)
        print("Usage: neo construct {list|show|search|index}", file=sys.stderr)
        sys.exit(1)


def handle_config(args):
    """Handle --config flag operations."""
    from neo.config import NeoConfig

    VALID_PROVIDERS = ['openai', 'anthropic', 'google', 'azure', 'ollama', 'local']
    EXPOSED_FIELDS = ['provider', 'model', 'api_key', 'base_url']

    def mask_secret(value: str) -> str:
        """Mask API keys and secrets for display."""
        if not value or len(value) < 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"

    # Load current config
    config = NeoConfig.load()

    if args.config == 'list':
        # Show all exposed fields
        print("Current configuration:")
        for field in EXPOSED_FIELDS:
            value = getattr(config, field, None)
            if value is None:
                display_value = "(not set)"
            elif field == 'api_key':
                display_value = mask_secret(value)
            else:
                display_value = value
            print(f"  {field}: {display_value}")

    elif args.config == 'get':
        # Get single field
        if not args.config_key:
            print("Error: --config-key required for 'get' operation", file=sys.stderr)
            sys.exit(1)

        if args.config_key not in EXPOSED_FIELDS:
            print(f"Error: Invalid config key. Valid keys: {', '.join(EXPOSED_FIELDS)}", file=sys.stderr)
            sys.exit(1)

        value = getattr(config, args.config_key, None)
        if value is None:
            print("(not set)")
        elif args.config_key == 'api_key':
            print(mask_secret(value))
        else:
            print(value)

    elif args.config == 'set':
        # Set field value
        if not args.config_key or not args.config_value:
            print("Error: --config-key and --config-value required for 'set' operation", file=sys.stderr)
            sys.exit(1)

        if args.config_key not in EXPOSED_FIELDS:
            print(f"Error: Invalid config key. Valid keys: {', '.join(EXPOSED_FIELDS)}", file=sys.stderr)
            sys.exit(1)

        # Validate provider
        if args.config_key == 'provider' and args.config_value not in VALID_PROVIDERS:
            print(f"Error: Invalid provider. Valid providers: {', '.join(VALID_PROVIDERS)}", file=sys.stderr)
            sys.exit(1)

        # Set the value
        setattr(config, args.config_key, args.config_value)
        config.save()
        print(f"✓ Set {args.config_key} = {args.config_value if args.config_key != 'api_key' else mask_secret(args.config_value)}")

    elif args.config == 'reset':
        # Reset to defaults
        default_config = NeoConfig()
        default_config.save()
        print("✓ Configuration reset to defaults")


def parse_args():
    """Parse command-line arguments."""
    import argparse
    import sys

    # Detect if 'construct' subcommand is being used
    if len(sys.argv) > 1 and sys.argv[1] == 'construct':
        # Parse construct subcommand with proper sub-subparsers
        p = argparse.ArgumentParser(
            prog="neo construct",
            description="Manage design pattern library"
        )
        subparsers = p.add_subparsers(dest='action', help='Construct actions')

        # construct list
        list_p = subparsers.add_parser('list', help='List all patterns')
        list_p.add_argument('--domain', help='Filter by domain')

        # construct show
        show_p = subparsers.add_parser('show', help='Show a pattern')
        show_p.add_argument('pattern_id', help='Pattern ID (e.g., caching/cache-aside)')

        # construct search
        search_p = subparsers.add_parser('search', help='Search patterns')
        search_p.add_argument('query', help='Search query')
        search_p.add_argument('--top-k', type=int, default=5, help='Number of results')

        # construct index
        index_p = subparsers.add_parser('index', help='Build search index')
        index_p.add_argument('--force', action='store_true', help='Force rebuild')

        # Add global --cwd to all
        for sp in [list_p, show_p, search_p, index_p]:
            sp.add_argument('--cwd', metavar="PATH", help="Working directory override")

        # Remove 'construct' from argv for parsing
        sys.argv.pop(1)
        args = p.parse_args()
        # Restore for compatibility
        args.command = 'construct'
        args.construct_action = args.action
        return args

    # Default argument parser (for reasoning mode)
    p = argparse.ArgumentParser(
        prog="neo",
        description="Neo - Reasoning helper for coding tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    p.add_argument("prompt", nargs="?", help="Plain text prompt (or use stdin)")
    p.add_argument("--json", action="store_true", help="Print JSONL events and final JSON")
    p.add_argument("--output-schema", metavar="NAME_OR_PATH", help="Control final response shape")
    p.add_argument("--cwd", metavar="PATH", help="Working directory override")
    p.add_argument("--max-bytes", type=int, default=300_000, help="Hard cap for total context bytes")
    p.add_argument("--max-files", type=int, default=30, help="Soft cap for number of context files")
    p.add_argument("--include", action="append", default=[], help="Allowlist glob patterns (repeatable)")
    p.add_argument("--exclude", action="append", default=[], help="Blocklist glob patterns (repeatable)")
    p.add_argument("--exts", metavar="CSV", help="Restrict to file extensions (comma-separated)")
    p.add_argument("--diff-since", metavar="REV", help="Prioritize files changed since git rev or duration")
    p.add_argument("--no-git", action="store_true", help="Skip git-aware heuristics")
    p.add_argument("--no-scan", action="store_true", help="Skip directory scan; use only JSON-provided context")
    p.add_argument("--semantic", action="store_true", help="Use semantic search (requires .neo/index.json)")
    p.add_argument("--stdin-json", action="store_true", help="Force JSON input mode")
    p.add_argument("--stdin-text", action="store_true", help="Force text input mode")
    p.add_argument("--dry-run", action="store_true", help="Show what would be sent to model and exit")
    p.add_argument("--version", "-v", action="store_true", help="Show version and learning progress")
    p.add_argument("--regenerate-embeddings", action="store_true", help="Regenerate all embeddings with current model (safe, with automatic backup)")
    p.add_argument("--index", action="store_true", help="Build semantic index for current directory")
    p.add_argument("--config", choices=['list', 'get', 'set', 'reset'], help="Manage configuration")
    p.add_argument("--config-key", help="Config key (for get/set)")
    p.add_argument("--config-value", help="Value (for set)")
    p.add_argument("--load-program", metavar="DATASET_ID", help="Load training pack from HuggingFace (e.g., mbpp)")
    p.add_argument("--split", default="train", help="Dataset split (train/test/validation)")
    p.add_argument("--columns", metavar="JSON", help="Column mapping JSON (e.g., '{\"text\":\"pattern\"}')")
    p.add_argument("--limit", type=int, default=1000, help="Max samples to import (default: 1000)")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p.parse_args()


def detect_input_mode(args):
    """Detect whether input is JSON or plain text."""
    import io

    if args.stdin_json:
        return "json"
    if args.stdin_text:
        return "text"

    # Auto-detect from stdin
    if not sys.stdin.isatty():
        buf = sys.stdin.read()
        stripped = buf.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                json.loads(buf)
                sys.stdin = io.StringIO(buf)
                return "json"
            except json.JSONDecodeError:
                sys.stdin = io.StringIO(buf)
                return "text"
        else:
            sys.stdin = io.StringIO(buf)
            return "text"

    return "text"


def read_prompt_from_argv_or_stdin(args):
    """Read prompt from argv or stdin."""
    if args.prompt and args.prompt != "-":
        return args.prompt

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    print("Error: No prompt provided. Use: neo \"your prompt\" or pipe via stdin", file=sys.stderr)
    sys.exit(2)


def main():
    """Main entry point for stdin/stdout interface."""
    # Parse arguments
    args = parse_args()

    # Handle construct subcommand
    if args.command == 'construct':
        handle_construct(args)
        sys.exit(0)

    # Handle --version flag
    if args.version:
        codebase_root = args.cwd or os.getcwd()
        show_version(codebase_root)
        sys.exit(0)

    # Handle --config flag
    if args.config:
        handle_config(args)
        sys.exit(0)

    # Handle --load-program flag
    if args.load_program:
        handle_load_program(args)
        sys.exit(0)

    # Handle --regenerate-embeddings flag
    if args.regenerate_embeddings:
        from neo.config import NeoConfig
        try:
            config = NeoConfig.load()
            codebase_root = args.cwd or os.getcwd()
            result = regenerate_embeddings(codebase_root=codebase_root, config=config)
            print(f"✓ Regenerated embeddings for {result['success']}/{result['total']} entries")
            print(f"  Model: {result['model']}")
            print(f"  Duration: {result['duration']:.1f}s")
            if result['failed'] > 0:
                print(f"  ⚠ Warning: {result['failed']} entries failed to regenerate")
            sys.exit(0)
        except RuntimeError as e:
            print(f"✗ Regeneration failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Handle --index flag
    if args.index:
        from src.index.project_index import ProjectIndex

        codebase_root = args.cwd or os.getcwd()
        print(f"[Neo] Building semantic index for {codebase_root}...")

        index = ProjectIndex(codebase_root)

        # Determine file patterns based on detected languages
        patterns = ['**/*.py']  # Default to Python

        max_files = 100  # Configurable later

        try:
            index.build_index(patterns, max_files=max_files)
            status = index.status()
            print(f"[Neo] Built index: {status['total_chunks']} chunks from {status['total_files']} files")
            print(f"[Neo] Index stored in {codebase_root}/.neo/")
            print(f"[Neo] Use '--semantic' flag to enable semantic search")
            sys.exit(0)
        except Exception as e:
            print(f"[Neo] Failed to build index: {e}", file=sys.stderr)
            sys.exit(1)

    # Detect input mode
    input_mode = detect_input_mode(args)

    # Parse input based on mode
    if input_mode == "json":
        try:
            input_data = json.loads(sys.stdin.read())
            neo_input = NeoInput(
                prompt=input_data["prompt"],
                task_type=TaskType(input_data.get("task_type", "feature")),
                context_files=[
                    ContextFile(**cf) for cf in input_data.get("context_files", [])
                ],
                error_trace=input_data.get("error_trace"),
                recent_commands=input_data.get("recent_commands", []),
                safe_read_paths=input_data.get("safe_read_paths", []),
                working_directory=input_data.get("working_directory"),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            error_output = {"error": f"Invalid JSON input: {e}"}
            print(json.dumps(error_output, indent=2))
            sys.exit(1)
    else:
        # Plain text mode
        prompt = read_prompt_from_argv_or_stdin(args)
        working_dir = args.cwd or os.getcwd()

        neo_input = NeoInput(
            prompt=prompt,
            task_type=TaskType.FEATURE,
            context_files=[],
            working_directory=working_dir,
            safe_read_paths=[working_dir],
        )

        # Gather context from working directory unless --no-scan
        if not args.no_scan:
            from neo.context_gatherer import gather_context, gather_context_semantic, GatherConfig, ContextFile as GatheredFile

            exts = args.exts.split(',') if args.exts else None

            config = GatherConfig(
                root=working_dir,
                prompt=prompt,
                exts=exts,
                includes=args.include,
                excludes=args.exclude,
                max_bytes=args.max_bytes,
                max_files=args.max_files,
                diff_since=args.diff_since,
                use_git=not args.no_git,
            )

            # Use semantic search if --semantic flag is set
            if args.semantic:
                gathered = gather_context_semantic(config)
            else:
                gathered = gather_context(config)

            # Convert gathered files to ContextFile format
            neo_input.context_files = [
                ContextFile(
                    path=gf.path,
                    content=gf.content,
                    line_range=(gf.start, gf.end) if gf.start else None
                )
                for gf in gathered
            ]

            # Print summary to stderr
            total_bytes = sum(gf.bytes for gf in gathered)
            print(f"[Neo] Gathered {len(gathered)} files ({total_bytes:,} bytes)", file=sys.stderr)
            print(f"[Neo] Invoking LLM inference...", file=sys.stderr)

            if args.dry_run:
                print("\n=== DRY RUN: Context that would be sent ===\n", file=sys.stderr)
                for gf in gathered:
                    lines_info = f" (lines {gf.start}-{gf.end})" if gf.start else ""
                    print(f"  {gf.rel_path}{lines_info} - {gf.bytes} bytes (score: {gf.score:.2f})", file=sys.stderr)
                print(f"\nPrompt: {prompt[:200]}...\n", file=sys.stderr)
                sys.exit(0)

    # Initialize adapter from environment
    # NO STUBS OR FALLBACKS - require real configuration
    from neo.adapters import create_adapter
    from neo.config import NeoConfig

    try:
        # Load config to get API key
        config = NeoConfig.load()
        adapter = create_adapter(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key
        )
    except Exception as e:
        error_output = {
            "error": f"Failed to initialize LM adapter: {e}",
            "hint": "Set NEO_PROVIDER and NEO_MODEL in config.json or environment, or set provider-specific API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)"
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)

    # Create engine and process (with codebase root for per-codebase learning)
    try:
        engine = NeoEngine(
            lm_adapter=adapter,
            codebase_root=neo_input.working_directory,
            config=config
        )
        output = engine.process(neo_input)
    except TimeoutError as e:
        error_output = {
            "error": "RequestTimeout",
            "message": "LLM request exceeded timeout limit",
            "timeout_seconds": 300,
            "details": str(e),
            "suggestions": [
                "Try simplifying your prompt",
                "Break complex queries into smaller parts",
                "Check your network connection"
            ]
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)
    except ValueError as e:
        error_msg = str(e)
        error_output = {
            "error": "ValidationError",
            "message": error_msg,
            "suggestions": []
        }

        # Provide specific suggestions based on error type
        if "schema" in error_msg.lower() or "validation" in error_msg.lower():
            error_output["suggestions"] = [
                "Check that LLM output includes required fields",
                "Verify schema_version is set to '3'",
                "Review structured_parser.py for validation rules"
            ]
        elif "parse" in error_msg.lower():
            error_output["suggestions"] = [
                "LLM may have produced invalid JSON",
                "Try re-running the query",
                "Check lm_logger output for raw response"
            ]
        else:
            error_output["suggestions"] = [
                "Review the error message for specific details",
                "Check Neo's logs for more context"
            ]

        print(json.dumps(error_output, indent=2))
        sys.exit(1)
    except Exception as e:
        # Import httpx to check for timeout errors
        try:
            import httpx
            if isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout)):
                error_output = {
                    "error": "NetworkTimeout",
                    "message": f"Network request timed out: {str(e)}",
                    "timeout_seconds": 300,
                    "suggestions": [
                        "Check your internet connection",
                        "Verify API endpoint is accessible",
                        "Try again in a moment"
                    ]
                }
                print(json.dumps(error_output, indent=2))
                sys.exit(1)
        except ImportError:
            pass

        # Generic error handler
        error_output = {
            "error": "ProcessingError",
            "message": f"Unexpected error during processing: {str(e)}",
            "error_type": type(e).__name__,
            "suggestions": [
                "Check Neo's logs for detailed stack trace",
                "Verify input format is correct",
                "Report this issue if it persists: https://github.com/Parslee-ai/neo/issues"
            ]
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)

    # Serialize output
    try:
        output_dict = {
            "plan": [
                {
                    "description": step.description,
                    "rationale": step.rationale,
                    "dependencies": step.dependencies,
                }
                for step in output.plan
            ],
            "simulation_traces": [
                {
                    "input_data": trace.input_data,
                    "expected_output": trace.expected_output,
                    "reasoning_steps": trace.reasoning_steps,
                    "issues_found": trace.issues_found,
                }
                for trace in output.simulation_traces
            ],
            "code_suggestions": [
                {
                    "file_path": sugg.file_path,
                    "unified_diff": sugg.unified_diff,
                    "description": sugg.description,
                    "confidence": sugg.confidence,
                    "tradeoffs": sugg.tradeoffs,
                }
                for sugg in output.code_suggestions
            ],
            "static_checks": [
                {
                    "tool_name": check.tool_name,
                    "diagnostics": check.diagnostics,
                    "summary": check.summary,
                }
                for check in output.static_checks
            ],
            "next_questions": output.next_questions,
            "confidence": output.confidence,
            "notes": output.notes,
            "metadata": output.metadata,
        }

        # Add confidence interpretation for better UX
        confidence_interpretation = _interpret_confidence(
            output.confidence,
            output.next_questions,
            output.plan,
            output.code_suggestions
        )
        output_dict["confidence_interpretation"] = confidence_interpretation

        # Output based on mode
        if args.json:
            # JSON mode: print structured output
            print(json.dumps(output_dict, indent=2))
        else:
            # Human-readable text mode
            print("\n" + "="*80)
            print(f"CONFIDENCE: {output.confidence:.2f}")
            print("="*80)

            if output.notes:
                print(f"\n{output.notes}\n")

            print("\nPLAN:")
            for i, step in enumerate(output.plan, 1):
                print(f"\n{i}. {step.description}")
                print(f"   Rationale: {step.rationale}")
                if step.dependencies:
                    print(f"   Dependencies: {step.dependencies}")

            if output.simulation_traces:
                print("\n" + "-"*80)
                print("SIMULATIONS:")
                for i, trace in enumerate(output.simulation_traces, 1):
                    print(f"\nScenario {i}:")
                    print(f"  Input: {trace.input_data}")
                    print(f"  Expected: {trace.expected_output}")
                    if trace.issues_found:
                        print(f"  Issues: {', '.join(trace.issues_found)}")

            if output.code_suggestions:
                print("\n" + "-"*80)
                print("CODE SUGGESTIONS:")
                for i, sugg in enumerate(output.code_suggestions, 1):
                    print(f"\n{i}. {sugg.file_path} (confidence: {sugg.confidence:.2f})")
                    print(f"   {sugg.description}")
                    if sugg.unified_diff:
                        print("\n" + sugg.unified_diff)

            if output.next_questions:
                print("\n" + "-"*80)
                print("NEXT QUESTIONS:")
                for q in output.next_questions:
                    print(f"  • {q}")

            print("\n" + "="*80 + "\n")
    except Exception as e:
        error_output = {
            "error": "SerializationError",
            "message": f"Failed to serialize output: {str(e)}",
            "error_type": type(e).__name__,
            "suggestions": [
                "Output may contain non-serializable data",
                "Check Neo's internal data structures",
                "Report this issue: https://github.com/Parslee-ai/neo/issues"
            ]
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()