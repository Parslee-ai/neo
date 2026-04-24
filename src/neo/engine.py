"""
Neo reasoning engine.

Contains the NeoEngine class which orchestrates the MapCoder/CodeSim-style
multi-agent reasoning pipeline.

Split from cli.py for modularity.
"""

import ast
import json
import logging
import os
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from neo.models import (
    CodeSuggestion,
    ContextFile,
    LMAdapter,
    NeoInput,
    NeoOutput,
    PlanStep,
    SimulationTrace,
    StaticCheckResult,
    TaskType,
)

from neo.pattern_extraction import generate_prevention_warnings, get_library

if TYPE_CHECKING:
    from neo.config import NeoConfig  # noqa: F401
    from neo.exemplar_index import ExemplarIndex  # noqa: F401
    from neo.memory.store import FactStore  # noqa: F401
    from neo.persistent_reasoning import PersistentReasoningMemory  # noqa: F401

# Initialize logger
logger = logging.getLogger(__name__)


class NeoEngine:
    """Main reasoning engine for Neo."""

    # Time budgets by difficulty (seconds) - Phase 5
    # Rationale for 30/60/120s budgets:
    # - Based on benchmark percentiles (easy: p75=30s, medium: p90=60s, hard: p95=120s)
    # - Prevents easy problems from wasting time
    # - Allocates more resources to hard problems
    TIME_BUDGETS = {
        "easy": 30,    # Simple problems with N <= 100
        "medium": 60,  # Standard problems with N <= 10,000
        "hard": 120    # Complex problems with N > 10,000 or algorithmic keywords
    }

    # Constants for magic numbers (Phase 5)
    EARLY_EXIT_CONFIDENCE = 0.8  # Skip static checks if confidence above this
    STATIC_CHECK_BUFFER = 0.9    # Reserve 10% of budget for static checks

    def __init__(
        self,
        lm_adapter: LMAdapter,
        exemplar_index: Optional["ExemplarIndex"] = None,
        enable_persistent_memory: bool = True,  # Persistent learning enabled by default
        codebase_root: Optional[str] = None,  # Root directory of the codebase being analyzed
        config: Optional[Any] = None,  # NeoConfig instance
    ):
        self.lm = lm_adapter
        self.exemplar_index = exemplar_index
        self.context: Optional[NeoInput] = None
        self.enable_persistent_memory = enable_persistent_memory
        self.codebase_root = codebase_root

        # Load beat deck for personality templates (no LLM call)
        self.beat_deck = self._load_beat_deck()

        # Initialize persistent memory (per-codebase)
        # Supports two backends: "fact_store" (new) and "legacy" (PersistentReasoningMemory)
        self.persistent_memory = None
        self.fact_store: Optional["FactStore"] = None
        if enable_persistent_memory:
            memory_backend = "fact_store"
            if config and hasattr(config, "memory_backend"):
                memory_backend = config.memory_backend

            if memory_backend == "fact_store":
                try:
                    from neo.memory.store import FactStore
                    self.fact_store = FactStore(
                        codebase_root=codebase_root,
                        config=config,
                        lm_adapter=lm_adapter,
                    )
                    # Set persistent_memory for backward compat in methods that check it
                    self.persistent_memory = self.fact_store
                except ImportError:
                    logger.warning("FactStore not available, falling back to legacy memory")
                    memory_backend = "legacy"

            if memory_backend == "legacy":
                try:
                    from neo.persistent_reasoning import PersistentReasoningMemory
                    self.persistent_memory = PersistentReasoningMemory(
                        codebase_root=codebase_root,
                        config=config
                    )
                except ImportError:
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

        # Extract verifiable constraints from the prompt so the LM sees them
        # explicitly and we can check them post-hoc.
        extracted_constraints = self._extract_prompt_constraints(neo_input.prompt)
        if extracted_constraints:
            enriched_context["verifiable_constraints"] = [
                {"type": c.type.value, "description": c.description}
                for c in extracted_constraints
            ]

        # Single LLM call for all 3 phases (plan + simulation + code)
        # This is 59% faster than the old 3-call approach (22s vs 55s)
        plan, simulation_traces, code_suggestions = self._process_combined(enriched_context)
        self.last_simulation_traces = simulation_traces

        # Phase 5: Run static checks BEFORE deciding early-exit.
        # LM self-reported confidence alone is self-validation — we require an
        # objective signal (no error-severity diagnostics) to skip the rest of
        # the pipeline. Static checks are cheap enough to always run when the
        # time budget allows.
        elapsed = time.time() - start_time
        static_checks = []
        if elapsed < time_budget * self.STATIC_CHECK_BUFFER:
            static_checks = self._run_static_checks(code_suggestions, extracted_constraints)
        else:
            logger.info(f"Skipping static checks (at {elapsed/time_budget*100:.0f}% budget utilization)")

        # Early exit only when BOTH signals agree: high self-confidence AND no
        # error-severity diagnostics. An objective signal (static analysis
        # actually ran and is clean) is required — we do NOT early-exit when
        # static_checks is empty because no tools were available.
        if code_suggestions and self._simulation_consensus(simulation_traces):
            max_confidence = max((s.confidence for s in code_suggestions), default=0.0)
            has_errors = any(
                d.get("severity") == "error"
                for check in static_checks
                for d in check.diagnostics
            )
            static_ran = len(static_checks) > 0
            if (
                max_confidence > self.EARLY_EXIT_CONFIDENCE
                and static_ran
                and not has_errors
            ):
                logger.info(
                    f"Early exit: confidence={max_confidence:.2f}, "
                    f"static_checks={len(static_checks)} clean, "
                    f"simulations agree"
                )
                return self._finalize_output(
                    neo_input=neo_input,
                    plan=plan,
                    simulation_traces=simulation_traces,
                    code_suggestions=code_suggestions,
                    static_checks=static_checks,
                    next_questions=[],
                    confidence=max_confidence,
                    enriched_context=enriched_context,
                    start_time=start_time,
                    difficulty=difficulty,
                    time_budget=time_budget,
                    early_exit=True,
                    extra_metadata={
                        "max_confidence": max_confidence,
                        "static_checks_clean": True,
                    },
                )

        next_questions = self._generate_questions(
            plan, simulation_traces, code_suggestions, static_checks
        )
        confidence = self._calculate_confidence(
            plan, simulation_traces, code_suggestions, static_checks
        )

        return self._finalize_output(
            neo_input=neo_input,
            plan=plan,
            simulation_traces=simulation_traces,
            code_suggestions=code_suggestions,
            static_checks=static_checks,
            next_questions=next_questions,
            confidence=confidence,
            enriched_context=enriched_context,
            start_time=start_time,
            difficulty=difficulty,
            time_budget=time_budget,
            early_exit=False,
        )

    def _finalize_output(
        self,
        *,
        neo_input: NeoInput,
        plan: list[PlanStep],
        simulation_traces: list[SimulationTrace],
        code_suggestions: list[CodeSuggestion],
        static_checks: list[StaticCheckResult],
        next_questions: list[str],
        confidence: float,
        enriched_context: dict[str, Any],
        start_time: float,
        difficulty: str,
        time_budget: float,
        early_exit: bool,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> NeoOutput:
        """Persist reasoning, log metrics, and build the NeoOutput.

        Single exit point for both early-exit and full-pipeline paths so the
        save/log/telemetry sequence can't drift between them.
        """
        fact = None
        if self.persistent_memory:
            fact = self._store_reasoning(
                neo_input, plan, code_suggestions, confidence, enriched_context
            )
        if self.fact_store is not None:
            ids = self._build_suggestion_fact_ids(fact, code_suggestions)
            self.fact_store.save_session(code_suggestions, neo_input.prompt, ids)

        elapsed = time.time() - start_time
        self._log_metrics(difficulty, time_budget, elapsed, early_exit=early_exit)

        metadata: dict[str, Any] = {"early_exit": early_exit} if early_exit else {}
        if extra_metadata:
            metadata.update(extra_metadata)

        output = NeoOutput(
            plan=plan,
            simulation_traces=simulation_traces,
            code_suggestions=code_suggestions,
            static_checks=static_checks,
            next_questions=next_questions,
            confidence=confidence,
            notes=self._generate_notes(plan, simulation_traces, static_checks),
            metadata=metadata,
        )
        self._log_usage_telemetry(output, neo_input)
        return output

    @staticmethod
    def _simulation_consensus(traces: list[SimulationTrace]) -> bool:
        """LM-independent sanity check: require ≥2 simulation traces with
        matching expected_output OR no reported issues.

        Returns True if we have enough trace agreement to trust the output,
        False if we should fall through to the full verification pipeline.
        When traces are unavailable (empty), returns True to avoid penalizing
        task types that don't produce them — the confidence+static_checks
        gates still apply.
        """
        if not traces:
            return True
        clean = [t for t in traces if not t.issues_found]
        if len(clean) < 2:
            return False
        outputs = [(t.expected_output or "").strip() for t in clean]
        # At least two clean traces with matching expected_output, OR all
        # clean traces have no expected_output (non-algorithmic tasks).
        non_empty = [o for o in outputs if o]
        if not non_empty:
            return True
        return outputs.count(non_empty[0]) >= 2

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
        base_dir = (Path(working_dir) if working_dir else Path.cwd()).resolve()

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
        if self.fact_store is not None:
            # Use FactStore: build_context + format_context_for_prompt
            prompt_text = context.get("prompt", "")
            k = self._adaptive_k_selection(prompt_text, context)
            fact_context = self.fact_store.build_context(prompt_text, environment=context, k=k)
            formatted = self.fact_store.format_context_for_prompt(fact_context)
            if formatted:
                past_learnings = [formatted]
            else:
                # Fallback: FactStore found nothing similar enough. Inject a
                # small set of community-curated globals so the prompt always
                # carries some memory-derived context.
                fallback = self._community_fallback_learnings(prompt_text)
                if fallback:
                    past_learnings = [fallback]

            # Log retrieval for measurement (Fact-compatible attributes)
            self._log_pattern_retrieval(
                [
                    type("_Entry", (), {
                        "source_hash": f.id,
                        "algorithm_type": f.kind.value if f.kind else "unknown",
                        "confidence": f.metadata.confidence if f.metadata else 0.0,
                        "_score": 0.0,
                        "use_count": f.metadata.access_count if f.metadata else 0,
                    })()
                    for f in fact_context.valid_facts
                ],
                context,
                k,
            )

        elif self.persistent_memory:
            # Legacy path: use PersistentReasoningMemory
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
                    pitfalls_str = "\n".join(f"  - {p}" for p in entry.common_pitfalls[:3])
                    learning += f"Known Pitfalls:\n{pitfalls_str}\n"
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

        prompt_formatters = {
            TaskType.ALGORITHM: self._format_algorithm_simulation_prompt,
            TaskType.REFACTOR: self._format_refactor_simulation_prompt,
            TaskType.BUGFIX: self._format_bugfix_simulation_prompt,
        }
        formatter = prompt_formatters.get(task_type, self._format_generic_simulation_prompt)
        return self._run_simulation(plan, context, formatter)

    def _run_simulation(
        self,
        plan: list[PlanStep],
        context: dict[str, Any],
        format_prompt,
    ) -> list[SimulationTrace]:
        """Run a simulation with the given prompt formatter."""
        messages = [
            {
                "role": "system",
                "content": self._get_simulation_system_prompt(),
            },
            {
                "role": "user",
                "content": format_prompt(plan, context),
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
        if self.fact_store is not None:
            prompt_text = context.get("prompt", "")
            k = self._adaptive_k_selection(prompt_text, context)
            fact_context = self.fact_store.build_context(prompt_text, environment=context, k=k)
            formatted = self.fact_store.format_context_for_prompt(fact_context)
            if formatted:
                past_learnings = [formatted]
        elif self.persistent_memory:
            # Legacy path: PersistentReasoningMemory
            k = self._adaptive_k_selection(context.get("prompt", ""), context)
            relevant = self.persistent_memory.retrieve_relevant(context, k=k)

            # Log pattern retrieval for measurement
            self._log_pattern_retrieval(relevant, context, k)

            past_learnings = []
            for e in relevant:
                learning = f"Pattern: {e.pattern}\nContext: {e.context}\nSuggestion: {e.suggestion}\n"
                # Phase 3: Surface known pitfalls (failure learning)
                if e.common_pitfalls:
                    pitfalls_str = "\n".join(f"  - {p}" for p in e.common_pitfalls[:3])
                    learning += f"Known Pitfalls:\n{pitfalls_str}\n"
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

        # Prevention warnings from learned patterns
        prevention_str = ""
        library = get_library()
        prevention_str = generate_prevention_warnings(
            context.get("prompt", ""),
            None,
            library
        )

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

TASK: {context_str}{past_learnings_str}{prevention_str}

RULES:
- Start with <<<NEO:SCHEMA=v3:KIND=plan>>> immediately
- NO text before, between, or after blocks
- id must be "ps_1", "ps_2" (not "p1")
- dependencies must be integers [0,1] (not strings)
- input_data and expected_output must be STRINGS (not JSON objects)"""

    def _run_static_checks(
        self,
        suggestions: list[CodeSuggestion],
        constraints: Optional[list] = None,
    ) -> list[StaticCheckResult]:
        """Run static analysis tools in read-only mode.

        Constraints are passed through explicitly (no instance state) so
        callers can compose the check without ordering-dependent setup.
        """
        from neo.static_analysis import run_static_checks
        from neo.config import NeoConfig

        config = NeoConfig.load()

        results = run_static_checks(
            suggestions,
            enable_ruff=config.enable_ruff,
            enable_pyright=config.enable_pyright,
            enable_mypy=config.enable_mypy,
            enable_eslint=config.enable_eslint,
        )

        # Phase 5.5: Static constraint verification (no code execution).
        if constraints:
            constraint_result = self._check_constraints_static(suggestions, constraints)
            if constraint_result is not None:
                results.append(constraint_result)

        return results

    def _community_fallback_learnings(self, prompt_text: str) -> str:
        """Return formatted community facts when the primary store misses.

        Reads ~/.neo/community_facts_cache.json directly and picks facts whose
        subject contains any token from the prompt. Falls back to the first 2
        facts if no token match.
        """
        cache = Path.home() / ".neo" / "community_facts_cache.json"
        if not cache.exists():
            return ""
        try:
            data = json.loads(cache.read_text())
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Community cache unreadable: {e}")
            return ""

        facts = data.get("data", {}).get("facts", [])
        if not facts:
            return ""

        tokens = {t for t in prompt_text.lower().split() if len(t) > 3}
        scored: list[tuple[int, dict]] = []
        for f in facts:
            text = (f.get("subject", "") + " " + f.get("body", "")).lower()
            hits = sum(1 for t in tokens if t in text)
            scored.append((hits, f))
        scored.sort(key=lambda x: x[0], reverse=True)

        picked = [f for h, f in scored if h > 0][:2] or [f for _, f in scored[:2]]
        if not picked:
            return ""

        lines = ["## Community Patterns (fallback)"]
        for f in picked:
            subject = f.get("subject", "")
            body = (f.get("body", "") or "")[:200]
            lines.append(f"- **{subject}**: {body}")
        return "\n".join(lines)

    def _extract_prompt_constraints(self, prompt: str) -> list:
        """Extract verifiable Constraint objects from the prompt (pattern-based).

        Returns typed Constraint objects so callers pass them explicitly to
        downstream checks rather than relying on instance state.
        """
        try:
            from neo.constraint_verification import ConstraintVerifier
        except ImportError:
            return []

        verifier = ConstraintVerifier()
        try:
            return verifier.extract_constraints(prompt)
        except Exception as e:
            logger.debug(f"Constraint extraction failed: {e}")
            return []

    @staticmethod
    def _strip_comments_and_strings(code: str) -> str:
        """Best-effort strip of Python comments and string/bytes literals.

        Used before substring-matching constraint markers so the word
        "sorted" inside a docstring or comment doesn't suppress a real
        warning. Falls back to the original code if tokenization fails
        (e.g. non-Python or syntactically broken input).
        """
        import io
        import tokenize

        try:
            out = []
            tokens = tokenize.generate_tokens(io.StringIO(code).readline)
            for tok in tokens:
                if tok.type in (tokenize.COMMENT, tokenize.STRING):
                    continue
                out.append(tok.string)
            return " ".join(out)
        except Exception:
            return code

    def _check_constraints_static(
        self,
        suggestions: list[CodeSuggestion],
        constraints: list,
    ) -> Optional[StaticCheckResult]:
        """Flag generated code that appears to ignore declared constraints.

        Pure textual checks with comment/string stripping. Does NOT execute
        code. Absence of a handler marker is a warning, not an error.
        """
        if not constraints or not suggestions:
            return None

        from neo.constraint_verification import CONSTRAINT_CODE_MARKERS

        diagnostics: list[dict[str, Any]] = []
        for sug in suggestions:
            raw = sug.code_block or sug.unified_diff or ""
            code = self._strip_comments_and_strings(raw).lower()
            for c in constraints:
                hints = CONSTRAINT_CODE_MARKERS.get(c.type)
                if not hints:
                    continue
                if not any(h.lower() in code for h in hints):
                    diagnostics.append({
                        "severity": "warning",
                        "message": (
                            f"Prompt declares constraint '{c.description}' but "
                            f"generated code shows no obvious handler "
                            f"(expected one of: {', '.join(hints)})"
                        ),
                        "constraint_type": c.type.value,
                        "file_path": sug.file_path,
                    })

        if not diagnostics:
            return None

        return StaticCheckResult(
            tool_name="constraint_verifier",
            diagnostics=diagnostics,
            summary=f"{len(diagnostics)} constraint(s) may not be handled",
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

    @staticmethod
    def _build_suggestion_fact_ids(
        fact, code_suggestions: list[CodeSuggestion]
    ) -> dict[str, str]:
        """Build file_path -> fact_id mapping for outcome linkage."""
        if fact is None:
            logger.warning("_build_suggestion_fact_ids: fact is None, no linkage possible")
            return {}
        ids: dict[str, str] = {}
        for s in code_suggestions:
            fp = getattr(s, "file_path", "")
            if fp and fp not in ("/", "N/A"):
                ids[fp] = fact.id
        logger.debug(f"_build_suggestion_fact_ids: linked {len(ids)} suggestions to fact {fact.id}")
        return ids

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
            logger.warning("_store_reasoning: no persistent_memory, returning None")
            return

        # Extract pattern from task type and prompt
        task_type_str = neo_input.task_type.value if neo_input.task_type else "unknown"
        pattern = f"{task_type_str}: {neo_input.prompt[:50]}"

        # Build context description
        context_desc = []
        if neo_input.error_trace:
            context_desc.append("has error trace")
        if neo_input.context_files:
            context_desc.append(f"{len(neo_input.context_files)} files")
        context_str = ", ".join(context_desc) if context_desc else "general task"

        # Build reasoning from plan
        reasoning = " -> ".join([step.description for step in plan[:3]])
        if len(plan) > 3:
            reasoning += f" ... ({len(plan)} steps total)"

        # Build suggestion summary
        suggestion = suggestions[0].description if suggestions else "No suggestions"
        if len(suggestions) > 1:
            suggestion += f" (+{len(suggestions)-1} more)"

        # Extract code skeleton from first suggestion (Kite-inspired AST approach)
        code_skeleton = ""
        if suggestions:
            code_source = suggestions[0].code_block or suggestions[0].unified_diff
            if code_source:
                code_skeleton = self._extract_code_skeleton(code_source)

        # Extract pitfalls from simulation traces
        pitfalls = []
        if hasattr(self, 'last_simulation_traces') and self.last_simulation_traces:
            for trace in self.last_simulation_traces:
                if hasattr(trace, 'issues_found') and isinstance(trace.issues_found, list):
                    pitfalls.extend(trace.issues_found)
                if hasattr(trace, 'errors') and isinstance(trace.errors, list):
                    pitfalls.extend(trace.errors)

        # Clean up temporary simulation traces to prevent memory leak
        sim_traces = getattr(self, 'last_simulation_traces', None)
        if hasattr(self, 'last_simulation_traces'):
            del self.last_simulation_traces

        # Route to FactStore or legacy memory
        if self.fact_store is not None:
            # Map task_type to FactKind
            from neo.memory.models import FactKind
            kind_map = {
                "algorithm": FactKind.PATTERN,
                "refactor": FactKind.ARCHITECTURE,
                "bugfix": FactKind.FAILURE,
                "feature": FactKind.DECISION,
                "explanation": FactKind.PATTERN,
            }
            fact_kind = kind_map.get(task_type_str, FactKind.PATTERN)

            # Combine reasoning + suggestion + pitfalls into body
            body_parts = [f"Reasoning: {reasoning}"]
            if suggestion:
                body_parts.append(f"Suggestion: {suggestion}")
            if code_skeleton:
                body_parts.append(f"Code skeleton: {code_skeleton}")
            if pitfalls:
                body_parts.append("Pitfalls: " + "; ".join(pitfalls[:5]))

            fact = self.fact_store.add_fact(
                subject=pattern,
                body="\n".join(body_parts),
                kind=fact_kind,
                confidence=confidence,
                source_prompt=neo_input.prompt[:200],
                tags=[task_type_str],
            )
            logger.info(f"_store_reasoning: created fact id={fact.id} subject={pattern[:50]}")
            return fact
        else:
            # Legacy path
            test_patterns = []
            if sim_traces:
                for trace in sim_traces:
                    if hasattr(trace, 'input_data') and trace.input_data is not None:
                        input_str = str(trace.input_data)[:100]
                        test_patterns.append(f"Input: {input_str}")
                    if hasattr(trace, 'test_case') and trace.test_case:
                        tc_str = str(trace.test_case)[:100]
                        test_patterns.append(tc_str)

            self.persistent_memory.add_reasoning(
                pattern=pattern,
                context=context_str,
                reasoning=reasoning,
                suggestion=suggestion,
                confidence=confidence,
                source_context=context,
                code_skeleton=code_skeleton,
                common_pitfalls=pitfalls[:5],
                test_patterns=test_patterns[:3],
            )
            return None

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
        1. Parse numeric constraints from prompt (N <= value) - HIGHEST PRIORITY (objective)
        2. Check for algorithmic keywords - MEDIUM PRIORITY (subjective but strong)
        3. Check for explicit difficulty markers - LOWEST PRIORITY (subjective)
        4. Return conservative estimate

        Returns:
            "easy", "medium", or "hard"

        Design decisions:
        - Why constraints first? Objective signal (N <= 100 vs N <= 1000000)
        - Why algorithmic keywords second? Subjective but strong indicator
        - Conservative estimate: Default to "medium" when uncertain
        """
        # Validate input
        if not neo_input.prompt:
            raise ValueError("Empty prompt - cannot estimate difficulty")

        prompt = neo_input.prompt.lower()

        # PRIORITY 1: Numeric constraints (HIGHEST - objective signal)
        import re
        # Match formats: N <= 100, N <= 100, N <= 10^5, N <= 10^5, N <= 1e6, N <= 1e6
        constraints = re.findall(
            r'n\s*(?:\u2264|<=)\s*(?:10\^(\d+)|(\d+)e(\d+)|(\d+))',
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

        Design decision: Log to local file only for privacy.
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

            # Telemetry is logged locally via lm_logger only.
            # Remote telemetry endpoint removed (security: env var allowed
            # exfiltration of usage data to arbitrary URLs).

        except Exception as e:
            # Silent failure - telemetry should never crash the main process
            logger.debug(f"Telemetry logging failed: {e}")

    def _adaptive_k_selection(self, prompt: str, context: dict[str, Any]) -> int:
        """
        Dynamically select k (number of patterns to retrieve) based on context.

        Uses heuristics (not ML) to optimize retrieval:
        - Broad/vague prompts -> more patterns (exploration)
        - Specific prompts -> fewer patterns (precision)
        - Error traces present -> focused retrieval
        - Large codebases -> more context needed

        Design decision: Simple heuristics before ML - measure if needed
        """
        # Check if adaptive k is enabled via env var
        if os.getenv("NEO_ADAPTIVE_K", "true").lower() != "true":
            return 3  # Default fallback

        prompt_tokens = len(prompt.split())
        task_type = context.get("task_type")
        has_error_trace = bool(context.get("error_trace"))
        context_files = len(context.get("files", []))

        # Heuristic 1 (highest priority): Specific bugfix with error trace -> laser focus
        if has_error_trace and task_type == TaskType.BUGFIX:
            return 1  # High precision, pattern should be very relevant

        # Heuristic 2: Large codebase -> comprehensive scan
        # (Check before vague prompt to avoid over-exploring large repos)
        if context_files > 20:
            return 5  # More files = need more context

        # Heuristic 3: Complex prompt -> more patterns
        if prompt_tokens > 50:
            return 5  # Detailed query suggests complex problem

        # Heuristic 4: Vague prompt -> exploration mode
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
3. Question assumptions; identify what to validate - deja vu moments reveal hidden patterns.
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
As you trace through scenarios, notice patterns and question assumptions like deja vu.

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
6. Call out risks, failure modes, and observability - "I've seen this before. Deja vu."
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
- unified_diff: max 5000 characters. REQUIRED non-empty whenever file_path is a real path. Empty "" ONLY when file_path="/" or "N/A" (pure review/analysis with no code change).
- code_block: complete executable code for the target file. REQUIRED non-empty whenever file_path is a real path — the feedback loop depends on it. Empty "" ONLY for pure reviews.
- description: max 1000 characters (be detailed for review findings!)
- confidence: float 0.0 to 1.0
- tradeoffs: array of strings, max 500 chars each
- Always include schema_version: "3"
- For review/analysis tasks without code changes: use file_path="/", unified_diff="", code_block=""
- For code changes: ALWAYS include BOTH unified_diff AND code_block. Never emit an empty diff for a real file path — it breaks Neo's learning.

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
            if self.fact_store is not None:
                from neo.memory.models import FactKind
                self.fact_store.add_fact(
                    subject=f"parse_failure:{result.error_code}",
                    body=f"Parse failed: {result.error_message}\n"
                         f"Raw: {result.raw_block[:200] if result.raw_block else 'N/A'}",
                    kind=FactKind.FAILURE,
                    confidence=0.9,
                    tags=["parse_failure", result.error_code or "unknown"],
                )
            elif self.persistent_memory:
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
            file_path = item.get("file_path", "")
            unified_diff = item.get("unified_diff", "")
            code_block = item.get("code_block", "")

            # Synthesize a diff when the LM omitted one for a real file path.
            # Without this, outcome detection collapses to UNVERIFIED and
            # success_count never grows — the learning loop stays dead.
            if not unified_diff and code_block and file_path and file_path not in ("/", "N/A"):
                synthesized = self._synthesize_unified_diff(file_path, code_block)
                if synthesized:
                    unified_diff = synthesized

            suggestions.append(CodeSuggestion(
                file_path=file_path,
                unified_diff=unified_diff,
                description=item.get("description", ""),
                confidence=item.get("confidence", 0.5),
                tradeoffs=item.get("tradeoffs", []),
                code_block=code_block
            ))

        return suggestions

    def _synthesize_unified_diff(self, file_path: str, code_block: str) -> str:
        """Build a unified diff from code_block vs. current on-disk content.

        Returns empty string if resolution or diffing fails (caller falls back
        to an empty diff, which just degrades outcome detection to UNVERIFIED).
        """
        import difflib

        try:
            p = Path(file_path)
            if not p.is_absolute() and self.codebase_root:
                p = Path(self.codebase_root) / file_path.lstrip("/")

            original = ""
            if p.exists() and p.is_file():
                try:
                    original = p.read_text()
                except (OSError, UnicodeDecodeError):
                    return ""

            rel_label = file_path.lstrip("/") or "file"
            from_label = f"a/{rel_label}" if original else "/dev/null"
            to_label = f"b/{rel_label}"

            original_lines = original.splitlines(keepends=True)
            new_lines = code_block.splitlines(keepends=True)
            if original_lines and not original_lines[-1].endswith("\n"):
                original_lines[-1] += "\n"
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            diff = "".join(difflib.unified_diff(
                original_lines, new_lines,
                fromfile=from_label, tofile=to_label, n=3,
            ))
            return diff[:5000]  # match schema cap
        except Exception as e:
            logger.debug(f"Diff synthesis failed for {file_path}: {e}")
            return ""
