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
import re
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from neo.models import (
    AppliedAction,
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
from neo.operating_mode import (
    ExecutionAdapter,
    ModeValidationError,
    OperatingMode,
    validate_agent_authority,
)

from neo.pattern_extraction import generate_prevention_warnings, get_library

if TYPE_CHECKING:
    from neo.config import NeoConfig  # noqa: F401
    from neo.exemplar_index import ExemplarIndex  # noqa: F401
    from neo.memory.store import FactStore  # noqa: F401
    from neo.persistent_reasoning import PersistentReasoningMemory  # noqa: F401
    from neo.reasoning_effort import MemorySignal  # noqa: F401

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
        execution_adapter: Optional[ExecutionAdapter] = None,
    ):
        self.lm = lm_adapter
        self.exemplar_index = exemplar_index
        self.context: Optional[NeoInput] = None
        self.enable_persistent_memory = enable_persistent_memory
        self.codebase_root = codebase_root
        self.config = config  # NeoConfig | None (kept for downstream lookups like reasoning_effort_cap)
        self.execution_adapter = execution_adapter

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

        # Per-call action log for loop-detection watchdog (G3).
        # Bounded ring; populated by _log_action; consumed by the
        # overseer check function spawned in process().
        self.action_log: deque = deque(maxlen=64)

        # Evidence ledger is separate from durable facts. It exists even when
        # persistent memory is disabled so every coding session can be traced.
        from neo.memory.episodes import LearningEpisodeStore
        if self.fact_store is not None:
            episode_project_id = self.fact_store.project_id or "unscoped"
        else:
            from neo.memory.scope import detect_org_and_project
            _, episode_project_id = detect_org_and_project(codebase_root)
        self.episode_store = LearningEpisodeStore(episode_project_id or "unscoped")
        self.current_learning_episode = None

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
        self._validate_operating_mode(neo_input)
        from neo.execution_context import resolve_execution_context
        self.resolved_execution_context = resolve_execution_context(neo_input)
        self.context = neo_input
        start_time = time.time()
        self.action_log.clear()
        self.last_applied_actions: list[AppliedAction] = []
        self.current_learning_episode = self._begin_learning_episode(neo_input)
        from neo.memory.metrics import record as record_memory_metric
        record_memory_metric(
            "execution_context_resolved",
            episode_id=self.current_learning_episode.episode_id,
            session_id=self.current_learning_episode.session_id,
            task_id=self.current_learning_episode.task_id,
            goal_origin=self.resolved_execution_context.goal.origin,
            goal_confidence=self.resolved_execution_context.goal.confidence,
            intent_origin=self.resolved_execution_context.intent.origin,
            intent_confidence=self.resolved_execution_context.intent.confidence,
            role=self.resolved_execution_context.role.value,
            iteration=self.resolved_execution_context.trajectory.iteration,
        )

        # Spawn the loop-detection watchdog for this run (G3 wire-up).
        # The overseer ticks on a deterministic schedule and emits
        # ``overseer_tick`` events to metrics.jsonl; it cancels nothing
        # in Neo's current single-shot engine but provides telemetry for
        # iterative paths (repair_loop, future multi-pass orchestrators)
        # and for callers wrapping NeoEngine in their own loops.
        from neo.overseer import StructuredOverseer
        overseer = StructuredOverseer(
            check_fn=self._overseer_loop_check,
            default_delay=10.0,
            min_delay=1.0,
        )
        overseer.start()
        self._log_action("process.start", neo_input.prompt[:60])

        try:
            return self._process_inner(neo_input, start_time)
        except Exception as exc:
            episode = self.current_learning_episode
            if episode is not None:
                episode.completed_at = time.time()
                episode.final_outcome = "engine_error"
                episode.outcome_details = {"error_type": type(exc).__name__}
                try:
                    self.episode_store.save(episode)
                except Exception as save_exc:
                    logger.debug("learning episode error-save failed: %s", save_exc)
            raise
        finally:
            overseer.stop()
            self._log_action("process.end", "")

    def _begin_learning_episode(self, neo_input: NeoInput):
        """Create the in-memory evidence record for one request."""
        from neo.agent_context import discover as discover_agent_docs
        from neo.memory.episodes import (
            ContextSelection,
            LearningEpisode,
            content_hash,
            redact_sensitive_text,
            repository_state,
        )

        revision, dirty = repository_state(self.codebase_root)
        context_selection = [
            ContextSelection(
                path=f.path,
                content_sha256=content_hash(f.content or ""),
                line_range=f.line_range,
                kind="repository_file",
            )
            for f in neo_input.context_files
        ]
        for doc in discover_agent_docs(self.codebase_root):
            context_selection.append(ContextSelection(
                path=doc.path,
                content_sha256=content_hash(doc.content),
                kind="project_instruction",
            ))

        provider = str(getattr(self.lm, "provider", "") or "")
        model = str(getattr(self.lm, "model", "") or "")
        if not model:
            try:
                model = str(self.lm.name())
            except Exception:
                model = ""

        return LearningEpisode(
            objective=redact_sensitive_text(neo_input.prompt),
            project_id=self.episode_store.project_id,
            repository_root=self.codebase_root or "",
            repository_revision=revision,
            repository_dirty=dirty,
            context_selection=context_selection,
            provider=provider,
            model=model,
            operating_mode=neo_input.operating_mode.value,
            authority=(neo_input.authority.public_summary() if neo_input.authority else {}),
            execution_context=self._safe_execution_context_dict(),
        )

    def _safe_execution_context_dict(self) -> dict[str, Any]:
        """Return a bounded, secret-redacted envelope safe for local evidence."""
        from neo.memory.episodes import redact_sensitive_text

        def clean(value, depth=0, key=""):
            if depth > 5:
                return "[TRUNCATED]"
            sensitive_payload_key = any(
                token in key.lower()
                for token in ("content", "source", "code", "diff", "patch")
            )
            if sensitive_payload_key and value not in (None, ""):
                serialized = json.dumps(value, sort_keys=True, default=str)
                return {
                    "sha256": content_hash(serialized),
                    "size": len(serialized),
                }
            if key == "current_state" and isinstance(value, dict):
                return {
                    str(item_key)[:100]: clean(item, depth + 1, str(item_key))
                    for item_key, item in list(value.items())[:50]
                }
            if isinstance(value, str):
                return redact_sensitive_text(value)[:2000]
            if isinstance(value, dict):
                return {
                    str(item_key)[:100]: clean(item, depth + 1, str(item_key))
                    for item_key, item in list(value.items())[:50]
                }
            if isinstance(value, list):
                return [clean(item, depth + 1, key) for item in value[:50]]
            if isinstance(value, (int, float, bool)) or value is None:
                return value
            return redact_sensitive_text(str(value))[:500]

        from neo.memory.episodes import content_hash
        return clean(self.resolved_execution_context.to_dict())

    def _validate_operating_mode(self, neo_input: NeoInput) -> None:
        """Fail closed before inference for invalid verification or agent requests."""
        if neo_input.operating_mode is OperatingMode.VERIFY:
            if not neo_input.proposed_changes:
                raise ModeValidationError(
                    "verify mode requires at least one caller-provided proposed change"
                )
            if any(
                not change.file_path or not (change.unified_diff or change.code_block)
                for change in neo_input.proposed_changes
            ):
                raise ModeValidationError(
                    "each proposed change requires file_path and unified_diff or code_block"
                )
        if neo_input.operating_mode is OperatingMode.AGENT:
            validate_agent_authority(
                neo_input.authority,
                has_executor=self.execution_adapter is not None,
            )

    def _capture_retrieval_context(self, fact_context, *, included: bool) -> None:
        """Record retrieved fact IDs and scores without changing fact authority."""
        episode = self.current_learning_episode
        if episode is None:
            return
        from neo.memory.episodes import RetrievedFactEvidence

        existing = {item.fact_id: item for item in episode.retrieved_facts}
        for fact in fact_context.valid_facts:
            score = fact_context.retrieval_scores.get(fact.id)
            item = existing.get(fact.id)
            if item is None:
                item = RetrievedFactEvidence(fact_id=fact.id, score=score)
                episode.retrieved_facts.append(item)
                existing[fact.id] = item
            elif score is not None:
                item.score = score
            item.included_in_context = item.included_in_context or included

    def _capture_detectable_fact_use(self, plan, simulations, suggestions) -> None:
        """Mark facts only when their stable citation survives into reasoning output."""
        episode = self.current_learning_episode
        if episode is None or not episode.retrieved_facts:
            return
        artifacts = [
            *(step.rationale for step in plan),
            *(step.description for step in plan),
            *(reason for trace in simulations for reason in trace.reasoning_steps),
            *(suggestion.description for suggestion in suggestions),
        ]
        text = "\n".join(str(item) for item in artifacts if item)
        for evidence in episode.retrieved_facts:
            evidence.used_in_reasoning = (
                evidence.included_in_context
                and f"[fact:{evidence.fact_id}]" in text
            )

    def _log_action(self, action: str, signature: str) -> None:
        """Append an action to the loop-detection log.

        Cheap (deque append, no I/O). Called at engine phase boundaries
        to give the overseer something to look at. Signature is a short
        string the watchdog compares for equality — typically the first
        chars of a prompt, a fact id, a phase name.
        """
        self.action_log.append((time.time(), action, signature[:64]))

    def _overseer_loop_check(self):
        """Check function for StructuredOverseer (G3).

        Inspects the last few entries in self.action_log. If the same
        (action, signature) tuple repeats LOOP_THRESHOLD times in a row,
        signals is_looping + force_cancel. Currently Neo's engine doesn't
        run iterative LLM cycles, so this is mostly telemetry for the
        repair_loop and future multi-pass paths — but the deterministic
        check itself fires on every tick.
        """
        from neo.overseer import OverseerCheck
        LOOP_THRESHOLD = 5
        recent = list(self.action_log)[-LOOP_THRESHOLD:]
        if len(recent) < LOOP_THRESHOLD:
            return OverseerCheck(making_progress=True, next_check_delay=10.0)
        keys = [(a, sig) for _, a, sig in recent]
        is_looping = len(set(keys)) == 1
        return OverseerCheck(
            making_progress=not is_looping,
            is_looping=is_looping,
            needs_notification=(f"Loop on {keys[0]}" if is_looping else None),
            force_cancel_agent=is_looping,
            next_check_delay=5.0 if is_looping else 10.0,
        )

    def _process_verification_only(
        self, neo_input: NeoInput, start_time: float
    ) -> NeoOutput:
        """Verify caller-provided changes deterministically without an LM call."""
        suggestions = [
            CodeSuggestion(
                file_path=change.file_path,
                unified_diff=change.unified_diff,
                code_block=change.code_block,
                description=change.description,
                confidence=0.0,
            )
            for change in neo_input.proposed_changes
        ]
        constraints = self._extract_input_constraints(neo_input)
        static_checks = self._run_static_checks(suggestions, constraints)
        statuses = [self._static_check_status(check) for check in static_checks]
        confidence = (
            0.0 if "failed" in statuses
            else 1.0 if statuses and all(status == "passed" for status in statuses)
            else 0.5
        )
        if self.current_learning_episode is not None:
            self.current_learning_episode.applied_actions.extend({
                "suggestion_id": suggestion.suggestion_id,
                "file_path": suggestion.file_path,
                "status": "provided_for_verification",
            } for suggestion in suggestions)
        plan = [PlanStep(
            description="Verify caller-provided changes",
            rationale="VERIFY mode performs deterministic checks without generation",
            verifier_checks=[check.tool_name for check in static_checks],
        )]
        self.last_reasoning_mode = "verification_only"
        self.last_reasoning_reason = "caller selected verify mode"
        return self._finalize_output(
            neo_input=neo_input,
            plan=plan,
            simulation_traces=[],
            code_suggestions=suggestions,
            static_checks=static_checks,
            next_questions=[],
            confidence=confidence,
            enriched_context={},
            start_time=start_time,
            difficulty="verification",
            time_budget=0.0,
            early_exit=False,
            extra_metadata={"verification_only": True, "lm_calls": 0},
        )

    def _execute_authorized_suggestions(
        self, neo_input: NeoInput, suggestions: list[CodeSuggestion]
    ) -> list[AppliedAction]:
        """Delegate pre-authorized actions to a host executor; never shell out here."""
        if neo_input.operating_mode is not OperatingMode.AGENT:
            return []
        policy = neo_input.authority
        if policy is None or self.execution_adapter is None:  # validated before inference
            raise ModeValidationError("agent execution authority is unavailable")
        unauthorized = [
            suggestion.file_path
            for suggestion in suggestions
            if suggestion.file_path and not policy.allows_path(suggestion.file_path)
        ]
        if unauthorized:
            raise ModeValidationError(
                "agent suggestion targets unauthorized path(s): " + ", ".join(unauthorized)
            )
        actions = self.execution_adapter.execute(suggestions, policy)
        suggestion_ids = {suggestion.suggestion_id for suggestion in suggestions}
        for action in actions:
            if action.suggestion_id not in suggestion_ids:
                raise ModeValidationError("executor returned an unknown suggestion_id")
            if action.file_path and not policy.allows_path(action.file_path):
                raise ModeValidationError("executor reported an unauthorized path")
        if self.current_learning_episode is not None:
            self.current_learning_episode.applied_actions.extend({
                "suggestion_id": action.suggestion_id,
                "file_path": action.file_path,
                "status": action.status,
                "summary": action.summary,
                "repository_revision": action.repository_revision,
            } for action in actions)
        return actions

    def _process_inner(self, neo_input: NeoInput, start_time: float) -> NeoOutput:
        """Internal process() body. Separated so process() can wrap
        with overseer.start()/stop() in a try/finally.
        """
        if neo_input.operating_mode is OperatingMode.VERIFY:
            return self._process_verification_only(neo_input, start_time)

        # Phase 5: Estimate difficulty and allocate time budget
        difficulty = self._estimate_difficulty(neo_input)
        time_budget = self._get_time_budget(difficulty)

        logger.info(f"Estimated difficulty: {difficulty}, time budget: {time_budget}s")

        # Store for outcome recording
        self.last_difficulty = difficulty

        # Phase 0: Detect implicit feedback from request history
        if self.persistent_memory and neo_input.operating_mode.allows_learning:
            current_request = {
                "prompt": neo_input.prompt,
                "timestamp": time.time(),
            }
            self.persistent_memory.detect_implicit_feedback(
                current_request, self.request_history
            )
            self.request_history.append(current_request)

        # Phase 1: Retrieve additional context
        self._log_action("retrieve_context", neo_input.prompt[:60])
        enriched_context = self._retrieve_context(neo_input)

        # Include difficulty in context for planning
        enriched_context["difficulty"] = difficulty
        enriched_context["time_budget"] = time_budget

        # Extract verifiable constraints from the prompt so the LM sees them
        # explicitly and we can check them post-hoc.
        extracted_constraints = self._extract_input_constraints(neo_input)
        if extracted_constraints:
            enriched_context["verifiable_constraints"] = [
                {"type": c.type.value, "description": c.description}
                for c in extracted_constraints
            ]

        # Decide the reasoning tier. Default: a single combined call (fast,
        # memory-primed). Novel queries with CAR + a diverse model pool escalate
        # to multi-agent deliberation. See
        # docs/solutions/tiered-reasoning-multi-agent.md.
        from neo.reasoning_mode import ReasoningMode
        self.last_deliberation = None
        decision, route_fn = self._decide_reasoning_mode(enriched_context, difficulty, neo_input)
        self.last_reasoning_mode = decision.mode.value
        self.last_reasoning_reason = decision.reason
        if self.current_learning_episode is not None:
            self.current_learning_episode.reasoning_mode = decision.mode.value
            self.current_learning_episode.reasoning_reason = decision.reason
        logger.info("Reasoning mode: %s — %s", decision.mode.value, decision.reason)

        plan = simulation_traces = code_suggestions = None
        if decision.mode is ReasoningMode.MULTI_AGENT:
            self._log_action("deliberate", neo_input.prompt[:60])
            plan, simulation_traces, code_suggestions, deliberation = self._deliberate(
                enriched_context, route_fn
            )
            if deliberation is not None and deliberation.confidence > 0.0 and code_suggestions:
                self.last_deliberation = deliberation
            else:
                logger.warning("Deliberation yielded no usable result; falling back to fast path")
                self.last_reasoning_mode = "fast"  # honest metadata: we fell back
                plan = None

        if plan is None:
            # Fast path (default, or fallback from a failed panel).
            self._log_action("lm_call", neo_input.prompt[:60])
            plan, simulation_traces, code_suggestions = self._process_combined(enriched_context)
        self._capture_detectable_fact_use(plan, simulation_traces, code_suggestions)
        code_suggestions = self._apply_role_boundary(code_suggestions)
        self.last_simulation_traces = simulation_traces
        self.last_applied_actions = self._execute_authorized_suggestions(
            neo_input, code_suggestions
        )

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
            check_statuses = [self._static_check_status(check) for check in static_checks]
            static_ran_clean = bool(check_statuses) and all(
                status == "passed" for status in check_statuses
            )
            if (
                max_confidence > self.EARLY_EXIT_CONFIDENCE
                and static_ran_clean
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

    def _apply_role_boundary(
        self, suggestions: list[CodeSuggestion]
    ) -> list[CodeSuggestion]:
        """Prevent advisory loop roles from silently becoming implementers."""
        from neo.execution_context import CallerRole

        restricted = {
            CallerRole.CRITIC,
            CallerRole.VERIFIER,
            CallerRole.STRATEGY_SELECTOR,
            CallerRole.MEMORY_RETRIEVER,
            CallerRole.POSTMORTEM_ANALYZER,
        }
        explicit_code_request = self.resolved_execution_context.requested_output in {
            "patch", "implementation", "code_change",
        }
        if self.resolved_execution_context.role in restricted and not explicit_code_request:
            return []
        return suggestions

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
        learning_allowed = neo_input.operating_mode.allows_learning and (
            neo_input.operating_mode is not OperatingMode.AGENT
            or neo_input.authority is None
            or neo_input.authority.allow_learning
        )
        fact = None
        if self.persistent_memory and learning_allowed:
            fact = self._store_reasoning(
                neo_input, plan, code_suggestions, confidence, enriched_context
            )
        simulation_facts = []
        if self.fact_store is not None and learning_allowed:
            ids = self._build_suggestion_fact_ids(fact, code_suggestions)
            episode = self.current_learning_episode
            candidates = {
                candidate.suggestion_id: {
                    "candidate_id": candidate.candidate_id,
                    "subject": candidate.subject,
                    "body": candidate.body,
                    "kind": candidate.kind,
                }
                for candidate in (episode.memory_candidates if episode else [])
            }
            self.fact_store.save_session(
                code_suggestions,
                neo_input.prompt,
                ids,
                learning_episode_id=episode.episode_id if episode else "",
                repository_revision=episode.repository_revision if episode else "",
                retrieved_fact_ids=[
                    evidence.fact_id for evidence in (episode.retrieved_facts if episode else [])
                ],
                used_fact_ids=[
                    evidence.fact_id
                    for evidence in (episode.retrieved_facts if episode else [])
                    if evidence.used_in_reasoning is True
                ],
                candidates_by_suggestion=candidates,
            )
            attempt_fact = self.fact_store.persist_attempt_outcome(
                execution_context=self._safe_execution_context_dict(),
                learning_episode_id=episode.episode_id if episode else "",
                repository_revision=episode.repository_revision if episode else "",
            )
            if attempt_fact is not None and episode is not None:
                import uuid
                from neo.memory.episodes import MemoryMutationEvidence

                episode.memory_mutations.append(MemoryMutationEvidence(
                    mutation_id=uuid.uuid4().hex,
                    operation="add_observed_attempt_episode_fact",
                    fact_id=attempt_fact.id,
                    reason="caller supplied both attempt and observed outcome",
                    after_state=self.fact_store._fact_learning_state(attempt_fact),
                ))

        elapsed = time.time() - start_time
        self._log_metrics(difficulty, time_budget, elapsed, early_exit=early_exit)

        metadata: dict[str, Any] = {"early_exit": early_exit} if early_exit else {}
        metadata["operating_mode"] = neo_input.operating_mode.value
        metadata["learning_enabled_for_request"] = learning_allowed
        metadata["repository_actions"] = len(self.last_applied_actions)
        metadata["execution_context"] = self._safe_execution_context_dict()
        # Reasoning-tier provenance — always explainable which path ran and why.
        metadata["reasoning_mode"] = getattr(self, "last_reasoning_mode", "fast")
        reason = getattr(self, "last_reasoning_reason", "")
        if reason:
            metadata["reasoning_reason"] = reason
        deliberation = getattr(self, "last_deliberation", None)
        if deliberation is not None:
            metadata["provenance"] = deliberation.provenance
            metadata["panel"] = {
                "consensus": round(deliberation.consensus, 3),
                "rounds": deliberation.rounds,
                "models_used": deliberation.models_used,
                **deliberation.meta,
            }
        if extra_metadata:
            metadata.update(extra_metadata)

        self._complete_learning_episode(
            code_suggestions=code_suggestions,
            static_checks=static_checks,
            reasoning_fact=fact,
            simulation_facts=simulation_facts,
            metadata=metadata,
        )

        from neo.execution_context import assess_loop
        goal_assessment, strategy_assessment = assess_loop(
            self.resolved_execution_context
        )
        from neo.memory.metrics import record as record_memory_metric
        record_memory_metric(
            "loop_assessed",
            episode_id=(
                self.current_learning_episode.episode_id
                if self.current_learning_episode else ""
            ),
            goal_status=goal_assessment.status,
            progress=goal_assessment.progress,
            strategy_decision=strategy_assessment.decision,
        )
        recommended_next_action: dict[str, Any] = {}
        if plan:
            recommended_next_action = {
                "description": plan[0].description,
                "rationale": plan[0].rationale,
            }
        if code_suggestions:
            recommended_next_action["suggestion_id"] = code_suggestions[0].suggestion_id
            recommended_next_action["file_path"] = code_suggestions[0].file_path

        output = NeoOutput(
            plan=plan,
            simulation_traces=simulation_traces,
            code_suggestions=code_suggestions,
            static_checks=static_checks,
            next_questions=next_questions,
            confidence=confidence,
            notes=self._generate_notes(plan, simulation_traces, static_checks),
            metadata=metadata,
            goal_assessment=goal_assessment,
            strategy_assessment=strategy_assessment,
            recommended_next_action=recommended_next_action,
        )
        self._log_usage_telemetry(output, neo_input)
        return output

    def _persist_simulation_episodes(
        self,
        traces: list[SimulationTrace],
        plan: list[PlanStep],
        prompt: str,
        code_suggestions: Optional[list[CodeSuggestion]] = None,
    ) -> list:
        """Write each simulation trace as an EPISODE fact for future retrieval.

        Only persists when we have a fact_store. Failure to write a single
        trace doesn't stop the others — and never propagates: simulation
        episodes are nice-to-have, not load-bearing.

        ``code_suggestions``'s file_paths are stashed as ``file:<path>``
        tags on each episode so future runs can ask "which files did past
        prompts like this one touch?" — closing the learning loop.
        """
        if not traces:
            return []
        # PlanStep has ``actions: list[str]`` (plural) and a top-level
        # ``description`` — both useful for a compact plan summary. Join
        # description preferentially; fall back to first action when missing.
        plan_summary = "; ".join(
            (step.description or (step.actions[0] if step.actions else ""))[:80]
            for step in plan
        )[:300]
        file_paths = list({s.file_path for s in (code_suggestions or []) if s.file_path})
        persisted = []
        for trace in traces:
            try:
                persisted.append(self.fact_store.persist_simulation_episode(
                    prompt=prompt,
                    input_data=trace.input_data or "",
                    expected_output=trace.expected_output or "",
                    reasoning_steps=list(trace.reasoning_steps or []),
                    issues_found=list(trace.issues_found or []),
                    plan_summary=plan_summary,
                    file_paths=file_paths,
                    codebase_ref=self.fact_store.codebase_root or "",
                ))
            except Exception as e:  # never block on episode-write failure
                logger.debug(f"persist_simulation_episode failed: {e}")
        return persisted

    def _complete_learning_episode(
        self, *, code_suggestions, static_checks, reasoning_fact,
        simulation_facts, metadata,
    ) -> None:
        """Finalize and persist the task evidence without promoting knowledge."""
        episode = self.current_learning_episode
        if episode is None:
            return
        from neo.memory.episodes import (
            aggregate_verification_status,
            MemoryMutationEvidence,
            SuggestionEvidence,
            VerificationEvidence,
            content_hash,
            redact_sensitive_text,
        )
        import uuid

        suggestion_ids: dict[str, str] = {}
        for suggestion in code_suggestions:
            suggestion_id = suggestion.suggestion_id
            suggestion_ids[suggestion.file_path] = suggestion_id
            episode.suggestions.append(SuggestionEvidence(
                suggestion_id=suggestion_id,
                file_path=suggestion.file_path,
                description=redact_sensitive_text(suggestion.description),
                confidence=suggestion.confidence,
                diff_sha256=content_hash(suggestion.unified_diff),
                code_sha256=content_hash(suggestion.code_block),
            ))

        if static_checks:
            for check in static_checks:
                severities = {str(d.get("severity", "")).lower() for d in check.diagnostics}
                status = check.status or (
                    "failed" if "error" in severities
                    else "warning" if check.diagnostics
                    else "passed"
                )
                episode.verification.append(VerificationEvidence(
                    verification_id=uuid.uuid4().hex,
                    kind=check.kind or "neo_static",
                    status=status,
                    tool_name=check.tool_name,
                    summary=check.summary,
                    diagnostics_count=len(check.diagnostics),
                    repository_revision=episode.repository_revision,
                ))
        else:
            episode.verification.append(VerificationEvidence(
                verification_id=uuid.uuid4().hex,
                kind="neo_static",
                status="skipped",
                summary="No deterministic static checker ran",
                repository_revision=episode.repository_revision,
            ))

        metadata["verification_verdict"] = aggregate_verification_status(
            episode.verification
        )
        if metadata["verification_verdict"] == "failed":
            for evidence in episode.retrieved_facts:
                if evidence.used_in_reasoning is True:
                    evidence.outcome_association = "failure"

        if reasoning_fact is not None:
            episode.memory_mutations.append(MemoryMutationEvidence(
                mutation_id=uuid.uuid4().hex,
                operation="legacy_fact_write",
                fact_id=reasoning_fact.id,
                reason="legacy memory backend only",
            ))
        for fact in simulation_facts:
            episode.memory_mutations.append(MemoryMutationEvidence(
                mutation_id=uuid.uuid4().hex,
                operation="add_simulation_episode_fact",
                fact_id=fact.id,
                reason="simulation trace persistence",
            ))

        episode.completed_at = time.time()
        mode = self.context.operating_mode if self.context else OperatingMode.LEARN
        episode.final_outcome = {
            OperatingMode.ADVISE: "advised_no_learning",
            OperatingMode.PATCH: "patch_proposed_no_learning",
            OperatingMode.VERIFY: "verification_complete",
            OperatingMode.LEARN: "suggested_pending_downstream_outcome",
            OperatingMode.AGENT: "agent_actions_pending_downstream_outcome",
        }[mode]
        metadata["learning_episode_id"] = episode.episode_id
        metadata["learning_task_id"] = episode.task_id
        metadata["suggestion_ids"] = suggestion_ids
        try:
            self.episode_store.save(episode)
        except Exception as exc:
            logger.warning("Learning episode persistence failed (non-fatal): %s", exc)

    @staticmethod
    def _simulation_consensus(traces: list[SimulationTrace]) -> bool:
        """LM-independent sanity check: require ≥2 simulation traces with
        matching expected_output OR no reported issues.

        Augmented by the CodeSim-style explicit decision token
        (paper 2502.05664): if ANY trace's reasoning emits **NO_MODIFY**
        the planner has explicitly approved its own plan; if any emits
        **MODIFY** the planner has explicitly flagged itself. The token
        check overrides the output-agreement heuristic — explicit > implicit.

        Returns True if we have enough trace agreement to trust the output,
        False if we should fall through to the full verification pipeline.
        When traces are unavailable (empty), returns True to avoid penalizing
        task types that don't produce them — the confidence+static_checks
        gates still apply.
        """
        if not traces:
            return True

        # Explicit decision token: planner-emitted MODIFY / NO_MODIFY beats
        # the implicit consensus heuristic. The simulator prompt may emit
        # either "**Plan Modification Needed**" / "**No Need to Modify Plan**"
        # (CodeSim canonical phrasing) or a bare MODIFY / NO_MODIFY token.
        decision = NeoEngine._extract_plan_decision(traces)
        if decision == "NO_MODIFY":
            return True
        if decision == "MODIFY":
            return False

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

    @staticmethod
    def _extract_plan_decision(traces: list[SimulationTrace]) -> Optional[str]:
        """Scan reasoning_steps for an explicit MODIFY / NO_MODIFY token.

        Returns "NO_MODIFY" if any trace approves the plan, "MODIFY" if
        any trace flags it for revision, or None when no token is present.
        Approval beats rejection within a single batch — the planner is
        usually right when it explicitly green-lights.
        """
        no_modify_re = re.compile(
            r"\b(NO_MODIFY|NO\s+NEED\s+TO\s+MODIFY|NO\s+MODIFICATION\s+NEEDED)\b",
            re.IGNORECASE,
        )
        modify_re = re.compile(
            r"\b(MODIFY|PLAN\s+MODIFICATION\s+NEEDED|REVISE\s+PLAN)\b",
            re.IGNORECASE,
        )
        for trace in traces:
            blob = " ".join(trace.reasoning_steps or [])
            if not blob:
                continue
            if no_modify_re.search(blob):
                return "NO_MODIFY"
        for trace in traces:
            blob = " ".join(trace.reasoning_steps or [])
            if modify_re.search(blob):
                return "MODIFY"
        return None

    def _retrieve_context(self, neo_input: NeoInput) -> dict[str, Any]:
        """Retrieve and enrich context from input payload."""
        context = {
            "prompt": neo_input.prompt,
            "task_type": neo_input.task_type,
            "files": neo_input.context_files,
            "error_trace": neo_input.error_trace,
            "commands": neo_input.recent_commands,
            "execution_context": self.resolved_execution_context,
            "execution_envelope_text": self.resolved_execution_context.prompt_section(),
        }

        # Optionally read additional files within safe allowlist
        if neo_input.safe_read_paths:
            additional_files = self._read_safe_files(
                neo_input.safe_read_paths,
                neo_input.working_directory,
            )
            context["additional_files"] = additional_files

        return context

    @staticmethod
    def _retrieval_query(context: dict[str, Any]) -> str:
        """Condition semantic retrieval on the larger goal and trajectory."""
        envelope = context.get("execution_context")
        if envelope is not None:
            return envelope.retrieval_query()
        return context.get("prompt", "")

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
            prompt_text = self._retrieval_query(context)
            k = self._adaptive_k_selection(prompt_text, context)
            fact_context = self.fact_store.build_context(prompt_text, environment=context, k=k)
            self._capture_retrieval_context(fact_context, included=True)
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
        from neo.reasoning_effort import (
            apply_cap,
            effort_from_memory,
        )

        # Build rich combined prompt + capture the memory signal that drove it
        # so we can size reasoning effort to query novelty.
        prompt, memory_signal = self._format_combined_prompt(context)

        difficulty = context.get("difficulty", "medium")
        effort = effort_from_memory(memory_signal, difficulty=difficulty)
        cap = getattr(self.config, "reasoning_effort_cap", None) if self.config else None
        effort = apply_cap(effort, cap)
        logger.info(
            f"Reasoning effort: {effort} "
            f"(patterns={memory_signal.pattern_count}, "
            f"avg_conf={memory_signal.avg_confidence:.2f}, "
            f"difficulty={difficulty}, cap={cap})"
        )

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

        response = self.lm.generate(
            messages,
            max_tokens=8192,
            temperature=0.3,
            reasoning_effort=effort,
        )  # Generous limit for complex multi-file changes

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

    # ------------------------------------------------------------------
    # Tiered reasoning: fast single call vs multi-agent deliberation.
    # docs/solutions/tiered-reasoning-multi-agent.md
    # ------------------------------------------------------------------

    def _compute_memory_signal(self, context: dict[str, Any]):
        """Novelty signal (pattern_count / avg_confidence) from retrieval — the
        same signal ``_format_combined_prompt`` uses, surfaced early to gate the
        reasoning tier."""
        from neo.reasoning_effort import MemorySignal, signal_from_facts
        if self.fact_store is not None:
            try:
                prompt_text = self._retrieval_query(context)
                k = self._adaptive_k_selection(prompt_text, context)
                fc = self.fact_store.build_context(prompt_text, environment=context, k=k)
                self._capture_retrieval_context(fc, included=False)
                return signal_from_facts(fc.valid_facts)
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("memory signal computation failed: %s", e)
        return MemorySignal()

    def _car_route_capability(self, prompt: str):
        """(car_available, capable_model_count, route_fn|None). Multi-agent needs
        CAR reachable AND a diverse pool; ``route_model`` (decision-only) reports
        the pool via its ``candidates`` ranking."""
        try:
            from neo.a2ui import is_daemon_reachable
            from neo.car_inference import is_available as car_available, get_runtime
        except Exception:
            return False, 0, None
        try:
            if not (car_available() and is_daemon_reachable()):
                return False, 0, None
            runtime = get_runtime()
            route_model = getattr(runtime, "route_model", None)
            if not callable(route_model):
                return True, 0, None
            route_fn = lambda p, intent_json: route_model(p, intent_json)  # noqa: E731
            from neo.panel import capable_model_count
            return True, capable_model_count(route_fn, probe=(prompt[:200] or "code task")), route_fn
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("CAR route capability check failed: %s", e)
            return False, 0, None

    def _decide_reasoning_mode(self, context: dict[str, Any], difficulty: str, neo_input):
        """Gate: fast vs multi-agent. Returns (ModeDecision, route_fn|None)."""
        from neo.reasoning_mode import decide_mode
        signal = self._compute_memory_signal(context)
        explicit = (getattr(self.config, "reasoning_mode", "auto") if self.config else "auto") or "auto"
        explicit = None if explicit.lower() == "auto" else explicit.lower()
        car_available, model_count, route_fn = self._car_route_capability(context.get("prompt", ""))
        decision = decide_mode(
            signal, difficulty=difficulty,
            car_available=car_available, capable_model_count=model_count, explicit=explicit,
        )
        return decision, route_fn

    def _build_car_role_factory(self, route_fn, prompt: str):
        """role_adapter(role) → a distinct CAR-pinned adapter per role (planner /
        coder / critic / judge), falling back to ``self.lm``. Diversity comes from
        the routing plan (``plan_role_models`` threads exclude_models)."""
        fallback = self.lm
        if route_fn is None:
            return lambda role: fallback
        from neo.panel import plan_role_models, build_role_factory
        try:
            role_models = plan_role_models(route_fn, prompt[:500])
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("role model planning failed: %s", e)
            role_models = {}
        from neo.adapters import create_adapter
        return build_role_factory(role_models, lambda m: create_adapter("car", model=m), fallback)

    def _deliberate(self, context: dict[str, Any], route_fn):
        """Run the multi-agent panel. Returns (plan, sims, code, DeliberationResult|None)."""
        from neo.multi_agent import MultiAgentReasoner
        prompt = context.get("prompt", "")
        ctx_str = ""
        for key in (
            "execution_envelope_text", "past_learnings", "verifiable_constraints",
        ):
            v = context.get(key)
            if v:
                ctx_str += (str(v)[:2000] + "\n\n")
        role_factory = self._build_car_role_factory(route_fn, prompt)
        try:
            result = MultiAgentReasoner(role_factory, k_plans=3, max_repair_rounds=1).deliberate(
                prompt, context=ctx_str.strip()[:6000]
            )
        except Exception as e:
            logger.warning("deliberation failed: %s", e)
            return None, None, None, None
        return result.plan, result.simulation_traces, result.code_suggestions, result

    def _format_combined_prompt(self, context: dict[str, Any]) -> tuple[str, "MemorySignal"]:
        """Format the combined prompt and return it alongside the memory signal.

        The signal is computed from the same retrieval used to build
        past_learnings — surfacing it lets the caller size reasoning effort
        without re-querying the store.
        """
        from neo.reasoning_effort import (
            MemorySignal,
            signal_from_facts,
            signal_from_legacy_entries,
        )

        # Get exemplars and past learnings (THE KEY CONTEXT WE WERE MISSING)
        exemplars = []
        if self.exemplar_index:
            similar = self.exemplar_index.search(context["prompt"], k=3)
            exemplars = [f"{ex.prompt} -> {ex.solution[:100]}..." for ex in similar]

        past_learnings = []
        memory_signal = MemorySignal()
        if self.fact_store is not None:
            prompt_text = self._retrieval_query(context)
            k = self._adaptive_k_selection(prompt_text, context)
            fact_context = self.fact_store.build_context(prompt_text, environment=context, k=k)
            self._capture_retrieval_context(fact_context, included=True)
            formatted = self.fact_store.format_context_for_prompt(fact_context)
            if formatted:
                past_learnings = [formatted]
            memory_signal = signal_from_facts(fact_context.valid_facts)
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
            memory_signal = signal_from_legacy_entries(relevant)

        # Build context
        task_type = context.get('task_type', 'unknown')
        task_type_str = task_type.value if hasattr(task_type, 'value') else str(task_type)
        parts = [f"Task: {context['prompt']}", f"Task Type: {task_type_str}"]
        if context.get("execution_envelope_text"):
            parts.append(context["execution_envelope_text"])

        # Inject project-local agent instructions (CLAUDE.md, .cursor/rules, etc.)
        # before the file dump so the model sees the team's written guidance
        # even when relevance ranking would have skipped those files.
        from neo.agent_context import (
            discover as discover_agent_docs,
            format_for_prompt as format_agent_docs,
        )
        agent_docs_section = format_agent_docs(discover_agent_docs(self.codebase_root))
        if agent_docs_section:
            parts.append(agent_docs_section)

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

        # Surface code smells in the gatherer-picked files so the model sees
        # known issues alongside the source. Scoped to `files[:20]` to match
        # what we actually showed the model above.
        from neo.code_smells import format_for_prompt, scan_files
        smells_section = format_for_prompt(scan_files(files[:20]))
        if smells_section:
            parts.append(smells_section)

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

        prompt = f"""Output 3 JSON blocks using this EXACT format:

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
- input_data and expected_output must be STRINGS (not JSON objects)
- The FINAL reasoning_step in the LAST simulation trace must be exactly
  either "**NO_MODIFY**" (if the plan correctly produces the expected
  output) or "**MODIFY: <brief reason>**" (if the plan needs revision).
  This token is parsed by Neo's consensus check and overrides the
  output-agreement heuristic."""
        return prompt, memory_signal

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

    @staticmethod
    def _static_check_status(check: StaticCheckResult) -> str:
        """Normalize legacy check results without treating absence as success."""
        if check.status:
            return check.status
        severities = {
            str(item.get("severity", "")).lower() for item in check.diagnostics
        }
        if "error" in severities:
            return "failed"
        if check.diagnostics:
            return "warning"
        return "passed"

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

    def _extract_input_constraints(self, neo_input: NeoInput) -> list:
        """Combine task text and explicit envelope constraints for static checks."""
        text = neo_input.prompt
        if neo_input.constraints:
            text += "\nConstraints:\n" + "\n".join(
                f"- {item}" for item in neo_input.constraints
            )
        return self._extract_prompt_constraints(text)

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
            # Generated reasoning is evidence, not durable knowledge. Create
            # one probationary candidate per suggestion in the separate
            # LearningEpisode; downstream verified outcomes decide whether a
            # candidate ever enters FactStore.
            from neo.memory.episodes import MemoryCandidateEvidence, redact_sensitive_text
            kind_map = {
                "algorithm": "pattern",
                "refactor": "architecture",
                "bugfix": "pattern",
                "feature": "decision",
                "explanation": "pattern",
            }
            candidate_kind = kind_map.get(task_type_str, "pattern")
            episode = self.current_learning_episode
            if episode is not None:
                import uuid

                for code_suggestion in suggestions:
                    body_parts = [f"Reasoning: {reasoning}"]
                    if code_suggestion.description:
                        body_parts.append(f"Suggestion: {code_suggestion.description}")
                    if pitfalls:
                        body_parts.append("Pitfalls: " + "; ".join(pitfalls[:5]))
                    episode.memory_candidates.append(MemoryCandidateEvidence(
                        candidate_id=uuid.uuid4().hex,
                        suggestion_id=code_suggestion.suggestion_id,
                        subject=redact_sensitive_text(
                            f"{pattern} [{code_suggestion.file_path}]"
                        ),
                        body=redact_sensitive_text("\n".join(body_parts)),
                        kind=candidate_kind,
                        supporting_episode_ids=[episode.episode_id],
                    ))
            return None
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

        Tuning anchor: MemMachine ablation (paper 2604.04853 §8.4.1) found
        k=20→30 yields +4.2 pts on multi-hop retrieval; k=50 regresses due
        to lost-in-the-middle. ContextAssembler still trims to budget, so
        these are candidate-pool sizes, not LLM-visible fact counts.

        Design decision: Simple heuristics before ML - measure if needed
        """
        # Check if adaptive k is enabled via env var
        if os.getenv("NEO_ADAPTIVE_K", "true").lower() != "true":
            return 10  # Default fallback

        prompt_tokens = len(prompt.split())
        task_type = context.get("task_type")
        has_error_trace = bool(context.get("error_trace"))
        context_files = len(context.get("files", []))

        # Heuristic 1 (highest priority): Specific bugfix with error trace -> laser focus
        if has_error_trace and task_type == TaskType.BUGFIX:
            return 3  # High precision, pattern should be very relevant

        # Heuristic 2: Large codebase -> comprehensive scan
        # (Check before vague prompt to avoid over-exploring large repos)
        if context_files > 20:
            return 20  # More files = need more context

        # Heuristic 3: Complex prompt -> more patterns
        if prompt_tokens > 50:
            return 20  # Detailed query suggests complex problem

        # Heuristic 4: Vague prompt -> exploration mode
        # (Lower priority - only if not a large codebase)
        if prompt_tokens < 5:
            return 30  # Need more context to understand intent

        # Default: balanced retrieval
        return 10

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

        if context.get("execution_envelope_text"):
            prompt_parts.append(f"\n{context['execution_envelope_text']}")

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
        return f"""{context.get('execution_envelope_text', '')}

Given this plan:
{self._format_plan_for_prompt(plan)}

Synthesize 3-5 test inputs and trace through expected outputs step by step.
Identify any edge cases or issues."""

    def _format_refactor_simulation_prompt(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> str:
        """Format prompt for refactor simulation."""
        return f"""{context.get('execution_envelope_text', '')}

Given this refactoring plan:
{self._format_plan_for_prompt(plan)}

Analyze dependency impact. What modules/functions will be affected?
What are the risks?"""

    def _format_bugfix_simulation_prompt(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> str:
        """Format prompt for bugfix simulation."""
        error_trace = context.get("error_trace", "No trace provided")
        return f"""{context.get('execution_envelope_text', '')}

Given this bugfix plan:
{self._format_plan_for_prompt(plan)}

Error Trace:
{error_trace}

Trace the execution path that leads to the error. What's the root cause?"""

    def _format_generic_simulation_prompt(
        self, plan: list[PlanStep], context: dict[str, Any]
    ) -> str:
        """Format prompt for generic simulation."""
        return f"""{context.get('execution_envelope_text', '')}

Given this plan:
{self._format_plan_for_prompt(plan)}

Reason through the execution step by step. Identify potential issues."""

    def _format_code_generation_prompt(
        self,
        plan: list[PlanStep],
        simulations: list[SimulationTrace],
        context: dict[str, Any],
    ) -> str:
        """Format prompt for code generation."""
        return f"""{context.get('execution_envelope_text', '')}

Plan:
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
