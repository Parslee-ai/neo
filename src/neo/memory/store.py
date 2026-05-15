"""
FactStore - Main fact-based memory system for Neo.

Replaces PersistentReasoningMemory with a scoped, supersession-based
fact store. No junk filter, no MinHash, no TF-IDF - just embeddings
and supersession chains.
"""

import hashlib
import json
import logging
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from neo.math_utils import cosine_similarity
from neo.memory.claude_memory import ClaudeMemoryIngester
from neo.memory.community import CommunityFeedIngester
from neo.memory.constraints import ConstraintIngester
from neo.memory.context import ContextAssembler
from neo.memory.seed import SeedIngester
from neo.languages import language_for_path
from neo.memory.models import ContextResult, Fact, FactKind, FactMetadata, FactScope, success_bonus
from neo.memory.outcomes import OutcomeTracker, OutcomeType
from neo.memory.scope import detect_org_and_project
from neo.pattern_extraction import extract_pattern_from_correction, get_library

logger = logging.getLogger(__name__)

# Embedding constants
EMBEDDING_CACHE_MAX_SIZE = 500
MAX_TEXT_LENGTH = 32000
SUPERSESSION_THRESHOLD = 0.85  # Cosine similarity threshold for supersession
SYNTHESIS_SIMILARITY = 0.80    # Cosine similarity threshold for review clustering

# Per-scope capacity limits
SCOPE_LIMITS: dict[str, int] = {
    FactScope.GLOBAL.value: 200,    # Cross-project patterns, grows slowly
    FactScope.ORG.value: 100,       # Team conventions, medium churn
    FactScope.PROJECT.value: 500,   # Codebase-specific, most active
    FactScope.SESSION.value: 50,    # Ephemeral, dies with session
}

# Pruning constants
STALE_MAX_CONFIDENCE = 0.4     # Facts below this confidence are stale candidates
STALE_MIN_AGE_DAYS = 14        # Must be this old before stale-pruning
DEMOTION_MIN_ACCESS = 5        # Minimum accesses before demotion kicks in
DEMOTION_PRUNE_ACCESS = 10     # Accesses threshold for hard pruning
DEMOTION_MIN_AGE_DAYS = 7      # Must be this old before demotion
DEMOTION_CONFIDENCE_PENALTY = 0.1  # Confidence reduction per demotion
DEMOTION_CONFIDENCE_FLOOR = 0.1    # Never demote below this
PROTECTION_HIT_RATE = 0.3      # Success/access ratio for protection boost
PROTECTION_BOOST = 0.05        # Confidence boost for consistently helpful facts

# Independent outcome limits (second layer — outcomes.py caps at 5/session)
MAX_INDEPENDENT_FACTS = 50     # Cap per project to prevent bloat from active repos

# Tags that protect facts from pruning/demotion (curated knowledge)
PROTECTED_TAGS = frozenset({"seed", "community", "synthesized"})

# Try importing fastembed
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

FACTS_DIR = Path.home() / ".neo" / "facts"


class FactStore:
    """Scoped fact store with supersession-based knowledge management.

    Replaces PersistentReasoningMemory. Key differences:
    - No junk filter (stores all facts)
    - No MinHash/TF-IDF (embeddings-only retrieval)
    - Supersession instead of merge/consolidation
    - Scoped storage (global, org, project)
    - Constraint ingestion from CLAUDE.md etc.
    """

    def __init__(
        self,
        codebase_root: Optional[str] = None,
        config: Optional[Any] = None,
        lm_adapter: Optional[Any] = None,
        eager_init: bool = True,
    ):
        self.codebase_root = codebase_root
        self._config = config
        self._lm_adapter = lm_adapter

        # Detect org and project
        self.org_id, self.project_id = detect_org_and_project(codebase_root)
        logger.info(f"FactStore: org={self.org_id}, project={self.project_id[:8] if self.project_id else 'none'}")

        # Storage paths
        FACTS_DIR.mkdir(parents=True, exist_ok=True)
        self._global_path = FACTS_DIR / "facts_global.json"
        self._org_path = FACTS_DIR / f"facts_org_{self.org_id}.json" if self.org_id != "unknown" else None
        self._project_path = FACTS_DIR / f"facts_project_{self.project_id}.json" if self.project_id else None

        # All facts in memory
        self._facts: list[Fact] = []
        self._cap_pending = False  # Set by _cap_independent_facts(save=False)

        # Embedding model (lazy-initialized on first use to avoid slow startup)
        self._embedder = None
        self._embedder_initialized = False
        self._embedding_cache: OrderedDict = OrderedDict()

        # Context assembler
        self._assembler = ContextAssembler()

        # Outcome tracker for learning from actual code changes
        self._outcome_tracker = OutcomeTracker(
            codebase_root=codebase_root,
            project_id=self.project_id,
        )

        # Load existing facts (required for basic operation)
        self.load()

        # Heavy I/O (migration, constraint/history ingestion) deferred to initialize()
        self._initialized = False
        if eager_init:
            self.initialize()

    def initialize(self) -> None:
        """Run post-construction I/O: migration, constraint ingestion, git history.

        Called explicitly after construction. Separating this from __init__
        ensures a partially-failed ingestion doesn't leave a broken object.
        """
        if self._initialized:
            return
        self._initialized = True

        self._maybe_migrate()
        self._ingest_seed_facts()
        self._ingest_community_feed()
        self._ingest_constraints()
        self._ingest_claude_memory()
        self._ingest_git_history()

        # Persist capping that load() deferred (save=False)
        if self._cap_pending:
            self.save()
            self._cap_pending = False

        # Lifecycle maintenance on every cold start. Previously these only
        # ran inside detect_implicit_feedback, so inactive projects accumulated
        # stale REVIEW facts indefinitely.
        try:
            self.prune_stale_facts()
            self.demote_unhelpful_facts()
            self.purge_dead_facts()
        except Exception as e:
            logger.warning(f"Lifecycle maintenance on init failed (non-fatal): {e}")

    def _ensure_embedder(self) -> None:
        """Lazy-initialize the embedding model on first use."""
        if self._embedder_initialized:
            return
        self._embedder_initialized = True
        if FASTEMBED_AVAILABLE:
            try:
                self._embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")
                logger.info("FactStore: Jina Code v2 embeddings initialized")
            except Exception as e:
                logger.warning(f"FactStore: Failed to initialize embedder: {e}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_fact(
        self,
        subject: str,
        body: str,
        kind: FactKind = FactKind.PATTERN,
        scope: FactScope = FactScope.PROJECT,
        confidence: float = 0.5,
        source_file: str = "",
        source_prompt: str = "",
        tags: Optional[list[str]] = None,
        depends_on: Optional[list[str]] = None,
        provenance: str = "inferred",
    ) -> Fact:
        """Add a new fact to the store.

        No junk filter - all facts are stored. If a supersession candidate
        is found (same scope+kind, cosine > 0.85), the old fact is superseded.

        Args:
            subject: Concise label for the fact.
            body: Full content.
            kind: Type of fact.
            scope: Scope level.
            confidence: Initial confidence (0.0-1.0).
            source_file: File that generated this fact.
            source_prompt: Prompt that triggered this fact.
            tags: Optional tags for filtering.
            depends_on: Fact IDs this derives from.
            provenance: Origin type - "structural", "inferred", or "observed".

        Returns:
            The newly created Fact.
        """
        # Generate embedding
        embed_text = f"{subject} {body}"
        embedding = self._embed_text(embed_text)

        fact = Fact(
            subject=subject,
            body=body,
            kind=kind,
            scope=scope,
            org_id=self.org_id,
            project_id=self.project_id,
            metadata=FactMetadata(
                source_file=source_file,
                source_prompt=source_prompt,
                confidence=confidence,
                provenance=provenance,
            ),
            embedding=embedding,
            tags=tags or [],
            depends_on=depends_on or [],
        )

        # Check for supersession candidate
        candidate = self._find_supersession_candidate(fact)
        if candidate:
            self._supersede(candidate, fact)

        self._facts.append(fact)
        self._enforce_scope_limit(fact.scope)
        self.save()
        return fact

    def retrieve_relevant(self, query: str, k: int = 5) -> list[Fact]:
        """Retrieve the most relevant valid facts for a query.

        Args:
            query: The search query.
            k: Maximum number of facts to return.

        Returns:
            List of relevant Fact objects, scored by similarity * confidence.
        """
        query_embedding = self._embed_text(query)

        valid_facts = [f for f in self._facts if f.is_valid and f.kind != FactKind.CONSTRAINT]

        if not valid_facts:
            return []

        scored: list[tuple[Fact, float]] = []
        now = time.time()

        for fact in valid_facts:
            sim = 0.5  # default when no embeddings
            if query_embedding is not None and fact.embedding is not None:
                sim = self._cosine_similarity(query_embedding, fact.embedding)

            confidence = fact.metadata.confidence
            # Provenance boost: structural facts are more trustworthy
            provenance_bonus = {
                "structural": 0.05,
                "observed": 0.02,
                "inferred": 0.0,
            }.get(fact.metadata.provenance, 0.0)
            score = (
                sim * confidence
                + success_bonus(fact.metadata.success_count)
                + provenance_bonus
            )
            scored.append((fact, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Update access metadata for returned facts
        results = []
        for fact, _ in scored[:k]:
            fact.metadata.last_accessed = now
            fact.metadata.access_count += 1
            results.append(fact)

        return results

    def build_context(
        self,
        query: str,
        environment: Optional[dict] = None,
        k: int = 5,
    ) -> ContextResult:
        """Build a full ContextResult for LLM injection.

        Delegates to ContextAssembler to organize facts into layers.
        """
        query_embedding = self._embed_text(query)
        result = self._assembler.assemble(
            facts=self._facts,
            query=query,
            query_embedding=query_embedding,
            environment=environment,
            k=k,
        )

        # Update access metadata on retrieved facts (mirrors retrieve_relevant)
        now = time.time()
        for fact in result.valid_facts:
            fact.metadata.last_accessed = now
            fact.metadata.access_count += 1

        return result

    def format_context_for_prompt(self, ctx: ContextResult) -> str:
        """Render ContextResult as a formatted string."""
        return self._assembler.format_context_for_prompt(ctx)

    def save_session(
        self,
        suggestions: list,
        prompt: str,
        suggestion_fact_ids: Optional[dict[str, str]] = None,
    ) -> None:
        """Persist current session for outcome detection on next invocation.

        Args:
            suggestions: List of CodeSuggestion objects.
            prompt: The user's prompt.
            suggestion_fact_ids: Mapping of file_path -> fact_id for outcome linkage.
        """
        try:
            self._outcome_tracker.save_session(suggestions, prompt, suggestion_fact_ids)
        except Exception as e:
            logger.warning(f"Session save failed: {e}")

    # ------------------------------------------------------------------ #
    # Backward compatibility
    # ------------------------------------------------------------------ #

    def memory_level(self) -> float:
        """Calculate memory level based on quality-weighted facts.

        Each valid fact contributes its confidence (boosted by success rate
        for well-tested facts). Sigmoid scaling maps total to 0.0-1.0.

        The reference quality scales with the total capacity of loaded scopes,
        so a project-only view and a multi-project view both reach "The One"
        at ~80% of their potential quality.
        """
        valid = [f for f in self._facts if f.is_valid]
        if not valid:
            return 0.0

        total_quality = 0.0
        for f in valid:
            score = f.metadata.confidence
            access = f.metadata.access_count
            if access > 0 and f.metadata.success_count > 0:
                success_rate = f.metadata.success_count / access
                score = score * 0.5 + success_rate * 0.5
            # Usage bonus: reward facts that have been applied
            usage_factor = min(1.0, access / 10)
            score *= (0.9 + 0.1 * usage_factor)
            total_quality += score

        # Reference quality = 20% of aggregate capacity across loaded scopes.
        # "The One" (0.80) requires ~80% of capacity filled with well-validated
        # facts (avg confidence ~0.7 + some success history).
        loaded_scopes = {f.scope for f in self._facts}
        capacity = sum(SCOPE_LIMITS.get(s.value, 200) for s in loaded_scopes)
        reference_quality = max(50.0, capacity * 0.2)

        return 1.0 - 1.0 / (1.0 + total_quality / reference_quality)

    def detect_implicit_feedback(self, current_request: dict, request_history: list) -> None:
        """Detect outcomes from previous session and update/create facts.

        For "accepted" outcomes with a linked suggestion fact: boost the original
        fact's confidence and increment success_count (no new REVIEW fact).
        Otherwise: fall back to creating a new REVIEW fact.
        """
        try:
            outcomes, suggestion_fact_ids = self._outcome_tracker.detect_outcomes()
        except Exception as e:
            logger.warning(f"Outcome detection failed: {e}")
            return

        # Architectural delta across the session batch — modulates how much
        # we trust each accept/modify outcome. Best-effort: returns None if
        # there's no baseline or if metrics computation fails.
        arch_delta = self._outcome_tracker.compute_arch_delta()
        arch_severity = arch_delta.severity() if arch_delta is not None else "neutral"
        if arch_delta is not None and arch_severity != "neutral":
            logger.info(
                f"Arch delta over session batch: {arch_severity} "
                f"(cycles={arch_delta.cycles_delta:+d}, "
                f"god_files={arch_delta.god_files_delta:+d}, "
                f"max_depth={arch_delta.max_depth_delta:+d})"
            )

        # Modulation amount: small enough not to overwhelm the base accept/
        # modify signal, large enough to be felt over many iterations.
        arch_mod = (
            -0.1 if arch_severity == "regression"
            else 0.1 if arch_severity == "improvement"
            else 0.0
        )

        facts_by_id = {f.id: f for f in self._facts}
        linked_count = 0

        def _lookup_fact_id(file_path: str) -> Optional[str]:
            """Look up fact_id with fallback for path normalization mismatches."""
            fid = suggestion_fact_ids.get(file_path)
            if fid:
                return fid
            # Try with/without leading slash
            if file_path.startswith("/"):
                fid = suggestion_fact_ids.get(file_path.lstrip("/"))
            else:
                fid = suggestion_fact_ids.get("/" + file_path)
            return fid

        for outcome in outcomes:
            if outcome.outcome_type == OutcomeType.ACCEPTED:
                # Try to link back to the original suggestion fact
                fact_id = _lookup_fact_id(outcome.file_path)
                original_fact = facts_by_id.get(fact_id) if fact_id else None

                if original_fact and original_fact.is_valid:
                    # Boost original fact instead of creating orphan REVIEW.
                    # Base +0.2; modulated by arch delta so a session that
                    # regressed structure earns less trust than one that didn't.
                    boost = max(-0.05, 0.2 + arch_mod)
                    original_fact.metadata.confidence = min(
                        1.0, max(0.0, original_fact.metadata.confidence + boost)
                    )
                    original_fact.metadata.success_count += 1
                    original_fact.metadata.last_accessed = time.time()
                    linked_count += 1
                    continue

                # Fallback: no linked fact, create REVIEW as before
                subject = f"outcome:accepted {outcome.file_path}"
                body_parts = [
                    f"User applied suggestion to {outcome.file_path}.",
                    f"Original suggestion: {outcome.suggestion_description}",
                    f"Original confidence: {outcome.suggestion_confidence:.2f}",
                ]
                if outcome.diff_summary:
                    body_parts.append(f"Actual changes:\n{outcome.diff_summary}")
                body = "\n".join(body_parts)
                tags = ["outcome", "accepted"]
                confidence = min(1.0, max(0.0, outcome.suggestion_confidence + 0.1 + arch_mod))
            elif outcome.outcome_type == OutcomeType.MODIFIED:
                # User corrected neo's suggestion - learn from the correction
                subject = f"outcome:modified {outcome.file_path}"
                body_parts = [
                    f"User modified neo's suggestion for {outcome.file_path}.",
                    f"Original suggestion: {outcome.suggestion_description}",
                    f"Original confidence: {outcome.suggestion_confidence:.2f}",
                ]
                if outcome.diff_summary:
                    body_parts.append(f"What user actually did:\n{outcome.diff_summary}")
                body = "\n".join(body_parts)
                tags = ["outcome", "modified"]
                confidence = 0.4

                # Demote the original suggestion fact since it was corrected.
                # arch_mod is negative for regression and positive for
                # improvement, so adding it gives: regression deepens the
                # penalty (-0.2 + -0.1 = -0.3), improvement softens it
                # (-0.2 + 0.1 = -0.1).
                fact_id = _lookup_fact_id(outcome.file_path)
                original_fact = facts_by_id.get(fact_id) if fact_id else None
                if original_fact and original_fact.is_valid:
                    penalty = min(-0.05, -0.2 + arch_mod)
                    original_fact.metadata.confidence = max(
                        0.1, original_fact.metadata.confidence + penalty
                    )
                    original_fact.metadata.last_accessed = time.time()
                    linked_count += 1
            elif outcome.outcome_type == OutcomeType.UNVERIFIED:
                # Suggested file changed, but no diff to compare — weak signal.
                # Only update linked fact; never create standalone REVIEW.
                fact_id = _lookup_fact_id(outcome.file_path)
                original_fact = facts_by_id.get(fact_id) if fact_id else None
                if original_fact and original_fact.is_valid:
                    boost = max(-0.05, 0.1 + arch_mod)
                    original_fact.metadata.confidence = min(
                        1.0, max(0.0, original_fact.metadata.confidence + boost)
                    )
                    original_fact.metadata.success_count += 1
                    original_fact.metadata.last_accessed = time.time()
                    linked_count += 1
                continue  # Never create a REVIEW fact for unverified outcomes
            elif outcome.outcome_type == OutcomeType.INDEPENDENT:
                subject = f"outcome:independent {outcome.file_path}"
                body_parts = [f"User changed {outcome.file_path} (not suggested by neo)."]
                if outcome.diff_summary:
                    body_parts.append(f"Changes:\n{outcome.diff_summary}")
                else:
                    body_parts.append("No diff content available.")
                body = "\n".join(body_parts)
                tags = ["outcome", "independent"]
                confidence = 0.2  # Low confidence so stale pruning cleans up faster
            else:
                continue

            self.add_fact(
                subject=subject,
                body=body,
                kind=FactKind.REVIEW,
                scope=FactScope.PROJECT,
                confidence=confidence,
                source_prompt=current_request.get("prompt", "")[:200],
                tags=tags,
            )

        if linked_count:
            self.save()
            logger.info(f"Boosted/demoted {linked_count} original fact(s) from outcomes")
        if outcomes:
            modified = sum(1 for o in outcomes if o.outcome_type == OutcomeType.MODIFIED)
            logger.info(f"Processed {len(outcomes)} outcome(s): modified={modified}")
            # Chain maintenance: synthesize -> prune stale -> demote unhelpful -> purge dead
            self.synthesize_reviews()
            self.prune_stale_facts()
            self.demote_unhelpful_facts()
            self.purge_dead_facts()

        # Extract prevention patterns from corrections
        modified_outcomes = [o for o in outcomes if o.outcome_type == OutcomeType.MODIFIED]
        if modified_outcomes and self._lm_adapter:
            for outcome in modified_outcomes[:3]:  # Limit to 3 per session
                try:
                    # Derive language from extension for accurate fence
                    # tagging in the extraction prompt.
                    language = language_for_path(outcome.file_path)

                    pattern = extract_pattern_from_correction(
                        problem_description=outcome.suggestion_description,
                        failed_code=outcome.suggestion_description,  # Best we have
                        corrected_code=outcome.diff_summary,
                        bug_category="suggestion-mismatch",
                        root_cause=f"Neo's suggestion for {outcome.file_path} was modified by user",
                        adapter=self._lm_adapter,
                        language=language,
                    )
                    library = get_library()
                    library.add_pattern(pattern)
                    logger.info(f"Learned prevention pattern from correction on {outcome.file_path}")
                except Exception as e:
                    logger.warning(f"Pattern extraction from correction failed: {e}")

    @property
    def entries(self) -> list[Fact]:
        """Backward-compatible access to all facts."""
        return self._facts

    def find_contributable(self, min_confidence: float = 0.8,
                           min_successes: int = 3) -> list[Fact]:
        """Find high-quality facts worth contributing to the community feed.

        Criteria: high confidence, real success validation, not already
        from seed/community feeds, not constraints (project-specific).
        """
        auto_tags = {"seed", "community", "constraint", "history", "git-commit"}
        results = []
        for f in self._facts:
            if not f.is_valid:
                continue
            if f.kind == FactKind.CONSTRAINT:
                continue
            if f.metadata.confidence < min_confidence:
                continue
            if f.metadata.success_count < min_successes:
                continue
            if auto_tags & set(f.tags):
                continue
            results.append(f)
        return results

    # ------------------------------------------------------------------ #
    # Scope capacity enforcement
    # ------------------------------------------------------------------ #

    def _enforce_scope_limit(self, scope: FactScope) -> int:
        """Evict lowest-quality valid facts when a scope exceeds its limit.

        Eviction score: confidence weighted by success rate. Constraints and
        synthesized facts are protected from eviction. The newest fact (just
        appended) is never evicted.

        Returns the number of evicted facts.
        """
        limit = SCOPE_LIMITS.get(scope.value)
        if limit is None:
            return 0

        scoped = [f for f in self._facts if f.scope == scope and f.is_valid]
        excess = len(scoped) - limit
        if excess <= 0:
            return 0

        def _eviction_score(fact: Fact) -> float:
            """Lower score = evict first."""
            base = fact.metadata.confidence
            total = fact.metadata.access_count
            if total > 0 and fact.metadata.success_count > 0:
                base = base * 0.5 + (fact.metadata.success_count / total) * 0.5
            return base

        # Never evict constraints or synthesized archetypes
        evictable = [
            f for f in scoped
            if f.kind != FactKind.CONSTRAINT and not (PROTECTED_TAGS & set(f.tags))
        ]
        evictable.sort(key=_eviction_score)

        evicted = 0
        for fact in evictable[:excess]:
            fact.is_valid = False
            self._cascade_needs_review(fact.id)
            evicted += 1

        if evicted:
            logger.info(
                f"Evicted {evicted} fact(s) from {scope.value} scope "
                f"(limit={limit}, was={len(scoped)})"
            )
        return evicted

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Save facts to scoped JSON files.

        Global facts use best-effort merge: re-reads the global file before
        writing to preserve facts added by other FactStore instances since
        load(). This is NOT concurrency-safe (no file locking). Neo is a
        single-user CLI tool — concurrent writes are rare but possible if
        multiple terminals invoke neo simultaneously. In that case, last
        writer wins for same-ID facts, but cross-project additions survive.
        """
        global_facts = [f for f in self._facts if f.scope == FactScope.GLOBAL]
        org_facts = [f for f in self._facts if f.scope == FactScope.ORG]
        project_facts = [f for f in self._facts if f.scope == FactScope.PROJECT]

        merged_global = self._merge_global_on_save(global_facts)
        self._save_file(self._global_path, merged_global)
        if self._org_path:
            self._save_file(self._org_path, org_facts)
        if self._project_path:
            self._save_file(self._project_path, project_facts)

    def _merge_global_on_save(self, our_facts: list[Fact]) -> list[Fact]:
        """Best-effort merge: keep disk facts we don't have in memory.

        Not concurrency-safe. Reduces data loss from sequential (not
        simultaneous) cross-project saves, which is the common case.
        """
        disk_facts = self._load_file(self._global_path)
        if not disk_facts:
            return our_facts

        our_ids = {f.id for f in our_facts}
        new_from_disk = [f for f in disk_facts if f.id not in our_ids]
        if new_from_disk:
            logger.info(f"Preserved {len(new_from_disk)} global fact(s) from disk")
        return our_facts + new_from_disk

    def purge_dead_facts(self) -> int:
        """Remove invalid facts whose chain resolves to a valid successor.

        Follows supersession chains to find the terminal fact. If the chain
        ends at a valid fact and the invalid fact hasn't been accessed in
        30+ days, it's safe to purge.

        Returns the number of purged facts.
        """
        facts_by_id = {f.id: f for f in self._facts}
        now = time.time()
        thirty_days = 30 * 86400

        def chain_resolves_to_valid(fact: Fact) -> bool:
            """Follow superseded_by chain; return True if it ends at a valid fact."""
            seen: set[str] = set()
            current = fact
            while current and not current.is_valid:
                if current.id in seen:
                    return False  # cycle
                seen.add(current.id)
                current = facts_by_id.get(current.superseded_by)  # type: ignore[arg-type]
            return current is not None and current.is_valid

        before = len(self._facts)
        self._facts = [
            f for f in self._facts
            if f.is_valid
            or (now - f.metadata.last_accessed) < thirty_days
            or not chain_resolves_to_valid(f)
        ]
        purged = before - len(self._facts)
        if purged:
            self.save()
            logger.info(f"Purged {purged} dead invalid facts")
        return purged

    def prune_stale_facts(self) -> int:
        """Remove valid-but-useless facts: low confidence, zero successes, old enough.

        Targets noise that was never validated. Does NOT touch CONSTRAINT facts
        or recently-created facts. INDEPENDENT-tagged facts are pure observation
        noise and prune twice as fast (7 days instead of 14).

        Returns the number of pruned facts.
        """
        now = time.time()
        default_stale_age = STALE_MIN_AGE_DAYS * 86400
        independent_stale_age = 7 * 86400
        pruned = 0

        for fact in self._facts:
            if not fact.is_valid:
                continue
            if fact.kind == FactKind.CONSTRAINT:
                continue
            if fact.metadata.success_count > 0:
                continue
            # Curated facts (seed, community, synthesized) are protected
            if PROTECTED_TAGS & set(fact.tags):
                continue

            is_independent = "independent" in fact.tags
            stale_age = independent_stale_age if is_independent else default_stale_age
            # INDEPENDENT facts are 0.2 confidence noise by construction; we
            # don't gate them on STALE_MAX_CONFIDENCE so they always prune
            # when their clock runs out.
            if not is_independent and fact.metadata.confidence >= STALE_MAX_CONFIDENCE:
                continue
            if (now - fact.metadata.created_at) < stale_age:
                continue

            fact.is_valid = False
            self._cascade_needs_review(fact.id)
            pruned += 1

        if pruned:
            self.save()
            logger.info(f"Pruned {pruned} stale unvalidated fact(s)")
        return pruned

    def demote_unhelpful_facts(self) -> int:
        """Demote or prune facts that are retrieved but never lead to accepted suggestions.

        Two tiers:
        - access_count 5-9, success 0: reduce confidence by 0.1 (floor 0.1)
        - access_count >= 10, success 0: mark invalid (actively unhelpful)

        Facts with hit rate > 30% get a small confidence boost (protection).

        Returns the total number of facts affected (demoted + pruned + boosted).
        """
        now = time.time()
        min_age = DEMOTION_MIN_AGE_DAYS * 86400
        affected = 0

        for fact in self._facts:
            if not fact.is_valid:
                continue
            if fact.kind == FactKind.CONSTRAINT:
                continue
            # Curated facts (seed, community, synthesized) are protected
            if PROTECTED_TAGS & set(fact.tags):
                continue
            if (now - fact.metadata.created_at) < min_age:
                continue
            if fact.metadata.access_count < DEMOTION_MIN_ACCESS:
                continue

            access = fact.metadata.access_count
            success = fact.metadata.success_count

            if success == 0:
                if access >= DEMOTION_PRUNE_ACCESS:
                    # Hard prune: accessed 10+ times, never helpful
                    fact.is_valid = False
                    self._cascade_needs_review(fact.id)
                    affected += 1
                else:
                    # Soft demotion: reduce confidence
                    fact.metadata.confidence = max(
                        DEMOTION_CONFIDENCE_FLOOR,
                        fact.metadata.confidence - DEMOTION_CONFIDENCE_PENALTY,
                    )
                    affected += 1
            elif success > 0 and (success / access) >= PROTECTION_HIT_RATE:
                # Protect consistently helpful facts
                new_conf = min(1.0, fact.metadata.confidence + PROTECTION_BOOST)
                if new_conf != fact.metadata.confidence:
                    fact.metadata.confidence = new_conf
                    affected += 1

        if affected:
            self.save()
            logger.info(f"Demote/protect cycle affected {affected} fact(s)")
        return affected

    def load(self) -> None:
        """Load facts from all scoped files and merge."""
        self._facts = []
        self._facts.extend(self._load_file(self._global_path))
        if self._org_path:
            self._facts.extend(self._load_file(self._org_path))
        if self._project_path:
            self._facts.extend(self._load_file(self._project_path))
        logger.info(f"FactStore: Loaded {len(self._facts)} facts")
        # Cap runs in-memory only; save is deferred to initialize()
        self._cap_independent_facts(save=False)

    def _cap_independent_facts(self, save: bool = True) -> None:
        """Prevent bloat: invalidate excess independent-tagged facts.

        Active repos can generate hundreds of independent outcomes per week.
        Keep only the newest MAX_INDEPENDENT_FACTS to maintain retrieval quality.
        Protected-tag facts are never capped.

        Args:
            save: If True, persist after capping. Set False when called from
                  load() to avoid save-during-load (deferred to initialize()).
        """
        indep = [
            f for f in self._facts
            if f.is_valid and "independent" in f.tags
            and not (PROTECTED_TAGS & set(f.tags))
        ]
        if len(indep) <= MAX_INDEPENDENT_FACTS:
            return
        indep.sort(key=lambda f: f.metadata.created_at, reverse=True)
        pruned = 0
        for f in indep[MAX_INDEPENDENT_FACTS:]:
            f.is_valid = False
            pruned += 1
        if pruned:
            if save:
                self.save()
            else:
                self._cap_pending = True
            logger.info(f"Capped independent facts: invalidated {pruned}, kept {MAX_INDEPENDENT_FACTS}")

    def _save_file(self, path: Path, facts: list[Fact]) -> None:
        """Save a list of facts to a JSON file atomically.

        Writes to a temp file in the same directory, then renames.
        Uses mkstemp to avoid collisions with concurrent processes.
        """
        import os
        import tempfile

        try:
            data = {
                "version": "2.0",
                "facts": [f.to_dict() for f in facts],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as fh:
                    json.dump(data, fh, indent=2)
                os.replace(tmp_name, str(path))
            except BaseException:
                os.unlink(tmp_name)
                raise
            logger.debug(f"Saved {len(facts)} facts to {path}")
        except Exception as e:
            logger.error(f"Failed to save facts to {path}: {e}")

    def _load_file(self, path: Optional[Path]) -> list[Fact]:
        """Load facts from a JSON file."""
        if path is None or not path.exists():
            return []
        try:
            with open(path) as fh:
                data = json.load(fh)
            return [Fact.from_dict(d) for d in data.get("facts", [])]
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load facts from {path}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Supersession
    # ------------------------------------------------------------------ #

    def _find_supersession_candidate(self, new_fact: Fact) -> Optional[Fact]:
        """Find an existing fact that the new fact should supersede.

        Criteria: same scope + kind, cosine similarity > threshold.
        """
        if new_fact.embedding is None:
            return None

        best_match: Optional[Fact] = None
        best_sim = 0.0

        for fact in self._facts:
            if not fact.is_valid:
                continue
            if fact.scope != new_fact.scope or fact.kind != new_fact.kind:
                continue
            if fact.embedding is None:
                continue

            sim = self._cosine_similarity(new_fact.embedding, fact.embedding)
            if sim > SUPERSESSION_THRESHOLD and sim > best_sim:
                best_sim = sim
                best_match = fact

        return best_match

    def _supersede(self, old: Fact, new: Fact) -> None:
        """Supersede an old fact with a new one and cascade needs_review."""
        old.is_valid = False
        old.superseded_by = new.id
        new.supersedes = old.id

        # Carry forward confidence with a small boost, but never downgrade
        carry_forward = min(1.0, old.metadata.confidence + 0.05)
        new.metadata.confidence = max(new.metadata.confidence, carry_forward)

        # Cascade: mark dependents as needing review
        self._cascade_needs_review(old.id)

        logger.info(f"Superseded fact '{old.subject[:40]}' with '{new.subject[:40]}'")

    def _cascade_needs_review(self, superseded_id: str) -> None:
        """Mark facts that depend on a superseded fact as needing review."""
        for fact in self._facts:
            if superseded_id in fact.depends_on and fact.is_valid:
                fact.needs_review = True
                logger.debug(f"Marked fact '{fact.subject[:40]}' as needs_review (dependency superseded)")

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using local Jina model."""
        if not text or not text.strip():
            return None

        self._ensure_embedder()

        cache_key = hashlib.sha256(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            self._embedding_cache.move_to_end(cache_key)
            return self._embedding_cache[cache_key]

        embedding = None
        if self._embedder:
            try:
                truncated = text[:MAX_TEXT_LENGTH]
                embeddings = list(self._embedder.embed([truncated]))
                if embeddings:
                    embedding = np.array(embeddings[0], dtype=np.float32)
                    if not np.isfinite(embedding).all():
                        logger.error("Embedding contains NaN or Inf values")
                        embedding = None
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")

        if embedding is not None:
            self._embedding_cache[cache_key] = embedding
            if len(self._embedding_cache) > EMBEDDING_CACHE_MAX_SIZE:
                self._embedding_cache.popitem(last=False)

        return embedding

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return cosine_similarity(a, b)

    # ------------------------------------------------------------------ #
    # Seed fact ingestion
    # ------------------------------------------------------------------ #

    def _ingest_seed_facts(self) -> None:
        """Load curated facts bundled with the neo package.

        Seed facts ship with every release and provide community-curated
        patterns (security, performance, reliability). Re-ingested only
        when the seed file changes (i.e., after a package update).
        """
        ingester = SeedIngester(
            org_id=self.org_id,
            project_id=self.project_id,
        )
        new_facts, superseded_facts = ingester.ingest(self._facts)

        if new_facts or superseded_facts:
            for fact in new_facts:
                if fact.embedding is None:
                    embed_text = f"{fact.subject} {fact.body}"
                    fact.embedding = self._embed_text(embed_text)

            self._facts.extend(new_facts)
            self.save()
            logger.info(
                f"Seed facts: {len(new_facts)} new, {len(superseded_facts)} superseded"
            )

    def _ingest_community_feed(self) -> None:
        """Fetch and ingest community-curated patterns from remote feed.

        Checks once per day, caches locally, falls back to cache on
        network failure. Patterns are contributed via PR and ship
        between releases.
        """
        try:
            ingester = CommunityFeedIngester(
                org_id=self.org_id,
                project_id=self.project_id,
            )
            new_facts, superseded_facts = ingester.ingest(self._facts)

            if new_facts or superseded_facts:
                for fact in new_facts:
                    if fact.embedding is None:
                        embed_text = f"{fact.subject} {fact.body}"
                        fact.embedding = self._embed_text(embed_text)

                self._facts.extend(new_facts)
                self.save()
                logger.info(
                    f"Community feed: {len(new_facts)} new, "
                    f"{len(superseded_facts)} superseded"
                )
        except Exception as e:
            # Network failures should never block neo startup
            logger.debug(f"Community feed ingestion failed: {e}")

    # ------------------------------------------------------------------ #
    # Constraint ingestion
    # ------------------------------------------------------------------ #

    def _ingest_constraints(self) -> None:
        """Scan constraint files and ingest/update constraint facts."""
        config_auto_scan = getattr(self._config, "constraint_auto_scan", True)
        if not config_auto_scan:
            return

        ingester = ConstraintIngester(
            codebase_root=self.codebase_root or "",
            org_id=self.org_id,
            project_id=self.project_id,
        )
        new_facts, superseded_facts = ingester.ingest(self._facts)

        if new_facts or superseded_facts:
            # Generate embeddings for new constraint facts
            for fact in new_facts:
                if fact.embedding is None:
                    embed_text = f"{fact.subject} {fact.body}"
                    fact.embedding = self._embed_text(embed_text)

            self._facts.extend(new_facts)
            self.save()
            logger.info(
                f"Constraints: {len(new_facts)} new, {len(superseded_facts)} superseded"
            )

    def _ingest_claude_memory(self) -> None:
        """Ingest curated knowledge from Claude Code's auto-memory system.

        Reads ~/.claude/projects/{project-id}/memory/*.md files that Claude
        Code has distilled from past conversations. These are pre-curated,
        high-quality facts about the project.
        """
        ingester = ClaudeMemoryIngester(
            codebase_root=self.codebase_root or "",
            org_id=self.org_id,
            project_id=self.project_id,
        )
        new_facts, superseded_facts = ingester.ingest(self._facts)

        if new_facts or superseded_facts:
            for fact in new_facts:
                if fact.embedding is None:
                    embed_text = f"{fact.subject} {fact.body}"
                    fact.embedding = self._embed_text(embed_text)

            self._facts.extend(new_facts)
            self.save()
            logger.info(
                f"Claude memory: {len(new_facts)} new, {len(superseded_facts)} superseded"
            )

    # ------------------------------------------------------------------ #
    # Git history ingestion
    # ------------------------------------------------------------------ #

    def _ingest_git_history(self) -> None:
        """Learn from git commit history that hasn't been ingested yet.

        Runs on each initialization to catch up on commits made since
        the last neo invocation. Uses a watermark to avoid re-ingesting.
        """
        try:
            records = self._outcome_tracker.ingest_git_history(max_commits=50)
        except Exception as e:
            logger.warning(f"Git history ingestion failed: {e}")
            return

        if not records:
            return

        for record in records:
            self.add_fact(
                subject=record["subject"],
                body=record["body"],
                kind=FactKind.REVIEW,
                scope=FactScope.PROJECT,
                confidence=0.5,
                tags=["history", "git-commit"],
            )

        logger.info(f"Ingested {len(records)} facts from git history")

    # ------------------------------------------------------------------ #
    # Review synthesis
    # ------------------------------------------------------------------ #

    def synthesize_reviews(self) -> int:
        """Distill clusters of REVIEW facts into higher-level PATTERN/FAILURE facts.

        Triggers every 10 new REVIEW facts after 20+ valid exist. Groups by tag,
        clusters by cosine similarity, and synthesizes clusters of 3+ into a
        single fact that supersedes the source facts.

        The watermark tracks total REVIEW facts ever seen (valid + invalidated)
        to avoid the count-drift bug where synthesis invalidates facts and
        the watermark comparison breaks.

        Returns:
            Number of synthesized facts created.
        """
        # Count ALL review facts (valid + invalid) for watermark comparison
        all_review_count = sum(
            1 for f in self._facts if f.kind == FactKind.REVIEW
        )
        valid_review_facts = [
            f for f in self._facts
            if f.is_valid and f.kind == FactKind.REVIEW
        ]

        if len(valid_review_facts) < 20:
            return 0

        # Check watermark against total count (not just valid)
        watermark = self._load_synthesis_watermark()
        if all_review_count - watermark < 10:
            return 0

        synthesized_count = 0

        # Group by primary tag (outcome:accepted, outcome:independent, history:*)
        groups: dict[str, list[Fact]] = {}
        for fact in valid_review_facts:
            key = self._synthesis_group_key(fact)
            groups.setdefault(key, []).append(fact)

        for group_key, facts in groups.items():
            if len(facts) < 3:
                continue

            clusters = self._cluster_by_similarity(facts, SYNTHESIS_SIMILARITY)

            for cluster in clusters:
                if len(cluster) < 3:
                    continue

                new_fact = self._synthesize_cluster(cluster, group_key)
                if new_fact:
                    self._facts.append(new_fact)
                    synthesized_count += 1

        if synthesized_count:
            self.save()
            logger.info(f"Synthesized {synthesized_count} fact(s) from REVIEW clusters")

        # Save total count as watermark (valid + invalidated)
        self._save_synthesis_watermark(all_review_count)
        return synthesized_count

    @staticmethod
    def _synthesis_group_key(fact: Fact) -> str:
        """Determine the grouping key for a REVIEW fact based on its tags."""
        tags = set(fact.tags)
        if "accepted" in tags:
            return "outcome:accepted"
        if "independent" in tags:
            return "outcome:independent"
        if "history" in tags:
            return "history"
        return "other"

    def _cluster_by_similarity(
        self, facts: list[Fact], threshold: float
    ) -> list[list[Fact]]:
        """Cluster facts by cosine similarity using complete-linkage.

        A candidate joins a cluster only if it meets the similarity threshold
        against ALL existing cluster members, ensuring all facts in a cluster
        are mutually similar.

        Returns list of clusters (each a list of facts).
        """
        embedded = [f for f in facts if f.embedding is not None]
        if not embedded:
            return []

        assigned: set[str] = set()
        clusters: list[list[Fact]] = []

        for fact in embedded:
            if fact.id in assigned:
                continue

            cluster = [fact]
            assigned.add(fact.id)

            for other in embedded:
                if other.id in assigned:
                    continue
                # Complete-linkage: must be similar to ALL cluster members
                if all(
                    self._cosine_similarity(member.embedding, other.embedding) >= threshold  # type: ignore[arg-type]
                    for member in cluster
                ):
                    cluster.append(other)
                    assigned.add(other.id)

            clusters.append(cluster)

        return clusters

    def _synthesize_cluster(
        self, cluster: list[Fact], group_key: str
    ) -> Optional[Fact]:
        """Create a single synthesized fact from a cluster of REVIEW facts.

        For clusters of 5+, attempts LLM-based synthesis for richer output.
        Falls back to mechanical synthesis on any error or for smaller clusters.
        """
        if not cluster:
            return None

        # Pick highest-confidence as base
        cluster.sort(key=lambda f: f.metadata.confidence, reverse=True)
        base = cluster[0]

        # Accepted suggestions become validated patterns; everything else stays REVIEW
        kind = FactKind.PATTERN if group_key == "outcome:accepted" else FactKind.REVIEW

        # Try LLM synthesis for large clusters
        subject = None
        body_text = None
        avg_conf = sum(f.metadata.confidence for f in cluster) / len(cluster)

        if len(cluster) >= 5 and self._lm_adapter:
            llm_result = self._llm_synthesize(cluster, group_key)
            if llm_result:
                subject, body_text, llm_conf = llm_result
                avg_conf = llm_conf

        # Mechanical fallback
        if subject is None:
            subject = self._extract_common_subject(cluster)
        if body_text is None:
            seen_lines: set[str] = set()
            merged_body_parts: list[str] = []
            for fact in cluster:
                if len(merged_body_parts) >= 20:
                    break
                for line in fact.body.split("\n"):
                    stripped = line.strip()
                    if stripped and stripped not in seen_lines:
                        seen_lines.add(stripped)
                        merged_body_parts.append(stripped)
                    if len(merged_body_parts) >= 20:
                        break
            body_text = "\n".join(merged_body_parts[:20])

        embedding = self._embed_text(f"{subject} {body_text}")
        if embedding is None:
            embedding = base.embedding

        new_fact = Fact(
            subject=subject,
            body=body_text,
            kind=kind,
            scope=FactScope.PROJECT,
            org_id=self.org_id,
            project_id=self.project_id,
            metadata=FactMetadata(
                confidence=avg_conf,
                source_prompt=f"synthesized from {len(cluster)} REVIEW facts",
            ),
            embedding=embedding,
            tags=["synthesized"] + list({t for f in cluster for t in f.tags}),
        )

        # Supersede all source facts and cascade dependency reviews
        for fact in cluster:
            fact.is_valid = False
            fact.superseded_by = new_fact.id
            self._cascade_needs_review(fact.id)

        return new_fact

    def _llm_synthesize(
        self, cluster: list[Fact], group_key: str
    ) -> Optional[tuple[str, str, float]]:
        """Use LLM to synthesize a cluster into a concise actionable insight.

        Returns (subject, body, confidence) or None on any error.
        """
        if not self._lm_adapter:
            return None

        try:
            # Build compact prompt from cluster data
            facts_text = "\n".join(
                f"- [{f.subject}] {f.body[:200]}"
                for f in cluster[:10]  # Cap at 10 to keep prompt small
            )
            group_label = group_key.replace("outcome:", "")

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Distill these {len(cluster)} related {group_label} code observations "
                        "into one actionable insight.\n"
                        "Return exactly 3 lines:\n"
                        "SUBJECT: <concise label, max 60 chars>\n"
                        "BODY: <actionable guidance, 1-3 sentences>\n"
                        "CONFIDENCE: <0.0-1.0 based on consistency of evidence>"
                    ),
                },
                {"role": "user", "content": facts_text},
            ]

            response = self._lm_adapter.generate(
                messages=messages,
                max_tokens=300,
                temperature=0.3,
                reasoning_effort="low",  # Distilling clustered facts; not reasoning.
            )

            return self._parse_llm_synthesis(response)

        except Exception as e:
            logger.debug(f"LLM synthesis failed (falling back to mechanical): {e}")
            return None

    @staticmethod
    def _parse_llm_synthesis(response: str) -> Optional[tuple[str, str, float]]:
        """Parse the 3-line LLM response into (subject, body, confidence)."""
        subject = None
        body = None
        confidence = None

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("SUBJECT:"):
                subject = line.split(":", 1)[1].strip()[:60]
            elif line.upper().startswith("BODY:"):
                body = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass

        if subject and body and confidence is not None:
            return subject, body, confidence
        return None

    @staticmethod
    def _extract_common_subject(facts: list[Fact]) -> str:
        """Find a shared file or area across cluster members for the subject line."""
        import re
        # Match file-like tokens: must contain / or have a code file extension
        file_pattern = re.compile(r'(\S+/\S+|\S+\.(?:py|js|ts|tsx|go|rs|java|c|cpp|rb|swift))\b')
        file_mentions: Counter = Counter()
        for fact in facts:
            for match in file_pattern.findall(fact.subject):
                file_mentions[match] += 1

        if file_mentions:
            most_common = file_mentions.most_common(1)[0][0]
            return f"pattern: {most_common}"

        # Fallback: use first few words of subjects
        if facts:
            return f"pattern: {facts[0].subject[:50]}"
        return "pattern: synthesized"

    def _load_synthesis_watermark(self) -> int:
        """Load the count of REVIEW facts at last synthesis run."""
        if not self._project_path:
            return 0
        watermark_path = self._project_path.parent / f"synthesis_watermark_{self.project_id}.json"
        if not watermark_path.exists():
            return 0
        try:
            data = json.loads(watermark_path.read_text())
            return data.get("review_count", 0)
        except (json.JSONDecodeError, OSError):
            return 0

    def _save_synthesis_watermark(self, count: int) -> None:
        """Save the current REVIEW fact count as synthesis watermark."""
        if not self._project_path:
            return
        watermark_path = self._project_path.parent / f"synthesis_watermark_{self.project_id}.json"
        try:
            watermark_path.write_text(json.dumps({
                "review_count": count,
                "updated_at": time.time(),
            }))
        except OSError as e:
            logger.debug(f"Failed to save synthesis watermark: {e}")

    # ------------------------------------------------------------------ #
    # Migration
    # ------------------------------------------------------------------ #

    def _maybe_migrate(self) -> None:
        """Migrate from old PersistentReasoningMemory format if needed.

        Only runs when facts/ is empty but old files exist.
        """
        # Only migrate if we have no facts yet (excluding constraints)
        non_constraint_facts = [f for f in self._facts if f.kind != FactKind.CONSTRAINT]
        if non_constraint_facts:
            return

        # Check for old files
        old_global = Path.home() / ".neo" / "global_memory.json"
        old_local = None
        if self.project_id:
            old_local = Path.home() / ".neo" / f"local_{self.project_id}.json"

        if not old_global.exists() and (old_local is None or not old_local.exists()):
            return

        try:
            from neo.memory.migration import migrate_from_legacy
            migrated = migrate_from_legacy(
                old_global_path=old_global,
                old_local_path=old_local,
                org_id=self.org_id,
                project_id=self.project_id,
            )
            if migrated:
                self._facts.extend(migrated)
                self.save()
                logger.info(f"Migrated {len(migrated)} facts from legacy format")
        except Exception as e:
            logger.warning(f"Migration failed (non-fatal): {e}")
