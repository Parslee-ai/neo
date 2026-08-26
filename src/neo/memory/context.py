"""
Context assembly for LLM prompt injection.

Filters and ranks facts into StateBench's four-layer model,
then renders them as a formatted string for prompt injection.
"""

import logging
import time
from typing import Optional

import numpy as np

from neo.math_utils import batched_cosine, cosine_similarity
from neo.memory.models import ContextResult, Fact, FactKind, FactScope, rank_score

logger = logging.getLogger(__name__)

class ContextAssembler:
    """Assembles a ContextResult from facts and query context.

    Layer ordering (following StateBench's winning approach):
    1. Constraints - sorted by scope (global first), always included
    2. Valid facts - ranked by cosine similarity, weighted by confidence and recency
    3. Invalidated facts - most recently superseded, capped at 3
    4. Known unknowns - facts with kind=KNOWN_UNKNOWN
    5. Environment - passed through as-is
    """

    def assemble(
        self,
        facts: list[Fact],
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        environment: Optional[dict] = None,
        k: int = 5,
        max_tokens: int = 12000,
    ) -> ContextResult:
        """Filter and rank facts into layers with token budget enforcement.

        Constraints are capped to 2/3 of the budget to reserve room for other
        layers. The remaining budget is shared across valid_facts, working_set,
        and known_unknowns in priority order.

        Args:
            facts: All facts from the store.
            query: The current query string.
            query_embedding: Embedding vector for the query (optional).
            environment: Git state and other environment info.
            k: Maximum number of valid facts to include.
            max_tokens: Token budget for non-constraint layers.

        Returns:
            ContextResult with facts organized into layers.
        """
        constraints: list[Fact] = []
        valid_candidates: list[Fact] = []
        invalidated: list[Fact] = []
        session_facts: list[Fact] = []
        known_unknowns: list[Fact] = []

        for fact in facts:
            if fact.kind == FactKind.CONSTRAINT and fact.is_valid:
                constraints.append(fact)
            elif fact.kind == FactKind.KNOWN_UNKNOWN and fact.is_valid:
                known_unknowns.append(fact)
            elif fact.scope == FactScope.SESSION and fact.is_valid:
                session_facts.append(fact)
            elif fact.is_valid:
                valid_candidates.append(fact)
            elif not fact.is_valid and fact.superseded_by:
                invalidated.append(fact)

        # Sort constraints: global first, then org, then project
        scope_order = {FactScope.GLOBAL: 0, FactScope.ORG: 1, FactScope.PROJECT: 2, FactScope.SESSION: 3}
        constraints.sort(key=lambda f: scope_order.get(f.scope, 99))

        # Cap constraints so they don't starve other layers.
        # Reserve at least 1/3 of budget for non-constraint content.
        constraint_cap = max_tokens * 2 // 3
        uncapped_total = sum(f.size_hint() for f in constraints)
        uncapped_count = len(constraints)

        # WHEN CONSTRAINTS OVERFLOW, RELEVANCE DECIDES WHICH SURVIVE.
        #
        # Scope order alone is a stable sort, so within a scope the order is
        # whatever the store yielded — effectively creation order. Combined
        # with a prefix cut that stops at the first fact that does not fit,
        # that made the injected set "the globals, plus the OLDEST project
        # constraints until the budget fills". Measured on a real store: 2,445
        # valid constraints against an 8,000-token cap, so ~1.4% were injected
        # and the newest were structurally unreachable no matter how well they
        # matched the query. The highest-authority memory layer was the only
        # one not consulting the query, while ordinary facts were ranked by
        # similarity right below (see _score_facts).
        #
        # Scope still leads: globals and org constraints are few and are
        # deliberately authoritative. Ranking applies WITHIN each scope, and
        # only matters at all once the layer overflows.
        if uncapped_total > constraint_cap and len(constraints) > 1:
            constraints = self._rank_constraints_by_scope_then_relevance(
                constraints, scope_order, query_embedding,
            )

        constraints = self._accumulate_within_budget(
            constraints, constraint_cap, at_least_one=True, skip_oversized=True,
        )
        constraint_tokens = sum(f.size_hint() for f in constraints)
        if uncapped_total > constraint_cap:
            logger.warning(
                "Constraints would consume %d tokens (cap %d); kept the %d most "
                "relevant of %d (%d tokens)",
                uncapped_total, constraint_cap, len(constraints),
                uncapped_count, constraint_tokens,
            )

        # Rank valid facts by similarity + confidence + recency
        scored_valid = self._score_facts(valid_candidates, query_embedding)

        # Budget-aware accumulation.
        # Valid facts get "at least one" guarantee; subsequent layers
        # only get what's left (no guarantee if budget is exhausted).
        budget_remaining = max(0, max_tokens - constraint_tokens)
        top_valid = self._accumulate_within_budget(
            [f for f, _ in scored_valid[:k]], budget_remaining, at_least_one=True,
        )
        budget_remaining = max(0, budget_remaining - sum(f.size_hint() for f in top_valid))

        session_capped = self._accumulate_within_budget(session_facts, budget_remaining)
        budget_remaining = max(0, budget_remaining - sum(f.size_hint() for f in session_capped))

        unknowns_capped = self._accumulate_within_budget(known_unknowns, budget_remaining)

        # Keep full invalidated list for annotation lookup.
        # Sorted by last_accessed as proxy for supersession time (no superseded_at field).
        invalidated.sort(key=lambda f: f.metadata.last_accessed, reverse=True)

        return ContextResult(
            constraints=constraints,
            valid_facts=top_valid,
            invalidated_facts=invalidated,
            working_set=session_capped,
            environment=environment or {},
            known_unknowns=unknowns_capped,
            retrieval_scores={f.id: score for f, score in scored_valid},
        )

    @staticmethod
    def _accumulate_within_budget(
        facts: list[Fact], budget: int, *, at_least_one: bool = False,
        skip_oversized: bool = False,
    ) -> list[Fact]:
        """Accumulate facts until budget is exhausted.

        Args:
            at_least_one: If True, always include the first fact even if it
                exceeds the budget. Only used for valid_facts (primary layer).
            skip_oversized: If True, a fact that does not fit is SKIPPED and
                accumulation continues. Default False stops at the first
                non-fitting fact, which is correct for a relevance-ordered
                list where everything after it ranks lower anyway.

                Constraints pass True: they are ordered by scope first, so a
                single large global constraint would otherwise truncate every
                project constraint behind it regardless of relevance — one
                verbose fact silently emptying the layer.
        """
        result: list[Fact] = []
        used = 0
        for fact in facts:
            cost = fact.size_hint()
            if used + cost > budget:
                if skip_oversized:
                    # Skip THIS fact, keep going. Deliberately does not consume
                    # budget for something that was not included.
                    continue
                if not result and at_least_one:
                    result.append(fact)
                break
            result.append(fact)
            used += cost

        # The "at least one" guarantee is a floor, not a priority claim. Under
        # skip_oversized it applies only when nothing fit at all — otherwise a
        # single oversized fact at the front (a verbose global constraint, say)
        # would be admitted over every smaller fact behind it and blow the cap
        # that exists to protect the other layers.
        if not result and at_least_one and facts:
            result.append(facts[0])
        return result

    def _rank_constraints_by_scope_then_relevance(
        self,
        constraints: list[Fact],
        scope_order: dict,
        query_embedding: Optional[np.ndarray],
    ) -> list[Fact]:
        """Order constraints by scope tier, then by relevance within the tier.

        With no query embedding there is nothing to rank on, so the incoming
        scope order is returned untouched — the pre-existing behaviour, not a
        silent fallback to something else.
        """
        if query_embedding is None:
            return constraints

        scored = self._score_facts(constraints, query_embedding)
        rank = {id(fact): position for position, (fact, _) in enumerate(scored)}
        return sorted(
            constraints,
            key=lambda f: (
                scope_order.get(f.scope, 99),
                rank.get(id(f), len(constraints)),
            ),
        )

    def _score_facts(
        self,
        facts: list[Fact],
        query_embedding: Optional[np.ndarray],
    ) -> list[tuple[Fact, float]]:
        """Score facts by sim * confidence + success_bonus + provenance_bonus.

        Vectorized cosine in one numpy pass, then rank_score per fact.
        Shares the ranking policy with FactStore.retrieve_relevant via
        memory.models.rank_score so the two retrieval paths stay consistent.
        """
        if not facts:
            return []

        now = time.time()
        sims = batched_cosine([f.embedding for f in facts], query_embedding)
        scored = [(f, rank_score(f, s, now)) for f, s in zip(facts, sims)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors. Kept for callers."""
        return cosine_similarity(a, b)

    def format_context_for_prompt(self, ctx: ContextResult) -> str:
        """Render ContextResult as a formatted string for LLM injection.

        Superseded facts are shown as inline change annotations on the
        facts that replaced them, rather than in a separate section.
        """
        sections: list[str] = []

        # Build lookup: old_fact.id → old_fact for inline annotations
        old_lookup: dict[str, Fact] = {f.id: f for f in ctx.invalidated_facts}

        # Use ``Fact.render_for_context()`` so a fact with an explicit
        # context_text (the MMS-style narrative form for episodes etc.)
        # gets that content, while plain facts fall back to subject+body.
        def _body_for_context(fact: Fact, limit: int) -> str:
            text = fact.context_text or fact.body
            return text[:limit] if limit > 0 and len(text) > limit else text

        if ctx.constraints:
            lines = ["## Project Constraints"]
            for fact in ctx.constraints:
                lines.append(f"### {fact.subject}")
                lines.append(fact.context_text or fact.body)
            sections.append("\n".join(lines))

        if ctx.valid_facts:
            lines = [
                "## Relevant Knowledge",
                "When a fact materially influences the answer, cite its "
                "[fact:<id>] inline AND list every id you used on a final line "
                "exactly as `Facts used: [<id>, <id>]` (use `Facts used: []` if "
                "none). This attribution is how neo learns which knowledge helps.",
            ]
            for fact in ctx.valid_facts:
                conf = fact.metadata.confidence
                line = (
                    f"- [fact:{fact.id}] **{fact.subject}** "
                    f"({fact.kind.value}, confidence={conf:.2f}): "
                    f"{_body_for_context(fact, 200)}"
                )
                # Inline change annotation. The `in old_lookup` guard is
                # load-bearing, not a style choice: purge_dead_facts reclaims
                # cold superseded tombstones, so a live fact's `supersedes` may
                # dangle. Membership-test, never a bare subscript.
                if fact.supersedes and fact.supersedes in old_lookup:
                    old = old_lookup[fact.supersedes]
                    line += f" (changed from: {_body_for_context(old, 80)})"
                lines.append(line)
            sections.append("\n".join(lines))

        if ctx.known_unknowns:
            lines = ["## Known Unknowns"]
            for fact in ctx.known_unknowns:
                lines.append(f"- {fact.subject}: {_body_for_context(fact, 150)}")
            sections.append("\n".join(lines))

        if ctx.working_set:
            lines = ["## Session Context"]
            for fact in ctx.working_set:
                lines.append(f"- {fact.subject}: {_body_for_context(fact, 200)}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
