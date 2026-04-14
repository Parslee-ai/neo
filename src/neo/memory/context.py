"""
Context assembly for LLM prompt injection.

Filters and ranks facts into StateBench's four-layer model,
then renders them as a formatted string for prompt injection.
"""

import logging
from typing import Optional

import numpy as np

from neo.math_utils import cosine_similarity
from neo.memory.models import ContextResult, Fact, FactKind, FactScope, success_bonus

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
        constraints = self._accumulate_within_budget(constraints, constraint_cap, at_least_one=True)
        constraint_tokens = sum(f.size_hint() for f in constraints)
        if uncapped_total > constraint_cap:
            logger.warning(
                "Constraints would consume %d tokens (cap %d); truncated to %d tokens",
                uncapped_total, constraint_cap, constraint_tokens,
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
        )

    @staticmethod
    def _accumulate_within_budget(
        facts: list[Fact], budget: int, *, at_least_one: bool = False,
    ) -> list[Fact]:
        """Accumulate facts until budget is exhausted.

        Args:
            at_least_one: If True, always include the first fact even if it
                exceeds the budget. Only used for valid_facts (primary layer).
        """
        result: list[Fact] = []
        used = 0
        for fact in facts:
            cost = fact.size_hint()
            if used + cost > budget:
                if not result and at_least_one:
                    result.append(fact)
                break
            result.append(fact)
            used += cost
        return result

    def _score_facts(
        self,
        facts: list[Fact],
        query_embedding: Optional[np.ndarray],
    ) -> list[tuple[Fact, float]]:
        """Score facts by cosine similarity * confidence + outcome bonus.

        Shares the ranking policy with FactStore.retrieve_relevant via
        memory.models.success_bonus so the two retrieval paths stay consistent.
        """
        scored: list[tuple[Fact, float]] = []

        for fact in facts:
            sim = 0.5
            if query_embedding is not None and fact.embedding is not None:
                sim = self._cosine_similarity(query_embedding, fact.embedding)

            confidence = fact.metadata.confidence
            score = sim * confidence + success_bonus(fact.metadata.success_count)
            scored.append((fact, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return cosine_similarity(a, b)

    def format_context_for_prompt(self, ctx: ContextResult) -> str:
        """Render ContextResult as a formatted string for LLM injection.

        Superseded facts are shown as inline change annotations on the
        facts that replaced them, rather than in a separate section.
        """
        sections: list[str] = []

        # Build lookup: old_fact.id → old_fact for inline annotations
        old_lookup: dict[str, Fact] = {f.id: f for f in ctx.invalidated_facts}

        if ctx.constraints:
            lines = ["## Project Constraints"]
            for fact in ctx.constraints:
                lines.append(f"### {fact.subject}")
                lines.append(fact.body)
            sections.append("\n".join(lines))

        if ctx.valid_facts:
            lines = ["## Relevant Knowledge"]
            for fact in ctx.valid_facts:
                conf = fact.metadata.confidence
                line = (
                    f"- **{fact.subject}** ({fact.kind.value}, confidence={conf:.2f}): "
                    f"{fact.body[:200]}"
                )
                # Inline change annotation
                if fact.supersedes and fact.supersedes in old_lookup:
                    old = old_lookup[fact.supersedes]
                    line += f" (changed from: {old.body[:80]})"
                lines.append(line)
            sections.append("\n".join(lines))

        if ctx.known_unknowns:
            lines = ["## Known Unknowns"]
            for fact in ctx.known_unknowns:
                lines.append(f"- {fact.subject}: {fact.body[:150]}")
            sections.append("\n".join(lines))

        if ctx.working_set:
            lines = ["## Session Context"]
            for fact in ctx.working_set:
                lines.append(f"- {fact.subject}: {fact.body[:200]}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)
