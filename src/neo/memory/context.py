"""
Context assembly for LLM prompt injection.

Filters and ranks facts into StateBench's four-layer model,
then renders them as a formatted string for prompt injection.
"""

import logging
import time
from typing import Optional

import numpy as np

from neo.memory.models import ContextResult, Fact, FactKind, FactScope

logger = logging.getLogger(__name__)

# Maximum invalidated facts to include for contrast
MAX_INVALIDATED_FACTS = 3


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
    ) -> ContextResult:
        """Filter and rank facts into layers.

        Args:
            facts: All facts from the store.
            query: The current query string.
            query_embedding: Embedding vector for the query (optional).
            environment: Git state and other environment info.
            k: Maximum number of valid facts to include.

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

        # Rank valid facts by similarity + confidence + recency
        scored_valid = self._score_facts(valid_candidates, query_embedding)
        top_valid = [f for f, _ in scored_valid[:k]]

        # Take most recently superseded for contrast
        invalidated.sort(key=lambda f: f.metadata.last_accessed, reverse=True)
        top_invalidated = invalidated[:MAX_INVALIDATED_FACTS]

        return ContextResult(
            constraints=constraints,
            valid_facts=top_valid,
            invalidated_facts=top_invalidated,
            working_set=session_facts,
            environment=environment or {},
            known_unknowns=known_unknowns,
        )

    def _score_facts(
        self,
        facts: list[Fact],
        query_embedding: Optional[np.ndarray],
    ) -> list[tuple[Fact, float]]:
        """Score facts by cosine similarity * confidence * recency.

        Returns sorted list of (fact, score) tuples, highest first.
        """
        scored: list[tuple[Fact, float]] = []
        now = time.time()

        for fact in facts:
            # Cosine similarity component
            sim = 0.5  # Default when no embedding available
            if query_embedding is not None and fact.embedding is not None:
                sim = self._cosine_similarity(query_embedding, fact.embedding)

            # Confidence component
            confidence = fact.metadata.confidence

            # Recency factor: half-life of 30 days
            age_days = (now - fact.metadata.last_accessed) / 86400
            recency = 0.5 ** (age_days / 30)

            score = sim * confidence * (0.5 + 0.5 * recency)
            scored.append((fact, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def format_context_for_prompt(self, ctx: ContextResult) -> str:
        """Render ContextResult as a formatted string for LLM injection.

        Produces a structured block that can be inserted into system
        or user prompts.
        """
        sections: list[str] = []

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
                lines.append(
                    f"- **{fact.subject}** ({fact.kind.value}, confidence={conf:.2f}): "
                    f"{fact.body[:200]}"
                )
            sections.append("\n".join(lines))

        if ctx.invalidated_facts:
            lines = ["## Recently Changed (for context)"]
            for fact in ctx.invalidated_facts:
                lines.append(
                    f"- ~~{fact.subject}~~ (superseded): {fact.body[:150]}"
                )
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
