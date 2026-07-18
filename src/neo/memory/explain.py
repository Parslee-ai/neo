"""Deterministic provenance explanations for Neo facts.

This module joins the durable FactStore with the per-task LearningEpisode
ledger. It performs no retrieval, embedding, or model call and never mutates
fact authority. The resulting JSON-compatible document is deliberately stable
enough for both the CLI and future local inspectors.
"""

from __future__ import annotations

from typing import Any, Optional

from neo.memory.episodes import LearningEpisode, LearningEpisodeStore
from neo.memory.models import Fact

EXPLANATION_SCHEMA_VERSION = 1


class FactLookupError(ValueError):
    """Raised when a fact identifier is missing or ambiguous."""


class LearningEpisodeCatalog:
    """Read-only aggregate over every bounded per-project episode store."""

    def __init__(self, base_dir):
        self.base_dir = base_dir

    def list(self) -> list[LearningEpisode]:
        if not self.base_dir.exists():
            return []
        episodes: list[LearningEpisode] = []
        for project_path in sorted(path for path in self.base_dir.iterdir() if path.is_dir()):
            episodes.extend(
                LearningEpisodeStore(project_path.name, base_dir=self.base_dir).list()
            )
        episodes.sort(key=lambda item: (item.started_at, item.episode_id))
        return episodes


def resolve_fact(facts: list[Fact], identifier: str) -> Fact:
    """Resolve an exact ID or an unambiguous ID prefix, including tombstones."""
    exact = [fact for fact in facts if fact.id == identifier]
    if exact:
        return exact[0]
    matches = [fact for fact in facts if fact.id.startswith(identifier)]
    if not matches:
        raise FactLookupError(f"fact not found: {identifier}")
    if len(matches) > 1:
        ids = ", ".join(sorted(fact.id for fact in matches)[:5])
        raise FactLookupError(f"ambiguous fact prefix {identifier!r}: {ids}")
    return matches[0]


def _fact_summary(fact: Fact) -> dict[str, Any]:
    return {
        "id": fact.id,
        "subject": fact.subject,
        "kind": fact.kind.value,
        "scope": fact.scope.value,
        "is_valid": fact.is_valid,
        "supersedes": fact.supersedes,
        "superseded_by": fact.superseded_by,
        "invalidation_reason": fact.invalidation_reason,
    }


def _episode_summary(episode: LearningEpisode) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "session_id": episode.session_id,
        "task_id": episode.task_id,
        "objective": episode.objective,
        "repository_revision": episode.repository_revision,
        "final_outcome": episode.final_outcome,
    }


def _supersession_chain(fact: Fact, by_id: dict[str, Fact]) -> dict[str, list[dict[str, Any]]]:
    previous: list[dict[str, Any]] = []
    seen = {fact.id}
    cursor = fact
    while cursor.supersedes and cursor.supersedes not in seen:
        prior = by_id.get(cursor.supersedes)
        if prior is None:
            previous.append({"id": cursor.supersedes, "missing": True})
            break
        previous.append(_fact_summary(prior))
        seen.add(prior.id)
        cursor = prior

    replacements: list[dict[str, Any]] = []
    cursor = fact
    while cursor.superseded_by and cursor.superseded_by not in seen:
        replacement = by_id.get(cursor.superseded_by)
        if replacement is None:
            replacements.append({"id": cursor.superseded_by, "missing": True})
            break
        replacements.append(_fact_summary(replacement))
        seen.add(replacement.id)
        cursor = replacement
    return {"previous": previous, "replacements": replacements}


def explain_fact(
    facts: list[Fact],
    identifier: str,
    *,
    episode_store: Optional[LearningEpisodeStore] = None,
) -> dict[str, Any]:
    """Build a complete local explanation for one durable or invalid fact."""
    fact = resolve_fact(facts, identifier)
    episodes = episode_store.list() if episode_store is not None else []
    episode_by_id = {episode.episode_id: episode for episode in episodes}

    retrieval_history: list[dict[str, Any]] = []
    mutation_history: list[dict[str, Any]] = []
    related_episode_ids = set(fact.supporting_episode_ids)
    related_episode_ids.update(fact.contradicting_episode_ids)

    for episode in episodes:
        for evidence in episode.retrieved_facts:
            if evidence.fact_id != fact.id:
                continue
            retrieval_history.append({
                **_episode_summary(episode),
                "score": evidence.score,
                "included_in_context": evidence.included_in_context,
                "used_in_reasoning": evidence.used_in_reasoning,
                "outcome_association": evidence.outcome_association,
                "reason": (
                    "ranked for this task and included in final context"
                    if evidence.included_in_context
                    else "ranked for this task but not included in final context"
                ),
            })
            related_episode_ids.add(episode.episode_id)
        for mutation in episode.memory_mutations:
            if mutation.fact_id != fact.id:
                continue
            mutation_history.append({
                **_episode_summary(episode),
                "mutation_id": mutation.mutation_id,
                "operation": mutation.operation,
                "reason": mutation.reason,
                "before_state": mutation.before_state,
                "after_state": mutation.after_state,
            })
            related_episode_ids.add(episode.episode_id)

    def evidence_rows(ids: list[str], relationship: str) -> list[dict[str, Any]]:
        rows = []
        for episode_id in ids:
            episode = episode_by_id.get(episode_id)
            if episode is None:
                rows.append({
                    "episode_id": episode_id,
                    "relationship": relationship,
                    "missing": True,
                })
                continue
            rows.append({
                **_episode_summary(episode),
                "relationship": relationship,
                "verification": [
                    {
                        "kind": item.kind,
                        "status": item.status,
                        "tool_name": item.tool_name,
                        "summary": item.summary,
                        "repository_revision": item.repository_revision,
                    }
                    for item in episode.verification
                ],
                "outcome_details": episode.outcome_details,
            })
        return rows

    related_episodes = []
    for episode_id in sorted(related_episode_ids):
        episode = episode_by_id.get(episode_id)
        related_episodes.append(
            _episode_summary(episode)
            if episode is not None
            else {"episode_id": episode_id, "missing": True}
        )

    by_id = {item.id: item for item in facts}
    return {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "fact": {
            **_fact_summary(fact),
            "body": fact.body,
            "domain": fact.domain,
            "org_id": fact.org_id,
            "project_id": fact.project_id,
            "tags": fact.tags,
            "provenance": fact.metadata.provenance,
            "source_file": fact.metadata.source_file,
            "source_prompt": fact.metadata.source_prompt,
            "source_candidate_id": fact.source_candidate_id,
            "confidence": fact.metadata.confidence,
            "success_count": fact.metadata.success_count,
            "effectiveness_c": fact.metadata.effectiveness_c,
            "effectiveness_n": fact.metadata.effectiveness_n,
            "effectiveness": fact.metadata.effectiveness_f,
            "access_count": fact.metadata.access_count,
            "recall_count": fact.metadata.recall_count,
        },
        "supporting_evidence": evidence_rows(fact.supporting_episode_ids, "supports"),
        "contradicting_evidence": evidence_rows(
            fact.contradicting_episode_ids, "contradicts"
        ),
        "retrieval_history": retrieval_history,
        "mutation_history": mutation_history,
        "related_episodes": related_episodes,
        "supersession": _supersession_chain(fact, by_id),
        "model_calls": 0,
    }
