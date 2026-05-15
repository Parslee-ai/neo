"""
SCM-style 4-D ValueTagger for facts (paper 2604.20943 §3).

Composite importance score per fact:

    I(c) = 0.30 · v_novelty
         + 0.20 · v_validation   (Neo substitute for v_emotional)
         + 0.35 · v_task
         + 0.15 · v_repetition

Each sub-score is in [0, 1]. Components:

  v_novelty    — 1 − max cosine of this fact to all OTHER valid facts.
                 Unique facts score high; near-duplicates score low.
  v_validation — success_count / (access_count + 1). Surrogate for the
                 paper's v_emotional (emotional weight is meaningless in a
                 code-fact context; "the user kept using this" is the
                 best Neo-shape analog).
  v_task       — average cosine to the "active corpus" (recently-accessed
                 facts). Captures relevance-to-current-interest.
  v_repetition — access_count / (global_max_access + 1).

Adaptive forgetting threshold (paper §3.6, with the corrected β_2 = 0.2):

    θ_f = μ_I − σ_I · (|G| / target_size), clip to ≥ 0.05

Facts below θ_f are pruning candidates. target_size defaults to 2000 —
Neo's documented soft cap.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from neo.math_utils import batched_cosine
from neo.memory.models import Fact

# Composite weights from paper 2604.20943 §3.4 (sum to 1.0).
W_NOVELTY = 0.30
W_VALIDATION = 0.20  # Neo substitute for v_emotional
W_TASK = 0.35
W_REPETITION = 0.15

# Adaptive forgetting (paper §3.6, β_2 = 0.2 per the paper's own bug-fix).
FORGETTING_FLOOR = 0.05
DEFAULT_TARGET_SIZE = 2000

# How many most-recently-accessed facts define the "active corpus" for
# v_task. Small enough to be specific, large enough to be stable.
ACTIVE_CORPUS_SIZE = 20


def _novelty(fact: Fact, others_embeddings: Sequence[np.ndarray]) -> float:
    """1 − max cosine(fact, other). 0 when this is a duplicate; 1 when unique."""
    if fact.embedding is None or not others_embeddings:
        return 0.5
    sims = batched_cosine(list(others_embeddings), fact.embedding, default=0.0)
    if not sims:
        return 1.0
    return max(0.0, 1.0 - max(sims))


def _validation(fact: Fact) -> float:
    """success_count / (access_count + 1), clamped to [0, 1]."""
    s = max(0, fact.metadata.success_count)
    a = max(0, fact.metadata.access_count)
    return min(1.0, s / (a + 1))


def _task(fact: Fact, active_corpus_centroid: np.ndarray | None) -> float:
    """Cosine to the centroid of recently-accessed facts; 0.5 when no signal."""
    if fact.embedding is None or active_corpus_centroid is None:
        return 0.5
    sims = batched_cosine([active_corpus_centroid], fact.embedding, default=0.5)
    return max(0.0, min(1.0, sims[0]))


def _repetition(fact: Fact, max_access: int) -> float:
    """access_count / (max_access + 1)."""
    a = max(0, fact.metadata.access_count)
    return min(1.0, a / (max_access + 1))


def _active_corpus_centroid(facts: Sequence[Fact]) -> np.ndarray | None:
    """Mean of the top ACTIVE_CORPUS_SIZE most-recently-accessed embeddings."""
    candidates = [f for f in facts if f.embedding is not None and f.is_valid]
    candidates.sort(key=lambda f: f.metadata.last_accessed, reverse=True)
    if not candidates:
        return None
    top = candidates[:ACTIVE_CORPUS_SIZE]
    matrix = np.asarray([f.embedding for f in top], dtype=np.float32)
    return matrix.mean(axis=0)


def compute_value(fact: Fact, all_facts: Sequence[Fact]) -> float:
    """Composite value score I(c) for one fact in the context of all_facts."""
    others = [f.embedding for f in all_facts if f is not fact and f.embedding is not None]
    max_access = max((f.metadata.access_count for f in all_facts), default=0)
    centroid = _active_corpus_centroid(all_facts)

    return (
        W_NOVELTY * _novelty(fact, others)
        + W_VALIDATION * _validation(fact)
        + W_TASK * _task(fact, centroid)
        + W_REPETITION * _repetition(fact, max_access)
    )


def compute_value_scores(facts: Sequence[Fact]) -> dict[str, float]:
    """Batch compute I(c) for every fact. Returns id → score."""
    if not facts:
        return {}
    max_access = max((f.metadata.access_count for f in facts), default=0)
    centroid = _active_corpus_centroid(facts)

    scores: dict[str, float] = {}
    for fact in facts:
        others = [f.embedding for f in facts if f is not fact and f.embedding is not None]
        scores[fact.id] = (
            W_NOVELTY * _novelty(fact, others)
            + W_VALIDATION * _validation(fact)
            + W_TASK * _task(fact, centroid)
            + W_REPETITION * _repetition(fact, max_access)
        )
    return scores


def forgetting_threshold(
    scores: Iterable[float],
    *,
    corpus_size: int,
    target_size: int = DEFAULT_TARGET_SIZE,
) -> float:
    """θ_f = μ_I − σ_I · (|G| / target_size), clipped to [FORGETTING_FLOOR, 1].

    When |G| ≤ target_size the threshold is at the mean — half the facts
    are below. When |G| ≫ target_size it lifts above the mean, pruning
    aggressively. The floor stops the threshold collapsing to ~0 in tiny
    corpora.
    """
    arr = np.asarray(list(scores), dtype=np.float64)
    if arr.size == 0:
        return FORGETTING_FLOOR
    mu = float(arr.mean())
    sigma = float(arr.std())
    ratio = corpus_size / max(1, target_size)
    theta = mu - sigma * ratio
    if not math.isfinite(theta):
        return FORGETTING_FLOOR
    return max(FORGETTING_FLOOR, min(1.0, theta))
