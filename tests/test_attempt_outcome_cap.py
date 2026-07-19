"""Reproduction test for the attempt-outcome memory flood.

A stuck agent loop that keeps reporting the same failed attempt makes Neo persist
a fresh, high-confidence EPISODE fact every iteration. EPISODE facts bypass
dedup/supersession by design, so without a cap the flood fills the PROJECT scope
budget and the scope-limit evictor discards the lower-confidence REVIEW fact that
feeds synthesis, while the redundant 0.7 attempt episodes survive.

Two claims, each RED on the unpatched code and GREEN with the cap:
  (a) the attempt-outcome facts are bounded FAR below the flood — not merely at
      the scope limit;
  (b) a pre-existing, more-valuable REVIEW fact SURVIVES the flood.

The PROJECT scope budget is monkeypatched to a small value so the eviction harm
in (b) is reproduced quickly (the real budget is 500; driving past it unpatched
is correct but slow). The loop drives more iterations than that budget on purpose
so the unpatched evictor actually removes the REVIEW fact — making (b) genuinely
discriminating, and keeping the RED a behavioural failure, not an ImportError on
the fix's own constant.
"""

import neo.memory.store as store_mod
from neo.execution_context import (
    AttemptContext,
    OutcomeContext,
    resolve_execution_context,
)
from neo.memory.models import FactKind, FactScope
from neo.memory.store import FactStore
from neo.models import NeoInput

_SMALL_PROJECT_BUDGET = 100


def _flood(store, iterations):
    context = resolve_execution_context(NeoInput(
        prompt="Why is this timeout happening?",
        attempt=AttemptContext("Added a retry"),
        outcome=OutcomeContext(
            "failed", summary="Duplicates appeared", side_effects=["duplicate writes"],
        ),
    )).to_dict()
    for i in range(iterations):
        store.persist_attempt_outcome(
            execution_context=context,
            learning_episode_id=f"episode-{i}",
            repository_revision=f"rev{i}",
        )


def test_attempt_flood_is_capped_and_spares_review(tmp_path, monkeypatch):
    # Shrink the PROJECT scope budget so the eviction harm reproduces fast.
    monkeypatch.setitem(
        store_mod.SCOPE_LIMITS, FactScope.PROJECT.value, _SMALL_PROJECT_BUDGET
    )

    store = FactStore(
        codebase_root=str(tmp_path),
        eager_init=False,
        facts_dir=tmp_path / "facts",
        episodes_dir=tmp_path / "episodes",
        emit_metrics=False,
    )

    # A genuinely useful, lower-confidence REVIEW fact that feeds synthesis.
    review = store.add_fact(
        subject="Prefer retry-with-backoff for flaky network calls",
        body="Observed pattern worth synthesizing.",
        kind=FactKind.REVIEW,
        scope=FactScope.PROJECT,
        confidence=0.4,
    )

    # Drive MORE iterations than the (patched) PROJECT scope budget so the
    # unpatched (uncapped) code is forced to evict — and by eviction score the
    # 0.4 REVIEW fact goes before the 0.7 attempt episodes.
    iterations = _SMALL_PROJECT_BUDGET + 140
    _flood(store, iterations)

    valid_attempts = [
        f for f in store._facts
        if f.is_valid and "attempt" in f.tags and "observed-outcome" in f.tags
    ]

    # (a) BEHAVIOURAL red: bounded FAR below the flood. Unpatched, the only bound
    # is the scope limit, so this count sits near the budget (>> 50); with the cap
    # it sits at MAX_ATTEMPT_OUTCOME_FACTS (25). A value-tolerant threshold of 50
    # fails on base and passes on the fix without hard-coding the exact cap — and
    # is a behavioural assertion, not an ImportError on the fix's own constant.
    assert len(valid_attempts) <= 50, (
        f"attempt-outcome facts not capped: {len(valid_attempts)} valid "
        f"(unpatched code is bounded only by the scope limit)"
    )

    # Tighter check only when the cap constant exists (fix side); guarded so the
    # RED on base is the behavioural assertion above, never an import failure.
    try:
        from neo.memory.store import MAX_ATTEMPT_OUTCOME_FACTS
    except ImportError:
        MAX_ATTEMPT_OUTCOME_FACTS = None
    if MAX_ATTEMPT_OUTCOME_FACTS is not None:
        assert len(valid_attempts) <= MAX_ATTEMPT_OUTCOME_FACTS

    # (b) DISCRIMINATING harm: the valuable REVIEW fact survives the flood. On the
    # unpatched code the scope evictor removes it (lowest eviction score, evicted
    # once the flood pushes the PROJECT scope past its budget); with the cap the
    # scope never fills, so it survives.
    survivor = next((f for f in store._facts if f.id == review.id), None)
    assert survivor is not None and survivor.is_valid is True, (
        "the pre-existing REVIEW fact was evicted by the uncapped attempt flood"
    )
