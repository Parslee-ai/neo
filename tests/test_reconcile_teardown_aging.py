"""Reproduction tests for issue #9004.

The reconcile teardown that retracts an under-supported GLOBAL episode-derived
fact must fire ONLY on an active retraction (a candidate downgraded to
'contradicted'), never merely because supporting episodes aged out of a busy
project's bounded ring buffer. When it does fire, it must not permanently bar
the same pattern from being re-learned, and it must never block an unrelated
project from minting its own local project fact.

Each test asserts the correct behavior and therefore FAILS on the buggy code.
"""

from unittest.mock import patch

from neo.memory.models import FactScope
from neo.memory.outcomes import Outcome, OutcomeType
from neo.memory.store import FactStore

# A subject with NO file-path bracket, so the path-bearing project
# (_episode_signature) and path-agnostic global (_global_signature) identities
# coincide — needed to exercise the cross-project block (finding 3).
SUBJECT = "pattern: escape shell args"
BODY = "escape shell arguments before exec"


def _make_project(tmp_path, name, facts_dir, episodes_dir):
    root = tmp_path / name
    root.mkdir()
    return FactStore(
        codebase_root=str(root), eager_init=False,
        facts_dir=facts_dir, episodes_dir=episodes_dir,
    )


def _feed(store, episodes_dir, prefix, n, otype):
    from neo.memory.episodes import LearningEpisode, LearningEpisodeStore, MemoryCandidateEvidence

    es = LearningEpisodeStore(store.project_id, base_dir=episodes_dir)
    for i in range(n):
        ep_id, cid, sid = f"{prefix}-{i}", f"{prefix}c{i}", f"{prefix}s{i}"
        ep = LearningEpisode(episode_id=ep_id, project_id=store.project_id)
        ep.memory_candidates.append(MemoryCandidateEvidence(
            candidate_id=cid, suggestion_id=sid,
            subject=SUBJECT, body=BODY, kind="pattern"))
        es.save(ep)
        outcome = Outcome(
            outcome_type=otype, file_path="src/sh.py",
            suggestion_id=sid, learning_episode_id=ep_id, candidate_id=cid,
            candidate_subject=SUBJECT, candidate_body=BODY, candidate_kind="pattern")
        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=([outcome], {})):
            store.detect_implicit_feedback({"prompt": "n"}, [])


def _valid_globals(store):
    return [
        f for f in store.entries
        if f.is_valid and f.scope == FactScope.GLOBAL and "episode-derived" in f.tags
    ]


def test_aged_out_support_does_not_tear_down_healthy_global(tmp_path):
    """Finding 1: a healthy global must survive when a busy project's supporting
    episodes simply age out of the bounded ring buffer (no retraction)."""
    facts_dir = tmp_path / "facts"
    episodes_dir = tmp_path / "episodes"

    a = _make_project(tmp_path, "A", facts_dir, episodes_dir)
    _feed(a, episodes_dir, "acc", 2, OutcomeType.ACCEPTED)
    b = _make_project(tmp_path, "B", facts_dir, episodes_dir)
    _feed(b, episodes_dir, "bacc", 2, OutcomeType.ACCEPTED)  # A+B mint the global
    assert len(_valid_globals(b)) == 1

    # Simulate A's supporting episodes aging out of the bounded per-project
    # buffer (LearningEpisodeStore._enforce_limit unlinks the oldest). NO
    # retraction has occurred.
    from neo.memory.episodes import LearningEpisodeStore
    a_episode_dir = LearningEpisodeStore(a.project_id, base_dir=episodes_dir).path
    for stale in a_episode_dir.glob("*.json"):
        stale.unlink()

    # On the next reconcile the live tally shows only project B (below the bar),
    # but nobody retracted anything — the global must NOT be torn down.
    b.reconcile_cross_project_promotions()
    assert len(_valid_globals(b)) == 1, "healthy global was deleted after episodes aged out"


def test_torn_down_global_can_be_relearned_with_fresh_evidence(tmp_path):
    """Finding 2: after a real contradiction tears a global down, fresh strong
    cross-project evidence must be able to re-mint it."""
    facts_dir = tmp_path / "facts"
    episodes_dir = tmp_path / "episodes"

    a = _make_project(tmp_path, "A", facts_dir, episodes_dir)
    _feed(a, episodes_dir, "acc", 2, OutcomeType.ACCEPTED)
    b = _make_project(tmp_path, "B", facts_dir, episodes_dir)
    _feed(b, episodes_dir, "bacc", 2, OutcomeType.ACCEPTED)  # A+B mint the global
    assert len(_valid_globals(b)) == 1

    _feed(a, episodes_dir, "corr", 2, OutcomeType.MODIFIED)  # A actively retracts

    # Real contradiction present -> reconcile tears the under-supported global down.
    b.reconcile_cross_project_promotions()
    assert _valid_globals(b) == []

    # Fresh strong evidence: a brand-new project C accepts the same pattern twice,
    # and B still holds its two supporting episodes -> 2 projects / 4 episodes.
    c = _make_project(tmp_path, "C", facts_dir, episodes_dir)
    _feed(c, episodes_dir, "cacc", 2, OutcomeType.ACCEPTED)

    # The observer reconciles on a fresh reload each cycle (a stale store cannot
    # see globals another process just minted). Re-mint must now succeed.
    observer = _make_project(tmp_path, "observer", facts_dir, episodes_dir)
    observer.reconcile_cross_project_promotions()
    assert len(_valid_globals(observer)) == 1, "pattern permanently barred from re-promotion"


def test_global_teardown_does_not_block_unrelated_project_local_fact(tmp_path):
    """Finding 3: an unrelated project must still be able to mint its own local
    PROJECT fact after a global for the same signature was torn down."""
    facts_dir = tmp_path / "facts"
    episodes_dir = tmp_path / "episodes"

    a = _make_project(tmp_path, "A", facts_dir, episodes_dir)
    _feed(a, episodes_dir, "acc", 2, OutcomeType.ACCEPTED)
    b = _make_project(tmp_path, "B", facts_dir, episodes_dir)
    _feed(b, episodes_dir, "bacc", 2, OutcomeType.ACCEPTED)  # A+B mint the global
    assert len(_valid_globals(b)) == 1

    _feed(a, episodes_dir, "corr", 2, OutcomeType.MODIFIED)  # A retracts
    b.reconcile_cross_project_promotions()                   # global torn down
    assert _valid_globals(b) == []

    # Unrelated project C never had any contradiction; it independently accepts
    # the same pattern twice and must get its own local project fact.
    c = _make_project(tmp_path, "C", facts_dir, episodes_dir)
    _feed(c, episodes_dir, "cacc", 2, OutcomeType.ACCEPTED)

    local = [
        f for f in c.entries
        if f.is_valid and f.scope == FactScope.PROJECT and "episode-derived" in f.tags
    ]
    assert local, "unrelated project blocked from minting its own local fact by global teardown"
