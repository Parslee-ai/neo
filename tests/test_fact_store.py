"""Tests for neo.memory.store - FactStore integration tests."""

import time
from unittest.mock import patch

import numpy as np
import pytest

from neo.memory.models import Fact, FactKind, FactMetadata, FactScope
from neo.memory.store import FactStore


@pytest.fixture
def tmp_facts_dir(tmp_path):
    """Override FACTS_DIR to use a temp directory."""
    facts_dir = tmp_path / "facts"
    facts_dir.mkdir()
    return facts_dir


@pytest.fixture
def store(tmp_facts_dir, tmp_path):
    """Create a FactStore with temp directories and no real embedder."""
    with patch("neo.memory.store.FACTS_DIR", tmp_facts_dir), \
         patch("neo.memory.store.detect_org_and_project", return_value=("testorg", "testproj1234")), \
         patch.object(FactStore, "_ingest_constraints"), \
         patch.object(FactStore, "_ingest_seed_facts"), \
         patch.object(FactStore, "_ingest_community_feed"), \
         patch.object(FactStore, "_ingest_claude_memory"), \
         patch.object(FactStore, "_maybe_migrate"), \
         patch("neo.memory.store.FASTEMBED_AVAILABLE", False):
        s = FactStore(codebase_root=str(tmp_path))
    return s


class TestAddFact:
    def test_basic_add(self, store):
        fact = store.add_fact(
            subject="Test fact",
            body="This is a test.",
            kind=FactKind.PATTERN,
        )
        assert fact.subject == "Test fact"
        assert fact.body == "This is a test."
        assert fact.kind == FactKind.PATTERN
        assert fact.is_valid is True
        assert len(store.entries) == 1

    def test_no_junk_filter(self, store):
        """Verify that entries that would be rejected by old junk filter are accepted."""
        fact = store.add_fact(
            subject="feature: Review authentication flow",
            body="You are given a Flask app with JWT tokens...",
            kind=FactKind.DECISION,
        )
        assert fact.is_valid is True
        assert len(store.entries) == 1

    def test_multiple_facts(self, store):
        store.add_fact(subject="Fact 1", body="Body 1")
        store.add_fact(subject="Fact 2", body="Body 2")
        store.add_fact(subject="Fact 3", body="Body 3")
        assert len(store.entries) == 3

    def test_tags_preserved(self, store):
        fact = store.add_fact(
            subject="Tagged",
            body="Body",
            tags=["python", "testing"],
        )
        assert "python" in fact.tags
        assert "testing" in fact.tags

    def test_confidence_set(self, store):
        fact = store.add_fact(
            subject="High conf",
            body="Body",
            confidence=0.95,
        )
        assert fact.metadata.confidence == 0.95


class TestSupersession:
    def test_supersession_with_embeddings(self, store):
        """Test that similar facts are superseded when embeddings match."""
        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        # Add a fact with a known embedding
        old = Fact(
            subject="Old fact",
            body="Old body",
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            org_id="testorg",
            project_id="testproj1234",
            embedding=emb,
            metadata=FactMetadata(confidence=0.5),
        )
        store._facts.append(old)

        # Create a very similar embedding (cosine > 0.85)
        similar_emb = emb + np.random.randn(768).astype(np.float32) * 0.01
        similar_emb = similar_emb / np.linalg.norm(similar_emb)

        # Mock _embed_text to return the similar embedding
        with patch.object(store, '_embed_text', return_value=similar_emb):
            new = store.add_fact(
                subject="New fact",
                body="Updated body",
                kind=FactKind.PATTERN,
                scope=FactScope.PROJECT,
            )

        assert old.is_valid is False
        assert old.superseded_by == new.id
        assert new.supersedes == old.id

    def test_no_supersession_different_kind(self, store):
        """Facts of different kinds should not supersede each other."""
        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        old = Fact(
            subject="Old",
            body="Body",
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            embedding=emb,
            metadata=FactMetadata(confidence=0.5),
        )
        store._facts.append(old)

        with patch.object(store, '_embed_text', return_value=emb):
            store.add_fact(
                subject="New",
                body="Body",
                kind=FactKind.FAILURE,  # Different kind
            )

        assert old.is_valid is True  # Not superseded

    def test_dependency_cascade(self, store):
        """When a fact is superseded, its dependents should be marked needs_review."""
        parent = Fact(
            id="parent123",
            subject="Parent",
            body="Body",
            metadata=FactMetadata(confidence=0.8),
        )
        child = Fact(
            subject="Child",
            body="Body",
            depends_on=["parent123"],
            metadata=FactMetadata(confidence=0.5),
        )
        store._facts.extend([parent, child])

        store._supersede(parent, Fact(subject="New parent", body="New body"))
        assert child.needs_review is True


class TestRetrieveRelevant:
    def test_returns_valid_only(self, store):
        store._facts.append(Fact(
            subject="Valid",
            body="Body",
            is_valid=True,
            metadata=FactMetadata(confidence=0.8),
        ))
        store._facts.append(Fact(
            subject="Invalid",
            body="Body",
            is_valid=False,
            metadata=FactMetadata(confidence=0.8),
        ))
        results = store.retrieve_relevant("test", k=10)
        assert len(results) == 1
        assert results[0].subject == "Valid"

    def test_excludes_constraints(self, store):
        store._facts.append(Fact(
            subject="Constraint",
            body="Body",
            kind=FactKind.CONSTRAINT,
            metadata=FactMetadata(confidence=1.0),
        ))
        results = store.retrieve_relevant("test", k=10)
        assert len(results) == 0

    def test_updates_access_metadata(self, store):
        fact = Fact(
            subject="Accessed",
            body="Body",
            metadata=FactMetadata(confidence=0.8, access_count=0),
        )
        store._facts.append(fact)
        store.retrieve_relevant("test", k=1)
        assert fact.metadata.access_count == 1


class TestPersistence:
    def test_save_load_roundtrip(self, store):
        store.add_fact(subject="Persist me", body="Important content", kind=FactKind.DECISION)
        # Verify file exists
        assert store._project_path.exists() or store._global_path.exists()

        # Load into new store (same config)
        with patch("neo.memory.store.FACTS_DIR", store._global_path.parent), \
             patch("neo.memory.store.detect_org_and_project", return_value=("testorg", "testproj1234")), \
             patch("neo.memory.store.ConstraintIngester") as mock_ingester, \
             patch.object(FactStore, "_ingest_seed_facts"), \
             patch.object(FactStore, "_ingest_community_feed"), \
             patch.object(FactStore, "_ingest_claude_memory"), \
             patch("neo.memory.store.FASTEMBED_AVAILABLE", False):
            mock_ingester.return_value.ingest.return_value = ([], [])
            store2 = FactStore(codebase_root=store.codebase_root)

        found = [f for f in store2.entries if f.subject == "Persist me"]
        assert len(found) == 1
        assert found[0].body == "Important content"

    def test_scoped_files_created(self, store):
        store.add_fact(subject="Global", body="G", scope=FactScope.GLOBAL)
        store.add_fact(subject="Project", body="P", scope=FactScope.PROJECT)
        assert store._global_path.exists()
        assert store._project_path.exists()


class TestMemoryLevel:
    def test_empty_returns_zero(self, store):
        assert store.memory_level() == 0.0

    def test_with_facts_returns_positive(self, store):
        for i in range(10):
            store._facts.append(Fact(
                subject=f"Fact {i}",
                body="Body",
                metadata=FactMetadata(confidence=0.5),
            ))
        level = store.memory_level()
        assert 0.0 < level < 1.0

    def test_more_facts_higher_level(self, store):
        for i in range(5):
            store._facts.append(Fact(subject=f"f{i}", body="b", metadata=FactMetadata()))
        level5 = store.memory_level()

        for i in range(50):
            store._facts.append(Fact(subject=f"g{i}", body="b", metadata=FactMetadata()))
        level55 = store.memory_level()

        assert level55 > level5

    def test_higher_confidence_higher_level(self, store):
        """Quality-based: higher confidence facts should yield higher level."""
        for i in range(20):
            store._facts.append(Fact(
                subject=f"low {i}", body="b",
                scope=FactScope.PROJECT,
                metadata=FactMetadata(confidence=0.3),
            ))
        level_low = store.memory_level()

        store._facts.clear()
        for i in range(20):
            store._facts.append(Fact(
                subject=f"high {i}", body="b",
                scope=FactScope.PROJECT,
                metadata=FactMetadata(confidence=0.9),
            ))
        level_high = store.memory_level()

        assert level_high > level_low

    def test_validated_facts_higher_level(self, store):
        """Facts with success history should contribute more to level."""
        for i in range(20):
            store._facts.append(Fact(
                subject=f"unused {i}", body="b",
                scope=FactScope.PROJECT,
                metadata=FactMetadata(confidence=0.5, access_count=0, success_count=0),
            ))
        level_unused = store.memory_level()

        store._facts.clear()
        for i in range(20):
            store._facts.append(Fact(
                subject=f"validated {i}", body="b",
                scope=FactScope.PROJECT,
                metadata=FactMetadata(confidence=0.5, access_count=10, success_count=8),
            ))
        level_validated = store.memory_level()

        assert level_validated > level_unused

    def test_no_time_decay(self, store):
        """Memory level should NOT decay based on time since last access."""
        old_time = time.time() - 90 * 86400  # 90 days ago
        for i in range(20):
            store._facts.append(Fact(
                subject=f"old {i}", body="b",
                scope=FactScope.PROJECT,
                metadata=FactMetadata(confidence=0.7, last_accessed=old_time),
            ))
        level_old = store.memory_level()

        store._facts.clear()
        for i in range(20):
            store._facts.append(Fact(
                subject=f"new {i}", body="b",
                scope=FactScope.PROJECT,
                metadata=FactMetadata(confidence=0.7, last_accessed=time.time()),
            ))
        level_new = store.memory_level()

        assert level_old == level_new


class TestScopeLimits:
    def test_evicts_when_over_limit(self, store):
        """Adding facts beyond scope limit should evict lowest-quality entries."""
        from neo.memory.store import SCOPE_LIMITS
        limit = SCOPE_LIMITS[FactScope.SESSION.value]  # 50

        # Fill to limit with medium confidence
        for i in range(limit):
            store._facts.append(Fact(
                subject=f"session {i}", body="b",
                scope=FactScope.SESSION,
                metadata=FactMetadata(confidence=0.5),
            ))

        # Add one more — should trigger eviction
        store.add_fact(
            subject="new session fact", body="important",
            scope=FactScope.SESSION, confidence=0.9,
        )

        valid_session = [f for f in store._facts if f.scope == FactScope.SESSION and f.is_valid]
        assert len(valid_session) <= limit

    def test_evicts_lowest_quality_first(self, store):
        """Eviction should remove the lowest confidence facts first."""
        # Add a low-confidence fact
        low = Fact(
            subject="weak pattern", body="b",
            scope=FactScope.SESSION,
            metadata=FactMetadata(confidence=0.1),
        )
        store._facts.append(low)

        # Add a high-confidence fact
        high = Fact(
            subject="strong pattern", body="b",
            scope=FactScope.SESSION,
            metadata=FactMetadata(confidence=0.9),
        )
        store._facts.append(high)

        # Fill remaining capacity
        from neo.memory.store import SCOPE_LIMITS
        limit = SCOPE_LIMITS[FactScope.SESSION.value]
        for i in range(limit - 1):  # -1 because we already have 2, want to go 1 over
            store._facts.append(Fact(
                subject=f"filler {i}", body="b",
                scope=FactScope.SESSION,
                metadata=FactMetadata(confidence=0.5),
            ))

        store._enforce_scope_limit(FactScope.SESSION)

        # Low-confidence fact should be evicted
        assert low.is_valid is False
        # High-confidence fact should survive
        assert high.is_valid is True

    def test_constraints_protected_from_eviction(self, store):
        """Constraint facts should never be evicted even when over limit."""
        from neo.memory.store import SCOPE_LIMITS
        limit = SCOPE_LIMITS[FactScope.PROJECT.value]

        constraint = Fact(
            subject="project rule", body="must do X",
            kind=FactKind.CONSTRAINT,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(confidence=0.1),  # lowest possible
        )
        store._facts.append(constraint)

        # Fill past capacity
        for i in range(limit + 5):
            store._facts.append(Fact(
                subject=f"pattern {i}", body="b",
                scope=FactScope.PROJECT,
                metadata=FactMetadata(confidence=0.5),
            ))

        store._enforce_scope_limit(FactScope.PROJECT)

        # Constraint must survive
        assert constraint.is_valid is True

    def test_synthesized_protected_from_eviction(self, store):
        """Synthesized archetype facts should be protected from eviction."""
        from neo.memory.store import SCOPE_LIMITS
        limit = SCOPE_LIMITS[FactScope.SESSION.value]

        synth = Fact(
            subject="synthesized archetype", body="b",
            scope=FactScope.SESSION,
            metadata=FactMetadata(confidence=0.1),
            tags=["synthesized"],
        )
        store._facts.append(synth)

        for i in range(limit + 5):
            store._facts.append(Fact(
                subject=f"filler {i}", body="b",
                scope=FactScope.SESSION,
                metadata=FactMetadata(confidence=0.5),
            ))

        store._enforce_scope_limit(FactScope.SESSION)
        assert synth.is_valid is True

    def test_no_eviction_under_limit(self, store):
        """No eviction should happen when under the scope limit."""
        for i in range(5):
            store._facts.append(Fact(
                subject=f"fact {i}", body="b",
                scope=FactScope.GLOBAL,
                metadata=FactMetadata(confidence=0.5),
            ))

        evicted = store._enforce_scope_limit(FactScope.GLOBAL)
        assert evicted == 0
        assert all(f.is_valid for f in store._facts)

    def test_different_scopes_independent(self, store):
        """Filling one scope should not affect another scope's capacity."""
        from neo.memory.store import SCOPE_LIMITS
        session_limit = SCOPE_LIMITS[FactScope.SESSION.value]

        # Fill SESSION to capacity
        for i in range(session_limit):
            store._facts.append(Fact(
                subject=f"session {i}", body="b",
                scope=FactScope.SESSION,
                metadata=FactMetadata(confidence=0.5),
            ))

        # Add GLOBAL facts — should not be affected
        for i in range(10):
            store._facts.append(Fact(
                subject=f"global {i}", body="b",
                scope=FactScope.GLOBAL,
                metadata=FactMetadata(confidence=0.5),
            ))

        store._enforce_scope_limit(FactScope.GLOBAL)
        global_valid = [f for f in store._facts if f.scope == FactScope.GLOBAL and f.is_valid]
        assert len(global_valid) == 10


class TestRetrievalNoDecay:
    def test_old_facts_not_penalized(self, store):
        """Retrieval scoring should not penalize old facts."""
        old_time = time.time() - 180 * 86400  # 6 months ago
        old_fact = Fact(
            subject="old but good", body="important pattern",
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(confidence=0.9, last_accessed=old_time),
        )
        new_fact = Fact(
            subject="new but weak", body="tentative pattern",
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(confidence=0.4, last_accessed=time.time()),
        )
        store._facts.extend([old_fact, new_fact])

        results = store.retrieve_relevant("pattern", k=2)

        # Old high-confidence fact should rank above new low-confidence fact
        assert results[0].subject == "old but good"


class TestBuildContext:
    def test_returns_context_result(self, store):
        store._facts.append(Fact(
            subject="Test",
            body="Body",
            kind=FactKind.CONSTRAINT,
            metadata=FactMetadata(confidence=1.0),
        ))
        store._facts.append(Fact(
            subject="Pattern",
            body="Body",
            kind=FactKind.PATTERN,
            metadata=FactMetadata(confidence=0.8),
        ))
        ctx = store.build_context("query")
        assert len(ctx.constraints) == 1
        assert len(ctx.valid_facts) == 1


class TestPurgeDeadFacts:
    def test_purges_superseded_with_valid_successor(self, store):
        """Invalid facts whose chain resolves to a valid fact should be purged."""
        old = Fact(id="old1", subject="Old", body="B", is_valid=False,
                   superseded_by="new1",
                   metadata=FactMetadata(last_accessed=time.time() - 90 * 86400))
        new = Fact(id="new1", subject="New", body="B", is_valid=True,
                   supersedes="old1", metadata=FactMetadata())
        store._facts.extend([old, new])
        assert store.purge_dead_facts() == 1
        assert len(store._facts) == 1
        assert store._facts[0].id == "new1"

    def test_purges_entire_chain(self, store):
        """A chain A->B->C (A,B invalid, C valid) should purge both A and B."""
        a = Fact(id="a", subject="A", body="B", is_valid=False,
                 superseded_by="b",
                 metadata=FactMetadata(last_accessed=time.time() - 90 * 86400))
        b = Fact(id="b", subject="B", body="B", is_valid=False,
                 superseded_by="c",
                 metadata=FactMetadata(last_accessed=time.time() - 90 * 86400))
        c = Fact(id="c", subject="C", body="B", is_valid=True,
                 supersedes="b", metadata=FactMetadata())
        store._facts.extend([a, b, c])
        assert store.purge_dead_facts() == 2
        assert len(store._facts) == 1
        assert store._facts[0].id == "c"

    def test_keeps_recently_accessed_invalid(self, store):
        """Invalid facts accessed within 30 days should be kept for contrast."""
        old = Fact(id="old1", subject="Old", body="B", is_valid=False,
                   superseded_by="new1",
                   metadata=FactMetadata(last_accessed=time.time() - 5 * 86400))
        new = Fact(id="new1", subject="New", body="B", is_valid=True,
                   supersedes="old1", metadata=FactMetadata())
        store._facts.extend([old, new])
        assert store.purge_dead_facts() == 0
        assert len(store._facts) == 2

    def test_keeps_invalid_without_valid_successor(self, store):
        """Invalid facts whose chain doesn't resolve to a valid fact should be kept."""
        orphan = Fact(id="orph", subject="Orphan", body="B", is_valid=False,
                      superseded_by="gone",
                      metadata=FactMetadata(last_accessed=time.time() - 90 * 86400))
        store._facts.append(orphan)
        assert store.purge_dead_facts() == 0
        assert len(store._facts) == 1


class TestOutcomeLinkage:
    """Tests for Step 1: outcome->suggestion fact linkage."""

    def test_accepted_outcome_boosts_original_fact(self, store):
        """When an accepted outcome has a linked fact, boost that fact's confidence."""
        from neo.memory.outcomes import Outcome

        # Create an original suggestion fact
        original = store.add_fact(
            subject="bugfix: fix validation",
            body="Suggestion: add input validation",
            kind=FactKind.PATTERN,
            confidence=0.7,
        )
        original_id = original.id
        assert original.metadata.success_count == 0

        # Simulate detect_outcomes returning a linked outcome
        outcomes = [Outcome(
            outcome_type="accepted",
            file_path="src/foo.py",
            suggestion_description="add input validation",
            suggestion_confidence=0.7,
        )]
        suggestion_fact_ids = {"src/foo.py": original_id}

        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=(outcomes, suggestion_fact_ids)):
            store.detect_implicit_feedback({"prompt": "test"}, [])

        # Original fact should be boosted, not a new REVIEW fact created
        assert original.metadata.confidence == pytest.approx(0.8)
        assert original.metadata.success_count == 1
        # No new REVIEW facts should exist
        review_facts = [f for f in store._facts if f.kind == FactKind.REVIEW]
        assert len(review_facts) == 0

    def test_accepted_outcome_without_linkage_creates_review(self, store):
        """When no fact ID is linked, fall back to creating a REVIEW fact."""
        from neo.memory.outcomes import Outcome

        outcomes = [Outcome(
            outcome_type="accepted",
            file_path="src/bar.py",
            suggestion_description="refactor method",
            suggestion_confidence=0.6,
        )]
        # Empty suggestion_fact_ids = no linkage
        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=(outcomes, {})):
            store.detect_implicit_feedback({"prompt": "test"}, [])

        review_facts = [f for f in store._facts if f.kind == FactKind.REVIEW]
        assert len(review_facts) == 1
        assert "accepted" in review_facts[0].tags

    def test_independent_outcome_creates_review(self, store):
        """Independent changes always create REVIEW facts."""
        from neo.memory.outcomes import Outcome

        outcomes = [Outcome(
            outcome_type="independent",
            file_path="src/baz.py",
            diff_summary="+new code",
        )]
        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=(outcomes, {})):
            store.detect_implicit_feedback({"prompt": "test"}, [])

        review_facts = [f for f in store._facts if f.kind == FactKind.REVIEW]
        assert len(review_facts) == 1
        assert "independent" in review_facts[0].tags

    def test_multiple_boosts_accumulate(self, store):
        """Multiple accepted outcomes should stack confidence boosts."""
        from neo.memory.outcomes import Outcome

        original = store.add_fact(
            subject="pattern: error handling",
            body="Wrap I/O in try/except",
            kind=FactKind.PATTERN,
            confidence=0.5,
        )

        for _ in range(3):
            outcomes = [Outcome(
                outcome_type="accepted",
                file_path="src/store.py",
                suggestion_description="wrap in try/except",
                suggestion_confidence=0.5,
            )]
            with patch.object(store._outcome_tracker, "detect_outcomes",
                              return_value=(outcomes, {"src/store.py": original.id})):
                store.detect_implicit_feedback({"prompt": "test"}, [])

        assert original.metadata.success_count == 3
        assert original.metadata.confidence == pytest.approx(0.8)


class TestSynthesizeReviews:
    """Tests for Step 2: periodic review synthesis."""

    def _make_review_fact(self, store, subject, body, tags, embedding=None):
        """Helper to add a REVIEW fact directly."""
        fact = Fact(
            subject=subject,
            body=body,
            kind=FactKind.REVIEW,
            scope=FactScope.PROJECT,
            org_id="testorg",
            project_id="testproj1234",
            metadata=FactMetadata(confidence=0.5),
            embedding=embedding,
            tags=tags,
        )
        store._facts.append(fact)
        return fact

    def test_no_synthesis_below_threshold(self, store):
        """Should not synthesize when fewer than 20 REVIEW facts exist."""
        for i in range(15):
            self._make_review_fact(store, f"review {i}", f"body {i}", ["outcome", "accepted"])
        assert store.synthesize_reviews() == 0

    def test_synthesis_creates_pattern_from_accepted_cluster(self, store):
        """A cluster of 3+ similar accepted REVIEW facts should synthesize into PATTERN."""
        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        # Create 25 REVIEW facts; 5 of them are a tight cluster
        for i in range(20):
            random_emb = np.random.randn(768).astype(np.float32)
            random_emb = random_emb / np.linalg.norm(random_emb)
            self._make_review_fact(
                store, f"outcome:accepted other_{i}.py", f"body {i}",
                ["outcome", "accepted"], embedding=random_emb,
            )

        # Add a tight cluster of 5 similar facts
        for i in range(5):
            slight_variation = emb + np.random.randn(768).astype(np.float32) * 0.01
            slight_variation = slight_variation / np.linalg.norm(slight_variation)
            self._make_review_fact(
                store, "outcome:accepted store.py", f"wrap I/O in try/except variant {i}",
                ["outcome", "accepted"], embedding=slight_variation,
            )

        count = store.synthesize_reviews()
        assert count >= 1

        # The synthesized fact should be PATTERN kind
        patterns = [f for f in store._facts if f.kind == FactKind.PATTERN and f.is_valid]
        assert len(patterns) >= 1
        assert "synthesized" in patterns[0].tags

    def test_synthesis_supersedes_source_facts(self, store):
        """Source REVIEW facts should be marked invalid after synthesis."""
        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        # 20 filler + 3 clustered
        for i in range(20):
            random_emb = np.random.randn(768).astype(np.float32)
            random_emb = random_emb / np.linalg.norm(random_emb)
            self._make_review_fact(
                store, f"filler {i}", f"body {i}",
                ["outcome", "accepted"], embedding=random_emb,
            )

        cluster_facts = []
        for i in range(4):
            slight = emb + np.random.randn(768).astype(np.float32) * 0.005
            slight = slight / np.linalg.norm(slight)
            f = self._make_review_fact(
                store, "outcome:accepted store.py", f"same pattern {i}",
                ["outcome", "accepted"], embedding=slight,
            )
            cluster_facts.append(f)

        store.synthesize_reviews()

        # All cluster source facts should be invalidated
        for f in cluster_facts:
            assert f.is_valid is False
            assert f.superseded_by is not None

    def test_watermark_prevents_rerun(self, store):
        """Synthesis should not re-run until 10 more REVIEW facts accumulate."""
        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        for i in range(25):
            random_emb = np.random.randn(768).astype(np.float32)
            random_emb = random_emb / np.linalg.norm(random_emb)
            self._make_review_fact(
                store, f"review {i}", f"body {i}",
                ["outcome", "accepted"], embedding=random_emb,
            )

        # First run sets watermark
        store.synthesize_reviews()

        # Second run without new facts should skip
        count = store.synthesize_reviews()
        assert count == 0


class TestPruneStaleFacts:
    """Tests for quality pruning of stale unvalidated facts."""

    def test_prunes_old_low_confidence_zero_success(self, store):
        """Old facts with low confidence and no successes should be pruned."""
        old_time = time.time() - 20 * 86400  # 20 days old
        fact = Fact(
            subject="stale",
            body="never validated",
            kind=FactKind.REVIEW,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(confidence=0.3, success_count=0, created_at=old_time),
        )
        store._facts.append(fact)
        assert store.prune_stale_facts() == 1
        assert fact.is_valid is False

    def test_keeps_constraint_facts(self, store):
        """CONSTRAINT facts should never be pruned by stale check."""
        old_time = time.time() - 20 * 86400
        fact = Fact(
            subject="rule",
            body="project rule",
            kind=FactKind.CONSTRAINT,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(confidence=0.2, success_count=0, created_at=old_time),
        )
        store._facts.append(fact)
        assert store.prune_stale_facts() == 0
        assert fact.is_valid is True

    def test_keeps_recent_facts(self, store):
        """Facts less than 14 days old should not be pruned."""
        recent_time = time.time() - 5 * 86400  # 5 days old
        fact = Fact(
            subject="recent",
            body="new",
            kind=FactKind.REVIEW,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(confidence=0.2, success_count=0, created_at=recent_time),
        )
        store._facts.append(fact)
        assert store.prune_stale_facts() == 0
        assert fact.is_valid is True

    def test_keeps_facts_with_successes(self, store):
        """Facts with success_count > 0 should not be pruned regardless of confidence."""
        old_time = time.time() - 20 * 86400
        fact = Fact(
            subject="validated",
            body="was helpful once",
            kind=FactKind.REVIEW,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(confidence=0.2, success_count=1, created_at=old_time),
        )
        store._facts.append(fact)
        assert store.prune_stale_facts() == 0
        assert fact.is_valid is True

    def test_keeps_high_confidence_facts(self, store):
        """Facts with confidence >= 0.4 should not be pruned."""
        old_time = time.time() - 20 * 86400
        fact = Fact(
            subject="confident",
            body="good enough",
            kind=FactKind.REVIEW,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(confidence=0.5, success_count=0, created_at=old_time),
        )
        store._facts.append(fact)
        assert store.prune_stale_facts() == 0

    def test_cascades_needs_review(self, store):
        """Pruned facts should cascade needs_review to dependents."""
        old_time = time.time() - 20 * 86400
        parent = Fact(
            id="stale_parent",
            subject="stale parent",
            body="old",
            kind=FactKind.REVIEW,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(confidence=0.2, success_count=0, created_at=old_time),
        )
        child = Fact(
            subject="child",
            body="depends on parent",
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            depends_on=["stale_parent"],
            metadata=FactMetadata(confidence=0.8),
        )
        store._facts.extend([parent, child])
        store.prune_stale_facts()
        assert child.needs_review is True


class TestDemoteUnhelpfulFacts:
    """Tests for success/failure-based demotion."""

    def test_demotes_accessed_but_unsuccessful(self, store):
        """Facts accessed 5-9 times with 0 successes should lose confidence."""
        old_time = time.time() - 10 * 86400
        fact = Fact(
            subject="unhelpful",
            body="retrieved but never accepted",
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(
                confidence=0.6, access_count=7, success_count=0, created_at=old_time,
            ),
        )
        store._facts.append(fact)
        assert store.demote_unhelpful_facts() == 1
        assert fact.metadata.confidence == pytest.approx(0.5)
        assert fact.is_valid is True  # Soft demotion, not pruned

    def test_prunes_heavily_accessed_unsuccessful(self, store):
        """Facts accessed 10+ times with 0 successes should be invalidated."""
        old_time = time.time() - 10 * 86400
        fact = Fact(
            subject="actively bad",
            body="lots of chances, never helped",
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(
                confidence=0.6, access_count=12, success_count=0, created_at=old_time,
            ),
        )
        store._facts.append(fact)
        assert store.demote_unhelpful_facts() == 1
        assert fact.is_valid is False

    def test_protects_successful_facts(self, store):
        """Facts with good hit rate should get a confidence boost."""
        old_time = time.time() - 10 * 86400
        fact = Fact(
            subject="helpful",
            body="consistently useful",
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(
                confidence=0.7, access_count=10, success_count=4, created_at=old_time,
            ),
        )
        store._facts.append(fact)
        assert store.demote_unhelpful_facts() == 1
        assert fact.metadata.confidence == pytest.approx(0.75)
        assert fact.is_valid is True

    def test_skips_constraints(self, store):
        """CONSTRAINT facts should never be demoted."""
        old_time = time.time() - 10 * 86400
        fact = Fact(
            subject="rule",
            body="project rule",
            kind=FactKind.CONSTRAINT,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(
                confidence=0.5, access_count=15, success_count=0, created_at=old_time,
            ),
        )
        store._facts.append(fact)
        assert store.demote_unhelpful_facts() == 0

    def test_skips_young_facts(self, store):
        """Facts less than 7 days old should not be demoted."""
        recent = time.time() - 3 * 86400
        fact = Fact(
            subject="new",
            body="too young",
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(
                confidence=0.5, access_count=8, success_count=0, created_at=recent,
            ),
        )
        store._facts.append(fact)
        assert store.demote_unhelpful_facts() == 0

    def test_skips_low_access_facts(self, store):
        """Facts with fewer than 5 accesses should not be touched."""
        old_time = time.time() - 10 * 86400
        fact = Fact(
            subject="barely used",
            body="not enough data",
            kind=FactKind.PATTERN,
            scope=FactScope.PROJECT,
            metadata=FactMetadata(
                confidence=0.5, access_count=3, success_count=0, created_at=old_time,
            ),
        )
        store._facts.append(fact)
        assert store.demote_unhelpful_facts() == 0


class TestLLMSynthesis:
    """Tests for LLM-based synthesis enhancement."""

    def _make_review_fact(self, store, subject, body, tags, embedding=None):
        fact = Fact(
            subject=subject,
            body=body,
            kind=FactKind.REVIEW,
            scope=FactScope.PROJECT,
            org_id="testorg",
            project_id="testproj1234",
            metadata=FactMetadata(confidence=0.5),
            embedding=embedding,
            tags=tags,
        )
        store._facts.append(fact)
        return fact

    def test_llm_used_for_large_cluster(self, store):
        """Clusters of 5+ should use LLM synthesis when adapter is available."""
        # Set up a mock LM adapter
        mock_lm = type("MockLM", (), {
            "generate": lambda self, **kwargs: (
                "SUBJECT: error handling pattern in store.py\n"
                "BODY: Wrap all I/O operations in try/except with logger.debug for resilience.\n"
                "CONFIDENCE: 0.85"
            )
        })()
        store._lm_adapter = mock_lm

        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        # 20 filler + 5 tight cluster
        for i in range(20):
            random_emb = np.random.randn(768).astype(np.float32)
            random_emb = random_emb / np.linalg.norm(random_emb)
            self._make_review_fact(
                store, f"outcome:accepted other_{i}.py", f"body {i}",
                ["outcome", "accepted"], embedding=random_emb,
            )
        for i in range(5):
            slight = emb + np.random.randn(768).astype(np.float32) * 0.005
            slight = slight / np.linalg.norm(slight)
            self._make_review_fact(
                store, "outcome:accepted store.py", f"wrap I/O variant {i}",
                ["outcome", "accepted"], embedding=slight,
            )

        store.synthesize_reviews()

        patterns = [f for f in store._facts if f.kind == FactKind.PATTERN and f.is_valid]
        assert len(patterns) >= 1
        # Should use the LLM-generated subject
        llm_pattern = [p for p in patterns if "error handling" in p.subject]
        assert len(llm_pattern) == 1
        assert llm_pattern[0].metadata.confidence == pytest.approx(0.85)

    def test_mechanical_fallback_for_small_cluster(self, store):
        """Clusters of 3-4 should use mechanical synthesis even with adapter."""
        mock_lm = type("MockLM", (), {
            "generate": lambda self, **kwargs: "should not be called"
        })()
        store._lm_adapter = mock_lm

        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        for i in range(20):
            random_emb = np.random.randn(768).astype(np.float32)
            random_emb = random_emb / np.linalg.norm(random_emb)
            self._make_review_fact(
                store, f"filler {i}", f"body {i}",
                ["outcome", "accepted"], embedding=random_emb,
            )
        # Only 3 in cluster (below LLM threshold of 5)
        for i in range(3):
            slight = emb + np.random.randn(768).astype(np.float32) * 0.005
            slight = slight / np.linalg.norm(slight)
            self._make_review_fact(
                store, "outcome:accepted store.py", f"pattern {i}",
                ["outcome", "accepted"], embedding=slight,
            )

        count = store.synthesize_reviews()
        assert count >= 1  # Mechanical synthesis still works

    def test_mechanical_fallback_on_llm_error(self, store):
        """LLM errors should fall back to mechanical synthesis gracefully."""
        mock_lm = type("MockLM", (), {
            "generate": lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("API down"))
        })()
        store._lm_adapter = mock_lm

        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        for i in range(20):
            random_emb = np.random.randn(768).astype(np.float32)
            random_emb = random_emb / np.linalg.norm(random_emb)
            self._make_review_fact(
                store, f"filler {i}", f"body {i}",
                ["outcome", "accepted"], embedding=random_emb,
            )
        for i in range(5):
            slight = emb + np.random.randn(768).astype(np.float32) * 0.005
            slight = slight / np.linalg.norm(slight)
            self._make_review_fact(
                store, "outcome:accepted store.py", f"variant {i}",
                ["outcome", "accepted"], embedding=slight,
            )

        count = store.synthesize_reviews()
        assert count >= 1  # Falls back to mechanical

    def test_mechanical_fallback_no_adapter(self, store):
        """No adapter should mean pure mechanical synthesis."""
        assert store._lm_adapter is None  # Default fixture has no adapter

        emb = np.random.randn(768).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        for i in range(20):
            random_emb = np.random.randn(768).astype(np.float32)
            random_emb = random_emb / np.linalg.norm(random_emb)
            self._make_review_fact(
                store, f"filler {i}", f"body {i}",
                ["outcome", "accepted"], embedding=random_emb,
            )
        for i in range(5):
            slight = emb + np.random.randn(768).astype(np.float32) * 0.005
            slight = slight / np.linalg.norm(slight)
            self._make_review_fact(
                store, "outcome:accepted store.py", f"variant {i}",
                ["outcome", "accepted"], embedding=slight,
            )

        count = store.synthesize_reviews()
        assert count >= 1

    def test_parse_llm_response(self, store):
        """Test parsing of various LLM response formats."""
        from neo.memory.store import FactStore

        # Good response
        result = FactStore._parse_llm_synthesis(
            "SUBJECT: error handling in store.py\n"
            "BODY: Always wrap I/O in try/except.\n"
            "CONFIDENCE: 0.9"
        )
        assert result is not None
        assert result[0] == "error handling in store.py"
        assert "wrap I/O" in result[1]
        assert result[2] == pytest.approx(0.9)

        # Bad response (missing fields)
        assert FactStore._parse_llm_synthesis("just some text") is None

        # Confidence out of range gets clamped
        result = FactStore._parse_llm_synthesis(
            "SUBJECT: test\nBODY: test body\nCONFIDENCE: 1.5"
        )
        assert result is not None
        assert result[2] == pytest.approx(1.0)


class TestConstraintEmbeddings:
    def test_ingest_generates_embeddings(self, tmp_facts_dir, tmp_path):
        """Constraint ingestion should generate embeddings for new facts."""
        # Create a CLAUDE.md in the temp project
        claude_md = tmp_path / "CLAUDE.md"
        claude_md.write_text("## Rule One\nDo the thing.\n\n## Rule Two\nDo another thing.\n")

        fake_emb = np.ones(768, dtype=np.float32)
        with patch("neo.memory.store.FACTS_DIR", tmp_facts_dir), \
             patch("neo.memory.store.detect_org_and_project", return_value=("testorg", "testproj1234")), \
             patch.object(FactStore, "_maybe_migrate"), \
             patch.object(FactStore, "_ingest_seed_facts"), \
             patch.object(FactStore, "_ingest_community_feed"), \
             patch.object(FactStore, "_ingest_claude_memory"), \
             patch.object(FactStore, "_embed_text", return_value=fake_emb):
            s = FactStore(codebase_root=str(tmp_path))

        constraints = [f for f in s._facts if f.kind == FactKind.CONSTRAINT]
        for c in constraints:
            assert c.embedding is not None, f"Constraint '{c.subject}' has no embedding"
