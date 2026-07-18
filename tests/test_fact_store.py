"""Tests for neo.memory.store - FactStore integration tests."""

import time
from unittest.mock import patch

import numpy as np
import pytest

from neo.memory.models import Fact, FactKind, FactMetadata, FactScope
from neo.memory.outcomes import Outcome, OutcomeType
from neo.memory.store import FactStore


@pytest.fixture
def tmp_facts_dir(tmp_path):
    """Override FACTS_DIR to use a temp directory."""
    facts_dir = tmp_path / "facts"
    facts_dir.mkdir()
    return facts_dir


@pytest.fixture
def tmp_checksum_dir(tmp_path):
    """Temp checksum dir to prevent test pollution of real ~/.neo/constraints/."""
    d = tmp_path / "checksums"
    d.mkdir()
    return d


@pytest.fixture
def store(tmp_facts_dir, tmp_path, tmp_checksum_dir):
    """Create a FactStore with temp directories and no real embedder.

    Patches remain active for the lifetime of the test (generator fixture),
    so any method calls during the test (save, load, re-ingestion) hit
    mocked paths, not real ~/.neo state.
    """
    from contextlib import ExitStack
    with ExitStack() as stack:
        stack.enter_context(patch("neo.memory.store.FACTS_DIR", tmp_facts_dir))
        stack.enter_context(patch("neo.memory.store.detect_org_and_project", return_value=("testorg", "testproj1234")))
        stack.enter_context(patch.object(FactStore, "_ingest_constraints"))
        stack.enter_context(patch.object(FactStore, "_ingest_seed_facts"))
        stack.enter_context(patch.object(FactStore, "_ingest_community_feed"))
        stack.enter_context(patch.object(FactStore, "_ingest_claude_memory"))
        stack.enter_context(patch.object(FactStore, "_maybe_migrate"))
        stack.enter_context(patch("neo.memory.store.FASTEMBED_AVAILABLE", False))
        # Prevent test pollution of real checksums.json
        for mod in ("constraints", "seed", "community", "claude_memory"):
            stack.enter_context(patch(f"neo.memory.{mod}.CHECKSUM_DIR", tmp_checksum_dir))
            stack.enter_context(patch(f"neo.memory.{mod}.CHECKSUM_FILE", tmp_checksum_dir / "checksums.json"))
        s = FactStore(codebase_root=str(tmp_path))
        yield s


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

    def test_domain_filter_excludes_other_domains(self, store):
        store._facts.append(Fact(
            subject="Testing fact",
            body="Use pytest",
            domain="testing",
            metadata=FactMetadata(confidence=0.8),
        ))
        store._facts.append(Fact(
            subject="Style fact",
            body="Snake case",
            domain="code-style",
            metadata=FactMetadata(confidence=0.8),
        ))
        results = store.retrieve_relevant("anything", k=10, domain="testing")
        assert len(results) == 1
        assert results[0].subject == "Testing fact"

    def test_domain_filter_none_returns_all(self, store):
        store._facts.append(Fact(
            subject="A", body="b", domain="testing",
            metadata=FactMetadata(confidence=0.8),
        ))
        store._facts.append(Fact(
            subject="B", body="b", domain="code-style",
            metadata=FactMetadata(confidence=0.8),
        ))
        store._facts.append(Fact(
            subject="C", body="b", domain=None,
            metadata=FactMetadata(confidence=0.8),
        ))
        results = store.retrieve_relevant("anything", k=10)
        assert len(results) == 3

    def test_domain_filter_excludes_unset_domain(self, store):
        """Facts with domain=None must NOT match a specific domain filter."""
        store._facts.append(Fact(
            subject="Untagged", body="b", domain=None,
            metadata=FactMetadata(confidence=0.8),
        ))
        store._facts.append(Fact(
            subject="Tagged", body="b", domain="testing",
            metadata=FactMetadata(confidence=0.8),
        ))
        results = store.retrieve_relevant("anything", k=10, domain="testing")
        assert len(results) == 1
        assert results[0].subject == "Tagged"


class TestPersistence:
    def test_save_load_roundtrip(self, store):
        store.add_fact(subject="Persist me", body="Important content", kind=FactKind.DECISION)
        # Verify file exists
        assert store._project_path.exists() or store._global_path.exists()

        # Load into new store (same config)
        with patch("neo.memory.store.FACTS_DIR", store._global_path.parent), \
             patch("neo.memory.store.detect_org_and_project", return_value=("testorg", "testproj1234")), \
             patch.object(FactStore, "_ingest_constraints"), \
             patch.object(FactStore, "_ingest_seed_facts"), \
             patch.object(FactStore, "_ingest_community_feed"), \
             patch.object(FactStore, "_ingest_claude_memory"), \
             patch.object(FactStore, "_maybe_migrate"), \
             patch("neo.memory.store.FASTEMBED_AVAILABLE", False):
            store2 = FactStore(codebase_root=store.codebase_root)

        found = [f for f in store2.entries if f.subject == "Persist me"]
        assert len(found) == 1
        assert found[0].body == "Important content"

    def test_corrupt_fact_file_is_backed_up_before_empty_load(self, tmp_facts_dir, tmp_path):
        corrupt_path = tmp_facts_dir / "facts_global.json"
        corrupt_path.write_text('{"facts": [')

        with patch("neo.memory.store.FACTS_DIR", tmp_facts_dir), \
             patch("neo.memory.store.detect_org_and_project", return_value=("testorg", "testproj1234")), \
             patch.object(FactStore, "_ingest_constraints"), \
             patch.object(FactStore, "_ingest_seed_facts"), \
             patch.object(FactStore, "_ingest_community_feed"), \
             patch.object(FactStore, "_ingest_claude_memory"), \
             patch.object(FactStore, "_maybe_migrate"), \
             patch("neo.memory.store.FASTEMBED_AVAILABLE", False):
            store = FactStore(codebase_root=str(tmp_path), eager_init=False)

        backups = list(tmp_facts_dir.glob("facts_global.json.corrupt-*"))
        assert store.entries == []
        assert len(backups) == 1
        assert backups[0].read_text() == '{"facts": ['

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

    def test_eviction_strips_embedding_without_cascade(self, store):
        """Eviction invalidates + strips the vector but does NOT flag dependents
        (cascade=False): eviction-for-capacity isn't supersession."""
        low = Fact(
            id="evictme", subject="weak", body="b", scope=FactScope.SESSION,
            embedding=np.random.randn(768).astype(np.float32),
            metadata=FactMetadata(confidence=0.05),
        )
        dependent = Fact(
            id="dep", subject="depends", body="b", scope=FactScope.SESSION,
            depends_on=["evictme"], metadata=FactMetadata(confidence=0.9),
        )
        store._facts.extend([low, dependent])

        from neo.memory.store import SCOPE_LIMITS
        limit = SCOPE_LIMITS[FactScope.SESSION.value]
        for i in range(limit):  # push well over so `low` is evicted
            store._facts.append(Fact(
                subject=f"filler {i}", body="b", scope=FactScope.SESSION,
                metadata=FactMetadata(confidence=0.6),
            ))

        store._enforce_scope_limit(FactScope.SESSION)

        assert low.is_valid is False
        assert low.embedding is None            # stripped at the transition
        assert dependent.needs_review is False  # cascade=False — no noise flag

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


class TestStripTombstoneEmbeddings:
    def test_strips_embedding_from_invalid_fact(self, store):
        """Invalid facts should lose their embedding (dead weight); valid keep it."""
        emb = np.random.randn(768).astype(np.float32)
        dead = Fact(id="dead", subject="Dead", body="B", is_valid=False,
                    embedding=emb.copy(), metadata=FactMetadata())
        live = Fact(id="live", subject="Live", body="B", is_valid=True,
                    embedding=emb.copy(), metadata=FactMetadata())
        store._facts.extend([dead, live])

        assert store.strip_tombstone_embeddings() == 1
        assert dead.embedding is None
        assert live.embedding is not None

    def test_idempotent(self, store):
        """Second run strips nothing — no embedding left on tombstones."""
        dead = Fact(id="dead", subject="Dead", body="B", is_valid=False,
                    embedding=np.random.randn(768).astype(np.float32),
                    metadata=FactMetadata())
        store._facts.append(dead)
        assert store.strip_tombstone_embeddings() == 1
        assert store.strip_tombstone_embeddings() == 0

    def test_survives_save_reload(self, store):
        """A stripped tombstone reloads with embedding None (key omitted on disk)."""
        dead = Fact(id="dead", subject="Dead", body="B", is_valid=False,
                    scope=FactScope.PROJECT, org_id="testorg",
                    project_id="testproj1234",
                    embedding=np.random.randn(768).astype(np.float32),
                    metadata=FactMetadata())
        store._facts.append(dead)
        store.strip_tombstone_embeddings()
        store.load()
        reloaded = [f for f in store._facts if f.id == "dead"]
        assert reloaded and reloaded[0].embedding is None

    def test_save_false_defers_persist_but_mutates(self, store):
        """save=False strips in-memory and returns the count, but does not save —
        lets the cold-start chain collapse four janitor saves into one."""
        dead = Fact(id="dead", subject="Dead", body="B", is_valid=False,
                    embedding=np.random.randn(768).astype(np.float32),
                    metadata=FactMetadata())
        store._facts.append(dead)
        with patch.object(store, "save") as mock_save:
            assert store.strip_tombstone_embeddings(save=False) == 1
        assert dead.embedding is None      # mutated in memory
        mock_save.assert_not_called()      # but not persisted


class TestInvalidate:
    def test_strips_embedding_and_cascades(self, store):
        """_invalidate marks invalid, drops the vector, and flags dependents."""
        parent = Fact(id="p", subject="Parent", body="B", is_valid=True,
                      embedding=np.random.randn(768).astype(np.float32),
                      metadata=FactMetadata())
        child = Fact(id="c", subject="Child", body="B", is_valid=True,
                     depends_on=["p"], metadata=FactMetadata())
        store._facts.extend([parent, child])

        store._invalidate(parent)

        assert parent.is_valid is False
        assert parent.embedding is None
        assert child.needs_review is True

    def test_cascade_false_skips_dependents(self, store):
        """cascade=False (the independent-fact cap) strips but doesn't flag."""
        parent = Fact(id="p", subject="Parent", body="B", is_valid=True,
                      embedding=np.random.randn(768).astype(np.float32),
                      metadata=FactMetadata())
        child = Fact(id="c", subject="Child", body="B", is_valid=True,
                     depends_on=["p"], metadata=FactMetadata())
        store._facts.extend([parent, child])

        store._invalidate(parent, cascade=False)

        assert parent.is_valid is False
        assert parent.embedding is None
        assert child.needs_review is False

    def test_strip_ingester_tombstones_nulls_embeddings(self, store):
        """Ingester-superseded facts (off the _invalidate path) get their
        vectors dropped at the consumption site."""
        a = Fact(id="a", subject="A", body="b", is_valid=False,
                 embedding=np.random.randn(768).astype(np.float32),
                 metadata=FactMetadata())
        b = Fact(id="b", subject="B", body="b", is_valid=False,
                 embedding=np.random.randn(768).astype(np.float32),
                 metadata=FactMetadata())
        store._strip_ingester_tombstones([a, b])
        assert a.embedding is None and b.embedding is None

    def test_supersede_strips_old_embedding_and_keeps_pointer(self, store):
        """_supersede routes through _invalidate: old fact loses its vector but
        keeps the superseded_by pointer for chain integrity."""
        old = Fact(id="old", subject="Old", body="B", is_valid=True,
                   embedding=np.random.randn(768).astype(np.float32),
                   metadata=FactMetadata())
        new = Fact(id="new", subject="New", body="B", metadata=FactMetadata())
        store._facts.append(old)

        store._supersede(old, new)

        assert old.is_valid is False
        assert old.embedding is None
        assert old.superseded_by == "new"


class TestJanitorSaveDeferral:
    def test_purge_records_deletions_when_save_deferred(self, store):
        """purge(save=False) must still record _deleted_ids so a later trailing
        save() flushes them (else merge-on-save could resurrect the tombstone)."""
        orphan = Fact(id="orph", subject="Orphan", body="B", is_valid=False,
                      metadata=FactMetadata(last_accessed=time.time() - 90 * 86400))
        store._facts.append(orphan)
        with patch.object(store, "save") as mock_save:
            assert store.purge_dead_facts(save=False) == 1
        assert "orph" in store._deleted_ids   # recorded for the trailing flush
        mock_save.assert_not_called()

    def test_all_four_janitors_accept_save_flag(self, store):
        """Signature guard: every chained janitor takes save= and defaults True."""
        import inspect
        for name in ("prune_stale_facts", "demote_unhelpful_facts",
                     "purge_dead_facts", "strip_tombstone_embeddings"):
            sig = inspect.signature(getattr(store, name))
            assert sig.parameters["save"].default is True


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

    def test_purges_cold_eviction_orphan(self, store):
        """Eviction orphans (invalid, no successor) cold for 30+ days are purged.

        Regression test: cold-start purge previously retained these forever
        because their supersession chain never resolves to a valid fact, so
        inactive projects' fact files bloated indefinitely.
        """
        orphan = Fact(id="orph", subject="Orphan", body="B", is_valid=False,
                      metadata=FactMetadata(last_accessed=time.time() - 90 * 86400))
        store._facts.append(orphan)
        assert store.purge_dead_facts() == 1
        assert len(store._facts) == 0

    def test_purges_cold_broken_chain_tombstone(self, store):
        """Invalid facts pointing at a missing successor are also purged when cold."""
        orphan = Fact(id="orph", subject="Orphan", body="B", is_valid=False,
                      superseded_by="gone",
                      metadata=FactMetadata(last_accessed=time.time() - 90 * 86400))
        store._facts.append(orphan)
        assert store.purge_dead_facts() == 1
        assert len(store._facts) == 0

    def test_keeps_recent_eviction_orphan(self, store):
        """Recently-evicted orphans (< 30 days) are kept for contrast."""
        orphan = Fact(id="orph", subject="Orphan", body="B", is_valid=False,
                      metadata=FactMetadata(last_accessed=time.time() - 5 * 86400))
        store._facts.append(orphan)
        assert store.purge_dead_facts() == 0
        assert len(store._facts) == 1


class TestOutcomeLinkage:
    """Tests for Step 1: outcome->suggestion fact linkage."""

    def test_accepted_outcome_boosts_original_fact(self, store):
        """When an accepted outcome has a linked fact, boost that fact's confidence."""

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
            outcome_type=OutcomeType.ACCEPTED,
            file_path="src/foo.py",
            suggestion_description="add input validation",
            suggestion_confidence=0.7,
        )]
        suggestion_fact_ids = {"src/foo.py": original_id}

        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=(outcomes, suggestion_fact_ids)):
            store.detect_implicit_feedback({"prompt": "test"}, [])

        # Original fact should be boosted, not a new REVIEW fact created
        # +0.2 boost: 0.7 -> 0.9
        assert original.metadata.confidence == pytest.approx(0.9)
        assert original.metadata.success_count == 1
        assert original.metadata.effectiveness_n == 1
        assert original.metadata.effectiveness_c == pytest.approx(1.5)
        # No new REVIEW facts should exist
        review_facts = [f for f in store._facts if f.kind == FactKind.REVIEW]
        assert len(review_facts) == 0

    def test_linked_outcome_normalizes_absolute_suggestion_path(self, store):
        """Absolute saved suggestion paths should match relative outcome paths."""

        original = store.add_fact(
            subject="bugfix: normalize linkage",
            body="Suggestion: fix path lookup",
            kind=FactKind.PATTERN,
            confidence=0.7,
        )
        absolute_path = f"{store.codebase_root}/src/foo.py"
        outcomes = [Outcome(
            outcome_type=OutcomeType.ACCEPTED,
            file_path="src/foo.py",
            suggestion_description="fix path lookup",
            suggestion_confidence=0.7,
        )]

        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=(outcomes, {absolute_path: original.id})):
            store.detect_implicit_feedback({"prompt": "test"}, [])

        assert original.metadata.confidence == pytest.approx(0.9)
        assert original.metadata.success_count == 1
        assert original.metadata.effectiveness_n == 1
        review_facts = [f for f in store._facts if f.kind == FactKind.REVIEW]
        assert len(review_facts) == 0

    def test_accepted_outcome_without_linkage_creates_review(self, store):
        """When no fact ID is linked, fall back to creating a REVIEW fact."""

        outcomes = [Outcome(
            outcome_type=OutcomeType.ACCEPTED,
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

        outcomes = [Outcome(
            outcome_type=OutcomeType.INDEPENDENT,
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

        original = store.add_fact(
            subject="pattern: error handling",
            body="Wrap I/O in try/except",
            kind=FactKind.PATTERN,
            confidence=0.5,
        )

        for _ in range(3):
            outcomes = [Outcome(
                outcome_type=OutcomeType.ACCEPTED,
                file_path="src/store.py",
                suggestion_description="wrap in try/except",
                suggestion_confidence=0.5,
            )]
            with patch.object(store._outcome_tracker, "detect_outcomes",
                              return_value=(outcomes, {"src/store.py": original.id})):
                store.detect_implicit_feedback({"prompt": "test"}, [])

        assert original.metadata.success_count == 3
        assert original.metadata.effectiveness_n == 3
        assert original.metadata.effectiveness_c == pytest.approx(4.5)
        # 3 x +0.2 = 0.6, starting from 0.5, capped at 1.0
        assert original.metadata.confidence == pytest.approx(1.0)


    def test_unverified_outcome_does_not_mutate_linked_fact(self, store):
        """Missing verification is not success, even for legacy linked facts."""

        original = store.add_fact(
            subject="bugfix: add retry",
            body="Suggestion: add retry logic",
            kind=FactKind.PATTERN,
            confidence=0.7,
        )

        outcomes = [Outcome(
            outcome_type=OutcomeType.UNVERIFIED,
            file_path="src/foo.py",
            suggestion_description="add retry logic",
            suggestion_confidence=0.7,
        )]
        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=(outcomes, {"src/foo.py": original.id})):
            store.detect_implicit_feedback({"prompt": "test"}, [])

        assert original.metadata.confidence == pytest.approx(0.7)
        assert original.metadata.success_count == 0
        assert original.metadata.effectiveness_n == 0
        assert original.metadata.effectiveness_c == pytest.approx(0.0)
        # No REVIEW facts created
        review_facts = [f for f in store._facts if f.kind == FactKind.REVIEW]
        assert len(review_facts) == 0

    def test_unverified_outcome_no_linkage_skipped(self, store):
        """Unverified outcomes with no linked fact are silently skipped."""

        initial_count = len(store._facts)
        outcomes = [Outcome(
            outcome_type=OutcomeType.UNVERIFIED,
            file_path="src/bar.py",
            suggestion_description="some change",
            suggestion_confidence=0.5,
        )]
        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=(outcomes, {})):
            store.detect_implicit_feedback({"prompt": "test"}, [])

        # No new facts created
        assert len(store._facts) == initial_count

    def test_single_accepted_episode_remains_candidate_only(self, store):
        """One verified result is insufficient to create durable knowledge."""
        from neo.memory.episodes import (
            LearningEpisode,
            LearningEpisodeStore,
            MemoryCandidateEvidence,
        )

        episode = LearningEpisode(episode_id="ep-one", project_id=store.project_id)
        episode.memory_candidates.append(MemoryCandidateEvidence(
            candidate_id="candidate-one",
            suggestion_id="suggestion-one",
            subject="pattern: validate input [src/api.py]",
            body="Suggestion: validate input before processing",
            kind="pattern",
        ))
        LearningEpisodeStore(store.project_id).save(episode)
        outcome = Outcome(
            outcome_type=OutcomeType.ACCEPTED,
            file_path="src/api.py",
            suggestion_id="suggestion-one",
            learning_episode_id="ep-one",
            candidate_id="candidate-one",
            candidate_subject="pattern: validate input [src/api.py]",
            candidate_body="Suggestion: validate input before processing",
            candidate_kind="pattern",
        )

        with patch.object(store._outcome_tracker, "detect_outcomes", return_value=([outcome], {})):
            store.detect_implicit_feedback({"prompt": "next"}, [])

        assert not [f for f in store.entries if "episode-derived" in f.tags]
        persisted = LearningEpisodeStore(store.project_id).load("ep-one")
        assert persisted.memory_candidates[0].status == "supported_once"

    def test_repeated_accepted_episodes_promote_durable_pattern(self, store):
        """Two independent accepted episodes promote one attributed project fact."""
        from neo.memory.episodes import (
            LearningEpisode,
            LearningEpisodeStore,
            MemoryCandidateEvidence,
        )

        episode_store = LearningEpisodeStore(store.project_id)
        outcomes = []
        for suffix in ("one", "two"):
            episode_id = f"ep-{suffix}"
            candidate_id = f"candidate-{suffix}"
            suggestion_id = f"suggestion-{suffix}"
            episode = LearningEpisode(episode_id=episode_id, project_id=store.project_id)
            episode.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=candidate_id,
                suggestion_id=suggestion_id,
                subject="pattern: validate input [src/api.py]",
                body="Suggestion: validate input before processing",
                kind="pattern",
            ))
            episode_store.save(episode)
            outcomes.append(Outcome(
                outcome_type=OutcomeType.ACCEPTED,
                file_path="src/api.py",
                suggestion_id=suggestion_id,
                learning_episode_id=episode_id,
                candidate_id=candidate_id,
                candidate_subject="pattern: validate input [src/api.py]",
                candidate_body="Suggestion: validate input before processing",
                candidate_kind="pattern",
            ))

        for outcome in outcomes:
            with patch.object(
                store._outcome_tracker, "detect_outcomes", return_value=([outcome], {})
            ):
                store.detect_implicit_feedback({"prompt": "next"}, [])

        promoted = [f for f in store.entries if "episode-derived" in f.tags]
        assert len(promoted) == 1
        assert promoted[0].metadata.success_count == 2
        assert set(promoted[0].supporting_episode_ids) == {"ep-one", "ep-two"}
        assert "durable" in promoted[0].tags
        assert "probation" not in promoted[0].tags

    def test_attributed_contradictions_demote_then_roll_back_only_derived_fact(self, store):
        """Repeated corrections invalidate their fact and preserve unrelated memory."""
        from neo.memory.episodes import (
            LearningEpisode,
            LearningEpisodeStore,
            MemoryCandidateEvidence,
        )

        unrelated = store.add_fact(
            subject="unrelated verified convention",
            body="Keep audit timestamps in UTC.",
            kind=FactKind.PATTERN,
            confidence=0.8,
            tags=["durable"],
        )
        episode_store = LearningEpisodeStore(store.project_id)
        accepted = []
        for index in range(2):
            episode_id = f"rollback-{index}"
            candidate_id = f"rollback-candidate-{index}"
            suggestion_id = f"rollback-suggestion-{index}"
            episode = LearningEpisode(episode_id=episode_id, project_id=store.project_id)
            episode.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=candidate_id,
                suggestion_id=suggestion_id,
                subject="pattern: validate input [src/api.py]",
                body="Suggestion: validate input before processing",
                kind="pattern",
            ))
            episode_store.save(episode)
            accepted.append(Outcome(
                outcome_type=OutcomeType.ACCEPTED,
                file_path="src/api.py",
                suggestion_id=suggestion_id,
                learning_episode_id=episode_id,
                candidate_id=candidate_id,
                candidate_subject="pattern: validate input [src/api.py]",
                candidate_body="Suggestion: validate input before processing",
                candidate_kind="pattern",
            ))

        for outcome in accepted:
            with patch.object(
                store._outcome_tracker, "detect_outcomes", return_value=([outcome], {})
            ):
                store.detect_implicit_feedback({"prompt": "next"}, [])

        promoted = next(f for f in store.entries if "episode-derived" in f.tags)
        original_confidence = promoted.metadata.confidence

        for index, accepted_outcome in enumerate(accepted):
            correction = Outcome(
                outcome_type=OutcomeType.MODIFIED,
                file_path=accepted_outcome.file_path,
                suggestion_id=accepted_outcome.suggestion_id,
                learning_episode_id=accepted_outcome.learning_episode_id,
                candidate_id=accepted_outcome.candidate_id,
                candidate_subject=accepted_outcome.candidate_subject,
                candidate_body=accepted_outcome.candidate_body,
                candidate_kind=accepted_outcome.candidate_kind,
                diff_summary="User restored the required behavior.",
            )
            with patch.object(
                store._outcome_tracker, "detect_outcomes", return_value=([correction], {})
            ):
                store.detect_implicit_feedback({"prompt": "next"}, [])

            if index == 0:
                assert promoted.is_valid is True
                assert promoted.metadata.confidence < original_confidence

        assert promoted.is_valid is False
        assert promoted.invalidation_reason == "repeated_attributed_contradiction"
        assert set(promoted.contradicting_episode_ids) == {"rollback-0", "rollback-1"}
        assert promoted.embedding is None
        assert unrelated.is_valid is True
        assert unrelated.metadata.confidence == pytest.approx(0.8)
        assert unrelated.contradicting_episode_ids == []

        for index in range(2):
            episode = episode_store.load(f"rollback-{index}")
            assert episode.memory_candidates[0].status == "contradicted"
            operations = [mutation.operation for mutation in episode.memory_mutations]
            expected = "rollback_contradicted_fact" if index == 1 else "demote_contradicted_fact"
            assert expected in operations
            mutation = next(item for item in episode.memory_mutations if item.operation == expected)
            assert mutation.before_state["confidence"] > mutation.after_state["confidence"]
            assert mutation.after_state["is_valid"] is (index == 0)

    def _promote_pattern_from_two_episodes(self, store, subject, body, prefix="accept"):
        """Helper: promote an episode-derived PATTERN fact from two accepted episodes."""
        from neo.memory.episodes import (
            LearningEpisode, LearningEpisodeStore, MemoryCandidateEvidence,
        )
        episode_store = LearningEpisodeStore(store.project_id)
        for index in range(2):
            ep_id = f"{prefix}-{index}"
            cand_id = f"{prefix}-cand-{index}"
            episode = LearningEpisode(episode_id=ep_id, project_id=store.project_id)
            episode.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=cand_id, suggestion_id=f"{prefix}-sug-{index}",
                subject=subject, body=body, kind="pattern",
            ))
            episode_store.save(episode)
            outcome = Outcome(
                outcome_type=OutcomeType.ACCEPTED, file_path="src/api.py",
                suggestion_id=f"{prefix}-sug-{index}", learning_episode_id=ep_id,
                candidate_id=cand_id, candidate_subject=subject,
                candidate_body=body, candidate_kind="pattern",
            )
            with patch.object(store._outcome_tracker, "detect_outcomes",
                              return_value=([outcome], {})):
                store.detect_implicit_feedback({"prompt": "next"}, [])
        return episode_store

    def test_canonical_signature_frozen_survives_text_drift(self, store):
        """T6: a promoted fact stores its canonical signature at mint, and
        rollback-resolve prefers it — so a later generalize() change (simulated
        here by mutating the fact's text) can't orphan the fact on the rollback
        path."""
        subject = "pattern: validate input [src/api.py]"
        body = "Suggestion: validate input before processing"
        self._promote_pattern_from_two_episodes(store, subject, body)
        promoted = next(f for f in store.entries if "episode-derived" in f.tags)
        assert promoted.canonical_signature  # frozen at mint
        frozen = promoted.canonical_signature

        # Simulate generalize() drift: the fact's live text now generalizes
        # somewhere else, so a recompute would miss it. The frozen signature must
        # still resolve it.
        promoted.subject = "completely unrelated wording that generalizes elsewhere"
        promoted.body = "nothing to do with the original"
        assert promoted.canonical_signature == frozen

        resolved = store._resolve_promoted_fact_by_signature(subject, body)
        assert resolved is promoted  # found via the frozen signature, not recompute

    def test_corrections_on_new_episodes_roll_back_promoted_fact(self, store):
        """C1: a correction arriving on a NEW episode (its candidate carries no
        promoted_fact_id) must still resolve the promoted fact by canonical
        signature and demote -> roll it back. This path was previously a no-op —
        the fact was never found, so nothing was demoted or rolled back."""
        from neo.memory.episodes import (
            LearningEpisode, MemoryCandidateEvidence,
        )
        subject = "pattern: validate input [src/api.py]"
        body = "Suggestion: validate input before processing"
        episode_store = self._promote_pattern_from_two_episodes(store, subject, body)

        promoted = next(f for f in store.entries if "episode-derived" in f.tags)
        assert promoted.is_valid is True
        original_conf = promoted.metadata.confidence

        # Corrections arrive on DISTINCT NEW episodes; their candidates carry NO
        # promoted_fact_id — the exact real-world case the old code couldn't reach.
        for index in range(2):
            ep_id = f"correct-{index}"
            cand_id = f"correct-cand-{index}"
            episode = LearningEpisode(episode_id=ep_id, project_id=store.project_id)
            episode.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=cand_id, suggestion_id=f"correct-sug-{index}",
                subject=subject, body=body, kind="pattern",
            ))
            assert episode.memory_candidates[0].promoted_fact_id == ""  # the broken case
            episode_store.save(episode)
            correction = Outcome(
                outcome_type=OutcomeType.MODIFIED, file_path="src/api.py",
                suggestion_id=f"correct-sug-{index}", learning_episode_id=ep_id,
                candidate_id=cand_id, candidate_subject=subject,
                candidate_body=body, candidate_kind="pattern",
                diff_summary="User corrected the suggestion.",
            )
            with patch.object(store._outcome_tracker, "detect_outcomes",
                              return_value=([correction], {})):
                store.detect_implicit_feedback({"prompt": "next"}, [])
            if index == 0:
                assert promoted.is_valid is True
                assert promoted.metadata.confidence < original_conf

        assert promoted.is_valid is False
        assert promoted.invalidation_reason == "repeated_attributed_contradiction"
        assert set(promoted.contradicting_episode_ids) == {"correct-0", "correct-1"}

    def test_rolled_back_pattern_is_not_resurrected_by_new_acceptances(self, store):
        """C1 secondary: once a pattern is rolled back by repeated contradiction,
        new accepted episodes for the same signature must NOT mint a fresh valid
        fact (which would resurrect the retracted knowledge)."""
        subject = "pattern: validate input [src/api.py]"
        body = "Suggestion: validate input before processing"
        self._promote_pattern_from_two_episodes(store, subject, body, prefix="accept")
        promoted = next(f for f in store.entries if "episode-derived" in f.tags)

        # Roll it back via two corrections on new episodes.
        from neo.memory.episodes import (
            LearningEpisode, LearningEpisodeStore, MemoryCandidateEvidence,
        )
        episode_store = LearningEpisodeStore(store.project_id)
        for index in range(2):
            ep_id = f"kill-{index}"
            episode = LearningEpisode(episode_id=ep_id, project_id=store.project_id)
            episode.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=f"kill-cand-{index}", suggestion_id=f"kill-sug-{index}",
                subject=subject, body=body, kind="pattern",
            ))
            episode_store.save(episode)
            correction = Outcome(
                outcome_type=OutcomeType.MODIFIED, file_path="src/api.py",
                suggestion_id=f"kill-sug-{index}", learning_episode_id=ep_id,
                candidate_id=f"kill-cand-{index}", candidate_subject=subject,
                candidate_body=body, candidate_kind="pattern",
            )
            with patch.object(store._outcome_tracker, "detect_outcomes",
                              return_value=([correction], {})):
                store.detect_implicit_feedback({"prompt": "next"}, [])
        assert promoted.is_valid is False

        valid_before = {f.id for f in store.entries if f.is_valid and "episode-derived" in f.tags}
        # Two more acceptances for the same signature must NOT resurrect it.
        self._promote_pattern_from_two_episodes(store, subject, body, prefix="revive")
        valid_after = {f.id for f in store.entries if f.is_valid and "episode-derived" in f.tags}
        assert valid_after == valid_before  # no new valid episode-derived fact
        assert promoted.is_valid is False

    def test_retracted_tombstone_survives_purge_so_block_is_durable(self, store):
        """The retracted-signature record must outlive the 30-day purge window,
        otherwise the re-promotion block silently expires and the pattern re-mints."""
        subject = "pattern: validate input [src/api.py]"
        body = "Suggestion: validate input before processing"
        self._promote_pattern_from_two_episodes(store, subject, body, prefix="accept")
        promoted = next(f for f in store.entries if "episode-derived" in f.tags)

        from neo.memory.episodes import (
            LearningEpisode, LearningEpisodeStore, MemoryCandidateEvidence,
        )
        episode_store = LearningEpisodeStore(store.project_id)
        for index in range(2):
            ep_id = f"kill-{index}"
            episode = LearningEpisode(episode_id=ep_id, project_id=store.project_id)
            episode.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=f"kill-cand-{index}", suggestion_id=f"kill-sug-{index}",
                subject=subject, body=body, kind="pattern",
            ))
            episode_store.save(episode)
            correction = Outcome(
                outcome_type=OutcomeType.MODIFIED, file_path="src/api.py",
                suggestion_id=f"kill-sug-{index}", learning_episode_id=ep_id,
                candidate_id=f"kill-cand-{index}", candidate_subject=subject,
                candidate_body=body, candidate_kind="pattern",
            )
            with patch.object(store._outcome_tracker, "detect_outcomes",
                              return_value=([correction], {})):
                store.detect_implicit_feedback({"prompt": "next"}, [])
        assert promoted.is_valid is False

        # Age the tombstone well past the 30-day purge window and purge.
        promoted.metadata.last_accessed = time.time() - 90 * 86400
        store.purge_dead_facts()
        survivor = next((f for f in store._facts if f.id == promoted.id), None)
        assert survivor is not None, "retracted tombstone must survive purge"
        assert survivor.invalidation_reason == "repeated_attributed_contradiction"

    def test_durable_fact_protected_by_structure_not_forgeable_tag(self, store):
        """T7: promoted facts survive the janitors on STRUCTURAL provenance
        (supporting_episode_ids, set only at promotion), NOT on the 'durable'
        tag — which a future importer could forge to mint permanent immunity."""
        # Genuinely promoted (carries supporting_episode_ids) -> protected.
        promoted = store.add_fact(
            subject="episode-derived verified pattern",
            body="always validate input at the boundary",
            kind=FactKind.PATTERN, scope=FactScope.PROJECT, confidence=0.2,
            tags=["episode-derived", "durable"],
            supporting_episode_ids=["ep-1", "ep-2"],
        )
        promoted.metadata.success_count = 0
        promoted.metadata.created_at = time.time() - 60 * 86400
        promoted.metadata.confidence = 0.2

        # Same tags, but NO promotion provenance -> the forgeable case, NOT shielded.
        forged = store.add_fact(
            subject="forged durable claim",
            body="just attach the durable tag and survive forever",
            kind=FactKind.PATTERN, scope=FactScope.PROJECT, confidence=0.2,
            tags=["episode-derived", "durable"],
        )
        forged.metadata.success_count = 0
        forged.metadata.created_at = time.time() - 60 * 86400
        forged.metadata.confidence = 0.2

        store.prune_stale_facts()
        store.demote_unhelpful_facts()

        assert promoted.is_valid is True   # protected by structure
        assert forged.is_valid is False    # tag alone does not shield

    def test_protected_facts_do_not_consume_the_eviction_budget(self, store):
        """T8: durable/protected facts are durable EXTRA storage — they don't
        count against the ordinary eviction budget, so ordinary facts keep their
        full limit and the scope isn't pinned over-limit by protected growth."""
        from neo.memory.store import SCOPE_LIMITS
        limit = SCOPE_LIMITS[FactScope.SESSION.value]

        protected = []
        for i in range(5):
            p = Fact(id=f"prot{i}", subject=f"promoted {i}", body="v",
                     kind=FactKind.PATTERN, scope=FactScope.SESSION,
                     supporting_episode_ids=[f"e{i}a", f"e{i}b"],
                     metadata=FactMetadata(confidence=0.3))
            store._facts.append(p)
            protected.append(p)
        for i in range(limit):  # ordinary facts exactly at the limit
            store._facts.append(Fact(
                id=f"ord{i}", subject=f"ordinary {i}", body="b",
                kind=FactKind.PATTERN, scope=FactScope.SESSION,
                metadata=FactMetadata(confidence=0.5)))

        store._enforce_scope_limit(FactScope.SESSION)

        # All protected survive, and ordinary facts kept their full budget — the
        # 5 protected facts did NOT force 5 ordinary evictions (the old bug).
        assert all(p.is_valid for p in protected)
        ordinary_valid = [
            f for f in store._facts
            if f.is_valid and f.scope == FactScope.SESSION and not store._is_protected(f)
        ]
        assert len(ordinary_valid) == limit

    def test_dedup_collapses_duplicate_global_signatures(self, store):
        """T9: two global facts minted for the same canonical signature (the
        concurrency window merge-on-save can't dedup by id) are collapsed — the
        richest kept, the duplicate superseded + invalidated, episodes merged."""
        sig = "validate input signature"
        common = dict(
            subject="validate input", body="do it", kind=FactKind.PATTERN,
            scope=FactScope.GLOBAL, tags=["episode-derived", "durable"],
            canonical_signature=sig,
        )
        a = Fact(id="ga", supporting_episode_ids=["e1", "e2"],
                 metadata=FactMetadata(confidence=0.6), **common)
        b = Fact(id="gb", supporting_episode_ids=["e3", "e4", "e5"],
                 metadata=FactMetadata(confidence=0.6), **common)
        store._facts.extend([a, b])

        assert store._dedup_global_signatures() == 1
        # b has more supporting episodes -> kept; a superseded + invalidated.
        assert b.is_valid is True and a.is_valid is False
        assert a.superseded_by == "gb"
        assert set(b.supporting_episode_ids) == {"e1", "e2", "e3", "e4", "e5"}
        assert store._dedup_global_signatures() == 0  # idempotent

    def test_single_acceptance_leaves_promoted_fact_id_empty(self, store):
        """T2: promoted_fact_id is written ONLY by promotion. One acceptance
        (below the 2-episode bar) must leave it empty, not point at an unrelated
        mutation-target fact."""
        from neo.memory.episodes import (
            LearningEpisode, LearningEpisodeStore, MemoryCandidateEvidence,
        )
        episode_store = LearningEpisodeStore(store.project_id)
        ep = LearningEpisode(episode_id="solo", project_id=store.project_id)
        ep.memory_candidates.append(MemoryCandidateEvidence(
            candidate_id="solo-cand", suggestion_id="solo-sug",
            subject="pattern: cache results [src/a.py]", body="memoize the call",
            kind="pattern",
        ))
        episode_store.save(ep)
        outcome = Outcome(
            outcome_type=OutcomeType.ACCEPTED, file_path="src/a.py",
            suggestion_id="solo-sug", learning_episode_id="solo",
            candidate_id="solo-cand", candidate_subject="pattern: cache results [src/a.py]",
            candidate_body="memoize the call", candidate_kind="pattern",
        )
        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=([outcome], {})):
            store.detect_implicit_feedback({"prompt": "next"}, [])
        cand = episode_store.load("solo").memory_candidates[0]
        assert cand.status == "supported_once"
        assert cand.promoted_fact_id == ""  # not promoted -> no fact id

    def test_cross_project_scan_gated_on_project_promotion(self, store):
        """T3: the all-projects cross-project scan is off the per-request path —
        it must NOT run on an acceptance that doesn't project-promote, only when
        this project newly clears the project bar."""
        from neo.memory.episodes import (
            LearningEpisode, LearningEpisodeStore, MemoryCandidateEvidence,
        )
        episode_store = LearningEpisodeStore(store.project_id)
        subject, body = "pattern: x [a.py]", "do x safely"

        def _accept(idx):
            ep_id, cand_id, sug_id = f"g{idx}", f"g{idx}c", f"g{idx}s"
            ep = LearningEpisode(episode_id=ep_id, project_id=store.project_id)
            ep.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=cand_id, suggestion_id=sug_id,
                subject=subject, body=body, kind="pattern"))
            episode_store.save(ep)
            return Outcome(
                outcome_type=OutcomeType.ACCEPTED, file_path="a.py",
                suggestion_id=sug_id, learning_episode_id=ep_id, candidate_id=cand_id,
                candidate_subject=subject, candidate_body=body, candidate_kind="pattern")

        # First acceptance: not enough to promote -> no cross-project scan.
        with patch.object(store, "_promote_cross_project_candidate") as mock_cp, \
                patch.object(store._outcome_tracker, "detect_outcomes",
                             return_value=([_accept(0)], {})):
            store.detect_implicit_feedback({"prompt": "n"}, [])
            mock_cp.assert_not_called()

        # Second acceptance: project-promotes -> exactly one cross-project scan.
        with patch.object(store, "_promote_cross_project_candidate") as mock_cp, \
                patch.object(store._outcome_tracker, "detect_outcomes",
                             return_value=([_accept(1)], {})):
            store.detect_implicit_feedback({"prompt": "n"}, [])
            mock_cp.assert_called_once()

    def test_later_regression_api_records_normalized_evidence_and_rolls_back(self, store):
        """Delayed failures use stable attribution rather than similarity fanout."""
        from neo.memory.episodes import (
            LearningEpisode,
            LearningEpisodeStore,
            MemoryCandidateEvidence,
            SuggestionEvidence,
        )

        episode_store = LearningEpisodeStore(store.project_id)
        accepted = []
        for index in range(2):
            episode_id = f"regression-{index}"
            suggestion_id = f"regression-suggestion-{index}"
            candidate_id = f"regression-candidate-{index}"
            episode = LearningEpisode(episode_id=episode_id, project_id=store.project_id)
            episode.suggestions.append(SuggestionEvidence(
                suggestion_id=suggestion_id,
                description="validate input before processing",
                file_path="src/api.py",
                confidence=0.7,
            ))
            episode.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=candidate_id,
                suggestion_id=suggestion_id,
                subject="pattern: validate input [src/api.py]",
                body="Suggestion: validate input before processing",
                kind="pattern",
            ))
            episode_store.save(episode)
            accepted.append(Outcome(
                outcome_type=OutcomeType.ACCEPTED,
                file_path="src/api.py",
                suggestion_id=suggestion_id,
                learning_episode_id=episode_id,
                candidate_id=candidate_id,
                candidate_subject="pattern: validate input [src/api.py]",
                candidate_body="Suggestion: validate input before processing",
                candidate_kind="pattern",
            ))

        for outcome in accepted:
            with patch.object(
                store._outcome_tracker, "detect_outcomes", return_value=([outcome], {})
            ):
                store.detect_implicit_feedback({"prompt": "next"}, [])

        promoted = next(f for f in store.entries if "episode-derived" in f.tags)
        for index in range(2):
            affected = store.record_later_regression(
                learning_episode_id=f"regression-{index}",
                suggestion_id=f"regression-suggestion-{index}",
                summary="A later test reproduced data corruption.",
                repository_revision=f"bad-revision-{index}",
            )
            assert affected is promoted

        assert promoted.is_valid is False
        assert promoted.invalidation_reason == "repeated_attributed_contradiction"
        for index in range(2):
            episode = episode_store.load(f"regression-{index}")
            regression = [
                evidence for evidence in episode.verification
                if evidence.kind == "later_regression"
            ]
            assert len(regression) == 1
            assert regression[0].status == "failed"
            assert regression[0].repository_revision == f"bad-revision-{index}"
            assert episode.final_outcome == "regression"
            assert episode.outcome_details["evidence_summary"] == (
                "A later test reproduced data corruption."
            )
            assert len(episode.outcome_details["evidence_sha256"]) == 64

    def test_failed_verification_blocks_repeated_acceptance_promotion(self, store):
        """Acceptance cannot turn deterministically rejected output into a fact."""
        from neo.memory.episodes import (
            LearningEpisode,
            LearningEpisodeStore,
            MemoryCandidateEvidence,
            VerificationEvidence,
        )

        episode_store = LearningEpisodeStore(store.project_id)
        outcomes = []
        for index in range(2):
            episode_id = f"failed-{index}"
            candidate_id = f"failed-candidate-{index}"
            episode = LearningEpisode(episode_id=episode_id, project_id=store.project_id)
            episode.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=candidate_id,
                suggestion_id=f"failed-suggestion-{index}",
                subject="pattern: skip validation [src/api.py]",
                body="Suggestion: remove the required validation",
                kind="pattern",
            ))
            episode.verification.append(VerificationEvidence(
                verification_id=f"check-{index}",
                kind="static_check",
                status="failed",
                tool_name="constraint_verifier",
                summary="required validation was removed",
            ))
            episode_store.save(episode)
            outcomes.append(Outcome(
                outcome_type=OutcomeType.ACCEPTED,
                file_path="src/api.py",
                suggestion_id=f"failed-suggestion-{index}",
                learning_episode_id=episode_id,
                candidate_id=candidate_id,
                candidate_subject="pattern: skip validation [src/api.py]",
                candidate_body="Suggestion: remove the required validation",
                candidate_kind="pattern",
            ))

        for outcome in outcomes:
            with patch.object(
                store._outcome_tracker, "detect_outcomes", return_value=([outcome], {})
            ):
                store.detect_implicit_feedback({"prompt": "next"}, [])

        assert not [f for f in store.entries if "episode-derived" in f.tags]
        for index in range(2):
            persisted = episode_store.load(f"failed-{index}")
            assert persisted.memory_candidates[0].status == "rejected_by_verification"

    def test_repeated_architecture_candidates_do_not_auto_promote(self, store):
        """Architecture knowledge requires a stronger policy than coding patterns."""
        from neo.memory.episodes import (
            LearningEpisode,
            LearningEpisodeStore,
            MemoryCandidateEvidence,
        )

        episode_store = LearningEpisodeStore(store.project_id)
        for index in range(3):
            episode_id = f"arch-{index}"
            candidate_id = f"arch-candidate-{index}"
            episode = LearningEpisode(episode_id=episode_id, project_id=store.project_id)
            episode.memory_candidates.append(MemoryCandidateEvidence(
                candidate_id=candidate_id,
                suggestion_id=f"arch-suggestion-{index}",
                subject="architecture: split the service",
                body="Suggestion: introduce a new service boundary",
                kind="architecture",
            ))
            episode_store.save(episode)
            outcome = Outcome(
                outcome_type=OutcomeType.ACCEPTED,
                file_path="src/service.py",
                learning_episode_id=episode_id,
                candidate_id=candidate_id,
                candidate_subject="architecture: split the service",
                candidate_body="Suggestion: introduce a new service boundary",
                candidate_kind="architecture",
            )
            with patch.object(
                store._outcome_tracker, "detect_outcomes", return_value=([outcome], {})
            ):
                store.detect_implicit_feedback({"prompt": "next"}, [])

        assert not [f for f in store.entries if "episode-derived" in f.tags]

    def test_reconcile_catches_asymmetric_cross_project_evidence(self, tmp_path):
        """T3: when a minority (1-episode) project supplies the episode that
        completes the global bar, the request-path gate never fires (that project
        doesn't project-promote), so the global fact is stranded. The observer
        reconcile sweep mints it. Idempotent on a second run."""
        from neo.memory.episodes import (
            LearningEpisode, LearningEpisodeStore, MemoryCandidateEvidence,
        )
        facts_dir = tmp_path / "facts"
        episodes_dir = tmp_path / "episodes"
        subject = "pattern: validate input [src/api.py]"
        body = "Validate input before processing."

        def _project(name, n_accepts):
            root = tmp_path / name
            root.mkdir()
            st = FactStore(codebase_root=str(root), eager_init=False,
                           facts_dir=facts_dir, episodes_dir=episodes_dir)
            es = LearningEpisodeStore(st.project_id, base_dir=episodes_dir)
            for i in range(n_accepts):
                ep_id, cid, sid = f"{name}-{i}", f"{name}-c-{i}", f"{name}-s-{i}"
                ep = LearningEpisode(episode_id=ep_id, project_id=st.project_id)
                ep.memory_candidates.append(MemoryCandidateEvidence(
                    candidate_id=cid, suggestion_id=sid,
                    subject=subject, body=body, kind="pattern"))
                es.save(ep)
                outcome = Outcome(
                    outcome_type=OutcomeType.ACCEPTED, file_path="src/api.py",
                    suggestion_id=sid, learning_episode_id=ep_id, candidate_id=cid,
                    candidate_subject=subject, candidate_body=body, candidate_kind="pattern")
                with patch.object(st._outcome_tracker, "detect_outcomes",
                                  return_value=([outcome], {})):
                    st.detect_implicit_feedback({"prompt": "n"}, [])
            return st

        _project("alpha", 3)          # project-promotes, but beta is empty then
        beta = _project("beta", 1)    # 1 episode -> supported_once, no promotion event

        # Gate missed it: 4 episodes across 2 projects on disk, but no global fact.
        assert [f for f in beta.entries if f.scope == FactScope.GLOBAL] == []

        # Observer reconcile catches the stranded evidence.
        assert beta.reconcile_cross_project_promotions() == 1
        globals_after = [
            f for f in beta.entries
            if f.scope == FactScope.GLOBAL and "episode-derived" in f.tags
        ]
        assert len(globals_after) == 1
        assert len(set(globals_after[0].supporting_episode_ids)) >= 4
        # Idempotent: nothing new on a second sweep.
        assert beta.reconcile_cross_project_promotions() == 0

        # Rollback regression: if the global fact is retracted by repeated
        # attributed contradiction, reconcile must NOT resurrect it — the
        # supporting candidates still carry promoted_global_fact_id, so the
        # (validity-independent) idempotency guard keeps it retracted.
        promoted_global = globals_after[0]
        promoted_global.is_valid = False
        promoted_global.invalidation_reason = "repeated_attributed_contradiction"
        beta.save()
        assert beta.reconcile_cross_project_promotions() == 0
        still_valid_globals = [
            f for f in beta.entries
            if f.is_valid and f.scope == FactScope.GLOBAL and "episode-derived" in f.tags
        ]
        assert still_valid_globals == []  # not resurrected

    def test_global_promotion_requires_four_episodes_across_two_projects(self, tmp_path):
        """One project is insufficient; cross-project evidence is generalized safely."""
        from neo.memory.episodes import (
            LearningEpisode,
            LearningEpisodeStore,
            MemoryCandidateEvidence,
        )

        facts_dir = tmp_path / "global-facts"
        episodes_dir = tmp_path / "global-episodes"
        subject = "pattern: validate credentials [src/private.py]"
        body = "Validate credentials with api_key=super-secret-value before processing."

        stores = []
        for project in ("alpha", "beta"):
            root = tmp_path / project
            root.mkdir()
            project_store = FactStore(
                codebase_root=str(root),
                eager_init=False,
                facts_dir=facts_dir,
                episodes_dir=episodes_dir,
            )
            stores.append(project_store)
            episode_store = LearningEpisodeStore(
                project_store.project_id, base_dir=episodes_dir
            )
            for index in range(2):
                episode_id = f"{project}-{index}"
                candidate_id = f"candidate-{project}-{index}"
                suggestion_id = f"suggestion-{project}-{index}"
                episode = LearningEpisode(
                    episode_id=episode_id,
                    project_id=project_store.project_id,
                )
                episode.memory_candidates.append(MemoryCandidateEvidence(
                    candidate_id=candidate_id,
                    suggestion_id=suggestion_id,
                    subject=subject,
                    body=body,
                    kind="pattern",
                ))
                episode_store.save(episode)
                outcome = Outcome(
                    outcome_type=OutcomeType.ACCEPTED,
                    file_path="src/private.py",
                    suggestion_id=suggestion_id,
                    learning_episode_id=episode_id,
                    candidate_id=candidate_id,
                    candidate_subject=subject,
                    candidate_body=body,
                    candidate_kind="pattern",
                )
                with patch.object(
                    project_store._outcome_tracker,
                    "detect_outcomes",
                    return_value=([outcome], {}),
                ):
                    project_store.detect_implicit_feedback({"prompt": "next"}, [])

            if project == "alpha":
                assert not [
                    fact for fact in project_store.entries
                    if fact.scope == FactScope.GLOBAL
                ]

        global_facts = [
            fact for fact in stores[1].entries if fact.scope == FactScope.GLOBAL
        ]
        assert len(global_facts) == 1
        global_fact = global_facts[0]
        assert len(global_fact.supporting_episode_ids) == 4
        assert global_fact.project_id == ""
        assert "src/private.py" not in global_fact.subject
        assert "super-secret-value" not in global_fact.body
        assert "[REDACTED]" in global_fact.body
        for project_store, project in zip(stores, ("alpha", "beta")):
            episode_store = LearningEpisodeStore(
                project_store.project_id, base_dir=episodes_dir
            )
            for index in range(2):
                episode = episode_store.load(f"{project}-{index}")
                assert episode.memory_candidates[0].promoted_global_fact_id == global_fact.id
                assert any(
                    mutation.operation == "promote_cross_project_candidate"
                    and mutation.fact_id == global_fact.id
                    for mutation in episode.memory_mutations
                )

    def test_cap_independent_facts(self, store):
        """_cap_independent_facts invalidates excess independent facts."""
        from neo.memory.store import MAX_INDEPENDENT_FACTS

        # Create more than MAX_INDEPENDENT_FACTS independent facts
        for i in range(MAX_INDEPENDENT_FACTS + 20):
            fact = Fact(
                subject=f"outcome:independent src/file{i}.py",
                body=f"User changed src/file{i}.py",
                kind=FactKind.REVIEW,
                scope=FactScope.PROJECT,
                org_id="testorg",
                project_id="testproj1234",
                metadata=FactMetadata(
                    confidence=0.2,
                    created_at=time.time() - (MAX_INDEPENDENT_FACTS + 20 - i),
                ),
                tags=["outcome", "independent"],
            )
            store._facts.append(fact)

        store._cap_independent_facts()

        valid_indep = [f for f in store._facts if f.is_valid and "independent" in f.tags]
        assert len(valid_indep) == MAX_INDEPENDENT_FACTS

    def test_replay_linked_feedback_updates_only_linked_facts(self, store):
        """Maintenance replay updates linked facts without creating REVIEW noise."""

        original = store.add_fact(
            subject="bugfix: replay linkage",
            body="Suggestion: add guard",
            kind=FactKind.PATTERN,
            confidence=0.6,
        )
        outcomes = [
            Outcome(
                outcome_type=OutcomeType.ACCEPTED,
                file_path="src/foo.py",
                suggestion_description="add guard",
                suggestion_confidence=0.6,
            ),
            Outcome(
                outcome_type=OutcomeType.INDEPENDENT,
                file_path="src/bar.py",
                diff_summary="+noise",
            ),
        ]

        with (
            patch.object(store._outcome_tracker, "collect_outcomes",
                         return_value=(outcomes, {"src/foo.py": original.id})) as collect,
            patch.object(store._outcome_tracker, "_clear_session_log") as clear,
        ):
            stats = store.replay_linked_feedback()

        collect.assert_called_once_with(clear_processed=False, include_fallback=False)
        clear.assert_called_once()
        assert stats["linked_updates"] == 1
        assert stats["skipped_independent"] == 1
        assert original.metadata.confidence == pytest.approx(0.8)
        assert original.metadata.success_count == 1
        assert original.metadata.effectiveness_n == 1
        review_facts = [f for f in store._facts if f.kind == FactKind.REVIEW]
        assert len(review_facts) == 0

    def test_replay_linked_feedback_dry_run_does_not_mutate_or_clear(self, store):
        """Dry-run reports linked outcomes without changing memory."""

        original = store.add_fact(
            subject="bugfix: dry run",
            body="Suggestion: add guard",
            kind=FactKind.PATTERN,
            confidence=0.6,
        )
        outcomes = [Outcome(
            outcome_type=OutcomeType.MODIFIED,
            file_path="src/foo.py",
            suggestion_description="add guard",
            suggestion_confidence=0.6,
        )]

        with (
            patch.object(store._outcome_tracker, "collect_outcomes",
                         return_value=(outcomes, {"src/foo.py": original.id})),
            patch.object(store._outcome_tracker, "_clear_session_log") as clear,
        ):
            stats = store.replay_linked_feedback(dry_run=True)

        clear.assert_not_called()
        assert stats["linked_updates"] == 1
        assert stats["modified"] == 1
        assert original.metadata.confidence == pytest.approx(0.6)
        assert original.metadata.effectiveness_n == 0

    def test_replay_unverified_feedback_does_not_reinforce_fact(self, store):
        """Maintenance replay cannot reinterpret absent evidence as success."""
        original = store.add_fact(
            subject="bugfix: no evidence",
            body="Suggestion: add guard",
            kind=FactKind.PATTERN,
            confidence=0.6,
        )
        outcome = Outcome(
            outcome_type=OutcomeType.UNVERIFIED,
            file_path="src/foo.py",
            suggestion_description="add guard",
            suggestion_confidence=0.6,
        )

        with patch.object(
            store._outcome_tracker,
            "collect_outcomes",
            return_value=([outcome], {"src/foo.py": original.id}),
        ):
            stats = store.replay_linked_feedback()

        assert stats["unverified"] == 1
        assert stats["linked_updates"] == 0
        assert original.metadata.confidence == pytest.approx(0.6)
        assert original.metadata.success_count == 0


class TestRetrievedFactAttribution:
    def test_only_explicitly_used_fact_receives_success_credit(self, store):
        used = store.add_fact(
            subject="project convention",
            body="Use typed identifiers",
            kind=FactKind.PATTERN,
            confidence=0.6,
        )
        merely_retrieved = store.add_fact(
            subject="unrelated convention",
            body="Use UTC timestamps",
            kind=FactKind.PATTERN,
            confidence=0.6,
        )
        outcome = Outcome(
            outcome_type=OutcomeType.ACCEPTED,
            file_path="src/new.py",
            retrieved_fact_ids=[used.id, merely_retrieved.id],
            used_fact_ids=[used.id],
        )

        with patch.object(
            store._outcome_tracker,
            "detect_outcomes",
            return_value=([outcome], {}),
        ), patch.object(store._outcome_tracker, "compute_arch_delta", return_value=None):
            store.detect_implicit_feedback({}, [])

        assert used.metadata.success_count == 1
        assert used.metadata.effectiveness_n == 1
        assert used.metadata.confidence == pytest.approx(0.6)
        assert merely_retrieved.metadata.success_count == 0
        assert merely_retrieved.metadata.effectiveness_n == 0

    def test_attributed_modification_applies_bounded_demotion(self, store):
        used = store.add_fact(
            subject="project convention",
            body="Use typed identifiers",
            kind=FactKind.PATTERN,
            confidence=0.6,
        )
        outcome = Outcome(
            outcome_type=OutcomeType.MODIFIED,
            file_path="src/new.py",
            retrieved_fact_ids=[used.id],
            used_fact_ids=[used.id],
        )

        with patch.object(
            store._outcome_tracker,
            "detect_outcomes",
            return_value=([outcome], {}),
        ), patch.object(store._outcome_tracker, "compute_arch_delta", return_value=None):
            store.detect_implicit_feedback({}, [])

        assert used.metadata.confidence == pytest.approx(0.55)
        assert used.metadata.success_count == 0
        assert used.metadata.effectiveness_n == 1
        assert used.metadata.effectiveness_c < 0


class TestArchDeltaModulation:
    """Verify ArchDelta from outcomes changes the confidence adjustment math."""

    def _setup(self, store):
        from neo.architecture_metrics import ArchDelta
        original = store.add_fact(
            subject="bugfix: cycle-introducing change",
            body="Suggestion: cross-import",
            kind=FactKind.PATTERN,
            confidence=0.7,
        )
        outcome = Outcome(
            outcome_type=OutcomeType.ACCEPTED,
            file_path="src/foo.py",
            suggestion_description="introduce shared helper",
            suggestion_confidence=0.7,
        )
        return original, outcome, ArchDelta

    def test_neutral_arch_delta_uses_full_boost(self, store):
        """Existing baseline behavior: +0.2 boost when arch is neutral."""
        original, outcome, _ = self._setup(store)
        with (
            patch.object(store._outcome_tracker, "detect_outcomes",
                         return_value=([outcome], {"src/foo.py": original.id})),
            patch.object(store._outcome_tracker, "compute_arch_delta",
                         return_value=None),
        ):
            store.detect_implicit_feedback({"prompt": "x"}, [])
        # 0.7 + 0.2 = 0.9 (the existing accepted-linkage behavior)
        assert original.metadata.confidence == pytest.approx(0.9)

    def test_regression_arch_delta_softens_boost(self, store):
        """A session that introduced a cycle earns less trust on accept."""
        original, outcome, ArchDelta = self._setup(store)
        regression = ArchDelta(cycles_delta=1, god_files_delta=0, max_depth_delta=0)
        with (
            patch.object(store._outcome_tracker, "detect_outcomes",
                         return_value=([outcome], {"src/foo.py": original.id})),
            patch.object(store._outcome_tracker, "compute_arch_delta",
                         return_value=regression),
        ):
            store.detect_implicit_feedback({"prompt": "x"}, [])
        # 0.7 + (0.2 - 0.1) = 0.8 — regression weakens the accept signal.
        assert original.metadata.confidence == pytest.approx(0.8)

    def test_improvement_arch_delta_amplifies_boost(self, store):
        """A session that removed a cycle earns extra trust on accept."""
        original, outcome, ArchDelta = self._setup(store)
        improvement = ArchDelta(cycles_delta=-1, god_files_delta=0, max_depth_delta=0)
        with (
            patch.object(store._outcome_tracker, "detect_outcomes",
                         return_value=([outcome], {"src/foo.py": original.id})),
            patch.object(store._outcome_tracker, "compute_arch_delta",
                         return_value=improvement),
        ):
            store.detect_implicit_feedback({"prompt": "x"}, [])
        # 0.7 + (0.2 + 0.1) = 1.0 (clamped by min(1.0, ...))
        assert original.metadata.confidence == pytest.approx(1.0)

    def test_modified_outcome_regression_strengthens_penalty(self, store):
        """MODIFIED + regression = stronger demote (accept already failed)."""
        original = store.add_fact(
            subject="bugfix: failed attempt", body="x",
            kind=FactKind.PATTERN, confidence=0.7,
        )
        outcome = Outcome(
            outcome_type=OutcomeType.MODIFIED,
            file_path="src/foo.py",
            suggestion_description="x",
            suggestion_confidence=0.7,
        )
        from neo.architecture_metrics import ArchDelta
        regression = ArchDelta(cycles_delta=1, god_files_delta=0, max_depth_delta=0)
        with (
            patch.object(store._outcome_tracker, "detect_outcomes",
                         return_value=([outcome], {"src/foo.py": original.id})),
            patch.object(store._outcome_tracker, "compute_arch_delta",
                         return_value=regression),
        ):
            store.detect_implicit_feedback({"prompt": "x"}, [])
        # 0.7 + (-0.2 - 0.1) = 0.4 — regression deepens the demotion.
        assert original.metadata.confidence == pytest.approx(0.4)
        assert original.metadata.effectiveness_n == 1
        assert original.metadata.effectiveness_c == pytest.approx(-0.5)


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


class TestLegacyProjectIdMigration:
    """Verifies that fact + watermark files written under the pre-remote-hash
    project ID get renamed to the new (git-remote-hashed) ID on FactStore init.
    """

    def test_renames_legacy_fact_and_watermark_files(self, tmp_facts_dir, tmp_path):
        """Legacy path-hashed files should be moved to the remote-hashed key."""
        from contextlib import ExitStack

        from neo.memory.scope import _compute_legacy_project_id

        # Codebase root is a fake path; what matters is that the mocked git
        # remote yields a remote-hashed ID distinct from the path-hashed one.
        codebase_root = str(tmp_path / "fake-repo")
        (tmp_path / "fake-repo").mkdir()
        remote_url = "git@github.com:parslee-ai/neo.git"

        legacy_id = _compute_legacy_project_id(codebase_root)
        # Pre-populate the legacy fact + watermark files
        legacy_facts = tmp_facts_dir / f"facts_project_{legacy_id}.json"
        legacy_facts.write_text('{"facts": [], "version": 1}')
        legacy_wm = tmp_facts_dir / f"synthesis_watermark_{legacy_id}.json"
        legacy_wm.write_text('{"watermark": 0}')

        with ExitStack() as stack:
            stack.enter_context(patch("neo.memory.store.FACTS_DIR", tmp_facts_dir))
            stack.enter_context(patch("neo.memory.scope._get_git_remote_url",
                                     return_value=remote_url))
            stack.enter_context(patch.object(FactStore, "_ingest_constraints"))
            stack.enter_context(patch.object(FactStore, "_ingest_seed_facts"))
            stack.enter_context(patch.object(FactStore, "_ingest_community_feed"))
            stack.enter_context(patch.object(FactStore, "_ingest_claude_memory"))
            stack.enter_context(patch.object(FactStore, "_maybe_migrate"))
            stack.enter_context(patch("neo.memory.store.FASTEMBED_AVAILABLE", False))

            store = FactStore(codebase_root=codebase_root, eager_init=True)

        # New (remote-hashed) ID must differ from the legacy one
        assert store.project_id != legacy_id
        # Legacy files moved to new ID
        assert not legacy_facts.exists()
        assert not legacy_wm.exists()
        assert (tmp_facts_dir / f"facts_project_{store.project_id}.json").exists()
        assert (tmp_facts_dir / f"synthesis_watermark_{store.project_id}.json").exists()

    def test_no_op_when_no_remote(self, tmp_facts_dir, tmp_path):
        """Without a remote, legacy ID == new ID, so nothing should be renamed."""
        from contextlib import ExitStack

        codebase_root = str(tmp_path / "no-remote")
        (tmp_path / "no-remote").mkdir()

        with ExitStack() as stack:
            stack.enter_context(patch("neo.memory.store.FACTS_DIR", tmp_facts_dir))
            stack.enter_context(patch("neo.memory.scope._get_git_remote_url",
                                     return_value=""))
            stack.enter_context(patch.object(FactStore, "_ingest_constraints"))
            stack.enter_context(patch.object(FactStore, "_ingest_seed_facts"))
            stack.enter_context(patch.object(FactStore, "_ingest_community_feed"))
            stack.enter_context(patch.object(FactStore, "_ingest_claude_memory"))
            stack.enter_context(patch.object(FactStore, "_maybe_migrate"))
            stack.enter_context(patch("neo.memory.store.FASTEMBED_AVAILABLE", False))

            store = FactStore(codebase_root=codebase_root, eager_init=True)

        from neo.memory.scope import _compute_legacy_project_id
        assert store.project_id == _compute_legacy_project_id(codebase_root)

    def test_does_not_clobber_new_file(self, tmp_facts_dir, tmp_path):
        """If both legacy and new files exist, leave the new one alone."""
        from contextlib import ExitStack

        from neo.memory.scope import _compute_legacy_project_id

        codebase_root = str(tmp_path / "repo")
        (tmp_path / "repo").mkdir()
        remote_url = "git@github.com:parslee-ai/neo.git"

        legacy_id = _compute_legacy_project_id(codebase_root)
        legacy_facts = tmp_facts_dir / f"facts_project_{legacy_id}.json"
        legacy_facts.write_text('{"legacy": true}')

        with ExitStack() as stack:
            stack.enter_context(patch("neo.memory.store.FACTS_DIR", tmp_facts_dir))
            stack.enter_context(patch("neo.memory.scope._get_git_remote_url",
                                     return_value=remote_url))
            stack.enter_context(patch.object(FactStore, "_ingest_constraints"))
            stack.enter_context(patch.object(FactStore, "_ingest_seed_facts"))
            stack.enter_context(patch.object(FactStore, "_ingest_community_feed"))
            stack.enter_context(patch.object(FactStore, "_ingest_claude_memory"))
            stack.enter_context(patch.object(FactStore, "_maybe_migrate"))
            stack.enter_context(patch("neo.memory.store.FASTEMBED_AVAILABLE", False))

            # Pre-create the new fact file so the migration must not overwrite it
            from neo.memory.scope import _compute_project_id
            new_id = _compute_project_id(codebase_root)
            new_facts = tmp_facts_dir / f"facts_project_{new_id}.json"
            new_facts.write_text('{"new": true}')

            # Construct for its side effect (migration check); the assertions
            # below verify the on-disk state, not the store object.
            FactStore(codebase_root=codebase_root, eager_init=True)

        # New file untouched, legacy file left in place
        assert legacy_facts.exists()
        assert legacy_facts.read_text() == '{"legacy": true}'
        assert new_facts.read_text() == '{"new": true}'


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
    def test_ingest_generates_embeddings(self, tmp_facts_dir, tmp_path, tmp_checksum_dir):
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
             patch.object(FactStore, "_embed_text", return_value=fake_emb), \
             patch("neo.memory.constraints.CHECKSUM_DIR", tmp_checksum_dir), \
             patch("neo.memory.constraints.CHECKSUM_FILE", tmp_checksum_dir / "checksums.json"):
            s = FactStore(codebase_root=str(tmp_path))

        constraints = [f for f in s._facts if f.kind == FactKind.CONSTRAINT]
        for c in constraints:
            assert c.embedding is not None, f"Constraint '{c.subject}' has no embedding"

    def test_ingests_uppercase_agents_md(self, tmp_facts_dir, tmp_path, tmp_checksum_dir):
        """AGENTS.md is the Codex-standard project instruction file."""
        agents_md = tmp_path / "AGENTS.md"
        agents_md.write_text("## Codex Rules\nRespect the project instructions.\n")

        fake_emb = np.ones(768, dtype=np.float32)
        with patch("neo.memory.store.FACTS_DIR", tmp_facts_dir), \
             patch("neo.memory.store.detect_org_and_project", return_value=("testorg", "testproj1234")), \
             patch.object(FactStore, "_maybe_migrate"), \
             patch.object(FactStore, "_ingest_seed_facts"), \
             patch.object(FactStore, "_ingest_community_feed"), \
             patch.object(FactStore, "_ingest_claude_memory"), \
             patch.object(FactStore, "_embed_text", return_value=fake_emb), \
             patch("neo.memory.constraints.CHECKSUM_DIR", tmp_checksum_dir), \
             patch("neo.memory.constraints.CHECKSUM_FILE", tmp_checksum_dir / "checksums.json"):
            s = FactStore(codebase_root=str(tmp_path))

        constraints = [f for f in s._facts if f.kind == FactKind.CONSTRAINT]
        assert any(f.metadata.source_file == str(agents_md) for f in constraints)


class TestConcurrentSaveMerge:
    """save() must not let one process clobber facts another just added —
    the observer-vs-request-path clobber that erased linked reasoning facts."""

    def test_project_addition_survives_concurrent_save(self, store, tmp_path):
        A = store  # "process A", project = testproj1234
        # "process B" loads the (currently empty) project file.
        B = FactStore(codebase_root=str(tmp_path))

        # A adds a project fact and persists it to disk.
        fx = A.add_fact(subject="Fact X from process A", body="b",
                        kind=FactKind.REVIEW, scope=FactScope.PROJECT)
        # B, which never saw X, adds its own fact and saves on top.
        fy = B.add_fact(subject="Fact Y from process B", body="b",
                        kind=FactKind.REVIEW, scope=FactScope.PROJECT)

        # A fresh load from disk must contain BOTH.
        C = FactStore(codebase_root=str(tmp_path))
        ids = {f.id for f in C._facts}
        assert fx.id in ids, "process A's project fact was clobbered by B's save"
        assert fy.id in ids

    def test_global_addition_still_survives_concurrent_save(self, store, tmp_path):
        # Regression guard that generalizing the merge didn't break the
        # original global-scope behavior.
        A = store
        B = FactStore(codebase_root=str(tmp_path))
        fx = A.add_fact(subject="Global X", body="b", kind=FactKind.PATTERN,
                        scope=FactScope.GLOBAL)
        fy = B.add_fact(subject="Global Y", body="b", kind=FactKind.PATTERN,
                        scope=FactScope.GLOBAL)
        C = FactStore(codebase_root=str(tmp_path))
        ids = {f.id for f in C._facts}
        assert fx.id in ids and fy.id in ids

    def test_purge_not_resurrected_by_merge(self, store, tmp_path):
        """A physically purged fact must stay gone — the merge-on-save re-read
        must not re-append it from the about-to-be-overwritten file."""
        f = store.add_fact(subject="dead fact", body="b", kind=FactKind.REVIEW,
                           scope=FactScope.PROJECT)
        f.is_valid = False
        f.metadata.last_accessed = time.time() - 40 * 86400  # cold (>30d)
        store.save()
        assert store.purge_dead_facts() == 1
        # Reload from disk: the purged fact must not be present.
        C = FactStore(codebase_root=str(tmp_path))
        assert f.id not in {x.id for x in C._facts}, "purged fact was resurrected by merge"

    def test_concurrent_purge_not_resurrected_via_merge(self, store, tmp_path):
        """When another process writes the file (forcing a merge re-read), a
        fact this process purged must still be excluded via _deleted_ids."""
        A = store
        dead = A.add_fact(subject="cold dead", body="b", kind=FactKind.REVIEW,
                          scope=FactScope.PROJECT)  # disk now has `dead`
        # B loads (sees `dead`) and writes its own fact -> changes file mtime,
        # so A's next save can no longer take the mtime fast-path.
        B = FactStore(codebase_root=str(tmp_path))
        fy = B.add_fact(subject="B fact", body="b", kind=FactKind.REVIEW,
                        scope=FactScope.PROJECT)
        # A purges `dead`; its save must re-read (mtime changed) yet not
        # resurrect `dead`, while still preserving B's addition.
        dead.is_valid = False
        dead.metadata.last_accessed = time.time() - 40 * 86400
        assert A.purge_dead_facts() == 1
        C = FactStore(codebase_root=str(tmp_path))
        ids = {x.id for x in C._facts}
        assert dead.id not in ids, "purged fact resurrected through concurrent-merge re-read"
        assert fy.id in ids, "concurrent process's fact was lost"

    def test_concurrent_success_bump_survives_stale_save(self, store, tmp_path):
        """A success_count bump committed by one process must not be lost when
        another process holds a stale copy and saves later (field reconcile)."""
        A = store
        fact = A.add_fact(subject="shared fact", body="b", kind=FactKind.PATTERN,
                          scope=FactScope.PROJECT)  # disk has fact, success=0
        # B loads the fact (stale copy, success=0).
        B = FactStore(codebase_root=str(tmp_path))
        bfact = next(f for f in B._facts if f.id == fact.id)
        # A bumps success_count and persists.
        fact.metadata.success_count += 1
        fact.metadata.confidence = min(1.0, fact.metadata.confidence + 0.2)
        A.save()
        # B, unaware of the bump, saves its stale copy on top.
        bfact.metadata.last_accessed = time.time()
        B.save()
        # The bump must survive on disk.
        C = FactStore(codebase_root=str(tmp_path))
        merged = next(f for f in C._facts if f.id == fact.id)
        assert merged.metadata.success_count == 1, "concurrent success bump was lost"

    def test_reconcile_never_resurrects_locally_invalidated(self, store, tmp_path):
        """If we invalidated a fact this session, a higher-success disk copy
        must not flip it back to valid."""
        A = store
        fact = A.add_fact(subject="to invalidate", body="b", kind=FactKind.PATTERN,
                          scope=FactScope.PROJECT)
        # Another process bumps success on disk.
        B = FactStore(codebase_root=str(tmp_path))
        bfact = next(f for f in B._facts if f.id == fact.id)
        bfact.metadata.success_count = 5
        B.save()
        # A invalidates its copy, then saves (must re-read & reconcile).
        fact.is_valid = False
        A.save()
        C = FactStore(codebase_root=str(tmp_path))
        merged = next(f for f in C._facts if f.id == fact.id)
        assert merged.is_valid is False, "locally-invalidated fact was resurrected"

    def test_reconcile_preserves_our_independent_edits(self, store, tmp_path):
        """Reconcile must keep OURS as the base, not adopt the disk record
        wholesale — so a concurrent edit doesn't discard our own field changes
        (e.g. a confidence demotion at equal success, or a tag we added)."""
        A = store
        fact = A.add_fact(subject="shared", body="b", kind=FactKind.PATTERN,
                          scope=FactScope.PROJECT)
        fact.metadata.success_count = 2
        fact.metadata.confidence = 0.8
        A.save()
        # B loads the fact (success=2, conf=0.8).
        B = FactStore(codebase_root=str(tmp_path))
        bfact = next(f for f in B._facts if f.id == fact.id)
        # A writes something else, changing the file mtime so B's next save
        # must re-read + reconcile (not take the fast path).
        A.add_fact(subject="other", body="b", kind=FactKind.PATTERN, scope=FactScope.PROJECT)
        # B demotes confidence (success unchanged) and tags it, then saves.
        bfact.metadata.confidence = 0.5
        bfact.tags.append("b-only-tag")
        B.save()
        C = FactStore(codebase_root=str(tmp_path))
        merged = next(f for f in C._facts if f.id == fact.id)
        assert merged.metadata.confidence == 0.5, "our demotion was clobbered by wholesale adopt"
        assert "b-only-tag" in merged.tags, "our tag edit was discarded"
        assert merged.metadata.success_count == 2

    def test_file_lock_does_not_break_single_process_saves(self, store, tmp_path):
        """The cross-process lock must not deadlock or corrupt normal saves;
        a sidecar .lock file is created and adds/reloads still work."""
        f1 = store.add_fact(subject="locked one", body="b", kind=FactKind.PATTERN,
                            scope=FactScope.PROJECT)
        f2 = store.add_fact(subject="locked two", body="b", kind=FactKind.REVIEW,
                            scope=FactScope.PROJECT)
        store.save()
        # Sidecar lock file exists alongside the project fact file.
        assert any(p.name.endswith(".json.lock")
                   for p in (tmp_path / "facts").iterdir())
        C = FactStore(codebase_root=str(tmp_path))
        ids = {f.id for f in C._facts}
        assert f1.id in ids and f2.id in ids

    def test_lock_serialized_concurrent_writes_lose_nothing(self, store, tmp_path):
        """Even interleaved across two store instances, both processes' added
        facts survive (lock serializes the read-modify-write)."""
        A = store
        B = FactStore(codebase_root=str(tmp_path))
        fa = A.add_fact(subject="A1", body="b", kind=FactKind.PATTERN, scope=FactScope.PROJECT)
        fb = B.add_fact(subject="B1", body="b", kind=FactKind.PATTERN, scope=FactScope.PROJECT)
        fa2 = A.add_fact(subject="A2", body="b", kind=FactKind.PATTERN, scope=FactScope.PROJECT)
        C = FactStore(codebase_root=str(tmp_path))
        ids = {f.id for f in C._facts}
        assert {fa.id, fb.id, fa2.id} <= ids

    def test_detect_feedback_dedupes_multi_suggestion_fan_out(self, store, tmp_path):
        """Multiple ACCEPTED outcomes that resolve to the SAME reasoning fact
        (one invocation links all its suggestions to one fact) must reinforce
        that fact ONCE, not once per suggestion."""
        f = store.add_fact(subject="reasoning", body="b", kind=FactKind.DECISION,
                           scope=FactScope.PROJECT, confidence=0.5)
        before_succ = f.metadata.success_count
        # Three accepted outcomes for three suggested files, all linked to f.
        sfi = {"a.py": f.id, "b.py": f.id, "c.py": f.id}
        outcomes = [
            Outcome(outcome_type=OutcomeType.ACCEPTED, file_path=p, diff_summary="+x",
                    suggestion_description="d", suggestion_confidence=0.8)
            for p in ("a.py", "b.py", "c.py")
        ]
        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=(outcomes, sfi)), \
             patch.object(store._outcome_tracker, "compute_arch_delta", return_value=None):
            store.detect_implicit_feedback({"prompt": "p"}, [])
        assert f.metadata.success_count == before_succ + 1, "fan-out double-counted success"

    def test_detect_feedback_modified_fan_out_demotes_once_one_review(self, store, tmp_path):
        """Multiple MODIFIED outcomes resolving to one reasoning fact must demote
        it ONCE and create exactly ONE correction REVIEW — not one per file."""
        f = store.add_fact(subject="reasoning-mod", body="b", kind=FactKind.DECISION,
                           scope=FactScope.PROJECT, confidence=0.8)
        sfi = {p: f.id for p in ("a.py", "b.py", "c.py")}
        outcomes = [
            Outcome(outcome_type=OutcomeType.MODIFIED, file_path=p, diff_summary="+x",
                    suggestion_description="d", suggestion_confidence=0.8)
            for p in ("a.py", "b.py", "c.py")
        ]
        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=(outcomes, sfi)), \
             patch.object(store._outcome_tracker, "compute_arch_delta", return_value=None):
            store.detect_implicit_feedback({"prompt": "p"}, [])
        assert f.metadata.confidence == pytest.approx(0.6), "fan-out demoted more than once"
        modified_reviews = [x for x in store._facts if x.is_valid and "modified" in x.tags]
        assert len(modified_reviews) == 1, "duplicate MODIFIED REVIEW per file"

    def test_detect_feedback_modified_without_link_still_reviews(self, store, tmp_path):
        """A MODIFIED outcome with no linked fact must still record its REVIEW
        (the dedup guard must not suppress the no-link fallback)."""
        outcomes = [
            Outcome(outcome_type=OutcomeType.MODIFIED, file_path="unlinked.py",
                    diff_summary="+x", suggestion_description="d", suggestion_confidence=0.8)
        ]
        with patch.object(store._outcome_tracker, "detect_outcomes",
                          return_value=(outcomes, {})), \
             patch.object(store._outcome_tracker, "compute_arch_delta", return_value=None):
            store.detect_implicit_feedback({"prompt": "p"}, [])
        modified_reviews = [x for x in store._facts if x.is_valid and "modified" in x.tags]
        assert len(modified_reviews) == 1
