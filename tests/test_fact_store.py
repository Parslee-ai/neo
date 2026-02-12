"""Tests for neo.memory.store - FactStore integration tests."""

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
