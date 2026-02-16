"""Tests for neo.memory.models - data model serialization and enums."""

import numpy as np

from neo.memory.models import (
    ContextResult,
    Fact,
    FactKind,
    FactMetadata,
    FactScope,
)


class TestFactKind:
    def test_all_values(self):
        assert FactKind.CONSTRAINT.value == "constraint"
        assert FactKind.ARCHITECTURE.value == "architecture"
        assert FactKind.PATTERN.value == "pattern"
        assert FactKind.REVIEW.value == "review"
        assert FactKind.DECISION.value == "decision"
        assert FactKind.KNOWN_UNKNOWN.value == "known_unknown"
        assert FactKind.FAILURE.value == "failure"

    def test_roundtrip(self):
        for kind in FactKind:
            assert FactKind(kind.value) == kind


class TestFactScope:
    def test_all_values(self):
        assert FactScope.GLOBAL.value == "global"
        assert FactScope.ORG.value == "org"
        assert FactScope.PROJECT.value == "project"
        assert FactScope.SESSION.value == "session"


class TestFactMetadata:
    def test_to_dict_from_dict_roundtrip(self):
        meta = FactMetadata(
            created_at=1000.0,
            last_accessed=2000.0,
            access_count=5,
            source_file="/path/to/file.py",
            source_prompt="test prompt",
            confidence=0.85,
        )
        d = meta.to_dict()
        restored = FactMetadata.from_dict(d)
        assert restored.created_at == 1000.0
        assert restored.last_accessed == 2000.0
        assert restored.access_count == 5
        assert restored.source_file == "/path/to/file.py"
        assert restored.source_prompt == "test prompt"
        assert restored.confidence == 0.85

    def test_from_dict_defaults(self):
        meta = FactMetadata.from_dict({})
        assert meta.access_count == 0
        assert meta.source_file == ""
        assert meta.confidence == 0.5


class TestFact:
    def test_to_dict_from_dict_roundtrip(self):
        embedding = np.random.randn(768).astype(np.float32)
        fact = Fact(
            id="abc123def456gh78",
            subject="Test pattern",
            body="This is a test body.",
            kind=FactKind.ARCHITECTURE,
            scope=FactScope.ORG,
            org_id="myorg",
            project_id="proj1234567890ab",
            is_valid=True,
            superseded_by=None,
            supersedes="old_fact_id",
            depends_on=["dep1", "dep2"],
            needs_review=False,
            metadata=FactMetadata(confidence=0.9),
            embedding=embedding,
            tags=["python", "architecture"],
        )
        d = fact.to_dict()
        restored = Fact.from_dict(d)

        assert restored.id == "abc123def456gh78"
        assert restored.subject == "Test pattern"
        assert restored.body == "This is a test body."
        assert restored.kind == FactKind.ARCHITECTURE
        assert restored.scope == FactScope.ORG
        assert restored.org_id == "myorg"
        assert restored.project_id == "proj1234567890ab"
        assert restored.is_valid is True
        assert restored.superseded_by is None
        assert restored.supersedes == "old_fact_id"
        assert restored.depends_on == ["dep1", "dep2"]
        assert restored.needs_review is False
        assert restored.metadata.confidence == 0.9
        assert restored.tags == ["python", "architecture"]
        np.testing.assert_array_almost_equal(restored.embedding, embedding, decimal=5)

    def test_to_dict_no_embedding(self):
        fact = Fact(subject="No embedding")
        d = fact.to_dict()
        assert "embedding" not in d

    def test_from_dict_no_embedding(self):
        d = {"subject": "test", "kind": "pattern", "scope": "global"}
        fact = Fact.from_dict(d)
        assert fact.embedding is None
        assert fact.kind == FactKind.PATTERN
        assert fact.scope == FactScope.GLOBAL

    def test_id_auto_generated(self):
        f1 = Fact()
        f2 = Fact()
        assert f1.id != f2.id
        assert len(f1.id) == 16

    def test_defaults(self):
        fact = Fact()
        assert fact.is_valid is True
        assert fact.needs_review is False
        assert fact.superseded_by is None
        assert fact.depends_on == []
        assert fact.tags == []


class TestContextResult:
    def test_empty(self):
        ctx = ContextResult()
        assert ctx.constraints == []
        assert ctx.valid_facts == []
        assert ctx.invalidated_facts == []
        assert ctx.working_set == []
        assert ctx.environment == {}
        assert ctx.known_unknowns == []

    def test_with_data(self):
        c = Fact(kind=FactKind.CONSTRAINT)
        v = Fact(kind=FactKind.PATTERN)
        ku = Fact(kind=FactKind.KNOWN_UNKNOWN)
        ctx = ContextResult(
            constraints=[c],
            valid_facts=[v],
            known_unknowns=[ku],
            environment={"branch": "main"},
        )
        assert len(ctx.constraints) == 1
        assert len(ctx.valid_facts) == 1
        assert len(ctx.known_unknowns) == 1
        assert ctx.environment["branch"] == "main"
