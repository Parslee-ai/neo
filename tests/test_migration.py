"""Tests for neo.memory.migration - legacy format migration."""

import json

import numpy as np
import pytest

from neo.memory.migration import _convert_entry, migrate_from_legacy
from neo.memory.models import FactKind, FactScope


@pytest.fixture
def old_global_file(tmp_path):
    """Create a mock old-format global_memory.json."""
    path = tmp_path / "global_memory.json"
    entries = [
        {
            "pattern": "feature: Add user auth",
            "context": "Flask app",
            "reasoning": "Use JWT tokens for stateless auth",
            "suggestion": "Implement middleware for token verification",
            "confidence": 0.7,
            "use_count": 3,
            "created_at": 1000.0,
            "last_used": 2000.0,
            "algorithm_type": "",
            "code_template": "def verify_token(token): ...",
            "common_pitfalls": ["Token expiry not handled", "Missing CORS headers"],
            "when_to_use": "REST APIs needing authentication",
        },
        {
            "pattern": "algorithm: Two pointer sliding window",
            "context": "Array problems",
            "reasoning": "Efficient O(n) approach",
            "suggestion": "Use left/right pointers",
            "confidence": 0.85,
            "use_count": 10,
            "created_at": 500.0,
            "last_used": 1500.0,
            "algorithm_type": "two-pointer",
            "algorithm_category": "array",
        },
        {
            "pattern": "bugfix: Memory leak in worker pool",
            "context": "Python asyncio",
            "reasoning": "Tasks not being awaited properly",
            "suggestion": "Use asyncio.gather with return_exceptions=True",
            "confidence": 0.6,
            "use_count": 1,
            "created_at": 800.0,
            "last_used": 900.0,
        },
    ]
    data = {"entries": entries, "version": "1.0"}
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def old_local_file(tmp_path):
    """Create a mock old-format local file."""
    path = tmp_path / "local_abc123.json"
    entries = [
        {
            "pattern": "refactor: Extract service layer",
            "context": "Django monolith",
            "reasoning": "Separate business logic from views",
            "suggestion": "Create services/ directory",
            "confidence": 0.5,
            "use_count": 2,
            "created_at": 600.0,
            "last_used": 700.0,
        },
    ]
    data = {"entries": entries, "version": "1.0"}
    path.write_text(json.dumps(data))
    return path


class TestConvertEntry:
    def test_feature_maps_to_decision(self):
        entry = {"pattern": "feature: Add button", "reasoning": "UI improvement", "suggestion": "Use React"}
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert fact.kind == FactKind.DECISION

    def test_bugfix_maps_to_failure(self):
        entry = {"pattern": "bugfix: Fix crash", "reasoning": "Null check", "suggestion": "Add guard"}
        fact = _convert_entry(entry, FactScope.PROJECT, "org", "proj")
        assert fact.kind == FactKind.FAILURE

    def test_refactor_maps_to_architecture(self):
        entry = {"pattern": "refactor: Extract module", "reasoning": "Separation", "suggestion": "Move"}
        fact = _convert_entry(entry, FactScope.PROJECT, "org", "proj")
        assert fact.kind == FactKind.ARCHITECTURE

    def test_algorithm_maps_to_pattern(self):
        entry = {"pattern": "algorithm: BFS", "reasoning": "Graph traversal", "suggestion": "Queue"}
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert fact.kind == FactKind.PATTERN

    def test_unknown_maps_to_pattern(self):
        entry = {"pattern": "something: else", "reasoning": "R", "suggestion": "S"}
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert fact.kind == FactKind.PATTERN

    def test_preserves_embedding(self):
        emb = np.random.randn(768).astype(np.float32).tolist()
        entry = {"pattern": "test", "reasoning": "R", "suggestion": "S", "embedding": emb}
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert fact.embedding is not None
        np.testing.assert_array_almost_equal(fact.embedding, np.array(emb, dtype=np.float32))

    def test_body_combines_fields(self):
        entry = {
            "pattern": "test",
            "reasoning": "Use X",
            "suggestion": "Apply Y",
            "code_template": "def foo(): pass",
            "common_pitfalls": ["Don't forget Z"],
            "when_to_use": "When A",
        }
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert "Use X" in fact.body
        assert "Apply Y" in fact.body
        assert "def foo(): pass" in fact.body
        assert "Don't forget Z" in fact.body
        assert "When A" in fact.body

    def test_tags_include_migrated(self):
        entry = {"pattern": "test", "reasoning": "R", "suggestion": "S"}
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert "migrated" in fact.tags

    def test_algorithm_type_in_tags(self):
        entry = {"pattern": "algorithm: DP", "reasoning": "R", "suggestion": "S",
                 "algorithm_type": "dynamic-programming", "algorithm_category": "optimization"}
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert "dynamic-programming" in fact.tags
        assert "optimization" in fact.tags

    def test_confidence_preserved(self):
        entry = {"pattern": "test", "confidence": 0.92}
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert fact.metadata.confidence == 0.92

    def test_confidence_clamped(self):
        entry = {"pattern": "test", "confidence": 1.5}
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert fact.metadata.confidence == 1.0

    def test_timestamps_preserved(self):
        entry = {"pattern": "test", "created_at": 1234.0, "last_used": 5678.0, "use_count": 42}
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert fact.metadata.created_at == 1234.0
        assert fact.metadata.last_accessed == 5678.0
        assert fact.metadata.access_count == 42

    def test_empty_pattern_returns_none(self):
        entry = {"pattern": "", "reasoning": "R"}
        fact = _convert_entry(entry, FactScope.GLOBAL, "org", "proj")
        assert fact is None


class TestMigrateFromLegacy:
    def test_migrates_global_file(self, old_global_file):
        facts = migrate_from_legacy(old_global_file, org_id="org", project_id="proj")
        assert len(facts) == 3
        assert all(f.scope == FactScope.GLOBAL for f in facts)

    def test_migrates_local_file(self, old_global_file, old_local_file):
        facts = migrate_from_legacy(old_global_file, old_local_file, org_id="org", project_id="proj")
        assert len(facts) == 4  # 3 global + 1 local
        local_facts = [f for f in facts if f.scope == FactScope.PROJECT]
        assert len(local_facts) == 1

    def test_does_not_modify_old_files(self, old_global_file, old_local_file):
        original_global = old_global_file.read_text()
        original_local = old_local_file.read_text()

        migrate_from_legacy(old_global_file, old_local_file, org_id="org", project_id="proj")

        assert old_global_file.read_text() == original_global
        assert old_local_file.read_text() == original_local

    def test_missing_files_returns_empty(self, tmp_path):
        nonexistent = tmp_path / "does_not_exist.json"
        facts = migrate_from_legacy(nonexistent, org_id="org", project_id="proj")
        assert facts == []

    def test_corrupt_file_returns_empty(self, tmp_path):
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("not valid json{{{")
        facts = migrate_from_legacy(corrupt, org_id="org", project_id="proj")
        assert facts == []
