"""Tests for neo.memory.seed - bundled seed fact ingestion."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from neo.memory.models import Fact, FactKind, FactMetadata, FactScope
from neo.memory.seed import SEED_CONFIDENCE, SeedIngester


@pytest.fixture
def tmp_checksum_dir(tmp_path):
    checksum_dir = tmp_path / "constraints"
    checksum_dir.mkdir()
    return checksum_dir


@pytest.fixture
def seed_file(tmp_path):
    """Create a temporary seed facts file."""
    seed = tmp_path / "seed_facts.json"
    seed.write_text(json.dumps({
        "version": "0.1.0",
        "facts": [
            {
                "subject": "N+1 query detection",
                "body": "Batch your queries.",
                "kind": "pattern",
                "tags": ["performance", "database"],
            },
            {
                "subject": "SQL injection prevention",
                "body": "Use parameterized queries.",
                "kind": "pattern",
                "tags": ["security"],
            },
        ],
    }))
    return seed


@pytest.fixture
def ingester(seed_file, tmp_checksum_dir):
    with patch("neo.memory.seed.SEED_FILE", seed_file), \
         patch("neo.memory.seed.CHECKSUM_DIR", tmp_checksum_dir), \
         patch("neo.memory.seed.CHECKSUM_FILE", tmp_checksum_dir / "checksums.json"):
        yield SeedIngester(org_id="testorg", project_id="testproj")


class TestSeedIngestion:
    def test_loads_seed_facts(self, ingester):
        new_facts, _ = ingester.ingest([])
        assert len(new_facts) == 2
        assert new_facts[0].subject == "N+1 query detection"
        assert new_facts[1].subject == "SQL injection prevention"

    def test_facts_are_global_scope(self, ingester):
        new_facts, _ = ingester.ingest([])
        for f in new_facts:
            assert f.scope == FactScope.GLOBAL

    def test_confidence_is_0_6(self, ingester):
        new_facts, _ = ingester.ingest([])
        for f in new_facts:
            assert f.metadata.confidence == SEED_CONFIDENCE

    def test_tagged_as_seed(self, ingester):
        new_facts, _ = ingester.ingest([])
        for f in new_facts:
            assert "seed" in f.tags
            assert "auto-ingested" in f.tags

    def test_preserves_entry_tags(self, ingester):
        new_facts, _ = ingester.ingest([])
        assert "performance" in new_facts[0].tags
        assert "database" in new_facts[0].tags
        assert "security" in new_facts[1].tags

    def test_kind_mapping(self, ingester):
        new_facts, _ = ingester.ingest([])
        assert new_facts[0].kind == FactKind.PATTERN


class TestChecksumSkip:
    def test_skips_unchanged_seed(self, ingester):
        new1, _ = ingester.ingest([])
        assert len(new1) == 2

        new2, _ = ingester.ingest([])
        assert len(new2) == 0

    def test_reingests_on_change(self, ingester, seed_file):
        new1, _ = ingester.ingest([])
        assert len(new1) == 2

        # Simulate package update with new seed
        seed_file.write_text(json.dumps({
            "version": "0.2.0",
            "facts": [
                {"subject": "New pattern", "body": "New body.", "kind": "pattern", "tags": []},
            ],
        }))

        new2, superseded = ingester.ingest(new1)
        assert len(new2) == 1
        assert len(superseded) == 2
        assert new2[0].subject == "New pattern"


class TestSupersession:
    def test_supersedes_old_seed_facts(self, ingester, seed_file):
        new1, _ = ingester.ingest([])

        seed_file.write_text(json.dumps({
            "version": "0.2.0",
            "facts": [
                {"subject": "Updated", "body": "New.", "kind": "pattern", "tags": []},
            ],
        }))

        _, superseded = ingester.ingest(new1)
        for f in superseded:
            assert f.is_valid is False

    def test_does_not_supersede_non_seed_facts(self, ingester, seed_file):
        """Only seed-tagged facts from the same file should be superseded."""
        other = Fact(
            subject="user pattern",
            body="unrelated",
            metadata=FactMetadata(source_file=str(seed_file)),
            tags=["claude-memory"],
        )

        seed_file.write_text(json.dumps({
            "version": "0.2.0",
            "facts": [
                {"subject": "X", "body": "Y.", "kind": "pattern", "tags": []},
            ],
        }))

        # First ingest to set checksum, then change file
        ingester.ingest([])
        seed_file.write_text(json.dumps({
            "version": "0.3.0",
            "facts": [
                {"subject": "Z", "body": "W.", "kind": "pattern", "tags": []},
            ],
        }))

        _, superseded = ingester.ingest([other])
        assert other.is_valid is True
        assert other not in superseded


class TestEdgeCases:
    def test_missing_seed_file(self, tmp_checksum_dir):
        with patch("neo.memory.seed.SEED_FILE", Path("/nonexistent/seed.json")), \
             patch("neo.memory.seed.CHECKSUM_DIR", tmp_checksum_dir), \
             patch("neo.memory.seed.CHECKSUM_FILE", tmp_checksum_dir / "checksums.json"):
            ing = SeedIngester()
            new, sup = ing.ingest([])
        assert new == []
        assert sup == []

    def test_empty_facts_array(self, ingester, seed_file):
        seed_file.write_text(json.dumps({"version": "0.1.0", "facts": []}))
        # Reset checksum so it re-reads
        ingester._checksums.clear()
        new, _ = ingester.ingest([])
        assert len(new) == 0

    def test_skips_entries_without_subject(self, ingester, seed_file):
        seed_file.write_text(json.dumps({
            "version": "0.1.0",
            "facts": [
                {"body": "no subject", "kind": "pattern", "tags": []},
                {"subject": "has subject", "body": "ok", "kind": "pattern", "tags": []},
            ],
        }))
        ingester._checksums.clear()
        new, _ = ingester.ingest([])
        assert len(new) == 1
        assert new[0].subject == "has subject"

    def test_unknown_kind_defaults_to_pattern(self, ingester, seed_file):
        seed_file.write_text(json.dumps({
            "version": "0.1.0",
            "facts": [
                {"subject": "test", "body": "body", "kind": "banana", "tags": []},
            ],
        }))
        ingester._checksums.clear()
        new, _ = ingester.ingest([])
        assert new[0].kind == FactKind.PATTERN
