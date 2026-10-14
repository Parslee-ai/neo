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

    def test_supersedes_seeds_minted_from_another_install_path(self, ingester):
        """pipx, Homebrew and an editable checkout each reach the seed file
        through a different absolute path; one store must hold one copy."""
        stale = [
            Fact(
                subject=f"N+1 query detection ({where})",
                body="Batch your queries.",
                scope=FactScope.GLOBAL,
                metadata=FactMetadata(source_file=f"/{where}/neo/config/seed/seed_facts.json"),
                tags=["seed", "auto-ingested"],
            )
            for where in ("pipx", "homebrew")
        ]

        new, superseded = ingester.ingest(stale)

        assert len(new) == 2
        assert {f.subject for f in superseded} == {f.subject for f in stale}
        assert all(not f.is_valid for f in stale)

    def test_checksum_is_not_keyed_on_install_path(self, ingester, seed_file, tmp_path):
        """A second install of the same seed content must not re-ingest."""
        ingester.ingest([])
        moved = tmp_path / "other_install" / "seed_facts.json"
        moved.parent.mkdir()
        moved.write_bytes(seed_file.read_bytes())

        with patch("neo.memory.seed.SEED_FILE", moved):
            new, superseded = ingester.ingest([])

        assert new == [] and superseded == []

    def test_does_not_supersede_non_seed_facts(self, ingester, seed_file):
        """Only seed-tagged facts are superseded, whatever their source file."""
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


class TestSupersessionScope:
    def test_project_scoped_seed_tag_is_not_superseded(self, ingester):
        """Supersession is for the global seed corpus only."""
        local = Fact(subject="local", body="b", scope=FactScope.PROJECT,
                     tags=["seed"])

        _, superseded = ingester.ingest([local])

        assert local.is_valid is True
        assert local not in superseded


class TestIngestOrdering:
    def test_unreadable_seed_file_retires_nothing(self, ingester, seed_file):
        current = Fact(subject="in use", body="b", scope=FactScope.GLOBAL,
                       tags=["seed", "auto-ingested"])
        seed_file.write_text("{not json")

        new, superseded = ingester.ingest([current])

        assert (new, superseded) == ([], [])
        assert current.is_valid is True

    def test_checksum_reaches_disk_only_when_the_caller_persists(
        self, ingester, tmp_checksum_dir
    ):
        new, _ = ingester.ingest([])
        assert new
        assert not (tmp_checksum_dir / "checksums.json").exists()

        ingester.save_checksums()
        assert (tmp_checksum_dir / "checksums.json").exists()


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


class TestReadOnlyStoreDoesNotIngest:
    """A read-only store (`--dry-run`, a Lattice lookup) must not record the
    seed file as ingested: its save() is a no-op, so a recorded checksum with
    unsaved facts meant the real store never received that seed generation."""

    def _store(self, tmp_path, *, read_only):
        from contextlib import ExitStack

        from neo.memory.store import FactStore

        checksum_dir = tmp_path / "constraints"
        checksum_dir.mkdir(exist_ok=True)
        with ExitStack() as stack:
            stack.enter_context(patch("neo.memory.store.FACTS_DIR", tmp_path / "facts"))
            stack.enter_context(patch("neo.memory.store.detect_org_and_project",
                                      return_value=("testorg", "testproj")))
            stack.enter_context(patch("neo.memory.store.FASTEMBED_AVAILABLE", False))
            for name in ("_ingest_community_feed", "_ingest_constraints",
                         "_ingest_claude_memory", "_ingest_git_history", "_maybe_migrate"):
                stack.enter_context(patch.object(FactStore, name))
            stack.enter_context(patch("neo.memory.seed.CHECKSUM_DIR", checksum_dir))
            stack.enter_context(patch("neo.memory.seed.CHECKSUM_FILE", checksum_dir / "checksums.json"))
            return FactStore(codebase_root=str(tmp_path), read_only=read_only)

    @staticmethod
    def _seeds(store):
        return [f for f in store._facts if f.is_valid and "seed" in f.tags]

    def test_read_only_store_leaves_the_seed_for_the_real_one(self, tmp_path):
        assert self._seeds(self._store(tmp_path, read_only=True)) == []

        assert self._seeds(self._store(tmp_path, read_only=False)), (
            "the real store must still ingest the seed file"
        )
