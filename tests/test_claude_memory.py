"""Tests for neo.memory.claude_memory - Claude Code auto-memory ingestion."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from neo.memory.claude_memory import (
    CHECKSUM_FILE,
    CLAUDE_MEMORY_CONFIDENCE,
    CLAUDE_PROJECTS_DIR,
    ClaudeMemoryIngester,
)
from neo.memory.models import Fact, FactKind, FactMetadata, FactScope


@pytest.fixture
def tmp_claude_dir(tmp_path):
    """Create a fake Claude Code projects directory."""
    claude_dir = tmp_path / ".claude" / "projects"
    claude_dir.mkdir(parents=True)
    return claude_dir


@pytest.fixture
def tmp_checksum_dir(tmp_path):
    """Override checksum storage to temp directory."""
    checksum_dir = tmp_path / "constraints"
    checksum_dir.mkdir()
    return checksum_dir


@pytest.fixture
def ingester(tmp_claude_dir, tmp_checksum_dir):
    """Create an ingester with temp directories."""
    with patch("neo.memory.claude_memory.CLAUDE_PROJECTS_DIR", tmp_claude_dir), \
         patch("neo.memory.claude_memory.CHECKSUM_DIR", tmp_checksum_dir), \
         patch("neo.memory.claude_memory.CHECKSUM_FILE", tmp_checksum_dir / "checksums.json"):
        yield ClaudeMemoryIngester(
            codebase_root="/Users/testuser/git/myproject",
            org_id="testorg",
            project_id="testproj1234",
        )


def _make_memory_dir(claude_dir: Path, codebase_root: str) -> Path:
    """Create a Claude Code memory directory for a given codebase root."""
    project_id = codebase_root.replace("/", "-")
    memory_dir = claude_dir / project_id / "memory"
    memory_dir.mkdir(parents=True)
    return memory_dir


class TestPathMapping:
    def test_maps_codebase_root_to_claude_dir(self, ingester, tmp_claude_dir):
        """Codebase root should map to Claude's dash-separated project dir."""
        memory_dir = ingester._resolve_memory_dir()
        expected = tmp_claude_dir / "-Users-testuser-git-myproject" / "memory"
        assert memory_dir == expected

    def test_empty_codebase_root_returns_none(self, tmp_checksum_dir):
        """No codebase root means no memory dir."""
        with patch("neo.memory.claude_memory.CHECKSUM_DIR", tmp_checksum_dir), \
             patch("neo.memory.claude_memory.CHECKSUM_FILE", tmp_checksum_dir / "checksums.json"):
            ing = ClaudeMemoryIngester(codebase_root="")
        assert ing._resolve_memory_dir() is None


class TestFrontmatterParsing:
    def test_parses_yaml_frontmatter(self, ingester, tmp_claude_dir):
        """Files with YAML frontmatter should extract name, type, and body."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        (memory_dir / "test_fact.md").write_text(
            "---\n"
            "name: test-memory\n"
            "description: A test memory\n"
            "type: project\n"
            "---\n\n"
            "This is the body of the memory.\n"
        )

        new_facts, _ = ingester.ingest([])
        assert len(new_facts) == 1
        assert new_facts[0].subject == "test-memory"
        assert "body of the memory" in new_facts[0].body
        assert new_facts[0].kind == FactKind.DECISION  # project -> DECISION

    def test_no_frontmatter_uses_filename(self, ingester, tmp_claude_dir):
        """Files without frontmatter should use stem as subject."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        (memory_dir / "plain_notes.md").write_text("Just plain markdown content.\n")

        new_facts, _ = ingester.ingest([])
        assert len(new_facts) == 1
        assert new_facts[0].subject == "plain_notes"
        assert new_facts[0].kind == FactKind.DECISION  # default type="project" → DECISION

    def test_malformed_yaml_treated_as_no_frontmatter(self, ingester, tmp_claude_dir):
        """Malformed YAML should not crash, fall back to plain text."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        (memory_dir / "bad_yaml.md").write_text(
            "---\n"
            "name: [invalid yaml\n"
            "---\n\n"
            "Body content.\n"
        )

        new_facts, _ = ingester.ingest([])
        assert len(new_facts) == 1
        assert new_facts[0].subject == "bad_yaml"  # Falls back to stem


class TestTypeMapping:
    @pytest.mark.parametrize("memory_type,expected_kind", [
        ("project", FactKind.DECISION),
        ("feedback", FactKind.PATTERN),
        ("reference", FactKind.ARCHITECTURE),
        ("user", FactKind.PATTERN),
    ])
    def test_known_types(self, ingester, tmp_claude_dir, memory_type, expected_kind):
        """Each Claude memory type should map to the correct FactKind."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        (memory_dir / f"test_{memory_type}.md").write_text(
            f"---\nname: test\ntype: {memory_type}\n---\n\nBody.\n"
        )

        new_facts, _ = ingester.ingest([])
        assert len(new_facts) == 1
        assert new_facts[0].kind == expected_kind

    def test_unknown_type_defaults_to_pattern(self, ingester, tmp_claude_dir):
        """Unknown types should default to PATTERN."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        (memory_dir / "test.md").write_text(
            "---\nname: test\ntype: banana\n---\n\nBody.\n"
        )

        new_facts, _ = ingester.ingest([])
        assert new_facts[0].kind == FactKind.PATTERN


class TestConfidenceAndTags:
    def test_confidence_is_0_7(self, ingester, tmp_claude_dir):
        """Claude memory facts should have confidence 0.7."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        (memory_dir / "test.md").write_text("---\nname: test\ntype: project\n---\n\nBody.\n")

        new_facts, _ = ingester.ingest([])
        assert new_facts[0].metadata.confidence == CLAUDE_MEMORY_CONFIDENCE

    def test_tags_include_claude_memory(self, ingester, tmp_claude_dir):
        """Facts should be tagged for identification."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        (memory_dir / "test.md").write_text("---\nname: test\ntype: project\n---\n\nBody.\n")

        new_facts, _ = ingester.ingest([])
        assert "claude-memory" in new_facts[0].tags
        assert "auto-ingested" in new_facts[0].tags


class TestSkipLogic:
    def test_skips_memory_md_index(self, ingester, tmp_claude_dir):
        """MEMORY.md is an index file and should be skipped."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        (memory_dir / "MEMORY.md").write_text("# Index\n- [Link](other.md)\n")
        (memory_dir / "real_fact.md").write_text("---\nname: real\ntype: project\n---\n\nBody.\n")

        new_facts, _ = ingester.ingest([])
        assert len(new_facts) == 1
        assert new_facts[0].subject == "real"

    def test_skips_empty_body(self, ingester, tmp_claude_dir):
        """Files with only frontmatter and no body should be skipped."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        (memory_dir / "empty.md").write_text("---\nname: empty\ntype: project\n---\n\n")

        new_facts, _ = ingester.ingest([])
        assert len(new_facts) == 0

    def test_missing_directory_returns_empty(self, ingester):
        """Missing Claude projects directory should return empty results."""
        new_facts, superseded = ingester.ingest([])
        assert new_facts == []
        assert superseded == []


class TestChecksumSkip:
    def test_skips_unchanged_files(self, ingester, tmp_claude_dir):
        """Second ingest of unchanged file should produce no new facts."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        (memory_dir / "test.md").write_text("---\nname: test\ntype: project\n---\n\nBody.\n")

        new1, _ = ingester.ingest([])
        assert len(new1) == 1

        new2, _ = ingester.ingest([])
        assert len(new2) == 0

    def test_reingests_changed_files(self, ingester, tmp_claude_dir):
        """Changed file should produce new facts and supersede old ones."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        md_file = memory_dir / "test.md"
        md_file.write_text("---\nname: test\ntype: project\n---\n\nOriginal body.\n")

        new1, _ = ingester.ingest([])
        assert len(new1) == 1

        # Simulate file change
        md_file.write_text("---\nname: test\ntype: project\n---\n\nUpdated body.\n")

        # Pass the first fact as existing so it gets superseded
        new2, superseded = ingester.ingest(new1)
        assert len(new2) == 1
        assert len(superseded) == 1
        assert superseded[0].is_valid is False
        assert "Updated body" in new2[0].body


class TestSupersession:
    def test_supersedes_old_claude_memory_facts(self, ingester, tmp_claude_dir):
        """Re-ingestion should mark old facts from same file as invalid."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        md_file = memory_dir / "test.md"
        md_file.write_text("---\nname: test\ntype: project\n---\n\nV1.\n")

        new1, _ = ingester.ingest([])

        md_file.write_text("---\nname: test\ntype: project\n---\n\nV2.\n")
        new2, superseded = ingester.ingest(new1)

        assert new1[0].is_valid is False
        assert len(new2) == 1
        assert "V2" in new2[0].body

    def test_does_not_supersede_non_claude_facts(self, ingester, tmp_claude_dir):
        """Only facts tagged 'claude-memory' from the same file should be superseded."""
        memory_dir = _make_memory_dir(tmp_claude_dir, "/Users/testuser/git/myproject")
        md_file = memory_dir / "test.md"
        md_file.write_text("---\nname: test\ntype: project\n---\n\nBody.\n")

        # Create a non-claude fact with the same source file
        other_fact = Fact(
            subject="other",
            body="unrelated",
            metadata=FactMetadata(source_file=str(md_file)),
            tags=["constraint"],
        )

        new_facts, superseded = ingester.ingest([other_fact])
        assert len(superseded) == 0  # other_fact not superseded
        assert other_fact.is_valid is True
