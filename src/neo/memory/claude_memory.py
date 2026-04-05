"""
Claude Code auto-memory ingestion.

Reads curated knowledge from Claude Code's memory system
(~/.claude/projects/{project-id}/memory/*.md) and converts
to neo facts. These files have YAML frontmatter with name,
description, and type fields followed by a markdown body.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

import yaml

from neo.memory.models import Fact, FactKind, FactMetadata, FactScope

logger = logging.getLogger(__name__)

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
CHECKSUM_DIR = Path.home() / ".neo" / "constraints"
CHECKSUM_FILE = CHECKSUM_DIR / "checksums.json"

# Map Claude Code memory types to neo FactKinds
TYPE_MAP: dict[str, FactKind] = {
    "project": FactKind.DECISION,
    "feedback": FactKind.PATTERN,
    "reference": FactKind.ARCHITECTURE,
    "user": FactKind.PATTERN,
}

CLAUDE_MEMORY_CONFIDENCE = 0.7


class ClaudeMemoryIngester:
    """Ingests Claude Code auto-memory files as neo facts.

    Claude Code stores curated knowledge in markdown files with YAML
    frontmatter under ~/.claude/projects/{project-id}/memory/. This
    ingester reads those files and creates scoped facts.
    """

    def __init__(self, codebase_root: Optional[str] = None,
                 org_id: str = "", project_id: str = ""):
        self.codebase_root = codebase_root or ""
        self.org_id = org_id
        self.project_id = project_id
        self._checksums = self._load_checksums()

    def ingest(self, existing_facts: list[Fact]) -> tuple[list[Fact], list[Fact]]:
        """Scan Claude Code memory files and return new/updated facts.

        Returns:
            Tuple of (new_facts, superseded_facts).
        """
        memory_dir = self._resolve_memory_dir()
        if memory_dir is None or not memory_dir.is_dir():
            return [], []

        new_facts: list[Fact] = []
        superseded_facts: list[Fact] = []

        for md_file in sorted(memory_dir.glob("*.md")):
            if md_file.name == "MEMORY.md":
                continue  # Index file, not a knowledge file

            current_checksum = self._file_checksum(md_file)
            stored_checksum = self._checksums.get(str(md_file), "")

            if current_checksum == stored_checksum:
                continue

            logger.info(f"Ingesting Claude memory: {md_file.name}")

            # Supersede old facts from this file
            for fact in existing_facts:
                if (fact.metadata.source_file == str(md_file)
                        and fact.is_valid
                        and "claude-memory" in fact.tags):
                    fact.is_valid = False
                    superseded_facts.append(fact)

            # Parse frontmatter + body
            subject, body, kind = self._parse_memory_file(md_file)
            if not body.strip():
                self._checksums[str(md_file)] = current_checksum
                continue

            fact = Fact(
                subject=subject,
                body=body.strip(),
                kind=kind,
                scope=FactScope.PROJECT,
                org_id=self.org_id,
                project_id=self.project_id,
                metadata=FactMetadata(
                    source_file=str(md_file),
                    confidence=CLAUDE_MEMORY_CONFIDENCE,
                ),
                tags=["claude-memory", "auto-ingested"],
            )
            new_facts.append(fact)
            self._checksums[str(md_file)] = current_checksum

        self._save_checksums()
        return new_facts, superseded_facts

    def _resolve_memory_dir(self) -> Optional[Path]:
        """Map codebase_root to Claude Code's memory directory."""
        if not self.codebase_root:
            return None

        # Claude Code derives project ID by replacing / with -
        claude_project_id = self.codebase_root.replace("/", "-")
        memory_dir = CLAUDE_PROJECTS_DIR / claude_project_id / "memory"
        return memory_dir

    def _parse_memory_file(self, path: Path) -> tuple[str, str, FactKind]:
        """Parse a Claude Code memory file with optional YAML frontmatter.

        Returns:
            Tuple of (subject, body, kind).
        """
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, IOError) as e:
            logger.warning(f"Failed to read {path}: {e}")
            return path.stem, "", FactKind.PATTERN

        # Parse YAML frontmatter between --- delimiters
        frontmatter_match = re.match(
            r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL
        )

        if frontmatter_match:
            try:
                meta = yaml.safe_load(frontmatter_match.group(1)) or {}
            except yaml.YAMLError:
                meta = {}
            body = frontmatter_match.group(2)
        else:
            meta = {}
            body = content

        subject = meta.get("name", path.stem)
        memory_type = meta.get("type", "project")
        kind = TYPE_MAP.get(memory_type, FactKind.PATTERN)

        return subject, body, kind

    def _file_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        h = hashlib.sha256()
        try:
            h.update(path.read_bytes())
        except (OSError, IOError):
            return ""
        return h.hexdigest()

    def _load_checksums(self) -> dict[str, str]:
        """Load stored checksums."""
        if CHECKSUM_FILE.exists():
            try:
                return json.loads(CHECKSUM_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_checksums(self) -> None:
        """Save checksums to disk."""
        CHECKSUM_DIR.mkdir(parents=True, exist_ok=True)
        CHECKSUM_FILE.write_text(json.dumps(self._checksums, indent=2))
