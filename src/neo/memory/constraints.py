"""
Constraint ingestion from project documentation.

Scans CLAUDE.md, agents.md, and similar files, splitting them into
individual constraint facts. Tracks file checksums to avoid re-ingestion
when files haven't changed.
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

from neo.memory.models import Fact, FactKind, FactMetadata, FactScope

logger = logging.getLogger(__name__)

# Files to scan for constraints, in priority order
CONSTRAINT_FILES = [
    ("~/.claude/CLAUDE.md", FactScope.GLOBAL),
    ("{project}/CLAUDE.md", FactScope.PROJECT),
    ("{project}/agents.md", FactScope.PROJECT),
    ("{project}/.cursor/rules", FactScope.PROJECT),
]

CHECKSUM_DIR = Path.home() / ".neo" / "constraints"
CHECKSUM_FILE = CHECKSUM_DIR / "checksums.json"


class ConstraintIngester:
    """Ingests project documentation as constraint facts.

    Scans markdown files, splits by ## headings, and creates one
    Fact per section. Uses checksums to detect file changes and
    supersede stale constraints.
    """

    def __init__(self, codebase_root: Optional[str] = None,
                 org_id: str = "", project_id: str = ""):
        self.codebase_root = codebase_root or ""
        self.org_id = org_id
        self.project_id = project_id
        self._checksums = self._load_checksums()

    def ingest(self, existing_facts: list[Fact]) -> tuple[list[Fact], list[Fact]]:
        """Scan constraint files and return new/updated facts.

        Args:
            existing_facts: Current facts list (for supersession).

        Returns:
            Tuple of (new_facts, superseded_facts).
        """
        new_facts: list[Fact] = []
        superseded_facts: list[Fact] = []

        for file_template, scope in CONSTRAINT_FILES:
            file_path = self._resolve_path(file_template)
            if not file_path or not file_path.exists():
                continue

            current_checksum = self._file_checksum(file_path)
            stored_checksum = self._checksums.get(str(file_path), "")

            if current_checksum == stored_checksum:
                logger.debug(f"Constraint file unchanged: {file_path}")
                continue

            logger.info(f"Ingesting constraints from: {file_path}")

            # Supersede old constraints from this file
            for fact in existing_facts:
                if (fact.kind == FactKind.CONSTRAINT
                        and fact.metadata.source_file == str(file_path)
                        and fact.is_valid):
                    fact.is_valid = False
                    superseded_facts.append(fact)

            # Parse new constraints
            sections = self._split_markdown(file_path)
            for heading, body in sections:
                if not body.strip():
                    continue

                fact = Fact(
                    subject=heading,
                    body=body.strip(),
                    kind=FactKind.CONSTRAINT,
                    scope=scope,
                    org_id=self.org_id,
                    project_id=self.project_id,
                    metadata=FactMetadata(
                        source_file=str(file_path),
                        confidence=1.0,
                    ),
                    tags=["constraint", "auto-ingested"],
                )
                new_facts.append(fact)

            # Update checksum
            self._checksums[str(file_path)] = current_checksum

        self._save_checksums()
        return new_facts, superseded_facts

    def _resolve_path(self, template: str) -> Optional[Path]:
        """Resolve a file path template."""
        resolved = template.replace("{project}", self.codebase_root)
        path = Path(resolved).expanduser()
        return path

    def _file_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        h = hashlib.sha256()
        try:
            h.update(path.read_bytes())
        except (OSError, IOError):
            return ""
        return h.hexdigest()

    def _split_markdown(self, path: Path) -> list[tuple[str, str]]:
        """Split a markdown file into (heading, body) sections by ## headings."""
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, IOError) as e:
            logger.warning(f"Failed to read {path}: {e}")
            return []

        sections: list[tuple[str, str]] = []
        current_heading = path.name  # Default heading is file name
        current_body_lines: list[str] = []

        for line in content.splitlines():
            heading_match = re.match(r"^#{1,3}\s+(.+)$", line)
            if heading_match:
                # Save previous section
                if current_body_lines:
                    sections.append((current_heading, "\n".join(current_body_lines)))
                current_heading = heading_match.group(1).strip()
                current_body_lines = []
            else:
                current_body_lines.append(line)

        # Save last section
        if current_body_lines:
            sections.append((current_heading, "\n".join(current_body_lines)))

        return sections

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
