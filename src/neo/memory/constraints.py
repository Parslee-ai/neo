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

from neo.memory.io_utils import atomic_write_json
from neo.memory.models import Fact, FactKind, FactMetadata, FactScope

logger = logging.getLogger(__name__)

# Files to scan for constraints, in priority order.
#
# Several entries deliberately name the same document under different
# spellings, because different tools write different ones. On a
# case-insensitive filesystem — macOS by default — `AGENTS.md` and `agents.md`
# are then not two candidates but ONE FILE listed twice, and a symlinked
# `CLAUDE.md -> AGENTS.md` is a third spelling of it. `_group_by_file_identity`
# collapses those before ingestion; the order here decides which spelling wins
# and becomes the `source_file` recorded on the facts.
CONSTRAINT_FILES = [
    ("~/.claude/CLAUDE.md", FactScope.GLOBAL),
    ("{project}/CLAUDE.md", FactScope.PROJECT),
    ("{project}/AGENTS.md", FactScope.PROJECT),
    ("{project}/agents.md", FactScope.PROJECT),
    ("{project}/.github/AGENTS.md", FactScope.PROJECT),
    ("{project}/.cursor/rules", FactScope.PROJECT),
    ("{project}/.cursorrules", FactScope.PROJECT),
    ("{project}/.windsurfrules", FactScope.PROJECT),
    ("{project}/.clinerules", FactScope.PROJECT),
    ("{project}/.github/copilot-instructions.md", FactScope.PROJECT),
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

        for file_path, aliases, scope in self._group_by_file_identity():
            # Retire facts recorded under a non-canonical spelling of this same
            # file, and do it BEFORE the unchanged-checksum short-circuit.
            #
            # This is the migration path, and the ordering is the whole of it.
            # Stores written before aliases were collapsed hold a full second
            # (and third) copy of every section under the other spellings —
            # valid, confidence 1.0, CONSTRAINT and therefore exempt from
            # recall decay, so nothing ages them out. Their file has not
            # changed, so the checksum matches, so a cleanup placed below the
            # short-circuit would never run and the duplicates would outlive
            # the fix indefinitely.
            self._retire_alias_facts(
                file_path, aliases, existing_facts, superseded_facts
            )

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

    def _group_by_file_identity(
        self,
    ) -> list[tuple[Path, list[Path], FactScope]]:
        """Collapse `CONSTRAINT_FILES` entries that name the same file.

        Returns one `(canonical_path, alias_paths, scope)` per distinct file,
        in `CONSTRAINT_FILES` order. `alias_paths` are the other spellings that
        reached it, and is empty for the ordinary case.

        The table lists `AGENTS.md` and `agents.md` separately because
        different tools write different ones. On a case-insensitive filesystem
        — macOS by default, which is where this was found — those are the same
        file, so the ingest loop visited it twice. Both the checksum cache and
        the supersession pass key on the literal path string, and
        `"…/AGENTS.md" != "…/agents.md"`, so neither pass saw the other's work:
        the second visit found no stored checksum, superseded nothing, and
        minted a complete second copy of every section. Measured on one machine
        before this fix: 51 duplicated subjects in a single project, and the
        same pattern in every project that had an `AGENTS.md` — with the
        control holding, since projects without one showed zero.

        Identity is `(st_dev, st_ino)`, not a normalized string. `realpath`
        resolves symlinks but preserves the case it was given, so on macOS it
        does not equate the two spellings at all, and `os.path.normcase` is a
        no-op outside Windows. Only the filesystem knows, so ask it — which
        catches the symlinked `CLAUDE.md -> AGENTS.md` layout for free.

        Falls back to the resolved path string when `stat` fails, which keeps a
        race between `exists()` and `stat()` from dropping a file entirely: the
        cost is a missed alias collapse, not a missed rule file.
        """
        groups: list[tuple[Path, list[Path], FactScope]] = []
        by_identity: dict[object, int] = {}

        for file_template, scope in CONSTRAINT_FILES:
            file_path = self._resolve_path(file_template)
            if not file_path or not file_path.exists():
                continue

            try:
                stat = file_path.stat()
                identity: object = (stat.st_dev, stat.st_ino)
            except OSError:
                identity = str(file_path.resolve())

            if identity in by_identity:
                # A later spelling of a file already claimed. The first entry
                # wins, so CONSTRAINT_FILES order is what decides the canonical
                # path — deliberately, since it is the documented priority
                # order and is stable across runs.
                canonical, aliases, _ = groups[by_identity[identity]]
                if file_path != canonical:
                    aliases.append(file_path)
                continue

            by_identity[identity] = len(groups)
            groups.append((file_path, [], scope))

        return groups

    def _retire_alias_facts(
        self,
        canonical: Path,
        aliases: list[Path],
        existing_facts: list[Fact],
        superseded_facts: list[Fact],
    ) -> None:
        """Invalidate constraints recorded under a non-canonical spelling.

        Also drops the aliases' checksum entries, so a stale cache row cannot
        keep a retired spelling alive in `checksums.json` forever.
        """
        alias_keys = {str(path) for path in aliases} - {str(canonical)}
        if not alias_keys:
            return

        for fact in existing_facts:
            if (fact.kind == FactKind.CONSTRAINT
                    and fact.is_valid
                    and fact.metadata.source_file in alias_keys):
                fact.is_valid = False
                superseded_facts.append(fact)

        for key in alias_keys:
            self._checksums.pop(key, None)

        logger.info(
            f"Collapsed {len(alias_keys)} alias spelling(s) of {canonical} "
            f"({', '.join(sorted(alias_keys))})"
        )

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
        atomic_write_json(CHECKSUM_FILE, self._checksums, indent=2)
