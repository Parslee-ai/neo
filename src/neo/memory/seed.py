"""
Seed fact ingestion from bundled package data.

Ships curated patterns with every neo release. On first run (or after
a package update), loads seed_facts.json as GLOBAL-scope facts.
Checksum-based skip avoids re-ingestion when the seed file hasn't changed.
"""

import hashlib
import json
import logging
from pathlib import Path

from neo.memory.io_utils import atomic_write_json
from neo.memory.models import Fact, FactKind, FactMetadata, FactScope

logger = logging.getLogger(__name__)

SEED_FILE = Path(__file__).parent.parent / "config" / "seed" / "seed_facts.json"
CHECKSUM_DIR = Path.home() / ".neo" / "constraints"
CHECKSUM_FILE = CHECKSUM_DIR / "checksums.json"

SEED_CONFIDENCE = 0.6

# Checksum key, deliberately NOT the file path. The same seed file is reached
# through a different absolute path from every install (a pipx venv, a
# Homebrew site-packages, an editable checkout), and keying on the path made
# each one ingest its own full copy into the shared global store: a live store
# held every seed fact three times, and retrieval injected all three copies of
# one fact into a single prompt.
SEED_CHECKSUM_KEY = "neo:seed_facts"

KIND_MAP: dict[str, FactKind] = {
    "constraint": FactKind.CONSTRAINT,
    "architecture": FactKind.ARCHITECTURE,
    "pattern": FactKind.PATTERN,
    "review": FactKind.REVIEW,
    "decision": FactKind.DECISION,
    "known_unknown": FactKind.KNOWN_UNKNOWN,
    "failure": FactKind.FAILURE,
}


class SeedIngester:
    """Ingests curated facts bundled with the neo package.

    Reads src/neo/config/seed/seed_facts.json and creates GLOBAL-scope
    facts. Uses checksum to detect package updates and re-ingest.
    """

    def __init__(self, org_id: str = "", project_id: str = ""):
        self.org_id = org_id
        self.project_id = project_id
        self._checksums = self._load_checksums()

    def ingest(self, existing_facts: list[Fact]) -> tuple[list[Fact], list[Fact]]:
        """Load seed facts if the seed file has changed.

        Returns:
            Tuple of (new_facts, superseded_facts).
        """
        if not SEED_FILE.exists():
            return [], []

        current_checksum = self._file_checksum(SEED_FILE)
        stored_checksum = self._checksums.get(SEED_CHECKSUM_KEY, "")

        if current_checksum == stored_checksum:
            return [], []

        logger.info("Ingesting seed facts from package")

        # Parse seed file
        try:
            data = json.loads(SEED_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read seed facts: {e}")
            return [], []

        new_facts: list[Fact] = []
        for entry in data.get("facts", []):
            subject = entry.get("subject", "")
            body = entry.get("body", "")
            if not subject or not body:
                continue

            kind = KIND_MAP.get(entry.get("kind", "pattern"), FactKind.PATTERN)
            tags = entry.get("tags", [])

            fact = Fact(
                subject=subject,
                body=body,
                kind=kind,
                scope=FactScope.GLOBAL,
                org_id=self.org_id,
                project_id=self.project_id,
                metadata=FactMetadata(
                    source_file=str(SEED_FILE),
                    confidence=SEED_CONFIDENCE,
                ),
                tags=["seed", "auto-ingested"] + tags,
            )
            new_facts.append(fact)

        # Supersede every earlier seed generation, whichever install path
        # minted it (see SEED_CHECKSUM_KEY), only once the replacement parsed:
        # an unreadable seed file must not retire the generation in use. Only
        # this ingester mints the tag.
        superseded_facts: list[Fact] = []
        for fact in existing_facts:
            if (fact.is_valid and "seed" in fact.tags
                    and fact.scope == FactScope.GLOBAL):
                fact.is_valid = False
                superseded_facts.append(fact)

        # Recorded in memory now, persisted by `save_checksums` only after the
        # caller has saved the facts: a checksum on disk for facts that never
        # reached it would skip this generation forever.
        self._checksums[SEED_CHECKSUM_KEY] = current_checksum

        logger.info(f"Seed facts: {len(new_facts)} loaded")
        return new_facts, superseded_facts

    def _file_checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        try:
            h.update(path.read_bytes())
        except (OSError, IOError):
            return ""
        return h.hexdigest()

    def _load_checksums(self) -> dict[str, str]:
        if CHECKSUM_FILE.exists():
            try:
                return json.loads(CHECKSUM_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def save_checksums(self) -> None:
        """Persist the ingested generation. Call after the facts are saved."""
        atomic_write_json(CHECKSUM_FILE, self._checksums, indent=2)
