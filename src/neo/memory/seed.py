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

from neo.memory.models import Fact, FactKind, FactMetadata, FactScope

logger = logging.getLogger(__name__)

SEED_FILE = Path(__file__).parent.parent / "config" / "seed" / "seed_facts.json"
CHECKSUM_DIR = Path.home() / ".neo" / "constraints"
CHECKSUM_FILE = CHECKSUM_DIR / "checksums.json"

SEED_CONFIDENCE = 0.6

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
        stored_checksum = self._checksums.get(str(SEED_FILE), "")

        if current_checksum == stored_checksum:
            return [], []

        logger.info("Ingesting seed facts from package")

        # Supersede old seed facts
        superseded_facts: list[Fact] = []
        for fact in existing_facts:
            if (fact.is_valid and "seed" in fact.tags
                    and fact.metadata.source_file == str(SEED_FILE)):
                fact.is_valid = False
                superseded_facts.append(fact)

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

        self._checksums[str(SEED_FILE)] = current_checksum
        self._save_checksums()

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

    def _save_checksums(self) -> None:
        CHECKSUM_DIR.mkdir(parents=True, exist_ok=True)
        CHECKSUM_FILE.write_text(json.dumps(self._checksums, indent=2))
