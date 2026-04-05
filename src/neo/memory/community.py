"""
Community fact feed ingestion.

Fetches curated patterns from a remote JSON feed (GitHub-hosted) and
ingests them as GLOBAL-scope facts. Checks once per day, caches locally,
and uses content hashing to avoid re-ingestion when the feed hasn't changed.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from neo.memory.models import Fact, FactKind, FactMetadata, FactScope

logger = logging.getLogger(__name__)

COMMUNITY_FEED_URL = (
    "https://raw.githubusercontent.com/Parslee-ai/neo/main/community_facts.json"
)
CACHE_DIR = Path.home() / ".neo"
CACHE_FILE = CACHE_DIR / "community_facts_cache.json"
CHECKSUM_DIR = Path.home() / ".neo" / "constraints"
CHECKSUM_FILE = CHECKSUM_DIR / "checksums.json"

FETCH_INTERVAL = 86400  # 24 hours
FETCH_TIMEOUT = 5  # seconds

COMMUNITY_CONFIDENCE = 0.6

KIND_MAP: dict[str, FactKind] = {
    "constraint": FactKind.CONSTRAINT,
    "architecture": FactKind.ARCHITECTURE,
    "pattern": FactKind.PATTERN,
    "review": FactKind.REVIEW,
    "decision": FactKind.DECISION,
    "known_unknown": FactKind.KNOWN_UNKNOWN,
    "failure": FactKind.FAILURE,
}


class CommunityFeedIngester:
    """Fetches and ingests community-curated facts from a remote feed.

    Checks the feed once per day. Caches the response locally so neo
    works offline. Uses content hashing to skip re-ingestion when the
    feed hasn't changed.
    """

    def __init__(self, org_id: str = "", project_id: str = ""):
        self.org_id = org_id
        self.project_id = project_id
        self._checksums = self._load_checksums()

    def ingest(self, existing_facts: list[Fact]) -> tuple[list[Fact], list[Fact]]:
        """Fetch community feed and return new/updated facts.

        Returns:
            Tuple of (new_facts, superseded_facts).
        """
        feed_data = self._get_feed()
        if feed_data is None:
            return [], []

        content_hash = hashlib.sha256(
            json.dumps(feed_data, sort_keys=True).encode()
        ).hexdigest()

        stored_hash = self._checksums.get("community_feed", "")
        if content_hash == stored_hash:
            return [], []

        logger.info("Ingesting community fact feed")

        # Supersede old community facts
        superseded_facts: list[Fact] = []
        for fact in existing_facts:
            if (fact.is_valid and "community" in fact.tags
                    and "auto-ingested" in fact.tags):
                fact.is_valid = False
                superseded_facts.append(fact)

        # Parse facts
        new_facts: list[Fact] = []
        for entry in feed_data.get("facts", []):
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
                    source_file="community_feed",
                    confidence=COMMUNITY_CONFIDENCE,
                ),
                tags=["community", "auto-ingested"] + tags,
            )
            new_facts.append(fact)

        self._checksums["community_feed"] = content_hash
        self._save_checksums()

        logger.info(f"Community feed: {len(new_facts)} facts loaded")
        return new_facts, superseded_facts

    def _get_feed(self) -> dict | None:
        """Get the feed data, fetching from remote if stale.

        Returns cached data if within FETCH_INTERVAL, otherwise fetches
        fresh. Falls back to cache on network failure.
        """
        cached = self._read_cache()

        if cached and not self._is_stale(cached):
            return cached.get("data")

        fresh = self._fetch_remote()
        if fresh is not None:
            self._write_cache(fresh)
            return fresh

        # Network failed — use stale cache if available
        if cached:
            logger.debug("Using stale community feed cache")
            return cached.get("data")

        return None

    def _is_stale(self, cached: dict) -> bool:
        fetched_at = cached.get("fetched_at", 0)
        return (time.time() - fetched_at) >= FETCH_INTERVAL

    def _fetch_remote(self) -> dict | None:
        """Fetch community_facts.json from the remote URL."""
        try:
            request = Request(
                COMMUNITY_FEED_URL,
                headers={"User-Agent": "neo-reasoner"},
            )
            with urlopen(request, timeout=FETCH_TIMEOUT) as response:
                data = json.loads(response.read().decode("utf-8"))
                logger.info("Fetched community feed from remote")
                return data
        except (URLError, json.JSONDecodeError, OSError) as e:
            logger.debug(f"Community feed fetch failed: {e}")
            return None

    def _read_cache(self) -> dict | None:
        if not CACHE_FILE.exists():
            return None
        try:
            return json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, data: dict) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = {"fetched_at": time.time(), "data": data}
        try:
            CACHE_FILE.write_text(json.dumps(cache))
        except OSError as e:
            logger.debug(f"Failed to write community feed cache: {e}")

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
