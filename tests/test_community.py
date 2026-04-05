"""Tests for neo.memory.community - community fact feed ingestion."""

import json
import time
from unittest.mock import patch

import pytest

from neo.memory.community import (
    COMMUNITY_CONFIDENCE,
    CommunityFeedIngester,
    FETCH_INTERVAL,
)
from neo.memory.models import Fact, FactMetadata, FactScope


SAMPLE_FEED = {
    "version": "1",
    "updated": "2026-04-05",
    "facts": [
        {
            "subject": "Docker layer caching",
            "body": "Copy deps before source.",
            "kind": "pattern",
            "tags": ["docker"],
        },
        {
            "subject": "GraphQL N+1",
            "body": "Use DataLoader.",
            "kind": "pattern",
            "tags": ["graphql"],
        },
    ],
}


@pytest.fixture
def tmp_dirs(tmp_path):
    cache_dir = tmp_path / "neo"
    cache_dir.mkdir()
    checksum_dir = tmp_path / "constraints"
    checksum_dir.mkdir()
    return cache_dir, checksum_dir


@pytest.fixture
def ingester(tmp_dirs):
    cache_dir, checksum_dir = tmp_dirs
    with patch("neo.memory.community.CACHE_DIR", cache_dir), \
         patch("neo.memory.community.CACHE_FILE", cache_dir / "community_facts_cache.json"), \
         patch("neo.memory.community.CHECKSUM_DIR", checksum_dir), \
         patch("neo.memory.community.CHECKSUM_FILE", checksum_dir / "checksums.json"), \
         patch.object(CommunityFeedIngester, "_fetch_remote", return_value=SAMPLE_FEED):
        yield CommunityFeedIngester(org_id="testorg", project_id="testproj")


class TestFeedIngestion:
    def test_loads_community_facts(self, ingester):
        new_facts, _ = ingester.ingest([])
        assert len(new_facts) == 2
        assert new_facts[0].subject == "Docker layer caching"
        assert new_facts[1].subject == "GraphQL N+1"

    def test_facts_are_global_scope(self, ingester):
        new_facts, _ = ingester.ingest([])
        for f in new_facts:
            assert f.scope == FactScope.GLOBAL

    def test_confidence_is_0_6(self, ingester):
        new_facts, _ = ingester.ingest([])
        for f in new_facts:
            assert f.metadata.confidence == COMMUNITY_CONFIDENCE

    def test_tagged_as_community(self, ingester):
        new_facts, _ = ingester.ingest([])
        for f in new_facts:
            assert "community" in f.tags
            assert "auto-ingested" in f.tags

    def test_preserves_entry_tags(self, ingester):
        new_facts, _ = ingester.ingest([])
        assert "docker" in new_facts[0].tags
        assert "graphql" in new_facts[1].tags


class TestCaching:
    def test_caches_response(self, ingester, tmp_dirs):
        cache_dir, _ = tmp_dirs
        ingester.ingest([])
        cache_file = cache_dir / "community_facts_cache.json"
        assert cache_file.exists()
        cached = json.loads(cache_file.read_text())
        assert "data" in cached
        assert "fetched_at" in cached

    def test_uses_cache_when_fresh(self, ingester):
        """Second call within FETCH_INTERVAL should use cache, not fetch."""
        ingester.ingest([])

        # Replace fetch with one that would return different data
        with patch.object(ingester, "_fetch_remote", return_value={"facts": [{"subject": "new", "body": "b", "kind": "pattern", "tags": []}]}):
            # Content hash hasn't changed (same cache), so no new facts
            new2, _ = ingester.ingest([])
            assert len(new2) == 0

    def test_falls_back_to_stale_cache(self, tmp_dirs):
        """When fetch fails, use stale cache."""
        cache_dir, checksum_dir = tmp_dirs

        # Pre-populate cache with stale data
        cache_file = cache_dir / "community_facts_cache.json"
        cache_file.write_text(json.dumps({
            "fetched_at": time.time() - FETCH_INTERVAL - 100,  # stale
            "data": SAMPLE_FEED,
        }))

        with patch("neo.memory.community.CACHE_DIR", cache_dir), \
             patch("neo.memory.community.CACHE_FILE", cache_file), \
             patch("neo.memory.community.CHECKSUM_DIR", checksum_dir), \
             patch("neo.memory.community.CHECKSUM_FILE", checksum_dir / "checksums.json"), \
             patch.object(CommunityFeedIngester, "_fetch_remote", return_value=None):
            ing = CommunityFeedIngester()
            new_facts, _ = ing.ingest([])

        assert len(new_facts) == 2  # got facts from stale cache


class TestContentHashing:
    def test_skips_unchanged_feed(self, ingester):
        new1, _ = ingester.ingest([])
        assert len(new1) == 2

        new2, _ = ingester.ingest([])
        assert len(new2) == 0

    def test_reingests_changed_feed(self, ingester):
        new1, _ = ingester.ingest([])
        assert len(new1) == 2

        updated_feed = {
            "version": "2",
            "facts": [{"subject": "New pattern", "body": "New.", "kind": "pattern", "tags": []}],
        }
        # Write updated cache to simulate feed change
        ingester._write_cache(updated_feed)

        with patch.object(ingester, "_fetch_remote", return_value=updated_feed):
            # Force stale so it re-fetches
            with patch.object(ingester, "_is_stale", return_value=True):
                new2, superseded = ingester.ingest(new1)

        assert len(new2) == 1
        assert new2[0].subject == "New pattern"
        assert len(superseded) == 2


class TestSupersession:
    def test_supersedes_old_community_facts(self, ingester):
        new1, _ = ingester.ingest([])

        updated_feed = {
            "version": "2",
            "facts": [{"subject": "V2", "body": "Updated.", "kind": "pattern", "tags": []}],
        }
        with patch.object(ingester, "_fetch_remote", return_value=updated_feed), \
             patch.object(ingester, "_is_stale", return_value=True):
            _, superseded = ingester.ingest(new1)

        for f in superseded:
            assert f.is_valid is False

    def test_does_not_supersede_non_community_facts(self, ingester):
        other = Fact(
            subject="user fact", body="mine",
            metadata=FactMetadata(source_file="community_feed"),
            tags=["seed"],
        )
        ingester.ingest([other])
        assert other.is_valid is True


class TestNetworkFailure:
    def test_returns_empty_on_no_cache_no_network(self, tmp_dirs):
        cache_dir, checksum_dir = tmp_dirs
        with patch("neo.memory.community.CACHE_DIR", cache_dir), \
             patch("neo.memory.community.CACHE_FILE", cache_dir / "community_facts_cache.json"), \
             patch("neo.memory.community.CHECKSUM_DIR", checksum_dir), \
             patch("neo.memory.community.CHECKSUM_FILE", checksum_dir / "checksums.json"), \
             patch.object(CommunityFeedIngester, "_fetch_remote", return_value=None):
            ing = CommunityFeedIngester()
            new, sup = ing.ingest([])

        assert new == []
        assert sup == []
