"""
Test embedding split/join logic for storage backends.

Tests the core logic of splitting text and embeddings.
"""

import hashlib


def test_split_logic():
    """Test the logic for splitting entries into text and embeddings."""

    # Simulate entries with embeddings
    entries = [
        {
            "pattern": "test-pattern-1",
            "context": "test-context-1",
            "reasoning": "test reasoning",
            "suggestion": "test suggestion",
            "source_hash": "testhash001",
            "embedding": [0.1, 0.2, 0.3] * 256,  # 768-dim
            "embedding_dim": 768,
            "embedding_model": "jinaai/jina-embeddings-v2-base-code"
        },
        {
            "pattern": "test-pattern-2",
            "context": "test-context-2",
            "reasoning": "test reasoning 2",
            "suggestion": "test suggestion 2",
            "source_hash": "testhash002",
            # No embedding
        },
        {
            "pattern": "test-pattern-3",
            "context": "test-context-3",
            "reasoning": "test reasoning 3",
            "suggestion": "test suggestion 3",
            # No source_hash - should generate stable one
            "embedding": [0.4, 0.5, 0.6] * 256,
            "embedding_dim": 768,
            "embedding_model": "test-model"
        }
    ]

    # Split logic for separating text and embeddings
    text_entries = []
    embedding_docs = []

    for entry in entries:
        # Generate stable ID
        if "source_hash" in entry and entry["source_hash"]:
            entry_id = entry["source_hash"]
        else:
            entry_id = hashlib.sha256(
                f"{entry.get('pattern', '')}:{entry.get('context', '')}".encode()
            ).hexdigest()[:16]

        # Split text and embedding
        text_entry = {**entry}
        if "embedding" in text_entry:
            embedding_vector = text_entry.pop("embedding")
            embedding_dim = text_entry.pop("embedding_dim", None)
            embedding_model = text_entry.pop("embedding_model", None)

            embedding_docs.append({
                "id": entry_id,
                "org_id": "test_org",
                "embedding": embedding_vector,
                "embedding_dim": embedding_dim,
                "embedding_model": embedding_model
            })

        text_entries.append(text_entry)

    # Verify split
    assert len(text_entries) == 3, f"Expected 3 text entries, got {len(text_entries)}"
    assert len(embedding_docs) == 2, f"Expected 2 embedding docs, got {len(embedding_docs)}"

    # Check entry 1 (has embedding)
    assert "embedding" not in text_entries[0], "Text entry should not have embedding"
    assert text_entries[0]["source_hash"] == "testhash001"
    assert embedding_docs[0]["id"] == "testhash001"
    assert len(embedding_docs[0]["embedding"]) == 768

    # Check entry 2 (no embedding)
    assert "embedding" not in text_entries[1]
    assert text_entries[1]["source_hash"] == "testhash002"

    # Check entry 3 (no source_hash, should generate stable one)
    assert "embedding" not in text_entries[2]
    generated_hash = hashlib.sha256("test-pattern-3:test-context-3".encode()).hexdigest()[:16]
    assert embedding_docs[1]["id"] == generated_hash

    print("✓ Split logic correctly separates text and embeddings")


def test_join_logic():
    """Test the logic for joining text entries with embeddings."""

    # Simulate text entries from main container
    text_entries = [
        {
            "pattern": "test-pattern-1",
            "context": "test-context-1",
            "source_hash": "hash001"
        },
        {
            "pattern": "test-pattern-2",
            "context": "test-context-2",
            "source_hash": "hash002"
        },
        {
            "pattern": "test-pattern-3",
            "context": "test-context-3",
            "source_hash": "hash003"
        }
    ]

    # Simulate embeddings from embedding container
    embedding_items = [
        {
            "id": "hash001",
            "org_id": "test_org",
            "embedding": [0.1] * 768,
            "embedding_dim": 768,
            "embedding_model": "model-1"
        },
        {
            "id": "hash003",
            "org_id": "test_org",
            "embedding": [0.3] * 768,
            "embedding_dim": 768,
            "embedding_model": "model-3"
        }
    ]

    # Build embedding map for joining back to entries
    embedding_map = {
        item["id"]: {
            "embedding": item.get("embedding"),
            "embedding_dim": item.get("embedding_dim"),
            "embedding_model": item.get("embedding_model")
        }
        for item in embedding_items
    }

    # Join embeddings to text entries
    for entry in text_entries:
        entry_id = entry.get("source_hash")
        if entry_id and entry_id in embedding_map:
            entry.update(embedding_map[entry_id])

    # Verify join
    assert "embedding" in text_entries[0], "Entry 1 should have embedding"
    assert len(text_entries[0]["embedding"]) == 768
    assert text_entries[0]["embedding_dim"] == 768

    assert "embedding" not in text_entries[1], "Entry 2 should not have embedding (backward compat)"

    assert "embedding" in text_entries[2], "Entry 3 should have embedding"
    assert text_entries[2]["embedding_model"] == "model-3"

    print("✓ Join logic correctly merges embeddings with text entries")


def test_round_trip():
    """Test full round-trip of split and join."""

    original = [
        {
            "pattern": "p1",
            "context": "c1",
            "source_hash": "h1",
            "embedding": [0.1] * 768,
            "embedding_dim": 768,
            "embedding_model": "m1",
            "other_field": "value1"
        },
        {
            "pattern": "p2",
            "context": "c2",
            "source_hash": "h2",
            "other_field": "value2"
        }
    ]

    # Split
    text_entries = []
    embedding_docs = []

    for entry in original:
        entry_id = entry["source_hash"]
        text_entry = {**entry}

        if "embedding" in text_entry:
            embedding_vector = text_entry.pop("embedding")
            embedding_dim = text_entry.pop("embedding_dim", None)
            embedding_model = text_entry.pop("embedding_model", None)

            embedding_docs.append({
                "id": entry_id,
                "org_id": "test",
                "embedding": embedding_vector,
                "embedding_dim": embedding_dim,
                "embedding_model": embedding_model
            })

        text_entries.append(text_entry)

    # Join
    embedding_map = {
        item["id"]: {
            "embedding": item.get("embedding"),
            "embedding_dim": item.get("embedding_dim"),
            "embedding_model": item.get("embedding_model")
        }
        for item in embedding_docs
    }

    for entry in text_entries:
        entry_id = entry.get("source_hash")
        if entry_id and entry_id in embedding_map:
            entry.update(embedding_map[entry_id])

    # Verify round-trip
    assert len(text_entries) == 2
    assert text_entries[0]["embedding"] == [0.1] * 768
    assert text_entries[0]["embedding_dim"] == 768
    assert text_entries[0]["embedding_model"] == "m1"
    assert text_entries[0]["other_field"] == "value1"

    assert "embedding" not in text_entries[1]
    assert text_entries[1]["other_field"] == "value2"

    print("✓ Round-trip preserves all data correctly")


if __name__ == "__main__":
    print("Testing Embedding Split/Join Logic...")
    print("=" * 60)

    test_split_logic()
    test_join_logic()
    test_round_trip()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
