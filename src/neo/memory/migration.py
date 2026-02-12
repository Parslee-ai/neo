"""
One-time migration from PersistentReasoningMemory to FactStore.

Reads old global_memory.json and local_*.json files (never modifies them)
and converts ReasoningEntry dicts to Fact objects.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from neo.memory.models import Fact, FactKind, FactMetadata, FactScope

logger = logging.getLogger(__name__)

# Mapping from old pattern prefixes to FactKind
_KIND_MAP = {
    "feature": FactKind.DECISION,
    "bugfix": FactKind.FAILURE,
    "refactor": FactKind.ARCHITECTURE,
    "algorithm": FactKind.PATTERN,
    "explanation": FactKind.PATTERN,
    "parse_failure": FactKind.FAILURE,
}


def migrate_from_legacy(
    old_global_path: Path,
    old_local_path: Optional[Path] = None,
    org_id: str = "",
    project_id: str = "",
) -> list[Fact]:
    """Migrate old ReasoningEntry dicts to Fact objects.

    Reads from old files but never modifies them, preserving rollback capability.

    Args:
        old_global_path: Path to ~/.neo/global_memory.json
        old_local_path: Path to ~/.neo/local_{hash}.json (optional)
        org_id: Organization ID for new facts.
        project_id: Project ID for new facts.

    Returns:
        List of migrated Fact objects.
    """
    migrated: list[Fact] = []

    # Migrate global entries
    global_entries = _load_old_file(old_global_path)
    for entry_dict in global_entries:
        fact = _convert_entry(entry_dict, FactScope.GLOBAL, org_id, project_id)
        if fact:
            migrated.append(fact)

    # Migrate local entries
    if old_local_path:
        local_entries = _load_old_file(old_local_path)
        for entry_dict in local_entries:
            fact = _convert_entry(entry_dict, FactScope.PROJECT, org_id, project_id)
            if fact:
                migrated.append(fact)

    logger.info(f"Migration: converted {len(migrated)} entries from legacy format")
    return migrated


def _load_old_file(path: Path) -> list[dict]:
    """Load entries from an old-format JSON file."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("entries", [])
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read legacy file {path}: {e}")
        return []


def _convert_entry(
    entry: dict,
    scope: FactScope,
    org_id: str,
    project_id: str,
) -> Optional[Fact]:
    """Convert a single ReasoningEntry dict to a Fact.

    Mapping:
    - pattern -> subject
    - reasoning + suggestion + code_template + pitfalls -> body
    - algorithm_type -> tags
    - Infers kind from pattern prefix
    - Preserves embeddings directly (same Jina model)
    """
    pattern = entry.get("pattern", "")
    if not pattern:
        return None

    # Infer kind from pattern prefix (e.g., "feature: Review..." -> DECISION)
    kind = FactKind.PATTERN  # default
    pattern_lower = pattern.lower()
    for prefix, fk in _KIND_MAP.items():
        if pattern_lower.startswith(prefix):
            kind = fk
            break

    # Build subject from pattern
    subject = pattern[:100]

    # Build body from multiple old fields
    body_parts = []
    if entry.get("reasoning"):
        body_parts.append(f"Reasoning: {entry['reasoning']}")
    if entry.get("suggestion"):
        body_parts.append(f"Suggestion: {entry['suggestion']}")
    if entry.get("code_template"):
        body_parts.append(f"Code template: {entry['code_template']}")
    if entry.get("code_skeleton"):
        body_parts.append(f"Code skeleton: {entry['code_skeleton']}")
    if entry.get("common_pitfalls"):
        pitfalls = entry["common_pitfalls"]
        if isinstance(pitfalls, list):
            body_parts.append("Pitfalls: " + "; ".join(pitfalls))
    if entry.get("when_to_use"):
        body_parts.append(f"When to use: {entry['when_to_use']}")

    body = "\n".join(body_parts) if body_parts else pattern

    # Build tags from algorithm_type and algorithm_category
    tags = ["migrated"]
    if entry.get("algorithm_type"):
        tags.append(entry["algorithm_type"])
    if entry.get("algorithm_category"):
        tags.append(entry["algorithm_category"])

    # Preserve embedding if present and valid
    embedding = None
    if "embedding" in entry and entry["embedding"] is not None:
        try:
            emb = entry["embedding"]
            if isinstance(emb, list):
                emb = np.array(emb, dtype=np.float32)
            if isinstance(emb, np.ndarray) and np.isfinite(emb).all():
                embedding = emb
        except (ValueError, TypeError):
            pass  # Skip invalid embeddings

    # Map confidence
    confidence = entry.get("confidence", 0.5)
    confidence = max(0.0, min(1.0, confidence))

    fact = Fact(
        subject=subject,
        body=body,
        kind=kind,
        scope=scope,
        org_id=org_id,
        project_id=project_id,
        metadata=FactMetadata(
            created_at=entry.get("created_at", 0.0),
            last_accessed=entry.get("last_used", 0.0),
            access_count=entry.get("use_count", 0),
            source_prompt=entry.get("context", ""),
            confidence=confidence,
        ),
        embedding=embedding,
        tags=tags,
    )

    return fact
