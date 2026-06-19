"""
Memory ingestion — import a peer tool's memory files into neo's store.

Phase 2 of cross-tool memory support (phase 1 is the read-only `memaudit`).
Reads Claude Code's per-project `memory/*.md` and admits each as a fact in
neo's own store, so what one agent learned becomes available to neo's reasoning.

Trust-first admission. A peer tool's memory can be wrong or stale, so imports:

- enter as **REVIEW** (a decaying kind), never as CONSTRAINT/ARCHITECTURE
  (which would bypass decay and be treated as curated truth);
- carry **INFERRED** provenance and an `imported:claude-memory` tag, so
  `add_fact` puts them on **probation** (they must earn promotion via access /
  success like any other fluid fact) and they get the lowest provenance bonus;
- are deduped/superseded by `add_fact`'s existing cosine≥0.85 machinery;
- are watermarked per (project, tool) by content hash, so re-running is
  idempotent and an edited memory re-imports (and supersedes) rather than
  duplicating.

Malformed entries (no frontmatter / no description) are skipped — only
well-formed memory is imported.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Optional

from neo.memory.memaudit import parse_memory_file, resolve_memory_dir
from neo.memory.models import FactKind, FactScope, Provenance

logger = logging.getLogger(__name__)

SOURCE_NAME = "claude-memory"
_IMPORT_CONFIDENCE = 0.4  # modest; probation + decay gate it further


@dataclass
class ImportStats:
    scanned: int = 0
    imported: int = 0
    deduped: int = 0
    skipped_existing: int = 0
    skipped_malformed: int = 0
    dry_run: bool = False
    note: str = ""


def _watermark_path(store):
    from neo.memory.outcomes import SESSIONS_DIR

    pid = getattr(store, "project_id", None) or "global"
    return SESSIONS_DIR / f"memimport_watermark_{SOURCE_NAME}_{pid}.json"


def _load_consumed(store) -> set:
    path = _watermark_path(store)
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("consumed", []))
    except (OSError, json.JSONDecodeError):
        return set()


def _persist_consumed(store, consumed: set) -> None:
    from neo.memory.io_utils import atomic_write_json
    from neo.memory.outcomes import SESSIONS_DIR

    path = _watermark_path(store)
    try:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"consumed": sorted(consumed)})
    except OSError as e:
        logger.warning("memimport: failed to persist watermark: %s", e)


def _is_importable(entry) -> bool:
    """Skip entries we can't trust to be well-formed memory."""
    if not entry.description:
        return False
    if "missing frontmatter" in entry.malformed or "unparseable frontmatter" in entry.malformed:
        return False
    return True


def import_memory(
    store,
    *,
    root: Optional[str] = None,
    confidence: float = _IMPORT_CONFIDENCE,
    dry_run: bool = False,
) -> ImportStats:
    """Import the project's Claude Code memory files into neo's store.

    Idempotent: content-hash watermark skips already-imported, unchanged files.
    """
    root = root or getattr(store, "codebase_root", None) or "."
    mem_dir = resolve_memory_dir(root)
    if mem_dir is None:
        return ImportStats(dry_run=dry_run, note="no memory directory found for this project")

    entries = [
        parse_memory_file(p)
        for p in sorted(mem_dir.glob("*.md"))
        if p.name != "MEMORY.md"
    ]
    if not entries:
        return ImportStats(dry_run=dry_run, note="memory directory is empty")

    consumed = _load_consumed(store)
    new_consumed: set = set()
    stats = ImportStats(dry_run=dry_run)

    for e in entries:
        stats.scanned += 1
        if not _is_importable(e):
            stats.skipped_malformed += 1
            continue
        key = f"{e.filename}:{hashlib.sha256(e.body.encode('utf-8')).hexdigest()[:12]}"
        if key in consumed:
            stats.skipped_existing += 1
            continue
        if dry_run:
            stats.imported += 1
            continue

        before = len(store._facts)
        store.add_fact(
            subject=e.description or e.name,
            body=e.body or e.description,
            kind=FactKind.REVIEW,
            scope=FactScope.PROJECT,
            confidence=confidence,
            source_file=e.path,
            tags=["imported", f"imported:{SOURCE_NAME}", f"memtype:{e.mtype or 'unknown'}"],
            provenance=Provenance.INFERRED,
        )
        if len(store._facts) > before:
            stats.imported += 1
        else:
            stats.deduped += 1
        new_consumed.add(key)

    if not dry_run and new_consumed:
        _persist_consumed(store, consumed | new_consumed)

    return stats
