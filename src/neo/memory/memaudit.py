"""
Memory-file audit — read-only hygiene inspection of an AI tool's memory store.

Distinct from `rulesync` (which compares human-authored rule files): this
inspects a tool's *accumulated, learned* memory. v1 targets Claude Code's
per-project `~/.claude/projects/<encoded>/memory/` dir — a `MEMORY.md` index
plus individual fact files with YAML frontmatter (`name`, `description`,
`metadata.type`) and a body.

It reports four hygiene problems, all read-only (never edits memory):

- **malformed**: missing frontmatter / description, or an invalid `type`.
- **duplicate**: two memories with near-identical bodies (redundant).
- **conflict**: two memories on the same topic that contradict (LM-judged; opt-out).
- **index**: a memory file absent from MEMORY.md, or MEMORY.md pointing at a
  missing file.

Dangling `[[links]]` are intentional per the memory spec ("marks something worth
writing later"), so they are NOT flagged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import yaml

from neo.math_utils import batched_cosine, cluster_by_similarity

logger = logging.getLogger(__name__)

VALID_TYPES = {"user", "feedback", "project", "reference"}

# Near-identical bodies -> redundant duplicate.
DUP_THRESHOLD = 0.93
# Same-topic-but-not-duplicate -> candidate contradiction (LM-judged).
ALIGN_THRESHOLD = 0.80
_MAX_CONFLICT_CHECKS = 40

_INDEX_LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


@dataclass
class MemoryEntry:
    path: str
    filename: str
    name: str
    description: str
    mtype: str
    body: str
    links: list[str] = field(default_factory=list)
    malformed: list[str] = field(default_factory=list)
    embedding: object = None


@dataclass
class Duplicate:
    names: list[str]


@dataclass
class Conflict:
    name_a: str
    name_b: str
    explanation: str = ""


@dataclass
class AuditReport:
    memory_dir: str = ""
    entry_count: int = 0
    malformed: list[tuple] = field(default_factory=list)   # (filename, issue)
    duplicates: list[Duplicate] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    index_issues: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def clean(self) -> bool:
        return not (self.malformed or self.duplicates or self.conflicts or self.index_issues)


def resolve_memory_dir(root: str) -> Optional[Path]:
    """The Claude Code memory dir for a codebase root, or None."""
    from neo.memory.transcript import resolve_transcript_dir

    tdir = resolve_transcript_dir(root)
    if tdir is None:
        return None
    mem = tdir / "memory"
    return mem if mem.is_dir() else None


def parse_memory_file(path: Path) -> MemoryEntry:
    """Parse a memory `.md` file into a MemoryEntry, recording malformations."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return MemoryEntry(str(path), path.name, path.stem, "", "", "", malformed=["unreadable"])

    fm: dict = {}
    body = raw
    malformed: list[str] = []
    if raw.lstrip().startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            try:
                loaded = yaml.safe_load(parts[1])
                fm = loaded if isinstance(loaded, dict) else {}
            except yaml.YAMLError:
                malformed.append("unparseable frontmatter")
            body = parts[2].strip()
    if not fm and "unparseable frontmatter" not in malformed:
        malformed.append("missing frontmatter")

    name = str(fm.get("name") or path.stem)
    description = str(fm.get("description") or "")
    meta = fm.get("metadata") if isinstance(fm.get("metadata"), dict) else {}
    mtype = str((meta or {}).get("type") or "")

    if not description:
        malformed.append("missing description")
    if mtype and mtype not in VALID_TYPES:
        malformed.append(f"invalid type {mtype!r}")
    elif not mtype:
        malformed.append("missing type")

    links = _WIKILINK_RE.findall(body)
    return MemoryEntry(
        path=str(path), filename=path.name, name=name, description=description,
        mtype=mtype, body=body, links=links, malformed=malformed,
    )


def _index_targets(memory_dir: Path) -> Optional[set[str]]:
    idx = memory_dir / "MEMORY.md"
    if not idx.is_file():
        return None
    try:
        raw = idx.read_text(encoding="utf-8")
    except OSError:
        return None
    return set(_INDEX_LINK_RE.findall(raw))


def audit_memories(
    entries: list[MemoryEntry],
    *,
    index_targets: Optional[set[str]] = None,
    existing_filenames: Optional[set[str]] = None,
    lm_adapter=None,
) -> AuditReport:
    """Compute hygiene findings over parsed memory entries."""
    report = AuditReport(entry_count=len(entries))

    for e in entries:
        for issue in e.malformed:
            report.malformed.append((e.filename, issue))

    # --- Index consistency vs MEMORY.md ------------------------------------
    if index_targets is not None:
        listed = index_targets
        present = existing_filenames or {e.filename for e in entries}
        for e in entries:
            if e.filename not in listed:
                report.index_issues.append(f"{e.filename} is not listed in MEMORY.md")
        for target in sorted(listed):
            if target not in present:
                report.index_issues.append(f"MEMORY.md references missing file {target}")

    # --- Duplicates & conflicts (embedding-based) --------------------------
    embedded = [e for e in entries if e.embedding is not None]
    if len(embedded) >= 2:
        clusters = cluster_by_similarity(
            embedded, embed_fn=lambda e: e.embedding, threshold=DUP_THRESHOLD
        )
        for cluster in clusters:
            if len(cluster) >= 2:
                report.duplicates.append(Duplicate(names=[e.name for e in cluster]))

        if lm_adapter is not None:
            report.conflicts = _detect_conflicts(embedded, lm_adapter)

    return report


def _detect_conflicts(embedded: list[MemoryEntry], lm_adapter) -> list[Conflict]:
    conflicts: list[Conflict] = []
    seen: set[tuple] = set()
    checks = 0
    for i, ea in enumerate(embedded):
        for eb in embedded[i + 1:]:
            if checks >= _MAX_CONFLICT_CHECKS:
                return conflicts
            sim = batched_cosine([eb.embedding], ea.embedding, default=0.0)[0]
            if ALIGN_THRESHOLD <= sim < DUP_THRESHOLD:
                sig = tuple(sorted((ea.name, eb.name)))
                if sig in seen:
                    continue
                seen.add(sig)
                checks += 1
                verdict = _judge_conflict(ea, eb, lm_adapter)
                if verdict and verdict.get("conflict"):
                    conflicts.append(
                        Conflict(
                            name_a=ea.name, name_b=eb.name,
                            explanation=str(verdict.get("explanation") or "").strip(),
                        )
                    )
    return conflicts


_CONFLICT_PROMPT = """Two persisted memory notes about the same project may \
contradict. Decide whether they state INCOMPATIBLE facts/guidance, or are \
compatible / about different things.

Note A (<<NA>>): <<A>>
Note B (<<NB>>): <<B>>

Respond with JSON only: {"conflict": true|false, "explanation": "<one sentence>"}"""


def _judge_conflict(ea: MemoryEntry, eb: MemoryEntry, lm_adapter) -> Optional[dict]:
    from neo.memory.transcript import _parse_json

    prompt = (
        _CONFLICT_PROMPT
        .replace("<<NA>>", ea.name).replace("<<A>>", (ea.description or ea.body)[:400])
        .replace("<<NB>>", eb.name).replace("<<B>>", (eb.description or eb.body)[:400])
    )
    try:
        out = lm_adapter.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.1,
        )
    except Exception as e:
        logger.warning("memaudit: conflict-judge LM call failed: %s", e)
        return None
    data = _parse_json(out)
    return data if isinstance(data, dict) else None


def find_memory_audit(
    store,
    *,
    root: Optional[str] = None,
    check_conflicts: bool = True,
    lm_adapter=None,
) -> AuditReport:
    """Discover and audit the project's memory dir. Read-only."""
    root = root or getattr(store, "codebase_root", None) or "."
    mem_dir = resolve_memory_dir(root)
    if mem_dir is None:
        return AuditReport(note="no memory directory found for this project")

    entries: list[MemoryEntry] = []
    for path in sorted(mem_dir.glob("*.md")):
        if path.name == "MEMORY.md":
            continue
        entries.append(parse_memory_file(path))

    if not entries:
        return AuditReport(memory_dir=str(mem_dir), note="memory directory is empty")

    embed: Optional[Callable] = getattr(store, "_embed_text", None)
    if embed is not None:
        for e in entries:
            e.embedding = _safe_embed(embed, f"{e.description}\n{e.body}")

    report = audit_memories(
        entries,
        index_targets=_index_targets(mem_dir),
        existing_filenames={e.filename for e in entries},
        lm_adapter=lm_adapter if check_conflicts else None,
    )
    report.memory_dir = str(mem_dir)
    return report


def _safe_embed(embed: Callable, text: str):
    try:
        return embed(text)
    except Exception:
        return None
