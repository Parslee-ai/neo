"""Project notes the rule file points to, indexed for `neo recall`.

A rule file (CLAUDE.md, AGENTS.md) keeps short rules and points at longer
notes for the reasons behind them ("read `docs/engineering-notes/` before
changing a subsystem"). The host loads the rule file, never the notes, so the
notes are what the model does not already have: recall offers the passage that
matches the prompt.

They are NOT facts. As facts they would count against the project's scope cap
and compete with mined lessons for it, and they would enter every full run's
context. They live in a sidecar index instead: the observer, which holds an
embedder, embeds each passage (`refresh_index`); recall, which may not load a
model, scores them with the query vector it already has (`load_index`). A note
edited since it was indexed is left out until the next sweep re-embeds it, so
recall never quotes text the file no longer says.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Import-time home capture: `tests/conftest.py::HOME_PATH_CONSTANTS` carries it.
NOTES_DIR = Path.home() / ".neo" / "notes"

# Project rule files whose backticked paths name notes.
RULE_FILES = ("CLAUDE.md", "AGENTS.md", "agents.md", ".github/AGENTS.md")
MAX_NOTE_FILES = 50
MAX_NOTE_BYTES = 512 * 1024
# One passage. Long enough to carry a rule's reason, short enough that its
# embedding is about one thing; a longer block is cut at sentence ends.
CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 40
# Bumped when the index layout changes; an index of another version is rebuilt.
INDEX_VERSION = 2

_BACKTICK = re.compile(r"`([^`\s]+)`")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def note_files(root: Path) -> list[Path]:
    """Markdown files or directories of them that a rule file names by a
    backticked path inside `root`; rule files themselves excluded."""
    root = root.resolve()
    rules = {(root / name).resolve() for name in RULE_FILES}
    found: list[Path] = []
    for name in RULE_FILES:
        try:
            text = (root / name).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for token in _BACKTICK.findall(text):
            if "/" not in token and not token.endswith(".md"):
                continue
            try:
                target = (root / token).resolve()
            except (OSError, RuntimeError):
                continue
            if root not in target.parents:
                continue
            candidates = sorted(target.glob("*.md")) if target.is_dir() else [target]
            for path in candidates:
                if (path.suffix == ".md" and path.is_file() and path not in rules
                        and path not in found):
                    found.append(path)
    return found[:MAX_NOTE_FILES]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def chunk_note(text: str) -> list[str]:
    """Passages: each top-level bullet or paragraph, long ones cut at
    sentence ends into windows of at most `CHUNK_CHARS`."""
    blocks: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("#"):
            blocks.append([])
            continue
        if not line.strip():
            if blocks and blocks[-1]:
                blocks.append([])
            continue
        if line.startswith("- ") or not blocks:
            blocks.append([])
        blocks[-1].append(line.strip())
    passages: list[str] = []
    for block in blocks:
        body = " ".join(block)
        window = ""
        for sentence in _SENTENCE_END.split(body):
            if window and len(window) + len(sentence) + 1 > CHUNK_CHARS:
                passages.append(window)
                window = ""
            window = f"{window} {sentence}".strip()
        if window:
            passages.append(window)
    return [p for p in passages if len(p) >= MIN_CHUNK_CHARS]


def index_path(root: Path) -> Optional[Path]:
    from neo.memory.scope import detect_org_and_project

    _, project_id = detect_org_and_project(str(root))
    return NOTES_DIR / f"{project_id}.json" if project_id else None


def refresh_index(root: Path, embed: Callable[[str], object], model: str) -> int:
    """Re-embed the notes that changed since the last sweep. Returns the
    number of passages embedded (0 when nothing changed)."""
    import numpy as np

    from neo.memory.io_utils import atomic_write_json

    root = Path(root).resolve()
    target = index_path(root)
    if target is None:
        return 0
    try:
        previous = json.loads(target.read_text())
        if previous.get("model") != model or previous.get("version") != INDEX_VERSION:
            previous = {}
    except (OSError, ValueError):
        previous = {}
    old_files = previous.get("files") or {}
    old_chunks = previous.get("chunks") or []

    files: dict[str, str] = {}
    chunks: list[dict] = []
    embedded = 0
    for path in note_files(root):
        try:
            data = path.read_bytes()[:MAX_NOTE_BYTES]
        except OSError:
            continue
        rel = str(path.relative_to(root))
        files[rel] = _sha(data)
        if old_files.get(rel) == files[rel]:
            chunks.extend(c for c in old_chunks if c.get("file") == rel)
            continue
        for passage in chunk_note(data.decode("utf-8", errors="replace")):
            vector = embed(f"{rel}: {passage}")
            if vector is None:
                return 0  # no embedder: keep the old index rather than a partial one
            # base64 float32: recall reads this file on every prompt, and as
            # JSON numbers it was 4x larger (3.8 MB for this repo's notes).
            raw = np.asarray(vector, dtype=np.float32).tobytes()
            chunks.append({"file": rel, "text": passage,
                           "embedding": base64.b64encode(raw).decode("ascii")})
            embedded += 1
    if previous and files == old_files and not embedded:
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(target, {"version": INDEX_VERSION, "model": model,
                               "files": files, "chunks": chunks})
    return embedded


def load_index(root: Path, model: str) -> list[tuple[str, str, object]]:
    """`(file, passage, embedding)` for every note whose file still matches
    what was embedded. Reads only; [] on any problem."""
    root = Path(root).resolve()
    target = index_path(root)
    try:
        index = json.loads(target.read_text()) if target else {}
    except (OSError, ValueError):
        return []
    if index.get("model") != model or index.get("version") != INDEX_VERSION:
        return []
    current: set[str] = set()
    for rel, sha in (index.get("files") or {}).items():
        try:
            if _sha((root / rel).read_bytes()[:MAX_NOTE_BYTES]) == sha:
                current.add(rel)
        except OSError:
            continue
    import numpy as np

    out = []
    for c in index.get("chunks") or []:
        if not (isinstance(c, dict) and c.get("file") in current
                and isinstance(c.get("text"), str) and isinstance(c.get("embedding"), str)):
            continue
        try:
            vector = np.frombuffer(base64.b64decode(c["embedding"]), dtype=np.float32)
        except (ValueError, TypeError):
            continue
        out.append((c["file"], c["text"], vector))
    return out
