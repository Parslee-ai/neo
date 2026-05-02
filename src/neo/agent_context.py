"""Discover AI-tool config / instruction files in a codebase root.

Many agent ecosystems drop project-local instruction documents into
well-known paths (CLAUDE.md, .cursor/rules/*.md, .github/copilot-
instructions.md, etc.). These are the highest-signal "how to work in this
codebase" docs the team has already invested in writing — much better
than anything neo could derive from raw source.

We surface them in the prompt unconditionally, separately from the
relevance-ranked file dump, because their value is global to the project
rather than tied to a specific query embedding.

Scope is intentionally narrow: markdown-style instruction content only.
Tool config (settings.json, hooks, JSON/YAML blobs) is not loaded — those
encode behavior of the tool, not guidance for this agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentDoc:
    """One discovered instruction document."""
    path: str           # path relative to the codebase root
    source: str         # short label of the originating tool (e.g. "claude", "cursor")
    content: str        # truncated to PER_FILE_CAP_BYTES

    def size(self) -> int:
        return len(self.content)


# Caps. Picked so a project with rich AI-tool docs across all the major
# ecosystems can't blow up the prompt — total injected bytes stay sub-32KB.
PER_FILE_CAP_BYTES = 6_000
TOTAL_CAP_BYTES = 32_000


# Discovery rules in priority order. Earlier sources outrank later ones
# when we hit the total byte cap. The list is the load-bearing surface — to
# add a new agent ecosystem, add an entry here.
#
# Each rule says: starting from the codebase root, look for `glob` patterns;
# tag matches with `source`. Globs use Path.glob semantics.
_DISCOVERY_RULES: list[tuple[str, list[str]]] = [
    # Root-level instruction files — the highest-signal slot, since users
    # who write them put them where every agent will see them.
    ("claude",     ["CLAUDE.md"]),
    ("agents",     ["AGENTS.md"]),
    ("cursor",     [".cursorrules"]),
    ("windsurf",   [".windsurfrules"]),
    # Dot-directory ecosystems — each tool's modern home.
    ("claude",     [".claude/CLAUDE.md", ".claude/agents/*.md", ".claude/commands/*.md"]),
    ("cursor",     [".cursor/rules/*.md", ".cursor/rules/*.mdc",
                    ".cursor/rules/**/*.md", ".cursor/rules/**/*.mdc"]),
    ("copilot",    [".github/copilot-instructions.md"]),
    ("agents",     [".github/AGENTS.md"]),
    ("continue",   [".continue/*.md", ".continue/**/*.md"]),
    ("augment",    [".augment/*.md", ".augment/**/*.md"]),
    ("specify",    [".specify/*.md", ".specify/**/*.md"]),
    ("aider",      [".aider/*.md"]),
    # Codex's primary instruction file is AGENTS.md (cross-tool standard,
    # already discovered above). The .codex/ dotdir is its own home for
    # tool-specific overrides and prompt fragments.
    ("codex",      [".codex/*.md", ".codex/**/*.md"]),
    ("codeium",    [".codeium/*.md"]),
]


def discover(root: Optional[Path | str]) -> list[AgentDoc]:
    """Walk the codebase root and return discovered instruction docs.

    Returns [] if `root` is None, missing, or unreadable. Never raises;
    discovery failure must not break neo's main path.
    """
    if root is None:
        return []

    root_path = Path(root).expanduser()
    if not root_path.is_dir():
        return []

    docs: list[AgentDoc] = []
    seen: set[Path] = set()
    total_bytes = 0

    for source, patterns in _DISCOVERY_RULES:
        for pattern in patterns:
            for match in _safe_glob(root_path, pattern):
                if match in seen:
                    continue
                seen.add(match)

                doc = _read_doc(root_path, match, source)
                if doc is None:
                    continue

                # Honor the global cap by stopping cleanly mid-stream.
                if total_bytes + doc.size() > TOTAL_CAP_BYTES:
                    logger.debug(
                        "agent_context: total cap reached at %d bytes; "
                        "stopping after %s",
                        total_bytes, doc.path,
                    )
                    return docs

                docs.append(doc)
                total_bytes += doc.size()

    return docs


def format_for_prompt(docs: list[AgentDoc]) -> str:
    """Render docs as a compact prompt section. Empty string when nothing
    was discovered, so the caller can concatenate without a guard.
    """
    if not docs:
        return ""

    parts: list[str] = ["", "PROJECT-LOCAL AGENT CONTEXT (instructions the team has written for AI agents):"]
    for d in docs:
        parts.append(f"\n--- {d.path} (source: {d.source}) ---\n{d.content}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _safe_glob(root: Path, pattern: str) -> Iterable[Path]:
    """Glob without letting permission errors or bad symlinks crash discovery."""
    try:
        # Path.glob handles ** correctly and yields lazily.
        return sorted(root.glob(pattern))
    except (OSError, ValueError) as exc:
        logger.debug("agent_context: glob %r failed under %s: %s", pattern, root, exc)
        return []


def _read_doc(root: Path, path: Path, source: str) -> Optional[AgentDoc]:
    """Read one file, truncated to PER_FILE_CAP_BYTES. Returns None on failure
    or if the file is not a regular file (e.g. broken symlink).
    """
    try:
        if not path.is_file():
            return None
        content = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError) as exc:
        logger.debug("agent_context: cannot read %s: %s", path, exc)
        return None

    if len(content) > PER_FILE_CAP_BYTES:
        content = content[:PER_FILE_CAP_BYTES] + "\n... [truncated]"

    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = str(path)

    return AgentDoc(path=rel, source=source, content=content)
