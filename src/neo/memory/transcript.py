"""
Claude Code transcript parsing for behavioral-signal ingestion.

Reads Claude Code session transcripts
(``~/.claude/projects/{codebase_root with / -> -}/*.jsonl``) and segments
them into *episodes* — a human request plus the assistant work that
followed — for downstream LM lesson extraction.

This is the schema-correct Stage A: Claude Code transcripts interleave
~10 record ``type`` values across multiple ``sessionId``s, and ~94% of
``user``-role records are ``tool_result`` envelopes rather than human
messages. We therefore parse defensively — filtering to ``user``/
``assistant`` records, skipping sidechains and meta records, and
distinguishing genuine human text from tool output by content-block
structure. Only ``user``/``assistant`` records (which always carry
``uuid``/``sessionId``/``timestamp``) anchor episodes, so every episode
has a stable watermark key.

Parsing only — no LM calls, no fact creation. Extraction and admission
live in the ingester that consumes these episodes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Record types that carry a conversational message we care about. Everything
# else (``ai-title``, ``last-prompt``, ``file-history-snapshot``,
# ``permission-mode``, ``queue-operation``, ``summary``, ...) is skipped — those
# are typeless/uuid-less metadata and never anchor an episode.
_MESSAGE_TYPES = frozenset({"user", "assistant"})

# Cap on the captured human ask, to bound stored size. Assistant/error text is
# bounded by the downstream extractor, not here.
_MAX_ASK_CHARS = 1500


@dataclass
class Episode:
    """A human request plus the assistant activity that followed it.

    ``session_id`` + ``last_uuid`` form the watermark key: an episode is
    only marked consumed after its derived facts are durably written.
    """

    session_id: str
    anchor_uuid: str          # uuid of the human message that opened the episode
    last_uuid: str            # uuid of the last record folded into the episode
    timestamp: str            # timestamp of the anchor message
    ask: str                  # the human request text
    assistant_text: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_substantive(self) -> bool:
        """True if there is enough here to be worth an extraction call."""
        return bool(self.ask) and bool(self.assistant_text or self.tools or self.errors)


def resolve_transcript_dir(codebase_root: Optional[str]) -> Optional[Path]:
    """Map a codebase root to its Claude Code transcript directory.

    Claude Code derives the project directory by replacing ``/`` with ``-``
    in the absolute path. NOTE: this is a *path* identity, distinct from
    neo's git-remote-hash ``project_id``; transcripts are located by path
    while the facts they produce are scoped by remote. On a worktree/clone
    with a different absolute path, this resolves to that path's transcripts.
    """
    if not codebase_root:
        return None
    encoded = str(codebase_root).replace("/", "-")
    return CLAUDE_PROJECTS_DIR / encoded


def _is_synthetic(text: str) -> bool:
    """True if a user-role string is tool/CLI plumbing, not human input.

    Claude Code injects string-content ``user`` records that are not human
    prose: XML-wrapped control/notification envelopes (``<command-name>``,
    ``<local-command-stdout>``, ``<task-notification>``, ``<bash-input>``, …)
    and the tool-use interrupt marker. Verified against real transcripts:
    every such record begins with ``<`` or the interrupt marker, and zero
    genuine human asks do — so an allowlist on the leading character is robust
    and won't drift the way a denylist of envelope names would.
    """
    s = text.lstrip()
    return s.startswith("<") or s.startswith("[Request interrupted")


def _human_text(content: object) -> str:
    """Extract genuine human-authored text from a user record's content.

    A ``user`` record is human only when its content is a plain string or
    contains ``text`` blocks. ``tool_result`` blocks are tool output wearing
    the user role; synthetic CLI/control strings (see ``_is_synthetic``) also
    wear the user role. Neither counts as human input.
    """
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "\n".join(p for p in parts if p).strip()
    else:
        return ""
    return "" if _is_synthetic(text) else text


def _assistant_parts(content: object) -> tuple[str, list[str]]:
    """Return (assistant_text, tool_names) from an assistant record's content."""
    if isinstance(content, str):
        return content.strip(), []
    text_parts, tools = [], []
    if isinstance(content, list):
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                text_parts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                tools.append(b.get("name", "?"))
    return "\n".join(p for p in text_parts if p).strip(), tools


def _tool_errors(content: object) -> list[str]:
    """Return error strings from tool_result blocks marked is_error."""
    errs: list[str] = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("is_error"):
                c = b.get("content")
                txt = c if isinstance(c, str) else json.dumps(c)
                errs.append(txt[:300])
    return errs


def iter_records(path: Path) -> Iterator[dict]:
    """Yield parsed JSON records from a transcript file, skipping bad lines."""
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # fail closed on malformed lines
    except OSError as e:
        logger.warning("transcript: cannot read %s: %s", path, e)


def build_episodes(path: Path) -> list[Episode]:
    """Parse one transcript file into episodes (schema-correct Stage A).

    Records are filtered to ``user``/``assistant``, sidechains and meta
    records dropped, and partitioned by ``sessionId``. Within a session a
    new episode opens on each genuine human message and absorbs the
    following assistant text, tool uses, and tool errors until the next
    human message.
    """
    # Group records by session, preserving file order. Invariant relied upon:
    # human-message anchors are monotonic in file order; only intra-episode
    # tool_result/assistant records jitter (sub-second), and those fold into the
    # current episode regardless of relative order — so file order never
    # reorders an anchor relative to its episode. (Confirmed across real files;
    # timestamp sorting would buy nothing and risks breaking causal order.)
    by_session: dict[str, list[dict]] = {}
    for r in iter_records(path):
        if r.get("type") not in _MESSAGE_TYPES:
            continue
        if r.get("isSidechain"):
            continue
        sid = r.get("sessionId")
        uuid = r.get("uuid")
        if not sid or not uuid:
            continue  # cannot watermark a record without identity
        by_session.setdefault(sid, []).append(r)

    episodes: list[Episode] = []
    for sid, recs in by_session.items():
        cur: Optional[Episode] = None
        for r in recs:
            msg = r.get("message", {}) or {}
            content = msg.get("content")
            uuid = r["uuid"]
            if r["type"] == "user":
                human = _human_text(content)
                if human:
                    if cur:
                        episodes.append(cur)
                    cur = Episode(
                        session_id=sid,
                        anchor_uuid=uuid,
                        last_uuid=uuid,
                        timestamp=r.get("timestamp", ""),
                        ask=human[:_MAX_ASK_CHARS],
                    )
                elif cur:
                    # tool_result-only user record: fold any errors into the episode
                    cur.errors += _tool_errors(content)
                    cur.last_uuid = uuid
            elif r["type"] == "assistant" and cur is not None:
                text, tools = _assistant_parts(content)
                if text:
                    cur.assistant_text.append(text)
                cur.tools += tools
                cur.last_uuid = uuid
        if cur:
            episodes.append(cur)
    return episodes


def collect_episodes(codebase_root: Optional[str]) -> list[Episode]:
    """Build episodes across all transcript files for a codebase root."""
    tdir = resolve_transcript_dir(codebase_root)
    if tdir is None or not tdir.is_dir():
        return []
    episodes: list[Episode] = []
    for fp in sorted(tdir.glob("*.jsonl")):
        episodes.extend(build_episodes(fp))
    return episodes
