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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

from neo.memory.io_utils import atomic_write_json
from neo.memory.metrics import record as metrics_record
from neo.memory.models import FactKind, FactScope, Provenance
from neo.memory.outcomes import SESSIONS_DIR

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


# ---------------------------------------------------------------------------
# Stage B/C: LM lesson extraction + verify-at-admission
# ---------------------------------------------------------------------------

# Bound downstream token usage per episode (parser leaves these unbounded).
_MAX_ASST_CHARS = 2500
_MAX_ERR_CHARS = 800
# Lessons are "1-2 sentences"; bound the body so a runaway extraction can't bloat
# the store or the embedding.
_MAX_BODY_CHARS = 600

# Transcript lessons are a single LM assertion grounded in observed behavior;
# cap their initial confidence so they never out-rank corroborated facts.
_MAX_LESSON_CONFIDENCE = 0.6
_TRANSCRIPT_TAG = "transcript-derived"

_EXTRACT_PROMPT = """You are mining a coding-assistant transcript for GENERALIZABLE engineering lessons that would help on FUTURE tasks in this or other codebases.

Given one episode (a user request + what the assistant did, including any tool errors), extract 0 to 3 transferable lessons. A good lesson is a reusable rule, pattern, gotcha, or correction — NOT a restatement of what happened, NOT project-trivia (file paths, line numbers, one-off values).

Return STRICT JSON: {"lessons":[{"kind":"pattern|failure","subject":"<=8 words","body":"1-2 sentences, generalizable","domain":"testing|debugging|git|architecture|performance|workflow|code-style|security|file-patterns|other","confidence":0.0-1.0,"evidence_span":"a SHORT verbatim quote (<=120 chars) copied EXACTLY from the episode text below that justifies the lesson"}]}
If there is no transferable lesson, return {"lessons":[]}.

EPISODE:
USER ASK: <<ASK>>
ASSISTANT DID: <<ASST>>
TOOLS USED: <<TOOLS>>
ERRORS: <<ERRS>>
"""

_VERIFY_PROMPT = """You are a skeptical reviewer guarding a long-term memory store. Default to REJECT unless the lesson is clearly worth keeping.

Reject if the lesson is: a restatement of one episode rather than a transferable rule; project-trivia; vague; obvious boilerplate; or not actually supported by the evidence quote.

LESSON: <<SUBJECT>> — <<BODY>>
EVIDENCE QUOTE: <<EVIDENCE>>

Return STRICT JSON: {"keep": true|false, "reason": "<=15 words"}
"""


def _parse_json(text: str) -> Optional[dict]:
    """Parse the first JSON object from an LM response, tolerantly.

    Uses ``raw_decode`` from the first ``{``, which parses one complete object
    and ignores any surrounding prose — including trailing text that itself
    contains braces (where a first-brace/last-brace slice would fail). Fails
    closed (returns ``None``) rather than risk bad data.
    """
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _normalize_ws(s: str) -> str:
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Source adapters — one per AI tool. Each yields the common Episode shape so the
# extract→verify→admit pipeline and per-source watermark are reused unchanged.
# ---------------------------------------------------------------------------


class TranscriptSource(Protocol):
    """A tool whose transcripts neo mines.

    ``name`` namespaces the watermark; ``scope`` is the FactScope for facts
    derived from this source (PROJECT for repo-bound tools, GLOBAL for
    cross-agent tools that aren't tied to one repo).
    """

    name: str
    scope: FactScope

    def collect_episodes(self) -> list[Episode]:
        ...


class ClaudeCodeSource:
    """Claude Code session transcripts for one project."""

    name = "claude-code"
    scope = FactScope.PROJECT

    def __init__(self, codebase_root: Optional[str]):
        self.codebase_root = codebase_root

    def collect_episodes(self) -> list[Episode]:
        return collect_episodes(self.codebase_root)


class TranscriptIngester:
    """Extract verified, generalizable lessons from transcript episodes and
    admit them directly as PATTERN/FAILURE facts.

    Stage B (extract) and Stage C (verify) both call the configured LM
    adapter. Admission is gated by two hard filters: the verifier must keep
    the lesson, and its ``evidence_span`` must appear verbatim in the source
    episode (provable evidence, not claimed).
    """

    def __init__(self, store, lm_adapter, codebase_root: Optional[str] = None,
                 sources: Optional[list] = None):
        self._store = store
        self._lm = lm_adapter
        self.codebase_root = codebase_root or getattr(store, "codebase_root", None)
        # Default source set; add Codex/etc. here as adapters land.
        self.sources = sources if sources is not None else [
            ClaudeCodeSource(self.codebase_root),
        ]

    # -- Stage B ---------------------------------------------------------
    def extract_lessons(self, ep: Episode) -> list[dict]:
        prompt = (
            _EXTRACT_PROMPT
            .replace("<<ASK>>", ep.ask)
            .replace("<<ASST>>", (" ".join(ep.assistant_text)[:_MAX_ASST_CHARS]) or "(no text)")
            .replace("<<TOOLS>>", ", ".join(dict.fromkeys(ep.tools)) or "(none)")
            .replace("<<ERRS>>", (" | ".join(ep.errors)[:_MAX_ERR_CHARS]) or "(none)")
        )
        data = self._lm_json(prompt)
        if not data:
            return []
        return [L for L in data.get("lessons", []) if isinstance(L, dict) and L.get("body")]

    # -- Stage C ---------------------------------------------------------
    def verify(self, lesson: dict, ep: Episode) -> bool:
        """Two hard gates: verbatim evidence present, then adversarial judge."""
        span = _normalize_ws(str(lesson.get("evidence_span", "")))
        if not span:
            return False
        # Include tool names: the extract prompt shows them, so a lesson may
        # legitimately cite one as evidence.
        haystack = _normalize_ws(" ".join([ep.ask, *ep.assistant_text, *ep.errors, *ep.tools]))
        if span not in haystack:
            return False  # hallucinated / non-verbatim evidence
        prompt = (
            _VERIFY_PROMPT
            .replace("<<SUBJECT>>", str(lesson.get("subject", "")))
            .replace("<<BODY>>", str(lesson.get("body", "")))
            .replace("<<EVIDENCE>>", span)
        )
        data = self._lm_json(prompt)
        return bool(data and data.get("keep") is True)

    # -- admission -------------------------------------------------------
    def admit(self, lesson: dict, ep: Episode, scope: FactScope = FactScope.PROJECT):
        kind = FactKind.FAILURE if lesson.get("kind") == "failure" else FactKind.PATTERN
        try:
            raw_conf = float(lesson.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            raw_conf = 0.5  # non-numeric LM confidence ("high") -> conservative default
        confidence = min(raw_conf, _MAX_LESSON_CONFIDENCE)
        domain = lesson.get("domain") or ""
        if domain == "other":
            domain = ""
        return self._store.add_fact(
            subject=str(lesson.get("subject", ""))[:120],
            body=str(lesson.get("body", ""))[:_MAX_BODY_CHARS],
            kind=kind,
            scope=scope,
            confidence=confidence,
            source_prompt=ep.ask[:200],
            tags=[_TRANSCRIPT_TAG],
            provenance=Provenance.INFERRED,  # LM generalization, no bonus over corroborated facts
            domain=domain,  # first-class field so retrieve_relevant(domain=...) matches
        )

    def ingest_episode(self, ep: Episode, scope: FactScope = FactScope.PROJECT) -> int:
        """Full per-episode pipeline; returns number of facts admitted.

        ``store.add_fact`` saves to disk on each call, so an episode's facts
        are durably written before ``ingest`` advances the watermark.
        """
        if not ep.is_substantive:
            return 0
        admitted = 0
        for lesson in self.extract_lessons(ep):
            if self.verify(lesson, ep):
                self.admit(lesson, ep, scope)
                admitted += 1
        return admitted

    # -- incremental ingest with per-source watermark --------------------
    def ingest(self, max_episodes: Optional[int] = None,
               max_seconds: Optional[float] = None,
               should_stop: Optional[Callable[[], bool]] = None) -> dict:
        """Mine all configured sources, advancing each source's watermark per
        episode.

        Idempotent per source: an episode's ``anchor_uuid`` is recorded as
        consumed only *after* its facts are durably written, so a re-run skips it
        and a crash mid-episode simply reprocesses it (dedup absorbs any partial).
        Watermarks are namespaced by source and (for project-scoped sources) by
        project_id, so sources never collide and the key survives worktrees/clones
        that share the same git remote.

        ``max_episodes`` / ``max_seconds`` / ``should_stop`` are SHARED across all
        sources (a single per-cycle budget). They stop *dispatching new* episodes
        — an LM call already in flight runs to its own timeout — collapsing a hung
        pass from N×(call timeout) to ~1 and keeping the supervised observer
        responsive to shutdown. The watermark drains the remaining backlog next
        cycle.
        """
        stats = {"episodes_total": 0, "episodes_new": 0,
                 "episodes_processed": 0, "facts_admitted": 0}
        start = time.monotonic()

        def stop_now() -> bool:
            if max_episodes is not None and stats["episodes_processed"] >= max_episodes:
                return True
            if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
                return True
            return bool(should_stop is not None and should_stop())

        for source in self.sources:
            if stop_now():
                break
            consumed = self._load_consumed(source)
            try:
                episodes = source.collect_episodes()
            except Exception as e:  # one bad source must not sink the others
                logger.warning("transcript: source %s collect failed: %s", source.name, e)
                continue
            new = [e for e in episodes if e.anchor_uuid not in consumed]
            stats["episodes_total"] += len(episodes)
            stats["episodes_new"] += len(new)
            for ep in new:
                if stop_now():
                    break
                stats["facts_admitted"] += self.ingest_episode(ep, source.scope)
                consumed.add(ep.anchor_uuid)
                self._persist_consumed(source, consumed)  # advance only after durable
                stats["episodes_processed"] += 1
        if stats["episodes_processed"]:
            metrics_record("transcript_ingest", **stats)
        return stats

    def _watermark_path(self, source) -> Optional[Path]:
        if source.scope == FactScope.PROJECT:
            pid = getattr(self._store, "project_id", None)
            if not pid:
                return None
            suffix = pid
        else:
            suffix = "global"
        return SESSIONS_DIR / f"transcript_watermark_{source.name}_{suffix}.json"

    def _load_consumed(self, source) -> set:
        path = self._watermark_path(source)
        if not path or not path.exists():
            return set()
        try:
            return set(json.loads(path.read_text(encoding="utf-8")).get("consumed", []))
        except (OSError, json.JSONDecodeError):
            return set()

    def _persist_consumed(self, source, consumed: set) -> None:
        path = self._watermark_path(source)
        if not path:
            return
        try:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, {"consumed": sorted(consumed)})
        except OSError as e:
            logger.warning("transcript: failed to persist watermark: %s", e)

    def _lm_json(self, prompt: str) -> Optional[dict]:
        try:
            out = self._lm.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.2,
            )
        except Exception as e:
            logger.warning("transcript: LM call failed: %s", e)
            return None
        return _parse_json(out)
