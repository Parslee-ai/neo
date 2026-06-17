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

import datetime
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

from neo.math_utils import cosine_similarity
from neo.memory.io_utils import atomic_write_json
from neo.memory.metrics import record as metrics_record
from neo.memory.models import FactKind, FactScope, Provenance
from neo.memory.outcomes import (
    MAX_MINED_OUTCOMES_PER_CYCLE,
    OUTCOME_CORRELATION_SIMILARITY,
    OUTCOME_CORRELATION_WINDOW_SECONDS,
    SESSIONS_DIR,
)

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


def _episode_epoch(ts: str) -> Optional[float]:
    """Parse an Episode timestamp to epoch seconds.

    Transcript timestamps are ISO-8601 (``2026-06-17T15:37:23.572Z``); some
    sources may already store epoch floats. Returns None if unparseable.
    """
    if not ts:
        return None
    try:
        return float(ts)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


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


CAR_SESSIONS_DIR = Path.home() / ".car" / "sessions"


class CarSource:
    """CAR agent session transcripts (``~/.car/sessions/*.json``).

    CAR sessions are ``{id, task, messages:[{role, content}], ...}`` task
    conversations. They are cross-agent and not bound to a git repo, so derived
    facts are GLOBAL-scoped. Each session maps to one episode — there are no
    per-message ids, so the session id is the watermark anchor. (The sibling
    ``~/.car/journals`` are thin action-lifecycle audit logs with no extractable
    content, so they are intentionally not a source.)
    """

    name = "car"
    scope = FactScope.GLOBAL

    def __init__(self, sessions_dir: Optional[Path] = None):
        self._dir = sessions_dir or CAR_SESSIONS_DIR

    def collect_episodes(self) -> list[Episode]:
        if not self._dir.is_dir():
            return []
        episodes: list[Episode] = []
        seen_asks: set[str] = set()
        for fp in sorted(self._dir.glob("*.json")):
            ep = self._session_to_episode(fp)
            if ep is None:
                continue
            # CAR's multi-agent fan-out creates many sessions with identical
            # tasks (e.g. dozens of "What is 6*7?"). Collapse by normalized ask
            # so the toy-duplicate flood never reaches the (global, 200-cap) store.
            sig = ep.ask.strip().lower()
            if sig in seen_asks:
                continue
            seen_asks.add(sig)
            episodes.append(ep)
        return episodes

    @staticmethod
    def _session_to_episode(fp: Path) -> Optional[Episode]:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(d, dict):
            return None
        # Only finished sessions are immutable. An in-flight session can gain
        # turns across runs, and we watermark by session id — so mining it early
        # would permanently skip the finished, substantive version. Skip until done.
        if d.get("finished") is not True:
            return None
        sid = str(d.get("id") or fp.stem)
        user_texts, asst_texts = [], []
        for m in d.get("messages") or []:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if content is None:
                continue  # skip empty messages rather than emit a literal "null"
            if not isinstance(content, str):
                content = json.dumps(content)
            if m.get("role") == "user":
                user_texts.append(content)
            elif m.get("role") == "assistant":
                asst_texts.append(content)
        # The first user message is the actual prompt; fall back to the task field.
        ask = (user_texts[0] if user_texts else str(d.get("task") or "")).strip()
        if not ask:
            return None
        return Episode(
            session_id=sid,
            anchor_uuid=sid,
            last_uuid=sid,
            timestamp=str(d.get("created_at") or ""),
            ask=ask[:_MAX_ASK_CHARS],
            assistant_text=asst_texts,
        )


CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"
# Sanity ceiling only. Rollouts are parsed line-by-line (bounded memory), and
# real working sessions legitimately reach hundreds of MB (inline command
# output), so this is set high — it exists to skip a truly broken file, not to
# drop real data. Logged, not silent.
_CODEX_MAX_ROLLOUT_BYTES = 2 * 1024 * 1024 * 1024
# Codex injects a synthetic "agent history" review wrapper as a user_message;
# it is not a human ask and must not anchor an episode.
_CODEX_SYNTHETIC_PREFIX = "The following is the Codex agent history"
_CODEX_EXIT_RE = re.compile(r"exited with code (\d+)")


def _codex_output_error(output: str) -> Optional[str]:
    """Return a failure snippet from a function_call_output blob, else None.

    Codex command results land in ``function_call_output.output`` as text like
    ``Process exited with code 1\n...`` or ``command timed out after ...`` — the
    only error channel in the modern rollout format. Success blobs say
    ``exited with code 0`` and are ignored.
    """
    if not output:
        return None
    m = _CODEX_EXIT_RE.search(output)
    failed = (m is not None and m.group(1) != "0") or "timed out" in output.lower()
    return output.strip()[:300] if failed else None


class CodexSource:
    """Codex CLI session rollouts (``~/.codex/sessions/**/rollout-*.jsonl``).

    Rollouts are ``{timestamp, type, payload}`` JSONL whose first record is a
    ``session_meta`` carrying ``cwd``. Unlike CAR, that makes them
    project-attributable, so this is PROJECT-scoped: only rollouts whose cwd is
    within the current ``codebase_root`` are ingested. Conversation lives in
    ``event_msg`` records (``user_message`` anchors an episode, ``agent_message``
    is assistant text), tools in ``response_item`` ``function_call`` names, and
    errors in ``exec_command_end`` with a non-zero ``exit_code``. Rollouts are
    append-only, so ``(session_id, record-timestamp)`` is a stable watermark
    anchor.
    """

    name = "codex"
    scope = FactScope.PROJECT

    def __init__(self, codebase_root: Optional[str], sessions_dir: Optional[Path] = None):
        self.codebase_root = codebase_root
        self._dir = sessions_dir or CODEX_SESSIONS_DIR

    def collect_episodes(self) -> list[Episode]:
        root = self.codebase_root
        if not root or not self._dir.is_dir():
            return []
        episodes: list[Episode] = []
        for fp in sorted(self._dir.glob("**/rollout-*.jsonl")):
            if not self._cwd_within_root(fp, root):
                continue
            try:
                if fp.stat().st_size > _CODEX_MAX_ROLLOUT_BYTES:
                    logger.warning("transcript: skipping oversized codex rollout %s", fp.name)
                    continue
            except OSError:
                continue
            episodes.extend(self._rollout_to_episodes(fp))
        return episodes

    @staticmethod
    def _cwd_within_root(fp: Path, root: str) -> bool:
        """Cheap project filter: read only the first record (session_meta)."""
        try:
            with fp.open(encoding="utf-8") as f:
                first = f.readline()
            r = json.loads(first)
        except (OSError, json.JSONDecodeError):
            return False
        if r.get("type") != "session_meta":
            return False
        cwd = (r.get("payload") or {}).get("cwd") or ""
        return cwd == root or cwd.startswith(root.rstrip("/") + "/")

    @staticmethod
    def _rollout_to_episodes(fp: Path) -> list[Episode]:
        episodes: list[Episode] = []
        sid = fp.stem
        msg_idx = 0  # monotonic per-session index keeps anchors unique even if
        # two records share a timestamp; stable across runs (append-only order).
        cur: Optional[Episode] = None
        try:
            handle = fp.open(encoding="utf-8")
        except OSError:
            return []
        with handle as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = r.get("type")
                p = r.get("payload")
                if not isinstance(p, dict):
                    continue
                if t == "session_meta":
                    sid = str(p.get("id") or sid)
                elif t == "event_msg":
                    pt = p.get("type")
                    if pt == "user_message":
                        msg = str(p.get("message") or "").strip()
                        if msg and not _is_synthetic(msg) \
                                and not msg.startswith(_CODEX_SYNTHETIC_PREFIX):
                            if cur:
                                episodes.append(cur)
                            # (sid, msg_idx) is already unique + stable under the
                            # append-only ordering; no timestamp needed.
                            anchor = f"{sid}:{msg_idx}"
                            msg_idx += 1
                            cur = Episode(
                                session_id=sid, anchor_uuid=anchor, last_uuid=anchor,
                                timestamp=str(r.get("timestamp", "")),
                                ask=msg[:_MAX_ASK_CHARS],
                            )
                    elif pt == "agent_message" and cur is not None:
                        # Use agent_message (not response_item/message, role=assistant)
                        # for assistant text — they are byte-identical duplicates;
                        # reading both would double every assistant turn.
                        m = str(p.get("message") or "").strip()
                        if m:
                            cur.assistant_text.append(m)
                    elif pt == "exec_command_end" and cur is not None:
                        # Older-format error channel; modern rollouts use
                        # function_call_output below.
                        if p.get("exit_code") not in (0, None):
                            err = str(p.get("stderr") or "").strip()[:300] or f"exit {p.get('exit_code')}"
                            cur.errors.append(err)
                    elif pt == "patch_apply_end" and cur is not None:
                        if p.get("success") is False:
                            err = str(p.get("stderr") or "").strip()[:300] or "patch apply failed"
                            cur.errors.append(err)
                elif t == "response_item" and cur is not None:
                    pt = p.get("type")
                    if pt == "function_call" and p.get("name"):
                        cur.tools.append(str(p["name"]))
                    elif pt == "function_call_output":
                        # The primary error channel in the modern rollout format.
                        err = _codex_output_error(str(p.get("output") or ""))
                        if err:
                            cur.errors.append(err)
        if cur:
            episodes.append(cur)
        return episodes


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
        # Default source set; add new tool adapters here as they land.
        self.sources = sources if sources is not None else [
            ClaudeCodeSource(self.codebase_root),
            CodexSource(self.codebase_root),
            CarSource(),
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

        all_episodes: list[Episode] = []
        for source in self.sources:
            if stop_now():
                break
            consumed = self._load_consumed(source)
            try:
                episodes = source.collect_episodes()
            except Exception as e:  # one bad source must not sink the others
                logger.warning("transcript: source %s collect failed: %s", source.name, e)
                continue
            all_episodes.extend(episodes)
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

        # Correlate neo's own past suggestions (durable ledger) against all
        # collected episodes to derive real accept/modify outcomes. Best-effort:
        # a failure here must not sink the lesson-ingest stats above.
        try:
            stats["outcomes_mined"] = self.mine_suggestion_outcomes(all_episodes)
        except Exception as e:
            logger.warning("transcript: suggestion-outcome mining failed: %s", e)
            stats["outcomes_mined"] = 0

        if stats["episodes_processed"] or stats.get("outcomes_mined"):
            metrics_record("transcript_ingest", **stats)
        return stats

    # -- Stage D: suggestion-outcome mining ------------------------------
    def _match_episode(self, description: str, candidates: list) -> Optional["Episode"]:
        """Return the candidate episode whose text best matches a suggestion
        description above the similarity floor, or None.

        Uses the store's embedder (Jina) — same vectors as fact retrieval, so no
        extra model is loaded and no LM call is made. Cost is
        O(entries × candidates) embed lookups per cycle; bounded because the
        ledger is compacted every cycle and matching is capped at
        MAX_MINED_OUTCOMES_PER_CYCLE.
        """
        embed = getattr(self._store, "_embed_text", None)
        if embed is None or not description or not candidates:
            return None
        try:
            dvec = embed(description)
        except Exception:
            return None
        if dvec is None:
            return None
        best, best_sim = None, OUTCOME_CORRELATION_SIMILARITY
        for ep in candidates:
            text = " ".join([ep.ask, *ep.assistant_text])[:_MAX_ASST_CHARS]
            if not text.strip():
                continue
            try:
                evec = embed(text)
            except Exception:
                continue
            if evec is None:
                continue
            sim = cosine_similarity(dvec, evec)
            if sim >= best_sim:
                best, best_sim = ep, sim
        return best

    def mine_suggestion_outcomes(self, episodes: list) -> int:
        """Weakly reinforce facts whose past suggestion recurs in later work.

        The durable-ledger, semantically-correlated complement to
        OutcomeTracker._detect_non_git_outcomes: a suggestion whose description
        matches a *subsequent* transcript episode is evidence the suggestion's
        area recurred in later work — weak implicit acceptance, not verified
        diff-overlap. So a match earns the same weak UNVERIFIED delta as neo's
        other non-git signals (see store.apply_mined_outcomes), NOT the strong
        ACCEPTED reward the git matcher reserves for proven acceptance. The
        episode does not contain neo's suggestion text, so this is topic
        recurrence — deliberately not classified as accept-vs-modify (a matched
        episode's tool errors are its own process noise, unrelated to the
        suggestion's fate). Entries that find no match before their correlation
        window lapses are dropped so the ledger stays bounded. Returns the number
        of facts reinforced.
        """
        tracker = getattr(self._store, "_outcome_tracker", None)
        if tracker is None or not hasattr(tracker, "load_suggestion_ledger"):
            return 0
        ledger = tracker.load_suggestion_ledger()
        if not ledger:
            return 0

        dated = [(t, e) for e in episodes if (t := _episode_epoch(e.timestamp)) is not None]
        now = time.time()
        matched_fact_ids: set[str] = set()  # dedup: one reinforcement per fact per cycle
        done_ids: set[str] = set()

        # Ledger order is append order (oldest first) — the right drain order,
        # since oldest entries are closest to window-expiry.
        for entry in ledger:
            if len(matched_fact_ids) >= MAX_MINED_OUTCOMES_PER_CYCLE:
                break
            eid = entry.get("id", "")
            fid = entry.get("fact_id", "")
            ets = float(entry.get("ts", 0) or 0)
            if not fid:
                done_ids.add(eid)
                continue
            window_end = ets + OUTCOME_CORRELATION_WINDOW_SECONDS
            cands = [e for t, e in dated if ets <= t <= window_end]
            if self._match_episode(entry.get("description", ""), cands) is not None:
                matched_fact_ids.add(fid)
                done_ids.add(eid)
            elif now > window_end:
                done_ids.add(eid)  # window lapsed with no match — give up

        # Fan-out dedup: one neo invocation links ALL its suggestions to a
        # SINGLE reasoning fact, so the ledger holds many entries per fact_id.
        # We reinforce each fact at most once per cycle (set above); consume the
        # *other* still-pending entries for any reinforced fact too, so they
        # can't drip extra bumps for the same fact in later cycles. (Genuinely
        # later, post-compaction re-logging can still reinforce — that's a new
        # recurrence, not this fan-out.)
        if matched_fact_ids:
            for entry in ledger:
                if entry.get("fact_id") in matched_fact_ids:
                    done_ids.add(entry.get("id", ""))

        # Compact BEFORE applying, and unconditionally:
        #  - before, because apply_mined_outcomes is NOT idempotent (it bumps the
        #    monotonic success_count); a crash between apply and compaction would
        #    re-mine the entry and double-count. Dropping the ledger entry first
        #    means a crash loses a (noisy, plentiful) signal rather than
        #    corrupting the counter that gates community contribution.
        #  - unconditionally, so the TTL cutoff inside compact_suggestion_ledger
        #    always runs; gating it on done_ids let a ledger of young-unmatched
        #    entries grow without the backstop ever firing.
        tracker.compact_suggestion_ledger(drop_ids=done_ids)
        return self._store.apply_mined_outcomes(list(matched_fact_ids)) if matched_fact_ids else 0

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
