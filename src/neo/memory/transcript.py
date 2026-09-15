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
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

from neo.math_utils import cosine_similarity
from neo.memory.io_utils import atomic_write_json
from neo.memory.metrics import record as metrics_record
from neo.memory.models import FactKind, FactScope, Provenance
from neo.memory.scope import _get_git_remote_url, _normalize_remote_url
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


def collect_episodes(codebase_root: Optional[str],
                     since: Optional[float] = None) -> list[Episode]:
    """Build episodes across all transcript files for a codebase root.

    ``since`` skips files untouched since that epoch time. Parsing dominates the
    observer's memory: one project measured 104 MB for this source alone, and the
    sweep does 25 projects a cycle. The watermark only gates *admission*, so
    without this an unchanged project still paid the full parse every cycle —
    which is what the docs' "unchanged projects do near-zero work" claimed but
    did not do.
    """
    tdir = resolve_transcript_dir(codebase_root)
    if tdir is None or not tdir.is_dir():
        return []
    episodes: list[Episode] = []
    for fp in sorted(tdir.glob("*.jsonl")):
        if _unchanged_since(fp, since):
            continue
        episodes.extend(build_episodes(fp))
    return episodes


def _unchanged_since(path: "Path", since: Optional[float]) -> bool:
    """True when ``path`` has not been modified since ``since``.

    Conservative by construction: any error, or no ``since`` at all, returns
    False so the file is parsed. Skipping is a pure optimization — a wrong
    "skip" would silently drop learning, so it only ever happens on a positive
    mtime comparison.
    """
    if not since:
        return False
    try:
        return path.stat().st_mtime < since
    except OSError:
        return False


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

#: How many passes an episode may fail with a NON-transient LM error before it
#: is given up. Such an error — a 400 for an oversized prompt, a reply cut off
#: by max_output_tokens, output that is not JSON — recurs identically on every
#: retry, so without a cap it is retried forever.
MAX_EPISODE_LM_FAILURES = 3

#: The same cap for TRANSIENT errors (timeouts, 429s, 5xx), deliberately much
#: higher: they are only ever charged when the provider answered other calls in
#: the same pass (see ``ingest``), which is partial rate-limiting or an episode
#: whose own prompt always times out. Uncapped, that episode would cost a full
#: timeout on every visit forever; capped, it costs hours of retries first.
MAX_EPISODE_TRANSIENT_FAILURES = 20


class LMUnavailable(Exception):
    """The LM call for an episode failed, as opposed to answering "no lessons".

    ``_lm_json`` used to swallow every exception and return None, which
    ``extract_lessons`` turned into ``[]`` — so a DNS outage was recorded as an
    episode with nothing to learn and its watermark advanced for good. A live
    observer logged 610 such failures, every one an episode consumed unmined.
    """

    def __init__(self, cause: BaseException):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.transient = _is_transient_lm_error(cause)


#: HTTP statuses that mean "the provider could not serve this now". Narrower
#: than the SDKs' own retry set on purpose: 409 and 501/505 are deterministic
#: for a given request, so retrying them is the per-episode cap's job, not an
#: outage's.
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


def _is_transient_lm_error(exc: BaseException) -> bool:
    """True when ``exc`` says the provider could not be reached or served, and
    nothing about the request itself.

    Conservative: a transient verdict stops the pass and retries under the much
    larger ``MAX_EPISODE_TRANSIENT_FAILURES``, so only errors that are about
    reachability qualify — socket/connection/timeout errors, the SDKs'
    connection errors, and retryable HTTP statuses. Anything unrecognised
    (including a CAR adapter error) is non-transient and hits the per-episode
    cap, which bounds the cost of a wrong guess. The cause chain is walked
    because SDKs wrap the transport error that actually happened.
    """
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (ConnectionError, TimeoutError)):
            return True
        # openai/anthropic carry `status_code`; google-genai's APIError carries
        # `code`, and the Google adapter re-raises it as ValueError("Rate limit
        # exceeded: ...") from inside the except block, so it is on the chain.
        for attr in ("status_code", "code"):
            status = getattr(cur, attr, None)
            if isinstance(status, int) and status in _TRANSIENT_STATUSES:
                return True
        if isinstance(cur, socket.gaierror):
            return True
        # Looked up in sys.modules rather than imported: an exception can only
        # be an instance of an SDK's class if that SDK is already loaded.
        for module, name in (("openai", "APIConnectionError"),
                             ("anthropic", "APIConnectionError"),
                             ("httpx", "TransportError")):
            cls = getattr(sys.modules.get(module), name, None)
            if isinstance(cls, type) and isinstance(cur, cls):
                return True
        cur = cur.__cause__ or cur.__context__
    return False

def _source_root_key(source) -> str:
    """Which transcript directory a source's drain mark describes.

    The watermark FILE is keyed by project_id, and two clones of one remote
    share it (`flyx/fms` and `flyx/fms2`). Sharing ``consumed`` is right — anchors
    are uuids — but a drain mark is a claim about one directory's files: keyed
    per file, `fms` draining would arm the skip against `fms2`'s backlog."""
    return str(getattr(source, "codebase_root", None) or "")


def _aware_epoch(timestamp: str) -> Optional[float]:
    """``_episode_epoch``, but None for an ISO time without a zone.

    ``_episode_epoch`` reads a naive time as LOCAL, which is fine for the
    outcome window it was written for and wrong for a drain mark: a naive UTC
    stamp read as local on a UTC-4 machine lands the mark four hours late —
    past the one-hour skew margin — and skips files still holding unconsumed
    episodes. Claude Code and Codex write ``Z`` stamps today; a mark must not
    depend on that staying true.
    """
    try:
        float(timestamp)
        return _episode_epoch(timestamp)  # epoch floats name an instant
    except (TypeError, ValueError):
        pass
    try:
        aware = datetime.datetime.fromisoformat(
            str(timestamp).replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return None
    return _episode_epoch(timestamp) if aware else None


def _drain_mark(collect_started: float, pending: list) -> Optional[float]:
    """Epoch time before which a source's files hold no unconsumed episode.

    ``min(collect_started, earliest pending episode timestamp)``. Safe because
    transcripts are append-only: a file's mtime is never earlier than the
    timestamps of the records inside it, so every file holding a pending
    episode has mtime >= the mark and is read next pass. It keeps ADVANCING
    while a backlog drains — a mark that moved only when a pass consumed
    everything would never move for a project producing more episodes per
    rotation than the budget, which is the full-parse-every-visit cost the gate
    exists to avoid. None (leave the previous mark) when a pending episode's
    time cannot be read: its file was collected this pass, so the previous
    mark already admits it."""
    mark = collect_started
    for ep in pending:
        ts = _aware_epoch(ep.timestamp)
        if ts is None:
            return None
        mark = min(mark, ts)
    return mark


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

    Optional trust overrides (a source may define neither, either, or both):
    ``fact_kind`` forces the FactKind for derived facts (e.g. REVIEW for
    lower-trust, decaying material that synthesis later promotes) instead of
    the default PATTERN/FAILURE inferred per-lesson; ``extra_tags`` are appended
    to the standard transcript tag so the source is identifiable and dedup-able.
    When unset, ``admit`` keeps today's behavior. See ``GitHubPRSource``.
    """

    name: str
    scope: FactScope
    # collect_episodes(since: Optional[float] = None) -> list[Episode]
    # collect_episodes may optionally accept `since` (epoch seconds) to skip
    # inputs untouched since then. Sources that omit it are called without it.
    # Optional; read via getattr in the ingester so existing sources that don't
    # define them keep the default PATTERN/FAILURE + transcript-tag behavior.
    fact_kind: Optional["FactKind"]
    extra_tags: Optional[list[str]]

    def collect_episodes(self) -> list[Episode]:
        ...


class ClaudeCodeSource:
    """Claude Code session transcripts for one project."""

    name = "claude-code"
    scope = FactScope.PROJECT

    def __init__(self, codebase_root: Optional[str]):
        self.codebase_root = codebase_root

    def collect_episodes(self, since: Optional[float] = None) -> list[Episode]:
        return collect_episodes(self.codebase_root, since=since)


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

    def collect_episodes(self, since: Optional[float] = None) -> list[Episode]:
        # `since` accepted for a uniform source Protocol; this corpus is ~1 MB,
        # so skipping buys nothing and the filter would only add a failure mode.
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

    def __init__(self, codebase_root: Optional[str], sessions_dir: Optional[Path] = None,
                 peer_roots: Optional[list[str]] = None):
        self.codebase_root = codebase_root
        self._dir = sessions_dir or CODEX_SESSIONS_DIR
        # Only peers *nested under* our root can out-specify us for a given cwd,
        # so narrow once here rather than re-scanning every peer per rollout.
        # Built through the same containment primitive the ownership test uses,
        # so there is one rule for "is X under Y" rather than two that can drift.
        self._nested_peers: list[str] = []
        if codebase_root and peer_roots:
            self._nested_peers = [
                p for p in peer_roots
                if Path(p) != Path(codebase_root)
                and self._cwd_within_root(p, codebase_root)
            ]

    def collect_episodes(self, since: Optional[float] = None) -> list[Episode]:
        root = self.codebase_root
        if not root or not self._dir.is_dir():
            return []
        episodes: list[Episode] = []
        for fp in sorted(self._dir.glob("**/rollout-*.jsonl")):
            # mtime check first: it is a stat, while _owns_rollout opens the
            # file. This source measured 224 MB for a single project's collect.
            if _unchanged_since(fp, since):
                continue
            if not self._owns_rollout(fp, root):
                continue
            try:
                if fp.stat().st_size > _CODEX_MAX_ROLLOUT_BYTES:
                    logger.warning("transcript: skipping oversized codex rollout %s", fp.name)
                    continue
            except OSError:
                continue
            episodes.extend(self._rollout_to_episodes(fp))
        return episodes

    def _owns_rollout(self, fp: Path, root: str) -> bool:
        """Does ``root`` own this rollout? Most-specific root wins.

        A rollout belongs to the *deepest* known project root containing its
        cwd. Without that, a container root (``~/git``, or worse ``/``) claims
        every nested project's sessions too — re-ingesting the machine's whole
        Codex history under a junk scope on every observer cycle.
        """
        cwd = self._rollout_cwd(fp)
        if cwd is None or not self._cwd_within_root(cwd, root):
            return False
        return not any(self._cwd_within_root(cwd, p) for p in self._nested_peers)

    @staticmethod
    def _rollout_cwd(fp: Path) -> Optional[str]:
        """The session's recorded cwd, or None if it can't be determined.

        Reads only the first record (``session_meta``) — a rollout is often
        megabytes and this runs per file per sweep. Returns None for an
        unreadable/malformed file, a first record that isn't session_meta, and
        a session_meta carrying no cwd: all three mean "unattributable", and
        collapsing them keeps callers from reasoning about a third state.
        """
        try:
            with fp.open(encoding="utf-8") as f:
                first = f.readline()
            r = json.loads(first)
        except (OSError, json.JSONDecodeError):
            return None
        if r.get("type") != "session_meta":
            return None
        return (r.get("payload") or {}).get("cwd") or None

    @staticmethod
    def _cwd_within_root(cwd: str, root: str) -> bool:
        """Component-aware containment.

        ``Path.is_relative_to`` rather than a string prefix: it makes the
        ``"/".rstrip("/") == ""`` empty-base footgun structurally impossible,
        handles trailing slashes, and refuses the false-prefix sibling case
        (``/work/github/x`` is NOT inside ``/work/git``) by construction.
        """
        try:
            return Path(cwd).is_relative_to(Path(root))
        except (ValueError, OSError):
            return False

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


# ---------------------------------------------------------------------------
# GitHub PR source helpers (task #2): derive owner/repo from the repo's git
# remote and fetch merged PRs + their review threads via the `gh` CLI. Every
# failure path returns empty/None so a missing/unauthenticated gh, a non-GitHub
# remote, an offline box, or a rate-limit never raises into the ingest loop.
# ---------------------------------------------------------------------------

# One GraphQL call pulls this many most-recently-updated merged PRs. This is a
# fixed window, NOT paginated: PRs older than the window at first run are not
# backfilled. Forward coverage still holds — a newly-merged PR surfaces at the
# top (merge bumps updatedAt) and is mined before it falls out, as long as the
# fetch cadence outpaces the merge rate.
_GH_PR_PAGE = 25
_GH_API_TIMEOUT = 20  # seconds for a single gh invocation
# PRs change far slower than transcripts. Fetch at most once per this interval
# per repo so the observer's all-projects sweep keeps its "unchanged projects do
# near-zero work" property instead of firing a gh subprocess every cycle.
_GH_PR_FETCH_INTERVAL = 3600  # seconds

# One GraphQL query gets each merged PR with its review summaries, conversation
# comments, and inline review-thread comments — bounded fan-out (1 call/repo)
# instead of REST's 1 + 3N.
_GH_PR_QUERY = """
query($owner:String!, $repo:String!, $n:Int!) {
  repository(owner:$owner, name:$repo) {
    pullRequests(states:MERGED, first:$n, orderBy:{field:UPDATED_AT, direction:DESC}) {
      nodes {
        number title body mergedAt updatedAt
        author { login }
        reviews(first:50) { nodes { body state author { login } } }
        comments(first:50) { nodes { body author { login } } }
        reviewThreads(first:50) { nodes {
          isResolved
          comments(first:20) { nodes { body path author { login } } }
        } }
      }
    }
  }
}
"""


def _owner_repo_from_remote(codebase_root: Optional[str]) -> Optional[tuple[str, str]]:
    """``(owner, repo)`` for a github.com remote, or None.

    Reuses ``scope._normalize_remote_url`` (the same normalization that derives
    ``project_id``), so PR facts co-scope with that repo's transcript facts.
    GitHub Enterprise hosts (``github.<company>.com``) are intentionally not
    matched yet — only public github.com.
    """
    norm = _normalize_remote_url(_get_git_remote_url(codebase_root))
    m = re.match(r"github\.com/([^/]+)/([^/]+)$", norm)
    if not m:
        return None
    return m.group(1), m.group(2)


def _gh_available() -> bool:
    """True if the ``gh`` CLI is on PATH. Auth/network failures are handled at
    call time (a fetch just returns []), so we don't pay a `gh auth status`
    subprocess on every cycle."""
    return shutil.which("gh") is not None


def _gh_graphql(query: str, str_vars: Optional[dict] = None,
                int_vars: Optional[dict] = None) -> Optional[dict]:
    """Run a ``gh api graphql`` query, returning parsed ``data`` or None.

    None on any failure — gh absent, not authenticated, offline, rate-limited,
    timeout, or malformed JSON — so callers degrade to an empty source.

    ``str_vars`` go via ``-f`` (raw string); ``int_vars`` via ``-F`` (typed).
    The split matters: ``-F`` coerces values (a bare integer becomes an Int, a
    leading ``@`` reads a file), so a String! variable like a repo literally
    named ``2048`` MUST use ``-f`` or it fails the schema and silently mines
    nothing.
    """
    cmd = ["gh", "api", "graphql", "-f", f"query={query}"]
    for k, v in (str_vars or {}).items():
        cmd += ["-f", f"{k}={v}"]
    for k, v in (int_vars or {}).items():
        cmd += ["-F", f"{k}={v}"]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_GH_API_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("github-pr: gh graphql failed: %s", e)
        return None
    if result.returncode != 0:
        logger.debug("github-pr: gh graphql rc=%s: %s",
                     result.returncode, (result.stderr or "")[:200])
        return None
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    return data if isinstance(data, dict) else None


def _fetch_merged_prs(owner: str, repo: str, limit: int = _GH_PR_PAGE) -> list[dict]:
    """The ``limit`` most-recently-updated merged PRs for ``owner/repo`` with
    their reviews and comments, or [] on any failure."""
    data = _gh_graphql(_GH_PR_QUERY,
                       str_vars={"owner": owner, "repo": repo},
                       int_vars={"n": limit})
    if not data:
        return []
    try:
        nodes = data["repository"]["pullRequests"]["nodes"]
    except (KeyError, TypeError):
        return []
    return [n for n in nodes if isinstance(n, dict)]


# Authored-by-automation noise: GitHub App accounts end in "[bot]"; the rest are
# common review/CI bots posting under plain logins. Their comments are high
# volume, low lesson density.
_PR_BOT_LOGINS = {"dependabot", "renovate", "github-actions", "codecov",
                  "coderabbitai", "sonarcloud", "netlify", "vercel"}


def _is_bot(login: str) -> bool:
    low = (login or "").lower()
    return low.endswith("[bot]") or low in _PR_BOT_LOGINS


def _node_list(obj: dict, key: str) -> list[dict]:
    """Safely pull ``obj[key]['nodes']`` as a list of dicts ([] if absent)."""
    try:
        nodes = obj[key]["nodes"]
    except (KeyError, TypeError):
        return []
    return [n for n in nodes if isinstance(n, dict)]


def _author_login(obj: dict) -> str:
    a = obj.get("author") if isinstance(obj, dict) else None
    return a.get("login", "") if isinstance(a, dict) else ""


class GitHubPRSource:
    """Merged GitHub PRs + their review threads for one repo, via the ``gh`` CLI.

    Repo-bound, so derived facts are PROJECT-scoped and co-scope with that
    repo's transcript facts (same git-remote ``project_id``). Facts enter as
    REVIEW on probation (trust-first): PR reviews are other people's opinions on
    code, topic-correlated and not neo's own validated outcomes, so they're
    low-trust, decaying material. They are NOT promoted by recurrence — synthesis
    consolidates them into another REVIEW, and only an independent git-verified
    acceptance of a suggestion neo itself surfaced ever mints a PATTERN. That is
    deliberate: promoting other people's opinions by repetition is exactly the
    false-accept vector neo guards against. ``fact_kind``/``extra_tags`` carry
    this posture to the ingester's ``admit`` (task #1).

    Each merged PR is mined once (watermark keyed on PR number); post-merge
    thread growth is not re-mined — merged-PR discussion is effectively final,
    and mine-once keeps the watermark bounded (one entry/PR, like the other
    sources) instead of accreting one per edit.
    """

    name = "github-pr"
    scope = FactScope.PROJECT
    fact_kind = FactKind.REVIEW
    extra_tags = ["imported:github-pr"]

    def __init__(self, codebase_root: Optional[str]):
        self.codebase_root = codebase_root

    def collect_episodes(self, since: Optional[float] = None) -> list[Episode]:
        # `since` accepted for a uniform source Protocol; this source is already
        # throttled to one `gh` fetch per repo per _GH_PR_FETCH_INTERVAL.
        repo = _owner_repo_from_remote(self.codebase_root)
        if repo is None or not _gh_available():
            return []  # non-GitHub remote / no gh on PATH → silent no-op
        owner, name = repo
        if not self._fetch_due(owner, name):
            return []  # throttled: a gh subprocess at most once per interval/repo
        episodes: list[Episode] = []
        for pr in _fetch_merged_prs(owner, name):
            ep = self._pr_to_episode(pr)
            if ep is not None:
                episodes.append(ep)
        # Advance the throttle only after a real fetch — if the budget skipped us
        # this cycle (collect never ran), we retry next cycle rather than waiting
        # a full interval, so a busy repo isn't starved on first mining.
        self._mark_fetched(owner, name)
        return episodes

    @staticmethod
    def _fetch_stamp_path(owner: str, repo: str) -> "Path":
        # Sanitize: owner/repo are GitHub handles (no "/"), but be defensive.
        slug = f"{owner}_{repo}".replace("/", "_")
        return SESSIONS_DIR / f"github_pr_fetch_{slug}.stamp"

    def _fetch_due(self, owner: str, repo: str) -> bool:
        try:
            age = time.time() - self._fetch_stamp_path(owner, repo).stat().st_mtime
        except OSError:
            return True  # no stamp yet → due
        return age >= _GH_PR_FETCH_INTERVAL

    def _mark_fetched(self, owner: str, repo: str) -> None:
        try:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            self._fetch_stamp_path(owner, repo).write_text(str(time.time()))
        except OSError as e:
            logger.debug("github-pr: could not write fetch stamp: %s", e)

    @staticmethod
    def _pr_to_episode(pr: dict) -> Optional[Episode]:
        try:
            number = int(pr["number"])
        except (KeyError, TypeError, ValueError):
            return None
        title = str(pr.get("title") or "").strip()
        if not title:
            return None
        body = str(pr.get("body") or "").strip()
        ask = f"PR #{number}: {title}"
        if body:
            ask = f"{ask}\n\n{body}"
        ask = ask[:_MAX_ASK_CHARS]

        texts: list[str] = []
        errors: list[str] = []
        # Review summaries, carrying the verdict state.
        for r in _node_list(pr, "reviews"):
            login = _author_login(r)
            if _is_bot(login):
                continue
            rb = str(r.get("body") or "").strip()
            state = str(r.get("state") or "").strip()
            if state == "CHANGES_REQUESTED":
                errors.append(f"changes requested by {login}: {rb}"[:300])
            if rb:
                texts.append(f"[review {state} by {login}] {rb}")
        # Conversation (issue-style) comments.
        for c in _node_list(pr, "comments"):
            login = _author_login(c)
            cb = str(c.get("body") or "").strip()
            if _is_bot(login) or not cb:
                continue
            texts.append(f"[comment by {login}] {cb}")
        # Inline review-thread comments (line-level discussion).
        for th in _node_list(pr, "reviewThreads"):
            for c in _node_list(th, "comments"):
                login = _author_login(c)
                cb = str(c.get("body") or "").strip()
                if _is_bot(login) or not cb:
                    continue
                path = str(c.get("path") or "").strip()
                loc = f" {path}" if path else ""
                texts.append(f"[inline{loc} by {login}] {cb}")

        # A merged PR with no human review discussion has nothing to teach — the
        # lesson lives in the review, not the description. Skip it (and leave it
        # un-watermarked so it ingests later if review activity appears).
        if not texts:
            return None

        # Mine-once: anchor on the PR number alone, so re-fetching a
        # already-consumed PR is a no-op and the watermark stays bounded at one
        # entry per PR (no per-edit accretion).
        anchor = f"pr-{number}"
        return Episode(
            session_id=f"pr-{number}",
            anchor_uuid=anchor,
            last_uuid=anchor,
            timestamp=str(pr.get("mergedAt") or ""),
            ask=ask,
            assistant_text=texts,
            errors=errors,
        )


class TranscriptIngester:
    """Extract verified, generalizable lessons from transcript episodes and
    admit them directly as PATTERN/FAILURE facts.

    Stage B (extract) and Stage C (verify) both call the configured LM
    adapter. Admission is gated by two hard filters: the verifier must keep
    the lesson, and its ``evidence_span`` must appear verbatim in the source
    episode (provable evidence, not claimed).
    """

    def __init__(self, store, lm_adapter, codebase_root: Optional[str] = None,
                 sources: Optional[list] = None, peer_roots: Optional[list[str]] = None):
        self._store = store
        self._lm = lm_adapter
        # Successful LM responses so far; see `ingest` for why it is counted.
        self._lm_answers = 0
        self.codebase_root = codebase_root or getattr(store, "codebase_root", None)
        # Default source set; add new tool adapters here as they land. No env
        # toggles: GitHubPRSource self-disables (returns []) when the repo has no
        # github.com remote or `gh` isn't on PATH, so it costs nothing where it
        # doesn't apply rather than needing an opt-out flag.
        # ``peer_roots`` (the observer's full sweep set) lets cwd-attributed
        # sources give a nested project's sessions to that project, not to an
        # enclosing container root. Absent it, single-project callers keep the
        # plain within-root behavior.
        if sources is not None and peer_roots:
            # Explicit sources are used verbatim, so peer_roots would be a
            # silent no-op. Say so rather than pretending it applied.
            raise ValueError(
                "peer_roots does not apply when explicit sources are supplied; "
                "construct the sources with peer_roots= directly"
            )
        self.sources = sources if sources is not None else [
            ClaudeCodeSource(self.codebase_root),
            CodexSource(self.codebase_root, peer_roots=peer_roots),
            CarSource(),
            GitHubPRSource(self.codebase_root),
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
    def admit(self, lesson: dict, ep: Episode, scope: FactScope = FactScope.PROJECT,
              fact_kind: Optional[FactKind] = None,
              extra_tags: Optional[list[str]] = None):
        # ``fact_kind`` (when a source declares one) overrides the per-lesson
        # PATTERN/FAILURE split — e.g. a PR source enters everything as REVIEW
        # (low-trust, decaying) and lets synthesis promote clusters later.
        if fact_kind is not None:
            kind = fact_kind
        else:
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
            tags=[_TRANSCRIPT_TAG, *(extra_tags or [])],
            provenance=Provenance.INFERRED,  # LM generalization, no bonus over corroborated facts
            domain=domain,  # first-class field so retrieve_relevant(domain=...) matches
        )

    def ingest_episode(self, ep: Episode, scope: FactScope = FactScope.PROJECT,
                       fact_kind: Optional[FactKind] = None,
                       extra_tags: Optional[list[str]] = None) -> int:
        """Full per-episode pipeline; returns number of facts admitted.

        ``store.add_fact`` saves to disk on each call, so an episode's facts
        are durably written before ``ingest`` advances the watermark.
        """
        if not ep.is_substantive:
            return 0
        # Every LM call happens BEFORE the first admission. If a judge call
        # raises LMUnavailable part-way, nothing has been written and the
        # episode is retried whole; admitting the lessons verified so far would
        # re-extract them on the retry in different LM wording, which
        # exact-signature dedup does not recognise as the same lesson.
        kept = [lesson for lesson in self.extract_lessons(ep) if self.verify(lesson, ep)]
        for lesson in kept:
            self.admit(lesson, ep, scope, fact_kind=fact_kind, extra_tags=extra_tags)
        return len(kept)

    # -- incremental ingest with per-source watermark --------------------
    def ingest(self, max_episodes: Optional[int] = None,
               max_seconds: Optional[float] = None,
               should_stop: Optional[Callable[[], bool]] = None,
               provider_known_good: bool = False) -> dict:
        """Mine all configured sources, advancing each source's watermark per
        episode.

        Idempotent per source: an episode's ``anchor_uuid`` is recorded as
        consumed only *after* its facts are durably written, so a re-run skips it
        and a crash mid-episode reprocesses it. An LM failure is NOT "no
        lessons", and it is charged to the episode only when the provider is
        known to be answering — it answered another call in this pass, or the
        caller says so via ``provider_known_good``. Two failures with no answer
        stop the pass with nothing charged (``stats["lm_unavailable"]``, which
        the observer uses to stop its sweep). Each kind has its own per-episode
        cap (``MAX_EPISODE_LM_FAILURES``,
        ``MAX_EPISODE_TRANSIENT_FAILURES``) so neither a poison episode nor an
        always-timing-out one can wedge its source. A source's drain mark — what
        lets unchanged transcripts be skipped — never passes the earliest
        episode still unconsumed (``_drain_mark``).
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
                 "episodes_processed": 0, "facts_admitted": 0,
                 "episodes_failed": 0, "episodes_abandoned": 0,
                 "lm_unavailable": False}
        start = time.monotonic()

        def stop_now() -> bool:
            if max_episodes is not None and stats["episodes_processed"] >= max_episodes:
                return True
            if max_seconds is not None and (time.monotonic() - start) >= max_seconds:
                return True
            return bool(should_stop is not None and should_stop())

        all_episodes: list[Episode] = []
        outage = False
        # Failures not yet attributable to their episode (see the except).
        held: list[tuple] = []
        unanswered = 0
        answers_at_start = self._lm_answers
        for source in self.sources:
            if outage or stop_now():
                break
            state = self._load_watermark(source)
            consumed = state["consumed"]
            root_key = _source_root_key(source)
            # Taken BEFORE collecting, so a transcript written while this pass
            # runs is newer than the drain mark and is read next time.
            collect_started = time.time()
            try:
                episodes = self._collect(source)
            except Exception as e:  # one bad source must not sink the others
                logger.warning("transcript: source %s collect failed: %s", source.name, e)
                continue
            all_episodes.extend(episodes)
            new = [e for e in episodes if e.anchor_uuid not in consumed]
            stats["episodes_total"] += len(episodes)
            stats["episodes_new"] += len(new)
            # Optional per-source trust overrides (default None → today's behavior).
            src_kind = getattr(source, "fact_kind", None)
            src_tags = getattr(source, "extra_tags", None)
            for ep in new:
                if stop_now():
                    break
                try:
                    stats["facts_admitted"] += self.ingest_episode(
                        ep, source.scope, fact_kind=src_kind, extra_tags=src_tags)
                except LMUnavailable as e:
                    stats["episodes_processed"] += 1  # it spent LM calls
                    stats["episodes_failed"] += 1
                    if self._lm_answers > answers_at_start:
                        # The provider has answered in this pass, so the
                        # failure is this episode's — and so is any held one.
                        unanswered = 0
                        held.append((source, state, ep, e))
                        if not all([self._charge(*h, stats) for h in held]):
                            outage = True  # watermark unwritable
                            break
                        held.clear()
                        continue
                    unanswered += 1
                    if unanswered >= 2:
                        # Two failures and not one answer: the provider is not
                        # serving us, WHATEVER the error class says — a 401,
                        # 403 or quota 429 fails every episode alike, and
                        # charging them would write off the backlog one cap at
                        # a time. Nothing further is charged.
                        logger.warning(
                            "transcript: LM unavailable (%s); stopping the pass", e)
                        stats["lm_unavailable"] = True
                        outage = True
                        break
                    if provider_known_good:
                        # The caller saw this provider answer recently (an
                        # earlier project this cycle), so a lone failure is the
                        # episode's. Without this a poison episode that is the
                        # only one left in its project would never be charged
                        # and would be retried on every visit forever.
                        if not self._charge(source, state, ep, e, stats):
                            outage = True
                            break
                    else:
                        # One failure proves nothing: an unreachable provider,
                        # a revoked key and a bad episode look the same. Hold
                        # it and try one more episode.
                        held.append((source, state, ep, e))
                    continue
                if self._lm_answers > answers_at_start:
                    # An answer: the provider works, so any held failure was
                    # that episode's after all. (A non-substantive episode
                    # "succeeds" with no LM call and proves nothing — hence
                    # the answer count, not success.)
                    unanswered = 0
                    for h in held:
                        self._charge(*h, stats)
                    held.clear()
                consumed.add(ep.anchor_uuid)
                state["failures"].pop(ep.anchor_uuid, None)
                state["transient_failures"].pop(ep.anchor_uuid, None)
                stats["episodes_processed"] += 1
                if not self._persist_watermark(source, state):  # advance only after durable
                    outage = True
                    break
            mark = _drain_mark(collect_started,
                               [e for e in new if e.anchor_uuid not in consumed])
            if mark is not None:
                state["drained_at"][root_key] = mark
                self._persist_watermark(source, state)

        # Correlate neo's own past suggestions (durable ledger) against all
        # collected episodes to derive real accept/modify outcomes. Best-effort:
        # a failure here must not sink the lesson-ingest stats above.
        try:
            stats["outcomes_mined"] = self.mine_suggestion_outcomes(all_episodes)
        except Exception as e:
            logger.warning("transcript: suggestion-outcome mining failed: %s", e)
            stats["outcomes_mined"] = 0

        stats["lm_answers"] = self._lm_answers - answers_at_start
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
        """Record when a past suggestion's topic recurs in later work.

        The durable-ledger, semantically-correlated complement to
        OutcomeTracker._detect_non_git_outcomes: a suggestion whose description
        matches a *subsequent* transcript episode is evidence the suggestion's
        area recurred in later work, not acceptance or correctness. A match is
        therefore observation-only (see store.apply_mined_outcomes): it never
        changes confidence, success, effectiveness, or probation state. The
        episode does not contain neo's suggestion text, so this is topic
        recurrence — deliberately not classified as accept-vs-modify (a matched
        episode's tool errors are its own process noise, unrelated to the
        suggestion's fate). Entries that find no match before their correlation
        window lapses are dropped so the ledger stays bounded. Returns the number
        of attributed topic-recurrence observations.
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

        # Compact BEFORE recording, and unconditionally. A crash may lose a
        # low-authority recurrence metric, but can never corrupt durable fact
        # confidence or success state.
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

    #: Subtracted from the drain mark before using it as a skip threshold. A transcript written *while* the previous ingest was running
    #: could otherwise be judged "already seen" and skipped forever. One hour is
    #: far longer than any observed ingest, and over-parsing is free where
    #: under-parsing silently loses learning.
    _COLLECT_SKEW_SECONDS = 3600.0

    def _collect(self, source) -> list:
        """Call a source's collector, passing ``since`` only if it accepts it.

        The parameter was added later, and sources are an open interface — the
        issues diagnostic builds its own, and third parties may too. Calling
        with an unexpected kwarg would raise TypeError, which the caller's
        per-source guard swallows as "source failed", silently returning zero
        episodes. Detect support instead of relying on that.
        """
        import inspect

        try:
            accepts = "since" in inspect.signature(source.collect_episodes).parameters
        except (TypeError, ValueError):  # builtins / C callables
            accepts = False
        if accepts:
            return source.collect_episodes(since=self._collected_through(source))
        return source.collect_episodes()

    def _collected_through(self, source) -> Optional[float]:
        """Epoch time before which this source's inputs are fully ingested.

        The source's drain mark (see ``_drain_mark``), less the skew margin. It
        is deliberately not the watermark file's mtime, which is what it used to
        be: that file is rewritten after every consumed episode, including by a
        pass that stops on its episode budget with a backlog left in the same
        transcript — and the next pass skipped that transcript as "untouched
        since the watermark", stranding the backlog for good. Measured live: 152
        of 1,800 episodes unconsumed, every one in a file the gate had stopped
        reading.

        Returns None — parse everything — when this source's directory has no
        mark yet, which includes every watermark written before marks existed.
        That first full read is also what recovers episodes already stranded.
        """
        mark = self._load_watermark(source)["drained_at"].get(_source_root_key(source))
        if not isinstance(mark, (int, float)):
            return None
        return float(mark) - self._COLLECT_SKEW_SECONDS

    def _load_watermark(self, source) -> dict:
        """``consumed`` (set), ``failures`` / ``transient_failures``
        ({anchor: attempts}) and ``drained_at`` ({root key: epoch}).

        Tolerates the legacy ``{"consumed": [...]}`` shape and a corrupt file
        (treated as empty, as before)."""
        state: dict = {"consumed": set(), "failures": {}, "transient_failures": {},
                       "drained_at": {}}
        path = self._watermark_path(source)
        if not path or not path.exists():
            return state
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return state
        if not isinstance(raw, dict):
            return state
        state["consumed"] = set(raw.get("consumed", []))
        for key in ("failures", "transient_failures"):
            book = raw.get(key)
            if isinstance(book, dict):
                state[key] = {str(k): v for k, v in book.items() if isinstance(v, int)}
        marks = raw.get("drained_at")
        if isinstance(marks, dict):
            state["drained_at"] = {str(k): v for k, v in marks.items()
                                   if isinstance(v, (int, float))}
        return state

    def _load_consumed(self, source) -> set:
        return self._load_watermark(source)["consumed"]

    def _load_failures(self, source) -> dict:
        return self._load_watermark(source)["failures"]

    def _charge(self, source, state: dict, ep: Episode, error: LMUnavailable,
                stats: dict) -> bool:
        """Charge a failure to ``ep`` and persist. False when the watermark
        could not be written."""
        if self._record_lm_failure(source, state, ep, error):
            stats["episodes_abandoned"] += 1
        return self._persist_watermark(source, state)

    def _record_lm_failure(self, source, state: dict, ep: Episode,
                           error: LMUnavailable) -> bool:
        """Count a failed attempt at ``ep``; give it up at its cap.

        Returns True when the episode was abandoned (marked consumed unmined).
        The two caps are counted separately so an outage's attempts never
        spend an episode's small non-transient budget. ``failures`` entries for
        anchors whose transcripts are later deleted are never pruned: a
        throttled source (the PR source returns nothing between fetches) makes
        "not collected this pass" mean nothing, and the growth is one int per
        failing episode, next to a ``consumed`` set that grows the same way."""
        book_key, cap = (("transient_failures", MAX_EPISODE_TRANSIENT_FAILURES)
                         if error.transient
                         else ("failures", MAX_EPISODE_LM_FAILURES))
        book = state[book_key]
        attempts = book.get(ep.anchor_uuid, 0) + 1
        if attempts < cap:
            book[ep.anchor_uuid] = attempts
            logger.warning(
                "transcript: LM call failed for %s episode %s (%s attempt %d of %d): %s",
                source.name, ep.anchor_uuid,
                "transient" if error.transient else "non-transient",
                attempts, cap, error)
            return False
        logger.warning(
            "transcript: giving up on %s episode %s after %d %s LM failures: %s",
            source.name, ep.anchor_uuid, attempts,
            "transient" if error.transient else "non-transient", error)
        state["failures"].pop(ep.anchor_uuid, None)
        state["transient_failures"].pop(ep.anchor_uuid, None)
        state["consumed"].add(ep.anchor_uuid)
        return True

    def _persist_watermark(self, source, state: dict) -> bool:
        """Write the watermark atomically. False when it could not be written,
        so the caller stops rather than mining episodes whose consumption it
        cannot record (each would be re-mined as a duplicate next pass)."""
        path = self._watermark_path(source)
        if not path:
            return True
        payload: dict = {"consumed": sorted(state["consumed"])}
        for key in ("failures", "transient_failures", "drained_at"):
            if state.get(key):
                payload[key] = dict(sorted(state[key].items()))
        try:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            atomic_write_json(path, payload)
        except OSError as e:
            logger.warning("transcript: failed to persist watermark: %s", e)
            return False
        return True

    def _lm_json(self, prompt: str) -> Optional[dict]:
        try:
            out = self._lm.generate(
                messages=[{"role": "user", "content": prompt}],
                # A reasoning model spends this budget on reasoning tokens
                # BEFORE it emits a visible character, so 1024 — which predates
                # them — left no room for the answer: a live observer log shows
                # `incomplete_details.reason = max_output_tokens` with 850 of
                # 1024 spent reasoning and the lessons array cut mid-object, so
                # the whole extraction was discarded. 4096 is the adapters' own
                # default; this call site was the outlier, not the target.
                max_tokens=4096,
                temperature=0.2,
            )
        except Exception as e:
            # Not logged here: the caller knows which episode this was.
            raise LMUnavailable(e) from e
        # The provider answered, even if what it said is unusable: this is the
        # evidence `ingest` needs before blaming an episode for a failure.
        self._lm_answers += 1
        data = _parse_json(out)
        if data is None:
            # Truncated or non-JSON output is not "no lessons" either: consumed
            # on the first pass, the episode would never be mined. Counted
            # against the non-transient cap, since a prompt that yields
            # garbage tends to yield it again.
            raise LMUnavailable(ValueError(
                f"LM output is not a JSON object ({len(out or '')} chars)"))
        return data
