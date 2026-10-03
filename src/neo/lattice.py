"""Neo as a player on the CAR Lattice.

The Lattice is CAR's collaboration layer: Claude Code, Codex, CAR Coder and
humans on one daemon declare what they are good at (``players.join``), find
each other by need (``players.find``), coordinate with expiring work claims
(``work.claim``), and talk with typed peer messages (``agents.message`` with a
``kind`` of ``question``, ``answer``, ``review_request`` or ``handoff``).

Neo joins as a **read-only** player. What it brings that nobody else on the
Lattice has:

- **Project memory across runtimes.** The observer mines Claude Code, Codex,
  CAR and GitHub-PR history into one fact store, so a Codex session can ask
  what was learned about a subsystem in a Claude Code session last week.
  ``memory:`` lookups are answered from retrieval alone, with no model call.
- **Reasoning on request.** A ``question`` runs the normal Neo pipeline in the
  node's repository: file selection, retrieved facts, plan, suggested diffs.
- **Review.** A ``review_request`` carrying a diff gets the deterministic
  VERIFY checks (no model call) and then a semantic review in ADVISE mode.
- **It is always listening.** A supervised CAR agent receives
  ``agent.peer_message`` as a push, unlike an MCP coding session, which only
  sees its mail when it next polls. Neo answers while the asker keeps working.

What Neo deliberately does not do:

- **Claim or take work.** It writes no files, so it never ``work.claim``\\ s
  and declines ``handoff`` with a pointer to ``players.find``.
- **Route.** Matching needs to players is the daemon's deterministic job;
  intelligence stays at the nodes.
- **Answer an answer.** Two auto-responders replying to each other form a loop
  no policy rule catches, so ``kind: answer`` is acknowledged and dropped.
- **Learn from a peer's patch as if it were its own.** Reviews run in ADVISE
  mode, which records no learning episode; questions run in the default LEARN
  mode, so Neo's own suggestions are verified against git exactly as CLI ones
  are. A peer saying "thanks" is not an acceptance.

One node serves one repository, as one supervised agent
(``neo-lattice-<project_id12>``), because a player has exactly one project and
Neo's memory is scoped by repository. When neo and CAR are installed together
this is automatic: every neo CLI run in a git repository registers that
repository's node (``maybe_autojoin``), the same way the observer autostarts.
``neo lattice leave`` takes a repository off and records the choice so
autostart does not put it back; ``neo lattice join`` reverses that. An idle
node is ~35 MB; one that has answered re-execs itself after
``IDLE_RECYCLE_SECS`` idle to drop the loaded engine.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from neo.text_budget import truncate_marked

logger = logging.getLogger(__name__)

AGENT_PREFIX = "neo-lattice-"
RUNTIME = "neo"

# Tags a `players.find` need is matched against. The daemon scores a tag only
# when EVERY word in it appears in the need, so multi-word tags are precise and
# single words are broad; both shapes are here on purpose.
BASE_CAPABILITIES: tuple[str, ...] = (
    "code-reasoning",
    "code-review",
    "patch-verification",
    "architecture",
    "debugging",
    "root-cause-analysis",
    "performance-optimization",
    "refactoring",
    "design-patterns",
    "project-memory",
    "past-solutions",
    "codebase-questions",
)
# At most this many repository languages are appended to the base tags.
MAX_LANGUAGE_TAGS = 4
# A language needs this many tracked files to be advertised: one stray script
# does not make Neo a player for that language.
MIN_LANGUAGE_FILES = 5
# Requests waiting behind the one being answered. Past this a sender is told
# Neo is busy rather than being queued indefinitely.
MAX_QUEUE = 8
# Characters in one answer. The daemon's cap is ~1 MB; an answer a coding
# session has to read back into its context should be far smaller.
ANSWER_BUDGET = 12_000
MEMORY_RESULTS = 8
MEMORY_BODY_BUDGET = 700
# A node that has answered holds a loaded engine (fact store, embedding model,
# several hundred MB) that CPython does not return to the OS. After this long
# with nothing to do, the node re-execs itself back to its ~35 MB idle size.
IDLE_RECYCLE_SECS = 600.0
QUESTION_OUTPUT = (
    "answer for another agent: answer the question directly in prose, citing "
    "files and lines; propose a code change only if the question asks for one"
)
RECONNECT_MIN_SECS = 1.0
RECONNECT_MAX_SECS = 30.0
# A peer name is 1-128 of [A-Za-z0-9._-], not starting with `.` or `-`.
_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

USAGE = (
    "Neo is a read-only reasoning player. Send me:\n"
    "- kind=question: a plain-language question about this repository.\n"
    "- kind=question, body starting `memory:`: a lookup in Neo's project "
    "memory (no model call).\n"
    "- kind=review_request: a unified diff, optionally with a note; I run "
    "deterministic checks and then review it.\n"
    "A JSON body {\"op\": \"reason\"|\"memory\"|\"review\", \"prompt\": ..., "
    "\"diff\": ..., \"error_trace\": ...} is also accepted."
)


# ---------------------------------------------------------------------------
# Identity and capabilities
# ---------------------------------------------------------------------------


def sanitize_name(raw: str, fallback: str = "project") -> str:
    """Coerce `raw` into a valid player name ([A-Za-z0-9._-], 1-128 chars)."""
    name = _NAME_RE.sub("-", raw.strip()).lstrip(".-")[:128].rstrip("-")
    return name or fallback


def derive_project_name(root: str) -> str:
    """The player project for `root`: the git remote's repo name, else the dir.

    Other players name a project the way a person would ("car", "neo"), and
    two clones of one repo must join the SAME project, so the remote's repo
    name wins over a local directory name like `car2`.
    """
    from neo.memory.scope import _get_git_remote_url

    remote = ""
    try:
        remote = _get_git_remote_url(root) or ""
    except Exception:  # noqa: BLE001 — identity falls back to the dir name
        remote = ""
    if remote:
        tail = remote.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1]
        if tail.endswith(".git"):
            tail = tail[:-4]
        if tail:
            return sanitize_name(tail)
    return sanitize_name(Path(root).resolve().name)


def repository_languages(root: str) -> list[str]:
    """The repository's dominant languages, most files first.

    Counted over `git ls-files`, so ignored and vendored trees do not count.
    Not a git repo, or git unavailable: no language tags, never an error.
    """
    from neo.languages import language_for_path, normalize_language_name

    try:
        out = subprocess.run(
            ["git", "-C", root, "ls-files"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    counts: Counter[str] = Counter()
    for line in out.splitlines():
        lang = language_for_path(line)
        if lang:
            counts[normalize_language_name(lang)] += 1
    return [
        lang for lang, n in counts.most_common(MAX_LANGUAGE_TAGS)
        if n >= MIN_LANGUAGE_FILES and lang
    ]


_TAG_RE = re.compile(r"[^a-z0-9 \-_.+#/]+")


def capabilities_for(root: str) -> list[str]:
    tags = list(BASE_CAPABILITIES)
    for lang in repository_languages(root):
        # The daemon's tag charset (letters, digits, space, `-_.+#/`), which
        # keeps `c++` and `c#` intact where a name sanitizer would not.
        tag = _TAG_RE.sub("", lang.lower()).strip()[:64]
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def agent_id_for(root: str) -> str:
    from neo.memory.scope import _compute_project_id

    return f"{AGENT_PREFIX}{_compute_project_id(root)[:12]}"


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


@dataclass
class PeerRequest:
    """One `agent.peer_message` push, as the daemon delivered it."""

    message_id: str
    sender: str
    body: str
    kind: str = ""
    no_reply: bool = False
    in_reply_to: str = ""

    @classmethod
    def from_params(cls, params: dict) -> "PeerRequest":
        return cls(
            message_id=str(params.get("id") or ""),
            sender=str(params.get("from") or ""),
            body=str(params.get("body") or ""),
            # The daemon sends `message` for an untyped message; an older
            # daemon omits the field. Both mean the same thing here.
            kind=_normalize_kind(params.get("kind")),
            no_reply=bool(params.get("no_reply")),
            in_reply_to=str(params.get("in_reply_to") or ""),
        )


def _normalize_kind(raw: Any) -> str:
    kind = str(raw or "").strip().lower()
    return "" if kind == "message" else kind


@dataclass
class Task:
    """What Neo will do about a request."""

    op: str  # reason | review | memory | decline | ignore
    prompt: str = ""
    diff: str = ""
    error_trace: Optional[str] = None
    reason: str = ""  # why a request was declined or ignored


_DIFF_START_RE = re.compile(r"^(diff --git |--- \S)", re.M)


def looks_like_diff(text: str) -> bool:
    return bool(re.search(r"^--- \S.*\n\+\+\+ \S", text, re.M)) or (
        "diff --git " in text
    )


def plan_task(req: PeerRequest) -> Task:
    """Decide what to do with a request. Pure: no I/O, no model."""
    if req.kind == "answer":
        # Never answer an answer: two auto-responders would loop forever.
        return Task(op="ignore", reason="answers are not answered")
    if req.kind == "handoff":
        return Task(op="decline", reason=(
            "Neo is read-only and does not take work. Use players.find with "
            "the capability you need to reach a player that writes code. "
            "Ask me a question or send a review_request instead.\n\n" + USAGE
        ))

    body = req.body.strip()
    op = "review" if req.kind == "review_request" else "reason"
    prompt, diff, error_trace = body, "", None

    data = _json_object(body)
    if data is not None:
        requested = str(data.get("op") or "").strip().lower()
        if requested in {"reason", "memory", "review"}:
            op = requested
        prompt = str(
            data.get("prompt") or data.get("query") or data.get("question") or ""
        ).strip()
        diff = str(data.get("diff") or "")
        trace = data.get("error_trace")
        error_trace = str(trace) if trace else None
    else:
        lowered = body.lower()
        for prefix in ("memory:", "/memory"):
            if lowered.startswith(prefix):
                op, prompt = "memory", body[len(prefix):].strip()
                break
        if op == "review":
            match = _DIFF_START_RE.search(body)
            if match and looks_like_diff(body[match.start():]):
                prompt, diff = body[:match.start()].strip(), body[match.start():]

    if op == "review" and not diff and looks_like_diff(prompt):
        diff, prompt = prompt, ""
    if not prompt and not diff:
        return Task(op="decline", reason="Empty request.\n\n" + USAGE)
    return Task(op=op, prompt=prompt, diff=diff, error_trace=error_trace)


def _json_object(text: str) -> Optional[dict]:
    if not text.startswith("{"):
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def split_unified_diff(diff: str) -> list[tuple[str, str]]:
    """Split a multi-file unified diff into `(file_path, file_diff)` pairs.

    The path comes from the `+++` header (`b/` stripped), or from `---` when
    the file is deleted. Timestamps after a tab are dropped. Text that has no
    file header yields nothing — VERIFY needs a path to check against.
    """
    lines = diff.splitlines(keepends=True)
    starts: list[int] = []
    for i, line in enumerate(lines):
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            start = i - 1 if i > 0 and lines[i - 1].startswith("diff --git ") else i
            starts.append(start)
    pairs: list[tuple[str, str]] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        chunk = lines[start:end]
        minus = next(line for line in chunk if line.startswith("--- "))
        plus = next(line for line in chunk if line.startswith("+++ "))
        path = _header_path(plus)
        if path == "/dev/null":
            path = _header_path(minus)
        if path and path != "/dev/null":
            pairs.append((path, "".join(chunk)))
    return pairs


def _header_path(line: str) -> str:
    path = line[4:].split("\t", 1)[0].strip()
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------


class Answerer:
    """Turns a planned task into answer text, against one repository.

    Synchronous and single-threaded: `NeoEngine` handles one request at a time
    and the node feeds it from a single worker.
    """

    def __init__(self, root: str, engine_factory: Optional[Callable[[], Any]] = None,
                 store_factory: Optional[Callable[[], Any]] = None):
        self.root = root
        self._engine_factory = engine_factory or self._default_engine
        self._store_factory = store_factory or self._default_store
        self._engine: Any = None
        self._store: Any = None

    def _default_engine(self) -> Any:
        from neo.adapters import resolve_adapter
        from neo.config import NeoConfig
        from neo.engine import NeoEngine

        config = NeoConfig.load()
        return NeoEngine(
            lm_adapter=resolve_adapter(config),
            codebase_root=self.root,
            config=config,
        )

    @property
    def engine_loaded(self) -> bool:
        return self._engine is not None

    @property
    def engine(self) -> Any:
        if self._engine is None:
            self._engine = self._engine_factory()
        return self._engine

    def answer(self, task: Task) -> str:
        if task.op == "memory":
            return self.memory(task.prompt)
        if task.op == "review":
            return self.review(task)
        return self.reason(task)

    # -- memory ---------------------------------------------------------------

    def _memory_store(self) -> Any:
        """The fact store, WITHOUT building an engine when none exists yet.

        A memory lookup is advertised as model-free, and building the engine
        resolves an LM adapter, which fails on a machine with no API key. Once
        the engine exists its store is used, so the two never diverge.
        """
        if self._engine is not None:
            return getattr(self._engine, "fact_store", None)
        if self._store is None:
            self._store = self._store_factory()
        return self._store

    def _default_store(self) -> Any:
        from neo.config import NeoConfig
        from neo.memory.store import FactStore

        return FactStore(codebase_root=self.root, config=NeoConfig.load())

    def memory(self, query: str) -> str:
        store = self._memory_store()
        if store is None:
            return "Neo's project memory is not available on this node."
        facts = _distinct(store.retrieve_relevant(query, k=MEMORY_RESULTS))
        if not facts:
            return f"Neo's memory has nothing on: {query}"
        lines = [
            f"Neo's memory for: {query}",
            "(Recorded lessons, not re-checked against the current tree. "
            "Confidence is Neo's, provenance is in brackets.)",
            "",
        ]
        for n, fact in enumerate(facts, 1):
            kind = getattr(fact.kind, "value", str(fact.kind))
            confidence = getattr(fact.metadata, "confidence", None)
            conf = f", confidence {confidence:.2f}" if isinstance(confidence, (int, float)) else ""
            provenance = [
                t for t in fact.tags
                if t.startswith("imported:") or t in {"transcript-derived", "community", "seed", "synthesized"}
            ]
            source = f" [{', '.join(provenance)}]" if provenance else ""
            lines.append(f"{n}. {fact.subject} ({kind}{conf}){source}")
            body = fact.body or ""
            # Commit-history facts carry the whole patch after `Changes:`;
            # the message and file list are what a reader can use.
            if "\nChanges:\n" in body:
                body = body.split("\nChanges:\n", 1)[0]
            if body:
                lines.append("   " + truncate_marked(body, MEMORY_BODY_BUDGET).replace("\n", "\n   "))
        return "\n".join(lines)

    # -- reasoning ------------------------------------------------------------

    def reason(self, task: Task) -> str:
        from neo.models import NeoInput

        output = self.engine.process(NeoInput(
            prompt=task.prompt,
            error_trace=task.error_trace,
            working_directory=self.root,
            # A peer's question wants an answer it can act on. Left at the
            # default `next_action`, an explanatory question came back as a
            # "change" whose diff created a file holding the prose answer.
            requested_output=QUESTION_OUTPUT,
        ))
        return render_output(output)

    def review(self, task: Task) -> str:
        from neo.models import NeoInput, ProposedChange
        from neo.operating_mode import OperatingMode

        sections: list[str] = []
        changes = [
            ProposedChange(file_path=path, unified_diff=chunk, description="peer change under review")
            for path, chunk in split_unified_diff(task.diff)
        ]
        if changes:
            verified = self.engine.process(NeoInput(
                prompt=task.prompt or "Verify the proposed change.",
                working_directory=self.root,
                operating_mode=OperatingMode.VERIFY,
                proposed_changes=changes,
            ))
            sections.append(render_verification(verified))
        elif task.diff:
            sections.append("Deterministic checks: skipped — no `---`/`+++` file headers found in the diff.")

        note = f"\n\nThe requester's note: {task.prompt}" if task.prompt else ""
        prompt = (
            "Review this proposed change for correctness bugs, regressions, "
            "missed edge cases and missing tests. Name concrete problems with "
            "file and line; say plainly if you find none." + note
            + ("\n\n```diff\n" + task.diff + "\n```" if task.diff else "")
        )
        reviewed = self.engine.process(NeoInput(
            prompt=prompt,
            working_directory=self.root,
            # ADVISE: the patch is the peer's, so it must not enter Neo's
            # learning loop as a Neo suggestion awaiting acceptance.
            operating_mode=OperatingMode.ADVISE,
        ))
        sections.append("Review:\n" + render_output(reviewed))
        return "\n\n".join(sections)


def _distinct(facts: list) -> list:
    """Drop facts whose subject and body repeat an earlier one.

    Seed facts are installed per scope, so one lesson can be retrieved three
    times; a reader gains nothing from the second copy and loses a slot.
    """
    seen: set[tuple[str, str]] = set()
    kept = []
    for fact in facts:
        key = (fact.subject.strip().lower(), (fact.body or "").strip())
        if key not in seen:
            seen.add(key)
            kept.append(fact)
    return kept


# The engine's self-report of retrieved facts, appended to model prose for
# citation tracking. An empty one tells a peer nothing.
_EMPTY_FACTS_USED_RE = re.compile(r"\n*\s*Facts used:\s*\[\s*\]\s*$")


def _is_answer(suggestion: Any) -> bool:
    """A suggestion with no diff is not a change; it is (part of) the answer.

    Asked a question, the model often returns its prose answer as a "change"
    to `/` or to the file it read, with the text in `code_block` and no diff.
    Rendered as a change it reads "I'd change 1 thing in /", which is false.
    """
    return not suggestion.unified_diff and bool(suggestion.code_block)


def render_verification(output: Any) -> str:
    checks = list(getattr(output, "static_checks", []) or [])
    if not checks:
        return "Deterministic checks: none applied to these files."
    lines = ["Deterministic checks (no model call):"]
    for check in checks:
        status = check.status or "reported"
        summary = f" — {check.summary}" if check.summary else ""
        lines.append(f"- {check.tool_name}: {status}{summary}")
    return "\n".join(lines)


def render_output(output: Any) -> str:
    """Plain-text rendering of a NeoOutput for a peer to read.

    An answer leads with the answer. Cautions are always kept: a caution
    exists so a confident-sounding answer cannot bury it.
    """
    orch = getattr(output, "orchestrator", None)
    parts: list[str] = []
    suggestions = getattr(output, "code_suggestions", []) or []
    answers = [s for s in suggestions if _is_answer(s)]
    changes = [s for s in suggestions if not _is_answer(s)]
    for a in answers:
        text = _EMPTY_FACTS_USED_RE.sub("", a.code_block.strip())
        path = (a.file_path or "").strip()
        if path.strip("/."):
            text = f"(re {path})\n{text}"
        parts.append(text)
    summary = getattr(orch, "summary", "") if orch else ""
    # The summary counts "changes"; with only an answer it would misdescribe it.
    if summary and (changes or not answers):
        parts.append(summary)
    cautions = getattr(orch, "cautions", []) if orch else []
    if cautions:
        parts.append("Cautions:\n" + "\n".join(f"- {c}" for c in cautions))
    plan = getattr(output, "plan", []) or []
    if plan and not answers:
        parts.append("Plan:\n" + "\n".join(
            f"{n}. {step.description}" for n, step in enumerate(plan, 1)
        ))
    for s in changes:
        conf = f" (confidence {s.confidence:.2f})" if isinstance(s.confidence, (int, float)) and s.confidence else ""
        block = f"### {s.file_path}{conf}\n{s.description}".rstrip()
        if s.unified_diff:
            block += f"\n```diff\n{s.unified_diff.rstrip()}\n```"
        elif s.code_block:
            block += f"\n```\n{s.code_block.rstrip()}\n```"
        parts.append(block)
    questions = getattr(output, "next_questions", []) or []
    if questions:
        parts.append("Open questions:\n" + "\n".join(f"- {q}" for q in questions))
    if not parts:
        notes = getattr(output, "notes", "") or ""
        parts.append(notes or "Neo had nothing to add.")
    return "\n\n".join(parts)


def format_reply(req: PeerRequest, text: str) -> str:
    """The body of an answer. The first line names the message it answers,
    because `agents.message` has no correlation field of its own."""
    header = f"re: {req.message_id}" if req.message_id else "re: your message"
    return truncate_marked(f"{header}\n\n{text}", ANSWER_BUDGET)


# ---------------------------------------------------------------------------
# The node: a supervised CAR agent that joins, listens and answers
# ---------------------------------------------------------------------------


@dataclass
class NodeConfig:
    root: str
    project: str
    agent_id: str
    token: str
    url: str
    display_name: str = ""
    capabilities: list[str] = field(default_factory=list)


class LatticeNode:
    """Neo's presence on the Lattice for one repository.

    Acknowledges each pushed `agent.peer_message` immediately, queues it, and
    answers from one worker so the engine sees one request at a time. Status
    reads `busy` while answering, so `players.find` routes elsewhere unless
    the asker opts into unavailable players.
    """

    def __init__(self, config: NodeConfig, answerer: Optional[Answerer] = None,
                 client_factory: Optional[Callable[[str], Any]] = None):
        self.config = config
        self.answerer = answerer or Answerer(config.root)
        self._client_factory = client_factory or _default_client
        self.client: Any = None
        self.queue: Optional[asyncio.Queue] = None
        self._stop = asyncio.Event()
        self._connected = asyncio.Event()
        self.answered = 0
        self.last_activity = time.monotonic()
        self.recycle = False
        self.working = False

    def should_recycle(self, now: Optional[float] = None) -> bool:
        """True once a loaded engine has sat idle past `IDLE_RECYCLE_SECS`."""
        if not self.answerer.engine_loaded:
            return False
        if self.working or (self.queue is not None and self.queue.qsize()):
            return False
        return ((now or time.monotonic()) - self.last_activity) >= IDLE_RECYCLE_SECS

    # -- profile --------------------------------------------------------------

    def profile(self, status: str, note: str = "") -> dict:
        params: dict[str, Any] = {
            "project": self.config.project,
            "capabilities": self.config.capabilities,
            "display_name": self.config.display_name or self.config.agent_id,
            "runtime": RUNTIME,
            "status": status,
        }
        params["note"] = (note or "Read-only: ask a question, look up project "
                          "memory, or send a review_request.")[:280]
        return params

    async def set_status(self, status: str, note: str = "") -> None:
        try:
            await self.client.call("players.join", self.profile(status, note))
        except Exception as e:  # noqa: BLE001 — status is advisory
            logger.warning("players.join (%s) failed: %s", status, e)
            await self._drop_if_superseded(e)

    async def _drop_if_superseded(self, error: Exception) -> None:
        """A superseded binding answers every call with an error while the
        socket stays open, so nothing else would ever notice. Closing it sends
        the run loop through a fresh attach and join."""
        if "superseded" in str(error) and self.client is not None:
            logger.warning("agent binding superseded; re-attaching")
            await self.client.close()

    # -- inbound --------------------------------------------------------------

    async def on_peer_message(self, params: dict) -> dict:
        """Acknowledge a push. Never blocks: answering happens in the worker."""
        req = PeerRequest.from_params(params)
        task = plan_task(req)
        if task.op == "ignore":
            return {"received": True, "action": "ignored", "reason": task.reason}
        assert self.queue is not None
        try:
            self.queue.put_nowait((req, task))
        except asyncio.QueueFull:
            asyncio.create_task(self._reply(req, (
                f"Neo is answering {MAX_QUEUE + 1} requests already. "
                "Try again in a few minutes."
            )))
            return {"received": True, "action": "busy"}
        return {"received": True, "action": "queued", "queued": self.queue.qsize()}

    # -- outbound -------------------------------------------------------------

    async def _reply(self, req: PeerRequest, text: str) -> None:
        if req.no_reply or not req.sender:
            return
        params: dict[str, Any] = {
            "to": req.sender, "kind": "answer", "body": format_reply(req, text),
            "summary": f"neo answer to {req.message_id or req.sender}",
        }
        if req.message_id:
            # The daemon's correlation field; the `re:` line in the body
            # carries the same id for readers that only see text.
            params["in_reply_to"] = req.message_id
        for attempt in range(2):
            try:
                await self.client.call("agents.message", params)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("reply to %s failed (attempt %d): %s", req.sender, attempt + 1, e)
                await self._drop_if_superseded(e)
                if self.client is not None and self.client.connected:
                    # The daemon refused it on a live connection (guard,
                    # policy, recipient gone): a retry would be refused too.
                    return
                # One retry after a reconnect, so a daemon restart mid-answer
                # does not lose the answer the user waited for.
                self._connected.clear()
                try:
                    await asyncio.wait_for(self._connected.wait(), timeout=60)
                except asyncio.TimeoutError:
                    return

    async def worker(self) -> None:
        assert self.queue is not None
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            req, task = await self.queue.get()
            self.last_activity = time.monotonic()
            self.working = True
            try:
                if task.op == "decline":
                    await self._reply(req, task.reason)
                    continue
                await self.set_status("busy", f"answering a {req.kind or 'message'} from {req.sender}")
                try:
                    text = await loop.run_in_executor(None, self.answerer.answer, task)
                except Exception as e:  # noqa: BLE001 — the asker gets told
                    logger.exception("answering %s failed", req.message_id)
                    text = f"Neo failed to answer: {type(e).__name__}: {e}"
                await self._reply(req, text)
                self.answered += 1
            finally:
                self.working = False
                self.last_activity = time.monotonic()
                self.queue.task_done()
                if self.queue.empty() and self.client is not None:
                    await self.set_status("available")

    # -- connection -----------------------------------------------------------

    async def connect_once(self) -> None:
        client = self._client_factory(self.config.url)
        client.on_request("agent.peer_message", self.on_peer_message)
        await client.connect({"token": self.config.token, "agent_id": self.config.agent_id})
        self.client = client
        try:
            from neo import __version__
            await client.call("server.handshake", {
                "protocol_version": 3, "client_version": f"neo-{__version__}",
                "required_capabilities": [], "optional_capabilities": [],
            })
        except Exception as e:  # noqa: BLE001 — players.* does not need it
            logger.info("server.handshake not negotiated: %s", e)
        await client.call("players.join", self.profile("available"))
        self._connected.set()
        logger.info("joined project %s as %s", self.config.project, self.config.agent_id)

    async def run(self) -> None:
        self.queue = asyncio.Queue(maxsize=MAX_QUEUE)
        worker = asyncio.create_task(self.worker())
        delay = RECONNECT_MIN_SECS
        try:
            while not self._stop.is_set():
                try:
                    await self.connect_once()
                    delay = RECONNECT_MIN_SECS
                    while self.client.connected and not self._stop.is_set():
                        await asyncio.sleep(1.0)
                        if self.should_recycle():
                            logger.info("idle with a loaded engine; recycling")
                            self.recycle = True
                            self._stop.set()
                    if not self._stop.is_set():
                        logger.warning("daemon connection lost; reconnecting")
                except Exception as e:  # noqa: BLE001 — reconnect with backoff
                    logger.warning("lattice connection failed: %s", e)
                self._connected.clear()
                if self.client is not None:
                    await self.client.close()
                if self._stop.is_set():
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_SECS)
        finally:
            worker.cancel()

    async def leave(self) -> None:
        self._stop.set()
        if self.client is not None and self.client.connected:
            try:
                await self.client.call("players.leave", {})
            except Exception:  # noqa: BLE001
                pass


def _default_client(url: str) -> Any:
    from neo.a2ui import DaemonClient

    return DaemonClient(url)


def daemon_url() -> str:
    from neo.a2ui import DAEMON_URL

    return os.environ.get("CAR_DAEMON_URL") or DAEMON_URL


# ---------------------------------------------------------------------------
# Lifecycle: register Neo with CAR's supervisor
# ---------------------------------------------------------------------------

# Forwarded into the supervised child; nothing else from the caller's env is.
_FORWARDED_ENV = ("CAR_DAEMON_URL", "NEO_PROFILE", "NEO_METRICS", "NEO_LOG_LEVEL")


def build_spec(root: str, project: str) -> dict:
    agent_id = agent_id_for(root)
    return {
        "id": agent_id,
        "name": f"Neo ({project})",
        "command": sys.executable,
        "args": ["-m", "neo.lattice", "run", "--cwd", root, "--project", project],
        "cwd": root,
        "env": {k: os.environ[k] for k in _FORWARDED_ENV if k in os.environ},
        "restart": "on_failure",
        "max_restarts": 10,
        "backoff_secs": 5,
        "auto_start": True,
    }


def _left_marker(root: str) -> Path:
    """Records that the user took this repo off the Lattice.

    Resolved at call time (not import time) so it follows `Path.home()`.
    """
    from neo.memory.scope import _compute_project_id

    return Path.home() / ".neo" / "lattice" / f"left-{_compute_project_id(root)[:16]}"


def repository_root(path: str) -> Optional[str]:
    """The git top-level containing `path`, or None outside a repository.

    Auto-join requires a repository: a session started from `$HOME` or `/`
    must not mint a player whose "project" is the whole machine.
    """
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    top = out.stdout.strip()
    return top if out.returncode == 0 and top else None


def maybe_autojoin(cwd: Optional[str] = None) -> Optional[str]:
    """Put this repository on the Lattice when CAR is present.

    Called once per neo CLI run, beside the observer autostart: when neo and
    CAR are installed together, every repository neo is used in gets a
    supervised, read-only neo player. No-op outside a git repository, when
    the daemon is unreachable, when the node is already registered, or when
    the user ran `neo lattice leave` here. Never raises. Returns the agent id
    it registered, else None.

    `NEO_OBSERVER_AUTOSTART=0` turns this off too: it is the one existing
    switch for "neo registers no CAR agents on its own", and a second switch
    for the second agent would let a caller silence one and not the other.
    """
    try:
        if os.getenv("NEO_OBSERVER_AUTOSTART", "").strip() == "0":
            return None
        from neo.memory.observer import (
            _car_server_reachable,
            _find_managed_agent,
            _require_car_runtime,
        )

        root = repository_root(cwd or os.getcwd())
        if root is None or _left_marker(root).exists():
            return None
        if not _car_server_reachable():
            return None
        car = _require_car_runtime()
        agent_id = agent_id_for(root)
        if _find_managed_agent(car, agent_id) is not None:
            return None  # registered; the supervisor owns its lifecycle
        spec = build_spec(root, derive_project_name(root))
        car.agents_upsert(json.dumps(spec))
        car.agents_start(agent_id)
        logger.debug("auto-joined %s to the lattice as %s", root, agent_id)
        return agent_id
    except Exception as e:  # noqa: BLE001 — must never break a neo command
        logger.debug("lattice autojoin skipped: %s", e)
        return None


def join(root: str, project: Optional[str] = None) -> dict:
    from neo.memory.observer import _find_managed_agent, _require_car_runtime

    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    try:
        _left_marker(root).unlink()
    except FileNotFoundError:
        pass
    project = project or derive_project_name(root)
    spec = build_spec(root, project)
    agent_id = spec["id"]
    try:
        car.agents_upsert(json.dumps(spec))
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"agents_upsert failed: {e}"}
    existing = _find_managed_agent(car, agent_id)
    if existing and existing.get("status") == "running":
        try:
            # Restart so a changed spec (project, capabilities) takes effect.
            managed = json.loads(car.agents_restart(agent_id))
        except Exception as e:  # noqa: BLE001
            return {"status": "error", "message": f"agents_restart failed: {e}"}
        return {"status": "restarted", "agent_id": agent_id, "project": project,
                "pid": managed.get("pid")}
    try:
        managed = json.loads(car.agents_start(agent_id))
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"agents_start failed: {e}"}
    return {"status": "started", "agent_id": agent_id, "project": project,
            "pid": managed.get("pid"),
            "log": f"~/.car/logs/{agent_id}.stderr.log"}


def leave(root: str) -> dict:
    from neo.memory.observer import _find_managed_agent, _require_car_runtime

    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    agent_id = agent_id_for(root)
    # Recorded first, so autostart honours the choice even if CAR is down.
    marker = _left_marker(root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{root}\n")
    if _find_managed_agent(car, agent_id) is None:
        return {"status": "not_joined", "agent_id": agent_id}
    for op in (car.agents_stop, car.agents_remove):
        try:
            op(agent_id)
        except Exception as e:  # noqa: BLE001 — stop fails on a stopped agent
            logger.debug("%s(%s): %s", op.__name__, agent_id, e)
    return {"status": "left", "agent_id": agent_id}


def status(root: str) -> dict:
    from neo.memory.observer import _find_managed_agent, _require_car_runtime

    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    agent_id = agent_id_for(root)
    managed = _find_managed_agent(car, agent_id)
    if managed is None:
        return {"status": "not_joined", "agent_id": agent_id}
    return {"status": managed.get("status", "unknown"), "agent_id": agent_id,
            "pid": managed.get("pid")}


def _run_foreground(root: str, project: str) -> int:
    # Taken OUT of the environment, not just read. The car_runtime bindings
    # also read these, so anything in this process that touches CAR (the
    # CarAdapter, discovery) would attach as this same agent and supersede
    # the node's own connection, silently ending its player profile mid-answer.
    # Without them the bindings connect as an ordinary client.
    agent_id = os.environ.pop("CAR_AGENT_ID", "")
    token = os.environ.pop("CAR_AGENT_TOKEN", "")
    if not agent_id or not token:
        print(
            "neo lattice run must be started by CAR's supervisor, which sets "
            "CAR_AGENT_ID and CAR_AGENT_TOKEN. Use `neo lattice join`.",
            file=sys.stderr,
        )
        return 2
    config = NodeConfig(
        root=root, project=project, agent_id=agent_id, token=token,
        url=daemon_url(), display_name=f"neo-{project}",
        capabilities=capabilities_for(root),
    )
    node = LatticeNode(config)

    async def _main() -> None:
        import signal

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.ensure_future(node.leave()))
            except (NotImplementedError, RuntimeError):
                pass
        await node.run()

    started = time.monotonic()
    asyncio.run(_main())
    logger.info("lattice node exiting after %.0fs, %d answered",
                time.monotonic() - started, node.answered)
    if node.recycle:
        # exec keeps the pid and CAR_AGENT_ID/CAR_AGENT_TOKEN, so the
        # supervisor sees the same agent; the old socket closes with the old
        # image and the new one re-attaches and re-joins.
        sys.stdout.flush()
        sys.stderr.flush()
        env = dict(os.environ, CAR_AGENT_ID=agent_id, CAR_AGENT_TOKEN=token)
        os.execve(sys.executable, [sys.executable, "-m", "neo.lattice", "run",
                                   "--cwd", root, "--project", project], env)
    return 0


def cli_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="neo lattice",
        description="Make Neo a read-only player on the CAR Lattice for this repository.",
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for name, text in (
        ("join", "register and start Neo's lattice node for this repository"),
        ("leave", "stop and unregister it, and keep autostart from re-adding it"),
        ("status", "show the node's supervisor state"),
        ("run", "the node itself (started by CAR's supervisor)"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--cwd", default=None, help="repository root (default: cwd)")
        if name in {"join", "run"}:
            p.add_argument("--project", default=None,
                           help="player project name (default: the git remote's repo name)")
    args = parser.parse_args(argv)
    root = os.path.abspath(args.cwd or os.getcwd())

    if args.action == "run":
        logging.basicConfig(
            level=os.environ.get("NEO_LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        return _run_foreground(root, args.project or derive_project_name(root))

    result = {"join": lambda: join(root, args.project), "leave": lambda: leave(root),
              "status": lambda: status(root)}[args.action]()
    print(json.dumps(result, indent=2))
    return 1 if result.get("status") == "error" else 0


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
