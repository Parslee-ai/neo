"""Neo as a player on the CAR Lattice.

The Lattice is CAR's collaboration layer: Claude Code, Codex, CAR Coder and
humans on one daemon declare what they are good at (``lattice.join``), find
each other by need (``lattice.find``), coordinate with expiring work claims
(``lattice.claim``), and talk with typed peer messages (``agents.message`` with a
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

- **Claim or take work.** It writes no files, so it never ``lattice.claim``\\ s
  and declines ``handoff`` with a pointer to ``lattice.find``.
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
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from neo.text_budget import truncate_marked

logger = logging.getLogger(__name__)

AGENT_PREFIX = "neo-lattice-"
RUNTIME = "neo"

# Tags a `lattice.find` need is matched against. The daemon scores a tag only
# when EVERY word in it appears in the need, so multi-word tags are precise and
# single words are broad; both shapes are here on purpose.
BASE_CAPABILITIES: tuple[str, ...] = (
    "code-reasoning",
    "code-review",
    "patch-verification",
    "architecture",
    "debugging",
    "root-cause-analysis",
    # Qualified: Neo advises on these and writes nothing, so a need for
    # someone to DO the refactor must not match it.
    "performance-analysis",
    "refactoring-advice",
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
# Model-backed answers one sender may ask for per hour. MAX_QUEUE bounds how
# many run at once, not how much a chatty auto-responder can spend on the
# operator's API key.
SENDER_BUDGET_PER_HOUR = 12
# Follow-ups to a Neo answer, chained by `in_reply_to`, before Neo stops.
# Ignoring kind=answer stops the obvious loop; a peer that replies to every
# answer with a new question is the less obvious one.
MAX_FOLLOWUP_DEPTH = 4
QUESTION_OUTPUT = (
    "answer for another agent: answer the question directly in prose, citing "
    "files and lines; propose a code change only if the question asks for one"
)
RECONNECT_MIN_SECS = 1.0
# Every daemon call times out; a daemon that stops replying on an open socket
# is a lost connection, not a reason to wait forever.
RPC_TIMEOUT_SECS = 20.0
POLL_SECS = 1.0
# A node is reaped once neo has not run in its repository for this long.
STALE_NODE_DAYS = 14
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
            # NUL-separated: a quoted `"café.py"` ends in `.py"` and has no language.
            ["git", "-C", root, "ls-files", "-z"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    counts: Counter[str] = Counter()
    for path in out.split("\0"):
        lang = language_for_path(path)
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
    depth: int = 0  # follow-ups deep in a chain started by a Neo answer


# Kinds Neo acts on. Everything else (`answer`, and any kind a newer peer
# invents: ack, status, result, error, ...) is acknowledged and dropped. An
# allowlist, not a denylist, because the failure of a denylist is an unbounded
# conversation between two auto-responders, billed per round trip.
_ANSWERED_KINDS = {"", "question", "review_request"}

# A file header pair, `--- x` then `+++ y`, at the start of a line. Both lines
# are required: a lone `--- Summary` in prose is not a diff.
_HEADER_PAIR_RE = re.compile(r"^--- \S[^\n]*\n\+\+\+ \S", re.M)
_GIT_HEADER_RE = re.compile(r"^diff --git a/\S", re.M)


def _diff_start(text: str) -> Optional[int]:
    """Offset where a unified diff begins in `text`, or None."""
    hits = [m.start() for m in (_GIT_HEADER_RE.search(text), _HEADER_PAIR_RE.search(text)) if m]
    return min(hits) if hits else None


def looks_like_diff(text: str) -> bool:
    return _diff_start(text) is not None


def plan_task(req: PeerRequest) -> Task:
    """Decide what to do with a request. Pure: no I/O, no model."""
    if req.kind == "handoff":
        return Task(op="decline", reason=(
            "Neo is read-only and does not take work. Use lattice.find with "
            "the capability you need to reach a player that writes code. "
            "Ask me a question or send a review_request instead.\n\n" + USAGE
        ))
    if req.kind not in _ANSWERED_KINDS:
        return Task(op="ignore", reason=f"kind `{req.kind}` is not answered")
    if not req.kind and req.in_reply_to:
        # An untyped message that answers something ("received, thanks") is
        # a reply, not a request, whatever its kind field says.
        return Task(op="ignore", reason="replies are not answered")

    body = req.body.strip()
    op = "review" if req.kind == "review_request" else "reason"
    prompt, diff, error_trace = body, "", None

    data = _json_object(body)
    if data is not None:
        requested = str(data.get("op") or "").strip().lower()
        # Under review_request the body may narrow to a memory lookup but not
        # widen to `reason`: that runs the learning path, and the peer's diff
        # would enter it as if it were Neo's suggestion.
        allowed = {"review", "memory"} if req.kind == "review_request" else {"reason", "memory", "review"}
        if requested in allowed:
            op = requested
        prompt = str(data.get("prompt") or "").strip()
        diff = str(data.get("diff") or "")
        trace = data.get("error_trace")
        error_trace = str(trace) if trace else None
    elif body.lower().startswith("memory:"):
        op, prompt = "memory", body[len("memory:"):].strip()
    elif op == "review":
        at = _diff_start(body)
        if at is not None:
            prompt, diff = body[:at].strip(), body[at:]

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


_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


def split_unified_diff(diff: str) -> list[tuple[str, str]]:
    """Split a multi-file unified diff into `(file_path, file_diff)` pairs.

    Lines are split on `\\n` only: `str.splitlines` also breaks on form feeds
    and other separators that occur inside real source, which cuts a hunk
    line in two. Inside a hunk the `@@` line counts are tracked, so a diff OF
    a patch file, whose hunk lines read `--- x` / `+++ y`, is not taken for a
    new file. Every `diff --git` line starts a section; a section with no
    hunks (rename, mode change, binary) carries no content to verify and is
    skipped. The path is the `+++` side, or `---` for a deletion; C-quoted
    paths are unquoted. Text with no file header yields nothing.
    """
    lines = diff.split("\n")
    sections: list[list[str]] = []
    current: Optional[list[str]] = None
    old_left = new_left = 0  # hunk lines still expected on each side
    i = 0
    while i < len(lines):
        line = lines[i]
        in_hunk = old_left > 0 or new_left > 0
        if in_hunk and _ends_hunk(lines, i):
            # The header overstated its line count, which hand-edited and
            # model-written diffs do routinely. Trusting the count would
            # swallow the next file's headers into this file's chunk.
            old_left = new_left = 0
            in_hunk = False
        if in_hunk:
            if line.startswith("-"):
                old_left -= 1
            elif line.startswith("+"):
                new_left -= 1
            elif line.startswith("\\"):
                pass  # "\ No newline at end of file"
            else:
                old_left -= 1
                new_left -= 1
            if current is not None:
                current.append(line)
            i += 1
            continue
        if line.startswith("diff --git "):
            current = [line]
            sections.append(current)
        elif line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            if current is None or any(x.startswith("--- ") for x in current):
                current = []
                sections.append(current)
            current.extend([line, lines[i + 1]])
            i += 2
            continue
        else:
            hunk = _HUNK_RE.match(line)
            if hunk and current is not None:
                old_left = int(hunk.group(1)) if hunk.group(1) is not None else 1
                new_left = int(hunk.group(2)) if hunk.group(2) is not None else 1
            if current is not None:
                current.append(line)
        i += 1

    pairs: list[tuple[str, str]] = []
    for section in sections:
        minus = next((x for x in section if x.startswith("--- ")), None)
        plus = next((x for x in section if x.startswith("+++ ")), None)
        if minus is None or plus is None or not any(_HUNK_RE.match(x) for x in section):
            continue
        path = _header_path(plus)
        if path == "/dev/null":
            path = _header_path(minus)
        if path and path != "/dev/null":
            text = "\n".join(section).rstrip("\n") + "\n"
            pairs.append((path, text))
    return pairs


def _ends_hunk(lines: list[str], i: int) -> bool:
    """Whether line `i` cannot be content of the hunk still being read."""
    line = lines[i]
    if line.startswith(("diff --git ", "@@ ")):
        return True
    if line and line[0] not in " -+\\":
        return True
    # `--- x` / `+++ y` / `@@` is the next file's header, not a removed line
    # followed by an added one: a hunk line cannot be followed by `@@`.
    return (line.startswith("--- ") and i + 2 < len(lines)
            and lines[i + 1].startswith("+++ ") and lines[i + 2].startswith("@@ "))


def _header_path(line: str) -> str:
    raw = line[4:].split("\t", 1)[0].strip()
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        # git C-quotes paths with special bytes: "b/caf\303\251.py"
        try:
            raw = raw[1:-1].encode("latin-1").decode("unicode_escape").encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            raw = raw[1:-1]
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    return raw


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------


class Answerer:
    """Turns a planned task into answer text, against one repository.

    Synchronous and single-threaded: `NeoEngine` handles one request at a time
    and the node feeds it from a single worker.
    """

    def __init__(self, root: str, engine_factory: Optional[Callable[[], Any]] = None,
                 store_factory: Optional[Callable[[], Any]] = None, scoped_consumer=None):
        self.root = root
        self.scoped_consumer = scoped_consumer
        if scoped_consumer is not None and (engine_factory is not None or store_factory is not None):
            raise ValueError("Scoped Neo requires its bound engine and private store")
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
            lm_adapter=resolve_adapter(config, scoped_consumer=self.scoped_consumer),
            codebase_root=self.root,
            config=config,
            scoped_consumer=self.scoped_consumer,
        )

    @property
    def loaded(self) -> bool:
        """Holding an engine or a fact store: the memory a recycle releases."""
        return self._engine is not None or self._store is not None

    @property
    def engine(self) -> Any:
        if self._engine is None:
            self._engine = self._engine_factory()
            # From here on the engine's store answers memory lookups; keeping
            # the standalone one would hold two copies of the same facts.
            self._store = None
        return self._engine

    def answer(self, task: Task) -> str:
        if self.scoped_consumer is not None:
            self.scoped_consumer.authorize()
        if task.op == "memory":
            result = self.memory(task.prompt)
        elif task.op == "review":
            result = self.review(task)
        else:
            result = self.reason(task)
        if self.scoped_consumer is not None:
            self.scoped_consumer.authorize()
        return result

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

        # read_only: a lookup must not run the prune/demote chain and save.
        scope = self.scoped_consumer
        if scope is not None:
            scope.prepare_state_root()
        return FactStore(codebase_root=self.root, config=NeoConfig.load(), read_only=True,
                         **({"facts_dir": scope.state_root / "facts",
                             "episodes_dir": scope.state_root / "episodes",
                             "emit_metrics": False} if scope is not None else {}))

    def memory(self, query: str) -> str:
        if self.scoped_consumer is not None:
            self.scoped_consumer.authorize()
        result = self._memory(query)
        if self.scoped_consumer is not None:
            self.scoped_consumer.authorize()
        return result

    def _memory(self, query: str) -> str:
        store = self._memory_store()
        if store is None:
            return "Neo's project memory is not available on this node."
        from neo.memory.models import FactScope

        # No access stamps (a lookup cannot earn the success that offsets
        # one), and PROJECT scope only: global and org facts are mined from
        # every repository on this machine, and any peer on the daemon can
        # send this query.
        facts = _distinct([
            f for f in store.retrieve_relevant(query, k=MEMORY_RESULTS * 3, record_access=False)
            if f.scope is FactScope.PROJECT
        ])[:MEMORY_RESULTS]
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

        from neo.models import TaskType, classify_task_type
        from neo.operating_mode import OperatingMode

        # Only a bugfix/algorithm suggestion can ever be promoted by a later
        # git-verified acceptance; anything else would just leave an episode
        # pending forever. Those run in ADVISE and record nothing.
        promotable = classify_task_type(task.prompt, task.error_trace) in {
            TaskType.BUGFIX, TaskType.ALGORITHM,
        }
        output = self.engine.process(NeoInput(
            prompt=task.prompt,
            error_trace=task.error_trace,
            working_directory=self.root,
            operating_mode=(OperatingMode.ADVISE if self.scoped_consumer is not None
                            else OperatingMode.LEARN if promotable else OperatingMode.ADVISE),
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
            if self.scoped_consumer is not None:
                from types import SimpleNamespace
                from neo.models import CodeSuggestion
                from neo.static_analysis import run_static_checks

                self.scoped_consumer.authorize()
                root = Path(self.root).resolve()
                suggestions = []
                for change in changes:
                    path = (root / change.file_path).resolve()
                    if not path.is_relative_to(root):
                        raise PermissionError("Peer review path escapes the original workspace")
                    suggestions.append(CodeSuggestion(
                        file_path=str(path), unified_diff=change.unified_diff,
                        description=change.description, confidence=0.0,
                    ))
                checks = run_static_checks(suggestions)
                request = NeoInput(prompt=task.prompt or "Verify the proposed change.",
                                   working_directory=self.root,
                                   operating_mode=OperatingMode.ADVISE)
                constraint_check = self.engine._check_constraints_static(
                    suggestions, self.engine._extract_input_constraints(request))
                if constraint_check is not None:
                    checks.append(constraint_check)
                self.scoped_consumer.authorize()
                verified = SimpleNamespace(static_checks=checks)
            else:
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

    Cautions come FIRST: a caution exists so a confident-sounding answer
    cannot bury it, and the reply is cut from the tail when it is too long.
    Then the answer, which leads everything else.
    """
    orch = getattr(output, "orchestrator", None)
    parts: list[str] = []
    cautions = getattr(orch, "cautions", []) if orch else []
    if cautions:
        parts.append("Cautions:\n" + "\n".join(f"- {c}" for c in cautions))
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
    work_context_json: Optional[str] = None


def budget_key(sender: str) -> str:
    """The identity a sender's model budget is charged to.

    A node on another CAR arrives as `<node>@<car>`, and the `<node>` part is
    chosen by that CAR, so keying on the full address would let it rotate
    names past `SENDER_BUDGET_PER_HOUR`. Charge the CAR instead, as `@<car>`
    so it cannot share a budget with a local node of the same name. Local names
    cannot contain `@`, so they are their own key.
    """
    return "@" + sender.rpartition("@")[2] if "@" in sender else sender


# This explicit native parent profile also enables CAR's task incarnation lifecycle.
SCOPED_PARENT_METHODS = tuple(sorted((
    "agents.message", "infer", "lattice.join", "lattice.leave", "models.route",
    "session.consumer.issue", "session.consumer.revoke",
)))
CONSUMER_TTL_SECS = 900
CONSUMER_RENEW_MARGIN_SECS = 60


def _original_receipt(auth: Any, agent_id: str, context: dict) -> tuple[dict, tuple[str, ...]]:
    """Validate and canonicalize the native receipt before publishing pins."""
    if (not isinstance(auth, dict) or auth.get("ok") is not True
            or auth.get("auth_enabled") is not True or auth.get("agent_id") != agent_id
            or auth.get("work_context") != context):
        raise PermissionError("Original Neo coordination receipt changed")
    task = auth.get("task_binding")
    if (not isinstance(task, dict) or set(task) != {"version", "task_incarnation", "resource_key"}
            or type(task["version"]) is not int or task["version"] != 1):
        raise PermissionError("Original Neo task/resource receipt unavailable")
    try:
        task_id = uuid.UUID(task["task_incarnation"])
    except (ValueError, TypeError, AttributeError):
        raise PermissionError("Invalid original Neo task incarnation") from None
    if task_id.int == 0 or str(task_id) != task["task_incarnation"]:
        raise PermissionError("Invalid original Neo task incarnation")
    if not isinstance(task["resource_key"], str) or re.fullmatch(r"context-private:work-[0-9a-f]{64}", task["resource_key"]) is None:
        raise PermissionError("Invalid original Neo resource")
    methods = auth.get("method_allowlist")
    if (not isinstance(methods, list) or any(not isinstance(m, str) or not m or len(m) > 256
            or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in m) for m in methods)):
        raise PermissionError("Invalid original Neo method scope")
    canonical = tuple(sorted(set(methods)))
    if canonical != SCOPED_PARENT_METHODS:
        raise PermissionError("Neo requires its explicit original parent method profile")
    return dict(task), canonical


class LatticeNode:
    """Neo's presence on the Lattice for one repository.

    Acknowledges each pushed `agent.peer_message` immediately, queues it, and
    answers from one worker thread so the engine sees one request at a time.
    Status reads `busy` while answering, so `lattice.find` routes elsewhere
    unless the asker opts into unavailable players.

    Every daemon call has a timeout, and a timeout, a send failure or a
    superseded binding is treated as a lost connection: the socket is closed
    and the run loop re-attaches and re-joins. A daemon that stops answering
    on an open socket would otherwise wedge the node with nothing to notice.
    """

    def __init__(self, config: NodeConfig, answerer: Optional[Answerer] = None,
                 client_factory: Optional[Callable[[str], Any]] = None):
        self.config = config
        self.answerer = answerer or Answerer(config.root)
        self._scoped_consumer = None
        self._original_task_binding = None
        self._original_method_allowlist = None
        self._consumer_id = None
        self._consumer_renew_at = 0.0
        self._consumer_lock = asyncio.Lock()
        self._consumer_refused = False
        self._expected_work_context_json = config.work_context_json
        self._original_agent_id = config.agent_id
        self._original_token = config.token
        self._original_url = config.url
        if config.work_context_json is not None:
            context = json.loads(config.work_context_json)
            if (not isinstance(context, dict) or set(context) != {"api_base", "account_id", "organization_id"}
                    or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in context.values())):
                raise ValueError("Scoped Neo launch requires its complete original context")
        if config.work_context_json is not None and answerer is not None:
            raise ValueError("Scoped nodes cannot substitute an unbound answerer")
        self._client_factory = client_factory or _default_client
        self.client: Any = None
        self.queue: Optional[asyncio.Queue] = None
        self._stop = asyncio.Event()
        self._connected = asyncio.Event()
        self.answered = 0
        self.last_activity = time.monotonic()
        self.recycle = False
        self.signalled = False
        self.unavailable = False
        self.orphaned = False
        self.working = False
        # CAR's supervisor spawns the node directly, so it is our parent; the
        # recycle execs in place and keeps it. A daemon that dies without
        # stopping its children (SIGKILL, a crash, a test daemon torn down)
        # reparents us, and the reconnect loop would otherwise retry forever.
        self._supervisor_pid = os.getppid()
        self._spend: dict[str, list[float]] = {}
        # Ids of answers Neo sent -> how many follow-ups deep each is.
        self._answer_depth: dict[str, int] = {}
        # Strong references: the loop holds tasks weakly, so an unreferenced
        # fire-and-forget task can be collected mid-flight.
        self._tasks: set[asyncio.Task] = set()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="neo-lattice")

    def supervisor_gone(self) -> bool:
        return os.getppid() != self._supervisor_pid

    def should_recycle(self, now: Optional[float] = None) -> bool:
        """True once a loaded engine or store has sat idle past `IDLE_RECYCLE_SECS`."""
        if not self.answerer.loaded:
            return False
        if self.working or (self.queue is not None and self.queue.qsize()):
            return False
        return ((now or time.monotonic()) - self.last_activity) >= IDLE_RECYCLE_SECS

    def _spawn(self, coro: Any) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # -- daemon calls ---------------------------------------------------------

    async def _rpc(self, method: str, params: dict) -> Any:
        client = self.client
        if client is None:
            raise ConnectionError("not connected")
        try:
            return await client.call(method, params, timeout=RPC_TIMEOUT_SECS)
        except ConnectionError as e:
            await self._lose(client, f"{method}: {e}")
            raise
        except asyncio.TimeoutError:
            # Not raised as ConnectionError: a timed-out request may still
            # have been carried out, which matters to a caller deciding
            # whether to resend it.
            await self._lose(client, f"{method}: no reply in {RPC_TIMEOUT_SECS:.0f}s")
            raise
        except RuntimeError as e:
            if "superseded" in str(e):
                # The binding moved to another connection; this socket stays
                # open and refuses everything, so close it and re-attach.
                await self._lose(client, f"{method}: {e}")
                raise ConnectionError(str(e)) from e
            raise

    async def _lose(self, client: Any, why: str) -> None:
        logger.warning("dropping daemon connection (%s)", why)
        self._connected.clear()
        await self._drop_consumer(client)
        await client.close()

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
                          "memory (body `memory: ...`), or send a review_request "
                          "with a diff.")[:280]
        return params

    async def set_status(self, status: str, note: str = "") -> None:
        try:
            await self._rpc("lattice.join", self.profile(status, note))
        except Exception as e:  # noqa: BLE001 — status is advisory
            logger.warning("lattice.join (%s) failed: %s", status, e)

    # -- inbound --------------------------------------------------------------

    async def on_peer_message(self, params: dict) -> dict:
        """Acknowledge a push. Never blocks: answering happens in the worker."""
        req = PeerRequest.from_params(params)
        task = self.admit(req, plan_task(req))
        if task.op == "ignore":
            return {"received": True, "action": "ignored", "reason": task.reason}
        assert self.queue is not None
        try:
            self.queue.put_nowait((req, task))
        except asyncio.QueueFull:
            self._refund(req, task)
            # Bounded like the queue: under a flood the overflow notices must
            # not become their own unbounded backlog.
            if len(self._tasks) < MAX_QUEUE:
                self._spawn(self._reply(req, (
                    f"Neo is answering {MAX_QUEUE + 1} requests already. "
                    "Try again in a few minutes."
                )))
            return {"received": True, "action": "busy"}
        return {"received": True, "action": "queued", "queued": self.queue.qsize()}

    def _refund(self, req: PeerRequest, task: Task) -> None:
        """Return the budget slot of a request that was turned away unanswered."""
        key = budget_key(req.sender)
        if task.op in {"reason", "review"} and self._spend.get(key):
            self._spend[key].pop()

    def admit(self, req: PeerRequest, task: Task, now: Optional[float] = None) -> Task:
        """Apply the follow-up depth cap and the per-sender budget."""
        if task.op not in {"reason", "review"}:
            return task  # memory lookups and declines cost no model call
        depth = self._answer_depth.get(req.in_reply_to, 0) + 1 if req.in_reply_to else 0
        if depth > MAX_FOLLOWUP_DEPTH:
            return Task(op="ignore", reason=f"follow-up chain deeper than {MAX_FOLLOWUP_DEPTH}")
        now = now if now is not None else time.monotonic()
        key = budget_key(req.sender)
        recent = [t for t in self._spend.get(key, []) if now - t < 3600]
        if len(recent) >= SENDER_BUDGET_PER_HOUR:
            self._spend[key] = recent
            return Task(op="decline", reason=(
                f"Neo has answered {SENDER_BUDGET_PER_HOUR} model-backed requests from you "
                "in the last hour. memory: lookups still work; try again later."
            ))
        recent.append(now)
        self._spend[key] = recent
        task.depth = depth
        return task

    # -- outbound -------------------------------------------------------------

    async def _reply(self, req: PeerRequest, text: str, depth: int = 0) -> None:
        if req.no_reply or not req.sender:
            return
        params: dict[str, Any] = {
            "to": req.sender, "kind": "answer", "body": format_reply(req, text),
        }
        if req.message_id:
            # The daemon's correlation field; the `re:` line in the body
            # carries the same id for readers that only see text.
            params["in_reply_to"] = req.message_id
        for attempt in range(2):
            try:
                sent = await self._rpc("agents.message", params)
            except asyncio.TimeoutError:
                # The daemon may have delivered it and been slow to say so;
                # resending could hand the peer the same answer twice.
                logger.warning("reply to %s unconfirmed (timeout); not resending", req.sender)
                return
            except ConnectionError as e:
                # Not sent at all: one retry once re-attached, so a daemon
                # restart mid-answer does not lose a finished answer.
                logger.warning("reply to %s not sent (attempt %d): %s", req.sender, attempt + 1, e)
                try:
                    await asyncio.wait_for(self._connected.wait(), timeout=60)
                except asyncio.TimeoutError:
                    return
                continue
            except Exception as e:  # noqa: BLE001 — refused (guard, policy, recipient gone)
                logger.warning("reply to %s refused: %s", req.sender, e)
                return
            if isinstance(sent, dict) and sent.get("id"):
                if len(self._answer_depth) >= 1024:
                    self._answer_depth.clear()  # bounded; worst case a chain restarts its count
                self._answer_depth[str(sent["id"])] = depth
            return

    async def worker(self) -> None:
        assert self.queue is not None
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            req, task = await self.queue.get()
            self.last_activity = time.monotonic()
            self.working = True
            advertised_busy = False
            try:
                if task.op == "decline":
                    await self._reply(req, task.reason)
                    continue
                await self.set_status("busy", f"answering a {req.kind or 'message'} from {req.sender}")
                advertised_busy = True
                try:
                    async with self._consumer_lock:
                        await self._ensure_consumer()
                        text = await loop.run_in_executor(self._executor, self.answerer.answer, task)
                        if self._expected_work_context_json is not None:
                            await self._authorize_consumer()
                except Exception as e:  # noqa: BLE001 — the asker gets told
                    logger.exception("answering %s failed", req.message_id)
                    if self._expected_work_context_json is not None:
                        try:
                            await self._authorize_consumer()
                        except PermissionError:
                            self._stop.set()
                        except (ConnectionError, OSError, asyncio.TimeoutError):
                            if self.client is not None:
                                await self._lose(self.client, "original child transport unavailable")
                        except Exception:
                            self._stop.set()
                    text = ("Neo could not answer under its original authority."
                            if self._expected_work_context_json is not None
                            else f"Neo failed to answer: {type(e).__name__}: {e}")
                await self._reply(req, text, depth=task.depth)
                self.answered += 1
                # Serving peers is use: a node nobody runs the neo CLI beside
                # but that answers every day must not be reaped as unused.
                touch_seen(self.config.agent_id)
            finally:
                self.working = False
                self.last_activity = time.monotonic()
                self.queue.task_done()
                if advertised_busy and self.queue.empty() and self.client is not None:
                    await self.set_status("available")

    # -- connection -----------------------------------------------------------

    def _close_consumer(self) -> None:
        scope, self._scoped_consumer = self._scoped_consumer, None
        self._consumer_id = None
        if scope is not None:
            scope.runtime.close()

    async def _drop_consumer(self, client: Any) -> None:
        async with self._consumer_lock:
            if self._consumer_id is not None and client is not None and client.connected:
                try:
                    await self._retire_consumer(client, self._consumer_id)
                except Exception:
                    logger.warning("Neo child retirement unconfirmed; grant expiry remains enforced")
            self._close_consumer()

    async def _authorize_consumer(self) -> None:
        if self._consumer_refused or self._scoped_consumer is None:
            raise PermissionError("Original Neo consumer unavailable or refused")
        try:
            await asyncio.to_thread(self._scoped_consumer.authorize)
        except PermissionError:
            self._consumer_refused = True
            raise
        except (ConnectionError, OSError, asyncio.TimeoutError):
            raise
        except Exception as error:
            message = str(error)
            if (type(error) is RuntimeError and (message in {
                    "rpc channel closed for models.route", "daemon closed connection",
                    "send scoped consumer authentication failed",
                    "scoped consumer authentication timed out",
                    "daemon closed during scoped consumer authentication",
                    "receive scoped consumer authentication failed",
                } or message.startswith(f"connect daemon at {self._original_url}: ")
                or message.startswith(f"connect daemon at {self._original_url} timed out after "))):
                raise ConnectionError("Original Neo child transport unavailable") from error
            self._consumer_refused = True
            raise

    async def _retire_consumer(self, client: Any, consumer_id: str) -> None:
        result = await client.call("session.consumer.revoke", {"consumer_id": consumer_id}, timeout=RPC_TIMEOUT_SECS)
        if not isinstance(result, dict) or result.get("revoked") is not True:
            raise PermissionError("Original Neo child retirement unconfirmed")

    async def _issue_consumer(self, client: Any, context: dict) -> None:
        try:
            await self._issue_consumer_once(client, context)
        except PermissionError:
            self._consumer_refused = True
            raise
        except (ConnectionError, OSError, asyncio.TimeoutError):
            raise
        except Exception as error:
            message = str(error)
            if (type(error) is RuntimeError and (message in {
                    "rpc channel closed for models.route", "daemon closed connection",
                    "send scoped consumer authentication failed",
                    "scoped consumer authentication timed out",
                    "daemon closed during scoped consumer authentication",
                    "receive scoped consumer authentication failed",
                } or message.startswith(f"connect daemon at {self._original_url}: ")
                or message.startswith(f"connect daemon at {self._original_url} timed out after "))):
                raise ConnectionError("Original Neo child transport unavailable") from error
            self._consumer_refused = True
            raise

    async def _issue_consumer_once(self, client: Any, context: dict) -> None:
        from neo.scoped_consumer import ScopedConsumer
        if self._consumer_refused:
            raise PermissionError("Original Neo consumer refused; new enrollment required")
        issuance = await client.call("session.consumer.issue", {
            "methods": ["infer", "models.route"], "ttl_secs": CONSUMER_TTL_SECS,
        }, timeout=RPC_TIMEOUT_SECS)
        now_ms = int(time.time() * 1000)
        consumer_id = issuance.get("consumer_id") if isinstance(issuance, dict) else None
        issued = issuance.get("issued_at_ms") if isinstance(issuance, dict) else None
        expires = issuance.get("expires_at_ms") if isinstance(issuance, dict) else None
        if (not isinstance(consumer_id, str) or not consumer_id or type(issued) is not int
                or type(expires) is not int or issued > now_ms or expires <= now_ms
                or expires <= issued or expires - issued > CONSUMER_TTL_SECS * 1000):
            if isinstance(consumer_id, str) and consumer_id:
                try:
                    await self._retire_consumer(client, consumer_id)
                except Exception:
                    logger.warning("Malformed Neo child retirement unconfirmed; grant expiry remains enforced")
            raise PermissionError("Invalid Neo child lifetime receipt")
        # Cancellation cannot abandon a still-authenticating SDK socket in a thread.
        factory = asyncio.create_task(asyncio.to_thread(ScopedConsumer.from_issued,
            self._original_url, json.dumps(issuance), self._original_agent_id,
            self._original_task_binding["resource_key"], context, Path.home() / ".neo" / "scoped"))
        scope = None
        old_scope, old_id = self._scoped_consumer, self._consumer_id
        try:
            scope = await asyncio.shield(factory)
            now_ms = int(time.time() * 1000)
            if expires <= now_ms:
                raise PermissionError("Neo child expired while authenticating")
            if old_id is not None:
                await self._retire_consumer(client, old_id)
        except BaseException:
            if scope is None:
                try:
                    scope = await factory
                except Exception:
                    pass
            if scope is not None:
                scope.runtime.close()
            try:
                await self._retire_consumer(client, consumer_id)
            except Exception:
                logger.warning("Unused Neo child retirement unconfirmed; grant expiry remains enforced")
            raise
        self._scoped_consumer = scope
        self._consumer_id = consumer_id
        self._consumer_renew_at = time.monotonic() + max(0, (expires - now_ms) / 1000 - CONSUMER_RENEW_MARGIN_SECS)
        self.answerer = Answerer(self.config.root, scoped_consumer=scope)
        if old_scope is not None:
            old_scope.runtime.close()

    async def _ensure_consumer(self) -> None:
        """Caller holds the consumer lock across renewal and answering."""
        if self._expected_work_context_json is None:
            return
        await self._authorize_consumer()
        if time.monotonic() >= self._consumer_renew_at:
            if self.client is None or not self.client.connected:
                raise ConnectionError("Original Neo coordination socket unavailable")
            await self._issue_consumer(self.client, json.loads(self._expected_work_context_json))

    async def renew_consumer_if_due(self) -> None:
        if self._expected_work_context_json is not None and time.monotonic() >= self._consumer_renew_at:
            async with self._consumer_lock:
                await self._ensure_consumer()

    async def connect_once(self) -> None:
        client = self._client_factory(self._original_url)
        client.on_request("agent.peer_message", self.on_peer_message)
        # Assigned before connecting, so a failed auth is closed by the run
        # loop instead of leaking a socket and a receive task per retry.
        self.client = client
        auth = await asyncio.wait_for(
            client.connect({"token": self._original_token, "agent_id": self._original_agent_id}),
            RPC_TIMEOUT_SECS,
        )
        if isinstance(auth, dict) and auth.get("work_context") is not None and self._expected_work_context_json is None:
            raise PermissionError("Bound Neo launch requires its original context")
        if self._expected_work_context_json is not None:
            context = json.loads(self._expected_work_context_json)
            task, methods = _original_receipt(auth, self._original_agent_id, context)
            if self._original_task_binding is not None and self._original_task_binding != task:
                raise PermissionError("Original Neo task/resource changed on reconnect")
            if self._original_method_allowlist is not None and self._original_method_allowlist != methods:
                raise PermissionError("Original Neo method scope changed on reconnect")
            self._original_task_binding = task
            self._original_method_allowlist = methods
            async with self._consumer_lock:
                await self._issue_consumer(client, context)
        try:
            from neo import __version__
            await client.call("server.handshake", {
                "protocol_version": 3, "client_version": f"neo-{__version__}",
                "required_capabilities": [], "optional_capabilities": [],
            }, timeout=RPC_TIMEOUT_SECS)
        except Exception as e:  # noqa: BLE001 — lattice.* does not need it
            logger.info("server.handshake not negotiated: %s", e)
        try:
            await client.call("lattice.join", self.profile("available"), timeout=RPC_TIMEOUT_SECS)
        except RuntimeError as e:
            if getattr(e, "unknown_method", False):
                raise LatticeUnavailable(str(e)) from e
            raise
        self._connected.set()
        logger.info("joined project %s as %s", self.config.project, self.config.agent_id)

    async def run(self) -> None:
        """Serve until stopped. Raises `WorkerDied` if the answering worker
        exits, so the process fails and CAR's supervisor restarts it."""
        self.queue = asyncio.Queue(maxsize=MAX_QUEUE)
        worker = asyncio.create_task(self.worker())
        delay = RECONNECT_MIN_SECS
        try:
            while not self._stop.is_set():
                if self.supervisor_gone():
                    logger.warning("supervisor exited; exiting rather than reconnect forever")
                    self.orphaned = True
                    break
                try:
                    if not await self._unless_stopped(self.connect_once()):
                        break
                    delay = RECONNECT_MIN_SECS
                    while self.client.connected and not self._stop.is_set():
                        await asyncio.sleep(POLL_SECS)
                        if self.supervisor_gone():
                            break
                        await self.renew_consumer_if_due()
                        if worker.done():
                            break
                        if self.should_recycle():
                            logger.info("idle with memory loaded; recycling")
                            self.recycle = True
                            self._stop.set()
                except LatticeUnavailable as e:
                    # This daemon predates the Lattice. Retrying cannot help;
                    # a clean exit is not restarted under `on_failure`, and the
                    # next daemon start (an upgraded one) tries again.
                    logger.warning("this CAR daemon has no Lattice (%s); exiting", e)
                    self.unavailable = True
                    self._stop.set()
                except Exception as e:  # noqa: BLE001 — reconnect with backoff
                    logger.warning("lattice connection failed: %s", e)
                    if self._consumer_refused:
                        logger.warning("Original Neo consumer refused; explicit re-enrollment required")
                        self._stop.set()
                if worker.done() and not self._stop.is_set():
                    # Acking into a queue nothing drains looks healthy and
                    # answers no one; fail so the supervisor restarts us.
                    raise WorkerDied(repr(worker.exception()) if not worker.cancelled() else "cancelled")
                self._connected.clear()
                await self._drop_consumer(self.client)
                if self.client is not None:
                    await self.client.close()
                if self._stop.is_set():
                    break
                logger.warning("reconnecting in %.0fs", delay)
                await self._unless_stopped(asyncio.sleep(delay))
                delay = min(delay * 2, RECONNECT_MAX_SECS)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            await self._drop_consumer(self.client)
            if self.client is not None:
                await self.client.close()
            self._executor.shutdown(wait=False, cancel_futures=True)

    async def _unless_stopped(self, coro: Any) -> bool:
        """Run `coro` unless a stop arrives first; True if it completed.

        Reconnect waits (a 30 s backoff, three 20 s calls) would otherwise
        outlast the supervisor's SIGTERM grace and end in a SIGKILL.
        """
        work = asyncio.ensure_future(coro)
        stop = asyncio.ensure_future(self._stop.wait())
        done, _ = await asyncio.wait({work, stop}, return_when=asyncio.FIRST_COMPLETED)
        if work in done:
            stop.cancel()
            work.result()  # re-raise its exception, if any
            return True
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        return False

    async def leave(self) -> None:
        self._stop.set()
        if self.client is not None and self.client.connected:
            try:
                await self.client.call("lattice.leave", {}, timeout=5)
            except Exception:  # noqa: BLE001
                pass


class WorkerDied(RuntimeError):
    """The answering worker exited; the node must not keep acknowledging."""


class LatticeUnavailable(RuntimeError):
    """The daemon does not implement `lattice.*`."""


def daemon_has_lattice(url: str, timeout: float = 5.0) -> Optional[bool]:
    """Whether the daemon at `url` implements the Lattice's `lattice.*`.

    True or False when the daemon answered; None when it could not be asked.
    A plain client calling `lattice.nodes` is refused either way. A daemon
    WITH the API refuses it for lacking a peer address; one without it says
    "unknown method", and that is the only answer that means "no".
    """
    async def probe() -> Optional[bool]:
        from neo.a2ui import DaemonClient

        client = DaemonClient(url)
        try:
            await asyncio.wait_for(client.connect(), timeout)
            await client.call("lattice.nodes", {}, timeout=timeout)
            return True
        except RuntimeError as e:
            return not getattr(e, "unknown_method", False)
        finally:
            await client.close()

    try:
        return asyncio.run(probe())
    except Exception as e:  # noqa: BLE001 — unreachable, no websockets, auth
        logger.debug("lattice probe failed: %s", e)
        return None


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


def _state_dir() -> Path:
    """`~/.neo/lattice`, resolved at call time so it follows `Path.home()`."""
    return Path.home() / ".neo" / "lattice"


def node_argv(root: str, project: str) -> list[str]:
    """Interpreter arguments for the node, shared by the spec and the recycle.

    `-P` (3.11+) keeps the working directory off `sys.path`; the spec's cwd
    is Neo's own state dir for the same reason on older interpreters. The
    repository is passed as `--cwd`, never as the process cwd: `python -m`
    puts the cwd first on `sys.path`, so a node started IN a repository
    would import that repository's `json.py` or `neo/` package instead of
    the real ones, while holding an agent token, merely because neo was
    once run there.
    """
    safe = ["-P"] if sys.version_info >= (3, 11) else []
    return [*safe, "-m", "neo.lattice", "run", "--cwd", root, "--project", project]


def build_spec(root: str, project: str, work_context: Optional[dict] = None) -> dict:
    agent_id = agent_id_for(root)
    if work_context is not None:
        import hashlib
        agent_id += "-" + hashlib.sha256(json.dumps(work_context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    home = Path.home() / ".neo"
    home.mkdir(parents=True, exist_ok=True)
    spec = {
        "id": agent_id,
        "name": f"Neo ({project})",
        "command": sys.executable,
        "args": node_argv(root, project),
        "cwd": str(home),
        "env": {k: os.environ[k] for k in _FORWARDED_ENV if k in os.environ},
        "restart": "on_failure",
        "max_restarts": 10,
        "backoff_secs": 5,
        "auto_start": True,
    }
    if work_context is not None:
        spec["work_context"] = json.loads(json.dumps(work_context))
        spec["method_allowlist"] = list(SCOPED_PARENT_METHODS)
    return spec


def _project_key(root: str) -> str:
    from neo.memory.scope import _compute_project_id

    return _compute_project_id(root)[:16]


def _left_marker(root: str) -> Path:
    """Records that the user took this repo off the Lattice."""
    return _state_dir() / f"left-{_project_key(root)}"


def _seen_marker(agent_id: str) -> Path:
    """Touched whenever neo runs in the node's repository; its age decides
    when an unused node is reaped."""
    return _state_dir() / f"seen-{agent_id}"


def touch_seen(agent_id: str) -> None:
    try:
        marker = _seen_marker(agent_id)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError as e:
        logger.debug("seen marker for %s: %s", agent_id, e)


def repository_root(path: str) -> Optional[str]:
    """The MAIN checkout of the repository containing `path`, or None.

    Not `--show-toplevel`: inside a linked worktree that names the worktree,
    and the node's agent id is keyed on the remote, so whichever checkout neo
    ran in first would own the node, and deleting that worktree (agent
    worktrees are deleted routinely) would leave a node restart-looping on a
    missing directory. The main checkout is the parent of the common git dir.

    Auto-join requires a repository: a session started from `$HOME` or `/`
    must not mint a player whose "project" is the whole machine.
    """
    def git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        value = out.stdout.strip()
        return value if out.returncode == 0 and value else None

    common = git("rev-parse", "--path-format=absolute", "--git-common-dir")
    if common and Path(common).name == ".git":
        return str(Path(common).parent)
    return git("rev-parse", "--show-toplevel")


def _list_agents(car: Any) -> Optional[list[dict]]:
    """The supervisor's agents, or None when the call failed.

    None is not "none registered": treating a failed listing as empty would
    re-register the node on every CLI run.
    """
    try:
        agents = json.loads(car.agents_list())
    except Exception as e:  # noqa: BLE001
        logger.debug("agents_list failed: %s", e)
        return None
    return agents if isinstance(agents, list) else None


def _node_root(agent: dict) -> Optional[str]:
    args = agent.get("args") or []
    for i, arg in enumerate(args[:-1]):
        if arg == "--cwd":
            return args[i + 1]
    return None


def _node_is_broken(agent: dict) -> bool:
    """Its repository or its interpreter is gone, so it can only crash-loop.

    Only evidence counts: a record WITHOUT `args` or `command` is unknown,
    not broken. Both answers here are destructive (re-register and restart
    this node, stop and remove other ones), so a listing that merely omits
    a field must not trigger them.
    """
    root, command = _node_root(agent), agent.get("command")
    return bool((root and not os.path.isdir(root)) or (command and not os.path.exists(command)))


def _reap_stale_nodes(car: Any, agents: list[dict], keep: str, now: float) -> list[str]:
    """Remove nodes whose repository is gone or that neo has not run beside
    in `STALE_NODE_DAYS`. Each node is an auto-started process; without this,
    every repository neo ever touched would hold one forever."""
    reaped: list[str] = []
    for agent in agents:
        agent_id = str(agent.get("id") or "")
        if (not agent_id.startswith(AGENT_PREFIX) or agent_id == keep
                or "-" in agent_id[len(AGENT_PREFIX):] or agent.get("work_context") is not None):
            continue
        seen = _seen_marker(agent_id)
        try:
            stale = now - seen.stat().st_mtime > STALE_NODE_DAYS * 86400
        except FileNotFoundError:
            stale = False  # never stamped (registered by hand): leave it
        if not (stale or _node_is_broken(agent)):
            continue
        for op in (car.agents_stop, car.agents_remove):
            try:
                op(agent_id)
            except Exception as e:  # noqa: BLE001
                logger.debug("%s(%s): %s", op.__name__, agent_id, e)
        seen.unlink(missing_ok=True)
        reaped.append(agent_id)
    return reaped


def maybe_autojoin(cwd: Optional[str] = None) -> Optional[str]:
    """Put this repository on the Lattice when CAR is present.

    Called once per neo CLI run, beside the observer autostart: when neo and
    CAR are installed together, every repository neo is used in gets a
    supervised, read-only neo player. No-op outside a git repository, when
    the daemon is unreachable, or when the user ran `neo lattice leave` here.
    A registered node whose repository or interpreter has disappeared is
    re-registered; nodes for vanished or long-unused repositories are reaped.
    Never raises. Returns the agent id it (re)registered, else None.

    `NEO_OBSERVER_AUTOSTART=0` turns this off too: it is the one existing
    switch for "neo registers no CAR agents on its own", and a second switch
    for the second agent would let a caller silence one and not the other.
    """
    try:
        # A bound consumer must not register an ambient personal node. Its
        # trusted launcher registers the context-bearing spec explicitly.
        if os.environ.get("CAR_WORK_CONTEXT") is not None:
            return None
        if os.getenv("NEO_OBSERVER_AUTOSTART", "").strip() == "0":
            return None
        from neo.memory.observer import _car_server_reachable, _require_car_runtime

        root = repository_root(cwd or os.getcwd())
        if root is None or _left_marker(root).exists():
            return None
        if not _car_server_reachable():
            return None
        car = _require_car_runtime()
        agents = _list_agents(car)
        if agents is None:
            return None
        agent_id = agent_id_for(root)
        touch_seen(agent_id)
        _reap_stale_nodes(car, agents, keep=agent_id, now=time.time())
        existing = next((a for a in agents if a.get("id") == agent_id), None)
        if existing is not None and not _node_is_broken(existing):
            return None  # registered and viable; the supervisor owns it
        # Only when about to register, so a registered node costs nothing
        # here. Against a daemon without the API, registering would leave a
        # process that can never join: autojoin waits until the daemon has it.
        if daemon_has_lattice(daemon_url()) is not True:
            return None
        car.agents_upsert(json.dumps(build_spec(root, derive_project_name(root))))
        if existing is not None and existing.get("status") == "running":
            car.agents_restart(agent_id)
        else:
            car.agents_start(agent_id)
        logger.debug("auto-joined %s to the lattice as %s", root, agent_id)
        return agent_id
    except Exception as e:  # noqa: BLE001 — must never break a neo command
        logger.debug("lattice autojoin skipped: %s", e)
        return None


def _pid_file(agent_id: str) -> Path:
    return _state_dir() / f"{agent_id}.pid"


def _live_node_pid(agent_id: str) -> Optional[int]:
    """The pid the node itself recorded, if that process is still a node.

    The one liveness signal that does not route through CAR: under a
    client/daemon protocol skew `agents_list` falls back to the manifest and
    reports every agent `stopped`, which would make `status` lie and `join`
    start a second copy of a running node.
    """
    try:
        pid_text, _, recorded_root = _pid_file(agent_id).read_text().partition("\n")
        pid = int(pid_text.strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return None
    from neo.memory.observer import _pid_cmdline

    # A stale file's pid can be reused by ANOTHER repository's node, so the
    # command line must name this node's repository, not just be a node.
    cmd = _pid_cmdline(pid) or ""
    root = recorded_root.strip()
    return pid if "neo.lattice" in cmd and root and f"--cwd {root}" in cmd else None


def _skew_message(agent_id: str, managed: Optional[dict]) -> Optional[str]:
    pid = _live_node_pid(agent_id)
    if pid is not None and (managed is None or managed.get("status") != "running"):
        return (f"CAR reports {agent_id} as "
                f"{managed.get('status') if managed else 'unregistered'}, but node pid {pid} "
                "is alive. The CAR client and daemon probably disagree on protocol "
                "version (update car-runtime or CarHost); refusing to act on CAR's view.")
    return None


def join(root: str, project: Optional[str] = None, *, work_context: Optional[dict] = None,
         project_ids: Optional[list[str]] = None, native_owner: Optional[dict] = None) -> dict:
    if native_owner is not None and work_context is None:
        return {"status": "error", "message": "Native owner requires explicit organization work context"}
    if work_context is not None:
        from neo.lattice_enrollment import operation
        return operation("join", root, project, work_context, project_ids, native_owner=native_owner)
    if project_ids is not None:
        return {"status": "error", "message": "Shared project approval requires an explicit organization work context"}
    from neo.memory.observer import _require_car_runtime

    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    root = repository_root(root) or root
    agents = _list_agents(car)
    if agents is None:
        return {"status": "error", "message": "could not list CAR's agents"}
    agent_id = agent_id_for(root)
    existing = next((a for a in agents if a.get("id") == agent_id), None)
    skew = _skew_message(agent_id, existing)
    if skew:
        return {"status": "error", "agent_id": agent_id, "message": skew}
    _left_marker(root).unlink(missing_ok=True)
    if daemon_has_lattice(daemon_url()) is False:
        return {"status": "error", "agent_id": agent_id, "message": (
            "this CAR daemon has no Lattice (lattice.*) yet; neo will join "
            "automatically once CAR is updated")}
    project = project or derive_project_name(root)
    try:
        car.agents_upsert(json.dumps(build_spec(root, project)))
        if existing and existing.get("status") == "running":
            # Restart so a changed spec (project, capabilities) takes effect.
            managed = json.loads(car.agents_restart(agent_id))
            state = "restarted"
        else:
            managed = json.loads(car.agents_start(agent_id))
            state = "started"
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "message": f"registering {agent_id} failed: {e}"}
    touch_seen(agent_id)
    return {"status": state, "agent_id": agent_id, "project": project, "root": root,
            "pid": managed.get("pid") if isinstance(managed, dict) else None}


def leave(root: str, *, work_context: Optional[dict] = None, native_owner: Optional[dict] = None) -> dict:
    if native_owner is not None and work_context is None:
        return {"status": "error", "message": "Native owner requires explicit organization work context"}
    if work_context is not None:
        from neo.lattice_enrollment import operation
        return operation("leave", root, None, work_context, native_owner=native_owner)
    from neo.memory.observer import _require_car_runtime

    root = repository_root(root) or root
    agent_id = agent_id_for(root)
    # Recorded first, so autostart honours the choice even if CAR is down.
    marker = _left_marker(root)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{root}\n")
    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "message": f"opt-out recorded; {e}"}
    agents = _list_agents(car)
    if agents is None:
        return {"status": "error", "message": "opt-out recorded; could not list CAR's agents"}
    existing = next((a for a in agents if a.get("id") == agent_id), None)
    skew = _skew_message(agent_id, existing)
    if skew:
        return {"status": "error", "agent_id": agent_id, "message": "opt-out recorded; " + skew}
    if existing is None:
        return {"status": "not_joined", "agent_id": agent_id}
    for op in (car.agents_stop, car.agents_remove):
        try:
            op(agent_id)
        except Exception as e:  # noqa: BLE001 — stop fails on a stopped agent
            logger.debug("%s(%s): %s", op.__name__, agent_id, e)
    return {"status": "left", "agent_id": agent_id}


def status(root: str, *, work_context: Optional[dict] = None, native_owner: Optional[dict] = None) -> dict:
    if native_owner is not None and work_context is None:
        return {"status": "error", "message": "Native owner requires explicit organization work context"}
    if work_context is not None:
        from neo.lattice_enrollment import operation
        return operation("status", root, None, work_context, native_owner=native_owner)
    from neo.memory.observer import _require_car_runtime

    try:
        car = _require_car_runtime()
    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    root = repository_root(root) or root
    agent_id = agent_id_for(root)
    agents = _list_agents(car)
    if agents is None:
        return {"status": "error", "agent_id": agent_id, "message": "could not list CAR's agents"}
    managed = next((a for a in agents if a.get("id") == agent_id), None)
    skew = _skew_message(agent_id, managed)
    if skew:
        return {"status": "unverified", "agent_id": agent_id, "message": skew}
    if managed is None:
        state = "left" if _left_marker(root).exists() else "not_joined"
        return {"status": state, "agent_id": agent_id}
    return {"status": managed.get("status", "unknown"), "agent_id": agent_id,
            "pid": managed.get("pid"), "root": _node_root(managed)}


def _websockets_available() -> bool:
    try:
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


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
    if not _websockets_available():
        # Exit non-zero so the supervisor marks the agent errored. Retrying a
        # connection that can never be made would look alive and answer no one.
        print("neo lattice needs the `websockets` package: pip install 'neo-reasoner[car]'",
              file=sys.stderr)
        return 3
    config = NodeConfig(
        root=root, project=project, agent_id=agent_id, token=token,
        url=daemon_url(), display_name=f"neo-{project}",
        capabilities=capabilities_for(root),
        work_context_json=os.environ.get("CAR_WORK_CONTEXT"),
    )
    node = LatticeNode(config)
    pid_file = _pid_file(agent_id)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{os.getpid()}\n{root}\n")
    touch_seen(agent_id)

    async def _main() -> None:
        import signal

        loop = asyncio.get_running_loop()

        def _on_signal() -> None:
            node.signalled = True
            node._spawn(node.leave())

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, _on_signal)
            except (NotImplementedError, RuntimeError):
                pass
        await node.run()

    started = time.monotonic()
    code = 0
    try:
        asyncio.run(_main())
    except WorkerDied as e:
        logger.error("answering worker died (%s); exiting for a supervised restart", e)
        code = 1
    logger.info("lattice node exiting after %.0fs, %d answered",
                time.monotonic() - started, node.answered)
    sys.stdout.flush()
    sys.stderr.flush()
    if node.recycle:
        # exec keeps the pid, so the supervisor sees the same agent; the old
        # socket closes with the old image and the new one re-attaches. The
        # credentials go back into the NEW image's environment only.
        env = dict(os.environ, CAR_AGENT_ID=agent_id, CAR_AGENT_TOKEN=token)
        os.execve(sys.executable, [sys.executable, *node_argv(root, project)], env)
    if node.signalled:
        # A signal mid-answer leaves the engine thread inside a model call;
        # a normal exit would join it and outlast the supervisor's SIGTERM
        # grace. The answer is abandoned either way.
        pid_file.unlink(missing_ok=True)
        os._exit(code)
    pid_file.unlink(missing_ok=True)
    return code


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
        if name in {"join", "leave", "status"}:
            p.add_argument("--work-context-file", default=None,
                           help="explicit original issuer/account/organization JSON; CAR verifies membership")
        if name == "join":
            p.add_argument("--project-id", action="append", default=None,
                           help="approve a canonical repo:<64 lowercase hex> for organization work (repeatable)")
        if name in {"join", "run"}:
            p.add_argument("--project", default=None,
                           help="player project name (default: the git remote's repo name)")
    for name in ("join", "leave", "status"):
        sub.choices[name].add_argument("--native-owner-config", type=Path, default=None)
    args = parser.parse_args(argv)
    root = os.path.abspath(args.cwd or os.getcwd())

    if args.action == "run":
        logging.basicConfig(
            level=os.environ.get("NEO_LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        return _run_foreground(root, args.project or derive_project_name(root))

    context = None
    if args.work_context_file is not None:
        try:
            from neo.lattice_enrollment import work_context
            context = work_context(json.loads(Path(args.work_context_file).read_text()))
        except (OSError, ValueError) as error:
            print(json.dumps({"status": "error", "message": str(error)}))
            return 1
    native_owner = None
    if args.native_owner_config is not None:
        try:
            native_owner = json.loads(args.native_owner_config.read_text())
        except (OSError, ValueError) as error:
            print(json.dumps({"status": "error", "message": str(error)}))
            return 1
    result = {"join": lambda: join(root, args.project, work_context=context, project_ids=args.project_id, native_owner=native_owner),
              "leave": lambda: leave(root, work_context=context, native_owner=native_owner),
              "status": lambda: status(root, work_context=context, native_owner=native_owner)}[args.action]()
    print(json.dumps(result, indent=2))
    return 1 if result.get("status") == "error" else 0


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
