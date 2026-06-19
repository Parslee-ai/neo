"""
Conversation-mined issue diagnostic.

Surfaces *recurring frictions* mined from the same multi-source transcript
episodes the memory ingester consumes (Claude Code / Codex / CAR), but as a
read-only VIEW: it never extracts facts, never calls an LM, and never touches
the ingester watermark. Re-running it is idempotent and mutates nothing.

Pipeline (all LM-free in v1):

    collect episodes  ->  tag per-episode signals  ->  cluster by ask embedding
    ->  gate (>= min_cluster members, >= 2 distinct sessions, friction present,
        verbatim evidence)  ->  categorize + score  ->  ranked Issue list

Signals are derived from ``Episode`` fields (``errors``, ``assistant_text``).
The clarification vocabulary is lifted from :mod:`neo.prompt.analyzer`, but we
operate on ``Episode`` objects (the watermarked, multi-tool substrate) rather
than the analyzer's Claude-only message dicts.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Callable, Optional

from neo.math_utils import cluster_by_similarity

logger = logging.getLogger(__name__)

# Same threshold the REVIEW->PATTERN synthesis uses (store.SYNTHESIS_SIMILARITY);
# kept as a local constant so this module imports nothing heavy at import time.
CLUSTER_SIMILARITY = 0.85

# Issue categories, mapped to the harness failure taxonomy ("most agent failures
# are configuration failures: a missing tool, a vague rule, an absent guardrail").
CATEGORY_MISSING_TOOL = "missing-tool"
CATEGORY_ABSENT_GUARDRAIL = "absent-guardrail"
CATEGORY_VAGUE_RULE = "vague-rule"

# Confidence weights: cluster size / session spread / recency.
_W_SIZE = 0.4
_W_SPREAD = 0.3
_W_RECENCY = 0.3
_RECENCY_HALFLIFE_DAYS = 14.0

# Clarification/confusion phrases an assistant uses when the ask was
# under-specified (lifted from prompt.analyzer.CONFUSION_PATTERNS).
_CLARIFICATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"could you clarify",
        r"can you clarify",
        r"i'm not sure",
        r"i am not sure",
        r"i need more information",
        r"can you provide more context",
        r"what do you mean by",
        r"please specify",
        r"could you be more specific",
        r"it's unclear",
        r"i cannot determine",
        r"which (?:file|one|approach|option) (?:did you mean|do you want)",
    )
]

# Claude Code's own tool-protocol guards surface as <tool_use_error> envelopes
# (read-before-edit, replace_all, file-modified-since-read, sleep-blocked, …).
# They are agent/harness churn, not project issues, so they never count as
# friction.
_HARNESS_ENVELOPE_RE = re.compile(r"^\s*<tool_use_error>", re.IGNORECASE)

# Generic command-outcome banners carry no actionable error class on their own
# (a bare "Exit code 2" says nothing about what failed). When an error is only
# banners, it is dropped; when a banner precedes real output (Codex emits
# "Process exited with code 1\n<traceback>"), the banner is skipped and the
# substantive line is used.
_GENERIC_BANNER_RE = re.compile(
    r"^\s*(?:process )?exit(?:ed with)? code \d+"
    r"|^\s*command (?:timed out|failed)$"
    r"|^\s*\[request interrupted",
    re.IGNORECASE,
)

# A non-zero exit captures the command's whole output, which is often not an
# error message at all (section headers, diffs, code being edited). Require a
# line to look like a diagnostic, not merely contain the substring "error" — a
# type like `Result<T, WorkflowError>` is code, not an error occurrence. So we
# match error words only at diagnostic positions (prefix-with-colon/paren, or a
# typed exception followed by a colon), lint codes, and a curated set of
# standalone failure phrases that rarely appear as code identifiers.
_ERROR_SIGNAL_RE = re.compile(
    # "error:", "fatal:", "panic(", "traceback (" as a line/segment prefix
    r"(?:^|[\s>|\]])(?:error|errno|fatal|panic|exception|traceback)\b\s*[:(]"
    # typed exception carrying a message: "TypeError: …", "ModuleNotFoundError: …"
    r"|\b\w*(?:error|exception)\b\s*:"
    # linter / compiler codes: E402, C0301, …
    r"|\b[A-Za-z]\d{3,}\b"
    # curated standalone failure phrases
    r"|command not found|no such file|not a git repository|would be overwritten"
    r"|permission denied|access denied|segmentation fault|stack overflow"
    r"|connection (?:refused|reset|timed out)|operation not permitted"
    r"|cannot find|could not|couldn'?t|unable to|failed to|not recognized"
    r"|undefined (?:reference|symbol|variable|method)"
    r"|(?:tests?|build|compilation|assertion) failed",
    re.IGNORECASE,
)

# Error shapes that point specifically at a missing/unavailable *tool* (vs. a
# missing file or a logic bug). Kept narrow: "no such file or directory" is a
# missing path, not a missing tool, so it is deliberately excluded and falls to
# absent-guardrail.
_MISSING_TOOL_RE = re.compile(
    r"command not found|: not found$|unknown command"
    r"|is not recognized|executable file not found|no such command",
    re.IGNORECASE,
)

# How many evidence spans to attach per issue.
_MAX_EVIDENCE = 3
# Window (chars) captured around a clarification match for the evidence span.
_CLARIFICATION_SPAN_CHARS = 160


@dataclass
class IssueEvidence:
    """A single, provable citation for an issue.

    ``span`` is a verbatim substring of the source episode (Stage-C discipline:
    provable evidence, not a paraphrase).
    """

    session_id: str
    timestamp: str
    span: str


@dataclass
class Issue:
    """A recurring friction surfaced from transcript history."""

    title: str
    category: str
    confidence: float
    evidence: list[IssueEvidence]
    session_count: int
    member_count: int
    suggested_rule: Optional[str] = None  # filled by --suggest-rules (post-v1)


@dataclass
class EpisodeSignals:
    """Per-episode friction signals, derived purely from episode fields."""

    has_tool_error: bool = False
    error_class: Optional[str] = None
    asst_asked_clarification: bool = False

    @property
    def has_friction(self) -> bool:
        return self.has_tool_error or self.asst_asked_clarification


def _substantive_error_line(text: str) -> Optional[str]:
    """First error-like line in an error blob, or None.

    Skips generic command-outcome banners ("Exit code 2", "command timed out")
    and non-error command output (section headers, diffs), returning the first
    line that actually looks like an error. So a banner+traceback (Codex:
    "Process exited with code 1\\nModuleNotFoundError: …") yields the traceback
    line, while a non-zero exit whose output is just "=== section ===" yields
    None and is dropped as friction.
    """
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or _GENERIC_BANNER_RE.match(s):
            continue
        if _ERROR_SIGNAL_RE.search(s):
            return s
    return None


def _episode_errors(ep) -> list[str]:
    """Project-relevant error strings for an episode.

    Drops (a) empty/whitespace entries, (b) Claude Code ``<tool_use_error>``
    tool-protocol guards (agent/harness churn, not project issues), and
    (c) banner-only outcomes (a bare exit code says nothing about what failed).
    What remains is substantive enough to anchor an issue.
    """
    out: list[str] = []
    for e in getattr(ep, "errors", None) or []:
        if not e or not e.strip():
            continue
        if _HARNESS_ENVELOPE_RE.match(e):
            continue
        if _substantive_error_line(e) is None:
            continue
        out.append(e)
    return out


def _first_line(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return text.splitlines()[0].strip()


def _normalize_error(text: str) -> Optional[str]:
    """Reduce an error message to a stable class key for grouping.

    Strips the volatile bits (line numbers, hex addresses, absolute paths) so
    the same error recurring with different coordinates groups together. Uses
    the first substantive (non-banner) line.
    """
    line = _substantive_error_line(text)
    if not line:
        return None
    s = re.sub(r"0x[0-9a-fA-F]+", "", line)
    s = re.sub(r"(?:/[^\s:'\"]+)+", "", s)  # absolute paths
    s = re.sub(r"\b\d+\b", "", s)  # line numbers / counts
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s[:80] or None


def tag_signals(ep) -> EpisodeSignals:
    """Derive friction signals from one ``Episode``. Pure; no LM, no I/O."""
    errors = _episode_errors(ep)
    has_tool_error = bool(errors)
    error_class = _normalize_error(errors[0]) if has_tool_error else None

    clarified = any(
        turn and any(p.search(turn) for p in _CLARIFICATION_PATTERNS)
        for turn in (getattr(ep, "assistant_text", None) or [])
    )

    return EpisodeSignals(
        has_tool_error=has_tool_error,
        error_class=error_class,
        asst_asked_clarification=clarified,
    )


def _clarification_span(ep) -> Optional[str]:
    """A verbatim window around the first clarification phrase, or None.

    Matches within a single assistant turn so the returned span is a true
    substring of one source field (not a join-reconstructed string).
    """
    for turn in getattr(ep, "assistant_text", None) or []:
        if not turn:
            continue
        for pat in _CLARIFICATION_PATTERNS:
            m = pat.search(turn)
            if m:
                start = max(0, m.start() - 40)
                end = min(len(turn), m.end() + (_CLARIFICATION_SPAN_CHARS - 40))
                return turn[start:end].strip()
    return None


def _evidence_span(ep, sig: EpisodeSignals) -> Optional[str]:
    """Build a verbatim evidence span for a frictional episode, or None."""
    if sig.has_tool_error:
        errors = _episode_errors(ep)
        if errors:
            line = _substantive_error_line(errors[0]) or _first_line(errors[0])
            if line:
                return line[:_CLARIFICATION_SPAN_CHARS]
    if sig.asst_asked_clarification:
        return _clarification_span(ep)
    return None


def _episode_epoch_safe(ep) -> Optional[float]:
    from neo.memory.transcript import _episode_epoch

    return _episode_epoch(getattr(ep, "timestamp", "") or "")


def _representative_topic(members: list) -> str:
    """A short topic label from the shortest substantive ask in the cluster."""
    asks = [(_first_line(getattr(m, "ask", "")) or "") for m in members]
    asks = [a for a in asks if a]
    if not asks:
        return "(unknown)"
    topic = min(asks, key=len)
    return topic[:60] + ("…" if len(topic) > 60 else "")


def _categorize(members: list, sigs: list[EpisodeSignals]) -> tuple[str, str]:
    """Return (category, title) for a frictional cluster.

    Precedence: missing-tool > absent-guardrail > vague-rule (most structural
    signal wins, so one cluster maps to exactly one category).
    """
    tool_errs = [s for s in sigs if s.has_tool_error]
    topic = _representative_topic(members)

    if tool_errs:
        if any(_MISSING_TOOL_RE.search(s.error_class or "") for s in tool_errs):
            label = _most_common_error(tool_errs) or topic
            return CATEGORY_MISSING_TOOL, f"Repeated tool failure: {label}"
        label = _most_common_error(tool_errs) or topic
        return CATEGORY_ABSENT_GUARDRAIL, f"Recurring error: {label}"

    return CATEGORY_VAGUE_RULE, f"Repeated clarification: {topic}"


def _most_common_error(sigs: list[EpisodeSignals]) -> Optional[str]:
    counts: dict[str, int] = {}
    for s in sigs:
        if s.error_class:
            counts[s.error_class] = counts.get(s.error_class, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


def _confidence(member_count: int, session_count: int, members: list, now: float) -> float:
    size_term = min(1.0, member_count / 6.0)
    spread_term = min(1.0, session_count / 3.0)

    epochs = [e for m in members if (e := _episode_epoch_safe(m)) is not None]
    if epochs:
        age_days = max(0.0, (now - max(epochs)) / 86400.0)
        # True half-life: term == 0.5 at exactly _RECENCY_HALFLIFE_DAYS.
        recency_term = math.exp(-age_days * math.log(2) / _RECENCY_HALFLIFE_DAYS)
    else:
        recency_term = 0.5  # unknown age -> neutral

    score = _W_SIZE * size_term + _W_SPREAD * spread_term + _W_RECENCY * recency_term
    return round(score, 2)


def detect_issues(
    episodes: list,
    signals: list[EpisodeSignals],
    embeddings: list,
    *,
    min_cluster: int,
    now: float,
) -> list[Issue]:
    """Cluster episodes by ask embedding and emit gated, scored issues.

    Gate per cluster: >= ``min_cluster`` members, >= 2 distinct sessions,
    >= 2 members carrying a friction signal, and at least one verbatim
    evidence span. Clusters that are merely recurring topics with no friction
    are dropped (conservative: a wrong issue wastes scarce human attention).
    """
    indices = list(range(len(episodes)))
    clusters = cluster_by_similarity(
        indices, embed_fn=lambda i: embeddings[i], threshold=CLUSTER_SIMILARITY
    )

    issues: list[Issue] = []
    for cluster in clusters:
        if len(cluster) < min_cluster:
            continue

        members = [episodes[i] for i in cluster]
        sigs = [signals[i] for i in cluster]

        sessions = {getattr(m, "session_id", "") for m in members}
        if len(sessions) < 2:
            continue

        frictional = [(m, s) for m, s in zip(members, sigs) if s.has_friction]
        if len(frictional) < 2:
            continue

        # Evidence: most recent frictional members first, verbatim spans only.
        frictional.sort(key=lambda ms: _episode_epoch_safe(ms[0]) or 0.0, reverse=True)
        evidence: list[IssueEvidence] = []
        for m, s in frictional:
            span = _evidence_span(m, s)
            if not span:
                continue
            evidence.append(
                IssueEvidence(
                    session_id=getattr(m, "session_id", ""),
                    timestamp=getattr(m, "timestamp", "") or "",
                    span=span,
                )
            )
            if len(evidence) >= _MAX_EVIDENCE:
                break
        if not evidence:
            continue

        category, title = _categorize(members, sigs)
        confidence = _confidence(len(members), len(sessions), members, now)
        issues.append(
            Issue(
                title=title,
                category=category,
                confidence=confidence,
                evidence=evidence,
                session_count=len(sessions),
                member_count=len(members),
            )
        )

    issues.sort(key=lambda iss: iss.confidence, reverse=True)
    return issues


def _embed_text_for(ep, embed: Optional[Callable]) -> Optional[object]:
    """Embed the ask, sharpened with the first error line when present.

    Terse asks ("fix it") cluster poorly; folding in the error line gives the
    clusterer more signal for frictional episodes.
    """
    if embed is None:
        return None
    text = getattr(ep, "ask", "") or ""
    errors = _episode_errors(ep)
    if errors:
        line = _substantive_error_line(errors[0]) or _first_line(errors[0])
        if line:
            text = f"{text}\n{line}" if text else line
    if not text:
        return None
    try:
        return embed(text)
    except Exception:
        return None


def find_issues(
    store,
    *,
    since_seconds: Optional[float] = None,
    min_cluster: int = 3,
    sources: Optional[list] = None,
    now: Optional[float] = None,
) -> list[Issue]:
    """Collect recent episodes across all transcript sources and detect issues.

    Read-only: collects episodes within the ``since_seconds`` window but never
    consults or writes the ingester watermark, so it is fully decoupled from
    fact admission and safe to run repeatedly.
    """
    import time

    from neo.memory.transcript import (
        CarSource,
        ClaudeCodeSource,
        CodexSource,
        _episode_epoch,
    )

    if now is None:
        now = time.time()

    root = getattr(store, "codebase_root", None)
    if sources is None:
        sources = [ClaudeCodeSource(root), CodexSource(root), CarSource()]

    cutoff = (now - since_seconds) if since_seconds else None

    episodes: list = []
    for src in sources:
        try:
            collected = src.collect_episodes()
        except Exception:
            continue
        for ep in collected:
            if not ep.is_substantive:
                continue
            if cutoff is not None:
                ts = _episode_epoch(getattr(ep, "timestamp", "") or "")
                if ts is not None and ts < cutoff:
                    continue
            episodes.append(ep)

    if len(episodes) < min_cluster:
        return []

    embed = getattr(store, "_embed_text", None)
    embeddings = [_embed_text_for(ep, embed) for ep in episodes]
    # Distinguish a genuine "no recurring frictions" from an embedder that
    # never loaded (offline / missing model) — the latter silently drops every
    # episode in clustering and would otherwise look like a clean result.
    if all(e is None for e in embeddings):
        logger.warning(
            "memory issues: no embeddings produced for %d episode(s); "
            "cannot cluster (embedder unavailable?)",
            len(episodes),
        )
        return []
    signals = [tag_signals(ep) for ep in episodes]

    return detect_issues(
        episodes, signals, embeddings, min_cluster=min_cluster, now=now
    )
