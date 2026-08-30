"""
Outcome-based learning for Neo.

Two learning modes:
1. Session-based: Tracks what the user did after neo's suggestions (between invocations)
2. History-based: Ingests git commit history to learn from past code evolution

Session data is persisted to ~/.neo/sessions/ so outcomes can be detected
across invocations. History watermarks are persisted to avoid re-ingesting.
"""

import datetime
import enum
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

from neo.memory.io_utils import atomic_write_json

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path.home() / ".neo" / "sessions"

# Cap independent outcomes per session to avoid flooding facts with noise.
# This is the first layer of defense; store.py's MAX_INDEPENDENT_FACTS (50)
# caps the total across sessions. Together: 5/session * ~10 sessions before
# cap kicks in, then oldest are invalidated.
MAX_INDEPENDENT_OUTCOMES = 5
CODE_OVERLAP_ACCEPTED_THRESHOLD = 0.8

# Durable suggestion ledger + transcript outcome mining.
# The session log is consumed (cleared) by git-based outcome detection on the
# next invocation, so it can't be relied on for asynchronous, transcript-driven
# correlation. The ledger is an append-only record of linked suggestions that
# the async miner (TranscriptIngester.mine_suggestion_outcomes) drains on its
# own cadence, independent of how soon neo is re-invoked.
OUTCOME_CORRELATION_WINDOW_SECONDS = 2 * 3600  # episode must follow a suggestion within 2h
OUTCOME_CORRELATION_SIMILARITY = 0.6           # min cosine(description, episode) to call it a match
MAX_MINED_OUTCOMES_PER_CYCLE = 20              # bound observer work + confidence churn per cycle
SUGGESTION_LEDGER_TTL_DAYS = 30                # ledger entries older than this are compacted away


class OutcomeType(enum.Enum):
    """Classification of what the user did after a neo suggestion."""
    ACCEPTED = "accepted"       # User applied the suggestion (diff overlap > 0.3)
    MODIFIED = "modified"       # User changed the file differently (overlap <= 0.3)
    REGRESSION = "regression"   # Later evidence showed an accepted change was harmful
    UNVERIFIED = "unverified"   # File changed but no diff to compare
    INDEPENDENT = "independent" # User changed a file neo didn't suggest


class OutcomeIndicator(enum.Enum):
    """Trajectory-Memory semantic indicator (paper 2603.10600 §4).

    Independent from OutcomeType (which describes *event shape* — did the
    diff land, did the user edit elsewhere). OutcomeIndicator describes
    the *semantic shape* of what happened — failure, recovery from a
    prior failure, inefficient-but-correct, or clean success.
    """
    FAILURE = "failure"           # error/test-fail/exception signals
    RECOVERY = "recovery"         # failure followed by clean completion
    INEFFICIENCY = "inefficiency" # works but repeats / batches poorly
    SUCCESS = "success"           # clean completion, no error trail


_FAILURE_RE = re.compile(
    r"\b(error|exception|traceback|fail(?:ed|ure)?|assert(?:ion)?error|"
    r"raised|panic|aborted|invalid|missing|undefined|broken|crash)\b",
    re.IGNORECASE,
)
_RECOVERY_HINTS = ("fix", "fixed", "fixes", "resolve", "resolved", "patch",
                    "corrected", "recovers", "recover", "fall back", "fallback")
_INEFFICIENCY_RE = re.compile(
    r"\b(loop[ _-]?in[ _-]?loop|n\+1|nplus(?:one)?|repeat(?:ed|edly)?|"
    r"redundant|slow|inefficient|quadratic)\b",
    re.IGNORECASE,
)


class CodeOutcome(enum.Enum):
    """LessonL objective code-outcome categorizer (paper 2505.23946 §3).

    For code suggestions with static-check / compile / test signals,
    bucket the observed result deterministically. Distinct from
    OutcomeIndicator (which works on prose) and OutcomeType (event shape).
    """
    SPEED_UP = "speed_up"
    SLOW_DOWN = "slow_down"
    FUNCTIONAL_INCORRECTNESS = "functional_incorrectness"
    SYNTAX_ERROR = "syntax_error"
    UNKNOWN = "unknown"  # not enough signal


_SYNTAX_ERR_RE = re.compile(
    r"\b(SyntaxError|ParseError|parse error|unexpected token|"
    r"unterminated|invalid syntax)\b",
    re.IGNORECASE,
)
_FUNCTIONAL_ERR_RE = re.compile(
    r"\b(assertion ?error|expected .* got|test (failed|fail)|"
    r"wrong (answer|output)|incorrect|FAIL(?:ED)?)\b",
    re.IGNORECASE,
)


def classify_code_outcome(
    *,
    diagnostics: Optional[list[dict]] = None,
    runtime_log: str = "",
    speedup_ratio: Optional[float] = None,
) -> CodeOutcome:
    """Deterministic LessonL outcome bucket from compiler/runtime evidence.

    Order matters: explicit syntax errors first (compiler-shaped),
    functional incorrectness second (runtime/test-shaped), then
    speed signals if a measured ratio is supplied. Tests are
    deliberately NOT included for the incorrectness path (paper §3
    rationale: keeps lessons from overfitting to specific tests).
    """
    blob_parts = [runtime_log]
    for d in diagnostics or []:
        msg = d.get("message") or d.get("text") or ""
        blob_parts.append(str(msg))
    blob = "\n".join(p for p in blob_parts if p)

    if _SYNTAX_ERR_RE.search(blob):
        return CodeOutcome.SYNTAX_ERROR
    if _FUNCTIONAL_ERR_RE.search(blob):
        return CodeOutcome.FUNCTIONAL_INCORRECTNESS

    if speedup_ratio is not None:
        if speedup_ratio > 1.0:
            return CodeOutcome.SPEED_UP
        if speedup_ratio < 1.0:
            return CodeOutcome.SLOW_DOWN

    return CodeOutcome.UNKNOWN


def classify_outcome_indicator(
    *,
    diff_text: str = "",
    issues_found: Optional[list[str]] = None,
    error_trace: str = "",
    prior_failure: bool = False,
    reasoning_steps: Optional[list[str]] = None,
) -> OutcomeIndicator:
    """Deterministic semantic classification (paper 2603.10600 §4-5).

    Pure pattern matching on the action log — no LLM. Order matters:

      1. error_trace OR explicit "error|exception|fail" in diff/issues
         → FAILURE  (unless prior_failure is True AND we now see clean
         completion in reasoning_steps → RECOVERY)
      2. INEFFICIENCY tokens in diff/reasoning → INEFFICIENCY
      3. Otherwise → SUCCESS
    """
    blob_parts = [diff_text, error_trace]
    blob_parts.extend(issues_found or [])
    blob_parts.extend(reasoning_steps or [])
    blob = " ".join(p for p in blob_parts if p)

    has_failure = bool(error_trace.strip()) or bool(_FAILURE_RE.search(blob))
    if has_failure:
        # Recovery only when we'd previously seen failure AND now the
        # current message looks like a fix/recovery — distinguishes
        # "we landed a fix" from "still broken".
        if prior_failure and any(h in blob.lower() for h in _RECOVERY_HINTS):
            return OutcomeIndicator.RECOVERY
        return OutcomeIndicator.FAILURE

    if _INEFFICIENCY_RE.search(blob):
        return OutcomeIndicator.INEFFICIENCY

    return OutcomeIndicator.SUCCESS


@dataclass
class SuggestionRecord:
    """Minimal record of a suggestion for outcome matching."""
    file_path: str
    description: str
    confidence: float = 0.0


@dataclass
class SessionRecord:
    """Persisted state from a neo invocation, used for outcome detection on next run."""
    timestamp: float = 0.0
    codebase_root: str = ""
    project_id: str = ""
    prompt: str = ""
    suggestions: list[dict] = field(default_factory=list)
    suggestion_fact_ids: dict[str, str] = field(default_factory=dict)
    # Architectural snapshot at save time. Empty dict when computation
    # failed or codebase_root was unavailable.
    architecture_snapshot: dict = field(default_factory=dict)
    learning_episode_id: str = ""
    repository_revision: str = ""
    retrieved_fact_ids: list[str] = field(default_factory=list)
    used_fact_ids: list[str] = field(default_factory=list)
    # How far this record has already been scanned for outcomes. Set when a
    # partially-resolved session is retained; queries then start here rather
    # than at `timestamp`. Without it a retained record re-scans the whole
    # window since the suggestion on every invocation, forever, because the
    # anchor never advances — measured at ~8s per invocation, persisting for
    # the full 14-day TTL.
    scanned_through: float = 0.0
    # Paths this session suggested that already produced an outcome. Retained
    # even though their suggestions are dropped, so the independent-change
    # detector still recognizes them as ours. Otherwise an ACCEPTED suggestion
    # comes back as INDEPENDENT — "user changed a file neo didn't suggest" —
    # naming a file neo suggested and the user demonstrably applied.
    resolved_paths: list[str] = field(default_factory=list)


@dataclass
class Outcome:
    """A detected outcome from comparing suggestions to actual changes."""
    outcome_type: OutcomeType
    file_path: str
    diff_summary: str = ""  # actual git diff content (truncated)
    suggestion_description: str = ""  # empty for independent changes
    suggestion_confidence: float = 0.0
    suggestion_id: str = ""
    learning_episode_id: str = ""
    repository_revision: str = ""
    retrieved_fact_ids: list[str] = field(default_factory=list)
    used_fact_ids: list[str] = field(default_factory=list)
    candidate_id: str = ""
    candidate_subject: str = ""
    candidate_body: str = ""
    candidate_kind: str = "pattern"


def normalize_suggestion_path(path: str, codebase_root: Optional[str]) -> str:
    """Normalize a suggestion's file path to the form outcome matching uses.

    Handles three cases:
    1. True absolute paths under codebase_root -> relative
    2. Bare leading slash under codebase_root (/src/bar.py) -> src/bar.py
    3. Everything else unchanged

    Module-level and shared: anything that predicts whether a suggestion can be
    verified MUST resolve paths the same way attribution does, or it will gate
    on a different answer than the code it is predicting.
    """
    if not path:
        return path

    if codebase_root:
        try:
            p = Path(path)
            root = Path(codebase_root)
            if p.is_absolute():
                return str(p.relative_to(root))
        except (ValueError, TypeError):
            pass

        # Strip bare leading slash if the result exists relative to codebase_root
        # (common in suggestion file_path values like "/src/foo.py")
        if path.startswith("/"):
            stripped = path.lstrip("/")
            candidate = Path(codebase_root) / stripped
            if candidate.exists() or candidate.parent.exists():
                return stripped

    return path


def suggestion_is_verifiable(
    file_path: str, has_change_text: bool, codebase_root: Optional[str]
) -> bool:
    """Could a downstream git diff ever confirm this suggestion?

    Promotion to a durable fact is gated on a git-verified outcome, which is
    detected by diffing the suggested path. So a suggestion qualifies only when
    the path is one a diff could name AND there is code/diff text to compare the
    change against — without the latter the outcome can only ever be UNVERIFIED.

    A path that doesn't exist yet still qualifies when its parent directory does
    and it sits inside the repo: proposing a *new* file is a real suggestion, and
    once committed it appears in ``git log --name-only``. What this rejects is
    the path a model invents for an advisory answer, whose parent doesn't exist
    or which points outside the repo ("/review/commit-<sha>.md", "/dev/null").
    Directories are rejected — a diff never names one.

    Resolution goes through ``normalize_suggestion_path`` deliberately: an
    independent reimplementation missed bare-leading-slash paths and so called
    two genuinely promotable candidates unverifiable.
    """
    if not file_path or not has_change_text:
        return False
    try:
        normalized = normalize_suggestion_path(file_path, codebase_root)
        path = Path(normalized)
        if not path.is_absolute():
            if not codebase_root:
                return False
            path = Path(codebase_root) / normalized
        if path.is_dir():
            return False
        if path.is_file():
            return True
        # Doesn't exist yet: only a plausible new file inside the repo.
        if not codebase_root:
            return False
        root = Path(codebase_root).resolve()
        parent = path.parent.resolve()
        return parent.is_dir() and (parent == root or parent.is_relative_to(root))
    except (OSError, ValueError):
        return False


class OutcomeTracker:
    """Detects what the user actually did after neo's suggestions.

    On each invocation:
    1. Load previous session (if any) for this project
    2. Run git diff to see what changed since last session
    3. Match changes against previous suggestions
    4. Return outcomes for fact creation
    5. Save current session for next invocation
    """

    def __init__(self, codebase_root: Optional[str] = None, project_id: str = ""):
        self.codebase_root = codebase_root
        self.project_id = project_id
        self._session_path = self._get_session_path()
        self._session_log_path = self._get_session_log_path()
        self._suggestion_ledger_path = self._get_suggestion_ledger_path()
        # Sessions still awaiting user action after the last collect_outcomes.
        # Lets a caller that inspected with clear_processed=False consume the
        # log afterwards without re-running git or nuking pending work.
        self._last_pending: list[SessionRecord] = []
        # Identities of the sessions that collect_outcomes actually read. The
        # rewrite keeps anything NOT in this set, so a session appended by a
        # peer process after the read survives instead of being erased.
        self._last_loaded_keys: set[tuple] = set()

    def _get_session_path(self) -> Optional[Path]:
        if not self.project_id:
            return None
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        return SESSIONS_DIR / f"session_{self.project_id}.json"

    def _get_session_log_path(self) -> Optional[Path]:
        """Path for the append-only session log that accumulates across invocations."""
        if not self.project_id:
            return None
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        return SESSIONS_DIR / f"session_log_{self.project_id}.jsonl"

    def _get_suggestion_ledger_path(self) -> Optional[Path]:
        """Path for the durable, append-only suggestion ledger.

        Distinct from the session log: the session log is consumed by
        git-based outcome detection, while the ledger persists until the
        transcript outcome miner drains it (or it ages out), so asynchronous
        correlation never races the request path.
        """
        if not self.project_id:
            return None
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        return SESSIONS_DIR / f"suggestion_ledger_{self.project_id}.jsonl"

    def append_suggestion_ledger(
        self, suggestions: list, prompt: str,
        suggestion_fact_ids: Optional[dict[str, str]] = None,
    ) -> None:
        """Append linked suggestions to the durable ledger for later mining.

        Only suggestions that carry a linked ``fact_id`` are recorded — the
        miner can only reinforce a fact it can resolve. Best-effort: a write
        failure must never break the request path.
        """
        path = self._suggestion_ledger_path
        if not path or not suggestion_fact_ids:
            return
        ts = time.time()
        rows = []
        for s in suggestions:
            file_path = getattr(s, "file_path", "")
            fid = suggestion_fact_ids.get(file_path) if file_path else None
            if not fid:
                continue
            rows.append({
                "id": f"{ts:.6f}:{fid}:{file_path}",
                "ts": ts,
                "project_id": self.project_id,
                "prompt": prompt[:200],
                "file_path": file_path,
                "description": getattr(s, "description", "")[:500],
                "confidence": getattr(s, "confidence", 0.0),
                "fact_id": fid,
            })
        if not rows:
            return
        try:
            with open(path, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        except OSError as e:
            logger.warning(f"Failed to append suggestion ledger: {e}")

    def load_suggestion_ledger(self) -> list[dict]:
        """Return all current ledger entries (skipping any corrupt lines)."""
        path = self._suggestion_ledger_path
        if not path or not path.exists():
            return []
        entries: list[dict] = []
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning(f"Failed to read suggestion ledger: {e}")
        return entries

    def compact_suggestion_ledger(
        self, *, drop_ids: Optional[set] = None,
        max_age_days: int = SUGGESTION_LEDGER_TTL_DAYS,
    ) -> None:
        """Rewrite the ledger, dropping mined/expired entries to keep it bounded.

        Load→filter→tmp→replace. The replace is atomic (no corruption), but this
        is unlocked: a save_session append (request process) landing between the
        load and the replace is lost. That's acceptable here — the mined signal
        is weak and plentiful, so losing the occasional suggestion is harmless;
        not worth a cross-process file lock.
        """
        path = self._suggestion_ledger_path
        if not path or not path.exists():
            return
        drop_ids = drop_ids or set()
        cutoff = time.time() - max_age_days * 86400
        kept = [
            e for e in self.load_suggestion_ledger()
            if e.get("id") not in drop_ids and float(e.get("ts", 0) or 0) >= cutoff
        ]
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp, "w") as f:
                for e in kept:
                    f.write(json.dumps(e) + "\n")
            tmp.replace(path)
        except OSError as e:
            logger.warning(f"Failed to compact suggestion ledger: {e}")

    def save_session(
        self,
        suggestions: list,
        prompt: str,
        suggestion_fact_ids: Optional[dict[str, str]] = None,
        *,
        learning_episode_id: str = "",
        repository_revision: str = "",
        retrieved_fact_ids: Optional[list[str]] = None,
        used_fact_ids: Optional[list[str]] = None,
        candidates_by_suggestion: Optional[dict[str, dict]] = None,
    ) -> None:
        """Persist current session for outcome detection on next run.

        Args:
            suggestions: List of CodeSuggestion objects from current invocation.
            prompt: The user's prompt.
            suggestion_fact_ids: Mapping of file_path -> fact_id for linking outcomes.
        """
        if not self._session_path:
            logger.warning(
                "save_session: skipped — no project_id (codebase_root not set). "
                "Memory will NOT persist. Pass --cwd or working_directory in JSON input."
            )
            return

        records = []
        candidates_by_suggestion = candidates_by_suggestion or {}
        for s in suggestions:
            file_path = getattr(s, "file_path", "")
            if not file_path or file_path in ("/", "N/A"):
                continue
            suggestion_id = getattr(s, "suggestion_id", "")
            candidate = candidates_by_suggestion.get(suggestion_id, {})
            records.append({
                "suggestion_id": suggestion_id,
                "file_path": file_path,
                "description": getattr(s, "description", "")[:500],
                "confidence": getattr(s, "confidence", 0.0),
                "suggested_diff": getattr(s, "unified_diff", "")[:2000],
                "suggested_code": getattr(s, "code_block", "")[:4000],
                "candidate_id": candidate.get("candidate_id", ""),
                "candidate_subject": candidate.get("subject", ""),
                "candidate_body": candidate.get("body", ""),
                "candidate_kind": candidate.get("kind", "pattern"),
            })

        session = SessionRecord(
            timestamp=time.time(),
            codebase_root=self.codebase_root or "",
            project_id=self.project_id,
            prompt=prompt[:200],
            suggestions=records,
            suggestion_fact_ids=suggestion_fact_ids or {},
            architecture_snapshot=self._snapshot_architecture(),
            learning_episode_id=learning_episode_id,
            repository_revision=repository_revision,
            retrieved_fact_ids=list(retrieved_fact_ids or []),
            used_fact_ids=list(used_fact_ids or []),
        )

        try:
            atomic_write_json(self._session_path, asdict(session), indent=2)

            # Also append to session log so we don't lose prior sessions.
            # Locked: `_rewrite_session_log` replaces this file wholesale, so an
            # unlocked append can land between that rewrite's re-read and its
            # `os.replace` and vanish. The lock is held for one line of I/O.
            log_path = self._session_log_path
            if log_path:
                from neo.memory.store import scope_file_lock

                with scope_file_lock(log_path):
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(session)) + "\n")
                        f.flush()
                        os.fsync(f.fileno())

            # Durable ledger for async transcript outcome mining (survives the
            # session-log clearing done by git-based outcome detection).
            self.append_suggestion_ledger(suggestions, prompt, suggestion_fact_ids)

            logger.info(
                f"Saved session: {len(records)} suggestion(s), "
                f"{len(suggestion_fact_ids or {})} linked fact(s)"
            )
        except OSError as e:
            logger.warning(f"Failed to save session: {e}")

    def _snapshot_architecture(self) -> dict:
        """Capture an ArchSnapshot of the codebase. Always best-effort."""
        if not self.codebase_root:
            return {}
        try:
            from neo.architecture_metrics import compute
            return compute(self.codebase_root).to_dict()
        except Exception as exc:
            logger.debug("architecture snapshot failed (non-fatal): %s", exc)
            return {}

    def compute_arch_delta(self):
        """Return an ArchDelta for the oldest unprocessed session, or None.

        We use the oldest session's snapshot as the baseline because it
        captures the longest-running drift across the batch we're about to
        process. The current state is computed fresh.

        Returns None when no baseline exists (first run, missing snapshot,
        or computation failure on either side). Never raises.
        """
        try:
            from neo.architecture_metrics import ArchSnapshot, compare, compute
        except ImportError:
            return None

        sessions = self._load_unprocessed_sessions()
        baseline_dict = None
        for prev in sessions:
            if prev.project_id != self.project_id:
                continue
            if prev.architecture_snapshot:
                baseline_dict = prev.architecture_snapshot
                break  # oldest match wins

        if not baseline_dict:
            return None

        try:
            baseline = ArchSnapshot.from_dict(baseline_dict)
            current = compute(self.codebase_root) if self.codebase_root else ArchSnapshot()
        except Exception as exc:
            logger.debug("arch delta computation failed (non-fatal): %s", exc)
            return None

        # No useful comparison if we never managed to scan anything either round.
        if baseline.files_scanned == 0 or current.files_scanned == 0:
            return None

        return compare(baseline, current)

    def collect_outcomes(
        self, *, clear_processed: bool = True, include_fallback: bool = True
    ) -> tuple[list[Outcome], dict[str, str]]:
        """Collect outcomes by comparing previous suggestions to actual git changes.

        Checks the session log for ALL unprocessed sessions, not just the most recent.
        This prevents loss of outcome signals when neo is invoked multiple times
        before the user acts on suggestions.

        Args:
            clear_processed: If True, clear persisted session state after
                collection. Set False for dry-runs or maintenance commands that
                need to inspect outcomes before deciding whether to consume them.
            include_fallback: If True, use the legacy single-session file when
                the append-only log is absent. Maintenance replay defaults this
                off because old fallback files may already have been processed
                by a prior version that only cleared the log.

        Returns:
            Tuple of (outcomes list, merged suggestion_fact_ids from all sessions).
        """
        sessions = self._load_unprocessed_sessions(include_fallback=include_fallback)
        if not sessions:
            logger.debug("No unprocessed sessions found for outcome detection")
            # Reset explicitly: a stale _last_pending from an earlier call would
            # otherwise be what a later consume_sessions_keeping_pending() writes
            # back. Unreachable today only because replay gates on non-zero
            # stats — make it structural, not incidental.
            self._last_pending = []
            self._last_loaded_keys = set()
            return [], {}

        all_outcomes: list[Outcome] = []
        merged_fact_ids: dict[str, str] = {}
        # The working-tree half of the changed-file query is timestamp
        # independent, so it is computed ONCE rather than per session. Retention
        # means the log now holds many sessions; recomputing it per session cost
        # a measured 0.88s of pure `git` forking at 40 pending sessions, on the
        # request hot path, growing linearly.
        working_tree = self._get_working_tree_changes()
        # Host-recorded edits, loaded once and filtered per session below —
        # hoisted for the same reason `working_tree` is.
        host_edits = self._load_host_edit_events()
        # Watermark stamped onto retained records. Backed off a little because
        # `git log --since` is second-granular: anchoring exactly at "now"
        # could skip a commit made in this same second.
        scan_started = time.time() - self.SCAN_WATERMARK_SKEW_SECONDS
        # Sessions whose suggestions are still awaiting a user action. Retained
        # rather than cleared — see the note at the clearing step below.
        pending: list[SessionRecord] = []

        for prev in sessions:
            if prev.project_id != self.project_id:
                continue

            # Merge fact-id links for EVERY same-project session, including
            # those with no git changes: review-only / UNVERIFIED outcomes
            # (below) resolve their linked fact through this map, and those
            # sessions never reach the git-matching branch.
            merged_fact_ids.update(prev.suggestion_fact_ids)

            # Start from how far this record has already been scanned, not from
            # when the suggestion was made. A retained record's anchor never
            # advances otherwise: it re-queries the whole window every run.
            since = max(prev.timestamp, prev.scanned_through)
            changed_files = self._get_changed_files_since(
                since, working_tree=working_tree
            )
            # Union in the host's own record of what it edited. Strictly ADDS
            # evidence: git already reports anything committed or dirty, and the
            # ledger additionally covers an edit that was applied and then left
            # alone with no later Neo run in that repository — the case
            # `_get_changed_files_since` structurally cannot see, and the one
            # that ages a real acceptance out at PENDING_SESSION_TTL_SECONDS.
            # `>=` not `>`: `since` is a session's own timestamp, and an edit
            # applied in the same clock second as the suggestion is an
            # acceptance, not a pre-existing change.
            changed_files |= {
                path for ts, path in host_edits if ts >= since
            }

            # Retain a REDUCED record holding only what is still outstanding.
            # Keeping the whole session would re-emit an outcome for every
            # already-resolved suggestion on each later invocation.
            remaining = self._unresolved_suggestions(prev, changed_files)
            if remaining and not self._session_expired(prev):
                resolved_now = [
                    s.get("file_path", "") for s in prev.suggestions
                    if s.get("file_path") and s not in remaining
                ]
                pending.append(replace(
                    prev,
                    suggestions=remaining,
                    scanned_through=scan_started,
                    resolved_paths=sorted(set(prev.resolved_paths) | set(resolved_now)),
                ))

            if not changed_files:
                continue

            suggested = [s.get("file_path", "") for s in prev.suggestions]
            logger.info(
                f"Outcome detection (session {prev.timestamp:.0f}): "
                f"{len(changed_files)} changed files, "
                f"{len(suggested)} suggestions, "
                f"{len(prev.suggestion_fact_ids)} linked fact(s)"
            )

            all_outcomes.extend(self._match_to_suggestions(changed_files, prev))

        # Also check for non-git outcomes (weak implicit acceptance for paths the
        # git matcher can't see: review docs, /dev/null, docs/).
        all_outcomes.extend(self._detect_non_git_outcomes(sessions))

        # Collapse to one outcome per file path, strongest signal winning, so a
        # path that is BOTH git-changed and non-git-trackable doesn't double-bump
        # the same fact from a single user action.
        all_outcomes = self._dedup_outcomes(all_outcomes)

        # Clear PROCESSED sessions, keep the ones still waiting on the user.
        #
        # This used to delete the whole log whenever any session existed, even
        # when nothing had changed and no outcome was produced. That silently
        # destroyed every pending suggestion the moment neo was invoked again
        # before the user acted — which is the normal way people use it. The
        # multi-session read directly above was added to prevent exactly that
        # loss, and the unconditional clear defeated it one level up.
        #
        # Measured consequence over 30 days of real traffic: 108 episodes, 58
        # stuck at `suggested_pending_downstream_outcome`, and ZERO `accepted`
        # outcomes ever recorded — so the promote path, which needs two
        # git-verified acceptances, could never fire no matter how correct its
        # own gates were.
        #
        # A session is processed once its files have changed (git-matched) or
        # it produced a non-git outcome; anything else is still pending.
        # Remembered so `replay_linked_feedback` can consume the log with the
        # same retention rule instead of deleting it wholesale.
        self._last_pending = pending
        # Recorded from what we actually read, so the rewrite can distinguish
        # "processed by us" from "appended by a peer since".
        self._last_loaded_keys = {self._session_key(s) for s in sessions}
        if clear_processed:
            self._rewrite_session_log(pending)

        accepted = sum(1 for o in all_outcomes if o.outcome_type == OutcomeType.ACCEPTED)
        modified = sum(1 for o in all_outcomes if o.outcome_type == OutcomeType.MODIFIED)
        independent = sum(1 for o in all_outcomes if o.outcome_type == OutcomeType.INDEPENDENT)
        logger.info(
            f"Outcomes total: {accepted} accepted, {modified} modified, "
            f"{independent} independent (from {len(sessions)} session(s))"
        )
        return all_outcomes, merged_fact_ids

    def detect_outcomes(self) -> tuple[list[Outcome], dict[str, str]]:
        """Detect outcomes and consume processed session state."""
        return self.collect_outcomes(clear_processed=True, include_fallback=True)

    def _load_previous_session(self) -> Optional[SessionRecord]:
        """Load the previous session record from disk."""
        if not self._session_path or not self._session_path.exists():
            return None
        try:
            with open(self._session_path) as f:
                data = json.load(f)
            return SessionRecord(**data)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to load previous session: {e}")
            self._backup_corrupt_file(self._session_path)
            return None
        except (OSError, TypeError) as e:
            logger.warning(f"Failed to load previous session: {e}")
            return None

    @staticmethod
    def _backup_corrupt_file(path: Path) -> None:
        """Preserve a corrupt session file before later writes replace it."""
        backup = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
        try:
            shutil.copy2(path, backup)
            logger.warning(f"Backed up corrupt session file to {backup}")
        except OSError as backup_error:
            logger.warning(f"Failed to back up corrupt session file {path}: {backup_error}")

    def _load_unprocessed_sessions(self, *, include_fallback: bool = True) -> list[SessionRecord]:
        """Load all unprocessed sessions from the session log.

        Falls back to the single session file if no log exists (backward compat).
        """
        log_path = self._session_log_path
        sessions: list[SessionRecord] = []

        if log_path and log_path.exists():
            try:
                with open(log_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            sessions.append(SessionRecord(**data))
                        except (json.JSONDecodeError, TypeError):
                            continue
            except OSError as e:
                logger.warning(f"Failed to read session log: {e}")

        # Fallback: use single session file if log is empty/missing.
        if include_fallback and not sessions:
            prev = self._load_previous_session()
            if prev and prev.timestamp:
                sessions.append(prev)

        return sessions

    def consume_sessions_keeping_pending(self) -> None:
        """Rewrite the log, retaining whatever the last collect_outcomes found
        still outstanding.

        For callers that inspect with `clear_processed=False` and then decide to
        consume. `replay_linked_feedback` did this by calling the old
        `_clear_session_log`, which deleted everything — so the documented
        repair command for a broken memory loop destroyed every pending
        suggestion the moment you ran it.
        """
        self._rewrite_session_log(self._last_pending)

    @staticmethod
    def _is_review_only_path(file_path: str) -> bool:
        """True for suggestion paths that are review/analysis output, not a code edit.

        neo's review workload (Linus/Liotta/code-review) emits suggestions whose
        ``file_path`` names a review *document* (``REVIEW.md``,
        ``ARCHITECTURAL_REVIEW.md``), lives under a ``review``/``reviews`` path
        segment, or is the exact ``NO_MODIFY`` / ``/NO_MODIFY`` sentinel. These
        never correspond to a tracked file, so the git-diff outcome matcher is
        structurally blind to them — yet they are the most common kind of
        suggestion. Recognising them lets ``_detect_non_git_outcomes`` emit the
        same weak-acceptance signal it already gives ``docs/`` and ``/dev/null``
        suggestions.

        Deliberately narrow to avoid matching real source files (e.g.
        ``src/review_service.py`` is NOT review-only).
        """
        fp = (file_path or "").strip()
        if not fp:
            return False
        if fp in ("NO_MODIFY", "/NO_MODIFY"):
            return True
        low = fp.lower()
        base = low.rsplit("/", 1)[-1]
        # A markdown review document: REVIEW.md, ARCHITECTURAL_REVIEW.md, ...
        if base.endswith(".md") and "review" in base:
            return True
        # A 'review'/'reviews' directory segment, or an explicit *-review dir
        # (architecture-review/, code-review/). Only directory segments count,
        # so review_service.py (a real file) is excluded.
        dir_segments = low.strip("/").split("/")[:-1]
        if any(seg in ("review", "reviews") or seg.endswith("-review")
               for seg in dir_segments):
            return True
        return False

    def _dedup_outcomes(self, outcomes: list[Outcome]) -> list[Outcome]:
        """One outcome per (normalized) file path, strongest signal winning.

        The git matcher's ACCEPTED/MODIFIED/INDEPENDENT are verified, stronger
        signals than the non-git weak-acceptance heuristic (UNVERIFIED). When a
        suggestion path is both git-changed and non-git-trackable (e.g. a
        committed ``docs/`` or review ``.md``), the git outcome wins — preventing
        a double success_count/confidence bump on one fact from a single user
        action. Collapsing by path also bounds per-session-in-batch accrual: the
        same review path queued across several unprocessed sessions counts once.
        """
        priority = {
            OutcomeType.ACCEPTED: 3,
            OutcomeType.MODIFIED: 3,
            OutcomeType.REGRESSION: 3,
            OutcomeType.INDEPENDENT: 2,
            OutcomeType.UNVERIFIED: 1,
        }
        best: dict[str, Outcome] = {}
        for o in outcomes:
            key = self._normalize_path(o.file_path)
            cur = best.get(key)
            if cur is None or priority.get(o.outcome_type, 0) > priority.get(cur.outcome_type, 0):
                best[key] = o
        return list(best.values())

    # A pending suggestion is kept alive across invocations, but not forever:
    # a suggestion nobody acted on in this long is abandoned, not pending, and
    # retaining it would grow the log without bound.
    PENDING_SESSION_TTL_SECONDS = 14 * 24 * 3600

    # `git log --since` resolves to whole seconds, so a watermark set at exactly
    # "now" can miss a commit landing in the same second. Back it off slightly;
    # the cost of overlap is re-seeing a change, which `resolved_paths` and the
    # per-path resolution check already absorb.
    SCAN_WATERMARK_SKEW_SECONDS = 2

    _NON_TRACKABLE_PATHS = frozenset({"/dev/null"})

    def _is_non_git_trackable(self, file_path: str, normalized: str) -> bool:
        """True when git can never show us what the user did with this path.

        Shared by the non-git weak-acceptance detector and the pending-session
        retention rule, so the two cannot disagree about which suggestions are
        still worth waiting on.
        """
        return (
            file_path in self._NON_TRACKABLE_PATHS
            or normalized.startswith("docs/")
            or "/docs/" in normalized
            or self._is_review_only_path(file_path)
        )

    def _unresolved_suggestions(
        self, session: SessionRecord, changed_files: set[str]
    ) -> list[dict]:
        """The suggestions in `session` that are still waiting on the user.

        Pendingness is a property of **each suggested path**, not of "did the
        repository move at all". The first version of this asked whether
        `changed_files` was empty, which is only true in a repo with no commits
        and a spotless working tree since the suggestion — so one unrelated
        dirty file dropped the whole session and the acceptance was lost
        exactly as before the fix. Every test passed because they all ran
        against a pristine tree, which is the one state neo is never invoked in.

        Dropped here, and therefore NOT retained:
          - paths git has since touched (they produced their outcome this round)
          - review-only / docs / /dev/null paths (their weak UNVERIFIED already
            fired; retaining them re-emits it on every later invocation)
        """
        remaining: list[dict] = []
        for sugg in session.suggestions:
            file_path = sugg.get("file_path", "")
            if not file_path:
                continue
            normalized = normalize_suggestion_path(
                file_path, session.codebase_root or self.codebase_root
            )
            if self._is_non_git_trackable(file_path, normalized):
                continue
            # Same predicate _match_to_suggestions uses, so "resolved" here and
            # "matched" there cannot drift apart.
            if normalized in changed_files or file_path in changed_files:
                continue
            remaining.append(sugg)
        return remaining

    @staticmethod
    def _session_key(session: SessionRecord) -> tuple:
        """Stable identity for one saved session.

        Used to tell "a record this process already read and processed" from
        "a record some other process appended while we were working". No
        `session_id` field exists on the record, and adding one would not help
        for entries written by an older neo — `time.time()` is microsecond
        resolution, so two sessions of the same project colliding here is not a
        realistic possibility.
        """
        return (
            round(session.timestamp, 6),
            session.learning_episode_id,
            session.prompt,
        )

    def _session_expired(self, session: SessionRecord) -> bool:
        return (time.time() - session.timestamp) > self.PENDING_SESSION_TTL_SECONDS

    def _rewrite_session_log(self, keep: list[SessionRecord]) -> None:
        """Replace the log with `keep`.

        Read-modify-write across processes, so it takes the same per-scope lock
        `FactStore.save()` uses — and, crucially, **re-reads the log under that
        lock** and preserves anything it did not itself process.

        Locking the write alone was not enough. Between one process's read and
        its replace, a peer's `save_session` append was erased without trace
        (reproduced). The fix is merge-on-write rather than a wider lock: the
        lock is held only for re-read + write, never across the git and LM work
        in `collect_outcomes` / `replay_linked_feedback`, so a slow reasoning
        run cannot block another process's save.

        `save_session` takes the same lock for its append, so an append can
        never interleave with this rewrite.

        Durability: temp file + `os.replace`, with `flush`+`fsync` before the
        rename so a machine crash can't leave a rename pointing at unwritten
        data. `mkstemp` rather than a fixed `.tmp` name, since two concurrent
        rewrites would otherwise interleave into one file — the same reason
        `FactStore._save_file` uses it.
        """
        log_path = self._session_log_path
        if not log_path:
            return

        # Imported here, not at module scope: `store` imports this module, so a
        # top-level import would be circular. `scope_file_lock` is already
        # best-effort — it proceeds unlocked when fcntl is unavailable — so it
        # needs no fallback of its own.
        from neo.memory.store import scope_file_lock

        with scope_file_lock(log_path):
            self._rewrite_session_log_locked(log_path, keep)

    def _sessions_appended_since_read(self, log_path: Path) -> list[SessionRecord]:
        """Records now in the log that this process never read.

        Called under the lock, immediately before the rewrite. Anything whose
        identity is absent from `_last_loaded_keys` arrived from a peer while we
        were doing git and LM work, and must survive our write.
        """
        if not log_path.exists():
            return []
        fresh: list[SessionRecord] = []
        try:
            with open(log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = SessionRecord(**json.loads(line))
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if self._session_key(record) not in self._last_loaded_keys:
                        fresh.append(record)
        except OSError as e:
            # Better to skip the merge than to lose the rewrite entirely; the
            # worst case degrades to the previous behavior.
            logger.warning(f"Could not re-read session log before rewrite: {e}")
            return []
        if fresh:
            logger.info(
                "Preserving %d session(s) appended by another process", len(fresh)
            )
        return fresh

    def _rewrite_session_log_locked(
        self, log_path: Path, keep: list[SessionRecord]
    ) -> None:
        # Merge in anything a peer appended after our read. Without this the
        # rewrite is a lost update: we would write back a view of the file that
        # predates their save.
        keep = list(keep) + self._sessions_appended_since_read(log_path)
        # The single-session file is a backward-compat fallback that
        # `_load_unprocessed_sessions` reads only when the log is absent.
        # Removed only AFTER the log is safely in place: unlinking first and
        # then failing the write would drop both copies.
        session_path = self._session_path

        try:
            if keep:
                import tempfile

                fd, tmp_name = tempfile.mkstemp(
                    dir=str(log_path.parent), prefix=log_path.name + ".", suffix=".tmp"
                )
                try:
                    with os.fdopen(fd, "w") as f:
                        for session in keep:
                            f.write(json.dumps(asdict(session)) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_name, log_path)
                except BaseException:
                    # Includes SIGINT/SIGTERM-driven exits: leave no stray temp.
                    try:
                        os.unlink(tmp_name)
                    except OSError:
                        pass
                    raise
                logger.info(
                    "Retained %d pending session(s) awaiting user action", len(keep)
                )
            elif log_path.exists():
                log_path.unlink()

            if session_path and session_path.exists():
                session_path.unlink()
        except OSError as e:
            logger.warning(f"Failed to rewrite session log: {e}")

    def _detect_non_git_outcomes(
        self, sessions: list[SessionRecord]
    ) -> list[Outcome]:
        """Detect implicit acceptance for suggestions that can't be tracked by git.

        When a suggestion targets /dev/null, a docs path, or review/analysis
        output (see _is_review_only_path), and the user invoked neo again
        afterward, that's a weak acceptance signal — the user continued working
        rather than abandoning the tool. Review/analysis is neo's most common
        workload and is invisible to the git-diff matcher, so without this the
        bulk of suggestions never produce any outcome signal.

        All sessions in the log are from PREVIOUS invocations (the current
        session hasn't been written yet when detect_outcomes runs), so we
        process all of them.
        """
        if not sessions:
            return []

        outcomes: list[Outcome] = []

        for prev in sessions:
            for sugg in prev.suggestions:
                file_path = sugg.get("file_path", "")
                # Same reasoning as _match_to_suggestions: resolve against the
                # root this suggestion was recorded under, not the current run's.
                normalized = normalize_suggestion_path(
                    file_path, prev.codebase_root or self.codebase_root
                )

                if not self._is_non_git_trackable(file_path, normalized):
                    continue

                # User came back and ran neo again — weak acceptance signal.
                # Use the raw recorded path (falling back to normalized) so
                # detect_implicit_feedback resolves the linked
                # suggestion_fact_id, which is keyed by the path as recorded.
                outcomes.append(Outcome(
                    outcome_type=OutcomeType.UNVERIFIED,
                    file_path=file_path or normalized,
                    diff_summary="",
                    suggestion_description=sugg.get("description", ""),
                    suggestion_confidence=sugg.get("confidence", 0.0),
                    suggestion_id=sugg.get("suggestion_id", ""),
                    learning_episode_id=prev.learning_episode_id,
                    repository_revision=prev.repository_revision,
                    retrieved_fact_ids=list(prev.retrieved_fact_ids),
                    used_fact_ids=list(prev.used_fact_ids),
                    candidate_id=sugg.get("candidate_id", ""),
                    candidate_subject=sugg.get("candidate_subject", ""),
                    candidate_body=sugg.get("candidate_body", ""),
                    candidate_kind=sugg.get("candidate_kind", "pattern"),
                ))

        return outcomes

    def _load_host_edit_events(self) -> list[tuple[float, str]]:
        """Edits the HOST recorded, as ``(timestamp, repo-relative path)``.

        The `neo hook record` PostToolUse hook appends one line per
        Edit/Write/MultiEdit/NotebookEdit to ``~/.neo/sessions/host_events.jsonl``.
        This is the consumer for it, and it exists because the git-based
        detector cannot see an edit that was never followed by another Neo run
        in that repository: a suggestion applied and then left alone ages out at
        ``PENDING_SESSION_TTL_SECONDS`` and its acceptance is lost. The ledger
        records the edit at the moment it happens.

        **Attribution is by ``file_path`` and never by ``cwd`` or ``head``.**
        Those two name the directory the HOST was launched in, which is not the
        repository the edited file belongs to — measured directly: an edit to a
        scratch repository made from a Claude Code session rooted in the neo
        checkout recorded neo's own HEAD. Only the path can say which project an
        edit belongs to, so a record is ours when its file resolves inside
        ``codebase_root``.

        Read ONCE per ``collect_outcomes`` and filtered per session in memory,
        for the same reason ``_get_working_tree_changes`` is hoisted out of that
        loop: retention means many pending sessions, and re-reading a
        multi-megabyte ledger per session would put a linear cost on the request
        hot path.

        Returns an empty list on every failure — a missing, unreadable or
        malformed ledger must never break outcome detection, which is the same
        contract the hook that writes it keeps. A single malformed LINE is
        skipped rather than discarding the file: the ledger is append-only, so a
        torn final write is the expected corruption and the records before it
        are still good.
        """
        if not self.codebase_root:
            return []
        try:
            from neo.hook import HOOK_LEDGER
        except Exception:  # pragma: no cover - import guard only
            return []

        events: list[tuple[float, str]] = []
        root = Path(self.codebase_root)
        # The rotated generation is read too. Rotation happens at 8 MB, and it
        # moves the RECENT records into `.1` while leaving the active file
        # nearly empty — so reading only the active file would silently lose the
        # window immediately after a rotation, which is exactly the evidence
        # this consumer exists to stop losing.
        for ledger in (HOOK_LEDGER.with_name(HOOK_LEDGER.name + ".1"), HOOK_LEDGER):
            try:
                if not ledger.exists():
                    continue
                with ledger.open("r", encoding="utf-8", errors="replace") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line.startswith("{"):
                            continue
                        try:
                            record = json.loads(line)
                        except (ValueError, TypeError):
                            continue
                        if not isinstance(record, dict):
                            continue
                        raw = record.get("file_path")
                        ts = record.get("ts")
                        if not isinstance(raw, str) or not raw:
                            continue
                        if not isinstance(ts, (int, float)):
                            continue
                        try:
                            resolved = Path(raw)
                            if not resolved.is_absolute():
                                continue
                            relative = str(resolved.relative_to(root))
                        except (ValueError, TypeError, OSError):
                            continue  # not under this project's root
                        events.append((float(ts), relative))
            except OSError as exc:
                logger.debug("host-edit ledger unreadable (non-fatal): %s", exc)
                continue
        return events

    def _get_working_tree_changes(self) -> set[str]:
        """Files dirty in the working tree right now.

        Split out because it does not depend on a session timestamp: callers
        iterating many sessions compute it once and pass it in.
        """
        if not self.codebase_root:
            return set()
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=self.codebase_root,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            if result.returncode != 0:
                return set()
            return {line.strip() for line in result.stdout.strip().split("\n") if line.strip()}
        except (subprocess.SubprocessError, FileNotFoundError, OSError, UnicodeDecodeError) as e:
            logger.debug(f"Git working-tree query failed (non-fatal): {e}")
            return set()

    def _get_changed_files_since(
        self, since_timestamp: float, working_tree: Optional[set[str]] = None
    ) -> set[str]:
        """Get files that changed in git since a timestamp.

        Uses git log --since with ISO timestamp for reliable cross-platform behavior.
        `working_tree` lets a caller supply the timestamp-independent half once
        instead of re-forking `git diff` for every session.
        """
        if not self.codebase_root:
            return set()

        try:
            # Convert timestamp to ISO format for git
            since_iso = datetime.datetime.fromtimestamp(
                since_timestamp, tz=datetime.timezone.utc
            ).isoformat()

            # Get committed changes since timestamp
            result = subprocess.run(
                ["git", "log", "--since", since_iso, "--name-only", "--pretty=format:"],
                cwd=self.codebase_root,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            committed = set()
            if result.returncode == 0:
                committed = {
                    line.strip()
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                }

            working = (
                working_tree if working_tree is not None
                else self._get_working_tree_changes()
            )
            return committed | working

        except (subprocess.SubprocessError, FileNotFoundError, OSError, UnicodeDecodeError) as e:
            logger.debug(f"Git diff failed (non-fatal): {e}")
            return set()

    def _match_to_suggestions(
        self, changed_files: set[str], session: SessionRecord
    ) -> list[Outcome]:
        """Match changed files against previous suggestions to determine outcomes."""
        outcomes: list[Outcome] = []
        suggested_files: set[str] = set()

        for sugg in session.suggestions:
            file_path = sugg.get("file_path", "")
            if not file_path:
                continue

            # Normalize: suggestions may have absolute paths, git diff has relative.
            # Resolve against the root the suggestion was RECORDED under, not the
            # current run's. They differ whenever the previous run happened in a
            # different working tree of the same repo — the common case being a
            # Claude Code session in `.claude/worktrees/agent-*`, whose absolute
            # paths never relative_to() the main checkout, so every suggestion
            # silently failed to match and its learning was orphaned (14 of 65
            # recorded sessions on a live install). project_id already spans
            # worktrees and clones, so the session is found; only the path root
            # was wrong. Falls back to the current root for older records.
            normalized = normalize_suggestion_path(
                file_path, session.codebase_root or self.codebase_root
            )
            suggested_files.add(normalized)

            if normalized in changed_files or file_path in changed_files:
                diff = self._get_file_diff_since(normalized, session.timestamp)
                suggested_diff = sugg.get("suggested_diff", "")
                suggested_code = sugg.get("suggested_code", "")

                # Determine if user applied our suggestion or did something different
                if suggested_diff and diff:
                    overlap = self._compute_diff_overlap(suggested_diff, diff)
                    outcome_type = OutcomeType.ACCEPTED if overlap > 0.3 else OutcomeType.MODIFIED
                elif suggested_code and diff:
                    overlap = self._compute_code_overlap(suggested_code, diff)
                    outcome_type = (
                        OutcomeType.ACCEPTED
                        if overlap >= CODE_OVERLAP_ACCEPTED_THRESHOLD
                        else OutcomeType.MODIFIED
                    )
                else:
                    # Missing suggested_diff or actual diff — can't verify
                    outcome_type = OutcomeType.UNVERIFIED

                outcomes.append(Outcome(
                    outcome_type=outcome_type,
                    file_path=normalized,
                    diff_summary=diff,
                    suggestion_description=sugg.get("description", ""),
                    suggestion_confidence=sugg.get("confidence", 0.0),
                    suggestion_id=sugg.get("suggestion_id", ""),
                    learning_episode_id=session.learning_episode_id,
                    repository_revision=session.repository_revision,
                    retrieved_fact_ids=list(session.retrieved_fact_ids),
                    used_fact_ids=list(session.used_fact_ids),
                    candidate_id=sugg.get("candidate_id", ""),
                    candidate_subject=sugg.get("candidate_subject", ""),
                    candidate_body=sugg.get("candidate_body", ""),
                    candidate_kind=sugg.get("candidate_kind", "pattern"),
                ))

        # Detect independent changes (user changed files neo didn't suggest).
        # Rate-limited: keep only the top MAX_INDEPENDENT_OUTCOMES by diff size
        # to avoid flooding facts with low-value noise in active repos.
        # Paths already resolved for this session count as ours. They are no
        # longer in `session.suggestions` (dropped on retention so they cannot
        # re-match), so without this an ACCEPTED suggestion reappears here as
        # INDEPENDENT — a record asserting neo never suggested a file it did.
        known_ours = set(suggested_files)
        for path in session.resolved_paths:
            known_ours.add(path)
            known_ours.add(self._normalize_path(path))

        independent_candidates: list[Outcome] = []
        for changed in changed_files:
            normalized = self._normalize_path(changed)
            if normalized not in known_ours and changed not in known_ours:
                if self._is_code_file(changed):
                    diff = self._get_file_diff_since(changed, session.timestamp)
                    if not diff:
                        continue  # No diff content = no learning signal
                    independent_candidates.append(Outcome(
                        outcome_type=OutcomeType.INDEPENDENT,
                        file_path=changed,
                        diff_summary=diff,
                    ))

        # Keep only the most informative independent changes (deterministic: size desc, path asc)
        independent_candidates.sort(key=lambda o: (-len(o.diff_summary), o.file_path))
        outcomes.extend(independent_candidates[:MAX_INDEPENDENT_OUTCOMES])

        return outcomes

    def _is_untracked(self, file_path: str) -> bool:
        """Is this path present on disk but unknown to git?

        `ls-files --error-unmatch` exits non-zero for a path git does not
        track. Used only to decide whether the `--no-index` diff is the right
        source; any failure answers False, so the caller falls back to
        reporting no diff rather than inventing one.
        """
        try:
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", "--", file_path],
                cwd=self.codebase_root,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            return result.returncode != 0
        except (subprocess.SubprocessError, FileNotFoundError, OSError,
                UnicodeDecodeError) as exc:
            logger.debug("untracked check failed for %s (non-fatal): %s",
                         file_path, exc)
            return False

    def _get_file_diff_since(self, file_path: str, since_timestamp: float) -> str:
        """Get the actual diff content for a file since a timestamp.

        Returns a truncated diff summary (max 2000 chars) showing what changed.
        Tries committed diff first, falls back to working tree diff.
        """
        if not self.codebase_root:
            return ""

        MAX_DIFF_CHARS = 2000

        try:
            since_iso = datetime.datetime.fromtimestamp(
                since_timestamp, tz=datetime.timezone.utc
            ).isoformat()

            # Try committed changes first
            result = subprocess.run(
                ["git", "log", "--since", since_iso, "-p", "--", file_path],
                cwd=self.codebase_root,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            diff = ""
            if result.returncode == 0 and result.stdout.strip():
                diff = result.stdout.strip()

            # Also check working tree changes
            result2 = subprocess.run(
                ["git", "diff", "HEAD", "--", file_path],
                cwd=self.codebase_root,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            if result2.returncode == 0 and result2.stdout.strip():
                if diff:
                    diff += "\n" + result2.stdout.strip()
                else:
                    diff = result2.stdout.strip()

            # Third source: a file that exists on disk but is UNTRACKED. Neither
            # query above can see one — `git log -p` needs a commit and
            # `git diff HEAD` reports tracked modifications only — so a
            # suggestion to create a NEW file, applied and not yet committed,
            # produced an empty diff and was classified UNVERIFIED, which
            # mutates nothing. `suggestion_is_verifiable` explicitly admits a
            # not-yet-existing path as legitimate, so this is a case the system
            # invites and then could not verify. `--no-index` exits 1 when the
            # files differ, which is the normal result here, so a non-zero
            # return is not an error; only stdout decides.
            if not diff:
                target = Path(self.codebase_root) / file_path
                if target.is_file() and self._is_untracked(file_path):
                    result3 = subprocess.run(
                        ["git", "diff", "--no-index", "--", os.devnull, file_path],
                        cwd=self.codebase_root,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=10,
                    )
                    if result3.stdout.strip():
                        diff = result3.stdout.strip()

            if not diff:
                return ""

            # Extract meaningful parts: headers, hunks, and change lines
            summary_lines = [
                line for line in diff.split("\n")
                if line.startswith(("+++", "---", "@@"))
                or (line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
            ]
            summary = "\n".join(summary_lines)
            if len(summary) > MAX_DIFF_CHARS:
                summary = summary[:MAX_DIFF_CHARS] + "\n... (truncated)"

            return summary

        except (subprocess.SubprocessError, FileNotFoundError, OSError, UnicodeDecodeError) as e:
            logger.debug(f"File diff failed for {file_path} (non-fatal): {e}")
            return ""

    @staticmethod
    def _compute_code_overlap(suggested_code: str, actual_diff: str) -> float:
        """Estimate overlap between a suggested code block and actual changed lines.

        This supports Neo's code-first output mode, where suggestions may include
        executable code but no unified diff. We compare the normalized added lines
        from the actual diff against the normalized code block lines.
        """
        code_lines = {
            line.strip()
            for line in suggested_code.splitlines()
            if line.strip()
        }
        changed_lines = {
            line[1:].strip()
            for line in actual_diff.splitlines()
            if line.startswith("+") and not line.startswith("+++")
            and line[1:].strip()
        }

        if not code_lines and not changed_lines:
            return 1.0
        if not code_lines or not changed_lines:
            return 0.0

        overlap = len(code_lines & changed_lines)
        return overlap / min(len(code_lines), len(changed_lines))

    def _normalize_path(self, path: str) -> str:
        """Normalize a file path to relative form for comparison."""
        return normalize_suggestion_path(path, self.codebase_root)

    @staticmethod
    def _compute_diff_overlap(suggested: str, actual: str) -> float:
        """Compute line-level overlap between suggested and actual diffs.

        Returns 0.0-1.0 where 1.0 means identical changes.
        Preserves the +/- prefix so additions and removals of the same
        content are not conflated.
        """
        def extract_change_lines(diff_text: str) -> set[str]:
            return {
                line.strip()
                for line in diff_text.split("\n")
                if line.strip()
                and line.strip()[0] in ("+", "-")
                and not line.strip().startswith(("+++", "---", "@@"))
            }

        suggested_lines = extract_change_lines(suggested)
        actual_lines = extract_change_lines(actual)

        if not suggested_lines and not actual_lines:
            return 1.0
        if not suggested_lines or not actual_lines:
            return 0.0

        intersection = suggested_lines & actual_lines
        union = suggested_lines | actual_lines
        return len(intersection) / len(union) if union else 0.0

    # ------------------------------------------------------------------ #
    # Git history ingestion
    # ------------------------------------------------------------------ #

    def ingest_git_history(self, max_commits: int = 50) -> list[dict]:
        """Learn from git commit history that hasn't been ingested yet.

        Reads commits since the last ingestion watermark (or last 50 commits
        on first run). For each commit, extracts the commit message, changed
        files, and diff summary. Returns structured records ready for fact
        creation.

        Args:
            max_commits: Maximum number of commits to ingest per run.

        Returns:
            List of dicts with keys: subject, body, commit_hash, timestamp.
        """
        if not self.codebase_root:
            return []

        watermark = self._load_watermark()
        commits = self._get_commits_since(watermark, max_commits)

        if not commits:
            return []

        records = []
        for commit in commits:
            if not self._is_meaningful_commit(commit["message"]):
                continue

            diff = self._get_commit_diff(commit["hash"])
            if not diff:
                continue

            # Build a learnable record from the commit
            subject = f"history:{commit['hash'][:8]} {commit['message'][:60]}"
            body_parts = [
                f"Commit: {commit['hash'][:12]}",
                f"Message: {commit['message']}",
                f"Files: {', '.join(commit['files'][:10])}",
            ]
            if diff:
                body_parts.append(f"Changes:\n{diff}")

            records.append({
                "subject": subject,
                "body": "\n".join(body_parts),
                "commit_hash": commit["hash"],
                "timestamp": commit["timestamp"],
            })

        # Update watermark to most recent commit
        if commits:
            self._save_watermark(commits[0]["hash"])

        logger.info(f"Ingested {len(records)} commits from git history")
        return records

    def _get_commits_since(
        self, since_hash: Optional[str], max_commits: int
    ) -> list[dict]:
        """Get commit metadata since a watermark hash.

        Returns commits in reverse chronological order (newest first).
        """
        if not self.codebase_root:
            return []

        try:
            # Build git log command
            cmd = [
                "git", "log",
                f"-{max_commits}",
                "--pretty=format:%H\t%at\t%s",  # hash, timestamp, subject
                "--name-only",
            ]
            if since_hash:
                cmd.append(f"{since_hash}..HEAD")

            result = subprocess.run(
                cmd,
                cwd=self.codebase_root,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=15,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []

            # Parse: each commit block is header line + file lines + blank line
            commits = []
            current = None
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                if "\t" in line and len(line.split("\t")) >= 3:
                    # This is a header line: hash\ttimestamp\tsubject
                    parts = line.split("\t", 2)
                    current = {
                        "hash": parts[0],
                        "timestamp": float(parts[1]),
                        "message": parts[2],
                        "files": [],
                    }
                    commits.append(current)
                elif current is not None:
                    # This is a file path
                    current["files"].append(line)

            return commits

        except (subprocess.SubprocessError, FileNotFoundError, OSError, UnicodeDecodeError) as e:
            logger.debug(f"Git log failed (non-fatal): {e}")
            return []

    def _get_commit_diff(self, commit_hash: str) -> str:
        """Get the diff for a specific commit, filtered to code files only."""
        if not self.codebase_root:
            return ""

        MAX_DIFF_CHARS = 2000

        try:
            result = subprocess.run(
                ["git", "show", "--stat", "--patch", "--format=", commit_hash],
                cwd=self.codebase_root,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return ""

            # Extract meaningful parts: headers, hunks, and change lines
            summary_lines = [
                line for line in result.stdout.split("\n")
                if line.startswith(("+++", "---", "@@"))
                or (line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
            ]
            summary = "\n".join(summary_lines)
            if len(summary) > MAX_DIFF_CHARS:
                summary = summary[:MAX_DIFF_CHARS] + "\n... (truncated)"
            return summary

        except (subprocess.SubprocessError, FileNotFoundError, OSError, UnicodeDecodeError) as e:
            logger.debug(f"Commit diff failed for {commit_hash} (non-fatal): {e}")
            return ""

    @staticmethod
    def _is_meaningful_commit(message: str) -> bool:
        """Filter out commits that aren't useful for learning.

        Skip merge commits, version bumps, and auto-generated commits.
        Keep: bug fixes, features, refactors, and anything with substance.
        """
        msg = message.lower().strip()

        # Skip noise
        skip_prefixes = (
            "merge ", "merge pull request", "merge branch",
            "bump version", "release v", "update changelog",
            "chore(deps)", "chore(release)",
            "initial commit",
        )
        if any(msg.startswith(p) for p in skip_prefixes):
            return False

        # Skip very short messages (likely not informative)
        if len(msg) < 10:
            return False

        return True

    def _load_watermark(self) -> Optional[str]:
        """Load the last-ingested commit hash for this project."""
        watermark_path = self._get_watermark_path()
        if not watermark_path or not watermark_path.exists():
            return None
        try:
            data = json.loads(watermark_path.read_text())
            return data.get("last_commit_hash")
        except (json.JSONDecodeError, OSError):
            return None

    def _save_watermark(self, commit_hash: str) -> None:
        """Save the last-ingested commit hash."""
        watermark_path = self._get_watermark_path()
        if not watermark_path:
            return
        try:
            atomic_write_json(watermark_path, {
                "last_commit_hash": commit_hash,
                "updated_at": time.time(),
            })
        except OSError as e:
            logger.debug(f"Failed to save watermark: {e}")

    def _get_watermark_path(self) -> Optional[Path]:
        if not self.project_id:
            return None
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        return SESSIONS_DIR / f"watermark_{self.project_id}.json"

    @staticmethod
    def _is_code_file(path: str) -> bool:
        """Check if a file looks like source code (not config, docs, etc.)."""
        code_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
            ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".swift", ".kt",
            ".scala", ".sql", ".sh", ".bash", ".zsh",
        }
        return Path(path).suffix.lower() in code_extensions
