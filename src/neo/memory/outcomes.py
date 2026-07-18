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
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
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

            # Also append to session log so we don't lose prior sessions
            log_path = self._session_log_path
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(asdict(session)) + "\n")

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
            return [], {}

        all_outcomes: list[Outcome] = []
        merged_fact_ids: dict[str, str] = {}

        for prev in sessions:
            if prev.project_id != self.project_id:
                continue

            # Merge fact-id links for EVERY same-project session, including
            # those with no git changes: review-only / UNVERIFIED outcomes
            # (below) resolve their linked fact through this map, and those
            # sessions never reach the git-matching branch.
            merged_fact_ids.update(prev.suggestion_fact_ids)

            changed_files = self._get_changed_files_since(prev.timestamp)
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

        # Clear processed sessions from the log
        if clear_processed and (all_outcomes or sessions):
            self._clear_session_log()

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

    def _clear_session_log(self) -> None:
        """Clear processed session state after outcomes have been processed."""
        log_path = self._session_log_path
        if log_path and log_path.exists():
            try:
                log_path.unlink()
            except OSError:
                pass
        # The single-session file is a backward-compat fallback when the
        # append-only log is missing. Remove it too, otherwise the same
        # processed session can replay on the next invocation.
        session_path = self._session_path
        if session_path and session_path.exists():
            try:
                session_path.unlink()
            except OSError:
                pass

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
        non_trackable = {"/dev/null"}

        for prev in sessions:
            for sugg in prev.suggestions:
                file_path = sugg.get("file_path", "")
                normalized = self._normalize_path(file_path)

                is_non_trackable = (
                    file_path in non_trackable
                    or normalized.startswith("docs/")
                    or "/docs/" in normalized
                    or self._is_review_only_path(file_path)
                )
                if not is_non_trackable:
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

    def _get_changed_files_since(self, since_timestamp: float) -> set[str]:
        """Get files that changed in git since a timestamp.

        Uses git log --since with ISO timestamp for reliable cross-platform behavior.
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

            # Also get currently staged/unstaged changes
            result2 = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=self.codebase_root,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=10,
            )
            working = set()
            if result2.returncode == 0:
                working = {
                    line.strip()
                    for line in result2.stdout.strip().split("\n")
                    if line.strip()
                }

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

            # Normalize: suggestions may have absolute paths, git diff has relative
            normalized = self._normalize_path(file_path)
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
        independent_candidates: list[Outcome] = []
        for changed in changed_files:
            normalized = self._normalize_path(changed)
            if normalized not in suggested_files and changed not in suggested_files:
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
        """Normalize a file path to relative form for comparison.

        Handles three cases:
        1. True absolute paths under codebase_root -> relative
        2. Bare leading slash under codebase_root (/src/bar.py) -> src/bar.py
        3. Everything else unchanged
        """
        if not path:
            return path

        if self.codebase_root:
            try:
                p = Path(path)
                root = Path(self.codebase_root)
                if p.is_absolute():
                    return str(p.relative_to(root))
            except (ValueError, TypeError):
                pass

            # Strip bare leading slash if the result exists relative to codebase_root
            # (common in suggestion file_path values like "/src/foo.py")
            if path.startswith("/"):
                stripped = path.lstrip("/")
                candidate = Path(self.codebase_root) / stripped
                if candidate.exists() or candidate.parent.exists():
                    return stripped

        return path

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
