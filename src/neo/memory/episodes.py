"""Versioned evidence records for Neo coding sessions.

Learning episodes are an append-only evidence surface, separate from the
generalized knowledge in :mod:`neo.memory.store`.  They intentionally retain
identifiers, hashes, and outcomes rather than raw repository source so the
causal chain remains inspectable without duplicating sensitive code into a
second memory corpus.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from neo.memory.io_utils import atomic_write_json

logger = logging.getLogger(__name__)

EPISODE_SCHEMA_VERSION = 3
MAX_EPISODES_PER_PROJECT = 500

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(
        r"(?i)\b(password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*([^\s,;]+)"
    ),
)


def content_hash(value: str) -> str:
    """Return a stable digest without persisting the potentially sensitive text."""
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def redact_sensitive_text(value: str) -> str:
    """Best-effort removal of common credential shapes before persistence."""
    redacted = value or ""
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(lambda match: f"{match.group(1)}=[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


@dataclass
class ContextSelection:
    """One repository or instruction artifact selected for the model context."""

    path: str
    content_sha256: str = ""
    line_range: Optional[tuple[int, int]] = None
    kind: str = "repository_file"


@dataclass
class RetrievedFactEvidence:
    """A fact considered during retrieval for this task."""

    fact_id: str
    score: Optional[float] = None
    included_in_context: bool = False
    used_in_reasoning: Optional[bool] = None
    outcome_association: str = ""


@dataclass
class SuggestionEvidence:
    """Privacy-conscious description of a proposed change."""

    suggestion_id: str
    file_path: str
    description: str
    confidence: float
    diff_sha256: str = ""
    code_sha256: str = ""


@dataclass
class VerificationEvidence:
    """Normalized check result. ``status`` never aliases skipped to passed."""

    verification_id: str
    kind: str
    status: str  # passed | failed | warning | unavailable | skipped
    tool_name: str = ""
    summary: str = ""
    diagnostics_count: int = 0
    repository_revision: str = ""


VERIFICATION_KINDS = frozenset({
    "test", "lint", "type_check", "parser", "compile", "neo_static",
    "user_acceptance", "user_modification", "later_regression",
})
VERIFICATION_STATUSES = frozenset({
    "passed", "failed", "warning", "unavailable", "skipped",
})


def aggregate_verification_status(evidence: list[VerificationEvidence]) -> str:
    """Return a deterministic, fail-closed verdict for heterogeneous evidence."""
    statuses = {item.status for item in evidence}
    for status in ("failed", "warning", "unavailable", "skipped", "passed"):
        if status in statuses:
            return status
    return "skipped"


@dataclass
class MemoryMutationEvidence:
    """One fact-store mutation attributable to the episode."""

    mutation_id: str
    operation: str
    fact_id: str
    reason: str = ""
    before_state: dict[str, Any] = field(default_factory=dict)
    after_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryCandidateEvidence:
    """Probationary knowledge candidate derived from one proposed suggestion."""

    candidate_id: str
    suggestion_id: str
    subject: str
    body: str
    kind: str
    status: str = "observed_unverified"
    promoted_fact_id: str = ""
    promoted_global_fact_id: str = ""
    supporting_episode_ids: list[str] = field(default_factory=list)
    contradicting_episode_ids: list[str] = field(default_factory=list)


@dataclass
class LearningEpisode:
    """Versioned causal record for one Neo task.

    Optional/default fields make v1 records forward-compatible: readers can
    load older or partially populated records without treating absent evidence
    as successful verification.
    """

    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    schema_version: int = EPISODE_SCHEMA_VERSION
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    objective: str = ""
    project_id: str = ""
    repository_root: str = ""
    repository_revision: str = ""
    repository_dirty: Optional[bool] = None
    context_selection: list[ContextSelection] = field(default_factory=list)
    retrieved_facts: list[RetrievedFactEvidence] = field(default_factory=list)
    reasoning_mode: str = ""
    reasoning_reason: str = ""
    provider: str = ""
    model: str = ""
    operating_mode: str = "learn"
    authority: dict[str, Any] = field(default_factory=dict)
    execution_context: dict[str, Any] = field(default_factory=dict)
    suggestions: list[SuggestionEvidence] = field(default_factory=list)
    applied_actions: list[dict[str, Any]] = field(default_factory=list)
    verification: list[VerificationEvidence] = field(default_factory=list)
    final_outcome: str = "pending"
    outcome_details: dict[str, Any] = field(default_factory=dict)
    memory_mutations: list[MemoryMutationEvidence] = field(default_factory=list)
    memory_candidates: list[MemoryCandidateEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningEpisode":
        """Load v1 or older partial records with conservative defaults."""
        if not isinstance(data, dict):
            raise TypeError("learning episode must be a JSON object")
        version = int(data.get("schema_version", 0))
        if version > EPISODE_SCHEMA_VERSION:
            raise ValueError(f"unsupported learning episode schema version: {version}")
        verification = []
        for item in data.get("verification", []):
            normalized = dict(item)
            if normalized.get("kind") == "static_check":
                normalized["kind"] = "neo_static"
            elif normalized.get("kind") == "downstream_outcome":
                normalized["kind"] = (
                    "user_modification"
                    if normalized.get("status") == "failed"
                    else "user_acceptance"
                )
            verification.append(VerificationEvidence(**normalized))
        return cls(
            episode_id=str(data.get("episode_id") or uuid.uuid4().hex),
            session_id=str(data.get("session_id") or uuid.uuid4().hex),
            task_id=str(data.get("task_id") or uuid.uuid4().hex),
            schema_version=EPISODE_SCHEMA_VERSION,
            started_at=float(data.get("started_at", time.time())),
            completed_at=data.get("completed_at"),
            objective=str(data.get("objective", "")),
            project_id=str(data.get("project_id", "")),
            repository_root=str(data.get("repository_root", "")),
            repository_revision=str(data.get("repository_revision", "")),
            repository_dirty=data.get("repository_dirty"),
            context_selection=[ContextSelection(**item) for item in data.get("context_selection", [])],
            retrieved_facts=[RetrievedFactEvidence(**item) for item in data.get("retrieved_facts", [])],
            reasoning_mode=str(data.get("reasoning_mode", "")),
            reasoning_reason=str(data.get("reasoning_reason", "")),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            operating_mode=str(data.get("operating_mode", "learn")),
            authority=dict(data.get("authority", {})),
            execution_context=dict(data.get("execution_context", {})),
            suggestions=[SuggestionEvidence(**item) for item in data.get("suggestions", [])],
            applied_actions=list(data.get("applied_actions", [])),
            verification=verification,
            final_outcome=str(data.get("final_outcome", "pending")),
            outcome_details=dict(data.get("outcome_details", {})),
            memory_mutations=[MemoryMutationEvidence(**item) for item in data.get("memory_mutations", [])],
            memory_candidates=[MemoryCandidateEvidence(**item) for item in data.get("memory_candidates", [])],
        )


class LearningEpisodeStore:
    """Atomic, bounded, per-project local storage for learning episodes."""

    def __init__(self, project_id: str, *, base_dir: Optional[Path] = None):
        self.project_id = project_id or "unscoped"
        root = base_dir or (Path.home() / ".neo" / "episodes")
        self.path = root / self.project_id

    def save(self, episode: LearningEpisode) -> Path:
        """Atomically write one episode and enforce the per-project bound."""
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / f"{episode.episode_id}.json"
        atomic_write_json(target, episode.to_dict(), indent=2)
        self._enforce_limit()
        return target

    def load(self, episode_id: str) -> Optional[LearningEpisode]:
        """Load one record; preserve malformed input and fail safely."""
        target = self.path / f"{episode_id}.json"
        if not target.exists():
            return None
        try:
            return LearningEpisode.from_dict(json.loads(target.read_text()))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self._preserve_corrupt(target)
            logger.warning("Malformed learning episode %s: %s", target.name, exc)
            return None

    def list(self) -> list[LearningEpisode]:
        """Return readable episodes oldest-first, skipping malformed records."""
        if not self.path.exists():
            return []
        episodes = []
        for target in sorted(self.path.glob("*.json")):
            episode = self.load(target.stem)
            if episode is not None:
                episodes.append(episode)
        episodes.sort(key=lambda item: item.started_at)
        return episodes

    def _enforce_limit(self) -> None:
        records = sorted(
            self.path.glob("*.json"),
            key=lambda item: item.stat().st_mtime_ns,
            reverse=True,
        )
        for stale in records[MAX_EPISODES_PER_PROJECT:]:
            try:
                stale.unlink()
            except OSError as exc:
                logger.debug("Failed to prune learning episode %s: %s", stale, exc)

    @staticmethod
    def _preserve_corrupt(path: Path) -> None:
        try:
            path.rename(path.with_name(f"{path.name}.corrupt-{time.time_ns()}"))
        except OSError:
            pass


def repository_state(root: Optional[str]) -> tuple[str, Optional[bool]]:
    """Return ``(HEAD revision, dirty)`` without raising or making changes."""
    if not root:
        return "", None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, timeout=5,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True,
            text=True, timeout=5,
        )
        rev = revision.stdout.strip() if revision.returncode == 0 else ""
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return rev, dirty
    except (OSError, subprocess.SubprocessError):
        return "", None
