"""
Core data models for Neo's fact-based memory system.

Replaces the flat ReasoningEntry list with a scoped, supersession-based
fact store inspired by StateBench's four-layer state model.
"""

import math
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from neo.math_utils import g_n_update, recall_probability


# Ranking policy shared across retrieval paths (FactStore.retrieve_relevant
# and ContextAssembler._score_facts). MUST stay in one place — if the two
# paths diverge, outcome learning ranks inconsistently.
SUCCESS_BONUS_WEIGHT = 0.1
SUCCESS_BONUS_CAP = 0.2  # Caps bonus so a narrow historical winner can't
                         # dominate cosine similarity (bounded in [0,1]).


def success_bonus(success_count: int) -> float:
    """Log-scaled, capped bonus for facts with validated outcomes.

    1 success → +0.10, 3 → +0.20 (cap), 10 → +0.20 (cap).
    """
    if success_count <= 0:
        return 0.0
    return min(SUCCESS_BONUS_CAP, SUCCESS_BONUS_WEIGHT * math.log2(success_count + 1))


class Provenance(str, Enum):
    """Source attribution for a fact, ordered by trust.

    Mirrors the user_statement > tool_output > agent_inference taxonomy
    from the memory-systems survey (paper 2603.07670 §7.3) using the
    existing Neo vocabulary:

      STRUCTURAL — parsed from code/config/CLAUDE.md (authored by a human).
      OBSERVED   — saw it happen at runtime (outcome detection / tool output).
      INFERRED   — derived by the LLM, no direct evidence.

    String-valued so existing JSON dumps round-trip unchanged.
    """
    STRUCTURAL = "structural"
    OBSERVED = "observed"
    INFERRED = "inferred"


_PROVENANCE_BONUS = {
    Provenance.STRUCTURAL.value: 0.05,
    Provenance.OBSERVED.value: 0.02,
    Provenance.INFERRED.value: 0.0,
}


def provenance_bonus(provenance: str) -> float:
    """Tiny bonus reflecting how the fact was sourced (see ``Provenance``)."""
    return _PROVENANCE_BONUS.get(provenance, 0.0)


SECONDS_PER_DAY = 86400.0

# Tags that mark a fact as curated knowledge — seeded, community-sourced,
# or distilled by synthesis. Curated facts skip Ebbinghaus decay because
# their relevance isn't a function of how often the user happens to query
# them: a CLAUDE.md rule still applies after a year of silence.
_CURATED_TAGS = frozenset({"seed", "community", "synthesized"})


def _decays(fact: "Fact") -> bool:
    """True iff this fact's similarity should be transformed by recall decay.

    CONSTRAINT/ARCHITECTURE/DECISION are stable knowledge — they don't
    decay. Curated-tagged facts are explicitly marked as do-not-touch and
    also skip decay. Everything else (PATTERN, REVIEW, FAILURE,
    KNOWN_UNKNOWN) is fluid and decays.
    """
    if fact.kind in (FactKind.CONSTRAINT, FactKind.ARCHITECTURE, FactKind.DECISION):
        return False
    if _CURATED_TAGS & set(fact.tags):
        return False
    return True


def rank_score(fact: "Fact", similarity: float, now: Optional[float] = None) -> float:
    """Single source of truth for fact ranking.

    Used by FactStore.retrieve_relevant and ContextAssembler._score_facts so
    a query gets the same ordering whichever path runs it.

        s_recall = recall_probability(sim, t, g_n) if fact decays else sim
        score    = s_recall * confidence
                 + success_bonus(success_count)
                 + provenance_bonus

    The Ebbinghaus transform (Hou et al., 2404.00573) gives spaced-repetition
    semantics: frequently-recalled fluid facts decay slower, dormant ones
    decay faster, and the gap between two recalls shapes future decay.
    Curated/stable facts (see ``_decays``) bypass the transform entirely.
    """
    if _decays(fact):
        ts = fact.metadata.last_recall_ts
        if ts is None:
            ts = fact.metadata.created_at
        elapsed_days = max(0.0, ((now if now is not None else time.time()) - ts) / SECONDS_PER_DAY)
        sim = recall_probability(
            similarity,
            days_since_recall=elapsed_days,
            g_n=fact.metadata.g_n,
        )
    else:
        sim = similarity

    return (
        sim * fact.metadata.confidence
        + success_bonus(fact.metadata.success_count)
        + provenance_bonus(fact.metadata.provenance)
    )


def update_recall(fact: "Fact", now: Optional[float] = None) -> None:
    """Bookkeeping when retrieval has surfaced a fact.

    Increments recall_count and stamps last_recall_ts. Strengthens g_n on
    the *second* recall onward (the gap between two real recalls is the
    spaced-repetition signal). On the first recall there is no gap to
    reward, so g_n stays put — only the recall_count and timestamp move.

    Skipped entirely for non-decaying facts: curated/stable facts don't
    use these fields for ranking, so we don't churn them.
    """
    if not _decays(fact):
        return

    if now is None:
        now = time.time()

    if fact.metadata.last_recall_ts is not None:
        elapsed_days = max(0.0, (now - fact.metadata.last_recall_ts) / SECONDS_PER_DAY)
        fact.metadata.g_n = g_n_update(fact.metadata.g_n, elapsed_days)

    fact.metadata.last_recall_ts = now
    fact.metadata.recall_count += 1


class FactKind(Enum):
    """Type of fact stored in memory."""
    CONSTRAINT = "constraint"       # Project rules (from CLAUDE.md etc.)
    ARCHITECTURE = "architecture"   # Architectural decisions and patterns
    PATTERN = "pattern"             # Reusable code/design patterns
    REVIEW = "review"               # Code review learnings
    DECISION = "decision"           # Feature/design decisions made
    KNOWN_UNKNOWN = "known_unknown" # Explicit gaps in knowledge
    FAILURE = "failure"             # Failed approaches and their reasons


class FactScope(Enum):
    """Scope hierarchy for facts."""
    GLOBAL = "global"     # Cross-project (e.g., language idioms)
    ORG = "org"           # Organization-wide (e.g., team conventions)
    PROJECT = "project"   # Project-specific (e.g., architecture choices)
    SESSION = "session"   # Current session only (ephemeral)


@dataclass
class FactMetadata:
    """Metadata attached to a fact."""
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source_file: str = ""      # File that generated this fact
    source_prompt: str = ""    # Prompt that triggered this fact
    confidence: float = 0.5    # 0.0 to 1.0
    success_count: int = 0     # Times this suggestion was validated
    provenance: str = Provenance.INFERRED.value  # see Provenance enum
    # Ebbinghaus-style spaced-repetition fields. recall_count tracks how
    # often this fact has been returned by retrieval; g_n is the per-fact
    # decay-constant denominator (starts at 1.0, grows on each recall so
    # consistent re-use slows future decay); last_recall_ts is the last
    # time retrieval surfaced the fact (None = never recalled).
    # See math_utils.recall_probability.
    recall_count: int = 0
    g_n: float = 1.0
    last_recall_ts: Optional[float] = None
    # Bi-temporal model (Zep/AriGraph pattern; see paper 2512.13564 §5.2.2).
    # ``event_time`` answers "when did the fact represent?" — typically the
    # commit/observation time of the world-event the fact describes. For
    # facts ingested at the same moment as the underlying event, it equals
    # created_at; for facts retro-inserted from git history, it should be
    # the commit timestamp.
    # ``ingest_time`` answers "when did Neo learn it?" — equals created_at
    # by default. The split lets us soft-delete by stamping event_time_end
    # instead of dropping the row, and to resolve contradictions by the
    # newer event_time without losing audit trail.
    event_time: Optional[float] = None     # None ⇒ falls back to created_at
    event_time_end: Optional[float] = None # set for soft-deleted/superseded
    ingest_time: Optional[float] = None    # None ⇒ falls back to created_at

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "source_file": self.source_file,
            "source_prompt": self.source_prompt,
            "confidence": self.confidence,
            "success_count": self.success_count,
            "provenance": self.provenance,
            "recall_count": self.recall_count,
            "g_n": self.g_n,
            "last_recall_ts": self.last_recall_ts,
            "event_time": self.event_time,
            "event_time_end": self.event_time_end,
            "ingest_time": self.ingest_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FactMetadata":
        # Legacy: pre-A2 dumps used 0.0 as the "never recalled" sentinel.
        last_recall = data.get("last_recall_ts")
        if last_recall == 0.0:
            last_recall = None
        return cls(
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            access_count=data.get("access_count", 0),
            source_file=data.get("source_file", ""),
            source_prompt=data.get("source_prompt", ""),
            confidence=data.get("confidence", 0.5),
            success_count=data.get("success_count", 0),
            provenance=data.get("provenance", Provenance.INFERRED.value),
            recall_count=data.get("recall_count", 0),
            g_n=data.get("g_n", 1.0),
            last_recall_ts=last_recall,
            event_time=data.get("event_time"),
            event_time_end=data.get("event_time_end"),
            ingest_time=data.get("ingest_time"),
        )

    @property
    def effective_event_time(self) -> float:
        """The event_time, falling back to created_at when unset."""
        return self.event_time if self.event_time is not None else self.created_at

    @property
    def effective_ingest_time(self) -> float:
        """The ingest_time, falling back to created_at when unset."""
        return self.ingest_time if self.ingest_time is not None else self.created_at


@dataclass
class Fact:
    """A single piece of knowledge in the fact store.

    Replaces ReasoningEntry with scoped, supersession-based tracking.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    subject: str = ""              # Concise label (what this fact is about)
    body: str = ""                 # Full content
    kind: FactKind = FactKind.PATTERN
    scope: FactScope = FactScope.PROJECT
    org_id: str = ""               # From git remote (e.g., "parslee-ai")
    project_id: str = ""           # SHA256[:16] of codebase root
    is_valid: bool = True          # True until superseded
    superseded_by: Optional[str] = None   # ID of replacing fact
    supersedes: Optional[str] = None      # ID of replaced fact
    depends_on: list[str] = field(default_factory=list)
    needs_review: bool = False     # Set when a dependency is superseded
    metadata: FactMetadata = field(default_factory=FactMetadata)
    embedding: Optional[np.ndarray] = None
    tags: list[str] = field(default_factory=list)

    def size_hint(self) -> int:
        """Approximate token count. Uses len//4 heuristic — not precise, just monotonic."""
        return len(self.subject + self.body) // 4

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "subject": self.subject,
            "body": self.body,
            "kind": self.kind.value,
            "scope": self.scope.value,
            "org_id": self.org_id,
            "project_id": self.project_id,
            "is_valid": self.is_valid,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "depends_on": self.depends_on,
            "needs_review": self.needs_review,
            "metadata": self.metadata.to_dict(),
            "tags": self.tags,
        }
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Fact":
        embedding = None
        if "embedding" in data and data["embedding"] is not None:
            embedding = np.array(data["embedding"], dtype=np.float32)

        return cls(
            id=data.get("id", uuid.uuid4().hex[:16]),
            subject=data.get("subject", ""),
            body=data.get("body", ""),
            kind=FactKind(data.get("kind", "pattern")),
            scope=FactScope(data.get("scope", "project")),
            org_id=data.get("org_id", ""),
            project_id=data.get("project_id", ""),
            is_valid=data.get("is_valid", True),
            superseded_by=data.get("superseded_by"),
            supersedes=data.get("supersedes"),
            depends_on=data.get("depends_on", []),
            needs_review=data.get("needs_review", False),
            metadata=FactMetadata.from_dict(data.get("metadata", {})),
            embedding=embedding,
            tags=data.get("tags", []),
        )


@dataclass
class ContextResult:
    """Assembled context for LLM injection, following StateBench's four-layer model."""
    constraints: list[Fact] = field(default_factory=list)          # Layer 1
    valid_facts: list[Fact] = field(default_factory=list)          # Layer 2
    invalidated_facts: list[Fact] = field(default_factory=list)    # Layer 2b (superseded, for contrast)
    working_set: list[Fact] = field(default_factory=list)          # Layer 3 (session-scoped)
    environment: dict = field(default_factory=dict)                # Layer 4 (git state, passed through)
    known_unknowns: list[Fact] = field(default_factory=list)       # Hallucination prevention
