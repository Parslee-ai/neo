"""
Core data models for Neo's fact-based memory system.

Replaces the flat ReasoningEntry list with a scoped, supersession-based
fact store inspired by StateBench's four-layer state model.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


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

    def to_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "source_file": self.source_file,
            "source_prompt": self.source_prompt,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FactMetadata":
        return cls(
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            access_count=data.get("access_count", 0),
            source_file=data.get("source_file", ""),
            source_prompt=data.get("source_prompt", ""),
            confidence=data.get("confidence", 0.5),
        )


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
