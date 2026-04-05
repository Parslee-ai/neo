"""
Neo's fact-based memory system.

Replaces PersistentReasoningMemory with a scoped, supersession-based
fact store inspired by StateBench's four-layer state model.
"""

from neo.memory.claude_memory import ClaudeMemoryIngester
from neo.memory.community import CommunityFeedIngester
from neo.memory.models import (
    ContextResult,
    Fact,
    FactKind,
    FactMetadata,
    FactScope,
)
from neo.memory.outcomes import OutcomeTracker
from neo.memory.seed import SeedIngester
from neo.memory.store import FactStore

__all__ = [
    "ClaudeMemoryIngester",
    "CommunityFeedIngester",
    "ContextResult",
    "Fact",
    "FactKind",
    "FactMetadata",
    "FactScope",
    "FactStore",
    "OutcomeTracker",
    "SeedIngester",
]
