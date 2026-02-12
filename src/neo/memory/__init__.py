"""
Neo's fact-based memory system.

Replaces PersistentReasoningMemory with a scoped, supersession-based
fact store inspired by StateBench's four-layer state model.
"""

from neo.memory.models import (
    ContextResult,
    Fact,
    FactKind,
    FactMetadata,
    FactScope,
)
from neo.memory.store import FactStore

__all__ = [
    "ContextResult",
    "Fact",
    "FactKind",
    "FactMetadata",
    "FactScope",
    "FactStore",
]
