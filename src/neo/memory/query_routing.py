"""
Query-routing classifier + decomposer (paper 2604.04853 §5.3).

Three query shapes, detected by cheap regex/heuristic:

  DIRECT       — single fact lookup; just run the retriever once
  CHAIN        — multi-hop dependency ("of the X of the Y"); decompose into
                 ordered sub-queries
  SPLIT        — multi-entity fanout ("compare A, B, and C"); decompose into
                 parallel sub-queries

All routing is deterministic — pattern matching on the query string,
never an LLM call. The retrieval caller dispatches each sub-query
through the existing retrieve_relevant path and merges results.
"""

from __future__ import annotations

import enum
import re


class QueryShape(enum.Enum):
    DIRECT = "direct"
    CHAIN = "chain"
    SPLIT = "split"


# "of the X of the Y", "X's Y's Z", "the X that owns the Y", etc.
_CHAIN_PATTERNS = [
    re.compile(r"\b(?:of|in|on|by|from|under) the [\w-]+ (?:of|in|on|by|from|under) the\b", re.IGNORECASE),
    re.compile(r"\b\w+'s \w+'s\b"),  # "Alice's manager's email"
    re.compile(r"\b(?:that|which|who) (?:owns|has|holds|contains) the \w+\b", re.IGNORECASE),
]

# "A, B, and C" / "A vs B" / "compare/contrast X and Y"
_SPLIT_PATTERNS = [
    re.compile(r"\bcompare \w+", re.IGNORECASE),
    re.compile(r"\bcontrast \w+", re.IGNORECASE),
    re.compile(r"\bdiff(?:erence)? between\b", re.IGNORECASE),
    re.compile(r"\bvs\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
]

# Splitting tokens used to fan out a SPLIT query.
_SPLIT_DELIM_RE = re.compile(
    r"\s*(?:,| and | or | vs | versus | between )\s*",
    re.IGNORECASE,
)


def classify(query: str) -> QueryShape:
    """Deterministic shape detection from the query string."""
    text = (query or "").strip()
    if not text:
        return QueryShape.DIRECT

    for pat in _CHAIN_PATTERNS:
        if pat.search(text):
            return QueryShape.CHAIN
    for pat in _SPLIT_PATTERNS:
        if pat.search(text):
            return QueryShape.SPLIT
    return QueryShape.DIRECT


def decompose_chain(query: str, *, max_steps: int = 3) -> list[str]:
    """Break a CHAIN query into ordered sub-queries.

    Splits on the canonical "of the X of the Y" pivot and yields up to
    ``max_steps`` sub-questions, innermost first. Falls back to a single
    element when no clean pivot exists.
    """
    parts = re.split(r"\s+of the\s+", query, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [query.strip()]
    # Innermost referent first: paper §5.3 routes "of the X of the Y" as
    # first resolve Y, then resolve X-of-Y, then the outer.
    return parts[::-1][:max_steps]


def decompose_split(query: str, *, max_branches: int = 6) -> list[str]:
    """Break a SPLIT query into parallel sub-queries.

    Splits the query body on comma / "and" / "or" / "vs" / "versus" /
    "between" and returns up to ``max_branches`` sub-queries. Single
    short tokens are filtered out.
    """
    # Drop the leading "compare/contrast/difference between" prefix so
    # the remaining clause is what we actually fan out on.
    body = re.sub(
        r"^(?:compare|contrast|diff(?:erence)? between)\s+",
        "",
        query.strip(),
        flags=re.IGNORECASE,
    )
    branches = _SPLIT_DELIM_RE.split(body)
    branches = [b.strip(" ?.") for b in branches if b.strip(" ?.")]
    if len(branches) <= 1:
        return [query.strip()]
    return branches[:max_branches]


def decompose(query: str) -> tuple[QueryShape, list[str]]:
    """Classify + decompose. Returns (shape, sub_queries).

    For DIRECT, sub_queries is [query].
    """
    shape = classify(query)
    if shape is QueryShape.CHAIN:
        return shape, decompose_chain(query)
    if shape is QueryShape.SPLIT:
        return shape, decompose_split(query)
    return shape, [query.strip()]
