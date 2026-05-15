"""
Description generalization for synthesis clustering (paper 2603.10600 §7).

Three normalization passes applied to fact text before clustering, so
specific instances cluster with their abstract twin:

  entity_abstraction  — collapse identifiers, emails, hashes, file paths,
                        version numbers to canonical placeholders
  action_normalization — fold near-synonym verbs to one canonical form
                        ("get/fetch/retrieve/obtain" → "fetch", etc.)
  context_removal     — drop task-specific qualifiers and quoted strings

All three are pure-Python regex; no LLM in the loop. Cheap to run on
every synthesis candidate.
"""

from __future__ import annotations

import re
from typing import Iterable

# Verb synonyms folded to a canonical form.
_VERB_SYNONYMS = {
    "fetch": ("get", "fetch", "retrieve", "obtain", "pull", "download", "load"),
    "send": ("send", "submit", "post", "push", "upload", "transmit"),
    "create": ("create", "make", "build", "add", "new"),
    "delete": ("delete", "remove", "drop", "destroy", "erase"),
    "update": ("update", "modify", "change", "edit", "patch", "alter"),
    "validate": ("validate", "verify", "check", "test", "confirm", "assert"),
    "authenticate": ("authenticate", "auth", "login", "log in", "sign in", "signin"),
}

_REVERSE_VERBS = {syn: canon for canon, syns in _VERB_SYNONYMS.items() for syn in syns}

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_HASH_RE = re.compile(r"\b[0-9a-f]{7,64}\b")
_PATH_RE = re.compile(r"(?:[A-Za-z]:)?(?:[\\/][\w.-]+){2,}")
_VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+){1,3}(?:-[\w.+-]+)?\b")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
# Bare integers/floats are too aggressive; only kill big numeric IDs.
_BIG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
# Anything inside backticks or double quotes is usually task-specific.
_QUOTED_RE = re.compile(r"`[^`]*`|\"[^\"]*\"")


def entity_abstraction(text: str) -> str:
    """Replace IDs, emails, hashes, paths, versions with placeholders."""
    text = _UUID_RE.sub("<uuid>", text)
    text = _EMAIL_RE.sub("<email>", text)
    text = _PATH_RE.sub("<path>", text)
    text = _HASH_RE.sub("<hash>", text)
    text = _VERSION_RE.sub("<version>", text)
    text = _BIG_NUMBER_RE.sub("<n>", text)
    return text


def action_normalization(text: str) -> str:
    """Fold near-synonym verbs to one canonical form.

    Word-boundary aware. Multi-word phrases (e.g. "log in") are handled
    before single-word synonyms so they don't get split.
    """
    # Multi-word phrases first so "log in" doesn't split on "log".
    out = text
    for canon, syns in _VERB_SYNONYMS.items():
        for syn in syns:
            if " " not in syn:
                continue
            out = re.sub(rf"\b{re.escape(syn)}\b", canon, out, flags=re.IGNORECASE)
    # Then word-by-word.
    def _replace(match: re.Match) -> str:
        word = match.group(0).lower()
        return _REVERSE_VERBS.get(word, match.group(0))

    out = re.sub(r"\b[A-Za-z]+\b", _replace, out)
    return out


def context_removal(text: str) -> str:
    """Strip quoted strings and collapse whitespace."""
    text = _QUOTED_RE.sub("<q>", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generalize(text: str) -> str:
    """Apply all three passes in order."""
    return context_removal(action_normalization(entity_abstraction(text)))


def generalize_many(items: Iterable[str]) -> list[str]:
    """Vectorized convenience."""
    return [generalize(s) for s in items]
