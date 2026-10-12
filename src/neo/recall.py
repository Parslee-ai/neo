"""`neo recall`: project memory relevant to a prompt, for a host to inject.

The Claude Code mod runs this on EVERY user prompt and hands the output to the
model as additional context, so three rules decide its shape:

1. **No model call, no model load.** Only the observer's resident embedder is
   used (`neo.memory.embed_service`). If it does not answer, recall prints
   nothing: loading the ~1.3 GB model per prompt is the cost the resident
   embedder exists to remove.
2. **Nothing is mutated.** The store is opened `read_only` without
   `initialize()` (no ingestion, pruning or saves), and access is not stamped.
3. **Only what the model does not already have.** Rule-file constraints, the
   host's own memory files and git history are already in its context; seed
   and community patterns are textbook; outcome logs and simulation episodes
   are bookkeeping. Measured on this repo's store, 19 of 203 valid facts
   survive, and they are the lessons mined from real sessions.

It never fails the host: every error path prints nothing and exits 0.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Below this cosine a fact is noise for the prompt. Calibrated on a live store
# (jina-v2-base-code): relevant lessons scored 0.52-0.68 against prompts they
# answer, while unrelated prompts topped out at 0.33.
MIN_SIMILARITY = 0.45
MAX_FACTS = 3
MAX_BODY_CHARS = 400

# Provenance the model already has in context, or that carries no lesson.
_REDUNDANT_TAGS = frozenset({
    "history",                  # git log subjects
    "claude-memory",            # the host's own memory files
    "imported:claude-memory",
    "seed",                     # generic textbook patterns
    "community",
    "outcome",                  # suggestion-outcome bookkeeping
})
_REDUNDANT_KINDS = frozenset({"constraint", "episode"})


def _project_root(cwd: str) -> str:
    try:
        top = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=5,
        )
        if top.returncode == 0 and top.stdout.strip():
            return top.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return cwd


def _resident_embedder():
    """The observer's embedder, or None. Never loads a model locally."""
    from neo.memory import embed_service
    from neo.memory.store import DEFAULT_EMBED_MODEL

    return embed_service.connect(DEFAULT_EMBED_MODEL, local_factory=lambda: None)


def recall(prompt: str, cwd: str, embedder=None) -> str:
    """Text block of relevant learned lessons, or "" when there are none."""
    import numpy as np

    from neo.math_utils import batched_cosine
    from neo.memory.store import MAX_TEXT_LENGTH, FactStore
    from neo.text_budget import truncate_marked

    if not prompt.strip():
        return ""
    embedder = embedder if embedder is not None else _resident_embedder()
    if embedder is None:
        return ""
    vectors = list(embedder.embed([prompt[:MAX_TEXT_LENGTH]]))
    if not vectors:
        return ""
    query = np.asarray(vectors[0], dtype=np.float32)

    store = FactStore(
        codebase_root=_project_root(cwd), eager_init=False,
        read_only=True, emit_metrics=False,
    )
    candidates = [
        fact for fact in store.entries
        if fact.is_valid
        and fact.embedding is not None
        and fact.kind.value not in _REDUNDANT_KINDS
        and not _REDUNDANT_TAGS.intersection(fact.tags)
    ]
    if not candidates:
        return ""
    scores = batched_cosine([fact.embedding for fact in candidates], query)
    ranked = sorted(zip(scores, candidates), key=lambda pair: -pair[0])
    hits = [fact for score, fact in ranked[:MAX_FACTS] if score >= MIN_SIMILARITY]
    if not hits:
        return ""

    lines = ["Neo memory: lessons learned in earlier sessions that match this request."]
    for fact in hits:
        body = " ".join(truncate_marked(fact.body, MAX_BODY_CHARS).split())
        lines.append(f"- {fact.subject} [{fact.kind.value}, {fact.scope.value}]: {body}")
    return "\n".join(lines)


def run_recall(argv: list[str], stdin=None) -> int:
    """`neo recall [--cwd DIR]`, prompt on stdin. Always exits 0."""
    try:
        logging.disable(logging.WARNING)
        parser = argparse.ArgumentParser(prog="neo recall")
        parser.add_argument("--cwd", default=str(Path.cwd()))
        args = parser.parse_args(argv)
        prompt = (stdin or sys.stdin).read()
        text = recall(prompt, args.cwd)
        if text:
            print(text)
    except BaseException:  # a host hook must never fail on our account
        pass
    return 0

