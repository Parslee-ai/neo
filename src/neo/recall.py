"""`neo recall`: project memory relevant to a prompt, for a host to inject.

The Claude Code mod runs this on EVERY user prompt and hands the output to the
model as additional context, so three rules decide its shape:

1. **No model call, no model load, a hard deadline.** One request to the
   observer's resident embedder (`embed_service.embed_query`): no retries, no
   local fallback, `EMBED_DEADLINE_S` over the whole exchange. If it does not
   answer in time, recall prints nothing: loading the ~1.3 GB model, or
   queueing behind the observer's own sweep, per prompt is not affordable.
2. **Nothing is written.** The fact files are read directly. Constructing a
   `FactStore` is NOT read-only even with `read_only=True`: it creates
   directories, migrates legacy file names and backs up a corrupt file, all
   outside `save()`. Nothing here writes a metrics line either.
3. **Only what the model does not already have.** Rule-file constraints, the
   host's own memory files and git history are already in its context; seed
   and community patterns are textbook; outcome logs and simulation episodes
   are bookkeeping. Legacy unverified suggestion facts and low-confidence
   facts, which a store's cold start would retire or prune, are skipped here
   because nothing here runs that cold start. Measured on this repo's store,
   19 of 203 valid facts survive, and they are the lessons mined from real
   sessions.

It never fails the host: every error path prints nothing and exits 0.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

# Below this cosine a fact is noise for the prompt. Calibrated on a live store
# (jina-v2-base-code): relevant lessons scored 0.52-0.68 against prompts they
# answer, while unrelated prompts topped out at 0.33.
MIN_SIMILARITY = 0.45
MAX_FACTS = 3
MAX_BODY_CHARS = 400
# The whole embedding exchange. A short query takes ~0.16 s on an idle server;
# past this the observer is busy with its own sweep and we stay silent.
EMBED_DEADLINE_S = 1.0

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


def _resident_embed(text: str):
    from neo.memory import embed_service
    from neo.memory.store import DEFAULT_EMBED_MODEL

    return embed_service.embed_query(text, DEFAULT_EMBED_MODEL, EMBED_DEADLINE_S)


def _load_facts(root: str) -> list:
    """Global, org and project facts, parsed straight from disk."""
    from neo.memory.models import Fact
    from neo.memory.scope import _compute_legacy_project_id, detect_org_and_project

    org_id, project_id = detect_org_and_project(root)
    base = Path.home() / ".neo" / "facts"
    paths = [base / "facts_global.json"]
    if org_id != "unknown":
        paths.append(base / f"facts_org_{org_id}.json")
    if project_id:
        project = base / f"facts_project_{project_id}.json"
        if not project.exists():
            project = base / f"facts_project_{_compute_legacy_project_id(root)}.json"
        paths.append(project)
    facts = []
    for path in paths:
        try:
            records = json.loads(path.read_text()).get("facts", [])
        except (OSError, ValueError, AttributeError):
            continue
        for record in records:
            try:
                facts.append(Fact.from_dict(record))
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
    return facts


_RECALL_OPENER_FALLBACK = "I remember this. From earlier sessions:"


def _recall_opener(facts: list) -> str:
    """Neo's opener for the stage his memory has reached, from the beat deck.

    The engine phrases its summary by the same stage (`memory_stage` of
    `memory_level_of` the loaded facts), so recall sounds like the Neo whose
    findings follow it. A missing or malformed deck degrades to the
    Unplugged line rather than failing the prompt."""
    import yaml

    from neo.memory.store import memory_level_of, memory_stage

    try:
        deck_path = Path(__file__).parent / "config" / "beats" / "neo_matrix.yaml"
        deck = yaml.safe_load(deck_path.read_text()) or {}
        stages = deck.get("orchestrator_voice", {}).get("stages", {})
        stage = stages.get(memory_stage(memory_level_of(facts))) or {}
        return str(stage.get("recall_opener") or _RECALL_OPENER_FALLBACK)
    except Exception:
        return _RECALL_OPENER_FALLBACK


def recall(prompt: str, cwd: str,
           embed: Optional[Callable[[str], object]] = None) -> str:
    """Text block of relevant learned lessons, or "" when there are none."""
    import numpy as np

    from neo.math_utils import batched_cosine
    from neo.memory.models import is_legacy_suggestion_fact
    from neo.memory.store import MAX_TEXT_LENGTH, STALE_MAX_CONFIDENCE
    from neo.text_budget import truncate_marked

    if not prompt.strip():
        return ""
    vector = (embed or _resident_embed)(prompt[:MAX_TEXT_LENGTH])
    if vector is None:
        return ""
    query = np.asarray(vector, dtype=np.float32)

    facts = _load_facts(_project_root(cwd))
    candidates = [
        fact for fact in facts
        if fact.is_valid
        and fact.embedding is not None
        and np.shape(fact.embedding) == query.shape  # a foreign model's vector
        and fact.metadata.confidence >= STALE_MAX_CONFIDENCE
        and fact.kind.value not in _REDUNDANT_KINDS
        and not _REDUNDANT_TAGS.intersection(fact.tags)
        and not is_legacy_suggestion_fact(fact)
    ]
    if not candidates:
        return ""
    scores = batched_cosine([fact.embedding for fact in candidates], query)
    ranked = sorted(zip(scores, candidates), key=lambda pair: -pair[0])
    hits = [fact for score, fact in ranked[:MAX_FACTS] if score >= MIN_SIMILARITY]
    if not hits:
        return ""

    lines = [f"Neo memory: {_recall_opener(facts)}"]
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

