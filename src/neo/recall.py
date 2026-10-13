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


STRONG_MATCH = 0.6
FAINT_MATCH = 0.52


def _recall_triggers(scored_hits: list) -> set:
    """The situation, read from what was recalled (see `recall_beats`)."""
    triggers = set()
    top = max(score for score, _ in scored_hits)
    if top >= STRONG_MATCH:
        triggers.add("strong_match")
    elif top < FAINT_MATCH:
        triggers.add("faint_match")
    for _, fact in scored_hits:
        if fact.kind.value == "failure":
            triggers.add("failure_lesson")
        if fact.metadata.success_count > 0 or fact.canonical_signature:
            triggers.add("proven_lesson")
        if fact.scope.value == "global":
            triggers.add("elsewhere")
    return triggers


def _recall_beat(facts: list, scored_hits: list, recent: set) -> dict:
    """The recall beat for this situation at Neo's current stage, or {}.

    Most matching triggers wins, ties in deck order; a beat Neo spoke
    recently in this session is skipped (the host passes them back), and no
    fit means silence. The stage is the engine's own (`memory_stage` of
    `memory_level_of` the loaded facts), so the beats available change as
    Neo remembers more. A missing or malformed deck means silence.
    """
    import yaml

    from neo.memory.store import memory_level_of, memory_stage

    try:
        deck_path = Path(__file__).parent / "config" / "beats" / "neo_matrix.yaml"
        beats = (yaml.safe_load(deck_path.read_text()) or {}).get("recall_beats") or []
        stage = memory_stage(memory_level_of(facts))
        triggers = _recall_triggers(scored_hits)
        best, best_score = {}, 0
        for beat in beats:
            if stage not in (beat.get("stages") or []) or beat.get("beat_id") in recent:
                continue
            score = len(triggers & set(beat.get("trigger_contexts") or []))
            if score > best_score and beat.get("line"):
                best, best_score = beat, score
        return best
    except Exception:
        return {}


def recall(prompt: str, cwd: str,
           embed: Optional[Callable[[str], object]] = None,
           recent_beats: Optional[set] = None) -> dict:
    """`{"text", "beat_id"}`: relevant learned lessons, opened by Neo's beat
    for the situation when one fits. Empty text when nothing is relevant."""
    import numpy as np

    from neo.math_utils import batched_cosine
    from neo.memory.models import is_legacy_suggestion_fact
    from neo.memory.store import MAX_TEXT_LENGTH, STALE_MAX_CONFIDENCE
    from neo.text_budget import truncate_marked

    silent = {"text": "", "beat_id": ""}
    if not prompt.strip():
        return silent
    vector = (embed or _resident_embed)(prompt[:MAX_TEXT_LENGTH])
    if vector is None:
        return silent
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
        return silent
    scores = batched_cosine([fact.embedding for fact in candidates], query)
    ranked = sorted(zip(scores, candidates), key=lambda pair: -pair[0])
    scored_hits = [(score, fact) for score, fact in ranked[:MAX_FACTS]
                   if score >= MIN_SIMILARITY]
    if not scored_hits:
        return silent

    beat = _recall_beat(facts, scored_hits, recent_beats or set())
    lines = [f"Neo: {beat['line']}" if beat else "Neo:"]
    for _, fact in scored_hits:
        body = " ".join(truncate_marked(fact.body, MAX_BODY_CHARS).split())
        lines.append(f"- {fact.subject} [{fact.kind.value}, {fact.scope.value}]: {body}")
    return {"text": "\n".join(lines), "beat_id": beat.get("beat_id", "") if beat else ""}


def run_recall(argv: list[str], stdin=None) -> int:
    """`neo recall [--cwd DIR] [--recent-beats IDS] [--json]`, prompt on stdin.
    Always exits 0."""
    try:
        logging.disable(logging.WARNING)
        parser = argparse.ArgumentParser(prog="neo recall")
        parser.add_argument("--cwd", default=str(Path.cwd()))
        parser.add_argument("--recent-beats", default="",
                            help="comma-separated beat ids Neo spoke recently")
        parser.add_argument("--json", action="store_true")
        args = parser.parse_args(argv)
        prompt = (stdin or sys.stdin).read()
        recent = {b for b in args.recent_beats.split(",") if b}
        result = recall(prompt, args.cwd, recent_beats=recent)
        if args.json:
            print(json.dumps(result))
        elif result["text"]:
            print(result["text"])
    except BaseException:  # a host hook must never fail on our account
        pass
    return 0
