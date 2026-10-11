"""Read-only, project-local data for host workspace integrations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

WORKSPACE_SCHEMA_VERSION = 1


def memory_evidence_for(engine) -> list[dict[str, Any]]:
    """Surface recorded retrieval distinctions; do not infer use from similarity."""
    episode = engine.current_learning_episode
    if episode is None:
        return []
    store = engine.fact_store
    facts = {fact.id: fact for fact in store.entries} if store is not None else {}
    result = []
    for item in episode.retrieved_facts:
        row = asdict(item)
        fact = facts.get(item.fact_id)
        if fact is not None:
            row.update({"scope": fact.scope.value, "subject": fact.subject})
        result.append(row)
    return result


def project_snapshot(root: str) -> tuple[str, list, list, list[str]]:
    """Read persisted evidence without store initialization or migrations."""
    from neo.memory.episodes import LearningEpisode
    from neo.memory.models import Fact
    from neo.memory.scope import _compute_legacy_project_id, detect_org_and_project

    _, project_id = detect_org_and_project(root)
    base = Path.home() / ".neo"
    facts = []
    episodes = []
    cautions = []
    path = base / "facts" / f"facts_project_{project_id}.json"
    if not path.exists():
        path = base / "facts" / f"facts_project_{_compute_legacy_project_id(root)}.json"
    if path.exists():
        try:
            document = json.loads(path.read_text())
            for record in document.get("facts", []):
                try:
                    # The inspector never needs to allocate embeddings.
                    fact = Fact.from_dict({**record, "embedding": None})
                    if fact.scope.value == "project" and fact.project_id in {
                        project_id, _compute_legacy_project_id(root)
                    }:
                        facts.append(fact)
                except (ValueError, TypeError, AttributeError):
                    cautions.append("Skipped a malformed project fact.")
        except (OSError, ValueError, AttributeError):
            cautions.append("Project facts could not be read.")
    episode_path = base / "episodes" / project_id
    if episode_path.exists():
        for path in sorted(episode_path.glob("*.json")):
            try:
                episodes.append(LearningEpisode.from_dict(json.loads(path.read_text())))
            except (OSError, ValueError, TypeError, AttributeError):
                cautions.append(f"Skipped unreadable episode {path.stem}.")
    return project_id, facts, episodes, sorted(set(cautions))


def workspace_status(root: str) -> dict[str, Any]:
    """Summarize project facts without aging, learning, or model calls."""
    from neo import __version__

    project_id, facts, episodes, cautions = project_snapshot(root)
    valid = [fact for fact in facts if fact.is_valid]
    return {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "neo_version": __version__,
        "project_id": project_id,
        "checkout": str(Path(root).resolve()),
        "model_calls": 0,
        "cautions": cautions,
        "fact_count": len(valid),
        "probation_count": sum("probation" in fact.tags for fact in valid),
        "episode_count": len(episodes),
        "facts": [
            {"id": fact.id, "subject": fact.subject, "kind": fact.kind.value,
             "probation": "probation" in fact.tags, "is_valid": fact.is_valid}
            for fact in facts
        ],
    }


def workspace_explain(root: str, identifier: str) -> dict[str, Any]:
    """Use the existing explanation join, confined to this project."""
    from neo.memory.explain import explain_fact

    _, facts, episodes, cautions = project_snapshot(root)

    class Snapshot:
        def list(self):
            return episodes

    result = explain_fact(facts, identifier, episode_store=Snapshot())
    result["cautions"] = cautions
    return result


def run_workspace(argv: list[str]) -> int:
    """Host protocol; never bootstrap observers or an inference provider."""
    parser = argparse.ArgumentParser(prog="neo workspace")
    parser.add_argument("action", choices=["status", "explain"])
    parser.add_argument("fact_id", nargs="?")
    parser.add_argument("--cwd", default=str(Path.cwd()))
    args = parser.parse_args(argv)
    try:
        result = (
            workspace_status(args.cwd) if args.action == "status"
            else workspace_explain(args.cwd, args.fact_id or "")
        )
        print(json.dumps(result))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
