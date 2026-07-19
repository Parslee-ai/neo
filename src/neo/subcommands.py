"""
Neo CLI subcommand handlers.

Contains all handler functions for CLI subcommands (version, help, config,
construct, prompt, etc.) and their supporting utilities.

Split from cli.py for modularity.
"""

import contextlib
import copy
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from neo.models import RegenerateStats

if TYPE_CHECKING:
    from neo.config import NeoConfig
    from neo.persistent_reasoning import PersistentReasoningMemory

# Initialize logger
logger = logging.getLogger(__name__)

# Embedding regeneration configuration
MIN_EMBEDDING_SUCCESS_RATE = 0.8  # Require 80% success to prevent mass data corruption
VALID_EMBEDDING_DIMENSIONS = {384, 768, 1536}  # BGE-small, Jina-v2, OpenAI


def show_version(codebase_root: Optional[str] = None):
    """Show Neo's current state and journey progress."""
    from neo.config import NeoConfig
    from neo.storage import FileStorage
    import importlib.metadata
    import yaml

    # Get package version
    try:
        version = importlib.metadata.version("neo-reasoner")
    except Exception:
        version = "unknown"

    config = NeoConfig.load()
    memory_backend = getattr(config, "memory_backend", "fact_store")

    if memory_backend == "fact_store":
        from neo.memory.store import FactStore
        memory = FactStore(codebase_root=codebase_root, config=config, eager_init=False)
    else:
        from neo.persistent_reasoning import PersistentReasoningMemory
        memory = PersistentReasoningMemory(codebase_root=codebase_root, config=config)

    level = memory.memory_level()
    entries = memory.entries
    total_entries = len(entries)
    avg_confidence = sum(
        e.metadata.confidence if hasattr(e, 'metadata') else getattr(e, 'confidence', 0.0)
        for e in entries
    ) / total_entries if total_entries > 0 else 0.0

    # Determine stage (1-5)
    if level < 0.2:
        stage_num = 1
        stage = "Sleeper"
    elif level < 0.4:
        stage_num = 2
        stage = "Glitch"
    elif level < 0.6:
        stage_num = 3
        stage = "Unplugged"
    elif level < 0.8:
        stage_num = 4
        stage = "Training"
    else:
        stage_num = 5
        stage = "The One"

    # Load beat deck to get personality quote
    quote = "What is real? How do you define 'real'?"  # Default fallback
    try:
        beat_deck_path = Path(__file__).parent / "config" / "beats" / "neo_matrix.yaml"
        if beat_deck_path.exists():
            with open(beat_deck_path, "r") as f:
                beat_deck = yaml.safe_load(f)
                if beat_deck and "base_expressions" in beat_deck:
                    stage_expr = beat_deck["base_expressions"].get(stage_num, {})
                    quote = stage_expr.get("internal", quote)
    except Exception:
        pass  # Use fallback quote if loading fails

    # Detect storage backend type
    storage_info = ""
    storage_backend = getattr(memory, 'storage_backend', None)
    if storage_backend is not None and isinstance(storage_backend, FileStorage):
        base_path = getattr(storage_backend, 'base_path', 'unknown')
        storage_info = f"FileStorage (path: {base_path})"
    elif memory_backend == "fact_store":
        storage_info = f"FactStore (path: {Path.home() / '.neo' / 'facts'})"
    else:
        storage_info = f"{type(memory).__name__}"

    # Display personality quote first
    print(f'"{quote}"\n')

    # Then technical output
    bar_filled = int(level * 40)
    bar = '\u2588' * bar_filled + '\u2591' * (40 - bar_filled)

    print(f"neo {version}")
    print(f"Provider: {config.provider} | Model: {config.model}")
    print(f"Storage: {storage_info}")
    try:
        from neo.car_discovery import discover_car
        print(f"CAR: {discover_car().summary()}")
    except Exception:
        pass
    print(f"Stage: {stage} | Memory: {level:.1%}")
    print(f"{bar}")
    print(f"{total_entries} patterns | {avg_confidence:.2f} avg confidence")

    # Community contribution status
    if memory_backend == "fact_store" and hasattr(memory, 'find_contributable'):
        contributable = memory.find_contributable()
        if contributable:
            print(f"\n\u2728 {len(contributable)} pattern(s) ready to share with the community")
            print("   Run: neo contribute")
        else:
            # Show progress toward contribution
            valid = [f for f in entries if getattr(f, 'is_valid', True)]
            near = [f for f in valid
                    if hasattr(f, 'metadata')
                    and f.metadata.confidence >= 0.6
                    and f.metadata.success_count >= 1]
            if near:
                print(f"\n\u26a1 {len(near)} pattern(s) approaching contribution (need 0.8 confidence + 3 successes)")
            elif valid:
                best_conf = max((f.metadata.confidence for f in valid if hasattr(f, 'metadata')), default=0)
                best_succ = max((f.metadata.success_count for f in valid if hasattr(f, 'metadata')), default=0)
                print(f"\n\u26a1 {len(valid)} pattern(s), none yet contributable (best: {best_conf:.0%} confidence, {best_succ} successes — need 0.8 + 3)")
            else:
                print("\n\u26a1 No patterns yet. Use neo to build patterns — validated ones can be shared via: neo contribute")
    print()


def handle_contribute(args):
    """Export high-quality patterns and open a GitHub PR draft."""
    from neo.config import NeoConfig
    from neo.memory.store import FactStore

    codebase_root = getattr(args, 'cwd', None) or os.getcwd()
    config = NeoConfig.load()
    memory = FactStore(codebase_root=codebase_root, config=config, eager_init=False)

    contributable = memory.find_contributable()
    if not contributable:
        print("No patterns ready to contribute yet.")
        print("Patterns qualify when they reach high confidence (>0.8) with 3+ successes.")
        return

    print(f"Found {len(contributable)} pattern(s) ready to contribute:\n")

    # Anonymize and format for community_facts.json
    exported = []
    for fact in contributable:
        # Strip identifying info
        entry = {
            "subject": fact.subject,
            "body": fact.body,
            "kind": fact.kind.value,
            "tags": [t for t in fact.tags
                     if t not in ("auto-ingested", "claude-memory", "synthesized")],
        }
        exported.append(entry)
        conf = fact.metadata.confidence
        successes = fact.metadata.success_count
        print(f"  [{conf:.0%} confidence, {successes} successes] {fact.subject}")
        print(f"    {fact.body[:100]}{'...' if len(fact.body) > 100 else ''}")
        print()

    # Write to temp file
    import json
    import tempfile
    export_data = {"contributed_by": "neo-user", "facts": exported}
    export_file = Path(tempfile.gettempdir()) / "neo_contribution.json"
    export_file.write_text(json.dumps(export_data, indent=2))
    print(f"Exported to: {export_file}")
    print("\nTo contribute, open a PR adding these to community_facts.json:")
    print("  https://github.com/Parslee-ai/neo/edit/main/community_facts.json")


def _linked_feedback_projects(*, include_fallback: bool = False) -> list[tuple[str, str]]:
    """Return unique (project_id, root) pairs with linked session feedback."""
    sessions_dir = Path.home() / ".neo" / "sessions"
    projects: dict[str, tuple[str, float]] = {}
    if not sessions_dir.exists():
        return []

    session_files = list(sessions_dir.glob("session_log_*.jsonl"))
    if include_fallback:
        session_files += list(sessions_dir.glob("session_*.json"))
    for path in session_files:
        if path.name.startswith("session_log_"):
            docs = []
            try:
                for line in path.read_text(errors="replace").splitlines():
                    if line.strip():
                        docs.append(json.loads(line))
            except (OSError, json.JSONDecodeError):
                continue
        else:
            try:
                docs = [json.loads(path.read_text(errors="replace"))]
            except (OSError, json.JSONDecodeError):
                continue

        for data in docs:
            pid = data.get("project_id")
            root = data.get("codebase_root")
            links = data.get("suggestion_fact_ids") or {}
            ts = data.get("timestamp", 0.0)
            if not pid or not root or not links:
                continue
            if not Path(root).exists():
                continue
            prev = projects.get(pid)
            if prev is None or ts > prev[1]:
                projects[pid] = (root, ts)

    return [(pid, root_ts[0]) for pid, root_ts in sorted(projects.items(), key=lambda item: item[1][1], reverse=True)]


def _compact_fact_file(path: Path, *, max_invalid_age_days: int = 30, dry_run: bool = False) -> dict:
    """Drop old invalid facts (and strip retained tombstones' embeddings) from
    one scoped fact JSON file.

    Serializes the read-modify-write under the same sidecar ``<file>.lock`` that
    ``FactStore.save`` holds, so a concurrent observer/request-path save can't be
    clobbered — this compactor is an out-of-class writer, and stripping now makes
    it write on nearly every run instead of rarely. Dry-run takes no lock (it
    never writes).
    """
    from neo.memory.store import scope_file_lock

    lock = contextlib.nullcontext() if dry_run else scope_file_lock(path)
    with lock:
        return _compact_fact_file_locked(
            path, max_invalid_age_days=max_invalid_age_days, dry_run=dry_run
        )


def _compact_fact_file_locked(path: Path, *, max_invalid_age_days: int, dry_run: bool) -> dict:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"status": "error", "path": str(path), "error": str(exc)}

    facts = data.get("facts", [])
    if not isinstance(facts, list):
        return {"status": "error", "path": str(path), "error": "facts is not a list"}

    now = time.time()
    cutoff = max(0, max_invalid_age_days) * 86400
    kept = []
    removed = 0
    stripped = 0
    for fact in facts:
        if not isinstance(fact, dict):
            kept.append(fact)
            continue
        is_valid = fact.get("is_valid", True)
        metadata = fact.get("metadata") if isinstance(fact.get("metadata"), dict) else {}
        last_accessed = metadata.get("last_accessed") or metadata.get("created_at") or 0
        try:
            age = now - float(last_accessed)
        except (TypeError, ValueError):
            age = 0
        # Retracted-signature ledger: tombstones invalidated by repeated
        # attributed contradiction are kept indefinitely so a rolled-back
        # pattern can't be re-minted after the 30-day window. Mirrors
        # FactStore.purge_dead_facts.
        retracted = fact.get("invalidation_reason") == "repeated_attributed_contradiction"
        if not is_valid and age >= cutoff and not retracted:
            removed += 1
            continue
        # Retained tombstone (invalid but too young to purge): its embedding is
        # dead weight — invalid facts are never retrieved or deduped — so drop
        # the ~24 KB vector while keeping the row for supersession/audit. Mirrors
        # FactStore.strip_tombstone_embeddings on the cold-start path.
        if not is_valid and fact.get("embedding") is not None:
            fact = {k: v for k, v in fact.items() if k != "embedding"}
            stripped += 1
        kept.append(fact)

    if (removed or stripped) and not dry_run:
        updated = dict(data)
        updated["facts"] = kept
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(updated, indent=2))
        tmp.replace(path)

    return {
        "status": "ok",
        "path": str(path),
        "total": len(facts),
        "removed": removed,
        "stripped": stripped,
        "remaining": len(kept),
    }


def _fact_files_for_prune(*, all_files: bool, cwd: Optional[str] = None) -> list[Path]:
    """Return fact JSON files to compact."""
    facts_dir = Path.home() / ".neo" / "facts"
    if all_files:
        return sorted(facts_dir.glob("facts_*.json"))

    from neo.memory.scope import detect_org_and_project

    root = cwd or os.getcwd()
    org_id, project_id = detect_org_and_project(root)
    files = [facts_dir / "facts_global.json"]
    if org_id != "unknown":
        files.append(facts_dir / f"facts_org_{org_id}.json")
    if project_id:
        files.append(facts_dir / f"facts_project_{project_id}.json")
    return [p for p in files if p.exists()]


def _handle_observer(args) -> None:
    from neo.memory.observer import (
        kick_observer,
        observer_status,
        start_observer,
        stop_observer,
    )

    sub = getattr(args, "observer_action", None)
    cwd = getattr(args, "cwd", None) or os.getcwd()

    if sub == "start":
        result = start_observer(cwd)
    elif sub == "stop":
        result = stop_observer(cwd)
    elif sub == "kick":
        result = kick_observer(cwd)
    elif sub == "status":
        result = observer_status(cwd)
    else:
        print("Usage: neo memory observer {start|stop|status|kick} [--cwd PATH]")
        return

    if result.get("status") == "error":
        print(f"[Neo] observer error: {result.get('message')}", file=sys.stderr)
        return

    parts = [f"[Neo] observer {sub}: {result.get('status')}"]
    if result.get("agent_id"):
        parts.append(f"id={result['agent_id']}")
    if result.get("pid"):
        parts.append(f"pid={result['pid']}")
    if result.get("project_id"):
        parts.append(f"project={result['project_id'][:8]}")
    rc = result.get("restart_count")
    if rc is not None and rc > 0:
        parts.append(f"restarts={rc}")
    if result.get("log_file"):
        parts.append(f"log={result['log_file']}")
    print(" ".join(parts))

    orphans = result.get("orphans") or []
    if orphans:
        pidlist = " ".join(str(p) for p in orphans)
        print(
            f"[Neo] WARNING: {len(orphans)} orphaned observer process(es) for this "
            f"project not supervised by CAR (pid {pidlist}) — a leftover from a prior "
            f"car-server (pre-0.18.0 footgun). It runs redundant synthesis cycles; "
            f"reap with: kill {pidlist}",
            file=sys.stderr,
        )


def _parse_since(value: str) -> Optional[float]:
    """Parse a duration like ``14d`` / ``48h`` / ``30m`` / ``3600s`` to seconds.

    Returns None for an empty value or ``all`` (no window). A bare number is
    treated as seconds. Raises ValueError on anything else.
    """
    if not value:
        return None
    v = value.strip().lower()
    if v in ("all", "0"):
        return None
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
    if v[-1] in units and v[:-1].replace(".", "", 1).isdigit():
        return float(v[:-1]) * units[v[-1]]
    if v.replace(".", "", 1).isdigit():
        return float(v)
    raise ValueError(f"invalid --since value: {value!r} (use e.g. 14d, 48h, 30m)")


def _handle_import(args) -> None:
    """Import a peer tool's memory files into neo's store (on probation)."""
    from neo.config import NeoConfig
    from neo.memory.memimport import import_memory
    from neo.memory.store import FactStore

    root = getattr(args, "cwd", None) or os.getcwd()
    config = NeoConfig.load()
    store = FactStore(codebase_root=root, config=config, eager_init=False)

    stats = import_memory(
        store,
        root=root,
        confidence=getattr(args, "confidence", 0.4),
        dry_run=getattr(args, "dry_run", False),
    )

    if stats.note and stats.scanned == 0:
        print(f"[Neo] memory import: {stats.note}")
        return

    verb = "would import" if stats.dry_run else "imported"
    print(
        f"[Neo] memory import — scanned {stats.scanned}; {verb} {stats.imported}"
        + (f", deduped {stats.deduped}" if stats.deduped else "")
        + (f", skipped {stats.skipped_existing} already-imported" if stats.skipped_existing else "")
        + (f", skipped {stats.skipped_malformed} malformed" if stats.skipped_malformed else "")
    )
    if not stats.dry_run and stats.imported:
        print("  imported as REVIEW facts on probation (decaying, must earn promotion); "
              "tagged 'imported:claude-memory'")


def _handle_audit(args) -> None:
    """Audit an AI tool's memory files for hygiene issues (read-only)."""
    from neo.config import NeoConfig
    from neo.memory.memaudit import find_memory_audit
    from neo.memory.store import FactStore

    root = getattr(args, "cwd", None) or os.getcwd()
    config = NeoConfig.load()
    store = FactStore(codebase_root=root, config=config, eager_init=False)

    check_conflicts = not getattr(args, "no_conflicts", False)
    lm = None
    if check_conflicts:
        try:
            from neo.adapters import resolve_adapter

            lm = resolve_adapter(config)
        except Exception as e:
            print(
                f"[Neo] audit: no LM adapter for the conflict check ({e}); "
                "reporting other findings only.",
                file=sys.stderr,
            )

    report = find_memory_audit(
        store, root=root, check_conflicts=check_conflicts, lm_adapter=lm
    )

    if getattr(args, "json", False):
        payload = {
            "memory_dir": report.memory_dir,
            "entry_count": report.entry_count,
            "clean": report.clean,
            "note": report.note,
            "malformed": [{"file": f, "issue": i} for f, i in report.malformed],
            "duplicates": [{"names": d.names} for d in report.duplicates],
            "conflicts": [
                {"name_a": c.name_a, "name_b": c.name_b, "explanation": c.explanation}
                for c in report.conflicts
            ],
            "index_issues": report.index_issues,
        }
        print(json.dumps(payload, indent=2))
        return

    if report.note and report.entry_count == 0:
        print(f"[Neo] memory audit: {report.note}")
        return

    print(f"[Neo] memory audit — {report.entry_count} memories in {report.memory_dir}")
    if report.clean:
        print("  ✓ no hygiene issues found")
        return
    if report.malformed:
        print(f"\n  Malformed ({len(report.malformed)}):")
        for fname, issue in report.malformed:
            print(f"    • {fname}: {issue}")
    if report.duplicates:
        print(f"\n  Near-duplicates ({len(report.duplicates)}):")
        for d in report.duplicates:
            print(f"    • {', '.join(d.names)}")
    if report.conflicts:
        print(f"\n  Conflicts ({len(report.conflicts)}):")
        for c in report.conflicts:
            print(f"    • {c.name_a} ↔ {c.name_b}: {c.explanation}")
    if report.index_issues:
        print(f"\n  MEMORY.md index ({len(report.index_issues)}):")
        for issue in report.index_issues:
            print(f"    • {issue}")
    print()


def _handle_rules(args) -> None:
    """Flag drift between AGENTS.md / CLAUDE.md / GEMINI.md (read-only)."""
    from neo.config import NeoConfig
    from neo.memory.rulesync import find_rule_sync
    from neo.memory.store import FactStore

    root = getattr(args, "cwd", None) or os.getcwd()
    config = NeoConfig.load()
    store = FactStore(codebase_root=root, config=config, eager_init=False)

    check_conflicts = not getattr(args, "no_conflicts", False)
    lm = None
    if check_conflicts:
        try:
            from neo.adapters import resolve_adapter

            lm = resolve_adapter(config)
        except Exception as e:
            print(
                f"[Neo] rules: no LM adapter for the conflict check ({e}); "
                "reporting gaps only.",
                file=sys.stderr,
            )

    report = find_rule_sync(
        store, root=root, check_conflicts=check_conflicts, lm_adapter=lm
    )

    if getattr(args, "json", False):
        payload = {
            "files": [
                {"tool": f.tool, "path": f.path, "rule_count": len(f.units)}
                for f in report.files
            ],
            "in_sync": report.in_sync,
            "note": report.note,
            "gaps": [
                {
                    "rule": g.rule,
                    "present_in": g.present_in,
                    "missing_from": g.missing_from,
                    "suggestion": g.suggestion,
                }
                for g in report.gaps
            ],
            "conflicts": [
                {
                    "tool_a": c.tool_a, "text_a": c.text_a,
                    "tool_b": c.tool_b, "text_b": c.text_b,
                    "explanation": c.explanation,
                    "suggestion": c.suggestion,
                }
                for c in report.conflicts
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    files_desc = ", ".join(f"{f.tool.upper()}.md ({len(f.units)} rules)" for f in report.files) or "none"
    print(f"[Neo] memory rules — files: {files_desc}")
    if report.note:
        print(f"  {report.note}")
    if report.in_sync:
        if not report.note:
            print("  ✓ rule files are in sync (no gaps or conflicts)")
        return

    if report.gaps:
        print(f"\n  Gaps ({len(report.gaps)}):")
        for g in report.gaps:
            print(f"    • present in {', '.join(g.present_in)}; missing from {', '.join(g.missing_from)}")
            print(f"      {g.rule}")
            print(f"      → {g.suggestion}")
    if report.conflicts:
        print(f"\n  Conflicts ({len(report.conflicts)}):")
        for c in report.conflicts:
            print(f"    • {c.tool_a.upper()}.md vs {c.tool_b.upper()}.md: {c.explanation}")
            print(f"      → {c.suggestion}")
    print()


def _handle_issues(args) -> None:
    """Surface recurring frictions mined from transcript history (read-only)."""
    from neo.config import NeoConfig
    from neo.memory.issues import find_issues
    from neo.memory.store import FactStore

    root = getattr(args, "cwd", None) or os.getcwd()
    try:
        since_seconds = _parse_since(getattr(args, "since", "14d"))
    except ValueError as e:
        print(f"[Neo] {e}", file=sys.stderr)
        return
    min_cluster = getattr(args, "min_cluster", 3)

    config = NeoConfig.load()
    store = FactStore(codebase_root=root, config=config, eager_init=False)
    issues = find_issues(store, since_seconds=since_seconds, min_cluster=min_cluster)

    if getattr(args, "suggest_rules", False) and issues:
        try:
            from neo.adapters import resolve_adapter
            from neo.memory.issues import suggest_rules

            adapter = resolve_adapter(config)
        except Exception as e:
            print(
                f"[Neo] --suggest-rules: could not build an LM adapter ({e}); "
                "reporting issues without suggested rules.",
                file=sys.stderr,
            )
        else:
            suggest_rules(issues, adapter)

    if getattr(args, "json", False):
        payload = [
            {
                "title": iss.title,
                "category": iss.category,
                "confidence": iss.confidence,
                "session_count": iss.session_count,
                "member_count": iss.member_count,
                "evidence": [
                    {"session_id": ev.session_id, "timestamp": ev.timestamp, "span": ev.span}
                    for ev in iss.evidence
                ],
                "suggested_rule": iss.suggested_rule,
            }
            for iss in issues
        ]
        print(json.dumps(payload, indent=2))
        return

    window = getattr(args, "since", "14d")
    if not issues:
        print(f"[Neo] memory issues: no recurring frictions found (window: {window}).")
        return

    print(f"[Neo] memory issues — {len(issues)} issue(s) (window: {window})\n")
    for iss in issues:
        print(f"[{iss.confidence:.2f}] {iss.title}    ·{iss.category}")
        print(f"  {iss.member_count} episodes across {iss.session_count} sessions")
        for ev in iss.evidence:
            print(f"    {ev.timestamp or '?'}  {ev.span}")
        if iss.suggested_rule:
            print(f"  suggested rule: {iss.suggested_rule}")
        print()


def _handle_explain(args) -> None:
    """Render one fact's persisted causal chain without initializing an LM."""
    from neo.config import NeoConfig
    from neo.memory.episodes import LearningEpisodeStore
    from neo.memory.explain import (
        FactLookupError,
        LearningEpisodeCatalog,
        explain_fact,
        resolve_fact,
    )
    from neo.memory.models import FactScope
    from neo.memory.store import FactStore

    root = getattr(args, "cwd", None) or os.getcwd()
    store = FactStore(
        codebase_root=root,
        config=NeoConfig.load(),
        eager_init=False,
    )
    try:
        fact = resolve_fact(store.entries, getattr(args, "fact_id", ""))
        episode_source = (
            LearningEpisodeCatalog(Path.home() / ".neo" / "episodes")
            if fact.scope == FactScope.GLOBAL
            else LearningEpisodeStore(store.project_id or "unscoped")
        )
        explanation = explain_fact(
            store.entries,
            fact.id,
            episode_store=episode_source,
        )
    except FactLookupError as exc:
        print(f"[Neo] memory explain: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if getattr(args, "json", False):
        print(json.dumps(explanation, indent=2, sort_keys=True))
        return

    fact = explanation["fact"]
    state = "valid" if fact["is_valid"] else "invalid"
    print(f"[Neo] fact {fact['id']} — {state}")
    print(f"  {fact['subject']}")
    print(
        f"  {fact['kind']} / {fact['scope']} · confidence={fact['confidence']:.2f} "
        f"· successes={fact['success_count']} · effectiveness={fact['effectiveness']:.2f}"
    )
    print(f"  provenance={fact['provenance']} candidate={fact['source_candidate_id'] or '-'}")
    if fact["invalidation_reason"]:
        print(f"  invalidation={fact['invalidation_reason']}")

    print(f"\nSupporting evidence ({len(explanation['supporting_evidence'])})")
    for item in explanation["supporting_evidence"]:
        if item.get("missing"):
            print(f"  {item['episode_id']} · missing local episode")
        else:
            print(
                f"  {item['episode_id']} · {item['final_outcome']} "
                f"· revision={item['repository_revision'] or '-'}"
            )

    print(f"\nContradicting evidence ({len(explanation['contradicting_evidence'])})")
    for item in explanation["contradicting_evidence"]:
        if item.get("missing"):
            print(f"  {item['episode_id']} · missing local episode")
        else:
            print(
                f"  {item['episode_id']} · {item['final_outcome']} "
                f"· revision={item['repository_revision'] or '-'}"
            )

    print(f"\nRetrieval history ({len(explanation['retrieval_history'])})")
    for item in explanation["retrieval_history"]:
        score = "?" if item["score"] is None else f"{item['score']:.4f}"
        print(
            f"  {item['episode_id']} · score={score} · "
            f"included={str(item['included_in_context']).lower()} · "
            f"outcome={item['outcome_association'] or '-'}"
        )

    print(f"\nMemory mutations ({len(explanation['mutation_history'])})")
    for item in explanation["mutation_history"]:
        before = item["before_state"]
        after = item["after_state"]
        transition = ""
        if before or after:
            transition = f" · {before or '{}'} -> {after or '{}'}"
        print(f"  {item['episode_id']} · {item['operation']}{transition}")

    previous = explanation["supersession"]["previous"]
    replacements = explanation["supersession"]["replacements"]
    print(f"\nSupersession · previous={len(previous)} replacements={len(replacements)}")


def _handle_evaluate_learning(args) -> None:
    """Run and render the deterministic learning-loop acceptance benchmark."""
    from neo.memory.evaluation import run_learning_evaluation

    corpus = Path(args.corpus).resolve() if getattr(args, "corpus", None) else None
    workspace = (
        Path(args.workspace).resolve() if getattr(args, "workspace", None) else None
    )
    report = run_learning_evaluation(corpus_path=corpus, workspace=workspace)
    payload = report.to_dict()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        status = "PASS" if report.accepted else "FAIL"
        print(f"[Neo] evidence-learning evaluation — {status}")
        print(f"  benchmark={report.benchmark_id}")
        for mode in report.modes:
            metrics = mode.metrics
            print(
                f"  {mode.mode}: success={metrics.task_success_rate:.2f} "
                f"adherence={metrics.constraint_adherence:.2f} "
                f"precision={metrics.retrieval_precision:.2f} "
                f"harmful={metrics.harmful_memory_rate:.2f} "
                f"unsupported={metrics.unsupported_promotion_rate:.2f} "
                f"repeat-error={metrics.repeat_error_rate:.2f} "
                f"latency={metrics.latency_ms:.1f}ms "
                f"model-calls={metrics.model_calls} tokens={metrics.token_usage}"
            )
        for scenario in report.modes[-1].scenarios:
            marker = "PASS" if scenario.passed else "FAIL"
            print(f"    [{marker}] {scenario.id}")
        for failure in report.acceptance_failures:
            print(f"  failure: {failure}")
    if not report.accepted:
        raise SystemExit(1)


def _handle_citation_stats(args) -> None:
    """Summarize citation_survival events from metrics.jsonl: how often a
    retrieved [fact:id] survives into reasoning, and WHICH detector earns the
    credit (marker vs structured self-report vs subject overlap). Answers the
    open question of whether the reliable self-report carries the retrieved-fact
    reinforcement path or the softer overlap signal is doing the work."""
    path = Path.home() / ".neo" / "metrics.jsonl"

    cutoff = None
    if getattr(args, "since", None):
        try:
            window = _parse_since(args.since)
        except ValueError as exc:
            print(f"[Neo] {exc}", file=sys.stderr)
            return
        cutoff = (time.time() - window) if window else None

    agg = {k: 0 for k in (
        "requests", "retrieved", "included", "used",
        "by_marker", "by_self_report", "by_overlap", "by_overlap_only",
    )}
    by_model: dict[str, dict[str, int]] = {}
    if path.exists():
        with path.open() as handle:
            for line in handle:
                # Cheap pre-filter before parsing every metrics line.
                if '"citation_survival"' not in line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Valid-but-non-object JSON (a bare scalar/list that happens to
                # contain the substring) would crash .get() — skip it.
                if not isinstance(event, dict) or event.get("event") != "citation_survival":
                    continue
                if cutoff is not None and float(event.get("ts", 0) or 0) < cutoff:
                    continue
                agg["requests"] += 1
                for key in ("retrieved", "included", "used", "by_marker",
                            "by_self_report", "by_overlap", "by_overlap_only"):
                    agg[key] += int(event.get(key, 0) or 0)
                model = event.get("model") or "unknown"
                stats = by_model.setdefault(model, {"requests": 0, "included": 0, "used": 0})
                stats["requests"] += 1
                stats["included"] += int(event.get("included", 0) or 0)
                stats["used"] += int(event.get("used", 0) or 0)

    survival = (agg["used"] / agg["included"] * 100) if agg["included"] else 0.0
    window_label = f" (since {args.since})" if getattr(args, "since", None) else ""

    if getattr(args, "json", False):
        print(json.dumps({
            **agg,
            "survival_rate_pct": round(survival, 2),
            "by_model": by_model,
        }, indent=2))
        return

    if agg["requests"] == 0:
        print(f"[Neo] citation-survival{window_label}: "
              "no citation_survival events recorded yet")
        return
    print(f"[Neo] citation-survival{window_label}: {agg['requests']} request(s)")
    print(f"  retrieved={agg['retrieved']}  included={agg['included']}  used={agg['used']}")
    print(f"  survival rate (used/included): {survival:.1f}%")
    denom = agg["used"] or 1
    print("  which detector earned it (share of 'used'; a fact can match >1):")
    for label, key in (("marker", "by_marker"),
                       ("self-report", "by_self_report"),
                       ("overlap", "by_overlap")):
        count = agg[key]
        print(f"    {label:<12} {count:>7}  ({count / denom * 100:5.1f}%)")
    # The decision number: facts credited by overlap and nothing else.
    only = agg["by_overlap_only"]
    print(f"  overlap-only credited: {only}  ({only / denom * 100:.1f}% of used) "
          f"-> {'overlap is dead weight, safe to drop' if only == 0 else 'overlap is load-bearing, tune not cut'}")
    if by_model:
        print("  by model:")
        for model, stats in sorted(by_model.items(), key=lambda kv: -kv[1]["requests"]):
            rate = (stats["used"] / stats["included"] * 100) if stats["included"] else 0.0
            print(f"    {model:<20} {stats['requests']:>5} req  "
                  f"used={stats['used']}  ({rate:.1f}% survival)")


def handle_memory(args):
    """Memory maintenance subcommands."""
    action = getattr(args, "memory_action", None)
    if action == "observer":
        _handle_observer(args)
        return
    if action == "issues":
        _handle_issues(args)
        return
    if action == "rules":
        _handle_rules(args)
        return
    if action == "audit":
        _handle_audit(args)
        return
    if action == "import":
        _handle_import(args)
        return
    if action == "explain":
        _handle_explain(args)
        return
    if action == "evaluate-learning":
        _handle_evaluate_learning(args)
        return
    if action == "citation-stats":
        _handle_citation_stats(args)
        return
    if action == "prune":
        files = _fact_files_for_prune(
            all_files=getattr(args, "all", False),
            cwd=getattr(args, "cwd", None),
        )
        limit = getattr(args, "limit", None)
        if limit is not None:
            files = files[:limit]

        totals = {"files": 0, "removed": 0, "stripped": 0, "errors": 0}
        for path in files:
            totals["files"] += 1
            stats = _compact_fact_file(
                path,
                max_invalid_age_days=getattr(args, "max_invalid_age_days", 30),
                dry_run=getattr(args, "dry_run", False),
            )
            if stats.get("status") == "error":
                totals["errors"] += 1
                print(f"[Neo] prune error {path}: {stats.get('error')}", file=sys.stderr)
                continue
            removed = int(stats.get("removed", 0))
            stripped = int(stats.get("stripped", 0))
            totals["removed"] += removed
            totals["stripped"] += stripped
            if removed or stripped or getattr(args, "verbose", False):
                mode = "would remove" if getattr(args, "dry_run", False) else "removed"
                detail = f"{mode} {removed} old invalid fact(s)"
                if stripped:
                    verb = "would strip" if getattr(args, "dry_run", False) else "stripped"
                    detail += f", {verb} {stripped} tombstone embedding(s)"
                print(f"[Neo] {detail} from {path.name}")

        mode = "dry run" if getattr(args, "dry_run", False) else "prune"
        print(
            f"[Neo] memory {mode}: {totals['removed']} old invalid fact(s) removed, "
            f"{totals['stripped']} tombstone embedding(s) stripped "
            f"across {totals['files']} file(s)"
        )
        if totals["errors"]:
            print(f"[Neo] {totals['errors']} file(s) failed", file=sys.stderr)
        return

    if action != "replay-feedback":
        print(
            "Usage: neo memory "
            "{replay-feedback|prune|observer|issues|rules|audit|import|explain|"
            "evaluate-learning} ..."
        )
        return

    from neo.config import NeoConfig
    from neo.memory.store import FactStore

    config = NeoConfig.load()
    if getattr(args, "all", False):
        projects = _linked_feedback_projects(
            include_fallback=getattr(args, "include_legacy_fallback", False)
        )
    else:
        root = getattr(args, "cwd", None) or os.getcwd()
        projects = [("", root)]

    limit = getattr(args, "limit", None)
    if limit is not None:
        projects = projects[:limit]

    totals = {
        "projects": 0,
        "updated_projects": 0,
        "linked_updates": 0,
        "accepted": 0,
        "modified": 0,
        "unverified": 0,
        "skipped_independent": 0,
        "skipped_unlinked": 0,
        "errors": 0,
    }

    for pid, root in projects:
        totals["projects"] += 1
        store = FactStore(codebase_root=root, config=config, eager_init=False)
        stats = store.replay_linked_feedback(
            dry_run=getattr(args, "dry_run", False),
            include_fallback=getattr(args, "include_legacy_fallback", False),
        )
        if stats.get("status") == "error":
            totals["errors"] += 1
            print(f"[Neo] replay error {root}: {stats.get('error')}")
            continue

        updates = int(stats.get("linked_updates", 0))
        if updates:
            totals["updated_projects"] += 1
        for key in ("linked_updates", "accepted", "modified", "unverified", "skipped_independent", "skipped_unlinked"):
            totals[key] += int(stats.get(key, 0))
        label = pid or getattr(store, "project_id", "")
        if updates or getattr(args, "verbose", False):
            mode = "would update" if getattr(args, "dry_run", False) else "updated"
            print(
                f"[Neo] {mode} {updates} linked outcome(s) for {label or root} "
                f"(accepted={stats.get('accepted', 0)}, modified={stats.get('modified', 0)}, "
                f"unverified={stats.get('unverified', 0)})"
            )

    mode = "dry run" if getattr(args, "dry_run", False) else "replay"
    print(
        f"[Neo] feedback {mode}: {totals['linked_updates']} linked update(s) "
        f"across {totals['updated_projects']}/{totals['projects']} project(s)"
    )
    if totals["errors"]:
        print(f"[Neo] {totals['errors']} project(s) failed", file=sys.stderr)


def show_help():
    """Show help documentation."""
    help_text = """
neo - AI-powered code reasoning assistant

USAGE:
    neo [OPTIONS]
    echo '<json>' | neo
    neo < input.json

OPTIONS:
    --help, -h       Show this help message
    --version, -v    Show Neo's current learning progress

INPUT FORMAT (via stdin):
    {
      "prompt": "string (REQUIRED)",
      "task_type": "algorithm|refactor|bugfix|feature|explanation (optional)",
      "context_files": [
        {
          "path": "string",
          "content": "string",
          "line_range": [start, end]  // optional
        }
      ],
      "error_trace": "string (optional)",
      "recent_commands": ["cmd1", "cmd2"],
      "safe_read_paths": ["*.py", "*.js"],
      "working_directory": "/path/to/project",
      "operating_mode": "advise|patch|verify|learn|agent",
      "proposed_changes": [
        {"file_path": "src/app.py", "unified_diff": "...", "code_block": "..."}
      ],
      "authority": {
        "workspace_root": "/path/to/project",
        "allowed_write_paths": ["src/**"],
        "allowed_commands": [],
        "allow_learning": false
      }
    }

ENVIRONMENT VARIABLES:
    ANTHROPIC_API_KEY    Anthropic API key
    OPENAI_API_KEY       OpenAI API key
    GOOGLE_API_KEY       Google API key
    NEO_PROVIDER         LLM provider (openai|anthropic|google|ollama)
    NEO_MODEL            Model name
    NEO_API_KEY          Generic API key override (takes precedence)

EXAMPLES:
    # Simple query
    echo '{"prompt": "Write a function to check if a number is prime"}' | neo

    # With context
    echo '{"prompt": "Fix this bug", "task_type": "bugfix", "context_files": [...]}' | neo

    # From file
    neo < input.json

    # Check learning progress
    neo --version

DOCUMENTATION:
    https://github.com/Parslee-ai/neo
"""
    print(help_text)

def _interpret_confidence(
    confidence: float,
    next_questions: list[str],
    plan: list,
    code_suggestions: list
) -> dict:
    """
    Interpret confidence score and provide actionable guidance.

    Helps users understand what the confidence score means and what to do next.
    """
    interpretation = {}

    # Determine action guidance based on confidence level
    if confidence >= 0.7:
        interpretation["action"] = "READY_TO_IMPLEMENT"
        interpretation["message"] = "High confidence - plan is well-structured and data-driven"
        interpretation["next_steps"] = [
            "Review the plan and code suggestions carefully",
            "Implement with standard monitoring and rollback procedures",
            "Consider the tradeoffs mentioned in code suggestions"
        ]
    elif confidence >= 0.4:
        interpretation["action"] = "PROCEED_WITH_CAUTION"
        interpretation["message"] = "Medium confidence - plan is sound but has some gaps or uncertainties"
        interpretation["next_steps"] = [
            "Review next_questions for areas needing clarification",
            "Consider gathering additional data before full implementation",
            "Implement incrementally with careful monitoring"
        ]
    else:  # confidence < 0.4
        interpretation["action"] = "GATHER_MORE_DATA"
        interpretation["message"] = "Low confidence - plan framework is provided but critical data is missing"
        interpretation["next_steps"] = []

        # Analyze next_questions to identify what's missing
        blocking_issues = []
        if next_questions:
            # Categorize the gaps
            has_missing_constraints = any("constraint" in q.lower() or "missing" in q.lower() for q in next_questions)
            has_missing_metrics = any("metric" in q.lower() or "quantify" in q.lower() for q in next_questions)
            has_missing_observability = any("observability" in q.lower() or "monitoring" in q.lower() for q in next_questions)

            if has_missing_constraints:
                blocking_issues.append("Missing or incomplete constraints")
            if has_missing_metrics:
                blocking_issues.append("Lacking quantitative metrics or baselines")
            if has_missing_observability:
                blocking_issues.append("No observability/monitoring strategy")

            # If we couldn't categorize, just note that there are gaps
            if not blocking_issues:
                blocking_issues.append("Data gaps identified in next_questions")

        interpretation["blocking_issues"] = blocking_issues if blocking_issues else ["Insufficient data to proceed with confidence"]

        # Provide specific next steps
        if plan:
            interpretation["next_steps"].append("Follow the plan to gather missing data and requirements")
        if next_questions:
            interpretation["next_steps"].append("Address the issues listed in next_questions")
        interpretation["next_steps"].append("Re-run Neo with complete data for higher confidence decision")

        # Important clarification
        interpretation["note"] = "The plan itself may be valuable - low confidence indicates missing input data, not plan quality"

    # Add confidence scale reference
    interpretation["confidence_scale"] = {
        "0.0-0.4": "Gather more data - critical information missing",
        "0.4-0.7": "Proceed with caution - some uncertainties remain",
        "0.7-1.0": "Ready to implement - high confidence in approach"
    }

    return interpretation

def _restore_from_backup(memory: 'PersistentReasoningMemory', backup: list) -> None:
    """Restore memory entries from backup (used on failure)."""
    memory.entries = backup


def _regenerate_entry_embeddings(
    memory: 'PersistentReasoningMemory',
    backup: list
) -> tuple[int, int, str]:
    """
    Regenerate embeddings for all entries in memory.

    Returns:
        Tuple of (success_count, failed_count, model_used)

    Raises:
        RuntimeError: If success rate < MIN_EMBEDDING_SUCCESS_RATE
    """
    total_entries = len(memory.entries)
    success_count = 0
    failed_count = 0
    model_used = "unknown"

    for i, entry in enumerate(memory.entries):
        # Build text from entry
        text = f"{entry.pattern}\n{entry.context}\n{entry.suggestion}"

        # Generate new embedding
        embedding = memory._embed_text(text)

        # Validate embedding
        if embedding is not None and len(embedding) in VALID_EMBEDDING_DIMENSIONS:
            entry.embedding = embedding
            entry.embedding_dim = len(embedding)

            # Extract model name from cache
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in memory.embedding_cache:
                _, cached_model, _ = memory.embedding_cache[cache_key]
                entry.embedding_model = cached_model
                if model_used == "unknown":
                    model_used = cached_model

            success_count += 1
        else:
            failed_count += 1
            if embedding is not None:
                logger.warning(
                    f"Entry {i} has invalid embedding dimension {len(embedding)}, "
                    f"expected one of {VALID_EMBEDDING_DIMENSIONS}"
                )
            else:
                logger.warning(f"Entry {i} failed to generate embedding: {entry.pattern[:50]}")

    # Validate success rate
    success_rate = success_count / total_entries if total_entries > 0 else 0.0

    if success_rate < MIN_EMBEDDING_SUCCESS_RATE:
        _restore_from_backup(memory, backup)
        raise RuntimeError(
            f"Embedding regeneration failed: only {success_count}/{total_entries} "
            f"({success_rate:.1%}) succeeded. Need at least {MIN_EMBEDDING_SUCCESS_RATE:.0%}. "
            f"Backup restored."
        )

    return success_count, failed_count, model_used


def regenerate_embeddings(
    codebase_root: Optional[str] = None,
    config: Optional['NeoConfig'] = None
) -> RegenerateStats:
    """
    Regenerate all embeddings with current configured model.

    Safe operation with automatic backup and failure detection.
    Use when switching embedding models or fixing mixed-model state.

    Args:
        codebase_root: Path to codebase (for local memory)
        config: NeoConfig instance (for embedding model selection)

    Returns:
        RegenerateStats dict with operation metrics

    Raises:
        RuntimeError: If backup fails, success rate < 80%, or save fails
    """
    start_time = time.time()

    # Initialize memory
    from neo.persistent_reasoning import PersistentReasoningMemory
    memory = PersistentReasoningMemory(codebase_root=codebase_root, config=config)

    total_entries = len(memory.entries)
    if total_entries == 0:
        logger.info("No entries to regenerate")
        return RegenerateStats(
            total=0,
            success=0,
            failed=0,
            success_rate=1.0,
            model="none",
            duration=0.0
        )

    logger.info(f"Regenerating embeddings for {total_entries} entries...")

    # Create backup (deep copy to prevent mutation)
    backup = copy.deepcopy(memory.entries)

    # Regenerate embeddings
    try:
        success_count, failed_count, model_used = _regenerate_entry_embeddings(memory, backup)
    except RuntimeError:
        # Backup already restored by _regenerate_entry_embeddings
        raise

    # Save updated entries
    try:
        memory.save()
    except (IOError, OSError, PermissionError) as e:
        _restore_from_backup(memory, backup)
        raise RuntimeError(f"Failed to save regenerated embeddings: {e}. Backup restored.") from e

    duration = time.time() - start_time
    success_rate = success_count / total_entries if total_entries > 0 else 0.0

    logger.info(
        f"Embedding regeneration complete: {success_count}/{total_entries} succeeded "
        f"in {duration:.1f}s using {model_used}"
    )

    return RegenerateStats(
        total=total_entries,
        success=success_count,
        failed=failed_count,
        success_rate=success_rate,
        model=model_used,
        duration=duration
    )


def handle_load_program(args):
    """Handle --load-program flag operations (The Operator)."""
    from neo.config import NeoConfig
    from neo.program_loader import ProgramLoader
    from neo.persistent_reasoning import PersistentReasoningMemory

    try:
        # Load config
        config = NeoConfig.load()
        codebase_root = args.cwd or os.getcwd()

        # Initialize memory
        memory = PersistentReasoningMemory(
            codebase_root=codebase_root,
            config=config
        )

        # Initialize loader
        loader = ProgramLoader(memory)

        # Parse column mapping if provided
        column_mapping = None
        if args.columns:
            try:
                column_mapping = json.loads(args.columns)
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON in --columns: {e}", file=sys.stderr)
                sys.exit(1)

        # Load program
        result = loader.load_program(
            dataset_id=args.load_program,
            split=args.split,
            column_mapping=column_mapping,
            limit=args.limit,
            dry_run=args.dry_run,
            quiet=args.quiet
        )

        # Print Matrix-style output
        print()
        print(loader.format_result(result))

    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Install with: pip install datasets", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        logger.exception("Failed to load program")
        print(f"Error: Unexpected failure: {e}", file=sys.stderr)
        sys.exit(1)


def handle_update(args):
    """Handle the 'neo update' command to perform self-update."""
    from neo.update_checker import perform_update

    success = perform_update()
    sys.exit(0 if success else 1)


def handle_serve(args) -> int:
    """Handle the 'neo serve' command — host Neo over A2A via CAR.

    Returns the process exit code (0 on clean shutdown, 1 on init
    failure). Blocks until SIGINT/SIGTERM.
    """
    try:
        from neo.car_host import run_server
    except ImportError as e:
        print(
            f"Failed to load CAR host: {e}\n"
            "The [car] extra is required:\n"
            "    pip install 'neo-reasoner[car]'",
            file=sys.stderr,
        )
        return 1

    return run_server(
        bind=args.a2a_bind,
        public_url=getattr(args, "public_url", None),
        agent_name=getattr(args, "agent_name", "neo"),
    )


def handle_car(args) -> int:
    """Handle `neo car` discovery commands."""
    action = getattr(args, "car_action", None)
    if action not in ("status", None):
        print("Usage: neo car status", file=sys.stderr)
        return 1

    from neo.car_discovery import discover_car

    info = discover_car()
    print(f"CAR: {info.summary()}")
    if info.cli_path:
        print(f"  cli: {info.cli_path}")
        if info.cli_version:
            print(f"  version: {info.cli_version}")
    else:
        print("  cli: not found")
    print(f"  server: {info.server_path or 'not found'}")
    print(f"  python binding: {info.python_binding or 'not found'}")
    print(f"  daemon: {'running' if info.daemon_running else 'not detected'} ({info.daemon_url})")
    if info.available and not info.has_python_runtime:
        print(
            "  note: CLI/server detected. Install neo-reasoner[car] or car-runtime "
            "in this Python environment to enable CAR-native `neo serve`."
        )
    return 0


def handle_construct(args):
    """Handle construct subcommand operations."""
    from neo.construct import ConstructIndex
    from pathlib import Path

    # Determine construct root
    cwd = Path(args.cwd) if hasattr(args, 'cwd') and args.cwd else Path.cwd()

    # Try to find construct directory in repo
    construct_root = None
    if (cwd / 'construct').exists():
        construct_root = cwd / 'construct'
    elif (cwd.parent / 'construct').exists():
        construct_root = cwd.parent / 'construct'
    else:
        # Check if we're in the neo repo
        current = cwd
        while current != current.parent:
            if (current / 'construct').exists():
                construct_root = current / 'construct'
                break
            current = current.parent

    index = ConstructIndex(construct_root=construct_root)

    if args.construct_action == 'list':
        patterns = index.list_patterns(domain=args.domain)
        if not patterns:
            print("No patterns found.")
            if args.domain:
                print(f"(Domain filter: {args.domain})")
            return

        # Group by domain
        by_domain = {}
        for p in patterns:
            by_domain.setdefault(p.domain, []).append(p)

        for domain in sorted(by_domain.keys()):
            print(f"\n{domain}:")
            for p in by_domain[domain]:
                print(f"  {p.pattern_id:<40} {p.name}")

        print(f"\nTotal: {len(patterns)} patterns")

    elif args.construct_action == 'show':
        pattern = index.show_pattern(args.pattern_id)
        if not pattern:
            print(f"Error: Pattern '{args.pattern_id}' not found", file=sys.stderr)
            sys.exit(1)

        # Display pattern
        print(f"# Pattern: {pattern.name}")
        print(f"Author: {pattern.author}")
        print(f"Domain: {pattern.domain}")
        print(f"ID: {pattern.pattern_id}\n")
        print(f"## Intent\n{pattern.intent}\n")
        print(f"## Forces\n{pattern.forces}\n")
        print(f"## Solution\n{pattern.solution}\n")
        print(f"## Consequences\n{pattern.consequences}\n")
        if pattern.references:
            print(f"## References\n{pattern.references}\n")

    elif args.construct_action == 'search':
        results = index.search(args.query, top_k=args.top_k)
        if not results:
            print(f"No results found for: {args.query}")
            return

        print(f"Search results for: {args.query}\n")
        for i, (pattern, score) in enumerate(results, 1):
            print(f"{i}. {pattern.pattern_id} (score: {score:.3f})")
            print(f"   {pattern.name}")
            print(f"   {pattern.intent[:100]}...")
            print()

    elif args.construct_action == 'index':
        print("Building construct pattern index...")
        result = index.build_index(force_rebuild=args.force)

        if result['status'] == 'success':
            print(f"\u2713 Indexed {result['pattern_count']} patterns in {result['elapsed_seconds']:.2f}s")
            print(f"  Index: {result['index_path']}")
        elif result['status'] == 'skipped':
            print("Index is recent, skipping rebuild (use --force to rebuild)")
        else:
            print(f"\u2717 Index build failed: {result.get('reason', 'unknown error')}", file=sys.stderr)
            sys.exit(1)

    else:
        print("Error: No construct action specified", file=sys.stderr)
        print("Usage: neo construct {list|show|search|index}", file=sys.stderr)
        sys.exit(1)


def handle_prompt(args):
    """Handle prompt subcommand operations."""
    from neo.prompt import PromptSystem

    system = PromptSystem()

    if args.prompt_action == 'analyze':
        results = system.analyze(project=args.project, since=args.since)
        _print_prompt_analysis(results)

    elif args.prompt_action == 'enhance':
        # Read prompt from argument or stdin
        prompt = getattr(args, 'prompt_text', None)
        if not prompt:
            if not sys.stdin.isatty():
                try:
                    prompt = sys.stdin.read().strip()
                except EOFError:
                    print("Error: No input received (Ctrl+D pressed)", file=sys.stderr)
                    sys.exit(1)
                except KeyboardInterrupt:
                    print("\nOperation cancelled", file=sys.stderr)
                    sys.exit(130)
            else:
                print("Error: No prompt provided. Use: neo prompt enhance \"your prompt\" or pipe via stdin", file=sys.stderr)
                sys.exit(1)

        enhancement = system.enhance(prompt)

        if getattr(args, 'auto', False):
            print(enhancement.enhanced)
        else:
            _print_prompt_enhancement(enhancement)

    elif args.prompt_action == 'patterns':
        patterns = system.get_patterns(
            search=getattr(args, 'search', None),
            limit=getattr(args, 'limit', 10)
        )
        _print_prompt_patterns(patterns)

    elif args.prompt_action == 'suggest':
        suggestions = system.suggest_improvements(project=args.project)
        _print_prompt_suggestions(suggestions)

    elif args.prompt_action == 'history':
        evolutions = system.get_evolution_history(path=getattr(args, 'path', None))
        _print_prompt_evolutions(evolutions)

    elif args.prompt_action == 'stats':
        stats = system.get_stats()
        _print_prompt_stats(stats)

    else:
        print("Error: No prompt action specified", file=sys.stderr)
        print("Usage: neo prompt {analyze|enhance|patterns|suggest|history|stats}", file=sys.stderr)
        sys.exit(1)


def _print_prompt_analysis(results: dict) -> None:
    """Format and print prompt analysis results."""
    print("\n" + "=" * 60)
    print("PROMPT EFFECTIVENESS ANALYSIS")
    print("=" * 60)
    print(f"\nSessions analyzed: {results['total_sessions']}")
    print(f"Prompts scored: {results['total_prompts']}")
    print(f"Average effectiveness: {results['avg_effectiveness']:.2f}")

    avg = results["avg_effectiveness"]
    if avg >= 0.7:
        rating = "Excellent - prompts are highly effective"
    elif avg >= 0.4:
        rating = "Good - most prompts work well"
    elif avg >= 0.0:
        rating = "Fair - room for improvement"
    else:
        rating = "Needs work - many prompts cause issues"
    print(f"Rating: {rating}")

    if results.get("common_issues"):
        print("\nCommon Issues:")
        for issue in results["common_issues"]:
            signal = issue.get("signal", "").replace("_", " ").title()
            count = issue.get("count", 0)
            print(f"  - {signal}: {count} occurrences")

    if results.get("recommendations"):
        print("\nRecommendations:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. {rec}")
    print()


def _print_prompt_enhancement(enhancement) -> None:
    """Format and print prompt enhancement results."""
    print("\n" + "=" * 60)
    print("PROMPT ENHANCEMENT")
    print("=" * 60)
    print(f"\nOriginal:\n  {enhancement.original}")
    print(f"\nEnhanced:\n  {enhancement.enhanced}")

    if enhancement.improvements:
        print("\nImprovements made:")
        for imp in enhancement.improvements:
            print(f"  - {imp.replace('_', ' ').title()}")

    print(f"\nExpected benefit: {enhancement.expected_benefit}")
    print(f"Confidence: {enhancement.confidence:.0%}")
    print()


def _print_prompt_patterns(patterns: list) -> None:
    """Format and print effective patterns."""
    if not patterns:
        print("No patterns found. Patterns are extracted from effective prompts over time.")
        return

    print("\n" + "=" * 60)
    print("EFFECTIVE PROMPT PATTERNS")
    print("=" * 60)

    for i, pattern in enumerate(patterns, 1):
        name = pattern.name if hasattr(pattern, 'name') else pattern.get('name', 'Unknown')
        score = pattern.effectiveness_score if hasattr(pattern, 'effectiveness_score') else pattern.get('effectiveness_score', 0.0)
        template = pattern.template if hasattr(pattern, 'template') else pattern.get('template', '')

        print(f"\n{i}. {name} (score: {score:.2f})")
        if template:
            print(f"   Template: {template}")

    print(f"\nTotal: {len(patterns)} patterns")
    print()


def _print_prompt_suggestions(suggestions: list) -> None:
    """Format and print improvement suggestions."""
    if not suggestions:
        print("No improvement suggestions at this time.")
        return

    print("\n" + "=" * 60)
    print("CLAUDE.MD IMPROVEMENT SUGGESTIONS")
    print("=" * 60)

    for i, sugg in enumerate(suggestions, 1):
        stype = sugg.get("type", "unknown").replace("_", " ").title()
        suggestion = sugg.get("suggestion", "")
        reason = sugg.get("reason", "")
        confidence = sugg.get("confidence", 0.0)

        conf_label = "HIGH" if confidence >= 0.8 else "MEDIUM" if confidence >= 0.5 else "LOW"
        print(f"\n{i}. [{conf_label}] {stype}")
        print(f"   Suggestion: {suggestion}")
        if reason:
            print(f"   Reason: {reason}")

    print(f"\nTotal: {len(suggestions)} suggestions")
    print()


def _print_prompt_evolutions(evolutions: list) -> None:
    """Format and print CLAUDE.md evolution history."""
    if not evolutions:
        print("No evolution history found.")
        return

    print("\n" + "=" * 60)
    print("CLAUDE.MD EVOLUTION HISTORY")
    print("=" * 60)

    for i, evo in enumerate(evolutions, 1):
        path = str(evo.path) if hasattr(evo, 'path') else evo.get('path', '')
        timestamp = evo.timestamp if hasattr(evo, 'timestamp') else evo.get('timestamp', '')
        change_type = evo.change_type if hasattr(evo, 'change_type') else evo.get('change_type', '')
        reason = evo.inferred_reason if hasattr(evo, 'inferred_reason') else evo.get('inferred_reason', '')

        ts_str = timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(timestamp, 'strftime') else str(timestamp)[:16]

        print(f"\n{i}. {path}")
        print(f"   Time: {ts_str}")
        print(f"   Type: {change_type}")
        if reason:
            print(f"   Reason: {reason}")

    print(f"\nTotal: {len(evolutions)} changes recorded")
    print()


def _print_prompt_stats(stats: dict) -> None:
    """Format and print knowledge base statistics."""
    print("\n" + "=" * 60)
    print("PROMPT KNOWLEDGE BASE STATISTICS")
    print("=" * 60)
    print(f"\nTotal entries: {stats.get('total_entries', 0):,}")
    print(f"Patterns: {stats.get('patterns', 0):,}")
    print(f"Effectiveness scores: {stats.get('scores', 0):,}")
    print(f"Evolution records: {stats.get('evolutions', 0):,}")
    print(f"Pending suggestions: {stats.get('pending_suggestions', 0):,}")
    print(f"Projects tracked: {stats.get('projects_tracked', 0):,}")

    components = stats.get("components_available", {})
    if components:
        print("\nComponent Status:")
        for name, available in components.items():
            status = "OK" if available else "NOT AVAILABLE"
            print(f"  - {name.replace('_', ' ').title()}: {status}")
    print()


def handle_config(args):
    """Handle --config flag operations."""
    from neo.config import NeoConfig, store_api_key_in_keychain

    VALID_PROVIDERS = ['openai', 'anthropic', 'google', 'azure', 'ollama', 'local', 'claude-code']
    VALID_MEMORY_BACKENDS = ['fact_store', 'legacy']
    EXPOSED_FIELDS = [
        'provider', 'model', 'api_key', 'base_url',
        'memory_backend', 'auto_install_updates', 'constraint_auto_scan',
        'log_level', 'reasoning_effort_cap',
    ]

    def mask_secret(value: str) -> str:
        """Mask API keys and secrets for display."""
        if not value or len(value) < 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"

    # Load current config
    config = NeoConfig.load()

    if args.config == 'list':
        # Show all exposed fields
        print("Current configuration:")
        for field in EXPOSED_FIELDS:
            value = getattr(config, field, None)
            if value is None:
                display_value = "(not set)"
            elif field == 'api_key':
                display_value = mask_secret(value)
            else:
                display_value = value
            print(f"  {field}: {display_value}")

    elif args.config == 'get':
        # Get single field
        if not args.config_key:
            print("Error: --config-key required for 'get' operation", file=sys.stderr)
            sys.exit(1)

        if args.config_key not in EXPOSED_FIELDS:
            print(f"Error: Invalid config key. Valid keys: {', '.join(EXPOSED_FIELDS)}", file=sys.stderr)
            sys.exit(1)

        value = getattr(config, args.config_key, None)
        if value is None:
            print("(not set)")
        elif args.config_key == 'api_key':
            print(mask_secret(value))
        else:
            print(value)

    elif args.config == 'set':
        # Set field value
        if not args.config_key:
            print("Error: --config-key required for 'set' operation", file=sys.stderr)
            sys.exit(1)

        if args.config_key not in EXPOSED_FIELDS:
            print(f"Error: Invalid config key. Valid keys: {', '.join(EXPOSED_FIELDS)}", file=sys.stderr)
            sys.exit(1)

        # Validate provider
        if args.config_key == 'provider' and args.config_value not in VALID_PROVIDERS:
            print(f"Error: Invalid provider. Valid providers: {', '.join(VALID_PROVIDERS)}", file=sys.stderr)
            sys.exit(1)
        if args.config_key == 'memory_backend' and args.config_value not in VALID_MEMORY_BACKENDS:
            print(f"Error: Invalid memory backend. Valid values: {', '.join(VALID_MEMORY_BACKENDS)}", file=sys.stderr)
            sys.exit(1)
        if args.config_key == 'log_level':
            value_upper = str(args.config_value).upper() if args.config_value else ""
            if value_upper not in ("DEBUG", "INFO", "WARNING", "ERROR"):
                print("Error: Invalid log level. Use: DEBUG, INFO, WARNING, ERROR", file=sys.stderr)
                sys.exit(1)

        if args.config_key == 'api_key' and not args.config_value:
            import getpass
            value = getpass.getpass(f"API key for {config.provider}: ")
            if not value:
                print("Error: API key cannot be empty", file=sys.stderr)
                sys.exit(1)
        elif not args.config_value:
            print("Error: --config-value required for this config key", file=sys.stderr)
            sys.exit(1)
        else:
            value = args.config_value

        # Convert boolean values
        if args.config_key in ('auto_install_updates', 'constraint_auto_scan'):
            if str(value).lower() in ('true', '1', 'yes', 'on'):
                value = True
            elif str(value).lower() in ('false', '0', 'no', 'off'):
                value = False
            else:
                print("Error: Invalid boolean value. Use: true/false, 1/0, yes/no, on/off", file=sys.stderr)
                sys.exit(1)
        elif args.config_key == 'reasoning_effort_cap':
            from neo.reasoning_effort import validate_effort
            try:
                value = validate_effort(None if str(value).lower() in ('none', 'null', '') else str(value))
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
        elif args.config_key == 'log_level':
            value = str(value).upper()

        # Set the value
        if args.config_key == 'api_key':
            if os.environ.get("NEO_ALLOW_PLAINTEXT_API_KEY"):
                config.api_key = str(value)
                config.save()
                print("\u2713 Stored api_key in config.json (plaintext enabled)")
                return
            else:
                try:
                    store_api_key_in_keychain(config.provider, str(value))
                except Exception as exc:
                    print(f"Error: {exc}", file=sys.stderr)
                    sys.exit(1)
                value = None
        else:
            setattr(config, args.config_key, value)
        config.save()
        if args.config_key == 'api_key':
            print(f"\u2713 Stored api_key in Keychain for provider {config.provider}")
        else:
            print(f"\u2713 Set {args.config_key} = {value}")

    elif args.config == 'reset':
        # Reset to defaults
        default_config = NeoConfig()
        default_config.save()
        print("\u2713 Configuration reset to defaults")
