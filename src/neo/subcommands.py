"""
Neo CLI subcommand handlers.

Contains all handler functions for CLI subcommands (version, help, config,
construct, prompt, etc.) and their supporting utilities.

Split from cli.py for modularity.
"""

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
    print(f"Stage: {stage} | Memory: {level:.1%}")
    print(f"{bar}")
    print(f"{total_entries} patterns | {avg_confidence:.2f} avg confidence")

    # Check for contributable facts
    if memory_backend == "fact_store" and hasattr(memory, 'find_contributable'):
        contributable = memory.find_contributable()
        if contributable:
            print(f"\n\u2728 {len(contributable)} pattern(s) ready to share with the community")
            print("   Run: neo contribute")
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
      "working_directory": "/path/to/project"
    }

ENVIRONMENT VARIABLES:
    ANTHROPIC_API_KEY    Anthropic API key
    OPENAI_API_KEY       OpenAI API key
    GOOGLE_API_KEY       Google API key
    NEO_PROVIDER         LLM provider (openai|anthropic|google|ollama)
    NEO_MODEL            Model name
    NEO_API_KEY          Generic API key (provider-specific keys take precedence)

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
    from neo.config import NeoConfig

    VALID_PROVIDERS = ['openai', 'anthropic', 'google', 'azure', 'ollama', 'local', 'claude-code']
    EXPOSED_FIELDS = ['provider', 'model', 'api_key', 'base_url', 'auto_install_updates']

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
        if not args.config_key or not args.config_value:
            print("Error: --config-key and --config-value required for 'set' operation", file=sys.stderr)
            sys.exit(1)

        if args.config_key not in EXPOSED_FIELDS:
            print(f"Error: Invalid config key. Valid keys: {', '.join(EXPOSED_FIELDS)}", file=sys.stderr)
            sys.exit(1)

        # Validate provider
        if args.config_key == 'provider' and args.config_value not in VALID_PROVIDERS:
            print(f"Error: Invalid provider. Valid providers: {', '.join(VALID_PROVIDERS)}", file=sys.stderr)
            sys.exit(1)

        # Convert boolean values
        if args.config_key == 'auto_install_updates':
            if args.config_value.lower() in ('true', '1', 'yes', 'on'):
                value = True
            elif args.config_value.lower() in ('false', '0', 'no', 'off'):
                value = False
            else:
                print("Error: Invalid boolean value. Use: true/false, 1/0, yes/no, on/off", file=sys.stderr)
                sys.exit(1)
        else:
            value = args.config_value

        # Set the value
        setattr(config, args.config_key, value)
        config.save()
        print(f"\u2713 Set {args.config_key} = {value if args.config_key != 'api_key' else mask_secret(value)}")

    elif args.config == 'reset':
        # Reset to defaults
        default_config = NeoConfig()
        default_config.save()
        print("\u2713 Configuration reset to defaults")
