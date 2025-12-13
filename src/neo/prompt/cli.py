"""
CLI command handlers for the Prompt Enhancement System.

Provides the `neo prompt` subcommand with subcommands:
- analyze: Analyze prompt effectiveness
- enhance: Enhance a prompt
- patterns: Show effective patterns
- suggest: Suggest CLAUDE.md improvements
- history: Show CLAUDE.md evolution history
- stats: Show prompt knowledge base stats
"""

import argparse
import sys
from typing import Optional


def register_prompt_commands(subparsers: argparse._SubParsersAction) -> None:
    """
    Register the 'prompt' subcommand with all its subcommands.

    Args:
        subparsers: The subparsers action from the main parser
    """
    # Create the 'prompt' subparser
    prompt_parser = subparsers.add_parser(
        "prompt",
        help="Prompt enhancement and analysis tools",
        description="Tools for analyzing prompt effectiveness, enhancing prompts, and suggesting improvements",
    )

    # Add subcommands to 'prompt'
    prompt_subparsers = prompt_parser.add_subparsers(
        dest="prompt_command",
        help="Prompt enhancement commands",
    )

    # neo prompt analyze
    analyze_parser = prompt_subparsers.add_parser(
        "analyze",
        help="Analyze prompt effectiveness",
        description="Analyze prompt effectiveness for a project based on session history",
    )
    analyze_parser.add_argument(
        "--project",
        metavar="PATH",
        help="Specific project path to analyze",
    )
    analyze_parser.add_argument(
        "--since",
        metavar="DATE",
        help="Analyze sessions since date (ISO format: YYYY-MM-DD)",
    )

    # neo prompt enhance <prompt>
    enhance_parser = prompt_subparsers.add_parser(
        "enhance",
        help="Enhance a prompt",
        description="Enhance a prompt using learned patterns and rules",
    )
    enhance_parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to enhance (or read from stdin)",
    )
    enhance_parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-apply enhancement (output only the enhanced prompt)",
    )

    # neo prompt patterns
    patterns_parser = prompt_subparsers.add_parser(
        "patterns",
        help="Show effective patterns",
        description="List effective prompt patterns from the knowledge base",
    )
    patterns_parser.add_argument(
        "--search",
        metavar="QUERY",
        help="Search for specific patterns",
    )
    patterns_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of patterns to show (default: 10)",
    )

    # neo prompt suggest
    suggest_parser = prompt_subparsers.add_parser(
        "suggest",
        help="Suggest CLAUDE.md improvements",
        description="Suggest improvements to CLAUDE.md based on usage patterns",
    )
    suggest_parser.add_argument(
        "--project",
        metavar="PATH",
        help="Project to analyze",
    )
    suggest_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply suggestions interactively",
    )

    # neo prompt history
    history_parser = prompt_subparsers.add_parser(
        "history",
        help="Show CLAUDE.md evolution history",
        description="Show the evolution history of CLAUDE.md files",
    )
    history_parser.add_argument(
        "--path",
        metavar="FILE",
        help="Specific file to show history for",
    )

    # neo prompt stats
    prompt_subparsers.add_parser(
        "stats",
        help="Show prompt knowledge base stats",
        description="Display statistics about the prompt knowledge base",
    )


def handle_prompt_command(args: argparse.Namespace) -> None:
    """
    Handle prompt subcommands by dispatching to appropriate PromptSystem method.

    Args:
        args: Parsed command line arguments
    """
    from neo.prompt import PromptSystem

    system = PromptSystem()

    if args.prompt_command == "analyze":
        results = system.analyze(project=args.project, since=args.since)
        _print_analysis(results)

    elif args.prompt_command == "enhance":
        # Read prompt from argument or stdin
        prompt = args.prompt
        if not prompt:
            if not sys.stdin.isatty():
                prompt = sys.stdin.read().strip()
            else:
                print("Error: No prompt provided. Use: neo prompt enhance \"your prompt\" or pipe via stdin", file=sys.stderr)
                sys.exit(1)

        enhancement = system.enhance(prompt)

        if args.auto:
            # Just output the enhanced prompt
            print(enhancement.enhanced)
        else:
            _print_enhancement(enhancement)

    elif args.prompt_command == "patterns":
        patterns = system.get_patterns(search=args.search, limit=args.limit)
        _print_patterns(patterns)

    elif args.prompt_command == "suggest":
        suggestions = system.suggest_improvements(project=args.project)
        if args.apply:
            _apply_suggestions_interactive(suggestions)
        else:
            _print_suggestions(suggestions)

    elif args.prompt_command == "history":
        evolutions = system.get_evolution_history(path=args.path)
        _print_evolutions(evolutions)

    elif args.prompt_command == "stats":
        stats = system.get_stats()
        _print_stats(stats)

    else:
        # No subcommand provided
        print("Error: No prompt subcommand specified", file=sys.stderr)
        print("Usage: neo prompt {analyze|enhance|patterns|suggest|history|stats}", file=sys.stderr)
        sys.exit(1)


# =============================================================================
# Output Formatting Functions
# =============================================================================


def _print_analysis(results: dict) -> None:
    """Format and print analysis results."""
    print("\n" + "=" * 60)
    print("PROMPT EFFECTIVENESS ANALYSIS")
    print("=" * 60)

    print(f"\nSessions analyzed: {results['total_sessions']}")
    print(f"Prompts scored: {results['total_prompts']}")
    print(f"Average effectiveness: {results['avg_effectiveness']:.2f}")

    # Effectiveness interpretation
    avg = results["avg_effectiveness"]
    if avg >= 0.7:
        interpretation = "Excellent - prompts are highly effective"
    elif avg >= 0.4:
        interpretation = "Good - most prompts work well"
    elif avg >= 0.0:
        interpretation = "Fair - room for improvement"
    else:
        interpretation = "Needs work - many prompts cause issues"
    print(f"Rating: {interpretation}")

    if results["top_patterns"]:
        print("\nTop Effective Patterns:")
        for i, pattern in enumerate(results["top_patterns"], 1):
            name = pattern.get("name", "Unknown")
            score = pattern.get("score", 0.0)
            print(f"  {i}. {name} (score: {score:.2f})")

    if results["common_issues"]:
        print("\nCommon Issues:")
        for issue in results["common_issues"]:
            signal = issue.get("signal", "")
            count = issue.get("count", 0)
            # Make signal name more readable
            readable = signal.replace("_", " ").title()
            print(f"  - {readable}: {count} occurrences")

    if results["recommendations"]:
        print("\nRecommendations:")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"  {i}. {rec}")

    print()


def _print_enhancement(enhancement) -> None:
    """Format and print prompt enhancement results."""
    print("\n" + "=" * 60)
    print("PROMPT ENHANCEMENT")
    print("=" * 60)

    print(f"\nOriginal:\n  {enhancement.original}")
    print(f"\nEnhanced:\n  {enhancement.enhanced}")

    if enhancement.improvements:
        print("\nImprovements made:")
        for imp in enhancement.improvements:
            # Make improvement name more readable
            readable = imp.replace("_", " ").replace("-", " ").title()
            print(f"  - {readable}")

    print(f"\nExpected benefit: {enhancement.expected_benefit}")
    print(f"Confidence: {enhancement.confidence:.0%}")
    print()


def _print_patterns(patterns: list) -> None:
    """Format and print effective patterns."""
    if not patterns:
        print("No patterns found in knowledge base.")
        print("Patterns are extracted automatically from effective prompts over time.")
        return

    print("\n" + "=" * 60)
    print("EFFECTIVE PROMPT PATTERNS")
    print("=" * 60)

    for i, pattern in enumerate(patterns, 1):
        # Handle both object and dict patterns
        if hasattr(pattern, "name"):
            name = pattern.name
            description = pattern.description
            template = pattern.template
            score = pattern.effectiveness_score
            examples = pattern.examples
            use_cases = pattern.use_cases
            anti_patterns = pattern.anti_patterns
        else:
            name = pattern.get("name", "Unknown")
            description = pattern.get("description", "")
            template = pattern.get("template", "")
            score = pattern.get("effectiveness_score", 0.0)
            examples = pattern.get("examples", [])
            use_cases = pattern.get("use_cases", [])
            anti_patterns = pattern.get("anti_patterns", [])

        print(f"\n{i}. {name} (score: {score:.2f})")
        if description:
            print(f"   Description: {description}")
        if template:
            print(f"   Template: {template}")
        if examples:
            print("   Examples:")
            for ex in examples[:3]:  # Limit to 3 examples
                print(f"     - {ex}")
        if use_cases:
            print(f"   Use cases: {', '.join(use_cases[:3])}")
        if anti_patterns:
            print("   Avoid:")
            for ap in anti_patterns[:2]:  # Limit to 2 anti-patterns
                print(f"     - {ap}")

    print(f"\nTotal: {len(patterns)} patterns")
    print()


def _print_suggestions(suggestions: list) -> None:
    """Format and print improvement suggestions."""
    if not suggestions:
        print("No improvement suggestions at this time.")
        print("Suggestions are generated based on repeated patterns in your sessions.")
        return

    print("\n" + "=" * 60)
    print("CLAUDE.MD IMPROVEMENT SUGGESTIONS")
    print("=" * 60)

    for i, sugg in enumerate(suggestions, 1):
        stype = sugg.get("type", "unknown")
        target = sugg.get("target", "")
        suggestion = sugg.get("suggestion", "")
        reason = sugg.get("reason", "")
        confidence = sugg.get("confidence", 0.0)

        # Confidence label
        if confidence >= 0.8:
            conf_label = "HIGH"
        elif confidence >= 0.5:
            conf_label = "MEDIUM"
        else:
            conf_label = "LOW"

        print(f"\n{i}. [{conf_label}] {stype.replace('_', ' ').title()}")
        if target:
            print(f"   Target: {target}")
        print(f"   Suggestion: {suggestion}")
        if reason:
            print(f"   Reason: {reason}")

    print(f"\nTotal: {len(suggestions)} suggestions")
    print()


def _apply_suggestions_interactive(suggestions: list) -> None:
    """Interactively apply suggestions to CLAUDE.md files."""
    if not suggestions:
        print("No suggestions to apply.")
        return

    print("\nInteractive suggestion application is not yet implemented.")
    print("For now, please review the suggestions and apply them manually:\n")
    _print_suggestions(suggestions)


def _print_evolutions(evolutions: list) -> None:
    """Format and print CLAUDE.md evolution history."""
    if not evolutions:
        print("No evolution history found.")
        print("Evolution history is recorded when CLAUDE.md files change.")
        return

    print("\n" + "=" * 60)
    print("CLAUDE.MD EVOLUTION HISTORY")
    print("=" * 60)

    for i, evo in enumerate(evolutions, 1):
        # Handle both object and dict
        if hasattr(evo, "path"):
            path = str(evo.path)
            timestamp = evo.timestamp
            change_type = evo.change_type
            diff = evo.diff
            reason = evo.inferred_reason
        else:
            path = evo.get("path", "")
            timestamp = evo.get("timestamp", "")
            change_type = evo.get("change_type", "")
            diff = evo.get("diff", "")
            reason = evo.get("inferred_reason", "")

        # Format timestamp
        if hasattr(timestamp, "strftime"):
            ts_str = timestamp.strftime("%Y-%m-%d %H:%M")
        else:
            ts_str = str(timestamp)[:16] if timestamp else "Unknown"

        print(f"\n{i}. {path}")
        print(f"   Time: {ts_str}")
        print(f"   Type: {change_type}")
        if reason:
            print(f"   Reason: {reason}")
        if diff:
            # Show abbreviated diff
            diff_lines = diff.split("\n")
            if len(diff_lines) > 10:
                print("   Diff (truncated):")
                for line in diff_lines[:10]:
                    print(f"     {line}")
                print(f"     ... ({len(diff_lines) - 10} more lines)")
            else:
                print("   Diff:")
                for line in diff_lines:
                    print(f"     {line}")

    print(f"\nTotal: {len(evolutions)} changes recorded")
    print()


def _print_stats(stats: dict) -> None:
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
            # Make name readable
            readable_name = name.replace("_", " ").title()
            print(f"  - {readable_name}: {status}")

    print()
