#!/usr/bin/env python3
"""
Neo - A read-only reasoning helper for interactive CLI tools.

Receives context via stdin, performs MapCoder/CodeSim-style reasoning,
and returns structured output via stdout. No writes, single-call architecture.

This module serves as the thin CLI entry point. Core logic has been split into:
- neo.models: Data classes and LMAdapter ABC
- neo.engine: NeoEngine class
- neo.subcommands: CLI handler functions
"""

import json
import logging
import os
import sys

# Disable tokenizer parallelism warning (fastembed uses HuggingFace tokenizers)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Load environment variables from .env file
try:
    from neo.load_env import load_env
    load_env()
except ImportError:
    pass  # load_env.py not available, skip

# Initialize logger
logger = logging.getLogger(__name__)

# Backward-compat re-exports (moved to neo.models, neo.engine, neo.subcommands)
from neo.models import (  # noqa: E402, F401
    TaskType, ContextFile, NeoInput, PlanStep, SimulationTrace,
    CodeSuggestion, StaticCheckResult, NeoOutput, RegenerateStats, LMAdapter,
)
from neo.engine import NeoEngine  # noqa: E402, F401
from neo.subcommands import (  # noqa: E402, F401
    show_version, show_help, _interpret_confidence, _restore_from_backup,
    _regenerate_entry_embeddings, regenerate_embeddings, handle_load_program,
    handle_update, handle_construct, handle_prompt, handle_config,
    _print_prompt_analysis, _print_prompt_enhancement, _print_prompt_patterns,
    _print_prompt_suggestions, _print_prompt_evolutions, _print_prompt_stats,
)

# Re-export constants that were at module level
from neo.subcommands import MIN_EMBEDDING_SUCCESS_RATE, VALID_EMBEDDING_DIMENSIONS  # noqa: E402, F401


def parse_args():
    """Parse command-line arguments."""
    import argparse
    import sys

    # Create parent parser for global flags (shared across all parsers)
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument('--version', '-v', action='store_true', help='Show version and learning progress')
    global_parser.add_argument('--config', choices=['list', 'get', 'set', 'reset'], help='Manage configuration')
    global_parser.add_argument('--config-key', help='Config key (for get/set)')
    global_parser.add_argument('--config-value', help='Value (for set)')
    global_parser.add_argument('--load-program', metavar='DATASET_ID', help='Load training pack from HuggingFace (e.g., mbpp)')
    global_parser.add_argument('--regenerate-embeddings', action='store_true', help='Regenerate all embeddings with current model (safe, with automatic backup)')
    global_parser.add_argument('--index', action='store_true', help='Build semantic index for current directory')
    global_parser.add_argument('--languages', metavar='CSV', help='Languages to index (e.g., python,csharp,typescript)')
    global_parser.add_argument('--cwd', metavar='PATH', help='Working directory override')
    global_parser.add_argument('--verbose', action='store_true', help='Enable verbose logging (INFO level) to stderr')
    global_parser.add_argument('--debug', action='store_true', help='Enable debug logging (DEBUG level) to stderr')

    # Detect if 'contribute' subcommand is being used
    if len(sys.argv) > 1 and sys.argv[1] == 'contribute':
        p = argparse.ArgumentParser(
            prog="neo contribute",
            description="Export high-quality patterns for community contribution",
            parents=[global_parser]
        )
        args = p.parse_args(sys.argv[2:])
        args.command = 'contribute'
        return args

    # Detect if 'update' subcommand is being used
    if len(sys.argv) > 1 and sys.argv[1] == 'update':
        p = argparse.ArgumentParser(
            prog="neo update",
            description="Update neo to the latest version",
            parents=[global_parser]
        )
        args = p.parse_args(sys.argv[2:])  # Parse remaining args after 'update'
        args.command = 'update'
        return args

    # Detect if 'construct' subcommand is being used
    if len(sys.argv) > 1 and sys.argv[1] == 'construct':
        # Parse construct subcommand with proper sub-subparsers
        p = argparse.ArgumentParser(
            prog="neo construct",
            description="Manage design pattern library",
            parents=[global_parser]
        )

        subparsers = p.add_subparsers(dest='action', help='Construct actions')

        # construct list
        list_p = subparsers.add_parser('list', help='List all patterns')
        list_p.add_argument('--domain', help='Filter by domain')

        # construct show
        show_p = subparsers.add_parser('show', help='Show a pattern')
        show_p.add_argument('pattern_id', help='Pattern ID (e.g., caching/cache-aside)')

        # construct search
        search_p = subparsers.add_parser('search', help='Search patterns')
        search_p.add_argument('query', help='Search query')
        search_p.add_argument('--top-k', type=int, default=5, help='Number of results')

        # construct index
        index_p = subparsers.add_parser('index', help='Build search index')
        index_p.add_argument('--force', action='store_true', help='Force rebuild')

        # Parse remaining args (skip 'neo construct' prefix)
        args = p.parse_args(sys.argv[2:])
        args.command = 'construct'
        args.construct_action = args.action
        return args

    # Detect if 'prompt' subcommand is being used
    if len(sys.argv) > 1 and sys.argv[1] == 'prompt':
        # Parse prompt subcommand with proper sub-subparsers
        p = argparse.ArgumentParser(
            prog="neo prompt",
            description="Prompt enhancement and analysis tools",
            parents=[global_parser]
        )

        subparsers = p.add_subparsers(dest='action', help='Prompt actions')

        # prompt analyze
        analyze_p = subparsers.add_parser('analyze', help='Analyze prompt effectiveness')
        analyze_p.add_argument('--project', metavar='PATH', help='Project path to analyze')
        analyze_p.add_argument('--since', metavar='DATE', help='Analyze since date (ISO format)')

        # prompt enhance
        enhance_p = subparsers.add_parser('enhance', help='Enhance a prompt')
        enhance_p.add_argument('prompt_text', nargs='?', help='Prompt to enhance (or stdin)')
        enhance_p.add_argument('--auto', action='store_true', help='Output only enhanced prompt')

        # prompt patterns
        patterns_p = subparsers.add_parser('patterns', help='Show effective patterns')
        patterns_p.add_argument('--search', metavar='QUERY', help='Search query')
        patterns_p.add_argument('--limit', type=int, default=10, help='Max patterns to show')

        # prompt suggest
        suggest_p = subparsers.add_parser('suggest', help='Suggest CLAUDE.md improvements')
        suggest_p.add_argument('--project', metavar='PATH', help='Project to analyze')
        suggest_p.add_argument('--apply', action='store_true', help='Apply interactively')

        # prompt history
        history_p = subparsers.add_parser('history', help='Show CLAUDE.md evolution history')
        history_p.add_argument('--path', metavar='FILE', help='Specific file path')

        # prompt stats
        subparsers.add_parser('stats', help='Show knowledge base stats')

        # Parse remaining args (skip 'neo prompt' prefix)
        args = p.parse_args(sys.argv[2:])
        args.command = 'prompt'
        args.prompt_action = args.action
        return args

    # Default argument parser (for reasoning mode)
    p = argparse.ArgumentParser(
        prog="neo",
        description="Neo - Reasoning helper for coding tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[global_parser]
    )

    p.add_argument("prompt", nargs="?", help="Plain text prompt (or use stdin)")
    p.add_argument("--json", action="store_true", help="Print JSONL events and final JSON")
    p.add_argument("--output-schema", metavar="NAME_OR_PATH", help="Control final response shape")
    p.add_argument("--max-bytes", type=int, default=300_000, help="Hard cap for total context bytes")
    p.add_argument("--max-files", type=int, default=30, help="Soft cap for number of context files")
    p.add_argument("--include", action="append", default=[], help="Allowlist glob patterns (repeatable)")
    p.add_argument("--exclude", action="append", default=[], help="Blocklist glob patterns (repeatable)")
    p.add_argument("--exts", metavar="CSV", help="Restrict to file extensions (comma-separated)")
    p.add_argument("--diff-since", metavar="REV", help="Prioritize files changed since git rev or duration")
    p.add_argument("--no-git", action="store_true", help="Skip git-aware heuristics")
    p.add_argument("--no-scan", action="store_true", help="Skip directory scan; use only JSON-provided context")
    p.add_argument("--semantic", action="store_true", help="Use semantic search (requires .neo/index.json)")
    p.add_argument("--stdin-json", action="store_true", help="Force JSON input mode")
    p.add_argument("--stdin-text", action="store_true", help="Force text input mode")
    p.add_argument("--dry-run", action="store_true", help="Show what would be sent to model and exit")
    p.add_argument("--split", default="train", help="Dataset split (train/test/validation)")
    p.add_argument("--columns", metavar="JSON", help="Column mapping JSON (e.g., '{\"text\":\"pattern\"}')")
    p.add_argument("--limit", type=int, default=1000, help="Max samples to import (default: 1000)")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return p.parse_args()


def detect_input_mode(args):
    """Detect whether input is JSON or plain text."""
    import io

    if args.stdin_json:
        return "json"
    if args.stdin_text:
        return "text"

    # Auto-detect from stdin
    if not sys.stdin.isatty():
        buf = sys.stdin.read()
        stripped = buf.lstrip()
        if stripped.startswith(("{", "[")):
            try:
                json.loads(buf)
                sys.stdin = io.StringIO(buf)
                return "json"
            except json.JSONDecodeError:
                sys.stdin = io.StringIO(buf)
                return "text"
        else:
            sys.stdin = io.StringIO(buf)
            return "text"

    return "text"


def read_prompt_from_argv_or_stdin(args):
    """Read prompt from argv or stdin."""
    if args.prompt and args.prompt != "-":
        return args.prompt

    if not sys.stdin.isatty():
        return sys.stdin.read().strip()

    print("Error: No prompt provided. Use: neo \"your prompt\" or pipe via stdin", file=sys.stderr)
    sys.exit(2)


def _configure_logging(args) -> None:
    """Configure logging based on CLI flags, env var, or config.

    Priority: --debug > --verbose > NEO_LOG_LEVEL env > config.log_level > WARNING
    Output goes to stderr so it doesn't interfere with JSON on stdout.
    """
    # Determine level from flags
    if getattr(args, "debug", False):
        level = logging.DEBUG
    elif getattr(args, "verbose", False):
        level = logging.INFO
    else:
        # Check env, then fall back to default (config loaded later)
        env_level = os.environ.get("NEO_LOG_LEVEL", "").upper()
        level = getattr(logging, env_level, None) if env_level else None

    if level is None:
        # Will be reconfigured after config is loaded if needed
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(name)s %(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
    # Quiet noisy third-party loggers
    for name in ("httpx", "httpcore", "openai", "anthropic", "urllib3", "fastembed", "onnxruntime"):
        logging.getLogger(name).setLevel(max(level, logging.WARNING))


def main():
    """Main entry point for stdin/stdout interface."""
    # Parse arguments
    args = parse_args()

    # Configure logging early so all subsequent code can log
    _configure_logging(args)

    # Check for updates (non-blocking, silent on failure)
    # Skip if running certain commands to avoid noise
    skip_update_check = (
        hasattr(args, 'version') and args.version or
        hasattr(args, 'config') and args.config or
        hasattr(args, 'command') and args.command == 'update'
    )
    if not skip_update_check:
        try:
            from neo.update_checker import check_for_updates
            from neo.config import NeoConfig

            # Load config to check if auto-install is enabled
            config = NeoConfig.load()
            check_for_updates(auto_install=config.auto_install_updates)
        except Exception as e:
            logger.debug(f"Update check failed: {e}")

    # Handle global flags first (exist on all parsers, must check before subcommand-specific attributes)

    # Handle --version flag
    if args.version:
        codebase_root = args.cwd or os.getcwd()
        show_version(codebase_root)
        sys.exit(0)

    # Handle --config flag
    if args.config:
        handle_config(args)
        sys.exit(0)

    # Handle --load-program flag
    if args.load_program:
        handle_load_program(args)
        sys.exit(0)

    # Handle --regenerate-embeddings flag
    if args.regenerate_embeddings:
        from neo.config import NeoConfig
        try:
            config = NeoConfig.load()
            codebase_root = args.cwd or os.getcwd()
            result = regenerate_embeddings(codebase_root=codebase_root, config=config)
            print(f"\u2713 Regenerated embeddings for {result['success']}/{result['total']} entries")
            print(f"  Model: {result['model']}")
            print(f"  Duration: {result['duration']:.1f}s")
            if result['failed'] > 0:
                print(f"  \u26a0 Warning: {result['failed']} entries failed to regenerate")
            sys.exit(0)
        except RuntimeError as e:
            print(f"\u2717 Regeneration failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Handle --index flag
    if args.index:
        from neo.index.project_index import ProjectIndex

        codebase_root = args.cwd or os.getcwd()
        print(f"[Neo] Building semantic index for {codebase_root}...")

        index = ProjectIndex(codebase_root)

        # Parse languages if provided
        languages = None
        if hasattr(args, 'languages') and args.languages:
            languages = [lang.strip() for lang in args.languages.split(',')]
            print(f"[Neo] Indexing languages: {', '.join(languages)}")

        max_files = 100  # Configurable later

        try:
            index.build_index(languages=languages, max_files=max_files)
            status = index.status()
            print(f"[Neo] Built index: {status['total_chunks']} chunks from {status['total_files']} files")
            print(f"[Neo] Index stored in {codebase_root}/.neo/")
            print("[Neo] Supported languages: Python, C#, TypeScript, JavaScript, Java, Go, Rust, C/C++")
            print("[Neo] Use '--semantic' flag to enable semantic search")
            sys.exit(0)
        except Exception as e:
            print(f"[Neo] Failed to build index: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Handle contribute subcommand
    if hasattr(args, 'command') and args.command == 'contribute':
        from neo.subcommands import handle_contribute
        handle_contribute(args)
        sys.exit(0)

    # Handle update subcommand
    if hasattr(args, 'command') and args.command == 'update':
        handle_update(args)
        sys.exit(0)

    # Handle construct subcommand
    if hasattr(args, 'command') and args.command == 'construct':
        handle_construct(args)
        sys.exit(0)

    # Handle prompt subcommand
    if hasattr(args, 'command') and args.command == 'prompt':
        handle_prompt(args)
        sys.exit(0)

    # Detect input mode
    input_mode = detect_input_mode(args)

    # Parse input based on mode
    if input_mode == "json":
        try:
            input_data = json.loads(sys.stdin.read())
            working_dir = input_data.get("working_directory") or args.cwd or os.getcwd()
            neo_input = NeoInput(
                prompt=input_data["prompt"],
                task_type=TaskType(input_data.get("task_type", "feature")),
                context_files=[
                    ContextFile(**cf) for cf in input_data.get("context_files", [])
                ],
                error_trace=input_data.get("error_trace"),
                recent_commands=input_data.get("recent_commands", []),
                safe_read_paths=input_data.get("safe_read_paths", []),
                working_directory=working_dir,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            error_output = {"error": f"Invalid JSON input: {e}"}
            print(json.dumps(error_output, indent=2))
            sys.exit(1)
    else:
        # Plain text mode
        prompt = read_prompt_from_argv_or_stdin(args)
        working_dir = args.cwd or os.getcwd()

        neo_input = NeoInput(
            prompt=prompt,
            task_type=TaskType.FEATURE,
            context_files=[],
            working_directory=working_dir,
            safe_read_paths=[working_dir],
        )

        # Gather context from working directory unless --no-scan
        if not args.no_scan:
            from neo.context_gatherer import gather_context, gather_context_semantic, GatherConfig

            exts = args.exts.split(',') if args.exts else None

            config = GatherConfig(
                root=working_dir,
                prompt=prompt,
                exts=exts,
                includes=args.include,
                excludes=args.exclude,
                max_bytes=args.max_bytes,
                max_files=args.max_files,
                diff_since=args.diff_since,
                use_git=not args.no_git,
            )

            # Use semantic search if --semantic flag is set
            if args.semantic:
                gathered = gather_context_semantic(config)
            else:
                gathered = gather_context(config)

            # Convert gathered files to ContextFile format
            neo_input.context_files = [
                ContextFile(
                    path=gf.path,
                    content=gf.content,
                    line_range=(gf.start, gf.end) if gf.start else None
                )
                for gf in gathered
            ]

            # Print summary to stderr
            total_bytes = sum(gf.bytes for gf in gathered)
            print(f"[Neo] Gathered {len(gathered)} files ({total_bytes:,} bytes)", file=sys.stderr)
            print("[Neo] Invoking LLM inference...", file=sys.stderr)

            if args.dry_run:
                print("\n=== DRY RUN: Context that would be sent ===\n", file=sys.stderr)
                for gf in gathered:
                    lines_info = f" (lines {gf.start}-{gf.end})" if gf.start else ""
                    print(f"  {gf.rel_path}{lines_info} - {gf.bytes} bytes (score: {gf.score:.2f})", file=sys.stderr)
                print(f"\nPrompt: {prompt[:200]}...\n", file=sys.stderr)
                sys.exit(0)

    # Initialize adapter from environment
    # NO STUBS OR FALLBACKS - require real configuration
    from neo.adapters import create_adapter
    from neo.config import NeoConfig

    try:
        # Load config to get API key
        config = NeoConfig.load()

        # Upgrade log level from config if no CLI flag was set
        if not getattr(args, "debug", False) and not getattr(args, "verbose", False):
            if not os.environ.get("NEO_LOG_LEVEL"):
                cfg_level = getattr(logging, config.log_level.upper(), None)
                if cfg_level is not None:
                    logging.getLogger().setLevel(cfg_level)

        adapter = create_adapter(
            provider=config.provider,
            model=config.model,
            api_key=config.api_key
        )
    except Exception as e:
        error_output = {
            "error": f"Failed to initialize LM adapter: {e}",
            "hint": "Set NEO_PROVIDER and NEO_MODEL in config.json or environment, or set provider-specific API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)"
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)

    # Create engine and process (with codebase root for per-codebase learning)
    try:
        engine = NeoEngine(
            lm_adapter=adapter,
            codebase_root=neo_input.working_directory,
            config=config
        )
        output = engine.process(neo_input)
    except TimeoutError as e:
        error_output = {
            "error": "RequestTimeout",
            "message": "LLM request exceeded timeout limit",
            "timeout_seconds": 300,
            "details": str(e),
            "suggestions": [
                "Try simplifying your prompt",
                "Break complex queries into smaller parts",
                "Check your network connection"
            ]
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)
    except ValueError as e:
        error_msg = str(e)
        error_output = {
            "error": "ValidationError",
            "message": error_msg,
            "suggestions": []
        }

        # Provide specific suggestions based on error type
        if "schema" in error_msg.lower() or "validation" in error_msg.lower():
            error_output["suggestions"] = [
                "Check that LLM output includes required fields",
                "Verify schema_version is set to '3'",
                "Review structured_parser.py for validation rules"
            ]
        elif "parse" in error_msg.lower():
            error_output["suggestions"] = [
                "LLM may have produced invalid JSON",
                "Try re-running the query",
                "Check lm_logger output for raw response"
            ]
        else:
            error_output["suggestions"] = [
                "Review the error message for specific details",
                "Check Neo's logs for more context"
            ]

        print(json.dumps(error_output, indent=2))
        sys.exit(1)
    except Exception as e:
        # Import httpx to check for timeout errors
        try:
            import httpx
            if isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout)):
                error_output = {
                    "error": "NetworkTimeout",
                    "message": f"Network request timed out: {str(e)}",
                    "timeout_seconds": 300,
                    "suggestions": [
                        "Check your internet connection",
                        "Verify API endpoint is accessible",
                        "Try again in a moment"
                    ]
                }
                print(json.dumps(error_output, indent=2))
                sys.exit(1)
        except ImportError:
            pass

        # Generic error handler
        error_output = {
            "error": "ProcessingError",
            "message": f"Unexpected error during processing: {str(e)}",
            "error_type": type(e).__name__,
            "suggestions": [
                "Check Neo's logs for detailed stack trace",
                "Verify input format is correct",
                "Report this issue if it persists: https://github.com/Parslee-ai/neo/issues"
            ]
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)

    # Serialize output
    try:
        output_dict = {
            "plan": [
                {
                    "description": step.description,
                    "rationale": step.rationale,
                    "dependencies": step.dependencies,
                }
                for step in output.plan
            ],
            "simulation_traces": [
                {
                    "input_data": trace.input_data,
                    "expected_output": trace.expected_output,
                    "reasoning_steps": trace.reasoning_steps,
                    "issues_found": trace.issues_found,
                }
                for trace in output.simulation_traces
            ],
            "code_suggestions": [
                {
                    "file_path": sugg.file_path,
                    "unified_diff": sugg.unified_diff,
                    "description": sugg.description,
                    "confidence": sugg.confidence,
                    "tradeoffs": sugg.tradeoffs,
                }
                for sugg in output.code_suggestions
            ],
            "static_checks": [
                {
                    "tool_name": check.tool_name,
                    "diagnostics": check.diagnostics,
                    "summary": check.summary,
                }
                for check in output.static_checks
            ],
            "next_questions": output.next_questions,
            "confidence": output.confidence,
            "notes": output.notes,
            "metadata": output.metadata,
        }

        # Add confidence interpretation for better UX
        confidence_interpretation = _interpret_confidence(
            output.confidence,
            output.next_questions,
            output.plan,
            output.code_suggestions
        )
        output_dict["confidence_interpretation"] = confidence_interpretation

        # Output based on mode
        if args.json:
            # JSON mode: print structured output
            print(json.dumps(output_dict, indent=2))
        else:
            # Human-readable text mode
            print("\n" + "="*80)
            print(f"CONFIDENCE: {output.confidence:.2f}")
            print("="*80)

            if output.notes:
                print(f"\n{output.notes}\n")

            print("\nPLAN:")
            for i, step in enumerate(output.plan, 1):
                print(f"\n{i}. {step.description}")
                print(f"   Rationale: {step.rationale}")
                if step.dependencies:
                    print(f"   Dependencies: {step.dependencies}")

            if output.simulation_traces:
                print("\n" + "-"*80)
                print("SIMULATIONS:")
                for i, trace in enumerate(output.simulation_traces, 1):
                    print(f"\nScenario {i}:")
                    print(f"  Input: {trace.input_data}")
                    print(f"  Expected: {trace.expected_output}")
                    if trace.issues_found:
                        print(f"  Issues: {', '.join(trace.issues_found)}")

            if output.code_suggestions:
                print("\n" + "-"*80)
                print("CODE SUGGESTIONS:")
                for i, sugg in enumerate(output.code_suggestions, 1):
                    print(f"\n{i}. {sugg.file_path} (confidence: {sugg.confidence:.2f})")
                    print(f"   {sugg.description}")
                    if sugg.unified_diff:
                        print("\n" + sugg.unified_diff)

            if output.next_questions:
                print("\n" + "-"*80)
                print("NEXT QUESTIONS:")
                for q in output.next_questions:
                    print(f"  - {q}")

            print("\n" + "="*80 + "\n")
    except Exception as e:
        error_output = {
            "error": "SerializationError",
            "message": f"Failed to serialize output: {str(e)}",
            "error_type": type(e).__name__,
            "suggestions": [
                "Output may contain non-serializable data",
                "Check Neo's internal data structures",
                "Report this issue: https://github.com/Parslee-ai/neo/issues"
            ]
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
