# Changelog

## Unreleased

### Added

- **Full multi-language coverage for Ruby, Kotlin, Swift, and PHP.** Previously these extensions were recognized but produced no analysis. Now they get:
  - **God-file detection** via LOC and function/method counts from tree-sitter queries.
  - **Empty catch/rescue detection.** PHP uses the standard `catch_clause`; Kotlin and Swift use `catch_block` (empty = no `statements` child); Ruby uses `rescue` (empty = no `then` child), and the finding message reads "empty rescue block" so it isn't mis-labeled as a catch.
  - **Semantic chunking + import edges** for Kotlin (`import_header`), Swift (`import_declaration`), and PHP (`namespace_use_declaration`). Ruby imports are runtime method calls (`require`, `require_relative`) so they're left out of edge extraction for now.

### Changed (breaking)

- **`tree-sitter` is now a required dependency, not optional.** Empty-catch detection, multi-language god-file metrics, and the semantic-chunking code path it powers are core features — the previous `[tree-sitter]` extra was a fiction, and on Python 3.13+ it could not install at all because `tree-sitter-languages` had no wheels. Switched to `tree-sitter-language-pack` (>=0.13.0,<1.0, the maintained successor) and promoted both `tree-sitter` and the language pack into the base `dependencies` block. The graceful-degrade `TREE_SITTER_AVAILABLE` scaffolding in `language_parser.py`, `code_smells.py`, and `architecture_metrics.py` has been removed; install neo and tree-sitter installs with it.

### Removed

- **`neo.self_correction` and `neo.input_templates` modules deleted.** Both dead since v0.7.0 — commit 6e35f5d acknowledged the pattern-learning system had been unreachable since the initial public release, but the "fix" only revived sibling modules. `self_correction.py` had no callers; `input_templates.py` was imported only from `self_correction.py` and an import-health smoke test, so deleting one orphaned the other. The live learning path runs through `neo.pattern_extraction.extract_pattern_from_correction` via `memory/store.py`. If you were importing from either module, the surviving surface lives in `neo.pattern_extraction` and `neo.algorithm_design`.

### Changed

- **`static_analysis`: pyright and mypy now both run when both are enabled.** Previously the dispatch had a hidden `pyright … elif mypy` mutex that suppressed mypy whenever pyright was enabled. They flag different things, so the coupling cost real diagnostics. Users running both intentionally now get both sets of findings.

### Added

- **`neo serve` — CAR-hosted tool surface for `neo.process` (Phase 1 of the CAR migration).** Opt-in via the new `[car]` extra (`pip install 'neo-reasoner[car]'`); requires a running `car-server` daemon. Boots a `CarRuntime`, registers `neo.process` with a JSON-Schema-typed `params` shape derived from `NeoInput`, installs a Python `tools.execute` handler (closes the consumer-facing half of Parslee-ai/car-releases#38 — first downstream user of the new PyO3 surface), and binds car-server's A2A HTTP listener so an Agent Card is served at `/.well-known/agent-card.json`. CAR-native callers can submit proposals against `neo.process` today and reach the handler end-to-end. **Known limitation**: car-server's `start_a2a` spins up an isolated Runtime, so FFI-registered tools don't yet appear in the Agent Card's `skills` list and A2A peer dispatch for `neo.process` doesn't route back to Python — pending a CAR-side runtime-sharing fix in `car-server-core::a2a::start_a2a`. Tracked CAR-side; the CAR-native path is unaffected.
  - New: `src/neo/car_tool_schema.py` (tool schema + NeoInput/NeoOutput dict converters), `src/neo/car_host.py` (server entry, per-codebase NeoEngine cache, signal-driven shutdown), `tests/test_car_tool_schema.py` (15 unit tests), `tests/test_car_host_smoke.py` (live end-to-end smoke; auto-skipped without daemon).
  - CLI: `neo serve --a2a-bind HOST:PORT [--public-url URL] [--agent-name NAME]`.

## [0.16.1] - 2026-05-04

### Fixed
- Auto-updater now routes through the install method that owns neo (#89). Previously it unconditionally shelled out to `pip install --upgrade --break-system-packages --ignore-installed`, which on Homebrew Python wrote a duplicate copy without removing the old one — `importlib.metadata` resolved the stale copy on next start, producing an infinite "upgrading from X to Y ✓ Success" loop in `~/.neo/auto_update.log`. Detection now distinguishes pipx / pip-venv / brew-formula / external (PEP-668) installs and uses the correct upgrade command (or prints user guidance, throttled per version, when pip would do harm).
- Drop update-check interval from 24h to 1h with stale-while-revalidate so users on auto-update receive new releases within ~1 hour of publication instead of up to 24h (#87).
- Disable exploration in ranking tests to eliminate the last 3% test flake (#88).
- Stabilize ReasoningBank tests across Python versions (#86).

## [0.16.0] - 2026-05-02

### Added
- **Codex plugin** at `plugins/neo/` exposes the same six skills the Claude Code plugin offers (`$neo`, `$neo-review`, `$neo-optimize`, `$neo-architect`, `$neo-debug`, `$neo-pattern`), packaged via `.codex-plugin/plugin.json` plus a repo marketplace at `.agents/plugins/marketplace.json`. Skills wrap the local `neo` CLI; persistent memory in `~/.neo/` is shared across both plugins.
- Memory-driven reasoning effort for OpenAI gpt-5* models. Each query's `reasoning.effort` is sized from the strength of the memory hit: high-confidence pattern match → `low` (cheap), no relevant memory → `high`, no memory + hard difficulty → `xhigh`. Cap with `NEO_REASONING_EFFORT` for cost control. End-to-end measurement on a familiar query: `Reasoning effort: low (patterns=5, avg_conf=0.91)` — neo's learning monetizes directly into inference cost.
- Code-smell detection during context assembly. Surfaces TODO/FIXME/HACK/XXX markers, Python stubs (pass / `...` / `raise NotImplementedError`), bare `except:`, swallowed exceptions, and hardcoded credentials (OpenAI / AWS / GitHub / Slack token shapes) in the prompt under "KNOWN ISSUES IN NEARBY CODE." Per-file cap of 8 + global cap of 20 keeps growth bounded.
- Automatic discovery of project-local AI agent instruction docs: `CLAUDE.md`, `AGENTS.md`, `.cursorrules`, `.windsurfrules`, `.claude/`, `.cursor/rules/`, `.github/copilot-instructions.md`, `.continue/`, `.augment/`, `.specify/` (Spec Kit), `.aider/`, `.codeium/`. Markdown content surfaces unconditionally in the prompt under "PROJECT-LOCAL AGENT CONTEXT" — works whether you use Claude Code, Cursor, Copilot, Aider, Continue, Augment, or Windsurf.
- Native architectural quality delta closes the outcome learning loop. At session save time neo snapshots three structural metrics — import-graph cycles (Tarjan SCC), god files (LOC + function-count thresholds), max nesting depth — and at outcome detection time it diffs against the current state. A regression weakens the accept/boost or strengthens the modify/penalty by 0.1; an improvement does the reverse. Failure-tolerant; collapses to neutral if metrics computation hits any error.

### Changed
- Default OpenAI model bumped to `gpt-5.5` (previous: `gpt-5.3-codex`). gpt-5.5 routes through the existing `/v1/responses` path; reasoning + message output shape handled unchanged.

## [0.15.5] - 2026-04-15

### Fixed
- Revive the learning feedback loop after the recent gpt-5 routing changes had quietly broken it (#84).

## [0.15.4] - 2026-04-14

### Fixed
- Raise the bar above the base LM by closing four reasoning gaps that caused Neo to collapse to "LLM + logging":
  - Objective early-exit gate: require self-confidence >0.8 AND static checks actually ran clean AND simulation traces agree, instead of trusting self-reported confidence alone (was bailing in 47.7% of sessions)
  - Share `success_bonus(n)` between `FactStore.retrieve_relevant` and `ContextAssembler._score_facts` via `memory.models`, so outcome learning influences ranking on the main retrieval path; cap bonus at 0.2 so a narrow historical winner can't dominate cosine similarity
  - Wire `ConstraintVerifier.extract_constraints` into the engine's main path with typed constraints threaded as parameters (no instance state), marker dict keyed by `ConstraintType` enum, comment/string stripping before substring match
  - Community-facts fallback in `_retrieve_context` so prompts always carry some memory-derived context when `FactStore` retrieval returns empty

### Changed
- Extracted `NeoEngine._finalize_output` as single exit point for both early-exit and full-pipeline paths (eliminates duplicated save/log/telemetry blocks)
- Narrowed silent `except Exception` in community fallback to `(OSError, json.JSONDecodeError)`

### Removed
- Unsafe `ConstraintVerifier.verify_code` subprocess path (executed arbitrary generated code; never called)

## [0.15.3] - 2026-04-12

### Fixed
- Repair broken feedback loop so patterns actually rise in confidence: session log accumulation (JSONL) instead of single-session overwrite, path normalization for bare leading slashes, fallback path lookup with/without leading `/`, stronger confidence boosts (+0.2 accepted, +0.1 unverified), and log2-scaled success bonus in retrieval scoring
- Non-git outcome detection now processes all previous sessions correctly (detect_outcomes runs before save_session, so all log entries are from prior invocations)

## [0.15.2] - 2026-04-08

### Fixed
- Outcome learning now persists `code_block` suggestions in session records and uses them to classify accepted vs modified edits when `unified_diff` is empty, restoring feedback learning for Neo's code-first output mode

## [0.15.1] - 2026-04-07

### Fixed
- Auto-updater failing on Homebrew Python due to broken dependency RECORD files (e.g., pillow installed by Homebrew lacks RECORD metadata, causing "Cannot uninstall" errors). Added `--ignore-installed` flag for externally-managed environments.

## [0.15.0] - 2026-04-07

### Fixed
- Close broken feedback loop: false "accepted" signals when suggested diffs were empty, unbounded independent outcome flooding (2800+ noise facts in active repos), and community facts silently lost on cross-project saves
- Fix diff overlap conflating additions and removals (`+foo` and `-foo` treated as same line)
- Fix empty actual diff fallthrough incorrectly classifying outcomes as "accepted"
- Fix test suite polluting real `~/.neo/constraints/checksums.json` with pytest temp paths
- Fix `_cap_independent_facts` calling `save()` during `load()` (deferred via `_cap_pending` flag)

### Added
- `OutcomeType` enum replacing stringly-typed outcome classification (ACCEPTED, MODIFIED, UNVERIFIED, INDEPENDENT)
- "Unverified" outcome type for suggestions where no diff comparison is possible (+0.05 confidence boost, no success_count increment)
- Rate-limit independent outcomes to 5 per session, sorted by diff size with deterministic tiebreaking
- Cap independent facts at 50 per project with automatic invalidation of excess on load
- `PROTECTED_TAGS` (seed, community, synthesized) — protected from pruning, demotion, and eviction
- Best-effort merge on global fact save to prevent cross-project data loss
- Cross-referenced documentation between `MAX_INDEPENDENT_OUTCOMES` (5/session) and `MAX_INDEPENDENT_FACTS` (50/project)

### Changed
- Independent fact confidence lowered from 0.3 to 0.2 for faster stale pruning
- Diff filter logic simplified from nested if/elif to single list comprehension
- Test fixture converted to generator (yield) for proper patch lifetime
- Inline `import datetime` moved to module level

## [0.14.0] - 2026-04-05

### Fixed
- Wire pattern learning pipeline that has been dead code since v0.7.0 — bare module imports silently caught by try/except ImportError disabled the entire self-correction and prevention pattern system
- Fix all internal module imports (pattern_extraction, algorithm_design, input_templates) to use package-qualified `neo.X` paths
- Remove phantom module imports (enhanced_simulation, iterative_refinement) that never existed
- Delete ghost CONSTRAINT_VERIFICATION try/except block containing no actual import
- Add validation to PatternLibrary.add_pattern rejecting junk entries with empty keywords or placeholder rules
- Fix wrong import paths across 4 files (neo.cli/neo → neo.models for LMAdapter, CodeSuggestion, etc.)
- Remove misleading ImportError guards on required dependencies (sklearn, jsonschema)
- Fix flaky CLI subprocess test timeout (10s → 30s for construct subcommand with heavy imports)

### Added
- Prevention warnings from learned patterns now injected into every engine prompt
- "Modified" outcome type detects when users correct neo's suggestions (compares suggested diff vs actual diff using Jaccard overlap at 30% threshold)
- Prevention pattern extraction from user corrections via LLM analysis — neo learns from its mistakes
- Original fact confidence demoted when suggestions are modified by users
- Import health test suite (49 tests) verifying all modules import and no phantom modules exist — catches silent import failures in CI
- Diff overlap and modified outcome tests (11 tests)
- Neo-character greeting printed before context gathering, contextual to prompt and memory level (Matrix-themed beat deck, no LLM call)

### Changed
- Internal neo module imports converted from try/except fallback to direct imports — broken features now fail loudly at import time instead of silently disabling
- Session records now persist suggested_diff for outcome comparison

## [0.13.3] - 2026-04-05

### Changed
- `neo --version` contribution status now shows pattern count and progress toward contribution threshold instead of a generic message

## [0.13.2] - 2026-04-05

### Fixed
- Run full initialization in `neo --version` so seed and community facts load on first run
- Always show community contribution status in version output (not just when ready)
- Migrate existing users to auto-update on upgrade (old default was false, now true)
- Seed RNG in reasoning bank tests to eliminate flaky CI rankings
- Add pre-commit hook (ruff + pytest) and CI gate on publish workflow

## [0.13.1] - 2026-04-05

### Added
- Ship 20 curated seed facts (security, performance, reliability, correctness) with every release — `pip install` is now a knowledge upgrade
- Community fact feed: neo fetches crowd-curated patterns from GitHub daily, updated between releases via PR
- `neo contribute` command: exports anonymized high-quality patterns for community contribution
- `neo --version` shows contribution hint when local patterns qualify (confidence >0.8, 3+ successes)
- Auto-update enabled by default (`auto_install_updates: true`)

### Fixed
- Fix CI: patch all ingesters in test fixtures to prevent seed/community facts from leaking into unit tests

## [0.13.0] - 2026-04-05

### Fixed
- Remove time-based memory decay that punished inactivity instead of poor quality — vacations and project switches no longer crush memory level
- Remove 14-day half-life from `persistent_reasoning.score()`, 30-day half-life from `store.retrieve_relevant()` and `context._score_facts()`
- Replace count-based `memory_level()` with quality-weighted sigmoid that reflects actual fact validation

### Added
- Per-scope capacity limits: GLOBAL (200), ORG (100), PROJECT (500), SESSION (50) with quality-based eviction when full
- Claude Code auto-memory ingestion: reads curated knowledge from `~/.claude/projects/{id}/memory/*.md` as neo facts
- AI tool instruction file ingestion: `.cursorrules`, `.windsurfrules`, `.clinerules`, `.github/copilot-instructions.md` now ingested as constraints alongside CLAUDE.md
- New `ClaudeMemoryIngester` class with YAML frontmatter parsing and type mapping (project→DECISION, feedback→PATTERN, reference→ARCHITECTURE)

### Changed
- Retrieval scoring simplified to `similarity * confidence` — old facts rank by quality, not recency
- Memory level now scales reference quality to loaded scope capacity so per-project views are meaningful

## [0.12.0] - 2026-04-02

### Fixed
- Fix dead code / memory leak: `_store_reasoning` cleanup was unreachable after two early returns
- Fix path traversal in `_read_safe_files`: resolve `base_dir` before containment check
- Fix `show_version` displaying stale stats from legacy memory backend instead of configured FactStore
- Remove telemetry exfiltration risk: `NEO_TELEMETRY_ENDPOINT` env var allowed sending data to arbitrary URLs
- Fix `sys.argv` mutation in `parse_args` — now uses `sys.argv[2:]` slicing instead of `pop(1)`
- Fix `iter_paths` return type annotation (was 2-tuple, actually 3-tuple)
- Fix broken `sys.path` hack for `ProjectIndex` import in `context_gatherer.py`
- Replace MD5 with SHA256 for embedding cache keys

### Changed
- **Breaking**: Split `cli.py` (3609 lines) into `models.py`, `engine.py`, `subcommands.py` with backward-compat re-exports
- Consolidate 4 copy-paste `_simulate_*` methods into single `_run_simulation` with dispatch table
- Extract `cosine_similarity` into `math_utils.py`, replacing 3 duplicate implementations
- Extract `FactStore.initialize()` from `__init__` with `eager_init` parameter for lightweight construction
- Atomic writes for fact persistence via `tempfile.mkstemp` + `os.replace` with cleanup on failure
- Upgrade memory pipeline exception handlers from debug to warning level

## [0.11.3] - 2026-04-02

### Fixed
- Fix silent memory pipeline failure: JSON input mode left `working_directory` as None when not provided, causing empty `project_id` and all memory operations (session saves, outcome detection, fact persistence) to silently no-op. Now falls back to `--cwd` or `os.getcwd()`
- Elevate silent `debug`-level exception handlers in `FactStore.save_session()` and `detect_implicit_feedback()` to `warning` level so memory failures are visible
- Add warning logs when `project_id` is empty in `OutcomeTracker.save_session()` and `_compute_project_id()` instead of returning silently

### Added
- Configurable logging infrastructure: `--verbose` (INFO), `--debug` (DEBUG), `NEO_LOG_LEVEL` env var, and `config.log_level` setting for diagnosing memory pipeline issues
- Token budget enforcement and inline change annotations (#75)

## [0.11.2] - 2026-03-18

### Added
- Outcome linkage: accepted suggestions now boost the original fact's confidence and success_count instead of creating orphan REVIEW facts
- Review synthesis: clusters of similar REVIEW facts are distilled into single PATTERN facts via embedding-based complete-linkage clustering
- LLM-based synthesis: clusters of 5+ facts optionally use an LLM for richer distillation (falls back to mechanical synthesis)
- Quality pruning: stale facts (low confidence, zero successes, >14 days old) are automatically removed
- Success/failure-based demotion: facts retrieved 5+ times without validation lose confidence; 10+ times get invalidated; consistently helpful facts get protected
- Full maintenance chain in `detect_implicit_feedback`: synthesize → prune stale → demote unhelpful → purge dead

### Changed
- `detect_outcomes()` now returns suggestion_fact_ids for outcome-to-fact linkage
- `SessionRecord` carries `suggestion_fact_ids` mapping for cross-invocation tracking
- `FactMetadata` gains `success_count` field for tracking validated suggestions
- `FactStore` accepts optional `lm_adapter` for LLM-based synthesis

## [0.11.1] - 2026-03-17

### Fixed
- Fix memory facts decaying to zero: `build_context()` now updates `last_accessed` and `access_count` on retrieved facts, matching the behavior of `retrieve_relevant()`
- Isolate tests from live `~/.neo/` memory files (#76)
- Handle PEP 668 externally-managed environments in auto-update
- Remove unused `Outcome` import in test_outcomes

## [0.11.0] - 2026-03-14

### Added
- Outcome-based learning from git history and code changes — neo now learns from what actually happens in the codebase, not just its own reasoning output
- Git history ingestion: on each invocation, ingests new commits since a watermark (commit messages, changed files, diffs) to learn from all code evolution
- Session-based outcome detection: tracks neo's suggestions between invocations and compares git diff against previous suggestions to detect accepted vs independent changes with actual diff content
- Replaces no-op `detect_implicit_feedback` stub in FactStore with real implementation

## [0.10.0] - 2026-02-16

### Added
- Replace memory system with fact-based store — scoped facts (global/org/project), supersession-based deduplication, four-layer context assembly inspired by StateBench (#71)
- Prompt enhancement system for analyzing Claude Code effectiveness (#67)
- Fully automatic update system with opt-in auto-install (#64)
- Tree-sitter multi-language code indexing with FAISS-backed semantic search (#63)

### Changed
- Add autonomous agent slash commands for bug, feature, and chore workflows
- Add fix-ci slash command for CI failure repair

### Documentation
- Update all documentation (README, INSTALL, QUICKSTART, CONTRIBUTING, LOAD_PROGRAM, SECURITY) to reflect fact-based memory system

### CI
- Bump actions/upload-artifact from 5 to 6 (#68)
- Bump actions/download-artifact from 6 to 7 (#69)
- Bump actions/checkout from 4 to 6 (#66)

## [0.9.0] - 2025-11-19

### BREAKING CHANGES

**Python Version Requirement**
- Minimum Python version increased from 3.9 to 3.10
- Required for google-genai SDK compatibility

**Google Gemini SDK Migration**
- Migrated from deprecated google-generativeai to official google-genai SDK
- google-generativeai reaches EOL on November 30, 2025
- Hard cutover approach - no backward compatibility with old SDK

### Upgrading from 0.8.x

**For Python 3.9 users:**
- This release requires Python 3.10+
- Upgrade Python before installing v0.9.0

**For Google Gemini users:**
```bash
# Upgrade Neo (pip automatically handles the SDK migration)
pip install --upgrade neo-reasoner[google]

# Verify installation
neo --version
```

**Model name changes (if you explicitly specify models):**
- Old: `gemini-pro` → New: `gemini-2.0-flash` (recommended default)
- Old: `gemini-pro-vision` → New: `gemini-2.0-flash`

**For OpenAI/Anthropic/Ollama users:**
```bash
# Just upgrade normally
pip install --upgrade neo-reasoner
```

No additional action required - pip handles all package dependencies automatically.

### Changed

**GoogleAdapter Updates**
- Replaced google-generativeai with google-genai>=0.2.0
- Updated client initialization to use `genai.Client(api_key=...)`
- Migrated to new `client.models.generate_content()` API
- Updated message format to use `types.GenerateContentConfig`
- Changed default model from "gemini-pro" to "gemini-2.0-flash"

**Dependencies**
- Updated pyproject.toml to require Python 3.10+
- Removed Python 3.9 classifier
- Updated tool configurations (black, ruff, mypy) to target Python 3.10

### Added

**CLI Enhancements**
- Enhanced `neo --version` output to display current provider and model (#61)
- Updated default OpenAI model to gpt-5.1-codex-max for improved performance (#61)

**Test Coverage**
- Added comprehensive test suite for GoogleAdapter (tests/test_google_adapter.py)
- Tests cover initialization, API key validation, message formatting, and response extraction
- All tests use mocks to avoid real API calls

**Documentation**
- Updated README.md with Python 3.10+ requirement for Google provider
- Updated model names to latest Gemini 2.0 models
- Added migration notes for google-genai SDK

## [0.8.1] - 2025-10-29

### Fixed

**Critical Bug Fixes**
- Prevent JSON serialization failure causing data loss in persistent reasoning (#44)
- Normalize empty strings before schema validation to prevent parser errors (#48, #56)
- Resolve ModuleNotFoundError for --index flag (issue #38) (#40)
- Reorder CLI flag checks to prevent AttributeError on --version (#37)
- Check for command attribute existence before accessing in CLI
- Check pattern file modifications for index freshness in Construct (#41)

**Model Compatibility**
- Upgrade deprecated Anthropic model to claude-sonnet-4-5-20250929 (#55)

**Test Stability**
- Correct compositional strategy boundary condition to 70% (#54)
- Transform flaky latency test to behavioral semantic test (#53)
- Use set comparison for consistency test to handle score ties
- Resolve 5 failing tests in reasoning bank and failure learning (#42)

**Code Quality**
- Resolve 17 ruff linting violations for code quality compliance (#52)

### Added

**Dependencies**
- Add missing jsonschema dependency to pyproject.toml for schema validation

**CI/CD**
- Add GitHub Actions CI workflow for automated testing (#50)
- Bump actions/upload-artifact from 4 to 5 (#36)
- Bump actions/download-artifact from 4 to 6 (#35)
- Bump actions/checkout from 4 to 5 (#4)
- Bump actions/setup-python from 5 to 6 (#2)

### Changed

**Development**
- Update autonomous commands for Neo codebase (#39)
- Update .gitignore to exclude specs directory

## [0.8.0] - 2025-10-21

### Added

**Release Automation**
- Added `/prepare-release` command for automated version bumping and changelog updates (#23)
- Added `/ship-release` command for complete release workflow with PR creation and PyPI publishing (#23)
- Automated version updates across pyproject.toml, __init__.py, and plugin.json

**The Construct - Semantic Pattern Discovery**
- Added semantic pattern discovery system for extracting reusable patterns from successful code (#24)
- Pattern extraction with confidence scoring and similarity-based clustering
- Integration with Neo's semantic memory for pattern recall and reuse
- Enables learning from successful implementations across projects

**Executable Artifacts & Incremental Planning**

*Grounded in recent code generation research (Liu ICLR 2023, Zhang 2023, Huang 2025, Yao NAACL 2024)*

**Executable Artifacts for CodeSuggestion**
- Added 7 optional fields to CodeSuggestion schema for actionable outputs:
  - `patch_content`: Full unified diff content (not truncated)
  - `apply_command`: Shell command to apply change (ADVISORY - validate before execution)
  - `rollback_command`: Shell command to undo change (ADVISORY)
  - `test_command`: Shell command to verify change (ADVISORY)
  - `dependencies`: Array of suggestion IDs this depends on (execution order)
  - `estimated_risk`: Enum (low/medium/high) for risk assessment
  - `blast_radius`: Float 0.0-100.0 percentage of codebase files affected (files changed / total files × 100)
- Security warnings: All command fields documented as ADVISORY ONLY (never use shell=True)
- Backward compatible: All new fields optional, schema version remains v3

**Incremental Planning for PlanStep**
- Added 8 optional fields to PlanStep schema for as-needed decomposition:
  - `preconditions[]`: Conditions that must be met before execution
  - `actions[]`: Concrete actions to perform in this step
  - `exit_criteria[]`: Success verification criteria
  - `risk`: Step-specific risk level (low/medium/high)
  - `retrieval_keys[]`: Keywords for step-scoped memory retrieval (CodeSim-style)
  - `failure_signatures[]`: Known failure patterns from past attempts (ReasoningBank)
  - `verifier_checks[]`: Validation checks (MapCoder's Solver-Critic-Verifier pattern)
  - `expanded`: Boolean tracking if step was expanded from seed plan
- Enables seed plan → expand when blocked workflow (Yao et al., NAACL 2024)
- Step-level failure learning for ReasoningBank integration (Chen et al., 2025)

**Testing & Quality**
- Added 8 comprehensive schema validation tests using jsonschema
- All tests use actual `jsonschema.validate()` (not mocks)
- Test coverage: 100% of new schema fields validated
- Tests verify enum constraints, range validation, and backward compatibility
- Code review: Linus agent ACCEPT (kernel-level quality standards met)

**Documentation**
- Enhanced README with detailed schema documentation
- Expanded Research & References section with 8 academic papers
- Added proper links to papers, GitHub repos, and datasets
- Included citation block for academic use

### Changed

**Schema Enhancements**
- `blast_radius`: Changed from integer (1-100) to float (0.0-100.0) for precision
  - Allows sub-1% impact representation (e.g., 0.5% for large codebases)
- Command field descriptions: Added security warnings about safe execution
- Schema validation: Maintained strict `additionalProperties: False` for safety

### Performance

- Schema validation overhead: <10ms per suggestion/step (O(1) constant time)
- Memory footprint: ~50 bytes per new field with default values (negligible)
- Backward compatibility: Zero impact on existing code (optional fields)

### Research References

This release implements concepts from:
- Liu et al., ICLR 2023 - Planning-guided code generation (preconditions, exit criteria)
- Zhang et al., 2023 - Self-planning workflow (+7% HumanEval improvement)
- Huang et al., 2025 - AdaCoder adaptive multi-agent framework (risk assessment)
- Islam et al., 2024 - MapCoder Solver-Critic-Verifier (verifier_checks)
- Xu et al., 2023 - CodeSim step-level retrieval (retrieval_keys)
- Yao et al., NAACL 2024 - As-needed decomposition (expanded flag, incremental planning)
- Chen et al., 2025 - ReasoningBank failure learning (failure_signatures)
- Wang et al., 2024 - Multi-agent survey (architectural foundations)

## [0.7.6] - 2025-10-14

### Fixed
- Python 3.9 compatibility: Replaced Python 3.10+ union syntax (X | Y) with Optional/Union for broader compatibility (#21)
- Added missing `source_context` field to ReasoningEntry dataclass (#20)

### Documentation
- Updated documentation files to latest standards

## [0.7.5] - 2025-10-10

### Changed
- Bumped version to 0.7.5 to match plugin version for consistency

### Fixed
- Plugin file paths: Ensured all file paths are correctly relative to the plugin root (#15)
- Plugin file paths: Fixed to be relative to repository root (#14)

### Added
- Updated plugin version to 0.7.5 and removed redundant README.md file (#13)
- Load program feature: HuggingFace dataset import (#12)
- Required YAML front matter to command files for Claude Code compatibility (#11)
- Plugin install step to README (#10)

### Changed
- Increased default max_entries from 200 to 2000 for larger memory capacity (#7)

### Fixed
- Claude Code plugin manifest schema validation errors (#9)

## [0.7.4] - 2025-10-10

### Fixed
- ImportError: Export CodeSuggestion, PlanStep, SimulationTrace, and StaticCheckResult from neo package (Fixes #5)
- Version sync: Updated __version__ in __init__.py from 0.7.0 to 0.7.4 to match pyproject.toml

### Added
- GitHub community files for open source management (#6):
  - SECURITY.md with vulnerability reporting policy
  - PR template with comprehensive checklist
  - dependabot.yml for automated dependency updates

## [0.7.0] - 2025-10-10

### Added - ReasoningBank Implementation (Phases 2-5)

*Based on ReasoningBank paper (arXiv:2509.25140v1)*

**Phase 2: Semantic Anchor Embedding**
- Implemented semantic anchor strategy: embeddings now use pattern+context only (not full reasoning)
- Reduces noise in similarity matching by focusing on WHAT+WHEN instead of HOW
- Backward compatible with existing embeddings (no re-embedding required)

**Phase 3: Systematic Failure Learning**
- Added failure root cause extraction when confidence < 0.5
- LLM-based failure analysis with heuristic fallback for reliability
- Failure patterns stored in `common_pitfalls` and surfaced in Neo output
- Tracks WHY patterns fail, not just that they failed

**Phase 4: Self-Contrast Consolidation**
- Added `problem_outcomes` tracking for contrastive learning
- Archetypal patterns (consistent winners) get +0.2 confidence boost
- Spurious patterns (lucky once, fail elsewhere) get -0.2 penalty
- Enables learning "which patterns work WHERE OTHERS FAIL"

**Phase 5: Strategy Evolution Tracking**
- Added strategy level inference: procedural, adaptive, compositional
- Difficulty-aware retrieval boosts (compositional +0.15 on hard problems)
- Procedural strategies penalized -0.10 on hard problems to prevent poor suggestions
- Zero new schema fields - pure algorithmic leverage from existing difficulty_affinity data

**Testing & Quality**
- Added 39 comprehensive tests (all passing)
- Integration test suite validates all phases working together
- Performance benchmarks: 12.3ms avg retrieval (target <100ms)
- Kernel-quality code review by Linus agent

**Documentation**
- Phase-specific documentation for each improvement (phases 2-5)
- Production readiness checklist with deployment plan
- Benchmark impact analysis and performance validation
- Linus review findings and fixes documented

### Changed

**Performance Optimizations**
- Replaced recursive DFS with iterative to eliminate RecursionError risk
- Extracted magic numbers to named class constants for tunability
- Consistent difficulty validation across all code paths

**Code Quality**
- Added named constants for all tunable parameters:
  - `AFFINITY_BONUS_WEIGHT = 0.2`
  - `CONTRASTIVE_SCALE = 0.4`
  - `STRATEGY_BOOST_HARD_COMPOSITIONAL = 0.15`
  - `CONFIDENCE_BOOST_SUCCESS = 0.1`
- Improved confidence reinforcement from ±0.02 to ±0.1 (stronger learning signals)

### Fixed
- RecursionError risk in clustering DFS (now uses iterative approach)
- Inconsistent difficulty validation (now defaults invalid values to "medium")
- Zero-vector edge case in cosine similarity (already handled, verified)

### Performance Metrics
- Retrieval latency: 12.3ms avg (87% faster than 100ms target)
- Consolidation: <50ms for 5-entry clusters
- Strategy inference: 66.7% accuracy on test cases
- Contrastive boost: ±0.4 difference (archetypal vs spurious)

### Technical Debt (Documented & Acceptable)
- O(n³) contrastive boost complexity (acceptable for <200 entries)
- Hardcoded strategy thresholds (66.7% accuracy acceptable for v1)
- Both items tracked for future optimization if needed

## [0.2.0] - 2025-09-30

### Added
- Plain text input mode with smart context gathering (CLI ergonomics like Claude Code)
- Context gathering with .gitignore-aware file discovery and git-based prioritization
- Keyword-based relevance scoring for context files
- Refactoring warnings for files >50KB (god object detection)
- Warning headers in LLM context for large files to enable specific refactoring suggestions
- Missing datasketch dependency for MinHash-based similarity detection

### Changed
- Lowered default max_bytes from 300KB to 100KB for better gpt-5-codex performance
- Strengthened size penalty: 10KB=-0.1, 50KB=-0.5, 100KB=-1.0 (favor smaller modules)
- Fixed OpenAI adapter to support gpt-5-codex /v1/responses endpoint
- Increased HTTP timeout from 60s to 300s for complex prompts

### Fixed
- Added context_gatherer module to package distribution
- OpenAI adapter now uses correct endpoint and minimal payload for gpt-5-codex

## [0.1.0] - Initial Release
