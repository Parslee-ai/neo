# Project: Neo - Semantic Reasoning Helper

## Quick Context
- **Purpose**: Read-only reasoning helper for CLI tools using MapCoder/CodeSim-style multi-agent reasoning with semantic memory
- **Tech Stack**: Python 3.10+, fastembed (Jina Code v2, 768d), faiss-cpu (legacy pattern matching), Anthropic/OpenAI/Google LMs
- **Installation**: `pip install -e ".[dev]"` for development

## Code Style
- Import convention: stdlib → third-party → local, specific imports
- Naming: PascalCase classes, snake_case functions, UPPER_SNAKE constants, _private methods
- Error handling: Try/except with specific exceptions, logger warnings, graceful fallbacks
- Testing: test_*.py pattern, pytest framework
- Type hints: Extensive with Optional, list[], dict[]
- Docstrings: Triple quotes, brief description first

## Project Rules
- Keep implementations simple first, enhance iteratively
- Test all changes before committing
- Use 3-5 minute timeout when executing `neo` commands
- Semantic memory: Local embeddings (Jina 768-dim) preferred over OpenAI (1536-dim)
- When creating a pull request, always use the PR template included in the repo.
- **Before changing a subsystem, read its notes** in `docs/engineering-notes/`.
  They hold the reasons, measurements and footguns behind every rule below;
  most rules exist because the obvious alternative was shipped and broke.
  Add new long-form notes THERE, not here — this file stays short.

## Subsystem rules (details: `docs/engineering-notes/<file>`)

### Memory store — `memory-store.md`
- Per-scope valid-fact caps (`SCOPE_LIMITS`): global 200, org 100, project 500,
  session 50. Supersession at cosine ≥ 0.85 (`SUPERSESSION_THRESHOLD`);
  pre-write dedup is canonical-signature EQUALITY, not cosine.
- Promotion to a durable PATTERN needs ≥2 git-verified ACCEPTED episodes sharing
  a correlation signature AND spanning ≥2 distinct sittings/base revisions
  (`applied_on_revision`, `FactStore._sitting_bases` union-find). Fails closed on
  a blank revision. **Never reintroduce a session-id fallback.** Any new producer
  of ACCEPTED outcomes must set `applied_on_revision` or it never promotes.
- Correlation signature = candidate SUBJECT + structural fingerprint
  (`_suggestion_fingerprint`, name-normalized inside the skeleton extractor),
  never the LM body. Frozen into `Fact.canonical_signature` at mint; teardown
  uses the frozen value only. Don't conflate `_episode_signature` /
  `_global_signature` with `_canonical_signature` (dedup, includes body).
- Only `pattern` kind promotes (`algorithm`/`bugfix` task types). Task type comes
  from `classify_task_type`; ties fail safe to non-promotable kinds.
- `suggestion_is_verifiable` must resolve paths via `normalize_suggestion_path`;
  it deliberately over-admits (kind is frozen at mint). `learning-stats`
  classification is reporting-only and stricter; keep its check ordering.
- REVIEW→PATTERN similarity synthesis was REMOVED; do not reintroduce an
  unverified clustering route.
- `_invalidate` is the single invalidation choke point and strips embeddings +
  bulk text (keeps subject/tags/metadata). Cold start runs prune → demote →
  purge → strip with `save=False` and one save.
- `project_id` = hash of the normalized git remote URL (path hash fallback).
- Retrieval rank formula lives only in `memory.models.rank_score`; audit
  `ContextAssembler._score_facts` when changing it. Four-layer context assembly
  must preserve the StateBench decision-accuracy contract.
- GitHubPRSource facts enter as REVIEW on probation and never promote by recurrence.

### Learning loop — `learning-loop.md`
- Pending sessions are RETAINED per suggested path, not cleared; the session log
  is merge-on-write under `scope_file_lock`. `replay_linked_feedback` must use
  `consume_sessions_keeping_pending()`; never reintroduce `_clear_session_log`.
- UNVERIFIED mutates nothing. New `OutcomeType` → update both `outcomes.py` and
  `store.detect_implicit_feedback`.
- Linked facts resolve via `find_durable_fact_for_candidate` against the FROZEN
  `canonical_signature`; never let an empty target match.
- **Every credit path must reach a `save()`**, with a test that reloads from disk.
- Legacy per-suggestion facts are invalidated on cold start; do NOT reclassify
  them to a decaying kind instead.
- Re-accepted durable patterns are reinforced in place; `durable` is terminal;
  the protection boost is capped by evidence. Contribution gates are
  single-sourced in `store.py`.
- Learning benchmark: `accepted` is correctness only — never gate it, or any
  test, on wall-clock time; pin budgets, never mock the clock.

### Measuring retrieval — `retrieval-evaluation.md`
- Harness is `tools/rank_mine_eval.py` (not `rank_eval.py`); measure with
  `--no-git`; pin `PYTHONPATH` to the tree under test; measure the real CLI, not
  an in-process replica. Use recall over the DELIVERED set for cap questions,
  never R@10. `--max-files` default 30 stays.

### Observer — `observer.md`
- Single global CAR agent sweeping all projects; autostarts when `car-server` is
  reachable (`NEO_OBSERVER_AUTOSTART=0` opts out — set it in `~/.zshenv`).
- Drain marks (not watermark mtime) gate re-parsing. An LM failure is not "no
  lessons" (`LMUnavailable`); never write off episodes on provider outages.
- Release the single-instance lock BEFORE re-exec. Interpreter must not live
  under a world-writable dir. Orphans are reaped; status cross-checks the lock
  when CAR's view disagrees.
- Resident embedder serves the model over `~/.neo/run/embed.sock`; model/dim/
  library versions must match the client exactly.

### Dry run — `dry-run.md`
- `neo --dry-run` runs the real engine and records the exact prompt via
  `RecordingLM`; it forces the panel off, never touches the CAR daemon, and must
  not modify the fact store (`FactStore(read_only=True)`). Both exits go through
  `cli._report_dry_run`. Use it before believing any claim about what Neo saw.

### Indexing — `indexing.md`
- Budgets are apportioned, never sliced; selection orders source before tests.
- Eligibility lives ONLY in `neo/eligibility.py` (pinned by invariant tests).
- A cap that fired must be reported, and never blame a cap for an absence it
  did not cause.
- Tree-sitter queries fail silently; every new query needs a behavioural test.
- Walk cache stores verdicts keyed on dir mtime+ctime; never cache sizes/mtimes.
  Content index (SQLite): `OperationalError` must be caught BEFORE
  `DatabaseError`; warm calls open no write transaction.

### Context selection — `context-selection.md`
- Files ranked by BM25 over CONTENT; the re-rank is load-bearing. Prompt-named
  paths are PINNED (≤ `PIN_BUDGET_SHARE` of bytes). One front door
  (`gather_context`); `--semantic` is only a weight hint.
- Every prompt-side cut must be marked; use `neo.text_budget`
  (`truncate_marked` / `elide_middle` / `shown_of`), never a slice, and never
  nest cuts. `_render_context_files` returns `(sections, banner, visible)` —
  feed `visible` to `code_smells.scan_files`.

### Hosts and plugins — `host-integration.md`
- Neo reports facts; hosts decide presentation. `--json`: stdout one document,
  stderr JSONL events. All user-facing wording lives in `neo_matrix.yaml`, not
  `engine.py`.
- Import-time `Path.home()` constants must be registered in
  `conftest.HOME_PATH_CONSTANTS`.
- `NeoEngine.process()` is one-request-at-a-time (`EngineBusyError`).
- Claude Code components live at the plugin ROOT; verify with
  `claude plugin list`, not `details`; never redeclare `hooks/hooks.json`.
- `neo hook record` never fails, stays cheap, records paths not contents.
- Claude / Codex / Cursor adapters must stay in parity
  (`tests/test_host_adapter_parity.py`). Version is single-sourced in
  `pyproject.toml` (`make sync-version`). The ruff invocation must stay identical
  across Makefile, CI and pre-commit.

### Providers and CAR — `providers.md`
- CAR router ids are pinned via `model_id` by provenance
  (`model_is_catalog_id=True`), verified by equality; typed names stay on
  `model`. No lenient fallback.
- No adapter branches on model names; optional params are learned via
  `_create_resilient`, persisted only after the adapted call succeeds.
  `OpenAIAdapter` uses the responses endpoint only.
