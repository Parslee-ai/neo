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
- Memory hygiene:
  - Per-scope valid-fact caps (`SCOPE_LIMITS` in `store.py`): global=200, org=100,
    project=500, session=50. Enforced per loaded scope set (project+org+global);
    invalidated facts persist as tombstones until `purge_dead_facts` runs.
  - Supersession & pre-write canonical-signature dedup at cosine ≥ 0.85
    (`SYNTHESIS_SIMILARITY`, `memory.generalize`).
  - REVIEW → PATTERN/FAILURE synthesis needs ≥20 valid REVIEWs and ≥3-member clusters;
    triple-trigger gate fires when ANY of: count-delta ≥10, elapsed ≥1h, or
    confidence-decile entropy >0.9.
  - Probation: new non-curated facts enter with a `probation` tag and a 3-day stale window
    (vs 7/14); promoted automatically on access_count ≥2 or success_count >0.
  - Independent-outcome facts capped at 5/session (`MAX_INDEPENDENT_OUTCOMES` in
    `outcomes.py`) and 50/project (`MAX_INDEPENDENT_FACTS` in `store.py`).
  - `prune_stale_facts` → `demote_unhelpful_facts` → `purge_dead_facts` run on every
    cold start (`store.py:190-192`). For on-demand compaction of tombstone bloat in a
    specific project's fact file, use `neo memory prune [--all] [--dry-run]`
    (`subcommands.py:_compact_fact_file`).
- Domain tags (`Fact.domain`, `memory.models.SUGGESTED_DOMAINS`): optional free-form
  area tag orthogonal to `FactKind` — `code-style`, `testing`, `git`, `debugging`,
  `workflow`, `security`, `file-patterns`, `architecture`, `performance` are the
  suggested vocabulary, but any string is valid. `retrieve_relevant(..., domain=...)`
  filters by exact match; `domain=None` returns all facts including unset ones.
- Outcomes (`memory.outcomes` + `store.detect_implicit_feedback`, ~`store.py:806-900`):
  ACCEPTED/MODIFIED/UNVERIFIED act on the linked original fact when present —
  confidence +0.2 / −0.2 / +0.1 (all ±arch_mod), and bump `success_count` (except
  MODIFIED). MODIFIED also writes a REVIEW at confidence 0.4; ACCEPTED falls back to a
  REVIEW (`suggestion_confidence + 0.1`) when no link is found; UNVERIFIED never creates
  a REVIEW. INDEPENDENT writes a REVIEW at confidence 0.2. **Footgun**: if you add a new
  `OutcomeType`, update both `outcomes.py` and `store.detect_implicit_feedback`.
- Retrieval: `rank_score = recall_decay(sim)·confidence + success_bonus·effectiveness_f
  + provenance_bonus`. `memory.models.rank_score` is the single source of truth — if you
  change the formula, audit `ContextAssembler._score_facts` too. Cosine is batched via
  `math_utils.batched_cosine`. Hybrid: 0.7·dense + 0.3·BM25; half the result slots ranked
  by `rank_score`, half by raw cosine. CONSTRAINT/ARCHITECTURE/DECISION and the
  `seed`/`community`/`synthesized` tags bypass decay. Branching prompts (CHAIN/SPLIT)
  get per-branch retrieval via `memory.query_routing`; each surfaced EPISODE pulls up to
  2 peer episodes from the same session.
- Local storage: per-scope JSON files in `~/.neo/facts/` with inline embeddings. Fine
  while any single scope file stays under ~10k facts; revisit the backend past that.
  `project_id` is `SHA256[:16]` of the **normalized git remote URL** (`scope._compute_project_id`)
  so the same repo on different clones / worktrees / machines hashes to the same ID.
  Falls back to a path hash for repos without a remote. Legacy path-hashed fact and
  watermark files are renamed in place on `FactStore` init
  (`store._migrate_legacy_project_id_files`).
- Context assembly four-layer model is from *Beyond Conversation: A State-Based Context
  Architecture for Enterprise AI Agents* (Liotta, 2025); the `ContextAssembler` token-budget
  enforcement is ported from *Memgine: A Deterministic Memory Engine for Stateful AI Agents*
  (Liotta, 2026). Both PDFs:
  [state-based-context-architecture](https://github.com/Parslee-ai/statebench/blob/main/docs/state-based-context-architecture.pdf)
  and
  [memgine-deterministic-memory-engine](https://github.com/Parslee-ai/statebench/blob/main/docs/memgine-deterministic-memory-engine.pdf).
  Both are evaluated by [StateBench](https://github.com/parslee-ai/statebench).
  Changes to layer ordering, the 2/3 constraint cap, or the inline `(changed from: X)`
  annotation should preserve the validated 95.8% decision-accuracy contract (GPT-5.2 on
  the v1.0 development split). See `docs/solutions/token-budget-enforcement.md`.
- Async synthesis observer (`memory.observer`): a per-project background process
  that runs `synthesize_reviews` on a wall-clock cadence, decoupled from the
  request path. *Additive* — the inline triple-trigger gate keeps firing too;
  the observer just makes synthesis more frequent. **Hard dep**: car-runtime
  ≥ 0.16.1 (for the `agents_*` lifecycle API) and a running `car-server`
  daemon — CAR's supervisor owns the spawn / restart-on-failure / log
  redirection / clean SIGTERM shutdown. Spec persisted to `~/.car/agents.json`
  (`auto_start: true` so it comes back on daemon boot); logs land at
  `~/.car/logs/neo-observer-<id8>.{stdout,stderr}.log`. Lifecycle:
  `neo memory observer {start|stop|status|kick}` — `kick` maps to
  `agents_restart` since CAR has no signal-passthrough primitive.
  Tunables: `NEO_OBSERVER_INTERVAL_SECONDS` (default 300),
  `NEO_OBSERVER_COOLDOWN` (default 60, per-process).
  **Known broken** on machines where CarHost.app is running:
  `car-runtime 0.16.x`'s module-level `agents_upsert` is still in-process
  and collides with the running supervisor's manifest lock — see
  [Parslee-ai/car-releases#54](https://github.com/Parslee-ai/car-releases/issues/54).
  The unit tests pass because they mock `car_runtime`; do not interpret
  green CI as "observer works". Revisit once #54 lands a wheel that
  routes `agents_*` over WS.
- Observability: retrieve / add_fact / lm_call / overseer_tick events land in
  `~/.neo/metrics.jsonl`. Gated by `NEO_PROFILE`:
  `off` (no emit), `minimal` (lm_call only), `standard` (default, all events),
  `strict` (reserved for future verbose events; currently == standard).
  `NEO_METRICS=off` is a legacy hard kill-switch that overrides `NEO_PROFILE`.
  Sessions and watermarks live in `~/.neo/sessions/`.
- Debugging: `neo --dry-run "your query"` assembles the full context (file selection,
  fact retrieval, constraints, four-layer assembly) and prints what *would* be sent to
  the LM, then exits without making the LLM call. Faster iteration on context-gatherer
  and retrieval changes than waiting for an inference round trip.
- CarAdapter defaults `intent_hint={"task":"code"}` so CAR's router picks a code-capable
  model rather than the chat default. This is the local workaround for
  [Parslee-ai/car-releases#52](https://github.com/Parslee-ai/car-releases/issues/52)
  (`route_model` is cost-biased for "simple" prompts and ranks `gpt-5.3-codex`/`o3`
  behind `gpt-4.1-mini`). If that upstream lands, revisit the default.
- When creating a pull request, always use the PR template included in the repo.