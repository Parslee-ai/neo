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
  - Invalidation choke point: `_invalidate(fact, *, cascade=True)` is the single
    path that sets `is_valid=False`. It **strips the 768-dim embedding at the
    transition** — a tombstone is never retrieved/deduped/clustered (all such
    paths pre-filter `is_valid`) but is retained up to 30 days for
    supersession/audit, so its embedding (~24 KB/fact) is immediate dead weight;
    stripping at the source keeps bloat from accumulating between sweeps. All six
    FactStore invalidation sites route through it (eviction, prune, demote,
    `_cap_independent_facts` with `cascade=False`, `_supersede`,
    `_synthesize_cluster`); `superseded_by`/`event_time_end` stay at the call
    site. Safe because invalidation is terminal (merge-on-save returns OURS when
    we hold it invalid); no current command re-embeds existing facts
    (`--regenerate-embeddings` targets the legacy ReasoningMemory cache), so the
    strip is one-way in practice.
  - `prune_stale_facts` → `demote_unhelpful_facts` → `purge_dead_facts` →
    `strip_tombstone_embeddings` run on every cold start (`store.py:190-192`),
    each taking `save=False` so the chain flushes **one** merge-on-save instead
    of four. `strip_tombstone_embeddings` is now a **backfill** — it only catches
    tombstones minted off the `_invalidate` path (an ingester superseding a fact;
    a peer process's still-embedded copy reconciled in) plus any legacy
    pre-strip rows; it self-heals across processes since every cold start /
    `detect_implicit_feedback` re-runs it. For on-demand compaction of tombstone
    bloat in a specific project's fact file, use `neo memory prune [--all]
    [--dry-run]` (`neo/subcommands.py:_compact_fact_file` — at the package root,
    not under `memory/`); it both drops 30-day-cold invalid rows and strips
    embeddings off the retained (<30-day) tombstones (reports `removed` +
    `stripped`), under the shared `scope_file_lock` so it can't clobber a
    concurrent observer/request-path `save()`.
  - `neo memory replay-feedback [--all] [--dry-run] [--include-legacy-fallback]
    [--limit N]` re-processes linked session outcomes (ACCEPTED/MODIFIED/UNVERIFIED)
    to update the linked facts' confidence + `success_count` — a manual re-run of
    the implicit-feedback pass, for after a memory-loop fix (`store.replay_linked_feedback`).
    `--dry-run` reports what would change without mutating; `--include-legacy-fallback`
    also inspects legacy `session_*.json` files (may re-replay already-processed
    sessions). Only touches linked, non-independent outcomes.
  - Diagnostics (read-only, flag-and-propose): `neo memory issues [--since 14d]
    [--min-cluster 3] [--suggest-rules] [--json]` surfaces recurring frictions mined from
    transcript history (Claude Code / Codex / CAR) as ranked, evidence-cited issues
    (`missing-tool` / `absent-guardrail` / `vague-rule`); `--suggest-rules` adds a bounded
    LM call per issue to draft a preventive rule. `neo memory rules [--json]
    [--no-conflicts]` flags drift between AGENTS.md / CLAUDE.md / GEMINI.md (gaps +
    LM-judged conflicts). `neo memory audit [--json] [--no-conflicts]` inspects an AI
    tool's memory files (Claude Code `memory/*.md`) for malformed entries, near-duplicates,
    conflicts, and MEMORY.md index drift. `neo memory import [--dry-run]` ingests a peer
    tool's memory files into neo's store as REVIEW facts on probation (trust-first;
    `imported:claude-memory` tag, content-hash watermark for idempotency).
    (`neo/memory/issues.py`, `neo/memory/rulesync.py`, `neo/memory/memaudit.py`,
    `neo/memory/memimport.py`)
    `neo memory citation-stats [--since 7d] [--json]` summarizes the
    `citation_survival` metric from `~/.neo/metrics.jsonl` — retrieved/included/used
    counts plus the per-signal split (`by_marker` / `by_self_report` / `by_overlap`)
    showing WHICH detector earns the retrieved-fact use-credit. Use it to decide
    whether the reliable structured self-report carries the reinforcement path or
    the softer subject-overlap heuristic is doing the work (and thus whether to
    keep/tune/drop overlap). Read-only, no LM call (`subcommands._handle_citation_stats`).
    `neo memory learning-stats [--since 7d] [--json]` is the promote-side pulse:
    it reads the episode ledger (`~/.neo/episodes`, no LM, no fact-store scan) and
    reports episodes, final outcomes, candidate statuses (durable / supported_once /
    contradicted / rejected_by_verification / …), and learning actions (promotions,
    rollbacks, demotions, reinforcements incl. cited-fact credit) from the ledger
    mutations. Scoped to the INTERACTIVE / attributed path: an IDLE reading means
    the accept-driven loop is quiet (suggestions not accepted downstream), NOT
    that neo isn't learning — the background promote engine (observer
    `synthesize_reviews`, transcript/GitHub-PR mining) mints facts with no episode
    footprint and is deliberately not counted here. Together with citation-stats it
    forms an "is it learning?" dashboard (`subcommands._handle_learning_stats`).
    - `issues` reuses the ingester's `TranscriptSource` episodes but never admits facts or
      touches the `transcript_watermark_*` watermark — decoupled from fact admission and
      idempotent (`find_issues`). Gate mirrors synthesis discipline (≥`min_cluster`
      members, ≥2 sessions, ≥2 frictional, verbatim evidence); clusters at
      `SYNTHESIS_SIMILARITY` via the shared `math_utils.cluster_by_similarity`. See
      `docs/solutions/conversation-mined-issues.md` and `docs/solutions/rule-file-sync.md`.
- Transcript sources (`memory.transcript`, the `TranscriptSource` Protocol): the
  `TranscriptIngester` mines lessons from four sources by default —
  `ClaudeCodeSource` (`~/.claude/projects/**/*.jsonl`), `CodexSource`
  (`~/.codex/sessions/**/rollout-*.jsonl`), `CarSource` (`~/.car/sessions/*.json`),
  and `GitHubPRSource` (merged PRs + review threads via the `gh` CLI). A source may
  declare optional `fact_kind` / `extra_tags` trust overrides that the ingester's
  `admit` reads (default = today's PATTERN/FAILURE + `transcript-derived` tag).
  `GitHubPRSource`: PROJECT-scoped (owner/repo derived from the git remote, so PR
  facts co-scope with that repo's transcript facts under the same `project_id`);
  enters facts as **REVIEW on probation** (`imported:github-pr` tag) — trust-first,
  and NOT promoted by recurrence (synthesis keys it `"other"` → stays REVIEW; only an
  independent git-verified acceptance ever mints PATTERN). Mine-once (watermark keyed
  on PR number, bounded); maps title+body→`ask`, reviews/comments/inline-thread
  comments→`assistant_text`, `CHANGES_REQUESTED`→`errors`; filters bot authors;
  skips PRs with no human discussion. Throttled to one `gh` fetch per repo per
  `_GH_PR_FETCH_INTERVAL` (3600s) so the all-projects sweep keeps near-zero work on
  unchanged repos. Self-disables (returns `[]`) when the remote isn't github.com or
  `gh` is absent — no env flag. Known limits (deferred): merged-only, no PR-diff
  ingestion (discussion text only), no historical backfill beyond the
  `_GH_PR_PAGE`(=25)-most-recently-updated window, GitHub Enterprise hosts and
  fork-origin upstreams not handled.
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
- A2UI memory inspector (`neo.a2ui`): a per-project A2UI v0.9 surface
  (`neo-<project_id8>`) registered with the running `car-server` daemon so any
  conformant renderer (CarHost.app, future webviews) can inspect neo's state
  live. Two tabs: **Observer** (status badge, pid, last cycle, recent cycles
  list, Kick/Stop buttons) and **Memory** (valid fact count, by kind, by scope,
  probation count). Updates pushed by the observer process at the end of each
  synthesis cycle — the same FactStore load powers both tabs, so the
  inspector adds zero hot-path cost. Kick/Stop buttons emit `a2ui.action`
  notifications which the observer dispatches to `kick_observer` /
  `stop_observer` — closes the loop with CAR's supervisor. **Footgun**:
  Python's `car_runtime.a2ui_*` helpers are in-process only; reaching the
  daemon's shared store (which renderers subscribe to) requires speaking
  JSON-RPC over its WebSocket. `neo.a2ui.DaemonClient` is that bridge.
  Activation: auto when `127.0.0.1:9100` is reachable; silent no-op
  otherwise. Adds `websockets>=12.0` to the `[car]` extra.
- Async synthesis observer (`memory.observer`): a **single global** background
  process (CAR agent `neo-observer`, daemon `--daemon --all`) that **sweeps all
  discovered projects** each cycle — round-robin/budgeted (`max_projects_per_cycle`,
  default 25; watermark-gated so unchanged projects do near-zero work) — running
  `synthesize_reviews` + transcript mining per project. *Additive* — the inline
  triple-trigger gate keeps firing too. **Not opt-in**: `maybe_autostart_observer()`
  (called from `cli.main`) auto-registers it whenever `car-server` is reachable;
  opt out with `NEO_OBSERVER_AUTOSTART=0`. No CAR → one-time hint, then silent.
  Projects are discovered from `~/.claude/projects/*` (decoded roots). On
  bootstrap/start, legacy **per-project** agents (`neo-observer-<id12>`, the old
  model) are stopped + `agents_remove`d. **Hard dep**: car-runtime ≥ 0.18.0
  (pin floor 0.27.0) + a running `car-server` — CAR's supervisor owns
  spawn / restart-on-failure / clean SIGTERM. Logs at
  `~/.car/logs/neo-observer.{stdout,stderr}.log`. Lifecycle/`status`/orphan-check
  all operate on the single global agent. (A2UI per-project inspector is skipped
  in global mode.)
  Lifecycle: `neo memory observer {start|stop|status|kick}` — `kick` maps to
  `agents_restart` since CAR has no signal-passthrough primitive. Status surfaces
  CAR's raw state verbatim (`running` | `stopped` | `starting` | `backoff` |
  `errored`) so restart-loops are diagnosable, and also flags **orphaned**
  observer processes — a `neo.memory.observer --daemon` reparented to
  init/launchd (`ppid==1`, or no live parent on Windows) by a dead prior
  car-server, which CAR's supervised view can't see
  (`observer._find_orphan_observers`; the `orphans` field + a `WARNING`).
  Orphans are now **auto-reaped**, not just reported: `_reap_orphan_observers`
  SIGTERMs them (re-checking each pid's cmdline right before the signal to
  defend against pid reuse) and is wired into `start`/`stop`/autostart and the
  daemon's own startup. A second guarantee backs it up — the daemon holds a
  cross-process **single-instance lock** (`_SingleInstanceLock`, `fcntl`/`msvcrt`
  on `~/.neo/observer.lock`) for its lifetime, so two observers can never run
  synthesis at once even in the handoff window; a contended daemon exits 0
  (benign no-op, no CAR backoff). If a straggler ignores SIGTERM past the
  `_LOCK_ESCALATE_AFTER` grace, the daemon escalates to SIGKILL so the kernel
  frees the lock — safe because `FactStore._save_file` is atomic (temp +
  `os.replace`), so a hard kill can only leave a stray `.tmp`, never a torn
  fact file. (This is belt-and-suspenders: `store.save()` already serializes
  writers with its own per-scope flock, so the orphan was never a corruption
  bug — just doubled LM spend and synthesis.) Tunables:
  `NEO_OBSERVER_INTERVAL_SECONDS` (default 300), `NEO_OBSERVER_COOLDOWN`
  (default 60, per-process). **Footgun**: the interpreter path (`sys.executable`)
  must not live under a world-writable directory (`/tmp`, `/private/tmp`,
  `/var/tmp`, `/dev/shm`) — the CAR daemon rejects such commands as a
  security measure. Use a venv under `$HOME` or a system install.
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
- CarAdapter defaults `intent_hint={"task":"code","prefer_quality":True}` so CAR's router
  routes neo to the most capable model, not the chat/cost default. This is the *intended*
  router API, not a hack:
  [Parslee-ai/car-releases#52](https://github.com/Parslee-ai/car-releases/issues/52)
  (router cost-bias on `task=code`) is **closed** — the router now prioritizes
  quality > speed > cost for `task=code`, and the 0.25–0.27 reworks add capability-honest
  routing + `exclude_models`. We keep `prefer_quality` explicit as belt-and-suspenders
  (CAR's *default* profile without it is still latency/cost-biased). Rationale lives on
  `CarAdapter.DEFAULT_INTENT_HINT` in `adapters.py`. Observer floor is car-runtime ≥0.18.0,
  now enforced at runtime by `_require_car_runtime` (version check, not just the `agents_*`
  attr); latest validated against car-runtime 0.27.0.
- Reasoning-model param compatibility (`adapters.py`): newer models reject standard
  chat params — Anthropic Opus 4.7+/Sonnet 5/Fable 5 reject `temperature`; OpenAI
  o-series/gpt-5, Azure reasoning deployments, and OpenAI-compatible reasoners (xAI
  Grok, DeepSeek) reject `temperature`, and the OpenAI-family require
  `max_completion_tokens` instead of `max_tokens`. There's no reliable model-string
  rule (opus-4-6 accepts `temperature`, opus-4-7 rejects it; Azure `model` is an
  arbitrary deployment name), so adapters **learn reactively**: catch the 400, drop/
  rename the param, retry, remember. The learnings persist in `_ModelParamCompat`
  (`~/.neo/model_param_compat.json`, keyed `"<provider>:<model>" → [flags]`) so the
  first-call retry penalty isn't re-paid every CLI invocation. Store is best-effort
  (I/O failure → in-memory only, never breaks inference), merge-on-write + atomic
  `os.replace`, path resolved at call time (per-test `Path.home()` stubs apply). The
  OpenAI-family adapters share `_chat_completion_resilient(client, kwargs, provider)`;
  Anthropic has its own inline learn-and-retry. **Footgun**: recovery keys on HTTP 400
  (`BadRequestError`); a provider returning 422 for a param error won't be caught.
- When creating a pull request, always use the PR template included in the repo.