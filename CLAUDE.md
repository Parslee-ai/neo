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
  - Supersession at cosine ≥ 0.85 (`SUPERSESSION_THRESHOLD`, `store.py:59`);
    pre-write dedup is canonical-signature **equality**, not cosine
    (`memory.generalize`). `SYNTHESIS_SIMILARITY` was a *separate* constant that
    happened to equal 0.85 and gated REVIEW clustering; it went away with
    `synthesize_reviews` (see below). The only 0.85 left in `store.py` is
    supersession. `memory.issues` keeps its own `CLUSTER_SIMILARITY = 0.85`,
    tunable independently.
  - Episode-derived promotion correlation (`store._episode_signature` /
    `_global_signature`): a candidate promotes to a durable fact only when ≥2
    verified-accepted episodes share a **correlation signature** AND those
    episodes span ≥2 distinct `repository_revision`s
    (`_supporting_episodes_span_distinct_revisions`). The revision requirement is
    what "independent" actually means here: distinct episode ids alone is no test
    at all, since every invocation mints a fresh one, so one operator applying the
    same patch twice minutes apart at the same HEAD promoted a durable PATTERN
    whose content was wrong (reproduced live). **A session-id fallback for non-git
    projects was written and REMOVED — do not reintroduce it.**
    `LearningEpisode.session_id` is a per-episode `uuid4()` that nothing ever
    assigns from a real session (121 episodes on a live ledger → 121 distinct
    ids), so "≥2 distinct sessions" was the very gate being replaced, renamed;
    and since acceptance detection is entirely git-based, a genuinely non-git
    project can never record an ACCEPTED outcome and could not reach the fallback
    anyway. Its only reachable trigger was a transient `rev-parse` failure, which
    made BOTH failing promote while ONE failing blocked — load-dependent
    non-determinism in the durable memory path. It now fails closed on a blank
    revision. Two accepted costs, documented on the predicate: the revision is
    captured when the episode BEGINS (HEAD when advice was asked for, not the
    commit the fix landed in), and applying one lesson across several files in a
    single sitting records ONE revision and promotes nothing (40% of
    revision-bearing episodes share a HEAD with another). Keying on the
    acceptance-carrying sha — already walked by `_get_changed_files_since` — is
    the obvious improvement. That
    signature is keyed on the candidate SUBJECT, never the body — the body is the
    LM's run-varying Reasoning/Suggestion prose, and including it (the old
    `generalize(subject+body)`) gave two acceptances of one task different
    signatures so promotion never fired (a live drill measured this). The subject
    is `f"{task_type}: {prompt[:50]} [{file_path}] [fp:{hash}]"`; the `[fp:<hash>]`
    is a structural fingerprint of the suggested change
    (`engine._suggestion_fingerprint` = sha256 of the AST-shaped
    `_extract_code_skeleton` called with `normalize_names=True`, so `def:<name>`
    becomes bare `def`, Python-only, "" for unparseable code → key degrades
    to subject-only). The name-normalization is load-bearing: the skeleton keeps
    `def:<name>` by default because it doubles as readable metadata on the fact,
    but *hashing* the name made the identical fix to `read_text` and `read_body`
    two different signatures, so a genuinely recurring lesson could never reach
    the two acceptances promotion requires. A live drill against real `neo` runs
    measured four git-verified acceptances and zero promotions; normalizing
    promoted on the next pair. It MUST happen inside the extractor, before the
    500-char truncation — `def:<name>` is the only unbounded-length token, so
    long identifiers are exactly what pushes a skeleton past the cut, and
    post-hoc string stripping still left identical shapes hashing differently.
    The rule for what survives is **bounded vocabulary**: the ~11 structural
    keywords, the 6-name method whitelist and the 9-name constructor whitelist
    all carry shape, while `def:<name>` was the only free user-chosen identifier
    (there is no `ClassDef` handler, no arg names, no bare `Name` loads) — which
    is what makes normalizing it both necessary and sufficient.
    **Honest limit**: post-normalization skeletons are coarse — the drill's fix
    hashes to `def return`, a guard-clause fix to `def if-stmt return` — so the
    fingerprint discriminates far less than "diff shape" suggests, and at the
    path-agnostic GLOBAL tier correlation is close to prompt-prefix plus a
    near-constant token. The real anti-collision protection is the double
    git-verified acceptance plus the kind gate, not the fingerprint;
    `test_structurally_distinct_fixes_do_not_collide` pins what it still buys. `_split_fingerprint` pulls the fp OUT before `generalize`
    (whose `_HASH_RE` would collapse the hex to `<hash>`) and appends it RAW after
    a `\x1f` separator, so two episodes correlate only when prompt-prefix AND
    diff-shape agree — read "diff-shape" with the coarseness caveat above. **Two tiers**: PROJECT uses `_episode_signature`
    (path-bearing, though `generalize` collapses deep multi-segment paths so it
    only discriminates SHALLOW `[file.py]` names); cross-project GLOBAL uses
    `_global_signature` (path-agnostic — strips all bracket qualifiers to match
    the bracket-stripped text `_mint_global_fact` stores, so the same lesson
    correlates across repos). The signature is **frozen** into
    `Fact.canonical_signature` at mint; all rollback/dedup/teardown sites use the
    frozen value only (never recompute a global fact's transformed subject).
    **Deliberate invariant**: single-project rollback keys on the path-bearing
    project signature and therefore can NEVER hard-retract a path-agnostic global
    fact — global teardown requires cross-project contradiction and is owned by
    `reconcile_cross_project_promotions`. Rollback-resolve is fingerprint-precise
    (a differently-shaped correction won't hard-retract; soft confidence demotion
    still applies). **Footgun**: `_episode_signature`/`_global_signature` (episode
    correlation, subject+fp) are distinct from `_canonical_signature` (pre-write
    exact-twin dedup, which DOES include body+kind+scope) — one keystroke apart,
    do not conflate.
  - Candidate KIND gate + task-type classification (`models.classify_task_type`,
    engine `kind_map`): a candidate promotes only when its kind is `pattern`. Kind
    derives from the task type — `algorithm`/`bugfix` → `pattern` (promotable);
    `feature` → `decision`, `refactor` → `architecture`, `explanation` → `review`,
    and the unknown-type default → `review` (all non-promotable, by design — auto-
    minting durable decisions/architecture/prose is riskier than patterns). The CLI
    used to hardcode `task_type=FEATURE` for every plain-text prompt, so nothing
    interactive could ever promote; it now calls `classify_task_type(prompt)` —
    deterministic keyword scoring (no LLM), highest distinct-match count wins, ties
    break by `_TASK_TYPE_PRIORITY` which is ordered so ALL non-promotable kinds
    precede the two promotable ones (ties FAIL SAFE to non-promotable; EXPLANATION
    must stay ahead of BUGFIX/ALGORITHM). No signal → `FEATURE`. JSON callers'
    explicit `task_type` still wins; only an omitted one is classified.
    **Known limitation** (contained by the double-acceptance + fingerprint promotion
    gate, not the kind map): a lone/multi promotable-noun in feature/explanation
    prose with no competing feature/refactor verb can win by score and raise
    *eligibility* (e.g. "improve the errors page" → BUGFIX; "summarize the
    performance of this algorithm" → ALGORITHM) — it still can't mint a durable fact
    without two independent git-verified acceptances of a matching diff shape.
    `classify_task_type(prompt, error_trace=None)` also takes an optional
    `error_trace` (wired on the JSON path): a supplied traceback adds
    `_FAILURE_TRACE_WEIGHT` (=2) to BUGFIX — strong but overridable (a 3-signal
    dominant intent still wins; a signal-less/empty prompt + trace → BUGFIX). The
    BUGFIX failure-symptom patterns are DERIVED from the shared
    `execution_context.FAILURE_SIGNAL_KEYWORDS` (`error/fail/exception/crash`) so
    this classifier and `_infer_intent` can't drift; `_infer_goal` keeps its own
    timeout-inclusive set by design. (If a third module needs the lexicon, extract
    it to a neutral `neo.lexicon` then.) The derived `\bcrash(?:…)?\b` is
    boundary-closed, so compound terms like `crashloop` no longer match — a
    deliberate precision/recall trade in the fail-safe direction.
  - **REVIEW → PATTERN synthesis has been REMOVED** (`synthesize_reviews` and
    its cluster/watermark/Hebbian machinery). It ran for four months and minted
    114 facts, **none of them a PATTERN** — the PATTERN branch required a
    `group_key == "outcome:accepted"` and a census found 0 accepted-tagged
    REVIEWs (against 97 `independent`, 3046 `history`), because
    `detect_implicit_feedback` boosts the linked fact or supports an episode
    candidate and both return before the fallback that would carry the tag.
    Meanwhile it re-consumed its own summaries as fresh evidence (68 synthesized
    vs 29 raw), and every run multiplied the whole corpus by 0.97. A live census
    also showed zero ≥3-member clusters at cosine 0.85 across all 1152 valid
    REVIEWs, so it could not fire on real data anyway. The git-verified episode
    ledger (`_promote_repeatedly_supported_candidate`, ≥2 verified acceptances
    spanning ≥2 revisions) is the learning path that remains — do not reintroduce an
    unverified similarity-clustered route beside it. Facts minted before the
    removal keep their `synthesized` tag and prune immunity; nothing mints it
    now. `SUPERSESSION_THRESHOLD` (0.85) and canonical-signature dedup are
    untouched — they were always separate from the deleted
    `SYNTHESIS_SIMILARITY`.
  - **Candidate verifiability gate** (`memory.outcomes.suggestion_is_verifiable`,
    applied in `engine._store_reasoning`): a candidate is minted with its
    task-type kind only when a downstream git diff could ever confirm it — the
    path must be one attribution could name AND there must be code/diff text to
    compare against. Otherwise it is downgraded to non-promotable `review`.
    Promotion is gated on git-verified acceptance, so an unverifiable suggestion
    was previously minted `pattern` and then sat pending forever. Measured live:
    only 23 of 65 recorded suggestions were verifiable at all; the rest are
    advisory prompts where the model answers with a topical pseudo-path
    (`/review/commit-<sha>.md`) instead of an edit target. **Footgun**: this
    predicate MUST resolve paths via `normalize_suggestion_path` — the same
    helper `OutcomeTracker._normalize_path` delegates to. An independent copy
    missed bare-leading-slash paths (`/src/foo.js`, which normalize to
    repo-relative) and wrongly rejected two genuinely promotable candidates.
    A not-yet-existing path still qualifies when its parent dir is inside the
    repo — proposing a NEW file is legitimate and shows up in `git log` once
    committed. Known limit: a bare-slash name at the repo ROOT
    (`/NO_CODE_PLANNING_ONLY` vs `/README.md`) is structurally indistinguishable
    from a real new file and is admitted. That asymmetry is deliberate —
    `kind` is frozen at mint, so under-admitting permanently kills a real
    lesson, while over-admitting only leaves a candidate pending.
    `neo memory learning-stats` buckets recorded suggestions FOUR ways
    (`subcommands._classify_suggestion`): `verifiable`, `advisory`,
    `unattributable` (the bug signal) and `root_unavailable`. The fourth exists
    because every resolution test runs against the LIVE filesystem, so a recorded
    `codebase_root` that no longer exists — mostly deleted Claude Code agent
    worktrees — makes them all meaningless; counted as unattributable those
    buried the one number the report exists to make actionable. **Ordering rule**:
    only the empty-input test is decidable from the arguments alone, so the root
    check runs immediately after it and every filesystem-dependent branch runs
    below. Placing the prose-suffix/sentinel/`docs/` branches above it looked
    string-decidable and was not — under a dead root all three degrade to "does
    not exist" and silently return advisory, which under-reported the integrity
    signal by 43% (8 against 14 real) and inflated `measurable` by the difference.
    Verdicts (ACTIVE / UNMEASURABLE / STARVED / IDLE) are computed over
    `measurable = total - root_unavailable`, never against a content bucket: the
    first version compared `root_unavailable > verifiable`, so a single dead root
    suppressed STARVED entirely and claimed "most recorded suggestions" off a
    count of one. An integrity note discloses the unmeasured share in EVERY
    branch, ACTIVE included. The bug signal is keyed on `_SOURCE_SUFFIXES`, not on
    a leading slash — slash-keying split `/review/x.json` from `review/x.json` on
    LM formatting alone and buried two real defects (a model-emitted
    `<placeholder>` segment and a new module in a new directory). This classifier
    is **reporting-only and deliberately stricter than
    `suggestion_is_verifiable`**, which stays untouched: its bias toward admitting
    a doubtful path is correct because `kind` is frozen at mint, while
    over-admitting in a report costs only an inaccurate number. Measured: the
    prose-suffix-before-verifiable ordering corrected `verifiable` from 21 to 10,
    11 of which were invented review docs (`/ARCHITECTURAL_REVIEW.md`) that the
    plausible-new-file rule had granted because their parent IS the repo root.
  - Probation: new non-curated facts enter with a `probation` tag and a 3-day stale window
    (vs 7/14); promoted automatically on access_count ≥2 or success_count >0.
  - Independent-outcome facts capped at 5/session (`MAX_INDEPENDENT_OUTCOMES` in
    `outcomes.py`) and 50/project (`MAX_INDEPENDENT_FACTS` in `store.py`).
  - Invalidation choke point: `_invalidate(fact, *, cascade=True)` is the single
    path that sets `is_valid=False`. It **strips the 768-dim embedding AND the
    bulk text (`body`, `context_text`, `retrieval_text`) at the transition**
    (`_strip_tombstone_text`; measured 11,421 tombstones holding 24.7 MB, of
    which `body` alone was 15.8 MB). `subject`/`tags` stay for audit and
    `metadata` stays because `purge_dead_facts` ages tombstones off
    `metadata.last_accessed` and reads `invalidation_reason` — dropping those
    would strand tombstones forever. `episode_context` is deliberately NOT
    stripped: it is a structured `EpisodeContext` with its own `to_dict`, and
    blanking it to `""` breaks serialization (caught against a copy of a real
    store). Safe because nothing reads a tombstone's text: dedup skips invalid
    facts (`_exact_canonical_match`), merge-on-save returns early for them
    (`_reconcile_fact`), retrieval and clustering pre-filter `is_valid`. The
    older wording said only the embedding — a tombstone is never retrieved/deduped/clustered (all such
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
    `strip_tombstone_embeddings` run on every cold start (in `FactStore.initialize`),
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
    transcript/GitHub-PR mining) mints facts with no episode footprint and is
    deliberately not counted here. Together with citation-stats it
    forms an "is it learning?" dashboard (`subcommands._handle_learning_stats`).
    - `issues` reuses the ingester's `TranscriptSource` episodes but never admits facts or
      touches the `transcript_watermark_*` watermark — decoupled from fact admission and
      idempotent (`find_issues`). Gate mirrors the old synthesis discipline (≥`min_cluster`
      members, ≥2 sessions, ≥2 frictional, verbatim evidence); clusters at
      `issues.CLUSTER_SIMILARITY` via the shared `math_utils.cluster_by_similarity`. See
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
  and NOT promoted by recurrence (only an independent git-verified acceptance
  ever mints PATTERN). Mine-once (watermark keyed
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
- **Pending sessions are RETAINED, not cleared** (`outcomes.collect_outcomes`).
  **Pendingness is per SUGGESTED PATH, not per session** — this is the whole
  subtlety. `_get_changed_files_since` returns every file changed anywhere in
  the repo, so the first version's `if not changed_files: pending.append(...)`
  only retained anything in a repo with no commits AND a spotless working tree
  since the suggestion. One unrelated dirty file dropped the session and lost
  the acceptance exactly as before the fix; every test passed because they all
  ran on a pristine tree, the one state neo is never invoked in.
  `_unresolved_suggestions` now keeps a **reduced** record holding only the
  suggestions still outstanding — git-trackable paths git hasn't touched yet.
  Resolved paths are removed so they can't re-emit, and review-only paths are
  removed because their weak UNVERIFIED already fired (retaining them re-emitted
  it every invocation, growing `VerificationEvidence` per episode without bound).
  Anything past `PENDING_SESSION_TTL_SECONDS` (14d) is dropped so the log stays
  bounded. `_get_working_tree_changes` is hoisted OUT of the per-session loop:
  it's timestamp-independent, and re-forking it per retained session measured
  0.88s of pure `git` forking at 40 pending sessions, on the request hot path,
  growing linearly.
  **The session log is merge-on-write, not last-writer-wins.**
  `_rewrite_session_log` takes `store.scope_file_lock`, then RE-READS the log
  under it and preserves every record whose `_session_key` is absent from
  `_last_loaded_keys` — i.e. appended by a peer since our read. Locking the
  write alone was NOT enough and the loss was reproduced: a second neo process
  saving between one process's read and its `os.replace` was erased without
  trace. `save_session` takes the same lock for its append. Merge-on-write
  rather than a wider lock on purpose — holding the lock across the git and LM
  work in `collect_outcomes`/`replay_linked_feedback` would let a slow
  reasoning run block another process's save.
  **`replay_linked_feedback` must use `consume_sessions_keeping_pending()`**,
  never a wholesale delete — it is the documented repair command for a broken
  memory loop, so nuking the log there destroyed every pending suggestion the
  moment you ran it. `_clear_session_log` is deleted; do not reintroduce it. This used
  to clear the WHOLE log whenever any session existed — so any neo invocation
  between "neo suggests X" and "user applies X" silently destroyed the pending
  suggestion. The multi-session read in `collect_outcomes` exists to prevent
  exactly that loss and the unconditional clear defeated it one level up.
  **Measured**: 30d of real traffic = 108 episodes, 58 stuck at
  `suggested_pending_downstream_outcome`, **zero `accepted` outcomes ever** —
  so the promote path (needs 2 git-verified acceptances) could never fire no
  matter how correct its own gates were. The July drill only worked because the
  operator applied the diff with no intervening run. Retention **rewrites** the
  log (`_rewrite_session_log`, temp + `os.replace`) rather than appending, or
  one suggestion would accrue a copy per invocation and bump its fact once per
  copy. `_is_non_git_trackable` is shared by the weak-acceptance detector and
  the retention rule so they can't disagree about what's still worth waiting on.
  **Test note**: `outcomes.SESSIONS_DIR` is resolved at IMPORT time from
  `Path.home()`. conftest now re-points it (and every other import-time home
  constant) at the fake home per test — see the home-isolation entry below —
  but tests sharing a fixed `project_id` still read each other's session logs
  within a run, so scope by a unique id.
- Outcomes (`memory.outcomes` + `store.detect_implicit_feedback`):
  ACCEPTED/MODIFIED act on the linked original fact when present — confidence
  +0.2 / −0.2 (both ±arch_mod); ACCEPTED also bumps `success_count` and sets
  effectiveness "better", MODIFIED sets "worse". **UNVERIFIED mutates nothing**:
  absence of verification is not success, so the evidence is preserved in the
  learning episode (candidate status → `unverified`) and neither confidence nor
  `success_count` moves — the live path and `replay_linked_feedback` share that
  invariant (`store.py:2148`, `store.py:2301`). MODIFIED also writes a REVIEW at
  confidence 0.4; ACCEPTED falls back to a REVIEW (`suggestion_confidence + 0.1`)
  when no link is found; UNVERIFIED never creates a REVIEW. INDEPENDENT writes a
  REVIEW at confidence 0.2. **Footgun**: if you add a new `OutcomeType`, update
  both `outcomes.py` and `store.detect_implicit_feedback`.
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
- Learning-loop benchmark (`memory/evaluation.py`, `benchmarks/learning_loop_v1.json`,
  `neo memory evaluate-learning`). **`accepted` is a correctness verdict and nothing
  else — never gate it on wall-clock time.** Everything it enforces is reproducible on
  any machine: twelve scenarios, four safety rates that evaluate to exactly `0.0`, the
  primary-metric comparison against the memory-disabled baseline, and a zero
  model-call assertion. Latency is a property of the hardware, so it lives in a
  separate `performance_budget` block and surfaces as `performance_within_budget` /
  `performance_notes`, advisory, with no effect on the exit code. It used to sit in
  `safety_thresholds`: a GitHub runner recorded **592.44ms against the 500ms limit with
  every scenario passing and every rate at zero**, while the same commit ran at ~53ms
  locally — an 11× spread with no code difference, so no threshold can be both
  sensitive enough to catch a regression and loose enough to survive a shared runner.
  Retuning the number only moves the coin-flip. Worse, `accepted` is the benchmark's
  *published* verdict, so a timing wobble invalidated a correctness claim, and the
  failure presented as `assert report.accepted is True` with a 9,000-character repr —
  it reads as "my change broke the learning benchmark" and costs a real detour to rule
  out. Corpus schema 2 moved the key; schema 1 still loads and its budget is read
  through the fallback in `_check_performance_budget`, because `--corpus` lets a caller
  supply their own file. **Footgun**: keeping the two verdicts apart is the whole fix,
  and the way to undo it is one `failures.extend(performance_notes)` —
  `test_no_latency_text_leaks_into_acceptance_failures` exists for exactly that edit,
  since the two obvious tests either side of it stay green when it happens (#183).
- A2UI memory inspector (`neo.a2ui`): a per-project A2UI v0.9 surface
  (`neo-<project_id8>`) registered with the running `car-server` daemon so any
  conformant renderer (CarHost.app, future webviews) can inspect neo's state
  live. Two tabs: **Observer** (status badge, pid, last cycle, recent cycles
  list, Kick/Stop buttons) and **Memory** (valid fact count, by kind, by scope,
  probation count). Updates pushed by the observer process at the end of each
  sweep cycle — the same FactStore load powers both tabs, so the
  inspector adds zero hot-path cost. Kick/Stop buttons emit `a2ui.action`
  notifications which the observer dispatches to `kick_observer` /
  `stop_observer` — closes the loop with CAR's supervisor. **Footgun**:
  Python's `car_runtime.a2ui_*` helpers are in-process only; reaching the
  daemon's shared store (which renderers subscribe to) requires speaking
  JSON-RPC over its WebSocket. `neo.a2ui.DaemonClient` is that bridge.
  Activation: auto when `127.0.0.1:9100` is reachable; silent no-op
  otherwise. Adds `websockets>=12.0` to the `[car]` extra.
- Async transcript-mining observer (`memory.observer`): a **single global**
  background process (CAR agent `neo-observer`, daemon `--daemon --all`) that
  **sweeps all discovered projects** each cycle — round-robin/budgeted
  (`max_projects_per_cycle`, default 25; watermark- AND mtime-gated so unchanged projects do near-zero work (the watermark alone gates only *admission*: sources still parsed every transcript each cycle, measured at 298 MB for one project, which is what drove multi-GB observer RSS. `_unchanged_since` now skips files untouched since the watermark file's mtime minus a 1h skew margin; every error path falls back to parsing, because a wrong skip loses learning silently)) — running transcript mining per project. (It also ran
  `synthesize_reviews` until that subsystem was removed.) Two roots can share a `project_id` (two
  clones of one remote, e.g. `flyx/fms` + `flyx/fms2`), meaning one fact file and
  one pid-keyed watermark; the sweep keeps a per-cycle `store_cache` so such a
  project loads and synthesizes **once**, while transcript ingest still runs per
  root — attribution is by cwd and the second clone has sessions the first would
  never see. **Not opt-in**: `maybe_autostart_observer()`
  (called from `cli.main`) auto-registers it whenever `car-server` is reachable;
  opt out with `NEO_OBSERVER_AUTOSTART=0`. No CAR → one-time hint, then silent.
  Projects are discovered from `~/.claude/projects/*` (decoded roots), minus
  **container roots** (`observer._is_container_root`): a decoded root that holds
  another discovered root and is not itself a repo. Claude Code mints a
  transcript dir for whatever cwd a session ran in, so ad-hoc sessions from `/`,
  `$HOME` or `~/git` created pseudo-projects that were ancestors of every real
  one — 26.3s of a ~40s cycle, and `CodexSource`'s cwd-prefix attribution then
  claimed the machine's entire Codex history under them (`/` matched everything
  because `"/".rstrip("/")` is `""`). The `.git` clause only ever *rescues* an
  ancestor (a monorepo is a real project); it is never a blanket requirement,
  since many real roots here have no repo. Nested real roots are disambiguated
  by `CodexSource(peer_roots=…)` — deepest root wins. Known limits: the
  inventory is Claude Code's, so a project only ever opened in another tool is
  invisible; comparisons are case-sensitive on a case-insensitive filesystem. On
  bootstrap/start, legacy **per-project** agents (`neo-observer-<id12>`, the old
  model) are stopped + `agents_remove`d. **Hard dep**: car-runtime ≥ 0.18.0
  (pin floor 0.27.0) + a running `car-server` — CAR's supervisor owns
  spawn / restart-on-failure / clean SIGTERM. Logs at
  `~/.car/logs/neo-observer.{stdout,stderr}.log`. Lifecycle/`status`/orphan-check
  all operate on the single global agent. (A2UI per-project inspector is skipped
  in global mode.)
  **RSS bounding**: the daemon re-execs itself every
  `NEO_OBSERVER_RECYCLE_CYCLES` cycles (default 48, ~4h; 0 disables). Its RSS is
  peak working set plus CPython arena fragmentation — each sweep deserializes a
  multi-MB fact file (79% of whose rows are tombstones retained by the 30-day
  policy) and the allocator never returns those arenas. Nothing leaks; RSS
  drifts to the high-water mark and stays (measured 0.5–0.7 GB). Embeddings are
  NOT the cost (1.9 MB in memory; they dominate the file on *disk* only, as JSON
  text). Re-exec, not exit-and-be-restarted: the CAR spec is
  `restart: "on_failure"` with `max_restarts: 10`, so a clean exit(0) would
  never restart and forcing a non-zero exit would exhaust the budget after ten
  recycles. **Footgun**: the single-instance lock MUST be released before the
  exec. Carrying the fd across looks safer but `flock` is owned by the open file
  description, so the inherited fd keeps the file locked and the new image can
  never take it — it exits as "contended" and the machine is left with no
  observer. That failure was measured directly. The floor is one cycle's working
  set (~380 MB), so recycling caps drift, not baseline.
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
  a sweep at once even in the handoff window; a contended daemon exits 0
  (benign no-op, no CAR backoff). If a straggler ignores SIGTERM past the
  `_LOCK_ESCALATE_AFTER` grace, the daemon escalates to SIGKILL so the kernel
  frees the lock — safe because `FactStore._save_file` is atomic (temp +
  `os.replace`), so a hard kill can only leave a stray `.tmp`, never a torn
  fact file. (This is belt-and-suspenders: `store.save()` already serializes
  writers with its own per-scope flock, so the orphan was never a corruption
  bug — just doubled LM spend and mining.) Tunables:
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
  The log rotates to `metrics.jsonl.1` at 32 MB (one generation retained); the
  size is sampled every 500th write so the steady state stays one `write` per
  event. Readers (`memory citation-stats`, `memory learning-stats`) window by
  `--since` and read only the active file — a `--since` older than the last
  rotation silently sees less history.
  Sessions and watermarks live in `~/.neo/sessions/`.
- Disk hygiene: `FactStore` reaps abandoned `*.tmp` atomic-write files older
  than 24h from `~/.neo/facts` on cold start (`_reap_stale_temp_files`). A save
  that is SIGKILLed between `mkstemp` and `os.replace` strands its temp file —
  `_save_file` unlinks on *exception*, but SIGKILL runs no handler, and the
  observer's own lock-escalation path SIGKILLs stragglers by design. 88 MB had
  accumulated this way with nothing to sweep it.
- `scope._get_git_remote_url` is memoized per root (`clear_remote_url_cache`
  resets it, called once per observer cycle). Resolving one project's identity
  asked git for the same remote 3× — ~100 forks/cycle, ~26k/week. Staleness is
  bounded to one cycle because a stale hit would write facts under the wrong
  `project_id`.
- **Measuring retrieval changes: pin `PYTHONPATH`, and measure the REAL pipeline.**
  Two traps, both hit during the BM25 work, both producing confident wrong numbers.
  (1) The venv installs neo EDITABLE against `src/`, so running
  `.venv/bin/python -m neo.cli` from a git worktree of another commit executes THIS
  tree's code against that tree's files — a "baseline" run that is not the baseline.
  Every A/B needs `PYTHONPATH=<that tree>/src`, and `rank_mine_eval` now REFUSES to
  run unless the tree it was handed is the tree `import neo` actually resolves to —
  mandatory is not the same as effective, since the editable `.pth` silently catches
  a typo'd path and measures the working checkout twice. Measured on the superseded
  harness generation: it made main look like MRR 0.613 against a real figure of
  0.082 (the current harness puts main at 0.304 — different instrument, same trap). (2) Calling `score_candidate` directly
  measures the FIRST-PASS ranking only; `gather_context` then re-ranks with
  `pi_boost + hist_boost + _symbol_score` and applies an adaptive limit and a byte
  budget. A first-pass harness overstated R@10 by 0.14 against the real CLI. Validate
  any in-process replica against `--dry-run` output before trusting a sweep.
- Debugging: `neo --dry-run "your query"` runs the real engine — file selection, fact
  retrieval, constraints, four-layer assembly — and prints the **exact messages** that
  would go to the provider, then exits without making the LLM call. Faster iteration on
  context-gatherer and retrieval changes than waiting for an inference round trip.
  **Use it before believing any claim about what Neo "saw"** — the two defects below
  were both invisible from the outside and presented as the model being unhelpful.
  This bullet described the tool's *intent* for a long time and not its behaviour: the
  flag used to exit in `cli.main` **before the engine was constructed**, so three of
  the four things listed above never ran and the output was the file list alone. The
  Execution Envelope, retrieved facts, and the REPOSITORY CONTEXT block with its
  truncation markers — the #178 work, whose entire point is that a cut be visible —
  were all uninspectable through the tool built for inspecting them. An instrument
  that under-reports sends the operator to the wrong knob, which is the same failure
  as a cap that blames itself for an absence it did not cause.
  **The prompt is recorded, never rebuilt** (`neo.dry_run.RecordingLM` is a real
  `LMAdapter` installed in the engine's own `self.lm` slot), because a renderer that
  walked the context dict would be a second implementation of the seven prompt
  builders, free to drift the moment one changed — the duplicated-rule shape that put
  `EXCLUDED_DIR_NAMES` in two places. What it shows is the adapter's INPUT, not the
  wire payload: Anthropic hoists `system` into a separate kwarg, Google remaps roles,
  Ollama flattens, CAR adds `intent_json`, and no provider is resolved at all because
  the flag deliberately requires no credentials. The output says so rather than
  claiming exactness it cannot have.
  `DryRunComplete` derives from `BaseException`,
  not `Exception`: `_process_guarded` converts anything its `except Exception` catches
  into a `FAILED` lifecycle event, and reporting a dry run as a crash would be one
  more way of misdescribing the run. An ordinary exception is NOT a safe substitute —
  `_deliberate` has its own `except Exception` that would swallow it and silently
  fall back to the fast path.
  **The panel is forced OFF** under `dry_run`. This is correctness, not tidiness:
  `_build_car_role_factory` calls `create_adapter("car", model=m)` per role and uses
  `self.lm` only as the fallback, so `RecordingLM` never intercepts it. With
  `car-server` reachable — the normal setup here, since the observer autostarts off
  it — a novel prompt under `--dry-run` ran the full panel against real models, spent
  real money, never raised `DryRunComplete`, and printed ordinary output. Measured: 4
  real adapters built. It is also the honest scope, since the panel's later prompts
  are built from earlier model responses and cannot be shown without making the calls
  the flag exists to avoid.
  **A dry run does not modify the fact store**, which is narrower than "mutates
  nothing" and is the claim that survives measurement. The old implementation got it
  for free by never constructing a `FactStore`; `FactStore.initialize` runs
  `prune_stale_facts` → `demote_unhelpful_facts` → `purge_dead_facts` and then
  **saves**, and `demote_unhelpful_facts` lowers confidence and invalidates facts — so
  reaching the engine at all meant a "read-only" inspection was aging the store it
  inspected. `FactStore(read_only=True)` makes `save()` a no-op at the single write
  choke point, which a new caller cannot forget; `dry_run` also skips
  `detect_implicit_feedback` and `_complete_learning_episode`. Retrieval still marks
  facts accessed in memory, and `metrics.jsonl` still records the run —
  deliberately: the two events it writes (`execution_context_resolved`, `retrieve`)
  are read by neither `citation-stats` (which filters `citation_survival`) nor
  `learning-stats` (which reads the episode ledger), so observability costs nothing.
  **That argument was measured on the fast path and briefly untrue on another.**
  VERIFY mode reasons without an LM call, so `process()` returns NORMALLY and never
  raises `DryRunComplete`; `_complete_learning_episode` then wrote an episode file
  AND a `citation_survival` metric — the exact two surfaces the sentence above
  claims are untouched. Measured on a clean HOME: 1 episode, 1 `citation_survival`.
  Gating that call is what makes the claim true; do not narrow the gate to the
  recorded-call path.
  Under `--json` the report is the single stdout document (`{dry_run, calls, note}`
  — a second schema, discriminated by `dry_run: true`, with no `orchestrator` key;
  `test_host_adapter_parity.py` does not know about it) and a terminal
  `phase_completed(reasoning)` + `completed` pair is emitted. Writing prose to
  stderr broke both `--json` invariants at once: zero documents on stdout, and every
  source line beginning with `{` became a counterfeit event. **Both dry-run exits
  route through `cli._report_dry_run`** — there are two, the recorded call and the
  normal return, and the second shipped without the `--json` handling the first had.
- Project index (`index/project_index.py`, `index/language_parser.py`; full
  notes in `docs/tree-sitter-setup.md`). Three invariants, each of which was
  violated and each of which produced an index that could not answer a question
  about its own repository:
  1. **Budgets are apportioned, never sliced.** `_select_files` groups eligible
     files by language and hands each a share of `--max-files` proportional to
     repo composition, floor of one slot per language (`_allocate_slots`); then
     `_cap_chunks` round-robins `MAX_CHUNKS_PER_REPO` across FILES so each keeps
     a chunk before any keeps a second. Both exist because a list slice is not a
     ranking: globbing `**/*.py` before `**/*.cs` and slicing gave a .NET repo of
     4,272 C# files an index of 83 Python files (95 from an in-repo worktree) and
     zero C#, exit 0. Fixing only the file cut left `chunks[:1000]` re-creating it
     one function later — chunks arrive grouped by file and files by language, so
     the slice kept 1000 C# chunks and dropped every other language, with 37 of
     the 100 selected files contributing nothing.
  2. **Exclusion is two layers and `bin`/`build`/`out`/`target`/`dist`/`vendor`
     belong to neither by name.** Both layers, and the walk that applies them,
     live in `neo/eligibility.py` — the ONE eligibility module, consumed by
     `context_gatherer`, `ProjectIndex` and `architecture_metrics` alike.
     `DEFAULT_IGNORE_PATTERNS` covers what repos forget to ignore (`.worktrees`,
     `.claude/worktrees`, `node_modules`, `obj`, virtualenvs);
     `load_ignore_patterns` layers the repo's own root `.gitignore`/`.ignore` on
     top, so a repo's `!negation` can re-include what a default excluded.
     `should_ignore` only tests the path handed to it, so ancestor directories
     must be handled separately — `walk` does that by PRUNING an ignored
     directory instead of descending, which is git's own rule and the reason a
     `!` beneath an excluded directory does not fire. Matching is exact and
     case-sensitive, against directory components only. The ambiguous names stay
     out because each is real source somewhere (`src/bin/main.rs`, vendored
     trees; 254 tracked files under `bin/`+`vendor/` across three local repos)
     and the asymmetry is one-sided: over-excluding hides code permanently,
     over-including only spends slots.
     **Three more exclusion classes are NOT gitignore, and conflating them
     sends an operator to the wrong file**: `WalkPolicy` knobs (symlink
     rejection, the gatherer's 512 KB `MAX_FILE_BYTES` ceiling, extension and
     per-language glob filters) are per-consumer policy; nested `.gitignore`
     files are not read, which under-excludes and is the accepted limit; and git
     applies ignore rules only to files it does not already TRACK, so a file
     added before a rule was written stays tracked while the walk still skips it
     (four `specs/*.md` on this checkout — recorded, deliberately not fixed in a
     pure refactor, since closing it means a `git ls-files` fork on the warm
     path). Two tests hold the line: `test_eligibility_single_source.py`
     AST-scans `src/` and fails on a second definition, a second `os.walk` or a
     second exclusion list (detected by CONTENT — three sentinel directory names
     in one literal — because a copy always renames the variable);
     `test_eligibility_differential.py` diffs the walk against `git check-ignore`
     over a fixture corpus AND this checkout, and fails on any tracked file
     skipped without an ignore rule accounting for it. Both are marked
     `invariants`, so they run in the Guard-invariant battery on every PR.
     **Footgun**: the index's `excluded` count is now excluded PATHS SEEN
     (`excluded_dirs` + `excluded_files`), not files under an excluded
     directory. A pruned subtree is one path; the walk does not descend and
     therefore does not know how many files are inside. The old "200 paths"
     number was only available because the old code globbed the whole repo and
     filtered afterwards — i.e. it walked every worktree copy in order to count
     what it was about to discard.
  3. **A cap that fired must be reported.** `selection_report` carries eligible /
     selected / excluded / duplicates / chunk counts and the CLI prints them.
     `truncated` means THE CAP BOUND US and is `examined < eligible` — whether
     any candidate went unlooked-at, which the per-language iterators already
     know. Both cheaper predicates name the wrong knob: `selected < eligible`
     made dedup print "2 of 7 eligible files (capped at --max-files=1000)", and
     `selected >= max_files` did the same whenever `--max-files` landed exactly
     on the unique-file count. Raising a cap that never bound fixes nothing.
     Separately, round-robin only represents every file while chunk slots ≥
     files; `MAX_CHUNKS_PER_REPO` is fixed at 1000 while `--max-files` is not,
     so `files_with_chunks` reports the shortfall — `truncated` is False there,
     because the FILE cap genuinely was not the constraint.
     **And the corollary the first version of this report broke: never blame a
     cap for an absence it did not cause.** A selected file is missing from the
     index for one of TWO unrelated reasons, and the console must not guess
     between them. Either it produced no chunks at all — no function, class,
     interface or struct for the grammar to match, as in an empty
     `__init__.py`, an enum-only `.cs`, a type-alias-only `.ts` — which no cap
     setting changes and which therefore gets a bare statement with NO remedy
     attached; or the cap took every chunk it produced, which gets the cap
     named and `lower --max-files`. `files_producing_chunks` is measured
     BEFORE `_cap_chunks` precisely so the two stay separable; subtracting the
     post-cap `files_with_chunks` from `selected` conflates them and is what
     printed "the 1000-chunk cap is below the 25 files selected" for a build
     that kept 559 of 559 chunks, with `chunks_capped` False on the same
     report. It fired on this repo, on any repo with an `__init__.py`. A report
     that invents a cause is worse than the silence it replaced, because
     silence at least does not send the operator to the wrong knob.
  **Query footgun** (`language_parser.py`; incident detail in the tree-sitter
  doc's "Why queries break silently"): a query that fails to compile is
  indistinguishable from one that matches nothing — `_get_query` returns None
  and `parse_file` moves on; edge failures log at DEBUG. Four were broken at
  once, so TS interfaces and ALL C# inheritance edges were absent from every
  index since they shipped. Three rules follow. Compile results are cached
  INCLUDING failures (the uncached retry warned per query *per file* — 9,699
  lines in one run). C# bases need all four of `identifier`, `generic_name`,
  and `qualified_name` with either as its `name:` field — across
  class/interface/record/struct — since a narrower pattern compiles fine and
  silently drops `: Repository<Order>`, `: System.Exception` and
  `interface IX : IY`. That query is GENERATED from `_CS_BASE_TYPES` over the
  four declaration kinds rather than written out four times: the hand-copied
  version is what left `interface_declaration` uncovered, and each widening
  since has had to be applied everywhere at once or not at all. And
  `test_every_chunk_query_compiles` /
  `test_every_edge_query_compiles` prove compilation ONLY, so a new query still
  needs its own behavioural assertion.
- Context selection (`context_gatherer`): **files are ranked by BM25 over their
  CONTENT** (`neo.file_retrieval`), not by their path. Until 2026-08 they were: the
  scorer took `(rel_path, size, prompt_tokens, git_recent, entry_points)` and the
  file was first opened *after* selection, only to chunk what had already been
  chosen. Every other defect in that scorer followed from having no content signal,
  and the dominant term was `score -= 0.01 * size_kb`, uncapped, against a realistic
  positive signal of +0.6 to +2.1 — so a file with one keyword hit was unrankable
  above 60 KB. `src/neo/memory/store.py` scored **0.000** and ranked 200th of 284 for
  "fix the fact store supersession threshold", because it is 162 KB. Ground truth ran
  31–177 KB against a corpus median of 10 KB: central files are large *because* they
  are central. That had already been noticed once and patched with a seven-name stem
  whitelist, which rescued `engine.py` (−0.13) and left `store.py` (−1.62) — a 12×
  disparity decided by whether someone had thought of the name. **The sign was wrong,
  not the magnitude**: BugLocator's rVSM (ICSE 2012) ranks larger files *higher* for
  this exact task, and BM25's `b` handles the concern with bounded, corpus-derived
  length normalization. Measured end-to-end over cases mined from git history
  (commit subject = query, changed non-test files = ground truth), R@10 / MRR:
  neo 0.301→0.742 / 0.304→0.771, car 0.180→0.472 / 0.162→0.425, quip 0.174→0.696 /
  0.158→0.643, at `CONTENT_WEIGHT = 3.0`. **`tools/rank_mine_eval.py` is the
  harness** — not `tools/rank_eval.py`, which is a different instrument (12
  hand-labelled prompts, this repo, recall@k, no MRR) and was named here in error
  while the real one went uncommitted, leaving the figures unreproducible. Quoting
  a number from one under the other's name is how `car` got into the record twice
  at 0.969 and 0.507; keep the generation attached. An earlier generation of these
  same figures read neo 0.078→0.603 / 0.082→0.655 — superseded, and not comparable,
  because the harness that produced it no longer exists to re-run.
  **Measure with `--no-git`, which is the default.** The scorer's recency signal
  reads `git status --porcelain` plus the last 50 commits and holds PATHS, not
  commits — so a case mined below the window whose truth file was touched again
  inside it is still handed its own answer key, as is every file dirty in the tree
  you are measuring from. `--skip-recent` does NOT fix this and a first version of
  that docstring wrongly said it did. `neo --dry-run --no-git` gates `git_recent`
  and nothing else (`_history_boost` and the rest of the re-rank stay live), which
  is what `tools/rank_eval.py` had been doing all along. `--with-git` measures the
  full pipeline and then reports `contaminated_cases` per run: with it on, 48 of 50
  neo cases are contaminated, and the figures move by ≤0.03 — the leak is real but
  was never what carried the result.
  **The re-rank is LOAD-BEARING, not redundant** (`pi_boost` + `_symbol_score` +
  `hist_boost`, applied after the first-pass score). Disabling it on the real CLI
  takes R@1 from 0.344 to **0.044** and MRR from 0.646 to **0.261** — a
  SUPERSEDED-generation ablation (pre-`--no-git`, uncommitted harness), so read
  those four numbers against each other and never against the table above. An earlier
  version of this note claimed the opposite, from a weight sweep that landed
  everywhere in 0.66–0.68 — measured at k=10, where every configuration is flat, and
  through an in-process replica that omits the byte budget and adaptive limit, i.e.
  exactly the stages that make the re-rank matter. Both errors are the ones
  `tools/rank_mine_eval.py` warns about in its own docstring (`rank_eval.py` was
  cited here and carries neither warning). Per channel, `hist_boost`
  contributes nothing measurable (identical results with it disabled) while
  `_symbol_score` carries most of the effect.
  RRF fusion with the dense channel LOSES to BM25 alone (0.596 best-weighted vs
  0.693, also superseded-generation and unstamped when first recorded) — dense
  returns ~25 files against BM25's ~180 and is half as accurate. Treat
  that as provisional: it was measured at k=10 before the same cutoff problem was
  understood, and the docstring deferring it to a chunk-allocation fix is stale
  because that fix landed in the same branch. Re-measure at k=3/k=5 before relying on
  it. Four filename-tuning fixes were separately measured and rejected; the filename
  is not the evidence, the file is.
  **A path named in the prompt is still pinned** (`EXPLICIT_PATH_BOOST=10.0`, chosen
  to exceed every organic signal combined — content caps at +3.0 (`CONTENT_WEIGHT`), the re-rank boosts
  at +1.0 and +1.2). Without it a spelled-out path competed on generic filename-token
  overlap and lost: `src/neo/subcommands.py` ranked **163rd of 296** on a prompt
  naming it, below its own test file, because an 86KB file took a heavy size penalty. The file never
  reached context, so the model correctly refused to patch code it had not seen and
  emitted NO diff — and a suggestion with no diff text can never be git-verified,
  which is a major reason only ~32% of suggestions were verifiable.
  `matches_explicit_path` tests containment in BOTH directions (an absolute path —
  what tracebacks, IDE copy-path and Neo's own output emit — must match the
  repo-relative candidate) and anchors on `/` so bare `subcommands.py` hits
  `src/neo/subcommands.py` but NOT `tests/test_subcommands.py`. A named path that
  matched nothing emits a WARNING; silence there is indistinguishable from "no path
  mentioned".
  `select_chunks` ranks windows by **per-file document frequency** and **matched-token
  length**, not file order or match count. It used to take `matching_idxs[:5]` — the
  first five matching lines in FILE ORDER — and since matching is a substring test and
  `extract_prompt_tokens` emits every 3+ character word, `"in"` matched
  `int`/`using`/`point` and virtually every line qualified. "First five matches"
  therefore meant **lines 1-5 for any prompt against any large file**: every large file
  contributed its import block, twice, as overlapping near-duplicates that consumed
  both slots of `MAX_CHUNKS_PER_FILE`. A length cutoff for "discriminative" is
  INVERTED on real prompts (English stopwords are long, identifiers are short — it
  keeps `does`/`here` and drops `db`/`fs`/`os`), hence document frequency
  (`DISCRIMINATIVE_MAX_LINE_FRACTION=0.25`). Length weighting matters because match
  COUNT let a keyword-bearing module docstring tie with the function body and win on
  the file-order tie-break. Window merging is bounded by `MAX_MERGED_WINDOW_LINES`:
  unbounded chain-merging measured **8.3× over `max_chunk_bytes`**, and because the
  caller admits a chunk all-or-nothing, an oversized chunk that no longer fits the
  global budget is DROPPED — the original bug returning through a different door.
  `_fit_to_budget` shrinks outward from the window's best line so truncation can never
  discard the line that earned the window.
- Prompt-side file rendering (`engine._render_context_files`): the REPOSITORY
  CONTEXT block caps each file at `_CONTEXT_FILE_CHARS` (3000), or
  `_IMPORTANT_FILE_CHARS` (8000) when the path contains one of
  `_IMPORTANT_FILE_PATTERNS`, and shows at most `_MAX_CONTEXT_FILES` (20).
  **Every cut MUST be marked and the banner MUST count post-truncation
  characters** — that is the whole contract, and both halves were broken.
  `content_preview = content[:char_limit]` appended nothing, so `--- path ---`
  followed by text that just stops was indistinguishable from a file that ends
  there, and the model answered questions about absence ("X is never called",
  "there is no null check") from a fragment. The dangerous case is not an
  obvious clip but a cut landing just past a method signature: enough to look
  answerable, not enough to be right — measured live, a claim confirmed at 0.90
  that moved to 0.99 *and reversed* once the body was supplied. Marked
  truncation converts that into a refusal, which neo already does well.
  Meanwhile the banner summed `len(f.content)` BEFORE the cut and called the
  result bytes, so "12 files, 340000 bytes" described a payload the model never
  received. It now reads `sent of total chars` — **chars, not bytes**: `len()`
  on a `str` counts code points and the caps are character counts, so claiming
  bytes was a second lie on any non-ASCII source. **Footgun**: the function
  returns THREE values — `(sections, banner, visible)`. `visible` is each shown
  file cut to what the model actually got, and `code_smells.scan_files` MUST be
  fed it rather than the originals. Scanning full content emitted findings like
  `src/Service.cs:401 [todo/warn] HACK: ...` for a file the model saw to line
  ~215: an unmarked cut invites a wrong inference about unseen code, but a
  line-numbered finding about unseen code asserts one.
  Use `neo --dry-run` to see the block.
- **Prompt-bound text is cut through `neo.text_budget`, never by a slice.**
  Three shapes, and the choice is a correctness question, not a style one:
  `truncate_marked` keeps the HEAD (source, problem statements, memory);
  `elide_middle` keeps BOTH ENDS (tracebacks); `shown_of` annotates an elided
  LIST. Picking wrong compiles, passes, and throws away the answer — a
  head-cut of a 40-frame traceback keeps forty `File "..."` headers and drops
  `ValueError: database is locked`, and a plain traceback puts the exception
  line last while a chained one puts the original cause first, so only a
  middle-elide serves both. A list is truncated as silently as a string: under
  a prompt saying "follow this exactly", three bullets read as three edge
  cases. Before the extraction the defect sat in NINETEEN individual cuts
  across nine prompt builders in six modules. Seventeen were found by the
  first sweep; two more (`_community_fallback_learnings`, one body cut and one
  list elision) came out of review, and the reason they were missed is worth
  keeping: that body cut is a `[:200]` on a `.get()` result rather than on a
  named variable, so it does not pattern-match the other seventeen. A sweep
  that looks for slices on identifiers will miss the same shape again.
  **Never nest cuts.** `_deliberate` used to cut each section to 2,000 and the
  concatenation to 6,000, which delivered `verifiable_constraints` as 1,890 of
  31,000 characters with its own marker sliced off, beneath an outer marker
  reporting 163 dropped — 29,110 characters gone behind a note asserting they
  were not, and always that section, because the order and caps were both
  fixed. A marker computed over a concatenated buffer is structurally
  incapable of describing source loss. Sections now share the budget via
  `apportion` (max-min fair) and are cut ONCE. Nesting also wasted the budget
  it was cutting: on a live store a 30-char envelope plus 33,144 chars of
  memory against 6,000 sent **2,030 under the old flat cap**, where
  apportionment sends the full 6,000. **Check whether a value arrives already
  cut** — `outcomes` caps `diff_summary` before `store` hands it to
  `pattern_extraction`, whose own cut drops that inner marker, leaving one
  that understates the real loss. The community fallback was a second instance
  of exactly this, inside `engine` itself: its unmarked body cut fed
  `past_learnings`, which `_deliberation_context` then apportioned and cut
  again, so the new marker would have reported the pre-cut length as the
  fact's real one. Marking the inner cut is what makes the outer one true —
  an apportioned section is only as honest as its least honest input. **Footguns**: `truncate_marked` and
  `apportion` raise on a budget that cannot do the job and no call site
  catches them narrowly — every caller passes a module constant, so it is a
  programmer error, and the one runtime-reachable path (the panel) degrades to
  the fast path through its existing broad handler. And the two helpers differ
  on the budget: `elide_middle` reserves its marker and lands **inside** the
  budget, while `truncate_marked` appends on top, so it bounds the CONTENT and
  its rendered section runs ~50 chars over.
- Host communication (`neo.events` + `models.OrchestratorMessage`): the engine
  reports lifecycle facts so a host (Claude Code, MCP, an IDE) doesn't have to
  reverse-engineer presentation from raw plans. **Governing rule: Neo reports
  facts about its process; the host decides how and when they become
  conversation** — never put host-specific wording in `engine.py`.
  `--json` writes **two streams**: stdout is exactly ONE JSON document (now with
  an `orchestrator` key), stderr is JSONL events. They are deliberately NOT
  interleaved — the old `--json` help text promised events on one stream, and
  honoring that literally would break every `neo --json | jq` consumer.
  `--json` implies `--quiet`, so stderr is *essentially* pure JSONL; logging
  warnings and CAR notices can still land there, so hosts parse lines starting
  with `{` and skip the rest. `--quiet` had been declared and never read — the
  `[Neo]` notices now route through `neo.progress` (`note()`/`set_quiet()`)
  instead of bare `print(file=sys.stderr)` calls scattered through
  `context_gatherer`. The flag is process-global on purpose (a display setting
  fixed once from argv, read five layers down); index-command **errors** stay
  unsuppressed, because `--quiet` silences progress, not failures.
  `safe_emit` swallows sink exceptions — an observer must never take down the
  run it observes — logging the FIRST at WARNING and the rest at DEBUG so a
  broken sink stays findable; `NullSink` is the default, so an unobserved run
  costs what it always did. Four **invariants**, all test-pinned: (1) findings
  (`hypothesis_formed`/`risk_found`) are emitted BEFORE their phase closes;
  (2) no event may claim a closed phase — `MEMORY_FOUND` fires from
  `_capture_retrieval_context` (the funnel all fact-retrieval paths pass
  through) which runs while REASONING is open, so it labels itself via
  `_current_phase()`; (3) every `phase_completed` has a `phase_started` —
  the budget-skip branch OPENS the phase before closing it `skipped`, and
  `_end_phase`'s synthesize-on-missing fallback logs a WARNING so the next
  caller bug is findable; (4) every run terminates with `completed` or
  `FAILED` — `process()` emits `FAILED` from its except branch and marks
  still-`running` records `failed`, because STARTED-then-silence is worse than
  no events (a host can't tell a crash from a hang). The first phase is
  `context` (file gathering), deliberately NOT `retrieval` — fact retrieval
  happens during REASONING, and the old name forced every emitter inside it to
  work around the overstatement. `_end_phase` reverse-searches for the newest
  open record because a failed panel closes `reasoning` as `fallback` and the
  fast path opens a second one.
  `_build_orchestrator_message` is **pure derivation** — no LM call, no new
  analysis; every claim must be defensible from the rest of the `NeoOutput`.
  **VERIFY mode gets its own sentence**: it makes no LM call and its
  `code_suggestions` are the CALLER's `proposed_changes` echoed back, so the
  generic "Neo reasoned … and proposes N change(s)" credited Neo with the
  caller's work; it also suppresses the low-confidence caution (VERIFY
  confidence is a pass/fail verdict, so "verify before acting" is circular).
  `cautions` distinguishes no-tools from skipped-for-budget (different
  remedies) and is CAPPED — a host is told never to drop one, so the list must
  stay relayable; anything dropped is reported as a count. `phase_summary`
  copies each record, not just the list (live engine state, foreign consumer).
  **Voice is staged and lives in the deck, NOT in code.** Every user-facing
  string comes from `orchestrator_voice.lines` in `neo_matrix.yaml` via
  `engine._voice(key, **fmt)`; `_voice_stage()` picks opener/hedge/terseness
  from `orchestrator_voice.stages` keyed on `_memory_level_to_stage()`, so the
  same facts read `Don't know this code. … , maybe. Confidence 0.88.` at stage 1
  and `src/parser.py. 1 change(s). 0.88.` at stage 5. A single fixed register
  throws away a signal the system already computes.
  `test_engine_holds_no_prose_of_its_own` pins the no-literals rule — a string
  in `engine.py` is a personality change hidden in a code diff. `_voice`
  degrades to `""` on a missing key/bad placeholder (a YAML slip must not kill a
  run) and `_cap_cautions` drops blanks so a failed template isn't relayed as an
  empty bullet. **Deliberate asymmetry**: cautions and progress messages do NOT
  vary by stage — a host is told never to drop a caution, so a warning must read
  the same at every stage; voice is not licence to soften a fact. LM-voiced
  summaries were considered and rejected (a call per run + fabrication risk);
  the system prompts already inject personality, so LM-authored prose is voiced
  by inference and only the derived scaffolding is templated.
  Beat surface metadata (`surface`/`importance`/`requires_finding`/
  `orchestrator_line` in `neo_matrix.yaml`) gates personality: `requires_finding`
  beats stay silent unless the run actually found something, and there is NO
  fallback line — silence is the default. **One beat per run** (`_run_beat`
  selects once and caches): `_generate_notes` and `_orchestrator_beat` both read
  it, and `--json` carries both, so independent selection let one character
  speak with two voices. `_select_beat` derives `memory_hit`/`no_memory_match`
  from real recall state — without them `unfamiliar_codebase` was configured,
  surfaceable and **unreachable**, i.e. dead text that a "declares its wording"
  test can't catch; `test_every_declared_beat_can_actually_fire` pins
  reachability itself. Beat lines must not outrun their trigger (a single
  traceback is not "multiple failures"; the keyword *fix* does not establish
  something "used to work"). No `cooldown`/`max_uses`: one process
  selects at most one beat, so in-process cooldown is a no-op and cross-run
  needs session state that doesn't exist. The plugin contract that consumes all
  this lives in `.claude-plugin/agents/neo.md`. See
  `docs/solutions/orchestrator-communication.md`.
- **Test home isolation is enforced by a table, not by `Path.home()` alone.**
  conftest patches `Path.home()`, but ~20 constants across the codebase capture
  their path at IMPORT time (`SESSIONS_DIR = Path.home() / ".neo" / "sessions"`)
  and pytest imports every module during collection — *before* any fixture
  runs. The fixture's docstring promise was therefore false, and measurably so:
  one run of test_outcomes + test_fact_store + test_transcript wrote
  `~/.neo/constraints/checksums.json` and
  `~/.neo/sessions/watermark_testproj1234.json` into live developer state.
  `conftest.HOME_PATH_CONSTANTS` re-points each one at the fake home.
  **Adding a new import-time `Path.home()` constant WILL fail
  `test_home_isolation.py`** — that test AST-scans `src/` and asserts the table
  is complete, which is the durable half; the table alone would rot. The scan
  strips `lambda`/deferred subtrees, because a `default_factory=lambda:
  Path.home() / …` resolves at instantiation and is already covered by the
  `Path.home()` patch (`prompt.scanner.claude_home` is the live example).
  `store.FASTEMBED_CACHE_DIR` is deliberately EXEMPT — a ~400 MB read-mostly
  model cache pinned to the real cache so it isn't re-downloaded per run.
  The scan knows FOUR spellings of "home" (`Path.home()`, `expanduser`,
  `environ["HOME"]`, `getenv("HOME")`) and recurses into module-level
  `if`/`try`; knowing only the first let `observer._CAR_HINT_FLAG` and
  `observer._LOCK_PATH` sit unprotected while the test reported success. It
  still CANNOT see derived constants (`CHECKSUM_FILE = CHECKSUM_DIR / ...`),
  re-imported bindings (`from ...outcomes import SESSIONS_DIR`), aliased
  imports, or non-`Name` targets — that limit is written into the test
  docstring rather than papered over.
  Note `transcript` does `from ...outcomes import SESSIONS_DIR`, binding a
  SECOND name that must be patched separately. **Do not** verify isolation by
  watching the filesystem: the observer daemon writes to `~/.neo` on its own
  schedule and will produce false positives (it did during this work).
- **`NeoEngine` is one-request-at-a-time, enforced.** `process()` takes a
  non-blocking `threading.Lock` and raises `EngineBusyError` (a `RuntimeError`
  subclass, so broad handlers still catch it) on overlap; the body moved to
  `_process_guarded`. Nearly all run state lives on the instance
  (`context`, `current_learning_episode`, `_phase_records`, `_findings`,
  `_selected_beat`, `resolved_execution_context`, `last_applied_actions`), so
  concurrent calls would cross-attribute suggestions/facts/episodes between
  unrelated requests — **silently**, which is why this fails loudly instead.
  Non-blocking is deliberate: queueing would hide a caller's design bug behind
  a latency mystery. **This is not theoretical** — `car_host._get_or_create_engine`
  caches engines per working-dir and REUSES them across calls, relying on CAR's
  drain task being single-threaded (its own comment calls that "an
  implementation detail", i.e. upstream and not ours to guarantee). `_handle_call`
  therefore maps `EngineBusyError` to a distinct retryable `EngineBusy`
  response — a peer that sees generic `ProcessingError` assumes its own request
  was malformed and stops retrying. The lock releases in a `finally`, so a
  failed run doesn't leave the engine permanently busy (pinned by a test).
- **Host adapters must stay in parity.** There are TWO checked-in integration
  surfaces, and they are easy to miss: `.claude-plugin/` (agent + 6 slash
  commands) and **`plugins/neo/`** (Codex CLI plugin + 6 skills, manifest at
  `plugins/neo/.codex-plugin/plugin.json`, registered by
  `.agents/plugins/marketplace.json`). `.agents/skills/` holds RELEASE
  maintenance skills, not the Neo capabilities — looking there and concluding
  the Codex plugin is missing is a documented wrong turn. Neither adapter owns
  an output format: both invoke `neo --json` and read the same `orchestrator`
  envelope. When the contract changes, BOTH change — the Codex skills sat on
  "parse the four structured sections: CONFIDENCE, PLAN, SIMULATIONS, CODE
  SUGGESTIONS" while the Claude side had already moved to `--json`.
  `tests/test_host_adapter_parity.py` pins it: both surfaces expose the same
  six capabilities, invoke `--json`, teach `orchestrator.summary`/`cautions`,
  document the error shape and attribution, and use phase/event names that
  actually exist in `events.py`. It also pins that both plugin manifests match
  the `pyproject.toml` version — `prepare-release` documents bumping them, but
  a documented step with no enforcement gets skipped (they had reached 0.19.0 /
  0.37.0 / 0.41.0). **The version is now single-sourced**: `pyproject.toml` is
  the truth and `tools/sync_version.py` (`make sync-version`) propagates it to
  `src/neo/__init__.py` and both manifests. Release bumps edit ONE file; never
  hand-edit the derived three. `--check` reports drift without writing. Edits
  are surgical regex replacements, not `json.dumps` re-serialization — a
  version bump must stay a one-line diff or it becomes unreviewable — and only
  the FIRST `"version"` key is rewritten so a nested one can't be clobbered.
  `tools/` is now covered by lint; it never was, which is why a dead import sat
  in `ab_controlled.py`. The ruff invocation is duplicated in `Makefile`
  (`lint` + `ci-local`), `.github/workflows/ci.yml` and `.githooks/pre-commit`
  — all four MUST stay identical or local green stops predicting CI green. Role differs even though protocol does not: Claude Code
  delegates to Neo as a subagent (visible boundary), Codex calls Neo mid-loop
  (no boundary), so the Codex skills carry stronger attribution wording and an
  explicit "Neo's result is an input, not the deliverable".
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
  attr); latest validated against car-runtime **0.40.0** (full `test_car_adapter`
  suite including the live calls, against a car-server 0.37.0 daemon). Note the
  pin `car-runtime>=0.27.0,<1.0` lets the client drift well ahead of a daemon
  that ships inside CarHost.app, so a client/daemon **version skew is the normal
  state**, not a fault: car-runtime prints a warning on every invocation, both
  sides speak wire protocol v1, and neo's usage is unaffected. Updating the
  daemon means updating CarHost.app — there is no `car` CLI to run and nothing
  neo can do about it. Do not paper over it with `CAR_NO_VERSION_WARNING=1`; the
  warning is accurate.
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