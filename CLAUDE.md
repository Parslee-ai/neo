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
- **"The linked original fact" means the DURABLE fact the suggestion re-applied,
  and the link nearly died in silence.** `suggestion_fact_ids` (file_path →
  fact_id) is the only input to the ACCEPTED reinforcement branch and to
  `replay_linked_feedback`. It is built by `engine._build_suggestion_fact_ids`
  from the fact `_store_reasoning` returns — and when episodes replaced
  immediate fact-writing (`412a174`, 2026-07-18) that function started
  returning `None` on **every** path (all three `return`s; the legacy branch
  too). The builder returns `{}` for a `None` fact, so the map was
  unconditionally empty from that commit on. Nothing raised: the ACCEPTED
  branch just fell through to the candidate path, `reinforce_legacy_fact`
  became unreachable, and `neo memory replay-feedback` — documented right here
  as the repair command for a broken memory loop — became a no-op that reports
  success. Measured on a live install: 51 sessions through 2026-06-19 carried
  one link per suggestion, then 9 consecutive sessions from 2026-07-19 carried
  zero, and **no fact's `success_count` moved again in 90 days** (`learning-stats`
  reinforcements = 0). Across 6,613 valid facts in every project, **zero have
  ever reached `success_count >= 3`**, so `find_contributable` has never once
  returned a fact and `neo contribute` has never been reachable.
  **That absence had a SECOND, independent cause, and restoring the link alone
  would not have lifted it** — the cited-credit path reached no `save()` at all,
  so the one mechanism that can carry a fact past the 2 that promotion writes
  was discarding its own work. See "A credited fact must reach a SAVE" below;
  read the two together before concluding the contribution gate is fixed.
  The fix does NOT restore per-suggestion fact minting — that is the unverified
  flood `412a174` existed to stop. It resolves the link through the episode
  candidate instead: `find_durable_fact_for_candidate(subject)` matches the
  candidate's `_episode_signature` against the FROZEN `canonical_signature`,
  which is written **only** by the two promotion paths, so a hit means "this
  exact lesson is already durable" and this suggestion is an application of it.
  A candidate with no durable fact yet contributes no link — its early
  acceptances are evidence toward promotion, which the candidate path already
  owns. **Footgun**: match the frozen field, never a recomputed signature, and
  never let an empty target match the empty default — every unpromoted fact
  would answer for every candidate.
- **A credited fact must reach a SAVE, and the cited-credit path had none.**
  `detect_implicit_feedback` credits two different populations: linked ORIGINAL
  facts (a suggestion re-applying a durable fact), which move `linked_count`;
  and CITED retrieved facts, credited by `_apply_used_fact_feedback`, which bump
  `success_count` on the live Fact objects and record an episode mutation and
  **nothing else**. The single save was `if linked_count: self.save()`, so a run
  that credited a cited fact with no linked original fact never wrote the store
  and the credit died with the process — the normal case, since
  `suggestion_fact_ids` is empty unless the candidate already resolves to a
  durable fact. The episode ledger still recorded the mutation, so
  `learning-stats` reported reinforcements the store never received; the
  reporter and the data disagreed and the REPORTER was the honest one.
  Measured live: **0 of 88 valid GLOBAL facts had ever reached
  `success_count > 0`** while 64 carried a non-zero `access_count` (global facts
  are almost never the linked original, so theirs were the credits always
  dropped), and PROJECT facts topped out at exactly **2** — the value promotion
  writes from its two supporting episodes — so no fact among 6,613 ever cleared
  the `success_count >= 3` contribution bar and `neo contribute` was
  mechanically unreachable, not merely starved. Gate is now
  `if linked_count or touched_fact_ids`.
  **Footgun — two neighbours save for their own reasons and will mask this.**
  The no-link ACCEPTED fallback calls `add_fact`, which saves; and the janitor
  chain under `if outcomes:` ends in `if changed: self.save()`. The credit
  survived whenever either happened to fire, which is what made the bug
  load-dependent. The real modern path reaches neither: an ACCEPTED outcome
  carrying an episode candidate takes the candidate branch and `continue`s past
  the fallback, and a FIRST acceptance promotes nothing.
  `test_cited_fact_credit_survives_the_process` therefore sets `candidate_id`,
  pins the janitor to "changed nothing", stubs promotion to `None`, and RELOADS
  FROM DISK — all four load-bearing; the first two cuts of that test passed
  against the broken code, once via `add_fact` and once via the janitor. Every
  other test in `TestRetrievedFactAttribution` asserts the mutated in-memory
  object and never reloads, which is how a suite thorough about attribution
  stayed silent about persistence. **Any new credit path needs its own save,
  and a test that reads the fact back off disk.**
- **A re-accepted durable pattern is reinforced in place, not re-minted.**
  `_promote_repeatedly_supported_candidate` looks for an existing valid PROJECT
  fact at the target signature before calling `add_fact`, and on a hit folds in
  the new supporting episodes, raises `success_count` to the support count and
  adds `REACCEPTANCE_BOOST` (0.05). `add_fact`'s pre-write dedup cannot catch
  this: its signature includes the BODY, which is one episode's run-varying LM
  prose, so a third acceptance worded differently wrote a SECOND durable fact
  for the same lesson. `collapse_duplicate_signature_facts` heals that after
  the fact but keeps the richest by supporting-episode count — the NEW fact, at
  its freshly capped mint confidence — silently discarding whatever the
  original had earned. This is also the only way past the mint cap: promotion
  mints at `min(0.75, 0.4 + 0.1·n)`, so without an in-place boost a
  repeatedly-verified pattern could never reach the 0.8 contribution bar no
  matter how many acceptances it collected.
- **`durable` is a terminal candidate status.** `detect_implicit_feedback`
  calls `_record_attributed_episode_outcome` a SECOND time right after a
  successful promotion, to record the mutation. That method used to assign the
  ACCEPTED status unconditionally, so it immediately walked the just-written
  `durable` back to `supported_once` — leaving every promoted candidate reading
  `supported_once` beside a populated `promoted_fact_id`, and `learning-stats`
  under-reporting the one number it exists to report. Promotion is not the only
  writer of that field, so the guard lives in the status loop, not the caller.
- **The protection boost is bounded by evidence, because it runs per PROCESS
  START.** `demote_unhelpful_facts` (cold-start chain) adds `PROTECTION_BOOST`
  to any fact with `success_count > 0` and hit rate ≥ `PROTECTION_HIT_RATE`.
  Unbounded, that compounds once per neo invocation, so confidence measured how
  often the process started rather than how often the fact was right — measured
  on a live store, the 93 facts with any success at all averaged **0.968**
  confidence and 57 sat at exactly **1.00**, several of them throwaway drill
  prompts holding a single success against 40-odd accesses. The boost now stops
  at `min(PROTECTION_CEILING_MAX, PROTECTION_CEILING_BASE +
  PROTECTION_CEILING_PER_SUCCESS · success_count)`. Verified outcomes may still
  carry a fact above that line; the comparison is strictly `>` so protection can
  never claw back confidence it did not grant. **This is why the contribution
  banner had it backwards**: confidence was the half satisfied almost
  accidentally, and successes the half nothing could move.
- Community contribution gates are single-sourced in `store.py`
  (`CONTRIBUTION_MIN_CONFIDENCE` / `CONTRIBUTION_MIN_SUCCESSES` /
  `CONTRIBUTION_EXCLUDED_TAGS`), and eligibility is split in two:
  `is_contribution_candidate` holds the PERMANENT disqualifiers (kind is
  CONSTRAINT, or provenance is a seed/community/history feed), while the two
  numeric thresholds are the only part a fact can grow out of. The split exists
  so a caller reporting *why* a fact is not contributable can name only the gate
  that binds — `subcommands._describe_contribution_gap`. The status banner used
  to print a fixed "need 0.8 confidence + 3 successes" at facts already sitting
  at 1.00, which is this repo's own rule about never blaming a cap for an
  absence it did not cause, broken in the one line a user reads most. It also
  filtered `near` differently from `find_contributable`, so it could advertise
  facts that were never contributable at all.
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
  **The gate came back once already, as a TEST rather than a threshold.**
  `test_within_budget_run_reports_clean` called `run_learning_evaluation` against
  the REAL 500ms budget and asserted `performance_within_budget is True` — i.e. it
  asserted the ambient machine is fast, which is the identical coin-flip one level
  up. It failed CI at **534.80ms** on a commit measured at ~50ms locally (three runs
  each on branch and main, pinned `PYTHONPATH`, ≤1ms apart — so not the change under
  review), with `accepted=True`, `acceptance_failures=[]`, all twelve scenarios
  passing and every safety rate 0.0. A correctness PR was red for a reason that had
  nothing to do with correctness, which is the whole defect #183 named. Every OTHER
  test in that class is deterministic because `_over_budget` forces
  `latency_ms_max = 0.001`, something no machine can meet; the clean-report case now
  goes through `_within_budget` (1e6 ms), something no machine can exceed. **Rule:
  no test in this class may depend on how fast the machine running it is** — pin the
  budget in whichever direction the case needs, and never mock the clock (a patched
  timer tests the patch).
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
  **Footgun — that export belongs in `~/.zshenv`, never `~/.zshrc`.** The gate is
  a plain `os.getenv` (`observer.py`), read by whichever process runs `neo`, and
  most of those are NOT interactive shells: an editor plugin, a CI step, a git
  hook, an agent tool call. zsh sources `.zshrc` for interactive shells only, so
  an export there leaves every programmatic invocation autostarting the observer
  while the terminal prints `0` and looks correct. Measured directly: with it in
  `.zshrc`, `zsh -c`, `zsh -lc` and a non-interactive tool call all read empty;
  only `zsh -ic` read `0`. **Verify without `-i`** — that flag forces the single
  mode that works, so the obvious check passes for the wrong reason and confirms
  nothing.
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
     **And the order WITHIN a language is source-before-tests, then depth, then
     alphabetical** — the test key being the load-bearing one. Depth alone is a
     centrality proxy that INVERTS on the conventional Python layout: with
     `src/<pkg>/…` beside `tests/…` every test sits at depth 2 and every source
     file at depth 3, so the whole test tree sorted ahead of the whole source
     tree and `--max-files` was spent before one source file was examined.
     Measured here: 131 Python files at depth 2, and the catalog came out
     **100% tests** while the build reported success — the cap had genuinely
     bound and the report was truthful about that, but nothing said the files
     it kept were all tests (#213). This module already treats "tests outrank
     the source they test" as a failure mode — `_embed_chunks` embeds a
     structured summary rather than the raw body precisely because assertion
     strings carry a query's keywords verbatim — but that mitigation runs at
     EMBEDDING time and so never got a chance; selection had already spent the
     budget. Tests are DEMOTED, not excluded: they fill the slots source does
     not need, so a test-only repo still indexes. `is_test_path` is IMPORTED
     from `context_gatherer`, never restated — a second copy of that rule would
     agree with the first only by coincidence, and its careful cases
     (`testdata/` and `testing/` are ordinary source; `Foo.Tests/` is not) are
     exactly what a re-implementation loses.
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
- Persistent eligibility walk (`index/walk_cache.py`, `eligibility.DirectoryListing`).
  The #208 walk is **kept on disk too**, in the same `.neo/`, because once the
  content index stopped rebuilding it became the largest single item in a warm
  call: 4.64 s on m365dotnet to re-derive that 9,348 files are eligible, on every
  invocation, in a repository that had not changed (#210). Warm walk now 0.16 s;
  the canonical M2 battery went 15.63 s → **8.57 s** median with byte-identical
  selection on all six prompts and byte-identical `rank_mine_eval` on all three
  flagships.
  - **The cost is the pattern matching, not the filesystem, and that decides the
    whole design.** Measured before the cache existed, on m365dotnet: the full
    walk **6.85 s**; the same traversal with the per-FILE ignore test removed
    **0.80 s**; `stat` over all 9,378 admitted files **0.10 s**. So 6.05 s of a
    6.85 s walk is `should_ignore` (11,219 calls), and caching the syscalls
    would have saved almost nothing. What is stored per directory is therefore
    the VERDICTS — which subdirectories survive, which filenames survive, and
    the exclusion counts — with the directory's stamps saying whether they hold.
  - **Directory mtime is the right key for exactly one reason**: on every POSIX
    filesystem it moves when an entry is created, deleted or renamed inside that
    directory, and does NOT move when the content of a file inside it changes.
    An edit can change what a file SAYS; it can never change whether it is
    eligible.
  - **mtime alone is not enough, because mtime is forgeable.** `touch -r`,
    `tar -x`, `rsync -a` and every snapshot restore write a directory's mtime
    back to a recorded value, so a restore that adds or deletes a file can land
    on exactly the mtime the cache holds and be reported `warm` — reproduced
    with two lines of `os.utime`, found by the fresh-verifier pass. `ctime_ns`
    is stored beside it: the inode change time moves on any metadata change, no
    API restores it, and it arrives in the same `stat`. On Windows `st_ctime` is
    a CREATION time and hence constant, which makes the extra comparison a no-op
    there rather than a false invalidation.
  - **Sizes and mtimes are never remembered.** They come fresh from the `stat`
    the walk owes its callers anyway (0.10 s), because the content index uses
    them as ITS freshness stamp — serving a remembered mtime would make an
    edited file look unedited and turn one cache's staleness into another's.
    `TestStampsAreNeverCached` pins it, mutation-verified.
  - **A `.gitignore` edit invalidates by CONTENT, not by any timestamp.** No
    directory's mtime moves when a pattern file is edited, so every stored
    verdict looks current while every one of them may now be wrong. The
    signature hashes the effective pattern list (shared defaults + the repo's own
    `.gitignore`/`.ignore`), plus a `MATCHER_VERSION` for the case the patterns
    are identical and `should_ignore` is not. Cost of an edit: one full walk,
    then warm again.
  - **`os.walk(followlinks=False)` had been doing work no one had named.** The
    traversal now recurses by hand, one directory at a time, so that flag
    protects only the single read it is passed — a link to an ancestor became an
    infinite descent and a link to `/` walked the machine. The refusal to
    descend into a symlinked DIRECTORY is restated explicitly and is not gated by
    `skip_symlinks`, which is about whether a symlinked FILE is delivered.
  - **A directory modified within `RACY_WINDOW_NS` (1 s) of being read is not
    trusted.** Timestamp granularity is not always finer than the interval
    between two events — HFS+ stamps whole seconds — so a directory read at E
    with mtime M can be modified afterwards into the same bucket whenever
    `E - M` is under one tick. Git carries the same guard under the name "racily
    clean". Its test had to pre-create `.neo/` to mean anything: writing the
    cache creates that directory, which moves the repository ROOT's mtime, so
    the call after a first-ever call re-lists the root whatever the guard does —
    the first cut of that test passed with the guard deleted.
  - **`extra_ignores` is never served from a cache and never writes one.** A
    caller's patterns are appended after the repo's own and gitignore is
    last-match-wins, so a `!negation` there can re-include what a stored verdict
    excluded — the stored verdict is not a stale answer, it is an answer to a
    different question. Nothing in Neo takes that path today (`--exclude` is
    applied after the walk); the guard exists so a future caller gets a correct
    answer rather than a fast wrong one. Reported as mode `bypassed`.
  - **JSON, not SQLite — the opposite conclusion from the same question.** Every
    directory is validated on every call, so the file is read whole every time,
    which is what JSON is for and what the semantic catalog beside it already
    uses. 507 KB and ~10 ms for m365dotnet. The neighbouring content index went
    to SQLite because a query touches ten terms of a few hundred thousand; the
    access shape decides, not the size.
  - The verdicts are the IGNORE layer only, so one cache serves every consumer:
    `--exts` / `match_globs` / `max_file_bytes` / `skip_symlinks` are applied on
    top, per call. `--index` and `architecture_metrics.compute` (which runs on
    the outcome-detection path of every real invocation) go through the same
    cache, so `--index` warms the walk as well as the catalog.
  - Degradations are loud and none is fatal: a corrupt, truncated, malformed or
    foreign-signature cache is discarded with a warning and the walk runs in
    full; an unwritable `.neo/` costs a warning and nothing else. A malformed
    ENTRY discards the whole file rather than the entry, because half a cache is
    half a repository, silently. `--dry-run` names which of cold / rebuilt /
    incremental (N of M directories) / warm / bypassed happened, in the
    selected-files block and in the `--json` payload's `walk_cache` key. A cold
    walk announces itself BEFORE it starts, since a first call on a large
    repository is seconds of otherwise-silent work.
- Persistent content index (`index/content_index.py`, `index/freshness.py`). The
  BM25 corpus above is **built once and kept on disk**, in the repository's own
  `.neo/` beside the semantic catalog, not re-derived per call. Rebuilding it per
  invocation was the entire cost of a Neo call on a large repo: the canonical M2
  battery on m365dotnet (9,348 eligible files) measured a **53.47 s** median wall
  and 1.95 GB peak RSS, against 18.50 s / 1.45 GB with the store (#195). Selection
  is UNCHANGED — byte-identical selected files and rank order on all six battery
  prompts, and byte-identical `rank_mine_eval` MRR / R@k on all three flagships
  (neo 0.712, aieweb 0.728, m365dotnet 0.669, 50 cases each).
  - **SQLite, not the catalog's JSON, and the access shape is the reason.** The
    catalog is read whole (every embedding participates in every query); a keyword
    query touches ten terms out of a few hundred thousand, so a parse-whole format
    would spend the warm budget deserializing postings nothing asked for. Index
    artifacts only — postings, doc lengths, hashes, a tokenizer/schema signature —
    never a file body; delivery still reads whole files fresh from disk. Cost is
    disk: 109 MB for m365dotnet, 5 MB for neo.
  - **The cheap stamp decides what to HASH, never that a file changed.** size +
    mtime come free from the walk's existing `stat` (`EligiblePath.mtime_ns`);
    only a content-hash mismatch counts as a change, so `touch` re-stamps and
    re-tokenizes nothing and is reported as `touched`, separately from `changed`.
    `UNRECORDED = -1` disables the cheap path for a store that persists hashes
    alone — and it is checked EXPLICITLY, because the first cut stamped both the
    store and the candidate with the sentinel and `-1 == -1` reported every
    changed file as unchanged.
  - **Eligibility arrives from the #208 walker and is never recomputed here** —
    `test_the_module_does_not_walk_the_filesystem` fails on an `os.walk`/`glob` in
    the module. A file the walker stops admitting (newly gitignored, deleted) has
    its postings dropped on the same pass, so a stale index cannot answer with an
    excluded file. Correspondingly the gatherer now walks ONCE, unfiltered, and
    applies `--exts`/`--include`/`--exclude` to the result: the index is a
    property of the REPOSITORY, and letting one `--exts py` call prune it would
    force the next call to rebuild. `--exclude` no longer prunes a directory
    subtree (it excludes every file under it via `should_ignore`, which matches a
    non-final component); the names that make pruning matter are in the walker's
    shared default list.
  - **Parity is enforced, not asserted.** `K1`/`B`/IDF are imported from
    `neo.memory.bm25` and the document is the same expression `FileIndex` used, so
    `TestParity` can score one corpus both ways and compare to floating point.
    Query-term MULTIPLICITY is preserved — deduping the query before hitting the
    postings table is a silent ranking change, since the scorer this replaces
    iterated the token LIST. **A filtered call gets filtered statistics**:
    `scores(prompt, candidates)` takes N, df and avgdl from `candidates`, not
    from the whole repository. Repo-global stats were the first cut and are the
    better IR design in the abstract; they are also a re-rank, and a
    fresh-verifier pass caught it — unflagged runs matched main exactly while
    `--exts py` changed all 25 selected lines. Now byte-identical under
    `--exts`, `--exclude` and `--include` as well.
  - Degradations are loud and none is fatal, and the **except-clause ORDER is the
    whole mechanism**: `sqlite3.OperationalError` is a SUBCLASS of
    `DatabaseError`, so `except DatabaseError` written first catches "database is
    locked" and runs the corruption handler — which `os.unlink`s a perfectly good
    store. Reproduced: the index was deleted AND the peer's committed transaction
    went into an unlinked inode and vanished with no error anywhere. Two Neo
    invocations in one repo is ordinary (an editor plugin and a shell), not
    exotic. So: **locked / read-only / full** → serve this call from memory via
    `FileIndex` and say so; **corrupt** (a `DatabaseError` that is NOT an
    `OperationalError`) → delete and rebuild, detected while OPENING because the
    memory fallback is permanent and a store that merely failed to open would pin
    that repo to the full per-call rebuild forever; **tokenizer/schema** bump →
    wipe and rebuild (no per-file hash can see it — the files did not change, the
    tokenizer did). `cold`/`rebuilt` are reported separately though both read
    everything: only one means something went wrong, and the corruption flag is
    CONSUMED, or a reused instance rebuilds on every refresh forever.
  - **A warm call opens no write transaction.** Rewriting the unchanged signature
    unconditionally made every invocation a writer, so two ordinary overlapping
    calls contended on the steady-state path rather than only during a rebuild —
    which is what made the clause-order bug reachable in normal use.
  - **A file that cannot be hashed keeps its path tokens.** `chmod 000` used to
    delete it from the corpus permanently, while the per-call index still ranked
    it on its name (its content simply read as empty). Permission was withdrawn
    from the CONTENT; the name is still a real name in the repository. The empty
    hash is safe as a comparison value because stamps are keyed by path —
    `"" == ""` is only ever asked of one file against its own previous state.
  - **The vocabulary is not preloaded to answer a query.** Reading every `terms`
    row to resolve ten of them cost a few hundred thousand rows on the warm path
    this module exists to make cheap; term ids resolve per query, and the full
    map is loaded only when writing.
  - Cold build is bounded and ANNOUNCED before it starts, with progress every 250
    files — 122 s for m365dotnet's 9,348, 2.3 s for neo's 307. `--dry-run` names
    which of cold / rebuilt / incremental (N files) / warm / memory happened, in
    the selected-files block and in the `--json` payload's `content_index` key
    (`--json` implies `--quiet`, so the stderr note is suppressed on exactly the
    path a machine consumer reads).
  - **M2's 500 MB target is NOT reachable from file selection and never was.**
    Warm profile on m365dotnet: imports 1.2 s, eligibility walk 4.6 s, content
    index refresh 0.4 s + scores 0.1 s, `_history_boost` 2.9 s **and +1.26 GB**.
    The RSS is the FactStore (152 MB of JSON with 768-dim embeddings inflating to
    ~1.3 GB of Python objects) — Goal 1's 1.43 GB baseline was never the
    gatherer's. What remains of the warm wall-clock is the walker and the memory
    system, in that order.
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
- **One retrieval front door** (`context_gatherer.gather_context`): every
  invocation goes through ONE pipeline with one priority order — (1) paths the
  prompt named, PINNED; (2) `--include`, pinned per ruling 1 with the scan
  continuing; (3) keyword BM25 over the persistent content index; (4) the
  embedding catalog, re-ranking and supplementing (3) whenever it exists.
  `gather_context_semantic` — a second gather function with its own candidate
  list, its own budget arithmetic and no idea what the prompt had named — is
  **deleted**; `--semantic` is now a HINT carried on `GatherConfig.semantic`
  that raises the catalog's weight (`SEMANTIC_WEIGHT` 1.0 →
  `SEMANTIC_HINT_WEIGHT` = `CONTENT_WEIGHT` = 3.0) and its retrieval depth
  (`SEMANTIC_HINT_DEPTH` = 3×). Which retrieval strategy you asked for must not
  decide whether a guarantee applies, and it did: the semantic lane ignored
  prompt-named paths entirely.
  **Stage 1 is a pin, not a boost.** `EXPLICIT_PATH_BOOST` makes a named path
  rank first, which is a weaker claim than "present" — the file could still be
  windowed into a fragment, and a prompt naming more files than the adaptive
  limit admits lost the last-named ones. `resolve_explicit_paths` now pins them
  on the same terms `--include` uses. The boost STAYS on the ranking: a file can
  only be pinned if the walk found it, so the boost covers the candidates the
  pin pool never held (an `--exts`-narrowed list).
  **The pin block cannot spend the whole ceiling** (`PIN_BUDGET_SHARE` = 0.5).
  Ruling 1 is "the named files AND keep scanning", and funding pins to the last
  byte satisfies the first clause by deleting the second. Measured on the M2
  battery after #214 merged: the prompt naming
  `src/Parslee.M365.Api/Program.cs` (442,867 bytes) pinned it, spent 299,959 of
  the 300,000-byte default, and delivered **one file** — the whole context was
  that file — where pre-#214 main delivered 22 and the fix delivers 30. The
  reserve binds only when pins would take more than half; with nothing else
  eligible it would fund nothing, so it does not apply and the pin arrives
  whole. The held-back cut is marked and announced, which the ruling permits
  ("whole, or with an explicit marker").
  **Each budget cap is charged for what IT removed.** `dropped_by_file_cap` is
  counted before the byte cap cuts and `dropped_by_byte_cap` after, and both are
  reported when both bind. Deriving the file cap's verdict from the full
  candidate list made the run print `pinned files filled the file budget
  (--max-files=30)` with 29 of 30 slots free and the BYTE ceiling holding the
  count at one — raising the named knob provably changed nothing, which is this
  repo's own rule about never blaming a cap for an absence it did not cause,
  broken inside the goal whose subject is selection truthfulness. In the
  scan-delivered-nothing branch the byte cap is tested FIRST, because it is
  applied second and therefore holds the margin.
  **Delivery is one entry per file, read whole from disk.** Chunking survives
  only as a RANKING internal — it chooses WHICH region of an over-budget file
  arrives, never how many entries a file contributes. `--max-bytes` is
  apportioned max-min fair (`text_budget.apportion`) across the selection rather
  than spent greedily in rank order, so a file's SIZE no longer decides how many
  other files reach the model; `MIN_FILE_SHARE_BYTES` (512) is the floor below
  which the ceiling reduces the file COUNT instead, and says by how many.
  Measured branch vs main at `d5adcbc`, 50 git-mined cases × 3 flagships:
  **MRR and R@k byte-identical in every cell**, while mean distinct files
  delivered per query went 18.4→28.9 (neo), 22.6→28.7 (aieweb), 22.1→29.3
  (m365dotnet) — **0 files lost, 1,190 gained, a strict superset in 150 of 150
  cases**. The metrics do not move because the gain sits below rank 10. On the
  M2 battery, within-prompt repeat entries went **45 → 0**: #197's "chunks
  counted as files" is now impossible rather than merely reported. M2 median
  wall 9.10 s → 8.97 s, peak RSS +0.24% — inside main's own 7.6–10.4 s spread.
  Full tables: `docs/goal8-front-door-measurements-2026-08-13.md`.
  **What the flagship M1 numbers CANNOT tell you**: none of neo, aieweb or
  m365dotnet has a `.neo/index.json`, so stage 4 returns `{}` on all 300 of
  those runs and is inert in both arms. Read "no regression" as measured and
  "no concept-shaped win" as unmeasured-here, not as absent.
  **Stage 4 is measured separately and the result is about robustness, not
  quality.** With a catalog built on neo, the OLD `--semantic` lane scored
  **MRR 0.000 / R@10 0.000 on all 50 cases** — it returned 1,224 files and
  **every one was a test file**, because it read `index.retrieve()` + MMR with
  none of the pipeline's judgement (no test demotion, no BM25, no pin) and
  because `neo --index` had built a catalog of 99 files that are 100% tests.
  That second half is a real upstream defect, **#213**:
  `ProjectIndex._select_files` ranks shallowest-path-first, so on `src/<pkg>/…`
  + `tests/…` every test is depth 2 and every source file depth 3 — 105 Python
  files at depth 2 here, first non-test at rank 102, `--max-files` default 100.
  Through the front door the same broken catalog yields 0.705 / 0.708, because
  it is one channel of four. **#213 is now FIXED** — selection ranks source
  before tests (see index invariant 1) and a rebuild of this repo's catalog went
  from 82% tests / 52 files to **100% source / 94 files**.
  **The deferred re-measurement has now been RUN, and it reverses the sign.**
  `tools/rank_mine_eval.py`, 50 git-mined cases on this repo at `d6ab226`,
  `--no-git`, clean tree, 0 failed cases, both arms the same commit and the same
  fresh catalog. `--semantic` against flag-off: MRR **0.778 → 0.841**, R@10
  0.708 → 0.767, H@1 0.680 → 0.760. Paired by case: **11 better, 2 worse, 37
  tied, two-sided sign p = 0.022**. The old **−0.007 MRR is superseded** — it
  was a property of the 100%-test catalog it was measured against, not of the
  weight, and that catalog no longer exists.
  **The win is the WEIGHT, not the depth.** `--semantic` moves two things, and
  they were separated by pinning `SEMANTIC_HINT_WEIGHT = 1.0` while
  `SEMANTIC_HINT_DEPTH` stayed at 3×: that arm scores MRR 0.769, *below*
  flag-off's 0.778 (1 better / 3 worse, p = 0.625). Retrieving 3× deeper into
  the catalog buys nothing on its own; what pays is weighing the catalog as
  heavily as the keyword index.
  **The weight is a broad plateau, and 3.0 is kept rather than re-tuned.**
  Sweep at 50 cases (MRR): 1.0 → 0.769, 2.0 → 0.803, **3.0 → 0.841**, 6.0 →
  0.853, 9.0 → 0.832 — an inverted U, degrading by 9.0. 6.0 is NOT
  distinguishable from 3.0 (paired 6 better / 2 worse of 50, p = 0.289), and
  its R@1 is lower (0.405 vs 0.430). Moving the default to the sweep's argmax
  would be selecting on the same 50 cases that produced it; the honest reading
  is that `SEMANTIC_HINT_WEIGHT = CONTENT_WEIGHT` now has evidence behind it
  rather than only a symmetry argument. **Still one repo and 50 cases**, and
  every caveat in the harness's own docstring applies — absolutes are upper
  bounds (the corpus has already seen the answer), the mined population is
  filtered upward, and `_history_boost` stays live. Direction survives those;
  magnitudes should not be quoted as precise. `aieweb` and `m365dotnet` still
  have no catalog, so the cross-repo half of this remains unmeasured. Full
  tables and setup: `docs/semantic-lane-remeasurement-2026-08-30.md`.
  **The renderer's per-file cap is NOT the front door's** and the two are
  deliberately still separate: `engine._render_context_files` keeps its 3,000
  character cut for unpinned files, with its own marker. Collapsing them would
  either grow every prompt by ~60% or push a relevance-unaware head-cut into the
  one place that currently keeps the relevant region.
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
- **Claude Code components live at the PLUGIN ROOT, never in `.claude-plugin/`.**
  Claude Code discovers `agents/`, `commands/`, `skills/` and `hooks/` at the
  plugin root and reads only manifests (`plugin.json`, `marketplace.json`) out
  of `.claude-plugin/`. Nested there they are silently not loaded: the plugin
  installs, `claude plugin validate` passes, and nothing fires — `claude plugin
  details` is the only check that proves a component loaded. This repo shipped
  the wrong layout, and so did CAR, which is why
  `test_the_manifest_directory_holds_no_components` fails on any non-manifest
  entry rather than merely asserting the components exist at the root (a stray
  copy left behind keeps every other assertion green).
- **The edit-recording hook (`neo hook record`, `neo/hook.py`)** is a
  `PostToolUse` hook on `Edit|Write|MultiEdit|NotebookEdit` that appends one
  line to `~/.neo/sessions/host_events.jsonl`: tool, path, host cwd, and HEAD at
  edit time. It exists because acceptance is otherwise inferred from a repo-wide
  git diff on the NEXT invocation, which cannot see an edit that was never
  followed by another Neo run. Three rules, two of them mutation-pinned:
  (1) **it never fails** — `run_hook` returns 0 on every path *by construction*,
  including an unknown action, because a hook exiting non-zero reports an error
  against a tool call that already succeeded. That means `except BaseException`,
  not `except Exception`: `KeyboardInterrupt`/`SystemExit`/`GeneratorExit` walk
  straight out of the narrower handler, and the failure-REPORTING path is
  guarded too (a closed stderr, or a custom `__str__` that raises, would defeat
  the guarantee from inside the code announcing it). Both gaps were found by
  running `neo` against this module and by no test — every case in
  `TestNeverFails` raised an `Exception` subclass, so none of them could
  distinguish the two handlers. Only SIGKILL and a library `os._exit` remain
  outside its reach, and the docstring says so; (2) **it stays cheap** —
  `import neo.cli` is 0.04s but `neo --version` is 0.36s, and the whole
  difference is `FactStore` construction, so `cli.main` dispatches to it before
  argument parsing, the update check and the observer autostart
  (`test_hook_stays_off_the_slow_path` fails if anything moves above it);
  (3) **paths, never contents** — `tool_input` carries the text being written.
  Opt out with `NEO_HOOKS=0`. `HOOK_LEDGER` captures `Path.home()` at import, so
  it is registered in `conftest.HOME_PATH_CONSTANTS`.
  **`collect_outcomes` reads the ledger** (`_load_host_edit_events`), unioning
  host-recorded edits into the git-derived `changed_files`. It ADDS evidence and
  removes none: git still reports everything committed or dirty.
  **Attribution is by `file_path` and never by `cwd` or `head`** — those name the
  directory the HOST was launched in, not the repository the edited file belongs
  to. Measured: an edit to a scratch repo, made from a Claude Code session rooted
  in the neo checkout, recorded neo's OWN head. A record is ours when its path
  resolves inside `codebase_root`.
  The read is hoisted OUT of the per-session loop for the same reason
  `_get_working_tree_changes` was — retention means many pending sessions, and a
  per-session re-read puts a linear cost on the request hot path
  (`test_the_ledger_is_read_once_per_call_not_once_per_session`). The rotated
  `.1` generation is read too: rotation moves the RECENT records there and leaves
  the active file nearly empty, so reading only the active file would lose
  exactly the window this exists to protect. A malformed LINE is skipped rather
  than discarding the file — the ledger is append-only, so a torn final write is
  the expected corruption.
  **The concrete gap it closes is the UNTRACKED new file.**
  `git diff --name-only HEAD` lists tracked modifications ONLY, so a suggestion
  to create a new file — which `suggestion_is_verifiable` explicitly admits as
  legitimate — was invisible to detection until someone committed it. Closing
  that needed BOTH halves: the ledger surfaces the path, and
  `_get_file_diff_since` gained a third source (`git diff --no-index` against
  `os.devnull`, gated on `_is_untracked`) so the classifier can actually see the
  content. With only the first half the outcome was UNVERIFIED, which mutates
  nothing — a file reported as changed that no diff could confirm. `--no-index`
  exits 1 when files differ, which is the normal result, so only stdout decides.
  Both halves are separately mutation-pinned in `tests/test_host_edit_ledger.py`.
  **Every test in that file runs against a DIRTY tree**, because a spotless
  working tree is the one state neo is never invoked in — the mistake that let
  the per-session retention bug ship.
  **Footgun — the plugin now has a hard CLI version floor.** The plugin updates
  from this repo; the CLI comes from PyPI. On a `neo` predating the subcommand,
  `neo hook record` is an argparse error exiting **2**, the one code Claude Code
  treats specially — the identical defect filed against CAR as
  Parslee-ai/car#993, reproduced here while writing that issue. Floor documented
  in README; `hooks.json` deliberately does NOT wrap the command in
  `|| true`, because that reintroduces the shell dependency (and `2>/dev/null`
  is invalid in `cmd.exe`) that choosing a subcommand was meant to avoid.
- **Host adapters must stay in parity.** There are TWO checked-in integration
  surfaces, and they are easy to miss: the Claude Code plugin (manifests in
  `.claude-plugin/`; agent, 6 slash commands and the hook at the REPO ROOT —
  see the layout rule above) and **`plugins/neo/`** (Codex CLI plugin + 6
  skills, manifest at
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