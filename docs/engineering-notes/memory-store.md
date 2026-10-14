# Memory store: hygiene, promotion gates, transcript sources, retrieval, storage

Detailed engineering notes moved verbatim from `CLAUDE.md` (2026-10-12). `CLAUDE.md` keeps the short rules; this file keeps the reasons, measurements and footguns behind them.

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
    revision. **The revision compared is what the accepted change was APPLIED
    ON TOP OF** (#254): `Outcome.applied_on_revision` — HEAD of the checkout
    holding it uncommitted, or the parent of the FIRST local-branch commit
    whose patch CARRIES the suggestion (same overlap test as ACCEPTED; merely
    touching the path let an unrelated sibling commit supply its own parent) —
    plus `carrier_revision`, that commit. Both are stored PER SUGGESTION in
    `episode.outcome_details["applied_on"]` (one episode can mint several
    candidates; a free-form dict, so no schema bump — a bump would make older
    running neo processes quarantine new records), read through
    `LearningEpisode.applied_on(suggestion_id)`, which returns "" (fail closed)
    when absent and never falls back to ask-time HEAD; pre-#254 episodes read
    their first passed `user_acceptance` revision. The entry is FILL-ONCE:
    `replay_linked_feedback` re-collects, and after a rebase (new carrier) or
    with HEAD moved under a still-dirty change an overwrite would turn a
    refused group promotable with no new evidence; a recorded "" base may be
    filled later ONLY from a found carrier (history), never from a dirty
    re-read (HEAD at replay time is a guess). Before the span gate,
    `FactStore._sitting_bases` groups supporting acceptances into SITTINGS as
    connected components (union-find): linked when they share a base (keeps
    labelling monotonic — without it a third acceptance continuing one of two
    parallel ones relabelled only its own component and promoted), when one's
    base IS the other's carrier (A committed as C1, B applied on C1), or one's base's PARENT is
    the other's base and the other was seen uncommitted (A edited at H and
    seen dirty, committed as C1, B on C1 — the usual editor flow, which has no
    carrier; hence the stored `parent`). Each sitting counts as ONE revision.
    **Components, not case-by-case folding**: two rounds of ordered folds each
    left a chain that escaped (a dirty B on a dirty A's commit; a parent link
    followed by a carrier link). Folding is fail-safe: a real second sitting
    applied directly on the first's commit, or on one unrelated commit atop a
    still-uncommitted first application, is grouped too and waits for a later
    recurrence. The promote path and `learning-stats` share it. Ask-time HEAD was wrong: a CAR Lattice node answers from a
    main checkout that sits still while peers commit on their own branches, so
    every episode shared one revision and nothing could promote. **The landing
    commit is ALSO wrong, and was tried first**: shas are unique per commit, so
    one sitting seen dirty by one run (H) and committed before the next (C)
    read as two revisions and promoted — the exact same-operator case the gate
    exists for. By base both are H, and parallel worktrees branched from one
    base committing one fix both read that base. Pre-#254 verifications hold
    ask-time HEAD in the same field, which equals the base for a change applied
    in place, so no migration. `evaluation.py` sets the field explicitly; any
    new producer of ACCEPTED outcomes must too, or it silently never promotes.
    Accepted costs: one lesson applied across several files in one sitting
    promotes nothing; and a known limit — an uncommitted acceptance takes HEAD
    at COLLECTION time, so one sitting collected either side of an unrelated
    commit can still read as two bases. **Acceptance detection also reads
    other checkouts**: every local branch (`--branches`, never `--all` — a
    fetched teammate commit is not an acceptance), every linked worktree's
    dirty files (only those written after the suggestion, since agents keep
    work in progress uncommitted in the very file they asked about), untracked
    files and host-ledger edits (attributed to the deepest checkout holding the
    path). That wider evidence only resolves paths Neo SUGGESTED; INDEPENDENT
    detection stays on the checkout's own HEAD and tree, or every agent
    worktree's edits would become candidates, each costing a diff fork. Separate
    clones share no refs and stay invisible. That
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
    **Recall deliveries lift it too** (`lift_probation_by_recall`, cold start,
    before the prune). `neo recall` is where mined lessons reach Claude, but it
    writes nothing, so it never touched `access_count`: measured on the live
    store, 76 of 112 transcript lessons were pruned at day 3 with zero accesses,
    and all 13 that got two accesses survived. The mod records each delivery as a
    `recall_delivered` ledger event (ids only); a `transcript-derived` lesson
    (never `imported:github-pr`) delivered in ≥ `PROBATION_PROMOTE_ACCESS`
    distinct sessions leaves probation tagged `recall-lifted`. Derived from the
    ledger every cold start (idempotent, no watermark; 18 ms on a full 2×8 MB
    ledger). **Deliveries are NOT accesses**: `demote_unhelpful_facts` reads
    access as retrieved-but-never-accepted, and a recalled lesson has no
    acceptance signal, so counting them would trade the probation cliff for the
    demotion one. **Delivered ≠ helped**, so a `recall-lifted` fact earns time,
    not permanence: its `last_accessed` follows its latest delivery and
    `prune_stale_facts` drops it after `STALE_MIN_AGE_DAYS` without one, with no
    confidence gate (recall only delivers ≥ `STALE_MAX_CONFIDENCE`, so the gate
    would make it immortal). Probation removal is one-way in `_reconcile_fact`.
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
- Retrieval: `rank_score = recall_decay(sim)·confidence + success_bonus·effectiveness_f
  + provenance_bonus`. `memory.models.rank_score` is the single source of truth — if you
  change the formula, audit `ContextAssembler._score_facts` too. Cosine is batched via
  `math_utils.batched_cosine`. Hybrid: 0.7·dense + 0.3·BM25; half the result slots ranked
  by `rank_score`, half by raw cosine. CONSTRAINT/ARCHITECTURE/DECISION and the
  `seed`/`community`/`synthesized` tags bypass decay. Branching prompts (CHAIN/SPLIT)
  get per-branch retrieval via `memory.query_routing`; each surfaced EPISODE pulls up to
  2 peer episodes from the same session.
  - **Recall decay counts in `DECAY_UNIT_DAYS` (30), not days** (`models.rank_score`,
    `update_recall`). Counted in days — the paper's companion-chat setting — a
    never-recalled lesson at cosine 0.6, confidence 0.6 ranked 0.028 at 3 days and
    0.0005 at 7, while a non-decaying seed at the same cosine and confidence kept 0.36. On a live store 65% of
    685 transcript lessons scored < 0.1 at cosine 0.6, so the memory source with
    the highest measured use rate (24-26% of injections used) got 19 of 736 engine
    slots in 21 days. **Footgun, measured when fixing it**: the old decay was also
    the only thing holding back `history` (git commit subjects, ~2% used); slowing
    decay alone moved 27 → 117 of 180 replayed slots to commit subjects and left
    lessons flat (10 → 9). It only works together with the next rule. Stored
    `g_n` values are reinterpreted, not migrated: on the live store every
    transcript lesson still had the default 1.0, so nothing learned is lost. 30
    is a calibration against the prune windows, not a derived constant.
  - **Only lessons are offered as learned memory** (`models.carries_lesson`, `NO_LESSON_TAGS` =
    seed / community / outcome / history, plus legacy suggestion facts). Applied to
    `ContextAssembler`'s valid layer (the engine prompt) and to `neo recall`, which
    additionally drops the host's own memory files. The facts stay in the store
    for their other uses, and the constraint / session / known-unknown layers and
    supersession annotations are untouched — this governs the valid layer only.
    It also shrinks `valid_facts`, which feeds the reasoning-tier memory signal
    (`signal_from_facts`) and citation credit. Replayed over 30 recent real objectives with both
    changes, lessons went from 10 to 57 of 180 slots; the rest were the
    operator's Claude memory, the one injected source with above-floor use
    (7.7%). **Honest limit**: the layer still has no relevance floor (`scored[:k]`,
    `at_least_one=True`), so on a store with lessons it is never empty; filler
    is replaced by the most similar lesson, weak or not. A floor needs its own
    calibration on engine-length prompts (recall's 0.45 was calibrated on typed
    prompts). The community fallback, which injected token-matched community
    patterns when the store returned nothing (0 of 141 community injections ever
    used), was removed. **Cost shift, accepted on purpose**: `valid_facts` feeds
    `signal_from_facts`, so a store with no lessons (a fresh install, a project
    with only seeds and history) now reads `pattern_count = 0` = novel →
    higher effort and, with CAR reachable, multi-agent. Before, seed filler made
    every prompt look "familiar" (86 of 93 recent runs went fast). Live, the
    shift runs the other way for stores with lessons: dropping low-confidence
    `outcome` facts raises the average. **Not filtered, deliberately**:
    `FactStore.retrieve_relevant` keeps every valid fact — Lattice `memory:`
    serves commit history by design (it strips the patch from commit bodies),
    and the gatherer's episode-history boost reads only `file:`-tagged episode
    facts, which decay exactly like history, so the unit change does not shift
    their balance. Seed and community facts now reach no prompt; their
    ingesters are candidates for removal.
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
