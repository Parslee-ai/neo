# Learning loop: outcomes, sessions, reinforcement, contribution, benchmark

Detailed engineering notes moved verbatim from `CLAUDE.md` (2026-10-12). `CLAUDE.md` keeps the short rules; this file keeps the reasons, measurements and footguns behind them.

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
- **Legacy per-suggestion facts are INVALIDATED on every cold start**
  (`store.retire_legacy_suggestion_facts`, `invalidation_reason=legacy_unverified_suggestion`).
  It stamps `last_accessed` before invalidating: `purge_dead_facts` runs later in the same cold
  start and hard-deletes tombstones untouched 30+ days, and every live match was 58–206 days
  stale, so without the stamp the audit tombstone never reached disk. It is kept 30 days, then
  purged like any other.
  Before episodes replaced immediate fact-writing (`412a174`), every feature suggestion was
  written straight to the store as an unverified DECISION — a stable kind that bypasses recall
  decay and that `update_recall` never stamps — and the since-bounded protection ratchet had
  already lifted them to 0.9–1.0. **Measured on a live install: 238 such facts were 49% of all
  memory injected into prompts over 30 days (1,217 of 2,477 inclusions), ~4% of those used in
  reasoning**, the most-injected being drill prompts. Transcript-mined lessons were 1.6%.
  The match is structural, not a date (first tag `feature`, subject `feature: `, body
  `Reasoning: `, no `canonical_signature`), because installs kept minting the shape until they
  upgraded. **Reclassifying them to a decaying kind was built first and is WRONG — do not
  reintroduce it.** The stale prune skips confidence >= `STALE_MAX_CONFIDENCE` (0.4) and any
  fact with a success, so 232 of 239 would have sat hidden on disk forever; and a decayed
  fact's similarity term goes to ~0 while `success_bonus` is similarity-independent, so the 93
  with a recorded success became constant-score fillers taking the same slots on every query.
  **That second half is a live ranking hazard for ANY decaying fact with successes**, not just
  these — today no other fact has one, which is the only reason it is not biting.
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
