# Context selection, retrieval front door, prompt rendering, text_budget

Detailed engineering notes moved verbatim from `CLAUDE.md` (2026-10-12). `CLAUDE.md` keeps the short rules; this file keeps the reasons, measurements and footguns behind them.

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
  the old semantic-lane gather — a second gather function, since deleted, with its own candidate
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
  **The deferred re-measurement has been RUN across all three flagships, and
  the honest answer is NULL — the semantic lane's value is a function of
  CATALOG COVERAGE, not of the weight.** `tools/rank_mine_eval.py`, 50
  git-mined cases per repo, `--no-git`, clean trees, 0 failed cases, each
  against a freshly built post-#213 catalog (100% source in all three, across
  Python, TS/TSX and C#). `--semantic` against flag-off, paired by case:

  | repo | MRR off → on | win/lose/tie | p | catalog coverage |
  |---|---|---|---|---|
  | neo | 0.778 → 0.841 (+0.063) | 11/2/37 | **0.022** | 49.5% |
  | aieweb | 0.729 → 0.706 (−0.023) | 4/10/36 | 0.180 | 13.2% |
  | m365dotnet | 0.536 → 0.542 (+0.006) | 3/6/41 | 0.508 | 1.8% |

  **Pooled over 150 cases: 18 better, 18 worse, sign p = 1.000.** Exactly null.
  A neo-only reading of this was written up first and was wrong to generalize:
  the one repo where the flag helps is the only one whose catalog covers a
  meaningful share of the repository.
  **Coverage is the mechanism, and it was tested causally rather than inferred
  from three points.** Rebuilding neo's OWN catalog at `--max-files 30` (49.5%
  → 14.7% coverage, same repo, same 50 cases, same code) collapses the effect:
  **+0.063 at p=0.022 becomes +0.011 at p=0.832**, win/lose 11/2 → 12/10. The
  lane is not weak, it is STARVED — `--max-files` defaults to 100, so on any
  real repository the catalog holds a rounding error of the codebase (84 of
  4,563 eligible on m365dotnet). Raising or repo-relativising that cap is the
  lever; the weight is not.
  So `SEMANTIC_HINT_WEIGHT = CONTENT_WEIGHT` stays, now for a sharper reason
  than symmetry: **at realistic coverage the weight cannot be tuned into
  mattering**, and at high coverage 3.0 is already on a broad plateau (sweep on
  neo, MRR: 1.0 → 0.769, 2.0 → 0.803, 3.0 → 0.841, 6.0 → 0.853, 9.0 → 0.832;
  6.0 is not distinguishable from 3.0, paired 6/2 of 50, p = 0.289, and its R@1
  is lower). The old −0.007 MRR is superseded either way — it was a property of
  the 100%-test catalog it was measured against.
  **The 3× depth does nothing on its own**: pinning the weight to 1.0 while
  `SEMANTIC_HINT_DEPTH` stays at 3× scores MRR 0.769, *below* flag-off's 0.778
  (1 better / 3 worse, p = 0.625). What pays — when anything does — is weighing
  the catalog, not reading further into it.
  Every caveat in the harness docstring still applies: absolutes are upper
  bounds (the corpus has already seen the answer), the mined population is
  filtered upward, and 50 cases per repo is a modest sample. Full tables,
  setup and limits: `docs/semantic-lane-remeasurement-2026-08-30.md`.
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
