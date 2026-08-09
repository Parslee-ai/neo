# Changelog

## [0.44.0] - 2026-08-09

Neo was discarding most of what it was asked to look at, and reporting success. The index dropped whole languages, the prompt clipped every file it showed the model, and four tree-sitter queries had never compiled at all. None of it surfaced: each failure produced a plausible-looking result and exit 0. This release is that investigation, plus two defects found while fixing it — one in the reporting added to make the caps visible, and one that had been quietly corrupting the repository on every commit made from a git worktree.

### Fixed

- **`neo --index` on a .NET repo produced an index containing zero C#.** `build_index` globbed the file patterns in list order, concatenated, and sliced to `--max-files`. Since `file_patterns` begins `**/*.py`, whichever language globbed first ate the entire budget: a backend of 4,272 C# files yielded an index of 83 Python files — 95 of them from a git worktree checked out inside the repo — and printed "Built index". `_select_files` now apportions the budget across languages by repo composition with a floor of one slot each, and `_cap_chunks` round-robins `MAX_CHUNKS_PER_REPO` across files rather than slicing. Both halves were needed: fixing only the file cut left `chunks[:1000]` re-creating the identical bug one function later, keeping 1000 C# chunks and dropping every TypeScript and Python chunk the allocator had just fought to include. Exclusion is two layers — `EXCLUDED_DIR_NAMES` for what repos forget to ignore, plus the repo's own `.gitignore` through the helper `context_gatherer` already used, which is also what finally makes `docs/tree-sitter-setup.md`'s long-standing "Neo honors .gitignore" claim true. `bin`, `build`, `out`, `target`, `dist` and `vendor` are deliberately in neither list: each is real source somewhere, and over-excluding hides code permanently while over-including only spends slots. (`index/project_index.py`, #159)

- **Four tree-sitter queries had never compiled, so their constructs were absent from every index ever built.** A query that fails to compile is indistinguishable from one that matches nothing — `_get_query` returns `None` and parsing moves on. `typescript`/`tsx` `interfaces` matched the interface body as `object_type` where the grammar says `interface_body`, so every TypeScript interface was missing; `c_sharp` `inheritance` used a `bases:` field the grammar does not have, so every C# inheritance edge was dropped with only a DEBUG line. The only outward signal was a warning emitted once per query *per parsed file* — 9,699 identical lines in one reported run, which is why compile results are now cached including failures. C# bases needed widening well past the reported fault: `identifier` alone drops `: Repository<Order>`, and adding `generic_name` still drops `: System.Exception`, so the pattern now covers qualified names too, across class, interface, record and struct. `python` `module_doc` was deleted rather than fixed — correcting it only revealed that nothing consumes a lone `@doc` capture, so it had never produced a chunk in any released version. `test_every_chunk_query_compiles` and `test_every_edge_query_compiles` are parametrized over every entry in both tables, so the next grammar rename fails a test instead of quietly shrinking the index. (`index/language_parser.py`, #158)

- **Every context file was clipped to 3,000 characters with no marker, and the banner reported the pre-truncation size.** The model received `--- path ---` followed by text that simply stops, which it cannot distinguish from a file that ends there — so it answered questions about absence ("X is not called here", "there is no null check") from a fragment. The dangerous case is not an obvious clip but a cut landing just past a method signature: enough to look answerable, not enough to be right. Measured live, a claim Neo confirmed at 0.90 moved to 0.99 *and reversed* once the body was supplied. Cuts are now marked, the banner counts what was actually sent and says **chars** rather than bytes (`len()` on a `str` counts code points, so the old label was wrong twice over), and `code_smells.scan_files` is fed the truncated text — scanning the originals emitted findings like `src/Service.cs:401 [todo/warn] HACK: ...` for a file the model was shown to line ~215. An unmarked cut invites a wrong inference about unseen code; a line-numbered finding about unseen code asserts one. (`engine.py`, #176)

- **The new cap report blamed the chunk cap for absences it did not cause.** Added in the same change as the fixes above, and caught in review before release. The shortfall line fired whenever `selected > files_with_chunks` and attributed the whole difference to `MAX_CHUNKS_PER_REPO` — but `_extract_chunks_from_file` returns nothing for any file holding no function, class, interface or struct, which no cap setting changes. Reproduced against this repository: 25 files selected, 559 of 559 chunks kept, `chunks_capped` False on the same report, and the console still printed "the 1000-chunk cap is below the 25 files selected; lower --max-files". The culprit was an empty `tests/__init__.py`, and the prescribed remedy would only have indexed less. `files_producing_chunks` is now recorded before the cap runs, so cap starvation and a constructs-free file are reported separately and only the former names a knob. (`cli.py`, `index/project_index.py`)

- **Nineteen more unmarked truncations across nine prompt builders.** #176 fixed one instance; auditing the tree for the same defect class found the rest. They are now cut through `neo.text_budget`, which holds three shapes because the right cut depends on where the information is: `truncate_marked` keeps the head, `elide_middle` keeps both ends, `shown_of` annotates an elided list. Picking wrong compiles, passes tests, and throws away the answer — a head-cut of a 40-frame traceback keeps forty `File "..."` headers and discards `ValueError: database is locked`.

  Two findings were worse than the bug. **A nested cut emitted a marker that understated its loss by 29,110 characters**: `_deliberate` cut each context section to 2,000 and the concatenation to 6,000, delivering `verifiable_constraints` as 1,890 of 31,000 with its own marker sliced off, beneath an outer marker reporting 163 characters dropped. A bare slice gives the model no information; that gave it false information in the confident register of an instrument reading — and it was always the same section, because the order and caps were both fixed. Sections now share the budget by max-min fair apportionment and are cut exactly once, which also recovered what the flat cap wasted: a 30-character envelope plus 33,144 characters of memory against a 6,000 budget was sending 2,030. **And the community-cache fallback was a second instance of the same nesting inside `engine` itself** — an unmarked 200-character body cut feeding `past_learnings`, which the new apportionment then cut again, so the outer marker would have reported the pre-cut length as the fact's real one. It also elided a five-item list of curated global rules down to two in silence. Both are marked now. An apportioned section is only as honest as its least honest input. (`text_budget.py`, `engine.py`, `algorithm_design.py`, `constraint_verification.py`, `pattern_extraction.py`, `persistent_reasoning.py`, `repair_loop.py`, #178)

- **Every rule file was ingested once per spelling of its name.** `CONSTRAINT_FILES` lists `{project}/AGENTS.md` and `{project}/agents.md` separately, because different tools write different ones — but on a case-insensitive filesystem those are the same file, and a symlinked `CLAUDE.md` is a third spelling of it. Both the checksum cache and the supersession pass keyed on the literal path string, so each visit was blind to the last: the second found no stored checksum, superseded nothing, and minted a complete duplicate set of `CONSTRAINT` facts at confidence 1.0. Those are exempt from recall decay and protected from eviction, so nothing aged them out and stale copies outranked corrected ones. Measured across the fact stores on one machine: 51 duplicated subjects in a single project, then 35, 35, 33, 27, 12 — with projects lacking an `AGENTS.md` showing zero, which is the control.

  Identity is now `(st_dev, st_ino)` rather than a normalized string, because `realpath` preserves the case it was given and `os.path.normcase` is a no-op outside Windows — only the filesystem can answer, and asking it catches the symlink layout for free. The alias cleanup runs *above* the unchanged-checksum short-circuit, which is the whole of the migration: existing stores hold the duplicates already and their files have not changed, so a cleanup below the short-circuit would never run. Every store self-heals on its next ingest. (`memory/constraints.py`, #138)

- **`git commit` from a linked worktree corrupted the repository.** A commit from a worktree exports `GIT_DIR` and an absolute `GIT_INDEX_FILE` to hooks; a commit from the main worktree exports neither. `.githooks/pre-commit` runs the full suite, and `cwd=` does not override `GIT_DIR` — so a fixture's `subprocess.run(["git", "init", "-q", "."], cwd=tmp_path / "repo")`, which looks perfectly isolated, reinitialized the real gitdir instead. That set `core.bare = true` in the `.git/config` worktrees *share* with the main repository, taking the main checkout offline entirely, and the fixture's `add`/`commit` wrote its two-file temp tree onto the worktree's branch as a real commit. An autouse fixture now scrubs `GIT_DIR`, `GIT_WORK_TREE`, `GIT_INDEX_FILE` and six more for the whole suite — scrubbed suite-wide rather than repaired in the one fixture that tripped it, because `_get_changed_files_since`, `_get_working_tree_changes`, `_get_git_remote_url` and the acceptance detector all shell out too, and under an inherited `GIT_DIR` they answer about the wrong repository while looking healthy. (`tests/conftest.py`, #179)

- **`construct show` accepted any path.** `pattern_id` was interpolated straight into a filesystem path with no containment check. The fix is resolved-path containment rather than a scan for `..`: a symlink planted inside the library escapes without the string ever containing `..`, while the substring test would reject a legitimate id like `retry..backoff` and miss the symlink entirely. An absolute id is rejected explicitly, because `Path.__truediv__` treats an absolute right-hand operand as a replacement — `construct_root / "/etc/passwd"` *is* `/etc/passwd`. (`construct.py`, #25)

- **`construct search --top-k` accepted any integer.** Both out-of-range directions failed silently and plausibly: a negative value reaches a list slice as `results[:-3]`, dropping the *last* three results and returning the rest, and zero returns nothing, which reads as "no patterns match". Now rejected at parse time with the bounds named. (`cli.py`, #28)

### Changed

- `neo --index` reports what the caps left out — eligible versus indexed with a per-language breakdown, excluded paths, skipped duplicates, a capped chunk total, and files that contributed nothing. Every line is conditional, so a build that indexed everything prints none of them and their presence is the signal. "Built index" on its own no longer means the index represents the repository.
- Byte-identical files are indexed once, and the freed slot refills — across languages, so dedup cannot shrink the index below the operator's cap.
- The C# inheritance query is generated from one alternation over the four declaration kinds rather than written out four times. The hand-copied version is what left `interface_declaration` uncovered, and each widening since would have had to be applied to all four by hand.

### Documentation

- `docs/tree-sitter-setup.md` gains a "Why queries break silently" section, and its "Adding a New Language" steps now require asserting that a construct is *extracted* rather than only that its query compiles — the compile matrix is a floor, not a contract.
- CLAUDE.md records the index apportionment, exclusion and cap-reporting invariants; the prompt-budget contract and its footguns; and the git-environment hazard above.

### Testing

- `test_construct_search_relevance_ordering` now calls `search()` and asserts the ranking. It previously built differentiated embeddings and asserted only that the embedder existed — scaffolding for an assertion it never made. Two properties of the stub turned out to be load-bearing, and each is a way the original could not have worked even with the assertion present: the vectors must be near-orthogonal, since FAISS normalizes to unit length and `[0.9]*768` and `[0.1]*768` normalize identically, and the discriminating token must be unique to one fixture, which `"cache"` is not. (#27)

### Upgrading

- **Rebuild your indexes** (`neo --index`). An existing `.neo/index.json` still carries the old composition, including whatever the glob-order slice happened to keep.
- The first ingest after upgrading retires duplicate constraint facts and logs `Collapsed N alias spelling(s) of <path>`. Fact counts in affected projects will drop; that is the fix working, and the tombstones age off under the existing 30-day policy.

## [0.43.0] - 2026-08-06

The interactive learning loop had never once reached `accepted` in 30 days of real traffic. This release is the end of that investigation: four separate defects, each of which independently prevented a suggestion from ever being verified, and each measured against a live install rather than argued from the code.

### Fixed

- **Neo never read the file you asked it about.** `score_candidate` had no handling for a path spelled out in the prompt, so an explicitly named file competed on generic filename-token overlap — capped at three hits — against every other candidate. Asked to fix a function in `src/neo/subcommands.py`, Neo ranked that file **163rd of 296 candidates**, below its own test file, because an 86KB file takes a heavy size penalty. The file never entered the context, so the model correctly declined to patch code it had not seen and returned no diff. A suggestion carrying no diff text can never be git-verified, so it can never be learned from — which is a large part of why only ~32% of recorded suggestions were verifiable at all.

  Named paths are now pinned (`EXPLICIT_PATH_BOOST`, set to exceed every organic signal combined). Matching tests containment in *both* directions, because absolute paths — what tracebacks, IDE "copy path" and Neo's own suggestion output emit — otherwise went unmatched, and anchors on `/` so a bare `subcommands.py` hits `src/neo/subcommands.py` but not `tests/test_subcommands.py`. A named path that matches nothing now warns instead of staying silent, which was indistinguishable from mentioning no path at all. (`context_gatherer.py`)

- **Every large file contributed its import block, twice, for any prompt.** `select_chunks` took the first five matching lines *in file order*. Matching is a substring test and `extract_prompt_tokens` emits every 3+ character word, so `"in"` matched `int`, `using` and `point` — virtually every line qualified, and "first five matches" meant lines 1-5 of any large file against any prompt. With a 40-line window that is the module docstring and imports, emitted as two overlapping near-duplicate windows that consumed both slots of `MAX_CHUNKS_PER_FILE`.

  Windows are now ranked by **per-file document frequency** and **matched-token length**. A length cutoff for "discriminative" was tried and is inverted on real prompts — English stopwords are long and identifiers are short, so `len >= 4` keeps "does" and "here" while discarding `db`, `fs`, `os` and `api`. Length weighting matters separately: match *count* let a keyword-bearing module docstring tie with the function body and win on the file-order tie-break. Merging is bounded, because unbounded chain-merging measured **8.3× over `max_chunk_bytes`** — and since the caller admits a chunk all-or-nothing, an oversized chunk that no longer fits the global budget is dropped and the named file contributes nothing, reintroducing the original bug through a different door. (`context_gatherer.py`)

- **Promotion accepted evidence that was not independent, and minted a fact that was wrong.** A candidate promoted on two supporting episodes with distinct `episode_id`s — but every invocation mints a fresh one, so one operator applying the same patch twice, minutes apart, at the same HEAD satisfied it. This was reproduced live, and the durable PATTERN it produced encoded an incorrect lesson. Supporting episodes must now span at least two distinct repository revisions.

  A session-id fallback for non-git projects was written and then removed rather than shipped: `LearningEpisode.session_id` is a per-episode `uuid4()` that nothing ever assigns from a real session (121 episodes on a live ledger, 121 distinct ids), so "two distinct sessions" was the very gate being replaced under a different name. Because acceptance detection is entirely git-based, a genuinely non-git project can never record an ACCEPTED outcome and could not reach the fallback anyway; its only reachable trigger was a transient `rev-parse` failure, which made *both* failing promote while *one* failing blocked. Two costs are accepted and documented on the predicate: the revision is captured when the episode begins, not when the fix lands, and one lesson applied across several files before a commit records a single revision and will not promote. (`memory/store.py`)

- **`neo memory learning-stats` misread its own data in three ways.** Suggestions whose recorded `codebase_root` no longer exists — mostly deleted Claude Code agent worktrees — are unmeasurable, not unattributable, and they dominated the bucket that exists specifically as a bug signal; they now have their own name, and every filesystem-dependent test runs after the root check so the count is honest. Invented review prose at repo root (`/ARCHITECTURAL_REVIEW.md`) counted as *verifiable*, because the plausible-new-file rule admits any path whose parent is the root — 11 of 21 supposedly verifiable suggestions were documents that do not exist. And verdicts compared the unmeasurable count against a content bucket, so a single dead root suppressed STARVED entirely and claimed "most recorded suggestions" off a count of one; they are now computed over the measurable subset, with an integrity note qualifying every branch including ACTIVE.

  On the live corpus this moves `{verifiable 21, advisory 34, unattributable 10}` to `{verifiable 10, advisory 40, unattributable 1, root_unavailable 14}` — and the single remaining unattributable is a genuine defect. `suggestion_is_verifiable` is deliberately untouched: its bias toward admitting a doubtful path is correct, because `kind` is frozen at mint and under-admitting there permanently kills a real lesson, whereas over-admitting in a report costs only an inaccurate number. (`subcommands.py`)

- **A peer process's saved session could be erased without trace.** `_rewrite_session_log` now takes the scope lock, re-reads the log under it, and preserves every record appended since its own read. Locking the write alone was not enough, and the loss was reproduced: a second Neo process saving between one process's read and its `os.replace` vanished. Merge-on-write rather than a wider lock, deliberately — holding the lock across the git and LM work in `collect_outcomes` would let a slow reasoning run block another process's save. (`memory/outcomes.py`, #170)

- **Pending suggestions were destroyed before the user could act on them.** `collect_outcomes` cleared the entire session log whenever any session existed, so *any* Neo invocation between "Neo suggests X" and "user applies X" discarded the pending suggestion and the later acceptance had nothing to attribute to. Pendingness is now tracked per suggested path rather than per repository — the first fix retained sessions only when no file anywhere in the repo had changed, which held solely in a pristine tree — with a `scanned_through` watermark so the window does not re-scan, and a 14-day TTL. `replay_linked_feedback`, the documented repair command for a broken memory loop, no longer nukes the log it exists to repair. (`memory/outcomes.py`, #166, #168)

- **Tests wrote into live developer state.** `Path.home()` was patched, but roughly 20 constants capture their path at import time and pytest imports every module during collection, before any fixture runs. `conftest.HOME_PATH_CONSTANTS` now re-points each one, and `test_home_isolation.py` AST-scans `src/` to assert the table stays complete — the durable half, since the table alone would rot. (`tests/conftest.py`, #168)

### Changed

- The Codex install instructions in the README were missing their second step. (#167)
- `.coverage` is untracked and coverage artifacts are gitignored. (#165)
- Dependency and CI action updates: `tree-sitter`, `ruff`, `actions/checkout` 6→7, `actions/setup-python` 6→7. (#153, #154, #155, #160)

## [0.42.0] - 2026-08-05

### Added

- **Neo can now tell a host what it is doing.** `engine.py` previously made *zero* `print`/stderr calls, so an orchestrator (Claude Code, Codex, an IDE, an MCP server) saw nothing between invoking Neo and receiving the final `NeoOutput`. Worse, the Claude Code plugin instructed the agent to parse Neo's **human-readable text output** — reverse-engineering presentation out of terminal formatting while a fully structured document sat one flag away.

  New `neo.events` module: `NeoEventType`, `NeoEvent`, a `NeoEventSink` Protocol, and `Null`/`Recording`/`Jsonl` sinks. `NeoOutput` gains an `orchestrator` envelope (`summary`, `personality`, `phase_summary`, `cautions`, `recommended_narration`) derived from state already computed — no LM call, no new analysis. The governing rule is that **Neo reports facts about its process and the host decides how and when they become conversation**; no host-specific wording lives in the engine.

  `--json` now writes two streams: stdout is exactly **one** JSON document (so `neo --json | jq` keeps working) and stderr is JSONL progress events. They are deliberately not interleaved — the old `--json` help text promised events on one stream and honoring that literally would have broken every existing consumer. (`events.py`, `engine.py`, `models.py`, `cli.py`)

- **Four invariants over the event stream**, each pinned by a test: findings are emitted *before* their phase closes; no event may claim a phase that has already closed; every `phase_completed` has a matching `phase_started`; and every run terminates with exactly one `completed` or `failed`. Emitting `started` and then going silent was worse than emitting nothing — a host cannot distinguish a crash from a hang.

- **Neo speaks as Neo, and the register follows memory level.** All user-facing wording now comes from `orchestrator_voice` in `neo_matrix.yaml` via `_voice(key, **fmt)`; `engine.py` holds no prose of its own, so the character can be retuned without editing code (`test_engine_holds_no_prose_of_its_own` pins that). The same facts render as `Don't know this code. I'd change 1 thing(s) in src/parser.py, maybe. Confidence 0.88.` at memory stage 1 and `src/parser.py. 1 change(s). 0.88.` at stage 5. Cautions and progress messages deliberately do **not** vary by stage: a host is told never to drop a caution, so a warning must read the same however terse Neo is. Voice is not licence to soften a fact.

### Fixed

- **`repair_loop` had been dead since the initial public release.** It imported `structured_parser` and `schemas` *unqualified*, which can never resolve inside the installed package. Every repair attempt died on `ModuleNotFoundError` — and the attempt loop's `except` swallowed it into a generic "failed to repair after 2 attempts", so the feature looked ineffective rather than broken. Nothing tested it; `tests/test_repair_loop.py` now does. (`repair_loop.py`)

- **The Codex plugin never learned the new contract.** All six skills under `plugins/neo/` still ran `neo --mode advise` and told the model to parse "four structured sections: `CONFIDENCE`, `PLAN`, `SIMULATIONS`, `CODE SUGGESTIONS`" out of terminal-formatted prose — the exact anti-pattern removed from the Claude side. One adapter got the change and the other silently did not. The contract itself was already host-neutral, so only the adapter needed updating. `tests/test_host_adapter_parity.py` now fails if one surface teaches a contract the other does not. (`plugins/neo/skills/*`)

- **`--quiet` was declared and never read**, so the `[Neo]` progress notices could not be turned off — which is why stderr under `--json` was a mixed stream every host had to filter. Notices now route through the new `neo.progress` module instead of bare `print(file=sys.stderr)` calls scattered across `context_gatherer`, and `--json` implies `--quiet`. Index-command *errors* stay unsuppressed: `--quiet` silences progress, not failures. (`progress.py`, `cli.py`, `context_gatherer.py`)

- **Concurrent `process()` calls on one `NeoEngine` are now rejected instead of corrupting state.** Nearly all run state lives on the instance, so overlapping calls cross-attributed suggestions, facts and learning episodes between unrelated requests — silently. `process()` takes a non-blocking lock and raises `EngineBusyError`; the lock releases in a `finally` so a failed run leaves the engine reusable. This is not theoretical: `car_host` caches engines per working directory and reuses them, relying on CAR's drain task being single-threaded, which its own comment calls "an implementation detail". The A2A host now answers an overlap with a retryable `EngineBusy` response rather than a stack trace — a peer seeing a generic `ProcessingError` would assume its own request was malformed and stop retrying. (`engine.py`, `car_host.py`)

- **The version is single-sourced.** It was written down in four files and `prepare-release` asked a human to edit all four by hand, which is how the package, the Claude manifest and the Codex manifest reached 0.41.0 / 0.37.0 / 0.19.0. `pyproject.toml` is now the source of truth and `make sync-version` propagates it; a release bump edits one file. Edits are surgical regex replacements rather than `json.dumps` re-serialization, so a version bump stays a one-line diff instead of a whole-file reformat, and only the first `"version"` key is rewritten so a nested one cannot be clobbered. (`tools/sync_version.py`)

- Guarded an unreachable-but-fatal `elapsed / time_budget` division in the static-check skip log. A zero budget is not reachable through `_get_time_budget` today, but a *log line* must never be what crashes a run. (`engine.py`)

### Changed

- `tools/` is now covered by lint. The Makefile, CI and the pre-commit hook all ran `ruff check src/ tests/`, so a dead `import time` had been sitting in `tools/ab_controlled.py` — and `tools/` is where the new release script lives. All four invocations must stay identical or local green stops predicting CI green. (`Makefile`, `.github/workflows/ci.yml`, `.githooks/pre-commit`)

- `.gitignore` gained `.venv/`. Only `venv/` was listed, which does not match it, while the Makefile and the pre-commit hook both look for `.venv/bin/python` — the project's own convention was one `git add -A` away from committing an entire interpreter.

- The first pipeline phase is named `context`, not `retrieval`: it covers file gathering only, and fact retrieval actually runs while `reasoning` is open. A phase whose name overstates its span forces every emitter inside it to either lie or compensate.

## [0.41.0] - 2026-07-26

### Fixed

- **Found and fixed the actual cause of the observer's multi-GB memory, rather than only capping it.** Cycles that mined nothing — 0.1s per project, "0 mined" — were still swinging RSS by gigabytes. The reason: `TranscriptIngester.ingest` called `source.collect_episodes()` and applied the watermark filter to the *result*. Every source fully parsed every transcript on every cycle, for every project, and `all_episodes` retained the lot. Measured on one project: **118 MB (ClaudeCodeSource) + 180 MB (CodexSource) = 298 MB**, times up to 25 projects a cycle. The documented "watermark-gated so unchanged projects do near-zero work" was simply not true — the watermark gated admission, not work.

  Sources now skip inputs untouched since the watermark file's mtime (minus a 1h skew margin, since a transcript written *during* the previous ingest must never be judged already-seen). Measured after: the same project's collect goes **298 MB → 0 MB** when nothing changed. No storage format change — the watermark file's own mtime is the marker.

  Every error path falls back to parsing: skipping is an optimization, but a wrong skip loses learning silently. `since` is passed only to sources whose signature accepts it, detected by inspection — sources are an open interface, and calling one with an unexpected kwarg raises a `TypeError` that the per-source guard swallows as "source failed", silently yielding zero episodes. That failure mode bit three separate test stubs during development. (`memory/transcript.py`)

### Changed

- **Recycle cadence 6 → 3 cycles (~15min).** With the parse spike fixed, peaks are far smaller, and a tighter cadence costs ~2s per 15min (<0.3%). (`memory/observer.py`)

## [0.40.3] - 2026-07-25

### Fixed

- **0.40.2's cadence fix was a no-op; this makes it real.** `ObserverConfig` declared the recycle default in two places — the dataclass field and a restated literal inside `from_env`. 0.40.2 moved the field 48 → 6 and left `from_env` saying 48, and the daemon only ever reads `from_env`, so the shipped default never changed. The guard test added alongside it checked `ObserverConfig()` rather than `ObserverConfig.from_env()`, so it passed against the broken path and gave false assurance. `from_env` now derives every fallback from the dataclass defaults instead of restating them, and the test asserts both construction paths plus their equality when no env is set. (`memory/observer.py`)

## [0.40.2] - 2026-07-25

### Fixed

- **The observer's recycle cadence was too lax to do its job.** 0.40.0 shipped a default of 48 cycles (~4h) on the assumption that RSS creeps slowly. It does not: a live daemon went from 103 MB to **693 MB in 7 cycles**, reaching its plateau within minutes. A 4-hour cadence therefore reset an already-plateaued process and bought essentially nothing. Recycling caps RSS at roughly whatever it reaches in `recycle_after_cycles`, which makes that number the actual tuning dial rather than a formality. Default is now **6 cycles (~30min)**; a re-exec costs ~2s of process restart, under 0.2% overhead at that interval. (`memory/observer.py`)

## [0.40.1] - 2026-07-25

### Changed

- **Tombstones no longer carry their bulk text.** An invalidated fact is retained up to 30 days for supersession and audit, and every byte of it was deserialized on each `FactStore` load — the single largest driver of observer RSS. Measured on a live store: 11,421 tombstones holding 24.7 MB, of which `body` alone was 15.8 MB. `_invalidate` now drops `body`, `context_text` and `retrieval_text` at the transition, mirroring the existing embedding strip at the same choke point, and the periodic sweep backfills tombstones created before this landed. Verified against a copy of a real 188 MB store: **zero valid facts lost or gained** at identity level, tombstone count unchanged, 15 MB reclaimed in one pass.

  `subject` and `tags` are kept for audit, and `metadata` is kept because `purge_dead_facts` ages tombstones off `metadata.last_accessed` — dropping it would strand them forever. `episode_context` is deliberately not stripped: it is a structured object with its own `to_dict`, and blanking it breaks serialization outright.

  The obvious bigger change — moving tombstones to a sidecar file so the hot path never parses them — was investigated and **rejected**: `_reconcile_fact` returns OURS when we hold a fact invalid, which is what makes an invalidation beat a concurrent writer's stale valid copy. Lazily-loaded tombstones would let merge-on-save silently resurrect invalidated facts. (`memory/store.py`)

## [0.40.0] - 2026-07-25

### Changed

- **The observer now bounds its own RSS by re-exec'ing on a cadence** (`NEO_OBSERVER_RECYCLE_CYCLES`, default 48 cycles ≈ 4h, 0 disables). Its memory is peak working set plus CPython arena fragmentation — every sweep deserializes a multi-MB fact file and the allocator never hands those arenas back — so RSS drifted to 0.5–0.7 GB and stayed. Nothing leaked. Measured with recycling on: 179–381 MB across 10 recycles.

  Two things worth recording, because both are counter-intuitive. Embeddings are **not** the cost: they total 1.9 MB in memory and only dominate the file on disk, as JSON text. And re-exec is used rather than exiting for the supervisor to restart, because the CAR spec is `restart: "on_failure"` with `max_restarts: 10` — a clean `exit(0)` would never be restarted, and forcing a non-zero exit would exhaust the budget after ten recycles, permanently. Re-exec keeps the pid, needs no supervisor cooperation, and consumes no restart budget.

  The single-instance lock is released immediately before the exec. Carrying the fd across looks safer and is not: `flock` is owned by the open file description, so the inherited fd keeps the file locked and the new image can never acquire it — it exits as "contended", leaving the machine with no observer at all. That was measured directly during development before being fixed. (`memory/observer.py`)

## [0.39.1] - 2026-07-25

### Changed

- **Local checks now actually predict CI.** `ruff` is pinned to an exact version instead of a range: it is pre-1.0 and changes both its default rule set and rule behavior in minor bumps, so a range let a contributor's linter and CI's disagree — which is how a green local tree turned `main` red twice during the 0.39.0 release. The `Makefile` now runs tools through the project venv rather than whatever `ruff`/`pytest` sit on `PATH` (a system ruff 0.15.1 was shadowing the pinned 0.16.0, so `make lint` was linting against the wrong rules). The repo's `.githooks/pre-commit` runs the same commands as CI and refuses to run at all on a linter version mismatch, and `make hooks` installs it so it is versioned with the code instead of drifting untracked in `.git/hooks/`. `make ci-local` runs the gate without committing. Documented in CONTRIBUTING, including the limit: CI is Linux across Python 3.10–3.13 while the hook is one machine and one interpreter, so green locally is necessary but not sufficient. (`pyproject.toml`, `Makefile`, `.githooks/pre-commit`, `CONTRIBUTING.md`)

## [0.39.0] - 2026-07-25

### Changed

- **CI lint is no longer at the mercy of ruff's shifting defaults.** `[dev]` pinned only `ruff>=0.1.0` and `[tool.ruff]` declared no `select`, so every CI run adopted whatever rule set the newest ruff shipped. The 0.15 → 0.16 bump turned the tree red with **1050 errors on the previously-released commit**, none from any code change. The lint rule set is now declared explicitly (`E4`, `E7`, `E9`, `F` — the historical default the project has actually been enforcing) and ruff carries an upper bound, so lint results depend on the code rather than on release timing. (`pyproject.toml`)

- **Learning stats now separate "no edit target" from "failed to verify".** The single verifiable/total ratio conflated a usage property with a bug: an advisory prompt legitimately has no file to diff, while a genuine code suggestion that didn't resolve is an attribution failure worth chasing. `learning-stats` now buckets recorded suggestions three ways — verifiable / advisory / unattributable — and flags the case where unattributable exceeds verifiable, since that skew is a bug signal rather than a usage one. The advisory rule is reporting-only and deliberately does NOT widen `OutcomeTracker._is_review_only_path`, which drives real weak-acceptance behavior; widening a production predicate to tidy a statistic would change what neo learns. (`subcommands.py`)



- **Docs and the A/B harnesses caught up to the shipped `gpt-5.6` default.** The runtime has defaulted to gpt-5.6 since 0.38.0, but README/QUICKSTART/INSTALL and `tools/ab_*.py` still said `gpt-5.5`. Both harnesses now derive `DEFAULT_MODEL` from `NeoConfig` so this cannot drift again, and take `--model` so the published gpt-5.5 tiered-reasoning baseline stays exactly reproducible rather than being silently re-pointed. (`README.md`, `QUICKSTART.md`, `INSTALL.md`, `tools/ab_controlled.py`, `tools/ab_reasoning.py`)

### Removed



- **REVIEW → PATTERN synthesis is deleted.** `synthesize_reviews` and its whole apparatus — cluster/LLM synthesis, the triple-trigger gate, synthesis watermarks, `_hebbian_strengthen`, and the corpus-wide `_global_confidence_downscale` — are gone (467 lines from `store.py`), along with both call sites and the `SYNTHESIS_SIMILARITY` constant.

  It ran for four months on this machine and minted 114 facts. **Not one was a PATTERN.** The PATTERN branch required `group_key == "outcome:accepted"`, and a census found **0** accepted-tagged REVIEWs against 97 `independent` and 3046 `history` — because `detect_implicit_feedback` handles an ACCEPTED outcome by boosting the linked fact or by supporting an episode candidate, and both return before the fallback that would carry the tag. So the subsystem's headline function never executed once.

  What it did do: re-consume its own summaries as fresh evidence (68 synthesized `independent` REVIEWs against 29 raw), and multiply the confidence of every decaying fact in the store by 0.97 on each run. It also could not have fired on real data — a census showed **zero** ≥3-member clusters at cosine 0.85 across all 1152 valid REVIEWs, in any project, in any group.

  Deletion is bounded to the failed subsystem. `SUPERSESSION_THRESHOLD` and `_supersede`, canonical-signature dedup, the janitor chain, and the git-verified episode ledger (`_promote_repeatedly_supported_candidate`, which requires ≥2 independent verified acceptances) are untouched — regression-tested in `TestSynthesisRemoved`. Facts minted before the removal keep their `synthesized` tag and its prune immunity; nothing mints the tag now, so `PROTECTED_TAGS` is closed rather than growing. Orphaned `synthesis_watermark_*.json` files are deleted rather than migrated. The observer no longer reports a synthesized count, and `_last_cycle_count` now means facts *mined*. (`memory/store.py`, `memory/observer.py`, `memory/issues.py`, `subcommands.py`)

### Fixed

- **Suggestions made inside a git worktree could never be attributed.** Outcome matching normalized a suggestion's path against the *current* run's `codebase_root` rather than the root it was recorded under. Those differ whenever the previous run happened in a different working tree of the same repo — the common case being a Claude Code session in `.claude/worktrees/agent-*`, whose absolute paths never `relative_to()` the main checkout. Every such suggestion silently failed to match and its learning was orphaned: 14 of 65 recorded sessions on a live install. `project_id` already spans worktrees and clones, so the session was found; only the path root was wrong. Verified end-to-end: a session recorded in a linked worktree, with the change landing in the main checkout, now yields an `accepted` outcome. (`memory/outcomes.py`)



- **The durable-promotion path could never fire for a recurring lesson, because the "structural" fingerprint included the function's name.** `_extract_code_skeleton` emits a `def:<name>` token, and `_suggestion_fingerprint` hashed it verbatim — so the *identical* fix applied to `read_text` and `read_body` produced two different correlation signatures, and the two independent acceptances promotion requires could never accumulate. This contradicted the skeleton's own stated purpose ("recognize patterns … even when variable names differ"). The fingerprint now normalizes `def:<name>` to bare `def`; `method:<name>` is deliberately kept, since which method is called is part of the shape, and the skeleton itself is unchanged because it doubles as readable metadata on the fact.

  Two further defects in the same mechanism, found while verifying the fix: the normalization had to move *inside* `_extract_code_skeleton`, because it runs before the 500-char truncation and `def:<name>` is the only unbounded-length token — stripping names from the finished string left 40 structurally identical functions truncated at different points (316 vs 500 chars) and still hashing differently. And `ast.AsyncFunctionDef` is not a subclass of `ast.FunctionDef`, so `async def` emitted no def token at all and an async fix could never correlate with its sync twin; both now share a shape.

  Scope note, stated plainly: post-normalization skeletons are coarse — the drill's fix hashes to `def return` — so the fingerprint discriminates considerably less than "diff shape" implies. The real protection against two unrelated fixes merging into one over-trusted fact is the double git-verified acceptance and the kind gate, not the fingerprint. `test_structurally_distinct_fixes_do_not_collide` pins what it still buys.

  Migration note: candidates already sitting at `supported_once` carry fingerprints computed the old way, so they will not pair with post-fix episodes and must be re-earned. That is bounded, one-time, and self-healing — no stored fact changes meaning, since the fingerprint is only a correlation key.

  Verified against real `neo` invocations rather than in the abstract: a drill repo with the same defect in several modules produced **four git-verified acceptances and zero promotions** before the fix, and after it a matching pair minted a durable `pattern` fact tagged `episode-derived, verified-repeated, durable`. The drill also confirmed the two promotion gates are independent and both real — wording that classifies as `feature` yields a non-promotable `decision` candidate no matter how well the shapes correlate. (`engine.py`)

- **The interactive learning loop was recording candidates that could never be verified, so durable promotion never fired.** The ledger showed 49 episodes and **0 acceptances**. A controlled drill proved the pipeline itself is sound end-to-end: a suggestion on a real file with suggested code, applied and re-detected, produces an `accepted` outcome carrying its `candidate_id`, moves the candidate to `supported_once`, and a second matching acceptance promotes it to `durable` with a real PATTERN fact minted. The loop was **starved, not broken** — only 23 of 65 recorded suggestions could ever be git-verified. The rest are advisory prompts where the model answers with a topical pseudo-path (`/review/commit-<sha>.md`, `/REVIEW_ONLY/no_code_change.md`) instead of an edit target, and a candidate on such a path was still minted `pattern`, leaving the ledger accruing permanently-pending episodes. Candidates are now downgraded to non-promotable `review` unless a downstream diff could confirm them (`memory.outcomes.suggestion_is_verifiable`). A not-yet-existing path still qualifies when its parent directory is inside the repo — proposing a new file is legitimate and appears in `git log` once committed. (`engine.py`, `memory/outcomes.py`)

- **Path resolution is no longer duplicated.** `OutcomeTracker._normalize_path` now delegates to a shared `normalize_suggestion_path`, which the verifiability predicate also uses. An independent reimplementation treated bare-leading-slash paths (`/src/foo.js`) as absolute and rejected two genuinely promotable candidates on live data. (`memory/outcomes.py`)

- **`learning-stats` could not distinguish a starved loop from a broken one.** It now reports how many recorded suggestions could ever be git-verified, and reports STARVED rather than IDLE when none can. It also no longer prints the self-contradictory "ACTIVE: 0 promoted, 0 rolled back, 0 reinforced" — `active` counts demotions, so a lone demotion flipped the verdict while every number shown was zero; demotions are now included in the line. (`subcommands.py`)



- **Two clones of one repo were swept twice per cycle.** `flyx/fms` and `flyx/fms2` share a git remote, so `_compute_project_id` gives both the same id — one fact file, one pid-keyed watermark, loaded and synthesized twice. The sweep now shares a `FactStore` per project_id within a cycle. Transcript ingest still runs **per root**: attribution is by cwd and the second clone has 56 sessions of its own that sweeping only the first would silently drop. (Checked and dismissed a related concern: shifting a rollout between same-pid roots cannot lose data, because the destination fact file and watermark are the same ones; across *different* pids the watermarks differ, so it re-ingests rather than skipping.) (`memory/observer.py`)

- **Synthesis consumed its own output.** `_synthesize_cluster` mints `kind=REVIEW` carrying the union of its members' tags, and the eligibility filter had no `synthesized` exclusion — so every summary was eligible input on the next pass, counted as fresh evidence. Live census: 68 valid synthesized `independent` REVIEWs against 29 raw ones. A restatement of already-counted evidence is not new evidence. (`memory/store.py`)

- **Every synthesis decayed facts belonging to other scopes.** `_global_confidence_downscale` iterated all loaded facts, but `self._facts` also holds ORG and GLOBAL — and a GLOBAL PATTERN, exactly what `reconcile_cross_project_promotions` mints from ≥2 projects' evidence, is decay-eligible. Unscoped, each of 37 projects' syntheses eroded those cross-project promotions for unrelated reasons. Now PROJECT-scoped, matching the population the Hebbian step potentiates. Latent rather than observed — no GLOBAL fact currently carries a decaying kind, and a live histogram confirmed the corpus is healthy (0 facts at the 0.05 floor). (`memory/store.py`)

- **A pinned CAR model that fails is now named in the error.** CAR's router can fall through to another candidate and report *that* backend's failure ("mode apple-foundation-models not implemented…"), so the model the caller actually asked for never appeared and the error wasn't diagnosable. `CarAdapter` now decorates such failures with the pinned model, preserving the original as `__cause__`, and leaves router-mode and already-named errors untouched. This fixes `test_live_bad_model_raises_actionable_error`, which failed on clean `main` and blocked the pre-commit hook — fixed by making the assertion true rather than by weakening it. (`adapters.py`)



- **The global observer swept the filesystem root and `$HOME` as if they were projects, costing ~26s of every ~40s cycle.** Claude Code mints a transcript dir for whatever cwd a session ran in, so ad-hoc sessions from `/` and `~` decoded back into "project" roots that are ancestors of every real one. `CodexSource` attributes a rollout by cwd prefix, and `"/".rstrip("/")` is `""` — so the `/` pseudo-project matched **every** Codex session on the machine (2968 episodes, 14s) and re-parsed them every cycle. Discovery now drops **container roots** structurally: a decoded root that holds another discovered root and is not itself a repo. The `.git` clause only ever *rescues* an ancestor (a monorepo is a real project) and is never a blanket requirement — many real roots have no repo. Nested real roots are disambiguated deepest-wins via `CodexSource(peer_roots=…)`, and containment is now `Path.is_relative_to` rather than a string prefix, which makes the empty-base footgun structurally impossible and refuses the false-prefix sibling (`/work/github/x` is not inside `/work/git`). Measured: 40 → 37 roots, all three containers gone, real projects unaffected. (`memory/observer.py`, `memory/transcript.py`, `memory/issues.py`)

- **Synthesis burned work every observer cycle on material that could never consolidate.** A census found all 1152 valid REVIEW facts across the 15 gate-passing projects clustering into groups of exactly **1** at `SYNTHESIS_SIMILARITY` 0.85 — no group, in any project, reached the 3-member minimum. 90% of that corpus (1039) is the `history` group, whose subjects are individual git commit messages: distinct events by construction, never a recurring lesson. They counted toward the ≥20 gate and were re-clustered every cycle, so the gate opened on material that could not produce output. `_is_synthesizable_review` now excludes them from both the gate count and the clustering, dropping projects doing pointless clustering work from 15 → 2. `SYNTHESIS_SIMILARITY` is deliberately **not** lowered — a threshold sweep showed 0.60 already yields a 40-member merge. (`memory/store.py`)

  **This is containment, not a cure, and the underlying subsystem is unsound.** Review surfaced three findings that this change does not address, recorded here so they aren't rediscovered: (1) the observer path has synthesized nothing in 265 cycles, but the inline trigger in `detect_implicit_feedback` has minted **114 synthesized facts** since 2026-03-24 — and **all 114 are `kind=REVIEW`; not one PATTERN has ever been produced**, because PATTERN requires a `group_key == "outcome:accepted"` and no REVIEW on this machine has ever carried the `accepted` tag. The subsystem's headline REVIEW→PATTERN function has never executed. (2) Synthesized output is itself a valid REVIEW carrying its members' tags, with no `synthesized` exclusion in the gate filter, so summaries re-enter as inputs. (3) Every successful synthesis runs `_global_confidence_downscale(0.97)` across the entire valid corpus. The real defect is upstream — minting one REVIEW per git commit — and the honest options are to give per-commit records their own kind or stop ingesting them. Deleting the subsystem outright is a live proposal, deliberately left as a separate decision rather than bundled into a waste-reduction change.

- **~100 subprocess forks per observer cycle (~26k/week) for an answer fixed within the cycle.** Resolving one project's identity asked git for the same remote 3× (org id, project id, transcript source lookup), each a fork+exec. `_get_git_remote_url` is now memoized per root, cleared once per cycle so a remote that changes under a multi-day daemon can't silently write facts under a stale `project_id`. Measured 4 → 2 forks per project. (`memory/scope.py`, `memory/observer.py`)

- **88 MB of stranded atomic-write temp files in `~/.neo/facts`, swept by nothing.** `_save_file` unlinks its temp file on exception, but a SIGKILL runs no handler — and the observer's own lock-escalation path SIGKILLs stragglers by design. `FactStore` now reaps `*.tmp` older than 24h on cold start (the age window makes it impossible to hit an in-flight write). (`memory/store.py`)

- **`metrics.jsonl` grew without bound** (17 MB on a live install). Rotates to one retained `.1` generation at 32 MB, size sampled every 500th write so the steady state stays one `write` syscall per event. (`memory/metrics.py`)

- **23,943 of ~24,000 observer stderr lines were libmalloc noise**, burying 29 real `ENOSPC` errors and a real LM failure. macOS emits "can't turn off malloc stack logging" once per forked child when `MallocStackLogging` is inherited but unset; the daemon now scrubs those vars at startup so no child inherits them. (`memory/observer.py`)

### Documented

- **`_synthesize_cluster`'s PATTERN branch is unreachable in normal operation**, and is now marked as such. It requires `group_key == "outcome:accepted"`, and a census found **0** `accepted`-tagged REVIEWs across the whole store (against 97 `independent`, 3046 `history`). Not a tag-writing bug: `detect_implicit_feedback` handles an ACCEPTED outcome by boosting the linked fact or by supporting an episode candidate, and both `continue` before the REVIEW fallback that would carry the tag. So the subsystem has produced 114 facts and **zero PATTERNs**. Deliberately not "fixed" by forcing the tag — that would reintroduce the orphan REVIEWs the boost path exists to avoid and stand up an unverified promotion route competing with the git-verified episode ledger. (`memory/store.py`)

## [0.38.0] - 2026-07-19

### Added

- **Evidence-driven learning loop: a per-request episode ledger, separate from the fact store, that promotes a candidate to a durable fact only on repeated, independently verified evidence.** Generated reasoning is now recorded as probationary `MemoryCandidateEvidence` in a `LearningEpisode` (`~/.neo/episodes/<project_id>/<id>.json`); downstream git-verified outcomes decide whether a candidate ever enters `FactStore`. Project-scope promotion needs ≥2 independent accepted episodes sharing a correlation signature; cross-project (global) promotion needs ≥4 episodes across ≥2 projects; attributed contradictions demote and eventually roll a promoted fact back. A live end-to-end drill (real model, real git-diff attribution) confirmed a durable PATTERN fact mints from two independent acceptances. (`memory/episodes.py`, `engine.py`, `memory/store.py`)
- **Goal-aware execution envelopes.** Explicit caller-supplied goal/intent/role are honored; otherwise conservative provisional defaults are inferred (`resolve_execution_context`), with a shared failure-symptom lexicon (`FAILURE_SIGNAL_KEYWORDS`) so intent inference and task-type classification can't drift. (`execution_context.py`, `models.py`)
- **An "is it learning?" dashboard: `neo memory citation-stats` and `neo memory learning-stats`.** citation-stats reads the `citation_survival` metric (retrieved/included/used, split by which detector earned the use-credit); learning-stats reads the episode ledger (episodes, outcomes, candidate statuses, and promote/rollback/demote/reinforce actions) for the INTERACTIVE/attributed path. Both read-only, no LM call. (`subcommands.py`)
- **Prompt→task-type classification (`models.classify_task_type`).** The plain-text CLI previously hardcoded `TaskType.FEATURE`, routing every interactive prompt to the non-promotable `decision` candidate kind — so interactive use could never mint a durable fact. It now classifies (deterministic keyword scoring, no LLM; ties fail SAFE toward non-promotable kinds; an optional `error_trace` strongly biases BUGFIX). Genuinely algorithmic/bugfix work reaches the promotable `pattern` kind while feature/refactor/explanation stay conservatively non-promotable. (`models.py`, `cli.py`)

### Fixed

- **The interactive episode→durable promotion path, provably inert, now fires.** A live drill measured zero promotions for two independent reasons, both fixed: (1) the correlation signature keyed on the run-varying LM reasoning body, so two acceptances of one task never matched — now keyed on the subject plus a structural diff **fingerprint** (`sha256` of the AST-shaped code skeleton), with a separate **path-agnostic** key for cross-project promotion so the same lesson correlates across repos; (2) the CLI hardcoded a non-promotable task_type (see Added). Rollback is fingerprint-precise; single-project rollback deliberately cannot reach a global fact (teardown is reconcile-owned). (`memory/store.py`, `engine.py`, `models.py`, `cli.py`)
- **Tombstone dead-embedding bloat and invalidation consolidation.** Invalidation now routes through a single choke point that strips the 768-dim embedding at the transition; a cold-start janitor chain (`prune → demote → purge → strip → dedup`) flushes one merge-on-save instead of several. Reclaimed hundreds of MB on a live store. (`memory/store.py`)
- **A project's retraction no longer feeds cross-project (global) promotion**, and duplicate same-signature facts are collapsed at every scope (not just via merge-by-id). (`memory/store.py`)

### Changed

- **Default static model `gpt-5.5` → `gpt-5.6`** (verified live: a real call succeeds, a bogus-model control fails, and the `reasoning.effort` levels still apply). Users on a pinned `config.json` are unaffected; this is the fresh-install default. (`config.py`)

## [0.37.0] - 2026-07-06

### Changed

- **Multi-agent deliberation now fires with ≥1 capable model, not ≥2 distinct ones — a controlled experiment overturned the "diversity is the value" assumption.** The tiered-reasoning gate (#129) offered the panel only when a query was novel AND CAR was reachable AND there were **≥2 genuinely-distinct capable models**, on the theory that a single-model panel just re-confirms one model's blind spots. A controlled A/B/A (n=8, three arms per task — single vs panel-with-gpt-critic vs panel-with-Claude-critic — scored in one judge call so the critic model is the only variable) measured: single **7.88**, panel same-model **9.00** (+1.12), panel distinct-critic **9.00** (diversity delta **~0**; head-to-head 1/1/6). The panel's win is the **orchestration structure** (plan-vote → code → adversarial critique → repair), which holds fully same-model; an adversarial critic in a fresh context catches errors via *reframing*, not different weights. So `DEFAULT_MIN_MODELS` drops 2 → 1: deliberation now fires on novelty + CAR + one capable model, and the parslee-family model-miscount no longer changes the decision. `min_models` stays tunable for callers who still want a diverse pool (harder/ambiguous tasks aren't yet measured). Internally `diversity_ok` → `can_deliberate`, with updated rationale/reason strings. Bounds: n=8 self-contained tasks, two strong critics. (`reasoning_mode.py`, `config.py`)

### Added

- **Controlled A/B/A reasoning harness with a pluggable critic provider.** `tools/ab_controlled.py` runs the three-arm, single-judge-call comparison above (isolating diversity from orchestration), and `tools/ab_reasoning.py` gains `--critic-provider {anthropic,openai,car}` so the panel critic can route to a different frontier model than the coder. (`tools/ab_controlled.py`, `tools/ab_reasoning.py`, `docs/solutions/tiered-reasoning-multi-agent.md`)

## [0.36.1] - 2026-07-06

### Documentation

- **Corrected neo's self-description of the synthesis observer, and documented `neo memory replay-feedback`.** The `observer.py` module docstring and `docs/solutions/async-observer.md` still described the background synthesis observer as a *per-project* process; it has been a **single global** CAR agent (`neo-observer`, `--daemon --all`) that sweeps every discovered project each cycle since the per-project agents (`neo-observer-<id12>`) were migrated away. The docstring now leads with the global model and the async-observer proposal's "open question" is marked resolved. Separately, the `neo memory replay-feedback` subcommand (re-processes linked ACCEPTED/MODIFIED/UNVERIFIED outcomes to update fact confidence + `success_count`) was undocumented and is now covered in the memory-hygiene notes. Docstring/doc-only — no behavior change. (#136)

## [0.36.0] - 2026-07-06

### Fixed

- **LM adapters no longer break on newer reasoning models that reject standard chat parameters.** Neo unconditionally sent `temperature` (and `max_tokens`) on every request, but newer reasoning models reject them with a 400 — surfacing as the reported `ProcessingError` / `BadRequestError`: Anthropic Opus 4.7+/Sonnet 5/Fable 5 reject `temperature` ("temperature is deprecated for this model"); OpenAI o-series/gpt-5, Azure reasoning deployments, and OpenAI-compatible reasoners (xAI Grok, DeepSeek behind a `base_url`) reject `temperature`, and the OpenAI-family additionally require `max_completion_tokens` instead of `max_tokens`. There is no reliable model-string rule (`opus-4-6` accepts `temperature`, `opus-4-7` rejects it; on Azure `model` is an arbitrary deployment name), so a hardcoded allow/deny list would silently break on the next model. Instead the adapters **adapt reactively**: catch the 400, drop/rename the offending param, retry, and remember the model so subsequent calls skip it up front. Any 400 for another reason re-raises untouched (fail-fast preserved). The three OpenAI-SDK adapters (OpenAI chat path, Azure, Local) share `_chat_completion_resilient`, which handles both the `temperature` drop and the `max_completion_tokens` rename, copies its kwargs, and is robust to minor error-wording changes; `AnthropicAdapter` has its own inline learn-and-retry. The OpenAI `gpt-5`/codex `/v1/responses` path (which already omits `temperature`) is unchanged. (`adapters.py`)

### Added

- **Learned reasoning-model parameter adaptations now persist across invocations.** The reactive learn-and-retry above recorded each model's needed adaptations in in-memory sets, so every short-lived CLI invocation re-paid the first-call 400-retry penalty (up to two wasted round-trips for a model that rejects both params). Learnings now persist in `_ModelParamCompat` (`~/.neo/model_param_compat.json`, keyed `"<provider>:<model>" → [flags]`) so a fresh process skips the bad params on the first call. The store is provider-keyed (a bare `gpt-4` behind a Local endpoint can't collide with Azure/OpenAI), uses merge-on-write + atomic `os.replace` without an flock (writes are rare and additive-only, so a lost update self-heals on the next reactive retry), resolves its path at call time (per-test `Path.home()` stubs apply), and is **best-effort and totally guarded** — a missing/corrupt/wrong-shape file or an unresolvable home degrades to "not learned" and never breaks inference. (`adapters.py`)

## [0.35.0] - 2026-07-04

### Added

- **Tiered reasoning: a real, CAR-gated multi-agent deliberation tier.** Neo's "MapCoder/CodeSim" reasoning was a single combined call playing three roles, and its "simulation" was self-narration the engine already distrusted. This adds a genuine panel — **plan-vote → code → adversarial critique → repair** — and gates it so the fast, memory-driven single call stays the default. Multi-agent fires only when a query is **novel** (the existing `effort_from_memory` signal, promoted from a token dimmer to a mode switch) **and** CAR is reachable **and** there are **≥2 genuinely-distinct capable models** — because a single-model panel just pays latency to re-confirm one model's blind spots. Model diversity is engineered via CAR's `route_model` candidates + `exclude_models` (car#358), so the critic routes to a *different* model than the coder; a decision-only routing-plan pass assigns roles up front. Confidence is a **consensus** of independent signals (plan agreement × coder confidence × critic verdict), not one model's self-report. `--deep`/`--fast` override the `reasoning_mode` config (`auto` default); output metadata always carries `reasoning_mode` + `reasoning_reason`, and panel runs add `provenance` + `panel` (models_used/consensus/rounds). A blind-judge A/B (`tools/ab_reasoning.py`) measured **+1.12 quality** (panel 9.25 vs single 8.12 / 10) with wins clustered on edge-case-heavy tasks — reinforcing that deliberation is worth its ~5× latency only on hard/novel work. Execution-grounded verification (L4) is deferred: most real code needs infra a sandbox can't conjure, so verification leans on the adversarial critic + static analysis. (`reasoning_mode.py`, `multi_agent.py`, `panel.py`, `engine.py`)
- **Deliberated outcomes are routed through probation with provenance.** Facts a panel produces enter memory tagged `[multi-agent, probation]` — recognizable and trust-first: promoted only by real outcomes (`access_count>=2` / `success_count>0`) via the existing pipeline, so a wrong panel answer never becomes confident memory unchecked. Stored confidence is the panel consensus. (`engine.py`)

### Fixed

- **The multi-agent diversity gate no longer miscounts same-backend model personas.** `capable_model_count` treated `parslee/advisor` and `parslee/reasoning` as two distinct models, but both are Azure gpt-5.5 (Parslee assistant personas) — so the "≥2 distinct models" gate could deliberate with no real diversity. It now counts distinct provider *families* (`parslee/*` → 1), biasing conservatively toward the fast path. (`panel.py`)

## [0.34.0] - 2026-07-01

### Fixed

- **`neo` can no longer hang forever on a non-tty stdin.** `detect_input_mode()` eagerly drained stdin (`sys.stdin.read()`, a read-until-EOF) whenever stdin wasn't a tty — even when the prompt was supplied on argv. Launched with a non-tty stdin that the peer holds open without data or EOF (a background job, a daemon, a mis-wired subprocess pipe), that read blocks indefinitely; a native stack trace caught the main thread parked in `_io_FileIO_readall_impl → read()` on fd 0 (an open unix socket). An argv prompt now skips the stdin read entirely (`neo "query"` — the common case, and the one that hung), and when stdin genuinely is the input, `_read_stdin_guarded()` `select()`s for readability with a deadline (`NEO_STDIN_TIMEOUT_SECONDS`, default 1.0s) before reading, treating a never-EOF stdin as empty rather than hanging. Real pipes and redirects report ready immediately and are unaffected; interactive terminal use was never affected (tty stdin → the read was already skipped). (`cli.py`)

### Added

- **CAR inference calls are now bounded and large-context calls are given room to finish.** Two complementary changes to `AutoAdapter`/`CarAdapter`: (1) a wall-clock watchdog around each CAR call (`NEO_CAR_TIMEOUT_SECONDS`, default 240s) — `car_runtime.infer_tracked` is a blocking FFI call, and a daemon restarted mid-call could leave the client's socket read blocked forever with the breaker's exception-only path unable to catch it; the call now runs in a daemon worker thread joined with a deadline, so a hang trips the breaker and fails over to the static provider. (2) `CarAdapter` raises CAR's per-call FFI read-timeout floor (`CAR_DAEMON_TIMEOUT`, default 180s, only when the operator hasn't set it) — a quality remote serving neo's large multi-file context legitimately takes ~140s, but CAR's 30s default cut those calls off mid-inference and forced a fallback every time. The watchdog sits above the FFI timeout so CAR's own clean timeout fires first and the watchdog only catches a true hang. (`adapters.py`)

## [0.33.0] - 2026-06-26

### Added

- **neo now learns from merged GitHub PRs and their review threads, not just AI-tool transcripts.** A fourth transcript source, `GitHubPRSource`, mines each repo's merged PRs via the `gh` CLI (one GraphQL call/repo) — review summaries, conversation comments, and inline line-comments — and turns the review discussion into memory facts. owner/repo is auto-derived from the git remote, so PR facts co-scope with that repo's transcript facts under the same `project_id`. Facts enter as **REVIEW on probation** (`imported:github-pr` tag) — trust-first, since PR reviews are other people's opinions, not neo's validated outcomes; they are not promoted by recurrence. The source filters bot authors, skips PRs with no human discussion, mines each PR once (bounded watermark keyed on PR number), and is throttled to one fetch per repo per hour so the all-projects observer sweep keeps near-zero work on unchanged repos. It self-disables (no env flag) when the remote isn't github.com or `gh` is absent. Known limits: merged-only, discussion text only (no PR diff), no historical backfill beyond the 25 most-recently-updated PRs, GitHub Enterprise and fork-origin remotes not yet handled. A new optional `fact_kind`/`extra_tags` on the `TranscriptSource` Protocol carries the trust posture into the ingester's `admit` (backward-compatible; existing sources unchanged). (`memory/transcript.py`)

## [0.32.0] - 2026-06-26

### Fixed

- **Orphaned observer processes are now auto-reaped instead of only reported, and a single-instance lock prevents two observers from ever running synthesis at once.** When a car-server died ungracefully (crash, force-quit, `kill -9`) its supervised observer reparented to init/launchd and kept running its synthesis loop forever; a new car-server then started a *second* supervised observer, so two ran concurrently (observed live: a straggler ran 2d18h). Previously `status` only *flagged* the orphan with a manual `kill` hint. Now `_reap_orphan_observers` SIGTERMs any unsupervised observer daemon — re-reading each pid's command line right before the signal to defend against pid reuse — wired into `start`/`stop`/autostart and the daemon's own startup. A second guarantee backs it up: the daemon holds a cross-platform single-instance file lock (`_SingleInstanceLock`, `fcntl`/`msvcrt` on `~/.neo/observer.lock`) for its lifetime, so even in the restart handoff window two observers can't both write; a daemon that can't get the lock exits 0 (benign no-op, no CAR backoff). If a straggler ignores SIGTERM past a short grace, the daemon escalates to SIGKILL so the kernel frees the lock — safe because `FactStore._save_file` is atomic (temp + `os.replace`), so a hard kill can only leave a stray `.tmp`, never a torn fact file. This was never a corruption bug (`store.save()` already serializes writers with its own per-scope flock) — just doubled LM spend and synthesis. (`memory/observer.py`)

## [0.31.1] - 2026-06-19

### Changed

- **The global observer sweep now logs per-project progress.** Previously a sweep logged only a `swept N/M …` summary at the *end*, so a long first sweep (catching up the transcript backlog of every previously-unobserved project) had no heartbeat and looked hung. The sweep now prints a `sweep start: N of M project(s)` line, then a `sweep [i/N] <project>: X synthesized, Y mined (Zs)` line as each project completes (errors per project logged to stderr). Subsequent watermark-fast sweeps stay quiet. (`memory/observer.py`)

## [0.31.0] - 2026-06-19

### Changed

- **The memory observer is now a single global, auto-started process instead of one opt-in process per project.** Previously you had to run `neo memory observer start` in each repo, and each got its own supervised daemon — so in practice only one project was ever observed. Now a single CAR agent (`neo-observer`, daemon `--daemon --all`) **sweeps all discovered projects** each cycle (round-robin, `max_projects_per_cycle` default 25; watermark-gated so projects with no new transcripts do near-zero work), running synthesis + transcript mining per project. **No opt-in**: `maybe_autostart_observer()` (called from `cli.main`) auto-registers it whenever `car-server` is reachable — opt out with `NEO_OBSERVER_AUTOSTART=0`; when CAR is absent it prints a one-time hint then stays silent. On bootstrap/start, legacy per-project agents (`neo-observer-<id12>`) are stopped and removed. Lifecycle (`start`/`stop`/`kick`/`status`) and orphan detection now operate on the single global agent; orphan detection flags *any* unsupervised neo observer daemon (global or stranded legacy). Per-project mode (`--daemon --cwd`) and the A2UI inspector remain for direct use. (`memory/observer.py`, `cli.py`)

## [0.30.0] - 2026-06-19

### Changed

- **Observer orphan detection is now cross-platform (Windows support).** The 0.29.0 orphan check used `ps` and `ppid == 1`, both POSIX-only. `_find_orphan_observers` now prefers **psutil** (added to the `[car]` extra) for parent-liveness introspection that works on macOS/Linux/Windows, falling back to `ps` on POSIX when psutil is absent. The portable orphan signal is "no live parent": POSIX reparents a dead parent's child to init/launchd (`ppid == 1`), while Windows does not reparent — psutil reports the vanished launching car-server as `parent() is None`. The supervised observer (parented by a live car-server) is never flagged on any OS. Verified live (psutil path doesn't false-positive the real supervised observer) and unit-tested for POSIX-orphan, Windows-orphan, supervised, and non-observer cases. (`memory/observer.py`, `pyproject.toml`)

## [0.29.0] - 2026-06-19

### Added

- **`neo memory observer status` now flags orphaned observer processes.** A prior car-server that died without reaping its child leaves an observer reparented to init/launchd (the pre-0.18.0 footgun) — running redundant synthesis cycles and invisible to `status`, which only knew about CAR's supervised agent. Status now scans for `neo.memory.observer --daemon` processes scoped to this project (matched by the daemon's `--cwd`, realpath-compared) whose parent is init/launchd (`ppid == 1`) — the supervised observer is always parented by car-server, so it's never falsely flagged — and prints a `WARNING` with the orphan pid(s) and a `kill` hint. Best-effort via `ps`; returns no orphans (never errors) on any platform/parse failure. The status result gains an `orphans` field. Found and reaped a real 4-day-old orphan during validation. (`memory/observer.py:_find_orphan_observers`, `subcommands.py`)

## [0.28.1] - 2026-06-19

### Changed

- **`[car]` extra floor raised to `car-runtime>=0.27.0,<1.0`** (was `>=0.18.0`). The spec already tracked latest *within* the 0.x line — a fresh install / `pipx upgrade` resolves to the newest compatible release, since car-runtime is on the package index — so this raises the floor to the newest version neo is validated against (a fresh install can no longer land on an old 0.x). The `<1.0` cap is kept deliberately: car-runtime is a sealed binary where 0.x minors can carry breaking changes, so a breaking 1.0 rewrite is not adopted silently on upgrade. The observer's *runtime* floor stays a separate, more permissive `0.18.0` (`memory.observer._CAR_MIN_VERSION`) — the functionally footgun-free minimum — so an existing 0.18–0.26 install keeps working while fresh installs ship the latest. (`pyproject.toml`)

## [0.28.0] - 2026-06-19

### Fixed

- **The observer now enforces its documented car-runtime ≥ 0.18.0 floor at runtime.** `_require_car_runtime` previously gated only on `hasattr(car_runtime, "agents_upsert")`, which 0.16.x/0.17.0 also satisfy — so the observer would silently run on an under-spec binding exposed to the pre-0.18.0 supervisor footguns (orphaned child / restart-storm on start-during-backoff / stale `last_exit_code`) that 0.18.0 fixed. It now also checks the installed version (via `importlib.metadata`) and raises an actionable error below 0.18.0. An unparseable/missing version is allowed through (lenient about unknown, strict about known-too-old) so a vendored build isn't falsely rejected. (`memory/observer.py`)

### Changed

- **Revisited the `CarAdapter` `task=code` intent hint now that [car-releases#52](https://github.com/Parslee-ai/car-releases/issues/52) is closed.** The router cost-bias that motivated the hint is fixed upstream (the router now prioritizes quality > speed > cost for `task=code`, plus the 0.25–0.27 capability-honest routing + `exclude_models` reworks), so the `{"task":"code","prefer_quality":True}` default is the *intended* router API rather than a bug workaround — kept, with the rationale in CLAUDE.md updated accordingly. Validated against car-runtime 0.27.0. (docs only; `adapters.py` default unchanged)

## [0.27.0] - 2026-06-19

### Added

- **`neo memory import` — ingest a peer tool's memory into neo's store (phase 2 of cross-tool memory).** Reads Claude Code's per-project `memory/*.md` and admits each well-formed entry as a fact in neo's own store, so what one agent learned becomes available to neo's reasoning. **Trust-first admission**: imports enter as **REVIEW** (a decaying kind, never CONSTRAINT/ARCHITECTURE which would bypass decay and read as curated truth), with **INFERRED** provenance and an `imported:claude-memory` tag — so `add_fact` puts them on **probation** (they must earn promotion via access/success like any fluid fact) and gives them the lowest provenance bonus. Dedup/supersession is handled by `add_fact`'s existing cosine ≥ 0.85 machinery; a per-(project, tool) content-hash watermark makes re-runs idempotent (an *edited* memory re-imports and supersedes rather than duplicating). Malformed entries (no frontmatter / no description) are skipped. `--dry-run` previews without mutating; `--confidence` tunes the initial value (default 0.4). `neo memory import [--dry-run] [--confidence 0.4] [--cwd]`. (`memory/memimport.py`, `cli.py`, `subcommands.py`; see `docs/solutions/memory-audit.md`)

## [0.26.0] - 2026-06-19

### Added

- **`neo memory audit` — read-only hygiene inspection of an AI tool's memory files.** Distinct from `neo memory rules` (which compares human-authored rule files), this inspects a tool's *accumulated, learned* memory. v1 targets Claude Code's per-project `~/.claude/projects/<proj>/memory/` dir (a `MEMORY.md` index plus fact files with YAML frontmatter). It reports four hygiene problems: **malformed** (missing frontmatter/description, invalid `type`), **near-duplicate** memories (embedding cosine ≥ 0.93 via the shared clusterer), **conflicts** (same-topic-but-divergent pairs, LM-judged; `--no-conflicts` to skip), and **index** drift (a memory file absent from `MEMORY.md`, or `MEMORY.md` pointing at a missing file). Dangling `[[links]]` are intentional per the memory spec and are not flagged. Read-only — never edits memory. Dogfooded across local projects (correctly flagged a memory file missing from its `MEMORY.md` index). `neo memory audit [--json] [--no-conflicts] [--cwd]`. (`memory/memaudit.py`, `cli.py`, `subcommands.py`; see `docs/solutions/memory-audit.md`) This is phase 1 of cross-tool memory support; phase 2 (ingesting peer memory into neo's store as a `MemorySource`, on probation) is deferred.

## [0.25.0] - 2026-06-19

### Added

- **`neo memory rules` — flag drift between `AGENTS.md` / `CLAUDE.md` / `GEMINI.md`.** A repo worked by multiple coding agents has multiple rule files, and teams update one while forgetting the others; no single tool notices because each reads only its own. This static cross-file diagnostic discovers the rule files at the repo root, parses each into rule units (bullets with wrapped continuations folded in; headings/fences/prose dropped), embeds them (Jina, shared infra), and reports two divergence kinds: **gaps** (a rule present in one file with no aligned equivalent in another, at cosine < 0.78) and **conflicts** (aligned-but-divergent pairs judged contradictory by a bounded, graceful LM judge — opt out with `--no-conflicts`). Read-only / flag-and-propose: it prints proposed reconciling edits (`Add to AGENTS.md: …` / `Reconcile: …`) but never writes files. Byte-identical files (e.g. a symlinked single source) and single-file repos report "in sync." Dogfooded on this repo, where it correctly surfaced one real gap (a diagnostics section added to CLAUDE.md but not AGENTS.md). `neo memory rules [--json] [--no-conflicts] [--cwd]`. (`memory/rulesync.py`, `cli.py`, `subcommands.py`; see `docs/solutions/rule-file-sync.md`)

## [0.24.0] - 2026-06-19

### Added

- **`neo memory issues --suggest-rules`** — for each surfaced friction, makes one bounded LM call to draft a preventive `AGENTS.md`/`CLAUDE.md` rule, populating the `Issue.suggested_rule` field (shown in both human and `--json` output). This closes the loop the issue diagnostic opened: v0.23.0 automated *detecting* recurring agent mistakes; this drafts the first version of the rule that would stop them, leaving the developer to review and apply (consistent with neo's read-only, advisory role). The LM-bearing step is confined to this flag — `find_issues` stays deterministic and LM-free — and is cost-bounded (one call per issue, highest-confidence first, capped at `_MAX_SUGGESTED_RULES`). Per-issue LM failure is graceful (leaves `suggested_rule` unset rather than aborting the batch); the adapter is built via `resolve_adapter` only when the flag is set. (`memory/issues.py`, `cli.py`, `subcommands.py`)

## [0.23.0] - 2026-06-19

### Added

- **`neo memory issues` — a read-only diagnostic that surfaces recurring frictions mined from transcript history.** The ingester already parses Claude Code / Codex / CAR transcripts into episodes and computes friction signals (tool errors, assistant clarification), but only ever used them to silently enrich retrieval. This new subcommand (`neo memory issues [--since 14d] [--min-cluster 3] [--json]`) flips that into actionable output: it clusters similar asks by embedding (Jina, the same vectors as fact retrieval — no extra model, no LM call) and reports recurring frictions as ranked, evidence-cited issues categorized `missing-tool` / `absent-guardrail` / `vague-rule`. **Read-only / no-consume**: `find_issues` never constructs `TranscriptIngester` and never touches the `transcript_watermark_*` files, so it is fully decoupled from fact admission and idempotent. The gate mirrors synthesis discipline (≥`min_cluster` members, ≥2 distinct sessions, ≥2 frictional members, ≥1 verbatim evidence span), deliberately biased toward precision: an error must look like a diagnostic (prefix `error:`/`fatal:`, typed exception with a colon, lint code, or a curated failure phrase) — Claude Code `<tool_use_error>` guards, bare exit-code banners, non-error command output, and code identifiers like `WorkflowError` are filtered out. Validated against real history across 8 local projects, where it turned 7 noise findings into 2 genuine cross-session frictions on the busiest repo. (`memory/issues.py`, `cli.py`, `subcommands.py`; see `docs/solutions/conversation-mined-issues.md`)

### Changed

- **Extracted the complete-linkage clusterer into `math_utils.cluster_by_similarity`.** The greedy complete-linkage loop in `store._cluster_by_similarity` is now a generic `cluster_by_similarity(items, embed_fn, threshold)` shared by REVIEW→PATTERN synthesis and the new issue diagnostic (behavior-identical; existing synthesis tests unchanged). (`math_utils.py`, `memory/store.py`)

## [0.22.4] - 2026-06-17

### Fixed

- **Outcome detection no longer over-reinforces a fact via suggestion fan-out.** One neo invocation links *all* its suggestions to a single reasoning fact (`engine._build_suggestion_fact_ids`), so both outcome paths could apply several reinforcements to that one fact from what is really a single acceptance/recurrence — inflating `success_count`, which gates probation-promotion *and* community contribution (`find_contributable`, `min_successes=3`). Now deduped per fact: the transcript miner (`mine_suggestion_outcomes`) reinforces each fact at most once per cycle (and compacts the fan-out sibling ledger entries so they can't drip later), and `detect_implicit_feedback` reinforces/demotes a given fact at most once per call (skipping duplicate MODIFIED REVIEW facts too). Mixed accept/modify across files of one fan-out resolves deterministically to the first outcome in suggestion order. Note: `MAX_MINED_OUTCOMES_PER_CYCLE` now bounds *distinct facts* per cycle rather than ledger entries — a more meaningful budget. Surfaced by having neo review its own codebase. (`memory/store.py`, `memory/transcript.py`)

## [0.22.3] - 2026-06-17

### Fixed

- **The save() torn-write window is closed with a cross-process file lock.** 0.22.1–0.22.2 made the request-path/observer merge-on-save correct but left a sub-microsecond residual: two processes could interleave between one's re-read and the other's atomic replace. `save()` now holds an exclusive `flock` on a per-scope sidecar `<file>.lock` across the whole read(merge)→write, so writers serialize and that window is gone. The lock is on a sidecar (not the data file, which `os.replace` swaps to a new inode mid-write), taken one scope at a time (no nesting/deadlock), released in `finally` on every path, and best-effort (degrades to the prior unlocked merge if `fcntl` is unavailable, never blocking a save). With this, a fact or a `success_count` reinforcement can no longer be lost under any save interleaving; the only non-lossless case left is two processes *independently* editing the same fact's confidence at once — both edits are always seen (no data loss), and reconciling them to one scalar is a documented policy choice (favor recorded reinforcement), which is a deliberate non-goal to make lossless (would need a per-fact operation log). (`memory/store.py`)

## [0.22.2] - 2026-06-17

### Fixed

- **Concurrent `success_count`/confidence reinforcements are no longer lost to last-writer-wins.** 0.22.1's merge-on-save preserved a concurrent writer's *added* facts but, for a fact present in both processes' memory, kept the in-memory copy wholesale — so a `success_count`/confidence bump one process committed could be erased when the other saved a stale copy. `_merge_on_save` now field-reconciles a same-id fact (`_reconcile_fact`) instead of overwriting: `success_count`/`access_count` (strictly monotonic) take the max of both copies, and confidence is lifted to the disk value only when that side recorded more successes. Crucially the reconcile keeps OUR record as the base, so our own independent edits (a MODIFIED confidence demotion, a supersession pointer, tags, effectiveness) survive — an "adopt the disk record wholesale" approach would have discarded them. A fact we invalidated this session still always wins (never resurrected). Remaining residual: when both processes independently edit confidence, resolution is deliberately best-effort (favoring recorded reinforcement) — lossless reconciliation there would require a per-fact version counter or a file lock. (`memory/store.py`)

## [0.22.1] - 2026-06-17

### Fixed

A concurrency fix that protects the learning loop's integrity — found by exercising 0.22.0 end-to-end.

- **`FactStore.save()` no longer lets the observer and a request-path invocation clobber each other's facts.** The async observer and an interactive `neo` run are separate processes writing the same `facts_project_<id>.json`. `save()` previously merged only the *global* scope on write and overwrote the *project*/*org* scopes with its in-memory snapshot — so whichever process saved last erased the other's just-added facts. Observed live: a `neo` invocation's linked reasoning fact was erased moments after creation because the observer (which had loaded the file earlier) saved its transcript facts on top, leaving the suggestion ledger pointing at a fact that no longer existed and silently breaking outcome linkage. The merge-on-save (re-read the file, preserve facts present on disk but absent from memory) now runs for **all** scopes, so a concurrent writer's additions survive. Two guards keep it correct and fast: facts physically removed this session (`_deleted_ids`, i.e. `purge_dead_facts`) are never resurrected by the re-read, so purge isn't undone; and an mtime fast-path skips the re-read+parse entirely when no other process has written since (avoiding a multi-MB re-parse on every `add_fact`). Same-id concurrent *field* edits (e.g. a `success_count` bump) remain last-writer-wins — a documented weak-signal residual that needs locking to close. (`memory/store.py`)

## [0.22.0] - 2026-06-17

### Added

neo's learning loop now closes: facts earn promotion from real outcomes instead of churning out of probation.

- **Transcript-driven outcome mining via a durable suggestion ledger.** The outcome detector only ever bumped a fact's `success_count` when a suggestion's `file_path` matched a git-changed file (ACCEPTED). But neo's dominant workload is review/analysis, whose suggestions carry synthetic paths (`/REVIEW.md`, …) that never match git — so on a real project only 3 of 123 facts had ever earned a success signal, and the rest were pruned within the probation window. neo now records linked suggestions to an append-only `suggestion_ledger_<project>.jsonl` (durable, so it survives the session-log clearing that git-based detection does on the next invocation), and the async observer mines it: each ledger entry is correlated against later AI-tool transcript episodes within a 2h window by **embedding similarity** (reusing the Jina vectors — no extra LM call). A match is evidence the suggestion's area recurred in subsequent work, so it earns a **weak** `UNVERIFIED` reinforcement (+0.1 confidence, +1 `success_count` — enough to promote a fact off probation), never the strong reward the git matcher reserves for verified diff-overlap. Compaction runs before the (non-idempotent) apply and unconditionally, so a crash loses a noisy signal rather than double-counting and the ledger stays bounded. (`memory/transcript.py`, `memory/outcomes.py`, `memory/store.py`)

### Fixed

Two feedback-loop and memory-hygiene fixes, plus a flaky test, all aimed at memory that actually grows instead of churning.

- **Outcome detection now sees review/analysis suggestions.** `_detect_non_git_outcomes` — which already emitted a weak acceptance signal for `/dev/null` and `docs/` paths — now also recognises review-document paths (`*REVIEW*.md`, `review/`/`reviews/` segments, the `NO_MODIFY` sentinel) via a deliberately narrow classifier, so the ~85% of suggestions that are structurally invisible to the git matcher finally produce a learning signal. Also fixes a pre-existing double-count (widened by the above): a path both git-changed and non-git-trackable now collapses to a single outcome (strongest signal wins) instead of bumping the same fact twice in one run. (`memory/outcomes.py`)
- **Cold-start purge reclaims scope-eviction orphans.** `purge_dead_facts` only removed tombstones whose supersession chain resolved to a valid successor, so facts invalidated by scope-cap *eviction* (no `superseded_by` pointer) were retained forever — per-project fact files bloated to 62–81% dead rows (up to 46 MB; neo's own project held 427 unpurgeable orphans despite constant cold starts). It now drops any fact that is invalid and untouched for 30+ days regardless of supersession, matching the `neo memory prune` compactor. (`memory/store.py`)
- **`test_live_oversized_prompt_does_not_crash_adapter` no longer flakes on a slow daemon.** The test whitelisted error-message substrings (`rpc`/`context`/`token`), so a graceful CAR daemon connect/read timeout — exactly the non-crash behaviour under test — failed it; it now accepts any readable `RuntimeError`. (`tests/test_car_adapter.py`)

## [0.21.0] - 2026-06-13

### Added

neo now learns from the behavioral record of AI coding tools — not just git — and can route inference through CAR when configured to.

- **Transcript learning (observer).** The synthesis observer was *starved*, not broken: on a real project, 92 diverse REVIEW facts produced 0 patterns because they don't cluster. A Phase 0 spike confirmed the fix isn't more clustering (diverse lessons don't cluster; the only clusters are paraphrase-dup artifacts) — it's richer ingestion. The observer now mines AI-tool session transcripts each cycle, extracts generalizable lessons via the configured LM, and admits the verified ones **directly as retrievable PATTERN/FAILURE facts** (no clustering step). Two hard admission gates: the cited `evidence_span` must appear **verbatim** in the source transcript, and an adversarial LM judge must keep the lesson. Lessons enter probationary at capped confidence (≤0.6) with `INFERRED` provenance, so they never out-rank corroborated facts and decay out if unhelpful. Bounded per cycle by an episode budget **and** a wall-clock deadline **and** a SIGTERM stop-check, so a hung provider can't stall the CAR-supervised process. (`memory/transcript.py`, `memory/observer.py`)
- **Multi-source ingestion via a `TranscriptSource` adapter interface.** Each tool is a small parser yielding a common `Episode`; the extract→verify→admit pipeline, the existing cosine supersession dedup, and a per-source watermark are reused unchanged. Three live sources: **Claude Code** (`~/.claude/projects`, project-scoped), **Codex** (`~/.codex/sessions` rollouts — project-scoped by `cwd`, errors read from `function_call_output`), and **CAR** (`~/.car/sessions`, global-scoped, finished-only + ask-deduped). Watermarks are namespaced per `(source, scope)`; the shared per-cycle budget spans sources. Codex/Cursor/etc. are drop-in adapters.
- **CAR-first inference mode** (`NeoConfig.inference_mode`, env `NEO_INFERENCE_MODE`). `"auto"` prefers CAR's dynamic router when `car-runtime` is importable **and** the daemon is reachable, falling back to the configured static provider on absence or runtime failure — via an `AutoAdapter` whose circuit breaker half-opens after a cooldown and whose static fallback is built lazily (so a CAR-only install with no static key still works). **Defaults to `"static"`** (the configured provider, e.g. gpt-5.5): CAR-first is a fully-built opt-in pending a CAR release that verifies the router's quality routing. (`adapters.py`, `config.py`)

### Changed

- **`add_fact` gains a `domain` parameter** threaded to `Fact.domain`, so transcript-derived lessons are matched by `retrieve_relevant(domain=...)` rather than lost in tags. (`memory/store.py`)
- Observer `car-runtime` version references aligned to the `>=0.18.0` floor (the gate is capability-based; the numbers are documentation), and the project-doc `subcommands.py` path corrected to `neo/subcommands.py`.

## [0.20.3] - 2026-05-26

### Fixed

Two observer crash modes and one upstream-pin bump — all aimed at the supervised observer surviving real-world macOS conditions.

- **`asyncio.get_event_loop()` → `get_running_loop()` in the observer coroutine** (`memory/observer.py:212`). On Python 3.14 the deprecated call has been observed raising `NameError: name 'asyncio' is not defined` from inside the coroutine launched by `asyncio.run(self._run_async())`, even though `asyncio` is imported at module scope and the same name resolves fine in `run()` a few lines up. `get_running_loop()` is the canonical API inside a running coroutine, dodges the deprecation, and stops the supervisor from chewing through `max_restarts: 10` on every restart.
- **fastembed cache pinned to `~/.cache/neo/fastembed/`** (`memory/store.py`). The default cache lives under `$TMPDIR/fastembed_cache/`, which on macOS is `/var/folders/<...>/T/` — periodically swept by the OS. After a sweep the in-cache manifest still points at a (now-deleted) `model.onnx` and `TextEmbedding(...)` raises `ONNXRuntimeError ... NO_SUCHFILE`, leaving FactStore silently running without embeddings until a human noticed. The new path survives reboots and tmp sweeps; on `NO_SUCHFILE` / missing-`model.onnx` the loader nukes the stale snapshot dir and retries once, so any future eviction self-heals.

### Changed

- **`car-runtime` pin bumped to `>=0.18.0`** for two upstream supervisor-reliability fixes that directly hit `neo memory observer`: `spawn_supervision` is now teardown-first with `kill_on_drop`, so a `start` issued during `Backoff` can no longer leave an orphaned child holding the agent's port; and restart backoff is exponential (base → ×2, capped at 60s) instead of flat, so a crash loop can no longer burn through `max_restarts` in under a minute. `last_exit_code` is also now cleared on successful (re)start, so `neo memory observer status` won't show a stale `1` next to a healthy `running` agent. No neo code changes required — the pin captures the upstream wins for new installs.

## [0.20.2] - 2026-05-25

### Fixed

The A2UI memory inspector ships against a real CarHost.app renderer for the first time. Three wire-schema bugs (silent failures the unit tests couldn't see) and one UX pass.

- **Tabs rendered empty** — A2UI v0.9's Tabs component takes **two parallel arrays** (`tabs: [{id, label}]` for metadata, `children: [contentId]` for content in matching order), per the renderer contract in `apps/car-a2ui-renderer/.../A2uiRenderer.swift:595`. The previous shape (a single `children` of nested tab-descriptor dicts) left the tab bar entirely blank — the renderer iterated nothing and showed no labels.
- **Badge rendered as an empty pill** — Badge's visible value lives on `text`, not `label` (`A2uiRenderer.swift:675`). With `label`, the renderer read empty string and drew the capsule with no content. Also pinned a literal `tone` because the renderer reads `obj["tone"].asString` directly — `{path: …}` references don't resolve there.
- **`List` with `forEach`/`itemTemplate` did nothing** — the v0.9 basic-catalog renderer maps `List` to `renderStack(.vertical)` (`A2uiRenderer.swift:465`); there's no template iteration. Dropped the dynamic recent-cycles list in favor of static stays-current Texts. A future surface revision can declare fixed slot Texts bound to indexed paths if a visible cycle log is desired.

### Changed (UX)

Inspector header redesigned to read like `neo --version` instead of supervisor jargon:

- **Header now mirrors `neo --version`**: personality quote (from `config/beats/neo_matrix.yaml`, stage-appropriate), repo display name (e.g. `Parslee-ai/neo` from the normalized git remote — not the SHA), `neo X.Y.Z` line, and a learning-stage summary (`Stage: Glitch · Memory 30.4% · 117 patterns · 0.58 avg confidence`). The CLI banner and the inspector header now read the same stage info — `a2ui._STAGE_TABLE` is kept in lockstep with `subcommands.show_version`.
- **Observer tab no longer redundant with the header** — dropped the "Idle, watching for patterns" status line that just restated the Observer card below it. Header stays summary-only; Observer card carries the operational detail (badge, last-check timestamp, cadence, action buttons).
- **Button labels say what they do**: `Kick` → `Run now`, `Stop` → `Pause`. The wire action names (`kick`, `stop`) are unchanged so handlers don't need to rename — only the user-visible label flipped.
- **Memory tab gains plain-English descriptions**: `50 patterns — stable techniques and conventions`, `58 reviews — recent observations being distilled into patterns`, `9 constraints — hard rules from project docs`. Pluralization handled (`1 pattern` vs `2 patterns`, `1 fact still being validated` vs `2 facts`). Scope line reads as `67 specific to this project · 50 from your global knowledge` instead of `project=67, global=50`.

## [0.20.1] - 2026-05-25

### Fixed

- **A2UI memory inspector surface rendered as "(no root component)"** in CarHost.app despite the daemon-side dataModel populating correctly. Root cause: A2UI v0.9's `CreateSurface` wire schema (`car-a2ui/src/lib.rs:120`) only carries `surfaceId` + catalog metadata — `components` and `dataModel` are silently dropped by serde (extra-fields-ignored). The agent-docs cookbook example shows them inside `createSurface`; the wire format doesn't actually accept that. `SurfaceManager.ensure_surface` now emits three envelopes in order: `createSurface` (surfaceId only) → `updateComponents` (the tree) → `updateDataModel` (initial state). For existing surfaces left empty by 0.20.0, re-emits ONLY `updateComponents` on reconnect — heals the tree without wiping live observer/memory state. Live-confirmed against CarHost.app: 17 components, correct root, observer + memory dataModel intact.

## [0.20.0] - 2026-05-24

Neo gains a long-lived presence: an out-of-band synthesis observer supervised by CAR's agent runtime, a live A2UI memory inspector that exposes both observer cycles and FactStore state to any conformant renderer, plus quieter foundational work on project-ID stability, metrics ergonomics, and a new fact-domain taxonomy.

### Added

- **Async synthesis observer (`neo memory observer {start|stop|status|kick}`)** — a per-project background process that runs `synthesize_reviews` on a wall-clock cadence (default 5 min), decoupled from the request path. **Additive**: the inline triple-trigger gate keeps firing too; the observer just makes synthesis more frequent. Lifecycle is owned by CAR's agent supervisor (car-runtime ≥ 0.17.0): spawn, restart-on-failure, log redirection to `~/.car/logs/`, clean SIGTERM shutdown, and auto-start at daemon boot via the persisted spec in `~/.car/agents.json`. Tunables: `NEO_OBSERVER_INTERVAL_SECONDS`, `NEO_OBSERVER_COOLDOWN`. Status surfaces CAR's raw state (`running | stopped | starting | backoff | errored`) so restart-loops are diagnosable. Hard footgun: the interpreter must not live under a world-writable directory (`/tmp`, `/private/tmp`, `/var/tmp`, `/dev/shm`) — the CAR daemon rejects such commands for security.
- **A2UI memory inspector surface (`neo.a2ui`)** — a per-project A2UI v0.9 surface (`neo-<project_id8>`) registered with the running `car-server` daemon so any conformant renderer (CarHost.app, future webviews) can inspect Neo's state live. Two tabs: **Observer** (status badge, pid, last cycle, recent cycles list, Kick/Stop buttons) and **Memory** (valid fact count, by kind, by scope, probation count). Both populate from the same `FactStore` load the observer's synthesis cycle already does — zero hot-path cost. Kick/Stop buttons emit `a2ui.action` notifications which the observer dispatches to `kick_observer` / `stop_observer`, closing the loop with CAR's supervisor. Activation: auto when `127.0.0.1:9100` is reachable; silent no-op otherwise.
- **`Fact.domain` taxonomy** — optional free-form area tag orthogonal to `FactKind`. `memory.models.SUGGESTED_DOMAINS` (`code-style`, `testing`, `git`, `debugging`, `workflow`, `security`, `file-patterns`, `architecture`, `performance`) is the recommended vocabulary; any string is valid. `retrieve_relevant(..., domain=...)` filters by exact match; `domain=None` returns all facts including unset ones; a specific filter excludes domain-unset facts.
- **`NEO_PROFILE` metrics gating** (`off | minimal | standard | strict`) replaces the single-knob `NEO_METRICS=off`. `minimal` emits only `lm_call` (audit trail without per-retrieval volume); `standard` emits everything (default, matches prior behavior); `strict` is reserved for future verbose events. `NEO_METRICS=off` remains a hard kill-switch that overrides `NEO_PROFILE`.

### Changed

- **`project_id` is now hashed from the normalized git remote URL** (`scope._compute_project_id`), not the codebase root path. The same repo on different clones / worktrees / machines now resolves to the same project ID. Falls back to a path hash when no remote is configured. Legacy path-hashed fact and watermark files in `~/.neo/facts/` are renamed in place on `FactStore` init (`store._migrate_legacy_project_id_files`) — transparent for upgrades.
- **`car-runtime>=0.17.0`** is now the floor for the `[car]` extra (bumped from `>=0.9.0`). The observer's `agents_*` lifecycle calls require the WS-routing fix that landed in v0.17.0 ([Parslee-ai/car-releases#54](https://github.com/Parslee-ai/car-releases/issues/54)) — earlier car-runtime versions still work for `CarAdapter`-only use, but `neo memory observer start` will refuse them with an actionable error pointing at the upgrade path.

### Fixed

- **Observer's `agents_list` shape parsing** — the CAR `ManagedAgent` shape uses a `status` *string* (`running | stopped | starting | backoff | errored`), not a `running` *bool*. Initial port checked `existing.get("running")` which was always falsy — meaning `stop_observer` returned "not_running" before ever calling `agents_stop`, leaking the supervised child. Fixed; `observer_status` now surfaces CAR's raw status verbatim. Regression test added.
- **`asyncio` missing from the observer module-scope import**. Surfaced only after the daemon spawn — caught by the live A2UI integration test, not unit tests.
- **`a2ui.apply` envelope was wrapped** under `params={"envelope": …}` instead of being passed as the envelope directly, silently creating an empty surface. Wire-shape regression test added (`tests/test_a2ui.py::TestWireShape`).
- **`pytest-asyncio` added to `[dev]` extra** — the new async a2ui tests failed in CI because the plugin was only in my local venv. Adds `pytest-asyncio>=0.21.0` to the manifest.
- **`websockets>=12.0` added to `[car]` extra** — needed by `neo.a2ui.DaemonClient` to speak JSON-RPC over the daemon's WebSocket (the Python `car_runtime.a2ui_*` helpers are in-process only).

### Docs

- CLAUDE.md / AGENTS.md gain bullets for: project-ID hashing + migration, domain taxonomy, profile gating, observer (incl. tunables and the world-writable-command footgun), and the A2UI surface architecture (including why the in-process `car_runtime.a2ui_*` helpers don't suffice).
- New `docs/solutions/async-observer.md` design proposal — borrows process-lifecycle ideas from ECC's `continuous-learning-v2` observer, ported onto CAR's `agents_*` supervisor. Includes an as-built note pointing readers at `src/neo/memory/observer.py`.

## [0.19.0] - 2026-05-17

CAR becomes a peer integration target alongside Claude Code and Codex; one OpenAI default-config regression fixed; docs realigned to what the code actually does.

### Added

- **CarAdapter** — route Neo's outbound inference through CAR's unified inference layer. With `model=None` (default), CAR's adaptive `route_model` picks local backends (Candle + MLX for Qwen3, Gemma 4) or remote providers (OpenAI, Anthropic, Google) per call. Pin via `model="gpt-5.3-codex"` etc. Activate with `neo --config set --config-key provider --config-value car` or env `NEO_PROVIDER=car`. Default `intent_hint={"task":"code"}` so the router knows it's serving code work; this is also Neo's local workaround for [Parslee-ai/car-releases#52](https://github.com/Parslee-ai/car-releases/issues/52), where `route_model` defaults are cost-biased rather than quality-biased for code prompts.
- **`car_inference.get_runtime()` singleton** — process-wide `CarRuntime` shared between the inbound A2A host (`neo serve`) and the outbound `CarAdapter`. Same state, same policies, same eventlog, single auth handshake. `car_host.run_server()` pulls from the singleton instead of constructing its own.
- **AGENTS.md ecosystem** — added Neo's own AGENTS.md (mirror of CLAUDE.md) and auto-ingestion of `{project}/AGENTS.md` / `{project}/.github/AGENTS.md` in `memory/constraints.py`. Codex-spec tools now see the same project rules as Claude Code.
- **`neo memory prune` subcommand** — compacts a project's facts file by dropping old invalid tombstones (default: >30 days since last access). Supports `--all`, `--dry-run`, `--limit`, `--max-invalid-age-days`.
- **`neo --dry-run "query"`** — print the assembled context without making the LLM call. Useful for debugging context-gatherer behavior.
- **OpenAIAdapter `/v1/responses` passes `max_output_tokens` and `reasoning.effort`** so gpt-5*/o-series generation knobs from NeoConfig (and engine memory-driven effort selection) actually take effect.
- **Construct library bundled into the wheel** as `neo/construct_library` via `[tool.hatch.build.targets.wheel.force-include]`. `neo construct list` now works for pip-installed users; was dev-tree only.

### Changed (migration)

- **`openai>=1.0.0` and `pyyaml>=6.0` are now core dependencies, not extras.** OpenAI is the default provider, so the base install is runnable on `OPENAI_API_KEY` alone. The `[openai]` extra still exists but is redundant; pinning workflows that install only with `[anthropic]` to avoid OpenAI dependencies should switch to a separate venv.
- **Provider env-var selection is now provider-matched, not first-found.** Previously `NeoConfig.load()` picked whichever provider env var was set first (so `OPENAI_API_KEY` was used even when `provider=anthropic`). Now it picks the env var that matches `config.provider` (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, or `AZURE_OPENAI_API_KEY`), with `NEO_API_KEY` as the explicit cross-provider override. Setups that relied on the old leak-through behavior must set the correct env var for their active provider.

### Fixed

- **OpenAIAdapter regression: `temperature` rejected by gpt-5*/o-series on `/v1/responses`** (introduced in `ad8718e`, fixed in `aa52ca0`). The hardening commit added `"temperature": temperature` unconditionally to the responses payload. gpt-5/codex/o-series reject it with `Unsupported parameter: 'temperature' is not supported with this model` — so default config (`provider=openai`, `model=gpt-5.5`) was broken on first call. Adapter still accepts the kwarg for `LMAdapter` ABC compat but no longer forwards it for these model families.
- **`neo --version` no longer eager-initializes FactStore.** Was paying the full embedding-pipeline cost (Jina model load + index build) just to read fact counts. Now `eager_init=False`.
- **`pipx` editable installs were misclassified as `INSTALL_EXTERNAL`** so auto-update silently no-op'd. `update_checker._detect_install_method()` now checks `sys.prefix` against the pipx venv path first.
- **`FileStorage.load()` backs up corrupt JSON before re-raising.** `JSONDecodeError` is handled separately from `PermissionError`/`IOError`: corrupt files trigger `_backup_corrupt_file()` so the original isn't silently destroyed by a subsequent save.

### Docs

- **README, QUICKSTART, INSTALL, and CONTRIBUTING restructured** so CAR, Claude Code, and Codex are documented symmetrically. CAR is the lead surface (`## Run as an Agent (CAR / A2A)` before the plugin sections); inbound (`neo serve`) and outbound (`provider=car`) directions both covered with Python and CLI examples; `intent_hint` knobs and `task=code` default documented in the LM Adapters CAR subsection.
- **`docs/tree-sitter-setup.md` rewrite**: migrated to `tree-sitter-language-pack` from the deprecated `tree-sitter-languages`; dropped the obsolete Python 3.13 exclusion; added Ruby/PHP/Swift/Kotlin to the supported list; fixed `MAX_CHUNK_LENGTH = 2000` chars (previously claimed `>100KB`); added the `code_smells._ERROR_SWALLOW_DETECTORS` registration step.
- **README Research & References rewritten** to match what's actually implemented. Previously cited five papers (Self-Planning, AdaCoder, As-Needed Decomposition, Multi-Agents Survey, Liu PG-TD) with "Implementation" claims that have zero code references. Now lists the 11 papers from the 0.18 memory architecture wave that ARE wired into code, with file-anchored citations. Wrong CodeSim paper (Xu 2023 vs the implemented Hou 2025) corrected; ReasoningBank moved to "Historical influences" since only the deprecated `persistent_reasoning.py` uses it. StateBench and memgine cited explicitly (was a one-word parenthetical).
- **CLAUDE.md memory hygiene rewritten with correct outcome deltas** (ACCEPTED +0.2 / MODIFIED −0.2 / UNVERIFIED +0.1, all ±arch_mod), the full `rank_score` formula including `effectiveness_f`, the triple-trigger consolidation gate, and two footgun warnings (new `OutcomeType` requires updating both `outcomes.py` and `store.detect_implicit_feedback`; `rank_score` is shared between FactStore and ContextAssembler). CLAUDE.md and AGENTS.md are now kept in sync as a deliberate invariant.

### Internal

- **CarAdapter test coverage**: 23 unit/integration tests in `tests/test_car_adapter.py` — 15 mocked (always run), 4 live integration gated on `car_inference.is_available()` (skipped if `car-runtime` not installed), 2 negative-case for runtime failure propagation, 2 wiring. Covers chat-message vs string-prompt paths, default vs explicit `intent_hint`, real-schema metrics (`model_used`/`prompt_tokens`/`completion_tokens`/`latency_ms`/`trace_id`), error propagation (ConnectionError, RuntimeError), bad model id, oversized prompt, and the singleton-shared-with-car_host identity invariant.
- **CarAdapter schema validated against `car-runtime 0.15.1`.** The first mocked-only release caught zero schema bugs by accident; running against the live daemon exposed three drift categories (`intent_hint`→`intent_json` JSON string; `temperature`/`stop` rejected; `model`/`input_tokens`/`output_tokens`/`cache_read_input_tokens` → `model_used`/`prompt_tokens`/`completion_tokens`/no-such-field). All fixed; FakeRuntime now mirrors the real shape.
- **Upstream bugs encountered while wiring CAR integration**: [Parslee-ai/car-releases#50](https://github.com/Parslee-ai/car-releases/issues/50) (classify mis-routes — CLOSED), [#51](https://github.com/Parslee-ai/car-releases/issues/51) (stale v0.8 reference in `open_session` error — OPEN), [#52](https://github.com/Parslee-ai/car-releases/issues/52) (router cost-biased for code prompts — OPEN; worked around in CarAdapter via `task=code` default).


## [0.18.1] - 2026-05-16

### Fixed

- **Linked memory feedback now actually improves retrieval ranking.** Accepted and unverified linked outcomes update LessonL-style effectiveness as positive feedback; modified linked outcomes update it as negative feedback. Previously facts could gain confidence/success counts without improving the effectiveness signal used by ranking.
- **Outcome linkage now survives absolute/relative path differences.** Suggestion fact IDs are normalized before matching outcome paths, so saved absolute suggestion paths match relative git outcome paths.
- **`neo memory replay-feedback` added for safe post-fix reprocessing.** Supports current-project replay, `--all` cross-project replay, `--dry-run`, `--limit`, and opt-in `--include-legacy-fallback`. Replays only linked outcomes and avoids independent-review synthesis/pruning side effects.
- **Processed session logs no longer replay through legacy fallback files.** Clearing a processed session now removes both `session_log_<project>.jsonl` entries and the legacy `session_<project>.json` fallback, preventing duplicate learning updates.
- **API keys no longer need to live in plaintext config.** `neo --config set --config-key api_key` securely prompts and stores provider keys in macOS Keychain; `NeoConfig.load()` reads them automatically. `config.json` writes `api_key: null` by default unless `NEO_ALLOW_PLAINTEXT_API_KEY` is explicitly set.
- **Pytest warning cleanup.** Script-style smoke tests now use normal pytest assertions/skips, and external SWIG deprecation noise is narrowly filtered.
- **Release artifact size.** Source distributions now exclude checked-in research PDFs and local coverage data, keeping the PyPI sdist small.

## [0.18.0] - 2026-05-15

This release is the result of a focused synthesis: 17 arxiv papers on multi-agent code generation, semantic memory, outcome detection, and consolidation were read, the deterministic techniques extracted, and the highest-leverage ones implemented end-to-end with neo + Linus agent reviews on the substantive changes. 38 commits, all paths verified against a real OpenAI run.

### Added — memory architecture (W1–W4)

- **Vectorized retrieval + unified scorer.** `FactStore.retrieve_relevant` and `ContextAssembler._score_facts` now share a single `rank_score(fact, similarity, now)` formula in `memory.models`. Cosine is batched in one numpy matrix-vector product (`math_utils.batched_cosine`) instead of per-fact loops. 3× speedup at 2000 facts; FAISS was prototyped and rejected after review showed numpy delivers the same win at this scale with half the code. Paper 2603.07670 §7.
- **Ebbinghaus recall-probability decay for fluid facts.** `FactMetadata` gains `recall_count`, `g_n`, `last_recall_ts`. Similarity through the transform `p_n(t) = (1 − exp(−r·exp(−t/g_n))) / (1 − e^-1)` so frequently-recalled facts decay slowly and dormant ones fast. Curated kinds (CONSTRAINT/ARCHITECTURE/DECISION + seed/community/synthesized tags) bypass the transform entirely. Paper 2404.00573.
- **LessonL effectiveness multiplier on `success_bonus`.** `FactMetadata.effectiveness_c` and `effectiveness_n` track per-fact reuse outcomes (`f = c/n` via `update_effectiveness(fact, outcome)`). f=1.0 default keeps legacy corpora ranking identically. Paper 2505.23946.
- **SCM 4-D ValueTagger composite + adaptive forgetting threshold.** `memory.value_score` module: `I(c) = 0.30·novelty + 0.20·validation + 0.35·task + 0.15·repetition` plus `θ_f = μ − σ·(|G|/target_size)` with floor 0.05. Paper 2604.20943.
- **NREM Hebbian strengthening + global downscale at synthesis.** Each cluster of ≥3 REVIEW facts bumps members' confidence by `min(0.1, 0.02·|cluster|)`; after the cycle, non-curated valid facts get a 0.97× global decay. Paper 2604.20943 §3.
- **Triple-trigger consolidation gate.** `synthesize_reviews` fires on count-delta ≥10 OR elapsed ≥1h OR Shannon-entropy of confidence deciles > 0.9. The count-only gate missed entropy-driven drift. Paper 2604.20943 §3.6.
- **Dual-buffer probation tagging.** New non-curated facts enter with a `probation` tag and the shortest stale-prune window (3 days vs 7 vs 14). Promoted out on `access_count ≥ 2`. Paper 2603.07670 §9.1.
- **Pre-write canonical-signature dedup.** `add_fact` computes a generalized signature (`memory.generalize`: entity abstraction + verb-synonym fold + context strip) and short-circuits identical canonical twins by bumping the existing fact's access count. Root-cause fix for the "independent flood" bug previously patched by post-hoc capping. Paper 2603.10600 §7.
- **Half-by-rank-score / half-by-cosine retrieval.** `retrieve_relevant` takes ⌈k/2⌉ by full rank_score and ⌊k/2⌋ by raw cosine — surfaces semantically-relevant facts with no track record alongside validated winners. Paper 2505.23946 Algorithm 1.
- **Bi-temporal `event_time` / `event_time_end` / `ingest_time`** on FactMetadata. Supersession soft-deletes by stamping `event_time_end` rather than dropping the row. Paper 2512.13564 §5.2.2.
- **Retrieval / context unit split.** `Fact.retrieval_text` (what we embed) and `Fact.context_text` (what we inject into the prompt) can now diverge — concise keywords for embedding, full narrative for context. Defaults to subject+body. Paper 2508.15294.
- **EPISODE FactKind + EpisodeContext.** Instance-specific events with `{when, where, why, with_whom}` that satisfy the 5-property test. Paper 2502.06975.
- **SimulationTrace persistence as EPISODE facts.** Every Neo run's simulator traces land in the fact store with retrieval/context split, episode_context, and bi-temporal stamps. Failure to persist is debug-logged, never propagates.
- **Provenance enum.** `STRUCTURAL > OBSERVED > INFERRED`, replacing free-string convention. Used in conflict-resolution precedence (newer event_time → higher provenance → higher cosine).
- **`memory.bm25` module + hybrid dense+BM25 retrieval.** Pure-stdlib BM25 (`k1=1.5, b=0.75`), min-max-normalized and weighted-summed `0.7·dense + 0.3·sparse`. Paper 2603.19935 §3.3.
- **Query-shape classifier + decomposer.** `memory.query_routing` distinguishes DIRECT / CHAIN / SPLIT prompts via regex (`"of the X of the Y"`, `"compare/contrast/vs"`); multi-hop and multi-entity get per-branch retrieval and merged. Paper 2604.04853 §5.3.
- **Nucleus episode expansion at retrieval.** Each surfaced EPISODE pulls up to 2 peer episodes from the same `source_prompt` (session), chronologically ordered. Paper 2604.04853 §4.6.
- **`outcomes.OutcomeIndicator` 4-class semantic classifier.** Pure regex on action logs: FAILURE / RECOVERY / INEFFICIENCY / SUCCESS, orthogonal to the existing event-shape OutcomeType. Paper 2603.10600 §4-5.
- **`outcomes.CodeOutcome` LessonL-style code classifier.** SPEED_UP / SLOW_DOWN / FUNCTIONAL_INCORRECTNESS / SYNTAX_ERROR from diagnostics + runtime logs + speedup ratios. Paper 2505.23946 §3.

### Added — smart file selection

- **ProjectIndex semantic boost in the gatherer.** `gather_context` now consults `.neo/index.json` (per-project FAISS over tree-sitter chunks) and projects top-k chunk hits back to per-file boosts up to +1.0 cosine. Test-file matches are demoted 0.4× unless the prompt itself mentions test/spec.
- **Tree-sitter symbol overlap as a scoring signal.** For the top 3× adaptive_limit filename-scored candidates, `_symbol_score` runs the parser, extracts function/class names + imports, and adds up to +1.2 (3 hits × 0.4) for substring matches against prompt tokens.
- **EPISODE-history feedback loop.** Each Neo run stashes touched file paths as `file:<rel>` tags on EPISODE facts. The gatherer's `_history_boost` queries the FactStore for similar past prompts and gives those files up to +0.5 boost on the *next* run. Closes the actual "Neo learns" loop — past behavior measurably influences future file selection.
- **Structured index embedding.** ProjectIndex now embeds `symbols + imports + first 600 chars of body` per chunk instead of raw chunk content. Eliminates the "tests outrank source files" bias caused by assertion strings containing prompt keywords verbatim.
- **Per-file chunk cap (2).** Large files no longer eat the adaptive-limit budget by splitting into 5+ chunks; the budget now produces +6 more unique files on representative prompts.
- **Score-weight fixes for main_impl files.** Halved the large-file penalty (0.001 vs 0.002 per KB-over-50) and made symbol matching substring-based with a length-3 floor. A 93 KB `engine.py` that was previously excluded from "fix the engine" prompts now reliably surfaces at rank 5.
- **First-run hint.** When `.neo/index.json` is absent, the gatherer prints `Tip: run 'neo --index' to enable semantic file selection` once.

### Added — engine pipeline

- **CodeSim-style MODIFY / NO_MODIFY decision token.** Simulator prompt instructs the model to emit `**NO_MODIFY**` or `**MODIFY: <reason>**` as the final reasoning step. `_simulation_consensus` parses the token (regex) and uses it as an explicit override of the agreement-of-outputs heuristic. Paper 2502.05664.
- **PlanStep.confidence + aggregate_confidence.** MapCoder-style per-step confidence as schema scaffolding for future multi-plan iteration. Paper 2405.11403.
- **`StructuredOverseer` watchdog wired into `process()`.** Daemon-thread tick loop emits `overseer_tick` events; detects 5-identical-actions-in-a-row as `is_looping=True`. Logged: `process.start`, `retrieve_context`, `lm_call`, `process.end`. Paper 2504.15228 §A.2.

### Added — observability

- **`memory.metrics` module — per-operation metrics jsonl.** Each retrieve / add_fact / lm_call / overseer_tick lands in `~/.neo/metrics.jsonl` with structured fields. Disable via `NEO_METRICS=off`. Path resolved lazily so test-harness HOME isolation works correctly.
- **LM-call token + cache-hit-rate observability.** Both AnthropicAdapter and OpenAIAdapter emit `lm_call` events with `input_tokens / cache_read_input_tokens / output_tokens / cache_hit_rate`. The OpenAI emitter handles both `/v1/chat/completions` and `/v1/responses` (gpt-5*/codex) shapes. Paper 2504.15228 Table 1.

### Fixed

- **UTF-8 strict decode dropped all git history ingest.** Six `subprocess.run` sites in `memory/outcomes.py` used `text=True` (strict UTF-8); a single non-UTF8 byte in any commit message raised `UnicodeDecodeError`, which the existing `except` block didn't catch. All six now pass `encoding="utf-8", errors="replace"`, and four `except` blocks now also catch `UnicodeDecodeError`. Before this fix Neo's own repo init silently ingested 0 git-history facts; after, it ingests all 50.
- **`engine._persist_simulation_episodes` referenced `step.action` (singular) but PlanStep declares `actions: list[str]`.** Every Neo run crashed at finalize, masking the simulation-episode persistence path entirely. Fixed.
- **Provenance string is now an enum.** `Provenance.STRUCTURAL / OBSERVED / INFERRED`. Backward-compatible — `add_fact(provenance=...)` accepts either the enum or its `.value` string. Paper 2603.07670 §7.3.
- **CLAUDE.md description of memory hygiene.** The "auto-consolidation every 10 entries" line was obsolete — replaced with accurate description of supersession + REVIEW-cluster synthesis + dual-buffer probation.

### Internal

- Single source of truth for fact ranking (`memory.models.rank_score`) used by both the FactStore and the ContextAssembler scoring paths.
- 17 cited arxiv papers checked into `papers/` for reproducibility.
- A 30-fact retrieval quality bench (`subject` + `body-snippet` queries) against a snapshot of the user's real `~/.neo` shows the new pipeline trades −3pp recall@1 for +4pp recall@5 vs pre-W1 baseline — matches the LessonL variance-reduction claim. Default `k=30` makes the breadth gain dominant.
- The `NEO_LEGACY_SCORING` env flag added during the A/B benchmark was ripped out after the bench ran. Flags are not for long-term retention.

## [0.17.0] - 2026-05-15

### Added

- **Full multi-language coverage for Ruby, Kotlin, Swift, and PHP.** Previously these extensions were recognized but produced no analysis. Now they get:
  - **God-file detection** via LOC and function/method counts from tree-sitter queries.
  - **Empty catch/rescue detection.** PHP uses the standard `catch_clause`; Kotlin and Swift use `catch_block` (empty = no `statements` child); Ruby uses `rescue` (empty = no `then` child), and the finding message reads "empty rescue block" so it isn't mis-labeled as a catch.
  - **Semantic chunking + import edges** for Kotlin (`import_header`), Swift (`import_declaration`), and PHP (`namespace_use_declaration`). Ruby imports are runtime method calls (`require`, `require_relative`) so they're left out of edge extraction for now.
- **Error-swallow detection for Go and Rust.** These languages don't have try/catch but do have idiomatic error-swallow patterns:
  - **Go:** `if err != nil { }` (and the `if err := f(); err != nil { }` initializer form) with an empty body. Detection matches any `if <expr-containing-nil> { }` empty block — catches both the canonical error pattern and the rarer nil-pointer-guard variant; both are smells in idiomatic Go.
  - **Rust:** `if let Err(...) = expr { }` with an empty body, and `match` arms with `Err(...)` patterns whose body is an empty block. `Err(_) => continue` and other non-empty-block bodies are not flagged. Discard idioms like `let _ = result;` and `result.ok();` are intentionally not flagged because they're often deliberate; flagging would produce too many false positives.
  - C error-handling patterns (`if (rc != 0) { }`, `(void)write(...)`) are deliberately skipped — reliable detection requires semantic knowledge about which functions return errors, which our structural-only detector can't provide.
- **Empty-catch detection for JS / TS / TSX / Java / C# / C++.** Tree-sitter-backed pass that flags `try { ... } catch (e) {}` (and the ES2019 optional-catch-binding `catch {}` form). Skips ERROR-subtree catches so mid-keystroke code doesn't false-positive. Surfaces alongside the existing Python `swallowed_except` detector in the "KNOWN ISSUES IN NEARBY CODE" prompt section.
- **Multi-language god-file detection in `architecture_metrics`.** LOC + function-count thresholds now fire for any language with tree-sitter queries (JS/TS/Java/C#/C++/Go/Rust/C/Ruby/Kotlin/Swift/PHP), not just Python. Cycle detection and max-nesting-depth remain Python-only (cross-language module-name semantics don't reconcile cleanly into one graph), so the `ArchSnapshot` now carries a separate `python_files_scanned` field and `ArchDelta.severity()` gates the cycle/depth channels on `python_coverage` to avoid reading no-coverage zeros as real signal.
- **`neo/languages.py` central language utility.** Single source of truth for extension → tree-sitter language, language → GFM fence tag, language → display name, and the accepted-fence-tag set used when parsing LM responses. Replaces four scattered per-module maps (`LANGUAGE_MAP`, `_TS_LANGUAGE_MAP`, `_FENCE_BY_LANGUAGE`, `_PROMPT_LANGUAGE_NAMES`). Includes `normalize_language_name()` so callers can pass any common form (`csharp`, `c#`, `c_sharp`, `py`, `rs`, etc.) and get the canonical tree-sitter name back.
- **Language-aware prompts in `algorithm_design.generate_code_from_design`.** Required keyword-only `language` parameter — the previous Python default would have silently mislabeled prompts as the self-correction loop gained non-Python paths. Includes a hardened code-block extractor that strips a leading variant fence tag (e.g. accepts `csharp` when we asked for `c_sharp`) only when it matches a known fence tag, preventing real code lines like `pass`, `done`, or `42` from being misread as tags.
- **Language-aware fence tagging in `pattern_extraction.extract_pattern_from_correction`.** Optional `language` parameter; `memory/store.py` derives it from `outcome.file_path` so the extraction prompt no longer hardcodes ```python when the code is JS/Go/Java.
- **`neo serve` — CAR-hosted tool surface for `neo.process` (Phase 1 of the CAR migration).** Opt-in via the new `[car]` extra (`pip install 'neo-reasoner[car]'`); requires a running `car-server` daemon. Boots a `CarRuntime`, registers `neo.process` with a JSON-Schema-typed `params` shape derived from `NeoInput`, installs a Python `tools.execute` handler (closes the consumer-facing half of Parslee-ai/car-releases#38 — first downstream user of the new PyO3 surface), and binds car-server's A2A HTTP listener so an Agent Card is served at `/.well-known/agent-card.json`. CAR-native callers can submit proposals against `neo.process` today and reach the handler end-to-end. **Known limitation**: car-server's `start_a2a` spins up an isolated Runtime, so FFI-registered tools don't yet appear in the Agent Card's `skills` list and A2A peer dispatch for `neo.process` doesn't route back to Python — pending a CAR-side runtime-sharing fix in `car-server-core::a2a::start_a2a`. Tracked CAR-side; the CAR-native path is unaffected.
  - New: `src/neo/car_tool_schema.py` (tool schema + NeoInput/NeoOutput dict converters), `src/neo/car_host.py` (server entry, per-codebase NeoEngine cache, signal-driven shutdown), `tests/test_car_tool_schema.py` (15 unit tests), `tests/test_car_host_smoke.py` (live end-to-end smoke; auto-skipped without daemon).
  - CLI: `neo serve --a2a-bind HOST:PORT [--public-url URL] [--agent-name NAME]`.

### Changed (breaking)

- **`tree-sitter` is now a required dependency, not optional.** Empty-catch detection, multi-language god-file metrics, and the semantic-chunking code path it powers are core features — the previous `[tree-sitter]` extra was a fiction, and on Python 3.13+ it could not install at all because `tree-sitter-languages` had no wheels. Switched to `tree-sitter-language-pack` (`>=0.13.0,<1.0`, the maintained successor) and promoted both `tree-sitter` and the language pack into the base `dependencies` block. The graceful-degrade `TREE_SITTER_AVAILABLE` scaffolding in `language_parser.py`, `code_smells.py`, and `architecture_metrics.py` has been removed; install neo and tree-sitter installs with it. Anyone who was pinning `tree-sitter-languages` directly needs to switch to `tree-sitter-language-pack`.

### Changed

- **`static_analysis`: pyright and mypy now both run when both are enabled.** Previously the dispatch had a hidden `pyright … elif mypy` mutex that suppressed mypy whenever pyright was enabled. They flag different things, so the coupling cost real diagnostics. Users running both intentionally now get both sets of findings.
- **`static_analysis` dispatch is registry-driven.** Hand-coded `if/elif` for Python and JS/TS was replaced with a `_LANGUAGE_CHECKERS` tuple pairing tool name, run function, and target extensions. `detect_available_tools()` derives from the registry so the PATH-detection list can't drift. Adding a new analyzer (e.g. a Go vet wrapper) is now one registry entry plus the run function, no dispatch changes.
- **`context_gatherer` scoring no longer biases toward neo's own filenames.** The `main_impl_patterns` list used to hardcode `'neo.py', 'persistent', 'context_gatherer', 'structured_parser', 'schemas.py'` — neo's own modules. When run against any other codebase that was either dead weight or actively wrong. Replaced with generic stem-equality matching (`main`, `index`, `app`, `server`, `lib`, `core`, `engine`).

### Removed

- **`neo.self_correction` and `neo.input_templates` modules deleted.** Both dead since v0.7.0 — commit 6e35f5d acknowledged the pattern-learning system had been unreachable since the initial public release, but the "fix" only revived sibling modules. `self_correction.py` had no callers; `input_templates.py` was imported only from `self_correction.py` and an import-health smoke test, so deleting one orphaned the other. The live learning path runs through `neo.pattern_extraction.extract_pattern_from_correction` via `memory/store.py`. If you were importing from either module, the surviving surface lives in `neo.pattern_extraction` and `neo.algorithm_design`.

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
