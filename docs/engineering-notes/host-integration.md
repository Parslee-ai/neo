# Host communication, test isolation, engine lock, plugins, edit hook, adapter parity

Detailed engineering notes moved verbatim from `CLAUDE.md` (2026-10-12). `CLAUDE.md` keeps the short rules; this file keeps the reasons, measurements and footguns behind them.

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
  **`headline` is for people, `summary` for models** (2026-10-18). The mod's
  band showed the summary, and a sharp review finding (trailing spaces passing
  a SQL Server CHECK) reached the person as `2 files. 2 change(s). 0.96. I've
  seen this shape before. Let me use what I remember.` — counts, a bare score
  and a canned line, nothing found. `_headline` is the first sentence of the
  first change's description (else the first plan step) through
  `models.spoken_line` (no severity tags, markdown or directories), led by the
  earned beat or the stage's `headline_lead`, with a stage-neutral tail when
  the confidence is low or no diff backs it. Beat lines are spoken AFTER the
  run in front of that finding, so they are short and never promise work to
  come.
  **The model writes the headline in the call it already makes** (same day).
  "LM-voiced summaries were rejected" (below) was about a SEPARATE call per
  run; the review, diagnosis and plan-check runs already make one, so the
  fast path asks for a fourth `headline` block and the panel's coder adds a
  `headline` field to its JSON — the coder's FINAL (post-repair) answer, and
  only when the engine accepts the panel, so a rejected draft never speaks.
  The instruction is `headline_instruction` in the deck, filled with the
  stage's `base_expressions` register. `models.voiced_headline` rejects a
  line over `SPOKEN_LIMIT`, with markup, or leaking a score, a count of the
  run's artifacts or a path; then the derived headline stands. It does NOT
  demand one sentence: Neo talks in short ones, and that rule threw away a
  live run's "Empty input raises IndexError. Is that intended, or should it
  return an empty string?" A cheaper second model was rejected: another round trip and
  failure mode inside the hooks' 90 s deadline, with less context than the
  call that found the problem. `neo recall` makes no model call and stays
  templated.
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
  installs, `claude plugin validate` passes, and nothing fires. This repo shipped
  the wrong layout, and so did CAR, which is why
  `test_the_manifest_directory_holds_no_components` fails on any non-manifest
  entry rather than merely asserting the components exist at the root (a stray
  copy left behind keeps every other assertion green).
  **`claude plugin list` is the check, and `claude plugin details` is NOT.**
  `details` reports the inventory it finds on disk and says nothing about load
  state — it prints all six skills, the agent and the hook for a plugin that
  failed to load, verified directly. Only `list` reports load state. This
  distinction is not pedantic: #221 verified the layout fix with `details`,
  read `Skills (6) Agents (1)` as success, and shipped a manifest that made
  the plugin fail to load in every release from 0.47.0 to 0.53.0 (#242).
  A second rule falls out of that one — **`manifest.hooks` must never name
  `hooks/hooks.json`**. Claude Code loads that path automatically and rejects
  the duplicate, failing the WHOLE plugin; the key is for *additional* hook
  files only. Pinned by
  `test_the_manifest_does_not_redeclare_the_auto_loaded_hooks_file`.
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
- **Host adapters must stay in parity.** There are THREE checked-in integration
  surfaces, and they are easy to miss: the Claude Code plugin (manifests in
  `.claude-plugin/`; agent, 6 slash commands and the hook at the REPO ROOT —
  see the layout rule above), **`plugins/neo/`** (Codex CLI plugin + 6 skills,
  manifest at `plugins/neo/.codex-plugin/plugin.json`, registered by
  `.agents/plugins/marketplace.json`), and **`plugins/cursor-neo/`** (Cursor
  plugin + 6 skills + Neo agent, manifest at
  `plugins/cursor-neo/.cursor-plugin/plugin.json`, registered by
  `.cursor-plugin/marketplace.json`). `.agents/skills/` holds RELEASE
  maintenance skills, not the Neo capabilities — looking there and concluding
  the Codex or Cursor plugin is missing is a documented wrong turn. No adapter
  owns an output format: all invoke `neo --json` and read the same `orchestrator`
  envelope. When the contract changes, ALL change — the Codex skills sat on
  "parse the four structured sections: CONFIDENCE, PLAN, SIMULATIONS, CODE
  SUGGESTIONS" while the Claude side had already moved to `--json`.
  `tests/test_host_adapter_parity.py` pins it: all three surfaces expose the same
  six capabilities, invoke `--json`, teach `orchestrator.summary`/`cautions`,
  document the error shape and attribution, and use phase/event names that
  actually exist in `events.py`. It also pins that all plugin manifests match
  the `pyproject.toml` version — `prepare-release` documents bumping them, but
  a documented step with no enforcement gets skipped (they had reached 0.19.0 /
  0.37.0 / 0.41.0). **The version is now single-sourced**: `pyproject.toml` is
  the truth and `tools/sync_version.py` (`make sync-version`) propagates it to
  `src/neo/__init__.py` and the Claude / Codex / Cursor manifests. Release bumps
  edit ONE file; never hand-edit the derived files. `--check` reports drift
  without writing. Edits are surgical regex replacements, not `json.dumps`
  re-serialization — a version bump must stay a one-line diff or it becomes
  unreviewable — and only the FIRST `"version"` key is rewritten so a nested
  one can't be clobbered.
  `tools/` is now covered by lint; it never was, which is why a dead import sat
  in `ab_controlled.py`. The ruff invocation is duplicated in `Makefile`
  (`lint` + `ci-local`), `.github/workflows/ci.yml` and `.githooks/pre-commit`
  — all four MUST stay identical or local green stops predicting CI green. Role
  differs even though protocol does not: Claude Code (and the Cursor **agent**)
  delegates to Neo as a subagent (visible boundary); Codex and Cursor **skills**
  call Neo mid-loop (no boundary), so those skills carry stronger attribution
  wording and an explicit "Neo's result is an input, not the deliverable".
  Cursor mid-loop skills use Codex's `--no-scan` / advise `--no-memory` defaults;
  the Cursor agent keeps Claude's looser advise invoke.
