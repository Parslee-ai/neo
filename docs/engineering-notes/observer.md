# Observer, resident embedder, A2UI inspector, observability, disk hygiene

Detailed engineering notes moved verbatim from `CLAUDE.md` (2026-10-12). `CLAUDE.md` keeps the short rules; this file keeps the reasons, measurements and footguns behind them.

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
  (`max_projects_per_cycle`, default 25; watermark- AND mtime-gated so unchanged projects do near-zero work (the watermark alone gates only *admission*: sources still parsed every transcript each cycle, measured at 298 MB for one project, which is what drove multi-GB observer RSS. `_unchanged_since` skips files untouched since the source's drain mark minus a 1h skew margin; every error path falls back to parsing, because a wrong skip loses learning silently. **The drain mark, never the watermark file's mtime** — that file is rewritten after every consumed episode, so a pass stopped by the 8-episode budget armed the skip against the very transcript still holding its backlog; measured live, 152 of 1,800 episodes were stranded that way. The mark (`transcript._drain_mark`) is `min(pass start, earliest timestamp of any episode still unconsumed)`: safe because a transcript's mtime is never earlier than its records, and it keeps advancing while a backlog drains (a mark set only on a fully-drained pass never moves for a busy project, i.e. full parse every visit). An unreadable pending timestamp leaves the previous mark. Marks are keyed **per transcript directory** (`codebase_root`) inside the project_id-keyed watermark, because two clones share that file and one clone draining must not arm the skip against the other. A legacy watermark has no mark and parses everything once, which also recovers the stranded)). **An LM failure is not "no lessons"** (`transcript.LMUnavailable`): `_lm_json` used to swallow every exception AND treat unparseable output as empty, so each of 610 live DNS failures consumed an episode unmined — unrecoverable, since nothing records which. **A failure is charged to an episode only when an answer vouches for the provider** — an answer during that episode (extract answered, then the judge call failed or the output was unusable) or in a LATER episode of the same pass; an answer from earlier in the pass proves nothing about calls made after it (a key revoked after episode 1 would otherwise strike every remaining episode). Failures wait in `held` until then. Two held failures in a row stop the pass AND the observer's sweep, whatever the error class — a revoked key (401), disabled billing (403) or quota 429 fails every episode alike, and charging per episode wrote off the whole backlog in three cycles. `ingest(provider_known_good=True)` (an earlier project answered this observer cycle; the observer clears it once a project reports unavailable) is the one exception: it charges a lone held failure at the end of a pass, and the two held failures at a stop, since two poison episodes are then likelier than an outage beginning at that instant and would otherwise stop the sweep on every visit. The sweep does NOT rewind its round-robin offset on that stop: a rewind re-ran the same batch in the same order, so two unanswered failures in a project nothing earlier vouched for wedged the sweep at the same place forever. Once charged, `_is_transient_lm_error` (connection/timeout/`gaierror`, SDK connection errors, 408/429/500/502/503/504 via `status_code` or google-genai's `code` on the cause chain — NOT 409/501/505) picks the cap: `MAX_EPISODE_LM_FAILURES` (3) or `MAX_EPISODE_TRANSIENT_FAILURES` (20), so neither a poison episode nor one whose prompt always times out is retried forever. Known limits: a lone failing episode with no evidence either way (overnight, or `--cwd` single-project mode) is never charged — one failing call per visit, and it pins its source's drain mark so that directory is re-parsed every visit; drain-mark safety assumes a transcript's mtime is not earlier than its records, which holds for Claude Code/Codex append-only files and for cp/rsync, not for a clock stepped back past the 1h margin; a Codex rollout that changes owning root when a nested peer leaves discovery can be skipped by the parent's mark; `_aware_epoch` (not `_episode_epoch`, which reads naive times as local) feeds the mark. All of an episode's LM calls run before its first admission — admitting part-way and retrying would re-extract the same lessons in new wording that exact-signature dedup does not catch — running transcript mining per project. (It also ran
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
  `NEO_OBSERVER_RECYCLE_CYCLES` cycles (default 3, ~15 min; 0 disables). Its RSS is
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
  **Resident embedder** (`memory.embed_service`, #211): the observer also loads
  the embedding model at start and serves it over `~/.neo/run/embed.sock`
  (0600, in a 0700 dir; newline JSON, protocol v1; hello carries model, dim and
  the fastembed/onnxruntime versions, all of which must match the client's, since a
  different preprocessing would drift query vectors from stored ones silently).
  The model is ~1.3 GB resident and WAS a neo run's whole memory floor:
  measured on a 2,209-fact store, a dry run peaked at 1,439 MB loading it and
  190 MB borrowing it. The observer therefore sits ~1.3 GB above the figures
  above, by design, and reloads the model (~0.3 s) on every recycle. The sweep
  shares the instance (`build_resilient_embedder` returns the process's one
  embedder, in `_Serialized` because server threads and the sweep call it
  concurrently). **Falling back costs 1.3 GB, so the client waits where waiting
  works**: requests are batched (64 texts, 16 KB) and the server embeds one
  text at a time, so a short query waits for one text, not another run's
  batch (64 texts of 800 chars hold the model ~15 s; one 32,000-char text
  33-75 s, since attention is quadratic). The deadline (90 s, above the worst
  single text, plus 3 ms per
  character) covers the whole exchange, so a trickling peer cannot stretch
  it. Texts are cut at 64K chars, past the model's 8,192-token window.
  Transport failures are retried for `RECOVER_S` (15 s) when they look like a
  recycle (socket gone) or a full backlog (refused while the observer's flock
  is held); a refusal with no observer alive is a socket left by a killed
  one and is abandoned at once, or every run would pay the wait. The backlog
  is `SOMAXCONN` (5 refused the sixth concurrent hello on macOS). The observer
  clears a stale socket before anything that can fail, so a successor that
  cannot load a model does not leave runs waiting on its predecessor's. A silent or
  incompatible server, or one that stays gone, ends in the run loading the
  model itself, once; a process that has loaded its own keeps it (an ONNX
  session's memory does not reliably go back to the OS), and a failed local
  load is remembered so a broken cache is healed once, not per call. Each
  process records `embedder_source` (remote / local / fallback) in metrics.
  The server is stopped in a `finally`, and restarted if a re-exec fails. The
  serving process never borrows from itself (`mark_server_process`). POSIX
  only; Windows always loads locally.
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
