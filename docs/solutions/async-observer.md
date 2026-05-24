# Async out-of-band synthesis observer

Proposal for moving REVIEW→PATTERN/FAILURE synthesis off the hot path, modeled
after ECC's `continuous-learning-v2` observer
(`affaan-m/ECC:skills/continuous-learning-v2/`).

> **As-built note (post-v0.19):** the first cut of this proposal rolled its own
> daemon (`subprocess.Popen` + PID file + SIGUSR1 kick + idle-exit). That
> implementation was replaced before any release with a port to CAR's
> `agents_*` lifecycle API (car-runtime ≥ 0.16.1). The motivation, ECC
> reference points, and design rationale below remain accurate; the
> "Process model" section's plumbing is **superseded** — CAR's supervisor
> owns spawn, restart-on-failure, log redirection, and clean shutdown.
> `~/.car/agents.json` is the persisted spec; `~/.car/logs/neo-observer-<id8>.{stdout,stderr}.log`
> is where output lands. `kick` maps to `agents_restart` since CAR has no
> signal-passthrough primitive. See `src/neo/memory/observer.py` for the
> current shape.

## Status quo

Synthesis runs inline in `memory/store.py:1722-1765` under a triple-trigger gate
(any of: `count_delta ≥ 10`, `elapsed ≥ 1h`, `entropy > 0.9`). It fires on the
caller's request path, blocking the return. Today that's tolerable because the
work is cheap (clustering REVIEW facts and writing summaries) — but it forces a
conservative gate. We can't cluster more aggressively without paying the latency
on every `add_fact`.

ECC's design demonstrates that moving this work to a background process with a
cheap model (Haiku) lets you analyze more frequently *and* with a richer
prompt-driven analysis than what the inline numeric path can do.

## Design

### Process model

A long-lived background process per project, started on demand, that:
1. Tails `~/.neo/metrics.jsonl` (filtering to the active `project_id`).
2. Wakes on an interval (`NEO_OBSERVER_INTERVAL_SECONDS`, default 300).
3. Runs a cooldown gate (`NEO_OBSERVER_COOLDOWN`, default 60s) to prevent
   rapid re-triggers from SIGUSR1.
4. Calls the synthesis pipeline (`memory.generalize` + `memory.store.consolidate`)
   with the last N events (`NEO_OBSERVER_MAX_LINES`, default 500).
5. Writes new PATTERN/FAILURE facts back through the normal `add_fact` path
   (which already handles dedup at cosine 0.85).
6. Self-exits after `NEO_OBSERVER_IDLE_SECONDS` (default 1800) without activity.

Lifted directly from `agents/observer-loop.sh`: re-entrancy guard,
analysis cooldown, idle exit, tail-based sampling, SIGUSR1 nudge.

### Lifecycle

- **Start**: `neo memory observer start` — daemonizes (`nohup`-style) and
  writes PID to `~/.neo/sessions/<project_id>/.observer.pid`.
- **Status**: `neo memory observer status` — checks PID liveness, prints
  observations-since-last-analysis count.
- **Stop**: `neo memory observer stop` — SIGTERM to PID.
- **Nudge**: `neo memory observer kick` — SIGUSR1 to PID, forces an analysis
  cycle subject to cooldown.

Auto-start: when `NEO_OBSERVER=auto` is set, the first `add_fact` of a session
spawns the observer if one isn't already running. Default is `off` (opt-in
during validation).

### File layout

```
~/.neo/
├── facts/                       # unchanged
├── metrics.jsonl                # unchanged — single source of events
└── sessions/
    └── <project_id>/
        ├── .observer.pid
        ├── observer.log
        ├── observer.last-analysis  # epoch of last successful run
        └── observer.watermark      # byte offset into metrics.jsonl
```

The watermark replaces neo's current synthesis watermark (kept in fact-file
headers) so the observer can stream forward without re-reading.

### Inline synthesis fallback

When the observer is running and healthy (PID alive, last-analysis < 2× interval
ago), the inline triple-trigger gate in `store.py:_maybe_consolidate` becomes a
no-op. When the observer is stopped or stale, inline synthesis continues to fire
as it does today. This keeps the system safe by default — turning off the
observer never causes synthesis to stop entirely.

## Why this is the right shape for neo

1. **Decouples cadence from request load.** Today synthesis fires on whatever
   call happens to cross a threshold. The user pays for it. Decoupling means we
   can run synthesis on the wall clock instead.

2. **Enables cheaper, more frequent passes.** Today we run synthesis at
   `count_delta ≥ 10`. With an observer we can drop that to ≥3 because the cost
   is amortized across the background process. ECC ships Haiku for this exact
   reason; neo can route through CarAdapter with `intent_hint={"task":"cheap"}`.

3. **Cleaner observability.** All synthesis lands in `observer.log` with
   timestamped cycles. Today it's interleaved with the request that triggered it.

4. **Matches existing neo primitives.** `~/.neo/sessions/` already exists.
   `metrics.jsonl` is already the event log. `add_fact` already handles dedup.
   The observer is plumbing on top of stable foundations — no new data model.

## What to borrow verbatim from ECC

Read while implementing — patterns that solved real production bugs:

| ECC file | What it teaches |
|---|---|
| `agents/observer-loop.sh:13-29` | Re-entrancy guard + signal-driven nudge with PENDING_ANALYSIS flag |
| `agents/observer-loop.sh:109-145` | Tail-based sampling (`MAX_ANALYSIS_LINES`) to prevent multi-MB LLM payloads — fix for ECC issue #521 |
| `agents/observer-loop.sh:71-85` | Idle-exit logic so observers don't run forever on dormant projects |
| `agents/session-guardian.sh` | Cheap-first gate ordering: time-window → cooldown → idle. Adopt the *pattern* not the time-window (we don't want active-hours gating) |
| `agents/start-observer.sh:218-227` | Confirmation-prompt sentinel: detect when the observer's LLM asks for permission and fail closed. Critical safety check |
| `hooks/observe.sh:148-181` | The 5-layer skip cascade (entrypoint, profile, env-var, agent_id, path-exclusion). Prevents the observer from observing its own runs |

## What NOT to borrow

- **Bash implementation.** Neo is Python — write the loop in Python with
  `signal.signal(SIGUSR1, …)` and `multiprocessing.Process` or a detached
  subprocess. Bash is the wrong substrate for our existing memory module
  imports.
- **YAML-frontmatter instinct files.** Neo already has JSON facts with
  embeddings. Synthesis output goes through `add_fact` — don't introduce a
  second schema.
- **CLI `claude --print` invocation.** Route LM calls through neo's existing
  `lm_logger` and CarAdapter so we get metrics emission and routing for free.
- **The Windows App Installer stub detection** (`observe.sh:42-67`). Neo
  doesn't currently support Windows; if we ever do, port this then.

## Phasing

1. **Phase 1 — Daemon scaffolding.** `neo memory observer {start,stop,status,kick}`,
   PID file, log file, watermark, idle-exit. No actual synthesis yet — just
   prints "would synthesize N events" each cycle. Validates the lifecycle on
   real sessions.
2. **Phase 2 — Wire synthesis.** Move the `_maybe_consolidate` call from
   `store.py` into the observer loop. Inline gate becomes a no-op when observer
   is healthy. Validate that facts/PATTERN counts converge to the same place as
   the inline path on a replay corpus.
3. **Phase 3 — Aggressive thresholds.** Drop `count_delta` gate from 10 to 3
   inside the observer (inline keeps 10 as a safety net). Measure synthesis
   quality on the StateBench corpus.
4. **Phase 4 — Auto-start.** Add `NEO_OBSERVER=auto` and `add_fact` spawn-on-first-use.

Each phase is shippable. We can stop at Phase 2 if Phase 3 doesn't show quality
gains.

## Open questions

- **Should the observer run per-project or per-machine?** ECC runs per-project
  (each has its own PID file). Neo's `metrics.jsonl` is global; per-project
  observers would need to filter by `project_id`. Easier: one observer per
  machine that handles all active projects. Recommend per-machine to start.
- **What's "active"?** ECC uses the metrics file's mtime. Same approach works
  here — if `metrics.jsonl` hasn't been touched in 30 minutes, exit.
- **Routing cost.** Each observer cycle costs ~1 cheap LM call. At 5-min
  intervals over an 8h day, that's ~96 calls. Worth measuring before defaulting
  on.
