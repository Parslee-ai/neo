# neo --dry-run

Detailed engineering notes moved verbatim from `CLAUDE.md` (2026-10-12). `CLAUDE.md` keeps the short rules; this file keeps the reasons, measurements and footguns behind them.

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
  **A dry run never touches the CAR daemon**: `cli.main` skips observer
  autostart and Lattice autojoin under `--dry-run` (pinned by
  `TestNoDaemonSideEffects`, which has a control so it cannot pass vacuously).
  **A dry run does not modify the fact store**, which is narrower than "mutates
  nothing" and is the claim that survives measurement. The old implementation got it
  for free by never constructing a `FactStore`; `FactStore.initialize` runs
  `prune_stale_facts` → `demote_unhelpful_facts` → `purge_dead_facts` and then
  **saves**, and `demote_unhelpful_facts` lowers confidence and invalidates facts — so
  reaching the engine at all meant a "read-only" inspection was aging the store it
  inspected. `FactStore(read_only=True)` makes `save()` a no-op at the single write
  choke point, which a new caller cannot forget. **It also skips every ingester**
  (seed, community, constraints, Claude memory, git history): each records "done"
  outside `save()` — a checksum file, a git watermark — so a read-only run that came
  first after a CLAUDE.md, memory-file or package change marked it ingested while
  its facts died with the no-op save, and the real store never got them (Lattice
  lookups are read-only too, so this was not dry-run-only). A read-only store reads
  only previously ingested inputs, so a dry run right after editing CLAUDE.md does
  not show the edit until a writable run ingests it; construction is still not
  filesystem-pure (directories, legacy renames) and maintenance still runs in
  memory; `dry_run` also skips
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
