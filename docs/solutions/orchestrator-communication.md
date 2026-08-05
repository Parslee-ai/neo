# Solution: Orchestrator Communication — Lifecycle Events + a Host-Facing Summary

**Date**: 2026-08-05
**Status**: Implemented
**Type**: design + implementation

## Implementation summary

| Piece | Module | Tests |
|---|---|---|
| Event vocabulary + sinks (`Null`/`Recording`/`Jsonl`) | `events.py` | `test_orchestrator_events.py` |
| Engine emission at phase boundaries | `engine.py` (`_emit`, `_begin_phase`, `_end_phase`, `_current_phase`) | `test_orchestrator_events.py` |
| Derived host summary | `engine.py` (`_build_orchestrator_message`), `models.py` (`OrchestratorMessage`) | `test_orchestrator_events.py` |
| CLI stream split | `cli.py` (`--json` → `JsonlSink(sys.stderr)`) | (live-verified; see below) |
| Staged voice + beat surface metadata | `config/beats/neo_matrix.yaml` (`orchestrator_voice`), `engine._voice` / `_voice_stage` / `_orchestrator_beat` | `test_orchestrator_events.py` |
| Progress-notice suppression (`--quiet`) | `progress.py`, `cli.py`, `context_gatherer.py` | `test_progress_quiet.py` |
| Plugin presentation contract | `.claude-plugin/agents/neo.md`, `commands/*.md` | — |

## The problem

`engine.py` made **zero** print or stderr calls. A host — Claude Code, an MCP
server, an IDE — saw nothing between invoking Neo and receiving the final
`NeoOutput`, which left it narrating a black box for 5–30 seconds.

Worse, the Claude Code plugin told the agent to run `neo --mode advise` and
parse the **human-readable text** output, even though `--json` already existed.
So the host was reverse-engineering presentation out of terminal formatting
while a fully structured document sat one flag away. Every host that did this
would reinvent the same inference, badly and differently.

## The contract

> Neo reports facts about its process. The host decides how and when those
> facts become conversation.

That line governs both halves. `events.py` and `OrchestratorMessage` carry
phase names, counts, and findings — never host-specific wording, never an
instruction to say something. `NeoEngine` does not know which host it is
talking to, and adding a second host must not require touching it.

## Stream split

`--json` writes **two** streams:

- **stdout** — exactly one JSON document. Unchanged shape plus an
  `orchestrator` key.
- **stderr** — JSONL events, one object per line, flushed per event.

Events deliberately do **not** interleave into stdout. The `--json` help text
used to promise "JSONL events and final JSON" on one stream; honoring that
literally would have broken every existing `neo --json | jq` consumer. The help
text was corrected to describe the split.

`--json` implies `--quiet`, so Neo's own `[Neo] …` progress notices are
suppressed and stderr is essentially pure JSONL. It is not *guaranteed* pure —
Python logging warnings and CAR version notices can still land there — so hosts
still parse lines beginning with `{` and ignore the rest.

`--quiet` had been declared in the parser and never read, so those notices
could not be turned off at all. They are now routed through `neo.progress`
(`note()` / `set_quiet()`) rather than bare `print(..., file=sys.stderr)` calls
scattered across `context_gatherer`. The suppression flag is process-global
rather than a threaded parameter: it is a display setting fixed once from argv
and read by leaf functions five layers below the CLI, and passing it down would
put a presentation argument into the signature of every scoring helper it
crosses. Index-command *errors* deliberately stay unsuppressed — `--quiet`
silences progress, not failures.

## Phase records and ordering

`_phase_records` accumulates `{name, status, message}` per run, reset at the
top of `process()`, and is replayed into `OrchestratorMessage.phase_summary`.
Four invariants are load-bearing and pinned by tests:

1. **Findings precede their phase's close.** A host replaying the stream must
   learn *what was found* before being told the phase finished. The reasoning
   phase therefore holds its summary in a local and calls `_end_phase` after
   emitting `hypothesis_formed` / `risk_found`, rather than closing inside each
   branch.
2. **No event claims a closed phase.** `MEMORY_FOUND` is emitted from
   `_capture_retrieval_context`, the funnel every fact-retrieval path passes
   through — but that runs while the *reasoning* phase is open. `_current_phase()`
   reports the open record instead of a hardcoded name. Caught in a live run.
3. **Every close has an open.** The budget-skip branch originally emitted
   `phase_completed` for `static_checks` with no `phase_started`, leaving a host
   tracking a close for something it never saw open. It now opens the phase and
   closes it `skipped`. `_end_phase`'s synthesize-on-missing fallback stays as a
   backstop but now logs a WARNING, so the next caller bug is findable rather
   than silently absorbed.
4. **Every run has a terminator.** `process()` emits `FAILED` from its except
   branch and marks any still-`running` record `failed`. Emitting `STARTED` and
   then going silent was worse than never emitting: a host could not distinguish
   a crash from a hang.

The first phase is named `context`, **not** `retrieval` — it covers file
gathering only. The original name overstated its span, which is what forced
`MEMORY_FOUND` to work around it. A phase whose name overstates what it covers
makes every emitter inside it either lie or compensate.

`_end_phase` searches records in reverse for the newest open one because a
phase can legitimately run twice: a failed panel closes `reasoning` with
`status="fallback"` and the fast path opens a second `reasoning` record.

## Sinks never break the run

`safe_emit` swallows sink exceptions. A closed pipe, a full disk, or a
third-party sink bug must degrade to "no progress reporting", never to a failed
reasoning run — the observer must not take down the thing it observes. The
default sink is `NullSink`, so an unobserved run costs exactly what it did
before events existed.

## Voice: staged, and authored in the deck

Neo speaks in the first person. Narrating himself in the third person ("Neo
reasoned over the request and proposes 1 change(s)") reads like a status board,
not like the character the beat deck already defines — and the summary is the
line a host is told to lead with, so it sets the register for the whole
response.

Two rules make that more than a find-and-replace:

**The register follows memory level.** `_voice_stage()` reads
`_memory_level_to_stage()` and selects an opener, a hedge, and a terseness flag
from `orchestrator_voice.stages`. The same facts render as
`Don't know this code. I'd change 1 thing(s) in src/parser.py, maybe.
Confidence 0.88.` at stage 1 and `src/parser.py. 1 change(s). 0.88.` at stage 5.
A single fixed register would have been a regression dressed as a feature: it
throws away a signal the system already computes, and the hedge at stage 1 is
genuinely informative — it tells the user Neo has no history here.

**No prose lives in `engine.py`.** Every user-facing string comes from
`orchestrator_voice.lines` via `_voice(key, **fmt)`, so the character can be
retuned by editing a YAML file. `test_engine_holds_no_prose_of_its_own` pins
that: a string literal in the engine is a personality change hidden inside a
code diff nobody reviews as one. A missing key or a bad placeholder degrades to
`""` rather than raising — a formatting slip in a personality file must never
take down a reasoning run — and `_cap_cautions` drops blanks so a failed
template can't become an empty bullet a host dutifully relays.

**Deliberate asymmetry: cautions do not vary by stage.** A host is instructed
never to drop a caution, so a warning must read the same whether Neo is a
Sleeper or The One. Voice is not licence to soften a fact. Same for phase and
lifecycle messages — they stay in Neo's voice but at one register, because five
variants of "Reading the code." would be upkeep with no reader-visible payoff.

An LM-voiced summary was considered and rejected for now: it would cost a call
and latency on every run and could fabricate or drop a claim — exactly the
"narrator that lies" failure the first review caught. The system prompts
(`_get_planning_system_prompt` and friends) already inject Neo's personality
into the model, so the plan descriptions and rationales the LM returns are
voiced by inference; only the derived scaffolding is templated.

## Personality is gated, not decorative

The beat deck already keyed beats on `category` + `trigger_contexts`, so beats
were never random flavor. What was missing was a decision about *audience*.
Each beat now declares:

- `surface` — `orchestrator` lets a host relay it; anything else (or absent)
  keeps it terminal-only.
- `importance` — advisory ranking for hosts that surface at most one line.
- `requires_finding` — when true, the beat is withheld unless the run actually
  produced a finding (a recalled fact, a simulation issue, a failed check).
  Beats that *claim* insight must earn it; beats that merely describe the
  situation need not.
- `orchestrator_line` — the host-facing wording.

`_orchestrator_beat` returns `""` — silence — whenever the deck did not match,
the beat is internal, or a finding was required and none exists. There is no
fallback line. Unattached flavor text is what makes a tool read as theater.

One line per beat rather than one per memory stage: the situation carries the
meaning, and five near-identical variants would be upkeep with no
reader-visible payoff.

**One beat per run.** `_run_beat` selects once and caches; both `_generate_notes`
and `_orchestrator_beat` read it. Selecting independently per surface let them
pick different beats from the same run — and `--json` carries both `notes` and
`orchestrator.personality`, so one character would speak with two voices in a
single response.

**Lines must not outrun their trigger.** Three beats originally asserted more
than their trigger established: `error_cascade` claimed "multiple failures, one
root" from a single traceback, `critical_bug` claimed "something broke that used
to work" from the keyword *fix*, and `pattern_match` claimed "I know where it
breaks" when the `requires_finding` gate only proves *something* was recalled —
possibly a style rule. All three were reworded to what the trigger supports.

**A configured beat that can never fire is dead text.** `unfamiliar_codebase`
declared `first_time_codebase` / `unfamiliar_domain` / `no_memory_match`, none
of which `_select_beat` ever produced — so its line was unreachable and the
"every surfaceable beat declares its wording" test gave false assurance.
`_select_beat` now derives `memory_hit` / `familiar_pattern` /
`no_memory_match` from actual recall state, which also makes `pattern_match`
reachable by memory rather than only by confidence.
`test_every_declared_beat_can_actually_fire` pins the reachability property
itself, not just the presence of a string.

**No `cooldown` / `max_uses`.** Each `neo` invocation is a fresh process that
selects at most one beat, so an in-process cooldown is a no-op and a cross-run
one needs session state that does not exist. Better absent than faked.

## Derived, never generated

`_build_orchestrator_message` makes no LM call and performs no new analysis. It
is pure derivation from state already computed — plan, suggestions, simulation
issues, static-check statuses, confidence, reasoning mode. Every claim must be
defensible from the rest of the `NeoOutput`, because a host that repeats an
unearned summary is worse than one that says nothing.

**VERIFY mode gets its own sentence.** That path makes no LM call, and its
`code_suggestions` are the *caller's* `proposed_changes` echoed back for
checking. The generic wording ("Neo reasoned over the request and proposes N
change(s)") credited Neo with work it did not do and would have been relayed
verbatim by a host told to lead with the summary. VERIFY also suppresses the
low-confidence caution: its confidence is a pass/fail verdict, not
self-assessed certainty, so "verify before acting" is circular there.

`cautions` is the field that matters most: low confidence, failed checks,
simulation issues, absent static analysis, open questions. The plugin contract
requires hosts to surface all of them. A confident-sounding recommendation must
not bury the reasons to doubt it. Two consequences:

- "No static analysis ran" is split into **no tools installed** versus
  **skipped for time budget**. Same absence, different remedies; the phase
  record already knows which happened.
- The list is **capped** (`MAX_CAUTIONS`, `MAX_CAUTION_CHARS`). A host is told
  never to drop a caution, so the list has to stay relayable — and failed-check
  summaries are unbounded LM prose. Anything dropped is reported as a count,
  never silently discarded, because "there are more problems" is itself
  something a host must not bury.

`phase_summary` copies each record, not just the list. These dicts are live
engine state and the object is handed to foreign consumers.

## One contract, two adapters

The contract is host-neutral by construction: `events.py` and
`OrchestratorMessage` know nothing about who is consuming them, and the
per-host wording lives entirely in the adapters.

| Surface | Location | Shape |
|---|---|---|
| Claude Code | `.claude-plugin/` | agent + 6 slash commands |
| Codex CLI | `plugins/neo/` | plugin manifest + 6 skills |
| CAR / A2A | `neo.a2ui`, `neo serve` | A2A status + artifact events |

(`.agents/skills/` holds *release-maintenance* skills, not Neo capabilities.
Looking there for the Codex plugin and concluding it is missing is an easy
wrong turn — the manifest is at `plugins/neo/.codex-plugin/plugin.json`,
registered by `.agents/plugins/marketplace.json`.)

**Adapters drift, and drift is invisible.** The Codex skills instructed the
model to parse "four structured sections: `CONFIDENCE`, `PLAN`, `SIMULATIONS`,
`CODE SUGGESTIONS`" out of terminal-formatted prose — the exact anti-pattern
the Claude side had just stopped doing. One adapter got the contract change;
the other did not, and nothing failed. `test_host_adapter_parity.py` now pins
the invariant: same six capabilities on both surfaces, both invoking `--json`,
both teaching `orchestrator.summary`/`cautions`, both documenting the error
shape and attribution, and every documented phase/event name checked against
`events.py` rather than against prose.

The same test pins manifest versions against `pyproject.toml`. `prepare-release`
documents bumping both plugin manifests, but a documented step with no
enforcement is a step that gets skipped: the package, the Claude manifest, and
the Codex manifest had reached 0.41.0 / 0.37.0 / 0.19.0.

**Role differs even though protocol does not.** Under Claude Code, Neo is
usually a delegated subagent — the user can see the boundary. Under Codex, Neo
is a step inside the same coding loop: Codex calls Neo, reads the result, then
edits files and runs tests in the same breath. Nothing marks where Neo's
reasoning ends and Codex's begins, so the Codex skills carry stronger
attribution wording ("Neo found …") and an explicit instruction that Neo's
result is an input, not the deliverable.

## Sink failures are findable

`safe_emit` logs the **first** failure at WARNING and the rest at DEBUG. Logging
every failure at DEBUG (the original) meant a permanently broken sink was
invisible at normal log levels — the swallow policy is right, but silence about
it is not.

Note what `safe_emit` does *not* cover: `NeoEvent(...)` is constructed in
`_emit` before the guard is reached, so a formatting bug in emission code still
raises. Message construction is kept to attribute reads on values already
proven non-None at that point.

## Verification

- Full suite: 1568 passed, 8 skipped. `ruff check src/ tests/` clean.
- 37 tests in `test_orchestrator_events.py` covering sink behavior, event
  ordering, the four phase invariants, summary derivation, beat gating and
  reachability, and the VERIFY / budget-skip / panel-fallback / failure
  branches. 4 more in `test_repair_loop.py`.
- Live end-to-end runs against a real LLM confirming the stdout/stderr split,
  single-parse stdout, and event ordering.

## Incidental fixes

- `repair_loop.py:149` imported `from structured_parser import …` —
  unqualified, and therefore unresolvable inside the installed package. Every
  repair attempt died on `ModuleNotFoundError` **and the attempt loop's
  `except` swallowed it**, so the feature reported a generic "failed to repair"
  and looked merely ineffective rather than broken. Dead since the initial
  public release because nothing tested it; `test_repair_loop.py` now does.
- The static-check skip logged `elapsed/time_budget*100` unguarded. A zero
  budget is not reachable through `_get_time_budget` today, but a *log line*
  must never be what crashes a run.

## Known limits

- A `Bash` tool call does not stream, so JSONL buys accurate **post-hoc**
  narration, not a live progress bar. Nothing short of a hook or an MCP server
  changes that. `JsonlSink` still flushes per event, for hosts that can stream.
- `NeoEngine` is single-request-at-a-time, and this is now **enforced**, not
  merely documented: `process()` takes a non-blocking lock and raises
  `EngineBusyError` on overlap. `_phase_records` / `_findings` /
  `_selected_beat` are per-request instance state, following the same pattern as
  `self.context`, `last_applied_actions`, and `current_learning_episode`, so
  overlapping runs would cross-attribute suggestions and learning episodes
  between unrelated requests — silently. `neo.car_host` caches engines per
  working directory and reuses them, relying on CAR's drain task being
  single-threaded, which its own comment calls "an implementation detail"; that
  detail is upstream and not ours to guarantee, so the host now answers an
  overlap with a retryable `EngineBusy` response rather than a stack trace.
  Non-blocking on purpose: queueing would hide a caller's design bug behind a
  latency mystery. The `StructuredOverseer` watchdog is *not* a hazard — it only
  reads `action_log`.
- `_timeout_response` builds a bare `NeoOutput` and so carries an empty
  orchestrator envelope. It is currently unreachable (neither it nor
  `_check_timeout` has a caller); wire the message in if it is ever revived.
- `JsonlSink` is not thread-safe; interleaved writes could split a line. Only
  the main thread emits today.
- stderr is *essentially* pure JSONL under `--json`, not guaranteed pure:
  Python logging warnings and CAR version notices still land there. Those are
  diagnostics, not progress, and `--quiet` correctly leaves them alone.
- The staged voice is templated, not generated. Within a stage the phrasing is
  fixed; only the stage varies. An LM-voiced summary would be more varied at
  the cost of a call per run and a fabrication risk.
