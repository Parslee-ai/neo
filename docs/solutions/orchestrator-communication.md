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
| Beat surface metadata | `config/beats/neo_matrix.yaml`, `engine._orchestrator_beat` | `test_orchestrator_events.py` |
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

stderr is a **mixed** stream — it also carries `[Neo] …` progress lines from
`context_gatherer`, library warnings, and CAR version notices. Hosts parse
lines beginning with `{` and ignore the rest. (`--quiet` exists in `cli.py` but
is never read; suppressing those lines is a separate fix.)

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
- `NeoEngine` is single-request-at-a-time. `_phase_records` / `_findings` /
  `_selected_beat` are per-request instance state, following the same pattern as
  `self.context`, `last_applied_actions`, and `current_learning_episode` — so
  this adds no new class of bug, but hosts are exactly the population likely to
  share one engine across a thread pool. The `StructuredOverseer` watchdog is
  *not* a hazard: it only reads `action_log`.
- `_timeout_response` builds a bare `NeoOutput` and so carries an empty
  orchestrator envelope. It is currently unreachable (neither it nor
  `_check_timeout` has a caller); wire the message in if it is ever revived.
- `--quiet` is defined but never read, so `[Neo]` progress lines cannot be
  suppressed under `--json`.
- `JsonlSink` is not thread-safe; interleaved writes could split a line. Only
  the main thread emits today.
