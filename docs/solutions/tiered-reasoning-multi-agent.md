# Solution: Tiered Reasoning — CAR-Gated Multi-Agent Deliberation with Memory Amortization

**Date**: 2026-07-04
**Status**: Implemented (L1–L3 reasoning tier + gate + memory provenance; L4 execution deferred)
**Type**: design + implementation

## Implementation summary

| Piece | Module | Tests |
|---|---|---|
| Novelty→mode gate | `reasoning_mode.py` (`decide_mode`) | `test_reasoning_mode.py` |
| Panel orchestrator (plan-vote → code → adversarial critique → repair) | `multi_agent.py` (`MultiAgentReasoner`) | `test_multi_agent.py` |
| Model-diversity planning (`route_model` + `exclude_models`) | `panel.py` | `test_panel.py` |
| Engine wiring (gate + branch + provenance metadata) | `engine.py` (`_decide_reasoning_mode`, `_deliberate`) | `test_engine_reasoning.py` |
| Config + CLI | `config.py` (`reasoning_mode`), `cli.py` (`--deep`/`--fast`) | — |
| A/B quality harness | `tools/ab_reasoning.py` | (see A/B result below) |

The gate stays **auto** by default; multi-agent fires only on novelty **and**
CAR-reachable **and** ≥2 distinct capable models — otherwise the fast single
call runs (a failed panel also falls back). Output `metadata` always carries
`reasoning_mode` + `reasoning_reason`, and panel runs add `provenance` +
`panel` (models_used, consensus, rounds).

## A/B result — does the panel actually produce better quality?

`tools/ab_reasoning.py`: 8 self-contained coding tasks, each solved two ways —
one-shot combined call (A) vs the panel (B) — then scored by a blind LLM judge
(order randomized per task). gpt-5.5, `effort=low`, `k_plans=2`,
`max_repair_rounds=1`.

| Metric | Value |
|---|---|
| Avg judge score — **panel** | **9.25 / 10** |
| Avg judge score — single | 8.12 / 10 |
| **Quality delta** | **+1.12** |
| Wins: panel / single / tie | 3 / 1 / 4 |
| Decisive win rate (panel) | **3 of 4 (0.75)** |
| Latency: panel vs single | 44.7s vs 8.4s (~5.3×) |

The panel's wins were on the **edge-case-heavy** tasks — dedup-a-list-of-dicts
(8 vs 4: the critic caught unhashable dicts), semver parsing (10 vs 7), debounce
(9 vs 6) — exactly where an adversarial second pass earns its keep. Its one loss
was the trivial "merge two sorted lists," where a one-shot answer is already
optimal and the extra machinery only added noise. **This reinforces the gate**:
the panel is worth ~5× the latency *on hard/novel work* and wasteful on easy
work — which is precisely what the novelty gate reserves it for.

**Honest scope of this measurement:** all four roles used the *same* model
(gpt-5.5) — so this isolates the value of the **orchestration structure** (plan
vote → adversarial critique → repair) with **zero model diversity**. Real
deployment adds distinct models per role (§4), which should widen the gap
further. Raw results: `/tmp/ab_low.json`.

### Model diversity: mechanism validated, quality lift not yet measurable

The **routing mechanism works** against real CAR: `plan_role_models` +
`exclude_models` routed the critic to a *different* model than the coder
(`coder → parslee/advisor`, `critic → parslee/reasoning`; `critic ≠ coder`
confirmed). But the **quality lift from diversity could not be measured on this
host**, for two concrete reasons — neither of which is an unreleased CAR
feature:

1. **`capable_model_count` overcounts.** It reported 3 models, but
   `parslee/advisor` and `parslee/reasoning` are **both Azure gpt-5.5 underneath**
   (Parslee assistant personas). So the "diverse pool" is really gpt-5.5 ×2 plus
   one local qwen — the `≥2 distinct models` gate can be *fooled* by same-backend
   personas. **Refinement needed:** dedupe the diversity count by backing model,
   not catalog ID (neo can't currently tell that `parslee/*` share a backend).
2. **No second *fast+capable+distinct* model exists here.** The only genuinely
   different family is a local MLX model, and it **timed out (180s) on every
   critique** — a 4B local model answers a one-word probe in ~8s but can't do a
   real critique (full solution + token budget) in time. And there's no
   Anthropic/Google credential wired up.

So a real frontier-vs-frontier diversity A/B needs a **second fast capable model
credential** (a Parslee/infra decision), or Parslee's `/inference/responses`
backend serving a genuinely different model family (its model-agnostic future).
The harness supports it now (`tools/ab_reasoning.py --critic-model ...`); it just
needs a viable second model to point at.

## Problem

Neo is described as "MapCoder/CodeSim-style multi-agent reasoning," but the
implementation collapses all three roles (plan → simulate → code) into a
**single LLM call** (`engine.py:_process_combined`, the comment: *"Single LLM
call for all 3 phases … 59% faster than the old 3-call approach (22s vs 55s)"*).
There was a real 3-call path once; it was deliberately flattened for speed.

Consequences of the flattening:

- **No inter-agent loop.** `engine.py` calls itself a *"single-shot engine."*
  Real MapCoder/CodeSim iterate (plan → code → verify → repair). Neo does not.
  Its `repair_loop.py` only repairs *malformed JSON*, not failing code.
- **"Simulation" is self-narration.** `simulation_traces` are the model
  describing how its plan *would* run — never executed. The engine itself
  distrusts this (`"LM self-reported confidence alone is self-validation"`) and
  bolts on static checks as the only *objective* signal before early-exit.
- **One model plays every role**, so the three "agents" share weights and
  therefore share blind spots — the critic can't catch what the planner missed.

The goal is a *real* multi-agent tier that adds genuine, uncorrelated scrutiny —
**without** paying its cost on every query, and **without** requiring CAR (neo
must keep working standalone).

## Core idea: two tiers, one memory

```
                      novelty signal (from memory)
                                 │
              familiar/confident │ novel / low-confidence / --deep
                    ┌────────────┴─────────────┐
                    ▼                           ▼
            FAST TIER (default)          DELIBERATION TIER (CAR only)
      single call + recalled memory   multi-agent panel: distinct models,
        (today's optimized path)       plan-vote → code → adversarial critique,
                    │                   repair loop, consensus confidence
                    │                           │
                    │                validated outcome → memory (provenance-tagged)
                    └───────────────────────────┘
                        both write/read the SAME fact store
```

**Deliberate once, recall forever.** The expensive tier runs where memory is
empty (novel issues), and its *job* is to manufacture the high-quality memory
that lets the fast tier be trusted on the *next* similar issue. The tiers feed
each other rather than compete.

## Design

### 1. The gate already exists — promote it from a dimmer to a switch

`reasoning_effort.effort_from_memory()` already maps a `MemorySignal`
(`pattern_count`, `avg_confidence`) to an effort level:

| Memory signal | today (effort) | proposed (mode) |
|---|---|---|
| ≥3 patterns, conf ≥ 0.8 | `low` | **fast recall** |
| some patterns, conf 0.5–0.8 | `medium` | fast |
| no patterns OR conf < 0.5 | `high` | **deliberation** |
| no patterns AND hard | `xhigh` | **deliberation, full** |

"No patterns" is neo's existing definition of *novel*. The change is to let the
top two rungs select a **mode** (deliberate) rather than only a bigger token
budget on the same single call. The bottom rungs are unchanged — already primed
by recalled patterns.

Three triggers, not one:

- **A priori** — gate on the novelty signal before running (main path).
- **Runtime escalation** — `reasoning_effort.escalate()` already exists
  (*"we know cheap-thinking failed, spend more"*). If the fast path trips a
  check (static analysis, low simulation consensus), escalate *into* the
  deliberation tier. This covers novelty the a-priori gate misclassifies.
- **Explicit override** — `--deep` always deliberates (and refreshes memory);
  `--fast` always takes the cheap path.

### 2. The deliberation tier is CAR-only — on the merits, not just packaging

Neo installs and runs **without** CAR (static provider path via
`resolve_adapter`). The multi-agent tier is offered **only when CAR is
reachable**, reusing the exact check `resolve_adapter` already makes
(`car_inference.is_available() and a2ui.is_daemon_reachable()`).

This is not a licensing quirk — it is *where the technique earns its cost*. The
value of a multi-agent panel is **model diversity**: different models with
different blind spots disagreeing and ranking each other. Without CAR's router
there is one configured provider, so "multi-agent" degrades to **one model
playing every role** — correlated errors, 5–15× the latency to re-confirm the
model's own blind spots. That is the self-validation weakness, more expensive.

Graceful degradation:

- **No CAR** → the novelty signal still fires; it just resolves to today's
  behavior (`high`/`xhigh` effort on the single call). Nothing regresses.
- **`--deep` without CAR** → warn and degrade to a max-effort single pass, don't
  error.
- **Shared memory across tiers.** The fast path doesn't care how a pattern was
  born. So **CAR sessions teach memories that non-CAR sessions recall**: same
  `project_id`, same fact store — deliberated patterns deposited during a
  CAR-enabled session are recalled later on a machine with no daemon.

### 3. What the tier actually *is* — reasoning, not execution

CAR already ships the orchestration; neo uses none of it for reasoning (its only
`agents_*` use today is registering the memory-observer sweep). Every CAR
`run_*` primitive takes an `agent_fn` callback — **you supply the model call,
CAR runs the pattern**:

| Need | CAR primitive |
|---|---|
| Generate *k* plans, rank/pick best | `run_vote` / `run_tournament` |
| Plan → code → verify stages | `run_pipeline` |
| Debug / repair feedback loop | `run_supervisor(max_rounds)` / `run_task_loop` |
| Per-role model choice | adaptive router via per-agent `intent` |

**Execution is NOT the core** — a correction worth recording. A container gives
you a filesystem + runtime, not an *environment*. Most real code needs
databases with seed data, external/internal APIs, auth, secrets, and network —
none of which a sandbox conjures, and CAR's sandbox is network-gated on purpose
(car#480). Neo, being read-only and credential-less, is the *least* equipped
caller to stand up a live environment.

MapCoder/CodeSim's execution loop works because their domain is **self-contained
algorithmic problems** (HumanEval/MBPP/CodeContests) with standalone tests. Neo
lifted the framing onto real repos, where that assumption collapses. So
execution-grounded verification is an **opportunistic add-on** that only fires
when neo detects a genuinely self-contained unit (a pure function, a generated
minimal repro) — worth doing, but a corner case, not the headline. The headline
is *diverse, adversarial reasoning*: plan-vote, cross-model critique, and a
model-judged repair loop, all of which run without any environment.

### 4. Ensuring distinct models per agent (the crux)

CAR's router optimizes **per-request fitness** ("best capable model for *this*
intent") and has no cross-request "make these N distinct" awareness — and
`run_vote` runs the same `agent_fn` for everyone. So identical intents →
**same model N times**, the exact failure to avoid. Diversity must be
engineered by the caller; CAR provides the primitive built for it.

**Division of labor:**

- **neo owns the diversity policy** — it names *roles*, their *capability
  intents*, and the *distinctness constraint*. It never hardcodes model IDs; it
  says "reasoning + quality," "code," "not that one." Catalog-agnostic, so new
  models auto-participate.
- **CAR owns capability-honest selection** — `task=code` → code-capable model,
  `require=[Reasoning]` → reasoning-capable (guaranteed by the router's
  Quality-workload / capability filter, car#469), honors `exclude_models`, and
  reports its work.

**The primitives** (both on `CarRuntime.route_model`, which is *decision-only —
no inference*, so it's cheap to ask up front):

- `exclude_models` — IntentHint field, documented as *"any capable model that is
  NOT this one — for adversarial-reviewer separation (car#358)."* Purpose-built.
- `route_model(...)` returns `candidates`: the full ranking of scored models
  (`{model_id, reliability, score, selected, in_band}`), so neo can see the
  whole capable pool and *what forcing a different model costs*.

**Flow** — a cheap "routing plan" pass first, then execute (parallelizable):

```
route(planner: task=reasoning, prefer_quality)          -> model A (+candidates)
route(coder:   task=code,      prefer_quality)          -> model B
route(critic:  task=reasoning, prefer_quality,
               exclude_models=[B])                       -> model C  (≠ coder)
route(judge:   task=classify,  prefer_fast)             -> model D  (small/cheap)
```

neo reads model IDs only transiently, to thread the next role's
`exclude_models` — it stays catalog-agnostic.

**Diversity is capped by the pool, and that becomes the real gate.** If only one
capable frontier model is wired up (e.g. only `parslee/gpt-5.5`, no
Anthropic/Google creds), `exclude_models` on the critic leaves nothing capable.
`candidates` reveals this *before* committing. So the deliberation trigger is
not merely *CAR reachable* but **CAR reachable AND ≥2 genuinely capable,
distinct models available**. Below that, neo degrades (single high-effort call,
or a same-model adversarial self-check) — it only pays for a "diverse panel"
when the diversity actually exists.

### 5. Memory amortization — feed the existing REVIEW→PATTERN pipeline

The "remember it so you don't redo it" loop is neo's *existing* synthesis
pipeline, just fed by a better source:

```
novel issue → deliberation → validated outcome
   → stored as a fact (provenance: multi-agent-validated)
   → synthesis clusters recurring ones → PATTERN
   → next similar issue: retrieval surfaces it
   → pattern_count≥3, conf≥0.8 → fast path, no agents
```

Deliberated outcomes enter through neo's **existing probation + outcome
mechanism** (`memory.outcomes`, `store.detect_implicit_feedback`) — they are
**not auto-trusted**. Their initial confidence should come from the panel's own
**consensus** (vote/tournament agreement), which is a far better signal than a
single model's self-report; real outcomes (ACCEPTED/MODIFIED) then promote or
demote them.

### 6. Guardrails

- **Novelty's asymmetric risk.** *False-novel* (deliberate when memory would
  have sufficed) = wasted time/money — safe. *False-familiar* (skip when you
  shouldn't) = a confident wrong answer from a mismatched pattern — dangerous.
  Tune the gate to prefer deliberation under ambiguity; require a recalled
  pattern to *actually match* (high similarity **and** same issue-kind/domain,
  not just nearby in vector space); treat "no strong match" as novel. Runtime
  escalation (§1) is the net for residual false-familiar cases.
- **Memory poisoning.** A wrong deliberation stored as high-confidence corrupts
  every future recall. Mitigated by probation + consensus-as-confidence +
  verify-before-promote (static checks / type-check must pass).
- **Ossification.** Patterns rot as code changes. Lean on `recall_decay` +
  probation staleness, and occasionally re-deliberate on a "seen" issue (when
  its recalled confidence has decayed, or at a low sampling rate).
- **One line to hold:** don't build a pure-Python "multi-agent lite" over the
  single configured provider for standalone neo — it re-creates correlated
  errors and adds a second orchestration codepath. Standalone = single-call +
  memory; deliberation = CAR-only. (A single-model *adversarial self-check* —
  generate, then a fresh context prompted to "find the flaw" — is the one
  cheap, no-CAR pattern worth considering, as a separate small feature.)

## Build levels (each shippable, flag-gated behind `reasoning_mode`)

- **L1 — Restore the 3 calls.** `_generate_plan` / `_simulate_plan` /
  `_generate_code_suggestions` still exist (dormant). Sequence them. No CAR.
  Modest gain, ~2.5× latency. Mostly a stepping stone.
- **L2 — Real planning.** `run_vote`/`run_tournament` over *k* candidate plans
  with distinct models (§4). This is MapCoder's core that neo discarded. CAR-only.
- **L3 — Repair loop.** `run_supervisor`/`run_task_loop`: critic critiques,
  coder repairs, iterate to budget; verifier = strong model + static/type checks
  (things that run without infra). CAR-only.
- **L4 — Opportunistic execution.** Only when a self-contained unit is detected:
  run it in `car-sandbox`, feed real results into the L3 loop. Narrow; not the
  headline.

The reusable core is a single **`agent_fn` bridge** (neo adapter ↔ CAR's
`(role, task) → str`, setting per-role intent + exclusions); all `run_*`
patterns compose over it.

## Open questions / to verify before committing

- **Novelty-match reliability** is the make-or-break research risk — the whole
  scheme's safety rests on rarely mistaking "looks similar" for "same fix
  applies." Needs measurement on real recall, not assumed.
- **Does `car-sandbox` run an arbitrary repo snapshot + its test command**, or
  only the `car-reason` Python-snippet verifier? This sets L4 at "wire an
  existing primitive" vs "extend car-sandbox."
- **Pattern granularity.** Recall generalizes only if PATTERNs are keyed on
  *issue-kind*, not exact code — a synthesis-clustering quality question.

## Non-goals / decisions

- The fast single-call + memory path stays the **default**. Deliberation is
  opt-in via novelty/escalation/`--deep`.
- Neo stays **catalog-agnostic**: it expresses capability intent + "not this
  one," never pins model IDs (except transiently, read back from `route_model`).
- Execution is **not** assumed available; verification leans on model consensus
  + static analysis, with sandboxed execution as a detected-when-possible bonus.

## Related

- `docs/solutions/token-budget-enforcement.md` — context assembly the fast tier
  relies on.
- CAR router: `car#469` (Quality-workload capability filter),
  `car#358` (`exclude_models` for adversarial-reviewer separation).
- `reasoning_effort.py` (`MemorySignal`, `effort_from_memory`, `escalate`),
  `engine.py` (`_process_combined` + dormant per-phase methods),
  `memory/outcomes.py`, `store.detect_implicit_feedback` (probation/outcomes).
