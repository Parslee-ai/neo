# CAR-first gate re-evaluation — car-runtime 0.56.1 (2026-10-07)

**Verdict: parity NOT met. `inference_mode` stays `static`.**

neo defaults to `inference_mode="static"` until a CAR release is shown not to
lose quality through its router. The previous evaluation, against car-runtime
0.23.0, found a routing defect: `task=code` cost-biased onto mini models
(gpt-5.4-mini, gpt-4.1-mini) and lost ~70% of head-to-heads. That eval was run
ad hoc and never committed. `tools/ab_car_router.py` is the re-runnable
replacement. Raw results are in `car-router-eval-2026-10-07.json`: routing,
usage, schema and judge reasons for each prompt. That run predates the harness
saving answer text.

## Setup

- **Arms are what production builds.** A = `OpenAIAdapter(gpt-6.1-sol)` with
  the engine's recorded `max_tokens` / `temperature` / `reasoning_effort`.
  B = `CarAdapter(model=None)`, which is exactly what `resolve_adapter` builds
  for `auto` (`task=code`, `prefer_quality`). B is called directly, never
  through `AutoAdapter`, so a CAR failure cannot quietly fall back to A.
- **Prompts are real.** Six requests about this repository were captured with
  `neo --dry-run --json`: a bugfix, an explanation, a refactor, a feature, a
  diagnosis and an optimization. Each is ~130K characters (~50K tokens)
  including repository context.
- **Daemon:** CarHost 0.56.1 / car-runtime 0.56.1.
- **Judge:** gpt-6.1-sol, blind and pairwise, with order randomized
  (seed 1234). It is judging its own arm, so any self-preference favors A.

## Results

| | CAR router | static gpt-6.1-sol |
|---|---|---|
| Model served | `anthropic/claude-opus-4-8:latest` on 6/6 | gpt-6.1-sol |
| Errors | 0 | 0 |
| neo schema parses (all 3 blocks) | 6/6 | 6/6 |
| Mean latency | 43 s | 28 s |
| Head-to-head wins | **0** | **6** (0 ties) |
| Mean judge score | 3.0 | 9.0 |

**The routing defect is fixed.** No prompt went to a mini or small model, and
all six went to a frontier model. Schema compliance is equal.

**The quality loss has a different cause from last time.** The judge gave the
same reason on all six prompts:

- The served model wrote concrete diffs against code it had only seen
  truncated, and invented APIs to fill the gap.
- The static model recognized the truncation, then proposed inspecting the
  code and a conditional fix.

This is the failure mode neo's truncation markers exist to prevent. Opus
answers past the markers and gpt-6.1-sol respects them.

**Spot-checked against source:** 4 of the 5 inventions the judge named
genuinely do not exist (`_adapt_for`, `_client_create`, `_recycle_via_exec`,
a `candidates=` parameter on `FactStore.retrieve_relevant`). One judge claim
was wrong: `_apply_flag` exists in `adapters.py`. So the judge is imperfect,
but the finding does not rest on it alone.

## Limits

- n = 6, from one repository. A 6–0 sweep is a strong signal, not a rate.
- The judge is self-preferring. That makes a CAR *loss* partly confounded,
  though the verified inventions are objective.
- CAR drops `reasoning_effort` and `temperature`. This is production behavior,
  so it is included deliberately, but it means the comparison is
  "CAR's choice under CAR's defaults", not "Opus vs GPT at equal settings".
- The judge scores correctness, not verifiable-suggestion yield. Every static
  answer scored exactly 9 for planning an inspection instead of patching, and a
  suggestion with no diff can never be git-verified (one stated reason the
  learning loop starves). The CAR arm was penalized for attempting diffs and
  getting them wrong; the static arm was never tested for being useful. Invented
  APIs are still objectively worse than a plan, so the verdict holds.
- Captured prompts are gitignored and depend on the repository's state, so a
  re-run measures today's six prompts, not these.
- The result describes the router's choice today. The router reroutes as its
  catalog changes, so re-run `python tools/ab_car_router.py` on each CAR
  release. `--probe` checks routing alone and costs no judge calls.

## Next gate

Flip to `auto` when a run shows CAR does not lose a majority of decided
head-to-heads, has equal schema compliance, and routes no prompt to a small
model. Plausible ways to get there are a router that weighs instruction
adherence for `task=code`, or `intent_hint` steering. Either is a CAR-side or
intent-design change, not a neo default flip.
