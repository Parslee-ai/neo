# How many files should neo deliver?

**Date:** 2026-08-30 · **Code:** `68223a7` · **Harness:** `tools/rank_mine_eval.py --max-files N`

The delivery cap (`--max-files`, default 30) had never been measured. This is the
sweep, on two repositories three orders of magnitude apart in candidate count.

## First, the knob did not work

The first sweep produced an incoherent curve — R@10 *higher* at cap 10 than at
the default 30, which cannot happen if the cap is a cap. It was not:
`calculate_adaptive_limit` returned its three broad-prompt buckets verbatim,
ignoring the caller's ceiling.

| prompt | `--max-files=5` | `=10` | `=30` |
|---|---|---|---|
| "review this" | **15** | **15** | 15 |
| "review this codebase" | **20** | **20** | 20 |
| "refactor memory delete synthesis" | **25** | **25** | 25 |
| "review ProjectIndex.retrieve() in src/…" | 5 | 10 | 30 |

Only the *specific* bucket honoured the ceiling, and it does so by construction
(it returns `default_max`) — which is why nothing caught it. Fixed in `68223a7`
(`min(floor, default_max)`), with no behaviour change at the default. Sweep
points taken before that fix were discarded rather than reported: a curve
measured across two code versions is not a curve.

## The curves

**neo** — 99 candidate files, 50 cases:

| cap | R@10 | MRR | H@10 | mean context bytes |
|---|---|---|---|---|
| 5 | 0.620 | 0.739 | 0.880 | 142,440 |
| **10** | **0.720** | 0.762 | **0.980** | **131,521** |
| 15 | 0.720 | 0.761 | 0.980 | — |
| 20 | 0.720 | 0.761 | 0.980 | — |
| 30 *(default)* | 0.720 | 0.763 | 0.980 | 197,382 |
| 50 | 0.720 | 0.752 | 0.980 | 253,296 |

**m365dotnet** — 2,882 candidate files, 50 cases:

| cap | R@10 | MRR | H@10 | mean context bytes |
|---|---|---|---|---|
| 5 | 0.607 | 0.528 | 0.760 | — |
| **10** | **0.678** | 0.532 | **0.800** | **201,796** |
| 20 | 0.678 | 0.534 | 0.800 | — |
| 30 *(default)* | 0.668 | 0.536 | 0.800 | 231,202 |
| 50 | 0.656 | 0.545 | 0.780 | — |

## What the numbers say

**The knee is ~10 files, and the shape replicates on a repo 29× larger.** R@10
and hit-rate@10 both peak at cap 10 on both repos and improve nowhere beyond it.

**Tripling the cap to the shipped default buys nothing measurable.** Pooled
paired over 100 cases, cap 30 vs cap 10: **6 better, 2 worse, 92 tied,
sign p = 0.289.** Tripling the delivered files changes the outcome in 8 cases
out of 100, in both directions.

**It costs real budget**: +50% context bytes on neo (131 KB → 197 KB), +15% on
m365dotnet (202 KB → 231 KB), on every invocation.

**Below 10 there is genuine loss.** neo 5 → 10 is 6 better / 0 worse,
**p = 0.031** (R@10 0.620 → 0.720, H@10 0.880 → 0.980). m365dotnet moves the
same direction (R@10 0.607 → 0.678) without reaching significance at n=50.

**Byte cost is not monotonic in file count.** cap 5 spends *more* bytes than
cap 10 on neo (142 KB vs 131 KB) while retrieving worse, because `--max-bytes`
(300 KB) is apportioned across whatever is admitted — five files each take a
large share. So 10 is simultaneously the saturation point of the quality curve
and the minimum of the cost curve.

## The default was NOT changed, and that is deliberate

The evidence above is retrieval-only. R@k asks whether the needed file is in the
delivered list; it cannot see whether the other 20 files help the model *reason*
about the task. `docs/effectiveness-evidence-2026-08-30.md` §2b shows context
presence is causally necessary for a usable patch — so cutting delivery could
degrade answers in a way this instrument is blind to.

Changing a shipped default on retrieval-only evidence would be measuring one
thing and acting on another. What is established:

- 10 files suffice **for retrieval** on both repos measured;
- the default of 30 costs 15–50% more context bytes for no measurable retrieval
  gain;
- 5 is too few.

Validating a change to 10 needs an answer-quality arm — the planted-bug design
in the effectiveness doc, run at cap 10 vs cap 30 — which has not been run.

## Limits

- Two repos, 50 cases each. `aieweb` was not swept.
- Every caveat in the harness docstring applies: absolutes are upper bounds
  (the corpus has already seen the answer), and the mined population is filtered
  upward toward small, well-described changes.
- "Not significantly different" with 92% ties is weak evidence *for* equivalence,
  not proof of it. The defensible claim is that no gain was measured, not that
  none exists.
- Byte figures are means over 7–11 prompts, not the full case set.
