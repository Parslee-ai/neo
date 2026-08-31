# How many files should neo deliver?

**Date:** 2026-08-30 (corrected same day) · **Harness:** `tools/rank_mine_eval.py --max-files N`

> ## Correction
>
> The first version of this document concluded **"the knee is ~10 files; the
> default of 30 costs 15–50% more for nothing."** That was **wrong**, and it
> shipped in v0.52.0's changelog and release notes.
>
> The error was the metric. R@10 and MRR only look at the top ten results, so
> they are **structurally blind to ranks 11+** — a correct answer delivered at
> rank 15 cannot move either number, no matter what the cap is. Those metrics
> were flat from cap 10 to cap 50 because they *could not vary*, not because
> extra files carried nothing. Measuring recall@10 to answer a question about
> delivery@30 is the same category of error this repo documents elsewhere:
> measuring one thing and concluding about another.
>
> Rescored over everything the model actually receives, the conclusion inverts:
> **the default of 30 is well chosen, and cutting to 10 would drop 13–21% of the
> answer files neo currently delivers.**

## Where the answer actually ranks

Position of each ground-truth file in the delivered list, cap 50, 50 cases:

| rank band | neo (107 truth files) | m365dotnet (112) |
|---|---|---|
| 1–10 | 70 (65%) | 70 (62%) |
| **11–30** | **17 (15%)** | **13 (11%)** |
| 31+ | 6 (5%) | 1 (0%) |
| never delivered | 14 (13%) | 28 (25%) |

**A cap of 10 would drop 23 of 107 answer files on neo and 14 of 112 on
m365dotnet.** Concretely, on neo: `src/neo/cli.py` at rank 14 and rank 23, and
`src/neo/index/project_index.py` at rank 12 — real answers, silently outside a
top-10 metric's field of view.

## Recall over everything delivered

The metric a delivery question requires: what fraction of the answer files does
the model actually receive?

| cap | neo recall | neo hit-rate | m365 recall | m365 hit-rate | neo bytes |
|---|---|---|---|---|---|
| 5 | 0.523 | 0.88 | 0.571 | 0.76 | 142,440 |
| 10 | 0.654 | 0.98 | 0.652 | 0.80 | 131,521 |
| 20 | 0.776 | 0.98 | 0.714 | 0.84 | — |
| **30** *(default)* | **0.813** | 0.98 | **0.741** | 0.88 | 197,382 |
| 50 | 0.869 | 0.98 | 0.750 | 0.90 | 253,296 |

Monotonic on both repos, with diminishing returns rather than a knee:

- **10 → 30**: recall 0.654 → 0.813 on neo (+24% relative) and 0.652 → 0.741 on
  m365dotnet (+14%), for +50% context bytes. A real trade, not "nothing".
- **30 → 50**: +7% relative recall on neo, +1% on m365dotnet, for a further
  +28% bytes — and top-of-list quality gets slightly *worse* (MRR 0.763 → 0.752
  on neo, R@1 0.403 → 0.377), so the extra files dilute the ranking they extend.

**The shipped default of 30 sits where the returns flatten without the dilution
that shows up by 50.** It was not measured when it was chosen; it is now, and it
holds up.

## What survives from the first analysis

- **`--max-files` was not a cap**, and fixing it was a genuine prerequisite:
  `calculate_adaptive_limit` returned its broad-prompt buckets verbatim, so
  `--max-files 5` on a vague prompt delivered 15. Sweeping the knob below 25 did
  nothing for any prompt that was not highly specific. Fixed and released in
  v0.52.0; that part of the finding is unaffected.
- **Byte cost is not monotonic in file count** — cap 5 spends *more* than cap 10
  (142 KB vs 131 KB) while delivering less, because `--max-bytes` is apportioned
  across whatever is admitted and five files each take a large share.
- **Below 10 there is real loss**, which both metrics agreed on.

## Limits

- Two repos, 50 cases each; `aieweb` was not swept.
- Absolutes are upper bounds — the corpus has already seen the answer — and the
  mined population is filtered upward toward small, well-described changes.
- Recall over delivered files measures whether the answer *reached* the model,
  not whether more files help it reason. A larger context can dilute as well as
  inform, and the MRR/R@1 dip at cap 50 is the only signal here that it does.
- Byte figures are means over 7–11 prompts, not the full case set.
