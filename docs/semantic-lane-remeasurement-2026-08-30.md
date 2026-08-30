# Semantic-lane re-measurement after #213

**Date:** 2026-08-30 · **Commit:** `d6ab226` · **Harness:** `tools/rank_mine_eval.py`

CLAUDE.md deferred this: `SEMANTIC_HINT_WEIGHT = CONTENT_WEIGHT` was "a defensible
default, NOT a tuned one — re-measure it against a catalog built after #213 lands".
#213 landed in `d6ab226`; this is that measurement.

## Setup

| | |
|---|---|
| repo / tree | `/Users/mliotta/git/neo` @ `d6ab226`, both arms |
| cases | 50 git-mined, `--skip-recent 50`, `--max-truth-files 5` |
| recency | `--no-git` (default) — `git_recent` gated off |
| working tree | clean (a dirty tree leaks into the recency signal) |
| failed cases | 0 in every arm |
| catalog | rebuilt after #213: **94 files, 100% source, 0 tests**, 1000 chunks |

The catalog was verified unchanged (same mtime, same composition) after the sweep,
so every arm read the same one.

## Result: the flag now helps, and the old figure is superseded

| metric | flag off | `--semantic` | delta |
|---|---|---|---|
| R@1 | 0.403 | 0.430 | +0.027 |
| R@3 | 0.572 | 0.628 | +0.056 |
| R@10 | 0.708 | 0.767 | +0.059 |
| H@1 | 0.680 | 0.760 | +0.080 |
| **MRR** | **0.778** | **0.841** | **+0.063** |

Paired by case: **11 better, 2 worse, 37 tied**, two-sided sign test **p = 0.022**.

The previously recorded **−0.007 MRR is superseded**. It was a property of the
100%-test catalog it was measured against — not of the weight — and that catalog
no longer exists.

## The win is the weight, not the depth

`--semantic` moves two things at once: the catalog's weight (1.0 → 3.0) and its
retrieval depth (1× → 3×). Pinning the weight to 1.0 while leaving depth at 3×
isolates them:

| arm | MRR | vs flag-off |
|---|---|---|
| flag off | 0.778 | — |
| depth 3× only (weight 1.0) | 0.769 | 1 better / 3 worse, p = 0.625 |
| depth 3× + weight 3.0 | 0.841 | 11 better / 2 worse, p = 0.022 |

Retrieving deeper into the catalog buys nothing on its own. What pays is weighing
the catalog as heavily as the keyword index.

## The weight is a broad plateau; 3.0 is kept, not re-tuned

| `SEMANTIC_HINT_WEIGHT` | R@1 | R@3 | R@10 | MRR | win/lose vs off | p |
|---|---|---|---|---|---|---|
| off | 0.403 | 0.572 | 0.708 | 0.778 | — | — |
| 1.0 | 0.383 | 0.592 | 0.720 | 0.769 | 1/3 | 0.625 |
| 2.0 | 0.410 | 0.620 | 0.755 | 0.803 | 8/2 | 0.109 |
| **3.0 (current)** | **0.430** | 0.628 | 0.767 | **0.841** | **11/2** | **0.022** |
| 6.0 | 0.405 | 0.668 | 0.778 | 0.853 | 12/3 | 0.035 |
| 9.0 | 0.365 | 0.648 | 0.778 | 0.832 | 11/5 | 0.210 |

An inverted U that degrades by 9.0. **6.0 is not distinguishable from 3.0** —
paired, 6 better / 2 worse of 50, p = 0.289 — and its R@1 is *lower* (0.405 vs
0.430).

**The default is unchanged, deliberately.** Moving it to the sweep's argmax would
be selecting on the same 50 cases that produced the sweep. The defensible
conclusion is narrower and more useful: `SEMANTIC_HINT_WEIGHT = CONTENT_WEIGHT`
now has evidence behind it rather than only a symmetry argument.

## Limits

Every caveat in the harness's own docstring applies and is not repeated here in
full. The load-bearing ones:

- **Absolutes are upper bounds.** Cases are queried with a commit subject and
  scored against files as they exist *after* that commit landed, so the corpus has
  already seen the answer. No flag removes this; both arms inherit it, so the
  *direction* survives but the magnitudes should not be quoted as precise.
- **The mined population is filtered upward** — well-described, small, surviving
  changes only. Read it as a proxy for focused maintenance prompts.
- **`_history_boost` stays live** under `--no-git`. Ungated is not the same as
  clean; CLAUDE.md measures its contribution as nil, but the provenance is the
  same family.
- **One repo, 50 cases.** `aieweb` and `m365dotnet` still have no `.neo/index.json`,
  so the cross-repo half of this is unmeasured. Building catalogs there and
  re-running is the obvious next step if the semantic lane is to be trusted
  broadly.
