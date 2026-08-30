# Semantic-lane re-measurement after #213

**Date:** 2026-08-30 · **Code:** `d6ab226` · **Harness:** `tools/rank_mine_eval.py`

CLAUDE.md deferred this: `SEMANTIC_HINT_WEIGHT = CONTENT_WEIGHT` was "a defensible
default, NOT a tuned one — re-measure it against a catalog built after #213 lands".
#213 landed in `d6ab226`; this is that measurement.

> **Correction.** A first version of this document reported the neo-only result
> (+0.063 MRR, p = 0.022) and concluded the flag now helps. Extending to all three
> flagships shows that does not generalize: **pooled, the effect is exactly null.**
> The single-repo write-up is superseded by the tables below.

## Setup

| | |
|---|---|
| cases | 50 git-mined per repo, `--skip-recent 50`, `--max-truth-files 5` |
| recency | `--no-git` (default) — `git_recent` gated off |
| code tree | `/Users/mliotta/git/neo` @ `d6ab226` for every arm |
| failed cases | 0 in all six arms |
| catalogs | rebuilt post-#213: **100% source in all three repos** |

`aieweb` and `m365dotnet` were measured in `git clone --local` copies under the
scratchpad, never in the working repos: `.neo/` is *not* gitignored in either, so
building in place would have left untracked artifacts in active checkouts (aieweb
had uncommitted work). A side benefit — a local clone's origin is the local path,
so both got fresh `project_id`s and no FactStore history, removing the
`_history_boost` contamination the harness docstring warns about.

## Headline: null across repos

`--semantic` vs flag-off, paired by case:

| repo | MRR off | MRR on | delta | R@1 off → on | win/lose/tie | sign p | catalog coverage |
|---|---|---|---|---|---|---|---|
| neo | 0.778 | 0.841 | **+0.063** | 0.403 → 0.430 | 11/2/37 | **0.022** | 49.5% |
| aieweb | 0.729 | 0.706 | −0.023 | 0.399 → 0.376 | 4/10/36 | 0.180 | 13.2% |
| m365dotnet | 0.536 | 0.542 | +0.006 | 0.211 → 0.231 | 3/6/41 | 0.508 | 1.8% |

**Pooled over 150 cases: 18 better, 18 worse, sign p = 1.000.**

The one repo where the flag helps is the only one whose catalog covers a
meaningful share of the repository.

## Coverage is the mechanism — tested, not inferred

Three points is a correlation, not a cause. So the mechanism was tested *within*
neo: rebuild its own catalog at `--max-files 30`, changing nothing else — same
repo, same 50 cases, same code, same weight.

| neo catalog | coverage | MRR off → on | delta | win/lose | p |
|---|---|---|---|---|---|
| `--max-files 100` | 49.5% | 0.778 → 0.841 | +0.063 | 11/2 | **0.022** |
| `--max-files 30` | 14.7% | 0.753 → 0.764 | +0.011 | 12/10 | 0.832 |

Starving coverage collapses a significant effect to nothing. **The semantic lane
is not weak; it is starved.** `--max-files` defaults to 100, so on any real
repository the catalog holds a rounding error of the codebase — 84 files of 4,563
eligible on m365dotnet, 1.8%.

**The lever is the cap, not the weight.** Raising `--max-files`, or making it
repo-relative, is what could make this lane earn its keep. No weight can fix a
catalog that does not contain the answer.

## Two secondary findings

**The 3× depth does nothing on its own.** `--semantic` moves both weight and
depth, so they were separated by pinning the weight to 1.0 while
`SEMANTIC_HINT_DEPTH` stayed at 3×: MRR 0.769, *below* flag-off's 0.778 (1 better
/ 3 worse, p = 0.625). What pays — when anything does — is weighing the catalog,
not reading further into it.

**At high coverage, 3.0 sits on a broad plateau.** Sweep on neo at 49.5%
coverage, 50 cases (MRR): 1.0 → 0.769, 2.0 → 0.803, **3.0 → 0.841**, 6.0 → 0.853,
9.0 → 0.832 — an inverted U degrading by 9.0. 6.0 is not distinguishable from 3.0
(paired 6 better / 2 worse of 50, p = 0.289) and its R@1 is lower (0.405 vs
0.430). Moving the default to the sweep's argmax would be selecting on the same
50 cases that produced the sweep.

**`SEMANTIC_HINT_WEIGHT = CONTENT_WEIGHT` is therefore unchanged**, now for a
sharper reason than symmetry: at realistic coverage the weight cannot be tuned
into mattering, and at high coverage it is already on a plateau.

The previously recorded **−0.007 MRR is superseded** regardless — it was a
property of the 100%-test catalog it was measured against, and that catalog no
longer exists.

## Limits

- **Absolutes are upper bounds.** Cases are queried with a commit subject and
  scored against files as they exist *after* that commit landed, so the corpus has
  already seen the answer. No flag removes this. Both arms inherit it, so
  direction survives; magnitudes should not be quoted as precise.
- **The mined population is filtered upward** — well-described, small, surviving
  changes only. Read it as a proxy for focused maintenance prompts, not a fair
  sample of developer questions.
- **50 cases per repo** is a modest sample; the per-repo p-values reflect that.
  The pooled null is the more reliable reading, and it is a null, not a proof of
  no effect.
- **Coverage was varied at two points in one repo.** The direction is clear and
  the within-repo design controls for everything else, but the shape of the
  coverage/benefit curve is not mapped.
