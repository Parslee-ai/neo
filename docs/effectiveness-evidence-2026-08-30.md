# Is neo effective? — evidence, and its limits

**Date:** 2026-08-30 · **Code:** `094ed5e` · **Repos:** neo, aieweb, m365dotnet

"Effective" is not a single claim, and absolute R@k figures cannot answer it —
they have no comparator. This document separates what is now demonstrated from
what is not, and names the baseline for every number.

---

## 1. Context selection beats every naive baseline — PROVEN

`tools/rank_baseline_eval.py` scores four naive rankers over the *identical*
candidate universe, queries and ground truth that `tools/rank_mine_eval.py`
used, so the ranking rule is the only thing that differs. 50 git-mined cases per
repo, 150 total.

### The control that matters

Baselines were chosen to be fair rather than strawmen, and one of them is
load-bearing. The harness's known leak — a file contains the commit's terms
*because* that commit put them there — helps any ranker that reads **content**
and does nothing for `size` or `recency`. Comparing only against content-blind
baselines would let the leak masquerade as ranker quality.

So the strongest baseline is **`grep`: count prompt-token occurrences in file
content** — literally what a developer does, and it inherits the same leak neo's
BM25 does. It is by far the best naive ranker, and it is the one the headline
number is quoted against.

### Result

neo vs `grep` (the strongest baseline), MRR:

| repo | candidates | neo | grep | ratio | vs random | win/lose/tie | sign p |
|---|---|---|---|---|---|---|---|
| neo | 99 | **0.778** | 0.584 | 1.3× | 6× | 25/11/14 | 0.029 |
| aieweb | 480 | **0.729** | 0.511 | 1.4× | 32× | 22/13/15 | 0.175 |
| m365dotnet | 2,882 | **0.536** | 0.219 | 2.4× | 43× | 34/13/3 | 0.003 |

**Pooled over 150 cases: 81 better, 37 worse, 32 tied — two-sided sign
p = 6.3 × 10⁻⁵.**

Full ranker table on m365dotnet (MRR): random 0.012 · recency 0.009 ·
filename 0.042 · size 0.098 · grep 0.219 · **neo 0.536**.

### The comparison is conservative toward neo

neo is scored on the ~30 files it would **actually deliver**; every baseline
ranks the **entire** candidate set (99 / 480 / 2,882). Baselines therefore get
far more chances to find the answer at depth, and neo still wins. Nothing here
is rigged in neo's favour.

## 2. The advantage grows with codebase size — PROVEN

vs `grep`: 1.3× → 1.4× → 2.4×. vs random: 6× → 32× → 43×.

Naive heuristics collapse as the candidate pool grows — `size` scores MRR 0.443
on neo's 99 files and 0.098 on m365dotnet's 2,882. neo degrades far more
gracefully (0.778 → 0.536). **The tool is worth most exactly where a human is
worst off**, which is the case a retrieval tool has to win.

## 3. The memory subsystem is correct and safe — PROVEN (on synthetic scenarios)

Two deterministic benchmarks, **zero model calls**, both PASS:

`neo memory evaluate-learning` — 12/12 scenarios, with the comparison built in:

| arm | success | precision | harmful | unsupported |
|---|---|---|---|---|
| memory disabled | 0.00 | 0.00 | 0.00 | 0.00 |
| naive immediate memory | 1.00 | 1.00 | **0.50** | **1.00** |
| evidence-driven (shipped) | **1.00** | **1.00** | **0.00** | **0.00** |

The shipped design matches the naive approach's success while eliminating its
harm: remembering everything immediately is 50% harmful and 100% unsupported;
the evidence gate keeps the wins and drops both.

`neo memory evaluate-execution` — 12/12 scenarios (fail-closed gates, stale
revisions pending, waivers explicit, unsupported confirmations downgraded).

**Scope:** these are hand-written scenarios from this project, exercising the
machinery deterministically. They prove the rules behave as designed. They are
not evidence about real tasks.

---

## What is NOT proven

Stated plainly, because a proof that overreaches is worth less than a narrow one.

- **The learning loop does not currently deliver value in practice.** Measured
  this session on a live install: 2 `accepted` outcomes in 208 episodes, *both
  from drills*; zero organic promotions in 30 days; 0 of 88 global facts had
  ever held a success. Two real defects were found and fixed (the cited-credit
  save gate, the unread host-edit ledger), which removes mechanical ceilings —
  but removing a ceiling is not the same as demonstrating a benefit. The loop is
  **starved**, and that is a measurement, not an excuse.
- **Answer quality is unmeasured.** Every number here is retrieval. Whether
  better context yields better answers needs an LM-graded evaluation that has
  not been run.
- **Developer productivity is unmeasured**, and is not the kind of claim this
  instrument can settle.

## Limits on the numbers that ARE proven

- **Absolutes are upper bounds.** Cases are queried with a commit subject and
  scored against files as they exist *after* that commit landed. The `grep`
  control equalises the leak between the two arms it matters for; it does not
  remove it. Read the *ratios*, not the absolute MRRs.
- **The mined population is filtered upward** — well-described, small, surviving
  changes. A proxy for focused maintenance prompts, not a fair sample of
  developer questions.
- **50 cases per repo.** aieweb alone does not clear p = 0.05 (0.175); the
  pooled result carries the claim.
- **Ground truth is what the commit changed** — not everything needed to
  understand the task, and not necessarily the only relevant files.
- **Two baselines are unstable and neither carries the claim.** `random` was
  seeded from Python's `hash()`, which is salted per process, so it did not
  reproduce across runs; it now uses `hashlib` and the whole table is
  byte-reproducible (verified by running it twice). `recency` is mtime-based, so
  editing files in the repo under measurement perturbs it — this harness lives
  in neo, and editing it moved its own score. Both are floor/weak comparators;
  the load-bearing baseline is `grep`, which is deterministic and reproduced
  exactly.

## Reproducing

```bash
# retrieval, per repo (writes per-case JSON)
python tools/rank_mine_eval.py --repo <repo> --tree <this tree> --cases 50 --json > base.json
# baselines over the same cases
python tools/rank_baseline_eval.py <repo> base.json
# memory subsystem
neo memory evaluate-learning
neo memory evaluate-execution
```

`--tree` must be the tree `import neo` resolves to; the harness enforces it.
aieweb and m365dotnet were measured in `git clone --local` copies, never in the
working checkouts.
