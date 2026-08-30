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

## 2b. Retrieval is causally necessary for a USABLE fix — PROVEN (n=3, and the mechanism is narrower than it looks)

Sections 1–2 measure retrieval. They do not show that better retrieval yields
better *answers*. Git-mined cases cannot show it either: the fix is already
present at HEAD, so neo reads the file, sees the code already does what the
subject describes, and correctly proposes nothing. Measuring answers needs a
pre-fix state, so one was constructed.

Three real bugs were planted in a scratch repo, each with a failing test that
defines "fixed" objectively — no judge, no opinion:

| bug | defect |
|---|---|
| `truncate` | appends `"..."` after slicing to `limit`, so output is `limit+3` |
| `mean` | `ZeroDivisionError` on an empty list |
| `is_under` | bare prefix match, so `/a/bc` reads as under `/a/b` |

Two arms per bug: **A** = neo's normal context; **B** = the same prompt with the
buggy file `--exclude`d (what a retriever that missed the file delivers). The
exclusion was verified via `--dry-run` before trusting the result. The patch is
applied with `git apply`, then the test is run.

| arm | patch applies | test passes |
|---|---|---|
| **A — file in context** | **3/3** | **3/3** |
| **B — file excluded** | **0/3** | 0/3 |

### What actually failed in arm B — the honest mechanism

Not the reasoning. In every case the *fix logic* was right; for `is_under` the
replacement line was byte-identical to the arm that worked. What differed was the
CONTEXT line: arm B invented the docstring as `Return whether path is under
root.` where the file actually reads `True when path lies inside directory
root.`, so `git apply` rejected the patch.

Without the file, neo **hallucinated the surrounding code**. The measured
contribution of retrieval here is **groundedness, not reasoning**: on textbook
defects the model knows the fix from priors either way, and what context buys is
a patch that matches the real file.

That is practically decisive — an unapplicable patch is worthless to a user — but
it is **not** evidence that retrieval improves the model's reasoning. On a
codebase-specific bug, where priors cannot supply the answer, the reasoning gap
would plausibly be larger; that was not tested. n=3, and the bugs were chosen to
be objectively checkable rather than representative.

### The chain

1. neo retrieves the needed file more often than any naive baseline (§1, 150
   cases, p = 6.3 × 10⁻⁵), and its margin grows with repo size (§2).
2. Having that file is what makes the emitted patch usable at all (§2b, 3/3 vs
   0/3).

So neo produces usable fixes more often than the alternatives do. That is the
effectiveness claim, and it is bounded by exactly the caveats above.

## 2c. Retrieval enables project-specific REASONING, and the failure mode is refusal — PROVEN (n=2)

§2b left a gap it named itself: its bugs were textbook, so priors supplied the
fix logic in both arms and retrieval's measured contribution was groundedness.
This closes that gap with defects whose answer **priors cannot know**, because
the convention exists only in the repository.

A scratch repo defines two project conventions:

- `app/errors.py` — `ConfigError(message, *, code)` where `code` is REQUIRED and
  restricted to three values.
- `app/keys.py` — `normalize_key()` lowercases **and** strips an `acme::` prefix.

Two files ignore them (`loader.py` raises bare `ValueError`; `registry.py` only
lowercases). The tests that define "fixed" assert the project-specific
behaviour: `ei.value.code == "missing"`, and that `"ACME::Foo"` resolves to
`foo`. A generic `ValueError`, or `ConfigError("...")` without the kwarg, fails.

**The arms isolate reasoning from groundedness.** Arm B excludes the
*convention-defining* file while leaving the file to be edited in context —
verified: `app/loader.py` is present at score 11.00 in arm B. So neo can reach
what it must change, and only the project knowledge varies.

| arm | patch | project-convention test |
|---|---|---|
| **A — convention visible** | applied 2/2 | **PASS 2/2** |
| **B — convention withheld** | no patch 2/2 | 0/2 |

### The failure mode is refusal, not hallucination

Arm B did not guess. Its own output:

> "No code change is proposed yet. First inspect `app/errors.py` and at least one
> existing usage, then replace ValueError with that verified project exception."

and it flagged: *"The exact configuration exception class and established import
convention cannot be verified."*

It identified the file it needed, declined to invent a convention, and said why.
That is the correct epistemic behaviour, and it contrasts instructively with
§2b: on **textbook** defects, priors gave the model false confidence and it
emitted a confident patch against hallucinated surrounding code; on
**project-specific** defects it recognised it could not know and stopped.

### A design flaw that had to be fixed first

The first run of this experiment reported arm B **passing** — because
`test_conv.py` lived in the repo and was retrieved into context at score 2.52.
It contains `from app.errors import ConfigError` and
`assert ei.value.code == "missing"`, so the no-context arm was reading the
answer out of the test rather than reasoning to it. The tests were moved outside
the repository entirely, where they cannot be retrieved, and the result
inverted. A benchmark whose answer key is inside the corpus measures nothing.

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
- **Answer reasoning is now measured (§2c) but at n=2.** Retrieval is shown to
  enable project-specific correctness that priors cannot supply, with refusal —
  not hallucination — as the failure mode. Two conventions in one scratch repo
  is an existence proof, not a rate. How often real tasks turn on
  project-specific knowledge, versus knowledge the model already has, is
  unmeasured.
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
