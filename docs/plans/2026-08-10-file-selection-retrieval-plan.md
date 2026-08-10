# File selection: replace path-string heuristics with content BM25

**Status:** ready to implement · **Date:** 2026-08-10 · **Author:** diagnosis session, no code changed

Neo picks which files the model sees by scoring the **path string and the byte
count**. It never reads the file. Measured over 506 real cases across three
repositories, the correct file reaches the top 10 **22% of the time** on this
repo. Replacing the scorer with BM25 over file content raises that to **70%**,
and the same change wins on every repo and every cutoff tested.

This document is the diagnosis, the evidence, the chosen design, the
alternatives that were measured and rejected, and the staged work. A fresh
session should be able to execute it without re-deriving anything.

---

## 1. The problem

### 1.1 File selection never reads the file

`score_candidate` (`src/neo/context_gatherer.py:338`) takes
`(rel_path, size, prompt_tokens, git_recent, entry_points)`. There is no
content parameter. File content is first read at `context_gatherer.py:848`,
**after** selection, only to chunk what was already chosen.

Everything else in this document follows from that. With no content signal,
the only available evidence is the filename and the file's size, so relevance
has to be inferred from a pile of hand-tuned additive bonuses:

| signal | value |
|---|---|
| docs bonus (`readme`, `docs/`, …) | +0.8 |
| keyword hits in path, capped at 3 | +0.6 each |
| `main_impl` stem whitelist | +0.4 |
| git recency | +0.3 |
| entry-point basename | +0.2 |
| depth | −0.05 per separator |
| **size, once over 10 KB** | **−0.01 per KB, uncapped** |

### 1.2 The size penalty has the sign backwards

A realistic source file earns +0.6 to +2.1. The size penalty is unbounded.
So **a file with one keyword hit becomes unrankable above 60 KB**, two hits
above 120 KB.

Ground-truth files in the labelled set are 31–177 KB. The corpus median is
10 KB. Central files are large *because* they are central.

Measured on this repo, for the prompt *"fix the fact store supersession
threshold"*:

```
src/neo/memory/store.py    keyword +0.60   depth −0.15   size −1.62   → 0.000
```

The most relevant file in the repository scores exactly zero, and ranks 200th
of 284.

**This was already diagnosed once.** The comment above the penalty
(`context_gatherer.py:487`) reads: *"they're large because they're central,
and the old 0.002 multiplier was pushing THE relevant file (93KB engine.py)
below threshold on prompts about it."* The fix applied was a seven-name stem
whitelist (`core, engine, main, index, app, server, lib`). It rescues
`engine.py` and not `store.py`:

| file | KB | size penalty | whitelisted |
|---|---|---|---|
| `src/neo/engine.py` | 177 | **−0.13** | yes |
| `src/neo/memory/store.py` | 162 | **−1.62** | no |

A 12× disparity between two files of near-identical size, decided by whether
someone thought of the name.

There is also a discontinuity: the penalty is `0.01 * size_kb`, not
`0.01 * (size_kb − 10)`, so a 9.9 KB file pays 0 and a 10.1 KB file pays
−0.101.

**The literature says the sign is wrong, not just the magnitude.** This task is
IR-based bug localization. BugLocator (Zhou et al., ICSE 2012) revises the
vector space model specifically to add a *logistic length function*
`1/(1+e^{−N(#terms)})` that ranks **larger files higher**, because larger files
are empirically more likely to contain the defect. BM25 handles the same
concern properly with bounded length normalization (the `b` parameter) rather
than an unbounded linear penalty.

### 1.3 The semantic path has the same bias, arrived at independently

`MAX_CHUNKS_PER_REPO = 1000` (`src/neo/index/project_index.py:52`), and
`_cap_chunks` round-robins so every file keeps a chunk before any file keeps a
second. That fairness was introduced by #159 for a good reason. Its effect
here:

| file | functions | indexed | coverage |
|---|---|---|---|
| `src/neo/memory/store.py` | 82 | 6 | **7%** |
| `src/neo/engine.py` | 95 | 6 | **6%** |
| `src/neo/index/project_index.py` | 61 | 6 | 10% |
| `src/neo/text_budget.py` | 4 | 4 | **100%** |

`src/` alone produces 1,344 chunks against the 1,000 cap, before tests and
docs compete for the same slots.

So both retrieval paths under-represent large central files — one by explicit
penalty, one by equal allocation. The index is **not** stale (`commit_hash`
matches HEAD) and **not** missing; it is truncated.

### 1.4 What this costs

Measured with `tools/rank_eval.py` against 506 cases mined from git history
(commit subject = query, changed non-test files = ground truth):

| repo | cases | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| neo | 207 | 0.048 | 0.143 | 0.223 | 0.143 |
| car | 75 | 0.109 | 0.369 | 0.382 | 0.296 |
| quip | 224 | 0.028 | 0.102 | 0.137 | 0.107 |

On quip, the correct file is in the top 10 **14% of the time**.

---

## 2. The solution

**Rank files by BM25 over their content, with code-aware tokenization.**

Neo already ships the BM25 (`src/neo/memory/bm25.py` — Lucene-style with IDF
smoothing) and already runs hybrid dense+sparse fusion for *fact* retrieval
(`store.py:751`, 0.7/0.3). File selection uses none of it.

Three properties matter, and each is why this is the right instrument rather
than another bonus:

1. **Length normalization is built in and bounded.** BM25's `b` parameter
   normalizes by document length against the corpus average. That is the
   principled version of what the size penalty was reaching for, and it cannot
   run away to −1.62.
2. **IDF is computed from the corpus.** A term's weight comes from how
   discriminating it actually is in *this* repository, which is what the two
   rejected document-frequency and length-floor fixes were hand-approximating.
   A repo is a small corpus, and BM25's relative advantage *rises* as the
   corpus shrinks.
3. **Code-aware tokenization is what makes it beat dense retrieval.** Split
   identifiers on camelCase and separators and keep both the whole identifier
   and its parts, so `getUserById` matches a query saying "user by id".

### 2.1 Measured result

| repo | cases | R@1 | R@5 | R@10 | MRR | MRR vs current |
|---|---|---|---|---|---|---|
| neo | 207 | 0.278 | 0.612 | 0.697 | 0.565 | **3.9×** |
| car | 75 | 0.347 | 0.838 | 0.969 | 0.621 | **2.1×** |
| quip | 224 | 0.268 | 0.638 | 0.762 | 0.544 | **5.1×** |

### 2.2 Do NOT add hybrid fusion yet

The literature recommends RRF hybrid as the minimum viable baseline. **On
neo's current index it makes things worse**, and the reason is diagnosed:

| variant | R@10 (neo, 207 cases) |
|---|---|
| bm25 only | **0.697** |
| RRF equal weights | 0.432 |
| RRF weighted 3:1 toward BM25 | 0.567 |
| RRF weighted 9:1 toward BM25 | 0.687 |

The dense channel returns ~33 files against BM25's 183. Equal-weight RRF lets
a short, low-quality list promote mediocre BM25 hits, and no weighting
recovers past BM25 alone. **The dense channel currently has negative marginal
value**, because of the chunk-allocation defect in §1.3.

Fusion becomes correct *after* §3 stage 3, and only if it beats BM25-only on
the harness. That is a gate, not a plan.

---

## 3. Staged work

Each stage is independently shippable and independently measurable. Run
`tools/rank_eval.py` before and after every one.

### Stage 1 — BM25 over content (the whole win)

- Add `neo.retrieval.code_tokens(text)`: split on non-alphanumerics and
  camelCase, emit both whole identifiers and parts, lowercase.
- Build a BM25 index over candidate files at gather time. Document =
  `code_tokens(rel_path) * 3 + code_tokens(content[:200_000])`. The path
  repetition keeps "a file named for the thing" as real evidence without
  making it the only evidence; the 200 KB read cap bounds cost.
- Rank by BM25. Keep `EXPLICIT_PATH_BOOST` (`context_gatherer.py:223`) — a
  path named outright in the prompt must still win outright.
- **Delete the size penalty** (`context_gatherer.py:487-497`) and the
  `main_impl_stems` whitelist that exists to patch it.
- Keep git recency and the docs bonus **as tie-breakers only**, scaled well
  below the BM25 signal, or delete them; measure both.

Acceptance: neo R@10 ≥ 0.65, quip R@10 ≥ 0.70, car R@10 ≥ 0.90, no repo
regressing on MRR.

Cost check: BM25 build is O(corpus) per invocation. Measure it. If reading
~300 files per run is too slow, cache the index keyed on a content hash — but
measure before optimizing, since the current code already reads every selected
file anyway.

### Stage 2 — Re-validate the two shipped changes against the new baseline

PR #188's test-file demotion and the `TEST_PENALTY` unification were tuned
against the old scorer. Under BM25 the tests may no longer outrank
implementations at all, in which case the demotion is either unnecessary or
actively harmful. Re-measure and keep only what earns its place.

### Stage 3 — Fix chunk allocation, then re-test fusion

- Allocate index chunks proportional to file size (or symbol count) with a
  floor, instead of equal round-robin. `store.py` at 7% coverage cannot be
  retrieved for most of what it contains.
- Reconsider `MAX_CHUNKS_PER_REPO = 1000`. `src/` alone needs 1,344.
- **Then** re-run the fusion variants. Adopt RRF only if it beats BM25-only.

### Stage 4 — Documentation

Update the CLAUDE.md context-selection section. It currently documents the
chunk-selection fixes in detail and never says that file selection reads no
content.

---

## 4. Alternatives measured and rejected

Do not retry these. Each was implemented and measured this session.

| alternative | result | why |
|---|---|---|
| Document-frequency filter on prompt tokens | R@10 5.58→5.75 *worse* (old metric) | `'a'` matches 49/85 basenames, but removing it changed only 2 of 12 prompts and both degraded. No measurable benefit. |
| `{3,}` length floor on the identifier regex | not shipped | Drops `db`, `os`, `fs`, `ui`, `id` — the short identifiers that carry most signal. Length is inverted on real prompts. |
| Whole-token matching for short tokens | MRR 0.583 vs 0.667 *worse* | Addressed the real mechanism (substring matching) and still lost, because it removes signal without adding any. |
| Cutting the `docs/` +0.8 bonus to +0.2 | helped code prompts, cost doc prompts 2.16→0.66 | A direct trade between prompt classes. |
| Cross-encoder reranking | not attempted, and should not be | Marginal below ~1,000 documents (arXiv 2604.01733). A repo is a small corpus. |

All four of the first alternatives share one flaw: they tune the *filename*
signal. The filename is not the evidence. The file is.

---

## 5. Evaluation harness

`tools/rank_eval.py` — recall@k and MRR against ground truth.

Two label sources:

1. **Git-derived (preferred, 506 cases).** `--build-from-git <repo>` mines
   commits: subject = query, changed non-test files = ground truth, keeping
   commits touching 1–3 files. Real queries, real labels, no opinion, and it
   scales to any repo.
2. **Hand-labelled (12 cases).** Retained for spot-checking; too small to
   support a conclusion on its own.

**Report several k.** Differences live at tight cutoffs, because per-file
character caps mean a file at rank 3 contributes far more content than the
same file at rank 9.

Known limits, stated so the numbers are not over-read: commit subjects are
terser and better-formed than real prompts; ground truth is what the commit
*changed*, which is a subset of what a developer needed to *read*; and file
content is read at HEAD while the commit is historical.

---

## 6. Open questions for the implementation session

1. **Is per-invocation BM25 build acceptable?** Measure wall-clock on a
   1,000-file repo before adding a cache.
2. **Should the docs bonus survive at all?** It exists for broad
   architecture prompts. Under content BM25 an architecture doc may rank on
   its own text. Measure with a doc-seeking prompt set.
3. **Does `select_chunks` need revisiting?** It has its own document-frequency
   logic over lines. Once file selection uses BM25, the two should probably
   share tokenization.
4. **Should content BM25 replace or complement `EXPLICIT_PATH_BOOST`?**
   Current recommendation: keep the boost, it encodes an unambiguous user
   instruction that no statistical signal should override.
