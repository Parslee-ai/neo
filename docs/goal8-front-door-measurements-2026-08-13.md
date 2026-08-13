# Goal 8 — one retrieval front door: measurements

**Measured:** 2026-08-13
**Goal:** Unified Store Plan, Goal 8 (One front door) — see `docs/unified-store-plan.md`
**Arms:** `main` = `d5adcbc4e848` (post-#207, the tip of `origin/main` at measurement
time) · `branch` = `0da5ecb8c845`

Measured **once, at the end**, against the final base — the plan's measurement
discipline after Goal 3's re-measure treadmill. Every number below was produced by
the command printed beside it; the raw harness output is committed under
`evidence/`, so the tables here are reproducible rather than transcribed.

Two placeholders, so the doc is not pinned to one developer's home directory:
`<platform-root>` is the `parslee-knowledge` checkout holding the child repos as
subdirectories, `<neo-worktree>` is the branch worktree this was produced from.

```bash
export platform_root=/path/to/parslee-knowledge
export neo_worktree=$platform_root/neo/.worktrees/g_msr9uzf2_114f50-unified-store-goal-8
export main_tree=$platform_root/neo/.worktrees/g_msr9uzf2_114f50-main-baseline
```

## Measurement conditions, disclosed

The repositories under test are separate from the source trees under test: `--repo`
says where cases are mined and files are read, `--tree` says which ranker executes.
Both arms of every comparison share `--repo`, so anything below is inherited equally
and cannot produce a delta — but it does bound what the ABSOLUTE numbers describe.

| repo | HEAD at measurement | working tree |
|---|---|---|
| neo | `5bbee46` (behind `origin/main` = `d5adcbc`) | 1 modified + 1 staged file, unrelated in-flight work, timestamped 20 h before the first run and unchanged throughout |
| aieweb | `26fff07e` | clean |
| m365dotnet | `61dc4a17` | clean |

The two flagship HEADs are the ones Goal 1's baseline doc records, so these figures
sit on the same corpus as the rest of the plan. The neo checkout is *not* at the
commit whose ranker the `main` arm runs — `repo_head` and `tree_head` are stamped
separately in every JSON for exactly this reason, and the neo table below reads "the
d5adcbc ranker against the 5bbee46 corpus", both arms alike.

Unlike Goal 6's m365dotnet run, no tree moved mid-sweep: the neo modifications
predate the first arm by 20 hours and both flagships were clean.

The stage-4 experiment further required building `.neo/index.json` in the neo
checkout. It was **removed after measurement**, restoring the directory to the
`content_index.sqlite3` + `walk_cache.json` it held before — leaving a tests-only
catalog in place would have re-ranked every subsequent real invocation in that repo,
on both `main` and this branch, for as long as #213 is open.

---

## Headline

| | main | branch | verdict |
|---|---|---|---|
| **M1** MRR / R@10, all three flagships | — | **identical, every cell** | no regression |
| **M1** files delivered per query (mean distinct) | 18.4 / 22.6 / 22.1 | **28.9 / 28.7 / 29.3** | +57% / +27% / +32% |
| **M1** files lost vs main, 150 cases | — | **0** | branch is a strict superset |
| **M2** battery median wall (m365dotnet) | 9.10 s | **8.97 s** | −1.4%, inside noise |
| **M2** battery peak `ru_maxrss` | 1387.2 MiB | **1390.5 MiB** | +0.24%, inside noise |
| **G1-inv** gitignored files selected | 0 / 111 | **0 / 125** | clean |
| **G1-inv** byte-identical duplicate copies | 0 | **0** | clean |
| **#197** within-prompt repeat entries (M2 battery) | 45 | **0** | structural |

---

## M1 — retrieval quality, branch vs main

```bash
$neo_worktree/.venv/bin/python $neo_worktree/tools/rank_mine_eval.py \
  --repo <repo> --tree <main_tree|neo_worktree> \
  --cases 50 --skip-recent 50 --k 1 3 10 --timeout 600 --json
```

Both arms of a repo share `--repo` (identical mined cases at an identical HEAD) and
differ only by `--tree`, so the delta is the ranker and nothing else. 50 cases per
arm, git recency off (the harness default; `--with-git` opts in). 0 failed cases in
all six runs. The scorer asserts the two arms' case lists are equal before scoring,
because two arms scored on different samples produce a delta that describes the
sample.

| repo | arm | n | failed | **MRR** | R@1 | R@3 | **R@10** | H@10 |
|---|---|---|---|---|---|---|---|---|
| neo | main | 50 | 0 | 0.712 | 0.307 | 0.502 | 0.708 | 0.980 |
| neo | **branch** | 50 | 0 | **0.712** | 0.307 | 0.502 | **0.708** | 0.980 |
| aieweb | main | 50 | 0 | 0.728 | 0.487 | 0.654 | 0.778 | 0.880 |
| aieweb | **branch** | 50 | 0 | **0.728** | 0.487 | 0.654 | **0.778** | 0.880 |
| m365dotnet | main | 50 | 0 | 0.669 | 0.312 | 0.497 | 0.771 | 0.860 |
| m365dotnet | **branch** | 50 | 0 | **0.669** | 0.312 | 0.497 | **0.771** | 0.860 |

Byte-identical in every cell, on every reported k. **No regression on MRR or R@10**,
which is the DoD.

### What did move: 1,190 files gained, none lost

Equal metrics could mean "nothing changed". It does not here, and the stronger
result is in the file sets rather than the scores.

| repo | mean distinct files delivered, main | branch | files LOST | files GAINED |
|---|---|---|---|---|
| neo | 18.38 | **28.90** | **0** | 526 |
| aieweb | 22.58 | **28.70** | **0** | 306 |
| m365dotnet | 22.14 | **29.30** | **0** | 358 |

In **150 of 150 cases the branch's file set is a strict superset of main's**, and
the top-10 is unchanged in 149 of them. The gain sits below rank 10, which is
exactly why the metrics do not move: R@10 cannot see a file that arrives at rank 24.

The cause is whole-file delivery. Under `main` a file over 15,000 characters was
delivered as up to two windows, each consuming a slot of the adaptive file cap — so
a 30-slot budget bought 18 to 22 files' worth of coverage. One entry per file spends
those slots on files.

### The one case in 150 whose top-10 moved

```
query : feat: neo memory rules — flag drift between AGENTS.md / CLAUDE.md / GEMINI.md
main  : 1. CLAUDE.md   2. AGENTS.md   3. docs/solutions/rule-file-sync.md ...
branch: 1. AGENTS.md   2. CLAUDE.md   3. docs/solutions/rule-file-sync.md ...
```

Stage 1 pinning, visible: both files are named by the query, so both are pinned and
delivered in path order rather than in score order. Ranks 3 onward are identical and
no metric moves — the two files that swapped are neither of them ground truth for
this case.

### Concept-shaped subset, reported separately

```bash
$neo_worktree/.venv/bin/python $neo_worktree/evidence/score_subsets.py
```

A case is **file-named** when its query names a path the repo actually has —
`extract_explicit_paths` + `matches_explicit_path` against `git ls-files`, i.e. the
gatherer's own resolution rather than a regex. That is the test that decides which
stage runs: a named path is PINNED (stage 1), a concept query is RANKED (stages 3
and 4).

| repo | subset | n | MRR main | MRR branch | R@10 main | R@10 branch |
|---|---|---|---|---|---|---|
| neo | concept | 49 | 0.721 | **0.721** | 0.717 | **0.717** |
| neo | file-named | 1 | 0.250 | **0.250** | 0.250 | **0.250** |
| aieweb | concept | 50 | 0.728 | **0.728** | 0.778 | **0.778** |
| m365dotnet | concept | 50 | 0.669 | **0.669** | 0.771 | **0.771** |

**The honest reading: this instrument cannot measure the hoped-for concept win, and
the reason is not the branch.** 149 of 150 mined cases are concept-shaped, so the
concept subset *is* the whole corpus and the split carries no information. More
importantly, the stage that was expected to produce the win — stage 4, the embedding
catalog — **does not exist on any flagship**: there is no `.neo/index.json` in neo,
aieweb or m365dotnet, so `_project_index_boost` returns `{}` on every one of these
300 runs and the semantic stage is inert in both arms.

That is not a gap in the branch; it is what the plan means by "WHEN the index
exists". The stage-4 change is measured separately below, against a repo that has
a catalog.

---

## M2 — wall-clock and peak RSS, branch vs main

```bash
$neo_worktree/tools/m2_battery.sh \
  $neo_worktree/evidence/bin/neo-<arm> $platform_root/m365dotnet <out-dir>
```

Canonical 6-prompt battery on m365dotnet, 3 timed runs per prompt after a discarded
warm-up, observer disabled. Arms run sequentially so they do not contend. The arm
launchers set `PYTHONPATH` to pin which tree executes; both were verified to load
their own `neo/__init__.py` before the run, and the `main` arm still has
`gather_context_semantic` while the `branch` arm does not.

| | main | branch | delta |
|---|---|---|---|
| battery median wall | 9.10 s | **8.97 s** | −1.4% |
| battery wall range | 7.62–10.44 s | 7.67–10.51 s | — |
| battery peak `ru_maxrss` | 1387.2 MiB | **1390.5 MiB** | +0.24% |

Both deltas are inside the noise floor — main's own runs span 7.62 s to 10.44 s
across the battery, so a 0.13 s median difference is not a signal. This is a
developer laptop, not an isolated box. **The front door does not re-add cost**,
which is the M2 requirement for this goal; the absolute figures are #212's warm
profile, unchanged.

Per prompt:

| id | shape | main median | branch median | main peak MiB | branch peak MiB |
|---|---|---|---|---|---|
| P1 | file-named | 9.37 | 10.50 | 1385.2 | 1390.5 |
| P2 | file-named | 10.19 | 9.49 | 1387.2 | 1387.5 |
| P3 | concept-only | 8.37 | 7.90 | 1380.5 | 1384.5 |
| P4 | concept-only | 9.14 | 9.02 | 1382.0 | 1384.6 |
| P5 | mixed | 8.36 | 8.92 | 1385.1 | 1385.2 |
| P6 | symptom | 9.31 | 9.14 | 1385.1 | 1385.2 |

P1 and P5 are slower and P2, P3, P4, P6 faster; every one of those differences is
smaller than the spread within a single prompt's own three runs. Read the battery
median, not a row.

### Selection, on the same battery

| | main | branch |
|---|---|---|
| context entries emitted | 180 | 151 |
| sum of per-prompt distinct files | 135 | 151 |
| **within-prompt repeat entries** | **45** | **0** |
| battery union, distinct files | 111 | **125** |

`entries == per-prompt distinct` on the branch, for all six prompts. That is #197's
defect closed at the source rather than reported: a file can no longer be two
entries, so the count that says "files" and the count that says "chunks" are the
same number by construction. The 45 repeat entries on `main` were 45 delivery slots
spent on second windows of files already present.

---

## G1-inv — the union of everything selected

```bash
cd $platform_root/m365dotnet && git check-ignore --stdin < <out-dir>/union.files
```

| | main | branch |
|---|---|---|
| union size | 111 | 125 |
| `git check-ignore` hits | **0** | **0** |
| byte-identical duplicate copies | **0** | **0** |

Duplicates are tested on CONTENT, not on basename. Three basenames repeat in the
branch union (`CURRENT.md` ×8, `INDEX.md` ×2, `AUTH_AND_SECURITY.md` ×2) and every
one is a genuinely different file — a per-product summary, a per-directory index.
Counting shared basenames as duplicates would report a G1-inv break on a repository
that has none.

---

## Stage 4 — the semantic supplement, on a repo that has a catalog

Separate experiment, separate repo state, **not comparable cell-for-cell with the M1
table above**: `neo --index` was run against the neo checkout first, so
`.neo/index.json` exists here and does not exist in any of the six M1 runs. All four
arms below share that one catalog and the same 50 cases.

```bash
$neo_worktree/evidence/bin/neo-branch --index --cwd $platform_root/neo   # 20m19s
$neo_worktree/evidence/run_semantic.sh
```

| arm | flag | MRR | R@1 | R@3 | R@10 | H@10 | mean files | test files delivered |
|---|---|---|---|---|---|---|---|---|
| main | — | 0.712 | 0.307 | 0.502 | 0.708 | 0.980 | 18.16 | 8.6% |
| main | `--semantic` | **0.000** | 0.000 | 0.000 | **0.000** | 0.000 | 24.48 | **100.0%** |
| **branch** | — | 0.712 | 0.307 | 0.502 | 0.708 | 0.980 | 28.90 | 10.6% |
| **branch** | `--semantic` | **0.705** | 0.307 | 0.472 | **0.708** | 0.980 | 28.90 | 23.0% |

### The lane scored zero on all 50 cases

`main --semantic` returned **1,224 files across 50 cases and every single one was a
test file** — 100.0%, and 100.0% within the top 10. Ground truth is non-test source
the commit changed, so the lane could not score above zero on any case.

Two causes, compounding, and only one of them is the lane's:

1. **The catalog contains no source.** `neo --index` selected 99 files and all 99
   are tests. `ProjectIndex._select_files` ranks shallowest-path-first, and on the
   `src/<pkg>/…` + `tests/…` layout every test is at depth 2 while every source file
   is at depth 3: this repo has 105 Python files at depth 2, the first non-test file
   appears at rank 102, and `--max-files` defaults to 100. Filed as **#213**; it is
   the index's selection rule, not this goal's, and is deliberately not fixed here.
2. **The lane applied none of the pipeline's judgement.** It read
   `index.retrieve(k=100)` and MMR-packed the result — no test demotion (which
   `_project_index_boost` has always applied), no keyword channel, no prompt-named
   pin. Whatever the catalog held, it shipped.

### What the front door does with the same broken catalog

`branch --semantic` reads that same all-tests catalog at three times the weight and
three times the depth, and scores **0.705 / R@10 0.708** — because the catalog is
one channel among four rather than the whole answer. Test demotion applies, BM25
still ranks over the real corpus, and pins still bind. Test files rise from 10.6% to
23.0%, which is the hint doing exactly what it was asked to do with the evidence it
was given.

**The honest reading, stated as a limit rather than a win.** This experiment
demonstrates *robustness* — the front door absorbs a catalog that is entirely wrong
where the lane amplified it — and it does **not** demonstrate the concept-shaped
retrieval win the plan hopes for. It cannot: with no source in the catalog, stage 4
has nothing relevant to contribute, so the −0.007 MRR / −0.030 R@3 under the flag
measures the catalog's contents, not the hint's weight. `SEMANTIC_HINT_WEIGHT`
should be re-measured against a catalog built after #213 is fixed, and until then it
is a defensible default rather than a tuned one. `--semantic` is opt-in and the
default path is byte-identical to main, so nothing ships on the strength of this
number.

## G-invariant battery

```bash
cd $neo_worktree && .venv/bin/python -m pytest -m invariants -q
```

`100 passed, 1 skipped` — gitignored, duplicates, the `--include` guarantee, the
prompt-named guarantee and the truncation markers, each asserted per language
(C#, TypeScript, Python).
