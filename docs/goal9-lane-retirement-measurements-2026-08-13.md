# Goal 9 — lane retirement: closing measurements

**Measured:** 2026-08-13
**Goal:** Unified Store Plan, Goal 9 (Lane retirement) — see `docs/unified-store-plan.md`
**Base:** `80a8cfc4a98b` = `origin/main` (`5c0ce5ddc79c`, post-#214) + this goal's one commit

Measured **once, at the final base**, per the plan's measurement discipline. The
raw harness output is committed under `evidence/`, so every table here is
reproducible rather than transcribed.

```bash
export platform_root=/path/to/parslee-knowledge
export neo_worktree=$platform_root/neo/.worktrees/g_msrhuemt_78c44a-unified-store-goal-9
$neo_worktree/evidence/run_goal9.sh          # M1 x3, M2, per-stage profile
```

## One arm, not two

Goals 6, 7 and 8 each ran a branch arm against a `main` arm, because each was
changing the ranker and had to show it had not regressed. **This goal changes no
runtime behaviour.** It deletes prose, rewrites four help/doc strings and adds a
guard test; `git diff origin/main -- src/` touches three lines of user-facing text
and one comment. A two-arm A/B would have spent two hours proving that identical
code performs identically.

The comparison that means something at the end of the climb is against the **Goal 1
trailhead**, and that is what this document reports. The goal-over-goal check —
"did Goal 9 regress Goal 8" — is answered below by the strongest available evidence:
every M1 cell reproduces Goal 8's branch arm exactly.

## Measurement conditions, disclosed

| repo | HEAD at measurement | working tree |
|---|---|---|
| neo | `5bbee46` (the corpus Goals 1 and 8 measured; the *ranker* is `80a8cfc`) | 4 modified/untracked, unrelated in-flight work |
| aieweb | `26fff07e` | 1 modified |
| m365dotnet | `61dc4a17` | 1 modified |

Same three flagship HEADs as Goal 1 and Goal 8, so this sits on the same corpus as
the rest of the plan.

### CPU contention, and why the battery was run three times

This is a shared developer laptop running several agent sessions. The first two M2
batteries were measured while a peer session ran repeated `tools/rank_mine_eval.py`
sweeps out of the Goal 8 worktree (PIDs 29801, 63178, 93876); that session then
cleared the machine, and the third battery ran with the box quiet. All three are
committed. The spread is the point:

| battery | 1-min load at start | median wall | peak `ru_maxrss` | entries / union |
|---|---|---|---|---|
| run 1 (contended) | ~20 | 9.89 s | 1390.4 MiB | 151 / 125 |
| run 2 (heavily contended) | 25.48 | 18.59 s | 1390.5 MiB | 151 / 125 |
| **run 3 (quiet)** | **8.40** | **8.41 s** | **1390.4 MiB** | **151 / 125** |

**Run 3 is the reading.** Contention is one-sided — it can only make a call slower,
never faster — so under a shared box the *minimum* is the better estimator of the
true warm cost, and a run that arrived 2.2× slower on identical code with an
identical file set and an identical peak RSS is measuring the machine.

The last two columns are the control that makes this safe to say. Across a 2.2×
wall-clock spread the selected file set never moved (151 entries, 125 distinct,
every run) and peak RSS never moved by more than 0.1 MiB. A prose deletion cannot
change a wall-clock while holding both of those fixed; a busy scheduler does exactly
that.

---

## M1 — retrieval quality at the final base

```bash
$neo_worktree/.venv/bin/python $neo_worktree/tools/rank_mine_eval.py \
  --repo <repo> --tree $neo_worktree \
  --cases 50 --skip-recent 50 --k 1 3 10 --timeout 600 --json
```

50 mined cases per repo, git recency off (harness default), **0 failed cases in all
three runs**.

| repo | n | failed | **MRR** | R@1 | R@3 | **R@10** | H@10 |
|---|---|---|---|---|---|---|---|
| neo | 50 | 0 | **0.712** | 0.307 | 0.502 | **0.708** | 0.980 |
| aieweb | 50 | 0 | **0.728** | 0.487 | 0.654 | **0.778** | 0.880 |
| m365dotnet | 50 | 0 | **0.669** | 0.312 | 0.497 | **0.771** | 0.860 |

**Byte-identical to Goal 8's branch arm in every cell, on every reported k.** That
is the intended result and the strongest form the no-regression claim can take: not
"within noise", but the same number.

### Against the Goal 1 trailhead

The trailhead numbers were produced by a **different instrument**, and the plan's own
baselines doc says so at length (`docs/eval-baselines-2026-08.md`, "Instrument
note"): #194 repurposed `tools/rank_eval.py` back to its 12-prompt hand-labelled
form and moved the git-mined MRR harness to `tools/rank_mine_eval.py`, which shells
out to `neo --dry-run` instead of calling `score_candidate` in-process. Quoting one
under the other's name is the error that doc exists to prevent, so the two are not
placed in the same column here.

The like-for-like comparison exists and was made by #194 itself, which re-ran the
**pinned trailhead instrument** (`git show 5bbee46:tools/rank_eval.py`, same mined
case files, same 209 / 221 / 173 evaluable cases) on both arms:

| repo | cases | MRR trailhead | MRR after #194 | R@10 trailhead | R@10 after #194 |
|---|---|---|---|---|---|
| neo | 209 | 0.136 | **0.714** | 0.212 | **0.751** |
| aieweb | 221 | 0.180 | **0.759** | 0.244 | **0.774** |
| **m365dotnet** | 173 | **0.051** | **0.738** | 0.097 | **0.883** |

**The plan's M1 gate is met on the instrument the plan named.** M1's "Done" column
reads *m365dotnet MRR ≥ 0.6*; on `5bbee46:tools/rank_eval.py` m365dotnet went
**0.051 → 0.738**, a 14.5× move past a 0.60 target that the trailhead doc had called
"a 11.8× gap, not a tuning gap".

Read the two tables as answering two different questions. The trailhead table says
**the climb worked**. The final table says **Goals 5–9 did not spend the win** —
0.712 / 0.728 / 0.669, unchanged since Goal 6 introduced the instrument.

### The second half of M1: no flagship regressed goal-over-goal

| repo | Goal 6 | Goal 7 | Goal 8 | **Goal 9** |
|---|---|---|---|---|
| neo MRR | 0.712 | 0.712 | 0.712 | **0.712** |
| aieweb MRR | 0.728 | 0.728 | 0.728 | **0.728** |
| m365dotnet MRR | 0.669 | 0.906 † | 0.669 | **0.669** |

† Goal 7 ran m365dotnet on **8** cases, not 50, and its own doc says the absolute
number is not comparable with the 50-case figure. It is in the row for completeness,
not as a reading. Goals 6, 8 and 9 are the comparable series, and they are equal.

---

## M2 — warm-call cost on m365dotnet

```bash
$neo_worktree/tools/m2_battery.sh \
  $neo_worktree/evidence/bin/neo-g9 $platform_root/m365dotnet $neo_worktree/evidence/m2_g9
```

The canonical 6-prompt battery, 3 timed runs per prompt after a discarded warm-up,
observer autostart disabled. Figures below are **run 3, the quiet one**; the two
contended runs are committed as `evidence/g9_m2_contended.txt` and
`evidence/g9_m2_run2.txt`.

| | Goal 1 trailhead | Goal 8 | **Goal 9** | target |
|---|---|---|---|---|
| battery median wall | 10.54 s | 8.97 s | **8.41 s** | ≤ 5 s |
| battery peak `ru_maxrss` | 1.43 GB | 1390.5 MiB | **1390.4 MiB** | ≤ 500 MB |
| entries / union distinct | — | 151 / 125 | **151 / 125** | — |
| within-prompt repeat entries (#197) | — ‡ | 0 | **0** | 0 |

‡ Not measured at the trailhead; the battery did not yet report the two file columns
separately. The number this row usually gets quoted against is **45**, which is Goal
8's `main` arm (`d5adcbc`, post-#207) — i.e. #207 closed the *count* half of #197 and
#214's whole-file delivery closed the *repeat* half. Both are zero here.

| id | shape | Goal 8 median | **Goal 9 median** | Goal 8 peak MiB | Goal 9 peak MiB |
|---|---|---|---|---|---|
| P1 | file-named | 10.50 | **8.40** | 1390.5 | 1390.4 |
| P2 | file-named | 9.49 | **8.89** | 1387.5 | 1387.3 |
| P3 | concept-only | 7.90 | **8.66** | 1384.5 | 1384.1 |
| P4 | concept-only | 9.02 | **8.28** | 1384.6 | 1381.6 |
| P5 | mixed | 8.92 | **7.28** | 1385.0 | 1385.1 |
| P6 | symptom | 9.14 | 23.81 ᵖ | 1385.2 | 1384.2 |

ᵖ P6's three runs were 7.27 / 23.81 / 54.16 s — monotonically escalating, with the
1-minute load average rising from 8.40 to 18.36 across the battery as another
session started work. Its first run is the in-family reading; the median is not.
Five of six prompts are **faster** than Goal 8 and the battery median is 6% below
it, which is the M2 requirement for this goal: **retirement does not re-add cost.**

The RSS column is the control: six prompts, six readings, every one within 3 MiB of
Goal 8's, and unchanged across a 2.2× wall-clock spread between batteries.

**M2's two absolute targets are not met and were never going to be met by this
goal.** The 1.39 GB floor is the memory system, not file selection — issue #211,
explicitly out of the plan's scope — and the profile below is where that is visible
rather than asserted.

### P1 delivers one file, and that is real, not noise

P1 names `src/Parslee.M365.Api/Program.cs`, which is 442,867 bytes. Stage 1 pins it,
the pin consumes 299,959 of the 300,000-byte `--max-bytes` default, and the scan is
left with 41 bytes — so **P1 delivers 1 file where pre-#214 main delivered 22**. The
battery's `entries`/`distinct_files` columns read `1, 1` on all three of P1's runs,
in every one of the three batteries.

This is merged behaviour introduced by #214, **not a Goal 9 regression** — Goal 8's
own branch arm produced the same 151-entry / 125-union totals, which is only
arithmetically possible if its P1 also delivered one file. It reads against standing
ruling 1 ("guarantee the named files **AND** keep scanning"): funding pins to the
last byte satisfies the first clause by deleting the second. The Goal 8 author
identified it in their own committed evidence after the merge and has a fix
(`PIN_BUDGET_SHARE`, capping the pin block at half of `--max-bytes` while the scan
still has candidates) on a branch, with a follow-up PR against post-#214 main. It is
recorded as an open exit in the plan ledger.

Two consequences for reading this document: P1's wall-clock is doing roughly a
thirtieth of the delivery work a pre-#214 P1 did, so P1-versus-old-P1 is not
like-for-like in either direction; and the Goal 8 ↔ Goal 9 comparison above *is*
like-for-like, because both sit on the same behaviour.

### Where the warm call goes

```bash
$neo_worktree/.venv/bin/python $neo_worktree/evidence/g9_stage_profile.py \
  $platform_root/m365dotnet
```

Timers wrapped around the stage functions of one real `neo --dry-run`, after a
discarded warm-up. No profiler, so the parts sum to a wall-clock a stopwatch would
agree with. (Goal 7's equivalent lived in `/tmp` and could not be re-run by anyone
reading its numbers; this one is committed beside them.)

Run five times across a load range of 6.1 to 20; the reading below is the
lowest-load run, for the same one-sided reason the battery quotes its quiet run.
The five WALL TOTALs were 6.47 / 7.09 / 7.12 / 9.05 / 35.52 s while peak RSS held at
1447 MB in **all five** — the memory figure is load-independent and the time figure
is not.

```
T imports                                    0.96s             <- 1.24s at Goal 7
T eligibility walk                           0.12s  (1 call)   <- 0.16s
T content index refresh                      0.08s  (1 call)   <- 0.11s
T content index scores                       0.09s  (1 call)   <- 0.12s
T project index boost                        0.08s  (1 call)   <- 0.10s
T fact store history boost                   1.94s  (1 call)   <- 2.25s
T fact store retrieve (memory layer)         1.49s  (1 call)   <- 1.65s
T GATHER TOTAL                               3.08s  (1 call)   <- 3.55s
T WALL TOTAL                                 6.47s   peak rss = 1447 MB   <- 7.67s / 1385 MB
T   selection only (GATHER minus history)    1.14s
T   wall excluding the memory system         3.04s             <- 3.77s
```

**File selection is 0.37 s of a 6.47 s call** — the eligibility walk, the content
index refresh, the content index scores and the project-index boost together, on a
9,348-file repository. The two fact-store rows are 3.43 s of it and carry the whole
1.39 GB.

**Excluding the memory system the warm call is 3.04 s, under the 5 s M2 target;
including it, 6.47 s.** Neither number is presented as the other. That is the same
split Goal 7 reported (3.77 s / 7.67 s), and it is the honest closing statement on
M2: *the unified store met its half of the target and the memory system owns the
rest.* Every stage the plan set out to fix is at or below its Goal 7 figure; the two
rows that are not file selection are the two rows issue #211 is about.

---

## G-invariants at the final base

```bash
cd $platform_root/m365dotnet
git check-ignore --stdin < $neo_worktree/evidence/m2_g9/union.files
```

| invariant | reading | verdict |
|---|---|---|
| **G1-inv** gitignored files selected | **0 / 125** | clean |
| **G1-inv** byte-identical duplicate copies | **0** | clean |
| **G1-inv** union paths absent from disk | **0** | clean |
| **G2-inv** prompt-named / `--include` files present | pinned by `tests/test_include_guarantee.py` (64 tests) | green |
| **G3-inv** no silent caps | pinned by the invariant battery | green |
| **G4-inv** one eligibility implementation | `tests/test_eligibility_single_source.py` + differential | green |
| **G5-inv** per-language LLM round trip | `tests/test_release_roundtrip.py`, runs in the release gate | green at v0.45.0 |

## Structural proof — the lane is gone, not just unused

New this goal: `tests/test_lane_retirement.py` (16 tests, `-m invariants`).

| claim | how it is proved |
|---|---|
| one eligibility implementation | `test_eligibility_single_source.py` (unchanged, Goal 5) |
| one gather path | `gather_context` defined once; no sibling `gather_*` at module level; `cli.py` calls that name and no other |
| `--semantic` never routes | AST scan: `cli.py` contains no `if`/`IfExp` whose test reads `.semantic` |
| zero references to deleted lane functions | text scan of `src/`, `.claude-plugin/`, `plugins/`, `README.md`, `QUICKSTART.md`, `INSTALL.md`, `AGENTS.md`, `docs/**` for `gather_context_semantic`, `mmr_pack_chunks`, `log_context_metrics` |
| the docs do not teach the old model | four named retired claims appear in no shipped source or doc |

Every one of the retired claims was live text in this repository at the start of this
goal, and each is verified to fail against `origin/main` — a guard that would have
passed before the change it guards is not a guard.

**Two exemptions, both named in the test's docstring.** Dated measurement records
under `docs/` (`goal<N>-*-measurements-<date>.md`, including this file) are evidence
of what was true on the day they were produced: the Goal 8 record compares an arm
whose code *did* contain `gather_context_semantic` against one that did not, and
editing that sentence to satisfy a grep would make the record lie about the
experiment. `tests/` is exempt from the same guard for the same reason in miniature —
a test asserting a symbol's absence has to be allowed to spell it.
