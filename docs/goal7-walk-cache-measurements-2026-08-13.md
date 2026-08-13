# Goal 7 — auto-freshness / persistent eligibility walk: measurements

**Measured:** 2026-08-13
**Goal:** Unified Store Plan, Goal 7 (Auto-freshness) — see `docs/unified-store-plan.md`
**Closes:** #210
**Arms:** `main` = `9b0c16d63cfc` (post-#208) · `base` = `260e56c` (PR #209 head, the
tree this branch is built on) · `branch` = `cc7d98c1eeb9`
**Companions:** `docs/goal6-content-index-measurements-2026-08-12.md` (Goal 6) and
`docs/eval-baselines-2026-08.md` (Goal 1), whose M2 battery and mining parameters are
canonical and unchanged here.

Every number below was produced by running the command printed beside it. Nothing is
estimated, projected or interpolated. `<platform-root>` is the checkout holding the
child repos.

**Three arms rather than two, because #209 is open and unmerged.** `base` is the tree
this branch was cut from, so `base → branch` is the delta this goal owns; `main` is
carried so the absolute number is comparable with the Goal 1 and Goal 6 tables.

---

## Headline

| Metric | main | base (#209) | branch | Target | Verdict |
|---|---|---|---|---|---|
| **M2** warm median wall, m365dotnet | 52.20 s | 15.63 s | **9.47 s** | ≤ 5 s | **not met including the memory system; met excluding it** — see the profile |
| **M2** peak RSS, m365dotnet | 1.98 GB | 1.45 GB | **1.45 GB** | ≤ 500 MB | not met — 1.26 GB of it is the FactStore (#211) |
| **eligibility walk**, warm, m365dotnet | 4.6–6.9 s | 4.6–6.9 s | **0.16 s** | — | the item this goal owns |
| **M3** walk + index refresh, 10 files edited | n/a | 0.79 s | **0.78 s** | ≤ 5 s | **met** |
| **M3** walk + index refresh, 10 files added | n/a | n/a | **1.05 s** | ≤ 5 s | **met** |
| **M1** MRR, three flagships | 0.712 / 0.728 / 0.906 | — | **identical** | no regression | **met** |
| **G1-inv** ignored / duplicate selections | 0 / 0 | 0 / 0 | **0 / 0** | 0 | **met** |
| cold walk, first call in a repository | n/a | n/a | 0.36 / 0.96 / **5.3 s** | bounded + announced | **met** |

---

## Why a cache of *verdicts* and not of anything else

The design follows from one measurement, taken before any of this existed. The walk's
cost is the ignore-pattern matching; the filesystem is nearly free.

```bash
<branch>/.venv/bin/python /tmp/g7_profile_walk.py <platform-root>/m365dotnet
```

| arm | time | what it did |
|---|---|---|
| raw `os.walk`, no ignore logic, no pruning | 10.394 s | 30,839 dirs / 307,115 files |
| `os.walk` + directory pruning only, no per-file test | **0.801 s** | 951 directories visited |
| full `eligibility.walk` as shipped | **6.853 s** | 9,378 admitted, 109 dirs + 763 files excluded |
| `os.stat` over exactly the 9,378 admitted files | **0.102 s** | — |

The second and third rows do the same traversal; the only difference is testing each
FILE entry against the pattern set (11,219 `should_ignore` calls, counted by
`cProfile` on the same run). That is **6.05 s of a 6.85 s walk**, and the `stat` a
cache cannot avoid is 0.102 s. So a cache of directory listings would have saved
almost nothing, and a cache of verdicts saves almost everything.

It also settles what must NOT be cached: file sizes and mtimes cost 0.1 s to read
fresh and are the content index's own freshness stamp, so they are read on every walk.

---

## M2 — warm call on m365dotnet

```bash
NEO_OBSERVER_AUTOSTART=0 RUNS=3 tools/m2_battery.sh \
  <arm>/.venv/bin/neo <platform-root>/m365dotnet /tmp/g7_m2_<arm>
```

| id | shape | main | base (#209) | **branch** |
|---|---|---|---|---|
| P1 | file-named | 58.25 s | 15.91 s | **10.31 s** |
| P2 | file-named | 50.95 s | 15.63 s | **10.05 s** |
| P3 | concept-only | 49.58 s | 15.23 s | **9.35 s** |
| P4 | concept-only | 51.45 s | 15.53 s | **9.73 s** |
| P5 | mixed | 53.14 s | 15.60 s | **8.73 s** |
| P6 | symptom | 52.96 s | 16.17 s | **9.40 s** |
| **BATTERY median** | | **52.20 s** | **15.63 s** | **9.47 s** |
| **BATTERY peak RSS** | | 1893.0 MiB | 1386.8 MiB | 1385.2 MiB |

**Disclosed measurement condition:** the `main` arm ran at `RUNS=1` (n=1 per prompt,
so min = median = max), the other two at the canonical `RUNS=3`. A 3-run `main` arm is
~16 minutes of wall clock and this harness caps a single foreground command at 10
minutes. The n=1 number is consistent with Goal 6's independently measured 53.47 s
median for the same arm on the same machine and repository, which is why it is quoted
rather than dropped — but it is one sample and should be read as one.

`base` → `branch` is **6.16 s of median wall removed**, which is the eligibility walk
and nothing else.

### Where the remaining 9.47 s goes

Real CLI invocation with timers wrapped around the stage functions — no profiler, so
the parts sum to a wall-clock a stopwatch would agree with:

```bash
<branch>/.venv/bin/python /tmp/g7_cli_stage_profile.py <platform-root>/m365dotnet
```

```
T imports                                  1.43s
T eligibility walk                         0.14s  (1 call)   <- 4.64s in #209's profile
T content index refresh                    0.12s  (1 call)
T content index scores                     0.11s  (1 call)
T project index boost                      0.07s  (1 call)
T fact store history boost                 2.30s  (1 call)
T fact store retrieve (memory layer)       1.79s  (1 call)
T   of which fastembed model load          3.22s  (2 calls, inside the two above)
T GATHER TOTAL                             3.57s  (1 call)
T WALL TOTAL                               8.52s   peak rss = 1381 MB
```

**File selection is 0.44 s of it.** The walk, the content index and the project index
together cost less than half a second on a 9,348-file repository.

**Against the ≤ 5 s target, exactly as the goal brief asks it to be reported.**
Excluding the memory system — `_history_boost` (2.30 s) and the engine's own fact
retrieval (1.79 s), which between them carry both fastembed model loads and all
1.26 GB of the RSS — the warm call is **4.43 s, under the 5 s target**. Including
them it is 8.52 s. Neither number is presented as the other. The memory system is
issue #211 and explicitly out of this goal's scope; nothing here touches it.

(The instrumented run's 8.52 s and the battery's 9.47 s median differ by process
start-up and printing the assembled prompt, which the battery pays and the profile
does not.)

---

## M3 — freshness cost, and staleness correctness, on the live repository

One script, three scenarios, each timing the FULL pipeline an invocation pays — the
eligibility walk plus the content index refresh — and then asserting the answer
actually changed. A unique token is written into the touched files and searched for
through the index; the same search after the restore must find nothing.

```bash
NEO_OBSERVER_AUTOSTART=0 <branch>/.venv/bin/python /tmp/g7_m3.py \
  <platform-root>/m365dotnet
```

```
baseline (settled)     walk= 0.18s [warm, 0/951 re-listed]  refresh= 0.13s [warm]                  TOTAL= 0.31s  files=9348
A: 10 files edited     walk= 0.18s [warm, 0/951 re-listed]  refresh= 0.60s [incremental, chg=10]   TOTAL= 0.78s  files=9348
   -> marker findable in 10 file(s)
A: restored            walk= 0.17s [warm, 0/951 re-listed]  refresh= 0.52s [incremental, chg=10]   TOTAL= 0.68s  files=9348
   -> marker findable in 0 file(s)
B: 10 files added      walk= 0.63s [incremental, 1/951]     refresh= 0.42s [incremental, add=10]   TOTAL= 1.05s  files=9358
   -> marker findable in 10 file(s): src/NeoGoal7Probe0.cs …
B: removed again       walk= 0.24s [incremental, 1/951]     refresh= 0.42s [incremental, rm=10]    TOTAL= 0.66s  files=9348
   -> marker findable in 0 file(s)
C: one file gitignored walk= 6.17s [rebuilt]   files=9347   still eligible: False
C: gitignore restored  walk= 5.29s [rebuilt]   files=9348   back: True

git status --porcelain identical before/after: True
```

Read the three scenarios as the three shapes of change, because they exercise three
different mechanisms:

- **A — an edit does not move any directory's mtime**, so the walk stays warm and
  re-lists nothing, and the file's new size and mtime are still read fresh. That is
  the cross-cache invariant: the content index sees the edit (`changed=10`) *through*
  a fully reused set of directory listings. Had the stamps been cached, this line
  would read `warm, 0 changed` and the index would answer with text the file no
  longer holds.
- **B — an add or a delete moves exactly one directory's mtime**, so one of 951
  directories is re-listed and the other 950 are not.
- **C — a `.gitignore` edit moves no mtime at all.** Every stored verdict looks
  current and every one of them may now be wrong, so the signature (a hash of the
  effective pattern list) discards the cache and the walk runs in full: 6.17 s once,
  then warm again. The newly-ignored file leaves the corpus on the next call and
  comes back when the pattern is removed.

**Both M3 shapes are met: 0.78 s and 1.05 s against ≤ 5 s.** The live tree was
restored byte-for-byte; `git status --porcelain` is identical before and after.

---

## M1 — retrieval quality: identical, on all three flagships

Persisting the walk must not re-rank, and it does not.

```bash
<branch>/.venv/bin/python <branch>/tools/rank_mine_eval.py \
  --repo <platform-root>/$repo --tree <arm> --json
```

| repo | arm | cases | MRR | R@1 | R@3 | R@10 | H@10 |
|---|---|---|---|---|---|---|---|
| neo | main | 50 | 0.712 | 0.307 | 0.502 | 0.708 | 0.980 |
| neo | **branch** | 50 | **0.712** | **0.307** | **0.502** | **0.708** | **0.980** |
| aieweb | main | 50 | 0.728 | 0.487 | 0.654 | 0.778 | 0.880 |
| aieweb | **branch** | 50 | **0.728** | **0.487** | **0.654** | **0.778** | **0.880** |
| m365dotnet | main | 8 | 0.906 | 0.583 | 0.729 | 0.958 | 1.000 |
| m365dotnet | **branch** | 8 | **0.906** | **0.583** | **0.729** | **0.958** | **1.000** |

Byte-identical in every cell. Repo HEADs at measurement: neo `5bbee46747c8`, aieweb
`26fff07e0f4a`, m365dotnet `61dc4a171bdf`. Zero failed cases on every arm, `--no-git`
(the default).

**Disclosed: m365dotnet ran 8 cases, not 50.** Its `main` arm costs ~52 s per case, so
50 cases is ~40 minutes in one foreground command against this harness's 10-minute cap.
The absolute MRR is therefore not comparable with Goal 6's 50-case 0.669 for that repo
— only the two arms of the same 8 cases are comparable with each other, which is what
this table claims. The stronger m365dotnet statement is the battery one below.

**Stronger than the MRR table, and the one to quote:** all six canonical M2 prompts
selected **the same files in the same rank order** on all three arms, compared as
text — 18 comparisons, no output.

```bash
for p in P1 P2 P3 P4 P5 P6; do
  diff /tmp/g7_m2_base/${p}_run1.files /tmp/g7_m2_branch/${p}_run1.files
done
diff /tmp/g7_m2_main/union.files /tmp/g7_m2_base/union.files
diff /tmp/g7_m2_base/union.files /tmp/g7_m2_branch/union.files
```

---

## G1-inv — the walker's verdict is not widened by caching it

The battery union, checked against `git check-ignore` in the target repository.

| label | distinct | `git check-ignore` excludes | duplicate content copies | inside `.worktrees/` | unresolvable |
|---|---|---|---|---|---|
| **BATTERY UNION**, main | 111 | 0 | 0 | 0 | 0 |
| **BATTERY UNION**, base | 111 | 0 | 0 | 0 | 0 |
| **BATTERY UNION**, branch | **111** | **0** | **0** | **0** | **0** |

The three union files are byte-identical, which is the point: a cache of the walker's
verdicts cannot widen them.

---

## Cold cost — what a fresh clone's first call pays, once

```bash
<branch>/.venv/bin/python -c "…cached_walk three times, timing each…" <repo>
```

| repo | files | directories | uncached walk | first call (cold) | second call | cache on disk |
|---|---|---|---|---|---|---|
| neo | 285 | 43 | 0.41 s | 0.36 s | **0.01 s** (warm) | 15 KB |
| aieweb | 902 | 221 | 1.05 s | 0.96 s | **0.06 s** (warm) | 64 KB |
| m365dotnet | 9,348 | 951 | 6.85 s | 5.3 s | **0.16 s** (warm) | 507 KB |

The first call in a repository announces the walk before it starts, for the same
reason the content index announces its cold build: seconds of silence are
indistinguishable from a hang. `neo --index` warms this cache as well as the semantic
catalog, so it remains an optional way to pay the cost deliberately — but it is
optional, because any invocation pays it once and every later one is warm.

The warm 0.16 s on m365dotnet is three things and nothing else: 951 directory `stat`s,
9,378 file `stat`s (0.102 s measured on its own, above) and a 507 KB JSON parse. No
directory is listed and no path is matched against a pattern.

---

## What is NOT claimed

- **The 500 MB M2 target is still not met and is still not reachable from file
  selection.** 1.26 GB of the 1.38 GB peak arrives in `_history_boost`, loading the
  FactStore. Goal 6 established this; nothing here changes it. Issue #211.
- **The ≤ 5 s M2 target is met only excluding the memory system.** Stated in both
  forms above rather than in the flattering one.
- **The `main` arm of M2 is n=1 and the m365dotnet M1 arms are 8 cases**, both for the
  same reason (a 10-minute foreground cap), both disclosed at the table rather than in
  a footnote.
- **A `.gitignore` edit costs a full walk.** Measured at 6.17 s on m365dotnet. The
  alternative — deriving which directories a pattern edit could have affected — is a
  large amount of machinery guarding a cost paid when someone edits a `.gitignore`.
- **Two clones of one repository do not share a cache.** It lives in each working
  tree's own `.neo/`, because it is keyed on that tree's directory mtimes.
