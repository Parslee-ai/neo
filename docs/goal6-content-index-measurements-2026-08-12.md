# Goal 6 — persistent content index: measurements

**Measured:** 2026-08-12
**Goal:** Unified Store Plan, Goal 6 (Persistent content index) — see `docs/unified-store-plan.md`
**Closes:** #195
**Arms:** `main` = `9b0c16d63cfc` (post-#208) · `branch` = `df8f670ebded`
**Companion:** `docs/eval-baselines-2026-08.md` (Goal 1), whose M2 battery and mining
parameters are canonical and unchanged here.

Every number below was produced by running the command printed beside it. Nothing is
estimated, projected, or interpolated. Two placeholders, as in the baselines doc:
`<platform-root>` is the checkout holding the child repos, `<branch>` / `<main>` are
the two neo worktrees.

---

## Headline

| Metric | main | branch | Target | Verdict |
|---|---|---|---|---|
| **M2** warm median wall, m365dotnet | 53.47 s | **18.50 s** | ≤ 5 s | **not met** (2.9× better, 3.7× over) |
| **M2** peak RSS, m365dotnet | 1.95 GB | **1.45 GB** | ≤ 500 MB | **not met** — see *Where M2 actually goes* |
| **M2** cold build, m365dotnet | n/a | 122.1 s / 1.47 GB | bounded + reported | **met** |
| **M3** re-index after 10 edited files | n/a (full rebuild) | **0.79 s** index work | ≤ 5 s | **met** |
| **M1** MRR, three flagships | 0.712 / 0.728 / 0.669 | **identical** | no regression | **met** |
| **G1-inv** ignored / duplicate selections | 0 / 0 | 0 / 0 | 0 | **met** |

Read the two M2 rows together with the profile below before treating them as a Goal 6
shortfall. The cost this goal owns — re-reading and re-tokenizing the corpus on every
call — is gone: it is **0.5 s of an 18.5 s** warm call. What remains belongs to two
other components, and one of them makes the 500 MB target unreachable from file
selection at all.

---

## M1 — retrieval quality: identical, on all three flagships

Persistence must not re-rank, and it does not. Same harness, same repos, same mined
cases, 50 per repo, zero failed cases, `--no-git` (the default).

| repo | arm | cases | MRR | R@1 | R@3 | R@10 | H@10 |
|---|---|---|---|---|---|---|---|
| neo | main | 50 | 0.712 | 0.307 | 0.502 | 0.708 | 0.980 |
| neo | **branch** | 50 | **0.712** | **0.307** | **0.502** | **0.708** | **0.980** |
| aieweb | main | 50 | 0.728 | 0.487 | 0.654 | 0.778 | 0.880 |
| aieweb | **branch** | 50 | **0.728** | **0.487** | **0.654** | **0.778** | **0.880** |
| m365dotnet | main | 50 | 0.669 | 0.312 | 0.497 | 0.771 | 0.860 |
| m365dotnet | **branch** | 50 | **0.669** | **0.312** | **0.497** | **0.771** | **0.860** |

Byte-identical in every cell. Repo HEADs at measurement: neo `5bbee46747c8`,
aieweb `26fff07e0f4a`, m365dotnet `61dc4a171bdf`.

**Measurement condition, disclosed because it happened mid-run.** m365dotnet is a live
working tree on a shared machine, and an unrelated session ran `git reset` in it at
20:58:59 (`git reflog --date=iso`) — between the m365dotnet `main` arm (20:53–21:27)
and the `branch` arm (21:27–21:42). That reverted one modified tracked file and removed
two untracked paths, so the two arms saw corpora differing by a handful of files. HEAD
did not move. The arms still agree in every cell, which says the perturbation was
immaterial to these 50 cases — it is not evidence that it could never matter, and a
re-measure wanting a stricter guarantee should pin the tree. The neo and aieweb arms
ran either side of no such event.

```bash
for repo in neo aieweb m365dotnet; do
  for arm in main branch; do
    tree=<main>; [ "$arm" = branch ] && tree=<branch>
    <branch>/.venv/bin/python <branch>/tools/rank_mine_eval.py \
      --repo <platform-root>/$repo --tree "$tree" --json
  done
done
```

A stronger statement is available on the M2 battery, and it is the one to quote: the
six canonical prompts on m365dotnet selected **the same files in the same rank order**
on both arms, compared as text.

```bash
for p in P1 P2 P3 P4 P5 P6; do
  diff /tmp/g6_m2_main/${p}_run1.files /tmp/g6_m2_branch/${p}_run1.files
done   # no output: identical selection AND order, all six
```

**Staleness, the check the plan asks for.** Ten `.cs` files in m365dotnet were each
given a unique marker, the index was refreshed, and the marker was searched for:

```
S3 files matching the unique edit token: 1 -> ['src/Ink.Core/Editing/ApplyEditPlanResult.cs']
```

New content is findable and attributed to exactly the file that now holds it. The
converse — old content ceasing to be findable — is pinned by
`test_an_edit_is_reflected_with_no_stale_hits`, which asserts both halves, because an
index that only ever ADDS postings passes the first check while answering with text
that is no longer in the file. All ten files were restored from a byte copy afterwards
and `git status --porcelain` on m365dotnet is identical before and after (5
pre-existing dirty entries, untouched).

---

## M2 — warm call cost on m365dotnet

`tools/m2_battery.sh`, unchanged, six canonical prompts, one discarded warm-up then
`RUNS=3` timed runs per prompt; wall-clock is the median, RSS is the maximum observed.

```bash
RUNS=3 tools/m2_battery.sh /tmp/g6bin/neo-<arm> <platform-root>/m365dotnet /tmp/g6_m2_<arm>
```

| id | shape | main median | **branch median** | main peak RSS | **branch peak RSS** |
|---|---|---|---|---|---|
| P1 | file-named | 60.25 s | **17.08 s** | 1615.6 MiB | **1384.7 MiB** |
| P2 | file-named | 54.25 s | **18.45 s** | 1686.2 MiB | **1386.4 MiB** |
| P3 | concept-only | 52.49 s | **19.00 s** | 1801.5 MiB | **1383.3 MiB** |
| P4 | concept-only | 53.52 s | **18.09 s** | 1698.0 MiB | **1383.6 MiB** |
| P5 | mixed | 52.46 s | **18.79 s** | 1851.3 MiB | **1384.2 MiB** |
| P6 | symptom | 39.55 s | **19.27 s** | 1856.8 MiB | **1384.2 MiB** |
| **BATTERY** | | **53.47 s** | **18.50 s** | **1856.8 MiB (1.95 GB)** | **1386.4 MiB (1.45 GB)** |

Selection identical on both arms: 180 context entries, 111 battery-union distinct
files, 69 battery-wide repeats, 45 within-prompt.

### Cold build, reported separately

First invocation in a repository with no store, timed with `/usr/bin/time -l`:

| repo | eligible files | cold build | peak RSS | store on disk |
|---|---|---|---|---|
| m365dotnet | 9,353 | **122.14 s** | 1.47 GB | 109 MB |
| neo | 307 | **2.3 s** | — | 5 MB |

It announces itself before it starts and reports progress every 250 files, so the
minutes are visible rather than silent:

```
[Neo] Content index: no usable index for this repository - building one over 9353 files.
      This runs once; later calls update only what changed.
[Neo] Content index: 250/9353 files tokenized
...
[Neo] Content index: cold build of 9353 files (first run for this repository) in 122.1s
```

### Where M2 actually goes

Warm profile, m365dotnet, one process, `ru_maxrss` sampled per stage:

```
T imports                      +  1.16s  cum   1.16s  rss=    82MB
T eligibility walk (9348)      +  4.64s  cum   5.80s  rss=    87MB
T content index refresh (warm) +  0.41s  cum   6.22s  rss=   127MB
T content index scores (7266)  +  0.12s  cum   6.33s  rss=   127MB
T project index boost          +  0.12s  cum   6.46s  rss=   140MB
T fact store history boost     +  2.94s  cum   9.40s  rss=  1404MB
```

Three readings, stated plainly:

1. **The cost Goal 6 owns is gone.** Content indexing is 0.53 s of a warm call, down
   from the ~35–45 s it contributed to main's 53.47 s median. It is no longer the
   dominant term, or close to it.
2. **The 500 MB RSS target is not reachable from file selection.** 1.26 GB of the
   1.4 GB arrives in a single step — `_history_boost` loading the FactStore, which on
   this machine is 152 MB of JSON carrying 768-dim embeddings that inflate to ~1.3 GB
   of Python objects. Goal 1's 1.43 GB M2 baseline was therefore never the gatherer's
   memory; it was the memory system's, measured through the gatherer. No change to
   how files are chosen can move it, and a Goal 6 that "hit 500 MB" would have done so
   by deleting a retrieval channel, not by indexing better.
3. **The next wall-clock item is the eligibility walk**, at 4.6 s for 9,348 admitted
   files out of ~345k filesystem entries — 97 ignore patterns evaluated per path
   component, CPU-bound and stable across repeats (5.88 / 6.91 / 6.63 s in three
   consecutive in-process calls). That is #208's shared walker, untouched here, and it
   is the largest remaining component of a warm call.

Neither (2) nor (3) is a defect introduced by this branch, and neither is in Goal 6's
scope. They are recorded here because the plan assigns M2's ≤5 s / ≤500 MB to this
goal, and the profile says which later goal can actually deliver each half.

---

## M3 — freshness cost

Ten `.cs` files in m365dotnet edited, then one refresh:

```
M3 walk=4.42s  refresh=0.79s  mode=incremental changed=10 added=0 removed=0 indexed=10 total=9353
```

**0.79 s of index work against a ≤5 s target — met.** Stated with its neighbour so the
number is not read as more than it is: the walk that precedes it costs 4.42 s, so
walk + refresh is 5.21 s. The walk is not re-indexing work, it is the eligibility scan
every invocation pays regardless of what changed (see reading 3 above), and Goal 6
neither added it nor can remove it.

---

## G1-invariant on m365dotnet

Computed over the full battery on **both** arms, with the script from
`docs/eval-baselines-2026-08.md` pointed at each arm's output directory.

| label | distinct | `git check-ignore` excludes | duplicate copies | inside `.worktrees/` | unresolvable |
|---|---|---|---|---|---|
| P1 | 22 | 0 | 0 | 0 | 0 |
| P2 | 23 | 0 | 0 | 0 | 0 |
| P3 | 24 | 0 | 0 | 0 | 0 |
| P4 | 22 | 0 | 0 | 0 | 0 |
| P5 | 24 | 0 | 0 | 0 | 0 |
| P6 | 20 | 0 | 0 | 0 | 0 |
| **BATTERY UNION** | **111** | **0** | **0** | **0** | **0** |

Identical on main and branch — which is the point: the index consumes the walker's
verdict and cannot widen it.

---

## What this document does not contain

| Missing | Reason |
|---|---|
| A `dense` / RRF lane comparison | Unbaselined on the flagships since Goal 1 and unchanged by this goal; see that document's own note. |
| M2 on neo or aieweb | The battery's six prompts are m365dotnet-specific by construction (they name m365dotnet paths). The plan measures M2 there. |
| A distinct count for main's cold build | main has no persistent store; its per-call rebuild IS the 53.47 s median, and quoting it as a separate "cold" number would double-count. |
| Concurrency measurements | Two Neo processes in one repository are handled (WAL, busy timeout, memory fallback) and unit-tested, but the contended path is not timed here. |
