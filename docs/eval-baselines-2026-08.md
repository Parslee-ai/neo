# Neo flagship baselines — August 2026

**Measured:** 2026-08-11
**Goal:** Unified Store Plan, Goal 1 (Trailhead) — see `docs/unified-store-plan.md`
**Neo under test:** v0.44.0 at `main` = `77251dff54d0726fb7d98e6e6daa525e1b8b2c6d`

These are the M1 / M2 / G1-invariant baselines every later goal in the unified-store
plan re-measures against. The mining parameters and the M2 prompt battery below are
**canonical**: change them and the goal-over-goal comparison is void.

Every number here was produced by running the command printed beside it. Where a
number could not be produced, the row says so and gives the blocking reason —
nothing in this document is estimated, projected, or interpolated.

**Two placeholders appear in the commands below**, deliberately, so the doc is not
pinned to one developer's home directory. `<platform-root>` is the `parslee-knowledge`
checkout that holds the child repos as subdirectories (`<platform-root>/m365dotnet`,
`<platform-root>/aieweb`, `<platform-root>/neo`); `<neo-worktree>` is the neo worktree
this document was produced from. Set them once before pasting anything:

```bash
export platform_root=/path/to/parslee-knowledge     # substitute for <platform-root>
export neo_worktree=$platform_root/neo/.worktrees/g_msp4yrhu_d974cb-unified-store-goal-1
```

---

## Headline

| Metric | neo | aieweb | m365dotnet | Target |
|---|---|---|---|---|
| **M1** MRR (`neo_current`) | 0.136 | 0.180 | **0.051** | m365dotnet ≥ 0.60 |
| **M1** recall@10 (`neo_current`) | 0.212 | 0.244 | **0.097** | — |
| **M2** wall-clock, median | — | — | **10.54 s** | ≤ 5 s |
| **M2** peak RSS (`ru_maxrss`) | — | — | **1.43 GB** | ≤ 500 MB |
| **G1-inv** gitignored files selected | — | — | **0 / 141** | 0 |
| **G1-inv** duplicate copies selected | — | — | **0 / 141** | 0 |

141 is the **union** of distinct files the selector put in context across all six
battery prompts (180 context entries in total). Count the union, never the sum of
per-prompt counts — see the G1 section.

Three readings, stated plainly:

1. **Retrieval on the largest flagship is the weakest.** m365dotnet's shipped ranker
   puts a correct file in the top 10 for roughly one query in ten (recall@10 = 0.097)
   and its MRR is 0.051 against a 0.60 target — a **11.8×** gap, not a tuning gap.
2. **The gap is not "retrieval is hard here."** On the same eval set, on the same
   machine, in the same process, plain BM25 over file content scores MRR 0.607 and
   recall@10 0.813. The signal is present in the repo; the shipped scorer is not
   using it.
3. **G1 is clean post-#186.** The 2026-08-10 reality was 14/16 ignored and 12/16
   duplicates. Measured now across the whole battery: 0 and 0. This is measured,
   not assumed.

---

## What was measured, and at which SHA

The goal asked for each repo at `origin/main`. Two of the three could not be moved
there: `m365dotnet` and `aieweb` both hold uncommitted in-flight work, and this goal
is forbidden from writing to either. Checking them out to `origin/main` is a write.
They were therefore measured **in place at their exact current HEAD**, recorded here
so any later goal can re-measure at the identical tree.

| Repo | Language | Measured at HEAD | Branch | Behind `origin/main` by | Working tree at measurement |
|---|---|---|---|---|---|
| neo | Python | `77251dff54d0726fb7d98e6e6daa525e1b8b2c6d` | `main` | 0 | clean (2 untracked: `.worktrees/`, `uv.lock`) |
| aieweb | TypeScript | `26fff07e0f4a1ec16ca7c9a8446e2bc2d2a4d406` | `feature/platform-nav-customer-tabs-core-rename` | 85 | 39 modified/staged, 14 untracked |
| m365dotnet | C# | `61dc4a171bdf16cdeb15123dd8790a19248175f9` | `main` | 166 | 1 modified, 3 untracked |

`origin/main` at fetch time, for the record: neo `77251df`, aieweb
`aa60b64a313b54aed3bf1e11d96e39a809f90f02`, m365dotnet
`89282099d2ad3e4e474ee22db9c1c7ef1e9f8eee`.

**Consequence to carry forward.** Content is read from the working tree, and the eval
set is mined from that same HEAD, so each repo is internally consistent — the
measurement is valid. What it is *not* is a measurement of `origin/main`. A later
goal re-measuring at a different SHA is comparing two things; re-measure at the SHAs
above, or re-baseline both sides.

### Machine

| | |
|---|---|
| CPU | Apple M4 Max, 16 cores |
| RAM | 64 GB |
| OS | macOS 26.5.2 |
| Python | 3.13.7 |

Wall-clock and RSS are properties of this machine. Treat M2 as a same-machine
before/after instrument, never as an absolute contract — Neo's own learning-loop
benchmark carries this lesson already (#183: an 11× spread between a laptop and a
shared CI runner on identical code).

### How Neo was invoked

The repo source at `main`, installed editable into a dedicated venv — **not** the
globally installed `neo` (which on this machine is a miniconda shim at
`/opt/homebrew/Caskroom/miniconda/base/bin/neo` and is a different version).

```bash
git worktree add .worktrees/g_msp4yrhu_d974cb-unified-store-goal-1 \
  -b goalpool/g_msp4yrhu_d974cb-unified-store-goal-1 origin/main
cd .worktrees/g_msp4yrhu_d974cb-unified-store-goal-1
python3 -m venv .venv-eval
.venv-eval/bin/pip install -e .
.venv-eval/bin/python -c "import neo; print(neo.__version__)"   # -> 0.44.0
```

Every command below uses `.venv-eval/bin/neo` or `.venv-eval/bin/python` explicitly.

---

## M1 — retrieval quality

Harness: `tools/rank_eval.py` (#192). Eval sets are mined from each repo's own git
history: commit subject becomes the query, the commit's changed non-test source files
become the ground truth.

### Canonical mining parameters

These are `tools/rank_eval.py`'s defaults as of `77251df`. They are hereby the
standard for the plan; a later goal that changes any of them must re-baseline all
three repos, not just the one it cares about.

| Parameter | Value |
|---|---|
| commits scanned | 400 most recent, `--no-merges` |
| files per commit | 1–3 (inclusive) after filtering |
| subject length | > 15 characters |
| extensions kept | `.py .ts .tsx .js .cs .go .rb .php .java` |
| test files | excluded (`tests?/`, `spec/`, `test_`, `_test.`) |
| queries dropped at eval | subject starting `release ` (case-insensitive) |
| cutoffs reported | k = 1, 3, 5, 10, 20 |

### Eval-set sizes

| Repo | Cases mined | After `release` filter | Evaluable (ground truth still in tree) |
|---|---|---|---|
| neo | 214 | 210 | 209 |
| aieweb | 225 | 225 | 221 |
| m365dotnet | 173 | 173 | 173 |

### Results

**neo** — 209 evaluable cases

| strategy | R@1 | R@3 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|---|
| `neo_current` | 0.043 | 0.098 | 0.146 | 0.212 | 0.418 | 0.136 |
| `bm25_content` | 0.266 | 0.541 | 0.612 | 0.695 | 0.740 | 0.559 |
| `dense` | *not measured* | | | | | |
| `rrf_bm25_dense` | *not measured* | | | | | |

**aieweb** — 221 evaluable cases

| strategy | R@1 | R@3 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|---|
| `neo_current` | 0.053 | 0.117 | 0.182 | 0.244 | 0.296 | 0.180 |
| `bm25_content` | 0.417 | 0.683 | 0.757 | 0.823 | 0.880 | 0.726 |
| `dense` | *not measured* | | | | | |
| `rrf_bm25_dense` | *not measured* | | | | | |

**m365dotnet** — 173 evaluable cases

| strategy | R@1 | R@3 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|---|
| `neo_current` | 0.006 | 0.029 | 0.035 | 0.097 | 0.146 | 0.051 |
| `bm25_content` | 0.284 | 0.559 | 0.684 | 0.813 | 0.877 | 0.607 |
| `dense` | *not measured* | | | | | |
| `rrf_bm25_dense` | *not measured* | | | | | |

Harness cost, for scheduling later runs: neo 22.8 s / 131 MB, aieweb 45.1 s / 165 MB,
m365dotnet 275.5 s / 800 MB.

### `dense` and `rrf_bm25_dense` are NOT baselined — blocking reason

The harness prints `0.000` across the board for `dense`. **That is not a measurement
of dense retrieval.** `evaluate()` falls back to an empty ranking when
`ProjectIndex(root).chunks` is empty, and it was empty for all three repos —
confirmed directly:

```bash
.venv-eval/bin/python -c "
from neo.index.project_index import ProjectIndex
for r in ['neo','aieweb','m365dotnet']:
    print(r, len(ProjectIndex('<platform-root>/'+r).chunks))"
# neo 0 / aieweb 0 / m365dotnet 0
```

Because `rrf_bm25_dense` degrades to `bmr` when `dn` is empty, its row would be a
byte-for-byte duplicate
of `bm25_content` and carries no independent information either. Both rows are
recorded as **not measured**, not as zero.

Building the index was attempted and is blocked two different ways:

- **aieweb, m365dotnet — blocked by this goal's constraints.** `ProjectIndex`
  persists to `<repo_root>/.neo/` (`src/neo/index/project_index.py:157`), i.e.
  inside the measured repo. This goal may not write to either repo, and neither has
  `.neo` in its ignore rules, so an index build would drop an untracked directory
  into a tree that already holds someone's in-flight work.
- **neo — attempted, did not complete.** `NEO_OBSERVER_AUTOSTART=0 neo --index --cwd
  <neo>` was run in the foreground and killed at **11 min 21 s** having produced no
  output and no `.neo/` directory, while holding **10,453,072 KB ≈ 9.97 GB RSS** at
  207% CPU (`ps -o etime,rss,%cpu`). It was killed because a 10 GB resident process
  contaminates the M2 wall-clock and RSS measurements that are this goal's other
  deliverable. No `.neo/` was left behind.

That second observation is itself a finding, and it lands on Goal 6 ("move #194's
per-call BM25 into the on-disk store") and Goal 7 ("cold first build bounded and
reported"): on this machine, a cold semantic-index build of the *smallest* flagship —
Neo's own ~300-file Python repo — did not finish inside 11 minutes and reached ~10 GB
resident. Goal 7's "cold build bounded and reported" is not a reporting nicety.

**To close this gap**, a later goal should either (a) measure `dense` on neo alone,
where writing `.neo/` is permitted, after the cold-build cost is understood, or
(b) relocate the index out of the repo root so a read-only target can be measured at
all. Until then the `dense` lane on the flagships is unbaselined and must not be
quoted as 0.

### Reproduce M1

```bash
cd <neo-worktree>
for r in neo aieweb m365dotnet; do
  .venv-eval/bin/python tools/rank_eval.py \
    --build-from-git <platform-root>/$r > /tmp/cases_$r.json
done

for r in neo aieweb m365dotnet; do
  /usr/bin/time -l .venv-eval/bin/python tools/rank_eval.py \
    --eval <platform-root>/$r /tmp/cases_$r.json
done
```

---

## M2 — warm-call cost on m365dotnet

Wall-clock and peak RSS from process invocation to context-assembled, via
`neo --dry-run`, which assembles the full context and exits **before** any LLM call.
The number is therefore the cost of selection, with no inference in it.

### Canonical prompt battery

Six prompts, fixed. Committed as `tools/m2_battery.sh` so the battery is an artifact,
not a paragraph someone has to re-key. The shapes were chosen to exercise the paths
that actually differ: two name a specific file (the pinning path,
`EXPLICIT_PATH_BOOST`), two are concept-only (pure ranking, no path to pin), one is
mixed, and one is a symptom/debug phrasing.

The prompt column below is the **verbatim** string the script sends — no backticks, no
markdown. That is not pedantry: `EXPLICIT_PATH_BOOST` matches on the path token, and a
stray backtick is part of the prompt text a caller would actually be sending. Copy from
`tools/m2_battery.sh`, which is the artifact of record; this table is a rendering of it.

| id | shape | prompt (verbatim) |
|---|---|---|
| P1 | file-named | Explain what src/Parslee.M365.Api/Program.cs does during startup. |
| P2 | file-named | Add a null check to the entitlements lookup in src/Parslee.M365.Api/Controllers/EntitlementsController.cs |
| P3 | concept-only | How does the backend authenticate requests from the web app? |
| P4 | concept-only | Where is chat history persisted and how is it retrieved? |
| P5 | mixed | Fix the retry logic for Cosmos DB throttling in the repository layer. |
| P6 | symptom | A Semantic Kernel tool call returns no result and logs nothing. Why? |

**Protocol:** one discarded warm-up run, then 3 timed runs per prompt (18 timed runs
total). Reported wall-clock is the **median** — a developer laptop takes scheduler
stalls, and a single stall skews a 6-sample mean. Reported RSS is the **maximum**
observed, because a peak is a peak. `NEO_OBSERVER_AUTOSTART=0` throughout, so a
background observer sweep cannot land inside a timed run.

### Results

| id | shape | median wall (s) | min | max | peak RSS (MiB) | context entries |
|---|---|---|---|---|---|---|
| P1 | file-named | 10.52 | 10.44 | 11.43 | 1367.8 | 30 |
| P2 | file-named | 10.42 | 9.93 | 10.55 | 1368.0 | 30 |
| P3 | concept-only | 10.48 | 10.10 | 10.83 | 1367.2 | 30 |
| P4 | concept-only | 9.69 | 7.45 | 9.90 | 1366.2 | 30 |
| P5 | mixed | 11.15 | 10.57 | 11.41 | 1367.8 | 30 |
| P6 | symptom | 11.26 | 11.06 | 11.97 | 1366.9 | 30 |

**Battery: median wall-clock 10.54 s** (n = 18, min 7.45, max 11.97).
**Battery: peak `ru_maxrss` 1,434,402,816 bytes = 1368.0 MiB = 1.43 GB.**

Against the M2 targets of ≤ 5 s and ≤ 500 MB: **2.1× over on time, 2.87× over on
memory.** (1,434,402,816 B ÷ 500 MB. The plan states the target in MB, so the ratio is
computed in MB — reading `ru_maxrss` as MiB against a MiB target gives 2.74× and
understates the overage by 5%. Pick one base; this document uses the plan's.)

Two things worth noting about the shape of the result:

- **Cost is flat across prompt shape.** The spread from the cheapest prompt to the
  dearest is 9.69 s → 11.26 s, and RSS varies by under 2 MiB across all six. Naming a
  file costs the same as naming nothing. That is the signature of a fixed scan-the-
  world cost that dominates whatever the query asks for — which is the premise the
  unified-store plan is built on, now measured rather than asserted.
- The plan's pre-measurement projection was ~27 s / ~1 GB post-#194. Actual at
  `77251df` (pre-#194) is 10.54 s / 1.43 GB: **faster than projected on time, higher
  on memory.** The projection is superseded by this row.

### Reproduce M2

```bash
cd <neo-worktree>
tools/m2_battery.sh "$PWD/.venv-eval/bin/neo" <platform-root>/m365dotnet /tmp/m2_m365
```

Prints one CSV row per timed run, then prints the **aggregates published above** —
per-prompt median/min/max wall, per-prompt peak RSS, the battery median and peak, and
the ratios against the M2 targets. The aggregation is in the script rather than done by
hand, so every number in the M2 tables comes out of this one command.

It also writes into the output directory: each run's raw `/usr/bin/time -l` capture
(`<id>_run<n>.txt`), each run's selected-file list (`<id>_run<n>.files`), the raw
`results.csv`, and `union.files` — the distinct union across the battery, which is the
input to the G1 check below.

### Reproducibility re-run — 2026-08-12

The tables above came from an ad-hoc first pass; `tools/m2_battery.sh` was written
afterwards to make that pass a command. A doc that publishes a script as its
reproduction contract owes you evidence that the *committed script* produces the
published numbers, so it was run end-to-end a second time into a clean directory
(3 min 28 s wall). What it showed, split by what is deterministic and what is not:

| quantity | first pass | committed-script re-run | verdict |
|---|---|---|---|
| context entries / per-prompt distinct / **union** | 180 / 171 / **141** | 180 / 171 / **141** | identical |
| per-prompt distinct (P1…P6) | 29, 28, 29, 29, 28, 28 | 29, 28, 29, 29, 28, 28 | identical |
| the union file set itself | — | byte-identical set | identical |
| repeat entries: battery-wide / within-prompt | 39 / 9 | 39 / 9 | identical |
| files picked by >1 prompt | 21 | 21 | identical |
| G1-inv: ignored / duplicate / `.worktrees/` / unresolvable | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | identical |
| peak `ru_maxrss` | 1368.0 MiB (2.87×) | 1368.0 MiB (2.87×) | identical |
| **battery median wall** | **10.54 s** | **10.22 s** | ±3%, see below |

**Selection is deterministic; wall-clock is not.** Every file-count, the G1 verdict and
the RSS peak land on the same value twice. Wall-clock moved 3%, and the whole of that
move is one scheduler stall: P4 run 1 took 19.27 s in the re-run against a 10.18 s
run 3 on the same prompt. That is precisely why the protocol reports a median and why
this document says to treat M2 as a same-machine before/after instrument. **The
published baseline stays 10.54 s** — re-baselining to the second sample would be
picking a number for no reason. A later goal comparing against it should read a
change under ~±0.5 s as noise, not as a result.

---

## G1-invariant on m365dotnet

> **G1-inv** Zero selected files that `git check-ignore` excludes; zero duplicate
> copies. (2026-08-10 reality: 14/16 ignored, 12/16 duplicates.)

Measured over the full canonical battery — every file the selector put in context for
all six prompts. m365dotnet is the right target for this: it carries **44 in-repo
worktrees** under `.worktrees/`, which is exactly the condition that produced the
2026-08-10 duplicate explosion.

| prompt | context entries | distinct files | `git check-ignore` excludes | duplicate copies | inside `.worktrees/` | unresolvable paths |
|---|---|---|---|---|---|---|
| P1 | 30 | 29 | 0 | 0 | 0 | 0 |
| P2 | 30 | 28 | 0 | 0 | 0 | 0 |
| P3 | 30 | 29 | 0 | 0 | 0 | 0 |
| P4 | 30 | 29 | 0 | 0 | 0 | 0 |
| P5 | 30 | 28 | 0 | 0 | 0 | 0 |
| P6 | 30 | 28 | 0 | 0 | 0 | 0 |
| *sum of rows* | 180 | 171 | 0 | 0 | 0 | 0 |
| **BATTERY UNION** | **180** | **141** | **0** | **0** | **0** | **0** |

**G1-inv holds.** Post-#186 this is not "near zero", it is zero, on the repo with the
worst duplicate exposure on this machine — and it holds computed either way, over the
141-file union and over the 171 per-prompt selections with repeats.

**Read the union row, not the sum row.** 171 is the sum of the six per-prompt distinct
counts, which counts a file once per prompt that selected it; the battery selected
**141** distinct files in total. The 30-file difference is 30 *excess selections*
spread over **21 files** that more than one prompt picked — not 30 files. The
sum row is kept only to make the double-count visible. A later goal that reports a true
union and compares it to 171 will see a 17% "coverage regression" that never happened.

Definitions, so a later re-measure counts the same things:

- **`git check-ignore` excludes** — the distinct selected paths piped through
  `git -C <repo> check-ignore --stdin`; the count is how many came back.
- **duplicate copies** — selected files sharing a byte-identical SHA-256 with another
  selected file, counted as the excess over one per content group. Zero groups had
  more than one member.
- **inside `.worktrees/`** — selected paths under a nested worktree, counted
  separately because that is the specific mechanism behind the 2026-08-10 numbers.
- **unresolvable paths** — a selected path that does not exist on disk. Zero. Worth
  keeping in the table: it is the check that catches a parsing error in the
  measurement itself rather than a defect in Neo, and it caught one during this run
  (see below).

Two observations that are *not* G1 violations but belong on the record:

- **180 context entries cover 141 distinct files.** Battery-wide, **39** of the 180
  entries repeat a file some prompt had already selected (180 − 141) — a mix of
  `MAX_CHUNKS_PER_FILE` second windows and the same file being picked by more than one
  prompt. Neither is duplication in the G1 sense. Within a single prompt the effect is
  far smaller: only **9** entries battery-wide are a second window of a file already
  selected *by that same prompt* (180 − 171), which is what turns 30 entries into 28–29
  distinct. So "30 files selected" overstates that prompt's coverage by ~5%, and a raw
  entry count overstates battery coverage by 28%. Any later goal reporting a file count
  must say which of the three it is counting: entries, per-prompt distinct, or union.
  (Do not confuse either number with the **23** *windowed* entries in the footgun note
  below — that is the count of entries carrying a `(lines a-b)` range, a different
  quantity that happens to sit in the same range.)
- **Named paths were pinned.** P1's `Program.cs` and P2's `EntitlementsController.cs`
  each appear (2 windows each). That is a G2-inv spot check, not a G2-inv
  measurement — the full G2 battery is Goal 3's deliverable.

### Reproduce G1

Run the M2 battery first (it writes the per-prompt selected-file lists), then:

```bash
python3 - <<'PY'
import subprocess, hashlib, os, collections, glob
REPO = "<platform-root>/m365dotnet"

def g1(label, paths):
    paths = sorted(set(paths))
    ig = subprocess.run(["git", "-C", REPO, "check-ignore", "--stdin"],
                        input="\n".join(paths), capture_output=True, text=True)
    ignored = [l for l in ig.stdout.splitlines() if l.strip()]
    h, missing = collections.defaultdict(list), []
    for rel in paths:
        try: h[hashlib.sha256(open(os.path.join(REPO, rel), 'rb').read()).hexdigest()].append(rel)
        except OSError: missing.append(rel)
    dup = sum(len(v) - 1 for v in h.values() if len(v) > 1)
    wt = [p for p in paths if p.startswith(".worktrees/") or "/.worktrees/" in p]
    print(label, len(paths), len(ignored), dup, len(wt), len(missing))

print("label distinct ignored dup in_worktrees unresolvable")
union = set()
for pid in ["P1", "P2", "P3", "P4", "P5", "P6"]:
    p = [l.strip() for l in open(f"/tmp/m2_m365/{pid}_run1.files") if l.strip()]
    union |= set(p)
    g1(pid, p)
g1("UNION", union)   # <- the row the headline quotes
PY
```

**Measurement footgun, recorded because it silently produced a wrong answer once.**
`--dry-run` emits a selected line as `  <rel_path> (lines a-b) - <n> bytes (score: x)`
— the line range is present only for windowed files. A first pass extracted the path
without stripping that suffix, so 23 of 180 entries carried `" (lines 1-168)"` on the
end. Those paths match nothing on disk, and `git check-ignore` then silently tested a
path that does not exist. The run still reported `0 ignored, 0 duplicates` — the right
answer, by luck, from a broken measurement. `tools/m2_battery.sh` strips the suffix,
and the `unresolvable paths` column exists so the same class of error announces itself
next time instead of passing.

---

## What this document does not contain

| Missing | Reason |
|---|---|
| `dense` / `rrf_bm25_dense` M1 baselines | Index build writes `<repo>/.neo/`; forbidden on aieweb and m365dotnet, and did not complete in 11 min at ~10 GB RSS on neo. See M1 section. |
| M1 / M2 at `origin/main` for aieweb and m365dotnet | Both hold uncommitted in-flight work; checking out `origin/main` is a write this goal is forbidden to make. Measured at the recorded HEADs instead. |
| M2 / G1 on neo and aieweb | Goal 1 scopes M2 and G1 to m365dotnet. |
| M3 (freshness cost) | Not in Goal 1's scope; it has no instrument yet — full rebuild every call. |
| A cold-vs-warm split for M2 | The protocol discards one warm-up run and reports warm figures. Cold first-call cost is Goal 7's deliverable and needs the index question resolved first. |
