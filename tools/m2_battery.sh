#!/usr/bin/env bash
# Canonical M2 / G1-invariant measurement battery for the unified-store plan.
#
#   tools/m2_battery.sh <neo-bin> <target-repo> <out-dir>
#
# Measures, per prompt: wall-clock and peak RSS (ru_maxrss) from process
# invocation to context-assembled, using `neo --dry-run` -- which exits after
# assembling context and before any LLM call, so the number is the cost of
# selection, not of inference.
#
# The prompt battery below is CANONICAL for the unified-store plan: every later
# goal re-measures with these exact prompts so numbers compare goal-over-goal.
# Two prompts name a specific file (the pinning path, EXPLICIT_PATH_BOOST), two
# are concept-only (the ranking path), one is mixed, one is a symptom/debug
# phrasing -- the four shapes a real caller actually sends.
#
# Protocol: one discarded warm-up run, then RUNS timed runs per prompt; the
# reported wall-clock is the median, the reported RSS is the maximum observed.
# Median over mean because a single scheduler stall skews a 6-sample mean and
# this runs on a developer laptop, not an isolated box.
#
# The observer is disabled (NEO_OBSERVER_AUTOSTART=0) so a background sweep
# cannot land inside a timed run. It is a separate process and would not enter
# ru_maxrss, but it does contend for CPU.
set -uo pipefail

NEO_BIN="${1:?usage: m2_battery.sh <neo-bin> <target-repo> <out-dir>}"
REPO="${2:?}"
OUT="${3:?}"
RUNS="${RUNS:-3}"

mkdir -p "$OUT"

# id|shape|prompt
PROMPTS=(
  "P1|file-named|Explain what src/Parslee.M365.Api/Program.cs does during startup."
  "P2|file-named|Add a null check to the entitlements lookup in src/Parslee.M365.Api/Controllers/EntitlementsController.cs"
  "P3|concept-only|How does the backend authenticate requests from the web app?"
  "P4|concept-only|Where is chat history persisted and how is it retrieved?"
  "P5|mixed|Fix the retry logic for Cosmos DB throttling in the repository layer."
  "P6|symptom|A Semantic Kernel tool call returns no result and logs nothing. Why?"
)

export NEO_OBSERVER_AUTOSTART=0

# Warm-up (page cache, interpreter start); result discarded.
"$NEO_BIN" --dry-run --cwd "$REPO" "warm up" >/dev/null 2>&1

# Header and rows go through the SAME pipe, or results.csv lands headerless and the
# aggregation below silently reads the first measurement as its column names.
{
echo "id,shape,run,wall_seconds,max_rss_bytes,files_selected"
for entry in "${PROMPTS[@]}"; do
  id="${entry%%|*}"; rest="${entry#*|}"
  shape="${rest%%|*}"; prompt="${rest#*|}"
  for run in $(seq 1 "$RUNS"); do
    raw="$OUT/${id}_run${run}.txt"
    /usr/bin/time -l "$NEO_BIN" --dry-run --cwd "$REPO" "$prompt" >/dev/null 2>"$raw"
    wall=$(awk '/ real /{print $1}' "$raw" | tail -1)
    rss=$(awk '/maximum resident set size/{print $1}' "$raw" | tail -1)
    # Selected lines look like:  "  <rel_path>[ (lines a-b)] - <n> bytes (score: x.xx)"
    # The optional line range MUST be stripped: a windowed file is emitted once
    # per window, so leaving the suffix on both corrupts the path (nothing on
    # disk matches it, and `git check-ignore` then tests a path that does not
    # exist) and hides that one file can occupy several of the slots.
    grep -E '^  .+ - [0-9]+ bytes' "$raw" \
      | sed -E 's/^  (.+) - [0-9]+ bytes.*/\1/; s/ \(lines [0-9]+-[0-9]+\)$//' \
      > "$OUT/${id}_run${run}.files"
    nfiles=$(sort -u "$OUT/${id}_run${run}.files" | wc -l | tr -d ' ')
    echo "$id,$shape,$run,$wall,$rss,$nfiles"
  done
done
} | tee "$OUT/results.csv"

# The published aggregates are derived here rather than by hand, so the doc's
# "exact command to reproduce" contract covers the headline numbers too.
#
# Median, not mean, for wall-clock (a single scheduler stall skews a small mean).
# Max, not mean, for RSS (a peak is a peak).
# Distinct file counts are reported BOTH as the per-prompt sum and as the battery
# UNION. They differ -- a file selected by two prompts is one file -- and quoting
# the sum as a distinct count inflates it. Report the union.
python3 - "$OUT" <<'PY'
import csv, statistics, collections, sys, pathlib
out = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(open(out / "results.csv")))
by = collections.defaultdict(list)
for r in rows:
    by[(r["id"], r["shape"])].append((float(r["wall_seconds"]), int(r["max_rss_bytes"])))

print("\n=== M2 aggregates ===")
print(f"{'id':<4}{'shape':<14}{'median_s':>10}{'min_s':>8}{'max_s':>8}{'peak_RSS_MiB':>14}")
walls, rsss = [], []
for (pid, shape), v in sorted(by.items()):
    w = [x[0] for x in v]; m = [x[1] for x in v]
    walls += w; rsss += m
    print(f"{pid:<4}{shape:<14}{statistics.median(w):>10.2f}{min(w):>8.2f}{max(w):>8.2f}"
          f"{max(m)/1048576:>14.1f}")
peak = max(rsss)
print(f"\nBATTERY median wall = {statistics.median(walls):.2f}s "
      f"(n={len(walls)}, min {min(walls):.2f}, max {max(walls):.2f})")
print(f"BATTERY peak ru_maxrss = {peak} B = {peak/1048576:.1f} MiB = {peak/1e9:.2f} GB")
print(f"vs M2 targets (<=5s / <=500MB): {statistics.median(walls)/5:.2f}x time, "
      f"{peak/500e6:.2f}x memory")

union, persum, entries = set(), 0, 0
for f in sorted(out.glob("P*_run1.files")):
    e = [l.strip() for l in f.read_text().splitlines() if l.strip()]
    entries += len(e); persum += len(set(e)); union |= set(e)
print(f"\nselected: {entries} entries | {persum} sum-of-per-prompt-distinct "
      f"| {len(union)} BATTERY UNION distinct")
(out / "union.files").write_text("\n".join(sorted(union)) + "\n")
print(f"union written to {out / 'union.files'} (feed this to the G1 check)")
PY
