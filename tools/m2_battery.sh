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
