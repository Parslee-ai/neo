#!/usr/bin/env bash
# Goal 8 stage-4 measurement: neo, WITH an embedding catalog present.
#
# Distinct from the M1 sweep, which ran with no catalog in any repo -- the
# state every flagship is actually in. Do not merge these tables: the repo is
# not in the same state, so a cell here is not comparable with a cell there.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
MAIN_TREE="$ROOT/.worktrees/g_msr9uzf2_114f50-main-baseline"
PY="$HERE/.venv/bin/python"
export NEO_OBSERVER_AUTOSTART=0 NEO_PROFILE=off

run() {  # name tree extra-flags...
  local name="$1" tree="$2"; shift 2
  echo "=== catalog/$name ===" >&2
  "$PY" "$HERE/tools/rank_mine_eval.py" --repo "$ROOT" --tree "$tree" \
    --cases 50 --skip-recent 50 --k 1 3 10 --timeout 600 --json "$@" \
    > "$HERE/evidence/cat_${name}.json" 2> "$HERE/evidence/cat_${name}.log"
  echo "  exit=$?" >&2
}

run main            "$MAIN_TREE"
run branch          "$HERE"
run branch_semantic "$HERE" --semantic
echo "=== catalog sweep done ===" >&2
