#!/usr/bin/env bash
# Goal 8 M1 sweep: 3 flagship repos x 2 arms, one harness, back to back.
#
#   evidence/run_m1.sh
#
# Arms differ only by --tree. The harness, the repo, the HEAD, the mined cases
# and every parameter are identical, so the delta is the ranker and nothing
# else. Runs are sequential so the two arms do not contend for CPU.
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"          # the neo repo
PLATFORM="$(cd "$ROOT/.." && pwd)"         # parslee-knowledge

BRANCH_TREE="$HERE"
MAIN_TREE="$ROOT/.worktrees/g_msr9uzf2_114f50-main-baseline"
PY="$HERE/.venv/bin/python"
OUT="$HERE/evidence"

export NEO_OBSERVER_AUTOSTART=0
export NEO_PROFILE=off

for repo_name in neo aieweb m365dotnet; do
  case "$repo_name" in
    neo) repo="$ROOT" ;;
    *)   repo="$PLATFORM/$repo_name" ;;
  esac
  for arm in main branch; do
    case "$arm" in
      main)   tree="$MAIN_TREE" ;;
      branch) tree="$BRANCH_TREE" ;;
    esac
    echo "=== M1 $repo_name / $arm ===" >&2
    "$PY" "$BRANCH_TREE/tools/rank_mine_eval.py" \
      --repo "$repo" --tree "$tree" \
      --cases 50 --skip-recent 50 --k 1 3 10 --timeout 600 --json \
      > "$OUT/m1_${repo_name}_${arm}.json" 2> "$OUT/m1_${repo_name}_${arm}.log"
    echo "  exit=$? -> $OUT/m1_${repo_name}_${arm}.json" >&2
  done
done
echo "=== M1 sweep done ===" >&2
