#!/usr/bin/env bash
# Goal 9 closing measurement: M1 on three flagships, M2 on m365dotnet, and the
# per-stage profile — one arm, one base, one pass.
#
#   evidence/run_goal9.sh
#
# ONE arm, unlike Goals 6-8. Those compared a branch against the main it would
# merge into, because each was changing the ranker and had to show it had not
# regressed. This goal deletes prose and adds a guard test; the runtime is
# `origin/main` plus zero behavioural change, so a two-arm A/B would spend an
# hour proving that identical code performs identically. The comparison that
# means something at the end of the climb is against the **Goal 1 trailhead**,
# which is the number in `docs/eval-baselines-2026-08.md`, not a second run.
#
# Everything is sequential: the M2 battery measures wall-clock on a developer
# laptop, so an M1 sweep running beside it would be measuring the sweep.
set -uo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"          # the goal-9 worktree
ROOT="$(cd "$HERE/../.." && pwd)"                 # the neo repo
PLATFORM="$(cd "$ROOT/.." && pwd)"                # parslee-knowledge
PY="$HERE/.venv/bin/python"
OUT="$HERE/evidence"
BIN="$OUT/bin"

export NEO_OBSERVER_AUTOSTART=0
export NEO_PROFILE=off

mkdir -p "$BIN"
# The arm launcher pins WHICH tree executes. `--cwd` says which repo is read;
# these are different questions and conflating them is how a measurement ends
# up describing the wrong ranker.
cat > "$BIN/neo-g9" <<LAUNCHER
#!/usr/bin/env bash
exec env PYTHONPATH="$HERE/src" NEO_OBSERVER_AUTOSTART=0 \\
  "$PY" -m neo.cli "\$@"
LAUNCHER
chmod +x "$BIN/neo-g9"

echo "=== arm identity ===" >&2
PYTHONPATH="$HERE/src" "$PY" -c \
  "import neo,sys; print('neo', neo.__version__, 'from', neo.__file__, file=sys.stderr)"
git -C "$HERE" rev-parse HEAD >&2

# ---- M1: three flagships, one arm -----------------------------------------
for repo_name in neo aieweb m365dotnet; do
  case "$repo_name" in
    neo) repo="$ROOT" ;;
    *)   repo="$PLATFORM/$repo_name" ;;
  esac
  echo "=== M1 $repo_name ===" >&2
  "$PY" "$HERE/tools/rank_mine_eval.py" \
    --repo "$repo" --tree "$HERE" \
    --cases 50 --skip-recent 50 --k 1 3 10 --timeout 600 --json \
    > "$OUT/g9_m1_${repo_name}.json" 2> "$OUT/g9_m1_${repo_name}.log"
  echo "  exit=$? -> $OUT/g9_m1_${repo_name}.json" >&2
done

# ---- M2: the canonical battery on m365dotnet ------------------------------
echo "=== M2 battery ===" >&2
"$HERE/tools/m2_battery.sh" "$BIN/neo-g9" "$PLATFORM/m365dotnet" \
  "$OUT/m2_g9" > "$OUT/g9_m2.txt" 2>&1
echo "  exit=$?" >&2

# ---- per-stage profile ----------------------------------------------------
# One discarded warm-up first: the profile is a WARM-call profile, and a cold
# process pays a fastembed model download/mmap that no steady-state invocation
# pays. Goal 7's figures are warm; an unwarmed re-run would read as a 20x
# regression that is entirely page cache.
echo "=== per-stage profile (warm-up, discarded) ===" >&2
"$BIN/neo-g9" --dry-run --quiet --cwd "$PLATFORM/m365dotnet" \
  "How does the backend authenticate requests from the web app?" \
  > /dev/null 2>&1
echo "=== per-stage profile (measured) ===" >&2
PYTHONPATH="$HERE/src" "$PY" "$OUT/g9_stage_profile.py" \
  "$PLATFORM/m365dotnet" > /dev/null 2> "$OUT/g9_profile.txt"
echo "  exit=$? -> $OUT/g9_profile.txt" >&2

echo "=== Goal 9 measurement done ===" >&2
