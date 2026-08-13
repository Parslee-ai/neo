#!/usr/bin/env bash
# Goal 8 M2: the canonical battery on m365dotnet, both arms, back to back.
set -uo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)/m365dotnet"
for arm in main branch; do
  echo "=== M2 $arm ===" >&2
  "$HERE/tools/m2_battery.sh" "$HERE/evidence/bin/neo-$arm" "$REPO" \
    "$HERE/evidence/m2_$arm" > "$HERE/evidence/m2_$arm.txt" 2>&1
  echo "  exit=$?" >&2
done
echo "=== M2 done ===" >&2
