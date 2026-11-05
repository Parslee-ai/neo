#!/usr/bin/env bash
# Drive the Neo CLI from THIS checkout in an isolated scratch HOME + fixture repo.
# Usage: verify.sh build | new | doctor RUN | neo RUN LABEL [neo args...]
#        | snapshot RUN LABEL | keep RUN RELPATH | cleanup RUN
# Evidence lives in $NEO_VERIFY_EVIDENCE_ROOT/<run> (default /tmp/neo-verify-evidence)
# and survives cleanup. Scratch state lives in /tmp/neo-verify-runs/<run>.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NEO="$REPO/.venv/bin/neo"
PY="$REPO/.venv/bin/python"
RUNS_ROOT="/tmp/neo-verify-runs"
EVIDENCE_ROOT="${NEO_VERIFY_EVIDENCE_ROOT:-/tmp/neo-verify-evidence}"
REAL_EMBED_CACHE="$HOME/.cache/neo/fastembed"

die() { echo "verify.sh: $*" >&2; exit 2; }

run_dir() {
  [[ "${1:-}" =~ ^run-[0-9]{8}-[0-9]{6}-[0-9]+$ ]] || die "bad RUN id '${1:-}'"
  echo "$RUNS_ROOT/$1"
}

cmd_build() {
  command -v uv >/dev/null || die "uv not on PATH"
  [ -x "$PY" ] || uv venv "$REPO/.venv" --python 3.12 -q || die "venv create failed"
  uv pip install -q --python "$PY" -e "$REPO[dev]" || die "install failed"
  echo "built: $("$PY" -c 'import neo; print(neo.__version__)') at $NEO"
}

cmd_new() {
  [ -x "$NEO" ] || die "no $NEO; run: verify.sh build"
  local id; id="run-$(date +%Y%m%d-%H%M%S)-$$"
  local d="$RUNS_ROOT/$id"
  mkdir -p "$d/home" "$d/repo/src/billing" "$EVIDENCE_ROOT/$id"
  # Fixture repo: small, deterministic, with a git remote so project_id is stable.
  cat > "$d/repo/src/billing/invoice.py" <<'PY'
"""Invoice totals for the fixture billing service."""

LATE_FEE_RATE = 0.015  # NEO_VERIFY_MARKER_7f3a


def invoice_total(subtotal: float, days_late: int) -> float:
    if days_late <= 0:
        return subtotal
    return round(subtotal * (1 + LATE_FEE_RATE * days_late), 2)
PY
  cat > "$d/repo/src/billing/discounts.py" <<'PY'
def apply_discount(total: float, code: str) -> float:
    if code == "WELCOME10":
        return total * 0.9
    return total
PY
  printf '# Fixture billing service\n\nRules: money is rounded to cents.\n' > "$d/repo/CLAUDE.md"
  ( cd "$d/repo" && git init -q && git remote add origin https://example.invalid/neo-verify/fixture.git \
      && git add -A && git -c user.name=verify -c user.email=verify@example.invalid commit -qm fixture ) \
      || die "fixture git init failed"
  echo "$id"
}

# Environment every drive runs under. HOME is scratch so ~/.neo is disposable;
# the ~400 MB embedding model is reused read-mostly from the real cache.
neo_env() {
  local d="$1"
  echo "HOME=$d/home" "NEO_OBSERVER_AUTOSTART=0" "NEO_AUTO_INSTALL_UPDATES=0" \
       "NEO_FASTEMBED_CACHE_DIR=$REAL_EMBED_CACHE" "PATH=$REPO/.venv/bin:/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"
  # Provider credentials pass through only when the caller exported them.
  local v; for v in OPENAI_API_KEY ANTHROPIC_API_KEY GOOGLE_API_KEY NEO_PROVIDER NEO_MODEL NEO_API_KEY NEO_BASE_URL; do
    [ -n "${!v:-}" ] && echo "$v=${!v}"
  done
}

cmd_doctor() {
  local d; d="$(run_dir "${1:-}")" || return 2; local ok=1
  [ -x "$NEO" ] && echo "ok   neo binary: $NEO" || { echo "FAIL no $NEO (verify.sh build)"; ok=0; }
  local want; want="$(sed -n 's/^version = "\(.*\)"/\1/p' "$REPO/pyproject.toml" | head -1)"
  local got; got="$(env -i $(neo_env "$d") "$PY" -c 'import neo,os;print(neo.__version__);print(os.path.dirname(neo.__file__))' 2>&1)"
  local gv; gv="$(echo "$got" | sed -n 1p)"; local gp; gp="$(echo "$got" | sed -n 2p)"
  [ "$gv" = "$want" ] && echo "ok   version $gv matches pyproject" || { echo "FAIL version '$gv' != pyproject '$want'"; ok=0; }
  [ "$gp" = "$REPO/src/neo" ] && echo "ok   imports from this tree ($gp)" || { echo "FAIL imports from '$gp', not $REPO/src/neo"; ok=0; }
  [ -d "$d/home" ] && [ -d "$d/repo/.git" ] && echo "ok   run dir $d (scratch HOME + fixture repo)" || { echo "FAIL run dir $d incomplete"; ok=0; }
  [ -d "$REAL_EMBED_CACHE" ] && echo "ok   embedding cache $REAL_EMBED_CACHE" || echo "warn no embedding cache; first embed downloads ~400 MB"
  [ -d "$EVIDENCE_ROOT/${1}" ] && echo "ok   evidence dir $EVIDENCE_ROOT/$1" || { echo "FAIL no evidence dir"; ok=0; }
  [ "$ok" = 1 ] && echo "DOCTOR OK" || { echo "DOCTOR FAILED"; return 1; }
}

# verify.sh neo RUN LABEL [args...]  — runs neo in the fixture repo, stdin passes through.
# Writes <label>.cmd/.stdout/.stderr/.exit under the evidence dir and echoes the exit code.
cmd_neo() {
  local id="${1:-}"; local d; d="$(run_dir "$id")" || return 2; local label="${2:-}"
  [ -n "$label" ] || die "missing LABEL"; shift 2
  local ev="$EVIDENCE_ROOT/$id"
  printf '%q ' neo "$@" > "$ev/$label.cmd"; echo >> "$ev/$label.cmd"
  ( cd "$d/repo" && env -i $(neo_env "$d") "$NEO" "$@" ) > "$ev/$label.stdout" 2> "$ev/$label.stderr"
  local rc=$?; echo "$rc" > "$ev/$label.exit"
  echo "exit=$rc evidence=$ev/$label.{cmd,stdout,stderr,exit}"
  return $rc
}

# Snapshot scratch-HOME state into evidence (side-effect proof). Lists FILES only:
# an empty directory does not appear, so it cannot prove a directory is absent.
cmd_snapshot() {
  local id="${1:-}"; local d; d="$(run_dir "$id")" || return 2; local label="${2:-state}"
  ( cd "$d" && find home repo -path repo/.git -prune -o -type f -print | sort ) > "$EVIDENCE_ROOT/$id/$label.files"
  echo "wrote $EVIDENCE_ROOT/$id/$label.files ($(wc -l < "$EVIDENCE_ROOT/$id/$label.files") files)"
}

# Copy a file from the run dir (relative to it, e.g. home/.neo/sessions/host_events.jsonl)
# into evidence, so proof read from scratch state survives cleanup.
cmd_keep() {
  local id="${1:-}"; local d; d="$(run_dir "$id")" || return 2; local rel="${2:-}"
  [ -f "$d/$rel" ] || die "no file $d/$rel"
  local dest="$EVIDENCE_ROOT/$id/kept/$rel"; mkdir -p "$(dirname "$dest")" && cp "$d/$rel" "$dest" && echo "kept $dest"
}

cmd_cleanup() {
  local id="${1:-}"; local d; d="$(run_dir "$id")" || return 2
  rm -rf "$d" && echo "removed $d; evidence kept at $EVIDENCE_ROOT/$id"
}

case "${1:-}" in
  build) cmd_build ;;
  new) cmd_new ;;
  doctor) shift; cmd_doctor "$@" ;;
  neo) shift; cmd_neo "$@" ;;
  snapshot) shift; cmd_snapshot "$@" ;;
  keep) shift; cmd_keep "$@" ;;
  cleanup) shift; cmd_cleanup "$@" ;;
  *) sed -n 2,6p "$0"; exit 2 ;;
esac
