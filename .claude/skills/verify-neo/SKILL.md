---
name: verify-neo
description: Drive the Neo CLI (`neo`) built from THIS checkout against a disposable fixture repo and scratch HOME, and capture evidence (stdout JSON, stderr JSONL events, exit codes, files written). Use when proving a Neo change works the way a user or host plugin hits it - dry-run prompt inspection, a real reasoning request, `neo recall`, `neo hook record`, memory stats - rather than trusting unit tests.
---

# Verify Neo

Neo's user surface is the `neo` CLI (hosts such as the Claude Code mod call the same CLI). There is no server to keep alive: build once, then every drive is one short process run inside a scratch HOME and a scratch git repo. Secondary surfaces not covered here: `neo serve` (A2A over CAR) and `neo memory observer` (CAR-supervised daemon); both touch the shared CAR daemon and must not be driven from this skill.

Everything goes through `.claude/skills/verify-neo/verify.sh` (run from the repo root). It is the only sanctioned way to run Neo here, because it guarantees the three isolation rules below.

## Isolation rules (why the helper exists)

- **Never use the `neo` on PATH.** It is a pipx install of an older release. The helper runs `.venv/bin/neo`, and `doctor` proves it imports from `./src/neo` at the `pyproject.toml` version.
- **Never use the real `~/.neo`.** Each run gets `HOME=/tmp/neo-verify-runs/<run>/home`; only the embedding model cache is shared (`NEO_FASTEMBED_CACHE_DIR=~/.cache/neo/fastembed`, read-mostly).
- **Never drive the Neo repo itself.** Neo writes a per-repo cache to `<repo>/.neo/`; drives run inside a fixture repo at `/tmp/neo-verify-runs/<run>/repo`. `NEO_OBSERVER_AUTOSTART=0` keeps drives off the CAR daemon. Runs are isolated from each other, so parallel runs are safe.

## Launch

```bash
.claude/skills/verify-neo/verify.sh build      # uv venv .venv + editable install; prints "built: <version>"
ID=$(.claude/skills/verify-neo/verify.sh new)  # prints run id, e.g. run-20261105-133155-15505
```

`new` creates the scratch HOME, an evidence dir, and the fixture repo (a git repo with remote `https://example.invalid/neo-verify/fixture.git`):

- `src/billing/invoice.py`: `invoice_total()`, `LATE_FEE_RATE = 0.015`, and the marker string `NEO_VERIFY_MARKER_7f3a`
- `src/billing/discounts.py`: `apply_discount()`
- `CLAUDE.md`: contains `money is rounded to cents`

Ready means `doctor` prints `DOCTOR OK`.

## Doctor

```bash
.claude/skills/verify-neo/verify.sh doctor "$ID"
```

Read-only. Checks the binary exists, its version equals `pyproject.toml`, it imports from this tree, the run dir and evidence dir exist, and the embedding cache is present. Run it first, and again whenever output looks wrong. Exits 1 and prints `DOCTOR FAILED` on any failure.

## Drive

```bash
.claude/skills/verify-neo/verify.sh neo "$ID" <label> [neo args...]   # stdin passes through
.claude/skills/verify-neo/verify.sh snapshot "$ID" <label>            # list files in scratch HOME + fixture repo
.claude/skills/verify-neo/verify.sh keep "$ID" <path-in-run-dir>      # copy a scratch file into evidence
```

`neo` runs inside the fixture repo under a clean `env -i` environment and records `<label>.cmd`, `.stdout`, `.stderr`, `.exit` in the evidence dir. It prints `exit=<rc> evidence=...` and returns Neo's exit code. Provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `NEO_PROVIDER`, `NEO_MODEL`, `NEO_API_KEY`, `NEO_BASE_URL`) pass through only if you export them; they are never written to `.cmd`, though they are visible in `ps` while a drive runs.

Use `--json` for assertions: stdout is exactly one JSON document and stderr carries JSONL events (a dry run prints a plain-text report on stderr first; parse from the first line starting with `{`). Per-feature recipes are in `features/` (start at `features/README.md`).

## Evidence

Evidence goes to `/tmp/neo-verify-evidence/<run>/` (override the root with `NEO_VERIFY_EVIDENCE_ROOT`) and survives cleanup. Proof standards:

- Drive the real CLI path a user or host uses. Never call Python internals or test fixtures as proof.
- Capture the command and the result: `.cmd`, `.stdout`, `.stderr`, `.exit`.
- Prove side effects with `snapshot` before and after (for example, a dry run writes no files under `home/.neo/facts/`; it does create the empty directory). `snapshot` lists files only, so it cannot prove a directory is absent.
- A file you read as proof from the run dir is deleted by cleanup. Copy it first: `verify.sh keep "$ID" home/.neo/sessions/host_events.jsonl` puts it under `<evidence>/kept/`.
- `--dry-run` is not network-free: the update check calls PyPI and writes `home/.neo/update_check.json`. It does skip the LM call, the CAR daemon, and fact-store writes.

## Cleanup

```bash
.claude/skills/verify-neo/verify.sh cleanup "$ID"
```

Removes only `/tmp/neo-verify-runs/<run>`; every command rejects an id that is not exactly `run-YYYYMMDD-HHMMSS-PID`. Evidence stays. Run it after every attempt, failed ones included. The helper starts no background processes, so there is nothing to kill. Leave `.venv` in place; it is gitignored and reused.
