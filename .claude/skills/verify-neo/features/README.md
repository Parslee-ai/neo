# Neo verification map

The maintained source for verifying Neo's user-facing CLI behavior. Read this index, then use the matching feature file as the recipe.

## Baseline preconditions

- From the repo root, set `V=.claude/skills/verify-neo/verify.sh`. Every recipe types `$V`; a bare `verify.sh` is not on PATH.
- `$V build` has run, and `ID=$($V new)` created a fresh run.
- `$V doctor "$ID"` prints `DOCTOR OK`.
- The fixture repo holds `src/billing/invoice.py` (marker `NEO_VERIFY_MARKER_7f3a`), `src/billing/discounts.py` and `CLAUDE.md` (`money is rounded to cents`).
- The scratch HOME starts empty: no facts, no sessions, no config.

## Driving conventions

- Run every command through `$V neo "$ID" <label> ...`, never a bare `neo`.
- In each recipe, the bullets after the `Preconditions:` list are the steps, in order.
- Labels are unique per run; they name the evidence files.
- Prefer `--json` and assert on parsed fields, not on prose.
- Commands are literal. Keep quoted prompts and flags unchanged.

## Proof and skip reporting

- CLI proof is the `.cmd`, `.stdout`, `.stderr` and `.exit` files for the label.
- Side-effect proof is a `$V snapshot` diff (files only; empty directories do not show) or a file copied with `$V keep`, which survives cleanup.
- Record the feature file and sub-feature ID with every artifact.
- An unreachable path is reported with the attempted command and the unmet precondition, never as verified through a different path.

## Features

- [Dry-run prompt inspection](./dry-run.md): what Neo would send, with no LM call and no fact-store writes.
- [Ask Neo (reasoning request)](./ask.md): a real answer from a provider. Needs a provider key.
- [Recall](./recall.md): the per-prompt lesson lookup the Claude Code mod calls.
- [Hook record](./hook-record.md): the edit telemetry hosts send after Edit/Write.
- [Status and memory stats](./status.md): `--version` and `memory learning-stats`.
