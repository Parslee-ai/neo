# Dry-run prompt inspection

`neo --dry-run` runs the real engine (file selection, fact retrieval, context assembly), prints the exact messages Neo would hand the provider adapter, and stops before any inference. It needs no credentials and must not write the fact store.

## Sub-features

- `dry-json` emits one stdout document with `dry_run: true`, `calls`, and `note`.
- `dry-pin` includes a file the prompt names by path.
- `dry-instructions` includes the repo's `CLAUDE.md`.
- `dry-no-writes` writes no files under `home/.neo/facts/` (the empty directory is created).
- `dry-events` ends stderr with `phase_completed` (status `dry_run`) then `completed`.

## How to get to it (user POV)

- Run `neo --dry-run "<question>"` in a repo (human-readable).
- Run `neo --dry-run --json "<question>"` (what hosts and this map use).

## Driving it with verify.sh

Preconditions:

- Baseline state from the README.

- **Snapshot before.** Run `$V snapshot "$ID" before`. Lists 3 fixture files and nothing under `home/`.
- **Dry run.** Run `$V neo "$ID" dryrun --dry-run --json "Why does invoice_total in src/billing/invoice.py round the late fee?"`. Prints `exit=0`.
- **Shape.** Parse `dryrun.stdout`: `dry_run` is `true` and `calls` has 1 entry with a `messages` list.
- **Pinned file and instructions.** Join every `calls[0].messages[*].content`. It contains `NEO_VERIFY_MARKER_7f3a` and `money is rounded to cents`.
- **Events.** The last two lines of `dryrun.stderr` are `phase_completed` with `"status": "dry_run"` and `completed` with message `dry run: context assembled, no inference performed`.
- **No store writes.** Run `$V snapshot "$ID" after` and diff the two `.files` lists. New files are limited to `home/.neo/metrics.jsonl`, `home/.neo/update_check.json`, onnxruntime files under `home/Library/`, and `repo/.neo/{content_index.sqlite3,walk_cache.json}`. Nothing under `home/.neo/facts/` or `home/.neo/episodes/`.

## Gotchas

- `--json` stderr starts with a plain-text `=== DRY RUN` report; only the later lines are JSON events. Parse from the first line starting with `{`.
- Not network-free: the update check calls PyPI. Offline, expect a slower run, not a failure.
- `repo/.neo/` is a per-repo index cache. Driving the Neo repo itself would write into the developer's own cache; use the fixture.
- The report shows the adapter's input, not the wire payload; providers restructure messages (the `note` field says so).
- A read-only store skips ingesters, so a `CLAUDE.md` edit made after the last writable run is not reflected. On a fresh run the fixture `CLAUDE.md` still appears because it is gathered as context.
