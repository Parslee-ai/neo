# Status and memory stats

`neo --version` shows the build, provider, store path and CAR detection. `neo memory learning-stats` summarizes the learning loop (episodes, promotions, gates) without an LM call.

## Sub-features

- `status-version` prints the build version and the scratch store path.
- `status-stats-json` prints learning stats as JSON.

## How to get to it (user POV)

- Run `neo --version` (or `neo -v`).
- Run `neo memory learning-stats --json`.

## Driving it with verify.sh

Preconditions:

- Baseline state from the README.

- **Version.** Run `$V neo "$ID" version --version`. Prints `exit=0`; stdout has `neo <pyproject version>` and `Storage: FactStore (path: /tmp/neo-verify-runs/<run>/home/.neo/facts)`.
- **Learning stats.** Run `$V neo "$ID" lstats memory learning-stats --json`. Prints `exit=0`; stdout parses as JSON with `"episodes": 0` and `"promotions": 0` on a fresh run.

## Gotchas

- `--version` opens a FactStore, which runs maintenance and saves; it is not read-only. Run it after any snapshot you diff for a different feature.
- The first stdout line is a random flavor quote. Match `neo <version>` on a later line.
- The CAR line reports the real machine's CAR install; it is detection only.
