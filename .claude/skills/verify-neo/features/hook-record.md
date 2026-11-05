# Hook record

Hosts pipe their PostToolUse payload to `neo hook record` after an Edit/Write so Neo can later tell whether a suggestion was acted on. It records the path and git HEAD, never file contents, and always exits 0.

## Sub-features

- `hook-edit` appends one record for an Edit of a file.
- `hook-ignore` records nothing for a tool it does not track (for example `Bash`).
- `hook-off` records nothing when `NEO_HOOKS=0`.

## How to get to it (user POV)

- Editing a file in Claude Code, Codex or Cursor with the Neo plugin installed.
- Run `echo '<payload json>' | neo hook record`.

## Driving it with verify.sh

Preconditions:

- Baseline state from the README.
- Shell variable set: `R=/tmp/neo-verify-runs/$ID`.

- **Edit.** Run `printf '{"tool_name":"Edit","tool_input":{"file_path":"%s/repo/src/billing/invoice.py"},"cwd":"%s/repo","session_id":"verify-s1"}' "$R" "$R" | $V neo "$ID" hook-record hook record`. Prints `exit=0`.
- **Proof.** Read `$R/home/.neo/sessions/host_events.jsonl`. It has one line with `"tool": "Edit"`, the fixture `file_path`, `"session_id": "verify-s1"`, and `head` equal to `git -C "$R/repo" rev-parse HEAD`. No file contents appear. Run `$V keep "$ID" home/.neo/sessions/host_events.jsonl` so the ledger survives cleanup.
- **Ignored tool.** Run `printf '{"tool_name":"Bash","tool_input":{"command":"ls"}}' | $V neo "$ID" hook-bash hook record`. Prints `exit=0`; the ledger still has one line.

## Gotchas

- Malformed JSON also exits 0 by design. Exit code alone proves nothing; read the ledger.
- `hook-off` cannot be driven through the helper's `env -i` without adding `NEO_HOOKS=0` to `neo_env`; report it as not driven rather than proving it another way.
