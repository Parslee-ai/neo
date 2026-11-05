# Recall

`neo recall` reads the user's prompt on stdin and returns lessons from memory the host should show. The Claude Code mod calls it on every typed prompt. It makes no model call, writes nothing, and always exits 0.

## Sub-features

- `recall-empty` returns an empty result on an empty store.
- `recall-hit` returns a stored lesson matching the prompt.
- `recall-never-fails` exits 0 on an unknown flag or a bad `--cwd`.

## How to get to it (user POV)

- Typing a prompt in Claude Code with the Neo mod installed.
- Run `echo "<prompt>" | neo recall --json`.

## Driving it with verify.sh

Preconditions:

- Baseline state from the README.

- **Empty store.** Run `echo "late fee rounding invoice" | $V neo "$ID" recall recall --json`. Prints `exit=0`; stdout is `{"text": "", "beat_id": "", "fact_ids": []}`.
- **Unknown flag never fails.** Run `echo x | $V neo "$ID" recall-badflag recall --json --no-such-flag`. Prints `exit=0`; stdout is empty and stderr has an argparse usage error.
- **Bad `--cwd` never fails.** Run `echo x | $V neo "$ID" recall-badcwd recall --json --cwd /nonexistent`. Prints `exit=0` with the empty result JSON.
- **Hit.** Not drivable from the baseline. See Gotchas.

## Gotchas

- `recall-hit` needs a lesson-carrying fact in the scratch store, and the baseline seeds none. There is no user-facing seeding command that creates one without an LM run (`memory import` facts enter on probation). Report `recall-hit` as unreachable with this reason; do not seed by writing fact JSON by hand and call it verified.
- Session context comes from env vars (`NEO_RECALL_TRANSCRIPT`, `NEO_RECALL_SEEN`), never flags. The helper's `env -i` drops them unless you add them.
