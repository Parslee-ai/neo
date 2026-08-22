# Host hooks for outcome detection

**Status**: recording half shipped; consuming half not started. See *Status* below.

## The problem, as measured

Neo's promote path needs two git-verified acceptances of a matching diff shape,
spanning two distinct `repository_revision`s. `CLAUDE.md` records what that path
has actually produced on a live install:

- 30 days of real traffic: **108 episodes, 58 stuck at
  `suggested_pending_downstream_outcome`, zero `accepted` outcomes ever**.
- Across **6,613 valid facts in every project, zero have ever reached
  `success_count >= 3`**, so `find_contributable` has never returned a fact and
  `neo contribute` has never been reachable.
- `neo --version` on this machine still prints *"No patterns yet"*.

Every gate downstream of acceptance has been audited, fixed and pinned. The
input to those gates is what is missing: acceptance is *inferred*, after the
fact, from git.

## Why inference is the weak link

`collect_outcomes` runs on the **next** Neo invocation and asks git what changed
since the suggestion. Three consequences, all documented in `CLAUDE.md` and none
fixable inside that design:

1. **No next invocation, no outcome.** A user who takes Neo's advice and moves on
   generates nothing. The July drill only worked "because the operator applied
   the diff with no intervening run."
2. **The revision is captured when the episode BEGINS** — HEAD when advice was
   asked for, not the commit the fix landed in. So applying one lesson across
   several files in one sitting records ONE revision and promotes nothing;
   **40% of revision-bearing episodes share a HEAD with another.** CLAUDE.md
   already names the fix: *"Keying on the acceptance-carrying sha — already
   walked by `_get_changed_files_since` — is the obvious improvement."*

   **A hook does not fix this, and an earlier draft of this note claimed it
   did.** Recording HEAD at edit time moves the stamp closer to the acceptance,
   but two applications of one lesson in a single sitting still share a HEAD if
   no commit happened between them — which is exactly the 40% case. The
   acceptance-carrying sha is knowable only after the commit, and Claude Code's
   event list has no commit event to hang that on. The ledger's `head` field is
   therefore *better evidence, not a solution*: it records where the tree
   actually stood when the edit landed, instead of where it stood when the
   question was asked. Closing defect (2) still needs the commit-walk CLAUDE.md
   describes, and that work is independent of this one.
3. **`_get_changed_files_since` returns every file changed anywhere in the
   repo**, so acceptance is a path-intersection guess, not an observation. One
   unrelated dirty file has already cost a real acceptance once.

A host hook does not replace the git verification. It supplies the one fact git
cannot: **that this specific file was edited, at this moment, at this HEAD**.

## What the payload actually carries

Confirmed against a shipped implementation rather than documentation —
`car-rs/crates/car-cli/src/policy_hook.rs` parses a Claude Code `PreToolUse`
payload and reads:

| Field | Use here |
|---|---|
| `tool_name` | `Edit` / `Write` / `MultiEdit` / `NotebookEdit` |
| `tool_input.file_path` | the edited path |
| `cwd` | **the host's project directory** — the hook process's own cwd is not necessarily the project, and CAR's code carries that warning explicitly |

`session_id` and `tool_output`/`tool_response` are also present per the hooks
reference; this design does not depend on them and they should be confirmed
empirically before any code reads them.

## Shape, as shipped

`hooks/hooks.json` at the **plugin root** (not `.claude-plugin/` — see
`tests/test_host_adapter_parity.py`), declared as `"hooks": "./hooks/hooks.json"`
in `.claude-plugin/plugin.json`:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write|MultiEdit|NotebookEdit",
        "hooks": [{ "type": "command", "command": "neo hook record", "timeout": 10 }]
      }
    ]
  }
}
```

`neo hook record` reads the payload on stdin and appends one line to
`~/.neo/sessions/host_events.jsonl`:

```json
{"ts": 1755800000.1, "tool": "Edit", "file_path": "src/neo/engine.py",
 "cwd": "...", "head": "a1b2c3d", "session_id": "..."}
```

`head` is `git rev-parse HEAD` **at edit time** — where the tree stood when the
edit landed, rather than when the question was asked. See the correction under
defect (2): this improves the stamp, it does not close that defect.

There is no `dirty` field. An earlier draft had one; it needs `git status
--porcelain`, which is the expensive call CLAUDE.md already measured at 0.88 s
of pure forking on the request path, and nothing needs it.

`collect_outcomes` is *intended* to read this ledger as evidence alongside git,
instead of inferring acceptance from a repo-wide diff. It does not yet.

## Constraints the implementation must respect

**Budget: under 100 ms.** Measured on this machine:

| | |
|---|---|
| `import neo.cli` | **0.04 s** |
| `neo --version` | **0.36 s** — constructs a `FactStore` |

The second number is the whole constraint. `neo hook record` must return
**before any `FactStore` is constructed**; it fires on every edit, and 0.36 s
per edit is not a tax worth paying for telemetry. This argues for an early
dispatch in `cli.main`, ahead of store setup, and a test that pins it.

**Fail open, always exit 0.** A `PostToolUse` hook that exits non-zero surfaces
as an error against a tool call that already succeeded. We are living the
cautionary example while writing this: CAR's hook documents that it "fails open
on anything it cannot evaluate — a missing binary — because a hook that failed
closed on an install problem would present it as a policy denial and wedge a
session with no obvious cause." That guarantee did not cover *binary present,
subcommand absent*: clap rejected the argv and exited non-zero before any
fail-open code ran, and it wedged every `Bash`/`Write`/`Edit` call in this
session. **Neo's hook must therefore be exit-0 on every path, including
unparseable stdin, no git, and an unwritable `~/.neo`.**

**This is not the removed session-id fallback.** `CLAUDE.md` is explicit that a
non-git independence signal "was written and REMOVED — do not reintroduce it."
Hook evidence is not that: it does not invent a notion of independence, it
records a real edit at a real revision. Promotion must continue to require two
acceptances spanning two distinct revisions — the hook improves *which* revision
gets recorded, never *whether* the span is required.

**Record paths, never contents.** CAR's hook filters `tool_input` down to
identifying string parameters precisely so file contents "must not leave the
process just to decide whether a call is allowed." The same rule applies here
with more force, since this one writes to disk.

## Status

**Shipped (recording half):** `neo hook record`, `hooks/hooks.json`, the
`"hooks"` declaration in `.claude-plugin/plugin.json`, and `tests/test_hook.py`.
Verified end to end at **0.06 s** per invocation against `neo --version`'s
0.36 s, exit 0 on every path, `claude plugin details` reporting
`Hooks (1) PostToolUse (harness-only — no model context cost)`.

**Not shipped (consuming half):** nothing reads `host_events.jsonl` yet.
`collect_outcomes` still infers acceptance from a repo-wide git diff.

The split is deliberate rather than partial delivery: a ledger is only worth
reading once it has history, and history cannot be recorded retroactively, so
the recorder has to land first regardless of when the consumer does. What the
consumer should claim is narrower than the first draft of this note implied —
see defect (2). The durable win is that the ledger cannot lose an acceptance
that git later obscures (a revert, a reformat, a commit that moves the
timestamp window), because it witnessed the edit rather than reconstructing it.

**The consuming half must be written against a dirty tree.** CLAUDE.md records
the precedent exactly: a previous fix here "passed every test because they all
ran on a pristine tree, the one state neo is never invoked in", and it lost
acceptances in production anyway.

## What this does not solve

- **Codex parity.** Codex plugins also support `hooks/`, but its advise skills
  run `--no-scan --no-memory`, so there is no suggestion to accept. Sequencing
  T4 before the Codex question is deliberate.
- **Attribution to a *Neo* suggestion.** The hook says a file changed; matching
  it to an outstanding suggestion is still `_unresolved_suggestions`' job. This
  makes that matching exact instead of repo-wide, and nothing more.
- **The 30-day backlog.** 58 pending episodes stay pending; this only improves
  what is recorded from here on.

## Open questions

1. Should `SessionEnd` also trigger a collection pass, or is reading the ledger
   on the next invocation sufficient? A collection pass costs a `FactStore`
   load at session end.
2. Should the hook ship **enabled** by default? It writes to `~/.neo` on every
   edit in every project where the plugin is installed, which is a real change
   in behaviour for anyone who installs Neo for the reasoning alone.
3. Does `tool_response` carry enough to distinguish a successful edit from a
   rejected one? If not, the ledger records attempts, and `dirty`/`head` do the
   disambiguation later.
