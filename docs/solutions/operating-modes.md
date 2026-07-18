# Operating modes and authority

**Status:** implemented shared contract

Neo separates repository authority from memory authority. Reading established
memory does not imply permission to learn, and producing a patch does not imply
permission to apply it.

| Mode | Inference | Repository result | Deterministic checks | Candidate/outcome learning | Execution |
|---|---|---|---|---|---|
| `advise` | yes | analysis and suggestions | generated suggestions may be checked | no | never |
| `patch` | yes | applicable patch/code artifacts | generated suggestions may be checked | no | never |
| `verify` | no | evaluates caller-provided changes | yes | no | never |
| `learn` | yes | read-only suggestions | yes | yes, through evidence pipeline | never |
| `agent` | yes | host-controlled | yes | only if authority allows | host adapter only |

`learn` is the backward-compatible standalone default. It may write Neo's local
episode/session/fact stores, but it never writes repository files or executes
generated commands. Codex and Claude analysis skills explicitly request
`advise`; the pattern-teaching skill explicitly requests `learn`.

## Verify mode

VERIFY requires `proposed_changes` in JSON/A2A input and builds deterministic
`CodeSuggestion` check inputs from them. It bypasses planning and generation,
uses a sentinel adapter that raises if inference is attempted, and records the
check result in a `verification_complete` episode. Missing changes and changes
without content fail before inference.

```json
{
  "prompt": "verify this patch",
  "operating_mode": "verify",
  "proposed_changes": [
    {"file_path": "src/app.py", "code_block": "value = 1"}
  ]
}
```

## Agent authority

Neo has no built-in repository or shell executor. An embedding host must pass a
typed execution adapter plus an explicit `AuthorityPolicy` containing:

- an absolute `workspace_root`;
- one or more workspace-relative `allowed_write_paths` globs;
- optional exact `allowed_commands` for enforcement by that host adapter;
- whether the resulting episode may participate in learning.

Neo resolves every generated and reported action path, rejects workspace
escapes and paths outside the allowlist before delegation, and persists the
public authority summary plus returned action evidence. Generated
`apply_command`, `test_command`, and `rollback_command` strings remain advisory;
Neo never invokes them. Standalone CLI and the default CAR host provide no
executor, so `agent` fails closed before provider inference.

All modes still create a bounded local LearningEpisode so a request is
inspectable. Only `learn`, or `agent` with `allow_learning=true`, performs
outcome detection, creates candidates, or writes session attribution.

