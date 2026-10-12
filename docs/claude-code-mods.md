# Neo workspace in Claude Code

Neo's Mod adds a direct workspace for reasoning, proposals and project memory.
The TypeScript module handles native UI and process transport. The Python CLI
owns context selection, inference, evidence and learning policy.

## Setup

Requires Claude Code terminal 2.1.287+ or Desktop 2.1.286+ with Mods support,
and Neo CLI 0.56.0+ matching the plugin version. Older CLI releases do
not implement `neo workspace`; updating only the plugin is insufficient.
The existing six delegated commands and Neo agent remain available.

For development, install the CLI in a persistent virtual environment and load
this checkout:

```sh
python -m venv .venv
.venv/bin/pip install -e '.[dev,openai]'
NEO_MOD_EXECUTABLE="$PWD/.venv/bin/neo" claude --plugin-dir "$PWD"
```

Configure your normal Neo provider and credentials before inference. The Mod
uses the plugin's `executable` user configuration (default `neo`); the
`NEO_MOD_EXECUTABLE` environment variable overrides it. The legacy command hook
still resolves `neo` from PATH, so keep that installation current too.

A distributed installation uses `/plugin marketplace add Parslee-ai/neo` then
`/plugin install neo`. Install or upgrade the CLI with `pip install -U neo-reasoner` and keep the
plugin and CLI versions aligned.

## Use

| Command | Behavior |
| --- | --- |
| `/neo-workspace` | Open Results, Memory and Evidence tabs |
| `/neo-ask [reason\|review\|debug\|architect\|optimize\|pattern] <question>` | Start direct advisory reasoning |
| `/neo-followup <question>` | Link a new analysis to the prior result in this session |
| `/neo-status` | Show current progress, or read project status without inference |
| `/neo-explain <project-fact-id-or-prefix>` | Inspect fact provenance and outcome evidence without inference |
| `/neo-cancel` | Stop the workspace's current invocation |

The pane offers the same six categories, question and context inputs, and a
**Use selection** control. Neo selects bounded repository context using its
existing Python gatherer and retrieves memory under its existing rules.
A follow-up includes a bounded copy of the prior findings and selected context;
its execution identity links to the prior task. **Share with Claude** fills an
editable draft; the user decides whether to send it.

Results show Neo's summary, every caution, notes, proposals, checks and validation
counts before optional reasoning details. Evidence distinguishes completion of
analysis from validation, application and independent acceptance. A failed,
skipped, unavailable or missing check is never displayed as a pass.

The pane uses native host scrolling and focus navigation. Results, Memory and
Evidence have `r`, `m`, `e` hotkeys; Ask, Follow up, details and Cancel use `a`,
`f`, `d`, `c`. Narrow panes stack navigation. The host handles text-input focus
and scrolling. While a run is active, a compact progress band adds to the host's
existing band. In VS Code and headless sessions the direct commands return text;
headless reasoning waits for completion and returns a nonzero exit on failure.

## Privacy, authority and learning

Opening a workspace, reading status and explaining a project fact make zero
model calls. They read persisted project files without initializing FactStore,
aging facts, embedding text, changing metrics, migrating storage or starting an
observer. Org/global facts and other projects are excluded from the inspector.
An analysis can retrieve other authorized scopes through Neo's normal engine;
those rows show their scope and cannot be opened through the project inspector.

Direct requests (`/neo-ask`, the pane) use `advise`: Neo proposes changes and
does not apply them, execute generated command strings, detect outcomes or
create learning candidates from the advisory run.

### Neo joining in on its own

Neo also takes part without being asked, at four moments (`hooks/auto.ts`):

| When | What Neo does | Model call | How Claude gets it |
| --- | --- | --- | --- |
| You type a prompt | `neo recall`: up to 3 learned lessons matching it | none | context on that prompt |
| A turn that edited files is about to end | reviews those edits (`git diff HEAD` per path; a new or untracked file as a whole-file addition) | one, up to 90 s | the turn continues with Neo's fix, only when it carries a diff or code |
| A test, build or lint command fails | diagnoses the failure, in the background | one | context at the next main-thread hook |
| A plan leaves plan mode | checks the plan | one, up to 90 s | context as the work starts |

- **Recall** makes no model call and never loads the embedding model: it asks
  the observer's resident embedder once, with a 1 s deadline, and says nothing
  if it does not answer. It reads fact files without constructing a store, so
  it writes nothing, and it leaves out what Claude already has (rule files,
  its own memory files, git history) along with generic seed and community
  patterns. It runs only for prompts a person typed, not machine-injected ones.
- **Reviews, diagnoses and plan checks** are real inference on Neo's configured
  provider: they send the diff, error output or plan, plus the source and
  memory Neo selects, and incur its normal cost. They are bounded by
  deduplication, not by a switch: one review per distinct set of edits, one
  diagnosis per distinct failure (command plus error), at most one running and
  six per session, one check per plan. A turn that any Stop hook continued is
  not reviewed then; its edits are kept and reviewed at the next Stop that is
  not a continuation. Only a review with a concrete change continues a
  turn; a diagnosis never does.
- **Subagents** contribute edits to the review but never trigger a diagnosis
  or consume findings meant for the main thread. A diagnosis that finishes
  after its session ended (`/clear`, resume) is dropped.
- **They run in `learn` mode.** Neo's suggestions are recorded, so when Claude
  applies one, git can verify the acceptance and the usual promotion rules
  apply. No task type is fixed; the CLI classifies the prompt.
- Findings reach Claude framed as data from a second model, not instructions,
  and with the same relay rule as the Neo agent: quote Neo's lines as written.
  His summary and recall's opener are phrased by his memory stage (`memory_stage`
  of `memory_level_of`, shared by the engine and `neo recall`; openers are
  `recall_opener` in `neo_matrix.yaml`), so the hedge at Sleeper and the
  terseness at The One are information, and the earned personality beat is
  shown in the pane and in every text form. A failed or timed-out run adds
  nothing and never blocks the turn.

The Mod does not intercept permissions, auto-approve tools, send prompts as the
user, or rewrite repository instructions.

Successful Edit/Write/MultiEdit/NotebookEdit observations and turn/invocation
lifecycle records use the existing host ledger. Records retain paths, checkout,
revision and host identity, never tool bodies, prompts or answers. The latest
Neo invocation is explicitly **recency correlation**, not causal attribution.
The outcome consumer preserves that diagnostic alongside its existing git-based
matching. Lifecycle completion cannot establish an edit. Same session/tool-call
IDs deduplicate Mod and legacy hook delivery; separate calls remain distinct.
A busy ledger lock has a 200 ms bound and drops telemetry rather than blocking
a successful host tool call. `NEO_HOOKS=0` disables both recorder paths.

No promotion or rollback rules change: generated suggestions, simulated success,
confidence and retrieved facts cannot substitute for deterministic verification
and independent acceptance. Read-only explanations use the existing provenance
join, including probation, tombstones, supporting/conflicting episodes,
supersession, metric mutations and rollback.

## Failure and lifecycle handling

Each reasoning invocation owns one `process.spawn` stream with a four-minute
deadline. Cancellation closes that stream; it never kills a process group.
Session end (/clear, resume, logout or exit) cancels the owned invocation and
clears its in-memory context and results. New requests capture their own session
and checkout identities. Module unload is also governed by the host's owned
process lifetime.

Progress is JSONL on stderr; final output is JSON on stdout. Chunk boundaries
and malformed progress lines are tolerated. Acceptance requires a completed
lifecycle, valid final result and successful process exit. Output exceeding
4 MiB fails explicitly. Missing executables, version drift, provider errors,
incomplete JSON, nonzero exits and timeouts remain visible, and later requests
can retry. Project inspector corruption is reported without quarantine/writes.

## Validation

```sh
claude plugin validate . --json
claude plugin test .
python -m pytest tests/test_workspace.py tests/test_hook.py tests/test_recall.py \
  tests/test_host_edit_ledger.py tests/test_memory_explain.py \
  tests/test_operating_modes.py tests/test_host_adapter_parity.py \
  tests/test_orchestrator_events.py -q
```

An interactive Claude Code session generates `.claude-plugin/types/` containing
its API declarations and TypeScript configuration; these generated files are
ignored by Git. `tsc -p tsconfig.json` uses them. The declarations are not plugin
components, which still live at the repository root.

Mod tests exercise both terminal and Desktop rendering. Real Desktop acceptance
requires an installed Desktop host and remains a separate check from rendering
tests. See [Anthropic's overview](https://code.claude.com/docs/en/plugins/mods/overview)
and [example Mods](https://github.com/anthropics/claude-code/tree/main/mods).
