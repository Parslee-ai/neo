---
description: >-
  Ask Neo for semantic reasoning over this codebase — multi-agent analysis
  with persistent memory. Use when a question spans several files or past
  decisions, or when you want a second independent opinion before committing
  to an approach. Skip for questions answerable from the file already open,
  factual lookups, and mechanical edits; each run costs 5-30s and an LLM
  call.
---

Ask Neo for semantic reasoning and code suggestions.

## Usage

```
/neo <your question or task>
```

## Description

Use this command when you need Neo's semantic reasoning for general questions, code suggestions, or architectural guidance.

## Examples

```
/neo How should I structure my new feature for user analytics?

/neo What's the best caching strategy for this API?

/neo Should I use REST or GraphQL for this data model?
```

## What Happens

Neo will:
1. Analyze your question using multi-agent reasoning (Solver, Critic, Verifier)
2. Search semantic memory for similar problems
3. Generate solutions with confidence scores
4. Provide actionable recommendations

This command uses `advise` mode. It reads and retrieves memory but does not create
learning candidates, modify repository files, or execute commands.

## Presentation

Invoke with `--json` and follow the agent's communication protocol
(`agents/neo.md`):

- Say what you are asking Neo before invoking it.
- Lead with `orchestrator.summary`, then the answer.
- Surface every entry in `orchestrator.cautions`.
- Relay `orchestrator.personality` verbatim when present, attached to the
  finding that earned it. Say nothing in its place when it is absent.
- Attribute clearly. "Neo found…" and "Looking at this myself…" are different
  claims with different reliability, and the user needs to be able to tell
  them apart.

## Parameters

- `<question>` - Your question or task description (required)

Provide as much context as possible for better results.
