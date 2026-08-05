---
description: "Get debugging assistance from Neo"
---

Get debugging assistance from Neo.

## Usage

```
/neo-debug <error message or bug description>
```

## Description

Use this command when debugging complex issues, especially intermittent bugs or race conditions. Neo uses semantic pattern matching to identify likely root causes.

## Examples

```
/neo-debug TypeError in data processing pipeline

/neo-debug Race condition in concurrent task processor

/neo-debug Memory leak happening after 1000+ requests
```

## What Happens

Neo will:
1. Analyze the error or bug description
2. Identify likely root causes using semantic patterns
3. Search memory for similar debugging scenarios
4. Suggest debugging strategies and fixes with confidence scores

## Presentation

Invoke with `--json` and follow the agent's communication protocol
(`agents/neo.md`). For debugging specifically:

- Lead with `orchestrator.summary`, then the leading hypothesis.
- Report what was **ruled out**, not just what was concluded. Every
  `hypothesis_rejected` event and every simulation issue in `risk_found`
  narrows the search for the user — that is the substance of a debugging
  session.
- Say what evidence supports the leading hypothesis and what would falsify it.
- If `memory_found` fired, say so: a prior occurrence of this failure is the
  single most useful thing Neo can contribute.
- Never present a hypothesis as a diagnosis. Neo does not run the code.

## Parameters

This command uses read-only `advise` mode. It does not apply fixes, execute
commands, or update durable memory.

- `<description>` - Error message or bug description (required)

Include: error messages, frequency, reproduction steps, environment details
