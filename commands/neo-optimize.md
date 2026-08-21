---
description: "Get optimization suggestions from Neo"
---

Get optimization suggestions from Neo.

## Usage

```
/neo-optimize <file path or function name>
```

## Description

Use this command to get performance optimization recommendations from Neo using semantic analysis and past optimization patterns.

## Examples

```
/neo-optimize process_large_dataset function

/neo-optimize src/data/processor.py

/neo-optimize the search algorithm
```

## What Happens

Neo will:
1. Analyze the code for algorithmic complexity
2. Identify bottlenecks and inefficiencies
3. Search memory for similar optimization patterns
4. Suggest improvements with confidence scores

## Presentation

Invoke with `--json` and follow the agent's communication protocol
(`agents/neo.md`). For optimization specifically:

- Lead with `orchestrator.summary`, then the identified bottleneck.
- Show the **evidence** for the bottleneck — the complexity argument or the
  code path Neo pointed at. An optimization claim with no evidence is a guess
  wearing a confidence score.
- State the expected impact and its basis. If Neo estimated rather than
  measured, say "estimated". Neo does not execute or benchmark anything.
- Recommend the user measure before and after. Any suggested benchmark
  commands are advisory and are never run by Neo.
- Surface `orchestrator.cautions`, especially correctness risks — a faster
  wrong answer is a regression.

## Parameters

This command uses read-only `advise` mode. Suggested benchmarks and commands are
advisory and are never executed by Neo.

- `<target>` - File path or function name (required)

Optionally include performance requirements (e.g., "needs <2s for 10k records")
