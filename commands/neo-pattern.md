---
description: >-
  Extract a reusable pattern from a solution that has proven itself, and
  record it in Neo's durable memory. Use deliberately, when a lesson should
  inform later runs. Skip for one-off code and speculative abstractions —
  this writes persistent memory, so a wrong lesson outlives the session.
---

Extract reusable patterns with Neo.

## Usage

```
/neo-pattern <code area or pattern type>
```

## Description

Use this command to extract reusable architectural patterns from your codebase. Neo will identify patterns, analyze their effectiveness, and suggest when to apply them.

## Examples

```
/neo-pattern repository pattern implementation

/neo-pattern error handling across the API

/neo-pattern caching strategies in the codebase
```

## What Happens

Neo will:
1. Analyze code to identify the pattern
2. Evaluate pattern effectiveness and consistency
3. Search memory for similar patterns
4. Suggest improvements or alternative patterns with confidence scores

This command uses explicit `learn` mode. The result is recorded only as an
episode candidate and requires independent verified support before promotion.

## Presentation

Invoke with `--json` and follow the agent's communication protocol
(`agents/neo.md`). For pattern extraction specifically:

- Lead with `orchestrator.summary`, then the pattern Neo identified.
- Cite the instances the pattern was drawn from. A "pattern" with one example
  is an anecdote — say how many occurrences Neo actually saw.
- Report `memory_found` when the pattern matched something Neo already knew.
  Recurrence across sessions is the strongest signal available here.
- Be accurate about learning: this run records an episode **candidate**. It
  does not mint a durable fact. Promotion needs two independent
  git-verified acceptances. Do not tell the user Neo "learned" this.

## Parameters

- `<target>` - Code area or pattern type (required)

Specify what pattern you want to analyze or where to look for patterns
