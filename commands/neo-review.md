---
description: >-
  Get Neo's code review with semantic matching against past findings in
  memory. Use on a diff or module before merge, especially where earlier
  mistakes in this codebase are likely to recur. Skip for formatting,
  lint-catchable issues, and single-line changes.
---

Get Neo's code review with semantic analysis.

## Usage

```
/neo-review <file path or code description>
```

## Description

Use this command to get Neo's code review with semantic pattern matching, security analysis, and optimization suggestions.

## Examples

```
/neo-review src/api/handlers.py

/neo-review authentication module

/neo-review the payment processing code
```

## What Happens

Neo will:
1. Gather context from the specified file or module
2. Analyze code for security vulnerabilities, edge cases, and performance issues
3. Check semantic memory for similar code review patterns
4. Provide improvements with confidence scores

This command uses read-only `advise` mode and performs no repository writes or
durable learning updates.

## Presentation

Invoke with `--json` and follow the agent's communication protocol
(`agents/neo.md`). For a review specifically:

- Lead with `orchestrator.summary`, then the findings themselves.
- Group findings by severity, not by file. A security issue and a naming nit
  are not peers.
- Surface `risk_found` events and every entry in `orchestrator.cautions` —
  a review that reads as clean because the caveats were trimmed is worse than
  no review.
- State which assumptions Neo challenged. "Neo disagreed with the existing
  retry logic" is the useful half of a review.
- Give the confidence number. A 0.55 review is a starting point for your own
  reading, not a verdict — say so.

## Parameters

- `<target>` - File path, module name, or code description (required)

Focus areas: security, edge cases, error handling, performance
