---
description: "Get architectural guidance from Neo on design decisions"
---

Get architectural guidance from Neo on design decisions.

## Usage

```
/neo-architect <your architectural question>
```

## Description

Use this command when you need help making architectural decisions. Neo will analyze tradeoffs, provide confidence scores, and reference similar systems from its memory.

## Examples

```
/neo-architect Should I use microservices or monolith for this project?

/neo-architect What's the best way to handle real-time notifications? WebSockets vs SSE vs polling?

/neo-architect How should I structure a multi-tenant SaaS database?
```

## What Happens

Neo will:
1. Analyze tradeoffs between different approaches
2. Consider scalability, maintainability, and complexity
3. Search memory for similar architectural decisions
4. Provide recommendations with confidence scores and risk analysis

## Presentation

Invoke with `--json` and follow the agent's communication protocol
(`agents/neo.md`). For architecture specifically:

- Lead with `orchestrator.summary`, then the recommendation.
- Name the **alternatives considered and why each lost**. An architectural
  recommendation without its rejected options is an opinion, not guidance.
  `hypothesis_rejected` events and the plan's rationale fields carry this.
- State the tradeoff Neo accepted, in the user's terms: what gets harder if
  they follow this advice.
- Surface `orchestrator.cautions` in full. Architecture mistakes are the
  expensive kind to reverse.
- Be explicit that this is advice, not a decision. Architectural context Neo
  cannot see — team, timeline, politics — routinely dominates.

## Parameters

This command uses `advise` mode. Architecture guidance is not automatically
promoted into policy or durable architecture memory.

- `<question>` - Your architectural question (required)

Include constraints: scalability needs, team size, infrastructure, timeline
