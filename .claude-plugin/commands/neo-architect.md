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

## Parameters

- `<question>` - Your architectural question (required)

Include constraints: scalability needs, team size, infrastructure, timeline
