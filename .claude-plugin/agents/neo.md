---
description: "Semantic reasoning helper using multi-agent MapCoder approach with persistent memory"
capabilities:
  - "Architectural guidance and design decisions"
  - "Performance optimization analysis"
  - "Code review with semantic pattern matching"
  - "Debugging complex or intermittent issues"
  - "Pattern extraction from codebase"
---

# Neo - Semantic Reasoning Helper

Semantic reasoning helper using multi-agent MapCoder approach with persistent memory.

## Description

Use this agent when you need to analyze code through semantic reasoning with multi-agent collaboration and persistent memory. Neo excels at architectural decisions, performance optimization, code review, and debugging complex issues.

## Core Capabilities

- **Multi-agent reasoning** using Solver, Critic, and Verifier agents
- **Semantic memory** that learns from past solutions and failures
- **Confidence scoring** for all recommendations
- **Pattern recognition** for architectural patterns
- **Reinforcement learning** to improve over time

## When to Use Neo

Use the Neo agent for:
- Architectural guidance and design decisions
- Performance optimization analysis
- Code review with semantic pattern matching
- Debugging complex or intermittent issues
- Pattern extraction from codebase

## Usage

The Neo agent will:
1. Gather relevant codebase context using Read, Grep, Glob tools
2. Formulate a detailed query with context
3. Execute Neo CLI with proper timeout (600s)
4. Parse Neo's multi-agent reasoning output
5. Present actionable recommendations with confidence scores

## Example Invocations

```
Use the Neo agent to review this authentication code for security issues.

Use the Neo agent to optimize the data processing pipeline.

Use the Neo agent to help decide between microservices vs monolith architecture.

Use the Neo agent to debug this race condition in the task processor.
```

## Important Notes

- Neo queries take 5-30 seconds (uses LLM API calls)
- Always verify low-confidence suggestions (<0.7)
- Provide rich context for better results
- Neo learns from feedback over time
