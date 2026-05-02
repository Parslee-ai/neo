---
name: neo-architect
description: Get Neo's architectural guidance for design decisions. Trade-off analysis for choices like microservices vs monolith, sync vs async, event-driven vs request-response — with persistent memory of how similar decisions played out.
---

# Neo Architectural Guidance

When the user invokes this skill (`$neo-architect <question>`), do the following:

1. **Restate the architectural question precisely.** "Should I use X or Y for Z?" with concrete constraints (scale expected, team size, existing stack, latency budget) yields better answers than open-ended questions.

2. **Gather codebase context.** Read `CLAUDE.md`, `AGENTS.md`, `README.md`, top-level config files, and any architecture docs under `docs/`. Neo's own context-assembly will pick these up too, but having you summarize the existing constraints up front helps.

3. **Invoke Neo with an architecture-framed prompt.** Allow up to 5 minutes.

   ```bash
   neo <<'QUERY'
   Architectural decision: <restate the question with constraints>.

   Current state of the codebase:
   <summarize tech stack, scale, team>

   Provide a recommendation with trade-offs explicit, plus alternatives ranked by fit.
   QUERY
   ```

4. **Present Neo's plan and simulations together.** Architecture answers benefit from the SIMULATIONS section especially — those describe how the recommendation would actually play out.

5. **Surface any architectural facts Neo retrieved from memory.** If past projects had similar decisions, Neo references them — those are higher-trust than fresh reasoning.

## Notes

- Architecture decisions are where Neo's persistent memory pays off most. The same question asked across multiple projects gradually accumulates trade-off learnings.
- Neo will not recommend "it depends" — it picks a default and explains the trade-off. If the user wants ambiguity preserved, ask them to phrase the question as "what are the trade-offs of X vs Y?" rather than "should I do X or Y?".
