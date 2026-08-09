---
name: neo-architect
description: Get Neo's architectural guidance for design decisions. Trade-off analysis for choices like microservices vs monolith, sync vs async, and event-driven vs request-response, with optional explicitly authorized memory recall.
---

# Neo Architectural Guidance

When the user invokes this skill (`$neo-architect <question>`), do the following:

1. **Restate the architectural question precisely.** "Should I use X or Y for Z?" with concrete constraints (scale expected, team size, existing stack, latency budget) yields better answers than open-ended questions.

2. **Gather codebase context.** Read `CLAUDE.md`, `AGENTS.md`, `README.md`, top-level config files, and any architecture docs under `docs/`. Neo's own context-assembly will pick these up too, but having you summarize the existing constraints up front helps.

3. **Apply the provider boundary.** Redact secrets, credentials, tokens,
   cookies, and session material. Before an external-provider call, tell the
   user which Neo provider will receive which files or data categories.
   Production, private, or customer architecture material requires explicit
   authorization for that provider and scope.

4. **Invoke Neo with an architecture-framed prompt.** Allow up to 5 minutes.
   Use Codex's approval flow for required network access, naming the provider
   and summarized data in the approval description.

   ```bash
   neo --json --no-scan --no-memory --mode advise <<'QUERY'
   Architectural decision: <restate the question with constraints>.

   Current state of the codebase:
   <summarize tech stack, scale, team>

   Provide a recommendation with trade-offs explicit, plus alternatives ranked by fit.
   QUERY
   ```

5. **Present Neo's plan and simulations together.** Architecture answers benefit from the SIMULATIONS section especially — those describe how the recommendation would actually play out.

6. **Surface any architectural facts Neo retrieved from memory.** If past projects had similar decisions, Neo references them — those are higher-trust than fresh reasoning.

`--no-scan` is mandatory: Codex already summarized the relevant architecture
context, so Neo must not silently add working-directory files to the provider
request.
`--no-memory` is the default so unrelated stored facts cannot enter the provider
prompt. Omit it only when the user explicitly authorizes relevant Neo memory;
include stored facts as a disclosed data category.

## Reading Neo's output

Invoke with `--json`. stdout is exactly one JSON document; stderr is JSONL
progress events (parse lines starting with `{`, ignore the rest). Never parse
the human-readable text output. On failure stdout is `{"error": ...}` with no
`orchestrator` key — check for `error` first.

Lead with `orchestrator.summary`, surface every entry in
`orchestrator.cautions`, and relay `orchestrator.personality` verbatim when
present. Neo writes in the first person and his register shifts with how much
he remembers about this project — keep his wording rather than translating it
into yours. See the `$neo` skill for the full contract.

**Attribute explicitly and keep going.** You are calling Neo inside your own
coding loop, so nothing marks where his reasoning ends and yours begins — say
"Neo found …" and keep your own analysis in your own voice. His result is an
input, not the deliverable: continue the task and report the combined outcome.

For architecture specifically: name the **alternatives considered and why each
lost** — a recommendation without its rejected options is an opinion, not
guidance, and `hypothesis_rejected` events plus the plan's rationale fields
carry it. State the tradeoff Neo accepted in the user's terms: what gets harder
if they follow this advice. Be explicit that this is advice — context Neo
cannot see (team, timeline, politics) routinely dominates.

## Notes

- This skill uses `advise`; architecture guidance is never promoted automatically. Deliberate policy learning requires explicit `learn` mode and stronger evidence.
- Neo will not recommend "it depends" — it picks a default and explains the trade-off. If the user wants ambiguity preserved, ask them to phrase the question as "what are the trade-offs of X vs Y?" rather than "should I do X or Y?".
