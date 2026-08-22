---
name: neo-pattern
description: Ask Neo to extract a reusable pattern from a piece of code, or to find existing patterns in the codebase that match a description. Useful for codifying conventions and finding duplicated logic. Skip for one-off code and speculative abstractions — this is the one Neo skill that writes durable memory, so a wrong lesson outlives the session.
---

# Neo Pattern Extraction

When the user invokes this skill (`$neo-pattern <code reference or description>`), do the following:

1. **Determine direction.** Is the user asking neo to:
   (a) **Extract** a pattern *from* a piece of code they're pointing at? — gather the code, then ask Neo to articulate the reusable pattern.
   (b) **Find** instances of a pattern *in* the codebase based on a description? — gather the description, then ask Neo to locate matching code.

2. **For extraction:** read the source code the user referenced. Include enough surrounding context that the pattern is intelligible.

3. **For pattern-finding:** translate the user's description into search terms. Use Grep/Glob to gather candidate files; pass them to Neo for semantic matching against the description.

4. **Apply the provider and learning boundary.** Redact secrets, credentials,
   tokens, cookies, and session material. Before an external-provider call,
   tell the user which Neo provider will receive which files or data categories.
   Production, private, or customer code requires explicit authorization for
   that provider and scope. Because this learning workflow uses shared Neo
   memory, also disclose that relevant stored facts may be selected for the
   provider prompt and that `learn` records an episode candidate.

5. **Invoke Neo with a pattern-framed prompt.** Allow up to 5 minutes. Use
   Codex's approval flow for required network access, naming the provider,
   summarized data, and candidate-learning effect in the approval description.

   ```bash
   neo --json --no-scan --mode learn <<'QUERY'
   <Extract a reusable pattern from> | <Find code matching this pattern>:

   <code or description here>

   Articulate: name, signature/shape, when to apply, when NOT to apply, common pitfalls.
   QUERY
   ```

6. **Present the pattern with concrete examples.** A named pattern with two example sites is more useful than an abstract description of one.

`--no-scan` is mandatory: Codex already selected the pattern evidence, so Neo
must not silently add working-directory files to the provider request.

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

For pattern extraction specifically: cite the instances the pattern was drawn
from — a "pattern" with one example is an anecdote, so say how many occurrences
Neo actually saw. Report `memory_found` when the pattern matched something Neo
already knew; recurrence across sessions is the strongest signal available here.
Be accurate about learning: this run records an episode **candidate**, not a
durable fact. Promotion needs two independent git-verified acceptances. Do not
tell the user Neo "learned" this.

## Notes

- `learn` records the extraction as an episode-local candidate. It is not trusted memory until independently verified and supported again.
- Patterns extracted from a single example are "PROVISIONAL" until Neo sees them confirmed in another part of the codebase. The user should treat single-example patterns as drafts.
