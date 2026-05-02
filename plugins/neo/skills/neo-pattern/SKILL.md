---
name: neo-pattern
description: Ask Neo to extract a reusable pattern from a piece of code, or to find existing patterns in the codebase that match a description. Useful for codifying conventions and finding duplicated logic.
---

# Neo Pattern Extraction

When the user invokes this skill (`$neo-pattern <code reference or description>`), do the following:

1. **Determine direction.** Is the user asking neo to:
   (a) **Extract** a pattern *from* a piece of code they're pointing at? — gather the code, then ask Neo to articulate the reusable pattern.
   (b) **Find** instances of a pattern *in* the codebase based on a description? — gather the description, then ask Neo to locate matching code.

2. **For extraction:** read the source code the user referenced. Include enough surrounding context that the pattern is intelligible.

3. **For pattern-finding:** translate the user's description into search terms. Use Grep/Glob to gather candidate files; pass them to Neo for semantic matching against the description.

4. **Invoke Neo with a pattern-framed prompt.** Allow up to 5 minutes.

   ```bash
   neo <<'QUERY'
   <Extract a reusable pattern from> | <Find code matching this pattern>:

   <code or description here>

   Articulate: name, signature/shape, when to apply, when NOT to apply, common pitfalls.
   QUERY
   ```

5. **Present the pattern with concrete examples.** A named pattern with two example sites is more useful than an abstract description of one.

## Notes

- Patterns Neo extracts are stored in its semantic memory and retrieved automatically on future related queries — so this skill is one of the highest-leverage ways to teach Neo about your codebase.
- Patterns extracted from a single example are "PROVISIONAL" until Neo sees them confirmed in another part of the codebase. The user should treat single-example patterns as drafts.
