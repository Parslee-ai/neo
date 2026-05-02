---
name: neo
description: Ask Neo for semantic reasoning and code suggestions over the current codebase. Use for general questions, code suggestions, or architectural guidance backed by Neo's persistent memory.
---

# Neo — Semantic Reasoning Helper

When the user invokes this skill (`$neo <question or task>`), do the following:

1. **Verify Neo is installed.** Run `neo --version` once. If the command is missing, tell the user: "Neo CLI not installed. Run `pip install neo-reasoner[openai]` and set `OPENAI_API_KEY`, then retry." Stop.

2. **Gather context.** Use your file-reading tools to collect the most relevant files for the user's question (typically 1–5 files). Include the user's full question text.

3. **Invoke Neo via stdin heredoc.** Allow up to 5 minutes — Neo runs multi-agent reasoning across LLM calls.

   ```bash
   neo <<'QUERY'
   <restate the user's question here, plus any short context excerpts>
   QUERY
   ```

4. **Parse Neo's output.** Neo returns four structured sections: `CONFIDENCE`, `PLAN`, `SIMULATIONS`, `CODE SUGGESTIONS`. Each suggestion carries its own confidence score.

5. **Present results to the user.** Lead with Neo's overall confidence and the highest-confidence suggestion. Quote Neo's reasoning verbatim where it's load-bearing. If confidence is below 0.7, flag it explicitly.

## Notes

- Neo learns from every session. Familiar queries return faster (Neo automatically picks lower reasoning effort when memory hits cleanly) and confidence rises over time.
- For code review, optimization, architectural decisions, debugging, or pattern extraction, prefer the more specific Neo skills (`$neo-review`, `$neo-optimize`, `$neo-architect`, `$neo-debug`, `$neo-pattern`).
- Always verify low-confidence suggestions (< 0.7) before applying them.
