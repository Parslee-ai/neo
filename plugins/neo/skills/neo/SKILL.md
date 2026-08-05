---
name: neo
description: Ask Neo for semantic reasoning and code suggestions over the current codebase. Use for general questions, code suggestions, or architectural guidance backed by Neo's persistent memory.
---

# Neo — Semantic Reasoning Helper

When the user invokes this skill (`$neo <question or task>`), do the following:

1. **Verify Neo is installed.** Run `neo --version` once. If the command is missing, tell the user: "Neo CLI not installed. Run `pip install neo-reasoner[openai]` and set `OPENAI_API_KEY`, then retry." Stop.

2. **Say what you are delegating.** One sentence, before the call: the user should know what question is in flight before a 5–30 second pause.

3. **Gather context.** Use your file-reading tools to collect the most relevant files for the user's question (typically 1–5 files). Include the user's full question text.

4. **Invoke Neo with `--json`.** Allow up to 5 minutes — Neo runs multi-agent reasoning across LLM calls.

   ```bash
   neo --json --mode advise <<'QUERY'
   <restate the user's question here, plus any short context excerpts>
   QUERY
   ```

   `--json` controls output only; a plain-text heredoc is still read as text.

5. **Read both streams.** Never parse Neo's human-readable text output — it is formatted for a terminal reader and omits the fields below.

## Output contract

**stdout** is exactly one JSON document. **stderr** is JSONL progress events,
one object per line. `--json` implies `--quiet`, so stderr is essentially pure
JSONL, but logging warnings can still appear — parse lines beginning with `{`
and ignore the rest. Do not discard stderr with `2>/dev/null`.

The stdout document carries an `orchestrator` object built for exactly this
purpose:

| Field | Use |
|---|---|
| `summary` | Lead with it. Neo's own account of what he did and concluded. |
| `personality` | Optional in-voice beat. Relay verbatim when present. |
| `phase_summary` | Per-phase records. Use to explain a slow or unusual run. |
| `cautions` | **Never drop these.** Low confidence, failed checks, open questions. |
| `recommended_narration` | Advisory progress lines. Reword freely. |

Useful event types on stderr: `memory_found` (prior facts recalled),
`hypothesis_formed`, `hypothesis_rejected` (an approach was discarded — usually
worth surfacing), `risk_found`, and exactly one of `completed` or `failed`
terminating every run. Phase names are stable: `context`, `reasoning`,
`static_checks`.

**On failure**, stdout is an error object with no `orchestrator` key:
`{"error": "RequestTimeout", "message": "...", "suggestions": [...]}`. Check for
`error` first. Say plainly that Neo did not complete, and do not substitute your
own analysis while implying it came from Neo.

## Communicating Neo's answer

1. Present `orchestrator.summary` before the detailed answer.
2. Surface every entry in `orchestrator.cautions`.
3. Mention significant rejected hypotheses and risks from the event stream.
   What Neo ruled out is often more useful than what he settled on.
4. Do not dump raw traces, full simulation output, or the event stream itself.
5. **Attribute explicitly.** You call Neo inside your own coding loop and keep
   working afterwards, so there is no visible boundary between his reasoning
   and yours. Without attribution the user will assume every word was yours.
   Write "Neo's take: …" or "Neo found …", and keep your own analysis in your
   own voice.
6. **Neo's result is an input, not the deliverable.** Continue the task —
   inspect files, make the change, run the tests — and report the combined
   outcome. `recommended_next_action` in the JSON names a concrete starting
   point.

## Neo's voice

Neo speaks in the first person, clipped and direct. His register shifts with how
much he remembers about this project: a hedge at low memory ("Don't know this
code… , maybe") is information, and terseness at high memory ("src/parser.py.
1 change(s). 0.88.") is the same signal inverted. Relay his wording rather than
translating it into your own register — that is also what keeps the attribution
legible. Cautions deliberately do **not** vary by register; a warning reads the
same however terse Neo is, and must not be softened.

`orchestrator.personality` is an additional beat, present only when Neo's beat
deck matched the situation and, for beats claiming insight, only when the run
actually found something. It is already filtered — relay it verbatim, attached
to the substance that earned it, or say nothing when the field is empty. There
is no fallback line.

## Notes

- This skill uses explicit `advise` mode: it may retrieve established memory but does not create candidates or update durable memory. Use `neo --mode learn` only when the user intends to contribute outcome evidence.
- For code review, optimization, architectural decisions, debugging, or pattern extraction, prefer the more specific Neo skills (`$neo-review`, `$neo-optimize`, `$neo-architect`, `$neo-debug`, `$neo-pattern`).
- Always verify low-confidence suggestions (< 0.7) before applying them.
