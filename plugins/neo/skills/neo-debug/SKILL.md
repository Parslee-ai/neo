---
name: neo-debug
description: Ask Neo to help debug intermittent, complex, or hard-to-reproduce issues. Particularly useful for race conditions, memory issues, distributed-systems bugs, and cases where the symptom is not the root cause.
---

# Neo Debug Assistant

When the user invokes this skill (`$neo-debug <bug description>`), do the following:

1. **Capture the bug context tightly.** Symptom, when it started, frequency, environment, and any reproduction steps. Vague bug reports get vague answers; Neo's memory is keyed on specifics.

2. **Gather observable evidence.** Recent error logs, stack traces, test failures. Read the file(s) implicated by the stack trace. If the user mentioned a specific function, locate it.

3. **Apply the provider boundary.** Redact secrets, credentials, tokens,
   cookies, session material, and unrelated log records. Before an
   external-provider call, tell the user which Neo provider will receive which
   files or data categories. Production, private, or customer diagnostics
   require explicit authorization for that provider and scope.

4. **Invoke Neo with a debug-framed prompt.** Allow up to 5 minutes. Use
   Codex's approval flow for required network access, naming the provider and
   summarized data in the approval description.

   ```bash
   neo --json --no-scan --no-memory --mode advise <<'QUERY'
   Debug this issue: <user's description>

   Symptoms: <what happens>
   Environment: <relevant context — concurrency model, OS, runtime, dependencies>
   Stack trace / logs:
   <paste evidence>

   Relevant code:
   <paste the function or module under suspicion>

   Provide ranked hypotheses about root cause with reasoning. For each, suggest a verification step.
   QUERY
   ```

5. **Present Neo's hypotheses ranked by confidence.** Lead with the verification step the user can take next — debugging is an iterative loop, not a one-shot answer.

6. **If Neo returns multiple competing hypotheses, surface them all.** Don't collapse to "the most likely one" — concurrent-systems bugs often have multiple contributing causes.

`--no-scan` is mandatory: Codex already selected the evidence, so Neo must not
silently add working-directory files to the provider request.
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

For debugging specifically: report what was **ruled out**, not just what was
concluded. Every `hypothesis_rejected` event and every simulation issue in
`risk_found` narrows the search — that is the substance of a debugging session.
Say what evidence supports the leading hypothesis and what would falsify it. If
`memory_found` fired, say so: a prior occurrence of this failure is the single
most useful thing Neo can contribute. Never present a hypothesis as a
diagnosis — Neo does not run the code, so verifying it is your job.

## Notes

- Race conditions, memory issues, and intermittent failures are where Neo's failure-pattern memory pays off most — past similar bugs add weight to matching hypotheses.
- If Neo's top hypothesis has confidence < 0.6, treat it as "worth investigating" rather than "probably right." Debugging is harder than greenfield reasoning; lower confidence is normal.
