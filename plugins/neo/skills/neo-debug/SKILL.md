---
name: neo-debug
description: Ask Neo to help debug intermittent, complex, or hard-to-reproduce issues. Particularly useful for race conditions, memory issues, distributed-systems bugs, and cases where the symptom is not the root cause.
---

# Neo Debug Assistant

When the user invokes this skill (`$neo-debug <bug description>`), do the following:

1. **Capture the bug context tightly.** Symptom, when it started, frequency, environment, and any reproduction steps. Vague bug reports get vague answers; Neo's memory is keyed on specifics.

2. **Gather observable evidence.** Recent error logs, stack traces, test failures. Read the file(s) implicated by the stack trace. If the user mentioned a specific function, locate it.

3. **Invoke Neo with a debug-framed prompt.** Allow up to 5 minutes.

   ```bash
   neo <<'QUERY'
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

4. **Present Neo's hypotheses ranked by confidence.** Lead with the verification step the user can take next — debugging is an iterative loop, not a one-shot answer.

5. **If Neo returns multiple competing hypotheses, surface them all.** Don't collapse to "the most likely one" — concurrent-systems bugs often have multiple contributing causes.

## Notes

- Race conditions, memory issues, and intermittent failures are where Neo's failure-pattern memory pays off most — past similar bugs add weight to matching hypotheses.
- If Neo's top hypothesis has confidence < 0.6, treat it as "worth investigating" rather than "probably right." Debugging is harder than greenfield reasoning; lower confidence is normal.
