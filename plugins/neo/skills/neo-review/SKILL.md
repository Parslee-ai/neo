---
name: neo-review
description: Get Neo's code review with semantic pattern matching. Focuses on security vulnerabilities, edge cases, error handling, and performance issues across the target file or module.
---

# Neo Code Review

When the user invokes this skill (`$neo-review <file or module>`), do the following:

1. **Identify the target.** It will usually be a file path (e.g. `src/api/handlers.py`), a module name, or a free-form description ("the payment processing code"). Use Read/Grep/Glob to resolve it to concrete file(s).

2. **Read the relevant code.** Up to 5 files at a time keeps Neo's context budget healthy. Prefer the files where the actual logic lives over generated/test files.

3. **Invoke Neo with a review-framed prompt.** Allow up to 5 minutes.

   ```bash
   neo <<'QUERY'
   Review the following code for: security vulnerabilities, edge cases, error handling, performance issues. Provide concrete suggestions with confidence scores.

   <paste relevant code or summarize what you read>
   QUERY
   ```

4. **Filter Neo's output to review-relevant findings.** Group by severity. Flag any finding with confidence ≥ 0.8 as actionable; treat lower-confidence findings as worth-checking-but-verify.

5. **Cross-reference with Neo's KNOWN ISSUES IN NEARBY CODE section if present.** Neo's context-assembly already surfaces TODOs, stubs, swallowed exceptions, hardcoded credentials — those overlap with review concerns and add weight to related findings.

## Notes

- Neo's confidence scores reflect both LLM self-assessment and pattern-match strength against past reviews in semantic memory.
- For security-critical code, escalate findings the user pushes back on — Neo's memory is updated with outcomes, so consistent rejections will demote weak patterns over time.
