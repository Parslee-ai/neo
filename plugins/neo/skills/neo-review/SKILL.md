---
name: neo-review
description: Get Neo's code review with semantic pattern matching. Focuses on security vulnerabilities, edge cases, error handling, and performance issues across the target file or module. Skip for formatting, lint-catchable issues, and single-line changes.
---

# Neo Code Review

When the user invokes this skill (`$neo-review <file or module>`), do the following:

1. **Identify the target.** It will usually be a file path (e.g. `src/api/handlers.py`), a module name, or a free-form description ("the payment processing code"). Use Read/Grep/Glob to resolve it to concrete file(s).

2. **Read the relevant code.** Up to 5 files at a time keeps Neo's context budget healthy. Prefer the files where the actual logic lives over generated/test files.

3. **Apply the provider boundary.** Redact secrets, credentials, tokens,
   cookies, and session material. Before an external-provider call, tell the
   user which Neo provider will receive which files or data categories.
   Production, private, or customer code requires explicit authorization for
   that provider and scope; invoking the skill is not blanket consent.

4. **Invoke Neo with a review-framed prompt.** Allow up to 5 minutes. Use
   Codex's approval flow for required network access, naming the provider and
   summarized data in the approval description.

   ```bash
   neo --json --no-scan --no-memory --mode advise <<'QUERY'
   Review the following code for: security vulnerabilities, edge cases, error handling, performance issues. Provide concrete suggestions with confidence scores.

   <paste relevant code or summarize what you read>
   QUERY
   ```

5. **Filter Neo's output to review-relevant findings.** Group by severity. Flag any finding with confidence ≥ 0.8 as actionable; treat lower-confidence findings as worth-checking-but-verify.

6. **Cross-reference with Neo's KNOWN ISSUES IN NEARBY CODE section if present.** Neo's context-assembly already surfaces TODOs, stubs, swallowed exceptions, hardcoded credentials — those overlap with review concerns and add weight to related findings.

`--no-scan` is mandatory: Codex already selected the review files, so Neo must
not silently add working-directory files to the provider request.
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

For a review specifically: group findings by severity, not by file — a security
issue and a naming nit are not peers. Surface `risk_found` events alongside the
cautions; a review that reads as clean because the caveats were trimmed is
worse than no review. Say which assumptions Neo challenged, and give the
confidence number rather than rounding it up in prose.

## Notes

- Neo's confidence scores reflect both LLM self-assessment and pattern-match strength against past reviews in semantic memory.
- For security-critical code, escalate findings the user pushes back on — Neo's memory is updated with outcomes, so consistent rejections will demote weak patterns over time.
