---
name: fix-ci
pattern: /fix-ci
description: Fix CI check failures on a PR by analyzing failure logs and implementing fixes
parameters:
  - name: pr_url
    description: GitHub PR URL
    required: true
argument-hint: <pr_url>
---

# Fix CI Failures

Analyze and fix CI check failures on a pull request.

## Workflow

### Step 1: Get CI Check Failures

Use @sentinel to fetch CI failure details:

1. Run `gh pr checks $PR_URL --json name,state,bucket,link,description`
2. For each failed check (bucket="fail"), get details
3. If available, fetch failure logs from the check link

### Step 2: Analyze Failures

Categorize the failures:
- **Test failures**: Which tests failed and why
- **Build errors**: Compilation or bundling issues
- **Lint errors**: Code style or static analysis issues
- **Other**: Deployment, security scans, etc.

### Step 3: Generate Fix Plan

Create a focused fix plan:
- List each failure with its cause
- Propose specific code fixes
- Prioritize by severity

### Step 4: Call /implement

Pass the fix plan to /implement to apply the fixes.

## Final Output (STRICT JSON)

```json
{
  "fixed": true|false,
  "failures_found": ["check-name-1", "check-name-2"],
  "fixes_applied": ["description of fix 1", "description of fix 2"],
  "summary": "Brief summary of what was fixed or why it couldn't be fixed"
}
```
