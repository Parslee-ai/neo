---
name: pr-review
pattern: /pr-review
description: Review a GitHub pull request for bugs, security issues, and code quality. Fetches complete PR context including description and all comments before performing a thorough code review. Posts review results as a PR comment tagging the author.
parameters:
  - name: pr_url
    description: GitHub PR URL (e.g., https://github.com/owner/repo/pull/123) or PR number for current repo
    required: true
  - name: base_branch
    description: Base branch to diff against (e.g., main, develop). Defaults to repository default branch if not specified.
    required: false
  - name: description
    description: Additional context about what this PR should accomplish. Supplements the PR description fetched from GitHub.
    required: false
argument-hint: <pr_url> [base_branch] [description]
---

# PR Review

Review a GitHub pull request for bugs, security issues, and code quality with complete context awareness.

## Variables

PR_URL: $1
BASE_BRANCH: $2
DESCRIPTION: $3

## Workflow

### Step 1: Use @sentinel to Fetch PR Context

Fetch complete PR context using the gh CLI:

1. **PR Metadata**: Run `gh pr view $PR_URL --json body,title,author,labels,baseRefName,number`
2. **PR Review Comments**: Run `gh api repos/{owner}/{repo}/pulls/{number}/comments`
3. **Review Threads**: Run `gh api repos/{owner}/{repo}/pulls/{number}/reviews`
4. **Issue Comments**: Run `gh api repos/{owner}/{repo}/issues/{number}/comments` (conversation tab)

Extract the following critical information:
- `baseRefName` - This is the base branch for the PR
- `author.login` - This is the PR author's GitHub username

Format the output as a comprehensive PR Context document that includes:
- **Title**: The PR title
- **Author**: @mention format (e.g., @username)
- **Base Branch**: The branch this PR targets
- **Labels**: Any labels applied to the PR
- **PR Description**: The full body of the PR
- **Conversation & Comments**: All comments from all sources, organized chronologically

This context will be passed to the code reviewer in Step 2.

### Step 2: Use @brutal-reviewer to Perform Code Review

Pass the following information to @brutal-reviewer:

**Input:**
- PR URL: $PR_URL
- Base Branch: Use $BASE_BRANCH if provided by user, otherwise use the baseRefName from Step 1
- User-Provided Context: Use $DESCRIPTION if provided by user, otherwise "None provided"
- Full PR Context: The complete context document from Step 1

**Instructions for brutal-reviewer:**

Review this GitHub pull request with full context:

**PR URL**: [PR_URL from above]
**Base Branch**: [BASE_BRANCH from above]
**User Context**: [DESCRIPTION from above]
**Full PR Context**: [Complete context from Step 1]

**Your Task:**
1. Checkout the PR using gh CLI or git fetch
2. Generate diff against base branch using git merge-base to find common ancestor
3. Review ONLY the changed lines (lines with '+' prefix in the diff)
4. Use the PR description and conversation context to understand the intent and context

**Report ONLY:**
- Actual bugs (logic errors, null pointer issues, incorrect algorithms, race conditions)
- Security vulnerabilities (injection attacks, auth bypass, data leaks, XSS, CSRF)
- Breaking changes that appear unintentional

**DO NOT Report:**
- Style preferences or formatting issues
- Pre-existing code issues (unless the changes make them worse)
- Theoretical edge cases without evidence they'll occur
- Suggestions that aren't fixing actual bugs
- Nitpicks about naming or organization

For each real issue found:
- Provide file:line reference
- Show code snippet demonstrating the problem
- Explain why it's wrong (not just "could be better")
- Suggest a specific fix

**Output Format:**

```
## Code Review

### Critical Issues
[List actual bugs and security vulnerabilities, or state "None found"]

### Code Quality Problems
[List real quality issues that impact maintainability or correctness, or state "None found"]

### Line-by-Line Feedback
[file:line] - [Problem description] - [Specific fix]

### Verdict
[APPROVE / NEEDS WORK / REJECT]
[One sentence justification for the verdict]
```

---

### Step 3: Use @sentinel to Post Review Comment

Post the review results as a PR comment using the gh CLI:

Run: `gh pr comment $PR_URL --body "[FORMATTED_REVIEW]"`

Format the comment body as follows:
- Start with "## Code Review by Claude"
- Tag the PR author (use the author.login from Step 1 in @mention format)
- Include the full review output from @brutal-reviewer
- Ensure proper GitHub markdown formatting

**IMPORTANT**: After posting the comment, capture the comment URL by running:
`gh api repos/{owner}/{repo}/issues/{number}/comments --jq '.[-1].html_url'`

This URL is needed for subsequent /gh-review-assess calls.

## Report Format

After all steps complete, provide a summary report to the user:

**PR Review Complete**

- **PR URL**: [url]
- **Author**: @[username]
- **Verdict**: [APPROVE/NEEDS WORK/REJECT]

**Summary**: [One paragraph summarizing the review findings]

**Issue Counts**:
- Critical Issues: [count]
- Code Quality Problems: [count]

**Action Taken**: Posted review as PR comment

**Full Review Details**:
[Include the complete review output from brutal-reviewer]

## Final Output (STRICT JSON)

After completing all steps, output ONLY this JSON structure as your final response:

```json
{
  "verdict": "APPROVED|CHANGES_REQUESTED|COMMENTED",
  "findings": [
    {"severity": "critical|high|medium|low", "file": "path/to/file.py", "line": 42, "description": "Issue description"}
  ],
  "review_comment_url": "https://github.com/owner/repo/pull/123#issuecomment-456789"
}
```

Where:
- `verdict`: One of "APPROVED", "CHANGES_REQUESTED", or "COMMENTED"
- `findings`: Array of issues found (empty array if none)
- `review_comment_url`: The URL to the PR comment that was posted (required for /gh-review-assess)
