---
name: gh-review-assess
pattern: /gh-review-assess
description: Automates the GitHub code review assessment workflow by fetching a PR comment or review from GitHub, using the Linus agent to evaluate which issues are legitimate vs nitpicking, and using the linus-kernel-planner agent to create pragmatic implementation plans for the valid issues. This command saves significant time when dealing with code reviews that may contain overly pedantic or unnecessary feedback, focusing effort only on changes that genuinely improve the codebase.
parameters:
  - name: comment_url
    description: |
      GitHub PR comment or review URL. Supports multiple formats:
      - Issue comment: https://github.com/owner/repo/pull/123#issuecomment-456789
      - Review comment: https://github.com/owner/repo/pull/123#discussion_r987654
      - Full review: https://github.com/owner/repo/pull/123#pullrequestreview-112233
      Must be a valid GitHub URL from a pull request. The gh CLI must be authenticated with sufficient permissions to read the repository and PR comments.
    required: true
allowed-tools: Task
---

## Usage

`/gh-review-assess <comment_url>`

Where:
- `comment_url` (required): GitHub PR comment, review comment, or full review URL

### Examples

#### Example 1: Basic Assessment of Single Comment

**Scenario**: You receive a code review comment suggesting that you refactor a working function to use a more "elegant" pattern. You want to assess whether this feedback is valuable or just stylistic preference.

**Command**:
```
/gh-review-assess https://github.com/acme/project/pull/456#issuecomment-789012
```

**Expected Behavior**:
1. Fetches the comment content from GitHub using gh CLI
2. Presents the comment to @agent-Linus for evaluation
3. Linus categorizes the feedback as legitimate vs nitpicking
4. For legitimate issues, @agent-linus-kernel-planner creates a simple implementation plan
5. Returns structured summary showing accepted/rejected issues with rationale

**Output**:
```
REVIEW ASSESSMENT COMPLETE
=========================

Original Comment:
"This function should use map/reduce instead of a for loop. It's more functional."

LINUS EVALUATION:
Status: REJECTED
Category: Style Nitpicking
Rationale: The existing for loop is clear, performant, and maintainable.
Changing to map/reduce provides no tangible benefit and may reduce readability
for team members less familiar with functional patterns. This is textbook
bikeshedding.

VERDICT: No action required.
```

**Notes**:
- If the comment contains multiple distinct issues, Linus will evaluate each separately
- The command uses gh CLI's JSON API to fetch comment metadata and content
- Authentication is required via `gh auth login` before first use

#### Example 2: Multi-Issue Review with Mixed Validity

**Scenario**: A reviewer posts a comment with five separate concerns. Some are valid security issues, others are stylistic preferences. You need to quickly determine which ones deserve attention.

**Command**:
```
/gh-review-assess https://github.com/acme/backend/pull/234#issuecomment-567890
```

**Expected Behavior**:
1. Extracts all distinct issues from the comment
2. Linus evaluates each issue independently
3. Valid issues are passed to linus-kernel-planner
4. Returns comprehensive breakdown showing decisions for each issue

**Output**:
```
REVIEW ASSESSMENT COMPLETE
=========================

Original Comment contained 5 issues:

ISSUE 1: "This endpoint doesn't validate user input"
-------------------------------------------------
LINUS EVALUATION: ACCEPTED
Category: Security Issue
Priority: HIGH
Rationale: User input validation is non-negotiable. This is a legitimate
security concern that could lead to injection attacks or data corruption.

IMPLEMENTATION PLAN (via linus-kernel-planner):
1. Add input validation middleware to endpoint (2 hours)
   - Validate required fields exist
   - Check field types and formats
   - Return 400 for invalid input
2. Add unit tests for validation cases (1 hour)
3. Test manually with malformed requests (30 min)

ISSUE 2: "Variable name 'data' is too generic"
---------------------------------------------
LINUS EVALUATION: REJECTED
Category: Nitpicking
Rationale: The variable 'data' is scoped to 5 lines and its purpose is
immediately clear from context. Renaming provides zero value and wastes time.

ISSUE 3: "Should extract this into a separate function"
------------------------------------------------------
LINUS EVALUATION: REJECTED
Category: Premature Abstraction
Rationale: This code appears once in the codebase. Creating an abstraction
before you have multiple use cases is premature optimization and violates YAGNI.
Wait until there's a second use case.

ISSUE 4: "Missing error handling for database connection failure"
----------------------------------------------------------------
LINUS EVALUATION: ACCEPTED
Category: Reliability Issue
Priority: MEDIUM
Rationale: Database failures will happen in production. Without proper error
handling, the application will crash or hang. This is a legitimate concern.

IMPLEMENTATION PLAN (via linus-kernel-planner):
1. Wrap database call in try/catch (15 min)
2. Log error with context (15 min)
3. Return 503 Service Unavailable to client (15 min)
4. Add integration test for DB failure scenario (1 hour)

ISSUE 5: "Should add JSDoc comments"
-----------------------------------
LINUS EVALUATION: REJECTED
Category: Documentation Preference
Rationale: The function is 8 lines, has a descriptive name, and clear
parameter names. The code is self-documenting. Adding JSDoc would be more
lines than the actual code and provides no value.

SUMMARY
=======
Accepted Issues: 2 (Issues #1, #4)
Rejected Issues: 3 (Issues #2, #3, #5)

Total Implementation Time: ~5-6 hours
Wasted Time Avoided: ~2-3 hours on non-issues

RECOMMENDED ACTION:
Respond to reviewer thanking them for catching the validation and error
handling issues. Implement those two fixes. Politely push back on the other
three items as outside the scope of this PR.
```

**Notes**:
- The command automatically identifies distinct issues within a single comment
- Time estimates help prioritize and schedule the work
- The recommended action provides diplomatic language for responding to reviewers

#### Example 3: Full Pull Request Review Assessment

**Scenario**: A senior engineer performed a full review of your PR with 15 comments across multiple files. You want to quickly assess which ones are worth addressing before requesting re-review.

**Command**:
```
/gh-review-assess https://github.com/acme/frontend/pull/789#pullrequestreview-112233
```

**Expected Behavior**:
1. Fetches the entire review including all comments
2. Groups comments by file and issue type
3. Linus evaluates each comment
4. Creates consolidated implementation plan for all accepted issues
5. Provides summary statistics

**Output**:
```
FULL REVIEW ASSESSMENT
=====================

Reviewer: @senior-dev
Files with Comments: 5
Total Comments: 15

ANALYSIS BY FILE
================

src/components/UserForm.tsx (6 comments)
-----------------------------------------
✓ ACCEPTED: "Missing PropTypes validation" (Priority: HIGH)
✓ ACCEPTED: "No error boundary, will crash parent on error" (Priority: HIGH)
✗ REJECTED: "Should use useCallback for handlers" (Premature optimization)
✗ REJECTED: "Component file too long, split into smaller pieces" (Arbitrary rule)
✗ REJECTED: "Add data-testid to every element" (Testing implementation detail)
✓ ACCEPTED: "Form submission doesn't disable button" (UX Issue)

src/api/users.ts (4 comments)
------------------------------
✓ ACCEPTED: "No timeout on API call" (Reliability)
✗ REJECTED: "Use axios interceptors instead" (Unnecessary refactor)
✗ REJECTED: "Add retry logic" (Feature creep, not in scope)
✓ ACCEPTED: "Error messages expose internal details" (Security)

src/utils/validation.ts (3 comments)
------------------------------------
✗ REJECTED: "Use Zod instead of manual validation" (Unnecessary dependency)
✗ REJECTED: "Extract regex to constants file" (Over-engineering)
✓ ACCEPTED: "Email validation regex has bug" (Correctness)

src/hooks/useUserData.ts (2 comments)
-------------------------------------
✗ REJECTED: "Use React Query instead" (Major refactor, wrong venue)
✓ ACCEPTED: "Race condition in state updates" (Correctness bug)

SUMMARY STATISTICS
==================
Total Issues: 15
Accepted: 7 (47%)
Rejected: 8 (53%)

Breakdown:
- HIGH Priority: 2
- MEDIUM Priority: 3
- LOW Priority: 2

CONSOLIDATED IMPLEMENTATION PLAN
=================================

Phase 1: Critical Issues (Must fix before merge)
1. Add PropTypes validation to UserForm (30 min)
2. Fix race condition in useUserData (1 hour)
3. Fix email validation regex bug (15 min)
4. Sanitize error messages in API layer (45 min)

Phase 2: Important Issues (Fix before merge if time permits)
5. Add error boundary around UserForm (1 hour)
6. Add timeout to API calls (30 min)
7. Disable submit button during form submission (15 min)

Total Estimated Time: 4-5 hours

RECOMMENDED RESPONSE
====================

"Thanks for the thorough review! I've addressed the following critical issues:
- PropTypes validation
- Race condition in useUserData hook
- Email validation bug
- Error message sanitization
- Added error boundary
- API timeout
- Form button state management

Regarding the other suggestions:
- useCallback/React Query/Zod: I agree these would be improvements, but they
  represent significant refactors beyond this PR's scope. Let's discuss in a
  separate planning session.
- Code organization suggestions: Appreciate the input, but current structure
  follows team conventions and is maintainable.
- Testing implementation: We prefer testing behavior over implementation details.

Ready for re-review!"
```

**Notes**:
- Full review assessment provides file-by-file breakdown
- Groups similar issues for easier mental model
- Provides ready-to-use response template
- Separates critical from nice-to-have fixes

#### Example 4: Handling Review Thread with Multiple Responses

**Scenario**: A review comment has turned into a discussion thread with multiple back-and-forth responses. You want to assess the entire thread to understand the core issue.

**Command**:
```
/gh-review-assess https://github.com/acme/api/pull/345#discussion_r876543
```

**Expected Behavior**:
1. Fetches the root comment and all replies in the thread
2. Synthesizes the discussion into core concerns
3. Linus evaluates the underlying issue, not just the surface discussion
4. Provides clarity on the actual technical decision needed

**Output**:
```
REVIEW THREAD ASSESSMENT
========================

Thread Summary:
- Root Comment: Reviewer suggests using dependency injection
- 3 replies debating implementation patterns
- Discussion veered into abstract architecture debate

LINUS ANALYSIS:

Core Issue Identified:
"The function has a hard-coded database connection string, making it
impossible to test without hitting the real database."

Discussion Quality: Low
Rationale: The thread devolved into abstract architecture discussions about
dependency injection patterns, IoC containers, and framework philosophy.
Nobody focused on solving the actual problem: untestable code.

LINUS EVALUATION: ACCEPTED (but simplified)
Category: Testability Issue
Priority: MEDIUM

Real Problem: Hard-coded dependency makes testing difficult
Proposed Solution: Complex DI framework
Linus Solution: Pass the connection as a parameter

IMPLEMENTATION PLAN (via linus-kernel-planner):
1. Add connection parameter to function (5 min)
   - function processUser(userId, connection) { ... }
2. Update all call sites to pass connection (15 min)
3. Write unit test with mock connection (30 min)

Total Time: 50 minutes

VERDICT:
The reviewers identified a real problem but proposed an over-engineered
solution. Follow the simple approach: make the dependency explicit via
parameters. No frameworks needed.

RECOMMENDED RESPONSE:
"Good catch on the testability issue! I've made the database connection
an explicit parameter. This solves the testing problem without adding
framework complexity. Tests now use a mock connection."
```

**Notes**:
- The command synthesizes multi-message threads into core issues
- Linus cuts through discussion noise to find the real problem
- Often reveals that a simple solution exists for a complex-sounding problem
- Helps avoid getting drawn into unproductive architectural debates

#### Example 5: Error Handling - Invalid URL Format

**Scenario**: You accidentally pass a malformed URL or non-PR URL to the command.

**Command**:
```
/gh-review-assess https://github.com/acme/project/issues/123
```

**Expected Behavior**:
1. Validates URL format before making any API calls
2. Provides clear error message
3. Shows examples of valid URL formats

**Output**:
```
ERROR: Invalid URL Format
========================

The provided URL does not appear to be a valid GitHub pull request comment URL.

Received: https://github.com/acme/project/issues/123

Valid formats:
- Issue comment: https://github.com/owner/repo/pull/123#issuecomment-456789
- Review comment: https://github.com/owner/repo/pull/123#discussion_r987654
- Full review: https://github.com/owner/repo/pull/123#pullrequestreview-112233

Notes:
- URL must be from a pull request, not an issue
- URL must include a comment/review fragment identifier (#issuecomment-...)
- Repository must be accessible with your current gh CLI authentication

To check your authentication:
  gh auth status

To find the correct URL:
  1. Navigate to the PR on GitHub
  2. Click on the specific comment or review
  3. Copy the URL from your browser's address bar
```

**Notes**:
- Early validation prevents wasted API calls
- Error messages are instructional, not just descriptive
- Includes debugging steps for authentication issues

#### Example 6: Private Repository Access

**Scenario**: You're working on a private repository and need to assess a code review. The gh CLI must be properly authenticated.

**Command**:
```
/gh-review-assess https://github.com/acme/private-repo/pull/567#issuecomment-890123
```

**Expected Behavior**:
1. Attempts to fetch comment using gh CLI
2. If authentication fails, provides clear instructions
3. If successful, proceeds with normal assessment workflow

**Output** (if auth fails):
```
ERROR: Authentication Required
=============================

Unable to access https://github.com/acme/private-repo/pull/567

Possible causes:
1. You're not authenticated with gh CLI
2. You don't have access to this private repository
3. Your authentication token has expired

To fix:

If not authenticated:
  gh auth login

If authenticated but no access:
  - Verify you have read access to acme/private-repo
  - Check with repository administrators

To verify current authentication:
  gh auth status

To test repository access:
  gh repo view acme/private-repo
```

**Output** (if auth succeeds):
```
[Proceeds with normal assessment as shown in previous examples]
```

**Notes**:
- The command respects GitHub's access controls
- Works seamlessly with both public and private repos if properly authenticated
- Provides actionable troubleshooting steps for auth failures

#### Example 7: Review Comment with Code Suggestions

**Scenario**: A reviewer provides inline code suggestions using GitHub's suggestion feature. You want to assess whether to accept, modify, or reject the suggestions.

**Command**:
```
/gh-review-assess https://github.com/acme/webapp/pull/678#discussion_r123456
```

**Expected Behavior**:
1. Extracts both the comment text and any code suggestions
2. Linus evaluates the suggested changes against the original code
3. Provides specific recommendation: accept, modify, or reject
4. If modification needed, provides the corrected approach

**Output**:
```
REVIEW COMMENT WITH CODE SUGGESTION
===================================

Reviewer's Comment:
"This loop is inefficient. Here's a better way:"

Suggested Code:
```javascript
const results = items
  .filter(item => item.active)
  .map(item => ({
    id: item.id,
    name: item.name,
    timestamp: Date.now()
  }));
```

Original Code:
```javascript
const results = [];
for (let i = 0; i < items.length; i++) {
  if (items[i].active) {
    results.push({
      id: items[i].id,
      name: items[i].name,
      timestamp: Date.now()
    });
  }
}
```

LINUS EVALUATION:

Code Quality: Both approaches work
Performance: Original is actually slightly faster (no intermediate arrays)
Readability: Functional style is cleaner, but Date.now() bug exists

VERDICT: MODIFY SUGGESTION

Issue Identified:
The suggested code calls Date.now() during array construction, meaning it
will be called once and all items get the same timestamp. If unique timestamps
per item are intended, both versions have this bug. If shared timestamp is
intended, neither version should call it inside the loop/map.

LINUS RECOMMENDATION:

If unique timestamps needed:
```javascript
const results = items
  .filter(item => item.active)
  .map(item => ({
    id: item.id,
    name: item.name,
    timestamp: Date.now()  // Actually called for each item
  }));
```
Wait, no - that's still wrong. Map evaluates Date.now() expression for each
item but as a single expression evaluation, not per-item. Correct version:

```javascript
const results = items
  .filter(item => item.active)
  .map(item => {
    return {
      id: item.id,
      name: item.name,
      timestamp: Date.now()  // Called for each item
    };
  });
```

If shared timestamp intended:
```javascript
const timestamp = Date.now();
const results = items
  .filter(item => item.active)
  .map(item => ({
    id: item.id,
    name: item.name,
    timestamp
  }));
```

IMPLEMENTATION PLAN:
1. Clarify requirement: unique or shared timestamps? (5 min)
2. Implement correct version based on requirement (10 min)
3. Add test verifying timestamp behavior (20 min)

RECOMMENDED RESPONSE:
"Good catch on the loop! Quick question: should each item have a unique
timestamp or should they all share the same timestamp? The current code and
your suggestion both have an ambiguity here. Once we clarify, I'll implement
the correct functional version."
```

**Notes**:
- Linus evaluates both the original and suggested code
- Often finds issues in both versions that the discussion missed
- Provides corrected implementation when needed
- Asks clarifying questions when requirements are ambiguous

## Purpose

The `/gh-review-assess` command is designed to save developers significant time and mental energy when dealing with code reviews, particularly those that mix legitimate concerns with nitpicking, stylistic preferences, or over-engineering suggestions.

Code reviews are essential for maintaining quality, but they often suffer from several problems: reviewers may nitpick minor stylistic issues while missing real bugs; they may suggest complex refactors that are out of scope for the current PR; they may apply personal preferences as if they were objective standards; or they may propose solutions that are more complex than the problem warrants. Developers receiving such reviews face a dilemma: they want to be collaborative and address legitimate feedback, but they also need to push back on unproductive suggestions without appearing defensive or difficult.

This command solves that problem by providing an objective, automated second opinion. The Linus agent is specifically designed to evaluate feedback through a pragmatic lens, asking "Does this change genuinely improve the codebase, or is it just different?" It applies the "Simple First" philosophy to identify when suggested solutions are over-engineered, and it recognizes legitimate issues around security, correctness, reliability, and user experience that truly deserve attention.

The workflow integrates two specialized agents: @agent-Linus for evaluation (determining what actually matters) and @agent-linus-kernel-planner for implementation planning (determining the simplest way to fix what matters). This two-phase approach ensures that you only spend time planning solutions for issues that are worth solving, and that those solutions are pragmatic rather than over-engineered.

The command is particularly valuable in several scenarios: when dealing with reviewers who are known to be pedantic or prefer complex solutions; when a PR has accumulated many comments and you need to triage them quickly; when review discussions have become lengthy or abstract and you need to cut through to the core issue; when you're uncertain whether feedback represents genuine concerns or personal preferences; and when you need to justify pushing back on certain feedback items with objective reasoning rather than subjective opinion.

By automating the assessment process, this command helps maintain healthy code review culture. It prevents the common pattern where developers start accepting all feedback uncritically to avoid conflict, which leads to over-engineered code and wasted time. It also provides diplomatic language for responding to reviewers, helping you address legitimate concerns while respectfully declining to implement non-essential changes. The result is better code, more efficient development cycles, and healthier team dynamics around code review.

## Context

### Prerequisites

Before using this command, ensure the following prerequisites are met:

1. **GitHub CLI Installation**: The `gh` CLI tool must be installed and available in your PATH. Install via `brew install gh` (macOS), `apt install gh` (Linux), or download from https://cli.github.com/

2. **GitHub Authentication**: You must be authenticated with gh CLI using `gh auth login`. The authentication token must have sufficient permissions to read the repository and its pull requests.

3. **Repository Access**: You must have read access to the repository containing the PR. For private repositories, this means you must be a collaborator or organization member.

4. **Agent Access**: The workflow depends on two agents being available:
   - `@agent-Linus`: For evaluating which review feedback is legitimate
   - `@agent-linus-kernel-planner`: For creating pragmatic implementation plans
   These agents must be configured in your Claude Code environment.

5. **Network Connectivity**: The command makes API calls to GitHub, so active internet connectivity is required.

### Environment Assumptions

- The command assumes you're working on a codebase that follows reasonable engineering practices where code review feedback should be balanced and constructive.
- The command works best when review comments are written in English, as the agents are optimized for English language analysis.
- The command expects GitHub's standard pull request and comment structure; it may not work correctly with heavily customized GitHub Enterprise instances.

### When to Use This Command

Use `/gh-review-assess` when:
- You've received code review feedback with multiple comments and need to triage quickly
- A reviewer's suggestions seem overly complex or out of scope
- You want an objective second opinion on whether feedback is actionable
- Review discussions have become lengthy and you need to identify the core issue
- You need to justify pushing back on certain feedback with clear reasoning
- You're working with a new team and unsure of their review culture/standards

### When NOT to Use This Command

Do not use this command when:
- The review feedback is clearly constructive and actionable (no assessment needed)
- You have security/compliance concerns about sending code to Claude's API
- The PR is in a domain requiring specialized expertise the agents don't have
- You need to maintain a specific relationship dynamic with the reviewer
- Your organization has policies against using AI tools for code review

<workflow name="gh-review-assess">

<meta_instructions>
CRITICAL RULES THAT MUST BE FOLLOWED THROUGHOUT THIS WORKFLOW:

1. URL VALIDATION IS NON-NEGOTIABLE: The GitHub URL must be validated before any API calls are made. Invalid URLs should fail fast with helpful error messages. WHY: Prevents wasted API calls and provides better user experience than cryptic API errors. CONSEQUENCE: If this is skipped, users will get confusing GitHub API errors instead of actionable guidance.

2. NEVER MODIFY THE USER'S CODE BASED ON REVIEW FEEDBACK: This workflow evaluates and plans, but does NOT implement changes. The user must review and approve the assessment before making any code changes. WHY: Automatically implementing review feedback would undermine the entire purpose of assessment. The user needs to understand and agree with the evaluation before taking action. CONSEQUENCE: Users would lose control of their codebase and potentially implement changes they disagree with.

3. DISTINGUISH BETWEEN TECHNICAL MERIT AND CULTURAL FIT: When Linus evaluates feedback, he must consider whether issues are objectively technical (security, correctness, performance) vs. culturally relative (style, organization preferences, tooling choices). WHY: Some feedback is universally valid (security bugs) while other feedback depends on team norms (tab vs spaces). CONSEQUENCE: Treating all feedback the same leads to either accepting bad advice or rejecting good advice.

4. PRESERVE THE ORIGINAL REVIEWER'S INTENT: When synthesizing multi-part comments or discussion threads, maintain fidelity to what the reviewer actually said, even if you disagree with it. WHY: Misrepresenting someone's feedback is both dishonest and counterproductive. The assessment must be based on what was actually said, not a strawman version. CONSEQUENCE: Users might push back on feedback based on misunderstanding, damaging relationships.

5. TIME ESTIMATES MUST BE REALISTIC: When linus-kernel-planner provides implementation time estimates, they should include thinking time, coding time, testing time, and some buffer for unexpected issues. WHY: Underestimating implementation time leads to missed commitments and rushed work. CONSEQUENCE: Users will lose trust in the tool if estimates are consistently wrong.

6. HANDLE AUTHENTICATION FAILURES GRACEFULLY: If gh CLI authentication fails or returns access denied errors, provide clear, actionable troubleshooting steps. WHY: Authentication issues are common and frustrating. Good error messages save users significant debugging time. CONSEQUENCE: Users will abandon the tool if authentication problems result in cryptic errors.

7. MULTIPLE ISSUES IN ONE COMMENT MUST BE EVALUATED SEPARATELY: If a single review comment contains multiple distinct concerns, each must be evaluated independently. WHY: A comment might contain one legitimate issue and three nitpicks. Evaluating them as a bundle leads to all-or-nothing decisions. CONSEQUENCE: Either good feedback gets rejected or bad feedback gets accepted because they're bundled together.

8. PROVIDE DIPLOMATIC RESPONSE TEMPLATES: When suggesting how to respond to reviewers, provide professional, non-confrontational language even when rejecting feedback. WHY: Code review is a social process. How you communicate matters as much as what you communicate. CONSEQUENCE: Blunt responses damage team relationships even when technically correct.

9. DEFER TO USER JUDGMENT ON EDGE CASES: When the evaluation is close call or depends on context the agents don't have, explicitly flag it as "USER DECISION REQUIRED" rather than guessing. WHY: Some decisions require business context, team history, or domain expertise that aren't available in the PR comment. CONSEQUENCE: Making these decisions on behalf of the user leads to wrong choices based on incomplete information.

10. AUDIT TRAIL IS MANDATORY: The final output must show what was evaluated, what was decided, and why. This allows users to understand and potentially disagree with the assessment. WHY: Transparency builds trust and allows users to calibrate the tool's judgment against their own. CONSEQUENCE: Black-box assessments without rationale don't help users learn or grow, they just shift the problem.
</meta_instructions>

<thinking_requirement>
Before proceeding with each phase, carefully think through:
- What is the current state of the workflow?
- What information do I have from previous phases?
- What are the inputs to this phase and have I validated them?
- What could go wrong in this phase?
- What is the expected outcome of this phase?
- How will I validate success before proceeding?
- Are there any edge cases or assumptions I need to verify?
- Do I have sufficient context to make good decisions, or do I need to ask for clarification?

Phase-specific considerations:
- Phase 1 (Validation): Does the URL match expected GitHub PR comment patterns? Have I tested both the happy path and error cases in my validation logic?
- Phase 2 (Fetch): Is my gh CLI command correct? Am I handling authentication errors? What if the comment was deleted?
- Phase 3 (Analysis): Have I correctly parsed the comment structure? Did I identify all distinct issues? Am I preserving the reviewer's original intent?
- Phase 4 (Linus Evaluation): Am I applying consistent evaluation criteria? Am I distinguishing technical merit from cultural preference? Am I being fair to both the reviewer and the author?
- Phase 5 (Planning): Are my plans realistic? Am I following "Simple First" philosophy? Have I estimated time appropriately?
- Phase 6 (Output): Is my output structured clearly? Have I provided rationale for all decisions? Is my response template diplomatic?
</thinking_requirement>

<phase number="1" name="Input Validation" mandatory="true">
<description>Validate the GitHub URL format and extract necessary identifiers before making any API calls</description>

1. Extract the comment URL from the command parameters
2. Validate that the URL matches one of the supported GitHub PR comment formats:
   - Issue comment: `https://github.com/[owner]/[repo]/pull/[number]#issuecomment-[id]`
   - Review comment: `https://github.com/[owner]/[repo]/pull/[number]#discussion_r[id]`
   - Full review: `https://github.com/[owner]/[repo]/pull/[number]#pullrequestreview-[id]`
3. Parse the URL to extract:
   - Repository owner
   - Repository name
   - Pull request number
   - Comment type (issue comment, review comment, or full review)
   - Comment ID
4. Verify the URL is from github.com (not GitHub Enterprise with different domain)
5. Construct the GitHub API endpoint needed for the next phase

<validation_gate>
STOP HERE. Confirm before proceeding:
✓ URL matches one of the three supported format patterns
✓ All required identifiers (owner, repo, PR number, comment ID) have been successfully extracted
✓ Comment type has been correctly identified (needed for choosing correct API endpoint)
✓ No malformed or incomplete URL components

If ANY check fails, STOP and report:
"Cannot proceed: Invalid URL format. Expected GitHub PR comment URL matching one of these patterns:
- https://github.com/owner/repo/pull/123#issuecomment-456789
- https://github.com/owner/repo/pull/123#discussion_r987654
- https://github.com/owner/repo/pull/123#pullrequestreview-112233

Received: [the actual URL provided]

Please provide a valid GitHub pull request comment URL."
</validation_gate>
</phase>

<phase number="2" name="Fetch Comment Content" mandatory="true">
<description>Use @sentinel to fetch the comment or review content from GitHub API</description>

1. Use @sentinel to verify gh CLI is installed and available by running `gh --version`
2. Use @sentinel to check authentication status by running `gh auth status`
3. Based on comment type identified in Phase 1, use @sentinel to construct and execute the appropriate gh API command:
   - For issue comments: `gh api repos/[owner]/[repo]/issues/comments/[comment-id]`
   - For review comments: `gh api repos/[owner]/[repo]/pulls/comments/[comment-id]`
   - For full reviews: `gh api repos/[owner]/[repo]/pulls/[pr-number]/reviews/[review-id]` followed by fetching review comments
4. @sentinel executes the gh CLI command to fetch comment content
5. Parse the JSON response to extract:
   - Comment body/content
   - Commenter's username
   - Comment creation timestamp
   - For review comments: associated code context (file path, line numbers, code snippet)
   - For full reviews: all associated comments in the review
6. Handle special cases:
   - Multi-line comments with code blocks
   - Comments with GitHub suggestions syntax
   - Comments that reference other issues/PRs
   - Deleted or edited comments (check for edit history)

<validation_gate>
STOP HERE. Confirm before proceeding:
✓ gh CLI is installed and accessible
✓ gh CLI authentication is valid and has access to the repository
✓ GitHub API call succeeded (status 200)
✓ Comment content was successfully extracted from JSON response
✓ Comment is not empty or deleted

If ANY check fails, STOP and report the specific error:
- If gh not installed: "Cannot proceed: gh CLI is not installed. Install via: brew install gh (macOS) or apt install gh (Linux)"
- If not authenticated: "Cannot proceed: gh CLI is not authenticated. Run: gh auth login"
- If access denied: "Cannot proceed: Access denied to repository [owner/repo]. Verify you have read access to this repository."
- If API error: "Cannot proceed: GitHub API error [error details]. The comment may have been deleted or the URL may be incorrect."
- If comment empty: "Cannot proceed: The comment appears to be empty or deleted."
</validation_gate>
</phase>

<phase number="3" name="Initial Analysis" mandatory="true">
<description>Analyze the comment structure to identify distinct issues and prepare for evaluation</description>

1. Read through the entire comment content to understand overall context
2. Identify if this is a single issue or multiple distinct issues:
   - Look for numbered lists
   - Look for paragraph breaks indicating separate concerns
   - Look for phrases like "Also," "Additionally," "Another thing," which signal multiple issues
   - Look for code suggestions vs. general feedback
3. For each distinct issue identified, extract:
   - The core concern or suggestion
   - Any supporting rationale provided by the reviewer
   - Any code snippets or specific examples
   - The tone/severity implied (urgent vs. optional suggestion)
4. Categorize each issue into preliminary buckets:
   - Security concerns
   - Correctness/bugs
   - Performance issues
   - Reliability/error handling
   - User experience
   - Code organization/structure
   - Naming/style
   - Testing
   - Documentation
   - Other
5. Identify any context needed for proper evaluation:
   - What file(s) are being discussed?
   - What is the surrounding code context?
   - Are there references to team standards or previous discussions?
6. Flag any issues that require additional information to evaluate properly

<validation_gate>
STOP HERE. Confirm before proceeding:
✓ The comment has been fully parsed and understood
✓ All distinct issues have been identified and separated (minimum 1)
✓ Each issue has been preliminarily categorized
✓ Any code snippets or examples have been preserved
✓ Context requirements have been identified

If ANY check fails, STOP and report:
"Cannot proceed with evaluation: [specific issue with parsing]"

If additional context is needed, note it for the user:
"Additional context may improve evaluation accuracy: [list what's needed]"
</validation_gate>
</phase>

<phase number="4" name="Linus Evaluation" mandatory="true">
<description>Invoke @agent-Linus to evaluate each issue and determine which are legitimate vs nitpicking</description>

1. For each issue identified in Phase 3, prepare a structured evaluation request for @agent-Linus containing:
   - The reviewer's original comment/concern
   - The preliminary category
   - Any code context from Phase 2
   - The PR context (what change is being reviewed)
2. Invoke @agent-Linus with the request, asking him to evaluate:
   - Is this a legitimate technical concern?
   - Does addressing this genuinely improve the codebase?
   - Is this a security, correctness, or reliability issue? (Higher priority)
   - Is this a preference, style, or organizational suggestion? (Lower priority)
   - Is the suggested solution proportional to the problem?
   - Does this follow "Simple First" philosophy or is it over-engineering?
3. For each issue, capture Linus's response including:
   - ACCEPT or REJECT decision
   - Category (Security, Correctness, Performance, Reliability, UX, Nitpicking, Over-engineering, Premature Optimization, etc.)
   - Priority level if accepted (HIGH, MEDIUM, LOW)
   - Detailed rationale explaining the decision
   - Any modifications to the reviewer's suggestion if the concern is valid but the proposed solution is not
4. Maintain the original reviewer's context and intent even when disagreeing
5. Ensure Linus distinguishes between:
   - Universal technical concerns (security, correctness) vs. team/cultural concerns (style, tooling preferences)
   - Real issues vs. hypothetical issues
   - Problems that affect users vs. problems that affect only developers
6. Track statistics:
   - Total issues evaluated
   - Number accepted vs. rejected
   - Breakdown by category
   - Overall time savings (rejected issues * estimated time to implement)

<validation_gate>
STOP HERE. Confirm before proceeding:
✓ Every issue has been evaluated by Linus
✓ Each evaluation includes clear ACCEPT/REJECT decision
✓ Each evaluation includes category and rationale
✓ Accepted issues have priority levels assigned
✓ The evaluation is fair to both reviewer and author
✓ Over-engineered solutions have been identified and simplified

If ANY check fails, STOP and report:
"Cannot proceed: Incomplete evaluation. [specific issues]"

If evaluation reveals ambiguity requiring user input:
"USER DECISION REQUIRED: [specific issue] cannot be evaluated without additional context: [what's needed]"
</validation_gate>
</phase>

<phase number="5" name="Implementation Planning" mandatory="true">
<description>For accepted issues, invoke @agent-linus-kernel-planner to create pragmatic implementation plans</description>

1. Filter to only issues marked ACCEPT by Linus in Phase 4
2. Group accepted issues by priority level (HIGH, MEDIUM, LOW)
3. For each accepted issue, prepare a planning request for @agent-linus-kernel-planner containing:
   - The original concern
   - Linus's evaluation and rationale
   - The category and priority
   - Any code context from Phase 2
4. Invoke @agent-linus-kernel-planner with the request, asking for:
   - A simple, pragmatic implementation plan
   - Step-by-step tasks following "Simple First" philosophy
   - Realistic time estimates for each step (including thinking, coding, testing)
   - Any prerequisites or dependencies between steps
   - Testing approach for verifying the fix
5. For each implementation plan, capture:
   - Numbered steps with descriptions
   - Time estimate per step
   - Total time estimate
   - Any risks or gotchas to watch for
   - Validation criteria for knowing the issue is resolved
6. Create a consolidated plan that sequences all accepted issues:
   - HIGH priority issues first
   - MEDIUM priority issues second
   - LOW priority issues last
   - Total estimated time across all issues
7. Calculate "time saved" by rejected issues (estimated time to implement unnecessary changes)

<validation_gate>
STOP HERE. Confirm before proceeding:
✓ Every accepted issue has an implementation plan
✓ Each plan follows "Simple First" philosophy (not over-engineered)
✓ Time estimates are realistic (include testing and buffer)
✓ Plans are sequenced by priority
✓ Dependencies between tasks are identified
✓ Total time estimate is calculated

If ANY check fails, STOP and report:
"Cannot proceed: Incomplete implementation planning. [specific issues]"

If plans seem over-engineered:
"WARNING: The following plans may be too complex. Reconsidering simpler approaches: [list issues]"
</validation_gate>
</phase>

<phase number="6" name="Generate Output Report" mandatory="true">
<description>Create comprehensive, structured output report with decisions, rationale, plans, and response template</description>

1. Create report header with:
   - Command execution summary
   - PR and comment metadata (reviewer, timestamp, link)
   - High-level statistics (total issues, accepted/rejected counts)
2. For each issue evaluated, include a section with:
   - The original reviewer's comment (verbatim or summarized)
   - Linus's evaluation decision (ACCEPT/REJECT)
   - Category and priority (if accepted)
   - Detailed rationale for the decision
   - Implementation plan (if accepted)
   - Or explanation for rejection (if rejected)
3. Create summary section with:
   - Total issues broken down by decision
   - Accepted issues grouped by priority
   - Total implementation time estimate
   - Total time saved by rejected issues
   - Overall assessment of the review quality
4. Create recommended action section with:
   - Prioritized list of what to implement
   - Suggested timeline or sequencing
   - Items to push back on (rejected issues)
5. Create recommended response template:
   - Professional, diplomatic tone
   - Thank reviewer for legitimate catches
   - List what will be addressed
   - Respectfully decline unnecessary items with brief rationale
   - Keep the relationship positive and collaborative
6. Include troubleshooting section if any issues:
   - If some issues needed more context
   - If any evaluations were edge cases
   - If user input is needed for final decisions
7. Format the output with:
   - Clear section headers and separators
   - Readable structure with proper whitespace
   - Code blocks where appropriate
   - Emphasis on key decisions

<validation_gate>
STOP HERE. Confirm before proceeding:
✓ Report includes all required sections
✓ Every issue is represented in the output
✓ Decisions and rationales are clear
✓ Implementation plans are actionable
✓ Response template is diplomatic and professional
✓ Statistics are accurate
✓ Output is well-formatted and readable

If ANY check fails, STOP and report:
"Cannot generate complete report: [specific issues]"

Once all checks pass, present the complete report to the user.
</validation_gate>
</phase>

<failure_handling>
You are ALLOWED and ENCOURAGED to fail gracefully at any phase:

- If URL validation fails, STOP immediately and provide format examples
- If gh CLI is not installed or authenticated, STOP and provide setup instructions
- If GitHub API access fails, STOP and provide troubleshooting steps
- If comment content cannot be parsed, STOP and explain what went wrong
- If agent invocation fails, STOP and report which agent failed and why
- If evaluation is ambiguous, STOP and ask user for clarification

Say: "Workflow stopped at Phase [X]: [specific reason and what to do about it]"

This is BETTER than guessing, skipping steps, or proceeding with incomplete information.

The user needs accurate assessment, not fast assessment. Take the time to do it right.
</failure_handling>

</workflow>

## Error Handling

### Error Type: Invalid URL Format

**Error Message**: `Cannot proceed: Invalid URL format. Expected GitHub PR comment URL matching one of these patterns...`

**Symptoms**:
- Command fails immediately without making any API calls
- Clear error message shows expected vs. received URL format

**Root Causes**:
1. User copied the wrong URL (e.g., main PR URL without comment fragment)
2. User copied from GitHub issue instead of pull request
3. URL from GitHub Enterprise with different domain
4. Typo in URL

**Resolution Steps**:
1. Navigate to the PR on GitHub in your browser
2. Click on the specific comment you want to assess
3. Ensure the URL in your address bar contains either `#issuecomment-`, `#discussion_r`, or `#pullrequestreview-`
4. Copy the complete URL including the fragment identifier
5. Run the command again with the corrected URL

**Prevention**:
- Always click on the specific comment before copying the URL
- Verify the URL contains a fragment identifier (#) before running the command
- Bookmark this command's documentation for URL format reference

**Related Issues**: Authentication Required, Comment Not Found

### Error Type: Authentication Required

**Error Message**: `Cannot proceed: gh CLI is not authenticated. Run: gh auth login`

**Symptoms**:
- Command fails during Phase 2 (Fetch Comment Content)
- Error occurs before any comment content is retrieved
- May see "HTTP 401" or "unauthorized" in error details

**Root Causes**:
1. gh CLI has never been authenticated on this machine
2. Authentication token has expired
3. gh CLI was authenticated but tokens were cleared

**Resolution Steps**:
1. Run `gh auth login` in your terminal
2. Choose "GitHub.com" (or your GitHub Enterprise instance)
3. Choose authentication method (web browser is easiest)
4. Follow the prompts to complete authentication
5. Verify with `gh auth status`
6. Run the command again

**Prevention**:
- Run `gh auth status` periodically to check authentication
- Re-authenticate before tokens expire if working with sensitive repos
- Consider using `gh auth refresh` to renew tokens proactively

**Related Issues**: Access Denied, Repository Not Found

### Error Type: Access Denied

**Error Message**: `Cannot proceed: Access denied to repository [owner/repo]. Verify you have read access to this repository.`

**Symptoms**:
- Command fails during Phase 2 when trying to fetch comment
- Authentication is valid but specific repository cannot be accessed
- May see "HTTP 403" or "forbidden" in error details

**Root Causes**:
1. Private repository that you're not a collaborator on
2. Organization repository requiring additional SSO authentication
3. Repository has been deleted or made private since comment URL was shared
4. gh CLI authenticated with account that doesn't have access

**Resolution Steps**:
1. Verify the repository exists: `gh repo view [owner/repo]`
2. Check if you're logged in with the correct GitHub account: `gh auth status`
3. For organization repositories, check if SSO is required: `gh auth status` (look for SSO notices)
4. Request access from repository owner if needed
5. If you recently gained access, try `gh auth refresh`
6. If using multiple GitHub accounts, switch to correct one: `gh auth switch`

**Prevention**:
- Verify repository access before attempting to assess comments
- Keep track of which GitHub account your gh CLI is authenticated with
- For organization repos, complete SSO authorization when prompted

**Related Issues**: Authentication Required, Repository Not Found

### Error Type: Comment Not Found

**Error Message**: `Cannot proceed: GitHub API error - Comment not found (404). The comment may have been deleted or the URL may be incorrect.`

**Symptoms**:
- Command fails during Phase 2 after authentication succeeds
- GitHub API returns 404 error
- User is confident URL format is correct

**Root Causes**:
1. Comment was deleted by author or repository maintainer
2. Pull request was closed and comments were hidden
3. Typo in comment ID within the URL
4. Comment ID from URL doesn't match the repository

**Resolution Steps**:
1. Navigate to the PR URL in your browser (without the comment fragment)
2. Verify the comment still exists on the page
3. If comment exists, click it and copy the URL again
4. If comment was deleted, there's nothing to assess
5. Check if you're looking at the right PR number

**Prevention**:
- Assess comments shortly after receiving them
- Keep a local copy of important review feedback before it might be deleted
- Verify URLs before running time-consuming commands

**Related Issues**: Invalid URL Format, Pull Request Closed

### Error Type: Agent Invocation Failure

**Error Message**: `Cannot proceed: Failed to invoke @agent-Linus. Agent may not be configured or available.`

**Symptoms**:
- Command succeeds through Phase 3 (fetching and parsing comment)
- Failure occurs when trying to invoke agent for evaluation
- Error mentions agent name specifically

**Root Causes**:
1. Agent is not configured in Claude Code environment
2. Agent configuration has incorrect name or path
3. Agent file has syntax errors
4. System resources exhausted (memory, API limits)

**Resolution Steps**:
1. Verify agent exists: check your `.claude/agents/` directory
2. Try invoking the agent manually: `@agent-Linus` in a regular conversation
3. Check agent configuration file for syntax errors
4. Restart Claude Code if agents were recently added
5. Check system resources and API quota
6. If problem persists, file an issue with agent configuration details

**Prevention**:
- Test agent availability before relying on automated workflows
- Keep agent configurations in version control
- Monitor API usage and quota limits
- Validate agent files after editing

**Related Issues**: Planning Agent Failure, Configuration Error

### Error Type: Ambiguous Review Content

**Error Message**: `USER DECISION REQUIRED: [specific issue] cannot be evaluated without additional context: [what's needed]`

**Symptoms**:
- Command completes most phases but flags certain issues as ambiguous
- Output includes specific questions for user
- Not a technical failure, but requires user input to proceed

**Root Causes**:
1. Review comment references team-specific conventions not visible in the PR
2. Comment uses ambiguous language that could be interpreted multiple ways
3. Suggested change depends on business requirements not in the code
4. Comment references prior discussions or context not in the PR

**Resolution Steps**:
1. Read the flagged issue and the context questions
2. Provide the missing information in a follow-up message
3. If information isn't available, make a judgment call based on your knowledge
4. Document the decision and reasoning for future reference
5. Consider asking the reviewer for clarification

**Prevention**:
- When requesting reviews, provide sufficient context in PR description
- Encourage reviewers to be explicit and self-contained in comments
- Link to relevant docs or prior discussions in PR description
- Keep team conventions documented and accessible

**Related Issues**: Evaluation Uncertainty, Context Insufficient

## Integration Points

### Related Commands

**`/pr-review`**: Complements `/gh-review-assess` by providing the initial code review. Typical workflow is:
1. Use `/pr-review` to conduct a thorough review of your own PR before submitting
2. Submit PR for team review
3. When you receive feedback, use `/gh-review-assess` to evaluate which feedback to act on
4. Make revisions and use `/pr-review` again to verify fixes

**`/project:github-bugfix`**: After using `/gh-review-assess` to identify legitimate bugs or issues from a review, use `/project:github-bugfix` to implement the fixes systematically with proper testing.

**`/code-review`**: An alternative local code review command. If you're pair programming or want a review before pushing to GitHub, use `/code-review`. Once code is on GitHub and has received feedback, use `/gh-review-assess`.

### Agent Dependencies

**@agent-Linus**: Core dependency. This agent provides the brutally honest evaluation that distinguishes legitimate issues from nitpicking. The agent applies pragmatic engineering judgment and "Simple First" philosophy. If this agent is unavailable, the command cannot function.

**@agent-linus-kernel-planner**: Secondary dependency. This agent creates implementation plans for accepted issues. If unavailable, you can still get the evaluation from Linus but won't get structured implementation plans.

### External Tool Dependencies

**gh CLI**: Hard dependency. The command uses `gh api` to fetch comment content from GitHub. Without gh CLI, the command cannot access GitHub's API. Install via package manager or from https://cli.github.com/

**GitHub API**: The command makes RESTful API calls to GitHub. It respects rate limits and requires appropriate authentication. If GitHub API is down or rate limited, the command will fail gracefully with appropriate error messages.

### Data Flow

```
User provides URL
    ↓
[Validation] → Extract repo, PR, comment identifiers
    ↓
[gh CLI] → Fetch comment content from GitHub API
    ↓
[Parsing] → Identify distinct issues in comment
    ↓
[@agent-Linus] → Evaluate each issue (ACCEPT/REJECT)
    ↓
[@agent-linus-kernel-planner] → Plan implementation for accepted issues
    ↓
[Output] → Structured report with decisions and plans
    ↓
User reviews and decides whether to act
```

### Configuration

The command requires no explicit configuration files. However, it inherits configuration from:
- gh CLI authentication state (via `gh auth login`)
- Claude Code agent configurations (agents must be defined in `.claude/agents/`)
- GitHub API rate limits (determined by your authentication type)

## Performance Considerations

### Execution Time

Typical execution time varies based on comment complexity:
- **Simple single-issue comment**: 30-60 seconds
  - Validation: <1 second
  - Fetch: 2-5 seconds
  - Analysis: 5-10 seconds
  - Linus evaluation: 15-30 seconds
  - Planning: 10-20 seconds
  - Output: <1 second

- **Multi-issue comment (3-5 issues)**: 1-2 minutes
  - Scales roughly linearly with number of issues
  - Each issue requires separate Linus evaluation

- **Full review (10+ comments)**: 3-5 minutes
  - Multiple API calls to fetch all review comments
  - Each comment evaluated separately
  - Consolidated planning at the end

### Resource Usage

**Network**: Makes 1-5 GitHub API calls depending on comment type:
- Single comment: 1 API call
- Review thread: 1 call per comment in thread
- Full review: 1 call for review metadata + 1 per comment

**API Quota**: Consumes GitHub API rate limit:
- Authenticated requests: 5,000 per hour
- This command typically uses 1-10 requests
- Negligible impact on quota for normal usage
- If assessing many reviews in quick succession, may hit rate limits

**Memory**: Minimal memory footprint:
- Comment content stored in memory (typically <10KB)
- Agent contexts maintained during evaluation
- Peak usage typically <50MB

**CPU**: Low CPU usage:
- Most time spent waiting on API calls and agent inference
- Parsing and validation are computationally trivial
- No intensive processing required

### Optimization Tips

1. **Batch similar assessments**: If assessing multiple comments from the same PR, do them in one session to reuse context
2. **Assess early**: Review comments when you receive them, before context becomes stale
3. **Use for complex reviews**: Don't use this command for obviously good or obviously bad feedback; save it for ambiguous cases
4. **Cache gh auth**: Keep gh CLI authenticated to avoid re-authentication overhead
5. **Filter before assessing**: If a review has 20 comments and you know 10 are fine, assess only the questionable ones

### Rate Limiting

GitHub API rate limits:
- **Authenticated**: 5,000 requests/hour
- **Unauthenticated**: 60 requests/hour (command won't work)

The command respects rate limits and will fail gracefully if limits are hit:
```
Cannot proceed: GitHub API rate limit exceeded.
Limit resets at: [timestamp]
Current usage: [X/5000] requests
```

If you hit rate limits:
1. Wait for the reset time (shown in error message)
2. Reduce frequency of assessments
3. Use command only for non-trivial reviews
4. Consider if you need to assess every single comment

## Security and Permissions

### GitHub Access

**Required Permissions**:
- Repository read access (to fetch PR and comment data)
- No write permissions required (command is read-only)

**Authentication**:
- Uses gh CLI authentication via `gh auth login`
- Supports personal access tokens (PATs) and OAuth apps
- Requires `repo` scope for private repositories
- Requires `public_repo` scope minimum for public repositories

**Private Repositories**:
- Full support for private repos if you have access
- Comment content is fetched via GitHub API with your credentials
- Command respects repository permissions and access controls

### Data Privacy

**What Data Leaves Your Machine**:
1. Comment content is sent to Claude AI for evaluation by @agent-Linus
2. Code snippets included in review comments are sent to Claude AI
3. Repository and PR metadata (names, numbers) are included in agent context

**What Data Stays Local**:
1. Your GitHub authentication tokens (managed by gh CLI)
2. The full repository code (only comment content is sent)
3. Evaluation results and plans (stored only in session)

**Compliance Considerations**:
- If your organization prohibits sending code to external AI services, DO NOT use this command
- Review your organization's AI usage policies before assessing reviews with proprietary code
- For public repositories, this is generally not a concern
- For private repositories with sensitive data, consider implications carefully

### Sensitive Information

**Risk**: Review comments may contain:
- API keys, passwords, or credentials (if reviewer inadvertently posted them)
- Proprietary algorithms or business logic
- Personally identifiable information (PII)
- Security vulnerability details

**Mitigation**:
- Manually review comment content before running assessment
- If comment contains secrets, redact them or don't assess
- For security issues, consider whether assessment via AI is appropriate
- Use discretion when dealing with embargoed security vulnerabilities

**Best Practices**:
- Don't assess comments that reference unreleased security vulnerabilities
- Don't assess comments containing credentials or API keys
- If in doubt about sensitivity, err on the side of caution and skip assessment
- Remember that AI service logs may retain data for some period

## Troubleshooting Guide

### Diagnostic Steps

If the command fails or produces unexpected results, follow these diagnostic steps in order:

#### Step 1: Verify URL Format
```bash
# The URL should match one of these patterns
# Issue comment:
https://github.com/owner/repo/pull/123#issuecomment-456789

# Review comment:
https://github.com/owner/repo/pull/123#discussion_r987654

# Full review:
https://github.com/owner/repo/pull/123#pullrequestreview-112233

# Check: Does your URL match one of these exactly?
# Check: Does it have the # fragment identifier?
# Check: Is it from a pull request (not an issue)?
```

#### Step 2: Verify gh CLI Installation and Authentication
```bash
# Check if gh CLI is installed
gh --version
# Expected: gh version X.X.X

# Check authentication status
gh auth status
# Expected: Logged in to github.com as [username]

# If not authenticated, run:
gh auth login
```

#### Step 3: Verify Repository Access
```bash
# Test access to the repository
gh repo view owner/repo
# Should show repository details

# Try fetching PR data
gh pr view 123 --repo owner/repo
# Should show PR details

# If access denied, verify:
# - You're logged in with correct account (gh auth status)
# - You have read access to the repository
# - For organization repos, SSO may be required
```

#### Step 4: Test Comment Fetch Manually
```bash
# For issue comments:
gh api repos/owner/repo/issues/comments/456789

# For review comments:
gh api repos/owner/repo/pulls/comments/987654

# For full reviews:
gh api repos/owner/repo/pulls/123/reviews/112233

# Should return JSON with comment content
# If 404: comment may be deleted or ID wrong
# If 403: authentication/access issue
```

#### Step 5: Verify Agent Availability
```bash
# In Claude Code, try manually invoking:
@agent-Linus
@agent-linus-kernel-planner

# Should respond without errors
# If error, check .claude/agents/ directory for agent files
```

#### Step 6: Check for Common Pitfalls
- Are you on a VPN or firewall that blocks GitHub API?
- Has the comment been deleted since you got the URL?
- Is the repository archived (read-only might affect API)?
- Are you hitting GitHub API rate limits? (Check headers in gh output)
- Is Claude Code's AI service accessible (network issues)?

### Common Issues and Solutions

**Issue**: "URL format invalid" but URL looks correct
- **Solution**: Check for extra characters or spaces. Copy URL fresh from GitHub. Ensure URL is from pull request, not issue.

**Issue**: "Comment not found" but comment exists on GitHub
- **Solution**: Click the comment and copy URL again. Comment ID in URL might be wrong. Try refreshing GitHub page.

**Issue**: "Access denied" but you can see the repo on GitHub
- **Solution**: Browser access doesn't guarantee API access. Verify gh CLI is authenticated with same account. Check for SSO requirements.

**Issue**: Command times out or hangs
- **Solution**: Check network connectivity. GitHub API might be slow or rate limited. Try again in a few minutes.

**Issue**: Linus evaluation seems wrong or incomplete
- **Solution**: This is subjective. Review the rationale. If you disagree, that's fine - you make the final call. Consider providing more context.

**Issue**: Implementation plan seems too complex
- **Solution**: The planner tries to be thorough. You can simplify further based on your judgment. Plans are guidelines, not requirements.

**Issue**: Can't assess comments from GitHub Enterprise
- **Solution**: Command currently supports github.com only. For GHE, you'd need to modify URL validation and gh CLI configuration.

### Getting Help

If you've followed all diagnostic steps and the command still doesn't work:

1. **Collect Debug Information**:
   - Exact command you ran
   - Error message received
   - Output of `gh auth status`
   - Output of `gh --version`
   - Whether repository is public or private

2. **Check Known Issues**:
   - Review this documentation for similar issues
   - Check Claude Code release notes for bugs

3. **Workarounds**:
   - Manually copy comment text and ask @agent-Linus to evaluate it
   - Use `/pr-review` as alternative for full PR assessment
   - Assess comments individually if bulk assessment fails

4. **Report Issues**:
   - If you believe this is a bug, report it with debug information
   - Include sanitized examples (no sensitive data)

## Summary

The `/gh-review-assess` command is a sophisticated automation tool designed to save developers significant time and mental energy when dealing with code reviews, particularly those containing a mix of legitimate feedback and unnecessary or overly pedantic suggestions.

**Core Functionality**: The command accepts a GitHub PR comment URL, fetches the comment content via gh CLI, analyzes it to identify distinct issues, invokes the Linus agent to evaluate each issue for legitimacy and value, invokes the linus-kernel-planner agent to create pragmatic implementation plans for accepted issues, and produces a comprehensive report showing which feedback should be acted upon and which should be respectfully declined.

**Key Workflow Phases**:
1. **Input Validation**: Ensures URL is properly formatted and extracts necessary identifiers
2. **Fetch Comment Content**: Uses gh CLI and GitHub API to retrieve comment data
3. **Initial Analysis**: Parses comment structure and identifies distinct issues
4. **Linus Evaluation**: Each issue is evaluated for technical merit and practical value
5. **Implementation Planning**: Accepted issues receive simple, pragmatic implementation plans
6. **Output Report**: Comprehensive report with decisions, rationale, plans, and response template

**Primary Benefits**:
- Saves time by filtering out nitpicky or unnecessary feedback items
- Provides objective evaluation criteria based on "Simple First" philosophy
- Creates actionable implementation plans for legitimate issues
- Supplies diplomatic language for responding to reviewers
- Helps maintain healthy code review culture by distinguishing signal from noise
- Reduces stress and conflict around pushing back on unproductive suggestions

**Integration**: The command integrates with gh CLI for GitHub access, @agent-Linus for evaluation, and @agent-linus-kernel-planner for planning. It complements other code review commands like `/pr-review` and `/code-review`, and can be followed by `/project:github-bugfix` for implementing the identified fixes.

**Prerequisites**: Requires gh CLI installed and authenticated, read access to the repository, and the two agent dependencies configured in Claude Code. Works with both public and private repositories.

**Typical Use Cases**:
- Triaging multi-issue review comments quickly
- Getting second opinion on whether feedback is actionable
- Assessing reviews from pedantic or over-engineering-prone reviewers
- Understanding core concerns when review discussions become lengthy
- Preparing diplomatic responses to feedback you plan to decline
- Validating your own instinct about which feedback matters

**Limitations**:
- Only supports github.com (not GitHub Enterprise at this time)
- Requires sending comment content to Claude AI (may not be suitable for highly sensitive code)
- Agent evaluations are judgment calls, not absolute truth
- Some context-dependent feedback may require user input to evaluate properly
- English language comments are handled best

**Success Criteria**: The command is successful when it provides clear, well-reasoned evaluations that help you make confident decisions about which review feedback to implement, saves you time by filtering out low-value suggestions, and provides communication templates that maintain positive working relationships with reviewers even when declining some of their suggestions.

**Philosophy**: The command embodies the "Simple First" philosophy by helping you resist pressure to over-engineer solutions, accept unnecessary refactors, or implement features that aren't in scope. It recognizes that not all feedback is equally valuable, and that spending time on low-value changes is an opportunity cost that prevents working on genuinely important improvements. By providing objective evaluation and clear rationale, it empowers you to push back constructively on suggestions that don't genuinely improve the codebase.

## Critical Success Factors

### Factor 1: Trust the Evaluation, But Verify

**Why It Matters**: The Linus agent provides opinions based on general engineering principles and the "Simple First" philosophy. These are educated opinions, not absolute truth. The agent doesn't have full context about your team's conventions, your codebase's history, or your product's business requirements.

**How to Apply**:
- Read the rationale for each evaluation decision
- If you disagree, that's completely valid - you make the final call
- Use the evaluation as input to your decision, not as the decision itself
- When evaluation seems wrong, ask yourself what context the agent might be missing
- Consider whether team-specific or domain-specific factors change the assessment

**Red Flags**:
- Blindly accepting all evaluations without reading rationale
- Never disagreeing with the agent's assessment
- Using evaluation as justification without understanding reasoning
- Ignoring your own engineering judgment in favor of agent judgment

### Factor 2: Maintain Reviewer Relationships

**Why It Matters**: Code review is fundamentally a social and collaborative process. Using this tool to dismiss feedback might feel efficient, but if it damages relationships with reviewers, you've created a bigger problem than you solved. The goal is better collaboration, not winning arguments.

**How to Apply**:
- Always thank reviewers for their time and attention
- Explain decisions with reasoning, not just "the AI said no"
- Pick your battles - sometimes implementing minor suggestions maintains goodwill
- Use the diplomatic response templates as starting points, not scripts
- Be especially careful with feedback from senior engineers or team leads
- Consider the reviewer's perspective and intent, even when declining suggestions

**Red Flags**:
- Using the tool to justify dismissing all feedback from certain reviewers
- Responding to reviewers with just the agent's output
- Creating perception that you're "automating away" human feedback
- Building reputation as someone who doesn't value code review

### Factor 3: Recognize the Limits of Automation

**Why It Matters**: Some aspects of code review require human judgment, domain expertise, team context, or business knowledge that AI agents don't have. Recognizing these limits prevents making poor decisions based on incomplete evaluation.

**How to Apply**:
- Flag evaluations that depend on information not in the PR
- Ask for human input when evaluation is ambiguous
- Don't use the tool for feedback requiring specialized domain knowledge
- Consider whether the reviewer knows something about the codebase you don't
- Be especially careful with security or compliance-related feedback
- Remember that team conventions and style guides are social contracts, not universal truths

**Red Flags**:
- Using the tool for all feedback without considering context
- Dismissing feedback from domain experts because agent disagreed
- Not recognizing when evaluation needs more information
- Treating agent evaluation as more authoritative than human expertise
- Applying general principles to cases requiring specific knowledge

### Factor 4: Optimize for Learning, Not Just Efficiency

**Why It Matters**: Code review is a learning opportunity. Quickly dismissing feedback might be efficient in the short term but could prevent you from learning valuable lessons or understanding different perspectives. The goal is not just to ship code faster, but to improve your skills and understanding.

**How to Apply**:
- Read the reviewer's full rationale even for rejected items
- Consider why they suggested what they did, even if you disagree
- Look for patterns in accepted vs. rejected feedback to calibrate judgment
- Use disagreements as opportunities to discuss and align on principles
- Ask questions when feedback introduces you to concepts you don't know
- Treat the evaluation as educational, not just operational

**Red Flags**:
- Only reading the ACCEPT/REJECT decision, not the rationale
- Never following up with reviewers to discuss evaluations
- Missing patterns in feedback that indicate knowledge gaps
- Using tool to avoid thinking about feedback yourself
- Not reflecting on whether your own code could be clearer to reduce feedback

### Factor 5: Balance Pragmatism with Quality

**Why It Matters**: The "Simple First" philosophy and the Linus evaluation style are pragmatic and results-oriented. This is valuable for avoiding over-engineering, but can be misapplied to justify cutting corners or ignoring real quality concerns. Pragmatism doesn't mean lowest possible effort.

**How to Apply**:
- Security, correctness, and reliability issues should almost always be accepted
- "Simple First" means simplest *good* solution, not simplest possible solution
- Consider whether rejecting feedback creates technical debt
- Balance shipping quickly with building maintainable systems
- Recognize that some "nice to have" improvements are worth the time
- Apply more scrutiny to rejected suggestions than accepted ones

**Red Flags**:
- Using "Simple First" to justify skipping testing or error handling
- Rejecting all architectural or organizational feedback
- Never accepting suggestions to improve code quality
- Building reputation for delivering quick-but-messy code
- Ignoring feedback about maintainability or readability
- Applying different standards to your code than you'd apply reviewing others

By keeping these critical success factors in mind, you'll use the `/gh-review-assess` command as a powerful tool for improving your code review workflow while maintaining quality code, healthy team dynamics, and continuous learning. The command is most effective when it augments your judgment rather than replacing it, and when it facilitates better collaboration rather than creating shortcuts around it.

## Final Output (STRICT JSON)

After completing all assessment phases, output ONLY this JSON structure as your final response:

```json
{
  "accepted": [
    {"issue": "Issue description", "priority": "high|medium|low", "rationale": "Why this should be fixed"}
  ],
  "rejected": [
    {"issue": "Issue description", "rationale": "Why this is rejected (nitpicking, out of scope, etc.)"}
  ],
  "plan_markdown": "# Implementation Plan\n\n## Step 1: ...\n\n## Step 2: ..."
}
```

Where:
- `accepted`: Array of review findings that will be addressed (may be empty)
- `rejected`: Array of review findings that will NOT be addressed (may be empty)
- `plan_markdown`: If `accepted` is non-empty, provide a markdown implementation plan. If `accepted` is empty, this field may be empty string or omitted.

**IMPORTANT**: After generating this JSON, you MUST also use @sentinel to post a comment to the PR summarizing the accepted/rejected decisions:
`gh pr comment <pr_url> --body "[SUMMARY_OF_DECISIONS]"`