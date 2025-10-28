---
allowed-tools: Task, SlashCommand, Write
description: Autonomously plan and fix bugs using Phase 0 multi-agent research, then auto-implement with /implement
argument-hint: [bug description]
---

# Autonomous Bug Planning & Implementation

Generate a bug fix plan using multi-agent research, then implement with full validation loop.

## Instructions

- You're autonomously planning and fixing a bug using Phase 0 research
- Phase 0: Use 3 agents to research and generate O1-O4 options
- CRITICAL: Phase 0 MUST output explicit Test Specification
- After planning, automatically proceed to /implement with the generated plan

## Bug
$ARGUMENTS

## Phase 0: PreflightExplain

**Step 0.1: Parse Bug**
* Extract bug description from $ARGUMENTS
* Classify severity and scope
* Define acceptance criteria (bug is fixed when...)

**Step 0.2: Codebase Research**
* INVOKE Task tool:
  - subagent_type: "codebase-researcher"
  - description: "Research codebase for bug context"
  - prompt: "Research the codebase for bug: {bug_description}. Identify: (1) Files likely related to the bug, (2) Existing tests for affected code, (3) Similar bugs previously fixed, (4) Dependencies and side effects. Provide structured report with file paths and line numbers."

**Step 0.3: Bug Analysis**
* INVOKE Task tool:
  - subagent_type: "debug-detective"
  - description: "Deep bug analysis"
  - prompt: "Analyze bug: {bug_description}. Investigate: (1) Root cause hypotheses, (2) Steps to reproduce, (3) Edge cases and variants, (4) Complexity assessment. Provide comprehensive analysis."

**Step 0.4: Generate Fix Options**
* INVOKE Task tool:
  - subagent_type: "linus-kernel-planner"
  - description: "Generate O1-O4 fix options"
  - prompt: "Create bug fix plan for: {bug_description}. Generate exactly 4 options (O1-O4), each with: approach overview, files to modify, complexity assessment (lines changed, risk level), risks and edge cases. Include: (1) Problem Assessment, (2) Root Cause Analysis, (3) Four Fix Options with tradeoffs, (4) Recommended Solution (simplest), (5) What NOT to Do (avoid over-engineering). Reject complex solutions. Prefer surgical fixes."

**Step 0.5: Synthesize Plan with Test Specification**
* Compile results from all agents
* Choose simplest option (usually O1 or O2)
* Produce BugFixPlan with REQUIRED sections:

```markdown
# Bug: [bug name from chosen option]

## Bug Description
[from Step 0.1 and 0.3]

## Problem Statement
[root cause from Step 0.4]

## Solution Statement
[chosen option approach]

## Steps to Reproduce
[from Step 0.3]

## Root Cause Analysis
[from Step 0.4]

## Test Specification
REQUIRED: Explicit test specification

### New Tests (Validate Fix)
[For chosen option, specify:]
- **File**: [from codebase research - existing test file or new]
- **Function**: test_[bug_scenario]()
- **Purpose**: Validates bug is fixed
- **Setup**: [reproduce bug scenario]
- **Assertions**: [verify fix works]

Example:
- **File**: tests/test_memory.py
- **Function**: test_memory_consolidation_respects_threshold()
- **Purpose**: Validates memory consolidation bug is fixed
- **Setup**: Create memory store with entries below threshold
- **Assertions**: Assert consolidation not triggered prematurely

### Existing Tests (Regression Prevention)
[From Step 0.2 codebase research:]
- **File**: [existing test file]
- **Function**: [test name] (line number)
- **Purpose**: Ensures [related functionality] still works

Example:
- **File**: tests/test_integration.py
- **Function**: test_reasoning_with_memory() (line 45)
- **Purpose**: Ensures normal reasoning flow unaffected

### Full Regression Suite
```bash
pytest
make lint
```

### Test Coverage Verification
- Run tests and confirm new test functions appear in output
- Check test file was modified (git diff should show new test functions)
- Manually verify test covers the code path that fixes the bug

## Relevant Files
[From Step 0.2 and chosen option]

## Step by Step Tasks
[From chosen option implementation steps]

REQUIRED: Include test creation step BEFORE validation:

### Step N: Write Tests
- Create test file: [from Test Specification]
- Write test function: [from Test Specification]
- Implement setup and assertions per Test Specification

### Step N+1: Run Validation Commands
[From Test Specification regression suite]

## Validation Commands
Execute every command to validate the bug is fixed with zero regressions.

```bash
pytest -v
make lint
```

## Notes
[From Step 0.2, 0.3, 0.4 - any gotchas or context]
```

* Write plan to specs/bug-[slug].md
* Proceed immediately to /implement with this plan

## Output Format

* **BugAnalysis**: Summary from debug-detective (root cause, severity, complexity)
* **ResearchFindings**: Files, tests, patterns from codebase-researcher
* **FixOptions**: O1-O4 from linus-kernel-planner with tradeoffs
* **ChosenOption**: Selected option with rationale
* **GeneratedPlan**: Full bug fix plan (written to specs/)
* **NextStep**: "Proceeding to /implement with generated plan"
