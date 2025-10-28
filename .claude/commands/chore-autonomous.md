---
allowed-tools: Task, SlashCommand, Write
description: Autonomously plan and execute chores using Phase 0 multi-agent research, then auto-implement with /implement
argument-hint: [chore description]
---

# Autonomous Chore Planning & Implementation

Generate a chore plan using multi-agent research, then implement with full validation loop.

## Instructions

- You're autonomously planning and executing a chore using Phase 0 research
- Phase 0: Use 2 agents to research and generate O1-O4 options
- CRITICAL: Phase 0 MUST output explicit Test Specification if code is modified
- After planning, automatically proceed to /implement with the generated plan

## Chore
$ARGUMENTS

## Phase 0: PreflightExplain

**Step 0.1: Parse Chore**
* Extract chore description from $ARGUMENTS
* Classify type (dependency update, refactor, cleanup, config change, docs)
* Define acceptance criteria (chore is complete when...)

**Step 0.2: Codebase Research**
* INVOKE Task tool:
  - subagent_type: "codebase-researcher"
  - description: "Research codebase for chore context"
  - prompt: "Research the codebase for chore: {chore_description}. Identify: (1) Files that need modification, (2) Existing tests that may be affected, (3) Dependencies and side effects, (4) Configuration files involved. Provide structured report with file paths and line numbers."

**Step 0.3: Generate Implementation Options**
* INVOKE Task tool:
  - subagent_type: "linus-kernel-planner"
  - description: "Generate O1-O4 implementation options"
  - prompt: "Create chore implementation plan for: {chore_description}. Generate exactly 4 options (O1-O4), each with: approach overview, files to modify, complexity assessment (lines changed, risk level), risks and edge cases. Include: (1) Task Assessment, (2) Rejected Approaches (over-engineered solutions), (3) Four Implementation Options with tradeoffs, (4) Recommended Solution (simplest), (5) What NOT to Do. Keep it simple - prefer minimal changes."

**Step 0.4: Synthesize Plan with Test Specification**
* Compile results from all agents
* Choose simplest option (usually O1 or O2)
* Produce ChorePlan with REQUIRED sections:

```markdown
# Chore: [chore name from chosen option]

## Chore Description
[from Step 0.1 and 0.3]

## Test Specification
If this chore modifies code behavior, REQUIRED: Explicitly define testing.

**Skip this section if:** Chore only affects docs, configs, or non-code files.

### New/Updated Tests (Validate Changes)
[If chore modifies code, specify:]
- **File**: [from codebase research - existing test file or new]
- **Function**: test_[chore_aspect]()
- **Purpose**: Validates chore changes work correctly
- **Changes**: [if updating existing test, describe changes]

Example:
- **File**: tests/test_integration.py
- **Function**: test_embedding_model_compatibility()
- **Purpose**: Validates upgrade to fastembed 0.4.0 works
- **Changes**: Update assertion for new embedding dimensions

### Existing Tests (Regression Prevention)
[From Step 0.2 codebase research:]
- **File**: [existing test file]
- **Function**: [test name] (line number)
- **Purpose**: Ensures [related functionality] still works

Example:
- **File**: tests/test_memory.py
- **Function**: test_memory_store_operations() (line 45)
- **Purpose**: Ensures memory operations unaffected by dependency update

### Full Regression Suite
```bash
pytest
make lint
```

## Relevant Files
[From Step 0.2 and chosen option]

## Step by Step Tasks
[From chosen option implementation steps]

If chore modifies code, REQUIRED: Include test step BEFORE validation:

### Step N: Write/Update Tests
- Create/modify test files per Test Specification section
- Write test functions with exact names from Test Specification
- Implement setup, assertions, and mocks as specified
- Verify test file appears in git diff

### Step N+1: Run Validation Commands
[From Test Specification regression suite]

## Validation Commands
Execute every command to validate the chore is complete with zero regressions.

```bash
pytest -v
make lint
make format
```

## Notes
[From Step 0.2, 0.3 - any gotchas or context]
```

* Write plan to specs/chore-[slug].md
* Proceed immediately to /implement with this plan

## Output Format

* **ChoreAnalysis**: Summary from linus-kernel-planner (type, complexity, scope)
* **ResearchFindings**: Files, tests, configs from codebase-researcher
* **ImplementationOptions**: O1-O4 with tradeoffs
* **ChosenOption**: Selected option with rationale
* **GeneratedPlan**: Full chore plan (written to specs/)
* **NextStep**: "Proceeding to /implement with generated plan"
