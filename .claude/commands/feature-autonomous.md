---
allowed-tools: Task, SlashCommand, Write
description: Autonomously plan and implement features using Phase 0 multi-agent research, then auto-implement with /implement
argument-hint: [feature description]
---

# Autonomous Feature Planning & Implementation

Generate a feature implementation plan using multi-agent research (Phase 0 from quick-ship-neo), then implement with full validation loop.

## Instructions

- You're autonomously planning and implementing a feature using Phase 0 research
- Phase 0: Use 3 agents to research and generate O1-O4 implementation options
- CRITICAL: Phase 0 MUST output explicit Test Specification
- After planning, automatically proceed to /implement with the generated plan

## Feature
$ARGUMENTS

## Phase 0: PreflightExplain

**Step 0.1: Parse Feature**
* Extract feature description from $ARGUMENTS
* Classify type (new functionality, enhancement, integration, infrastructure)
* Define acceptance criteria (feature is complete when...)
* Identify user story components (who, what, why)

**Step 0.2: Codebase Research**
* INVOKE Task tool:
  - subagent_type: "codebase-researcher"
  - description: "Research codebase for feature context"
  - prompt: "Research the codebase for feature: {feature_description}. Identify: (1) Existing patterns and conventions to follow, (2) Files that will need modification, (3) Similar features already implemented, (4) Architecture constraints and opportunities, (5) Dependencies and integration points. Provide structured report with file paths, line numbers, and code patterns."

**Step 0.3: Feature Analysis**
* INVOKE Task tool:
  - subagent_type: "debug-detective"
  - description: "Deep feature analysis"
  - prompt: "Analyze feature requirements: {feature_description}. Investigate: (1) Core functionality needed, (2) Edge cases and error scenarios, (3) User experience considerations, (4) Performance implications, (5) Security considerations, (6) Complexity assessment. Provide comprehensive analysis."

**Step 0.4: Generate Implementation Options**
* INVOKE Task tool:
  - subagent_type: "linus-kernel-planner"
  - description: "Generate O1-O4 implementation options"
  - prompt: "Create feature implementation plan for: {feature_description}. Generate exactly 4 options (O1-O4), each with: approach overview, files to create/modify, complexity assessment (LOC, risk level), architectural implications, risks and tradeoffs. Include: (1) Feature Assessment, (2) Rejected Approaches (over-engineered solutions), (3) Four Implementation Options with tradeoffs, (4) Recommended Solution (simplest that meets requirements), (5) What NOT to Do (avoid complexity). Follow existing patterns. Prefer simple, extensible solutions."

**Step 0.5: Synthesize Plan with Test Specification**
* Compile results from all agents
* Choose simplest viable option (usually O1 or O2 unless requirements demand more)
* Produce FeaturePlan with REQUIRED sections:

```markdown
# Feature: [feature name from chosen option]

## Feature Description
[from Step 0.1 and 0.3 - detailed description with purpose and value]

## User Story
As a [user type from Step 0.1]
I want to [action/goal from Step 0.1]
So that [benefit/value from Step 0.1]

## Problem Statement
[from Step 0.3 - problem or opportunity this addresses]

## Solution Statement
[chosen option approach from Step 0.4]

## Relevant Files
[From Step 0.2 and chosen option]

Use these files to implement the feature:
- **File**: [path]
  - **Why**: [reason - e.g., "Contains memory store, need to add consolidation method"]

### New Files
[If chosen option requires new files:]
- **File**: [path]
  - **Purpose**: [why this new file is needed - e.g., "New module for semantic pattern discovery"]

## Test Specification
REQUIRED: Explicit test specification

### New Tests (Validate Feature)
[For chosen option, specify all new tests needed:]

- **File**: [from codebase research - existing test file or new]
- **Function**: test_[feature_name]_[scenario]()
- **Purpose**: Validates [specific feature behavior]
- **Setup**: [test data, mocks, fixtures needed]
- **Assertions**: [what success looks like]

Example:
- **File**: tests/test_pattern_discovery.py
- **Function**: test_construct_discovers_semantic_patterns()
- **Purpose**: Validates pattern discovery from code samples
- **Setup**: Create reasoning bank with diverse code examples
- **Assertions**: Assert patterns extracted with correct similarity scores

[Include 3-5 test specifications covering:]
- Happy path (feature works as expected)
- Edge cases (empty data, max limits, etc.)
- Error handling (invalid input, failures)
- Integration (feature works with existing functionality)

### Existing Tests (Regression Prevention)
[From Step 0.2 codebase research:]
- **File**: [existing test file]
- **Function**: [test name] (line number)
- **Purpose**: Ensures [related functionality] still works

Example:
- **File**: tests/test_integration.py
- **Function**: test_reasoning_with_memory() (line 45)
- **Purpose**: Ensures reasoning flow unaffected by new feature

### Full Regression Suite
```bash
pytest
make lint
```

### Test Coverage Verification
- Run tests and confirm all new test functions appear in output
- Check test files were modified (git diff should show new test functions)
- Verify test coverage for new feature files is >80%
- Manually test feature end-to-end in browser/API

## Implementation Plan

### Phase 1: Foundation
[from chosen option - foundational work]
- Data structures and schemas
- Core algorithm scaffolding
- Storage layer preparation

### Phase 2: Core Implementation
[from chosen option - main implementation]
- Feature logic implementation
- CLI command integration
- Memory/storage integration

### Phase 3: Integration
[from chosen option - integration work]
- Connect with existing reasoning flow
- Update CLI help and documentation
- Error handling and edge cases

## Step by Step Tasks
IMPORTANT: Execute every step in order, top to bottom.

[From chosen option implementation steps - break down into detailed tasks]

### Step 1: [Foundation task]
- [detailed subtasks]

### Step 2: [Implementation task]
- [detailed subtasks]

REQUIRED: Include test creation step BEFORE validation:

### Step N: Write Tests
- Create test files per Test Specification section
- Write test functions with exact names from Test Specification
- Implement setup, assertions, and mocks as specified
- Verify test files appear in git diff
- Ensure test coverage >80% for new code

### Step N+1: Run Validation Commands
[From Test Specification regression suite]

## Testing Strategy

### Unit Tests
[from Test Specification - detail unit test approach]
- Test individual functions in isolation
- Mock external dependencies
- Cover edge cases and error paths

### Integration Tests
[from Step 0.3 - detail integration test approach]
- Test feature end-to-end
- Verify CLI command behavior
- Test with actual LLM providers (mocked)

### Edge Cases
[from Step 0.3 and chosen option]
- Empty/null data
- Maximum limits
- Concurrent operations
- Network failures
- Invalid permissions

## Acceptance Criteria
[from Step 0.1 and chosen option]
- [ ] Feature works as described in User Story
- [ ] All tests pass (unit, integration, regression)
- [ ] CLI command works with all required flags
- [ ] Help text and documentation clear
- [ ] Error messages are user-friendly
- [ ] Performance meets requirements (reasonable for CLI tool)
- [ ] No regressions in existing functionality

## Validation Commands
Execute every command to validate the feature works correctly with zero regressions.

```bash
pytest -v
make lint
make format
neo --help  # Verify CLI command appears
```

## Notes
[From Step 0.2, 0.3, 0.4 - any gotchas, dependencies, or future considerations]
- Libraries added (if any): [list with `pip install` commands]
- Breaking changes: [if any - especially for CLI arguments]
- Storage changes: [if memory/storage format changes]
- Documentation to update: [README.md, CHANGELOG.md, etc.]
- Future enhancements: [nice-to-haves deferred]
```

* Write plan to specs/feature-[slug].md
* Proceed immediately to /implement with this plan

## Output Format

* **FeatureAnalysis**: Summary from debug-detective (requirements, complexity, risks)
* **ResearchFindings**: Patterns, files, architecture from codebase-researcher
* **ImplementationOptions**: O1-O4 from linus-kernel-planner with tradeoffs
* **ChosenOption**: Selected option with rationale
* **GeneratedPlan**: Full feature plan (written to specs/)
* **NextStep**: "Proceeding to /implement with generated plan"
