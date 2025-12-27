---
name: implement
pattern: /implement
description: Fast implementation with neo protocol gates. Takes a pre-written plan and implements it with strategic review, economic analysis, risk assessment, code review, and testing. Skips research/analysis phases.
parameters:
  - name: plan_input
    description: Markdown plan block with approach, files to modify, acceptance criteria. Must include approach summary, files/symbols, constraints, and test plan.
    required: true
---

## Usage

```bash
# Paste plan, run command
/quick-implement

# Or inline (for single-line summaries)
/implement "Add user profile dropdown to sidebar: create ProfileMenu component, integrate with auth context, add tests, commit to feature/profile-menu"
```

## Purpose

Implements a pre-written plan with all neo protocol gates intact:
- Neo strategic review (catch over-engineering)
- Liotta economic analysis (validate leverage)
- Sentinel risk assessment (branch safety)
- Linus code review (kernel standards)
- Test validation (no regressions)
- Sentinel git ops (safe PR)

**No human gating.** Proceeds automatically unless stop rules trigger.

## Constraints

* No broken behavior
* Only @code-implementer writes code. Only @sentinel does git
* If branch risk > low → temp branch + PR + rollback
* Stop if complexity explodes or no safe path
* Each quick test ≤ 30s

## Turn Protocol

### Phase 1: Validate & ParallelPlan

**Step 1.0: Parse Input**
* Extract plan from parameter or last code block in conversation
* Restate: approach summary, files/symbols, constraints, acceptance criteria
* Classify risk: LOW/MEDIUM/HIGH (files changed, criticality, scope)

**Step 1.1: Neo Strategic Review** (parallel)
* INVOKE Task tool:
  - subagent_type: "neo:neo"
  - description: "Neo review of implementation plan"
  - prompt: "Review this implementation plan. Plan: {full_plan}. Provide: (1) Approach critique—is it the simplest viable path?, (2) Over-engineering red flags, (3) Simpler alternatives if any, (4) Key risks overlooked, (5) Missing edge cases or tests, (6) Concrete refinements. Be brutally honest."

**Step 1.2: Economic Analysis** (parallel)
* INVOKE Task tool:
  - subagent_type: "liotta"
  - description: "Economic analysis of plan"
  - prompt: "Analyze this plan for measurable impact. Plan: {full_plan}. Provide: (1) Scope & complexity assessment, (2) Impact on system (perf, maintainability, risk), (3) Measurable success metrics, (4) Risks and failure modes, (5) Recommend proceed/refine/stop."

**Step 1.3: Risk & Branch Strategy** (parallel)
* INVOKE Task tool:
  - subagent_type: "sentinel"
  - description: "Git risk assessment"
  - prompt: "Assess git risk. Files affected: {files_list}. Scope: {scope}. Provide: (1) Risk level (LOW/MEDIUM/HIGH), (2) Branch strategy (direct commit / feature branch / temp+PR), (3) Rollback plan, (4) PR checklist."

**Step 1.4: Compile ParallelPlan**
* Wait for all three agents
* Extract NeoDeltas: concrete refinements (simpler approaches, missing tests, edge cases)
* Summarize findings from all three
* If any agent recommends STOP or risk > MEDIUM, halt with evidence

### Phase 2: Decision

* Coordinator accepts plan if:
  - No stop signals from Neo/Liotta/Sentinel
  - Risk ≤ MEDIUM
  - Plan is clear and bounded
* State decision, acceptance criteria, stop rules
* Proceed to implementation loop

### Phase 3: Implementation Loop

Initialize: iteration = 1, reject_count = 0

Flow: 3.1 Implementation → 3.2 Code Review → 3.3 Outcome → 3.4a Test Verification → 3.4b Quick Tests → 3.5 Full Regression → 3.6 Git Operations

**Step 3.1: Code Implementation**
* INVOKE Task tool:
  - subagent_type: "code-implementer"
  - description: "Implement iteration #{iteration}"
  - prompt: "Implement the plan with mandatory test creation. Plan: {full_plan}. Iteration #{iteration}. Constraints: (1) Only files/symbols from plan, (2) Changes <100 lines per file if possible, (3) **CRITICAL: If plan has 'Test Specification' or 'Write Tests' section, you MUST create test code. Do NOT just document test approach - write actual test functions with the exact names from the plan.**, (4) Incorporate NeoDeltas only if constraints allow, (5) Comment non-obvious logic, (6) Follow project conventions and best practices for the detected language/framework. Test Creation Requirements (if applicable): Create/modify test files listed in plan's Test Specification, write test functions with exact names from plan, implement setup, mocks, and assertions as specified, test code must be executable (not pseudocode). Report: (1) Files modified (implementation + tests) with line counts, (2) Test files created/modified (if applicable), (3) Test functions added (list exact names from plan), (4) Approach summary, (5) Quick test method (≤30s), (6) Estimated diff. {If iteration > 1: Address @Linus feedback: {previous_feedback}}"

**Step 3.2: Code Review**
* INVOKE Task tool:
  - subagent_type: "Linus"
  - description: "Review iteration #{iteration}"
  - prompt: "Review this code. Plan: {full_plan}. Implementation (iteration #{iteration}): {implementation_summary}. Verdict: ACCEPT | NEEDS WORK | REJECT. Include: (1) Verdict, (2) Rationale, (3) If NEEDS WORK/REJECT: specific changes. Apply kernel standards & simplicity-first."

**Step 3.3: Outcome**
* **If ACCEPT:** → Step 3.4a
* **If NEEDS WORK:** → Step 3.1 (revision), increment iteration
* **If REJECT:**
  - Increment reject_count
  - If reject_count ≥ 5: HALT, Status: Stopped
  - Else: → Step 3.1, increment iteration

**Step 3.4a: Verify Test Coverage**
* Check if plan requires test creation:
  - Does plan have "Test Specification" section?
  - Does plan have "Write Tests" step?
* If NO test requirements → Skip to Step 3.4b
* If YES test requirements → INVOKE Task tool:
  - subagent_type: "test-validator"
  - description: "Verify tests exist for iteration #{iteration}"
  - prompt: "Verify test creation per plan requirements. Plan: {full_plan}. Implementation: {implementation_summary}. Verification Steps: (1) Extract test file paths from plan's Test Specification section, (2) Extract test function names from plan's Test Specification section, (3) Check if implementation report includes these test files, (4) Search for test function names in modified files, (5) Verify test functions match plan specifications. Verdict: PASS | FAIL. PASS: All required test files modified AND all test functions present. FAIL: Missing test files OR missing test functions OR tests don't match plan. Report: (1) Verdict (PASS/FAIL), (2) Expected test files (from plan): [list], (3) Actual test files modified: [list], (4) Expected test functions (from plan): [list], (5) Actual test functions found: [list], (6) Missing items (if FAIL): [list]"

* **If FAIL:** → Step 3.1 (NEEDS WORK - missing required tests)
* **If PASS or NO test requirements:** → Step 3.4b

**Step 3.4b: Run Quick Tests**
* INVOKE Task tool:
  - subagent_type: "test-validator"
  - description: "Run quick tests"
  - prompt: "Test the implementation (≤30s each). Plan: {full_plan}.

  IMPORTANT: Detect project type and run appropriate validation commands.

  Detection Strategy:
  1. Check for language-specific files (package.json, Cargo.toml, go.mod, requirements.txt, Gemfile, etc.)
  2. Look for framework indicators (next.config.js, vite.config.js, django settings, etc.)
  3. Examine plan's 'Validation Commands' section for quick test hints

  Common Quick Tests by Project Type (≤30s each):

  JavaScript/TypeScript:
    - Type checking: npx tsc --noEmit (if tsconfig.json exists)
    - Syntax validation: node --check <main-file> (fallback)
    - Import validation: Quick build test if build command exists

  Python:
    - Syntax check: python -m py_compile <changed-files>
    - Import check: python -c 'import <module>' for key modules
    - Type checking: mypy <files> (if mypy configured)

  Go:
    - Syntax/type check: go build -o /dev/null ./...
    - Format check: gofmt -l <changed-files>

  Rust:
    - Type check: cargo check
    - Format check: cargo fmt --check

  Ruby:
    - Syntax check: ruby -c <changed-files>
    - Load check: ruby -e 'require \"<file>\"'

  Generic Fallback:
    - File existence check for modified files
    - Basic syntax validation using language-specific linter if available

  Report: (1) Detected project type, (2) Commands executed, (3) Results (pass/fail), (4) Execution times, (5) Failures if any."

* **If FAIL:** → Step 3.1 (NEEDS WORK)
* **If PASS:** → Step 3.5

**Step 3.5: Full Regression Validation**
* INVOKE Task tool:
  - subagent_type: "test-validator"
  - description: "Run full regression suite"
  - prompt: "Execute comprehensive validation per plan's Validation Commands. Plan: {full_plan}.

  IMPORTANT: Run ALL commands from plan's 'Validation Commands' section. No time limits - this is comprehensive validation.

  Validation Strategy:
  1. FIRST: Look for 'Validation Commands' section in plan - if present, execute those commands exactly
  2. SECOND: If no validation commands in plan, detect project type and run standard validation suite

  Detection & Standard Validation by Project Type:

  JavaScript/TypeScript (package.json):
    - Full type check: npx tsc --noEmit (if TypeScript)
    - Linting: npm run lint (if configured)
    - Build: npm run build (if configured)
    - Tests: npm test -- --watchAll=false (if test script exists)

  Python (requirements.txt, setup.py, pyproject.toml):
    - Type check: mypy . (if mypy configured)
    - Linting: flake8 . or pylint <module> (if configured)
    - Tests: pytest or python -m unittest discover (if tests exist)
    - Format: black --check . (if black configured)

  Go (go.mod):
    - Build: go build ./...
    - Test: go test ./...
    - Vet: go vet ./...
    - Format: gofmt -l .

  Rust (Cargo.toml):
    - Check: cargo check
    - Test: cargo test
    - Clippy: cargo clippy
    - Format: cargo fmt --check

  Ruby (Gemfile):
    - Tests: bundle exec rspec or bundle exec rake test
    - Linting: bundle exec rubocop (if configured)

  Generic Fallback:
    - Run test command if detectable (make test, ./test.sh, etc.)
    - Run build command if detectable (make, make build, ./build.sh, etc.)
    - Basic linting if linter config found (.eslintrc, .rubocop.yml, etc.)

  Report: (1) Detected project type, (2) Commands executed (in order), (3) Results per command (exit code, duration), (4) Full output if any failures, (5) Verdict: PASS (all commands exit 0) or FAIL (any non-zero exit)"

* **If FAIL:** → Step 3.1 (NEEDS WORK - regression detected)
* **If PASS:** → Step 3.6

**Step 3.6: Commit & PR**
* INVOKE Task tool:
  - subagent_type: "sentinel"
  - description: "Commit and create PR"
  - prompt: "Git operations per branch strategy. Execute: (1) Create feature branch if needed, (2) Stage changes, (3) Commit with conventional message, (4) Push to remote, (5) Create PR with test results, metrics, rollback instructions. Provide: (1) Branch name, (2) Commit SHA, (3) PR URL, (4) Rollback one-liner."

### Phase 4: Completion

* Verify acceptance criteria met
* Report: PR URL, branch name, test summary, rollback instructions
* Status: Resolved

## Output Format

* **PlanSummary**: Approach, files, constraints, acceptance criteria (brief)
* **ParallelPlan**: Neo findings + NeoDeltas, Liotta analysis, Sentinel risk/strategy
* **Decision**: Rationale, stop rules, acceptance criteria
* **StepLog**: Iteration # → Implementer (code + tests) → Linus verdict → Test Coverage → Quick Tests → Full Regression → Git (concise)
* **TestCoverage**: Files tested, functions added, coverage verification
* **RegressionResults**: Full test suite output, build status
* **RiskAndRollback**: Top risks, blast radius, rollback command
* **Status**: Resolved (PR + branch + rollback) or Stopped (evidence)

## Validation & Rules

* Parse plan or halt for clarity
* **Verify Test Specification in plan - if missing, ask before proceeding**
* Missing acceptance criteria → ask before proceeding
* Neo/Liotta/Sentinel say STOP → Status: Stopped with evidence
* Linus rejects 5 times → Status: Stopped
* **Step 3.4a fails 3 times → Status: Stopped (implementer not writing tests)**
* Each quick test ≤ 30s (Step 3.4b)
* Full regression no time limit (Step 3.5)
* Enforce branch safety (Sentinel only)
* **Deliverables must include test coverage and regression results**
* Report deliverables: PlanSummary, ParallelPlan, Decision, StepLog, TestCoverage, RegressionResults, RiskAndRollback, Status

## Safety Contract

* **No human approval gates** — proceeds unless stop rules trigger
* **Plan validation required** — if unclear, ask for clarification before Phase 1
* **All agent boundaries respected** — code-implementer only writes code, sentinel only does git
* **Context guard** — if < 500 tokens remain, compress StepLog or stop
