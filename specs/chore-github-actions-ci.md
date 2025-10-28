# Chore: Add GitHub Actions CI for Automated Testing and Linting

## Chore Description

Add a GitHub Actions workflow to run tests and linting on every pull request, preventing broken code from being merged to main. Currently there is no automated CI pipeline - the project has comprehensive tests and linting tools configured, but they only run manually via `make lint` and `make test`.

**Problem**: Tests can fail and still be merged because there's no automated enforcement. The PR template has a manual checklist but developers can skip it.

**Solution**: Create a simple CI workflow that runs on all PRs and pushes to main, executing:
1. `ruff check src/ tests/` - Catch style violations and common bugs
2. `pytest` - Run the full test suite

This matches the existing local development commands and provides immediate feedback on PR status.

**Type**: Infrastructure/DevOps chore
**Impact**: Prevents test failures and style violations from reaching main branch

## Test Specification

**Skip this section** - This chore only creates a CI workflow file (infrastructure/config change, not code modification). The workflow itself will run existing tests, but we don't need new tests for the workflow file.

### Validation: Workflow Must Run Successfully

After creating the workflow, validate it works by:

1. **Push to test branch**: Workflow should trigger automatically
2. **Check workflow runs**: Verify it appears in Actions tab
3. **Verify steps pass**:
   - Checkout completes
   - Python 3.11 installs
   - Dependencies install
   - Ruff linting passes
   - Pytest passes
4. **Test failure detection**: Temporarily break a test, verify workflow fails with red X

### Existing Tests (Regression Prevention)

All 17 existing test files will be run by the new workflow:
- `tests/test_neo.py` - Core functionality
- `tests/test_integration.py` - Full pipeline
- `tests/test_schemas.py` - Schema validation
- `tests/test_storage_integration.py` - Storage layer
- `tests/test_structured_parser.py` - Parser logic
- `tests/test_construct.py` - Pattern library
- `tests/test_program_loader.py` - Dataset loading
- `tests/test_embedding_logic.py` - Embeddings
- `tests/test_cli_global_flags.py` - CLI flags
- `tests/test_cli_index_flag.py` - Index flag
- `tests/test_reasoningbank_end_to_end.py` - ReasoningBank E2E
- `tests/test_failure_learning.py` - Failure learning
- `tests/test_self_contrast.py` - Self-contrast
- `tests/test_strategy_evolution.py` - Strategy evolution
- `tests/test_security_fixes.py` - Security regressions
- `tests/test_json_serialization.py` - JSON serialization
- `tests/__init__.py` - Package init

### Full Regression Suite

```bash
# Run locally before pushing workflow
make lint    # Should pass (ruff check src/ tests/)
make test    # Should pass (pytest)

# After workflow is created, it will run these automatically on every PR
```

## Relevant Files

**Files to Create:**
- `.github/workflows/ci.yml` - New CI workflow file (25 lines)

**Files to Reference (not modify):**
- `pyproject.toml` - Contains ruff and pytest configuration (already correct)
- `Makefile` - Contains lint and test commands (workflow will replicate these)
- `tests/` - All 17 test files (will be run by workflow)

**Existing Infrastructure:**
- `.github/workflows/publish.yml` - Publishing workflow (will coexist with new CI workflow)
- `.github/dependabot.yml` - Already monitors GitHub Actions updates
- `.github/PULL_REQUEST_TEMPLATE.md` - Has manual checklist (now automated)

## Step by Step Tasks

### Step 1: Verify Current Tests Pass Locally

Before creating the workflow, confirm the current codebase passes all checks:

```bash
cd /Users/theleafnode/Documents/_projects/Parslee-AI/neo
make lint    # Run ruff linting
make test    # Run pytest
```

If either fails, fix issues first before proceeding. The workflow will run these exact commands.

### Step 2: Create CI Workflow File

Create `.github/workflows/ci.yml` with:

```yaml
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v6
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: pip install -e ".[dev]"

    - name: Lint with ruff
      run: ruff check src/ tests/

    - name: Run tests
      run: pytest
```

**Why this design:**
- Single job keeps it simple (not separate lint/test jobs)
- Python 3.11 matches the publish workflow
- Uses latest GitHub Actions (checkout@v4, setup-python@v6 per Dependabot)
- Installs dev dependencies to get pytest, ruff, black, mypy
- Runs ruff first for fast failure (faster than waiting for tests)
- Runs pytest with default config from pyproject.toml

### Step 3: Test Workflow on Branch

Don't push directly to main. Test the workflow first:

```bash
git checkout -b chore/add-ci-workflow
git add .github/workflows/ci.yml
git commit -m "chore: add GitHub Actions CI workflow for automated testing

- Runs ruff linting and pytest on all PRs
- Uses Python 3.11 on Ubuntu
- Prevents broken code from being merged
- Matches local 'make lint' and 'make test' commands

Resolves #47"
git push -u origin chore/add-ci-workflow
```

Then:
1. Open PR on GitHub
2. Navigate to Actions tab
3. Verify "CI" workflow appears and runs
4. Check all steps complete successfully (green checkmarks)
5. Confirm PR shows green status check

### Step 4: Verify Failure Detection

Test that the workflow correctly fails when tests fail:

```bash
# Temporarily break a test
echo "assert False, 'Testing CI failure detection'" >> tests/test_neo.py
git add tests/test_neo.py
git commit -m "test: verify CI failure detection"
git push
```

Verify:
- Workflow runs
- "Run tests" step fails with red X
- PR shows failing status check

Then revert:
```bash
git revert HEAD
git push
```

Verify workflow passes again.

### Step 5: Enable Branch Protection (Optional)

After confirming the workflow works, enable branch protection to enforce it:

1. Go to GitHub repo → Settings → Branches
2. Add branch protection rule for `main`
3. Check "Require status checks to pass before merging"
4. Search for and select "test" (the job name from ci.yml)
5. Check "Require branches to be up to date before merging"
6. Save changes

**Note**: If you're the only developer, you might want to skip this or allow admins to bypass. Branch protection can block your own PRs if you need to force-merge something.

### Step 6: Add CI Badge to README (Optional)

Add status badge to show CI health at a glance:

In `README.md`, add near the top (after title):

```markdown
[![CI](https://github.com/Parslee-ai/neo/actions/workflows/ci.yml/badge.svg)](https://github.com/Parslee-ai/neo/actions/workflows/ci.yml)
```

This shows a green badge when CI passes, red when it fails.

### Step 7: Merge and Document

After confirming everything works:

1. Merge the PR to main
2. Verify workflow runs on main branch
3. Close issue #47
4. Consider updating CONTRIBUTING.md to mention CI requirements

## Validation Commands

Execute these commands to validate the chore is complete with zero regressions:

```bash
# Local validation before pushing
make lint
make test

# After pushing, verify in GitHub:
# 1. Navigate to Actions tab
# 2. Confirm "CI" workflow exists
# 3. Check latest run shows all green checkmarks
# 4. Open a test PR and verify status check appears

# Verify workflow YAML is valid
cat .github/workflows/ci.yml  # Should match the template above

# Verify workflow runs on PR
gh pr list  # Find your PR number
gh pr checks <PR_NUMBER>  # Should show "test" check passing
```

## Notes

**Why O2 (Simple + Essential) was chosen:**

- **O1 rejected**: Just pytest without linting is too minimal. You already have ruff configured, enforce it.
- **O3 rejected**: Multi-version testing (Python 3.9-3.13) is valuable but adds complexity. You likely haven't tested on those versions yet and will discover compatibility issues. Better to get basic CI working first, then add version matrix in a follow-up.
- **O4 rejected**: Coverage reporting, branch protection with reviews, caching, and mypy is massive overkill. Gets you 2x value for 10x complexity. Do this incrementally later if needed.

**Gotchas:**

- **Existing code must pass**: If current code has lint violations or test failures, fix them FIRST before creating the workflow. Otherwise your first CI run will fail.
- **Don't add mypy yet**: Your `pyproject.toml` has `disallow_untyped_defs = false`, meaning typing is incomplete. Running mypy in CI will fail. Fix typing gradually in future PRs.
- **pytest-cov missing**: Documentation mentions coverage but `pytest-cov` isn't in dev dependencies. Don't add coverage to CI until you add that dependency.
- **Fast failure is good**: The workflow runs ruff before pytest because linting is faster (~5 seconds) than tests (~30 seconds). This gives faster feedback when code has style issues.

**Future enhancements** (do these later, not now):

- Add Python 3.9, 3.10, 3.12, 3.13 to test matrix
- Add pytest-cov and coverage reporting
- Add mypy type checking (after improving type coverage)
- Add caching for pip dependencies (saves ~15 seconds)
- Add pre-commit hooks for local enforcement
- Add Codecov or Coveralls integration

**What NOT to do:**

- Don't create separate workflows for lint/test/format (enterprise cosplay)
- Don't add coverage thresholds before measuring current coverage
- Don't require PR reviews if you're the only developer
- Don't add nightly scheduled runs yet
- Don't set up Slack/email notifications (PR status check is enough)
- Don't optimize pip caching before proving CI works

**Why this solves the problem:**

From issue #47: "This led to test failures being merged and not caught until later."

This workflow catches failures BEFORE merge by:
1. Running automatically on every PR (not optional)
2. Showing clear red/green status in PR UI
3. Blocking merge when tests fail (if branch protection enabled)
4. Running the same commands as local development (make lint && make test)

The workflow takes ~1-2 minutes to run and gives immediate feedback. No more "oops, I didn't run tests before merging."