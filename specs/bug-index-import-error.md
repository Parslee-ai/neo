# Bug: --index flag fails with ModuleNotFoundError on 'src' module

## Bug Description

Running `neo --index` fails with `ModuleNotFoundError: No module named 'src'` because the CLI uses an incorrect import path. The command crashes immediately when attempting to import the `ProjectIndex` class, preventing users from building semantic indexes for their codebases.

**Error Output:**
```
Traceback (most recent call last):
  File "/opt/homebrew/Caskroom/miniconda/base/bin/neo", line 8, in <module>
    sys.exit(main())
  File ".../neo/src/neo/cli.py", line 2778, in main
    from src.index.project_index import ProjectIndex
ModuleNotFoundError: No module named 'src'
```

**Impact:**
- Severity: HIGH - Complete feature failure
- Affects all users attempting to use the `--index` flag
- Feature has been non-functional since initial release
- No workaround available for end users

## Problem Statement

The root cause is an architectural mismatch between the directory structure and package configuration:

1. The `ProjectIndex` class is located at `src/index/project_index.py` (outside the `neo` package)
2. The package is configured in `pyproject.toml` to only include `src/neo` (line 68)
3. The import at `cli.py:2778` uses `from src.index.project_index import ProjectIndex`
4. When neo is installed, only the `neo` package is available - `src` is not a valid module

The bug was likely undetected during development because:
- pytest configuration adds `src/` to the Python path (pyproject.toml line 72)
- Editable installs make the import work in the development environment
- No integration tests validate the `--index` flag with an installed package

## Solution Statement

Move `project_index.py` into the `neo` package structure where it belongs:
1. Create `src/neo/index/` directory
2. Move `src/index/project_index.py` to `src/neo/index/project_index.py`
3. Update the import to use the correct package path: `from neo.index.project_index import ProjectIndex`

This follows Python packaging best practices and keeps all Neo code under a single namespace.

## Steps to Reproduce

1. Ensure neo is installed (not just editable install in dev environment)
2. Navigate to any directory: `cd /Users/theleafnode/Documents/_projects/Parslee-AI/neo`
3. Run: `neo --index`
4. Observe the crash with `ModuleNotFoundError: No module named 'src'`

## Root Cause Analysis

**Architectural Issue:**
The `src/index/` directory was created outside the `neo` package, but was never properly configured as a separate package. This creates an inconsistency:
- Development: Works because pytest adds `src/` to path and editable install includes source directory
- Production: Fails because only `neo` package is installed, not `src` or `index`

**Import Pattern:**
Only ONE location in the entire codebase uses this broken import pattern (verified by grep). All other imports correctly use the `neo.` prefix convention.

**Missing Package Configuration:**
The `src/index/` directory lacks:
- An `__init__.py` file (not a proper Python package)
- Entry in `pyproject.toml` packages list
- Integration tests validating it works when installed

## Test Specification

### New Tests (Validate Fix)

**File**: `tests/test_cli_index_flag.py` (new file)

**Function**: `test_index_flag_imports_successfully()`
- **Purpose**: Validates that the `--index` flag can import ProjectIndex without errors
- **Setup**: Import the ProjectIndex class as the CLI would
- **Assertions**: Import succeeds without ModuleNotFoundError

**Function**: `test_index_flag_basic_functionality(tmp_path)`
- **Purpose**: Validates that `--index` flag can build an index
- **Setup**: Create a temporary directory with a few Python files
- **Assertions**:
  - Command runs without crashing
  - `.neo/` directory is created
  - Index files exist (index.json, chunks.json, faiss.index)

**Function**: `test_project_index_in_neo_package()`
- **Purpose**: Validates that ProjectIndex is properly packaged under neo
- **Setup**: Attempt to import using the correct package path
- **Assertions**: `from neo.index.project_index import ProjectIndex` succeeds

### Existing Tests (Regression Prevention)

**File**: `tests/test_cli_global_flags.py`
- **Function**: `test_version_flag()` (line 10)
- **Purpose**: Ensures other CLI flags still work
- **Why**: Verifies we didn't break the CLI parser

**File**: `tests/test_integration.py`
- **Function**: `test_neo_help()` (line 8)
- **Purpose**: Ensures CLI help still displays correctly
- **Why**: Verifies basic CLI functionality unaffected

### Full Regression Suite

```bash
pytest -v
make lint
```

### Test Coverage Verification

After implementing the fix:
1. Run `pytest -v tests/test_cli_index_flag.py` and verify new test functions appear
2. Run `git diff tests/` to confirm test file was created with new test functions
3. Run `pytest -v` and verify all tests pass (both new and existing)
4. Manually test: `neo --index` in a sample directory to verify it works

## Relevant Files

**Primary Files:**
- `src/neo/cli.py` - Line 2778 has the broken import (1 line change)
- `src/index/project_index.py` - Needs to move to `src/neo/index/project_index.py`

**Supporting Files:**
- `pyproject.toml` - Package configuration (no changes needed - already includes src/neo)
- `src/neo/__init__.py` - Package initialization (no changes needed)

**New Files:**
- `src/neo/index/__init__.py` - Make index a proper subpackage (create empty file)
- `tests/test_cli_index_flag.py` - New test file for --index functionality

## Step by Step Tasks

### Step 1: Create neo/index directory structure
- Create directory: `src/neo/index/`
- Create package marker: `src/neo/index/__init__.py` (empty file)

### Step 2: Move project_index.py into neo package
- Use `git mv` to preserve history: `git mv src/index/project_index.py src/neo/index/project_index.py`
- Remove empty directory: `rm -rf src/index/`

### Step 3: Update import in cli.py
- Edit `src/neo/cli.py` line 2778
- Change: `from src.index.project_index import ProjectIndex`
- To: `from neo.index.project_index import ProjectIndex`

### Step 4: Write Tests
- Create test file: `tests/test_cli_index_flag.py`
- Write test function: `test_index_flag_imports_successfully()`
  - Purpose: Validates import works
  - Setup: `from neo.index.project_index import ProjectIndex`
  - Assertions: Import succeeds without exception
- Write test function: `test_index_flag_basic_functionality(tmp_path)`
  - Purpose: Validates index can be built
  - Setup: Create temp directory with sample Python files, run index creation
  - Assertions: Verify .neo/ directory and index files created
- Write test function: `test_project_index_in_neo_package()`
  - Purpose: Validates package structure
  - Setup: Import statement verification
  - Assertions: Correct import path works

### Step 5: Run Validation Commands
- Execute: `pytest -v tests/test_cli_index_flag.py`
- Execute: `pytest -v` (full suite)
- Execute: `make lint`
- Manual test: `neo --index` in a real directory

### Step 6: Verify fix manually
- Navigate to a Python project directory
- Run: `neo --index`
- Confirm: No ModuleNotFoundError
- Confirm: `.neo/` directory created with index files

## Validation Commands

Execute every command to validate the bug is fixed with zero regressions.

```bash
# Run new tests specifically
pytest -v tests/test_cli_index_flag.py

# Run full test suite
pytest -v

# Lint check
make lint

# Manual integration test
cd /tmp && mkdir test_neo_index && cd test_neo_index
echo "def hello(): print('world')" > sample.py
neo --index
ls -la .neo/  # Should show index files
cd - && rm -rf /tmp/test_neo_index
```

## Notes

**Why Option 2 was chosen:**
- Simplest architecturally - everything under one namespace (`neo`)
- Most Pythonic - follows standard package layout conventions
- Future-proof - if more index-related modules added, they have a home
- Clean imports - no sys.path hacks, no namespace pollution
- Low risk - only one file imports ProjectIndex (verified by grep)

**Alternative options considered:**
- O1: sys.path manipulation (creates technical debt)
- O3/O4: Separate `index` package (namespace pollution, architectural complexity)

**Testing gap identified:**
The codebase lacks integration tests that validate installed package behavior (as opposed to editable install behavior). This is why the bug went undetected. The new tests should help prevent similar issues.

**Git history preservation:**
Using `git mv` instead of manual move+delete preserves the file history for `project_index.py`, making it easier to track changes over time.

**No changes needed to pyproject.toml:**
Since we're moving the file INTO the existing `src/neo` package (which is already configured in line 68), no package configuration changes are required.
