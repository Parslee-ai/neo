# Bug: Index Freshness Check Ignores Pattern File Modifications

## Bug Description

The index freshness check at `src/neo/construct.py:348-352` only considers the modification time of the index file itself (checking if it's less than 1 hour old), but does not verify whether any pattern markdown files in the `/construct` directory have been updated since the index was last built.

This creates a scenario where users can edit pattern files, but the index won't rebuild automatically within the 1-hour window, causing semantic search to return stale results. The user's only workaround is to manually run `neo construct index --force`.

**Severity**: Medium - Silent data staleness where users see outdated search results without warning

**Impact**: Users editing patterns within the 1-hour TTL window will not see their changes reflected in search results

## Problem Statement

The current implementation uses **time-based freshness** - "is the index less than 1 hour old?" This is fundamentally flawed because it ignores the actual source of truth: the pattern markdown files themselves.

The correct approach is **content-based freshness** - "is the index newer than all source pattern files?" If ANY pattern file has a newer modification time than the index, the index is stale and must be rebuilt.

**Current Code (Lines 348-352):**
```python
# Check if index exists and is recent
if not force_rebuild and self.index_path.exists() and self.metadata_path.exists():
    age_seconds = time.time() - self.index_path.stat().st_mtime
    if age_seconds < 3600:  # Less than 1 hour old
        logger.info(f"Index is recent ({age_seconds:.0f}s old), skipping rebuild")
        return {'status': 'skipped', 'reason': 'index_recent'}
```

## Solution Statement

Add a simple file modification time comparison before the age check. Compare the index's modification time against every pattern file in `self.construct_root`. If ANY pattern file is newer than the index, treat the index as stale and proceed with rebuild.

This is a straightforward fix: scan pattern files with `rglob('*.md')`, check if any have `mtime > index_mtime`, and only skip rebuild if the index is newer than all pattern files.

## Steps to Reproduce

1. Build fresh index: `neo construct index`
2. Immediately modify a pattern file (within 1-hour window):
   ```bash
   echo "## New Section\nAdded content" >> construct/caching/cache-aside.md
   ```
3. Try to rebuild without force: `neo construct index`
4. **Expected**: Index rebuilds because pattern file was modified
5. **Actual**: Index skipped with message "Index is recent (30s old), skipping rebuild"
6. Search returns stale results that don't include the new content
7. Only workaround: `neo construct index --force`

## Root Cause Analysis

The original implementation optimized for performance by avoiding expensive index rebuilds using a simple TTL-based cache (1-hour). The reasoning likely was:

1. Pattern files change infrequently (assumed stable pattern library)
2. Embedding is expensive (building embeddings takes time and API calls)
3. Simple TTL is easy to implement
4. Assumed users would manually force rebuild when needed

However, this makes a **critical assumption**: pattern files don't change within the TTL window. This breaks in real-world scenarios:

- **Rapid iteration during development** - Developer edits pattern, runs search immediately
- **Multiple patterns updated in quick succession** - Batch edits within 1-hour window
- **Pattern deletion** - Deleted files still appear in index
- **Pattern file corruption fixes** - Quick fixes aren't reflected
- **Cross-team collaboration** - Pull updates that modify patterns within TTL

**Edge Cases:**
- Pattern file deleted after index built - search returns deleted pattern
- New pattern file added - won't appear in searches until TTL expires
- Pattern file renamed - old name in index, new file not indexed
- No pattern files exist - index continues to serve stale data

## Test Specification

### New Tests (Validate Fix)

**File**: `tests/test_construct.py`

**Function**: `test_construct_index_respects_pattern_file_changes()`
- **Purpose**: Validates that index rebuilds when pattern files are modified after index creation
- **Setup**:
  1. Create temp construct directory with sample pattern files
  2. Build initial index with `force_rebuild=True`
  3. Sleep 0.1 seconds to ensure different mtime
  4. Modify one pattern file (append new content)
  5. Call `build_index(force_rebuild=False)`
- **Assertions**:
  - Assert second build returns `status='success'` (not 'skipped')
  - Assert build does not return `reason='index_fresh'`
  - Verify modified pattern is in rebuilt index

**Function**: `test_construct_index_skips_when_fresh()`
- **Purpose**: Validates that index still skips rebuild when all patterns are older than index
- **Setup**:
  1. Create temp construct directory with sample pattern files
  2. Build initial index with `force_rebuild=True`
  3. Don't modify any files
  4. Call `build_index(force_rebuild=False)` immediately
- **Assertions**:
  - Assert returns `status='skipped'`
  - Assert returns `reason='index_fresh'`

**Function**: `test_construct_index_detects_new_pattern_file()`
- **Purpose**: Validates that new pattern files trigger rebuild
- **Setup**:
  1. Create temp construct directory with 2 pattern files
  2. Build initial index
  3. Add a 3rd pattern file
  4. Call `build_index(force_rebuild=False)`
- **Assertions**:
  - Assert rebuild occurs (status='success')
  - Assert new pattern appears in index

**Function**: `test_construct_index_handles_deleted_pattern()`
- **Purpose**: Validates that deleted patterns trigger rebuild
- **Setup**:
  1. Create temp construct directory with 3 pattern files
  2. Build initial index
  3. Delete one pattern file
  4. Call `build_index(force_rebuild=False)`
- **Assertions**:
  - Assert rebuild occurs
  - Assert deleted pattern not in rebuilt index

### Existing Tests (Regression Prevention)

**File**: `tests/test_construct.py`
- **Function**: `test_construct_index_build_performance()` (line 330-345)
  - **Purpose**: Ensures normal build performance unaffected
  - **Verification**: Run test, confirm timing thresholds still met

- **Function**: `test_construct_index_build()` (if exists in TestConstructIndex class)
  - **Purpose**: Ensures basic index building still works
  - **Verification**: All existing assertions pass

**File**: `tests/test_cli_index_flag.py`
- **Function**: `test_project_index_in_neo_package()` (line 75-78)
  - **Purpose**: Ensures CLI integration unaffected
  - **Verification**: Test passes without modification

### Full Regression Suite

```bash
pytest tests/test_construct.py -v
pytest tests/test_cli_index_flag.py -v
pytest -v  # Full suite
make lint
```

### Test Coverage Verification

After implementing the fix and tests:

1. **Run tests and verify new functions appear in output:**
   ```bash
   pytest tests/test_construct.py::test_construct_index_respects_pattern_file_changes -v
   ```

2. **Check test file was modified:**
   ```bash
   git diff tests/test_construct.py
   # Should show new test functions
   ```

3. **Manually verify test covers the code path:**
   - Set breakpoint at line 348 in construct.py
   - Run test in debug mode
   - Confirm new mtime comparison logic is executed

4. **Coverage report (optional):**
   ```bash
   pytest --cov=src/neo/construct --cov-report=term-missing tests/test_construct.py
   ```

## Relevant Files

**Primary Bug Location:**
- `src/neo/construct.py`
  - Lines 348-352: Buggy freshness check
  - Lines 335-429: `build_index()` method
  - Lines 355: Pattern file scan (already exists, can reuse)

**Test Files:**
- `tests/test_construct.py`
  - Lines 77-99: `temp_construct_dir` fixture (for test setup)
  - Lines 227-392: `TestConstructIndex` class (add new tests here)

**Reference Implementation (Correct Pattern):**
- `src/neo/index/project_index.py`
  - Lines 376-408: `check_staleness()` method (shows correct approach with file hash tracking)
  - Note: We'll use simpler mtime approach instead of hashes

**CLI Integration:**
- `src/neo/cli.py`
  - Lines 2521-2532: Index build handling (will benefit from fix, no changes needed)

## Step by Step Tasks

### Step 1: Understand Current Implementation
- Read `build_index()` method in construct.py (lines 335-429)
- Identify the freshness check logic (lines 348-352)
- Understand how pattern files are scanned (line 355)
- Review existing logging and return values

### Step 2: Implement File Modification Time Check
- Modify `build_index()` method at lines 348-352
- Add logic to scan pattern files: `pattern_files = list(self.construct_root.rglob('*.md'))`
- For each pattern file, check if `pattern_file.stat().st_mtime > index_mtime`
- If any pattern is newer, proceed with rebuild (don't skip)
- If all patterns are older than index, keep existing age check behavior
- Update log messages to reflect new logic

**Implementation (Option 1 - Recommended):**
```python
# Check if index exists and is fresh relative to source files
if not force_rebuild and self.index_path.exists() and self.metadata_path.exists():
    index_mtime = self.index_path.stat().st_mtime

    # Check if any pattern file is newer than the index
    pattern_files = list(self.construct_root.rglob('*.md'))
    stale = False
    for pattern_file in pattern_files:
        if pattern_file.stat().st_mtime > index_mtime:
            logger.info(f"Pattern file {pattern_file.name} modified after index, rebuilding")
            stale = True
            break

    if not stale:
        age_seconds = time.time() - index_mtime
        if age_seconds < 3600:  # Less than 1 hour old
            logger.info(f"Index is fresh ({age_seconds:.0f}s old), skipping rebuild")
            return {'status': 'skipped', 'reason': 'index_fresh'}

# Continue with existing build logic (line 355 onwards)...
```

**Lines Changed**: ~15 lines (adding ~10 new lines, modifying ~5 existing)

**Risk Level**: LOW
- Pure read operations (no writes)
- Fails safe (any error triggers rebuild)
- Early exit on first stale file (efficient)
- No schema changes to metadata

### Step 3: Write Tests
- Open `tests/test_construct.py`
- Add test function `test_construct_index_respects_pattern_file_changes()` to `TestConstructIndex` class
- Implement test setup using `temp_construct_dir` fixture
- Write assertions per Test Specification above
- Add additional test functions for edge cases (new files, deleted files)

**Test Implementation:**
```python
def test_construct_index_respects_pattern_file_changes(temp_construct_dir):
    """Test that index rebuilds when pattern files are modified."""
    index = ConstructIndex(construct_root=temp_construct_dir)

    # Build initial index
    with patch('neo.construct.FASTEMBED_AVAILABLE', True):
        with patch('neo.construct.TextEmbedding'):
            result1 = index.build_index(force_rebuild=True)
            assert result1['status'] == 'success'

            # Modify a pattern file
            pattern_file = temp_construct_dir / 'caching' / 'cache-aside.md'
            original_content = pattern_file.read_text()
            time.sleep(0.1)  # Ensure different mtime
            pattern_file.write_text(original_content + "\n## Updated\nNew content")

            # Build again without force_rebuild
            result2 = index.build_index(force_rebuild=False)

            # Should rebuild (not skip) because pattern file is newer
            assert result2['status'] == 'success'
            assert result2['reason'] != 'index_fresh'
```

### Step 4: Run Tests Locally
```bash
# Run new tests specifically
pytest tests/test_construct.py::test_construct_index_respects_pattern_file_changes -v

# Run full construct test suite
pytest tests/test_construct.py -v

# Verify no regressions
pytest -v
```

### Step 5: Manual Verification
```bash
# Test the actual CLI behavior
cd /Users/theleafnode/Documents/_projects/Parslee-AI/neo

# Build index
neo construct index --force

# Modify a pattern file
echo "## Test Section" >> construct/caching/cache-aside.md

# Rebuild without force - should detect change
neo construct index

# Verify log message shows pattern file was detected as modified
# Expected: "Pattern file cache-aside.md modified after index, rebuilding"
# NOT: "Index is recent (30s old), skipping rebuild"

# Search to verify updated content appears
neo construct search "cache aside"
```

### Step 6: Run Validation Commands

Execute every command to validate the bug is fixed with zero regressions:

```bash
# Run full test suite
pytest -v

# Run linting
make lint

# Verify specific construct tests pass
pytest tests/test_construct.py -v

# Check test coverage for modified code
pytest --cov=src/neo/construct --cov-report=term-missing tests/test_construct.py
```

## Validation Commands

Execute every command to validate the bug is fixed with zero regressions.

```bash
# Full test suite
pytest -v

# Linting
make lint

# Specific test for bug fix
pytest tests/test_construct.py::test_construct_index_respects_pattern_file_changes -v

# Coverage check
pytest --cov=src/neo/construct --cov-report=term tests/test_construct.py
```

## Notes

### Why This Fix Is Correct

1. **Solves the actual problem**: Checks if source files are newer than index
2. **Minimal complexity**: ~10 lines of straightforward code
3. **Fails safe**: Any error just triggers rebuild (correct behavior)
4. **No schema changes**: Doesn't modify metadata format
5. **Performance**: Early exit on first stale file, no unnecessary work

### What NOT to Do (Rejected Approaches)

- **Don't store content hashes**: File mtimes exist for this exact use case. Hashing adds complexity and CPU cost for zero benefit.
- **Don't build a config system for the 1-hour threshold**: It's a reasonable default.
- **Don't add file watching or background indexing**: This is a CLI tool used interactively.
- **Don't try to detect renames vs edits**: We care about "has content changed", not "what kind of change was it".
- **Don't make this async**: Premature optimization. With <100 patterns, this check takes <10ms.

### Edge Cases Handled

- **Empty pattern directory**: Loop doesn't execute, falls through to existing logic
- **Pattern file deleted**: Not newer than index, but will be caught by build step
- **New pattern added**: New file's mtime is after index build, triggers rebuild
- **Clock skew**: Worst case is unnecessary rebuild (safe failure mode)
- **Symlinks**: `stat()` follows symlinks by default (correct behavior)
- **File permissions**: Caught by general exception handling in build process

### Performance Impact

- **rglob('*.md')**: Fast (< 1ms for typical pattern library of ~100 files)
- **stat().st_mtime**: Cheap syscall (< 1µs per file)
- **Early exit**: Stops on first stale file found
- **Total overhead**: < 10ms for 100 patterns (negligible)

### Reference: ProjectIndex Comparison

The `ProjectIndex` class in `src/neo/index/project_index.py` has a superior implementation using file hash tracking (`check_staleness()` method, lines 376-408). However, for `ConstructIndex`:
- Pattern library is smaller (~100 files vs potentially thousands)
- Patterns change less frequently than source code
- Simple mtime check is sufficient and faster
- No need for hash-based change detection complexity
