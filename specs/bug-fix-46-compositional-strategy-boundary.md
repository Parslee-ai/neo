# Bug Fix: test_infer_compositional_strategy Boundary Condition Mismatch

## Bug Description

The test `test_infer_compositional_strategy` in `tests/test_strategy_evolution.py` consistently fails because of a boundary condition mismatch between test expectations and algorithm logic.

**Test Expectation**: Exactly 60% hard success rate (6/10 = 0.60) should classify as "compositional"
**Algorithm Implementation**: Uses `hard_rate > 0.60` (strictly greater than), requiring >60% for compositional classification
**Result**: Test fails because 0.60 is NOT > 0.60

## Problem Statement

The algorithm at `src/neo/persistent_reasoning.py:1330` uses strict inequality (`>`) when checking the hard success rate threshold for compositional strategy classification. The test was written expecting exactly 60% to qualify as "compositional", but the algorithm intentionally excludes the 60% boundary based on a comment that states: "Use > 0.60 to distinguish from adaptive patterns that have exactly 60% hard success".

This creates an ambiguity: Does ">60% hard success" mean "strictly greater than 60%" or "60% or higher"?

## Solution Statement

**Chosen Approach**: Fix the test to use 70% (7/10) hard success rate instead of 60% (6/10).

This is the simplest fix that:
1. Respects the documented algorithm design intent (strict >60%)
2. Aligns with other passing integration tests that use 70%+ hard success
3. Avoids changing production code
4. Preserves the semantic distinction between adaptive (40-60%) and compositional (>60%) strategies

## Steps to Reproduce

```bash
# Run the failing test
pytest tests/test_strategy_evolution.py::TestStrategyInference::test_infer_compositional_strategy -v

# Expected output: FAILED
# Expected: 'compositional'
# Actual: 'adaptive'
```

## Root Cause Analysis

### The Boundary Condition

The algorithm defines three strategy levels based on hard success rate:
- **Procedural**: `hard_rate < 0.40` (struggles with complexity)
- **Adaptive**: `0.40 <= hard_rate <= 0.60` (handles complexity reasonably)
- **Compositional**: `hard_rate > 0.60 AND merge_count > 3` (excels at complexity)

The test creates an entry with:
- `hard_rate = 6/10 = 0.60` (exactly 60%)
- `merge_count = 4` (satisfies >3 requirement)

The test expects "compositional", but the algorithm returns "adaptive" because 0.60 is not strictly greater than 0.60.

### Design Intent

The comment on line 1329 of `persistent_reasoning.py` explicitly states:
```python
# Use > 0.60 to distinguish from adaptive patterns that have exactly 60% hard success
```

This indicates the strict inequality is **intentional design**, not a bug. The algorithm reserves "compositional" classification for patterns that clearly excel (61%+) on hard problems.

### Supporting Evidence

Other tests confirm this design:
1. `test_strategy_inference_from_performance_patterns` (integration test) uses 70% hard success → compositional ✓
2. `test_strategy_inference_accuracy` (end-to-end test) uses 80% and 90% hard success → compositional ✓
3. The same end-to-end test has a hash table with 60% hard success but `merge_count=3` (not >3), which correctly classifies as "adaptive"

## Test Specification

### New Tests (Validate Fix)

**Primary Test Update**:
- **File**: `tests/test_strategy_evolution.py`
- **Function**: `test_infer_compositional_strategy()` (lines 15-36)
- **Purpose**: Validates compositional strategy classification
- **Changes**:
  - Line 31: Change `"hard": (6, 10)` to `"hard": (7, 10)` (70% hard success)
  - Update comment to clarify threshold is strict >60%
- **Setup**: Create entry with 70% easy, 70% medium, 70% hard success, merge_count=4
- **Assertions**: Assert strategy == "compositional"

**Optional Boundary Test** (documents the 60% boundary explicitly):
- **File**: `tests/test_strategy_evolution.py`
- **Function**: `test_infer_adaptive_at_60_percent_boundary()` (new test)
- **Purpose**: Documents that exactly 60% hard success is classified as "adaptive", not "compositional"
- **Setup**: Create entry with 60% hard success, merge_count=4
- **Assertions**: Assert strategy == "adaptive"

### Existing Tests (Regression Prevention)

All existing strategy inference tests should continue to pass:

- **File**: `tests/test_strategy_evolution.py`
  - `test_infer_procedural_strategy()` (line 38) - 90% easy, 20% hard → procedural
  - `test_infer_adaptive_strategy()` (line 61) - 70% easy, 50% hard → adaptive
  - `test_infer_strategy_no_data()` (line 84) - No data → adaptive (default)

- **File**: `tests/test_integration.py`
  - `test_strategy_inference_from_performance_patterns()` (line 195) - 70% hard → compositional
  - Lines 224-239 - 10% hard → procedural
  - Lines 241-256 - 50% hard → adaptive
  - Lines 460-476 - 100% hard → compositional

- **File**: `tests/test_reasoningbank_end_to_end.py`
  - `test_strategy_inference_accuracy()` (line 269) - DP (80%) and segment tree (90%) → compositional

### Full Regression Suite

```bash
# Run all strategy tests
pytest tests/test_strategy_evolution.py -v

# Run integration tests
pytest tests/test_integration.py -v

# Run end-to-end tests
pytest tests/test_reasoningbank_end_to_end.py -v

# Full test suite
pytest tests/ -v

# Linting
make lint
```

### Test Coverage Verification

After fixing the test:
1. Run `pytest tests/test_strategy_evolution.py -v` and confirm all 4 tests pass
2. Check git diff to verify only test file was modified
3. Verify test output shows the updated hard success rate (70%)
4. Optionally run full test suite to ensure no regressions

## Relevant Files

### Primary Files
- `tests/test_strategy_evolution.py` (line 31) - **FIX HERE**: Change test data from 60% to 70%
- `src/neo/persistent_reasoning.py` (line 1330) - Algorithm using strict `>` inequality (DO NOT CHANGE)

### Related Files (context only)
- `tests/test_integration.py` (lines 195-476) - Integration tests using 70%+ hard success
- `tests/test_reasoningbank_end_to_end.py` (lines 269-292) - End-to-end validation

## Step by Step Tasks

### Step 1: Update the Test Data
**File**: `tests/test_strategy_evolution.py`
**Line**: 31

Change:
```python
"hard": (6, 10),  # 60% hard success
```

To:
```python
"hard": (7, 10),  # 70% hard success - must be >60% (strict inequality)
```

**Why**: The test needs to use a value that clearly exceeds the 60% threshold to match the algorithm's strict inequality design.

### Step 2: Run the Fixed Test
```bash
pytest tests/test_strategy_evolution.py::TestStrategyInference::test_infer_compositional_strategy -v
```

**Expected**: Test should now PASS with "compositional" classification.

### Step 3: Run Full Strategy Test Suite
```bash
pytest tests/test_strategy_evolution.py -v
```

**Expected**: All 4 tests in the file should pass.

### Step 4: Run Integration Tests
```bash
pytest tests/test_integration.py -v
```

**Expected**: All integration tests should pass (no regressions).

### Step 5: Run End-to-End Tests
```bash
pytest tests/test_reasoningbank_end_to_end.py -v
```

**Expected**: All end-to-end tests should pass.

### Step 6: Run Full Test Suite
```bash
pytest tests/ -v
```

**Expected**: All tests pass with zero failures.

### Step 7: Run Linting
```bash
make lint
```

**Expected**: No linting errors.

### Step 8: Verify Git Diff
```bash
git diff tests/test_strategy_evolution.py
```

**Expected**: Should show only the change on line 31 (and possibly updated comment).

## Validation Commands

Execute these commands in order to validate the bug fix:

```bash
# 1. Run the specific failing test
pytest tests/test_strategy_evolution.py::TestStrategyInference::test_infer_compositional_strategy -v

# 2. Run all strategy tests
pytest tests/test_strategy_evolution.py -v

# 3. Run integration tests
pytest tests/test_integration.py -v

# 4. Run end-to-end tests
pytest tests/test_reasoningbank_end_to_end.py -v

# 5. Full test suite
pytest tests/ -v

# 6. Linting
make lint

# 7. Verify changes
git diff tests/test_strategy_evolution.py
```

## Notes

### Why Fix the Test Instead of the Algorithm?

1. **Design Intent**: The algorithm comment explicitly documents the strict >60% threshold
2. **Consistency**: Other passing tests use 70%, 80%, 90% hard success (all >60%)
3. **Risk**: Changing production code is riskier than changing test expectations
4. **Semantic Clarity**: Strict inequality creates clear separation between adaptive (≤60%) and compositional (>60%)

### Alternative Fix Considered

**Algorithm Change** (NOT recommended):
- Change `src/neo/persistent_reasoning.py:1330` from `hard_rate > 0.60` to `hard_rate >= 0.60`
- **Rejected because**:
  - Changes production behavior for all patterns with exactly 60% hard success
  - Contradicts the documented design intent (comment on line 1329)
  - Unknown impact on live memory classification
  - Higher risk than test-only change

### Boundary Condition Semantics

After this fix, the strategy classification boundaries are:
- **Procedural**: `hard_rate < 0.40` (exclusive upper bound)
- **Adaptive**: `0.40 <= hard_rate <= 0.60` (inclusive range)
- **Compositional**: `hard_rate > 0.60 AND merge_count > 3` (exclusive lower bound)

The 60% value sits at the boundary between "adaptive" (handling complexity reasonably) and "compositional" (excelling at complexity). The algorithm design intentionally treats exactly 60% as the upper limit of "adaptive" rather than the lower limit of "compositional".

### Test Data Rationale

Using 70% (7/10) for the test is appropriate because:
1. It clearly exceeds the 60% threshold (no ambiguity)
2. It matches the pattern used in integration tests
3. It represents a pattern that genuinely excels at hard problems (not a borderline case)
4. It maintains the same denominator (10) as other difficulty levels in the test

### Future Improvements

Consider adding an explicit test for the 60% boundary:
```python
def test_infer_adaptive_at_60_percent_boundary(self):
    """Test that exactly 60% hard success is classified as adaptive."""
    entry = ReasoningEntry(...)
    entry.difficulty_affinity = {
        "easy": (6, 10),
        "medium": (6, 10),
        "hard": (6, 10)  # Exactly 60%
    }
    entry.merge_count = 4  # >3, so would be compositional if hard_rate qualified

    bank = ReasoningBank(...)
    strategy = bank._infer_strategy_level(entry)
    assert strategy == "adaptive"  # Documents the boundary behavior
```

This test would explicitly document that 60% is treated as "adaptive", not "compositional".
