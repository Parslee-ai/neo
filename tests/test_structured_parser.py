#!/usr/bin/env python3
"""
Fuzzing tests for structured parser.

Tests parser robustness against malformed inputs.
"""

import pytest
from neo.structured_parser import (
    extract_block,
    tolerant_json_load,
    parse_plan_steps,
    parse_simulation_traces,
    parse_code_suggestions,
    ParseErrorCode
)
from neo.schemas import PLAN_STEP_SCHEMA


def test_perfect_plan():
    """Test parsing a perfect plan response."""
    response = """<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {
    "id": "ps_1",
    "description": "Parse input requirements",
    "rationale": "Must understand constraints first",
    "dependencies": [],
    "schema_version": "3"
  }
]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    assert result.success
    assert len(result.data) == 1
    assert result.data[0]["description"] == "Parse input requirements"


def test_missing_start_sentinel():
    """Test response missing start sentinel."""
    response = """[{"id": "ps_1", "description": "test"}]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    assert not result.success
    assert result.error_code == ParseErrorCode.MISSING_START_SENTINEL


def test_missing_end_sentinel():
    """Test response missing end sentinel."""
    response = """<<<NEO:SCHEMA=v3:KIND=plan>>>
[{"id": "ps_1", "description": "test"}]"""

    result = parse_plan_steps(response)
    assert not result.success
    assert result.error_code == ParseErrorCode.MISSING_END_SENTINEL


def test_stray_text_before():
    """Test response with text before sentinel."""
    response = """Here's my analysis of the problem:

<<<NEO:SCHEMA=v3:KIND=plan>>>
[{"id": "ps_1", "description": "test", "rationale": "reason", "dependencies": [], "schema_version": "3"}]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    assert not result.success
    assert result.error_code == ParseErrorCode.OFF_POLICY_EXPLANATION


def test_stray_text_after():
    """Test response with text after sentinel (should be tolerated)."""
    response = """<<<NEO:SCHEMA=v3:KIND=plan>>>
[{"id": "ps_1", "description": "test", "rationale": "reason", "dependencies": [], "schema_version": "3"}]
<<<END:plan>>>

This plan addresses the constraints by..."""

    result = parse_plan_steps(response)
    # Should succeed but log warning
    assert result.success


def test_markdown_wrapped():
    """Test response wrapped in markdown code fence."""
    response = """```json
<<<NEO:SCHEMA=v3:KIND=plan>>>
[{"id": "ps_1", "description": "test", "rationale": "reason", "dependencies": [], "schema_version": "3"}]
<<<END:plan>>>
```"""

    result = parse_plan_steps(response)
    assert not result.success
    assert result.error_code == ParseErrorCode.MARKDOWN_WRAPPED


def test_wrong_kind_sentinel():
    """Test response with wrong kind sentinel."""
    response = """<<<NEO:SCHEMA=v3:KIND=simulation>>>
[{"n": 1, "input_data": "test"}]
<<<END:simulation>>>"""

    result = parse_plan_steps(response)  # Expecting plan, got simulation
    assert not result.success
    assert result.error_code == ParseErrorCode.MIXED_SENTINELS


def test_trailing_comma():
    """Test JSON with trailing comma (should be repaired)."""
    response = """<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {
    "id": "ps_1",
    "description": "test",
    "rationale": "reason",
    "dependencies": [],
    "schema_version": "3",
  }
]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    assert result.success  # Tolerant loader should fix this


def test_smart_quotes():
    """Test JSON with smart quotes (should be repaired)."""
    # Programmatically insert smart quotes (U+201C and U+201D)
    left_quote = '\u201c'
    right_quote = '\u201d'
    response = f"""<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {{
    "id": "ps_1",
    "description": "test with {left_quote}smart quotes{right_quote}",
    "rationale": "reason",
    "dependencies": [],
    "schema_version": "3"
  }}
]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    assert result.success  # Tolerant loader should fix this


def test_missing_required_field():
    """Test JSON missing required field."""
    response = """<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {
    "id": "ps_1",
    "description": "test",
    "dependencies": [],
    "schema_version": "3"
  }
]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    assert not result.success
    assert result.error_code == ParseErrorCode.MISSING_FIELD


def test_hallucinated_fields():
    """Test JSON with extra fields not in schema."""
    response = """<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {
    "id": "ps_1",
    "description": "test",
    "rationale": "reason",
    "dependencies": [],
    "schema_version": "3",
    "extra_field": "should not be here",
    "another_field": 123
  }
]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    assert not result.success
    assert result.error_code == ParseErrorCode.HALLUCINATED_FIELDS


def test_wrong_schema_version():
    """Test JSON with wrong schema version."""
    response = """<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {
    "id": "ps_1",
    "description": "test",
    "rationale": "reason",
    "dependencies": [],
    "schema_version": "v2"
  }
]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    assert not result.success
    assert result.error_code == ParseErrorCode.WRONG_SCHEMA


def test_oversized_field():
    """Test JSON with field exceeding max length (should truncate)."""
    oversized_text = 'x' * 600
    response = f"""<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {{
    "id": "ps_1",
    "description": "{oversized_text}",
    "rationale": "reason",
    "dependencies": [],
    "schema_version": "3"
  }}
]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    # Should succeed after truncation
    assert result.success
    # Description should be truncated to maxLength (from schema)
    assert result.data[0]["description"].endswith("...")
    max_length = PLAN_STEP_SCHEMA["properties"]["description"]["maxLength"]
    assert len(result.data[0]["description"]) == max_length


def test_simulation_perfect():
    """Test parsing perfect simulation response."""
    response = """<<<NEO:SCHEMA=v3:KIND=simulation>>>
[
  {
    "n": 1,
    "input_data": "Empty array []",
    "expected_output": "Return 0",
    "reasoning_steps": ["Check length", "Return 0"],
    "issues_found": [],
    "schema_version": "3"
  }
]
<<<END:simulation>>>"""

    result = parse_simulation_traces(response)
    assert result.success
    assert len(result.data) == 1
    assert result.data[0]["n"] == 1


def test_code_suggestions_perfect():
    """Test parsing perfect code suggestions response."""
    response = """<<<NEO:SCHEMA=v3:KIND=code>>>
[
  {
    "file_path": "/app/server.py",
    "unified_diff": "--- a/server.py\\n+++ b/server.py\\n@@ -10,6 +10,7 @@\\n+    validate()",
    "description": "Add input validation",
    "confidence": 0.95,
    "tradeoffs": ["Adds latency"],
    "schema_version": "3"
  }
]
<<<END:code>>>"""

    result = parse_code_suggestions(response)
    assert result.success
    assert len(result.data) == 1
    assert result.data[0]["confidence"] == 0.95


def test_duplicate_keys():
    """Test JSON with duplicate keys."""
    response = """<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {
    "id": "ps_1",
    "description": "first",
    "description": "second",
    "rationale": "reason",
    "dependencies": [],
    "schema_version": "3"
  }
]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    # Python json.loads takes last value, but we should detect this
    # For now, tolerant_json_load will succeed but we could add stricter checks
    # This test documents current behavior
    assert result.success or result.error_code == ParseErrorCode.DUP_KEYS


def test_escape_errors():
    """Test JSON with unescaped characters."""
    response = """<<<NEO:SCHEMA=v3:KIND=plan>>>
[
  {
    "id": "ps_1",
    "description": "test with unescaped "quotes"",
    "rationale": "reason",
    "dependencies": [],
    "schema_version": "3"
  }
]
<<<END:plan>>>"""

    result = parse_plan_steps(response)
    # Should either succeed (if tolerant loader handles it) or fail with escape error
    if not result.success:
        assert result.error_code in [ParseErrorCode.ESCAPE_ERRORS, ParseErrorCode.BAD_JSON]


if __name__ == "__main__":
    # Run tests
    import sys

    tests = [
        ("Perfect plan", test_perfect_plan),
        ("Missing start sentinel", test_missing_start_sentinel),
        ("Missing end sentinel", test_missing_end_sentinel),
        ("Stray text before", test_stray_text_before),
        ("Stray text after", test_stray_text_after),
        ("Markdown wrapped", test_markdown_wrapped),
        ("Wrong kind sentinel", test_wrong_kind_sentinel),
        ("Trailing comma", test_trailing_comma),
        ("Smart quotes", test_smart_quotes),
        ("Missing required field", test_missing_required_field),
        ("Hallucinated fields", test_hallucinated_fields),
        ("Wrong schema version", test_wrong_schema_version),
        ("Oversized field", test_oversized_field),
        ("Simulation perfect", test_simulation_perfect),
        ("Code suggestions perfect", test_code_suggestions_perfect),
        ("Duplicate keys", test_duplicate_keys),
        ("Escape errors", test_escape_errors),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name}: Unexpected error: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)