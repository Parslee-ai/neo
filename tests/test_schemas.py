"""Tests for JSON schema validation."""
import pytest
from jsonschema import validate, ValidationError
from neo.schemas import CODE_SUGGESTION_SCHEMA, PLAN_STEP_SCHEMA, SCHEMA_VERSION


def test_code_suggestion_minimal():
    """Test CODE_SUGGESTION_SCHEMA with only required fields."""
    suggestion = {
        "file_path": "src/api.py",
        "unified_diff": "@@ -1,3 +1,4 @@\n+import rate_limit",
        "description": "Add rate limiting",
        "confidence": 0.85,
        "tradeoffs": ["Adds latency", "Requires Redis"],
        "schema_version": SCHEMA_VERSION,
    }
    # Should validate successfully (backward compatible)
    validate(instance=suggestion, schema=CODE_SUGGESTION_SCHEMA)


def test_code_suggestion_with_executable_fields():
    """Test CODE_SUGGESTION_SCHEMA with new executable artifact fields."""
    suggestion = {
        "file_path": "src/api.py",
        "unified_diff": "@@ -1,3 +1,4 @@\n+import rate_limit",
        "description": "Add rate limiting",
        "confidence": 0.85,
        "tradeoffs": ["Adds latency"],
        "schema_version": SCHEMA_VERSION,
        # New fields
        "patch_content": "@@ -1,3 +1,4 @@\n+import rate_limit\n",
        "apply_command": "git apply /tmp/patch.diff",
        "rollback_command": "git apply -R /tmp/patch.diff",
        "test_command": "pytest tests/test_api.py",
        "dependencies": ["suggestion_001"],
        "estimated_risk": "medium",
        "blast_radius": 15.5,
    }
    validate(instance=suggestion, schema=CODE_SUGGESTION_SCHEMA)


def test_estimated_risk_enum_validation():
    """Test that estimated_risk only accepts valid enum values."""
    valid_suggestion = {
        "file_path": "test.py",
        "unified_diff": "",
        "description": "Test",
        "confidence": 0.5,
        "tradeoffs": [],
        "schema_version": SCHEMA_VERSION,
        "estimated_risk": "low",
    }
    validate(instance=valid_suggestion, schema=CODE_SUGGESTION_SCHEMA)

    # Invalid enum should fail
    invalid_suggestion = valid_suggestion.copy()
    invalid_suggestion["estimated_risk"] = "critical"  # Not in enum
    with pytest.raises(ValidationError):
        validate(instance=invalid_suggestion, schema=CODE_SUGGESTION_SCHEMA)


def test_blast_radius_range_validation():
    """Test that blast_radius is within 0.0-100.0."""
    valid_suggestion = {
        "file_path": "test.py",
        "unified_diff": "",
        "description": "Test",
        "confidence": 0.5,
        "tradeoffs": [],
        "schema_version": SCHEMA_VERSION,
        "blast_radius": 50.5,
    }
    validate(instance=valid_suggestion, schema=CODE_SUGGESTION_SCHEMA)

    # Out of range should fail
    invalid_suggestion = valid_suggestion.copy()
    invalid_suggestion["blast_radius"] = 150.0
    with pytest.raises(ValidationError):
        validate(instance=invalid_suggestion, schema=CODE_SUGGESTION_SCHEMA)


def test_command_injection_documentation():
    """Verify command fields are present and documented.

    SECURITY NOTE: The apply_command, rollback_command, and test_command fields
    contain shell commands that MUST NEVER be executed directly without validation.

    These fields are ADVISORY ONLY and intended for:
    1. Human review before execution
    2. Display in UIs to show what WOULD be run
    3. Logging and audit trails

    DO NOT use shell=True when executing these commands.
    DO NOT concatenate user input into these commands.
    DO validate against allowed command patterns before execution.

    Example safe execution:
        # SAFE: Parse command, validate against allowlist
        if suggestion.apply_command.startswith("git apply "):
            subprocess.run(["git", "apply", patch_file], shell=False)

        # UNSAFE: Direct execution
        subprocess.run(suggestion.apply_command, shell=True)  # NEVER DO THIS
    """
    suggestion = {
        "file_path": "test.py",
        "unified_diff": "",
        "description": "Test",
        "confidence": 0.5,
        "tradeoffs": [],
        "schema_version": SCHEMA_VERSION,
        "apply_command": "git apply patch.diff",
        "rollback_command": "git apply -R patch.diff",
        "test_command": "pytest tests/",
    }
    validate(instance=suggestion, schema=CODE_SUGGESTION_SCHEMA)

    # Verify maxLength constraint prevents excessively long commands
    long_command_suggestion = suggestion.copy()
    long_command_suggestion["apply_command"] = "x" * 501  # Exceeds maxLength: 500
    with pytest.raises(ValidationError):
        validate(instance=long_command_suggestion, schema=CODE_SUGGESTION_SCHEMA)


def test_plan_step_minimal():
    """Test PLAN_STEP_SCHEMA with only required fields."""
    plan_step = {
        "id": "ps_1",
        "description": "Analyze codebase structure",
        "rationale": "Need to understand existing patterns",
        "dependencies": [],
        "schema_version": SCHEMA_VERSION,
    }
    validate(instance=plan_step, schema=PLAN_STEP_SCHEMA)


def test_plan_step_with_incremental_fields():
    """Test PLAN_STEP_SCHEMA with all incremental planning fields."""
    plan_step = {
        "id": "ps_2",
        "description": "Refactor authentication module",
        "rationale": "Current impl has security vulnerabilities",
        "dependencies": [1],
        "schema_version": SCHEMA_VERSION,
        # Incremental planning fields
        "preconditions": ["Tests pass", "No breaking changes in main"],
        "actions": ["Extract validate_token()", "Add rate limiting", "Update tests"],
        "exit_criteria": ["All tests pass", "Security scan clean", "Performance <100ms"],
        "risk": "high",
        "retrieval_keys": ["authentication", "security", "rate_limiting"],
        "failure_signatures": ["Token validation fails on edge case X"],
        "verifier_checks": ["pytest tests/test_auth.py", "bandit -r src/auth"],
        "expanded": True,
    }
    validate(instance=plan_step, schema=PLAN_STEP_SCHEMA)


def test_plan_step_risk_enum():
    """Test that PlanStep risk field validates enum values."""
    for risk_level in ["low", "medium", "high"]:
        plan_step = {
            "id": "ps_3",
            "description": "Test step",
            "rationale": "Testing",
            "dependencies": [],
            "schema_version": SCHEMA_VERSION,
            "risk": risk_level,
        }
        validate(instance=plan_step, schema=PLAN_STEP_SCHEMA)

    # Invalid risk should fail
    invalid_step = {
        "id": "ps_4",
        "description": "Test",
        "rationale": "Test",
        "dependencies": [],
        "schema_version": SCHEMA_VERSION,
        "risk": "critical",  # Not in enum
    }
    with pytest.raises(ValidationError):
        validate(instance=invalid_step, schema=PLAN_STEP_SCHEMA)
