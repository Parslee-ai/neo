"""
Structured block parser for Neo LM responses.

Replaces regex-based prose parsing with sentinel-bounded JSON blocks.
"""

import json
import re
import logging
from typing import Any, Optional
from dataclasses import dataclass

try:
    from jsonschema import validate, ValidationError
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    logging.warning("jsonschema not available, schema validation disabled")

from neo.schemas import (
    SCHEMA_VERSION,
    SENTINEL_START,
    SENTINEL_END,
    PLAN_STEP_SCHEMA,
    SIMULATION_TRACE_SCHEMA,
    CODE_SUGGESTION_SCHEMA,
    MULTI_PHASE_SCHEMA
)

logger = logging.getLogger(__name__)


@dataclass
class ParseResult:
    """Result of parsing with diagnostics."""
    success: bool
    data: Optional[Any] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    raw_block: Optional[str] = None


class ParseErrorCode:
    """Normalized error codes for parse failures."""
    MISSING_START_SENTINEL = "missing_start_sentinel"
    MISSING_END_SENTINEL = "missing_end_sentinel"
    STRAY_TEXT = "stray_text"
    BAD_JSON = "bad_json"
    WRONG_SCHEMA = "wrong_schema_version"
    MISSING_FIELD = "missing_required_field"
    DUP_KEYS = "duplicate_keys"
    TRUNCATION = "truncation"
    HALLUCINATED_FIELDS = "hallucinated_fields"
    MARKDOWN_WRAPPED = "markdown_wrapped"
    MIXED_SENTINELS = "mixed_sentinels"
    ESCAPE_ERRORS = "escape_errors"
    OVERSIZED_FIELD = "oversized_field"
    OFF_POLICY_EXPLANATION = "off_policy_explanation"


def extract_block(response: str, kind: str) -> ParseResult:
    """
    Extract structured JSON block from LM response.

    Args:
        response: Raw LM response
        kind: Expected block kind (plan, simulation, code, multi_phase)

    Returns:
        ParseResult with extracted block or error
    """
    # Build expected sentinels
    start_sentinel = SENTINEL_START.format(kind=kind)
    end_sentinel = SENTINEL_END.format(kind=kind)  # Format with kind for labeled end tags

    # Check for markdown wrapping FIRST (even if sentinels present)
    if "```json" in response or (response.startswith("```") and "```" in response[3:]):
        return ParseResult(
            success=False,
            error_code=ParseErrorCode.MARKDOWN_WRAPPED,
            error_message="Response wrapped in markdown code fence"
        )

    # Check for start sentinel
    if start_sentinel not in response:
        # Check for wrong kind sentinel
        if "<<<NEO:SCHEMA=" in response:
            return ParseResult(
                success=False,
                error_code=ParseErrorCode.MIXED_SENTINELS,
                error_message=f"Found sentinel but wrong kind, expected {kind}"
            )

        return ParseResult(
            success=False,
            error_code=ParseErrorCode.MISSING_START_SENTINEL,
            error_message=f"Missing start sentinel {start_sentinel}"
        )

    # Check for end sentinel
    if end_sentinel not in response:
        return ParseResult(
            success=False,
            error_code=ParseErrorCode.MISSING_END_SENTINEL,
            error_message=f"Missing end sentinel {end_sentinel}"
        )

    # Extract block between sentinels
    start_idx = response.index(start_sentinel) + len(start_sentinel)
    end_idx = response.index(end_sentinel)

    if start_idx >= end_idx:
        return ParseResult(
            success=False,
            error_code=ParseErrorCode.STRAY_TEXT,
            error_message="Sentinels out of order or empty block"
        )

    raw_block = response[start_idx:end_idx].strip()

    # Check for stray text before start or after end
    before_text = response[:response.index(start_sentinel)].strip()
    after_text = response[response.index(end_sentinel) + len(end_sentinel):].strip()

    if before_text and len(before_text) > 10:  # Allow small whitespace
        logger.warning(f"Stray text before sentinel: {before_text[:50]}...")
        return ParseResult(
            success=False,
            error_code=ParseErrorCode.OFF_POLICY_EXPLANATION,
            error_message=f"Text found before sentinel (length: {len(before_text)})",
            raw_block=raw_block
        )

    if after_text and len(after_text) > 10:
        logger.warning(f"Stray text after sentinel: {after_text[:50]}...")
        # Don't fail for this, but log it

    return ParseResult(
        success=True,
        data=raw_block,
        raw_block=raw_block
    )


def tolerant_json_load(json_str: str) -> ParseResult:
    """
    Load JSON with tolerance for common LM quirks.

    Handles:
    - Trailing commas
    - Smart quotes
    - Unescaped newlines in strings
    - Extra whitespace

    Args:
        json_str: JSON string to parse

    Returns:
        ParseResult with parsed data or error
    """
    # Try standard JSON first
    try:
        data = json.loads(json_str)
        return ParseResult(success=True, data=data)
    except json.JSONDecodeError as e:
        pass  # Try repairs below

    # Common repairs
    repaired = json_str

    # Fix smart quotes
    repaired = repaired.replace('"', '"').replace('"', '"')
    repaired = repaired.replace("'", "'").replace("'", "'")

    # Remove trailing commas before } or ]
    repaired = re.sub(r',\s*([}\]])', r'\1', repaired)

    # Try again
    try:
        data = json.loads(repaired)
        logger.info("JSON repaired successfully")
        return ParseResult(success=True, data=data)
    except json.JSONDecodeError as e:
        # Check for specific errors
        error_msg = str(e)

        if "Expecting value" in error_msg or "Unterminated string" in error_msg:
            error_code = ParseErrorCode.TRUNCATION
        elif "Duplicate" in error_msg:
            error_code = ParseErrorCode.DUP_KEYS
        elif "Expecting" in error_msg:
            error_code = ParseErrorCode.BAD_JSON
        else:
            error_code = ParseErrorCode.ESCAPE_ERRORS

        return ParseResult(
            success=False,
            error_code=error_code,
            error_message=f"JSON parse error: {error_msg}"
        )


def _truncate_oversized_fields(data: Any, schema: dict) -> Any:
    """
    Recursively truncate string fields that exceed maxLength.

    Walks dict/list/str structure and truncates strings to maxLength - 3,
    appending '...' to indicate truncation.

    Args:
        data: Data structure to truncate
        schema: JSON schema with maxLength constraints

    Returns:
        Modified data with truncated strings
    """
    if isinstance(data, dict):
        # Truncate dict values according to schema properties
        if "properties" in schema:
            result = {}
            for key, value in data.items():
                if key in schema["properties"]:
                    field_schema = schema["properties"][key]
                    result[key] = _truncate_oversized_fields(value, field_schema)
                else:
                    result[key] = value
            return result
        return data

    elif isinstance(data, list):
        # Truncate list items according to schema items
        if "items" in schema:
            item_schema = schema["items"]
            return [_truncate_oversized_fields(item, item_schema) for item in data]
        return data

    elif isinstance(data, str):
        # Truncate if exceeds maxLength
        if "maxLength" in schema:
            max_len = schema["maxLength"]
            if len(data) > max_len:
                # Truncate to maxLength - 3 and append '...' (total length = maxLength)
                # Example: maxLength=500 -> truncate to 497 chars + '...' = 500 chars
                return data[:max_len - 3] + "..."
        return data

    # Return unchanged for other types
    return data


def validate_schema(data: dict, schema: dict) -> ParseResult:
    """
    Validate data against JSON schema.

    Args:
        data: Parsed JSON data
        schema: JSON schema to validate against

    Returns:
        ParseResult with validation result
    """
    if not JSONSCHEMA_AVAILABLE:
        logger.warning("jsonschema not available, skipping validation")
        return ParseResult(success=True, data=data)

    # Truncate oversized fields before validation
    data = _truncate_oversized_fields(data, schema)

    try:
        validate(instance=data, schema=schema)

        return ParseResult(success=True, data=data)
    except ValidationError as e:
        # Parse error details
        error_path = ".".join(str(p) for p in e.path) if e.path else "root"

        # Determine error code
        if "is a required property" in e.message:
            error_code = ParseErrorCode.MISSING_FIELD
        elif "was unexpected" in e.message or "additional properties" in e.message.lower():
            error_code = ParseErrorCode.HALLUCINATED_FIELDS
        elif "does not match" in e.message:
            error_code = ParseErrorCode.WRONG_SCHEMA
        elif "is too long" in e.message or "is greater than" in e.message:
            error_code = ParseErrorCode.OVERSIZED_FIELD
        else:
            error_code = ParseErrorCode.WRONG_SCHEMA

        return ParseResult(
            success=False,
            error_code=error_code,
            error_message=f"Schema validation failed at {error_path}: {e.message}"
        )


def parse_structured_response(response: str, kind: str, schema: dict) -> ParseResult:
    """
    Parse structured response with full validation pipeline.

    Args:
        response: Raw LM response
        kind: Expected block kind
        schema: JSON schema to validate against

    Returns:
        ParseResult with parsed and validated data or error
    """
    # Step 1: Extract block between sentinels
    extract_result = extract_block(response, kind)
    if not extract_result.success:
        return extract_result

    # Step 2: Parse JSON with tolerance
    json_result = tolerant_json_load(extract_result.data)
    if not json_result.success:
        json_result.raw_block = extract_result.raw_block
        return json_result

    # Step 3: Validate schema
    validate_result = validate_schema(json_result.data, schema)
    if not validate_result.success:
        validate_result.raw_block = extract_result.raw_block
        return validate_result

    # Step 4: Check schema version
    data = validate_result.data
    if isinstance(data, dict):
        if "schema_version" in data and data["schema_version"] != SCHEMA_VERSION:
            return ParseResult(
                success=False,
                error_code=ParseErrorCode.WRONG_SCHEMA,
                error_message=f"Schema version mismatch: expected {SCHEMA_VERSION}, got {data['schema_version']}",
                raw_block=extract_result.raw_block
            )

    return ParseResult(success=True, data=data, raw_block=extract_result.raw_block)


# Convenience functions for specific types
def parse_plan_steps(response: str) -> ParseResult:
    """Parse plan steps from LM response."""
    result = parse_structured_response(response, "plan", {"type": "array", "items": PLAN_STEP_SCHEMA})
    return result


def parse_simulation_traces(response: str) -> ParseResult:
    """Parse simulation traces from LM response."""
    result = parse_structured_response(response, "simulation", {"type": "array", "items": SIMULATION_TRACE_SCHEMA})
    return result


def parse_code_suggestions(response: str) -> ParseResult:
    """Parse code suggestions from LM response."""
    result = parse_structured_response(response, "code", {"type": "array", "items": CODE_SUGGESTION_SCHEMA})
    return result


def parse_multi_phase(response: str) -> ParseResult:
    """Parse combined multi-phase output."""
    result = parse_structured_response(response, "multi_phase", MULTI_PHASE_SCHEMA)
    return result