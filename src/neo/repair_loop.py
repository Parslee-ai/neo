"""
Repair loop for malformed LM responses.

When parsing fails, attempts to repair the response using a formatter model.
"""

import logging
from typing import Optional
from dataclasses import dataclass

from neo.structured_parser import ParseResult, ParseErrorCode

logger = logging.getLogger(__name__)


@dataclass
class RepairResult:
    """Result of repair attempt."""
    success: bool
    repaired_response: Optional[str] = None
    parse_result: Optional[ParseResult] = None
    error_message: Optional[str] = None


def create_repair_prompt(
    bad_response: str,
    error_code: str,
    error_message: str,
    kind: str,
    original_prompt: str
) -> str:
    """
    Create a prompt for the formatter model to repair malformed output.

    Args:
        bad_response: The malformed LM response
        error_code: Normalized error code
        error_message: Detailed error message
        kind: Expected block kind (plan, simulation, code)
        original_prompt: The original user prompt that led to this response

    Returns:
        Prompt for formatter model
    """
    return f"""You are a JSON formatter. Your ONLY job is to fix malformed JSON responses.

**Original Task:** {original_prompt[:200]}...

**Expected Format:**
<<<NEO:SCHEMA=v3:KIND={kind}>>>
[valid JSON array matching the schema]
<<<END>>>

**Error Detected:** [{error_code}] {error_message}

**Malformed Response:**
{bad_response[:1000]}

**Your Task:**
1. Extract the semantic content from the malformed response
2. Re-emit it in the EXACT format above with proper sentinels
3. Ensure valid JSON (no trailing commas, proper quotes, escaped characters)
4. Include ONLY the JSON array between sentinels - NO explanations or text
5. Preserve all semantic meaning from the original

**Rules:**
- Start with <<<NEO:SCHEMA=v3:KIND={kind}>>>
- End with <<<END>>>
- Valid JSON only between sentinels
- No text before or after sentinels
- All fields must match schema requirements
- Include schema_version: "3" in each object

Re-emit the corrected response now:"""


def repair_response(
    bad_response: str,
    parse_result: ParseResult,
    kind: str,
    original_prompt: str,
    lm_adapter,
    max_attempts: int = 2
) -> RepairResult:
    """
    Attempt to repair a malformed LM response.

    Args:
        bad_response: The malformed response
        parse_result: The failed ParseResult
        kind: Expected block kind
        original_prompt: Original user prompt
        lm_adapter: LM adapter for calling formatter
        max_attempts: Maximum repair attempts (default 2)

    Returns:
        RepairResult with success status and repaired response
    """
    logger.info(f"Attempting to repair response for kind={kind}, error={parse_result.error_code}")

    # Don't attempt repair for certain error types that are unrecoverable
    unrecoverable_errors = {
        ParseErrorCode.TRUNCATION,  # Can't recover truncated output
        ParseErrorCode.OVERSIZED_FIELD,  # Would require removing content
    }

    if parse_result.error_code in unrecoverable_errors:
        logger.warning(f"Error code {parse_result.error_code} is unrecoverable, skipping repair")
        return RepairResult(
            success=False,
            error_message=f"Unrecoverable error: {parse_result.error_code}"
        )

    # Create repair prompt
    repair_prompt = create_repair_prompt(
        bad_response=bad_response,
        error_code=parse_result.error_code,
        error_message=parse_result.error_message,
        kind=kind,
        original_prompt=original_prompt
    )

    # Attempt repair with low temperature for determinism
    for attempt in range(max_attempts):
        try:
            logger.debug(f"Repair attempt {attempt + 1}/{max_attempts}")

            # Call formatter with strict parameters
            messages = [
                {
                    "role": "system",
                    "content": "You are a deterministic JSON formatter. Emit ONLY the corrected structured block. No explanations."
                },
                {
                    "role": "user",
                    "content": repair_prompt
                }
            ]

            repaired = lm_adapter.generate(
                messages,
                stop=["</neo>", "```"],
                max_tokens=4096,
                temperature=0.0  # Deterministic
            )

            # Parse the repaired response
            from structured_parser import parse_structured_response
            from schemas import (
                PLAN_STEP_SCHEMA,
                SIMULATION_TRACE_SCHEMA,
                CODE_SUGGESTION_SCHEMA
            )

            # Choose schema based on kind
            if kind == "plan":
                schema = {"type": "array", "items": PLAN_STEP_SCHEMA}
            elif kind == "simulation":
                schema = {"type": "array", "items": SIMULATION_TRACE_SCHEMA}
            elif kind == "code":
                schema = {"type": "array", "items": CODE_SUGGESTION_SCHEMA}
            else:
                raise ValueError(f"Unknown kind: {kind}")

            repaired_result = parse_structured_response(repaired, kind, schema)

            if repaired_result.success:
                logger.info(f"Repair successful on attempt {attempt + 1}")
                return RepairResult(
                    success=True,
                    repaired_response=repaired,
                    parse_result=repaired_result
                )
            else:
                logger.warning(
                    f"Repair attempt {attempt + 1} failed: {repaired_result.error_code}"
                )
                # Continue to next attempt

        except Exception as e:
            logger.error(f"Repair attempt {attempt + 1} raised exception: {e}")
            # Continue to next attempt

    # All attempts failed
    logger.error(f"All {max_attempts} repair attempts failed")
    return RepairResult(
        success=False,
        error_message=f"Failed to repair after {max_attempts} attempts"
    )


def parse_with_repair(
    response: str,
    kind: str,
    parser_func,
    original_prompt: str,
    lm_adapter,
    enable_repair: bool = True
) -> ParseResult:
    """
    Parse with automatic repair on failure.

    Args:
        response: LM response to parse
        kind: Expected block kind
        parser_func: Parser function to use
        original_prompt: Original user prompt
        lm_adapter: LM adapter for repair
        enable_repair: Whether to attempt repair on failure

    Returns:
        ParseResult (either from initial parse or successful repair)
    """
    # Try initial parse
    result = parser_func(response)

    if result.success:
        return result

    # If repair is disabled or no adapter, fail immediately
    if not enable_repair or not lm_adapter:
        return result

    # Attempt repair
    logger.info("Initial parse failed, attempting repair")
    repair_result = repair_response(
        bad_response=response,
        parse_result=result,
        kind=kind,
        original_prompt=original_prompt,
        lm_adapter=lm_adapter,
        max_attempts=2
    )

    if repair_result.success:
        logger.info("Repair successful, returning repaired ParseResult")
        return repair_result.parse_result
    else:
        logger.error(f"Repair failed: {repair_result.error_message}")
        # Return original parse failure
        return result