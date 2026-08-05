"""Tests for the malformed-response repair loop.

Regression coverage for an import bug that survived from the initial public
release: `repair_loop` imported `structured_parser` and `schemas` unqualified,
which can never resolve inside the installed package. Every repair attempt died
on ModuleNotFoundError and was swallowed by the attempt loop's `except`, so the
feature was silently dead and no test noticed.
"""

import json

from neo.repair_loop import repair_response
from neo.schemas import SCHEMA_VERSION
from neo.structured_parser import ParseErrorCode, ParseResult


def _valid_plan_block():
    payload = [{
        "id": "ps_1",
        "description": "Add a guard clause",
        "rationale": "avoid the crash",
        "dependencies": [],
        "schema_version": SCHEMA_VERSION,
    }]
    return (
        f"<<<NEO:SCHEMA=v3:KIND=plan>>>\n{json.dumps(payload)}\n<<<END:plan>>>"
    )


class _FormatterLM:
    """Returns a well-formed block, as a real formatter model would."""

    def __init__(self, response):
        self._response = response
        self.calls = 0

    def generate(self, messages, **kwargs):
        self.calls += 1
        return self._response

    def name(self):
        return "fake-formatter"


def _failed_parse():
    return ParseResult(
        success=False,
        error_code=ParseErrorCode.MISSING_START_SENTINEL,
        error_message="Missing start sentinel",
    )


def test_repair_recovers_a_malformed_plan_response():
    """The end-to-end path. Before the import fix this returned success=False
    with 'Failed to repair after 2 attempts' no matter what the formatter said."""
    lm = _FormatterLM(_valid_plan_block())

    result = repair_response(
        bad_response="here is your plan, roughly",
        parse_result=_failed_parse(),
        kind="plan",
        original_prompt="fix the parser",
        lm_adapter=lm,
    )

    assert result.success, result.error_message
    assert lm.calls == 1
    assert result.repaired_response


def test_repair_does_not_die_on_an_import_error():
    """Pins the specific failure mode: the attempt loop catches Exception, so a
    bad import surfaced only as a generic repair failure."""
    lm = _FormatterLM(_valid_plan_block())

    result = repair_response(
        bad_response="malformed",
        parse_result=_failed_parse(),
        kind="plan",
        original_prompt="fix the parser",
        lm_adapter=lm,
    )

    assert "No module named" not in (result.error_message or "")


def test_unrecoverable_errors_skip_the_formatter_call():
    lm = _FormatterLM(_valid_plan_block())

    result = repair_response(
        bad_response="truncated",
        parse_result=ParseResult(
            success=False,
            error_code=ParseErrorCode.TRUNCATION,
            error_message="output truncated",
        ),
        kind="plan",
        original_prompt="fix the parser",
        lm_adapter=lm,
    )

    assert not result.success
    assert lm.calls == 0


def test_repair_gives_up_after_max_attempts():
    lm = _FormatterLM("still not a structured block")

    result = repair_response(
        bad_response="malformed",
        parse_result=_failed_parse(),
        kind="plan",
        original_prompt="fix the parser",
        lm_adapter=lm,
        max_attempts=2,
    )

    assert not result.success
    assert lm.calls == 2
