"""Unit tests for neo.car_tool_schema.

Covers schema generation, NeoInput coercion (including malformed
inputs that come over A2A from peers we don't trust), and NeoOutput
serialization round-trip.

No CAR daemon required — these are pure-Python tests.
"""

from __future__ import annotations

import json

from neo.car_tool_schema import (
    TOOL_DESCRIPTION,
    TOOL_NAME,
    dict_to_neo_input,
    neo_output_to_dict,
    tool_schema,
    tool_schema_json,
)
from neo.models import (
    CodeSuggestion,
    NeoOutput,
    PlanStep,
    SimulationTrace,
    StaticCheckResult,
    TaskType,
)


class TestToolSchema:
    def test_schema_advertises_correct_tool_name(self):
        s = tool_schema()
        assert s["name"] == TOOL_NAME == "neo.process"

    def test_schema_includes_description(self):
        assert tool_schema()["description"] == TOOL_DESCRIPTION

    def test_prompt_is_required(self):
        params = tool_schema()["parameters"]
        assert params["type"] == "object"
        assert "prompt" in params["required"]
        assert params["properties"]["prompt"]["type"] == "string"

    def test_task_type_enum_covers_all_values(self):
        enum = tool_schema()["parameters"]["properties"]["task_type"]["enum"]
        assert set(enum) == {t.value for t in TaskType}

    def test_context_files_shape(self):
        cf = tool_schema()["parameters"]["properties"]["context_files"]
        assert cf["type"] == "array"
        item = cf["items"]
        assert set(item["required"]) == {"path", "content"}
        assert item["properties"]["line_range"]["minItems"] == 2

    def test_schema_is_not_idempotent(self):
        # Every call should re-run the pipeline (memory updates, etc.).
        # If we ever mark it idempotent, the daemon would cache results.
        assert tool_schema()["idempotent"] is False

    def test_schema_json_round_trips(self):
        assert json.loads(tool_schema_json()) == tool_schema()


class TestDictToNeoInput:
    def test_minimal_prompt_only(self):
        ni = dict_to_neo_input({"prompt": "hello"})
        assert ni.prompt == "hello"
        assert ni.task_type is None
        assert ni.context_files == []

    def test_full_payload(self):
        ni = dict_to_neo_input({
            "prompt": "fix the bug",
            "task_type": "bugfix",
            "context_files": [
                {"path": "/app/x.py", "content": "print('x')"},
                {"path": "/app/y.py", "content": "print('y')", "line_range": [10, 20]},
            ],
            "error_trace": "TypeError",
            "recent_commands": ["pytest -x"],
            "safe_read_paths": ["docs/"],
            "working_directory": "/home/me/project",
        })
        assert ni.task_type is TaskType.BUGFIX
        assert len(ni.context_files) == 2
        assert ni.context_files[1].line_range == (10, 20)
        assert ni.error_trace == "TypeError"
        assert ni.recent_commands == ["pytest -x"]
        assert ni.safe_read_paths == ["docs/"]
        assert ni.working_directory == "/home/me/project"

    def test_unknown_task_type_is_dropped_not_raised(self):
        # Forward-compat: peer using a newer enum value against an older
        # Neo build should still get a response, not a crash.
        ni = dict_to_neo_input({"prompt": "p", "task_type": "future-variant"})
        assert ni.task_type is None

    def test_malformed_context_files_are_skipped(self):
        # Defensive: A2A peers aren't trusted — drop malformed entries
        # rather than raising and rejecting the whole call.
        ni = dict_to_neo_input({
            "prompt": "p",
            "context_files": [
                {"path": "/ok.py", "content": "x"},
                "not a dict",
                {"path": "/no-content.py"},
                {"content": "no path"},
                {"path": 42, "content": "x"},
                None,
            ],
        })
        assert len(ni.context_files) == 1
        assert ni.context_files[0].path == "/ok.py"

    def test_non_string_strings_are_filtered(self):
        # recent_commands etc. could arrive with garbage entries.
        ni = dict_to_neo_input({
            "prompt": "p",
            "recent_commands": ["ls", 42, None, "pwd"],
            "safe_read_paths": [True, "ok"],
        })
        assert ni.recent_commands == ["ls", "pwd"]
        assert ni.safe_read_paths == ["ok"]

    def test_empty_prompt_returns_empty_string(self):
        # Validation that prompt is non-empty lives in the host handler,
        # not here. dict_to_neo_input is best-effort coercion.
        assert dict_to_neo_input({}).prompt == ""
        assert dict_to_neo_input({"prompt": None}).prompt == "None"


class TestNeoOutputToDict:
    def test_full_round_trip_is_json_serializable(self):
        output = NeoOutput(
            plan=[
                PlanStep(description="do x", rationale="why", dependencies=[]),
                PlanStep(description="then y", rationale="because", dependencies=[0]),
            ],
            simulation_traces=[
                SimulationTrace(
                    input_data="[1,2,3]",
                    expected_output="[1,2,3]",
                    reasoning_steps=["scan", "compare"],
                    issues_found=[],
                ),
            ],
            code_suggestions=[
                CodeSuggestion(
                    file_path="/app/x.py",
                    unified_diff="--- a\n+++ b\n@@ ...",
                    description="add nothing",
                    confidence=0.91,
                    tradeoffs=["readability"],
                ),
            ],
            static_checks=[
                StaticCheckResult(
                    tool_name="ruff",
                    diagnostics=[{"code": "F401", "message": "unused"}],
                    summary="1 finding",
                ),
            ],
            next_questions=["q1?"],
            confidence=0.85,
            notes="all good",
            metadata={"early_exit": True},
        )

        d = neo_output_to_dict(output)

        # Must be JSON-serializable as-is (this is the A2A artifact payload).
        encoded = json.dumps(d)
        decoded = json.loads(encoded)

        assert decoded["confidence"] == 0.85
        assert decoded["plan"][1]["dependencies"] == [0]
        assert decoded["code_suggestions"][0]["confidence"] == 0.91
        assert decoded["static_checks"][0]["diagnostics"][0]["code"] == "F401"
        assert decoded["metadata"]["early_exit"] is True

    def test_empty_output_serializes_cleanly(self):
        output = NeoOutput(
            plan=[],
            simulation_traces=[],
            code_suggestions=[],
            static_checks=[],
            next_questions=[],
            confidence=0.0,
            notes="",
        )
        d = neo_output_to_dict(output)
        json.dumps(d)  # must not raise
        assert d["plan"] == []
        assert d["metadata"] == {}
