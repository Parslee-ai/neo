"""JSON schema + converters for Neo's `neo.process` CAR tool.

Used by `neo serve` to declare the tool to car-runtime via
`register_tool_schema`. The schema feeds car-a2a's auto-generated
Agent Card, so any A2A peer can discover Neo's input/output shape
from `/.well-known/agent-card.json` without Neo-specific glue.

Wire shape:
  call: A2A peer sends `{tool:"neo.process", parameters:{...NeoInput}}`.
  return: handler returns JSON-encoded `NeoOutput`-shaped dict.
"""

from __future__ import annotations

import json
from typing import Any

from neo.models import (
    CodeSuggestion,
    ContextFile,
    NeoInput,
    NeoOutput,
    PlanStep,
    SimulationTrace,
    StaticCheckResult,
    TaskType,
)


TOOL_NAME = "neo.process"
TOOL_DESCRIPTION = (
    "Read-only code-reasoning helper. Returns a structured plan, "
    "simulation traces, and code suggestions (unified diffs) for the "
    "prompt + context. Does not modify files or run commands."
)


def tool_schema() -> dict[str, Any]:
    """Build the `ToolSchema` dict for `register_tool_schema`.

    Shape matches car-runtime's ToolSchema: name, description,
    parameters (JSON Schema object), returns (advisory), idempotent.
    No cache_ttl_secs — every call re-runs the pipeline.
    """
    return {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The user's request or question.",
                },
                "task_type": {
                    "type": "string",
                    "enum": [t.value for t in TaskType],
                    "description": (
                        "Optional task classification. Omit to let Neo infer."
                    ),
                },
                "context_files": {
                    "type": "array",
                    "description": "Files relevant to the request.",
                    "items": {
                        "type": "object",
                        "required": ["path", "content"],
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                            "line_range": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                    },
                },
                "error_trace": {
                    "type": "string",
                    "description": "Stack trace or error output, when debugging.",
                },
                "recent_commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Recent shell commands for environmental context.",
                },
                "safe_read_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Additional file paths Neo may read. Resolved relative to "
                        "working_directory; paths escaping it are dropped."
                    ),
                },
                "working_directory": {
                    "type": "string",
                    "description": "Caller's cwd. Anchors safe_read_paths and scope detection.",
                },
            },
        },
        "returns": {
            "type": "object",
            "description": "Structured NeoOutput. See neo_output_to_dict for the shape.",
        },
        "idempotent": False,
    }


def tool_schema_json() -> str:
    return json.dumps(tool_schema())


def dict_to_neo_input(params: dict[str, Any]) -> NeoInput:
    """Build a NeoInput from the A2A parameters dict.

    Tolerates missing optional fields. `task_type` accepts the enum
    value as a string; unknown values are dropped to None rather than
    raising, so a peer using a Neo build with extra TaskType variants
    against an older Neo still gets a useful response.
    """
    task_type: TaskType | None = None
    raw_task = params.get("task_type")
    if isinstance(raw_task, str):
        try:
            task_type = TaskType(raw_task)
        except ValueError:
            task_type = None

    context_files: list[ContextFile] = []
    for cf in params.get("context_files") or []:
        if not isinstance(cf, dict):
            continue
        path = cf.get("path")
        content = cf.get("content")
        if not isinstance(path, str) or not isinstance(content, str):
            continue
        line_range = cf.get("line_range")
        lr: tuple[int, int] | None = None
        if (
            isinstance(line_range, (list, tuple))
            and len(line_range) == 2
            and all(isinstance(x, int) for x in line_range)
        ):
            lr = (line_range[0], line_range[1])
        context_files.append(ContextFile(path=path, content=content, line_range=lr))

    def _str_list(key: str) -> list[str]:
        raw = params.get(key) or []
        return [x for x in raw if isinstance(x, str)]

    return NeoInput(
        prompt=str(params.get("prompt", "")),
        task_type=task_type,
        context_files=context_files,
        error_trace=params.get("error_trace") if isinstance(params.get("error_trace"), str) else None,
        recent_commands=_str_list("recent_commands"),
        safe_read_paths=_str_list("safe_read_paths"),
        working_directory=params.get("working_directory") if isinstance(params.get("working_directory"), str) else None,
    )


def neo_output_to_dict(output: NeoOutput) -> dict[str, Any]:
    """Serialize a NeoOutput to the A2A artifact payload shape.

    Mirrors cli.py's existing serialization (line ~678) so wire shape
    matches the stdin/stdout JSON consumers already see. Keeps the
    cross-transport contract stable.
    """
    return {
        "plan": [_plan_step_to_dict(s) for s in output.plan],
        "simulation_traces": [_sim_to_dict(t) for t in output.simulation_traces],
        "code_suggestions": [_suggestion_to_dict(s) for s in output.code_suggestions],
        "static_checks": [_static_check_to_dict(c) for c in output.static_checks],
        "next_questions": list(output.next_questions),
        "confidence": output.confidence,
        "notes": output.notes,
        "metadata": dict(output.metadata),
    }


def _plan_step_to_dict(step: PlanStep) -> dict[str, Any]:
    return {
        "description": step.description,
        "rationale": step.rationale,
        "dependencies": list(step.dependencies),
    }


def _sim_to_dict(trace: SimulationTrace) -> dict[str, Any]:
    return {
        "input_data": trace.input_data,
        "expected_output": trace.expected_output,
        "reasoning_steps": list(trace.reasoning_steps),
        "issues_found": list(trace.issues_found),
    }


def _suggestion_to_dict(s: CodeSuggestion) -> dict[str, Any]:
    return {
        "file_path": s.file_path,
        "unified_diff": s.unified_diff,
        "description": s.description,
        "confidence": s.confidence,
        "tradeoffs": list(s.tradeoffs),
    }


def _static_check_to_dict(c: StaticCheckResult) -> dict[str, Any]:
    return {
        "tool_name": c.tool_name,
        "diagnostics": list(c.diagnostics),
        "summary": c.summary,
    }
