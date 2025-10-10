"""
JSON Schemas for Neo structured output.

Schema version 3 - structured block format with sentinels.
"""

# Schema version constant
SCHEMA_VERSION = "3"

# Sentinel patterns
SENTINEL_START = "<<<NEO:SCHEMA=v3:KIND={kind}>>>"
SENTINEL_END = "<<<END:{kind}>>>"  # Labeled to prevent collisions in multi-block responses


# JSON Schema definitions
PLAN_STEP_SCHEMA = {
    "type": "object",
    "required": ["id", "description", "rationale", "dependencies", "schema_version"],
    "properties": {
        "id": {
            "type": "string",
            "pattern": "^ps_[0-9]+$",
            "description": "Plan step ID (e.g., ps_1)"
        },
        "description": {
            "type": "string",
            "minLength": 1,
            "maxLength": 500,
            "description": "Brief description of this step"
        },
        "rationale": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1000,
            "description": "Why this step is necessary"
        },
        "dependencies": {
            "type": "array",
            "items": {"type": "integer", "minimum": 0},
            "description": "List of step indices this depends on"
        },
        "schema_version": {
            "type": "string",
            "const": SCHEMA_VERSION,
            "description": "Schema version for validation"
        }
    },
    "additionalProperties": False
}


SIMULATION_TRACE_SCHEMA = {
    "type": "object",
    "required": ["n", "input_data", "expected_output", "reasoning_steps", "issues_found", "schema_version"],
    "properties": {
        "n": {
            "type": "integer",
            "minimum": 1,
            "description": "Simulation number"
        },
        "input_data": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1000,
            "description": "Input scenario for simulation"
        },
        "expected_output": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1000,
            "description": "Expected result"
        },
        "reasoning_steps": {
            "type": "array",
            "items": {"type": "string", "maxLength": 500},
            "minItems": 1,
            "maxItems": 20,
            "description": "Step-by-step reasoning"
        },
        "issues_found": {
            "type": "array",
            "items": {"type": "string", "maxLength": 500},
            "maxItems": 10,
            "description": "Issues discovered during simulation"
        },
        "schema_version": {
            "type": "string",
            "const": SCHEMA_VERSION,
            "description": "Schema version for validation"
        }
    },
    "additionalProperties": False
}


CODE_SUGGESTION_SCHEMA = {
    "type": "object",
    "required": ["file_path", "unified_diff", "description", "confidence", "tradeoffs", "schema_version"],
    "properties": {
        "file_path": {
            "type": "string",
            "maxLength": 500,
            "description": "Relative path to file (can be '/' or 'N/A' for review-style findings)"
        },
        "unified_diff": {
            "type": "string",
            "description": "Unified diff patch (can be empty for review-style suggestions)"
        },
        "code_block": {
            "type": "string",
            "description": "Executable Python code (optional, preferred over diff extraction)"
        },
        "description": {
            "type": "string",
            "minLength": 1,
            "maxLength": 3000,
            "description": "Description of changes"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence score 0.0-1.0"
        },
        "tradeoffs": {
            "type": "array",
            "items": {"type": "string", "maxLength": 500},
            "maxItems": 10,
            "description": "Tradeoffs to consider"
        },
        "schema_version": {
            "type": "string",
            "const": SCHEMA_VERSION,
            "description": "Schema version for validation"
        }
    },
    "additionalProperties": False
}


# Multi-phase schema for combined output
MULTI_PHASE_SCHEMA = {
    "type": "object",
    "required": ["plan_steps", "simulation_traces", "code_suggestions", "meta"],
    "properties": {
        "plan_steps": {
            "type": "array",
            "items": PLAN_STEP_SCHEMA,
            "minItems": 1,
            "maxItems": 20
        },
        "simulation_traces": {
            "type": "array",
            "items": SIMULATION_TRACE_SCHEMA,
            "maxItems": 10
        },
        "code_suggestions": {
            "type": "array",
            "items": CODE_SUGGESTION_SCHEMA,
            "maxItems": 10
        },
        "meta": {
            "type": "object",
            "required": ["request_id", "schema_version", "phase"],
            "properties": {
                "request_id": {"type": "string"},
                "schema_version": {"const": SCHEMA_VERSION},
                "phase": {
                    "enum": ["planning", "simulation", "codegen", "review"]
                }
            }
        }
    },
    "additionalProperties": False
}