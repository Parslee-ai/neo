"""
Strict parsers for LM responses.

NO FALLBACKS - If parsing fails, we want to see the real error so we can fix it.
"""

import re
from typing import Any, Optional

from neo import CodeSuggestion, PlanStep, SimulationTrace


# ============================================================================
# Plan Parser
# ============================================================================

def parse_plan(response: str) -> list[PlanStep]:
    """
    Parse plan from LM response.

    Expected format:
    1. Step description
       Rationale: explanation
       Dependencies: [1, 2] or None

    2. Next step...
    """
    steps = []

    # Split into numbered sections
    # Match patterns like "1.", "Step 1:", etc.
    step_pattern = r'(?:^|\n)(?:Step\s+)?(\d+)[\.\):\-]\s*(.*?)(?=(?:\n(?:Step\s+)?\d+[\.\):\-]|\Z))'
    matches = re.finditer(step_pattern, response, re.DOTALL | re.MULTILINE)

    for match in matches:
        step_num = int(match.group(1))
        step_content = match.group(2).strip()

        # Extract description (first line)
        lines = step_content.split('\n')
        description = lines[0].strip()

        # Extract rationale
        rationale = ""
        rationale_match = re.search(r'Rationale:\s*(.+?)(?=\n|Dependencies:|$)', step_content, re.IGNORECASE)
        if rationale_match:
            rationale = rationale_match.group(1).strip()

        # Extract dependencies
        dependencies = []
        deps_match = re.search(r'Dependencies:\s*\[([^\]]+)\]', step_content, re.IGNORECASE)
        if deps_match:
            deps_str = deps_match.group(1)
            dependencies = [int(d.strip()) for d in deps_str.split(',') if d.strip().isdigit()]

        steps.append(PlanStep(
            description=description,
            rationale=rationale or "No rationale provided",
            dependencies=dependencies,
        ))

    # NO FALLBACKS - if we can't parse, that's a real error to fix
    if not steps:
        raise ValueError(
            f"Failed to parse plan from LM response. "
            f"Response preview: {response[:200]}..."
        )

    return steps


# ============================================================================
# Simulation Trace Parser
# ============================================================================

def parse_simulation_traces(response: str) -> list[SimulationTrace]:
    """
    Parse simulation traces from LM response.

    Expected format:
    Simulation 1:
      Input: ...
      Expected Output: ...
      Reasoning:
        - Step 1
        - Step 2
      Issues: ...
    """
    traces = []

    # Split into simulation sections
    sim_pattern = r'(?:^|\n)Simulation\s+(\d+):\s*(.*?)(?=\nSimulation\s+\d+:|\Z)'
    matches = re.finditer(sim_pattern, response, re.DOTALL | re.IGNORECASE)

    for match in matches:
        sim_num = int(match.group(1))
        sim_content = match.group(2).strip()

        # Extract input
        input_data = ""
        input_match = re.search(r'Input:\s*(.+?)(?=\n\s*(?:Expected|Output|Reasoning)|$)', sim_content, re.IGNORECASE)
        if input_match:
            input_data = input_match.group(1).strip()

        # Extract expected output
        expected_output = ""
        output_match = re.search(r'(?:Expected\s*)?Output:\s*(.+?)(?=\n\s*(?:Reasoning|Issues)|$)', sim_content, re.IGNORECASE)
        if output_match:
            expected_output = output_match.group(1).strip()

        # Extract reasoning steps
        reasoning_steps = []
        reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=\n\s*Issues:|\Z)', sim_content, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1)
            # Extract bullet points or numbered items
            step_pattern = r'(?:^|\n)\s*[\*\-\d\.]+\s*(.+?)(?=\n\s*[\*\-\d\.]|\Z)'
            step_matches = re.finditer(step_pattern, reasoning_text, re.DOTALL)
            reasoning_steps = [m.group(1).strip() for m in step_matches if m.group(1).strip()]

        # Extract issues
        issues_found = []
        issues_match = re.search(r'Issues:\s*(.*?)(?=\Z)', sim_content, re.DOTALL | re.IGNORECASE)
        if issues_match:
            issues_text = issues_match.group(1)
            # Extract bullet points
            issue_pattern = r'(?:^|\n)\s*[\*\-]+\s*(.+?)(?=\n\s*[\*\-]|\Z)'
            issue_matches = re.finditer(issue_pattern, issues_text, re.DOTALL)
            issues_found = [m.group(1).strip() for m in issue_matches if m.group(1).strip()]

        if input_data or expected_output:  # Only add if we found something
            traces.append(SimulationTrace(
                input_data=input_data or "No input specified",
                expected_output=expected_output or "No output specified",
                reasoning_steps=reasoning_steps or ["Reasoning not provided"],
                issues_found=issues_found,
            ))

    # NO FALLBACKS - if we can't parse simulations, that's a real error
    if not traces:
        raise ValueError(
            f"Failed to parse simulation traces from LM response. "
            f"Expected format: 'Simulation N:\\n  Input: ...\\n  Expected Output: ...\\n  Reasoning: ...' "
            f"Response preview: {response[:300]}..."
        )

    return traces


# ============================================================================
# Code Suggestion Parser
# ============================================================================

def parse_code_suggestions(response: str) -> list[CodeSuggestion]:
    """
    Parse code suggestions from LM response.

    Expected format:
    File: path/to/file.py
    Description: ...
    Confidence: 0.9
    Tradeoffs:
      - Tradeoff 1
      - Tradeoff 2

    Diff:
    ```diff
    --- a/file.py
    +++ b/file.py
    @@ -10,5 +10,5 @@
    -old line
    +new line
    ```
    """
    suggestions = []

    # Try to find code blocks with file information
    # Pattern for file-based suggestions
    file_pattern = r'File:\s*([^\n]+)\s*\n(?:.*?\n)*?(?:```(?:diff)?\s*(.*?)```|Diff:\s*(.*?)(?=File:|$))'
    matches = re.finditer(file_pattern, response, re.DOTALL | re.IGNORECASE)

    for match in matches:
        file_path = match.group(1).strip()

        # Get diff content (either from code block or Diff: section)
        diff_content = match.group(2) or match.group(3) or ""
        diff_content = diff_content.strip()

        # Extract description
        description = ""
        desc_match = re.search(r'Description:\s*(.+?)(?=\n(?:Confidence|Tradeoffs|Diff)|$)', match.group(0), re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1).strip()

        # Extract confidence
        confidence = 0.8  # Default
        conf_match = re.search(r'Confidence:\s*([\d\.]+)', match.group(0), re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))

        # Extract tradeoffs
        tradeoffs = []
        tradeoffs_match = re.search(r'Tradeoffs:\s*(.*?)(?=\n\S|\Z)', match.group(0), re.DOTALL | re.IGNORECASE)
        if tradeoffs_match:
            tradeoffs_text = tradeoffs_match.group(1)
            tradeoff_pattern = r'(?:^|\n)\s*[\*\-]+\s*(.+?)(?=\n\s*[\*\-]|\Z)'
            tradeoff_matches = re.finditer(tradeoff_pattern, tradeoffs_text, re.DOTALL)
            tradeoffs = [m.group(1).strip() for m in tradeoff_matches if m.group(1).strip()]

        suggestions.append(CodeSuggestion(
            file_path=file_path,
            unified_diff=diff_content,
            description=description or "Code modification",
            confidence=confidence,
            tradeoffs=tradeoffs,
        ))

    # Alternative pattern: just look for diff blocks
    # NO FALLBACKS - if we can't parse code suggestions, that's a real error
    # This forces us to improve the prompts or the LM output format
    if not suggestions:
        raise ValueError(
            f"Failed to parse code suggestions from LM response. "
            f"Expected format: 'File: <path>\\nDescription: ...\\nConfidence: N\\n```diff...```' "
            f"Response preview: {response[:300]}..."
        )

    return suggestions


# ============================================================================
# Helper: Extract structured data with JSON fallback
# ============================================================================

def extract_json_block(response: str) -> Optional[dict[str, Any]]:
    """Extract JSON block from response if present."""
    json_pattern = r'```json\s*(.*?)```'
    match = re.search(json_pattern, response, re.DOTALL)

    if match:
        try:
            import json
            return json.loads(match.group(1))
        except:
            pass

    return None


# ============================================================================
# Unified Parser
# ============================================================================

def parse_response(response: str, expected_type: str) -> Any:
    """
    Unified parser that tries JSON first, then falls back to regex.

    Args:
        response: LM response text
        expected_type: "plan", "simulation", or "code"

    Returns:
        Parsed data structure
    """
    # Try JSON extraction first
    json_data = extract_json_block(response)
    if json_data:
        if expected_type == "plan" and "steps" in json_data:
            return [
                PlanStep(
                    description=step.get("description", ""),
                    rationale=step.get("rationale", ""),
                    dependencies=step.get("dependencies", []),
                )
                for step in json_data["steps"]
            ]
        elif expected_type == "simulation" and "simulations" in json_data:
            return [
                SimulationTrace(
                    input_data=sim.get("input", ""),
                    expected_output=sim.get("output", ""),
                    reasoning_steps=sim.get("reasoning", []),
                    issues_found=sim.get("issues", []),
                )
                for sim in json_data["simulations"]
            ]
        elif expected_type == "code" and "suggestions" in json_data:
            return [
                CodeSuggestion(
                    file_path=sugg.get("file", ""),
                    unified_diff=sugg.get("diff", ""),
                    description=sugg.get("description", ""),
                    confidence=sugg.get("confidence", 0.5),
                    tradeoffs=sugg.get("tradeoffs", []),
                )
                for sugg in json_data["suggestions"]
            ]

    # Fallback to regex parsing
    if expected_type == "plan":
        return parse_plan(response)
    elif expected_type == "simulation":
        return parse_simulation_traces(response)
    elif expected_type == "code":
        return parse_code_suggestions(response)
    else:
        raise ValueError(f"Unknown expected_type: {expected_type}")