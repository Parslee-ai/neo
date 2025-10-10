"""
Input Template Generation: The 10x Move
Separates input parsing (deterministic) from algorithm design (creative).

Expected impact: +20pp (75% → 95%), eliminates 40% of failures.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class InputTemplate:
    """Generated input parsing code."""
    code: str
    variables: list[str]  # Variable names extracted (e.g., ['nums', 'k'])
    format_description: str  # Human-readable format


def extract_input_template(problem_description: str, adapter) -> Optional[InputTemplate]:
    """
    Generate deterministic input parsing code from problem description.

    This is the 10x opportunity: Solve input parsing once, correctly.
    No pattern learning, no warnings - just correct code generation.

    Args:
        problem_description: Full problem text
        adapter: LLM adapter (use cheap model like GPT-3.5)

    Returns:
        InputTemplate with parsing code, or None if extraction fails
    """

    # Use cheap model for template extraction ($0.001 vs $0.01)
    extraction_prompt = f"""Extract the input format from this coding problem and generate parsing code.

Problem:
{problem_description[:1000]}

Identify:
1. What variables are in the input (e.g., "0-indexed array nums", "integer k")
2. How they're formatted (single line, multiple lines, space-separated, etc.)
3. The order they appear

Generate Python code that:
- Reads from stdin
- Parses the input into correct data types
- Stores in appropriately named variables
- Handles edge cases (empty input, whitespace, etc.)

Format your response as:

VARIABLES: [list variable names]
FORMAT: [describe input format]
CODE:
```python
[parsing code here]
```

Example:
VARIABLES: nums, k
FORMAT: First line contains space-separated integers for array nums, second line contains integer k
CODE:
```python
import sys
lines = sys.stdin.read().strip().split('\\n')
nums = list(map(int, lines[0].split()))
k = int(lines[1])
```

Be specific and defensive. Handle JSON-formatted input, multiline arrays, etc."""

    try:
        response = adapter.generate(
            [{"role": "user", "content": extraction_prompt}],
            temperature=0.0,
            max_tokens=500
        )

        # Parse response
        variables = []
        format_desc = ""
        code = ""

        lines = response.split('\n')
        current_section = None
        code_lines = []

        for line in lines:
            if line.startswith('VARIABLES:'):
                var_text = line.replace('VARIABLES:', '').strip()
                # Extract variable names
                variables = [v.strip().strip(',[]') for v in var_text.split() if v.strip() and v not in [',', '[', ']']]
            elif line.startswith('FORMAT:'):
                format_desc = line.replace('FORMAT:', '').strip()
            elif '```python' in line:
                current_section = 'code'
            elif '```' in line and current_section == 'code':
                current_section = None
            elif current_section == 'code':
                code_lines.append(line)

        code = '\n'.join(code_lines).strip()

        if code and variables:
            return InputTemplate(
                code=code,
                variables=variables,
                format_description=format_desc or "Input format extracted"
            )

        return None

    except Exception as e:
        # If template extraction fails, return None (fallback to standard approach)
        print(f"Warning: Template extraction failed: {e}")
        return None


def generate_solution_with_template(
    problem_description: str,
    template: InputTemplate,
    adapter,
    memory_context: str = "",
    prevention_warnings: str = ""
) -> str:
    """
    Generate solution with input parsing already solved.

    This is the leverage: LLM focuses on algorithm, not parsing.

    Args:
        problem_description: Problem text
        template: Pre-generated input parsing code
        adapter: LLM adapter
        memory_context: Past reasoning (if any)
        prevention_warnings: Pattern warnings (if any)

    Returns:
        Complete solution code
    """

    solution_prompt = f"""Solve this coding problem using the provided input parsing code.

Problem:
{problem_description[:800]}

INPUT PARSING (DO NOT MODIFY):
The input has been parsed for you. Use these variables:
{', '.join(template.variables)}

Input parsing code (already correct, don't change it):
```python
{template.code}
```

YOUR TASK:
Write the solution logic that uses the parsed variables.
The input parsing is already done - focus on the algorithm.

Generate ONLY the solution logic that comes after the input parsing.
Return executable Python code that:
1. Uses the variables from the input parsing above
2. Implements the solution algorithm
3. Prints the result

DO NOT regenerate input parsing. Use the variables provided.{memory_context}{prevention_warnings}"""

    code = adapter.generate(
        [{"role": "user", "content": solution_prompt}],
        temperature=0.0,
        max_tokens=2000
    )

    # Extract solution code
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    # Combine template + solution
    full_code = f"""import sys

# Input parsing (generated template)
{template.code}

# Solution logic
{code}"""

    return full_code


def should_use_template(problem_description: str) -> bool:
    """
    Decide if template generation is worth it.

    Heuristic: If problem mentions specific input format, use template.
    Otherwise, direct generation is fine.
    """
    indicators = [
        'input',
        'array',
        'integer',
        'string',
        'given',
        'first line',
        'second line',
        '0-indexed',
        '1-indexed',
    ]

    text = problem_description.lower()
    matches = sum(1 for indicator in indicators if indicator in text)

    # If multiple format indicators, use template
    return matches >= 3