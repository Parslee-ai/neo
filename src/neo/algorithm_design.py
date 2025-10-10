"""
Algorithm design phase for Neo.
Forces explicit algorithm design before code generation.
Prevents off-by-one and logic errors by thinking before coding.
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum


class AlgorithmClass(Enum):
    """Common algorithm classes for coding problems."""
    GREEDY = "greedy"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    TWO_POINTER = "two_pointer"
    SLIDING_WINDOW = "sliding_window"
    GRAPH = "graph"
    TREE = "tree"
    BACKTRACKING = "backtracking"
    DIVIDE_CONQUER = "divide_conquer"
    BINARY_SEARCH = "binary_search"
    SORTING = "sorting"
    SIMULATION = "simulation"
    MATH = "math"
    OTHER = "other"


@dataclass
class AlgorithmDesign:
    """Structured algorithm design before code generation."""
    algorithm_class: AlgorithmClass
    key_insight: str
    steps: List[str]
    edge_cases: List[str]
    data_structures: List[str]
    example_trace: str
    complexity: Optional[str] = None


def design_algorithm(problem_description: str, adapter) -> AlgorithmDesign:
    """
    Force explicit algorithm design before code generation.

    This prevents algorithmic bugs like:
    - Off-by-one errors (boundary conditions not thought through)
    - Logic errors (algorithm not validated on examples)
    - Wrong data structure choices (not considered alternatives)

    Returns structured design that guides code generation.
    """

    design_prompt = f"""Design an algorithm for this problem (DON'T write code yet):

{problem_description[:800]}

Think step-by-step. Answer in this EXACT format:

1. Algorithm class: [Choose ONE: greedy, dp, two-pointer, sliding-window, graph, tree, backtracking, divide-conquer, binary-search, sorting, simulation, math, other]

2. Key insight (one sentence):

3. Step-by-step approach (3-5 numbered steps):

4. Edge cases to handle (list 2-3):

5. Data structures needed (list):

6. Example walkthrough (trace algorithm on the sample input):

7. Time complexity:

Be specific about boundary conditions and loop invariants."""

    response = adapter.generate(
        [{"role": "user", "content": design_prompt}],
        temperature=0.0,
        max_tokens=800
    )

    # Parse response
    return _parse_design(response)


def _parse_design(response: str) -> AlgorithmDesign:
    """Parse LLM response into structured AlgorithmDesign."""

    lines = response.strip().split('\n')

    algorithm_class = AlgorithmClass.OTHER
    key_insight = ""
    steps = []
    edge_cases = []
    data_structures = []
    example_trace = ""
    complexity = ""

    current_section = None

    for line in lines:
        line_lower = line.lower().strip()

        # Detect sections
        if "algorithm class" in line_lower or line_lower.startswith("1."):
            current_section = "algorithm_class"
            # Extract class from this line or next
            class_text = line.split(':', 1)[-1].strip().lower()
            # Map text to enum
            if "greedy" in class_text:
                algorithm_class = AlgorithmClass.GREEDY
            elif "dp" in class_text or "dynamic" in class_text:
                algorithm_class = AlgorithmClass.DYNAMIC_PROGRAMMING
            elif "two" in class_text and "pointer" in class_text:
                algorithm_class = AlgorithmClass.TWO_POINTER
            elif "sliding" in class_text and "window" in class_text:
                algorithm_class = AlgorithmClass.SLIDING_WINDOW
            elif "graph" in class_text:
                algorithm_class = AlgorithmClass.GRAPH
            elif "tree" in class_text:
                algorithm_class = AlgorithmClass.TREE
            elif "backtrack" in class_text:
                algorithm_class = AlgorithmClass.BACKTRACKING
            elif "divide" in class_text or "conquer" in class_text:
                algorithm_class = AlgorithmClass.DIVIDE_CONQUER
            elif "binary" in class_text and "search" in class_text:
                algorithm_class = AlgorithmClass.BINARY_SEARCH
            elif "sort" in class_text:
                algorithm_class = AlgorithmClass.SORTING
            elif "simulation" in class_text or "simulate" in class_text:
                algorithm_class = AlgorithmClass.SIMULATION
            elif "math" in class_text:
                algorithm_class = AlgorithmClass.MATH

        elif "key insight" in line_lower or line_lower.startswith("2."):
            current_section = "key_insight"
            insight_text = line.split(':', 1)[-1].strip()
            if insight_text:
                key_insight = insight_text

        elif "step-by-step" in line_lower or "approach" in line_lower or line_lower.startswith("3."):
            current_section = "steps"

        elif "edge case" in line_lower or line_lower.startswith("4."):
            current_section = "edge_cases"

        elif "data structure" in line_lower or line_lower.startswith("5."):
            current_section = "data_structures"

        elif "example" in line_lower or "walkthrough" in line_lower or "trace" in line_lower or line_lower.startswith("6."):
            current_section = "example_trace"

        elif "complexity" in line_lower or line_lower.startswith("7."):
            current_section = "complexity"
            complexity_text = line.split(':', 1)[-1].strip()
            if complexity_text:
                complexity = complexity_text

        # Collect content for current section
        elif current_section and line.strip():
            if current_section == "key_insight":
                if not key_insight:
                    key_insight = line.strip()

            elif current_section == "steps":
                # Look for numbered/bulleted items
                step = line.strip().lstrip('0123456789.-*•) ').strip()
                if step and len(step) > 5:  # Filter out section headers
                    steps.append(step)

            elif current_section == "edge_cases":
                case = line.strip().lstrip('0123456789.-*•) ').strip()
                if case and len(case) > 5:
                    edge_cases.append(case)

            elif current_section == "data_structures":
                ds = line.strip().lstrip('0123456789.-*•) ').strip()
                if ds and len(ds) > 2:
                    data_structures.append(ds)

            elif current_section == "example_trace":
                example_trace += line + "\n"

            elif current_section == "complexity":
                if not complexity:
                    complexity = line.strip()

    return AlgorithmDesign(
        algorithm_class=algorithm_class,
        key_insight=key_insight or "Not specified",
        steps=steps if steps else ["Not specified"],
        edge_cases=edge_cases if edge_cases else ["Not specified"],
        data_structures=data_structures if data_structures else ["Not specified"],
        example_trace=example_trace.strip() if example_trace else "Not provided",
        complexity=complexity or "Not specified"
    )


def generate_code_from_design(
    problem_description: str,
    design: AlgorithmDesign,
    adapter,
    memory_context: str = "",
    prevention_warnings: str = ""
) -> str:
    """
    Generate code based on algorithm design.
    Uses design to guide implementation and avoid common pitfalls.
    """

    # Build guidance from design
    design_guidance = f"""
ALGORITHM DESIGN (follow this exactly):
Class: {design.algorithm_class.value}
Key insight: {design.key_insight}

Steps to implement:
{chr(10).join([f"{i+1}. {step}" for i, step in enumerate(design.steps[:5])])}

Edge cases to handle:
{chr(10).join([f"- {case}" for case in design.edge_cases[:3]])}

Data structures: {', '.join(design.data_structures[:3])}

Expected complexity: {design.complexity}
"""

    code_prompt = f"""Implement the following algorithm design in Python:

PROBLEM:
{problem_description[:500]}

{design_guidance}

Generate Python code that reads from stdin and prints to stdout.
Return ONLY executable Python code, no explanations.
Follow the algorithm design EXACTLY, paying special attention to:
- Boundary conditions (off-by-one errors)
- Loop invariants
- Edge cases listed above{memory_context}{prevention_warnings}"""

    code = adapter.generate(
        [{"role": "user", "content": code_prompt}],
        temperature=0.0,
        max_tokens=2000
    )

    # Extract code
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    return code