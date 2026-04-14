"""
Constraint verification layer for Neo.
Verifies solution constraints BEFORE test execution (O(n) vs O(1) LLM call).
This is the 10x opportunity: cheap verification vs expensive correction.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ConstraintType(Enum):
    """Types of verifiable constraints."""
    SORTED = "sorted"
    DIVISIBILITY = "divisibility"
    RANGE = "range"
    NON_NEGATIVE = "non_negative"
    UNIQUE_ELEMENTS = "unique_elements"
    LENGTH = "length"
    SUM_EQUALS = "sum_equals"
    INCREASING = "increasing"
    DECREASING = "decreasing"


@dataclass
class Constraint:
    """A verifiable constraint from problem description."""
    type: ConstraintType
    description: str
    parameters: Dict[str, Any]

    def to_check(self) -> str:
        """Generate Python check code for this constraint."""
        if self.type == ConstraintType.SORTED:
            var = self.parameters.get('variable', 'result')
            return f"{var} == sorted({var})"

        elif self.type == ConstraintType.DIVISIBILITY:
            var = self.parameters.get('variable', 'result')
            divisor = self.parameters.get('divisor', 1)
            return f"{var} % {divisor} == 0"

        elif self.type == ConstraintType.NON_NEGATIVE:
            var = self.parameters.get('variable', 'result')
            return f"{var} >= 0"

        elif self.type == ConstraintType.UNIQUE_ELEMENTS:
            var = self.parameters.get('variable', 'result')
            return f"len({var}) == len(set({var}))"

        elif self.type == ConstraintType.INCREASING:
            var = self.parameters.get('variable', 'result')
            return f"all({var}[i] <= {var}[i+1] for i in range(len({var})-1))"

        elif self.type == ConstraintType.DECREASING:
            var = self.parameters.get('variable', 'result')
            return f"all({var}[i] >= {var}[i+1] for i in range(len({var})-1))"

        elif self.type == ConstraintType.SUM_EQUALS:
            var = self.parameters.get('variable', 'result')
            target = self.parameters.get('target', 0)
            return f"sum({var}) == {target}"

        elif self.type == ConstraintType.LENGTH:
            var = self.parameters.get('variable', 'result')
            length = self.parameters.get('length', 0)
            return f"len({var}) == {length}"

        elif self.type == ConstraintType.RANGE:
            var = self.parameters.get('variable', 'result')
            min_val = self.parameters.get('min', float('-inf'))
            max_val = self.parameters.get('max', float('inf'))
            return f"{min_val} <= {var} <= {max_val}"

        return "True"


# Code-level markers that suggest a given constraint type is handled in the
# generated code. Used by the static (no-exec) checker in engine.py.
# Absence of a marker is a warning, not an error — the LM may satisfy the
# constraint through other means.
CONSTRAINT_CODE_MARKERS: Dict[ConstraintType, tuple] = {
    ConstraintType.SORTED: ("sorted(", ".sort(", "heappush", "heappop", "bisect"),
    ConstraintType.INCREASING: ("sorted(", ".sort(", "bisect"),
    ConstraintType.DECREASING: ("sorted(", ".sort(", "reverse=True"),
    ConstraintType.UNIQUE_ELEMENTS: ("set(", "dict.fromkeys"),
    ConstraintType.NON_NEGATIVE: ("abs(", "max(0"),
    ConstraintType.DIVISIBILITY: ("%",),
}


class ConstraintVerifier:
    """Extract and verify constraints from problem descriptions."""

    def extract_constraints(self, problem_description: str, adapter=None) -> List[Constraint]:
        """
        Parse problem description to extract verifiable constraints.
        Uses both pattern matching and LLM extraction.
        """
        constraints = []
        text = problem_description.lower()

        # Pattern-based extraction (fast, high-precision)

        # Sorted arrays
        if any(pattern in text for pattern in ['sorted array', 'sorted list', 'in sorted order', 'non-decreasing']):
            constraints.append(Constraint(
                type=ConstraintType.SORTED,
                description="Output must be sorted",
                parameters={'variable': 'result'}
            ))

        # Increasing sequence
        if 'increasing' in text and 'sorted' not in text:
            constraints.append(Constraint(
                type=ConstraintType.INCREASING,
                description="Output must be increasing",
                parameters={'variable': 'result'}
            ))

        # Decreasing sequence
        if 'decreasing' in text:
            constraints.append(Constraint(
                type=ConstraintType.DECREASING,
                description="Output must be decreasing",
                parameters={'variable': 'result'}
            ))

        # Divisibility
        divisibility_patterns = [
            r'divisible by (\d+)',
            r'multiple of (\d+)',
            r'modulo (\d+) (?:is|equals) 0'
        ]
        for pattern in divisibility_patterns:
            match = re.search(pattern, text)
            if match:
                divisor = int(match.group(1))
                constraints.append(Constraint(
                    type=ConstraintType.DIVISIBILITY,
                    description=f"Result must be divisible by {divisor}",
                    parameters={'variable': 'result', 'divisor': divisor}
                ))

        # Non-negative
        if any(pattern in text for pattern in ['non-negative', 'non negative', 'positive integer', '≥ 0', '>= 0']):
            constraints.append(Constraint(
                type=ConstraintType.NON_NEGATIVE,
                description="Result must be non-negative",
                parameters={'variable': 'result'}
            ))

        # Unique elements
        if any(pattern in text for pattern in ['unique', 'distinct', 'no duplicates', 'all different']):
            constraints.append(Constraint(
                type=ConstraintType.UNIQUE_ELEMENTS,
                description="Elements must be unique",
                parameters={'variable': 'result'}
            ))

        # LLM-based extraction (if no patterns found and adapter available)
        if not constraints and adapter:
            constraints = self._llm_extract_constraints(problem_description, adapter)

        return constraints

    def _llm_extract_constraints(self, problem_description: str, adapter) -> List[Constraint]:
        """Use LLM to extract constraints when patterns don't match."""
        prompt = f"""Extract verifiable constraints from this problem:

{problem_description[:500]}

List ONLY constraints that can be checked programmatically:
- sorted/increasing/decreasing order
- divisibility requirements
- range constraints
- uniqueness requirements
- length requirements

Format: One per line, like "sorted array" or "divisible by 3" or "non-negative integer"
If no clear constraints, return "none"."""

        try:
            response = adapter.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )

            constraints = []
            for line in response.strip().lower().split('\n'):
                line = line.strip('- ').strip()
                if not line or line == 'none':
                    continue

                # Parse LLM response into Constraint objects
                if 'sorted' in line or 'non-decreasing' in line:
                    constraints.append(Constraint(
                        type=ConstraintType.SORTED,
                        description=line,
                        parameters={'variable': 'result'}
                    ))
                elif 'increasing' in line:
                    constraints.append(Constraint(
                        type=ConstraintType.INCREASING,
                        description=line,
                        parameters={'variable': 'result'}
                    ))
                elif 'decreasing' in line:
                    constraints.append(Constraint(
                        type=ConstraintType.DECREASING,
                        description=line,
                        parameters={'variable': 'result'}
                    ))
                elif 'divisible' in line or 'multiple' in line:
                    # Try to extract number
                    match = re.search(r'\d+', line)
                    if match:
                        constraints.append(Constraint(
                            type=ConstraintType.DIVISIBILITY,
                            description=line,
                            parameters={'variable': 'result', 'divisor': int(match.group())}
                        ))
                elif 'unique' in line or 'distinct' in line:
                    constraints.append(Constraint(
                        type=ConstraintType.UNIQUE_ELEMENTS,
                        description=line,
                        parameters={'variable': 'result'}
                    ))
                elif 'non-negative' in line or 'positive' in line:
                    constraints.append(Constraint(
                        type=ConstraintType.NON_NEGATIVE,
                        description=line,
                        parameters={'variable': 'result'}
                    ))

            return constraints
        except Exception:
            return []

