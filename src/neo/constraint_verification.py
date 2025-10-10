"""
Constraint verification layer for Neo.
Verifies solution constraints BEFORE test execution (O(n) vs O(1) LLM call).
This is the 10x opportunity: cheap verification vs expensive correction.
"""

import re
import ast
from typing import Optional, List, Dict, Any
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


@dataclass
class Violation:
    """A constraint violation found in code."""
    constraint: Constraint
    explanation: str
    fix_suggestion: str


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

    def verify_code(self, code: str, constraints: List[Constraint], test_input: str, test_output: str) -> List[Violation]:
        """
        Verify code against constraints using test synthesis.
        Returns list of violations found.
        """
        if not constraints:
            return []

        violations = []

        # Add constraint checks to code
        check_code = code + "\n\n# Constraint verification\n"
        check_code += f"_test_input = '''{test_input}'''\n"
        check_code += "import sys\nfrom io import StringIO\n"
        check_code += "_old_stdin = sys.stdin\n"
        check_code += "sys.stdin = StringIO(_test_input)\n"

        # Run code to capture result
        check_code += "import io\n_output = io.StringIO()\n_old_stdout = sys.stdout\nsys.stdout = _output\n"
        check_code += "try:\n"
        check_code += "    exec(open(__file__).read().split('# Constraint verification')[0])\n"
        check_code += "except: pass\n"
        check_code += "sys.stdout = _old_stdout\n"
        check_code += "result = _output.getvalue().strip()\n"
        check_code += "sys.stdin = _old_stdin\n\n"

        # Add constraint checks
        for constraint in constraints:
            check = constraint.to_check()
            check_code += f"# Check: {constraint.description}\n"
            check_code += f"try:\n"
            check_code += f"    if isinstance(result, str) and result.strip():\n"
            check_code += f"        # Try parsing as int/float\n"
            check_code += f"        try:\n"
            check_code += f"            result = int(result)\n"
            check_code += f"        except:\n"
            check_code += f"            try:\n"
            check_code += f"                result = float(result)\n"
            check_code += f"            except:\n"
            check_code += f"                try:\n"
            check_code += f"                    result = eval(result)\n"
            check_code += f"                except: pass\n"
            check_code += f"    assert {check}, '{constraint.description}'\n"
            check_code += f"    print('✓ {constraint.type.value}')\n"
            check_code += f"except AssertionError:\n"
            check_code += f"    print('✗ {constraint.type.value}')\n"
            check_code += f"except Exception as e:\n"
            check_code += f"    print('? {constraint.type.value}')\n"

        # Execute verification
        import tempfile
        import subprocess
        import os

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(check_code)
            temp_file = f.name

        try:
            result = subprocess.run(
                ["python3", temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Parse results
            for line in result.stdout.split('\n'):
                if line.startswith('✗'):
                    constraint_type = line.split()[1]
                    constraint = next((c for c in constraints if c.type.value == constraint_type), None)
                    if constraint:
                        violation = Violation(
                            constraint=constraint,
                            explanation=f"Code output violates constraint: {constraint.description}",
                            fix_suggestion=self._get_fix_suggestion(constraint)
                        )
                        violations.append(violation)
        except Exception:
            pass
        finally:
            os.unlink(temp_file)

        return violations

    def _get_fix_suggestion(self, constraint: Constraint) -> str:
        """Generate fix suggestion for constraint violation."""
        if constraint.type == ConstraintType.SORTED:
            return "Add result.sort() or sorted(result) before returning"

        elif constraint.type == ConstraintType.DIVISIBILITY:
            divisor = constraint.parameters.get('divisor', 1)
            return f"Ensure result is divisible by {divisor}, or round to nearest multiple"

        elif constraint.type == ConstraintType.NON_NEGATIVE:
            return "Check for negative values and use abs() or max(0, value)"

        elif constraint.type == ConstraintType.UNIQUE_ELEMENTS:
            return "Remove duplicates using set() or check before adding"

        elif constraint.type == ConstraintType.INCREASING:
            return "Sort result or ensure values are added in increasing order"

        elif constraint.type == ConstraintType.DECREASING:
            return "Sort result in reverse or ensure values are added in decreasing order"

        elif constraint.type == ConstraintType.SUM_EQUALS:
            target = constraint.parameters.get('target', 0)
            return f"Verify sum of elements equals {target}"

        elif constraint.type == ConstraintType.LENGTH:
            length = constraint.parameters.get('length', 0)
            return f"Ensure result has exactly {length} elements"

        return "Verify constraint is satisfied before returning"