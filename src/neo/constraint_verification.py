"""
Constraint verification layer for Neo.
Verifies solution constraints BEFORE test execution (O(n) vs O(1) LLM call).
This is the 10x opportunity: cheap verification vs expensive correction.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from neo.text_budget import truncate_marked

# Cap on the problem text sent for constraint extraction. Marked, because the
# caller treats the result as the complete constraint set.
_PROBLEM_DESCRIPTION_CHARS = 500


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
#
# Absence of a marker is a warning, not an error — the LM may satisfy the
# constraint through other means. That reasoning covers a different *approach*;
# it never covered a different *language*. These tables were Python-only, so on
# every C#/TypeScript/markdown target the expectation was unsatisfiable by
# construction and the warning fired permanently (#196). Markers are therefore
# keyed by language, and a language with no table produces an honest "not
# checked" note instead of a false alarm.
#
# Markers are compared case-insensitively against the (comment- and
# string-stripped) code, so they are written here in their natural casing —
# `HashSet<` reads better in the operator-facing message than `hashset<`.
_PYTHON_MARKERS: Dict[ConstraintType, tuple] = {
    ConstraintType.SORTED: ("sorted(", ".sort(", "heappush", "heappop", "bisect"),
    ConstraintType.INCREASING: ("sorted(", ".sort(", "bisect"),
    ConstraintType.DECREASING: ("sorted(", ".sort(", "reverse=True"),
    ConstraintType.UNIQUE_ELEMENTS: ("set(", "dict.fromkeys"),
    ConstraintType.NON_NEGATIVE: ("abs(", "max(0"),
    ConstraintType.DIVISIBILITY: ("%",),
}

_CSHARP_MARKERS: Dict[ConstraintType, tuple] = {
    ConstraintType.SORTED: (
        "OrderBy", ".Sort(", "Array.Sort", "SortedSet<", "SortedList<",
        "SortedDictionary<", "PriorityQueue<", "BinarySearch",
    ),
    ConstraintType.INCREASING: ("OrderBy", ".Sort(", "SortedSet<", "BinarySearch"),
    ConstraintType.DECREASING: ("OrderByDescending", ".Sort(", ".Reverse("),
    ConstraintType.UNIQUE_ELEMENTS: (
        "HashSet<", "ToHashSet(", ".Distinct(", ".GroupBy(", "ISet<",
    ),
    ConstraintType.NON_NEGATIVE: ("Math.Abs", "Math.Max(0", "Math.Clamp("),
    ConstraintType.DIVISIBILITY: ("%",),
}

# TypeScript and JavaScript share a table: the constructs a constraint is
# satisfied with (`Array.prototype.sort`, `new Set`) are the runtime's, not the
# type layer's, so splitting them would duplicate every entry to no effect.
_TYPESCRIPT_MARKERS: Dict[ConstraintType, tuple] = {
    ConstraintType.SORTED: (".sort(", "sortBy(", "orderBy(", "toSorted("),
    ConstraintType.INCREASING: (".sort(", "sortBy(", "toSorted("),
    ConstraintType.DECREASING: (".sort(", ".reverse(", "toReversed(", "orderBy("),
    ConstraintType.UNIQUE_ELEMENTS: ("new Set", "uniq(", "uniqBy(", "Set<"),
    ConstraintType.NON_NEGATIVE: ("Math.abs(", "Math.max(0"),
    ConstraintType.DIVISIBILITY: ("%",),
}

LANGUAGE_CONSTRAINT_MARKERS: Dict[str, Dict[ConstraintType, tuple]] = {
    "python": _PYTHON_MARKERS,
    "csharp": _CSHARP_MARKERS,
    "typescript": _TYPESCRIPT_MARKERS,
    "javascript": _TYPESCRIPT_MARKERS,
}

# Back-compat: this name has always meant "the Python markers", and now says so.
CONSTRAINT_CODE_MARKERS: Dict[ConstraintType, tuple] = _PYTHON_MARKERS

# Extension → language label. Entries whose label is a key of
# LANGUAGE_CONSTRAINT_MARKERS are checkable; the rest exist so the "not
# checked" note can name what the file actually is ("not checked for
# markdown") rather than shrugging at it.
_EXTENSION_LANGUAGES: Dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".cs": "csharp", ".csx": "csharp",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".cjs": "javascript",
    ".md": "markdown", ".markdown": "markdown",
    ".json": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".txt": "text",
    ".rs": "rust", ".go": "go", ".java": "java", ".rb": "ruby",
    ".php": "php", ".kt": "kotlin", ".swift": "swift",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".sql": "sql", ".sh": "shell", ".html": "html", ".css": "css",
}

# What a path with no usable extension is called in the message. A caution
# saying "not checked for " reads as a bug in Neo, not as a limit of Neo.
UNKNOWN_LANGUAGE = "unknown"


def language_for_path(file_path: str) -> str:
    """Name the language of a suggested file, for marker lookup and messaging.

    Returns a canonical name when the extension is known (``python``,
    ``csharp``, …), the bare extension when it is not (so a note can still say
    which kind of file went unchecked), and ``UNKNOWN_LANGUAGE`` when there is
    no extension to go on — including the ``/`` and ``N/A`` placeholders the
    suggestion schema uses for analysis-only answers.
    """
    from pathlib import PurePosixPath

    path = (file_path or "").strip()
    if not path or path in ("/", "N/A", "n/a"):
        return UNKNOWN_LANGUAGE
    suffix = PurePosixPath(path).suffix.lower()
    if not suffix:
        return UNKNOWN_LANGUAGE
    return _EXTENSION_LANGUAGES.get(suffix, suffix.lstrip("."))


def markers_for_language(language: str) -> Dict[ConstraintType, tuple]:
    """Marker table for a language, or an empty mapping when there is none."""
    return LANGUAGE_CONSTRAINT_MARKERS.get(language, {})


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
        """Use LLM to extract constraints when patterns don't match.

        The description is cut with a marker. The caller treats what comes
        back as *the* constraint set and verifies solutions against it, so a
        constraint stated in a dropped tail does not merely go unextracted —
        it becomes a constraint the verifier reports as satisfied because it
        never knew to check. Marking the cut lets the model say it saw only
        part of the problem instead of answering as though it saw all of it.
        """
        prompt = f"""Extract verifiable constraints from this problem:

{truncate_marked(problem_description, _PROBLEM_DESCRIPTION_CHARS)}

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
                max_tokens=200,
                reasoning_effort="low",  # Constraint extraction; not a reasoning task.
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

