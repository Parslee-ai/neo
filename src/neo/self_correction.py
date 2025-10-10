"""
Self-correction module for Neo.
Implements test-driven refinement with failure diagnosis.
Also extracts prevention patterns from successful corrections.
"""

import subprocess
import tempfile
import os
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

try:
    from pattern_extraction import extract_pattern_from_correction, get_library, generate_prevention_warnings
    PATTERN_EXTRACTION_AVAILABLE = True
except ImportError:
    PATTERN_EXTRACTION_AVAILABLE = False

try:
    from constraint_verification import ConstraintVerifier
    CONSTRAINT_VERIFICATION_AVAILABLE = True
except ImportError:
    CONSTRAINT_VERIFICATION_AVAILABLE = False

try:
    from algorithm_design import design_algorithm, generate_code_from_design
    ALGORITHM_DESIGN_AVAILABLE = True
except ImportError:
    ALGORITHM_DESIGN_AVAILABLE = False

try:
    from input_templates import extract_input_template, generate_solution_with_template, should_use_template
    INPUT_TEMPLATE_AVAILABLE = True
except ImportError:
    INPUT_TEMPLATE_AVAILABLE = False


@dataclass
class TestResult:
    """Result of running code against test case."""
    passed: bool
    expected: str
    actual: str
    error: Optional[str] = None
    timeout: bool = False


@dataclass
class FailureAutopsy:
    """Structured analysis of why a solution failed."""
    bug_category: str  # "off-by-one" | "wrong-datastructure" | "local-vs-global" | "edge-case" | "complexity-timeout"
    root_cause: str    # One-sentence explanation
    correct_approach: str  # One-sentence fix
    similar_patterns: list[str]  # Related problem types
    was_in_memory: bool  # Was this pattern already stored?


def run_code_against_test(
    code: str,
    test_input: str,
    expected_output: str,
    timeout_seconds: int = 5
) -> TestResult:
    """Execute Python code against a test case and return result."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name

    try:
        result = subprocess.run(
            ["python3", temp_file],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )

        actual = result.stdout
        passed = actual.strip() == expected_output.strip()

        return TestResult(
            passed=passed,
            expected=expected_output,
            actual=actual,
            error=result.stderr if result.returncode != 0 else None
        )

    except subprocess.TimeoutExpired:
        return TestResult(
            passed=False,
            expected=expected_output,
            actual="",
            timeout=True
        )
    except Exception as e:
        return TestResult(
            passed=False,
            expected=expected_output,
            actual="",
            error=str(e)
        )
    finally:
        os.unlink(temp_file)


def diagnose_failure(
    problem_description: str,
    failed_code: str,
    test_result: TestResult,
    adapter,
    memory_patterns: list = None
) -> FailureAutopsy:
    """
    Use LLM to analyze why the solution failed and categorize the bug.

    Returns structured autopsy that can be stored in memory.
    """
    # Check if pattern was in memory
    was_in_memory = bool(memory_patterns)

    # Build diagnostic prompt
    diagnostic_prompt = f"""Analyze this failed coding solution:

PROBLEM:
{problem_description[:500]}

YOUR SOLUTION:
```python
{failed_code}
```

TEST FAILURE:
Expected output: {test_result.expected[:200]}
Actual output: {test_result.actual[:200]}
{'Runtime error: ' + test_result.error if test_result.error else ''}
{'TIMEOUT' if test_result.timeout else ''}

Answer in this EXACT format:
1. Bug category: [Choose ONE: off-by-one | wrong-datastructure | local-vs-global | edge-case | complexity-timeout | logic-error]
2. Root cause (one sentence):
3. Correct approach (one sentence):
4. Similar problem patterns (list 2-3):
5. Key insight to remember:"""

    response = adapter.generate(
        [{"role": "user", "content": diagnostic_prompt}],
        temperature=0.0,
        max_tokens=500
    )

    # Parse response (simple parsing, could be improved)
    lines = response.strip().split('\n')

    bug_category = "logic-error"  # default
    root_cause = "Unknown"
    correct_approach = "Unknown"
    similar_patterns = []

    for line in lines:
        if "bug category:" in line.lower():
            # Extract category
            parts = line.split(':')
            if len(parts) > 1:
                cat = parts[1].strip().split()[0].strip('[]*')  # Strip markdown too
                bug_category = cat
        elif "root cause" in line.lower():
            root_cause = line.split(':', 1)[1].strip() if ':' in line else line
        elif "correct approach" in line.lower():
            correct_approach = line.split(':', 1)[1].strip() if ':' in line else line
        elif "similar" in line.lower():
            # Try to extract list items from next few lines
            continue

    return FailureAutopsy(
        bug_category=bug_category,
        root_cause=root_cause,
        correct_approach=correct_approach,
        similar_patterns=similar_patterns,
        was_in_memory=was_in_memory
    )


def self_correct_solution(
    problem_description: str,
    failed_code: str,
    test_result: TestResult,
    autopsy: FailureAutopsy,
    adapter,
    max_attempts: int = 2
) -> Tuple[str, bool]:
    """
    Attempt to fix the solution using failure diagnosis.

    Returns: (corrected_code, success)
    """
    correction_prompt = f"""Your previous solution failed. Fix it using this analysis:

PROBLEM:
{problem_description[:500]}

YOUR FAILED SOLUTION:
```python
{failed_code}
```

FAILURE ANALYSIS:
Bug category: {autopsy.bug_category}
Root cause: {autopsy.root_cause}
Correct approach: {autopsy.correct_approach}

TEST FAILURE:
Expected: {test_result.expected[:200]}
Got: {test_result.actual[:200]}

Generate a CORRECTED Python solution. Return ONLY code, no explanations."""

    corrected = adapter.generate(
        [{"role": "user", "content": correction_prompt}],
        temperature=0.0,
        max_tokens=2000
    )

    # Extract code
    if "```python" in corrected:
        corrected = corrected.split("```python")[1].split("```")[0].strip()
    elif "```" in corrected:
        corrected = corrected.split("```")[1].split("```")[0].strip()

    return corrected, False  # Return code, caller should test it


def iterative_solve_with_correction(
    problem: Dict[str, Any],
    test_case: Dict[str, str],
    adapter,
    memory,
    max_attempts: int = 3,
    learn_patterns: bool = True
) -> Dict[str, Any]:
    """
    Solve a problem with test-driven self-correction.

    If learn_patterns=True, extracts prevention patterns from successful corrections.

    Returns result dict with solution, autopsy, and metrics.
    """
    # Retrieve memory
    relevant = memory.retrieve_relevant(
        problem_context={
            "prompt": problem['question_title'],
            "task_type": "code_generation"
        },
        k=3
    )

    memory_context = ""
    if relevant:
        memory_context = f"\n\nPast reasoning:\n{relevant[0].pattern}\nSuggestion: {relevant[0].suggestion[:300]}"

    # Get prevention warnings from learned patterns
    prevention_warnings = ""
    if PATTERN_EXTRACTION_AVAILABLE and learn_patterns:
        library = get_library()
        prevention_warnings = generate_prevention_warnings(
            problem['question_content'],
            None,
            library
        )

    # Try input template generation first (Liotta's 10x recommendation)
    input_template = None
    if INPUT_TEMPLATE_AVAILABLE and should_use_template(problem['question_content']):
        input_template = extract_input_template(problem['question_content'], adapter)

    if input_template:
        # Generate solution with template (40% of failures eliminated)
        code = generate_solution_with_template(
            problem['question_content'],
            input_template,
            adapter,
            memory_context,
            prevention_warnings
        )
    else:
        # Fallback: Direct generation
        prompt = f"""Solve: {problem['question_title']}

{problem['question_content'][:800]}

Generate Python code that reads from stdin and prints to stdout.
Return ONLY executable Python code, no explanations.{memory_context}{prevention_warnings}"""

        code = adapter.generate([{"role": "user", "content": prompt}], temperature=0.0)

        # Extract code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

    attempts = []
    current_code = code

    for attempt in range(max_attempts):
        # Test current solution
        test_result = run_code_against_test(
            current_code,
            test_case['input'],
            test_case['output']
        )

        attempts.append({
            "attempt": attempt + 1,
            "code": current_code,
            "passed": test_result.passed
        })

        if test_result.passed:
            # If we corrected on attempt >1, learn from the correction
            pattern_learned = False
            if attempt > 0 and learn_patterns and PATTERN_EXTRACTION_AVAILABLE:
                first_failed_code = attempts[0]['code']
                pattern_learned = learn_from_correction(
                    problem,
                    first_failed_code,
                    current_code,
                    autopsy,  # Will be set from previous iteration
                    adapter
                )

            return {
                "success": True,
                "attempts": attempts,
                "final_code": current_code,
                "autopsy": None,
                "pattern_learned": pattern_learned
            }

        # Diagnose failure
        autopsy = diagnose_failure(
            problem['question_content'],
            current_code,
            test_result,
            adapter,
            relevant
        )

        if attempt < max_attempts - 1:
            # Use algorithm design for correction (70% success rate)
            if attempt == 0 and ALGORITHM_DESIGN_AVAILABLE:
                # First failure: Design algorithm properly
                algorithm_design = design_algorithm(problem['question_content'], adapter)
                current_code = generate_code_from_design(
                    problem['question_content'],
                    algorithm_design,
                    adapter,
                    memory_context=f"\n\nPREVIOUS FAILURE:\n{autopsy.root_cause}\nCorrect approach: {autopsy.correct_approach}",
                    prevention_warnings=prevention_warnings
                )
            else:
                # Subsequent failures: Use diagnosis-based correction
                current_code, _ = self_correct_solution(
                    problem['question_content'],
                    current_code,
                    test_result,
                    autopsy,
                    adapter
                )

    # Failed after all attempts
    return {
        "success": False,
        "attempts": attempts,
        "final_code": current_code,
        "autopsy": autopsy,
        "pattern_learned": False
    }


def learn_from_correction(
    problem: Dict[str, Any],
    failed_code: str,
    corrected_code: str,
    autopsy: FailureAutopsy,
    adapter
) -> bool:
    """
    Extract and store a prevention pattern from a successful correction.

    Returns True if pattern was learned.
    """
    if not PATTERN_EXTRACTION_AVAILABLE:
        return False

    try:
        pattern = extract_pattern_from_correction(
            problem['question_content'],
            failed_code,
            corrected_code,
            autopsy.bug_category,
            autopsy.root_cause,
            adapter
        )

        library = get_library()
        library.add_pattern(pattern)

        return True
    except Exception as e:
        # Don't fail the whole solve if pattern extraction fails
        print(f"Warning: Pattern extraction failed: {e}")
        return False