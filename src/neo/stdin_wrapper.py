"""
Stdin Wrapper Generator: Transform Neo Functions → Stdin/Stdout Scripts

The 10x Move: Neo generates clean function definitions (good for learning/reuse).
This module adds minimal stdin/stdout boilerplate deterministically.

Algorithmic insight: The wrapper pattern is inferrable from test input structure.
- First line is count → loop that many times
- No count line → single test case
- Multiple values per line → parse and pass to function

No LLM needed - pure pattern matching on test structure.
"""

import ast
import re
from typing import Optional, Tuple


def infer_wrapper_pattern(sample_input: str, sample_output: str) -> dict:
    """
    Infer stdin reading pattern from sample test case.

    Returns dict with:
        - pattern: "multi_test" | "single_test" | "unknown"
        - test_count_var: name of loop variable (if multi_test)
        - input_per_test: how many lines per test case
    """
    input_lines = sample_input.strip().split('\n')
    output_lines = sample_output.strip().split('\n')

    # Pattern 1: First line is test count (most common in competitive programming)
    # Example: "3\nabc\ndef\nghi" → 3 test cases
    if len(input_lines) > 1:
        try:
            first_line = input_lines[0].strip()
            if first_line.isdigit():
                test_count = int(first_line)
                input_per_test = (len(input_lines) - 1) // test_count if test_count > 0 else 1

                return {
                    'pattern': 'multi_test',
                    'test_count_var': 't',
                    'input_per_test': input_per_test,
                    'has_count_line': True
                }
        except (ValueError, ZeroDivisionError):
            pass

    # Pattern 2: Output lines match input lines (no count line)
    # Example: "abc\ndef\nghi" → "YES\nNO\nYES"
    if len(output_lines) > 1 and len(input_lines) == len(output_lines):
        return {
            'pattern': 'multi_test',
            'test_count_var': None,
            'input_per_test': 1,
            'has_count_line': False,
            'line_count': len(input_lines)
        }

    # Pattern 3: Single test case
    return {
        'pattern': 'single_test',
        'test_count_var': None,
        'input_per_test': len(input_lines),
        'has_count_line': False
    }


def extract_function_name(code: str) -> str:
    """
    Extract function name from code using AST parsing.

    Args:
        code: Python code containing function definition

    Returns:
        Function name, or "solution" as fallback
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except SyntaxError:
        pass

    # Fallback to "solution" if AST parsing fails
    return "solution"


def extract_function_signature(code: str) -> Optional[Tuple[str, list[str]]]:
    """
    Extract function name and parameters from Neo's generated code using AST.

    Returns: (function_name, param_names) or None
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name
                params = []
                for arg in node.args.args:
                    params.append(arg.arg)
                return func_name, params
    except SyntaxError:
        pass

    # Fallback to regex if AST fails
    match = re.search(r'def\s+(\w+)\s*\(([^)]*)\)\s*:', code)
    if not match:
        return None

    func_name = match.group(1)
    params_str = match.group(2).strip()

    if not params_str:
        params = []
    else:
        # Split by comma, strip whitespace and type hints
        params = []
        for param in params_str.split(','):
            param = param.strip()
            # Remove type hints: "x: int" → "x"
            if ':' in param:
                param = param.split(':')[0].strip()
            # Remove default values: "x = 5" → "x"
            if '=' in param:
                param = param.split('=')[0].strip()
            if param:
                params.append(param)

    return func_name, params


def generate_stdin_wrapper(
    function_code: str,
    sample_input: str,
    sample_output: str
) -> Optional[str]:
    """
    Generate complete stdin/stdout script from Neo's function definition.

    Args:
        function_code: Clean function definition from Neo
        sample_input: Example input from test case
        sample_output: Example output from test case

    Returns:
        Complete executable script, or None if generation fails
    """
    # Extract function signature
    sig = extract_function_signature(function_code)
    if not sig:
        return None

    func_name, params = sig

    # Infer wrapper pattern from test structure
    pattern = infer_wrapper_pattern(sample_input, sample_output)

    # Generate wrapper based on pattern
    if pattern['pattern'] == 'multi_test' and pattern.get('has_count_line'):
        # Pattern: t = int(input()); for _ in range(t): ...
        num_params = len(params)

        if num_params == 0:
            # No parameters - function reads input itself (unusual)
            wrapper = f"""
{function_code}

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        result = {func_name}()
        print(result)
"""
        elif num_params == 1:
            # Single parameter - read one line per test
            wrapper = f"""
{function_code}

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        line = input().strip()
        result = {func_name}(line)
        print(result)
"""
        else:
            # Multiple parameters - read multiple lines per test
            read_lines = '\n        '.join(
                f"{param} = input().strip()" for param in params
            )
            call_args = ', '.join(params)

            wrapper = f"""
{function_code}

if __name__ == '__main__':
    t = int(input())
    for _ in range(t):
        {read_lines}
        result = {func_name}({call_args})
        print(result)
"""

    elif pattern['pattern'] == 'multi_test' and not pattern.get('has_count_line'):
        # Pattern: Loop over all input lines (no count)
        num_params = len(params)

        if num_params <= 1:
            wrapper = f"""
import sys

{function_code}

if __name__ == '__main__':
    for line in sys.stdin:
        line = line.strip()
        if line:
            result = {func_name}(line)
            print(result)
"""
        else:
            # Batch read all lines, process in groups
            wrapper = f"""
import sys

{function_code}

if __name__ == '__main__':
    lines = [line.strip() for line in sys.stdin if line.strip()]
    for i in range(0, len(lines), {num_params}):
        args = lines[i:i+{num_params}]
        if len(args) == {num_params}:
            result = {func_name}(*args)
            print(result)
"""

    else:
        # Pattern: Single test case
        num_params = len(params)

        if num_params == 0:
            wrapper = f"""
{function_code}

if __name__ == '__main__':
    result = {func_name}()
    print(result)
"""
        elif num_params == 1:
            # Read all input as single string
            wrapper = f"""
import sys

{function_code}

if __name__ == '__main__':
    data = sys.stdin.read().strip()
    result = {func_name}(data)
    print(result)
"""
        else:
            # Read multiple lines
            read_lines = '\n    '.join(
                f"{param} = input().strip()" for param in params
            )
            call_args = ', '.join(params)

            wrapper = f"""
{function_code}

if __name__ == '__main__':
    {read_lines}
    result = {func_name}({call_args})
    print(result)
"""

    return wrapper.strip()


def wrap_function_for_stdin(
    function_code: str,
    test_input: str,
    test_output: str
) -> str:
    """
    Main entry point: Transform Neo function → executable stdin/stdout script.

    Falls back to simple wrapper if pattern inference fails.
    """
    wrapper = generate_stdin_wrapper(function_code, test_input, test_output)

    if wrapper:
        return wrapper

    # Fallback: Assume single-parameter function reading one line
    # This handles most simple cases even if pattern matching fails
    return f"""
import sys

{function_code}

if __name__ == '__main__':
    # Fallback wrapper - adjust if needed
    for line in sys.stdin:
        line = line.strip()
        if line:
            # Try to extract function name
            import re
            match = re.search(r'def\\s+(\\w+)\\s*\\(', {repr(function_code)})
            if match:
                func_name = match.group(1)
                result = eval(f'{{func_name}}(line)')
                print(result)
"""