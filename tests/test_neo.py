#!/usr/bin/env python3
"""
Test Neo with a real LM provider (Anthropic).

Usage:
    # Set API key in .env or environment
    python test_neo.py
"""

import json
import os
import sys

# Load environment variables from .env file
try:
    from load_env import load_env
    load_env()
except ImportError:
    pass

from neo.adapters import create_adapter
from neo.cli import NeoEngine, NeoInput, ContextFile, TaskType
from neo.exemplar_index import create_exemplar_index


def test_simple_algorithm():
    """Test Neo on a simple algorithm task."""
    print("=" * 80)
    print("Test 1: Simple Algorithm Implementation")
    print("=" * 80 + "\n")

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY=sk-...")
        return False

    # Create adapter
    adapter = create_adapter("anthropic", model="claude-3-5-sonnet-20241022")
    print(f"Using adapter: {adapter.name()}\n")

    # Create engine
    engine = NeoEngine(lm_adapter=adapter)

    # Create input
    neo_input = NeoInput(
        prompt="Write a function to check if a string is a palindrome",
        task_type=TaskType.ALGORITHM,
        context_files=[
            ContextFile(
                path="utils.py",
                content="# String utilities\n\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n",
            )
        ],
    )

    # Process
    print("Processing request...\n")
    output = engine.process(neo_input)

    # Display results
    print(f"Confidence: {output.confidence:.2%}\n")

    print("Plan:")
    for i, step in enumerate(output.plan, 1):
        print(f"  {i}. {step.description}")
        print(f"     Rationale: {step.rationale}")
    print()

    print(f"Simulations: {len(output.simulation_traces)}")
    for i, trace in enumerate(output.simulation_traces, 1):
        print(f"  {i}. Input: {trace.input_data[:50]}...")
        print(f"     Output: {trace.expected_output[:50]}...")
        if trace.issues_found:
            print(f"     Issues: {', '.join(trace.issues_found)}")
    print()

    print(f"Code Suggestions: {len(output.code_suggestions)}")
    for i, sugg in enumerate(output.code_suggestions, 1):
        print(f"  {i}. {sugg.file_path}: {sugg.description}")
        print(f"     Confidence: {sugg.confidence:.2%}")
        if sugg.unified_diff:
            print(f"     Diff preview: {sugg.unified_diff[:100]}...")
    print()

    print(f"Notes: {output.notes}\n")

    return True


def test_bugfix():
    """Test Neo on a bugfix task."""
    print("=" * 80)
    print("Test 2: Bug Fix")
    print("=" * 80 + "\n")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return False

    adapter = create_adapter("anthropic")
    engine = NeoEngine(lm_adapter=adapter)

    buggy_code = """def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)  # Bug: division by zero if empty list
"""

    error_trace = """Traceback (most recent call last):
  File "test.py", line 5, in test_average
    result = calculate_average([])
  File "math_utils.py", line 5, in calculate_average
    return total / len(numbers)
ZeroDivisionError: division by zero
"""

    neo_input = NeoInput(
        prompt="Fix the division by zero error in calculate_average",
        task_type=TaskType.BUGFIX,
        context_files=[
            ContextFile(path="math_utils.py", content=buggy_code)
        ],
        error_trace=error_trace,
    )

    print("Processing bugfix request...\n")
    output = engine.process(neo_input)

    print(f"Confidence: {output.confidence:.2%}\n")
    print("Plan:")
    for i, step in enumerate(output.plan, 1):
        print(f"  {i}. {step.description}")
    print()

    if output.code_suggestions:
        print("Suggested fix:")
        print(output.code_suggestions[0].unified_diff[:300])
        print()

    if output.next_questions:
        print("Next questions:")
        for q in output.next_questions:
            print(f"  - {q}")
        print()

    return True


def test_with_exemplars():
    """Test Neo with exemplar retrieval."""
    print("=" * 80)
    print("Test 3: With Exemplar Learning")
    print("=" * 80 + "\n")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return False

    # Create exemplar index
    index = create_exemplar_index()

    # Add some exemplars
    print("Adding exemplars to index...")
    index.add(
        prompt="Write a function to reverse a string",
        solution="def reverse(s): return s[::-1]",
        task_type="algorithm",
        metadata={"language": "python"},
    )
    index.add(
        prompt="Create a function to check if string is palindrome",
        solution="def is_palindrome(s): return s == s[::-1]",
        task_type="algorithm",
        metadata={"language": "python"},
    )
    print(f"Index contains {len(index)} exemplars\n")

    # Create engine with exemplar index
    adapter = create_adapter("anthropic")
    engine = NeoEngine(lm_adapter=adapter, exemplar_index=index)

    # Test similar task
    neo_input = NeoInput(
        prompt="Write a function to check if a string reads the same forwards and backwards",
        task_type=TaskType.ALGORITHM,
    )

    print("Processing with exemplar retrieval...\n")
    output = engine.process(neo_input)

    print(f"Confidence: {output.confidence:.2%}\n")
    print("Plan (should benefit from exemplar knowledge):")
    for i, step in enumerate(output.plan, 1):
        print(f"  {i}. {step.description}")
    print()

    # Clean up
    index.clear()
    index.save()

    return True


def main():
    """Run all tests."""
    print("\nNeo Integration Tests with Anthropic")
    print("=" * 80 + "\n")

    tests = [
        ("Simple Algorithm", test_simple_algorithm),
        ("Bug Fix", test_bugfix),
        ("Exemplar Learning", test_with_exemplars),
    ]

    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\nERROR in {name}: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((name, False))

        print()

    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)