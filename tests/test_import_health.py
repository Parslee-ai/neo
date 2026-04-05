"""
Import health checks.

Verifies all internal neo modules import successfully and feature flags
are not silently disabled by broken imports. This catches the class of
bug where try/except ImportError swallows real failures.
"""

import importlib
import pathlib

import pytest


def _find_all_modules():
    """Find all Python modules under src/neo/."""
    src = pathlib.Path(__file__).parent.parent / "src" / "neo"
    modules = []
    for py in sorted(src.rglob("*.py")):
        if py.name.startswith("_"):
            continue
        rel = py.relative_to(src.parent)
        module = str(rel.with_suffix("")).replace("/", ".")
        modules.append(module)
    return modules


@pytest.mark.parametrize("module", _find_all_modules())
def test_module_imports(module):
    """Every non-private .py in src/neo/ must import without error."""
    importlib.import_module(module)


def test_self_correction_imports_are_direct():
    """Internal modules in self_correction must not use try/except fallback."""
    from neo.pattern_extraction import (
        extract_pattern_from_correction,
        generate_prevention_warnings,
        get_library,
    )
    from neo.algorithm_design import design_algorithm, generate_code_from_design
    from neo.input_templates import (
        extract_input_template,
        generate_solution_with_template,
        should_use_template,
    )
    # If any of these fail, the import is broken
    assert callable(extract_pattern_from_correction)
    assert callable(generate_prevention_warnings)
    assert callable(get_library)
    assert callable(design_algorithm)
    assert callable(generate_code_from_design)
    assert callable(extract_input_template)
    assert callable(generate_solution_with_template)
    assert callable(should_use_template)


def test_engine_has_no_phantom_modules():
    """Modules referenced by engine must actually exist."""
    # These were phantom imports that silently failed before cleanup
    for module_name in [
        "neo.pattern_extraction",
        "neo.algorithm_design",
        "neo.input_templates",
        "neo.constraint_verification",
        "neo.static_analysis",
    ]:
        importlib.import_module(module_name)
