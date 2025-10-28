"""Tests for the --index CLI flag functionality.

These tests validate that bug #38 is fixed:
- ProjectIndex can be imported from neo.index.project_index
- The --index flag works without ModuleNotFoundError
- Index files are created properly
"""

import subprocess
import sys

import pytest


def test_index_flag_imports_successfully():
    """Validates that the --index flag can import ProjectIndex without errors.

    This test ensures the import path is correct and the module is properly
    packaged under the neo namespace.
    """
    try:
        from neo.index.project_index import ProjectIndex
        assert ProjectIndex is not None
    except ModuleNotFoundError as e:
        pytest.fail(f"Failed to import ProjectIndex: {e}")


def test_index_flag_basic_functionality(tmp_path):
    """Validates that --index flag can build an index.

    Creates a temporary directory with Python files and verifies that:
    - The command runs without crashing
    - .neo/ directory is created
    - Index files exist (index.json, chunks.json, faiss.index)
    """
    # Create a simple Python file in the temp directory
    test_file = tmp_path / "sample.py"
    test_file.write_text("def hello():\n    print('world')\n")

    # Run neo --index in the temp directory
    result = subprocess.run(
        [sys.executable, "-m", "neo.cli", "--index"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Check command succeeded
    assert result.returncode == 0, f"Command failed: {result.stderr}"

    # Verify the actual bug (ModuleNotFoundError) does NOT appear
    assert "ModuleNotFoundError" not in result.stderr, (
        "ModuleNotFoundError found in stderr - bug #38 not fixed!"
    )
    assert "No module named 'src'" not in result.stderr, (
        "Import error found in stderr - bug #38 not fixed!"
    )

    # Check .neo directory was created
    neo_dir = tmp_path / ".neo"
    assert neo_dir.exists(), ".neo directory was not created"
    assert neo_dir.is_dir(), ".neo is not a directory"

    # Check index files exist
    index_json = neo_dir / "index.json"
    chunks_json = neo_dir / "chunks.json"
    faiss_index = neo_dir / "faiss.index"

    assert index_json.exists(), "index.json was not created"
    assert chunks_json.exists(), "chunks.json was not created"
    assert faiss_index.exists(), "faiss.index was not created"


def test_project_index_in_neo_package():
    """Validates that ProjectIndex is properly packaged under neo.

    This test ensures the correct package structure and import path work,
    preventing regression of bug #38.
    """
    # Test the correct import path
    from neo.index.project_index import ProjectIndex

    # Verify it's a class and has expected attributes
    assert callable(ProjectIndex), "ProjectIndex is not callable"
    assert hasattr(ProjectIndex, "__init__"), "ProjectIndex missing __init__"

    # Verify the module path is correct
    module_path = ProjectIndex.__module__
    assert module_path == "neo.index.project_index", (
        f"ProjectIndex module path is wrong: {module_path}"
    )
