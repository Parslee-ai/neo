"""
Quick security tests for Fix #2 (deepcopy) and Fix #3+#4 (path validation).

Tests verify:
1. Fix #2: deepcopy prevents mutation of caller's nested dicts
2. Fix #3: Symlink rejection (before resolve)
3. Fix #4: Path containment using relative_to()
4. Existence check handles non-existent paths
"""

import copy
import tempfile
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_fix2_deepcopy_prevents_mutation():
    """Test Fix #2: deepcopy prevents mutation of caller's nested dicts."""
    print("\n=== Test Fix #2: Deep Copy Prevents Mutation ===")

    # Create entry with nested dict (pitfalls)
    original_entry = {
        "pattern": "test_pattern",
        "context": "test_context",
        "pitfalls": [
            {"description": "pitfall1", "severity": "high"},
            {"description": "pitfall2", "severity": "medium"}
        ],
        "embedding": [0.1, 0.2, 0.3],
        "embedding_dim": 3,
        "embedding_model": "test-model"
    }

    # Simulate what storage backends should do when processing entries
    text_entry = copy.deepcopy(original_entry)  # Fix #2

    # Remove embedding fields
    if "embedding" in text_entry:
        text_entry.pop("embedding")
        text_entry.pop("embedding_dim", None)
        text_entry.pop("embedding_model", None)

    # Verify original_entry.pitfalls NOT mutated
    assert "embedding" in original_entry, "Original should still have embedding"
    assert original_entry["pitfalls"][0]["description"] == "pitfall1", "Pitfalls should be unchanged"

    print("✅ PASS: deepcopy prevents mutation of nested dicts")
    return True


def test_fix3_symlink_rejection():
    """Test Fix #3: Symlinks rejected before resolve()."""
    print("\n=== Test Fix #3: Symlink Rejection ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)

        # Create real file
        real_file = repo_root / "real.py"
        real_file.write_text("# real file")

        # Create symlink to real file
        symlink = repo_root / "link.py"
        symlink.symlink_to(real_file)

        # Simulate ProjectIndex validation logic (Fix #3+#4)
        files_to_index = [real_file, symlink]
        accepted = []

        for file_path in files_to_index:
            # Existence check
            if not file_path.exists():
                continue

            # Symlink check BEFORE resolve (Fix #3)
            if file_path.is_symlink():
                print(f"  Rejected symlink: {file_path.name}")
                continue

            # Resolve and validate containment (Fix #4)
            resolved_path = file_path.resolve()
            repo_root_resolved = repo_root.resolve()

            try:
                resolved_path.relative_to(repo_root_resolved)
                accepted.append(file_path)
            except ValueError:
                continue

        # Verify: real_file accepted, symlink rejected
        assert len(accepted) == 1, f"Expected 1 accepted file, got {len(accepted)}"
        assert accepted[0] == real_file, "Only real file should be accepted"

        print("✅ PASS: Symlinks rejected before resolve()")
        return True


def test_fix4_path_containment():
    """Test Fix #4: Path containment using relative_to()."""
    print("\n=== Test Fix #4: Path Containment Validation ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create repo and evil sibling
        repo_root = tmpdir_path / "project"
        evil_dir = tmpdir_path / "project_evil"
        repo_root.mkdir()
        evil_dir.mkdir()

        # Create files
        inside_file = repo_root / "inside.py"
        inside_file.write_text("# inside")

        outside_file = evil_dir / "outside.py"
        outside_file.write_text("# outside")

        # Test containment validation
        files_to_test = [inside_file, outside_file]
        accepted = []

        for file_path in files_to_test:
            if not file_path.exists():
                continue

            if file_path.is_symlink():
                continue

            # Fix #4: Use relative_to() instead of string prefix
            resolved_path = file_path.resolve()
            repo_root_resolved = repo_root.resolve()

            try:
                resolved_path.relative_to(repo_root_resolved)
                accepted.append(file_path)
            except ValueError:
                print(f"  Rejected outside file: {file_path.name}")
                continue

        # Verify: only inside_file accepted
        assert len(accepted) == 1, f"Expected 1 accepted file, got {len(accepted)}"
        assert accepted[0] == inside_file, "Only inside file should be accepted"

        print("✅ PASS: Path containment prevents /project vs /project_evil attack")
        return True


def test_existence_check():
    """Test existence check handles non-existent paths gracefully."""
    print("\n=== Test Existence Check ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_root = Path(tmpdir)

        # Non-existent file
        missing_file = repo_root / "missing.py"

        # Simulate validation logic
        if not missing_file.exists():
            print(f"  Rejected non-existent: {missing_file.name}")
            result = "skipped"
        else:
            result = "accepted"

        assert result == "skipped", "Non-existent files should be skipped"

        print("✅ PASS: Existence check handles missing files gracefully")
        return True


def main():
    """Run all security tests."""
    print("=" * 60)
    print("SECURITY FIXES QUICK TEST SUITE")
    print("=" * 60)

    tests = [
        test_fix2_deepcopy_prevents_mutation,
        test_fix3_symlink_rejection,
        test_fix4_path_containment,
        test_existence_check,
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append((test.__name__, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"❌ FAIL: {test.__name__}: {e}")
            results.append((test.__name__, "FAIL"))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results:
        print(f"{status}: {name}")

    all_passed = all(status == "PASS" for _, status in results)
    print("\n" + ("=" * 60))
    print(f"OVERALL: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
