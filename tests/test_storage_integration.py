#!/usr/bin/env python3
"""
Test storage adapter integration in PersistentReasoningMemory.
Verifies all issues from iteration #1 are fixed.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from neo.persistent_reasoning import PersistentReasoningMemory
from neo.storage_interface import StorageBackend


def test_storage_backend_integration():
    """Test that storage backend is properly integrated."""

    print("Testing Storage Backend Integration...")
    print("=" * 50)

    # Test 1: Type annotation is correct
    print("\n[1] Checking type annotation...")
    import inspect
    sig = inspect.signature(PersistentReasoningMemory.__init__)
    param = sig.parameters['storage_backend']

    # Check annotation includes Optional[StorageBackend]
    annotation_str = str(param.annotation)
    assert 'StorageBackend' in annotation_str or param.annotation == StorageBackend, \
        f"Expected StorageBackend in annotation, got: {annotation_str}"
    print("  ✓ Type annotation correct")

    # Test 2: Custom backend is used when provided
    print("\n[2] Testing custom backend usage...")
    mock_backend = Mock(spec=StorageBackend)
    mock_backend.load_entries.return_value = []

    memory = PersistentReasoningMemory(storage_backend=mock_backend)
    assert memory.storage_backend is mock_backend, "Custom backend not assigned"
    print("  ✓ Custom backend assigned correctly")

    # Test 3: Default backend is FileStorage
    print("\n[3] Testing default backend selection...")
    memory = PersistentReasoningMemory()
    backend_type = type(memory.storage_backend).__name__
    assert backend_type == 'FileStorage', f"Expected FileStorage, got {backend_type}"
    print("  ✓ Default backend is FileStorage")

    # Test 4: Save uses storage backend (no dual-write)
    print("\n[4] Testing save() uses only storage backend...")
    mock_backend = Mock(spec=StorageBackend)
    mock_backend.load_entries.return_value = []
    mock_backend.save_entries.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = PersistentReasoningMemory(
            storage_path=str(Path(tmpdir) / "test.json"),
            storage_backend=mock_backend
        )
        memory.save()

        # Check backend was called
        assert mock_backend.save_entries.called, "Backend save_entries not called"

        # Check no file was created (no dual-write)
        json_files = list(Path(tmpdir).glob("*.json"))
        assert len(json_files) == 0, f"Found JSON files (dual-write detected): {json_files}"
        print("  ✓ save() uses only storage backend (no dual-write)")

    # Test 5: Load uses storage backend (no fallback)
    print("\n[5] Testing load() uses only storage backend...")
    mock_backend = Mock(spec=StorageBackend)

    # First call returns empty for __init__, second call fails for explicit load()
    mock_backend.load_entries.side_effect = [[], Exception("Backend failed")]

    memory = PersistentReasoningMemory(storage_backend=mock_backend)

    try:
        memory.load()
        print("  ✗ FAILED: load() should have raised exception from backend")
        return False
    except Exception as e:
        if "Backend failed" in str(e):
            print("  ✓ load() propagated backend exception (no fallback)")
        else:
            print(f"  ✗ FAILED: Wrong exception: {e}")
            return False

    # Test 6: Helper methods removed
    print("\n[6] Checking helper methods are removed...")
    assert not hasattr(memory, '_save_to_path'), "_save_to_path still exists"
    assert not hasattr(memory, '_load_from_path'), "_load_from_path still exists"
    print("  ✓ Helper methods removed")

    # Test 7: local_storage_key initialized in storage_path branch
    print("\n[7] Testing local_storage_key initialization...")
    # Need fresh mock for this test
    fresh_mock = Mock(spec=StorageBackend)
    fresh_mock.load_entries.return_value = []

    with tempfile.TemporaryDirectory() as tmpdir:
        memory = PersistentReasoningMemory(
            storage_path=str(Path(tmpdir) / "test.json"),
            storage_backend=fresh_mock
        )
        assert hasattr(memory, 'local_storage_key'), "Missing local_storage_key attribute"
        assert memory.local_storage_key is None, "local_storage_key should be None with storage_path"
        print("  ✓ local_storage_key properly initialized")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED - Storage integration correct!")
    return True


if __name__ == "__main__":
    success = test_storage_backend_integration()
    sys.exit(0 if success else 1)