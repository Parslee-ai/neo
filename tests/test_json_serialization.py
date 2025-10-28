#!/usr/bin/env python3
"""
Test JSON serialization of ReasoningEntry with enum and dataclass objects.

This test reproduces the actual bug where TaskType enums and ContextFile
dataclasses in source_context dict cause TypeError during json.dumps().
"""

import json
import pytest
import tempfile
import os
from neo.persistent_reasoning import PersistentReasoningMemory, ReasoningEntry
from neo.cli import TaskType, ContextFile


def test_context_json_serialization_actual_bug():
    """
    Test that source_context with TaskType enum and ContextFile objects
    can be serialized through the actual storage path.

    This reproduces the actual bug scenario:
    1. NeoEngine._retrieve_context is called (returns dict with enums/dataclasses)
    2. Dict is passed to add_reasoning as source_context
    3. ReasoningEntry stores it in self.source_context
    4. save() calls to_dict() which must serialize source_context
    5. storage_backend.save_entries() calls json.dumps()

    The bug occurs at step 5 when json.dumps encounters non-serializable objects.
    """
    # Create temp directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup memory with temp storage
        old_home = os.environ.get('HOME')
        try:
            os.environ['HOME'] = temp_dir
            memory = PersistentReasoningMemory()

            # Create source_context exactly as _retrieve_context would (BEFORE fix)
            # This contains the actual enum and dataclass objects that cause the bug
            source_context = {
                "task_type": TaskType.ALGORITHM,  # Enum object
                "files": [
                    ContextFile(path="test.py", content="print('hello')", line_range=None)
                ],  # Dataclass objects
                "prompt": "test prompt",
                "error_trace": None,
            }

            # Add reasoning with problematic source_context
            memory.add_reasoning(
                pattern="test pattern",
                context="test context",
                reasoning="test reasoning",
                suggestion="test suggestion",
                confidence=0.8,
                source_context=source_context,
            )

            # This is where the bug would occur - save calls to_dict() then json.dumps()
            # If the fix is correct, this should succeed
            memory.save()

            # Verify the saved data is valid JSON
            memory2 = PersistentReasoningMemory()
            memory2.load()

            assert len(memory2.entries) == 1
            entry = memory2.entries[0]

            # Verify the source_context was properly serialized
            assert entry.source_context["task_type"] == "algorithm"  # String, not enum
            assert isinstance(entry.source_context["files"], list)
            assert entry.source_context["files"][0]["path"] == "test.py"  # Dict, not dataclass
            assert entry.source_context["files"][0]["content"] == "print('hello')"

        finally:
            if old_home:
                os.environ['HOME'] = old_home


def test_reasoning_entry_serializes_source_context_with_enums():
    """
    Test that ReasoningEntry.to_dict() correctly serializes source_context
    containing TaskType enums.

    This is a focused unit test on the serialization boundary.
    """
    # Create entry with enum in source_context
    entry = ReasoningEntry(
        pattern="test pattern",
        context="test context",
        reasoning="test reasoning",
        suggestion="test suggestion",
        confidence=0.7,
        source_context={
            "task_type": TaskType.BUGFIX,
            "prompt": "fix this bug",
        }
    )

    # Call to_dict - this is the serialization boundary
    entry_dict = entry.to_dict()

    # Verify source_context is JSON-serializable
    json_str = json.dumps(entry_dict)

    # Verify the serialized data is correct
    data = json.loads(json_str)
    assert data["source_context"]["task_type"] == "bugfix"
    assert data["source_context"]["prompt"] == "fix this bug"


def test_reasoning_entry_serializes_source_context_with_dataclasses():
    """
    Test that ReasoningEntry.to_dict() correctly serializes source_context
    containing ContextFile dataclasses.
    """
    # Create entry with dataclass in source_context
    entry = ReasoningEntry(
        pattern="test pattern",
        context="test context",
        reasoning="test reasoning",
        suggestion="test suggestion",
        confidence=0.7,
        source_context={
            "files": [
                ContextFile(path="file1.py", content="code1", line_range=(1, 10)),
                ContextFile(path="file2.py", content="code2", line_range=None),
            ],
            "prompt": "test",
        }
    )

    # Call to_dict - this is the serialization boundary
    entry_dict = entry.to_dict()

    # Verify source_context is JSON-serializable
    json_str = json.dumps(entry_dict)

    # Verify the serialized data is correct
    data = json.loads(json_str)
    assert len(data["source_context"]["files"]) == 2
    assert data["source_context"]["files"][0]["path"] == "file1.py"
    assert data["source_context"]["files"][0]["content"] == "code1"
    assert data["source_context"]["files"][0]["line_range"] == [1, 10]
    assert data["source_context"]["files"][1]["path"] == "file2.py"
    assert data["source_context"]["files"][1]["line_range"] is None


def test_reasoning_entry_serializes_mixed_source_context():
    """
    Test serialization with both enums and dataclasses in source_context.
    """
    entry = ReasoningEntry(
        pattern="test pattern",
        context="test context",
        reasoning="test reasoning",
        suggestion="test suggestion",
        confidence=0.7,
        source_context={
            "task_type": TaskType.FEATURE,
            "files": [ContextFile(path="app.py", content="# code", line_range=None)],
            "prompt": "add feature",
            "error_trace": "some error",
            "commands": ["cmd1", "cmd2"],
        }
    )

    # Call to_dict and verify JSON serialization
    entry_dict = entry.to_dict()
    json_str = json.dumps(entry_dict)
    data = json.loads(json_str)

    # Verify all fields serialized correctly
    assert data["source_context"]["task_type"] == "feature"
    assert data["source_context"]["files"][0]["path"] == "app.py"
    assert data["source_context"]["files"][0]["content"] == "# code"
    assert data["source_context"]["prompt"] == "add feature"
    assert data["source_context"]["error_trace"] == "some error"
    assert data["source_context"]["commands"] == ["cmd1", "cmd2"]


def test_empty_source_context_serializes():
    """Test that empty source_context serializes correctly."""
    entry = ReasoningEntry(
        pattern="test pattern",
        context="test context",
        reasoning="test reasoning",
        suggestion="test suggestion",
        confidence=0.7,
        source_context={}
    )

    entry_dict = entry.to_dict()
    json_str = json.dumps(entry_dict)
    data = json.loads(json_str)

    assert data["source_context"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
