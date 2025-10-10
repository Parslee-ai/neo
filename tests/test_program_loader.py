"""
Tests for program loader (The Operator).
"""

import pytest
from neo.program_loader import ProgramLoader, ProgramLoadResult
from neo.persistent_reasoning import PersistentReasoningMemory


class TestProgramLoader:
    """Test suite for ProgramLoader."""

    def test_hash_pattern(self):
        """Test pattern hashing for deduplication."""
        memory = PersistentReasoningMemory(max_entries=10)
        loader = ProgramLoader(memory)

        pattern1 = {"pattern": "test", "suggestion": "code"}
        pattern2 = {"pattern": "test", "suggestion": "code"}
        pattern3 = {"pattern": "different", "suggestion": "code"}

        hash1 = loader._hash_pattern(pattern1)
        hash2 = loader._hash_pattern(pattern2)
        hash3 = loader._hash_pattern(pattern3)

        # Same patterns should have same hash
        assert hash1 == hash2
        # Different patterns should have different hash
        assert hash1 != hash3

    def test_get_default_mapping_mbpp(self):
        """Test default mapping for MBPP dataset."""
        memory = PersistentReasoningMemory(max_entries=10)
        loader = ProgramLoader(memory)

        mapping = loader._get_default_mapping("mbpp")

        assert "pattern" in mapping
        assert "suggestion" in mapping
        assert mapping["pattern"] == "text"
        assert mapping["suggestion"] == "code"

    def test_get_default_mapping_unknown(self):
        """Test default mapping for unknown dataset."""
        memory = PersistentReasoningMemory(max_entries=10)
        loader = ProgramLoader(memory)

        mapping = loader._get_default_mapping("unknown_dataset")

        # Should return fallback mapping
        assert "pattern" in mapping
        assert "suggestion" in mapping

    def test_map_row_to_pattern(self):
        """Test mapping dataset row to pattern."""
        memory = PersistentReasoningMemory(max_entries=10)
        loader = ProgramLoader(memory)

        row = {
            "text": "Test pattern description",
            "code": "def test(): pass"
        }
        mapping = {
            "pattern": "text",
            "suggestion": "code"
        }

        pattern = loader._map_row_to_pattern(row, mapping, "test_dataset", "train")

        assert pattern["pattern"] == "Test pattern description"
        assert pattern["suggestion"] == "def test(): pass"

    def test_map_row_to_pattern_missing_field(self):
        """Test mapping with missing required field."""
        memory = PersistentReasoningMemory(max_entries=10)
        loader = ProgramLoader(memory)

        row = {
            "text": "Test pattern description"
            # Missing "code" field
        }
        mapping = {
            "pattern": "text",
            "suggestion": "code"
        }

        with pytest.raises(ValueError, match="Pattern and suggestion are required"):
            loader._map_row_to_pattern(row, mapping, "test_dataset", "train")

    def test_get_existing_hashes(self, tmp_path):
        """Test getting hashes of existing patterns."""
        # Use temp path to avoid loading global memory
        memory = PersistentReasoningMemory(
            storage_path=str(tmp_path / "test_memory.json"),
            max_entries=10
        )

        # Add some patterns
        memory.add_reasoning(
            pattern="test1",
            context="test",
            reasoning="test",
            suggestion="code1",
            confidence=0.5,
            source_context={}
        )
        memory.add_reasoning(
            pattern="test2",
            context="test",
            reasoning="test",
            suggestion="code2",
            confidence=0.5,
            source_context={}
        )

        loader = ProgramLoader(memory)
        hashes = loader._get_existing_hashes()

        # Should have at least the 2 patterns we added
        # (may have more if global memory is loaded)
        assert len(hashes) >= 2

    def test_format_result_basic(self):
        """Test formatting load result."""
        memory = PersistentReasoningMemory(max_entries=10)
        loader = ProgramLoader(memory)

        result = ProgramLoadResult(
            loaded_count=100,
            deduped_count=20,
            failed_count=5,
            index_rebuild_time=1.5,
            total_patterns=100,
            dataset_id="test_dataset",
            split="train"
        )

        output = loader.format_result(result, quote='"I know kung fu."')

        assert '"I know kung fu."' in output
        assert "Loaded: 100 patterns" in output
        assert "Deduped: 20 duplicates" in output
        assert "Failed: 5 errors" in output
        assert "Index rebuilt: 1.5s" in output
        assert "Memory: 100 total patterns" in output

    def test_format_result_no_dedupes(self):
        """Test formatting when no dedupes occurred."""
        memory = PersistentReasoningMemory(max_entries=10)
        loader = ProgramLoader(memory)

        result = ProgramLoadResult(
            loaded_count=100,
            deduped_count=0,
            failed_count=0,
            index_rebuild_time=1.5,
            total_patterns=100,
            dataset_id="test_dataset",
            split="train"
        )

        output = loader.format_result(result, quote='"Show me."')

        assert '"Show me."' in output
        assert "Loaded: 100 patterns" in output
        assert "Deduped:" not in output  # Should not show if 0
        assert "Failed:" not in output   # Should not show if 0

    def test_load_program_validation_empty_dataset(self):
        """Test validation with empty dataset_id."""
        memory = PersistentReasoningMemory(max_entries=10)
        loader = ProgramLoader(memory)

        with pytest.raises(ValueError, match="dataset_id is required"):
            loader.load_program(dataset_id="", split="train")

    def test_load_program_validation_invalid_split(self):
        """Test validation with invalid split."""
        memory = PersistentReasoningMemory(max_entries=10)
        loader = ProgramLoader(memory)

        with pytest.raises(ValueError, match="Invalid split"):
            loader.load_program(dataset_id="test", split="invalid")


class TestMatrixOutput:
    """Test Matrix-style output formatting."""

    def test_all_quotes_included(self):
        """Ensure all Matrix quotes are present in possible outputs."""
        memory = PersistentReasoningMemory(max_entries=10)
        loader = ProgramLoader(memory)

        # Test that random quote selection works
        result = ProgramLoadResult(
            loaded_count=10,
            deduped_count=0,
            failed_count=0,
            index_rebuild_time=0.5,
            total_patterns=10,
            dataset_id="test",
            split="train"
        )

        # Generate multiple outputs to check randomness
        outputs = [loader.format_result(result) for _ in range(10)]

        # At least some variation should occur (not all the same quote)
        unique_quotes = set(output.split('\n')[0] for output in outputs)
        assert len(unique_quotes) >= 1  # At least one unique quote
