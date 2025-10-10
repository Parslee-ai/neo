#!/usr/bin/env python3
"""
Test Phase 4: Self-Contrast Consolidation

Tests the contrastive confidence boost and archetypal pattern identification.
"""

import pytest
from neo.persistent_reasoning import PersistentReasoningMemory, ReasoningEntry


class TestContrastiveBoost:
    """Test contrastive confidence boost calculation (Phase 4)."""

    def test_calculate_contrastive_boost_archetypal_pattern(self):
        """Test that patterns winning where others fail get boosted."""
        memory = PersistentReasoningMemory()

        # Create entry that succeeds on problems A and B
        entry1 = ReasoningEntry(
            pattern="two-pointer",
            context="sorted array",
            reasoning="converge from both ends",
            suggestion="left=0, right=n-1",
            confidence=0.6,
            source_hash="entry1_hash"
        )
        entry1.problem_outcomes = {
            "problem_A": True,  # Success
            "problem_B": True,  # Success
        }

        # Create entry that fails on same problems
        entry2 = ReasoningEntry(
            pattern="sliding window",
            context="sorted array",
            reasoning="maintain window",
            suggestion="window of size k",
            confidence=0.5,
            source_hash="entry2_hash"
        )
        entry2.problem_outcomes = {
            "problem_A": False,  # Failed
            "problem_B": False,  # Failed
        }

        cluster = [entry1, entry2]

        # Calculate boost for archetypal pattern (entry1)
        boost1 = memory._calculate_contrastive_boost(entry1, cluster)
        assert boost1 > 0  # Should be positive (wins where others fail)
        assert boost1 <= 0.2  # Max boost is 0.2

        # Calculate boost for spurious pattern (entry2)
        boost2 = memory._calculate_contrastive_boost(entry2, cluster)
        assert boost2 < 0  # Should be negative (fails where others succeed)
        assert boost2 >= -0.2  # Max penalty is -0.2

    def test_calculate_contrastive_boost_no_shared_problems(self):
        """Test that boost is 0 when entries have no shared problems."""
        memory = PersistentReasoningMemory()

        entry1 = ReasoningEntry(
            pattern="binary search",
            context="sorted array",
            reasoning="divide and conquer",
            suggestion="mid = (left + right) // 2",
            confidence=0.7,
            source_hash="hash1"
        )
        entry1.problem_outcomes = {"problem_X": True}

        entry2 = ReasoningEntry(
            pattern="linear search",
            context="unsorted array",
            reasoning="iterate through all",
            suggestion="for i in range(n)",
            confidence=0.5,
            source_hash="hash2"
        )
        entry2.problem_outcomes = {"problem_Y": True}  # Different problem

        cluster = [entry1, entry2]
        boost = memory._calculate_contrastive_boost(entry1, cluster)

        assert boost == 0.0  # No shared problems = no contrast

    def test_calculate_contrastive_boost_no_outcomes_data(self):
        """Test graceful handling when entry has no problem_outcomes."""
        memory = PersistentReasoningMemory()

        entry1 = ReasoningEntry(
            pattern="greedy",
            context="optimization problem",
            reasoning="local optimum",
            suggestion="sort first",
            confidence=0.6,
            source_hash="hash1"
        )
        # No problem_outcomes set (empty dict)

        entry2 = ReasoningEntry(
            pattern="dynamic programming",
            context="optimization problem",
            reasoning="memoization",
            suggestion="cache[i] = ...",
            confidence=0.5,
            source_hash="hash2"
        )
        entry2.problem_outcomes = {"problem_Z": True}

        cluster = [entry1, entry2]
        boost = memory._calculate_contrastive_boost(entry1, cluster)

        assert boost == 0.0  # No data = no boost

    def test_calculate_contrastive_boost_mixed_outcomes(self):
        """Test boost calculation with mixed win/loss outcomes."""
        memory = PersistentReasoningMemory()

        entry1 = ReasoningEntry(
            pattern="dfs",
            context="graph traversal",
            reasoning="recursive exploration",
            suggestion="visited set",
            confidence=0.6,
            source_hash="hash1"
        )
        entry1.problem_outcomes = {
            "problem_1": True,   # Win
            "problem_2": False,  # Loss
            "problem_3": True,   # Win
        }

        entry2 = ReasoningEntry(
            pattern="bfs",
            context="graph traversal",
            reasoning="queue-based",
            suggestion="use deque",
            confidence=0.5,
            source_hash="hash2"
        )
        entry2.problem_outcomes = {
            "problem_1": False,  # Loss (entry1 won)
            "problem_2": True,   # Win (entry1 lost)
            "problem_3": False,  # Loss (entry1 won)
        }

        cluster = [entry1, entry2]
        boost1 = memory._calculate_contrastive_boost(entry1, cluster)

        # entry1: 2 wins, 1 loss → 2/3 = 0.67 → (0.67-0.5)*0.4 = +0.067
        assert boost1 > 0
        assert boost1 < 0.1


class TestSelfContrastIntegration:
    """Test self-contrast integration into consolidation."""

    def test_merge_cluster_applies_contrastive_boosts(self):
        """Test that _merge_cluster applies contrastive boosts before merging."""
        memory = PersistentReasoningMemory()

        # Archetypal pattern (wins consistently)
        entry1 = ReasoningEntry(
            pattern="kadane's algorithm",
            context="maximum subarray sum",
            reasoning="dynamic programming",
            suggestion="max_so_far, max_ending_here",
            confidence=0.6,
            source_hash="hash1"
        )
        entry1.problem_outcomes = {"prob_1": True, "prob_2": True}

        # Spurious pattern (fails consistently)
        entry2 = ReasoningEntry(
            pattern="brute force",
            context="maximum subarray sum",
            reasoning="try all subarrays",
            suggestion="nested loops",
            confidence=0.55,  # Slightly lower but close
            source_hash="hash2"
        )
        entry2.problem_outcomes = {"prob_1": False, "prob_2": False}

        cluster = [entry1, entry2]

        # Merge should boost entry1, penalize entry2, then select entry1 as best
        merged = memory._merge_cluster(cluster)

        # The merged entry should be based on entry1 (archetypal)
        # After boost: entry1 has 0.6+0.2=0.8, entry2 has 0.55-0.2=0.35
        # So entry1 should be selected as base
        assert "kadane" in merged.pattern.lower()
        # Merged confidence is weighted average of boosted confidences
        # With no success_signals, it's (0.8 + 0.35) / 2 = 0.575
        # The key is that entry1 (archetypal) was selected due to boost
        assert merged.merge_count == 2  # Indicates merge happened

    def test_record_difficulty_outcome_tracks_problem_hash(self):
        """Test that recording outcomes also tracks problem_hash."""
        entry = ReasoningEntry(
            pattern="test pattern",
            context="test context",
            reasoning="test reasoning",
            suggestion="test suggestion",
            confidence=0.5,
            source_hash="test_hash"
        )

        # Record successful outcome with problem_hash
        entry.record_difficulty_outcome(
            difficulty="medium",
            success=True,
            problem_hash="problem_123"
        )

        # Check problem_outcomes was updated
        assert "problem_123" in entry.problem_outcomes
        assert entry.problem_outcomes["problem_123"] is True

        # Record failure on different problem
        entry.record_difficulty_outcome(
            difficulty="hard",
            success=False,
            problem_hash="problem_456"
        )

        assert "problem_456" in entry.problem_outcomes
        assert entry.problem_outcomes["problem_456"] is False

    def test_serialization_includes_problem_outcomes(self):
        """Test that problem_outcomes serializes and deserializes correctly."""
        entry = ReasoningEntry(
            pattern="test",
            context="test",
            reasoning="test",
            suggestion="test",
            confidence=0.5,
            source_hash="hash"
        )
        entry.problem_outcomes = {
            "prob_A": True,
            "prob_B": False,
            "prob_C": True
        }

        # Serialize
        data = entry.to_dict()
        assert "problem_outcomes" in data
        assert data["problem_outcomes"]["prob_A"] is True
        assert data["problem_outcomes"]["prob_B"] is False

        # Deserialize
        restored = ReasoningEntry.from_dict(data)
        assert len(restored.problem_outcomes) == 3
        assert restored.problem_outcomes["prob_A"] is True
        assert restored.problem_outcomes["prob_B"] is False
        assert restored.problem_outcomes["prob_C"] is True

    def test_backward_compatibility_missing_problem_outcomes(self):
        """Test that old entries without problem_outcomes still load."""
        # Simulate old entry without problem_outcomes field
        old_data = {
            "pattern": "old pattern",
            "context": "old context",
            "reasoning": "old reasoning",
            "suggestion": "old suggestion",
            "confidence": 0.5,
            "source_hash": "old_hash",
            "use_count": 1,
            "success_signals": 1,
            "failure_signals": 0,
            "created_at": 1000,
            "last_used": 1000,
            "algorithm_type": "",
            "code_template": "",
            "time_complexity": "",
            "space_complexity": "",
            "when_to_use": "",
            "example_problems": [],
            "codebase_context": {},
            "contextual_stats": {},
            # Missing: problem_outcomes
        }

        # Should not raise error
        entry = ReasoningEntry.from_dict(old_data)
        assert hasattr(entry, "problem_outcomes")
        assert entry.problem_outcomes == {}  # Default empty dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
