#!/usr/bin/env python3
"""
Test Phase 5: Strategy Evolution Tracking

Tests the strategy level inference and difficulty-aware retrieval boost.
"""

import pytest
from neo.persistent_reasoning import PersistentReasoningMemory, ReasoningEntry


class TestStrategyInference:
    """Test strategy level inference from difficulty_affinity (Phase 5)."""

    def test_infer_compositional_strategy(self):
        """Test that high hard performance + merge_count → compositional."""
        memory = PersistentReasoningMemory()

        entry = ReasoningEntry(
            pattern="segment tree + binary search",
            context="range query optimization",
            reasoning="combine data structures",
            suggestion="build segment tree, binary search on ranges",
            confidence=0.8,
            source_hash="hash1"
        )
        # High hard success + evolved through consolidation
        entry.difficulty_affinity = {
            "easy": (8, 10),    # 80% easy
            "medium": (7, 10),  # 70% medium
            "hard": (6, 10)     # 60% hard
        }
        entry.merge_count = 4  # Evolved

        strategy = memory._infer_strategy_level(entry)
        assert strategy == "compositional"

    def test_infer_procedural_strategy(self):
        """Test that high easy, low hard → procedural."""
        memory = PersistentReasoningMemory()

        entry = ReasoningEntry(
            pattern="brute force nested loops",
            context="find all pairs",
            reasoning="check all combinations",
            suggestion="for i in range(n): for j in range(n)",
            confidence=0.5,
            source_hash="hash2"
        )
        # High easy, low hard
        entry.difficulty_affinity = {
            "easy": (9, 10),    # 90% easy
            "medium": (5, 10),  # 50% medium
            "hard": (2, 10)     # 20% hard
        }
        entry.merge_count = 1  # Not evolved

        strategy = memory._infer_strategy_level(entry)
        assert strategy == "procedural"

    def test_infer_adaptive_strategy(self):
        """Test that balanced performance → adaptive."""
        memory = PersistentReasoningMemory()

        entry = ReasoningEntry(
            pattern="hashmap for O(1) lookup",
            context="frequency counting",
            reasoning="use dict for constant time access",
            suggestion="freq = {}; for x in arr: freq[x] = freq.get(x, 0) + 1",
            confidence=0.7,
            source_hash="hash3"
        )
        # Balanced across difficulties
        entry.difficulty_affinity = {
            "easy": (7, 10),    # 70% easy
            "medium": (6, 10),  # 60% medium
            "hard": (5, 10)     # 50% hard
        }
        entry.merge_count = 2

        strategy = memory._infer_strategy_level(entry)
        assert strategy == "adaptive"

    def test_infer_strategy_no_data(self):
        """Test that no difficulty_affinity → adaptive (default)."""
        memory = PersistentReasoningMemory()

        entry = ReasoningEntry(
            pattern="new pattern",
            context="no history",
            reasoning="untested",
            suggestion="unknown",
            confidence=0.5,
            source_hash="hash4"
        )
        # No difficulty data

        strategy = memory._infer_strategy_level(entry)
        assert strategy == "adaptive"  # Default


class TestStrategyBoost:
    """Test strategy-difficulty matching boost calculation (Phase 5)."""

    def test_hard_problem_prefers_compositional(self):
        """Test that hard problems boost compositional strategies."""
        memory = PersistentReasoningMemory()

        boost = memory._calculate_strategy_boost("compositional", "hard")
        assert boost == 0.15  # Positive boost

        boost = memory._calculate_strategy_boost("procedural", "hard")
        assert boost == -0.10  # Penalty

        boost = memory._calculate_strategy_boost("adaptive", "hard")
        assert boost == 0.05  # Slight boost

    def test_easy_problem_accepts_procedural(self):
        """Test that easy problems don't penalize procedural."""
        memory = PersistentReasoningMemory()

        boost = memory._calculate_strategy_boost("procedural", "easy")
        assert boost == 0.0  # No penalty

        boost = memory._calculate_strategy_boost("adaptive", "easy")
        assert boost == 0.05  # Slight preference

    def test_medium_problem_prefers_adaptive(self):
        """Test that medium problems prefer adaptive strategies."""
        memory = PersistentReasoningMemory()

        boost = memory._calculate_strategy_boost("adaptive", "medium")
        assert boost == 0.10  # Positive boost

        boost = memory._calculate_strategy_boost("procedural", "medium")
        assert boost == 0.0  # Neutral

        boost = memory._calculate_strategy_boost("compositional", "medium")
        assert boost == 0.0  # Neutral


class TestStrategyAwareRetrieval:
    """Test that retrieval applies strategy boost correctly (Phase 5)."""

    def test_hard_problem_retrieves_compositional_first(self):
        """Test that hard problems rank compositional strategies higher."""
        memory = PersistentReasoningMemory()

        # Compositional strategy (good on hard)
        entry1 = ReasoningEntry(
            pattern="advanced dp + memoization",
            context="optimization problem",
            reasoning="dynamic programming with state compression",
            suggestion="dp[i][mask] = ...",
            confidence=0.7,
            source_hash="hash1"
        )
        entry1.difficulty_affinity = {
            "easy": (7, 10),
            "hard": (8, 10)  # 80% hard
        }
        entry1.merge_count = 5  # Highly evolved

        # Procedural strategy (bad on hard)
        entry2 = ReasoningEntry(
            pattern="brute force enumeration",
            context="try all possibilities",
            reasoning="enumerate all combinations",
            suggestion="for i in range(2**n): ...",
            confidence=0.7,  # Same base confidence
            source_hash="hash2"
        )
        entry2.difficulty_affinity = {
            "easy": (9, 10),  # 90% easy
            "hard": (2, 10)   # 20% hard
        }
        entry2.merge_count = 1

        memory.entries = [entry1, entry2]

        # Generate embeddings for retrieval
        import numpy as np
        entry1.embedding = np.random.rand(768).astype(np.float32)  # Mock embedding
        entry2.embedding = np.random.rand(768).astype(np.float32)  # Mock embedding

        # Retrieve for hard problem
        results = memory.retrieve_relevant(
            problem_context={
                "prompt": "optimize for large input n=10^6",
                "difficulty": "hard"
            },
            k=2
        )

        # Compositional should rank first
        # (gets +0.15 boost, procedural gets -0.10 penalty)
        if len(results) >= 1:
            # First result should be compositional (entry1)
            assert "dp" in results[0].pattern.lower() or "advanced" in results[0].pattern.lower()

    def test_easy_problem_accepts_procedural(self):
        """Test that easy problems don't penalize procedural strategies."""
        memory = PersistentReasoningMemory()

        # Procedural strategy
        entry1 = ReasoningEntry(
            pattern="simple linear search",
            context="find element",
            reasoning="iterate through array",
            suggestion="for x in arr: if x == target: return True",
            confidence=0.6,
            source_hash="hash1"
        )
        entry1.difficulty_affinity = {
            "easy": (9, 10),   # 90% easy
            "hard": (1, 10)    # 10% hard
        }

        memory.entries = [entry1]

        # Generate embedding for retrieval
        import numpy as np
        # Create similar embeddings so cosine similarity is high
        base_embedding = np.random.rand(768).astype(np.float32)
        entry1.embedding = base_embedding

        # Mock the _embed_text to return similar embedding for query
        original_embed = memory._embed_text
        memory._embed_text = lambda x: base_embedding + np.random.rand(768).astype(np.float32) * 0.1

        # Retrieve for easy problem
        try:
            results = memory.retrieve_relevant(
                problem_context={
                    "prompt": "find element in small array n=10",
                    "difficulty": "easy"
                },
                k=1
            )

            # Should retrieve procedural without penalty
            assert len(results) >= 1
        finally:
            # Restore original method
            memory._embed_text = original_embed


class TestGetSuccessRate:
    """Test success rate extraction utility."""

    def test_get_success_rate_normal(self):
        """Test normal success rate calculation."""
        from neo.persistent_reasoning import PersistentReasoningMemory

        rate = PersistentReasoningMemory._get_success_rate((7, 10))
        assert rate == 0.7

    def test_get_success_rate_none(self):
        """Test that None returns neutral 0.5."""
        from neo.persistent_reasoning import PersistentReasoningMemory

        rate = PersistentReasoningMemory._get_success_rate(None)
        assert rate == 0.5

    def test_get_success_rate_zero_total(self):
        """Test that zero total returns neutral 0.5."""
        from neo.persistent_reasoning import PersistentReasoningMemory

        rate = PersistentReasoningMemory._get_success_rate((0, 0))
        assert rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
