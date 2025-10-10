#!/usr/bin/env python3
"""
Integration Tests: All ReasoningBank Phases Working Together

Tests the complete pipeline with all 5 phases enabled:
- Phase 2: Semantic Anchor Embedding
- Phase 3: Systematic Failure Learning
- Phase 4: Self-Contrast Consolidation
- Phase 5: Strategy Evolution Tracking
"""

import pytest
import time
import numpy as np
from neo.persistent_reasoning import PersistentReasoningMemory, ReasoningEntry


class TestFullPipelineIntegration:
    """Test all phases working together in realistic scenarios."""

    def test_hard_problem_prefers_evolved_compositional_patterns(self):
        """
        End-to-end test: Hard problem should retrieve compositional strategy
        that evolved through consolidation and has contrastive advantage.

        Tests integration of:
        - Phase 4: Contrastive boost (archetypal vs spurious)
        - Phase 5: Strategy boost (compositional preferred for hard)
        """
        memory = PersistentReasoningMemory()

        # Create archetypal compositional pattern (evolved, good on hard)
        compositional = ReasoningEntry(
            pattern="segment tree + binary search",
            context="range query with updates",
            reasoning="combine data structures for O(log n) operations",
            suggestion="build segment tree, binary search on ranges",
            confidence=0.65,
            source_hash="comp_hash"
        )
        compositional.difficulty_affinity = {
            "easy": (7, 10),    # 70% easy
            "medium": (8, 10),  # 80% medium
            "hard": (8, 10)     # 80% hard (Phase 5: compositional)
        }
        compositional.merge_count = 5  # Highly evolved (Phase 5)
        compositional.problem_outcomes = {
            "prob_1": True,
            "prob_2": True,
            "prob_3": True,
        }  # Phase 4: archetypal (consistent winner)

        # Create procedural pattern (not evolved, fails on hard)
        procedural = ReasoningEntry(
            pattern="brute force nested loops",
            context="check all pairs",
            reasoning="enumerate all combinations",
            suggestion="for i in range(n): for j in range(n): ...",
            confidence=0.60,  # Similar base confidence
            source_hash="proc_hash"
        )
        procedural.difficulty_affinity = {
            "easy": (9, 10),    # 90% easy
            "medium": (5, 10),  # 50% medium
            "hard": (2, 10)     # 20% hard (Phase 5: procedural)
        }
        procedural.merge_count = 1  # Not evolved
        procedural.problem_outcomes = {
            "prob_1": False,
            "prob_2": False,
            "prob_3": False,
        }  # Phase 4: spurious (consistent loser)

        memory.entries = [compositional, procedural]

        # Generate embeddings (Phase 2: semantic anchor = pattern + context)
        base_embedding = np.random.rand(768).astype(np.float32)
        compositional.embedding = base_embedding.copy()
        procedural.embedding = base_embedding + np.random.rand(768).astype(np.float32) * 0.1

        # Mock _embed_text to return similar embedding for query
        original_embed = memory._embed_text
        memory._embed_text = lambda x: base_embedding + np.random.rand(768).astype(np.float32) * 0.05

        try:
            # Retrieve for hard problem
            results = memory.retrieve_relevant(
                problem_context={
                    "prompt": "optimize range query for n=10^6",
                    "difficulty": "hard"
                },
                k=2
            )
        finally:
            # Restore original method
            memory._embed_text = original_embed

        # Verify compositional ranks first
        # Expected boosts:
        # - Compositional: +0.2 (contrastive: 3/3 wins) +0.15 (strategy: compositional on hard)
        # - Procedural: -0.2 (contrastive: 0/3 wins) -0.10 (strategy: procedural on hard)
        assert len(results) >= 1
        assert "segment tree" in results[0].pattern.lower() or "binary search" in results[0].pattern.lower()

    def test_failure_learning_surfaces_pitfalls(self):
        """
        Test that failures are extracted and surfaced during retrieval.

        Tests integration of:
        - Phase 3: Failure extraction when confidence < 0.5
        - Pitfalls stored in common_pitfalls field
        """
        memory = PersistentReasoningMemory()

        # Simulate adding a low-confidence pattern with error
        context_with_error = {
            "prompt": "find shortest path",
            "error_trace": "TimeoutError: execution exceeded limit on input size n=10^6"
        }

        memory.add_reasoning(
            pattern="dijkstra with array",
            context="graph shortest path",
            reasoning="use priority queue",
            suggestion="dist = [inf] * n; for ...",
            confidence=0.4,  # Low confidence triggers Phase 3
            source_context=context_with_error
        )

        # Retrieve the entry
        entries = [e for e in memory.entries if "dijkstra" in e.pattern.lower()]
        assert len(entries) > 0

        entry = entries[0]

        # Phase 3: Should have extracted failure cause
        assert len(entry.common_pitfalls) > 0
        # Should mention timeout or performance
        pitfall_text = " ".join(entry.common_pitfalls).lower()
        assert "timeout" in pitfall_text or "performance" in pitfall_text or "complexity" in pitfall_text

    def test_self_contrast_consolidation_boosts_archetype(self):
        """
        Test that consolidation applies contrastive boosts before merging.

        Tests integration of:
        - Phase 4: Contrastive boost calculation
        - Consolidation selects archetypal pattern as base
        """
        memory = PersistentReasoningMemory()

        # Archetypal: wins consistently
        arch = ReasoningEntry(
            pattern="two-pointer technique",
            context="sorted array optimization",
            reasoning="converge from both ends",
            suggestion="left=0, right=len(arr)-1",
            confidence=0.60,
            source_hash="arch_hash"
        )
        arch.problem_outcomes = {
            "p1": True,
            "p2": True,
            "p3": True,
            "p4": True,
        }

        # Spurious: lucky once, fails otherwise
        spur = ReasoningEntry(
            pattern="two pointers",
            context="sorted array",
            reasoning="use two indices",
            suggestion="i=0, j=n-1",
            confidence=0.58,  # Close to archetypal
            source_hash="spur_hash"
        )
        spur.problem_outcomes = {
            "p1": False,
            "p2": False,
            "p3": True,   # Lucky once
            "p4": False,
        }

        cluster = [arch, spur]

        # Merge cluster (should apply Phase 4 boosts)
        merged = memory._merge_cluster(cluster)

        # Archetypal should be selected as base
        # After boost: arch gets +0.2, spur gets -0.15
        # So arch (0.80) should beat spur (0.43)
        assert "technique" in merged.pattern.lower()  # Archetypal pattern selected
        assert merged.merge_count == 2

    def test_strategy_inference_from_performance_patterns(self):
        """
        Test that strategy level is correctly inferred from difficulty affinity.

        Tests Phase 5 heuristics:
        - High hard + evolved → compositional
        - High easy, low hard → procedural
        - Balanced → adaptive
        """
        memory = PersistentReasoningMemory()

        # Test compositional inference
        comp_entry = ReasoningEntry(
            pattern="dp + memoization",
            context="optimization problem",
            reasoning="dynamic programming",
            suggestion="cache[i] = ...",
            confidence=0.7,
            source_hash="hash1"
        )
        comp_entry.difficulty_affinity = {
            "easy": (7, 10),
            "hard": (7, 10)  # 70% hard
        }
        comp_entry.merge_count = 4  # Evolved

        strategy = memory._infer_strategy_level(comp_entry)
        assert strategy == "compositional"

        # Test procedural inference
        proc_entry = ReasoningEntry(
            pattern="nested loops",
            context="find pairs",
            reasoning="check all",
            suggestion="for i: for j:",
            confidence=0.5,
            source_hash="hash2"
        )
        proc_entry.difficulty_affinity = {
            "easy": (9, 10),   # 90% easy
            "hard": (1, 10)    # 10% hard
        }

        strategy = memory._infer_strategy_level(proc_entry)
        assert strategy == "procedural"

        # Test adaptive inference
        adapt_entry = ReasoningEntry(
            pattern="hashmap",
            context="lookup",
            reasoning="O(1) access",
            suggestion="dict[key]",
            confidence=0.6,
            source_hash="hash3"
        )
        adapt_entry.difficulty_affinity = {
            "easy": (7, 10),
            "medium": (6, 10),
            "hard": (5, 10)
        }

        strategy = memory._infer_strategy_level(adapt_entry)
        assert strategy == "adaptive"


class TestPerformanceIntegration:
    """Test performance with all phases enabled."""

    def test_retrieval_latency_under_100ms(self):
        """
        Verify retrieval stays under 100ms with all boosts applied.

        Tests:
        - Phase 2: Semantic anchor embedding
        - Phase 4: Contrastive boost calculation
        - Phase 5: Strategy boost calculation
        """
        memory = PersistentReasoningMemory()

        # Create 20 realistic entries
        for i in range(20):
            entry = ReasoningEntry(
                pattern=f"pattern_{i}",
                context=f"context_{i}",
                reasoning=f"reasoning_{i}",
                suggestion=f"suggestion_{i}",
                confidence=0.5 + (i * 0.02),
                source_hash=f"hash_{i}"
            )
            entry.difficulty_affinity = {
                "easy": (7 + i % 3, 10),
                "hard": (4 + i % 4, 10)
            }
            entry.merge_count = i % 5
            entry.problem_outcomes = {
                f"prob_{j}": (i + j) % 2 == 0
                for j in range(3)
            }
            entry.embedding = np.random.rand(768).astype(np.float32)
            memory.entries.append(entry)

        # Measure retrieval time
        start = time.perf_counter()

        results = memory.retrieve_relevant(
            problem_context={
                "prompt": "test query",
                "difficulty": "hard"
            },
            k=5
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete under 100ms (generous target)
        # Note: First run may be slower due to embedding generation
        # In production with pre-embedded entries, should be <50ms
        assert elapsed_ms < 100, f"Retrieval took {elapsed_ms:.1f}ms, expected <100ms"
        assert len(results) > 0

    def test_consolidation_performance_with_all_boosts(self):
        """
        Test that consolidation completes quickly with Phase 4 contrastive boosts.
        """
        memory = PersistentReasoningMemory()

        # Create cluster of 5 similar entries
        entries = []
        for i in range(5):
            entry = ReasoningEntry(
                pattern=f"binary search variant {i}",
                context="sorted array search",
                reasoning="divide and conquer",
                suggestion=f"mid = (left + right) // 2 # variant {i}",
                confidence=0.55 + i * 0.02,
                source_hash=f"bs_hash_{i}"
            )
            entry.problem_outcomes = {
                "prob_1": i < 3,  # First 3 succeed
                "prob_2": i < 2,  # First 2 succeed
                "prob_3": i == 0, # Only first succeeds
            }
            entry.embedding = np.random.rand(768).astype(np.float32)
            entries.append(entry)

        # Measure consolidation time
        start = time.perf_counter()
        merged = memory._merge_cluster(entries)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete quickly (<50ms for 5 entries)
        assert elapsed_ms < 50, f"Consolidation took {elapsed_ms:.1f}ms, expected <50ms"
        assert merged.merge_count == 5


class TestBackwardCompatibility:
    """Test that old entries without new fields still work."""

    def test_old_entry_without_problem_outcomes(self):
        """Old entries without problem_outcomes should default gracefully."""
        memory = PersistentReasoningMemory()

        # Simulate old entry (no problem_outcomes)
        old_entry = ReasoningEntry(
            pattern="old pattern",
            context="old context",
            reasoning="old reasoning",
            suggestion="old suggestion",
            confidence=0.6,
            source_hash="old_hash"
        )
        # No problem_outcomes set

        # Should not crash during contrastive boost calculation
        cluster = [old_entry]
        boost = memory._calculate_contrastive_boost(old_entry, cluster)
        assert boost == 0.0  # No data = no boost

    def test_old_entry_without_difficulty_affinity(self):
        """Old entries without difficulty_affinity should default to adaptive."""
        memory = PersistentReasoningMemory()

        old_entry = ReasoningEntry(
            pattern="old pattern",
            context="old context",
            reasoning="old reasoning",
            suggestion="old suggestion",
            confidence=0.6,
            source_hash="old_hash"
        )
        # No difficulty_affinity set

        # Should infer as adaptive (default)
        strategy = memory._infer_strategy_level(old_entry)
        assert strategy == "adaptive"

        # Should get neutral boost
        boost = memory._calculate_strategy_boost(strategy, "hard")
        assert boost == 0.05  # Adaptive gets +0.05 on hard


class TestRealWorldScenarios:
    """Test realistic usage patterns."""

    def test_learning_from_repeated_failures(self):
        """
        Simulate learning cycle:
        1. Try pattern, fail
        2. Extract failure cause
        3. Try again, fail differently
        4. Accumulate pitfalls
        5. Pattern confidence decreases
        """
        memory = PersistentReasoningMemory()

        # First attempt: timeout failure
        memory.add_reasoning(
            pattern="recursive fibonacci",
            context="compute fib(n)",
            reasoning="recursion",
            suggestion="def fib(n): return fib(n-1) + fib(n-2)",
            confidence=0.45,
            source_context={
                "error_trace": "TimeoutError: fib(40) exceeded time limit"
            }
        )

        entries = [e for e in memory.entries if "fibonacci" in e.pattern.lower()]
        assert len(entries) > 0

        first_entry = entries[0]

        # Should have timeout-related pitfall
        assert len(first_entry.common_pitfalls) > 0
        assert any("timeout" in p.lower() or "exponential" in p.lower()
                   for p in first_entry.common_pitfalls)

        # Record failure outcome
        first_entry.record_difficulty_outcome(
            difficulty="medium",
            success=False,
            problem_hash="fib_problem_1"
        )

        # Check difficulty tracking
        assert "medium" in first_entry.difficulty_affinity
        success, total = first_entry.difficulty_affinity["medium"]
        assert success == 0
        assert total == 1

    def test_pattern_evolution_through_success(self):
        """
        Simulate pattern improving through multiple successes:
        1. Start with medium confidence
        2. Succeed on multiple problems
        3. Merge with similar patterns
        4. Become archetypal
        """
        memory = PersistentReasoningMemory()

        # Create successful pattern
        pattern = ReasoningEntry(
            pattern="sliding window",
            context="substring problem",
            reasoning="maintain window invariant",
            suggestion="left=0; for right in range(n): ...",
            confidence=0.60,
            source_hash="sliding_hash"
        )

        # Record multiple successes
        for i, diff in enumerate(["easy", "medium", "hard", "hard"]):
            pattern.record_difficulty_outcome(
                difficulty=diff,
                success=True,
                problem_hash=f"prob_{i}"
            )

        # Check difficulty affinity improved
        assert pattern.difficulty_affinity["easy"] == (1, 1)
        assert pattern.difficulty_affinity["medium"] == (1, 1)
        assert pattern.difficulty_affinity["hard"] == (2, 2)

        # Check problem outcomes tracked
        assert len(pattern.problem_outcomes) == 4
        assert all(pattern.problem_outcomes.values())  # All successes

        # Infer strategy (should be compositional due to hard success)
        pattern.merge_count = 4  # Simulate evolution
        strategy = memory._infer_strategy_level(pattern)
        assert strategy == "compositional"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
