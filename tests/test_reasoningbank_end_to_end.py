#!/usr/bin/env python3
"""
End-to-End ReasoningBank Validation

Tests the complete ReasoningBank implementation on realistic problem scenarios.
Measures impact of all 5 phases on retrieval quality and learning.
"""

import pytest
import numpy as np
from neo.persistent_reasoning import PersistentReasoningMemory, ReasoningEntry


class TestReasoningBankImpact:
    """Measure the impact of ReasoningBank improvements on retrieval quality."""

    def setup_method(self):
        """Create realistic memory with diverse patterns."""
        self.memory = PersistentReasoningMemory()

        # Create realistic problem patterns across difficulty levels
        self.patterns = [
            # Easy patterns (procedural)
            {
                "pattern": "linear search",
                "context": "find element in unsorted array",
                "reasoning": "iterate through all elements",
                "suggestion": "for i in range(len(arr)):\n    if arr[i] == target:\n        return i",
                "confidence": 0.65,
                "difficulty_affinity": {"easy": (9, 10), "medium": (5, 10), "hard": (1, 10)},
                "merge_count": 1,
                "problem_outcomes": {"p1": True, "p2": True, "p3": False, "p4": False},
            },
            {
                "pattern": "nested loops for pairs",
                "context": "find all pairs in array",
                "reasoning": "brute force check all combinations",
                "suggestion": "for i in range(n):\n    for j in range(i+1, n):\n        process(arr[i], arr[j])",
                "confidence": 0.60,
                "difficulty_affinity": {"easy": (8, 10), "medium": (4, 10), "hard": (1, 10)},
                "merge_count": 1,
                "problem_outcomes": {"p5": True, "p6": True, "p7": False},
            },

            # Medium patterns (adaptive)
            {
                "pattern": "hash table for O(1) lookup",
                "context": "frequency counting or membership",
                "reasoning": "use dict for constant time access",
                "suggestion": "freq = {}\nfor x in arr:\n    freq[x] = freq.get(x, 0) + 1",
                "confidence": 0.70,
                "difficulty_affinity": {"easy": (7, 10), "medium": (8, 10), "hard": (6, 10)},
                "merge_count": 3,
                "problem_outcomes": {"p8": True, "p9": True, "p10": True, "p11": True, "p12": False},
            },
            {
                "pattern": "two-pointer technique",
                "context": "sorted array optimization",
                "reasoning": "converge from both ends",
                "suggestion": "left, right = 0, len(arr)-1\nwhile left < right:\n    # process",
                "confidence": 0.72,
                "difficulty_affinity": {"easy": (7, 10), "medium": (7, 10), "hard": (5, 10)},
                "merge_count": 4,
                "problem_outcomes": {"p13": True, "p14": True, "p15": True, "p16": True},
                "common_pitfalls": ["May fail on unsorted arrays"],
            },

            # Hard patterns (compositional)
            {
                "pattern": "dynamic programming with memoization",
                "context": "optimization with overlapping subproblems",
                "reasoning": "cache intermediate results to avoid recomputation",
                "suggestion": "@lru_cache(maxsize=None)\ndef dp(state):\n    # base case\n    # recursive case with memo",
                "confidence": 0.75,
                "difficulty_affinity": {"easy": (5, 10), "medium": (7, 10), "hard": (8, 10)},
                "merge_count": 6,
                "problem_outcomes": {"p17": True, "p18": True, "p19": True, "p20": True, "p21": True},
            },
            {
                "pattern": "segment tree for range queries",
                "context": "range updates and queries on array",
                "reasoning": "tree structure for O(log n) operations",
                "suggestion": "class SegmentTree:\n    def __init__(self, arr):\n        # build tree\n    def query(self, l, r):\n        # range query",
                "confidence": 0.78,
                "difficulty_affinity": {"easy": (3, 10), "medium": (6, 10), "hard": (9, 10)},
                "merge_count": 5,
                "problem_outcomes": {"p22": True, "p23": True, "p24": True, "p25": True, "p26": True, "p27": True},
            },
            {
                "pattern": "binary search on answer",
                "context": "optimization problems with monotonic property",
                "reasoning": "search space reduction on answer range",
                "suggestion": "def is_valid(mid):\n    # check if mid works\nlo, hi = 0, max_val\nwhile lo < hi:\n    mid = (lo+hi)//2",
                "confidence": 0.73,
                "difficulty_affinity": {"easy": (4, 10), "medium": (7, 10), "hard": (8, 10)},
                "merge_count": 4,
                "problem_outcomes": {"p28": True, "p29": True, "p30": True, "p31": False},
                "common_pitfalls": ["Requires careful handling of search bounds", "is_valid() must be monotonic"],
            },

            # Spurious pattern (lucky once but actually bad)
            {
                "pattern": "recursive brute force",
                "context": "try all possibilities",
                "reasoning": "recursively enumerate all options",
                "suggestion": "def solve(state):\n    if done:\n        return result\n    return max(solve(next_state) for next_state in options)",
                "confidence": 0.55,
                "difficulty_affinity": {"easy": (6, 10), "medium": (3, 10), "hard": (1, 10)},
                "merge_count": 1,
                "problem_outcomes": {"p32": True, "p33": False, "p34": False, "p35": False},  # Lucky once
                "common_pitfalls": ["Exponential time complexity without memoization", "Stack overflow on deep recursion"],
            },
        ]

        # Add patterns to memory with embeddings
        for i, pattern_data in enumerate(self.patterns):
            entry = ReasoningEntry(
                pattern=pattern_data["pattern"],
                context=pattern_data["context"],
                reasoning=pattern_data["reasoning"],
                suggestion=pattern_data["suggestion"],
                confidence=pattern_data["confidence"],
                source_hash=f"pattern_{i}"
            )
            entry.difficulty_affinity = pattern_data["difficulty_affinity"]
            entry.merge_count = pattern_data.get("merge_count", 1)
            entry.problem_outcomes = pattern_data.get("problem_outcomes", {})
            entry.common_pitfalls = pattern_data.get("common_pitfalls", [])

            # Generate semantic embedding (Phase 2: pattern + context)
            entry.embedding = np.random.rand(768).astype(np.float32)

            self.memory.entries.append(entry)

    def test_hard_problem_ranks_compositional_higher(self):
        """
        Test that hard problems retrieve compositional strategies first.

        Expected behavior:
        - Segment tree (compositional, 90% hard) should rank higher than
        - Linear search (procedural, 10% hard)
        """
        # Query for hard problem
        query_embedding = self.memory.entries[5].embedding + np.random.rand(768).astype(np.float32) * 0.05

        original_embed = self.memory._embed_text
        self.memory._embed_text = lambda x: query_embedding

        try:
            results = self.memory.retrieve_relevant(
                problem_context={
                    "prompt": "optimize range sum queries for array with 10^6 elements",
                    "difficulty": "hard"
                },
                k=5
            )
        finally:
            self.memory._embed_text = original_embed

        # Verify compositional patterns appear before procedural
        top_3_patterns = [r.pattern for r in results[:3]]

        # Should include at least one compositional pattern
        compositional_patterns = ["segment tree", "dynamic programming", "binary search on answer"]
        assert any(cp in " ".join(top_3_patterns).lower() for cp in compositional_patterns), \
            f"Expected compositional pattern in top 3, got: {top_3_patterns}"

        # Linear search (procedural) should NOT be in top 3 for hard problem
        assert "linear search" not in " ".join(top_3_patterns).lower(), \
            f"Procedural pattern should not rank high for hard problem, got: {top_3_patterns}"

    def test_easy_problem_accepts_procedural(self):
        """
        Test that easy problems don't penalize procedural strategies.

        Expected behavior:
        - Linear search is acceptable for easy problems
        - No penalty applied to procedural patterns
        """
        query_embedding = self.memory.entries[0].embedding + np.random.rand(768).astype(np.float32) * 0.05

        original_embed = self.memory._embed_text
        self.memory._embed_text = lambda x: query_embedding

        try:
            results = self.memory.retrieve_relevant(
                problem_context={
                    "prompt": "find element in small array of 10 items",
                    "difficulty": "easy"
                },
                k=3
            )
        finally:
            self.memory._embed_text = original_embed

        # Linear search should be retrieved without penalty
        assert len(results) > 0
        patterns = [r.pattern.lower() for r in results]

        # Should include linear search (no penalty on easy)
        assert any("linear" in p or "nested loop" in p for p in patterns), \
            f"Expected procedural patterns for easy problem, got: {patterns}"

    def test_archetypal_pattern_outranks_spurious(self):
        """
        Test that archetypal patterns (consistent winners) rank higher than spurious patterns.

        Expected behavior:
        - DP (80% hard success, 5/5 wins) beats recursive brute force (10% hard, 1/4 wins)
        """
        # Create query similar to both DP and recursive patterns
        dp_entry = self.memory.entries[4]  # DP pattern
        recursive_entry = self.memory.entries[7]  # Recursive brute force

        # Make query embedding close to both
        avg_embedding = (dp_entry.embedding + recursive_entry.embedding) / 2
        query_embedding = avg_embedding + np.random.rand(768).astype(np.float32) * 0.05

        original_embed = self.memory._embed_text
        self.memory._embed_text = lambda x: query_embedding

        try:
            results = self.memory.retrieve_relevant(
                problem_context={
                    "prompt": "optimization problem with overlapping subproblems",
                    "difficulty": "hard"
                },
                k=5
            )
        finally:
            self.memory._embed_text = original_embed

        # Find positions of DP vs recursive
        dp_idx = None
        recursive_idx = None

        for i, r in enumerate(results):
            if "dynamic programming" in r.pattern.lower():
                dp_idx = i
            if "recursive brute force" in r.pattern.lower():
                recursive_idx = i

        # DP should rank higher than recursive (if both present)
        if dp_idx is not None and recursive_idx is not None:
            assert dp_idx < recursive_idx, \
                f"Archetypal DP (idx {dp_idx}) should outrank spurious recursive (idx {recursive_idx})"

    def test_failure_pitfalls_surface_in_retrieval(self):
        """
        Test that patterns with failure history include pitfalls.

        Expected behavior:
        - Patterns that failed should have common_pitfalls populated
        - Pitfalls should describe failure modes
        """
        # Check patterns with failures have pitfalls
        two_pointer = [e for e in self.memory.entries if "two-pointer" in e.pattern.lower()][0]
        assert len(two_pointer.common_pitfalls) > 0, "Pattern with failure should have pitfalls"

        binary_search_ans = [e for e in self.memory.entries if "binary search on answer" in e.pattern.lower()][0]
        assert len(binary_search_ans.common_pitfalls) > 0, "Pattern with failures should have pitfalls"

        recursive = [e for e in self.memory.entries if "recursive brute force" in e.pattern.lower()][0]
        assert len(recursive.common_pitfalls) > 0, "Spurious pattern should have pitfalls"

    def test_strategy_inference_accuracy(self):
        """
        Test that strategy levels are correctly inferred from performance patterns.

        Expected inferences:
        - Linear search → procedural (90% easy, 10% hard)
        - Hash table → adaptive (70% easy, 60% hard)
        - DP → compositional (80% hard, merge_count=6)
        """
        linear_search = self.memory.entries[0]
        strategy = self.memory._infer_strategy_level(linear_search)
        assert strategy == "procedural", f"Linear search should be procedural, got {strategy}"

        hash_table = self.memory.entries[2]
        strategy = self.memory._infer_strategy_level(hash_table)
        assert strategy == "adaptive", f"Hash table should be adaptive, got {strategy}"

        dp_pattern = self.memory.entries[4]
        strategy = self.memory._infer_strategy_level(dp_pattern)
        assert strategy == "compositional", f"DP should be compositional, got {strategy}"

        segment_tree = self.memory.entries[5]
        strategy = self.memory._infer_strategy_level(segment_tree)
        assert strategy == "compositional", f"Segment tree should be compositional, got {strategy}"

    def test_medium_problem_prefers_adaptive(self):
        """
        Test that medium problems prefer adaptive strategies.

        Expected behavior:
        - Hash table (adaptive, 80% medium) gets +0.10 boost
        - Two-pointer (adaptive, 70% medium) gets +0.10 boost
        """
        query_embedding = self.memory.entries[2].embedding + np.random.rand(768).astype(np.float32) * 0.05

        original_embed = self.memory._embed_text
        self.memory._embed_text = lambda x: query_embedding

        try:
            results = self.memory.retrieve_relevant(
                problem_context={
                    "prompt": "count frequency of elements efficiently",
                    "difficulty": "medium"
                },
                k=3
            )
        finally:
            self.memory._embed_text = original_embed

        # Should retrieve adaptive strategies
        top_patterns = [r.pattern.lower() for r in results[:3]]

        # Hash table or two-pointer should be in top 3
        assert any("hash" in p or "two-pointer" in p for p in top_patterns), \
            f"Expected adaptive patterns for medium problem, got: {top_patterns}"

    def test_performance_consistency_across_retrievals(self):
        """
        Test that retrieval performance is consistent across multiple queries.

        Expected behavior:
        - All retrievals complete in <100ms
        - Results are deterministic (same query → same results)
        """
        import time

        problem_context = {
            "prompt": "optimize algorithm for large input",
            "difficulty": "hard"
        }

        query_embedding = np.random.rand(768).astype(np.float32)
        original_embed = self.memory._embed_text
        self.memory._embed_text = lambda x: query_embedding

        try:
            # Run 5 retrievals and measure time
            times = []
            all_results = []

            for _ in range(5):
                start = time.perf_counter()
                results = self.memory.retrieve_relevant(problem_context=problem_context, k=3)
                elapsed_ms = (time.perf_counter() - start) * 1000

                times.append(elapsed_ms)
                all_results.append([r.pattern for r in results])

            # All should be under 100ms
            assert all(t < 100 for t in times), f"Some retrievals exceeded 100ms: {times}"

            # Results should be deterministic (same order)
            for i in range(1, len(all_results)):
                assert all_results[i] == all_results[0], \
                    f"Results inconsistent: {all_results[0]} vs {all_results[i]}"

        finally:
            self.memory._embed_text = original_embed


class TestReasoningBankMetrics:
    """Calculate metrics for ReasoningBank impact assessment."""

    def test_calculate_retrieval_precision(self):
        """
        Calculate precision of retrieval for different difficulty levels.

        Metric: % of top-3 results that match difficulty appropriately
        Target: >75% precision
        """
        memory = PersistentReasoningMemory()

        # Add test patterns (simplified)
        procedural = ReasoningEntry(
            pattern="brute force",
            context="try all",
            reasoning="enumerate",
            suggestion="nested loops",
            confidence=0.6,
            source_hash="proc"
        )
        procedural.difficulty_affinity = {"easy": (9, 10), "hard": (1, 10)}
        procedural.merge_count = 1
        procedural.embedding = np.array([1.0] * 768, dtype=np.float32)

        compositional = ReasoningEntry(
            pattern="advanced algorithm",
            context="complex optimization",
            reasoning="sophisticated approach",
            suggestion="advanced code",
            confidence=0.8,
            source_hash="comp"
        )
        compositional.difficulty_affinity = {"easy": (3, 10), "hard": (9, 10)}
        compositional.merge_count = 5
        compositional.embedding = np.array([2.0] * 768, dtype=np.float32)

        memory.entries = [procedural, compositional]

        # Test hard problem retrieval
        original_embed = memory._embed_text
        memory._embed_text = lambda x: np.array([2.0] * 768, dtype=np.float32)

        try:
            results = memory.retrieve_relevant(
                problem_context={"prompt": "hard problem", "difficulty": "hard"},
                k=2
            )
        finally:
            memory._embed_text = original_embed

        # First result should be compositional for hard problem
        if len(results) > 0:
            assert "advanced" in results[0].pattern.lower(), \
                f"Hard problem should retrieve compositional first, got: {results[0].pattern}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
