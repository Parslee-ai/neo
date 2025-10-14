"""
Persistent Reasoning Memory for Neo.

Key constraints:
1. Stateless calls - no direct success/failure feedback
2. Bounded storage - can't grow forever
3. Quality over quantity - only keep what helps
4. Self-improving - better entries replace worse ones

Design philosophy:
- Implicit feedback from context (re-asks mean failure)
- Confidence decay over time (old knowledge gets stale)
- Automatic consolidation (merge similar learnings)
- Competitive replacement (better patterns replace worse)
"""

import copy
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import hashlib
import logging
import os
import numpy as np
from datasketch import MinHash, MinHashLSH
from collections import OrderedDict
from neo.storage import FileStorage
from neo.storage_interface import StorageBackend

# Import FAISS for fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
    # Check for GPU support
    try:
        FAISS_GPU_AVAILABLE = faiss.get_num_gpus() > 0
    except:
        FAISS_GPU_AVAILABLE = False
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    # FAISS not installed - will fall back to O(n²) clustering

# Import OpenAI exception types at module level
try:
    from openai import OpenAIError, RateLimitError, APIError
except ImportError:
    # OpenAI not installed - exceptions won't be needed
    # because openai_client will be None
    pass

# Import fastembed for local embeddings (alternative to OpenAI)
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    # fastembed not installed - will fall back to OpenAI or MinHash

logger = logging.getLogger(__name__)

# Constants for embeddings (Phase 1)
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small dimension
EMBEDDING_CACHE_MAX_SIZE = 500  # LRU cache limit
# Rationale: 500 entries × 1536 floats × 8 bytes = ~6MB memory
# OrderedDict operations are O(1) up to 10K entries (performance OK)
# Typical usage: 2000 memory entries, each potentially embedded multiple times
MAX_TEXT_LENGTH = 32000  # ~8K tokens for OpenAI API
ALLOWED_DIFFICULTIES = frozenset(["easy", "medium", "hard"])


@dataclass
class ReasoningEntry:
    """A single piece of learned reasoning."""

    # Core data
    pattern: str  # What pattern this represents
    context: str  # When to apply it
    reasoning: str  # Why it works
    suggestion: str  # What to do

    # Solution pattern fields (NEW - for algorithmic patterns)
    algorithm_type: str = ""  # e.g., "two-pointer", "sliding-window", "dynamic-programming"
    code_template: str = ""  # Working code snippet/structure
    time_complexity: str = ""  # e.g., "O(n)", "O(n log n)"
    space_complexity: str = ""  # e.g., "O(1)", "O(n)"
    when_to_use: str = ""  # When to apply this pattern
    example_problems: list = field(default_factory=list)  # Problem IDs where this worked

    # Metadata
    confidence: float = 0.3  # 0.0 to 1.0, default to min_confidence
    use_count: int = 0
    success_signals: int = 0  # Implicit success indicators
    failure_signals: int = 0  # Implicit failure indicators
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)

    # Tracking
    source_hash: str = ""  # Hash of the problem it came from
    codebase_context: dict = field(default_factory=dict)
    contextual_stats: dict = field(default_factory=dict)  # tuple[str, ...] -> tuple[int, int]
    source_context: dict = field(default_factory=dict)  # Import provenance metadata

    # Semantic embeddings (NEW - Phase 1)
    embedding: Optional[np.ndarray] = None  # Embedding vector for semantic retrieval
    embedding_model: str = ""  # Model used: "openai-3-small" or "bge-small-en-v1.5"
    embedding_dim: int = 0  # Embedding dimension for validation

    # Multi-perspective knowledge (NEW - Phase 2)
    code_skeleton: str = ""  # Reusable code template/structure
    common_pitfalls: list[str] = field(default_factory=list)  # Known failure modes
    test_patterns: list[str] = field(default_factory=list)  # Standard test cases
    algorithm_category: str = ""  # High-level category: "two-pointer", "dp", "graph", etc.
    merge_count: int = 0  # How many entries merged into this archetype

    # Difficulty affinity (NEW - Phase 4)
    difficulty_affinity: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Maps difficulty → (success_count, total_count)
    # e.g., {"easy": (5, 5), "medium": (3, 8), "hard": (1, 10)}
    # This tracks how well this archetype performs at different difficulty levels

    # Problem-level outcomes (NEW - Phase 4: Self-Contrast)
    problem_outcomes: dict[str, bool] = field(default_factory=dict)
    # Maps source_hash → success/failure for contrastive learning
    # e.g., {"abc123": True, "def456": False}
    # This enables "pattern A works where pattern B fails" analysis

    # Constants
    MIN_CONTEXTUAL_SIGNALS = 10
    MAX_BUCKETS_PER_ENTRY = 50

    def get_contextual_confidence(self, task_type: str = "", error_state: str = "",
                                  file_ext: str = "") -> Optional[float]:
        """Get confidence for specific context with error-first fallback."""
        if not task_type and not error_state and not file_ext:
            return None

        fallback_order = [
            ("task", task_type, "error", error_state, "ext", file_ext),
            ("task", task_type, "error", error_state),
            ("error", error_state, "ext", file_ext),
            ("error", error_state),
            ("task", task_type),
        ]

        for bucket_key in fallback_order:
            # Skip if key components are empty
            if any(not v for v in bucket_key[1::2]):  # Check values in tuple
                continue

            if bucket_key in self.contextual_stats:
                success, total = self.contextual_stats[bucket_key]
                if total >= self.MIN_CONTEXTUAL_SIGNALS:
                    return success / total

        # Global fallback
        global_total = self.success_signals + self.failure_signals
        if global_total >= self.MIN_CONTEXTUAL_SIGNALS:
            return self.success_signals / global_total

        return None

    def record_contextual_outcome(self, task_type: str = "", error_state: str = "",
                                  file_ext: str = "", success: bool = True):
        """Record outcome with bucket limit enforcement."""
        if not task_type and not error_state and not file_ext:
            return  # No context to record

        # Update specific buckets (most to least specific)
        buckets = []
        if task_type and error_state and file_ext:
            buckets.append(("task", task_type, "error", error_state, "ext", file_ext))
        if task_type and error_state:
            buckets.append(("task", task_type, "error", error_state))
        if error_state and file_ext:
            buckets.append(("error", error_state, "ext", file_ext))
        if error_state:
            buckets.append(("error", error_state))
        if task_type:
            buckets.append(("task", task_type))

        for bucket_key in buckets:
            self._update_bucket(bucket_key, success)

    def _update_bucket(self, bucket_key: tuple, success: bool):
        """Update bucket with LRU eviction if needed."""
        if bucket_key in self.contextual_stats:
            old_success, old_total = self.contextual_stats[bucket_key]
            self.contextual_stats[bucket_key] = (
                old_success + (1 if success else 0),
                old_total + 1
            )
        elif len(self.contextual_stats) < self.MAX_BUCKETS_PER_ENTRY:
            self.contextual_stats[bucket_key] = (1 if success else 0, 1)
        else:
            # Evict least-used bucket
            min_bucket = min(self.contextual_stats.items(), key=lambda x: x[1][1])
            del self.contextual_stats[min_bucket[0]]
            self.contextual_stats[bucket_key] = (1 if success else 0, 1)

    def record_difficulty_outcome(self, difficulty: str, success: bool, problem_hash: str = ""):
        """
        Record outcome for specific difficulty level (Phase 4) and problem identity (Phase 4: Self-Contrast).

        This tracks how well this archetype performs on different difficulty levels
        and on specific problems (for contrastive learning).

        Args:
            difficulty: Difficulty level string (e.g., "easy", "medium", "hard")
            success: Whether the archetype succeeded for this difficulty
            problem_hash: Optional source_hash identifying the specific problem (Phase 4: Self-Contrast)

        Design decisions:
        - Why track per-difficulty? Enables adaptive archetype selection.
          Example: "two-pointer" archetype might have 95% success on easy,
          65% on medium, 30% on hard. This tells us to use it confidently
          for easy problems but consider alternatives for hard ones.

        - Why tuple (success, total) instead of just success_rate?
          Preserves sample size information. (5/5) is more confident than (1/1).
          We can apply statistical confidence intervals when selecting archetypes.

        - Why track problem_hash? Enables contrastive learning (Phase 4).
          Answer "which patterns work WHERE OTHERS FAIL" on same problems.
        """
        # Validate and normalize difficulty
        difficulty = difficulty.lower().strip() if difficulty else "medium"
        if difficulty not in ALLOWED_DIFFICULTIES:
            logger.warning(f"Unknown difficulty '{difficulty}', defaulting to 'medium'")
            difficulty = "medium"

        if difficulty not in self.difficulty_affinity:
            self.difficulty_affinity[difficulty] = (0, 0)

        success_count, total_count = self.difficulty_affinity[difficulty]
        if success:
            success_count += 1
        total_count += 1
        self.difficulty_affinity[difficulty] = (success_count, total_count)

        logger.debug(f"Difficulty affinity [{difficulty}]: {success_count}/{total_count}")

        # Phase 4: Track problem-level outcome for self-contrast
        if problem_hash:
            self.problem_outcomes[problem_hash] = success
            logger.debug(f"Recorded problem outcome: {problem_hash[:8]}... → {success}")


    def record_outcome(self, success: bool, context: dict = None):
        """
        Record outcome with STRONG reinforcement (Phase 4).

        This is the feedback loop that makes Neo learn from experience.
        Replaces weak ±0.02 adjustments with strong ±0.1 updates.

        Args:
            success: Whether the solution worked
            context: Optional context dict with keys:
                - task_type (str): Type of task
                - has_error (bool): Whether there was an error
                - file_extension (str): File extension
                - difficulty (str): Problem difficulty (for affinity tracking)

        Design decisions:
        - Why ±0.1 instead of ±0.02 or ±0.2?
          ±0.02 was too weak - barely moved the needle after 5-10 uses.
          ±0.2 would be too aggressive - one failure could tank a good archetype.
          ±0.1 is balanced: 10 successes → 1.0 confidence, 10 failures → 0.0.
          This means ~10 examples are needed to form a strong opinion, which
          matches typical ML sample size requirements.

        - Why clamp to [0.0, 1.0]? Confidence is a probability, must be bounded.
          Without clamping, we'd get confidence > 1.0 or < 0.0, which breaks
          scoring functions that multiply by confidence.

        - Why update last_used on both success and failure?
          Tracks recency regardless of outcome. An archetype that keeps failing
          will have recent last_used but low confidence, signaling "tried recently
          but didn't work." This prevents retrying bad patterns immediately.
        """
        if context:
            task_type = context.get("task_type", "general")
            error_state = "with_error" if context.get("has_error") else "no_error"
            file_ext = context.get("file_extension", "unknown")

            # Record contextual outcome
            self.record_contextual_outcome(task_type, error_state, file_ext, success)

            # Record difficulty outcome if available (Phase 4)
            # (Linus: Consistent with record_difficulty_outcome - default invalid to "medium")
            difficulty = context.get("difficulty", "").strip() if context else ""
            if difficulty:
                difficulty = difficulty.lower()
                if difficulty not in ALLOWED_DIFFICULTIES:
                    logger.warning(f"Unknown difficulty '{difficulty}', defaulting to 'medium'")
                    difficulty = "medium"
                self.record_difficulty_outcome(difficulty, success)

        # STRONG reinforcement (was ±0.02, now ±0.1)
        if success:
            self.success_signals += 1
            self.confidence = min(1.0, self.confidence + self.CONFIDENCE_BOOST_SUCCESS)  # Cap at 1.0
            logger.debug(f"Success: boosted confidence to {self.confidence:.3f}")
        else:
            self.failure_signals += 1
            self.confidence = max(0.0, self.confidence + self.CONFIDENCE_BOOST_FAILURE)  # Floor at 0.0
            logger.debug(f"Failure: decayed confidence to {self.confidence:.3f}")

        self.last_used = time.time()

    def score(self, task_type: str = "", error_state: str = "",
              file_ext: str = "", base_weight: float = 0.5) -> float:
        """Calculate contextual confidence score."""
        # Base confidence with global success rate
        base_score = self.confidence
        total_signals = self.success_signals + self.failure_signals
        if total_signals >= 10:  # MIN_CONTEXTUAL_SIGNALS
            base_success_rate = self.success_signals / total_signals
            base_score = base_score * 0.5 + base_success_rate * 0.5

        # Get contextual confidence
        contextual_conf = self.get_contextual_confidence(task_type, error_state, file_ext)

        # Blend or use base only
        if contextual_conf is None:
            score = base_score
        else:
            score = base_score * base_weight + contextual_conf * (1 - base_weight)

        # Stronger recency decay - penalize old entries more aggressively
        age_days = (time.time() - self.last_used) / 86400
        # Half-life of 14 days instead of 30, stronger penalty (0.5 base instead of 0.7)
        recency_factor = 0.5 ** (age_days / 14)
        score *= (0.5 + 0.5 * recency_factor)

        # Usage bonus (unchanged)
        usage_factor = min(1.0, self.use_count / 10)
        score *= (0.9 + 0.1 * usage_factor)

        return score

    def to_dict(self) -> dict:
        """Serialize to dict."""
        data = {
            "pattern": self.pattern,
            "context": self.context,
            "reasoning": self.reasoning,
            "suggestion": self.suggestion,
            "algorithm_type": self.algorithm_type,
            "code_template": self.code_template,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "when_to_use": self.when_to_use,
            "example_problems": self.example_problems,
            "confidence": self.confidence,
            "use_count": self.use_count,
            "success_signals": self.success_signals,
            "failure_signals": self.failure_signals,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "source_hash": self.source_hash,
            "codebase_context": self.codebase_context,
            "source_context": self.source_context,
            "contextual_stats": {
                "|".join(k): v for k, v in self.contextual_stats.items()
            } if self.contextual_stats else {},
            # Phase 2 fields (bounded at input time)
            "code_skeleton": self.code_skeleton,  # Already bounded at input
            "common_pitfalls": self.common_pitfalls,
            "test_patterns": self.test_patterns,
            "algorithm_category": self.algorithm_category,
            "merge_count": self.merge_count,
            # Phase 4 field
            "difficulty_affinity": {
                # Serialize difficulty affinity: difficulty -> [success_count, total_count]
                # Store as list instead of tuple for JSON compatibility
                difficulty: list(counts)
                for difficulty, counts in self.difficulty_affinity.items()
            } if self.difficulty_affinity else {},
            # Phase 4: Self-Contrast field
            "problem_outcomes": {
                problem_hash: success
                for problem_hash, success in self.problem_outcomes.items()
            } if self.problem_outcomes else {},
        }

        # Serialize embedding as list for JSON compatibility
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
            data["embedding_model"] = self.embedding_model
            data["embedding_dim"] = self.embedding_dim

        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ReasoningEntry":
        """Deserialize from dict."""
        # Convert serialized contextual_stats back to tuple keys
        if "contextual_stats" in data and data["contextual_stats"]:
            serialized = data["contextual_stats"]
            data["contextual_stats"] = {
                tuple(k.split("|")): tuple(v) if isinstance(v, list) else v
                for k, v in serialized.items()
            }
        else:
            data["contextual_stats"] = {}

        # Backward compatibility: set defaults for new fields if missing
        data.setdefault("algorithm_type", "")
        data.setdefault("code_template", "")
        data.setdefault("time_complexity", "")
        data.setdefault("space_complexity", "")
        data.setdefault("when_to_use", "")
        data.setdefault("example_problems", [])

        # Phase 2 backward compatibility
        data.setdefault("code_skeleton", "")
        data.setdefault("common_pitfalls", [])
        data.setdefault("test_patterns", [])
        data.setdefault("algorithm_category", "")
        data.setdefault("merge_count", 0)

        # Phase 4 backward compatibility: difficulty_affinity
        # Deserialize from list back to tuple: [success, total] -> (success, total)
        if "difficulty_affinity" in data and data["difficulty_affinity"]:
            serialized_affinity = data["difficulty_affinity"]
            data["difficulty_affinity"] = {
                difficulty: tuple(counts) if isinstance(counts, list) else counts
                for difficulty, counts in serialized_affinity.items()
            }
        else:
            data["difficulty_affinity"] = {}

        # Phase 4: Self-Contrast backward compatibility
        data.setdefault("problem_outcomes", {})

        # Import provenance backward compatibility
        data.setdefault("source_context", {})

        # Deserialize embedding from list back to numpy array with validation
        try:
            if "embedding" in data and data["embedding"] is not None:
                emb = np.array(data["embedding"])
                embedding_model = data.get("embedding_model", "unknown")
                embedding_dim = data.get("embedding_dim", len(emb))

                # Validate dimension matches
                if len(emb) != embedding_dim:
                    logger.warning(f"Embedding dimension mismatch: stored={embedding_dim}, actual={len(emb)}")

                # Validate dimension (accept OpenAI 1536, Jina 768, and old local 384)
                valid_dims = {384, 768, 1536}  # Supported embedding dimensions
                if emb.shape[0] not in valid_dims:
                    logger.warning(f"Invalid embedding dimension ({emb.shape[0]}), expected {valid_dims}, discarding")
                    data["embedding"] = None
                    data["embedding_model"] = ""
                    data["embedding_dim"] = 0
                elif not np.isfinite(emb).all():
                    logger.warning("Embedding contains NaN or Inf values, discarding")
                    data["embedding"] = None
                    data["embedding_model"] = ""
                    data["embedding_dim"] = 0
                else:
                    data["embedding"] = emb
                    data["embedding_model"] = embedding_model
                    data["embedding_dim"] = embedding_dim
            else:
                data["embedding"] = None
                data["embedding_model"] = ""
                data["embedding_dim"] = 0
        except Exception as e:
            logger.warning(f"Failed to deserialize embedding: {e}")
            data["embedding"] = None

        return cls(**data)


class PersistentReasoningMemory:
    """
    Manages persistent reasoning knowledge.

    Key features:
    - Bounded storage (max entries)
    - Automatic pruning (remove low-value entries)
    - Consolidation (merge similar entries)
    - Implicit feedback (detect success/failure from context)
    """

    # Memory consolidation constants
    MIN_CONSOLIDATION_ENTRIES = 10
    MAX_CONSOLIDATION_ENTRIES = 2000  # FAISS handles large scale efficiently (was 200)
    MAX_PITFALLS_PER_ENTRY = 20
    MAX_TESTS_PER_ENTRY = 15
    MAX_EXAMPLES_PER_ENTRY = 20

    # ReasoningBank scoring constants (Linus: Extracted from magic numbers for tunability)
    # Phase 4: Difficulty affinity and contrastive learning
    AFFINITY_BONUS_WEIGHT = 0.2  # How much difficulty-specific success affects retrieval score
    CONTRASTIVE_SCALE = 0.4  # Maps contrast ratio [0,1] to boost [-0.2, +0.2]

    # Phase 5: Strategy evolution boosts
    STRATEGY_BOOST_HARD_COMPOSITIONAL = 0.15  # Boost for compositional strategies on hard problems
    STRATEGY_BOOST_HARD_PROCEDURAL = -0.10  # Penalty for procedural strategies on hard problems
    STRATEGY_BOOST_HARD_ADAPTIVE = 0.05  # Slight boost for adaptive on hard
    STRATEGY_BOOST_EASY_ADAPTIVE = 0.05  # Prefer adaptive on easy
    STRATEGY_BOOST_MEDIUM_ADAPTIVE = 0.10  # Prefer adaptive on medium

    # Confidence reinforcement (strong signals for learning)
    CONFIDENCE_BOOST_SUCCESS = 0.1  # Boost confidence on success (was 0.02)
    CONFIDENCE_BOOST_FAILURE = -0.1  # Reduce confidence on failure (was -0.02)
    CONFIDENCE_BOOST_RETRIEVAL = 0.02  # Small boost when pattern is retrieved

    # Pattern detection keywords
    ALGORITHM_PATTERNS = {
        "two-pointer": ["two pointer", "left right", "start end", "both ends", "from edges"],
        "sliding-window": ["sliding window", "subarray", "substring", "consecutive", "window size"],
        "hash-map": ["hash map", "dictionary", "frequency", "count occurrences", "lookup"],
        "binary-search": ["binary search", "sorted array", "log n", "divide half", "mid point"],
        "dynamic-programming": ["dp", "dynamic programming", "memoization", "subproblem", "optimal substructure"],
        "greedy": ["greedy", "local optimum", "sorted first", "priority"],
        "backtracking": ["backtrack", "recursion", "try all", "explore paths", "permutation", "combination"],
        "graph": ["graph", "bfs", "dfs", "traversal", "adjacency", "connected"],
        "tree": ["tree", "binary tree", "level order", "inorder", "preorder", "postorder"],
        "stack": ["stack", "lifo", "push pop", "monotonic"],
        "queue": ["queue", "fifo", "deque"],
        "heap": ["heap", "priority queue", "max heap", "min heap"],
        "sort": ["sort", "merge sort", "quick sort", "bubble sort"],
    }

    # Problem description indicators (junk patterns)
    # NOTE: These should be full phrases, not fragments that appear in solution descriptions
    JUNK_INDICATORS = [
        "you are given",
        "given an array",
        "given a string",
        "return true if",
        "return the number of",
        "find the number of",
        "find the maximum",
        "find the minimum",
        "determine if the",
        "calculate the number",
        "check if the",
    ]

    def __init__(
        self,
        storage_path: Optional[str] = None,
        codebase_root: Optional[str] = None,
        min_confidence: float = 0.3,
        similarity_threshold: float = 0.8,
        reference_quality: float = 500.0,  # Quality threshold for sigmoid midpoint
        max_entries: int = 2000,  # Hard cap on memory size (FAISS handles this efficiently)
        storage_backend: Optional[StorageBackend] = None,  # Optional pluggable storage backend
        config: Optional[Any] = None,  # NeoConfig instance
    ):
        self.min_confidence = min_confidence
        self.similarity_threshold = similarity_threshold
        self.reference_quality = reference_quality
        self.max_entries = max_entries
        self.codebase_root = Path(codebase_root).resolve() if codebase_root else None

        # Storage backend: use FileStorage for local files
        if storage_backend:
            self.storage_backend = storage_backend
        else:
            logger.info("Using FileStorage (local ~/.neo directory)")
            self.storage_backend = FileStorage()

        # Storage keys for global and local memory
        self.global_storage_key = "global"

        # Two memory stores:
        # 1. Global memory - learns across all codebases
        # 2. Local memory - codebase-specific patterns

        if storage_path:
            self.global_path = Path(storage_path)
            self.local_path = None
            self.local_storage_key = None
        else:
            self.global_path = Path.home() / ".neo" / "global_memory.json"

            if codebase_root:
                # Hash the codebase path for local storage
                codebase_hash = hashlib.sha256(str(self.codebase_root).encode()).hexdigest()[:16]
                self.local_storage_key = f"local_{codebase_hash}"
                self.local_path = Path.home() / ".neo" / f"local_{codebase_hash}.json"
            else:
                self.local_storage_key = None
                self.local_path = None

        # Initialize entries list
        self.entries: list[ReasoningEntry] = []

        # Epsilon-greedy exploration (3% random sampling - reduced from 10%)
        self.exploration_rate: float = 0.03

        # Track last consolidation count to prevent race conditions
        self.last_consolidation_count = 0

        # Initialize embedding cache
        self.embedding_cache = OrderedDict()  # LRU cache with bounded size

        # Initialize local embedding model (PRIMARY) - code-specific model
        # Using Jina Code v2 for code similarity tasks (768 dims)
        self.local_embedder = None
        if FASTEMBED_AVAILABLE:
            try:
                # Use jinaai/jina-embeddings-v2-base-code - 768 dims, optimized for code
                # Downloads ~400MB model on first use, cached afterward
                self.local_embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")
                logger.info("Local code embedding model (jina-embeddings-v2-base-code) initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize local embedder: {e}")

        # Initialize OpenAI client for embeddings (OPTIONAL, only if explicitly enabled)
        # Only use if USE_OPENAI_EMBEDDINGS=true is set
        self.openai_client = None
        if os.getenv("USE_OPENAI_EMBEDDINGS") == "true" and os.getenv("OPENAI_API_KEY"):
            try:
                from openai import OpenAI
                self.openai_client = OpenAI()
                logger.info("OpenAI embedding client initialized (opt-in via USE_OPENAI_EMBEDDINGS=true)")
            except ImportError:
                logger.warning("OpenAI package not installed. Install with: pip install openai")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")

        # Log embedding availability
        if self.local_embedder:
            logger.info("Using local code embeddings (jina-embeddings-v2-base-code, 768 dims)")
        elif self.openai_client:
            logger.info("Using OpenAI embeddings (1536 dims)")
        else:
            logger.warning("No embedding model available, falling back to MinHash. Install fastembed with: pip install fastembed")

        # Log FAISS availability
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, will use O(n²) clustering fallback. Install with: pip install faiss-cpu")
        elif FAISS_GPU_AVAILABLE:
            logger.info(f"GPU-accelerated FAISS available ({faiss.get_num_gpus()} GPUs detected)")
        else:
            logger.info("CPU FAISS available (install faiss-gpu for 10-100x faster clustering)")

        # Load both global and local entries (after openai_client is set)
        self.load()

        # Check for model consistency
        models_in_use = set()
        for entry in self.entries:
            if entry.embedding_model:
                models_in_use.add(entry.embedding_model)

        if len(models_in_use) > 1:
            logger.warning(f"\033[93mWarning: Memory contains multiple embedding models: {models_in_use}\033[0m")
            logger.warning("\033[93mRun 'neo --regenerate-embeddings' to fix this and enable proper consolidation\033[0m")

        # Initialize last_consolidation_count after loading entries
        self.last_consolidation_count = len(self.entries)

        # Initialize LSH index for deduplication
        self.lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=128)
        self._rebuild_lsh_index()

        # TF-IDF for semantic similarity
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

            # Fit on all entry patterns
            if self.entries:
                corpus = [entry.pattern + " " + entry.context for entry in self.entries]
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    stop_words='english'
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
                self.tfidf_enabled = True
                logger.info(f"TF-IDF initialized with {len(corpus)} entries")
            else:
                self.tfidf_enabled = False
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None
                logger.info("TF-IDF initialization deferred until entries are loaded")
        except ImportError:
            self.tfidf_enabled = False
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            logger.warning("scikit-learn not available, using substring matching")
        except Exception as e:
            self.tfidf_enabled = False
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            logger.warning(f"TF-IDF init failed: {e}")

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for text using local model or OpenAI API.

        Priority:
        1. Local fastembed (code-specific, free, 768 dims) - DEFAULT
        2. OpenAI API (if USE_OPENAI_EMBEDDINGS=true, 1536 dims) - OPTIONAL
        3. None (fall back to MinHash)

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding, or None if unavailable
        """
        if not text or not text.strip():
            return None

        # Hash text for efficient cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()

        # Check cache first
        if cache_key in self.embedding_cache:
            self.embedding_cache.move_to_end(cache_key)
            cached = self.embedding_cache[cache_key]
            # Handle both old format (ndarray) and new format (tuple)
            if isinstance(cached, tuple):
                return cached[0]  # Return just the embedding
            else:
                return cached  # Old format, backward compatible

        embedding = None

        # Try local model first (free, fast, code-specific, 768 dims)
        if self.local_embedder:
            embedding = self._embed_with_local(text)

        # Fall back to OpenAI if explicitly enabled (higher dimensional, 1536 dims)
        elif self.openai_client:
            embedding = self._embed_with_openai(text)

        if embedding is not None:
            # Determine model and dimension
            if self.openai_client:
                model_name = "openai-3-small"
                dim = 1536
            elif self.local_embedder:
                model_name = "jina-embeddings-v2-base-code"
                dim = 768
            else:
                model_name = "unknown"
                dim = len(embedding)

            # Cache with metadata: (embedding, model, dim)
            self.embedding_cache[cache_key] = (embedding, model_name, dim)
            if len(self.embedding_cache) > EMBEDDING_CACHE_MAX_SIZE:
                self.embedding_cache.popitem(last=False)

        return embedding

    def _embed_with_openai(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using OpenAI API."""
        # Truncate if too long
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(f"Text too long ({len(text)} chars), truncating to {MAX_TEXT_LENGTH}")
            text = text[:MAX_TEXT_LENGTH]

        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )

            if not response.data or len(response.data) == 0 or not response.data[0].embedding:
                logger.error("Empty embedding response from OpenAI API")
                return None

            embedding = np.array(response.data[0].embedding)

            # Validate dimension and values
            if embedding.shape[0] != EMBEDDING_DIM:
                logger.error(f"Unexpected embedding dimension: {embedding.shape[0]}")
                return None
            if not np.isfinite(embedding).all():
                logger.error("Embedding contains NaN or Inf values")
                return None

            return embedding

        except Exception as e:
            exc_name = type(e).__name__
            if exc_name == 'RateLimitError':
                logger.warning(f"OpenAI rate limit hit: {e}")
            elif exc_name in ['APIError', 'OpenAIError']:
                logger.error(f"OpenAI error: {e}")
            else:
                logger.exception("Unexpected error generating OpenAI embedding")
            return None

    def _embed_with_local(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding using local fastembed model."""
        try:
            # fastembed returns list of embeddings, we want first one
            embeddings = list(self.local_embedder.embed([text]))

            if not embeddings or len(embeddings) == 0:
                logger.error("Empty embedding from local model")
                return None

            # Convert to numpy array (float32)
            embedding = np.array(embeddings[0], dtype=np.float32)

            # Validate
            if embedding.shape[0] != 768:  # jina-embeddings-v2-base-code dimension
                logger.error(f"Unexpected local embedding dimension: {embedding.shape[0]} (expected 768)")
                return None
            if not np.isfinite(embedding).all():
                logger.error("Local embedding contains NaN or Inf values")
                return None

            return embedding

        except Exception as e:
            logger.exception(f"Error generating local embedding: {e}")
            return None

    def _is_junk_entry(self, pattern: str, suggestion: str) -> bool:
        """
        Detect if entry is junk (problem description instead of solution pattern).

        Returns True if entry appears to be a problem description rather than a solution.
        """
        text_lower = (pattern + " " + suggestion).lower()
        suggestion_lower = suggestion.lower()

        # Check for problem description indicators
        junk_count = sum(1 for indicator in self.JUNK_INDICATORS if indicator in text_lower)

        # If has junk indicators AND suggestion is long (>150 chars), likely junk
        if junk_count >= 2 and len(suggestion) > 150:
            logger.debug(f"Rejected junk entry: pattern='{pattern[:50]}', junk_indicators={junk_count}")
            return True

        # Even one junk indicator with very long text is suspicious
        if junk_count >= 1 and len(suggestion) > 300:
            logger.debug(f"Rejected long junk entry: pattern='{pattern[:50]}', junk_indicators={junk_count}, len={len(suggestion)}")
            return True

        # Raw code without explanation (most common junk pattern from livecode)
        # If it's mostly code (has def/class) and no algorithmic keywords
        has_code = "def " in suggestion or "class " in suggestion
        has_algo_keywords = any(kw in text_lower for keywords in self.ALGORITHM_PATTERNS.values() for kw in keywords)

        if has_code and len(suggestion) > 100 and not has_algo_keywords:
            logger.debug(f"Rejected raw code without insights: pattern='{pattern[:50]}'")
            return True

        return False

    def _extract_algorithm_pattern(self, code: str, reasoning: str) -> dict:
        """
        Extract algorithmic pattern from code and reasoning.

        Returns dict with:
        - algorithm_type: Detected pattern type
        - code_template: Simplified code structure
        - time_complexity: Big-O time
        - space_complexity: Big-O space
        - when_to_use: Usage guidance
        """
        text_lower = (code + " " + reasoning).lower()
        result = {
            "algorithm_type": "",
            "code_template": "",
            "time_complexity": "",
            "space_complexity": "",
            "when_to_use": ""
        }

        # Detect algorithm type
        detected_patterns = []
        for pattern_name, keywords in self.ALGORITHM_PATTERNS.items():
            if any(kw in text_lower for kw in keywords):
                detected_patterns.append(pattern_name)

        if detected_patterns:
            result["algorithm_type"] = ", ".join(detected_patterns[:2])  # Top 2 patterns
            result["when_to_use"] = f"Use for problems involving {', '.join(detected_patterns[:2])}"

        # Extract complexity from reasoning
        import re
        time_match = re.search(r'O\([^)]+\)', reasoning)
        if time_match:
            result["time_complexity"] = time_match.group(0)

        # Extract simplified code template (first 200 chars of meaningful code)
        if code and "def " in code:
            # Get function signature and first few lines
            lines = code.split('\n')
            template_lines = []
            for line in lines[:5]:
                if line.strip() and not line.strip().startswith('#'):
                    template_lines.append(line)
            result["code_template"] = '\n'.join(template_lines)[:200]

        return result

    def add_reasoning(
        self,
        pattern: str,
        context: str,
        reasoning: str,
        suggestion: str,
        confidence: float,
        source_context: dict,
        # NEW: Phase 2 optional parameters (backward compatible)
        code_skeleton: str = "",
        common_pitfalls: list[str] = None,
        test_patterns: list[str] = None,
    ):
        """
        Add new reasoning with Bayesian skepticism and junk filtering.

        Key insight: Don't trust initial confidence - require evidence.

        Strategy:
        1. Reject junk entries (problem descriptions instead of solutions)
        2. Extract algorithmic patterns from valid entries
        3. New entries start at min_confidence (skeptical admission)
        4. Confidence grows with successful retrievals
        5. Confidence shrinks with failures
        6. Natural selection: good patterns survive, bad ones fade

        Phase 2 additions:
        - code_skeleton: Reusable code template/structure
        - common_pitfalls: Known failure modes
        - test_patterns: Standard test cases
        """
        # Handle mutable default arguments
        if common_pitfalls is None:
            common_pitfalls = []
        if test_patterns is None:
            test_patterns = []

        # Phase 3: Extract failure root causes if confidence is low
        # ReasoningBank insight: Learn from failures, not just successes
        if confidence < 0.5 and source_context.get("error_trace"):
            failure_causes = self._extract_failure_root_cause(source_context, suggestion)
            if failure_causes:
                common_pitfalls.extend(failure_causes)
                logger.info(f"Extracted {len(failure_causes)} failure patterns: {failure_causes}")

        # VALIDATION: Reject junk entries
        if self._is_junk_entry(pattern, suggestion):
            logger.info(f"Rejected junk entry: pattern='{pattern[:50]}'")
            return  # Do not store
        # Create hash of the pattern+context for deduplication
        content_hash = self._hash_content(pattern + context)

        # Check for duplicates or near-duplicates
        similar = self._find_similar(pattern, context)

        if similar:
            # Boost confidence of existing entry (evidence accumulation)
            # Use exponential moving average to gradually increase confidence
            alpha = 0.1  # Learning rate
            similar.confidence = (1 - alpha) * similar.confidence + alpha * confidence
            similar.confidence = min(0.95, similar.confidence)  # Cap at 0.95
            similar.reasoning = reasoning  # Update with latest reasoning
            similar.suggestion = suggestion
            similar.last_used = time.time()
            logger.debug(f"Boosted confidence for pattern '{pattern[:50]}' to {similar.confidence:.3f}")
            return

        # PATTERN EXTRACTION: Extract algorithmic insights
        algo_pattern = self._extract_algorithm_pattern(suggestion, reasoning)

        # Extract problem ID if available
        problem_id = source_context.get("problem_id", "")
        example_problems = [problem_id] if problem_id else []

        # New entries start skeptical - require evidence to gain confidence
        # Use min_confidence as starting point (typically 0.3)
        initial_confidence = self.min_confidence

        entry = ReasoningEntry(
            pattern=pattern,
            context=context,
            reasoning=reasoning,
            suggestion=suggestion,
            algorithm_type=algo_pattern["algorithm_type"],
            code_template=algo_pattern["code_template"],
            time_complexity=algo_pattern["time_complexity"],
            space_complexity=algo_pattern["space_complexity"],
            when_to_use=algo_pattern["when_to_use"],
            example_problems=example_problems[:self.MAX_EXAMPLES_PER_ENTRY],
            confidence=initial_confidence,  # Start skeptical
            source_hash=content_hash,
            codebase_context=self._extract_codebase_context(source_context),
            source_context=source_context,
            # Phase 2 fields
            code_skeleton=code_skeleton,
            common_pitfalls=common_pitfalls,
            test_patterns=test_patterns,
            algorithm_category=algo_pattern["algorithm_type"],  # Use detected algorithm type
            merge_count=0,  # New entries start with 0 merges
        )

        # Debug logging for source context
        if source_context:
            logger.debug(f"Adding reasoning entry with source context: {source_context.get('dataset', 'unknown')}")

        # Generate embedding for new entry if embedding model available
        # This enables immediate semantic retrieval without waiting for consolidation
        if self.openai_client or self.local_embedder:
            # ReasoningBank insight: Embed semantic anchor (WHAT + WHEN) not full content (HOW)
            # Clean signal: "Two-pointer: When array is sorted"
            # vs noisy: "Two-pointer: When array is sorted. Use convergent indices because..."
            # Result: Better clustering, +10-15% retrieval quality (see docs/phase2-semantic-anchor.md)
            entry_text = f"{pattern}: {context}"
            entry.embedding = self._embed_text(entry_text)
            if entry.embedding is not None:
                # Store model metadata
                cache_key = hashlib.md5(entry_text.encode()).hexdigest()
                if cache_key in self.embedding_cache:
                    cached = self.embedding_cache[cache_key]
                    if isinstance(cached, tuple):
                        _, model_name, dim = cached
                        entry.embedding_model = model_name
                        entry.embedding_dim = dim

                logger.debug(f"Generated {entry.embedding_dim}-dim embedding ({entry.embedding_model})")

        logger.debug(f"Added new pattern '{pattern[:50]}' with skeptical confidence {initial_confidence}, algo_type='{algo_pattern['algorithm_type']}'")

        # Create and cache MinHash signature
        try:
            text = pattern + " " + context
            minhash = self._create_minhash(text)
            entry.codebase_context['minhash_signature'] = [int(x) for x in minhash.hashvalues]

            # Insert into LSH index
            self.lsh.insert(entry.source_hash, minhash)
        except ValueError as e:
            logger.warning(f"Could not create MinHash for new entry: {e}")

        self.entries.append(entry)

        # Rebuild TF-IDF matrix to include new entry
        if self.tfidf_enabled and self.tfidf_vectorizer:
            try:
                corpus = [e.pattern + " " + e.context for e in self.entries]
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            except Exception as e:
                logger.warning(f"TF-IDF rebuild after add failed: {e}")

        # Quality-based pruning (remove low-value entries only)
        # No hard cap - prune only entries below min confidence with no usage
        self._prune_low_quality()

        # Phase 3: Automatic consolidation trigger
        self._maybe_consolidate()

        self.save()

    def detect_implicit_feedback(
        self,
        current_request: dict,
        recent_history: list[dict]
    ):
        """
        Detect implicit success/failure signals.

        Signals of FAILURE:
        - Same problem asked again within short time
        - Error trace mentions suggested code
        - User asks to "fix" or "undo"

        Signals of SUCCESS:
        - Related but different problem (moved on)
        - No errors in recent context
        - Building on previous suggestion
        """
        if not recent_history:
            return

        current_problem = self._hash_content(current_request.get("prompt", ""))
        task_type = current_request.get("task_type", "")

        # Adaptive thresholds based on task complexity
        if task_type == "bugfix":
            re_ask_threshold = 120  # 2 minutes (quick iteration)
            success_threshold = 600  # 10 minutes
        elif task_type == "refactor":
            re_ask_threshold = 600  # 10 minutes (needs thought)
            success_threshold = 1800  # 30 minutes
        else:  # feature, exploration, unknown
            re_ask_threshold = 1800  # 30 minutes (complex work)
            success_threshold = 3600  # 1 hour

        # Convert deque to list for slicing support
        history_list = list(recent_history) if not isinstance(recent_history, list) else recent_history

        for past_entry in history_list[-5:]:  # Last 5 requests
            past_problem = past_entry.get("problem_hash", "")
            time_delta = time.time() - past_entry.get("timestamp", 0)

            # Check for re-asks (failure signal)
            # Simple hash equality check - if exact same problem
            if current_problem == past_problem:
                if time_delta < re_ask_threshold:
                    # Weighted signal based on how quickly user re-asked
                    if time_delta < re_ask_threshold / 3:
                        self._record_failure_signal(past_problem, weight=2.0)  # Strong failure
                    else:
                        self._record_failure_signal(past_problem, weight=1.0)  # Weak failure
            else:
                # Moved on to different problem (success signal)
                if time_delta < success_threshold:
                    # Weighted signal based on time to resolution
                    if time_delta < success_threshold / 3:
                        self._record_success_signal(past_problem, weight=2.0)  # Strong success
                    else:
                        self._record_success_signal(past_problem, weight=1.0)  # Weak success

    def format_entry_as_guidance(self, entry: ReasoningEntry) -> str:
        """
        Format a ReasoningEntry as actionable guidance.

        Returns a concise, actionable description focusing on HOW to solve, not WHAT the problem is.
        """
        parts = []

        # Algorithm type (if available)
        if entry.algorithm_type:
            parts.append(f"Pattern: {entry.algorithm_type}")

        # When to use (if available)
        if entry.when_to_use:
            parts.append(entry.when_to_use)

        # Complexity info
        if entry.time_complexity:
            complexity_str = f"Time: {entry.time_complexity}"
            if entry.space_complexity:
                complexity_str += f", Space: {entry.space_complexity}"
            parts.append(complexity_str)

        # Code template (if available and short)
        if entry.code_template and len(entry.code_template) < 150:
            parts.append(f"Template:\n{entry.code_template}")

        # Fall back to reasoning if no pattern fields populated
        if not parts and entry.reasoning:
            parts.append(entry.reasoning)

        return " | ".join(parts) if parts else entry.suggestion[:200]

    def retrieve_relevant(
        self,
        problem_context: dict,
        k: int = 5
    ) -> list[ReasoningEntry]:
        """
        Retrieve most relevant reasoning entries.

        Uses epsilon-greedy exploration:
        - With probability exploration_rate: return random sample (explore)
        - Otherwise: return top-k by score (exploit)

        If OpenAI client is available, uses semantic embedding retrieval.
        Otherwise falls back to MinHash-based retrieval.

        Considers:
        - Similarity to current problem (cosine similarity or Jaccard)
        - Entry score (confidence, success rate, recency)
        - Codebase match
        """
        # Difficulty normalization: defaults to "medium" when missing or invalid
        # Rationale: Most problems are medium difficulty, reasonable fallback for missing data
        # Alternative considered: error on invalid difficulty, rejected for robustness
        difficulty = problem_context.get("difficulty", "").lower().strip()
        difficulty = difficulty if difficulty in ALLOWED_DIFFICULTIES else "medium"

        # Update problem_context with normalized difficulty
        problem_context = copy.deepcopy(problem_context)  # Don't mutate original
        problem_context["difficulty"] = difficulty

        # Epsilon-greedy exploration
        if random.random() < self.exploration_rate:
            # Explore: return random sample
            if len(self.entries) <= k:
                return self.entries[:]
            return random.sample(self.entries, k)

        # Choose retrieval strategy based on embedding availability
        if self.openai_client or self.local_embedder:
            return self._retrieve_by_embedding(problem_context, difficulty, k)
        else:
            return self._retrieve_by_minhash(problem_context, difficulty, k)

    def _retrieve_by_minhash(
        self,
        problem_context: dict,
        difficulty: str = "medium",
        k: int = 5
    ) -> list[ReasoningEntry]:
        """
        MinHash-based retrieval (original implementation).
        Used as fallback when embeddings are unavailable.
        """
        # Extract features from current problem
        current_pattern = self._extract_pattern(problem_context)
        current_codebase = self._extract_codebase_context(problem_context)

        # Score each entry
        scored = []
        for entry in self.entries:
            relevance = self._calculate_relevance(
                entry,
                current_pattern,
                current_codebase
            )

            # Combined score: relevance * entry quality
            quality = entry.score()

            # Phase 4: Difficulty affinity bonus
            # If we know the problem difficulty and this entry has affinity data,
            # boost quality score based on historical success rate at this difficulty
            if difficulty in entry.difficulty_affinity:
                success, total = entry.difficulty_affinity[difficulty]
                if total > 0:
                    # Bonus scales from 0 to 0.2 based on success rate
                    # Why 0.2 max? Keeps bonus meaningful but not dominant.
                    # 100% success rate → +0.2 quality boost
                    # 50% success rate → +0.1 quality boost
                    # This makes proven archetypes more likely to be selected
                    affinity_bonus = (success / total) * self.AFFINITY_BONUS_WEIGHT
                    quality = min(1.0, quality + affinity_bonus)  # CAP AT 1.0
                    logger.debug(f"Entry '{entry.pattern[:50]}' difficulty affinity [{difficulty}]: "
                               f"{success}/{total} = +{affinity_bonus:.3f} bonus")

            combined = relevance * quality

            # Smart threshold: Accept if EITHER highly relevant OR high confidence
            # This makes memory robust to noise - junk has low score, good patterns survive
            accept = (
                combined > 0.05 or  # Multiplicative: both matter (lowered from 0.3)
                relevance > 0.3 or  # High similarity alone
                quality > 0.6         # High confidence alone (proven pattern)
            )

            if accept:
                scored.append((combined, entry))
                # Mark as used
                entry.use_count += 1
                entry.last_used = time.time()

                # Boost confidence on successful retrieval (evidence accumulation)
                # Small incremental boost - successful use = pattern is valuable
                entry.confidence = min(0.95, entry.confidence + self.CONFIDENCE_BOOST_RETRIEVAL)
                entry.success_signals += 1

        # Sort and return top-k
        scored.sort(reverse=True, key=lambda x: x[0])
        results = [entry for score, entry in scored[:k]]

        # Penalize entries that didn't make the cut (implicit negative signal)
        # If an entry was considered but not selected, it's less valuable
        for entry in self.entries:
            if entry not in results and entry.use_count > 0:
                # Gradual confidence decay for unused entries
                entry.confidence = max(self.min_confidence, entry.confidence - 0.01)

        return results

    def _infer_strategy_level(self, entry: ReasoningEntry) -> str:
        """
        Infer strategy complexity from performance patterns (Phase 5: Strategy Evolution).

        Heuristics based on difficulty_affinity:
        - Procedural: High easy success (>75%), low hard success (<40%)
        - Compositional: High hard success (>60%), evolved through consolidation (merge_count >3)
        - Adaptive: Balanced performance (40-75% across difficulties) or default

        Args:
            entry: ReasoningEntry to classify

        Returns:
            Strategy level: "procedural", "adaptive", or "compositional"
        """
        if not entry.difficulty_affinity:
            return "adaptive"  # Default for unknown

        # Extract success rates
        easy_rate = self._get_success_rate(entry.difficulty_affinity.get("easy"))
        hard_rate = self._get_success_rate(entry.difficulty_affinity.get("hard"))

        # Compositional: excels on hard, evolved through consolidation
        if hard_rate >= 0.60 and entry.merge_count >= 3:
            return "compositional"

        # Procedural: works on easy, fails on hard
        if easy_rate >= 0.75 and hard_rate < 0.40:
            return "procedural"

        # Adaptive: everything else (balanced or insufficient data)
        return "adaptive"

    @staticmethod
    def _get_success_rate(affinity_tuple: Optional[tuple[int, int]]) -> float:
        """Extract success rate from (success, total) tuple."""
        if not affinity_tuple or affinity_tuple[1] == 0:
            return 0.5  # Neutral default
        success, total = affinity_tuple
        return success / total

    def _calculate_strategy_boost(
        self,
        strategy_level: str,
        problem_difficulty: str
    ) -> float:
        """
        Calculate boost based on strategy-difficulty matching (Phase 5).

        Rules:
        - Hard problems: prefer compositional (+0.15), penalize procedural (-0.10)
        - Easy problems: accept procedural (0.0), slight prefer adaptive (+0.05)
        - Medium problems: prefer adaptive (+0.10)

        Args:
            strategy_level: "procedural", "adaptive", or "compositional"
            problem_difficulty: "easy", "medium", or "hard"

        Returns:
            Boost value in range [-0.10, +0.15]
        """
        if problem_difficulty == "hard":
            if strategy_level == "compositional":
                return self.STRATEGY_BOOST_HARD_COMPOSITIONAL
            elif strategy_level == "procedural":
                return self.STRATEGY_BOOST_HARD_PROCEDURAL
            else:  # adaptive
                return self.STRATEGY_BOOST_HARD_ADAPTIVE

        elif problem_difficulty == "easy":
            if strategy_level == "adaptive":
                return self.STRATEGY_BOOST_EASY_ADAPTIVE
            else:
                return 0.0  # Procedural is fine for easy

        else:  # medium
            if strategy_level == "adaptive":
                return self.STRATEGY_BOOST_MEDIUM_ADAPTIVE
            else:
                return 0.0

    def _retrieve_by_embedding(
        self,
        problem_context: dict,
        difficulty: str = "medium",
        k: int = 5
    ) -> list[ReasoningEntry]:
        """
        Semantic embedding-based retrieval using cosine similarity (Phase 1).
        Enhanced with difficulty affinity scoring (Phase 4).

        Args:
            problem_context: Context dict containing problem information
            difficulty: Normalized difficulty level for affinity scoring
            k: Number of entries to retrieve

        Returns:
            Top-k most relevant entries by semantic similarity

        Design decisions:
        - Why cosine similarity? Captures semantic similarity better than Jaccard.
          "BFS on grid" and "shortest path in maze" have high cosine similarity
          but low Jaccard similarity (different tokens, same meaning).

        - Why difficulty affinity bonus? Boosts archetypes with proven track record
          at the specific difficulty level. A "two-pointer" archetype with 90%
          success on easy problems gets +0.18 bonus (0.9 * 0.2), making it more
          likely to be selected for easy problems.

        - How does affinity_bonus scale? Linearly from 0 to 0.2:
          - 100% success at this difficulty → +0.2 quality boost
          - 50% success → +0.1 boost
          - 0% success → +0.0 boost (no penalty to base quality)
          Affinity bonus: ranges from 0 to 0.2 (additive)
          This provides meaningful boost (up to +40% for quality=0.5) without dominating selection (capped at 1.0)
        """
        # Extract query text from problem context
        query_text = problem_context.get("prompt", "")
        if not query_text:
            # Fallback to MinHash if no query text
            return self._retrieve_by_minhash(problem_context, difficulty, k)

        # Generate query embedding
        query_embedding = self._embed_text(query_text)
        if query_embedding is None:
            # Fallback to MinHash if embedding fails
            logger.debug("Embedding generation failed, falling back to MinHash retrieval")
            return self._retrieve_by_minhash(problem_context, difficulty, k)

        # Score each entry by cosine similarity
        scored = []
        for entry in self.entries:
            # Skip entries without embeddings instead of blocking on API calls
            if entry.embedding is None:
                continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, entry.embedding)

            # Combined score: similarity * entry quality
            quality = entry.score()

            # Phase 4: Difficulty affinity bonus
            if difficulty in entry.difficulty_affinity:
                success, total = entry.difficulty_affinity[difficulty]
                if total > 0:
                    affinity_bonus = (success / total) * self.AFFINITY_BONUS_WEIGHT
                    quality = min(1.0, quality + affinity_bonus)  # CAP AT 1.0
                    logger.debug(f"Entry '{entry.pattern[:50]}' difficulty affinity [{difficulty}]: "
                               f"{success}/{total} = +{affinity_bonus:.3f} bonus")

            # Phase 5: Strategy evolution boost (prefer advanced strategies for hard problems)
            strategy_level = self._infer_strategy_level(entry)
            strategy_boost = self._calculate_strategy_boost(strategy_level, difficulty)
            if strategy_boost != 0.0:
                quality = min(1.0, max(0.0, quality + strategy_boost))  # CAP AT [0.0, 1.0]
                logger.debug(f"Entry '{entry.pattern[:50]}' strategy boost: "
                           f"{strategy_level} on {difficulty} = {strategy_boost:+.3f}")

            combined = similarity * quality

            # Lower threshold than MinHash (0.1 vs 0.05)
            if combined > 0.1:
                scored.append((combined, entry))
                # Mark as used
                entry.use_count += 1
                entry.last_used = time.time()

                # Boost confidence on successful retrieval
                entry.confidence = min(0.95, entry.confidence + self.CONFIDENCE_BOOST_RETRIEVAL)
                entry.success_signals += 1

        # Sort and return top-k
        scored.sort(reverse=True, key=lambda x: x[0])
        results = [entry for score, entry in scored[:k]]

        # Penalize entries that didn't make the cut
        for entry in self.entries:
            if entry not in results and entry.use_count > 0:
                entry.confidence = max(self.min_confidence, entry.confidence - 0.01)

        return results

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity in [0, 1], or 0.0 for invalid vectors
        """
        # Check for NaN or Inf values
        if not np.isfinite(vec1).all() or not np.isfinite(vec2).all():
            logger.warning("Cannot compute cosine similarity: NaN or Inf values")
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            logger.debug("Zero vector in cosine similarity")
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def consolidate_memory(self, similarity_threshold: float = 0.85, max_entries: int = 2000) -> int:
        """
        Consolidate similar reasoning entries using threshold-based clustering.

        Algorithm: Transitive Threshold Clustering
        - Builds graph where edge exists if similarity > threshold
        - Finds connected components via DFS
        - Merges all entries in each component into single archetype

        WARNING: Uses transitive closure, not hierarchical single-linkage.
        Entry A can cluster with C through intermediate B even if A-C similarity is low.
        Raise threshold to 0.85+ to prevent over-merging.

        This is the Phase 3 implementation that transforms scattered memory entries
        into consolidated archetypes. Uses semantic embeddings (cosine similarity)
        instead of MinHash (Jaccard similarity) for better conceptual clustering.

        Args:
            similarity_threshold: Cosine similarity threshold for clustering (0.85 recommended, was 0.75)
                                 Lower than MinHash threshold (0.8) to enable broader semantic grouping.
                                 - 0.85+: Very similar (near-duplicates)
                                 - 0.75: Reasonably similar (same algorithm pattern)
                                 - 0.65: Somewhat similar (related concepts)
            max_entries: Maximum entries to cluster (default 2000, FAISS provides O(n log n) performance)

        Returns:
            Number of entries merged (original_count - new_count)

        Algorithm:
            1. Compute embeddings for all entries (pattern + context + reasoning)
            2. Build similarity matrix using cosine similarity
            3. Find clusters using single-linkage clustering (transitive similarity)
            4. For each cluster, merge into single archetype:
               - Keep highest confidence reasoning
               - Combine code_skeleton (longest version)
               - Union of common_pitfalls (deduplicated)
               - Union of test_patterns (deduplicated)
               - Increment merge_count
               - Track original patterns in example_problems
            5. Replace cluster entries with merged archetype

        Design decisions:
            - Clustering algorithm: Single-linkage (if A~B and B~C, then A,B,C cluster together)
              Rationale: Captures transitive semantic relationships. "Two-pointer sorted" ~
              "left-right pointers" ~ "opposite ends technique" all become one archetype.

            - Similarity threshold (0.85): Raised from 0.75 to compensate for transitive clustering
              aggressiveness. We chose 0.85 conservatively to avoid over-merging.

            - Merge strategy (highest confidence wins): Best-performing pattern becomes
              the canonical representation. Lower-confidence variations get absorbed.

            - Category detection: Uses keyword matching on combined text. Not ML-based
              because we need deterministic, inspectable categories.

        Backward compatibility:
            - Falls back to MinHash consolidation if embeddings unavailable
            - Existing entries without embeddings are skipped (not deleted)
            - Next retrieval will backfill embeddings on-demand
        """
        if not self.openai_client:
            # Fallback to MinHash-based consolidation
            logger.info("No OpenAI client, falling back to MinHash consolidation")
            return self._consolidate_minhash()

        if len(self.entries) < self.MIN_CONSOLIDATION_ENTRIES:
            logger.debug(f"Too few entries to consolidate (< {self.MIN_CONSOLIDATION_ENTRIES})")
            return 0

        original_count = len(self.entries)
        logger.info(f"Starting semantic consolidation on {original_count} entries (threshold={similarity_threshold})")

        # Step 1: Ensure all entries have embeddings (backfill on-demand with retry)
        entries_with_embeddings = []
        for entry in self.entries:
            if entry.embedding is None:
                # Backfill: embed semantic anchor (pattern + context) for clean retrieval signal
                # ReasoningBank insight: Index on WHAT and WHEN, not implementation details
                text = f"{entry.pattern}: {entry.context}"

                # Try up to 3 times with exponential backoff
                for attempt in range(3):
                    entry.embedding = self._embed_text(text)
                    if entry.embedding is not None:
                        break
                    if attempt < 2:
                        time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s
                        logger.warning(f"Retry {attempt + 1}/3 for embedding: {entry.pattern[:50]}")

                if entry.embedding is None:
                    logger.error(f"Failed to embed after 3 attempts, skipping: {entry.pattern[:50]}")

            # Only include entries with valid embeddings
            if entry.embedding is not None:
                entries_with_embeddings.append(entry)
            else:
                logger.warning(f"Skipping entry without embedding: pattern='{entry.pattern[:50]}'")

        if len(entries_with_embeddings) < 2:
            logger.info("Not enough entries with embeddings to consolidate")
            return 0

        # FAISS scales to 1000+ entries efficiently
        # No artificial limit needed
        logger.debug(f"Consolidating {len(entries_with_embeddings)} entries using FAISS-accelerated clustering")

        # Step 2: Build similarity matrix and find clusters
        clusters = self._find_semantic_clusters(entries_with_embeddings, similarity_threshold)

        if not clusters:
            logger.info("No clusters found for consolidation")
            return 0

        logger.info(f"Found {len(clusters)} clusters: {[len(c) for c in clusters]}")

        # Step 3: Merge each cluster into an archetype
        merged_entries = []
        entries_to_remove = []  # Changed from set to list (ReasoningEntry not hashable)

        for cluster_indices in clusters:
            cluster = [entries_with_embeddings[i] for i in cluster_indices]

            if len(cluster) == 1:
                # No merge needed, keep as-is
                merged_entries.append(cluster[0])
            else:
                # Merge cluster into single archetype
                merged = self._merge_cluster(cluster)
                merged_entries.append(merged)

                # Mark original entries for removal
                entries_to_remove.extend(cluster)

                logger.info(f"Merged {len(cluster)} entries into archetype: pattern='{merged.pattern[:50]}', "
                          f"confidence={merged.confidence:.3f}, merge_count={merged.merge_count}")

        # Step 4: Update entries list
        # Strategy:
        # - Remove ALL entries that were processed (entries_with_embeddings)
        # - Add back merged archetypes
        # - Keep entries without embeddings (they weren't processed)

        # Use id() for comparison since ReasoningEntry is not hashable
        processed_ids = {id(e) for e in entries_with_embeddings}
        entries_without_embeddings = [e for e in self.entries if id(e) not in processed_ids]

        # New entry list: unprocessed entries + merged archetypes
        self.entries = entries_without_embeddings + merged_entries

        # Enforce hard memory limit after consolidation
        if len(self.entries) > max_entries:
            self.entries = sorted(
                self.entries,
                key=lambda e: e.score(),
                reverse=True
            )[:max_entries]
            logger.warning(f"Enforced hard memory limit: pruned to {max_entries} best entries")

        # Rebuild indices
        self._rebuild_lsh_index()
        if self.tfidf_enabled and self.tfidf_vectorizer:
            try:
                corpus = [e.pattern + " " + e.context for e in self.entries]
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            except Exception as e:
                logger.warning(f"TF-IDF rebuild after consolidation failed: {e}")

        self.save()

        merged_count = original_count - len(self.entries)
        logger.info(f"Consolidation complete: {original_count} → {len(self.entries)} entries ({merged_count} merged)")
        return merged_count

    def consolidate(self):
        """
        Legacy consolidation method for backward compatibility.
        Calls consolidate_memory() with default threshold.
        """
        return self.consolidate_memory(similarity_threshold=0.85)

    def _maybe_consolidate(self):
        """
        Automatic consolidation trigger for memory defragmentation.

        Runs consolidation periodically when memory is fragmented.
        This prevents the memory from growing with scattered entries
        that should be merged into archetypes.

        Trigger conditions:
        - Memory has >= 30 entries (enough data to find patterns)
        - At least 10 entries added since last consolidation
        - Example: triggers at 40, 50, 60, 70, ... entries

        Why these thresholds:
        - 30 minimum: Need enough entries to identify meaningful clusters
        - Every 10 entries: Frequent enough to prevent excessive fragmentation,
          rare enough to avoid constant consolidation overhead
        - Not on every add: Consolidation is O(n²) for similarity matrix,
          so we batch it to amortize cost

        Expected behavior on 46-entry memory:
        - First trigger: 40 entries (consolidates to ~20-25)
        - Second trigger: Won't hit 50 if consolidation worked
        - Result: Self-regulating memory that stays consolidated
        """
        current_count = len(self.entries)

        # Only trigger if we've added 10+ entries since last consolidation
        if current_count < 30:
            return

        entries_since_last = current_count - self.last_consolidation_count
        if entries_since_last < 10:
            return

        logger.info(f"Automatic consolidation triggered at {current_count} entries ({entries_since_last} new since last)")

        try:
            merged_count = self.consolidate_memory(similarity_threshold=0.85)

            if merged_count > 0:
                logger.info(f"Automatic consolidation successful: merged {merged_count} entries")
                self.last_consolidation_count = len(self.entries)  # Update after consolidation
            else:
                logger.debug("Automatic consolidation found no clusters to merge")

        except Exception as e:
            # Don't fail the add_reasoning() call if consolidation fails
            logger.error(f"Automatic consolidation failed: {e}", exc_info=True)

    def _prune_low_quality(self):
        """
        Natural selection pruning - no hard caps, only quality.

        Removes entries that have proven themselves worthless:
        1. Below min_confidence AND never used successfully
        2. Have more failures than successes (demonstrated negative value)
        3. Confidence has decayed below min_confidence

        Good patterns survive and grow, bad patterns naturally die.
        """
        # Track entries before pruning
        before_hashes = {e.source_hash for e in self.entries}

        # Remove entries that have proven worthless
        self.entries = [
            e for e in self.entries
            if not (
                # Low confidence and never proven useful
                (e.confidence < self.min_confidence and e.use_count == 0)
                or
                # Demonstrated negative value
                (e.failure_signals > 0 and e.success_signals < e.failure_signals)
            )
        ]

        # Track entries after pruning
        after_hashes = {e.source_hash for e in self.entries}

        # Remove pruned entries from LSH index
        pruned_hashes = before_hashes - after_hashes
        if pruned_hashes:
            logger.info(f"Pruned {len(pruned_hashes)} low-quality entries")
            for source_hash in pruned_hashes:
                try:
                    self.lsh.remove(source_hash)
                    logger.debug(f"Removed entry {source_hash} from LSH index")
                except KeyError:
                    # Entry wasn't in LSH index, that's fine
                    pass

    def _find_similar(
        self,
        pattern: str,
        context: str,
        threshold: float = 0.8
    ) -> Optional[ReasoningEntry]:
        """
        Find similar existing entry using LSH.

        Args:
            pattern: Pattern to search for
            context: Context to search for
            threshold: Similarity threshold (uses instance threshold if not provided)

        Returns:
            Most similar entry above threshold, or None
        """
        try:
            # Create query MinHash
            query_text = pattern + " " + context
            query_minhash = self._create_minhash(query_text)

            # Query LSH index
            candidates = self.lsh.query(query_minhash)

            if not candidates:
                return None

            # Find best match from candidates
            best_entry = None
            best_similarity = 0.0

            for source_hash in candidates:
                # Find entry with this source_hash
                for entry in self.entries:
                    if entry.source_hash == source_hash:
                        # Calculate exact similarity
                        entry_minhash = self._get_or_create_minhash(entry)
                        similarity = query_minhash.jaccard(entry_minhash)

                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_entry = entry
                        break

            # Return best match if above threshold
            if best_entry and best_similarity >= threshold:
                return best_entry

            return None

        except ValueError as e:
            logger.warning(f"Error finding similar entry: {e}")
            return None

    def _hash_content(self, content: str) -> str:
        """Create hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _create_minhash(self, text: str) -> MinHash:
        """
        Create MinHash signature from text.

        Args:
            text: Text to hash

        Returns:
            MinHash signature

        Raises:
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Cannot create MinHash from empty text")

        minhash = MinHash(num_perm=128)
        # Tokenize by whitespace
        tokens = text.split()
        for token in tokens:
            minhash.update(token.encode('utf-8'))

        return minhash

    def _get_or_create_minhash(self, entry: ReasoningEntry) -> MinHash:
        """
        Get cached MinHash signature or create new one.

        Args:
            entry: ReasoningEntry to get/create signature for

        Returns:
            MinHash signature
        """
        # Check if signature is cached
        if 'minhash_signature' in entry.codebase_context:
            # Reconstruct MinHash from cached data
            import numpy as np
            minhash = MinHash(num_perm=128)
            minhash.hashvalues = np.array(entry.codebase_context['minhash_signature'], dtype=np.uint64)
            return minhash

        # Create new signature
        try:
            text = entry.pattern + " " + entry.context
            minhash = self._create_minhash(text)

            # Cache it (convert numpy uint64 to Python int for JSON serialization)
            entry.codebase_context['minhash_signature'] = [int(x) for x in minhash.hashvalues]
            logger.debug(f"Generated MinHash for entry with source_hash {entry.source_hash}")

            return minhash
        except ValueError as e:
            logger.warning(f"Failed to create MinHash for entry {entry.source_hash}: {e}")
            raise

    def _rebuild_lsh_index(self):
        """
        Rebuild LSH index from all entries.

        Uses source_hash as stable keys.
        """
        self.lsh = MinHashLSH(threshold=self.similarity_threshold, num_perm=128)

        for entry in self.entries:
            try:
                minhash = self._get_or_create_minhash(entry)
                self.lsh.insert(entry.source_hash, minhash)
            except ValueError as e:
                # Duplicate hash - regenerate unique one
                import time
                entry.source_hash = self._hash_content(entry.pattern + entry.context + str(time.time()))
                try:
                    self.lsh.insert(entry.source_hash, minhash)
                except ValueError:
                    pass  # Still duplicate, skip silently
                continue

    def _record_failure_signal(self, problem_hash: str, weight: float = 1.0):
        """Record that a reasoning likely failed."""
        for entry in self.entries:
            if entry.source_hash == problem_hash:
                entry.failure_signals += weight

    def _record_success_signal(self, problem_hash: str, weight: float = 1.0):
        """Record that a reasoning likely succeeded."""
        for entry in self.entries:
            if entry.source_hash == problem_hash:
                entry.success_signals += weight

    def _extract_pattern(self, context: dict) -> str:
        """Extract pattern description from context."""
        prompt = context.get("prompt", "")
        task_type = context.get("task_type", "")
        return f"{task_type}:{prompt[:100]}"

    def _extract_codebase_context(self, context: dict) -> dict:
        """Extract codebase-specific context."""
        files = context.get("context_files", [])

        return {
            "file_patterns": [f.get("path", "") for f in files[:5]],
            "working_dir": context.get("working_directory", ""),
        }

    def _extract_failure_root_cause(self, context: dict, suggestion: str) -> list[str]:
        """
        Extract root cause of failure using LLM-as-judge (Phase 3: Failure Learning).

        ReasoningBank insight: Don't just decrement confidence, extract WHY it failed.
        This creates contrastive signals when same archetype has successes + failures.

        Args:
            context: Problem context with optional error_trace
            suggestion: The suggestion that failed

        Returns:
            List of failure patterns (max 3)
            Examples:
              - "Edge case: empty input not handled"
              - "Performance: O(n²) timeout on n>10k"
              - "Logic error: off-by-one in loop boundary"
        """
        # Only extract if we have error signals
        error_trace = context.get("error_trace", "")
        if not error_trace:
            return []

        # Graceful degradation: skip if no LLM available
        if not self.openai_client:
            # Fallback: extract simple patterns from error trace
            return self._extract_failure_heuristics(error_trace)

        # Quick prompt to LLM (use OpenAI client)
        prompt = f"""Analyze this failure and extract 1-3 root causes as bullet points.

ERROR TRACE:
{error_trace[:500]}

SUGGESTION THAT FAILED:
{suggestion[:200]}

Output format (1-3 lines only):
- Root cause 1
- Root cause 2
- Root cause 3

Focus on: edge cases, algorithm choice, complexity, logic errors."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",  # Fast, cheap for extraction
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
            )

            # Parse bullet points
            content = response.choices[0].message.content
            causes = [
                line.strip("- ").strip()
                for line in content.split("\n")
                if line.strip().startswith("-")
            ]
            return causes[:3]  # Max 3

        except Exception as e:
            logger.warning(f"Failure extraction failed: {e}")
            # Fallback to heuristics
            return self._extract_failure_heuristics(error_trace)

    def _extract_failure_heuristics(self, error_trace: str) -> list[str]:
        """
        Fallback: Extract failure patterns using heuristics (no LLM).

        Looks for common error patterns in trace:
        - TimeoutError → "Performance: timeout"
        - IndexError/KeyError → "Edge case: missing bounds check"
        - AssertionError → "Logic error: incorrect result"
        """
        causes = []
        error_lower = error_trace.lower()

        # Performance issues
        if "timeout" in error_lower or "time limit" in error_lower:
            causes.append("Performance: timeout on large input")

        # Edge cases
        if "indexerror" in error_lower or "out of range" in error_lower:
            causes.append("Edge case: array index out of bounds")
        if "keyerror" in error_lower or "not found" in error_lower:
            causes.append("Edge case: missing key/value")
        if "empty" in error_lower or "null" in error_lower or "none" in error_lower:
            causes.append("Edge case: empty or null input")

        # Logic errors
        if "assertionerror" in error_lower or "wrong answer" in error_lower:
            causes.append("Logic error: incorrect algorithm or calculation")

        return causes[:3]  # Max 3

    def _calculate_relevance(
        self,
        entry: ReasoningEntry,
        current_pattern: str,
        current_codebase: dict
    ) -> float:
        """Calculate how relevant an entry is to current problem."""
        relevance = 0.0

        # Semantic similarity (TF-IDF)
        if self.tfidf_enabled:
            try:
                from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

                # Find entry index
                entry_idx = self.entries.index(entry)

                # Transform query
                query_vec = self.tfidf_vectorizer.transform([current_pattern])
                entry_vec = self.tfidf_matrix[entry_idx]

                # Cosine similarity
                similarity = sklearn_cosine(query_vec, entry_vec)[0][0]
                relevance += similarity * 0.5
            except Exception as e:
                logger.warning(f"TF-IDF scoring failed: {e}")
                # Fallback to substring
                if any(word in current_pattern.lower()
                       for word in entry.pattern.lower().split()):
                    relevance += 0.5
        else:
            # Fallback to substring matching
            if any(word in current_pattern.lower()
                   for word in entry.pattern.lower().split()):
                relevance += 0.5

        # Codebase match (UNCHANGED - orthogonal to semantics)
        current_files = set(current_codebase.get("file_patterns", []))
        entry_files = set(entry.codebase_context.get("file_patterns", []))

        if current_files & entry_files:  # Overlap
            relevance += 0.3

        # Context similarity (TF-IDF)
        if self.tfidf_enabled:
            try:
                from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

                entry_idx = self.entries.index(entry)
                context_text = current_pattern
                context_vec = self.tfidf_vectorizer.transform([context_text])
                entry_vec = self.tfidf_matrix[entry_idx]
                context_similarity = sklearn_cosine(context_vec, entry_vec)[0][0]
                relevance += context_similarity * 0.2
            except Exception:
                # Fallback
                if any(word in current_pattern.lower()
                       for word in entry.context.lower().split()):
                    relevance += 0.2
        else:
            # Fallback to substring
            if any(word in current_pattern.lower()
                   for word in entry.context.lower().split()):
                relevance += 0.2

        return min(1.0, relevance)

    def _find_semantic_clusters(
        self,
        entries: list[ReasoningEntry],
        similarity_threshold: float
    ) -> list[list[int]]:
        """
        Find clusters of semantically similar entries using FAISS-accelerated nearest neighbors.

        Performance: O(n log n) with FAISS vs O(n²) brute force
        - 200 entries: <1s (was ~10s)
        - 500 entries: ~2s (was ~90s)
        - 1000 entries: ~5s (was ~6min)

        Args:
            entries: List of ReasoningEntry objects (must have embeddings)
            similarity_threshold: Cosine similarity threshold for clustering (0.85 default)

        Returns:
            List of clusters, where each cluster is list of entry indices
        """
        # Group entries by embedding model to prevent mixing incompatible embeddings
        by_model = {}
        for i, entry in enumerate(entries):
            model = entry.embedding_model or "unknown"
            if model not in by_model:
                by_model[model] = []
            by_model[model].append((i, entry))

        # Log if multiple models detected
        if len(by_model) > 1:
            logger.warning(f"Multiple embedding models detected: {list(by_model.keys())}")
            logger.warning("Entries will be clustered separately by model")

        # Cluster each model group separately
        all_clusters = []
        for model, model_entries in by_model.items():
            indices = [i for i, _ in model_entries]
            just_entries = [e for _, e in model_entries]

            logger.debug(f"Clustering {len(just_entries)} entries with model '{model}'")

            # Use existing FAISS or bruteforce clustering
            if FAISS_AVAILABLE and len(just_entries) > 50:
                clusters = self._find_semantic_clusters_faiss(just_entries, similarity_threshold)
            else:
                clusters = self._find_semantic_clusters_bruteforce(just_entries, similarity_threshold)

            # Map local indices back to global indices
            for cluster in clusters:
                global_cluster = [indices[local_idx] for local_idx in cluster]
                all_clusters.append(global_cluster)

        return all_clusters

    def _find_semantic_clusters_faiss(
        self,
        entries: list[ReasoningEntry],
        similarity_threshold: float
    ) -> list[list[int]]:
        """
        FAISS-based clustering using approximate nearest neighbors.

        Single-linkage clustering means: if A is similar to B, and B is similar to C,
        then A, B, C all go into the same cluster (transitive closure).

        Handles mixed embedding dimensions (1536 vs 384) by clustering separately.

        Args:
            entries: List of ReasoningEntry objects (must have embeddings)
            similarity_threshold: Cosine similarity threshold for edges

        Returns:
            List of clusters, where each cluster is a list of entry indices
        """
        n = len(entries)
        if n < 2:
            return [[0]] if n == 1 else []

        # Group entries by embedding dimension
        entries_by_dim = {}
        for i, entry in enumerate(entries):
            dim = entry.embedding.shape[0]
            if dim not in entries_by_dim:
                entries_by_dim[dim] = []
            entries_by_dim[dim].append(i)

        # Cluster each dimension group separately
        all_clusters = []

        for dim, indices in entries_by_dim.items():
            if len(indices) < 2:
                # Single entry, create singleton cluster
                all_clusters.append(indices)
                continue

            logger.debug(f"Clustering {len(indices)} entries with dimension {dim}")

            # Extract embeddings for this dimension group
            subset_entries = [entries[i] for i in indices]
            embeddings = np.array([e.embedding for e in subset_entries], dtype=np.float32)

            # Normalize vectors for cosine similarity
            faiss.normalize_L2(embeddings)

            # Build FAISS index
            index = faiss.IndexFlatIP(dim)

            # Move to GPU if available
            if FAISS_GPU_AVAILABLE:
                gpu_index = faiss.index_cpu_to_all_gpus(index)
                gpu_index.add(embeddings)
                index = gpu_index
            else:
                index.add(embeddings)

            # Search for neighbors
            k = len(indices)
            distances, search_indices = index.search(embeddings, k=k)

            # Build adjacency graph (using local indices)
            graph = {i: set() for i in range(len(indices))}
            for i in range(len(indices)):
                for j, dist in zip(search_indices[i], distances[i]):
                    if dist >= similarity_threshold and i != j:
                        graph[i].add(j)

            # DFS for connected components
            visited = set()
            dim_clusters = []

            def dfs(node, component):
                visited.add(node)
                component.append(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        dfs(neighbor, component)

            for node in range(len(indices)):
                if node not in visited:
                    component = []
                    dfs(node, component)
                    if len(component) > 0:
                        # Map local indices back to global indices
                        global_component = [indices[local_i] for local_i in component]
                        dim_clusters.append(global_component)

            all_clusters.extend(dim_clusters)

        logger.info(f"FAISS clustering: {n} entries → {len(all_clusters)} clusters")
        return all_clusters

    def _find_semantic_clusters_bruteforce(
        self,
        entries: list[ReasoningEntry],
        similarity_threshold: float
    ) -> list[list[int]]:
        """
        Fallback O(n²) clustering when FAISS unavailable.

        Single-linkage clustering means: if A is similar to B, and B is similar to C,
        then A, B, C all go into the same cluster (transitive closure).

        This is appropriate for semantic clustering because conceptually related patterns
        should merge even if not all pairs are directly similar. Example:
        - "two-pointer sorted array" ~ "left-right pointers on sorted data"
        - "left-right pointers on sorted data" ~ "opposite ends technique"
        - Result: all three cluster together as "two-pointer archetype"

        Args:
            entries: List of ReasoningEntry objects (must have embeddings)
            similarity_threshold: Cosine similarity threshold for edges

        Returns:
            List of clusters, where each cluster is a list of entry indices
            Example: [[0, 3, 7], [1, 2], [4], [5, 6, 8, 9]]
                    Means: entries 0,3,7 form a cluster, 1,2 form a cluster, etc.

        Algorithm:
            1. Build similarity graph: edge between i,j if cosine_sim(i,j) > threshold
            2. Find connected components (DFS/BFS on undirected graph)
            3. Return components as list of index lists
        """
        n = len(entries)
        if n < 2:
            return [[0]] if n == 1 else []

        # Step 1: Build adjacency list (similarity graph) - O(n²)
        graph = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                similarity = self._cosine_similarity(entries[i].embedding, entries[j].embedding)

                if similarity >= similarity_threshold:
                    # Add edge (undirected)
                    graph[i].add(j)
                    graph[j].add(i)
                    logger.debug(f"Cluster edge: entries {i},{j} (similarity={similarity:.3f})")

        # Step 2: Find connected components using iterative DFS
        # (Linus: Iterative to avoid RecursionError with Python's 1000-call limit)
        visited = set()
        clusters = []

        def dfs_iterative(start):
            """Iterative depth-first search to find connected component."""
            stack = [start]
            component = []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                stack.extend(n for n in graph[node] if n not in visited)
            return component

        for node in range(n):
            if node not in visited:
                component = dfs_iterative(node)
                clusters.append(component)

        logger.warning(f"Using O(n²) clustering fallback: {n} entries → {len(clusters)} clusters (install faiss-cpu for better performance)")
        return clusters

    def _cluster_entries(self) -> list[list[ReasoningEntry]]:
        """
        Legacy MinHash-based clustering for backward compatibility.

        Used by _consolidate_minhash() fallback when embeddings unavailable.
        """
        # Simple clustering: group by pattern prefix
        clusters = {}

        for entry in self.entries:
            prefix = entry.pattern[:20]  # First 20 chars
            if prefix not in clusters:
                clusters[prefix] = []
            clusters[prefix].append(entry)

        return [c for c in clusters.values() if len(c) > 1]

    def _calculate_contrastive_boost(self, entry: ReasoningEntry, cluster: list[ReasoningEntry]) -> float:
        """
        Calculate contrastive confidence boost based on comparative performance (Phase 4: Self-Contrast).

        ReasoningBank insight: Patterns that succeed WHERE OTHERS FAIL are archetypal.
        Patterns that succeed once but fail elsewhere are spurious.

        Args:
            entry: The entry to evaluate
            cluster: All entries in the same semantic cluster

        Returns:
            Confidence boost in range [-0.2, +0.2]
        """
        # Get problems this entry solved
        solved = {h for h, success in entry.problem_outcomes.items() if success}
        failed = {h for h, success in entry.problem_outcomes.items() if not success}

        if not solved and not failed:
            return 0.0  # No data

        # Compare against cluster peers
        contrast_wins = 0
        contrast_losses = 0

        for other in cluster:
            if other == entry:
                continue

            # Check shared problems
            shared_problems = (solved | failed) & (set(other.problem_outcomes.keys()))

            for problem_hash in shared_problems:
                this_success = entry.problem_outcomes.get(problem_hash, False)
                other_success = other.problem_outcomes.get(problem_hash, False)

                if this_success and not other_success:
                    contrast_wins += 1  # This pattern won where other failed
                elif not this_success and other_success:
                    contrast_losses += 1  # Other pattern won where this failed

        # Calculate boost
        total_contrasts = contrast_wins + contrast_losses
        if total_contrasts == 0:
            return 0.0

        # Scale: 100% win rate → +0.2, 0% → -0.2
        contrast_ratio = contrast_wins / total_contrasts
        boost = (contrast_ratio - 0.5) * self.CONTRASTIVE_SCALE  # Maps [0,1] to [-0.2, +0.2]

        logger.info(f"Contrastive boost for '{entry.pattern[:50]}': {contrast_wins}/{total_contrasts} = {boost:+.3f}")
        return boost

    def _merge_cluster(self, entries: list[ReasoningEntry]) -> ReasoningEntry:
        """
        Merge multiple semantically similar entries into a single archetype.

        This is the Phase 3 merging logic that combines knowledge from multiple
        entries while preserving the best reasoning and code patterns.

        Args:
            entries: List of ReasoningEntry objects to merge (must be similar)

        Returns:
            Merged ReasoningEntry archetype

        Merge strategy:
            - Pattern/reasoning/suggestion: Use highest-confidence entry (best performer)
            - Code skeleton: Use longest version (most detailed implementation)
            - Pitfalls/tests: Union of all (accumulated wisdom)
            - Confidence: Weighted average by success signals
            - Statistics: Sum across all entries
            - Algorithm category: Auto-detect from merged content
            - Example problems: Track all original patterns
        """
        if not entries:
            raise ValueError("Cannot merge empty list of entries")

        if len(entries) == 1:
            return entries[0]

        # Phase 4: Apply contrastive boosts BEFORE selecting best entry
        # This ensures archetypal patterns (wins where others fail) get prioritized
        for entry in entries:
            contrastive_boost = self._calculate_contrastive_boost(entry, entries)
            entry.confidence = max(0.0, min(1.0, entry.confidence + contrastive_boost))

        # Use highest-confidence entry as base (NOW includes contrastive boost)
        best = max(entries, key=lambda e: e.confidence)

        # Detect algorithm category from all merged content
        all_text = " ".join([
            e.pattern + " " + e.context + " " + e.reasoning
            for e in entries
        ])
        algorithm_category = self._detect_algorithm_category(all_text)

        # Extract common theme from patterns for archetype name
        # Simple heuristic: use the category if detected, otherwise use best pattern
        if algorithm_category and algorithm_category != "general":
            archetype_pattern = f"{algorithm_category}: {best.pattern}"
        else:
            archetype_pattern = f"Archetype: {best.pattern}"

        # Combine code skeletons: use longest (most detailed)
        code_skeleton = max(
            [e.code_skeleton for e in entries if e.code_skeleton],
            key=len,
            default=best.code_skeleton
        )

        # Union of pitfalls (deduplicate by lowercase comparison)
        all_pitfalls = []
        seen_pitfalls = set()
        for entry in entries:
            for pitfall in entry.common_pitfalls:
                pitfall_key = pitfall.lower().strip()
                if pitfall_key and pitfall_key not in seen_pitfalls:
                    all_pitfalls.append(pitfall)
                    seen_pitfalls.add(pitfall_key)

        # Union of test patterns (deduplicate by lowercase comparison)
        all_tests = []
        seen_tests = set()
        for entry in entries:
            for test in entry.test_patterns:
                test_key = test.lower().strip()
                if test_key and test_key not in seen_tests:
                    all_tests.append(test)
                    seen_tests.add(test_key)

        # Weighted average confidence (weighted by success signals)
        total_success = sum(e.success_signals for e in entries)
        if total_success > 0:
            weighted_confidence = sum(
                e.confidence * e.success_signals for e in entries
            ) / total_success
        else:
            # Fallback to simple average if no success signals
            weighted_confidence = sum(e.confidence for e in entries) / len(entries)

        # Track original patterns in example_problems
        example_problems = []
        for entry in entries:
            # Add this entry's pattern as an example
            example_problems.append(entry.pattern)
            # Also include any examples it already had
            if entry.example_problems:
                example_problems.extend(entry.example_problems)

        # Deduplicate example problems
        example_problems = list(dict.fromkeys(example_problems))  # Preserves order

        # Merge difficulty affinity (Phase 4)
        merged_affinity: dict[str, tuple[int, int]] = {}
        for entry in entries:
            for diff, (succ, tot) in entry.difficulty_affinity.items():
                if diff in merged_affinity:
                    old_succ, old_tot = merged_affinity[diff]
                    merged_affinity[diff] = (old_succ + succ, old_tot + tot)
                else:
                    merged_affinity[diff] = (succ, tot)

        # Re-embed the merged archetype
        merged_text = f"{archetype_pattern} {best.context} {best.reasoning}"
        merged_embedding = self._embed_text(merged_text)

        # Create merged archetype
        merged = ReasoningEntry(
            # Core identity (from best)
            pattern=archetype_pattern,
            context=best.context,
            reasoning=best.reasoning,
            suggestion=best.suggestion,

            # Phase 2 fields (combined knowledge)
            code_skeleton=code_skeleton,
            common_pitfalls=all_pitfalls[:self.MAX_PITFALLS_PER_ENTRY],
            test_patterns=all_tests[:self.MAX_TESTS_PER_ENTRY],
            algorithm_category=algorithm_category,
            merge_count=sum(e.merge_count for e in entries) + len(entries),  # Total merges

            # Legacy algorithm fields (from best)
            algorithm_type=best.algorithm_type or algorithm_category,
            code_template=best.code_template or code_skeleton[:200],
            time_complexity=best.time_complexity,
            space_complexity=best.space_complexity,
            when_to_use=best.when_to_use,
            example_problems=example_problems[:self.MAX_EXAMPLES_PER_ENTRY],

            # Statistics (summed)
            confidence=min(0.95, weighted_confidence),  # Cap at 0.95
            use_count=sum(e.use_count for e in entries),
            success_signals=sum(e.success_signals for e in entries),
            failure_signals=sum(e.failure_signals for e in entries),

            # Temporal tracking
            created_at=min(e.created_at for e in entries),  # Earliest creation
            last_used=max(e.last_used for e in entries),  # Most recent use

            # Metadata (from best)
            source_hash=best.source_hash,
            codebase_context=best.codebase_context,
            contextual_stats=best.contextual_stats,

            # Difficulty affinity (Phase 4)
            difficulty_affinity=merged_affinity,

            # Embedding (re-embedded)
            embedding=merged_embedding,
        )

        # Validate merged entry
        if merged.embedding is None:
            logger.error(f"Merged archetype has no embedding, using best entry")
            return best  # Fallback to best entry

        if not (0 <= merged.confidence <= 1.0):
            logger.warning(f"Invalid confidence {merged.confidence}, clamping to [0, 1]")
            merged.confidence = max(0.0, min(1.0, merged.confidence))

        if len(merged.common_pitfalls) > self.MAX_PITFALLS_PER_ENTRY:
            merged.common_pitfalls = merged.common_pitfalls[:self.MAX_PITFALLS_PER_ENTRY]

        if len(merged.test_patterns) > self.MAX_TESTS_PER_ENTRY:
            merged.test_patterns = merged.test_patterns[:self.MAX_TESTS_PER_ENTRY]

        if len(merged.example_problems) > self.MAX_EXAMPLES_PER_ENTRY:
            merged.example_problems = merged.example_problems[:self.MAX_EXAMPLES_PER_ENTRY]

        return merged

    def _detect_algorithm_category(self, text: str) -> str:
        """
        Detect common algorithm category from entry patterns.

        Uses keyword matching on ALGORITHM_PATTERNS class constant.
        Not ML-based for determinism and inspectability.

        Args:
            text: Combined text from pattern + context + reasoning

        Returns:
            Algorithm category string (e.g., "two-pointer", "dp", "graph")
            Returns "general" if no specific category detected

        Design decision: Returns FIRST matching category, not all.
        Rationale: Single primary category is cleaner for archetype naming.
        If multiple patterns apply (e.g., "graph + dp"), the dominant one
        (first match) becomes the category, others are preserved in reasoning.
        """
        text_lower = text.lower()

        # Check each category in order (most specific to least specific)
        for category, keywords in self.ALGORITHM_PATTERNS.items():
            if any(keyword in text_lower for keyword in keywords):
                logger.debug(f"Detected algorithm category: {category}")
                return category

        return "general"

    def _consolidate_minhash(self) -> int:
        """
        MinHash-based consolidation fallback for when embeddings unavailable.

        Uses the original consolidation logic based on Jaccard similarity.
        This is less effective than semantic consolidation but maintains
        backward compatibility when OpenAI API is unavailable.

        Returns:
            Number of entries merged
        """
        if len(self.entries) < 10:
            return 0

        original_count = len(self.entries)

        # Group by similar patterns using MinHash
        clusters = self._cluster_entries()

        for cluster in clusters:
            if len(cluster) >= 3:
                # Merge into a generalized entry
                merged = self._merge_entries(cluster)

                # Remove originals
                for entry in cluster:
                    self.entries.remove(entry)

                # Add merged
                self.entries.append(merged)

        self.save()

        merged_count = original_count - len(self.entries)
        if merged_count > 0:
            logger.info(f"MinHash consolidation: {original_count} → {len(self.entries)} entries ({merged_count} merged)")

        return merged_count

    def _merge_entries(self, entries: list[ReasoningEntry]) -> ReasoningEntry:
        """
        Legacy MinHash-based merge for backward compatibility.

        Used by _consolidate_minhash() fallback when embeddings unavailable.
        """
        # Take highest confidence as base
        best = max(entries, key=lambda e: e.confidence)

        # Combine statistics
        merged = ReasoningEntry(
            pattern=f"General: {best.pattern}",
            context=best.context,
            reasoning=best.reasoning,
            suggestion=best.suggestion,
            confidence=sum(e.confidence for e in entries) / len(entries),
            use_count=sum(e.use_count for e in entries),
            success_signals=sum(e.success_signals for e in entries),
            failure_signals=sum(e.failure_signals for e in entries),
            created_at=min(e.created_at for e in entries),
            last_used=max(e.last_used for e in entries),
            source_hash=best.source_hash,
            codebase_context=best.codebase_context
        )

        return merged

    def save(self):
        """Save to both global and local storage."""
        # Separate entries by scope
        global_entries = []
        local_entries = []

        for entry in self.entries:
            # Check if entry is codebase-specific
            if self._is_codebase_specific(entry):
                local_entries.append(entry)
            else:
                global_entries.append(entry)

        # Save using storage backend
        entry_dicts = [e.to_dict() for e in global_entries]
        self.storage_backend.save_entries(self.global_storage_key, entry_dicts)

        if self.local_storage_key and local_entries:
            local_dicts = [e.to_dict() for e in local_entries]
            self.storage_backend.save_entries(self.local_storage_key, local_dicts)

    def load(self):
        """Load from both global and local storage."""
        self.entries = []

        # Load using storage backend
        entry_dicts = self.storage_backend.load_entries(self.global_storage_key)
        self.entries.extend([ReasoningEntry.from_dict(e) for e in entry_dicts])

        if self.local_storage_key:
            local_dicts = self.storage_backend.load_entries(self.local_storage_key)
            self.entries.extend([ReasoningEntry.from_dict(e) for e in local_dicts])

        # Rebuild LSH index after loading all entries
        self._rebuild_lsh_index()

        # Rebuild TF-IDF after loading all entries
        if self.entries:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer

                corpus = [entry.pattern + " " + entry.context for entry in self.entries]
                if hasattr(self, 'tfidf_vectorizer') and self.tfidf_vectorizer is not None:
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
                    self.tfidf_enabled = True
                    logger.info(f"TF-IDF refitted with {len(corpus)} entries")
            except ImportError:
                pass  # Already logged in __init__
            except Exception as e:
                logger.warning(f"TF-IDF refit failed: {e}")
                self.tfidf_enabled = False

    def _is_codebase_specific(self, entry: ReasoningEntry) -> bool:
        """Determine if an entry is codebase-specific or general."""
        # Entry is codebase-specific if:
        # 1. It has codebase_context with specific file paths
        # 2. It matches the current codebase_root

        if not self.codebase_root or not entry.codebase_context:
            return False

        # Check if entry's files are in current codebase
        entry_files = entry.codebase_context.get("file_patterns", [])
        working_dir = entry.codebase_context.get("working_dir", "")

        if working_dir:
            try:
                entry_path = Path(working_dir).resolve()
                # Entry is local if it's in or related to current codebase
                return str(entry_path).startswith(str(self.codebase_root))
            except Exception:
                pass

        return False

    def memory_level(self) -> float:
        """
        Calculate memory level as a float from 0.0 to 1.0 using sigmoid scaling.

        Based on quality-weighted usage rather than raw count:
        - Weights high-confidence, frequently-used entries more
        - Returns 0.0 for empty memory
        - Approaches 1.0 asymptotically (infinite mastery)
        - Uses shifted sigmoid: level = 1 - 1/(1 + total_quality/reference_quality)

        With reference_quality=500:
        - 0 quality → 0.0
        - 250 quality → 0.33
        - 500 quality → 0.5
        - 1000 quality → 0.67
        - 2000 quality → 0.8 ("The One")
        - 5000 quality → 0.91
        - ∞ quality → 1.0 (asymptotic)
        """
        if not self.entries:
            return 0.0

        # Calculate quality-weighted memory usage
        # Each entry contributes based on its score (confidence * success rate * recency * usage)
        total_quality = sum(e.score() for e in self.entries)

        # Sigmoid scaling: approaches 1.0 asymptotically
        level = 1.0 - 1.0 / (1.0 + total_quality / self.reference_quality)

        return level

    def stats(self) -> dict:
        """Get statistics about the memory."""
        if not self.entries:
            return {"total": 0, "memory_level": 0.0}

        return {
            "total": len(self.entries),
            "memory_level": self.memory_level(),
            "avg_confidence": sum(e.confidence for e in self.entries) / len(self.entries),
            "avg_score": sum(e.score() for e in self.entries) / len(self.entries),
            "total_uses": sum(e.use_count for e in self.entries),
            "success_rate": (
                sum(e.success_signals for e in self.entries) /
                max(1, sum(e.success_signals + e.failure_signals for e in self.entries))
            )
        }