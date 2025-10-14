"""
Vector-based exemplar index for storing and retrieving similar past tasks.

Uses TF-IDF for simple text similarity without requiring heavy dependencies.
Can be upgraded to use sentence transformers for better semantic search.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Optional, Union

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# Exemplar Data Structure
# ============================================================================

class Exemplar:
    """An exemplar representing a past task."""

    def __init__(
        self,
        prompt: str,
        solution: str,
        task_type: str,
        metadata: dict[str, Any],
    ):
        self.prompt = prompt
        self.solution = solution
        self.task_type = task_type
        self.metadata = metadata

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "prompt": self.prompt,
            "solution": self.solution,
            "task_type": self.task_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Exemplar":
        """Create from dictionary."""
        return cls(
            prompt=data["prompt"],
            solution=data["solution"],
            task_type=data.get("task_type", "unknown"),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Vector-based Exemplar Index
# ============================================================================

class ExemplarIndex:
    """
    Vector index for storing and retrieving similar exemplars.

    Uses TF-IDF + cosine similarity for fast retrieval without heavy dependencies.
    Can be upgraded to use sentence transformers for better semantic search.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize exemplar index.

        Args:
            storage_path: Path to store the index (default: ~/.neo/exemplars.pkl)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "sklearn required for exemplar indexing. "
                "Install with: pip install scikit-learn numpy"
            )

        self.exemplars: list[Exemplar] = []
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english',
        )
        self.vectors = None

        # Storage
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".neo" / "exemplars.pkl"

        # Load existing exemplars if available
        self.load()

    def add(
        self,
        prompt: str,
        solution: str,
        task_type: str = "unknown",
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Add an exemplar to the index.

        Args:
            prompt: The task prompt/description
            solution: The solution (plan, code, explanation)
            task_type: Type of task (algorithm, refactor, bugfix, etc.)
            metadata: Additional metadata (language, confidence, etc.)
        """
        exemplar = Exemplar(
            prompt=prompt,
            solution=solution,
            task_type=task_type,
            metadata=metadata or {},
        )

        self.exemplars.append(exemplar)

        # Rebuild vectors
        self._rebuild_vectors()

    def search(
        self,
        query: str,
        k: int = 3,
        task_type: Optional[str] = None,
        min_similarity: float = 0.1,
    ) -> list[Exemplar]:
        """
        Search for similar exemplars.

        Args:
            query: The query text (user prompt)
            k: Number of results to return
            task_type: Optional filter by task type
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of similar exemplars, sorted by similarity
        """
        if not self.exemplars:
            return []

        # Filter by task type if specified
        candidates = self.exemplars
        if task_type:
            candidates = [e for e in self.exemplars if e.task_type == task_type]

        if not candidates:
            return []

        # Vectorize query
        query_vector = self.vectorizer.transform([query])

        # Get vectors for candidates
        if task_type:
            # Need to rebuild vectors for filtered candidates
            candidate_prompts = [e.prompt for e in candidates]
            candidate_vectors = self.vectorizer.transform(candidate_prompts)
        else:
            candidate_vectors = self.vectors

        # Compute similarities
        similarities = cosine_similarity(query_vector, candidate_vectors)[0]

        # Get top-k indices above threshold
        sorted_indices = similarities.argsort()[::-1]
        results = []

        for idx in sorted_indices:
            if len(results) >= k:
                break
            if similarities[idx] >= min_similarity:
                results.append(candidates[idx])

        return results

    def save(self):
        """Save index to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "exemplars": [e.to_dict() for e in self.exemplars],
            "vectorizer": self.vectorizer,
        }

        with open(self.storage_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        """Load index from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'rb') as f:
                data = pickle.load(f)

            self.exemplars = [Exemplar.from_dict(e) for e in data["exemplars"]]
            self.vectorizer = data["vectorizer"]

            # Rebuild vectors
            self._rebuild_vectors()
        except Exception:
            # If loading fails, start fresh
            self.exemplars = []
            self.vectors = None

    def _rebuild_vectors(self):
        """Rebuild TF-IDF vectors from exemplars."""
        if not self.exemplars:
            self.vectors = None
            return

        prompts = [e.prompt for e in self.exemplars]

        # Fit vectorizer if not already fitted
        if not hasattr(self.vectorizer, 'vocabulary_') or self.vectorizer.vocabulary_ is None:
            self.vectors = self.vectorizer.fit_transform(prompts)
        else:
            self.vectors = self.vectorizer.transform(prompts)

    def clear(self):
        """Clear all exemplars."""
        self.exemplars = []
        self.vectors = None

    def __len__(self) -> int:
        """Return number of exemplars."""
        return len(self.exemplars)


# ============================================================================
# Simple Fallback Index (no sklearn)
# ============================================================================

class SimpleExemplarIndex:
    """
    Simple exemplar index without vector search.

    Falls back to keyword matching when sklearn is not available.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """Initialize simple index."""
        self.exemplars: list[Exemplar] = []

        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".neo" / "exemplars.json"

        self.load()

    def add(
        self,
        prompt: str,
        solution: str,
        task_type: str = "unknown",
        metadata: Optional[dict[str, Any]] = None,
    ):
        """Add an exemplar."""
        exemplar = Exemplar(
            prompt=prompt,
            solution=solution,
            task_type=task_type,
            metadata=metadata or {},
        )
        self.exemplars.append(exemplar)

    def search(
        self,
        query: str,
        k: int = 3,
        task_type: Optional[str] = None,
        min_similarity: float = 0.1,
    ) -> list[Exemplar]:
        """
        Simple keyword-based search.

        Counts matching words between query and exemplar prompts.
        """
        if not self.exemplars:
            return []

        # Filter by task type
        candidates = self.exemplars
        if task_type:
            candidates = [e for e in self.exemplars if e.task_type == task_type]

        if not candidates:
            return []

        # Tokenize query
        query_words = set(query.lower().split())

        # Score candidates
        scored = []
        for exemplar in candidates:
            prompt_words = set(exemplar.prompt.lower().split())
            overlap = len(query_words & prompt_words)
            score = overlap / max(len(query_words), len(prompt_words))

            if score >= min_similarity:
                scored.append((score, exemplar))

        # Sort by score and return top-k
        scored.sort(reverse=True, key=lambda x: x[0])
        return [exemplar for score, exemplar in scored[:k]]

    def save(self):
        """Save to JSON."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = [e.to_dict() for e in self.exemplars]

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load from JSON."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            self.exemplars = [Exemplar.from_dict(e) for e in data]
        except Exception:
            self.exemplars = []

    def clear(self):
        """Clear all exemplars."""
        self.exemplars = []

    def __len__(self) -> int:
        """Return number of exemplars."""
        return len(self.exemplars)


# ============================================================================
# Factory
# ============================================================================

def create_exemplar_index(
    storage_path: Optional[str] = None,
    use_vectors: bool = True,
) -> Union[ExemplarIndex, SimpleExemplarIndex]:
    """
    Create an exemplar index.

    Args:
        storage_path: Path to store index
        use_vectors: Use vector-based search if sklearn available

    Returns:
        ExemplarIndex or SimpleExemplarIndex
    """
    if use_vectors and SKLEARN_AVAILABLE:
        return ExemplarIndex(storage_path)
    else:
        return SimpleExemplarIndex(storage_path)