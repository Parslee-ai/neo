"""
The Construct - Semantic pattern library for Neo.

Manages a curated collection of architecture/design patterns with semantic search.
Patterns are stored as markdown files with structured metadata.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Import FAISS for semantic indexing
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Import fastembed for local embeddings (reuse from persistent_reasoning)
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PatternSchema:
    """Structured representation of a design pattern."""

    # Identity
    pattern_id: str  # e.g., "rate-limiting/token-bucket"
    name: str  # e.g., "Token Bucket"
    author: str  # GitHub username or name

    # Content sections
    intent: str
    forces: str
    solution: str
    consequences: str
    references: str = ""

    # Metadata
    domain: str = ""  # e.g., "rate-limiting", "caching"
    file_path: str = ""
    line_count: int = 0
    created_at: float = field(default_factory=time.time)

    # Embedding (optional, computed during indexing)
    embedding: Optional[np.ndarray] = None
    embedding_model: str = ""
    embedding_dim: int = 0

    def to_dict(self):
        """Convert to JSON-serializable dict (exclude embedding)."""
        d = asdict(self)
        d.pop('embedding', None)  # Don't serialize numpy array
        return d

    def to_text(self) -> str:
        """Convert to searchable text representation."""
        parts = [
            f"# {self.name}",
            f"Author: {self.author}",
            f"Intent: {self.intent}",
            f"Forces: {self.forces}",
            f"Solution: {self.solution}",
            f"Consequences: {self.consequences}",
        ]
        if self.references:
            parts.append(f"References: {self.references}")
        return "\n\n".join(parts)


class PatternReader:
    """Parse markdown pattern files into PatternSchema objects."""

    @staticmethod
    def load(file_path: Path) -> Optional[PatternSchema]:
        """
        Load pattern from markdown file.

        Expected format:
        ```markdown
        # Pattern: <Name>
        Author: <author>

        ## Intent
        ...

        ## Forces
        ...

        ## Solution Sketch
        ...

        ## Consequences
        ...

        ## References
        ...
        ```

        Args:
            file_path: Path to markdown file

        Returns:
            PatternSchema object or None if parsing fails
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            line_count = len(lines)

            # Extract pattern ID from file path
            # e.g., /construct/rate-limiting/token-bucket.md -> "rate-limiting/token-bucket"
            parts = file_path.parts
            if 'construct' in parts:
                idx = parts.index('construct')
                domain = parts[idx + 1] if idx + 1 < len(parts) else ""
                filename = file_path.stem
                pattern_id = f"{domain}/{filename}"
            else:
                pattern_id = file_path.stem
                domain = ""

            # Parse sections
            sections = {}
            current_section = None
            section_content = []

            name = ""
            author = ""

            for line in lines:
                line = line.rstrip()

                # Extract pattern name from title
                if line.startswith('# Pattern:'):
                    name = line.replace('# Pattern:', '').strip()
                    continue

                # Extract author
                if line.startswith('Author:'):
                    author = line.replace('Author:', '').strip()
                    continue

                # Section headers
                if line.startswith('## '):
                    # Save previous section
                    if current_section:
                        sections[current_section] = '\n'.join(section_content).strip()

                    # Start new section
                    current_section = line.replace('##', '').strip().lower()
                    # Normalize section names
                    if 'solution' in current_section:
                        current_section = 'solution'
                    section_content = []
                    continue

                # Accumulate content
                if current_section:
                    section_content.append(line)

            # Save last section
            if current_section:
                sections[current_section] = '\n'.join(section_content).strip()

            # Validate required fields
            if not name:
                logger.warning(f"Pattern missing name: {file_path}")
                return None

            if not author:
                logger.warning(f"Pattern missing author: {file_path}")
                return None

            required_sections = ['intent', 'forces', 'solution', 'consequences']
            missing = [s for s in required_sections if s not in sections]
            if missing:
                logger.warning(f"Pattern missing sections {missing}: {file_path}")
                return None

            return PatternSchema(
                pattern_id=pattern_id,
                name=name,
                author=author,
                intent=sections.get('intent', ''),
                forces=sections.get('forces', ''),
                solution=sections.get('solution', ''),
                consequences=sections.get('consequences', ''),
                references=sections.get('references', ''),
                domain=domain,
                file_path=str(file_path),
                line_count=line_count,
            )

        except Exception as e:
            logger.error(f"Failed to parse pattern {file_path}: {e}")
            return None


class PatternValidator:
    """Validate pattern files against quality standards."""

    MAX_LINE_COUNT = 300

    @staticmethod
    def validate(pattern: PatternSchema) -> list[str]:
        """
        Validate pattern against quality standards.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check author field
        if not pattern.author or pattern.author.strip() == "":
            errors.append("Author field is required")

        # Check line count
        if pattern.line_count > PatternValidator.MAX_LINE_COUNT:
            errors.append(f"Pattern exceeds {PatternValidator.MAX_LINE_COUNT} lines ({pattern.line_count} lines)")

        # Check required sections have content
        if len(pattern.intent) < 10:
            errors.append("Intent section is too short (minimum 10 characters)")

        if len(pattern.forces) < 10:
            errors.append("Forces section is too short (minimum 10 characters)")

        if len(pattern.solution) < 10:
            errors.append("Solution section is too short (minimum 10 characters)")

        if len(pattern.consequences) < 10:
            errors.append("Consequences section is too short (minimum 10 characters)")

        return errors


class ConstructIndex:
    """
    Manages semantic indexing and retrieval of design patterns.

    Reuses embedding infrastructure from PersistentReasoningMemory.

    Thread Safety:
        NOT thread-safe for concurrent build/search operations. Callers must
        serialize access if using from multiple threads. Internal locks prevent
        index corruption during writes but do not protect read operations.
    """

    def __init__(self, construct_root: Optional[Path] = None, config=None):
        """
        Initialize ConstructIndex.

        Args:
            construct_root: Path to /construct directory (defaults to repo root)
            config: Optional NeoConfig instance for embedding settings
        """
        # Locate construct directory
        if construct_root:
            self.construct_root = Path(construct_root)
        else:
            # Default to repo root + /construct
            # Try to find git root, fall back to cwd
            cwd = Path.cwd()
            if (cwd / 'construct').exists():
                self.construct_root = cwd / 'construct'
            elif (cwd.parent / 'construct').exists():
                self.construct_root = cwd.parent / 'construct'
            else:
                # Fall back to ~/.neo/construct for testing
                self.construct_root = Path.home() / '.neo' / 'construct'
                self.construct_root.mkdir(parents=True, exist_ok=True)

        # Index storage location
        self.index_dir = Path.home() / '.neo'
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / 'construct_index.faiss'
        self.metadata_path = self.index_dir / 'construct_metadata.json'

        # Initialize embedder (reuse fastembed like PersistentReasoningMemory)
        self.embedder = None
        self.embedding_dim = 0
        self.embedding_model = ""

        if FASTEMBED_AVAILABLE:
            try:
                # Use same model as persistent_reasoning for consistency
                self.embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")
                self.embedding_dim = 768  # Jina code model dimension
                self.embedding_model = "jinaai/jina-embeddings-v2-base-code"
                logger.info("Construct index using local embeddings (jina-embeddings-v2-base-code)")
            except Exception as e:
                logger.warning(f"Failed to initialize embedder: {e}")

        if not self.embedder:
            logger.warning("No embedding model available. Semantic search will be unavailable.")
            logger.warning("Install fastembed with: pip install fastembed")

        # FAISS index (lazy loaded)
        self.index = None
        self.patterns = []  # Cached pattern metadata

        # Thread safety for index operations
        self._lock = threading.Lock()

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using fastembed."""
        if not self.embedder or not text.strip():
            return None

        try:
            # fastembed returns generator, take first result
            embeddings = list(self.embedder.embed([text]))
            if embeddings:
                return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")

        return None

    def build_index(self, force_rebuild: bool = False) -> dict:
        """
        Build FAISS index for all patterns in construct directory.

        Args:
            force_rebuild: If True, rebuild even if index exists

        Returns:
            Dict with build statistics
        """
        start_time = time.time()

        # Check if index exists and is recent
        if not force_rebuild and self.index_path.exists() and self.metadata_path.exists():
            age_seconds = time.time() - self.index_path.stat().st_mtime
            if age_seconds < 3600:  # Less than 1 hour old
                logger.info(f"Index is recent ({age_seconds:.0f}s old), skipping rebuild")
                return {'status': 'skipped', 'reason': 'index_recent'}

        # Scan construct directory for patterns
        pattern_files = list(self.construct_root.rglob('*.md'))
        if not pattern_files:
            logger.warning(f"No pattern files found in {self.construct_root}")
            return {'status': 'error', 'reason': 'no_patterns'}

        logger.info(f"Found {len(pattern_files)} pattern files")

        # Parse patterns
        patterns = []
        embeddings_list = []

        for path in pattern_files:
            pattern = PatternReader.load(path)
            if not pattern:
                continue

            # Validate pattern
            errors = PatternValidator.validate(pattern)
            if errors:
                logger.warning(f"Validation errors in {path}: {errors}")
                continue

            # Generate embedding
            text = pattern.to_text()
            embedding = self._embed_text(text)
            if embedding is None:
                logger.warning(f"Failed to embed pattern {pattern.pattern_id}")
                continue

            pattern.embedding = embedding
            pattern.embedding_model = self.embedding_model
            pattern.embedding_dim = self.embedding_dim

            patterns.append(pattern)
            embeddings_list.append(embedding)

        if not patterns:
            logger.warning("No valid patterns to index")
            return {'status': 'error', 'reason': 'no_valid_patterns'}

        # Build FAISS index
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available. Install with: pip install faiss-cpu")
            return {'status': 'error', 'reason': 'faiss_unavailable'}

        embeddings_array = np.vstack(embeddings_list).astype(np.float32)

        # Create L2 index (cosine similarity via normalization)
        faiss.normalize_L2(embeddings_array)
        index = faiss.IndexFlatL2(self.embedding_dim)
        index.add(embeddings_array)

        # Save index and metadata
        faiss.write_index(index, str(self.index_path))

        metadata = [p.to_dict() for p in patterns]
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        elapsed = time.time() - start_time

        logger.info(f"Built index for {len(patterns)} patterns in {elapsed:.2f}s")
        logger.info(f"Index saved to {self.index_path}")

        # Cache in memory (thread-safe update)
        with self._lock:
            self.index = index
            self.patterns = patterns

        return {
            'status': 'success',
            'pattern_count': len(patterns),
            'elapsed_seconds': elapsed,
            'index_path': str(self.index_path),
        }

    def load_index(self) -> bool:
        """
        Load FAISS index from disk.

        Returns:
            True if loaded successfully
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.info("Index not found, run 'neo construct index' to build")
            return False

        try:
            # Load from disk
            index = faiss.read_index(str(self.index_path))

            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)

            patterns = [
                PatternSchema(**m) for m in metadata
            ]

            # Update in-memory state (thread-safe)
            with self._lock:
                self.index = index
                self.patterns = patterns

            logger.info(f"Loaded index with {len(self.patterns)} patterns")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> list[tuple[PatternSchema, float]]:
        """
        Semantic search for patterns matching query.

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of (PatternSchema, similarity_score) tuples, sorted by relevance
        """
        start_time = time.time()

        # Ensure index is loaded
        if self.index is None:
            if not self.load_index():
                logger.warning("Index not available, building now...")
                result = self.build_index()
                if result['status'] != 'success':
                    return []

        # Generate query embedding
        query_embedding = self._embed_text(query)
        if query_embedding is None:
            logger.error("Failed to embed query")
            return []

        # Normalize for cosine similarity
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        # Search
        top_k = min(top_k, len(self.patterns))
        distances, indices = self.index.search(query_embedding, top_k)

        # Convert L2 distances to similarity scores (0-1 range)
        # L2 distance for normalized vectors ranges from 0 (identical) to 2 (opposite)
        # Convert to similarity: similarity = 1 - (distance / 2)
        similarities = 1.0 - (distances[0] / 2.0)

        results = []
        for idx, sim in zip(indices[0], similarities):
            if idx < len(self.patterns):
                results.append((self.patterns[idx], float(sim)))

        elapsed = time.time() - start_time
        logger.info(f"Search completed in {elapsed*1000:.1f}ms, found {len(results)} results")

        return results

    def list_patterns(self, domain: Optional[str] = None) -> list[PatternSchema]:
        """
        List all patterns, optionally filtered by domain.

        Args:
            domain: Optional domain filter (e.g., "rate-limiting")

        Returns:
            List of PatternSchema objects
        """
        pattern_files = list(self.construct_root.rglob('*.md'))
        patterns = []

        for path in pattern_files:
            pattern = PatternReader.load(path)
            if pattern:
                if domain is None or pattern.domain == domain:
                    patterns.append(pattern)

        # Sort by domain, then name
        patterns.sort(key=lambda p: (p.domain, p.name))

        return patterns

    def show_pattern(self, pattern_id: str) -> Optional[PatternSchema]:
        """
        Load and return a single pattern by ID.

        Args:
            pattern_id: Pattern identifier (e.g., "rate-limiting/token-bucket")

        Returns:
            PatternSchema or None if not found
        """
        # Construct file path from pattern_id
        # pattern_id format: "domain/filename"
        pattern_path = self.construct_root / f"{pattern_id}.md"

        if not pattern_path.exists():
            logger.warning(f"Pattern not found: {pattern_id}")
            return None

        return PatternReader.load(pattern_path)
