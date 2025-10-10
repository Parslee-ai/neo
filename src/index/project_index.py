"""
Project-specific semantic index for Neo.

Provides per-repository code context via FAISS-based semantic search.
This is separate from global memory - project index captures LOCAL codebase
knowledge, while global memory stores CROSS-PROJECT patterns.

Architecture:
- .neo/ directory per repository (can be checked in or synced)
- FAISS index for fast semantic retrieval of code chunks
- File hash tracking for staleness detection
- Opportunistic refresh during LLM wait time (no background daemons)

Design philosophy:
- Bounded storage (limit to top N most relevant chunks)
- Incremental updates (only re-embed changed files)
- Atomic writes (copy-on-write to prevent corruption)
- Zero hidden CPU work (all indexing is explicit or budgeted)
"""

import ast
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np

# Import FAISS for fast similarity search
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

# Import fastembed for local embeddings
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

logger = logging.getLogger(__name__)

# Constants
DEFAULT_EMBEDDING_DIM = 768  # Jina Code v2 dimension (same as global memory)
MAX_CHUNKS_PER_REPO = 1000  # Bounded storage
MAX_CHUNK_LENGTH = 2000  # Characters per chunk
STALENESS_THRESHOLD = 0.1  # 10% of files changed triggers full reindex warning
REFRESH_BUDGET_MS = 5000  # Max 5s for opportunistic refresh
REFRESH_MAX_CHUNKS = 100  # Max chunks to update during opportunistic refresh


@dataclass
class CodeChunk:
    """A semantic chunk of code from the repository."""

    # Identity
    file_path: str  # Relative to repo root
    chunk_id: str  # Unique ID within file (e.g., "func:calculate_total")

    # Content
    content: str  # The actual code
    chunk_type: str  # "function", "class", "module", etc.

    # Context
    start_line: int
    end_line: int
    symbols: List[str] = field(default_factory=list)  # Function/class names defined
    imports: List[str] = field(default_factory=list)  # Imported symbols

    # Embedding
    embedding: Optional[np.ndarray] = None

    # Retrieval metadata
    similarity: Optional[float] = None  # Similarity score from retrieve()

    # Metadata
    file_hash: str = ""  # Hash of source file for staleness detection
    indexed_at: float = field(default_factory=time.time)


@dataclass
class IndexSnapshot:
    """Snapshot metadata for .neo/index.json"""

    # Version tracking
    schema_version: str = "1"
    neo_version: str = ""

    # Repository state
    commit_hash: str = ""  # Git commit when indexed
    total_files: int = 0
    total_chunks: int = 0

    # Model info
    embedding_model: str = "jinaai/jina-embeddings-v2-base-code"
    embedding_dim: int = DEFAULT_EMBEDDING_DIM

    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # File tracking (for staleness detection)
    file_hashes: Dict[str, str] = field(default_factory=dict)  # rel_path -> hash


class ProjectIndex:
    """
    Project-specific semantic index for code retrieval.

    Stored in .neo/ directory:
    - index.json: Snapshot metadata
    - chunks.json: Code chunks with embeddings
    - faiss.index: FAISS index file (if FAISS available)
    """

    def __init__(self, repo_root: str):
        """
        Initialize project index for given repository.

        Args:
            repo_root: Absolute path to repository root
        """
        self.repo_root = Path(repo_root).resolve()
        self.neo_dir = self.repo_root / ".neo"
        self.snapshot_path = self.neo_dir / "index.json"
        self.chunks_path = self.neo_dir / "chunks.json"
        self.faiss_path = self.neo_dir / "faiss.index"

        # In-memory state
        self.chunks: List[CodeChunk] = []
        self.snapshot: Optional[IndexSnapshot] = None
        self.faiss_index: Optional[Any] = None
        self.embedding_model: Optional[TextEmbedding] = None

        # Load existing index if available
        if self.snapshot_path.exists():
            self._load()

    def _load(self):
        """Load index from disk."""
        try:
            # Load snapshot
            with open(self.snapshot_path) as f:
                snapshot_dict = json.load(f)
                self.snapshot = IndexSnapshot(**snapshot_dict)

            # Load chunks
            if self.chunks_path.exists():
                with open(self.chunks_path) as f:
                    chunks_data = json.load(f)
                    for chunk_dict in chunks_data:
                        # Deserialize embedding
                        embedding = None
                        if 'embedding' in chunk_dict and chunk_dict['embedding']:
                            embedding = np.array(chunk_dict['embedding'], dtype=np.float32)

                        chunk = CodeChunk(
                            file_path=chunk_dict['file_path'],
                            chunk_id=chunk_dict['chunk_id'],
                            content=chunk_dict['content'],
                            chunk_type=chunk_dict['chunk_type'],
                            start_line=chunk_dict['start_line'],
                            end_line=chunk_dict['end_line'],
                            symbols=chunk_dict.get('symbols', []),
                            imports=chunk_dict.get('imports', []),
                            embedding=embedding,
                            file_hash=chunk_dict.get('file_hash', ''),
                            indexed_at=chunk_dict.get('indexed_at', time.time())
                        )
                        self.chunks.append(chunk)

            # Load FAISS index if available
            if FAISS_AVAILABLE and self.faiss_path.exists():
                self.faiss_index = faiss.read_index(str(self.faiss_path))
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")

            logger.info(f"Loaded project index: {len(self.chunks)} chunks from {self.snapshot.total_files} files")

        except Exception as e:
            logger.error(f"Failed to load project index: {e}")
            # Reset to empty state
            self.chunks = []
            self.snapshot = None
            self.faiss_index = None

    def _save(self):
        """Save index to disk with atomic write."""
        try:
            # Create .neo/ directory
            self.neo_dir.mkdir(parents=True, exist_ok=True)

            # Update snapshot metadata
            if not self.snapshot:
                self.snapshot = IndexSnapshot()
            self.snapshot.last_updated = time.time()
            self.snapshot.total_chunks = len(self.chunks)

            # Atomic write: write to temp, then rename
            # Snapshot
            snapshot_tmp = self.snapshot_path.with_suffix('.tmp')
            with open(snapshot_tmp, 'w') as f:
                json.dump(self.snapshot.__dict__, f, indent=2)
            snapshot_tmp.rename(self.snapshot_path)

            # Chunks
            chunks_tmp = self.chunks_path.with_suffix('.tmp')
            chunks_data = []
            for chunk in self.chunks:
                chunk_dict = {
                    'file_path': chunk.file_path,
                    'chunk_id': chunk.chunk_id,
                    'content': chunk.content,
                    'chunk_type': chunk.chunk_type,
                    'start_line': chunk.start_line,
                    'end_line': chunk.end_line,
                    'symbols': chunk.symbols,
                    'imports': chunk.imports,
                    'embedding': chunk.embedding.tolist() if chunk.embedding is not None else None,
                    'file_hash': chunk.file_hash,
                    'indexed_at': chunk.indexed_at
                }
                chunks_data.append(chunk_dict)

            with open(chunks_tmp, 'w') as f:
                json.dump(chunks_data, f, indent=2)
            chunks_tmp.rename(self.chunks_path)

            # FAISS index
            if FAISS_AVAILABLE and self.faiss_index:
                faiss_tmp = self.faiss_path.with_suffix('.tmp')
                faiss.write_index(self.faiss_index, str(faiss_tmp))
                faiss_tmp.rename(self.faiss_path)

            logger.info(f"Saved project index: {len(self.chunks)} chunks")

        except Exception as e:
            logger.error(f"Failed to save project index: {e}")
            raise

    def build_index(self, file_patterns: List[str] = None, max_files: int = 100):
        """
        Build initial index for repository.

        Args:
            file_patterns: Glob patterns for files to index (default: ["**/*.py"])
            max_files: Maximum files to index (prevent runaway on large repos)
        """
        if not file_patterns:
            file_patterns = ["**/*.py"]  # Default to Python files

        logger.info(f"Building project index for {self.repo_root}")
        start_time = time.time()

        # Initialize snapshot
        self.snapshot = IndexSnapshot()
        self.snapshot.commit_hash = self._get_git_commit()

        # Find files to index
        files_to_index = []
        for pattern in file_patterns:
            files_to_index.extend(self.repo_root.glob(pattern))

        # Limit total files
        files_to_index = files_to_index[:max_files]
        self.snapshot.total_files = len(files_to_index)

        # Extract chunks from each file
        all_chunks = []
        for file_path in files_to_index:
            # Security: Reject symlinks and paths outside repo

            # Defensive: Check existence first (handles race conditions)
            if not file_path.exists():
                logger.warning(f"Skipping non-existent path: {file_path}")
                continue

            # Check symlinks before resolving (prevents info disclosure)
            if file_path.is_symlink():
                logger.warning(f"Skipping symlink: {file_path}")
                continue

            # Validate path containment using pathlib
            resolved_path = file_path.resolve()
            repo_root_resolved = self.repo_root.resolve()

            try:
                resolved_path.relative_to(repo_root_resolved)
            except ValueError:
                logger.warning(f"Skipping file outside repo: {file_path} (resolves to {resolved_path})")
                continue

            rel_path = file_path.relative_to(self.repo_root)
            chunks = self._extract_chunks_from_file(file_path, str(rel_path))
            all_chunks.extend(chunks)

            # Track file hash
            file_hash = self._compute_file_hash(file_path)
            self.snapshot.file_hashes[str(rel_path)] = file_hash

        # Limit total chunks (take top N by some scoring)
        if len(all_chunks) > MAX_CHUNKS_PER_REPO:
            logger.warning(f"Too many chunks ({len(all_chunks)}), limiting to {MAX_CHUNKS_PER_REPO}")
            all_chunks = all_chunks[:MAX_CHUNKS_PER_REPO]

        self.chunks = all_chunks

        # Generate embeddings
        self._embed_chunks(self.chunks)

        # Build FAISS index
        if FAISS_AVAILABLE:
            self._build_faiss_index()

        # Save to disk
        self._save()

        elapsed = time.time() - start_time
        logger.info(f"Built project index: {len(self.chunks)} chunks from {len(files_to_index)} files in {elapsed:.1f}s")

    def retrieve(self, query: str, k: int = 5) -> List[CodeChunk]:
        """
        Retrieve top-k most relevant code chunks for query.

        Args:
            query: Natural language or code query
            k: Number of chunks to retrieve

        Returns:
            List of CodeChunk objects ranked by relevance
        """
        if not self.chunks:
            return []

        # Generate query embedding
        query_embedding = self._embed_text(query)
        if query_embedding is None:
            logger.warning("Failed to generate query embedding")
            return []

        # Search using FAISS if available
        if FAISS_AVAILABLE and self.faiss_index:
            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                min(k, len(self.chunks))
            )
            results = []
            for i, dist in zip(indices[0], distances[0]):
                if i < len(self.chunks):
                    chunk = self.chunks[i]
                    chunk.similarity = float(dist)  # FAISS returns cosine similarity (after normalization)
                    results.append(chunk)
            return results

        # Fallback: brute-force cosine similarity
        similarities = []
        for i, chunk in enumerate(self.chunks):
            if chunk.embedding is not None:
                sim = np.dot(query_embedding, chunk.embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                )
                similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, sim in similarities[:k]:
            chunk = self.chunks[i]
            chunk.similarity = float(sim)
            results.append(chunk)
        return results

    def check_staleness(self) -> Tuple[bool, float, List[str]]:
        """
        Check if index is stale (files have changed).

        Returns:
            (is_stale, staleness_ratio, changed_files)
        """
        if not self.snapshot:
            return True, 1.0, []

        # Check git commit
        current_commit = self._get_git_commit()
        if current_commit != self.snapshot.commit_hash:
            logger.info(f"Commit changed: {self.snapshot.commit_hash[:7]} -> {current_commit[:7]}")

        # Check file hashes
        changed_files = []
        for rel_path, old_hash in self.snapshot.file_hashes.items():
            file_path = self.repo_root / rel_path
            if not file_path.exists():
                changed_files.append(rel_path)
                continue

            new_hash = self._compute_file_hash(file_path)
            if new_hash != old_hash:
                changed_files.append(rel_path)

        # Calculate staleness ratio
        total_files = len(self.snapshot.file_hashes)
        staleness_ratio = len(changed_files) / total_files if total_files > 0 else 0.0
        is_stale = staleness_ratio > STALENESS_THRESHOLD

        return is_stale, staleness_ratio, changed_files

    def refresh_changed_files(self, budget_ms: int = REFRESH_BUDGET_MS, max_chunks: int = REFRESH_MAX_CHUNKS):
        """
        Opportunistic refresh: re-embed only changed files within time budget.

        This is called during LLM wait time to keep index fresh without blocking.

        Args:
            budget_ms: Time budget in milliseconds
            max_chunks: Maximum chunks to update
        """
        start_time = time.time()
        budget_s = budget_ms / 1000.0

        is_stale, ratio, changed_files = self.check_staleness()
        if not changed_files:
            logger.debug("No changes detected, index is fresh")
            return

        logger.info(f"Refreshing {len(changed_files)} changed files (budget: {budget_ms}ms)")

        updated_chunks = []
        for rel_path in changed_files:
            # Check budget
            if time.time() - start_time > budget_s:
                logger.info(f"Refresh budget exceeded, stopping early")
                break

            if len(updated_chunks) >= max_chunks:
                logger.info(f"Max chunks reached ({max_chunks}), stopping early")
                break

            # Re-extract chunks from changed file
            file_path = self.repo_root / rel_path
            if not file_path.exists():
                # File deleted, remove chunks
                self.chunks = [c for c in self.chunks if c.file_path != rel_path]
                if rel_path in self.snapshot.file_hashes:
                    del self.snapshot.file_hashes[rel_path]
                continue

            # Extract new chunks
            new_chunks = self._extract_chunks_from_file(file_path, rel_path)

            # Remove old chunks for this file
            self.chunks = [c for c in self.chunks if c.file_path != rel_path]

            # Add new chunks
            self.chunks.extend(new_chunks)
            updated_chunks.extend(new_chunks)

            # Update file hash
            new_hash = self._compute_file_hash(file_path)
            self.snapshot.file_hashes[rel_path] = new_hash

        # Re-embed updated chunks
        if updated_chunks:
            self._embed_chunks(updated_chunks)

            # Rebuild FAISS index
            if FAISS_AVAILABLE:
                self._build_faiss_index()

            # Save changes
            self._save()

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Refreshed {len(updated_chunks)} chunks in {elapsed_ms:.0f}ms")

    def _extract_chunks_from_file(self, file_path: Path, rel_path: str) -> List[CodeChunk]:
        """
        Extract semantic chunks from a file using AST analysis.

        Chunks are:
        - Top-level functions
        - Top-level classes
        - Module docstring
        """
        chunks = []

        try:
            content = file_path.read_text(encoding='utf-8')
            file_hash = self._compute_file_hash(file_path)

            # Parse AST
            tree = ast.parse(content, filename=str(file_path))

            # Extract module docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                chunks.append(CodeChunk(
                    file_path=rel_path,
                    chunk_id="module_doc",
                    content=module_doc,
                    chunk_type="module_doc",
                    start_line=1,
                    end_line=len(module_doc.split('\n')),
                    file_hash=file_hash
                ))

            # Extract top-level functions and classes
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    func_content = ast.get_source_segment(content, node)
                    if func_content:
                        chunks.append(CodeChunk(
                            file_path=rel_path,
                            chunk_id=f"func:{node.name}",
                            content=func_content[:MAX_CHUNK_LENGTH],
                            chunk_type="function",
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            symbols=[node.name],
                            file_hash=file_hash
                        ))

                elif isinstance(node, ast.ClassDef):
                    class_content = ast.get_source_segment(content, node)
                    if class_content:
                        # Extract method names
                        methods = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)]

                        chunks.append(CodeChunk(
                            file_path=rel_path,
                            chunk_id=f"class:{node.name}",
                            content=class_content[:MAX_CHUNK_LENGTH],
                            chunk_type="class",
                            start_line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            symbols=[node.name] + methods,
                            file_hash=file_hash
                        ))

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Failed to extract chunks from {file_path}: {e}")

        return chunks

    def _embed_chunks(self, chunks: List[CodeChunk]):
        """Generate embeddings for chunks."""
        if not chunks:
            return

        # Initialize embedding model if needed
        if not self.embedding_model and FASTEMBED_AVAILABLE:
            self.embedding_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")

        if not self.embedding_model:
            logger.warning("No embedding model available, skipping embeddings")
            return

        # Prepare texts
        texts = [chunk.content for chunk in chunks]

        # Generate embeddings in batch
        try:
            embeddings = list(self.embedding_model.embed(texts))
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text."""
        if not self.embedding_model and FASTEMBED_AVAILABLE:
            self.embedding_model = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")

        if not self.embedding_model:
            return None

        try:
            embeddings = list(self.embedding_model.embed([text]))
            return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            return None

    def _build_faiss_index(self):
        """Build FAISS index from chunk embeddings."""
        if not FAISS_AVAILABLE:
            return

        # Collect embeddings
        embeddings = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                embeddings.append(chunk.embedding)

        if not embeddings:
            logger.warning("No embeddings available, skipping FAISS index")
            return

        # Build index
        embeddings_matrix = np.vstack(embeddings).astype(np.float32)
        dim = embeddings_matrix.shape[1]

        # Use IndexFlatIP for cosine similarity (inner product on normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(dim)

        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings_matrix)

        # Add to index
        self.faiss_index.add(embeddings_matrix)

        logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors (dim={dim})")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file contents."""
        try:
            content = file_path.read_bytes()
            return hashlib.sha256(content).hexdigest()
        except Exception as e:
            logger.error(f"Failed to hash {file_path}: {e}")
            return ""

    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Failed to get git commit: {e}")
        return ""

    def status(self) -> Dict[str, any]:
        """
        Get index status for display.

        Returns:
            Dict with keys: total_chunks, total_files, is_stale, staleness_ratio,
            commit_hash, last_updated, changed_files
        """
        if not self.snapshot:
            return {
                'exists': False,
                'message': 'No index found. Run: neo index'
            }

        is_stale, ratio, changed_files = self.check_staleness()

        return {
            'exists': True,
            'total_chunks': len(self.chunks),
            'total_files': self.snapshot.total_files,
            'is_stale': is_stale,
            'staleness_ratio': ratio,
            'changed_files': changed_files,
            'commit_hash': self.snapshot.commit_hash[:7] if self.snapshot.commit_hash else 'unknown',
            'last_updated': self.snapshot.last_updated,
            'embedding_model': self.snapshot.embedding_model,
            'embedding_dim': self.snapshot.embedding_dim
        }
