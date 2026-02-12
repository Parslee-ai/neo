"""
FactStore - Main fact-based memory system for Neo.

Replaces PersistentReasoningMemory with a scoped, supersession-based
fact store. No junk filter, no MinHash, no TF-IDF - just embeddings
and supersession chains.
"""

import hashlib
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from neo.memory.constraints import ConstraintIngester
from neo.memory.context import ContextAssembler
from neo.memory.models import ContextResult, Fact, FactKind, FactMetadata, FactScope
from neo.memory.scope import detect_org_and_project

logger = logging.getLogger(__name__)

# Embedding constants
EMBEDDING_CACHE_MAX_SIZE = 500
MAX_TEXT_LENGTH = 32000
SUPERSESSION_THRESHOLD = 0.85  # Cosine similarity threshold for supersession

# Try importing fastembed
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

FACTS_DIR = Path.home() / ".neo" / "facts"


class FactStore:
    """Scoped fact store with supersession-based knowledge management.

    Replaces PersistentReasoningMemory. Key differences:
    - No junk filter (stores all facts)
    - No MinHash/TF-IDF (embeddings-only retrieval)
    - Supersession instead of merge/consolidation
    - Scoped storage (global, org, project)
    - Constraint ingestion from CLAUDE.md etc.
    """

    def __init__(
        self,
        codebase_root: Optional[str] = None,
        config: Optional[Any] = None,
    ):
        self.codebase_root = codebase_root

        # Detect org and project
        self.org_id, self.project_id = detect_org_and_project(codebase_root)
        logger.info(f"FactStore: org={self.org_id}, project={self.project_id[:8] if self.project_id else 'none'}")

        # Storage paths
        FACTS_DIR.mkdir(parents=True, exist_ok=True)
        self._global_path = FACTS_DIR / "facts_global.json"
        self._org_path = FACTS_DIR / f"facts_org_{self.org_id}.json" if self.org_id != "unknown" else None
        self._project_path = FACTS_DIR / f"facts_project_{self.project_id}.json" if self.project_id else None

        # All facts in memory
        self._facts: list[Fact] = []

        # Embedding model (reuse existing Jina Code v2)
        self._embedder = None
        self._embedding_cache: OrderedDict = OrderedDict()
        self._init_embedder()

        # Context assembler
        self._assembler = ContextAssembler()

        # Load existing facts
        self.load()

        # Migrate from old format if needed
        self._maybe_migrate()

        # Ingest constraints
        self._ingest_constraints()

    def _init_embedder(self) -> None:
        """Initialize the embedding model."""
        if FASTEMBED_AVAILABLE:
            try:
                self._embedder = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-code")
                logger.info("FactStore: Jina Code v2 embeddings initialized")
            except Exception as e:
                logger.warning(f"FactStore: Failed to initialize embedder: {e}")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add_fact(
        self,
        subject: str,
        body: str,
        kind: FactKind = FactKind.PATTERN,
        scope: FactScope = FactScope.PROJECT,
        confidence: float = 0.5,
        source_file: str = "",
        source_prompt: str = "",
        tags: Optional[list[str]] = None,
        depends_on: Optional[list[str]] = None,
    ) -> Fact:
        """Add a new fact to the store.

        No junk filter - all facts are stored. If a supersession candidate
        is found (same scope+kind, cosine > 0.85), the old fact is superseded.

        Args:
            subject: Concise label for the fact.
            body: Full content.
            kind: Type of fact.
            scope: Scope level.
            confidence: Initial confidence (0.0-1.0).
            source_file: File that generated this fact.
            source_prompt: Prompt that triggered this fact.
            tags: Optional tags for filtering.
            depends_on: Fact IDs this derives from.

        Returns:
            The newly created Fact.
        """
        # Generate embedding
        embed_text = f"{subject} {body}"
        embedding = self._embed_text(embed_text)

        fact = Fact(
            subject=subject,
            body=body,
            kind=kind,
            scope=scope,
            org_id=self.org_id,
            project_id=self.project_id,
            metadata=FactMetadata(
                source_file=source_file,
                source_prompt=source_prompt,
                confidence=confidence,
            ),
            embedding=embedding,
            tags=tags or [],
            depends_on=depends_on or [],
        )

        # Check for supersession candidate
        candidate = self._find_supersession_candidate(fact)
        if candidate:
            self._supersede(candidate, fact)

        self._facts.append(fact)
        self.save()
        return fact

    def retrieve_relevant(self, query: str, k: int = 5) -> list[Fact]:
        """Retrieve the most relevant valid facts for a query.

        Args:
            query: The search query.
            k: Maximum number of facts to return.

        Returns:
            List of relevant Fact objects, scored by similarity * confidence * recency.
        """
        query_embedding = self._embed_text(query)

        valid_facts = [f for f in self._facts if f.is_valid and f.kind != FactKind.CONSTRAINT]

        if not valid_facts:
            return []

        scored: list[tuple[Fact, float]] = []
        now = time.time()

        for fact in valid_facts:
            sim = 0.5  # default when no embeddings
            if query_embedding is not None and fact.embedding is not None:
                sim = self._cosine_similarity(query_embedding, fact.embedding)

            confidence = fact.metadata.confidence
            age_days = (now - fact.metadata.last_accessed) / 86400
            recency = 0.5 ** (age_days / 30)

            score = sim * confidence * (0.5 + 0.5 * recency)
            scored.append((fact, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Update access metadata for returned facts
        results = []
        for fact, _ in scored[:k]:
            fact.metadata.last_accessed = now
            fact.metadata.access_count += 1
            results.append(fact)

        return results

    def build_context(
        self,
        query: str,
        environment: Optional[dict] = None,
        k: int = 5,
    ) -> ContextResult:
        """Build a full ContextResult for LLM injection.

        Delegates to ContextAssembler to organize facts into layers.
        """
        query_embedding = self._embed_text(query)
        return self._assembler.assemble(
            facts=self._facts,
            query=query,
            query_embedding=query_embedding,
            environment=environment,
            k=k,
        )

    def format_context_for_prompt(self, ctx: ContextResult) -> str:
        """Render ContextResult as a formatted string."""
        return self._assembler.format_context_for_prompt(ctx)

    # ------------------------------------------------------------------ #
    # Backward compatibility
    # ------------------------------------------------------------------ #

    def memory_level(self) -> float:
        """Calculate memory level (backward compatible with PersistentReasoningMemory).

        Returns 0.0-1.0 based on fact count using sigmoid scaling.
        """
        valid_count = sum(1 for f in self._facts if f.is_valid)
        if valid_count == 0:
            return 0.0
        # Sigmoid with reference point at 50 facts
        return 1.0 - 1.0 / (1.0 + valid_count / 50.0)

    @property
    def entries(self) -> list[Fact]:
        """Backward-compatible access to all facts."""
        return self._facts

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Save facts to scoped JSON files."""
        global_facts = [f for f in self._facts if f.scope == FactScope.GLOBAL]
        org_facts = [f for f in self._facts if f.scope == FactScope.ORG]
        project_facts = [f for f in self._facts if f.scope in (FactScope.PROJECT, FactScope.SESSION)]

        self._save_file(self._global_path, global_facts)
        if self._org_path:
            self._save_file(self._org_path, org_facts)
        if self._project_path:
            self._save_file(self._project_path, project_facts)

    def load(self) -> None:
        """Load facts from all scoped files and merge."""
        self._facts = []
        self._facts.extend(self._load_file(self._global_path))
        if self._org_path:
            self._facts.extend(self._load_file(self._org_path))
        if self._project_path:
            self._facts.extend(self._load_file(self._project_path))
        logger.info(f"FactStore: Loaded {len(self._facts)} facts")

    def _save_file(self, path: Path, facts: list[Fact]) -> None:
        """Save a list of facts to a JSON file."""
        try:
            data = {
                "version": "2.0",
                "facts": [f.to_dict() for f in facts],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as fh:
                json.dump(data, fh, indent=2)
            logger.debug(f"Saved {len(facts)} facts to {path}")
        except Exception as e:
            logger.error(f"Failed to save facts to {path}: {e}")

    def _load_file(self, path: Optional[Path]) -> list[Fact]:
        """Load facts from a JSON file."""
        if path is None or not path.exists():
            return []
        try:
            with open(path) as fh:
                data = json.load(fh)
            return [Fact.from_dict(d) for d in data.get("facts", [])]
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to load facts from {path}: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Supersession
    # ------------------------------------------------------------------ #

    def _find_supersession_candidate(self, new_fact: Fact) -> Optional[Fact]:
        """Find an existing fact that the new fact should supersede.

        Criteria: same scope + kind, cosine similarity > threshold.
        """
        if new_fact.embedding is None:
            return None

        best_match: Optional[Fact] = None
        best_sim = 0.0

        for fact in self._facts:
            if not fact.is_valid:
                continue
            if fact.scope != new_fact.scope or fact.kind != new_fact.kind:
                continue
            if fact.embedding is None:
                continue

            sim = self._cosine_similarity(new_fact.embedding, fact.embedding)
            if sim > SUPERSESSION_THRESHOLD and sim > best_sim:
                best_sim = sim
                best_match = fact

        return best_match

    def _supersede(self, old: Fact, new: Fact) -> None:
        """Supersede an old fact with a new one and cascade needs_review."""
        old.is_valid = False
        old.superseded_by = new.id
        new.supersedes = old.id

        # Carry forward confidence with a small boost
        new.metadata.confidence = min(1.0, old.metadata.confidence + 0.05)

        # Cascade: mark dependents as needing review
        self._cascade_needs_review(old.id)

        logger.info(f"Superseded fact '{old.subject[:40]}' with '{new.subject[:40]}'")

    def _cascade_needs_review(self, superseded_id: str) -> None:
        """Mark facts that depend on a superseded fact as needing review."""
        for fact in self._facts:
            if superseded_id in fact.depends_on and fact.is_valid:
                fact.needs_review = True
                logger.debug(f"Marked fact '{fact.subject[:40]}' as needs_review (dependency superseded)")

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text using local Jina model."""
        if not text or not text.strip():
            return None

        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            self._embedding_cache.move_to_end(cache_key)
            return self._embedding_cache[cache_key]

        embedding = None
        if self._embedder:
            try:
                truncated = text[:MAX_TEXT_LENGTH]
                embeddings = list(self._embedder.embed([truncated]))
                if embeddings:
                    embedding = np.array(embeddings[0], dtype=np.float32)
                    if not np.isfinite(embedding).all():
                        logger.error("Embedding contains NaN or Inf values")
                        embedding = None
            except Exception as e:
                logger.warning(f"Embedding failed: {e}")

        if embedding is not None:
            self._embedding_cache[cache_key] = embedding
            if len(self._embedding_cache) > EMBEDDING_CACHE_MAX_SIZE:
                self._embedding_cache.popitem(last=False)

        return embedding

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------------------ #
    # Constraint ingestion
    # ------------------------------------------------------------------ #

    def _ingest_constraints(self) -> None:
        """Scan constraint files and ingest/update constraint facts."""
        config_auto_scan = True
        if not config_auto_scan:
            return

        ingester = ConstraintIngester(
            codebase_root=self.codebase_root or "",
            org_id=self.org_id,
            project_id=self.project_id,
        )
        new_facts, superseded_facts = ingester.ingest(self._facts)

        if new_facts or superseded_facts:
            self._facts.extend(new_facts)
            self.save()
            logger.info(
                f"Constraints: {len(new_facts)} new, {len(superseded_facts)} superseded"
            )

    # ------------------------------------------------------------------ #
    # Migration
    # ------------------------------------------------------------------ #

    def _maybe_migrate(self) -> None:
        """Migrate from old PersistentReasoningMemory format if needed.

        Only runs when facts/ is empty but old files exist.
        """
        # Only migrate if we have no facts yet (excluding constraints)
        non_constraint_facts = [f for f in self._facts if f.kind != FactKind.CONSTRAINT]
        if non_constraint_facts:
            return

        # Check for old files
        old_global = Path.home() / ".neo" / "global_memory.json"
        old_local = None
        if self.project_id:
            old_local = Path.home() / ".neo" / f"local_{self.project_id}.json"

        if not old_global.exists() and (old_local is None or not old_local.exists()):
            return

        try:
            from neo.memory.migration import migrate_from_legacy
            migrated = migrate_from_legacy(
                old_global_path=old_global,
                old_local_path=old_local,
                org_id=self.org_id,
                project_id=self.project_id,
            )
            if migrated:
                self._facts.extend(migrated)
                self.save()
                logger.info(f"Migrated {len(migrated)} facts from legacy format")
        except Exception as e:
            logger.warning(f"Migration failed (non-fatal): {e}")
