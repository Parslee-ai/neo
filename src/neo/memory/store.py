"""
FactStore - Main fact-based memory system for Neo.

Replaces PersistentReasoningMemory with a scoped, supersession-based
fact store. No junk filter, no MinHash, no TF-IDF - just embeddings
and supersession chains.
"""

import contextlib
import hashlib
import json
import logging
import shutil
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Optional

try:
    import fcntl  # POSIX (macOS/Linux); used to serialize cross-process saves
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

import numpy as np

from neo.math_utils import batched_cosine, cluster_by_similarity, cosine_similarity
from neo.memory.bm25 import BM25, tokenize
from neo.memory.query_routing import QueryShape, decompose as _decompose_query
from neo.memory.claude_memory import ClaudeMemoryIngester
from neo.memory.community import CommunityFeedIngester
from neo.memory.constraints import ConstraintIngester
from neo.memory.context import ContextAssembler
from neo.memory.io_utils import atomic_write_json
from neo.memory.metrics import record as metrics_record, time_block
from neo.memory.seed import SeedIngester
from neo.languages import language_for_path
from neo.memory.models import (
    ContextResult,
    Fact,
    FactKind,
    FactMetadata,
    FactScope,
    Provenance,
    rank_score,
    update_effectiveness,
    update_recall,
)
from neo.memory.outcomes import OutcomeTracker, OutcomeType
from neo.memory.scope import _compute_legacy_project_id, detect_org_and_project
from neo.pattern_extraction import extract_pattern_from_correction, get_library

logger = logging.getLogger(__name__)

# Embedding constants
EMBEDDING_CACHE_MAX_SIZE = 500
MAX_TEXT_LENGTH = 32000
SUPERSESSION_THRESHOLD = 0.85  # Cosine similarity threshold for supersession
SYNTHESIS_SIMILARITY = 0.85    # Cosine similarity threshold for review clustering.
                                # Paper 2603.10600 (Trajectory-Memory) §7:
                                # τ = 0.85 was their empirical sweet spot for
                                # description-generalized clusters. Below this
                                # we over-merge unrelated REVIEWs; above this
                                # we get singleton clusters.

# Per-scope capacity limits
SCOPE_LIMITS: dict[str, int] = {
    FactScope.GLOBAL.value: 200,    # Cross-project patterns, grows slowly
    FactScope.ORG.value: 100,       # Team conventions, medium churn
    FactScope.PROJECT.value: 500,   # Codebase-specific, most active
    FactScope.SESSION.value: 50,    # Ephemeral, dies with session
}

# Pruning constants
STALE_MAX_CONFIDENCE = 0.4     # Facts below this confidence are stale candidates
STALE_MIN_AGE_DAYS = 14        # Must be this old before stale-pruning
DEMOTION_MIN_ACCESS = 5        # Minimum accesses before demotion kicks in
DEMOTION_PRUNE_ACCESS = 10     # Accesses threshold for hard pruning
DEMOTION_MIN_AGE_DAYS = 7      # Must be this old before demotion
DEMOTION_CONFIDENCE_PENALTY = 0.1  # Confidence reduction per demotion
DEMOTION_CONFIDENCE_FLOOR = 0.1    # Never demote below this
PROTECTION_HIT_RATE = 0.3      # Success/access ratio for protection boost
PROTECTION_BOOST = 0.05        # Confidence boost for consistently helpful facts

# Independent outcome limits (second layer — outcomes.py caps at 5/session)
MAX_INDEPENDENT_FACTS = 50     # Cap per project to prevent bloat from active repos

# Tags that protect facts from pruning/demotion (curated knowledge)
PROTECTED_TAGS = frozenset({"seed", "community", "synthesized"})

# Dual-buffer consolidation (paper 2603.07670 §9.1). New non-curated
# facts enter a "hot probation buffer" via the ``probation`` tag. They
# get half the normal stale-pruning grace period and only become
# permanent (probation tag removed) after re-verification: either
# explicit observation (success_count > 0), being re-retrieved
# (access_count ≥ 2), or surviving the probation window.
PROBATION_TAG = "probation"
PROBATION_AGE_DAYS = 3
PROBATION_PROMOTE_ACCESS = 2  # accesses needed to promote out of probation

# Try importing fastembed
try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False

FACTS_DIR = Path.home() / ".neo" / "facts"

# Persistent location for the fastembed ONNX cache. fastembed defaults
# to `$TMPDIR/fastembed_cache/` which on macOS lives under
# `/var/folders/<...>/T/` — periodically purged by the OS. After a
# purge, fastembed's local manifest still points at the (now-missing)
# `model.onnx` and `TextEmbedding(...)` blows up with
# `ONNXRuntimeError ... NO_SUCHFILE`. Pinning the cache under
# `~/.cache/neo/` keeps the model around across reboots and macOS tmp
# sweeps.
FASTEMBED_CACHE_DIR = Path.home() / ".cache" / "neo" / "fastembed"


@contextlib.contextmanager
def scope_file_lock(path: "Path"):
    """Exclusive cross-process lock for a scope file's read-modify-write.

    Locks a sidecar ``<file>.lock`` (never the data file itself — that gets
    atomically replaced by the writer, which would drop a lock held on the
    original inode). Best-effort: if ``fcntl`` is unavailable or locking fails,
    proceeds unlocked (degrades to the prior behavior) rather than blocking.

    Module-level so any writer that does its own read-modify-write on a fact
    file — ``FactStore.save`` and the ``neo memory prune`` compactor alike —
    serializes through the identical lock rather than each rolling its own.
    """
    if fcntl is None:
        yield
        return
    lock_path = path.with_suffix(path.suffix + ".lock")
    fd = None
    try:
        fd = open(lock_path, "a")  # 'a': never truncate; content is unused
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
    except OSError as e:
        logger.debug(f"scope file lock unavailable for {path.name}: {e}")
        if fd is not None:
            fd.close()
        yield
        return
    try:
        yield
    finally:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
        finally:
            fd.close()


class FactStore:
    """Scoped fact store with supersession-based knowledge management.

    Replaces PersistentReasoningMemory. Key differences:
    - No junk filter (stores all facts)
    - No MinHash/TF-IDF (embeddings-only retrieval)
    - Supersession instead of merge/consolidation
    - Scoped storage (global, org, project)
    - Constraint ingestion from CLAUDE.md etc.

    Not thread-safe. Retrieval mutates per-fact metadata (last_accessed,
    access_count, recall_count, g_n); concurrent retrievals over the same
    instance would race on those bumps.
    """

    def __init__(
        self,
        codebase_root: Optional[str] = None,
        config: Optional[Any] = None,
        lm_adapter: Optional[Any] = None,
        eager_init: bool = True,
    ):
        self.codebase_root = codebase_root
        self._config = config
        self._lm_adapter = lm_adapter

        # Detect org and project
        self.org_id, self.project_id = detect_org_and_project(codebase_root)
        logger.info(f"FactStore: org={self.org_id}, project={self.project_id[:8] if self.project_id else 'none'}")

        # Storage paths
        FACTS_DIR.mkdir(parents=True, exist_ok=True)
        self._global_path = FACTS_DIR / "facts_global.json"
        self._org_path = FACTS_DIR / f"facts_org_{self.org_id}.json" if self.org_id != "unknown" else None
        self._project_path = FACTS_DIR / f"facts_project_{self.project_id}.json" if self.project_id else None

        # Rename fact + watermark files written under the pre-remote-hash
        # project ID. Runs once per project; no-op when the legacy ID equals
        # the current ID (e.g. repos without a remote) or when no legacy
        # file exists.
        self._migrate_legacy_project_id_files(codebase_root)

        # All facts in memory
        self._facts: list[Fact] = []
        self._cap_pending = False  # Set by _cap_independent_facts(save=False)

        # Concurrency bookkeeping for the merge-on-save (see save()).
        # _deleted_ids: facts this instance physically removed (purge/prune) —
        #   excluded from the merge so a re-read can't resurrect them.
        # _scope_mtimes: last mtime_ns we observed per scope file, so the merge
        #   skips its re-read+parse when nothing else has written since (the
        #   common single-process case; avoids re-parsing a multi-MB file per add).
        self._deleted_ids: set[str] = set()
        self._scope_mtimes: dict[str, int] = {}

        # Embedding model (lazy-initialized on first use to avoid slow startup)
        self._embedder = None
        self._embedder_initialized = False
        self._embedding_cache: OrderedDict = OrderedDict()

        # Context assembler
        self._assembler = ContextAssembler()

        # Outcome tracker for learning from actual code changes
        self._outcome_tracker = OutcomeTracker(
            codebase_root=codebase_root,
            project_id=self.project_id,
        )

        # Load existing facts (required for basic operation)
        self.load()

        # Heavy I/O (migration, constraint/history ingestion) deferred to initialize()
        self._initialized = False
        if eager_init:
            self.initialize()

    def initialize(self) -> None:
        """Run post-construction I/O: migration, constraint ingestion, git history.

        Called explicitly after construction. Separating this from __init__
        ensures a partially-failed ingestion doesn't leave a broken object.
        """
        if self._initialized:
            return
        self._initialized = True

        self._maybe_migrate()
        self._ingest_seed_facts()
        self._ingest_community_feed()
        self._ingest_constraints()
        self._ingest_claude_memory()
        self._ingest_git_history()

        # Persist capping that load() deferred (save=False)
        if self._cap_pending:
            self.save()
            self._cap_pending = False

        # Lifecycle maintenance on every cold start. Previously these only
        # ran inside detect_implicit_feedback, so inactive projects accumulated
        # stale REVIEW facts indefinitely.
        try:
            # Defer each step's save and flush once at the end — a cold start on
            # a multi-MB fact file otherwise pays 4 full merge-on-save rewrites.
            changed = self.prune_stale_facts(save=False)
            changed += self.demote_unhelpful_facts(save=False)
            changed += self.purge_dead_facts(save=False)
            changed += self.strip_tombstone_embeddings(save=False)
            if changed:
                self.save()
        except Exception as e:
            logger.warning(f"Lifecycle maintenance on init failed (non-fatal): {e}")

    def _ensure_embedder(self) -> None:
        """Lazy-initialize the embedding model on first use."""
        if self._embedder_initialized:
            return
        self._embedder_initialized = True
        if not FASTEMBED_AVAILABLE:
            return
        FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            self._embedder = TextEmbedding(
                model_name="jinaai/jina-embeddings-v2-base-code",
                cache_dir=str(FASTEMBED_CACHE_DIR),
            )
            logger.info("FactStore: Jina Code v2 embeddings initialized")
        except Exception as e:
            # Recover from a stale local cache: if fastembed's
            # manifest points at a missing `model.onnx` (typical
            # after a `$TMPDIR` sweep on macOS), nuke the
            # snapshot dir and let fastembed re-download. Only
            # retry once — a second failure means something else
            # is wrong (network, disk, transient ONNX init bug).
            msg = str(e)
            if "NO_SUCHFILE" in msg or "model.onnx" in msg:
                logger.warning(
                    "FactStore: embedder cache appears stale "
                    "(%s); clearing and re-downloading", e
                )
                shutil.rmtree(FASTEMBED_CACHE_DIR, ignore_errors=True)
                FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    self._embedder = TextEmbedding(
                        model_name="jinaai/jina-embeddings-v2-base-code",
                        cache_dir=str(FASTEMBED_CACHE_DIR),
                    )
                    logger.info("FactStore: embedder re-downloaded successfully")
                    return
                except Exception as e2:
                    logger.warning(
                        "FactStore: Failed to initialize embedder "
                        "after cache reset: %s", e2
                    )
                    return
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
        provenance: "str | Provenance" = Provenance.INFERRED,
        retrieval_text: Optional[str] = None,
        context_text: Optional[str] = None,
        domain: str = "",
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
            provenance: Origin type — Provenance enum or its string value.

        Returns:
            The newly created Fact.
        """
        # Coerce enum to its string value so storage stays JSON-clean.
        prov_value = provenance.value if isinstance(provenance, Provenance) else provenance

        initial_tags = list(tags or [])
        # Dual-buffer: new fluid facts enter probation. Curated/structural
        # facts (CLAUDE.md, seed corpus, synthesis output) skip probation
        # because they're already vetted.
        is_curated = (
            kind in (FactKind.CONSTRAINT, FactKind.ARCHITECTURE)
            or prov_value == Provenance.STRUCTURAL.value
            or bool(PROTECTED_TAGS & set(initial_tags))
        )
        if not is_curated and PROBATION_TAG not in initial_tags:
            initial_tags.append(PROBATION_TAG)

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
                provenance=prov_value,
            ),
            tags=initial_tags,
            depends_on=depends_on or [],
            retrieval_text=retrieval_text,
            context_text=context_text,
            domain=domain or None,
        )

        # Pre-write dedup: filter -> canonicalize -> exact-match check.
        # Catches the "independent flood" failure mode where many near-
        # identical facts get written before scope-cap eviction has a
        # chance to fire. The supersession check below handles fuzzy
        # near-duplicates; this catches exact (post-generalize) twins
        # within the same scope, regardless of kind.
        existing = self._exact_canonical_match(fact)
        if existing is not None:
            # Bump access metadata on the existing record and return it
            # rather than writing a new row. Idempotent re-ingestion.
            existing.metadata.access_count += 1
            existing.metadata.last_accessed = time.time()
            self.save()
            metrics_record(
                "add_fact",
                kind=existing.kind.value,
                scope=existing.scope.value,
                confidence=existing.metadata.confidence,
                provenance=existing.metadata.provenance,
                corpus_size_after=len([f for f in self._facts if f.is_valid]),
                deduped=True,
            )
            return existing

        # Embed the retrieval_text (defaults to subject+body if unset).
        fact.embedding = self._embed_text(fact.embed_text())

        # Check for supersession candidate (fuzzy near-duplicate, same kind+scope).
        candidate = self._find_supersession_candidate(fact)
        if candidate:
            self._supersede(candidate, fact)

        self._facts.append(fact)
        self._enforce_scope_limit(fact.scope)
        self.save()
        metrics_record(
            "add_fact",
            kind=fact.kind.value,
            scope=fact.scope.value,
            confidence=fact.metadata.confidence,
            provenance=fact.metadata.provenance,
            corpus_size_after=len([f for f in self._facts if f.is_valid]),
        )
        return fact

    def retrieve_relevant(
        self,
        query: str,
        k: int = 30,
        domain: Optional[str] = None,
    ) -> list[Fact]:
        """Retrieve the most relevant valid facts for a query.

        Pipeline:
        1. Classify the query shape (DIRECT/CHAIN/SPLIT — paper 2604.04853
           §5.3). Multi-hop and multi-entity queries get decomposed; the
           per-branch results are merged with dedup by fact.id.
        2. For each (sub-)query: hybrid dense + BM25 over the full valid
           corpus, then rank_score (recall-probability * confidence +
           success_bonus * effectiveness_f + provenance_bonus).
        3. Half-by-rank-score / half-by-cosine selection over the merged
           candidate pool.

        At Neo's max scope of ~850 valid facts, all of this is in the
        single-millisecond range — no ANN index needed.

        If ``domain`` is given, only facts whose ``Fact.domain`` matches
        exactly are considered. See ``SUGGESTED_DOMAINS`` for the
        recommended vocabulary.
        """
        shape, sub_queries = _decompose_query(query)
        if shape is QueryShape.DIRECT or len(sub_queries) <= 1:
            return self._retrieve_single(query, k, domain=domain)

        # Multi-hop / multi-entity: per-branch retrieve, merge, dedup,
        # then take top-k by best per-fact rank_score across branches.
        per_branch_k = max(5, k // max(1, len(sub_queries)))
        merged: dict[str, tuple[Fact, float]] = {}
        for sq in sub_queries:
            for fact in self._retrieve_single(sq, per_branch_k, domain=domain):
                prev = merged.get(fact.id)
                if prev is None or fact.metadata.confidence > prev[1]:
                    merged[fact.id] = (fact, fact.metadata.confidence)
        merged_facts = [v[0] for v in merged.values()]
        merged_facts.sort(key=lambda f: rank_score(f, 1.0), reverse=True)
        metrics_record(
            "retrieve",
            path="retrieve_relevant.multi",
            shape=shape.value,
            sub_queries=len(sub_queries),
            k_requested=k,
            results=min(k, len(merged_facts)),
        )
        return merged_facts[:k]

    def _expand_episode_neighbors(
        self, hits: list[Fact], *, max_per_episode: int = 2,
    ) -> list[Fact]:
        """Nucleus expansion (paper 2604.04853 §4.6): for each EPISODE hit,
        pull a small neighborhood of peer episodes from the same session
        and inject them in chronological order.

        Sessions are keyed by ``metadata.source_prompt`` — the same prompt
        triggered all simulator traces in one Neo run, so all the EPISODE
        facts written by ``persist_simulation_episode`` for one run share
        it. Neighbors are sorted by ``effective_event_time`` and capped
        at ``max_per_episode`` to keep the prompt budget bounded.

        Non-episode hits pass through unchanged. Duplicates (peer already
        in ``hits``) are skipped. The result preserves the input ordering
        and appends each episode's neighbors immediately after.
        """
        if not hits:
            return hits

        existing_ids = {f.id for f in hits}
        episodes_by_prompt: dict[str, list[Fact]] = {}
        for fact in self._facts:
            if not fact.is_valid or fact.kind != FactKind.EPISODE:
                continue
            sp = fact.metadata.source_prompt
            if not sp:
                continue
            episodes_by_prompt.setdefault(sp, []).append(fact)

        expanded: list[Fact] = []
        for hit in hits:
            expanded.append(hit)
            if hit.kind != FactKind.EPISODE:
                continue
            peers = episodes_by_prompt.get(hit.metadata.source_prompt, [])
            if not peers:
                continue
            peers.sort(key=lambda f: f.metadata.effective_event_time)
            added = 0
            for peer in peers:
                if added >= max_per_episode:
                    break
                if peer.id == hit.id or peer.id in existing_ids:
                    continue
                expanded.append(peer)
                existing_ids.add(peer.id)
                added += 1
        return expanded

    def _retrieve_single(
        self, query: str, k: int, *, domain: Optional[str] = None
    ) -> list[Fact]:
        """Single-pass retrieval — what retrieve_relevant used to be.

        Split out so query-routing can call us per sub-query without
        recursing through the decomposer.
        """
        with time_block() as timed:
            query_embedding = self._embed_text(query)

            valid_facts = [f for f in self._facts if f.is_valid and f.kind != FactKind.CONSTRAINT]
            if domain is not None:
                valid_facts = [f for f in valid_facts if f.domain == domain]
            if not valid_facts:
                metrics_record(
                    "retrieve",
                    path="retrieve_relevant",
                    k_requested=k,
                    corpus_size=0,
                    results=0,
                    latency_ms=(time.perf_counter() - timed._t0) * 1000.0,
                )
                return []

            now = time.time()
            sims = batched_cosine([f.embedding for f in valid_facts], query_embedding)

            # Hybrid dense + sparse (paper 2603.19935 Memori §3.3). BM25
            # over the same corpus catches keyword matches the dense
            # embedding smooths over. Min-max normalize both signals to
            # [0, 1] then weighted-sum (0.7 dense + 0.3 sparse). Fused
            # similarity replaces the raw cosine inside rank_score so the
            # downstream Ebbinghaus decay and confidence multiplier still
            # apply the same way.
            fused_sims = self._fuse_dense_sparse(query, valid_facts, sims)
            scored = [(f, fs, rank_score(f, fs, now)) for f, fs in zip(valid_facts, fused_sims)]

            # Half-by-rank-score / half-by-cosine policy (paper 2505.23946
            # LessonL Algorithm 1): top ⌈k/2⌉ by full rank_score (which
            # captures confidence + success_bonus + provenance + decay),
            # then top ⌊k/2⌋ by raw cosine to the query — so a fresh fact
            # with no track record but high semantic match can still
            # surface alongside the validated winners. LessonL ablation
            # showed this cuts retrieval-quality variance from σ=0.28 to
            # σ=0.03 vs pure score-sort.
            half_score = (k + 1) // 2
            half_cos = k - half_score

            by_score = sorted(scored, key=lambda x: x[2], reverse=True)
            score_pick = by_score[:half_score]
            score_pick_ids = {f.id for f, _, _ in score_pick}

            by_cos = sorted(
                (s for s in scored if s[0].id not in score_pick_ids),
                key=lambda x: x[1],
                reverse=True,
            )
            cos_pick = by_cos[:half_cos]

            chosen = score_pick + cos_pick
            chosen.sort(key=lambda x: x[2], reverse=True)

            results: list[Fact] = []
            for fact, _sim, _score in chosen:
                self._mark_retrieved(fact, now)
                results.append(fact)

        # Nucleus expansion: when any retrieved fact is an EPISODE, pull
        # a small neighborhood of peer episodes from the same session so
        # the prompt sees the surrounding context, not just the single
        # turn whose cosine happened to score highest.
        results = self._expand_episode_neighbors(results)

        chosen_scores = [s for _, _, s in chosen]
        metrics_record(
            "retrieve",
            path="retrieve_relevant",
            k_requested=k,
            corpus_size=len(valid_facts),
            results=len(results),
            latency_ms=timed.elapsed_ms,
            top_score=chosen_scores[0] if chosen_scores else None,
            mean_top_k=sum(chosen_scores) / len(chosen_scores) if chosen_scores else None,
        )
        return results

    @staticmethod
    def _fuse_dense_sparse(
        query: str, facts: list[Fact], dense_sims: list[float],
        *, dense_weight: float = 0.7, sparse_weight: float = 0.3,
    ) -> list[float]:
        """Weighted fusion of dense cosine + BM25 sparse over the same corpus.

        Both signals are min-max normalized to [0, 1] before mixing so the
        weights are interpretable. Returns a list parallel to ``facts``.
        When the BM25 channel produces zero signal (empty query tokens or
        empty corpus), falls through to the raw dense sims unchanged.
        """
        if not facts:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return list(dense_sims)

        docs = [tokenize(f.embed_text()) for f in facts]
        index = BM25(docs)
        sparse = index.scores(query_terms)

        sparse_max = max(sparse) if sparse else 0.0
        if sparse_max <= 0.0:
            return list(dense_sims)
        sparse_norm = [s / sparse_max for s in sparse]

        # Dense sims are already in [-1, 1] from cosine; shift to [0, 1].
        dense_norm = [max(0.0, min(1.0, (s + 1.0) / 2.0)) for s in dense_sims]

        return [
            dense_weight * d + sparse_weight * sp
            for d, sp in zip(dense_norm, sparse_norm)
        ]

    @staticmethod
    def _mark_retrieved(fact: Fact, now: float) -> None:
        """Apply retrieval bookkeeping to a fact: access metadata + recall update.

        Single place that mutates per-fact retrieval-tracking state, so the
        two retrieval entry points (retrieve_relevant + build_context) can't
        drift in what they count.

        Probation promotion: a probation-tagged fact that's been accessed
        PROBATION_PROMOTE_ACCESS times has proven useful enough to leave
        the hot buffer.
        """
        fact.metadata.last_accessed = now
        fact.metadata.access_count += 1
        update_recall(fact, now)

        if (
            PROBATION_TAG in fact.tags
            and fact.metadata.access_count >= PROBATION_PROMOTE_ACCESS
        ):
            fact.tags = [t for t in fact.tags if t != PROBATION_TAG]

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
        result = self._assembler.assemble(
            facts=self._facts,
            query=query,
            query_embedding=query_embedding,
            environment=environment,
            k=k,
        )

        # Update access metadata on retrieved facts (mirrors retrieve_relevant)
        now = time.time()
        for fact in result.valid_facts:
            self._mark_retrieved(fact, now)

        metrics_record(
            "retrieve",
            path="build_context",
            k_requested=k,
            corpus_size=len([f for f in self._facts if f.is_valid]),
            results=len(result.valid_facts),
        )
        return result

    def format_context_for_prompt(self, ctx: ContextResult) -> str:
        """Render ContextResult as a formatted string."""
        return self._assembler.format_context_for_prompt(ctx)

    def persist_simulation_episode(
        self,
        *,
        prompt: str,
        input_data: str,
        expected_output: str,
        reasoning_steps: list[str],
        issues_found: list[str],
        plan_summary: str = "",
        codebase_ref: str = "",
        file_paths: Optional[list[str]] = None,
    ) -> Fact:
        """Persist a SimulationTrace as an EPISODE fact.

        Uses the retrieval/context split (B1) intentionally:
        - retrieval_text is concise (input → expected_output), so search
          can match on the *shape* of the simulation, not the full trace.
        - context_text holds the full reasoning_steps + issues for prompt
          injection later.

        EpisodeContext records when/where/why/with_whom so the EPISODE
        passes the 5-property test (paper 2502.06975).
        """
        import os  # local: top-level was dropped with the legacy flag
        from neo.memory.models import EpisodeContext  # local: keep import light

        clean = not issues_found
        subject = f"Simulation: {input_data[:60].strip()}"
        retrieval = (
            f"{input_data.strip()} -> {expected_output.strip()}"
            if expected_output
            else input_data.strip()
        )
        narrative_parts = []
        if plan_summary:
            narrative_parts.append(f"Plan: {plan_summary.strip()}")
        narrative_parts.append(f"Input: {input_data.strip()}")
        if expected_output:
            narrative_parts.append(f"Expected: {expected_output.strip()}")
        if reasoning_steps:
            narrative_parts.append("Reasoning:")
            narrative_parts.extend(f"  - {step.strip()}" for step in reasoning_steps if step.strip())
        if issues_found:
            narrative_parts.append("Issues:")
            narrative_parts.extend(f"  - {iss.strip()}" for iss in issues_found if iss.strip())
        context_text = "\n".join(narrative_parts)

        ts_now = time.time()
        episode_context = EpisodeContext(
            when=str(ts_now),
            where=codebase_ref or (self.codebase_root or ""),
            why=prompt[:200] if prompt else None,
            with_whom="simulator",
        )

        tags = ["simulation", "episode"]
        if clean:
            tags.append("simulation:clean")

        # Stash file paths that this simulation's code suggestions touched,
        # one per tag, normalized to relative paths against codebase_ref so
        # downstream consumers (gather_context history-boost) can compare
        # apples-to-apples with their own rel_path conventions. The "file:"
        # prefix gives a stable namespace future tag-based queries can grep.
        ref = codebase_ref or self.codebase_root or ""
        for fp in file_paths or []:
            if not fp:
                continue
            try:
                rel = os.path.relpath(fp, ref) if ref else fp
            except ValueError:  # absolute path on a different drive (Windows)
                rel = fp
            tag = f"file:{rel}"
            if tag not in tags:
                tags.append(tag)

        fact = self.add_fact(
            subject=subject,
            body=context_text,  # body kept in sync for backward callers
            kind=FactKind.EPISODE,
            scope=FactScope.PROJECT,
            confidence=0.6 if clean else 0.3,
            source_prompt=prompt,
            tags=tags,
            provenance=Provenance.OBSERVED,
            retrieval_text=retrieval,
            context_text=context_text,
        )
        fact.episode_context = episode_context
        # Stamp event_time/ingest_time explicitly so bi-temporal queries
        # know this represents a simulation that ran *now*.
        fact.metadata.event_time = ts_now
        fact.metadata.ingest_time = ts_now
        self.save()
        return fact

    def save_session(
        self,
        suggestions: list,
        prompt: str,
        suggestion_fact_ids: Optional[dict[str, str]] = None,
    ) -> None:
        """Persist current session for outcome detection on next invocation.

        Args:
            suggestions: List of CodeSuggestion objects.
            prompt: The user's prompt.
            suggestion_fact_ids: Mapping of file_path -> fact_id for outcome linkage.
        """
        try:
            self._outcome_tracker.save_session(suggestions, prompt, suggestion_fact_ids)
        except Exception as e:
            logger.warning(f"Session save failed: {e}")

    # ------------------------------------------------------------------ #
    # Backward compatibility
    # ------------------------------------------------------------------ #

    def memory_level(self) -> float:
        """Calculate memory level based on quality-weighted facts.

        Each valid fact contributes its confidence (boosted by success rate
        for well-tested facts). Sigmoid scaling maps total to 0.0-1.0.

        The reference quality scales with the total capacity of loaded scopes,
        so a project-only view and a multi-project view both reach "The One"
        at ~80% of their potential quality.
        """
        valid = [f for f in self._facts if f.is_valid]
        if not valid:
            return 0.0

        total_quality = 0.0
        for f in valid:
            score = f.metadata.confidence
            access = f.metadata.access_count
            if access > 0 and f.metadata.success_count > 0:
                success_rate = f.metadata.success_count / access
                score = score * 0.5 + success_rate * 0.5
            # Usage bonus: reward facts that have been applied
            usage_factor = min(1.0, access / 10)
            score *= (0.9 + 0.1 * usage_factor)
            total_quality += score

        # Reference quality = 20% of aggregate capacity across loaded scopes.
        # "The One" (0.80) requires ~80% of capacity filled with well-validated
        # facts (avg confidence ~0.7 + some success history).
        loaded_scopes = {f.scope for f in self._facts}
        capacity = sum(SCOPE_LIMITS.get(s.value, 200) for s in loaded_scopes)
        reference_quality = max(50.0, capacity * 0.2)

        return 1.0 - 1.0 / (1.0 + total_quality / reference_quality)

    def detect_implicit_feedback(self, current_request: dict, request_history: list) -> None:
        """Detect outcomes from previous session and update/create facts.

        For "accepted" outcomes with a linked suggestion fact: boost the original
        fact's confidence and increment success_count (no new REVIEW fact).
        Otherwise: fall back to creating a new REVIEW fact.
        """
        try:
            outcomes, suggestion_fact_ids = self._outcome_tracker.detect_outcomes()
        except Exception as e:
            logger.warning(f"Outcome detection failed: {e}")
            return

        # Architectural delta across the session batch — modulates how much
        # we trust each accept/modify outcome. Best-effort: returns None if
        # there's no baseline or if metrics computation fails.
        arch_delta = self._outcome_tracker.compute_arch_delta()
        arch_severity = arch_delta.severity() if arch_delta is not None else "neutral"
        if arch_delta is not None and arch_severity != "neutral":
            logger.info(
                f"Arch delta over session batch: {arch_severity} "
                f"(cycles={arch_delta.cycles_delta:+d}, "
                f"god_files={arch_delta.god_files_delta:+d}, "
                f"max_depth={arch_delta.max_depth_delta:+d})"
            )

        # Modulation amount: small enough not to overwhelm the base accept/
        # modify signal, large enough to be felt over many iterations.
        arch_mod = (
            -0.1 if arch_severity == "regression"
            else 0.1 if arch_severity == "improvement"
            else 0.0
        )

        facts_by_id = {f.id: f for f in self._facts}
        linked_count = 0
        # One neo invocation links ALL its suggestions to a SINGLE reasoning
        # fact, so several outcomes here can resolve to the same fact_id (each
        # suggested file that changed). Reinforce/demote a given fact at most
        # once per call — otherwise a multi-file suggestion ratchets one fact's
        # success_count/confidence (and spawns duplicate MODIFIED REVIEWs) from
        # what is really one acceptance/correction.
        touched_fact_ids: set[str] = set()

        normalized_fact_ids: dict[str, str] = {}
        for path, fid in suggestion_fact_ids.items():
            normalized_fact_ids[path] = fid
            normalized = self._outcome_tracker._normalize_path(path)
            normalized_fact_ids[normalized] = fid

        def _lookup_fact_id(file_path: str) -> Optional[str]:
            """Look up fact_id with fallback for path normalization mismatches."""
            fid = normalized_fact_ids.get(file_path)
            if fid:
                return fid
            normalized = self._outcome_tracker._normalize_path(file_path)
            fid = normalized_fact_ids.get(normalized)
            if fid:
                return fid
            # Try with/without leading slash
            if file_path.startswith("/"):
                fid = normalized_fact_ids.get(file_path.lstrip("/"))
            else:
                fid = normalized_fact_ids.get("/" + file_path)
            return fid

        for outcome in outcomes:
            if outcome.outcome_type == OutcomeType.ACCEPTED:
                # Try to link back to the original suggestion fact
                fact_id = _lookup_fact_id(outcome.file_path)
                original_fact = facts_by_id.get(fact_id) if fact_id else None

                if original_fact and original_fact.is_valid:
                    if original_fact.id in touched_fact_ids:
                        continue  # fan-out dedup: already reinforced this fact
                    # Boost original fact instead of creating orphan REVIEW.
                    # Base +0.2; modulated by arch delta so a session that
                    # regressed structure earns less trust than one that didn't.
                    boost = max(-0.05, 0.2 + arch_mod)
                    original_fact.metadata.confidence = min(
                        1.0, max(0.0, original_fact.metadata.confidence + boost)
                    )
                    original_fact.metadata.success_count += 1
                    update_effectiveness(original_fact, outcome="better")
                    original_fact.metadata.last_accessed = time.time()
                    touched_fact_ids.add(original_fact.id)
                    linked_count += 1
                    continue

                # Fallback: no linked fact, create REVIEW as before
                subject = f"outcome:accepted {outcome.file_path}"
                body_parts = [
                    f"User applied suggestion to {outcome.file_path}.",
                    f"Original suggestion: {outcome.suggestion_description}",
                    f"Original confidence: {outcome.suggestion_confidence:.2f}",
                ]
                if outcome.diff_summary:
                    body_parts.append(f"Actual changes:\n{outcome.diff_summary}")
                body = "\n".join(body_parts)
                tags = ["outcome", "accepted"]
                confidence = min(1.0, max(0.0, outcome.suggestion_confidence + 0.1 + arch_mod))
            elif outcome.outcome_type == OutcomeType.MODIFIED:
                # User corrected neo's suggestion - learn from the correction
                subject = f"outcome:modified {outcome.file_path}"
                body_parts = [
                    f"User modified neo's suggestion for {outcome.file_path}.",
                    f"Original suggestion: {outcome.suggestion_description}",
                    f"Original confidence: {outcome.suggestion_confidence:.2f}",
                ]
                if outcome.diff_summary:
                    body_parts.append(f"What user actually did:\n{outcome.diff_summary}")
                body = "\n".join(body_parts)
                tags = ["outcome", "modified"]
                confidence = 0.4

                # Demote the original suggestion fact since it was corrected.
                # arch_mod is negative for regression and positive for
                # improvement, so adding it gives: regression deepens the
                # penalty (-0.2 + -0.1 = -0.3), improvement softens it
                # (-0.2 + 0.1 = -0.1).
                fact_id = _lookup_fact_id(outcome.file_path)
                original_fact = facts_by_id.get(fact_id) if fact_id else None
                if original_fact and original_fact.is_valid:
                    if original_fact.id in touched_fact_ids:
                        continue  # fan-out dedup: skip duplicate demote + REVIEW
                    penalty = min(-0.05, -0.2 + arch_mod)
                    original_fact.metadata.confidence = max(
                        0.1, original_fact.metadata.confidence + penalty
                    )
                    update_effectiveness(original_fact, outcome="worse")
                    original_fact.metadata.last_accessed = time.time()
                    touched_fact_ids.add(original_fact.id)
                    linked_count += 1
            elif outcome.outcome_type == OutcomeType.UNVERIFIED:
                # Suggested file changed, but no diff to compare — weak signal.
                # Only update linked fact; never create standalone REVIEW.
                fact_id = _lookup_fact_id(outcome.file_path)
                original_fact = facts_by_id.get(fact_id) if fact_id else None
                if (
                    original_fact and original_fact.is_valid
                    and original_fact.id not in touched_fact_ids
                ):
                    boost = max(-0.05, 0.1 + arch_mod)
                    original_fact.metadata.confidence = min(
                        1.0, max(0.0, original_fact.metadata.confidence + boost)
                    )
                    original_fact.metadata.success_count += 1
                    update_effectiveness(original_fact, outcome="better")
                    original_fact.metadata.last_accessed = time.time()
                    touched_fact_ids.add(original_fact.id)
                    linked_count += 1
                continue  # Never create a REVIEW fact for unverified outcomes
            elif outcome.outcome_type == OutcomeType.INDEPENDENT:
                subject = f"outcome:independent {outcome.file_path}"
                body_parts = [f"User changed {outcome.file_path} (not suggested by neo)."]
                if outcome.diff_summary:
                    body_parts.append(f"Changes:\n{outcome.diff_summary}")
                else:
                    body_parts.append("No diff content available.")
                body = "\n".join(body_parts)
                tags = ["outcome", "independent"]
                confidence = 0.2  # Low confidence so stale pruning cleans up faster
            else:
                continue

            self.add_fact(
                subject=subject,
                body=body,
                kind=FactKind.REVIEW,
                scope=FactScope.PROJECT,
                confidence=confidence,
                source_prompt=current_request.get("prompt", "")[:200],
                tags=tags,
            )

        if linked_count:
            self.save()
            logger.info(f"Boosted/demoted {linked_count} original fact(s) from outcomes")
        if outcomes:
            modified = sum(1 for o in outcomes if o.outcome_type == OutcomeType.MODIFIED)
            logger.info(f"Processed {len(outcomes)} outcome(s): modified={modified}")
            # Chain maintenance: synthesize -> prune stale -> demote unhelpful
            # -> purge dead -> strip tombstone embeddings. The four janitors
            # defer their saves and flush once here.
            self.synthesize_reviews()
            changed = self.prune_stale_facts(save=False)
            changed += self.demote_unhelpful_facts(save=False)
            changed += self.purge_dead_facts(save=False)
            changed += self.strip_tombstone_embeddings(save=False)
            if changed:
                self.save()

        # Extract prevention patterns from corrections
        modified_outcomes = [o for o in outcomes if o.outcome_type == OutcomeType.MODIFIED]
        if modified_outcomes and self._lm_adapter:
            for outcome in modified_outcomes[:3]:  # Limit to 3 per session
                try:
                    # Derive language from extension for accurate fence
                    # tagging in the extraction prompt.
                    language = language_for_path(outcome.file_path)

                    pattern = extract_pattern_from_correction(
                        problem_description=outcome.suggestion_description,
                        failed_code=outcome.suggestion_description,  # Best we have
                        corrected_code=outcome.diff_summary,
                        bug_category="suggestion-mismatch",
                        root_cause=f"Neo's suggestion for {outcome.file_path} was modified by user",
                        adapter=self._lm_adapter,
                        language=language,
                    )
                    library = get_library()
                    library.add_pattern(pattern)
                    logger.info(f"Learned prevention pattern from correction on {outcome.file_path}")
                except Exception as e:
                    logger.warning(f"Pattern extraction from correction failed: {e}")

    def replay_linked_feedback(
        self, *, dry_run: bool = False, include_fallback: bool = False
    ) -> dict[str, Any]:
        """Replay persisted linked feedback without broad maintenance side effects.

        This is the update/migration path for repairing memory after feedback
        loop bugs. It only updates facts explicitly linked by
        ``suggestion_fact_ids`` and deliberately skips architecture deltas,
        independent-change review creation, synthesis, and pruning. That keeps
        replay fast and safe enough for a user-facing maintenance command.
        """
        try:
            outcomes, suggestion_fact_ids = self._outcome_tracker.collect_outcomes(
                clear_processed=False,
                include_fallback=include_fallback,
            )
        except Exception as e:
            logger.warning(f"Feedback replay outcome collection failed: {e}")
            return {"status": "error", "error": str(e)}

        facts_by_id = {f.id: f for f in self._facts}
        normalized_fact_ids: dict[str, str] = {}
        for path, fid in suggestion_fact_ids.items():
            normalized_fact_ids[path] = fid
            normalized_fact_ids[self._outcome_tracker._normalize_path(path)] = fid

        def _lookup_fact(file_path: str) -> Optional[Fact]:
            candidates = [
                file_path,
                self._outcome_tracker._normalize_path(file_path),
                file_path.lstrip("/") if file_path.startswith("/") else "/" + file_path,
            ]
            for candidate in candidates:
                fid = normalized_fact_ids.get(candidate)
                fact = facts_by_id.get(fid) if fid else None
                if fact and fact.is_valid:
                    return fact
            return None

        stats: dict[str, Any] = {
            "status": "ok",
            "outcomes_seen": len(outcomes),
            "linked_updates": 0,
            "accepted": 0,
            "modified": 0,
            "unverified": 0,
            "skipped_unlinked": 0,
            "skipped_independent": 0,
            "dry_run": dry_run,
        }

        for outcome in outcomes:
            if outcome.outcome_type == OutcomeType.INDEPENDENT:
                stats["skipped_independent"] += 1
                continue

            original_fact = _lookup_fact(outcome.file_path)
            if original_fact is None:
                stats["skipped_unlinked"] += 1
                continue

            if outcome.outcome_type == OutcomeType.ACCEPTED:
                stats["accepted"] += 1
                stats["linked_updates"] += 1
                if not dry_run:
                    original_fact.metadata.confidence = min(
                        1.0, max(0.0, original_fact.metadata.confidence + 0.2)
                    )
                    original_fact.metadata.success_count += 1
                    update_effectiveness(original_fact, outcome="better")
                    original_fact.metadata.last_accessed = time.time()
            elif outcome.outcome_type == OutcomeType.UNVERIFIED:
                stats["unverified"] += 1
                stats["linked_updates"] += 1
                if not dry_run:
                    original_fact.metadata.confidence = min(
                        1.0, max(0.0, original_fact.metadata.confidence + 0.1)
                    )
                    original_fact.metadata.success_count += 1
                    update_effectiveness(original_fact, outcome="better")
                    original_fact.metadata.last_accessed = time.time()
            elif outcome.outcome_type == OutcomeType.MODIFIED:
                stats["modified"] += 1
                stats["linked_updates"] += 1
                if not dry_run:
                    original_fact.metadata.confidence = max(
                        0.1, original_fact.metadata.confidence - 0.2
                    )
                    update_effectiveness(original_fact, outcome="worse")
                    original_fact.metadata.last_accessed = time.time()

        if not dry_run and stats["linked_updates"]:
            self.save()
            self._outcome_tracker._clear_session_log()

        return stats

    def apply_mined_outcomes(self, fact_ids: list[str]) -> int:
        """Weakly reinforce facts linked to transcript-mined suggestion matches.

        ``fact_ids`` come from ``TranscriptIngester.mine_suggestion_outcomes``,
        which correlates a past suggestion to a later episode by *topic
        similarity* — evidence the suggestion's area recurred in subsequent work,
        not verified diff-overlap acceptance. So each match earns the same weak
        delta neo gives its other non-git signals (UNVERIFIED): +0.1 confidence
        and a success_count bump — enough to promote a fact off probation —
        never the strong +0.2 the git matcher reserves for proven acceptance.
        The ledger records the suggestion→fact link directly, so this applies by
        fact_id without the path lookup ``detect_implicit_feedback`` needs.
        Returns the number of facts updated.
        """
        by_id = {f.id: f for f in self._facts}
        applied = 0
        for fid in fact_ids:
            fact = by_id.get(fid)
            if fact is None or not fact.is_valid:
                continue
            fact.metadata.confidence = min(1.0, fact.metadata.confidence + 0.1)
            fact.metadata.success_count += 1
            update_effectiveness(fact, outcome="better")
            fact.metadata.last_accessed = time.time()
            applied += 1
        if applied:
            self.save()
        return applied

    @property
    def entries(self) -> list[Fact]:
        """Backward-compatible access to all facts."""
        return self._facts

    def find_contributable(self, min_confidence: float = 0.8,
                           min_successes: int = 3) -> list[Fact]:
        """Find high-quality facts worth contributing to the community feed.

        Criteria: high confidence, real success validation, not already
        from seed/community feeds, not constraints (project-specific).
        """
        auto_tags = {"seed", "community", "constraint", "history", "git-commit"}
        results = []
        for f in self._facts:
            if not f.is_valid:
                continue
            if f.kind == FactKind.CONSTRAINT:
                continue
            if f.metadata.confidence < min_confidence:
                continue
            if f.metadata.success_count < min_successes:
                continue
            if auto_tags & set(f.tags):
                continue
            results.append(f)
        return results

    # ------------------------------------------------------------------ #
    # Scope capacity enforcement
    # ------------------------------------------------------------------ #

    def _enforce_scope_limit(self, scope: FactScope) -> int:
        """Evict lowest-quality valid facts when a scope exceeds its limit.

        Eviction score: confidence weighted by success rate. Constraints and
        synthesized facts are protected from eviction. The newest fact (just
        appended) is never evicted.

        Returns the number of evicted facts.
        """
        limit = SCOPE_LIMITS.get(scope.value)
        if limit is None:
            return 0

        scoped = [f for f in self._facts if f.scope == scope and f.is_valid]
        excess = len(scoped) - limit
        if excess <= 0:
            return 0

        def _eviction_score(fact: Fact) -> float:
            """Lower score = evict first."""
            base = fact.metadata.confidence
            total = fact.metadata.access_count
            if total > 0 and fact.metadata.success_count > 0:
                base = base * 0.5 + (fact.metadata.success_count / total) * 0.5
            return base

        # Never evict constraints or synthesized archetypes
        evictable = [
            f for f in scoped
            if f.kind != FactKind.CONSTRAINT and not (PROTECTED_TAGS & set(f.tags))
        ]
        evictable.sort(key=_eviction_score)

        evicted = 0
        for fact in evictable[:excess]:
            self._invalidate(fact)
            evicted += 1

        if evicted:
            logger.info(
                f"Evicted {evicted} fact(s) from {scope.value} scope "
                f"(limit={limit}, was={len(scoped)})"
            )
        return evicted

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        """Save facts to scoped JSON files.

        EVERY scope file is written with a best-effort merge (not just global):
        re-read the file and preserve any fact present on disk but absent from
        memory, so a concurrent writer's *additions* survive. This matters most
        for the PROJECT scope: the async observer and a request-path neo
        invocation are separate processes writing the same
        ``facts_project_<id>.json``. Without the merge, whichever saved last
        clobbered the other's just-added facts — e.g. a neo invocation's linked
        reasoning fact vanishing because the observer (which loaded the file
        earlier) saved its own transcript facts on top, silently breaking the
        outcome-linkage that feeds the learning loop.

        Same-id facts are field-reconciled, not blindly overwritten (see
        ``_reconcile_fact``), keeping our record as the base. What's preserved
        vs. resolved lossily, stated plainly:
          - ``success_count`` / ``access_count``: preserved losslessly (max of
            both — they're strictly monotonic).
          - ``confidence``: resolved LOSSILY when both sides edited it. We lift
            to the higher value only when the disk side had more successes;
            otherwise ours stands. Reconstructing intent from two confidence
            scalars (e.g. our MODIFIED demotion vs. a peer's ACCEPTED bump)
            needs a per-fact version counter or a file lock — neither is here,
            so this is a deliberate best-effort, favoring recorded reinforcement.
          - validity: a fact WE invalidated this session always wins; a peer's
            invalidation does not propagate mid-session (self-heals on cold start).

        Side effect: reconcile mutates our in-memory metadata in place, so this
        process's monotonic counters rise to match a concurrent writer after a
        save — intended (next decision uses the merged truth).

        The whole read(merge)→write runs under an exclusive cross-process lock
        (``_scope_file_lock``), so there is no torn-write window: a concurrent
        writer cannot interleave between our re-read and our atomic replace. The
        only thing not perfectly composed is two processes *independently*
        editing the SAME fact's confidence at once — both edits are always seen
        (no data loss), but reconciling them into one scalar is a policy choice
        (favor recorded reinforcement), not a lossless merge; lossless would
        need a per-fact operation log, which is a deliberate non-goal.
        """
        for path, scope in (
            (self._global_path, FactScope.GLOBAL),
            (self._org_path, FactScope.ORG),
            (self._project_path, FactScope.PROJECT),
        ):
            if not path:
                continue
            scoped = [f for f in self._facts if f.scope == scope]
            # Hold an exclusive cross-process lock across the whole
            # read(merge)→write so a concurrent writer can't interleave between
            # our re-read and our atomic replace — closing the torn-write window
            # entirely. Locks are taken one scope at a time and released before
            # the next, so there is no lock-ordering / deadlock concern.
            with self._scope_file_lock(path):
                self._save_file(path, self._merge_on_save(path, scoped))
                # Record the mtime we just wrote, so the next save can detect
                # whether another process wrote since (and skip the re-read).
                mt = self._file_mtime(path)
                if mt is not None:
                    self._scope_mtimes[str(path)] = mt

    def _scope_file_lock(self, path: "Path"):
        """Exclusive cross-process lock for a scope file's read-modify-write.

        Thin instance-level alias for the module-level :func:`scope_file_lock`
        so out-of-class writers (the ``neo memory prune`` compactor) serialize
        against ``FactStore.save()`` through the *same* sidecar lock — one
        locking contract, no divergence.
        """
        return scope_file_lock(path)

    @staticmethod
    def _file_mtime(path: "Path") -> Optional[int]:
        """Return a file's mtime in ns, or None if it doesn't exist."""
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return None

    def _merge_on_save(self, path: "Path", our_facts: list[Fact]) -> list[Fact]:
        """Best-effort merge for one scope file: keep disk facts we don't have
        in memory, so a concurrent writer's additions aren't clobbered.

        Fast path: if the file's mtime is unchanged since we last wrote/loaded
        it, no other process has touched it, so there is nothing to merge and we
        skip the (potentially multi-MB) re-read+parse entirely. Facts this
        instance deliberately removed this session (``_deleted_ids`` — purge) are
        never resurrected from disk, so a re-read can't undo a deletion.

        Same-id reconciliation: when a fact exists in BOTH memory and disk, the
        monotonic learning signals are reconciled rather than blindly
        overwritten (``_reconcile_fact``), so a ``success_count``/confidence bump
        another process committed isn't lost to last-writer-wins. A fact we
        invalidated this session always wins (never resurrected as valid).

        Callers hold ``_scope_file_lock`` across this read and the subsequent
        write, so no other process can write between them. The mtime check only
        compares against writes that completed *before* we took the lock; cross-
        process writes are serialized, so the fast path is taken only when the
        file genuinely hasn't changed since we last held the lock.
        """
        current = self._file_mtime(path)
        if current is not None and current == self._scope_mtimes.get(str(path)):
            return our_facts  # we were the last writer; disk has nothing new

        disk_facts = self._load_file(path)
        if not disk_facts:
            return our_facts

        disk_by_id = {f.id: f for f in disk_facts}
        our_ids = {f.id for f in our_facts}
        reconciled = [
            self._reconcile_fact(f, disk_by_id[f.id]) if f.id in disk_by_id else f
            for f in our_facts
        ]
        new_from_disk = [
            f for f in disk_facts
            if f.id not in our_ids and f.id not in self._deleted_ids
        ]
        if new_from_disk:
            logger.info(f"Preserved {len(new_from_disk)} fact(s) from {path.name} on save")
        return reconciled + new_from_disk

    @staticmethod
    def _reconcile_fact(ours: Fact, disk: Fact) -> Fact:
        """Field-merge a same-id fact present in both memory and disk so a
        concurrent process's learning gains aren't lost — WITHOUT discarding our
        own independent edits.

        Always keeps OURS as the base record (so a confidence demotion we
        applied, a supersession pointer we set, tags, effectiveness, etc. all
        survive) and reconciles only specific fields:
          1. If we invalidated it this session, our version wins outright —
             never resurrect a fact we dropped (validity isn't monotonic; ours
             is the intentional state). NOTE: a *peer's* invalidation does not
             propagate to us here (same stance as _deleted_ids — we can't tell a
             peer's intentional prune from a fact we simply never loaded); it
             self-heals when this process also prunes on a later cold start.
          2. success_count / access_count are strictly monotonic (only ever
             incremented), so take the max of both sides — neither counter can
             go backwards.
          3. confidence is NOT monotonic (MODIFIED / demote / prune lower it).
             We lift it via max() ONLY when the disk side recorded strictly more
             successes than we had (it saw an ACCEPTED we haven't, so its
             confidence reflects a reinforcement worth keeping). When both sides
             edited confidence independently this is a deliberate, lossy choice
             — there's no field-blind way to reconstruct intent from two scalars
             without a version counter — and we favor the recorded reinforcement.
        """
        if not ours.is_valid:
            return ours
        om, dm = ours.metadata, disk.metadata
        ours_success = om.success_count
        om.success_count = max(om.success_count, dm.success_count)
        om.access_count = max(om.access_count, dm.access_count)
        if disk.is_valid and dm.success_count > ours_success:
            om.confidence = max(om.confidence, dm.confidence)
        return ours

    def purge_dead_facts(self, save: bool = True) -> int:
        """Remove invalid facts that have gone cold (untouched 30+ days).

        Any fact that is invalid AND hasn't been accessed in 30+ days is
        dropped, regardless of how it died. Two distinct death modes both
        qualify:

          - *superseded* tombstones (``superseded_by`` set, replaced by a
            better fact), and
          - *eviction orphans* — facts marked invalid by
            ``_enforce_scope_limit`` for low quality, which carry **no**
            ``superseded_by`` pointer.

        This mirrors the on-demand compactor (``neo memory prune`` →
        ``subcommands._compact_fact_file``); keeping the two in sync ensures
        cold-start maintenance reclaims the same facts the manual command
        does. The 30-day age gate is the safety net — recently-invalidated
        facts are retained for contrast in retrieval.

        Previously this additionally required the supersession chain to
        resolve to a valid successor, which silently retained every eviction
        orphan forever (they never resolve to a valid fact), letting inactive
        projects' fact files bloat with dead rows indefinitely.

        Returns the number of purged facts.
        """
        now = time.time()
        thirty_days = 30 * 86400

        kept, removed_ids = [], []
        for f in self._facts:
            if f.is_valid or (now - f.metadata.last_accessed) < thirty_days:
                kept.append(f)
            else:
                removed_ids.append(f.id)
        purged = len(removed_ids)
        if purged:
            self._facts = kept
            # Record physical deletions so the merge-on-save can't resurrect
            # them from another process's stale copy on disk. Recorded even when
            # save is deferred, so the caller's single trailing save() flushes them.
            self._deleted_ids.update(removed_ids)
            if save:
                self.save()
            logger.info(f"Purged {purged} dead invalid facts")
        return purged

    def strip_tombstone_embeddings(self, save: bool = True) -> int:
        """Drop the 768-dim embedding from invalidated facts.

        An invalid fact is never retrieved, deduped against, or clustered —
        every such path pre-filters ``is_valid`` — yet ``purge_dead_facts``
        retains it up to 30 days for supersession-chain integrity, audit, and
        merge-on-save conflict resolution. During that window the fact's
        embedding is ~24 KB of JSON per fact; on an active repo the tombstone
        backlog is the overwhelming bulk of the fact file (embeddings on dead
        rows nobody reads). Stripping it preserves every field retrieval and
        merge actually touch while reclaiming ~95% of a tombstone's on-disk
        size.

        Safe because invalidation is terminal: no path revives a fact to valid
        (validity is one-way; the merge-on-save reconciler returns OURS
        outright when we hold it invalid), so the vector is never needed again.
        If some future code ever did need it, ``embed_text()`` re-derives it
        deterministically — but note no current command re-embeds existing
        facts (``--regenerate-embeddings`` targets the legacy ReasoningMemory
        cache, not FactStore facts), so this strip is one-way in practice.

        Not globally atomic across processes: a peer that still holds this fact
        valid-with-embedding in memory re-writes the vector on its next
        ``save()`` (the reconciler keeps OURS). It self-heals — every cold
        start and ``detect_implicit_feedback`` re-runs this strip — same
        eventual-consistency class as a peer's un-propagated invalidation.

        Idempotent. Returns the number of facts stripped.
        """
        stripped = 0
        for fact in self._facts:
            if not fact.is_valid and fact.embedding is not None:
                fact.embedding = None
                stripped += 1
        if stripped:
            if save:
                self.save()
            logger.info(f"Stripped embeddings from {stripped} tombstone(s)")
        return stripped

    def prune_stale_facts(self, save: bool = True) -> int:
        """Remove valid-but-useless facts: low confidence, zero successes, old enough.

        Targets noise that was never validated. Does NOT touch CONSTRAINT facts
        or recently-created facts. INDEPENDENT-tagged facts are pure observation
        noise and prune twice as fast (7 days instead of 14).

        Returns the number of pruned facts.
        """
        now = time.time()
        default_stale_age = STALE_MIN_AGE_DAYS * 86400
        independent_stale_age = 7 * 86400
        probation_stale_age = PROBATION_AGE_DAYS * 86400
        pruned = 0

        for fact in self._facts:
            if not fact.is_valid:
                continue
            if fact.kind == FactKind.CONSTRAINT:
                continue
            if fact.metadata.success_count > 0:
                continue
            # Curated facts (seed, community, synthesized) are protected
            if PROTECTED_TAGS & set(fact.tags):
                continue

            is_independent = "independent" in fact.tags
            is_probation = PROBATION_TAG in fact.tags
            # Probation has the shortest window — a hot-buffer fact that
            # wasn't re-accessed within PROBATION_AGE_DAYS gets dropped.
            if is_probation:
                stale_age = probation_stale_age
            elif is_independent:
                stale_age = independent_stale_age
            else:
                stale_age = default_stale_age
            # Promoted-out-of-probation facts AND independent facts share
            # the "low confidence by construction" behavior — they prune
            # purely on the clock, no STALE_MAX_CONFIDENCE gate.
            if (
                not is_independent
                and not is_probation
                and fact.metadata.confidence >= STALE_MAX_CONFIDENCE
            ):
                continue
            if (now - fact.metadata.created_at) < stale_age:
                continue

            self._invalidate(fact)
            pruned += 1

        if pruned:
            if save:
                self.save()
            logger.info(f"Pruned {pruned} stale unvalidated fact(s)")
        return pruned

    def demote_unhelpful_facts(self, save: bool = True) -> int:
        """Demote or prune facts that are retrieved but never lead to accepted suggestions.

        Two tiers:
        - access_count 5-9, success 0: reduce confidence by 0.1 (floor 0.1)
        - access_count >= 10, success 0: mark invalid (actively unhelpful)

        Facts with hit rate > 30% get a small confidence boost (protection).

        Returns the total number of facts affected (demoted + pruned + boosted).
        """
        now = time.time()
        min_age = DEMOTION_MIN_AGE_DAYS * 86400
        affected = 0

        for fact in self._facts:
            if not fact.is_valid:
                continue
            if fact.kind == FactKind.CONSTRAINT:
                continue
            # Curated facts (seed, community, synthesized) are protected
            if PROTECTED_TAGS & set(fact.tags):
                continue
            if (now - fact.metadata.created_at) < min_age:
                continue
            if fact.metadata.access_count < DEMOTION_MIN_ACCESS:
                continue

            access = fact.metadata.access_count
            success = fact.metadata.success_count

            if success == 0:
                if access >= DEMOTION_PRUNE_ACCESS:
                    # Hard prune: accessed 10+ times, never helpful
                    self._invalidate(fact)
                    affected += 1
                else:
                    # Soft demotion: reduce confidence
                    fact.metadata.confidence = max(
                        DEMOTION_CONFIDENCE_FLOOR,
                        fact.metadata.confidence - DEMOTION_CONFIDENCE_PENALTY,
                    )
                    affected += 1
            elif success > 0 and (success / access) >= PROTECTION_HIT_RATE:
                # Protect consistently helpful facts
                new_conf = min(1.0, fact.metadata.confidence + PROTECTION_BOOST)
                if new_conf != fact.metadata.confidence:
                    fact.metadata.confidence = new_conf
                    affected += 1

        if affected:
            if save:
                self.save()
            logger.info(f"Demote/protect cycle affected {affected} fact(s)")
        return affected

    def load(self) -> None:
        """Load facts from all scoped files and merge."""
        self._facts = []
        # Fresh load = fresh concurrency bookkeeping: forget prior deletions
        # and re-baseline the per-file mtimes we compare against on save.
        self._deleted_ids = set()
        self._scope_mtimes = {}
        for path in (self._global_path, self._org_path, self._project_path):
            if not path:
                continue
            self._facts.extend(self._load_file(path))
            mt = self._file_mtime(path)
            if mt is not None:
                self._scope_mtimes[str(path)] = mt
        logger.info(f"FactStore: Loaded {len(self._facts)} facts")
        # Cap runs in-memory only; save is deferred to initialize()
        self._cap_independent_facts(save=False)

    def _cap_independent_facts(self, save: bool = True) -> None:
        """Prevent bloat: invalidate excess independent-tagged facts.

        Active repos can generate hundreds of independent outcomes per week.
        Keep only the newest MAX_INDEPENDENT_FACTS to maintain retrieval quality.
        Protected-tag facts are never capped.

        Args:
            save: If True, persist after capping. Set False when called from
                  load() to avoid save-during-load (deferred to initialize()).
        """
        indep = [
            f for f in self._facts
            if f.is_valid and "independent" in f.tags
            and not (PROTECTED_TAGS & set(f.tags))
        ]
        if len(indep) <= MAX_INDEPENDENT_FACTS:
            return
        indep.sort(key=lambda f: f.metadata.created_at, reverse=True)
        pruned = 0
        for f in indep[MAX_INDEPENDENT_FACTS:]:
            # cascade=False preserves the cap's prior behavior: independent
            # facts are observation noise with no dependents worth flagging.
            self._invalidate(f, cascade=False)
            pruned += 1
        if pruned:
            if save:
                self.save()
            else:
                self._cap_pending = True
            logger.info(f"Capped independent facts: invalidated {pruned}, kept {MAX_INDEPENDENT_FACTS}")

    def _save_file(self, path: Path, facts: list[Fact]) -> None:
        """Save a list of facts to a JSON file atomically.

        Writes to a temp file in the same directory, then renames.
        Uses mkstemp to avoid collisions with concurrent processes.
        """
        import os
        import tempfile

        try:
            data = {
                "version": "2.0",
                "facts": [f.to_dict() for f in facts],
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as fh:
                    json.dump(data, fh, indent=2)
                os.replace(tmp_name, str(path))
            except BaseException:
                os.unlink(tmp_name)
                raise
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
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load facts from {path}: {e}")
            self._backup_corrupt_file(path)
            return []
        except OSError as e:
            logger.error(f"Failed to load facts from {path}: {e}")
            return []

    @staticmethod
    def _backup_corrupt_file(path: Path) -> None:
        """Preserve a corrupt fact file before future saves replace it."""
        backup = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
        try:
            shutil.copy2(path, backup)
            logger.warning(f"Backed up corrupt fact file to {backup}")
        except OSError as backup_error:
            logger.warning(f"Failed to back up corrupt fact file {path}: {backup_error}")

    # ------------------------------------------------------------------ #
    # Supersession
    # ------------------------------------------------------------------ #

    @staticmethod
    def _canonical_signature(fact: Fact) -> str:
        """Stable signature for exact-twin detection.

        Uses the generalize util (paper 2603.10600 §7) to strip identifiers,
        version numbers, paths, etc. before hashing, so "Updated 5 files"
        and "Updated 9 files" collapse to the same signature. Includes the
        kind+scope so a CONSTRAINT can't accidentally dedup against a
        PATTERN with similar text.
        """
        from neo.memory.generalize import generalize

        canonical = generalize(f"{fact.subject} {fact.body}")
        return f"{fact.kind.value}|{fact.scope.value}|{canonical}"

    def _exact_canonical_match(self, new_fact: Fact) -> Optional[Fact]:
        """Return an existing fact whose canonical signature matches new_fact.

        O(n) over valid facts — fine at the per-scope caps. Cached signatures
        would be a future optimization if write-throughput becomes a hotspot.
        """
        target = self._canonical_signature(new_fact)
        for fact in self._facts:
            if not fact.is_valid:
                continue
            if fact.scope != new_fact.scope or fact.kind != new_fact.kind:
                continue
            if self._canonical_signature(fact) == target:
                return fact
        return None

    def _find_supersession_candidate(self, new_fact: Fact) -> Optional[Fact]:
        """Find an existing fact that the new fact should supersede.

        Criteria: same scope + kind, cosine similarity > threshold.

        Tiebreaker among multiple candidates above threshold follows the
        survey-paper precedence (2603.07670 §7.3, 2603.19935 instr. #4):

          1. Newer event_time wins         (most-recent reality)
          2. Higher source-provenance wins (STRUCTURAL > OBSERVED > INFERRED)
          3. Higher cosine wins            (closer semantic match)

        This makes conflict resolution deterministic: when two old facts
        both look like supersession candidates, we replace the most stale
        one with the lower provenance — not just "whichever scored highest."
        """
        if new_fact.embedding is None:
            return None

        candidates: list[tuple[Fact, float]] = []
        for fact in self._facts:
            if not fact.is_valid:
                continue
            if fact.scope != new_fact.scope or fact.kind != new_fact.kind:
                continue
            if fact.embedding is None:
                continue
            sim = self._cosine_similarity(new_fact.embedding, fact.embedding)
            if sim > SUPERSESSION_THRESHOLD:
                candidates.append((fact, sim))

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0][0]

        # Deterministic precedence: oldest event_time gets superseded first
        # (it's the most stale), with ties broken by lower provenance and
        # lower cosine — so we replace the weakest match, not the strongest.
        provenance_rank = {
            Provenance.STRUCTURAL.value: 2,
            Provenance.OBSERVED.value: 1,
            Provenance.INFERRED.value: 0,
        }
        candidates.sort(
            key=lambda fc: (
                fc[0].metadata.effective_event_time,
                provenance_rank.get(fc[0].metadata.provenance, 0),
                fc[1],
            )
        )
        return candidates[0][0]

    def _supersede(self, old: Fact, new: Fact) -> None:
        """Supersede an old fact with a new one and cascade needs_review.

        Bi-temporal soft-delete: stamp event_time_end on the old fact at the
        new fact's event_time (or now), so the old row stays auditable. The
        is_valid flag still gates retrieval; event_time_end is for callers
        who want to ask "what was true at time T?" without losing history.
        """
        # Invalidate + strip + cascade dependents. superseded_by / event_time_end
        # are supersession-specific and set here, not in _invalidate.
        self._invalidate(old)
        old.superseded_by = new.id
        new.supersedes = old.id

        # Carry forward confidence with a small boost, but never downgrade
        carry_forward = min(1.0, old.metadata.confidence + 0.05)
        new.metadata.confidence = max(new.metadata.confidence, carry_forward)

        # Bi-temporal: end the old fact's validity at the new fact's event time
        if old.metadata.event_time_end is None:
            old.metadata.event_time_end = new.metadata.effective_event_time

        logger.info(f"Superseded fact '{old.subject[:40]}' with '{new.subject[:40]}'")

    def _invalidate(self, fact: Fact, *, cascade: bool = True) -> None:
        """Single choke point for marking a fact invalid.

        Sets ``is_valid = False`` and drops the embedding at the transition: a
        tombstone is never retrieved, deduped, or clustered (all such paths
        pre-filter ``is_valid``), so its 768-dim vector is immediately dead
        weight. Stripping here keeps bloat from accumulating between the
        periodic ``strip_tombstone_embeddings`` sweeps, which now only backfill
        tombstones created off this path — an ingester superseding a fact, or a
        peer process's still-embedded copy reconciled in.

        Cascades ``needs_review`` to dependents unless ``cascade=False``. The
        independent-fact cap opts out (observation noise, no dependents worth
        flagging) — matching its prior behavior exactly. Supersession-specific
        state (``superseded_by``, ``event_time_end``) stays at the call site;
        this owns only the invalidate + strip + cascade triple. Idempotent.
        """
        fact.is_valid = False
        fact.embedding = None
        if cascade:
            self._cascade_needs_review(fact.id)

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

        self._ensure_embedder()

        cache_key = hashlib.sha256(text.encode()).hexdigest()
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
        return cosine_similarity(a, b)

    # ------------------------------------------------------------------ #
    # Seed fact ingestion
    # ------------------------------------------------------------------ #

    def _ingest_seed_facts(self) -> None:
        """Load curated facts bundled with the neo package.

        Seed facts ship with every release and provide community-curated
        patterns (security, performance, reliability). Re-ingested only
        when the seed file changes (i.e., after a package update).
        """
        ingester = SeedIngester(
            org_id=self.org_id,
            project_id=self.project_id,
        )
        new_facts, superseded_facts = ingester.ingest(self._facts)

        if new_facts or superseded_facts:
            for fact in new_facts:
                if fact.embedding is None:
                    embed_text = f"{fact.subject} {fact.body}"
                    fact.embedding = self._embed_text(embed_text)

            self._facts.extend(new_facts)
            self.save()
            logger.info(
                f"Seed facts: {len(new_facts)} new, {len(superseded_facts)} superseded"
            )

    def _ingest_community_feed(self) -> None:
        """Fetch and ingest community-curated patterns from remote feed.

        Checks once per day, caches locally, falls back to cache on
        network failure. Patterns are contributed via PR and ship
        between releases.
        """
        try:
            ingester = CommunityFeedIngester(
                org_id=self.org_id,
                project_id=self.project_id,
            )
            new_facts, superseded_facts = ingester.ingest(self._facts)

            if new_facts or superseded_facts:
                for fact in new_facts:
                    if fact.embedding is None:
                        embed_text = f"{fact.subject} {fact.body}"
                        fact.embedding = self._embed_text(embed_text)

                self._facts.extend(new_facts)
                self.save()
                logger.info(
                    f"Community feed: {len(new_facts)} new, "
                    f"{len(superseded_facts)} superseded"
                )
        except Exception as e:
            # Network failures should never block neo startup
            logger.debug(f"Community feed ingestion failed: {e}")

    # ------------------------------------------------------------------ #
    # Constraint ingestion
    # ------------------------------------------------------------------ #

    def _ingest_constraints(self) -> None:
        """Scan constraint files and ingest/update constraint facts."""
        config_auto_scan = getattr(self._config, "constraint_auto_scan", True)
        if not config_auto_scan:
            return

        ingester = ConstraintIngester(
            codebase_root=self.codebase_root or "",
            org_id=self.org_id,
            project_id=self.project_id,
        )
        new_facts, superseded_facts = ingester.ingest(self._facts)

        if new_facts or superseded_facts:
            # Generate embeddings for new constraint facts
            for fact in new_facts:
                if fact.embedding is None:
                    embed_text = f"{fact.subject} {fact.body}"
                    fact.embedding = self._embed_text(embed_text)

            self._facts.extend(new_facts)
            self.save()
            logger.info(
                f"Constraints: {len(new_facts)} new, {len(superseded_facts)} superseded"
            )

    def _ingest_claude_memory(self) -> None:
        """Ingest curated knowledge from Claude Code's auto-memory system.

        Reads ~/.claude/projects/{project-id}/memory/*.md files that Claude
        Code has distilled from past conversations. These are pre-curated,
        high-quality facts about the project.
        """
        ingester = ClaudeMemoryIngester(
            codebase_root=self.codebase_root or "",
            org_id=self.org_id,
            project_id=self.project_id,
        )
        new_facts, superseded_facts = ingester.ingest(self._facts)

        if new_facts or superseded_facts:
            for fact in new_facts:
                if fact.embedding is None:
                    embed_text = f"{fact.subject} {fact.body}"
                    fact.embedding = self._embed_text(embed_text)

            self._facts.extend(new_facts)
            self.save()
            logger.info(
                f"Claude memory: {len(new_facts)} new, {len(superseded_facts)} superseded"
            )

    # ------------------------------------------------------------------ #
    # Git history ingestion
    # ------------------------------------------------------------------ #

    def _ingest_git_history(self) -> None:
        """Learn from git commit history that hasn't been ingested yet.

        Runs on each initialization to catch up on commits made since
        the last neo invocation. Uses a watermark to avoid re-ingesting.
        """
        try:
            records = self._outcome_tracker.ingest_git_history(max_commits=50)
        except Exception as e:
            logger.warning(f"Git history ingestion failed: {e}")
            return

        if not records:
            return

        for record in records:
            self.add_fact(
                subject=record["subject"],
                body=record["body"],
                kind=FactKind.REVIEW,
                scope=FactScope.PROJECT,
                confidence=0.5,
                tags=["history", "git-commit"],
            )

        logger.info(f"Ingested {len(records)} facts from git history")

    # ------------------------------------------------------------------ #
    # Review synthesis
    # ------------------------------------------------------------------ #

    def synthesize_reviews(self) -> int:
        """Distill clusters of REVIEW facts into higher-level PATTERN/FAILURE facts.

        Triple-trigger from SCM (paper 2604.20943 §3.6): consolidation fires
        when ANY of the following holds:

          1. count-delta  ≥ 10  — the legacy gate (preserved for back-compat)
          2. elapsed time ≥ 1h since last consolidation
          3. entropy(value-score over REVIEW facts) > 0.9  — high-uncertainty
             corpus state where consolidation usually pays off most

        Groups by tag, clusters by cosine similarity, synthesizes clusters of
        3+ into a single fact that supersedes the sources. The watermark
        tracks total REVIEW facts ever seen (valid + invalidated) to avoid
        the count-drift bug where synthesis invalidates facts and the
        watermark comparison breaks.

        Returns:
            Number of synthesized facts created.
        """
        # Count ALL review facts (valid + invalid) for watermark comparison
        all_review_count = sum(
            1 for f in self._facts if f.kind == FactKind.REVIEW
        )
        valid_review_facts = [
            f for f in self._facts
            if f.is_valid and f.kind == FactKind.REVIEW
        ]

        if len(valid_review_facts) < 20:
            return 0

        # Triple-trigger gate. Any single condition is enough.
        watermark = self._load_synthesis_watermark()
        count_delta = all_review_count - watermark
        elapsed_seconds = time.time() - self._load_synthesis_timestamp()
        entropy_score = self._review_entropy(valid_review_facts)

        # Logs are deliberately one-line so it's easy to grep which trigger
        # fired in production: most consolidations should be count-driven,
        # but a corpus that goes high-entropy without writes (e.g. lots of
        # outcome-driven confidence shifts) should still consolidate.
        if (
            count_delta < 10
            and elapsed_seconds < 3600.0
            and entropy_score <= 0.9
        ):
            return 0

        logger.info(
            "synthesis-trigger: count_delta=%d elapsed=%.1fs entropy=%.3f",
            count_delta, elapsed_seconds, entropy_score,
        )

        synthesized_count = 0

        # Group by primary tag (outcome:accepted, outcome:independent, history:*)
        groups: dict[str, list[Fact]] = {}
        for fact in valid_review_facts:
            key = self._synthesis_group_key(fact)
            groups.setdefault(key, []).append(fact)

        for group_key, facts in groups.items():
            if len(facts) < 3:
                continue

            clusters = self._cluster_by_similarity(facts, SYNTHESIS_SIMILARITY)

            for cluster in clusters:
                if len(cluster) < 3:
                    continue

                # NREM Hebbian strengthening (paper 2604.20943 §3, the
                # consolidation phase): facts that survived clustering
                # co-occurred — their mutual reinforcement is the cluster
                # itself. Bump each member's confidence by η · |cluster|
                # before synthesis runs, so the synthesized fact
                # inherits a strengthened lineage.
                self._hebbian_strengthen(cluster)

                new_fact = self._synthesize_cluster(cluster, group_key)
                if new_fact:
                    self._facts.append(new_fact)
                    synthesized_count += 1

        if synthesized_count:
            # Global downscale (paper 2604.20943 §3, α = 0.8 → too
            # aggressive at our scale; we use a gentler 0.97 multiplier
            # so an unused fact loses ~3% per consolidation cycle).
            # Keeps confidence values from drifting upward forever as
            # the Hebbian step accumulates.
            self._global_confidence_downscale(alpha=0.97)
            self.save()
            logger.info(f"Synthesized {synthesized_count} fact(s) from REVIEW clusters")

        # Save total count as watermark (valid + invalidated)
        self._save_synthesis_watermark(all_review_count)
        return synthesized_count

    @staticmethod
    def _hebbian_strengthen(cluster: list[Fact], *, eta: float = 0.02) -> None:
        """Bump each cluster member's confidence by η · cluster_size.

        Bounded by the [0, 1] interval. Small η keeps individual
        strengthens from dominating success_bonus.
        """
        boost = min(0.1, eta * len(cluster))
        for fact in cluster:
            fact.metadata.confidence = min(1.0, fact.metadata.confidence + boost)

    def _global_confidence_downscale(self, *, alpha: float) -> None:
        """Multiply all non-curated, valid facts' confidence by alpha.

        Curated/CONSTRAINT/ARCHITECTURE/DECISION facts skip the decay
        (mirrors the rank_score curated-bypass policy). Floor at 0.05 so
        nothing collapses to zero — a long-dormant fact stays visible.
        """
        from neo.memory.models import _decays  # local: cycle

        for fact in self._facts:
            if not fact.is_valid:
                continue
            if not _decays(fact):
                continue
            new_conf = max(0.05, fact.metadata.confidence * alpha)
            fact.metadata.confidence = new_conf

    @staticmethod
    def _synthesis_group_key(fact: Fact) -> str:
        """Determine the grouping key for a REVIEW fact based on its tags."""
        tags = set(fact.tags)
        if "accepted" in tags:
            return "outcome:accepted"
        if "independent" in tags:
            return "outcome:independent"
        if "history" in tags:
            return "history"
        return "other"

    def _cluster_by_similarity(
        self, facts: list[Fact], threshold: float
    ) -> list[list[Fact]]:
        """Cluster facts by cosine similarity using complete-linkage.

        A candidate joins a cluster only if it meets the similarity threshold
        against ALL existing cluster members, ensuring all facts in a cluster
        are mutually similar.

        Returns list of clusters (each a list of facts).
        """
        return cluster_by_similarity(
            facts, embed_fn=lambda f: f.embedding, threshold=threshold
        )

    def _synthesize_cluster(
        self, cluster: list[Fact], group_key: str
    ) -> Optional[Fact]:
        """Create a single synthesized fact from a cluster of REVIEW facts.

        For clusters of 5+, attempts LLM-based synthesis for richer output.
        Falls back to mechanical synthesis on any error or for smaller clusters.
        """
        if not cluster:
            return None

        # Pick highest-confidence as base
        cluster.sort(key=lambda f: f.metadata.confidence, reverse=True)
        base = cluster[0]

        # Accepted suggestions become validated patterns; everything else stays REVIEW
        kind = FactKind.PATTERN if group_key == "outcome:accepted" else FactKind.REVIEW

        # Try LLM synthesis for large clusters
        subject = None
        body_text = None
        avg_conf = sum(f.metadata.confidence for f in cluster) / len(cluster)

        if len(cluster) >= 5 and self._lm_adapter:
            llm_result = self._llm_synthesize(cluster, group_key)
            if llm_result:
                subject, body_text, llm_conf = llm_result
                avg_conf = llm_conf

        # Mechanical fallback
        if subject is None:
            subject = self._extract_common_subject(cluster)
        if body_text is None:
            seen_lines: set[str] = set()
            merged_body_parts: list[str] = []
            for fact in cluster:
                if len(merged_body_parts) >= 20:
                    break
                for line in fact.body.split("\n"):
                    stripped = line.strip()
                    if stripped and stripped not in seen_lines:
                        seen_lines.add(stripped)
                        merged_body_parts.append(stripped)
                    if len(merged_body_parts) >= 20:
                        break
            body_text = "\n".join(merged_body_parts[:20])

        embedding = self._embed_text(f"{subject} {body_text}")
        if embedding is None:
            embedding = base.embedding

        new_fact = Fact(
            subject=subject,
            body=body_text,
            kind=kind,
            scope=FactScope.PROJECT,
            org_id=self.org_id,
            project_id=self.project_id,
            metadata=FactMetadata(
                confidence=avg_conf,
                source_prompt=f"synthesized from {len(cluster)} REVIEW facts",
            ),
            embedding=embedding,
            tags=["synthesized"] + list({t for f in cluster for t in f.tags}),
        )

        # Supersede all source facts and cascade dependency reviews
        for fact in cluster:
            self._invalidate(fact)
            fact.superseded_by = new_fact.id

        return new_fact

    def _llm_synthesize(
        self, cluster: list[Fact], group_key: str
    ) -> Optional[tuple[str, str, float]]:
        """Use LLM to synthesize a cluster into a concise actionable insight.

        Returns (subject, body, confidence) or None on any error.
        """
        if not self._lm_adapter:
            return None

        try:
            # Build compact prompt from cluster data
            facts_text = "\n".join(
                f"- [{f.subject}] {f.body[:200]}"
                for f in cluster[:10]  # Cap at 10 to keep prompt small
            )
            group_label = group_key.replace("outcome:", "")

            messages = [
                {
                    "role": "system",
                    "content": (
                        f"Distill these {len(cluster)} related {group_label} code observations "
                        "into one actionable insight.\n"
                        "Return exactly 3 lines:\n"
                        "SUBJECT: <concise label, max 60 chars>\n"
                        "BODY: <actionable guidance, 1-3 sentences>\n"
                        "CONFIDENCE: <0.0-1.0 based on consistency of evidence>"
                    ),
                },
                {"role": "user", "content": facts_text},
            ]

            response = self._lm_adapter.generate(
                messages=messages,
                max_tokens=300,
                temperature=0.3,
                reasoning_effort="low",  # Distilling clustered facts; not reasoning.
            )

            return self._parse_llm_synthesis(response)

        except Exception as e:
            logger.debug(f"LLM synthesis failed (falling back to mechanical): {e}")
            return None

    @staticmethod
    def _parse_llm_synthesis(response: str) -> Optional[tuple[str, str, float]]:
        """Parse the 3-line LLM response into (subject, body, confidence)."""
        subject = None
        body = None
        confidence = None

        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("SUBJECT:"):
                subject = line.split(":", 1)[1].strip()[:60]
            elif line.upper().startswith("BODY:"):
                body = line.split(":", 1)[1].strip()
            elif line.upper().startswith("CONFIDENCE:"):
                try:
                    confidence = float(line.split(":", 1)[1].strip())
                    confidence = max(0.0, min(1.0, confidence))
                except ValueError:
                    pass

        if subject and body and confidence is not None:
            return subject, body, confidence
        return None

    @staticmethod
    def _extract_common_subject(facts: list[Fact]) -> str:
        """Find a shared file or area across cluster members for the subject line."""
        import re
        # Match file-like tokens: must contain / or have a code file extension
        file_pattern = re.compile(r'(\S+/\S+|\S+\.(?:py|js|ts|tsx|go|rs|java|c|cpp|rb|swift))\b')
        file_mentions: Counter = Counter()
        for fact in facts:
            for match in file_pattern.findall(fact.subject):
                file_mentions[match] += 1

        if file_mentions:
            most_common = file_mentions.most_common(1)[0][0]
            return f"pattern: {most_common}"

        # Fallback: use first few words of subjects
        if facts:
            return f"pattern: {facts[0].subject[:50]}"
        return "pattern: synthesized"

    def _load_synthesis_watermark(self) -> int:
        """Load the count of REVIEW facts at last synthesis run."""
        if not self._project_path:
            return 0
        watermark_path = self._project_path.parent / f"synthesis_watermark_{self.project_id}.json"
        if not watermark_path.exists():
            return 0
        try:
            data = json.loads(watermark_path.read_text())
            return data.get("review_count", 0)
        except (json.JSONDecodeError, OSError):
            return 0

    def _load_synthesis_timestamp(self) -> float:
        """Wall-clock time of the last completed consolidation, or 0 if none."""
        if not self._project_path:
            return 0.0
        watermark_path = self._project_path.parent / f"synthesis_watermark_{self.project_id}.json"
        if not watermark_path.exists():
            return 0.0
        try:
            data = json.loads(watermark_path.read_text())
            return float(data.get("updated_at", 0.0))
        except (json.JSONDecodeError, OSError):
            return 0.0

    @staticmethod
    def _review_entropy(facts: list[Fact]) -> float:
        """Shannon entropy of REVIEW facts' confidence distribution.

        Buckets confidence into deciles [0.0..1.0] and computes
        H = −Σ p_i log2 p_i, normalized by log2(num_nonzero_bins) so the
        result is in [0, 1]. High entropy ≈ uniform → consolidation gets a
        lot of leverage. Low entropy ≈ already-clustered → not much to gain.
        """
        if not facts:
            return 0.0
        import math as _math

        buckets = [0] * 10
        for fact in facts:
            c = max(0.0, min(0.999, fact.metadata.confidence))
            buckets[int(c * 10)] += 1
        total = sum(buckets)
        if total <= 0:
            return 0.0
        probs = [b / total for b in buckets if b > 0]
        if len(probs) <= 1:
            return 0.0
        h = -sum(p * _math.log2(p) for p in probs)
        return h / _math.log2(len(probs))

    def _save_synthesis_watermark(self, count: int) -> None:
        """Save the current REVIEW fact count as synthesis watermark."""
        if not self._project_path:
            return
        watermark_path = self._project_path.parent / f"synthesis_watermark_{self.project_id}.json"
        try:
            atomic_write_json(watermark_path, {
                "review_count": count,
                "updated_at": time.time(),
            })
        except OSError as e:
            logger.debug(f"Failed to save synthesis watermark: {e}")

    # ------------------------------------------------------------------ #
    # Migration
    # ------------------------------------------------------------------ #

    def _migrate_legacy_project_id_files(self, codebase_root: Optional[str]) -> None:
        """Rename fact + watermark files keyed by the legacy (path-only)
        project ID to the new (git-remote-hashed) project ID.

        No-op when there's no remote (legacy id equals current id) or when
        no legacy file exists. Runs before `load()` so the rename is
        transparent to callers.
        """
        if not self.project_id:
            return
        legacy_id = _compute_legacy_project_id(codebase_root)
        if not legacy_id or legacy_id == self.project_id:
            return

        legacy_facts = FACTS_DIR / f"facts_project_{legacy_id}.json"
        new_facts = FACTS_DIR / f"facts_project_{self.project_id}.json"
        if legacy_facts.exists() and not new_facts.exists():
            try:
                legacy_facts.rename(new_facts)
                logger.info(
                    f"Migrated project facts {legacy_id[:8]} → {self.project_id[:8]} "
                    "(now keyed by git remote URL)"
                )
            except OSError as e:
                logger.warning(f"Legacy fact-file rename failed (non-fatal): {e}")

        legacy_wm = FACTS_DIR / f"synthesis_watermark_{legacy_id}.json"
        new_wm = FACTS_DIR / f"synthesis_watermark_{self.project_id}.json"
        if legacy_wm.exists() and not new_wm.exists():
            try:
                legacy_wm.rename(new_wm)
            except OSError as e:
                logger.debug(f"Legacy watermark rename failed (non-fatal): {e}")

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
