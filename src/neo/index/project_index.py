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

import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
import numpy as np

from neo.index.language_parser import TreeSitterParser

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

# Path exclusions live in ONE place: `context_gatherer.load_gitignore_patterns`,
# which both this module and prompt assembly read. A second list used to sit
# here; see `_build_exclusion_filter` for what the duplication cost.


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


def _chunk_embed_text(chunk: CodeChunk) -> str:
    """Build a structured text from a chunk for embedding.

    Format: ``symbols imports head_of_body``. Each segment is space-
    separated; segments missing from the chunk are skipped. The head is
    capped at 600 chars to keep the embedding focused on signature +
    docstring rather than diluting with implementation details.

    Empty body or missing symbols/imports return the empty string —
    fastembed's tokenizer treats that as a zero vector, which is the
    correct behavior for a chunk we can't characterize.
    """
    parts: list[str] = []
    if chunk.symbols:
        parts.append(" ".join(chunk.symbols))
    if chunk.imports:
        parts.append(" ".join(chunk.imports))
    if chunk.content:
        head = chunk.content[:600]
        parts.append(head)
    return " ".join(parts)


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
        self.edges_path = self.neo_dir / "edges.json"
        self.faiss_path = self.neo_dir / "faiss.index"

        # In-memory state
        self.chunks: List[CodeChunk] = []
        self.edges: List[Dict[str, Any]] = []
        self.snapshot: Optional[IndexSnapshot] = None
        self.faiss_index: Optional[Any] = None
        self.embedding_model: Optional[TextEmbedding] = None
        self.parser: Optional[TreeSitterParser] = None  # Lazy-initialized
        # Populated by build_index; describes what the cap left out. Not
        # persisted — it describes one build, not the index on disk.
        self.selection_report: Optional[Dict[str, Any]] = None

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

            # Load edges
            if self.edges_path.exists():
                with open(self.edges_path) as f:
                    self.edges = json.load(f)
                logger.info(f"Loaded {len(self.edges)} edges")

            # Load FAISS index if available
            if FAISS_AVAILABLE and self.faiss_path.exists():
                self.faiss_index = faiss.read_index(str(self.faiss_path))
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")

            logger.info(f"Loaded project index: {len(self.chunks)} chunks, {len(self.edges)} edges from {self.snapshot.total_files} files")

        except Exception as e:
            logger.error(f"Failed to load project index: {e}")
            # Reset to empty state
            self.chunks = []
            self.edges = []
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

            # Atomic write: write to unique temp files, then replace.
            self._atomic_write_json(self.snapshot_path, self.snapshot.__dict__)

            # Chunks
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

            self._atomic_write_json(self.chunks_path, chunks_data)

            # Edges
            if self.edges:
                self._atomic_write_json(self.edges_path, self.edges)

            # FAISS index
            if FAISS_AVAILABLE and self.faiss_index:
                fd, tmp_name = tempfile.mkstemp(dir=self.faiss_path.parent, suffix=".tmp")
                os.close(fd)
                try:
                    faiss.write_index(self.faiss_index, tmp_name)
                    os.replace(tmp_name, self.faiss_path)
                except BaseException:
                    os.unlink(tmp_name)
                    raise

            logger.info(f"Saved project index: {len(self.chunks)} chunks, {len(self.edges)} edges")

        except Exception as e:
            logger.error(f"Failed to save project index: {e}")
            raise

    @staticmethod
    def _atomic_write_json(path: Path, data: Any) -> None:
        """Write JSON via a unique same-directory temp file."""
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_name, path)
        except BaseException:
            os.unlink(tmp_name)
            raise

    def _build_exclusion_filter(self):
        """Return a predicate deciding whether a path is unfit to index.

        ONE source of truth: `context_gatherer.load_gitignore_patterns`,
        which carries both the repo's own `.gitignore` and the defaults that
        apply when it says nothing. There used to be a second list here, and
        the duplication cost exactly what duplication costs — when the
        gatherer's copy was corrected to stop hiding 60 git-tracked skill
        implementations, this copy kept hiding them, and the two subsystems
        disagreed about `.claude` until someone diffed them. 27 of that
        list's 31 names were already in the shared defaults; the other four
        were the ones that had to go anyway.

        `should_ignore` only ever tests the path it is handed, so a file
        under an ignored directory does not match on its own — the walk in
        `context_gatherer.iter_paths` prunes those directories separately.
        This reproduces that by testing each ancestor directory, memoized so
        a directory is judged once no matter how many files sit under it.

        `should_ignore` honors negation for a path evaluated directly. This
        caller still cannot re-include a file beneath an excluded ancestor,
        because `dir_excluded` short-circuits first — which is also git's
        rule, so the short-circuit is the correct behaviour rather than a
        limitation.
        """
        from neo.context_gatherer import load_gitignore_patterns, should_ignore

        patterns = load_gitignore_patterns(str(self.repo_root))
        verdicts: Dict[tuple, bool] = {}

        def dir_excluded(parts: tuple) -> bool:
            if parts in verdicts:
                return verdicts[parts]
            verdicts[parts] = (
                (len(parts) > 1 and dir_excluded(parts[:-1]))
                or should_ignore("/".join(parts), patterns, is_dir=True)
            )
            return verdicts[parts]

        def excluded(path: Path) -> bool:
            try:
                rel = path.relative_to(self.repo_root)
            except ValueError:
                return True
            if rel.parts[:-1] and dir_excluded(rel.parts[:-1]):
                return True
            return should_ignore(rel.as_posix(), patterns)

        return excluded

    @staticmethod
    def _allocate_slots(counts: Dict[str, int], budget: int) -> Dict[str, int]:
        """Divide `budget` index slots across languages by repo composition.

        Proportional, with a floor of one slot per language present. Both
        halves matter, and each fixes a different way of producing an index
        that cannot answer a question about the repository:

        - Proportional, because glob order is not a ranking. Concatenating
          `**/*.py` before `**/*.cs` and slicing gave a .NET backend of 4,272
          C# files an index of 83 Python files and zero C#.
        - Floor of one, because proportion alone rounds small languages to
          nothing. The same repo's 18 Python files deserve a slot; they just
          do not deserve all 100 of them.

        A language allocated more slots than it has files hands the surplus
        back, largest language first, so no budget is left on the table.
        """
        langs = [lang for lang, n in counts.items() if n > 0]
        if not langs or budget <= 0:
            return {}

        # Not enough budget to seat every language: the biggest ones take it.
        if budget <= len(langs):
            ordered = sorted(langs, key=lambda lang: (-counts[lang], lang))
            return {lang: 1 for lang in ordered[:budget]}

        total = sum(counts[lang] for lang in langs)
        shared = budget - len(langs)  # one slot per language already reserved
        alloc = {
            lang: 1 + int(counts[lang] / total * shared) for lang in langs
        }

        # Nobody gets more slots than they have files; redistribute the rest.
        # This loop also absorbs the slack left by rounding every share down,
        # so no separate largest-remainder pass is needed — which language
        # receives a ±1 rounding slot is not observable against a within-
        # language ranking that is itself only a path heuristic.
        alloc = {lang: min(alloc[lang], counts[lang]) for lang in langs}
        free = budget - sum(alloc.values())
        while free > 0:
            hungry = sorted(
                (lang for lang in langs if alloc[lang] < counts[lang]),
                key=lambda lang: (-counts[lang], lang),
            )
            if not hungry:
                break
            for lang in hungry:
                if free == 0:
                    break
                alloc[lang] += 1
                free -= 1

        return alloc

    @staticmethod
    def _cap_chunks(chunks: List[CodeChunk], cap: int) -> List[CodeChunk]:
        """Trim to `cap` by PROPORTIONAL apportionment across files.

        Never by slicing. Chunks arrive grouped by file and files by language,
        so `chunks[:cap]` on a 300-file C# repo kept 1000 C# chunks and dropped
        every TypeScript and Python chunk the file allocator had just fought to
        include — 63 of 100 selected files contributing nothing.

        Round-robin fixed that and introduced a subtler version of the same
        bias. Taking every file's first chunk before any file's second gives
        every file the SAME share regardless of how much is in it, so a 9 KB
        utility was fully represented while the modules the repository is
        built on were not:

            src/neo/memory/store.py    82 symbols,  6 indexed,   7%
            src/neo/engine.py          95 symbols,  6 indexed,   6%
            src/neo/text_budget.py      4 symbols,  4 indexed, 100%

        A file cannot be retrieved for what was never indexed, so the semantic
        channel was blind to 93% of the two files most likely to be relevant —
        the same anti-correlation the file scorer's size penalty had, arrived
        at independently and for a defensible reason.

        Proportional-with-a-floor keeps what round-robin was protecting (every
        selected file is represented, so none is invisible) and spends the rest
        where the code is. `_allocate_slots` already implements exactly this
        apportionment for the file budget, so it is reused rather than
        reimplemented and the two cannot drift.
        """
        if len(chunks) <= cap:
            return chunks

        by_file: Dict[str, List[CodeChunk]] = {}
        for chunk in chunks:
            by_file.setdefault(chunk.file_path, []).append(chunk)

        allocation = ProjectIndex._allocate_slots(
            {path: len(group) for path, group in by_file.items()}, cap
        )

        kept: List[CodeChunk] = []
        for path, group in by_file.items():
            kept.extend(group[: allocation.get(path, 0)])
        return kept

    def _select_files(
        self, file_patterns: List[str], max_files: int
    ) -> Tuple[List[Tuple[Path, str]], Dict[str, Any]]:
        """Pick which files to index, and report on what was left out.

        Returns (selected, report) where `selected` is a list of
        (path, content_hash) pairs — the hash is computed here to reject
        duplicate content, and returned so the caller need not recompute it.

        The report is what makes truncation visible. `--index` previously
        printed "Built index" and exited 0 whether it had indexed the
        repository or 83 files of a stale worktree copy.
        """
        from neo.languages import language_for_path

        # Deduplicate at the path level first: overlapping patterns would
        # otherwise present the same file twice — and would double-count it
        # in the exclusion tally the operator reads.
        is_excluded = self._build_exclusion_filter()
        candidates: Dict[Path, None] = {}
        excluded_paths: set = set()
        for pattern in file_patterns:
            for path in self.repo_root.glob(pattern):
                if is_excluded(path):
                    excluded_paths.add(path)
                    continue
                candidates.setdefault(path, None)
        excluded = len(excluded_paths)

        # Security checks live here so a rejected file backfills from the
        # same language's remaining candidates rather than costing a slot.
        repo_root_resolved = self.repo_root.resolve()
        by_language: Dict[str, List[Path]] = {}
        for path in candidates:
            # Symlink check comes first and uses lstat: both is_file() and
            # resolve() follow the link, and the point of rejecting symlinks
            # is to not touch what they point at.
            if path.is_symlink():
                logger.warning(f"Skipping symlink: {path}")
                continue
            # Also covers the file-vanished-since-glob race; a missing path is
            # not a file, and the slot it would have taken backfills below.
            if not path.is_file():
                logger.debug(f"Skipping non-file path: {path}")
                continue
            try:
                path.resolve().relative_to(repo_root_resolved)
            except ValueError:
                logger.warning(f"Skipping file outside repo: {path}")
                continue
            language = language_for_path(path) or path.suffix.lstrip('.')
            by_language.setdefault(language, []).append(path)

        # Rank within a language: shallower paths first, then alphabetical.
        # This is a weak heuristic, not centrality — but it is deterministic,
        # and it beats whatever order the filesystem happened to yield.
        for paths in by_language.values():
            paths.sort(key=lambda p: (len(p.relative_to(self.repo_root).parts), str(p)))

        counts = {lang: len(paths) for lang, paths in by_language.items()}
        eligible = sum(counts.values())
        allocation = self._allocate_slots(counts, max_files)

        selected: List[Tuple[Path, str]] = []
        seen_hashes: Dict[str, Path] = {}
        duplicates = 0
        examined = 0
        per_language: Dict[str, Dict[str, int]] = {}

        # One iterator per language, consumed by `take` and then resumed by
        # the refill pass — so a file examined in the quota pass is never
        # examined again.
        pending = {
            lang: iter(by_language[lang])
            for lang in sorted(by_language, key=lambda lang: (-counts[lang], lang))
        }

        def take(language: str, quota: int) -> int:
            """Consume up to `quota` usable files from `language`."""
            nonlocal duplicates, examined
            taken = 0
            while taken < quota:
                path = next(pending[language], None)
                if path is None:
                    break
                examined += 1
                file_hash = self._compute_file_hash(path)
                if not file_hash:
                    # Unreadable, or vanished since the glob. Not counted
                    # against the quota, so the next candidate takes its slot.
                    continue
                # Content-identical files must not each consume a slot: they
                # produce identical chunks and identical embeddings, so the
                # second copy buys nothing and costs a file the index could
                # have held instead.
                #
                # Two accepted costs. A skipped duplicate never enters
                # `snapshot.file_hashes`, so `check_staleness` will not
                # notice if the copies later diverge — it is not indexed, so
                # there is nothing stale to refresh, and the next full build
                # re-decides. And genuinely distinct files that happen to be
                # byte-identical (barrel `index.ts`, empty `__init__.py`,
                # generated boilerplate) lose their path from the index; the
                # content is still there under the first path that had it.
                if file_hash in seen_hashes:
                    duplicates += 1
                    logger.debug(f"Skipping duplicate content: {path}")
                    continue
                seen_hashes[file_hash] = path
                selected.append((path, file_hash))
                taken += 1
            return taken

        for language in pending:
            per_language[language] = {
                'selected': take(language, allocation.get(language, 0)),
                'eligible': counts[language],
            }

        # Refill across languages. `take` backfills within a language, but a
        # language whose remaining candidates are all duplicates (or that ran
        # out entirely) leaves its quota unspent — and dedup must not quietly
        # shrink the index below what the operator allowed. Largest language
        # first, matching how `_allocate_slots` redistributes.
        for language in pending:
            if len(selected) >= max_files:
                break
            per_language[language]['selected'] += take(
                language, max_files - len(selected)
            )

        report = {
            'eligible': eligible,
            'selected': len(selected),
            'excluded': excluded,
            'duplicates': duplicates,
            'max_files': max_files,
            # Truncated means THE CAP BOUND US, and the test that cannot be
            # fooled is whether any candidate went unexamined: the loops stop
            # pulling from a language's iterator precisely when the budget is
            # spent. Counting instead on `selected < eligible` blamed dedup on
            # the cap — 7 files with 5 duplicates under a cap of 1000 reported
            # "2 of 7 eligible files (capped at --max-files=1000)", naming a
            # knob that was never reached and that raising cannot help. So did
            # `selected >= max_files`, whenever max_files landed exactly on the
            # unique-file count. Duplicates and exclusions get their own lines.
            'truncated': examined < eligible,
            'per_language': per_language,
        }
        return selected, report

    def build_index(self, file_patterns: List[str] = None, languages: List[str] = None, max_files: int = 100):
        """
        Build initial index for repository.

        Args:
            file_patterns: Glob patterns for files to index
            languages: Languages to index (e.g., ['python', 'csharp', 'typescript'])
            max_files: Maximum files to index (prevent runaway on large repos)
        """
        # Lazy-initialize parser
        if self.parser is None:
            try:
                self.parser = TreeSitterParser()
            except ImportError as e:
                raise RuntimeError(
                    "Tree-sitter is required for indexing. "
                    "Install with: pip install neo-reasoner"
                ) from e

        # Auto-generate file patterns from languages if specified
        if languages and not file_patterns:
            file_patterns = self._patterns_from_languages(languages)
            if not file_patterns:
                valid_languages = self.parser.get_supported_languages() if self.parser else []
                raise ValueError(
                    f"No valid languages found in {languages}. "
                    f"Supported languages: {', '.join(sorted(valid_languages))}"
                )
        elif not file_patterns:
            # Default to all supported languages
            file_patterns = [
                "**/*.py", "**/*.cs", "**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx",
                "**/*.java", "**/*.go", "**/*.rs", "**/*.c", "**/*.cpp", "**/*.h", "**/*.hpp"
            ]

        logger.info(f"Building project index for {self.repo_root}")
        logger.info(f"File patterns: {file_patterns}")
        start_time = time.time()

        # Initialize snapshot
        self.snapshot = IndexSnapshot()
        self.snapshot.commit_hash = self._get_git_commit()

        # Find files to index: excluded paths dropped, ranked across
        # languages by repo composition, then cut to `max_files`.
        files_to_index, selection = self._select_files(file_patterns, max_files)
        self.selection_report = selection
        self.snapshot.total_files = len(files_to_index)

        # Extract chunks and edges from each file
        all_chunks = []
        all_edges = []
        for file_path, file_hash in files_to_index:
            rel_path = file_path.relative_to(self.repo_root)
            chunks = self._extract_chunks_from_file(file_path, str(rel_path))
            all_chunks.extend(chunks)

            # Extract edges
            edges = self._extract_edges_from_file(file_path, str(rel_path))
            all_edges.extend(edges)

            # Hash was computed during selection (it is what rejects
            # duplicate content), so reuse it rather than re-reading.
            self.snapshot.file_hashes[str(rel_path)] = file_hash

        # Limit total chunks. Round-robin across files rather than slicing,
        # for exactly the reason the file budget is apportioned rather than
        # sliced — see _cap_chunks.
        total_chunks = len(all_chunks)
        # Measured BEFORE the cap, and that ordering is the whole point. A
        # selected file can end up absent from the index for two unrelated
        # reasons, and only one of them is a cap:
        #
        #   - it yielded no chunks at all, because it holds no function,
        #     class, interface or struct for the grammar to match. An empty
        #     `__init__.py`, an enum-only `.cs`, a type-alias-only `.ts`. No
        #     cap is involved and no cap setting changes it.
        #   - it yielded chunks and the cap took every one of them.
        #
        # Without this line the two are indistinguishable downstream, and the
        # report blamed the chunk cap for both. It printed "the 1000-chunk cap
        # is below the 25 files selected; lower --max-files" for a build that
        # kept 559 of 559 chunks — naming a cap that had not fired, on
        # evidence `chunks_capped` contradicted one key over, and prescribing
        # a remedy that indexes strictly less. Keep the causes separate here
        # so the console never has to guess between them.
        files_producing_chunks = len({c.file_path for c in all_chunks})
        all_chunks = self._cap_chunks(all_chunks, MAX_CHUNKS_PER_REPO)
        selection['chunks_extracted'] = total_chunks
        selection['chunks_kept'] = len(all_chunks)
        selection['chunks_capped'] = len(all_chunks) < total_chunks
        selection['max_chunks'] = MAX_CHUNKS_PER_REPO
        selection['files_producing_chunks'] = files_producing_chunks
        # Round-robin represents every file only while there are at least as
        # many chunk slots as files. `MAX_CHUNKS_PER_REPO` is fixed at 1000 and
        # `--max-files` is operator-settable, so above that the guarantee stops
        # holding — and `truncated` is False in exactly that case, because the
        # file cap was never the binding constraint. Report the shortfall
        # directly: an operator who raises --max-files to 3000 and is told
        # nothing was truncated should not have to discover that two thirds of
        # their files carry no chunks at all.
        selection['files_with_chunks'] = len({c.file_path for c in all_chunks})
        if selection['chunks_capped']:
            logger.warning(
                f"Too many chunks ({total_chunks}), limiting to {MAX_CHUNKS_PER_REPO}"
            )

        self.chunks = all_chunks
        self.edges = all_edges

        # Generate embeddings
        self._embed_chunks(self.chunks)

        # Build FAISS index
        if FAISS_AVAILABLE:
            self._build_faiss_index()

        # Save to disk
        self._save()

        elapsed = time.time() - start_time
        logger.info(f"Built project index: {len(self.chunks)} chunks, {len(self.edges)} edges from {len(files_to_index)} files in {elapsed:.1f}s")

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
                logger.info("Refresh budget exceeded, stopping early")
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
        Extract semantic chunks from a file using tree-sitter.

        Chunks include:
        - Functions/methods
        - Classes/structs
        - Interfaces (for languages that support them)
        """
        if not self.parser:
            logger.error("Tree-sitter parser not available")
            return []

        try:
            # Check if parser supports this file extension
            if not self.parser.supports_extension(file_path.suffix):
                logger.debug(f"Unsupported file extension: {file_path.suffix}")
                return []

            # Read file content
            content = file_path.read_text(encoding='utf-8')

            # Parse using tree-sitter
            parser_chunks = self.parser.parse_file(file_path, content)

            # Convert ParserCodeChunk to ProjectIndex CodeChunk
            chunks = []
            for pc in parser_chunks:
                # Update file_path to be relative
                chunk = CodeChunk(
                    file_path=rel_path,
                    chunk_id=pc.chunk_id,
                    content=pc.content,
                    chunk_type=pc.chunk_type,
                    start_line=pc.start_line,
                    end_line=pc.end_line,
                    symbols=pc.symbols,
                    imports=pc.imports,
                    file_hash=pc.file_hash,
                    indexed_at=pc.indexed_at
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            logger.error(f"Failed to extract chunks from {file_path}: {e}")
            return []

    def _extract_edges_from_file(self, file_path: Path, rel_path: str) -> List[Dict[str, Any]]:
        """Extract relationship edges from a file using tree-sitter."""
        if not self.parser:
            return []

        try:
            if not self.parser.supports_extension(file_path.suffix):
                return []

            content = file_path.read_text(encoding='utf-8')
            parser_edges = self.parser.extract_edges(file_path, content)

            return [
                {
                    'source_file': rel_path,
                    'source_symbol': e.source_symbol,
                    'target_symbol': e.target_symbol,
                    'edge_type': e.edge_type,
                    'line_number': e.line_number,
                }
                for e in parser_edges
            ]
        except Exception as e:
            logger.error(f"Failed to extract edges from {file_path}: {e}")
            return []

    def query_edges(
        self,
        symbol: str,
        direction: str = "outgoing",
        edge_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query edges by symbol name.

        Args:
            symbol: Symbol name to search for
            direction: "outgoing" (symbol is source) or "incoming" (symbol is target)
            edge_type: Filter by edge type (imports/inherits/implements)

        Returns:
            List of matching edge dicts
        """
        results = []
        for edge in self.edges:
            if direction == "outgoing":
                match = edge['source_symbol'] == symbol
            else:
                match = edge['target_symbol'] == symbol

            if match and (edge_type is None or edge['edge_type'] == edge_type):
                results.append(edge)
        return results

    def _embed_chunks(self, chunks: List[CodeChunk]):
        """Generate embeddings for chunks.

        Embeds a structured representation of each chunk (symbols +
        imports + first ~600 chars of body) rather than the raw body,
        for two reasons:

        - Test files contain a query's prompt-keywords verbatim as
          assertion strings, so embedding the raw body causes tests to
          systematically outrank the source files they test. Symbols
          and imports are the durable, non-keyword-heavy summary of
          what the chunk is *about*.
        - The first ~600 chars capture the docstring and function
          signature, which carry semantic weight the body's
          implementation details dilute.
        """
        if not chunks:
            return

        # Initialize embedding model if needed
        if not self.embedding_model and FASTEMBED_AVAILABLE:
            from neo.memory.store import build_resilient_embedder
            self.embedding_model = build_resilient_embedder(log_prefix="ProjectIndex")

        if not self.embedding_model:
            logger.warning("No embedding model available, skipping embeddings")
            return

        # Build structured texts: symbols + imports + head-of-body.
        texts = [_chunk_embed_text(chunk) for chunk in chunks]

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
            from neo.memory.store import build_resilient_embedder
            self.embedding_model = build_resilient_embedder(log_prefix="ProjectIndex")

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

    def _patterns_from_languages(self, languages: List[str]) -> List[str]:
        """Convert language names to file patterns."""
        from neo.languages import EXTENSION_TO_LANGUAGE, normalize_language_name

        # Invert the canonical language map to get extensions per language
        lang_to_exts: dict[str, list[str]] = {}
        for ext, lang in EXTENSION_TO_LANGUAGE.items():
            lang_to_exts.setdefault(lang, []).append(ext)

        patterns = []
        for lang in languages:
            lang_normalized = normalize_language_name(lang)

            if lang_normalized in lang_to_exts:
                for ext in lang_to_exts[lang_normalized]:
                    patterns.append(f"**/*{ext}")
            else:
                logger.warning(f"Unknown language: {lang}")

        return patterns

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

    def status(self) -> Dict[str, Any]:
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
            'total_edges': len(self.edges),
            'embedding_model': self.snapshot.embedding_model,
            'embedding_dim': self.snapshot.embedding_dim
        }
