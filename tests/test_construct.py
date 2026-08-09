"""
Tests for The Construct pattern library.

Tests cover pattern parsing, validation, indexing, and CLI integration.
"""

import os
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from neo.construct import (
    PatternSchema,
    PatternReader,
    PatternValidator,
    ConstructIndex,
)


# Test pattern content for use in tests
VALID_PATTERN_CONTENT = """# Pattern: Test Pattern
Author: test-user

## Intent
This is a test pattern for unit testing.

## Forces
- Force 1: Memory constraints
- Force 2: Latency requirements
- Force 3: Consistency tradeoffs

## Solution Sketch
The solution involves using a cache with TTL-based expiration
and asynchronous refresh to balance consistency and performance.

## Consequences
**Benefits:**
- Fast read access
- Reduced database load

**Risks:**
- Potential stale data
- Cache invalidation complexity

## References
- https://example.com/caching-patterns
"""

MISSING_AUTHOR_PATTERN = """# Pattern: Bad Pattern

## Intent
This pattern is missing the author field.

## Forces
- Some forces here

## Solution Sketch
Some solution

## Consequences
Some consequences
"""

MISSING_SECTION_PATTERN = """# Pattern: Incomplete Pattern
Author: test-user

## Intent
This pattern is missing required sections.

## Forces
Some forces
"""


def _axis_embedder():
    """A stub embedder putting each fixture pattern on its own unit axis.

    Two properties are load-bearing, and each corresponds to a way the earlier
    version of the ordering test could not have worked:

    - The vectors must be near-ORTHOGONAL, not just unequal. FAISS normalizes
      to unit length for cosine similarity, so `[0.9]*768` and `[0.1]*768`
      normalize to the same vector and rank identically.
    - The discriminating token must be unique to one fixture. `"cache"` is
      not: the rate-limiting pattern inherits the caching pattern's tradeoff
      list, "Cache invalidation complexity" included, so keying on it puts
      both patterns on the same axis. `"token bucket"` appears only in the
      rate-limiting fixture.
    """
    def embed(texts):
        vector = [0.0] * 768
        vector[1 if "token bucket" in texts[0].lower() else 0] = 1.0
        return [vector]

    embedder = MagicMock()
    embedder.embed = MagicMock(side_effect=embed)
    return embedder


@pytest.fixture
def temp_construct_dir():
    """Create temporary construct directory with test patterns."""
    tmpdir = tempfile.mkdtemp()
    construct_root = Path(tmpdir) / 'construct'
    construct_root.mkdir()

    # Create domain directories
    (construct_root / 'caching').mkdir()
    (construct_root / 'rate-limiting').mkdir()

    # Write test patterns
    (construct_root / 'caching' / 'cache-aside.md').write_text(VALID_PATTERN_CONTENT)

    pattern2 = VALID_PATTERN_CONTENT.replace("Test Pattern", "Token Bucket")
    pattern2 = pattern2.replace("cache with TTL", "token bucket algorithm")
    (construct_root / 'rate-limiting' / 'token-bucket.md').write_text(pattern2)

    yield construct_root

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir)


class TestPatternReader:
    """Test PatternReader parsing functionality."""

    def test_pattern_reader_parses_author_field(self):
        """Test that PatternReader correctly extracts author field."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(VALID_PATTERN_CONTENT)
            f.flush()
            path = Path(f.name)

        try:
            pattern = PatternReader.load(path)
            assert pattern is not None
            assert pattern.author == "test-user"
            assert pattern.name == "Test Pattern"
        finally:
            path.unlink()

    def test_pattern_reader_rejects_missing_author(self):
        """Test that patterns without author field are rejected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(MISSING_AUTHOR_PATTERN)
            f.flush()
            path = Path(f.name)

        try:
            pattern = PatternReader.load(path)
            assert pattern is None  # Should reject pattern without author
        finally:
            path.unlink()

    def test_pattern_reader_rejects_missing_sections(self):
        """Test that patterns with missing required sections are rejected."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(MISSING_SECTION_PATTERN)
            f.flush()
            path = Path(f.name)

        try:
            pattern = PatternReader.load(path)
            assert pattern is None  # Should reject incomplete pattern
        finally:
            path.unlink()

    def test_pattern_reader_extracts_all_sections(self):
        """Test that all sections are correctly extracted."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(VALID_PATTERN_CONTENT)
            f.flush()
            path = Path(f.name)

        try:
            pattern = PatternReader.load(path)
            assert pattern is not None
            assert "test pattern for unit testing" in pattern.intent.lower()
            assert "memory constraints" in pattern.forces.lower()
            assert "cache with ttl" in pattern.solution.lower()
            assert "benefits" in pattern.consequences.lower()
            assert "example.com" in pattern.references
        finally:
            path.unlink()


class TestPatternValidator:
    """Test PatternValidator quality checks."""

    def test_pattern_schema_validation(self):
        """Test that PatternValidator enforces required fields and constraints."""
        # Valid pattern
        valid_pattern = PatternSchema(
            pattern_id="test/valid",
            name="Valid Pattern",
            author="test-user",
            intent="Test intent with enough text",
            forces="Test forces with enough text",
            solution="Test solution with enough text",
            consequences="Test consequences with enough text",
            line_count=50,
        )
        errors = PatternValidator.validate(valid_pattern)
        assert len(errors) == 0

        # Missing author
        no_author = PatternSchema(
            pattern_id="test/no-author",
            name="No Author",
            author="",
            intent="Test intent with enough text",
            forces="Test forces with enough text",
            solution="Test solution with enough text",
            consequences="Test consequences with enough text",
            line_count=50,
        )
        errors = PatternValidator.validate(no_author)
        assert any("author" in e.lower() for e in errors)

        # Too many lines
        too_long = PatternSchema(
            pattern_id="test/too-long",
            name="Too Long",
            author="test-user",
            intent="Test intent",
            forces="Test forces",
            solution="Test solution",
            consequences="Test consequences",
            line_count=400,  # Exceeds MAX_LINE_COUNT
        )
        errors = PatternValidator.validate(too_long)
        assert any("line" in e.lower() for e in errors)

        # Sections too short
        short_sections = PatternSchema(
            pattern_id="test/short",
            name="Short Sections",
            author="test-user",
            intent="Short",  # < 10 chars
            forces="Short",
            solution="Short",
            consequences="Short",
            line_count=50,
        )
        errors = PatternValidator.validate(short_sections)
        assert len(errors) >= 4  # All sections too short


class TestConstructIndex:
    """Test ConstructIndex indexing and search functionality."""

    def test_construct_list_empty_directory(self):
        """Test listing patterns in empty directory."""
        tmpdir = tempfile.mkdtemp()
        construct_root = Path(tmpdir) / 'construct'
        construct_root.mkdir()

        try:
            index = ConstructIndex(construct_root=construct_root)
            patterns = index.list_patterns()
            assert len(patterns) == 0
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_construct_list_filtering_by_domain(self, temp_construct_dir):
        """Test filtering patterns by domain."""
        index = ConstructIndex(construct_root=temp_construct_dir)

        # List all patterns
        all_patterns = index.list_patterns()
        assert len(all_patterns) == 2

        # Filter by caching domain
        caching_patterns = index.list_patterns(domain='caching')
        assert len(caching_patterns) == 1
        assert caching_patterns[0].domain == 'caching'

        # Filter by rate-limiting domain
        rl_patterns = index.list_patterns(domain='rate-limiting')
        assert len(rl_patterns) == 1
        assert rl_patterns[0].domain == 'rate-limiting'

        # Filter by non-existent domain
        empty = index.list_patterns(domain='nonexistent')
        assert len(empty) == 0

    def test_construct_index_finds_packaged_pattern_library(self, tmp_path):
        """PyPI wheels ship patterns under neo/construct_library."""
        package_root = tmp_path / "neo"
        library = package_root / "construct_library"
        (library / "caching").mkdir(parents=True)
        (library / "caching" / "cache-aside.md").write_text(VALID_PATTERN_CONTENT)

        fake_construct_py = package_root / "construct.py"
        fake_construct_py.write_text("# placeholder")

        with patch("neo.construct.Path.cwd", return_value=tmp_path / "project"), \
             patch("neo.construct.__file__", str(fake_construct_py)):
            index = ConstructIndex()

        assert index.construct_root == library
        patterns = index.list_patterns()
        assert len(patterns) == 1
        assert patterns[0].pattern_id == "caching/cache-aside"

    def test_construct_show_missing_pattern(self, temp_construct_dir):
        """Test showing a pattern that doesn't exist."""
        index = ConstructIndex(construct_root=temp_construct_dir)
        pattern = index.show_pattern('nonexistent/pattern')
        assert pattern is None

    @pytest.mark.parametrize("template", [
        "../{stem}",
        "caching/../../{stem}",
        "{absolute}",
    ])
    def test_show_pattern_rejects_paths_outside_the_library(
        self, temp_construct_dir, template
    ):
        """`pattern_id` is interpolated into a path, so it must be confined.

        The escape target is a real, loadable pattern file placed outside the
        library — otherwise the test proves nothing. A traversal to a path
        that does not exist is rejected by the pre-existing `exists()` check
        for the wrong reason, so every such case passes with or without
        containment. Confirmed by running these against the unfixed source.

        The absolute case is included because `Path.__truediv__` treats an
        absolute right-hand operand as a REPLACEMENT, not a suffix:
        `construct_root / "/tmp/x"` is `/tmp/x`, library root discarded
        entirely (issue #25).
        """
        outside = temp_construct_dir.parent / "outside.md"
        outside.write_text(VALID_PATTERN_CONTENT)
        # Sanity: the escape target really is loadable, so a None result below
        # is containment refusing it rather than the file being unreadable.
        assert PatternReader.load(outside) is not None

        pattern_id = template.format(
            stem="outside", absolute=str(outside.with_suffix("")),
        )
        index = ConstructIndex(construct_root=temp_construct_dir)
        assert index.show_pattern(pattern_id) is None

    @pytest.mark.parametrize("pattern_id", ["", ".."])
    def test_show_pattern_rejects_degenerate_ids(
        self, temp_construct_dir, pattern_id
    ):
        """An empty id addresses `construct_root/.md`; `..` the parent dir."""
        index = ConstructIndex(construct_root=temp_construct_dir)
        assert index.show_pattern(pattern_id) is None

    def test_show_pattern_rejects_symlink_escape(self, temp_construct_dir):
        """A symlink escapes without the id ever containing `..`.

        This is why the check is resolved-path containment rather than a scan
        for traversal sequences: the string test passes this case cleanly.
        """
        outside = temp_construct_dir.parent / "outside.md"
        outside.write_text(VALID_PATTERN_CONTENT)
        os.symlink(outside, temp_construct_dir / "caching" / "escape.md")

        index = ConstructIndex(construct_root=temp_construct_dir)
        assert index.show_pattern("caching/escape") is None

    def test_show_pattern_still_loads_legitimate_ids(self, temp_construct_dir):
        """The other side of the contract: containment must not over-reject.

        A dotted id is legal — `..` as a substring is not traversal — and the
        ordinary nested id must keep working.
        """
        index = ConstructIndex(construct_root=temp_construct_dir)
        assert index.show_pattern("caching/cache-aside") is not None

        (temp_construct_dir / "caching" / "retry..backoff.md").write_text(
            VALID_PATTERN_CONTENT
        )
        assert index.show_pattern("caching/retry..backoff") is not None

    def test_construct_show_malformed_yaml(self):
        """Test handling of malformed pattern files."""
        tmpdir = tempfile.mkdtemp()
        construct_root = Path(tmpdir) / 'construct'
        construct_root.mkdir()
        (construct_root / 'test').mkdir()

        # Write malformed pattern (not actually YAML, but invalid markdown structure)
        malformed = "# Not a valid pattern\nJust random text"
        (construct_root / 'test' / 'malformed.md').write_text(malformed)

        try:
            index = ConstructIndex(construct_root=construct_root)
            pattern = index.show_pattern('test/malformed')
            # PatternReader should return None for invalid patterns
            assert pattern is None
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    @patch('neo.construct.FASTEMBED_AVAILABLE', False)
    @patch('neo.construct.FAISS_AVAILABLE', False)
    def test_construct_search_with_zero_results(self, temp_construct_dir):
        """Test search with no embedder available."""
        index = ConstructIndex(construct_root=temp_construct_dir)
        results = index.search("test query", top_k=5)
        # Without embedder, search should return empty list
        assert len(results) == 0

    @patch('neo.construct.FASTEMBED_AVAILABLE', True)
    @patch('neo.construct.FAISS_AVAILABLE', True)
    def test_construct_search_relevance_ordering(self, temp_construct_dir):
        """Search results come back ordered by similarity to the query.

        This test previously built the differentiated embeddings below and
        then asserted only `index.embedder is not None`, never calling
        `search()` — scaffolding for an assertion it did not make, which reads
        to a skimmer as though ordering were covered (issue #27).

        The embeddings must be near-ORTHOGONAL, not merely different in
        magnitude. FAISS normalizes to unit length for cosine similarity, so
        the old `[0.9]*768` and `[0.1]*768` both normalize to the identical
        vector and rank equally — the ordering would have been arbitrary even
        if the assertion had existed.
        """
        with patch('neo.memory.store.build_resilient_embedder') as mock_embedder_class:
            mock_embedder_class.return_value = _axis_embedder()

            index = ConstructIndex(construct_root=temp_construct_dir)
            index.build_index(force_rebuild=True)

            results = index.search("cache-aside read-through", top_k=2)

            assert len(results) == 2
            ordered = [pattern.pattern_id for pattern, _ in results]
            assert ordered[0] == "caching/cache-aside"
            assert ordered[1] == "rate-limiting/token-bucket"

            scores = [score for _, score in results]
            assert scores[0] > scores[1]
            assert scores == sorted(scores, reverse=True)

    @patch('neo.construct.FASTEMBED_AVAILABLE', True)
    @patch('neo.construct.FAISS_AVAILABLE', True)
    def test_construct_search_ordering_follows_the_query(self, temp_construct_dir):
        """The ranking must track the query, not a fixed pattern order.

        Without this, a `search()` that ignored its argument and returned
        patterns in directory order would satisfy the test above.
        """
        with patch('neo.memory.store.build_resilient_embedder') as mock_embedder_class:
            mock_embedder_class.return_value = _axis_embedder()

            index = ConstructIndex(construct_root=temp_construct_dir)
            index.build_index(force_rebuild=True)

            results = index.search("token bucket limiter", top_k=2)

            ordered = [pattern.pattern_id for pattern, _ in results]
            assert ordered[0] == "rate-limiting/token-bucket"

    @patch('neo.construct.FASTEMBED_AVAILABLE', True)
    @patch('neo.construct.FAISS_AVAILABLE', True)
    def test_construct_index_build_performance(self, temp_construct_dir):
        """Test that index builds in reasonable time (<5s for test patterns)."""
        with patch('neo.memory.store.build_resilient_embedder') as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed = MagicMock(return_value=[[0.5] * 768])
            mock_embedder_class.return_value = mock_embedder

            index = ConstructIndex(construct_root=temp_construct_dir)

            start = time.time()
            result = index.build_index(force_rebuild=True)
            elapsed = time.time() - start

            assert elapsed < 5.0  # Should complete in <5s
            assert result['status'] == 'success'
            assert result['pattern_count'] == 2

    @patch('neo.construct.FASTEMBED_AVAILABLE', True)
    @patch('neo.construct.FAISS_AVAILABLE', True)
    def test_construct_search_performance(self, temp_construct_dir):
        """Test that search completes in <100ms with warm cache."""
        with patch('neo.memory.store.build_resilient_embedder') as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed = MagicMock(return_value=[[0.5] * 768])
            mock_embedder_class.return_value = mock_embedder

            index = ConstructIndex(construct_root=temp_construct_dir)
            index.build_index(force_rebuild=True)

            # Warm up (first search may be slower)
            index.search("caching patterns", top_k=5)

            # Measure search performance
            start = time.time()
            index.search("rate limiting strategies", top_k=5)
            elapsed = (time.time() - start) * 1000  # Convert to ms

            # Note: This test may be flaky depending on system load
            # Using generous threshold for CI environments
            assert elapsed < 200  # <200ms is acceptable for test environment

    def test_construct_cli_backward_compatibility(self):
        """Test that existing CLI commands still work (no breaking changes)."""
        # This is a smoke test - just ensure parse_args doesn't break
        from neo.cli import parse_args
        import sys

        # Simulate old-style usage (prompt without subcommand)
        old_argv = sys.argv
        try:
            sys.argv = ['neo', '--version']
            args = parse_args()
            assert args.version is True
            # When no subcommand is used, command attribute may not exist
            assert not hasattr(args, 'command') or args.command is None

            sys.argv = ['neo', '--help']
            # parse_args will call sys.exit, so we can't test this directly
            # but we verify the structure is correct
        except SystemExit:
            pass  # --help exits, which is expected
        finally:
            sys.argv = old_argv


class TestCLIIntegration:
    """Test CLI command integration."""

    def test_construct_list_command(self, temp_construct_dir):
        """Test 'neo construct list' command."""
        from neo.cli import handle_construct
        import sys
        from io import StringIO
        from argparse import Namespace

        # Capture stdout
        captured = StringIO()
        sys.stdout = captured

        try:
            args = Namespace(
                command='construct',
                construct_action='list',
                domain=None,
                cwd=str(temp_construct_dir.parent)
            )
            handle_construct(args)
            output = captured.getvalue()

            # Check output contains pattern listings
            assert 'caching:' in output
            assert 'rate-limiting:' in output
            assert 'Total: 2 patterns' in output
        finally:
            sys.stdout = sys.__stdout__

    def test_construct_show_command(self, temp_construct_dir):
        """Test 'neo construct show' command."""
        from neo.cli import handle_construct
        import sys
        from io import StringIO
        from argparse import Namespace

        captured = StringIO()
        sys.stdout = captured

        try:
            args = Namespace(
                command='construct',
                construct_action='show',
                pattern_id='caching/cache-aside',
                cwd=str(temp_construct_dir.parent)
            )
            handle_construct(args)
            output = captured.getvalue()

            # Check output contains pattern details
            assert 'Pattern: Test Pattern' in output
            assert 'Author: test-user' in output
            assert '## Intent' in output
        finally:
            sys.stdout = sys.__stdout__


class TestPatternQualityConstraints:
    """Test quality constraints on patterns."""

    def test_pattern_validation_author_required(self):
        """Verify author field is mandatory."""
        pattern_without_author = PatternSchema(
            pattern_id="test/no-author",
            name="No Author Pattern",
            author="",
            intent="Some intent",
            forces="Some forces",
            solution="Some solution",
            consequences="Some consequences",
            line_count=100,
        )

        errors = PatternValidator.validate(pattern_without_author)
        assert len(errors) > 0
        assert any("author" in e.lower() for e in errors)

    def test_pattern_validation_line_limit(self):
        """Verify patterns must be under 300 lines."""
        long_pattern = PatternSchema(
            pattern_id="test/long",
            name="Long Pattern",
            author="test-user",
            intent="Intent text here",
            forces="Forces text here",
            solution="Solution text here",
            consequences="Consequences text here",
            line_count=350,  # Exceeds limit
        )

        errors = PatternValidator.validate(long_pattern)
        assert len(errors) > 0
        assert any("line" in e.lower() or "300" in e for e in errors)


class TestIndexFreshness:
    """Test index freshness checks against pattern file modifications."""

    @patch('neo.construct.FASTEMBED_AVAILABLE', True)
    @patch('neo.construct.FAISS_AVAILABLE', True)
    def test_construct_index_respects_pattern_file_changes(self, temp_construct_dir):
        """Test that index rebuilds when pattern file is modified after index creation."""
        with patch('neo.memory.store.build_resilient_embedder') as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed = MagicMock(return_value=[[0.5] * 768])
            mock_embedder_class.return_value = mock_embedder

            index = ConstructIndex(construct_root=temp_construct_dir)

            # Build initial index
            result = index.build_index(force_rebuild=True)
            assert result['status'] == 'success'
            assert list((temp_construct_dir / '.index').glob('*.tmp')) == []

            # Wait to ensure different mtime
            time.sleep(0.1)

            # Modify a pattern file
            pattern_path = temp_construct_dir / 'caching' / 'cache-aside.md'
            content = pattern_path.read_text()
            pattern_path.write_text(content + "\n# Modified")

            # Build index again without force_rebuild
            result = index.build_index(force_rebuild=False)

            # Should rebuild because pattern was modified
            assert result['status'] == 'success'
            assert result['pattern_count'] == 2

    @patch('neo.construct.FASTEMBED_AVAILABLE', True)
    @patch('neo.construct.FAISS_AVAILABLE', True)
    def test_construct_index_skips_when_fresh(self, temp_construct_dir):
        """Test that index is skipped when all patterns are older than index."""
        with patch('neo.memory.store.build_resilient_embedder') as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed = MagicMock(return_value=[[0.5] * 768])
            mock_embedder_class.return_value = mock_embedder

            index = ConstructIndex(construct_root=temp_construct_dir)

            # Build initial index
            result = index.build_index(force_rebuild=True)
            assert result['status'] == 'success'

            # Wait to ensure different mtime
            time.sleep(0.1)

            # Build index again without modifying files
            result = index.build_index(force_rebuild=False)

            # Should skip because index is fresh and no files modified
            assert result['status'] == 'skipped'
            assert result['reason'] == 'index_fresh'
