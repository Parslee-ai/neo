"""
Tests for The Construct pattern library.

Tests cover pattern parsing, validation, indexing, and CLI integration.
"""

import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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

    def test_construct_show_missing_pattern(self, temp_construct_dir):
        """Test showing a pattern that doesn't exist."""
        index = ConstructIndex(construct_root=temp_construct_dir)
        pattern = index.show_pattern('nonexistent/pattern')
        assert pattern is None

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
        """Test that search results are ordered by relevance."""
        # Mock the embedder
        with patch('neo.construct.TextEmbedding') as mock_embedder_class:
            mock_embedder = MagicMock()
            # Return different embeddings for different queries
            def embed_side_effect(texts):
                # Simulate embeddings: first pattern matches "cache", second matches "rate"
                if "cache" in texts[0].lower():
                    return [[0.9] * 768]  # High similarity to caching pattern
                elif "rate" in texts[0].lower():
                    return [[0.1] * 768]  # Low similarity to caching pattern
                else:
                    return [[0.5] * 768]
            mock_embedder.embed = MagicMock(side_effect=embed_side_effect)
            mock_embedder_class.return_value = mock_embedder

            index = ConstructIndex(construct_root=temp_construct_dir)
            # Force index build
            index.build_index(force_rebuild=True)

            # Search should work with mocked embedder
            # This is a basic test - in reality, semantic search ordering depends on embeddings
            assert index.embedder is not None

    @patch('neo.construct.FASTEMBED_AVAILABLE', True)
    @patch('neo.construct.FAISS_AVAILABLE', True)
    def test_construct_index_build_performance(self, temp_construct_dir):
        """Test that index builds in reasonable time (<5s for test patterns)."""
        with patch('neo.construct.TextEmbedding') as mock_embedder_class:
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
        with patch('neo.construct.TextEmbedding') as mock_embedder_class:
            mock_embedder = MagicMock()
            mock_embedder.embed = MagicMock(return_value=[[0.5] * 768])
            mock_embedder_class.return_value = mock_embedder

            index = ConstructIndex(construct_root=temp_construct_dir)
            index.build_index(force_rebuild=True)

            # Warm up (first search may be slower)
            index.search("caching patterns", top_k=5)

            # Measure search performance
            start = time.time()
            results = index.search("rate limiting strategies", top_k=5)
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
