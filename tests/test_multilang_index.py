"""Integration tests for multi-language project indexing."""

import pytest
from pathlib import Path
import tempfile
import shutil

from neo.index.project_index import ProjectIndex

# Check if tree-sitter is available
try:
    from neo.index.language_parser import TREE_SITTER_AVAILABLE
except ImportError:
    TREE_SITTER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE,
    reason="tree-sitter-languages not installed (requires Python 3.8-3.12)"
)


# Sample files for different languages
SAMPLE_FILES = {
    'main.py': '''
def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    return a + b

class DataProcessor:
    """Process data."""
    def process(self, data):
        return data.upper()
''',
    'Calculator.cs': '''
using System;

namespace MathLib
{
    public class Calculator
    {
        public int Add(int a, int b)
        {
            return a + b;
        }

        public int Subtract(int a, int b)
        {
            return a - b;
        }
    }

    public interface ICalculator
    {
        int Add(int a, int b);
    }
}
''',
    'utils.ts': '''
interface Config {
    apiUrl: string;
    timeout: number;
}

class ApiClient {
    constructor(private config: Config) {}

    async fetchData(endpoint: string): Promise<any> {
        const url = `${this.config.apiUrl}/${endpoint}`;
        return fetch(url);
    }
}

export function formatDate(date: Date): string {
    return date.toISOString();
}
''',
    'helper.js': '''
class StringHelper {
    capitalize(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

function reverseString(str) {
    return str.split('').reverse().join('');
}

module.exports = { StringHelper, reverseString };
''',
    'Main.java': '''
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }

    private static int multiply(int a, int b) {
        return a * b;
    }
}
''',
    'lib.go': '''
package mathlib

import "fmt"

func Add(a int, b int) int {
    return a + b
}

type Calculator struct {
    name string
}

func (c *Calculator) Multiply(a int, b int) int {
    return a * b
}
''',
}


@pytest.fixture
def temp_repo():
    """Create temporary repository with multi-language files."""
    tmpdir = tempfile.mkdtemp()
    repo_path = Path(tmpdir)

    # Create sample files
    for filename, content in SAMPLE_FILES.items():
        file_path = repo_path / filename
        file_path.write_text(content)

    yield repo_path

    # Cleanup
    shutil.rmtree(tmpdir)


def test_build_multilang_index(temp_repo):
    """Test building index for multi-language repository."""
    index = ProjectIndex(str(temp_repo))

    # Build index without specifying languages (should auto-detect)
    index.build_index(max_files=100)

    # Check that chunks were created
    assert len(index.chunks) > 0

    # Check that we got chunks from different file types
    file_paths = {chunk.file_path for chunk in index.chunks}
    assert len(file_paths) > 1  # Should have chunks from multiple files


def test_build_index_specific_languages(temp_repo):
    """Test building index for specific languages only."""
    index = ProjectIndex(str(temp_repo))

    # Index only Python and C#
    index.build_index(languages=['python', 'c_sharp'], max_files=100)

    # Check chunks
    assert len(index.chunks) > 0

    # All chunks should be from Python or C# files
    for chunk in index.chunks:
        assert chunk.file_path.endswith('.py') or chunk.file_path.endswith('.cs')


def test_index_persistence(temp_repo):
    """Test that index can be saved and loaded."""
    # Build index
    index1 = ProjectIndex(str(temp_repo))
    index1.build_index(max_files=100)

    initial_chunk_count = len(index1.chunks)
    assert initial_chunk_count > 0

    # Create new index instance (should load from disk)
    index2 = ProjectIndex(str(temp_repo))

    # Should have loaded the same chunks
    assert len(index2.chunks) == initial_chunk_count

    # Check .neo directory exists
    neo_dir = temp_repo / '.neo'
    assert neo_dir.exists()
    assert (neo_dir / 'index.json').exists()
    assert (neo_dir / 'chunks.json').exists()


def test_semantic_retrieval(temp_repo):
    """Test semantic code retrieval."""
    index = ProjectIndex(str(temp_repo))
    index.build_index(max_files=100)

    # Search for calculator-related code
    results = index.retrieve("calculator add function", k=5)

    # Should find relevant chunks
    assert len(results) > 0

    # Results should have similarity scores
    for result in results:
        assert result.similarity is not None


def test_chunk_content_quality(temp_repo):
    """Test that extracted chunks contain meaningful code."""
    index = ProjectIndex(str(temp_repo))
    index.build_index(max_files=100)

    # Check Python chunks
    python_chunks = [c for c in index.chunks if c.file_path.endswith('.py')]
    assert len(python_chunks) > 0

    for chunk in python_chunks:
        # Should contain actual code
        assert len(chunk.content) > 0

        # Should have proper metadata
        assert chunk.chunk_type in ['function', 'class', 'module_doc']
        assert chunk.start_line > 0
        assert chunk.end_line >= chunk.start_line

        # Should have symbols
        if chunk.chunk_type in ['function', 'class']:
            assert len(chunk.symbols) > 0


def test_csharp_chunks(temp_repo):
    """Test C# specific chunk extraction."""
    index = ProjectIndex(str(temp_repo))
    index.build_index(languages=['c_sharp'], max_files=100)

    cs_chunks = [c for c in index.chunks if c.file_path.endswith('.cs')]
    assert len(cs_chunks) > 0

    # Should find class and/or interface
    types = {c.chunk_type for c in cs_chunks}
    assert len(types) > 0


def test_typescript_chunks(temp_repo):
    """Test TypeScript specific chunk extraction."""
    index = ProjectIndex(str(temp_repo))
    index.build_index(languages=['typescript'], max_files=100)

    ts_chunks = [c for c in index.chunks if c.file_path.endswith('.ts')]
    assert len(ts_chunks) > 0

    # Should find class, interface, and/or function
    types = {c.chunk_type for c in ts_chunks}
    assert len(types) > 0


def test_index_status(temp_repo):
    """Test index status reporting."""
    index = ProjectIndex(str(temp_repo))
    index.build_index(max_files=100)

    status = index.status()

    assert status['exists'] is True
    assert status['total_chunks'] > 0
    assert status['total_files'] > 0
    assert 'is_stale' in status
    assert 'embedding_model' in status


def test_staleness_detection(temp_repo):
    """Test file change detection."""
    # Build initial index
    index = ProjectIndex(str(temp_repo))
    index.build_index(max_files=100)

    # Index should be fresh
    is_stale, ratio, changed = index.check_staleness()
    assert not is_stale
    assert ratio == 0.0
    assert len(changed) == 0

    # Modify a file
    py_file = temp_repo / 'main.py'
    py_file.write_text('# Modified\ndef new_func():\n    pass\n')

    # Now should be stale
    is_stale, ratio, changed = index.check_staleness()
    assert len(changed) > 0
    assert 'main.py' in changed


def test_file_patterns_from_languages(temp_repo):
    """Test language to file pattern conversion."""
    index = ProjectIndex(str(temp_repo))

    patterns = index._patterns_from_languages(['python', 'c_sharp'])

    assert '**/*.py' in patterns
    assert '**/*.cs' in patterns


def test_empty_repository():
    """Test indexing empty repository."""
    tmpdir = tempfile.mkdtemp()
    try:
        index = ProjectIndex(tmpdir)
        index.build_index(max_files=100)

        # Should handle gracefully
        assert len(index.chunks) == 0
        status = index.status()
        assert status['total_chunks'] == 0

    finally:
        shutil.rmtree(tmpdir)


def test_mixed_valid_invalid_files(temp_repo):
    """Test handling mix of valid and invalid files."""
    # Add an invalid Python file
    invalid_file = temp_repo / 'broken.py'
    invalid_file.write_text('def broken(:\n')

    # Add a non-code file that shouldn't be indexed
    readme = temp_repo / 'README.md'
    readme.write_text('# README\n')

    index = ProjectIndex(str(temp_repo))
    index.build_index(max_files=100)

    # Should still have chunks from valid files
    assert len(index.chunks) > 0

    # README should not be indexed
    md_chunks = [c for c in index.chunks if c.file_path.endswith('.md')]
    assert len(md_chunks) == 0


def test_max_files_limit(temp_repo):
    """Test max_files parameter limits indexing."""
    index = ProjectIndex(str(temp_repo))

    # Set very low limit
    index.build_index(max_files=2)

    # Should respect the limit
    file_paths = {chunk.file_path for chunk in index.chunks}
    assert len(file_paths) <= 2


def test_chunk_embeddings(temp_repo):
    """Test that embeddings are generated for chunks."""
    index = ProjectIndex(str(temp_repo))
    index.build_index(max_files=100)

    # Check that chunks have embeddings
    embedded_chunks = [c for c in index.chunks if c.embedding is not None]

    # Should have at least some embeddings (depends on fastembed availability)
    # This might be 0 if fastembed is not installed, which is OK
    if embedded_chunks:
        import numpy as np
        for chunk in embedded_chunks[:5]:  # Check first 5
            assert isinstance(chunk.embedding, np.ndarray)
            assert len(chunk.embedding) > 0


def test_faiss_index_creation(temp_repo):
    """Test FAISS index creation."""
    index = ProjectIndex(str(temp_repo))
    index.build_index(max_files=100)

    # FAISS index may or may not be available
    if index.faiss_index is not None:
        # Should have indexed all chunks with embeddings
        embedded_count = len([c for c in index.chunks if c.embedding is not None])
        if embedded_count > 0:
            assert index.faiss_index.ntotal == embedded_count
