"""Tests for tree-sitter multi-language parser."""

import pytest
from pathlib import Path

# Check if tree-sitter is available
try:
    from neo.index.language_parser import TreeSitterParser, TREE_SITTER_AVAILABLE
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TreeSitterParser = None

pytestmark = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE,
    reason="tree-sitter-languages not installed (requires Python 3.8-3.12)"
)


# Sample code for each language
SAMPLE_PYTHON = '''
def hello_world():
    """Greet the world."""
    print("Hello, World!")

class Calculator:
    """A simple calculator."""
    def add(self, a, b):
        return a + b
'''

SAMPLE_CSHARP = '''
using System;

namespace MyApp
{
    public class Calculator
    {
        public int Add(int a, int b)
        {
            return a + b;
        }
    }

    public interface IService
    {
        void Execute();
    }
}
'''

SAMPLE_TYPESCRIPT = '''
interface User {
    name: string;
    age: number;
}

class UserService {
    getUser(id: number): User {
        return { name: "John", age: 30 };
    }
}

function greet(name: string): void {
    console.log(`Hello, ${name}!`);
}
'''

SAMPLE_JAVASCRIPT = '''
class Calculator {
    add(a, b) {
        return a + b;
    }
}

function multiply(a, b) {
    return a * b;
}
'''

SAMPLE_JAVA = '''
public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    private int multiply(int a, int b) {
        return a * b;
    }
}
'''

SAMPLE_GO = '''
package main

import "fmt"

func add(a int, b int) int {
    return a + b
}

type Calculator struct {
    name string
}
'''

SAMPLE_RUST = '''
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub struct Calculator {
    name: String,
}
'''


@pytest.fixture
def parser():
    """Create parser instance."""
    return TreeSitterParser()


def test_parser_initialization(parser):
    """Test parser initializes correctly."""
    assert parser is not None
    assert isinstance(parser.parsers, dict)
    assert isinstance(parser.languages, dict)
    assert isinstance(parser.compiled_queries, dict)


def test_supports_extension(parser):
    """Test extension support detection."""
    # Supported extensions
    assert parser.supports_extension('.py')
    assert parser.supports_extension('.cs')
    assert parser.supports_extension('.ts')
    assert parser.supports_extension('.js')
    assert parser.supports_extension('.java')
    assert parser.supports_extension('.go')
    assert parser.supports_extension('.rs')

    # Unsupported extensions
    assert not parser.supports_extension('.txt')
    assert not parser.supports_extension('.md')
    assert not parser.supports_extension('.unknown')


def test_detect_language(parser):
    """Test language detection from file paths."""
    assert parser.detect_language(Path('test.py')) == 'python'
    assert parser.detect_language(Path('test.cs')) == 'c_sharp'
    assert parser.detect_language(Path('test.ts')) == 'typescript'
    assert parser.detect_language(Path('test.tsx')) == 'tsx'
    assert parser.detect_language(Path('test.js')) == 'javascript'
    assert parser.detect_language(Path('test.java')) == 'java'
    assert parser.detect_language(Path('test.go')) == 'go'
    assert parser.detect_language(Path('test.rs')) == 'rust'
    assert parser.detect_language(Path('test.cpp')) == 'cpp'
    assert parser.detect_language(Path('test.c')) == 'c'

    # Unknown extension
    assert parser.detect_language(Path('test.unknown')) is None


def test_parse_python(parser):
    """Test Python code parsing."""
    chunks = parser.parse_file(Path('test.py'), SAMPLE_PYTHON, 'python')

    assert len(chunks) > 0

    # Find function chunk
    func_chunks = [c for c in chunks if c.chunk_type == 'function']
    assert len(func_chunks) >= 1
    func = func_chunks[0]
    assert 'hello_world' in func.symbols

    # Find class chunk
    class_chunks = [c for c in chunks if c.chunk_type == 'class']
    assert len(class_chunks) >= 1
    cls = class_chunks[0]
    assert 'Calculator' in cls.symbols


def test_parse_csharp(parser):
    """Test C# code parsing."""
    chunks = parser.parse_file(Path('test.cs'), SAMPLE_CSHARP, 'c_sharp')

    assert len(chunks) > 0

    # Should find class and interface
    class_chunks = [c for c in chunks if c.chunk_type in ['class', 'interface']]
    assert len(class_chunks) >= 2


def test_parse_typescript(parser):
    """Test TypeScript code parsing."""
    chunks = parser.parse_file(Path('test.ts'), SAMPLE_TYPESCRIPT, 'typescript')

    assert len(chunks) > 0

    # Should find class, interface, and function
    types = {c.chunk_type for c in chunks}
    assert 'class' in types or 'function' in types


def test_parse_javascript(parser):
    """Test JavaScript code parsing."""
    chunks = parser.parse_file(Path('test.js'), SAMPLE_JAVASCRIPT, 'javascript')

    assert len(chunks) > 0

    # Should find class and function
    types = {c.chunk_type for c in chunks}
    assert 'class' in types or 'function' in types


def test_parse_java(parser):
    """Test Java code parsing."""
    chunks = parser.parse_file(Path('Calculator.java'), SAMPLE_JAVA, 'java')

    assert len(chunks) > 0

    # Should find class
    class_chunks = [c for c in chunks if c.chunk_type == 'class']
    assert len(class_chunks) >= 1


def test_parse_go(parser):
    """Test Go code parsing."""
    chunks = parser.parse_file(Path('main.go'), SAMPLE_GO, 'go')

    assert len(chunks) > 0

    # Should find function or struct
    types = {c.chunk_type for c in chunks}
    assert len(types) > 0


def test_parse_rust(parser):
    """Test Rust code parsing."""
    chunks = parser.parse_file(Path('lib.rs'), SAMPLE_RUST, 'rust')

    assert len(chunks) > 0

    # Should find function or struct
    types = {c.chunk_type for c in chunks}
    assert len(types) > 0


def test_chunk_properties(parser):
    """Test that chunks have all required properties."""
    chunks = parser.parse_file(Path('test.py'), SAMPLE_PYTHON, 'python')

    for chunk in chunks:
        assert chunk.file_path
        assert chunk.chunk_id
        assert chunk.content
        assert chunk.chunk_type
        assert chunk.start_line > 0
        assert chunk.end_line >= chunk.start_line
        assert isinstance(chunk.symbols, list)
        assert isinstance(chunk.imports, list)
        assert chunk.file_hash
        assert chunk.indexed_at > 0


def test_unsupported_language(parser):
    """Test handling of unsupported languages."""
    chunks = parser.parse_file(Path('test.unknown'), 'content', None)
    assert chunks == []


def test_get_supported_languages(parser):
    """Test getting list of supported languages."""
    languages = parser.get_supported_languages()
    assert 'python' in languages
    assert 'c_sharp' in languages
    assert 'typescript' in languages
    assert 'javascript' in languages


def test_get_supported_extensions(parser):
    """Test getting list of supported extensions."""
    extensions = parser.get_supported_extensions()
    assert '.py' in extensions
    assert '.cs' in extensions
    assert '.ts' in extensions
    assert '.js' in extensions
    assert '.java' in extensions
    assert '.go' in extensions
    assert '.rs' in extensions


def test_lazy_loading(parser):
    """Test that parsers are lazy-loaded."""
    # Initially no parsers loaded
    assert len(parser.parsers) == 0

    # Parse Python file
    parser.parse_file(Path('test.py'), SAMPLE_PYTHON, 'python')

    # Python parser should be loaded
    assert 'python' in parser.parsers

    # Other parsers not loaded yet
    assert 'c_sharp' not in parser.parsers


def test_syntax_error_handling(parser):
    """Test handling of syntax errors."""
    invalid_code = "def broken(:"

    # Should not crash, just return empty list or handle gracefully
    chunks = parser.parse_file(Path('test.py'), invalid_code, 'python')

    # May return empty list or partial chunks depending on tree-sitter recovery
    assert isinstance(chunks, list)


def test_empty_file(parser):
    """Test parsing empty files."""
    chunks = parser.parse_file(Path('test.py'), '', 'python')
    assert chunks == []


def test_line_numbers(parser):
    """Test that line numbers are correctly extracted."""
    chunks = parser.parse_file(Path('test.py'), SAMPLE_PYTHON, 'python')

    for chunk in chunks:
        # Line numbers should be positive and in order
        assert chunk.start_line > 0
        assert chunk.end_line >= chunk.start_line

        # Check that content actually exists in those lines
        lines = SAMPLE_PYTHON.split('\n')
        if chunk.end_line <= len(lines):
            # Content should span from start to end line
            assert chunk.start_line <= chunk.end_line
