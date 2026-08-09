"""Tests for tree-sitter multi-language parser."""

import logging
from pathlib import Path

import pytest

from neo.index.language_parser import EDGE_QUERIES, QUERIES, TreeSitterParser


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


SAMPLE_TSX = '''
export interface ButtonProps {
    label: string;
}

export class Button {
    render(): string {
        return "<button/>";
    }
}
'''


@pytest.fixture
def parser():
    """Create parser instance."""
    return TreeSitterParser()


def _query_ids(group):
    """Flatten a QUERIES-shaped dict into (language, query_name) pairs."""
    return [(lang, name) for lang, queries in group.items() for name in queries]


@pytest.mark.parametrize("language,query_name", _query_ids(QUERIES))
def test_every_chunk_query_compiles(parser, language, query_name):
    """Every query in QUERIES must compile against the installed grammar.

    A query that fails to compile is indistinguishable from one that matches
    nothing: `_get_query` returns None and `parse_file` moves on. That is how
    `typescript:interfaces` sat broken from the day it shipped, dropping every
    interface in every TypeScript repo out of the index without an error.
    Grammars move; this is the assertion that notices when they do.
    """
    assert parser._get_query(language, query_name) is not None, (
        f"QUERIES[{language!r}][{query_name!r}] does not compile against the "
        f"installed grammar — every construct it matches is silently missing "
        f"from the index."
    )


@pytest.mark.parametrize("language,query_name", _query_ids(EDGE_QUERIES))
def test_every_edge_query_compiles(language, query_name):
    """Same guarantee for the edge queries, which fail even more quietly.

    `_extract_edges` logs compile failures at DEBUG, so a broken edge query
    produces no console output at all — `c_sharp:inheritance` dropped every
    inheritance edge in every C# repo with nothing but a debug line to show.

    Compiled against the grammar directly rather than through the parser's
    private cache key: keying on `f"{language}:edge:{name}"` here would keep
    passing if `_extract_edges` changed its convention, which is exactly the
    breakage this test exists to notice.
    """
    from tree_sitter import Query
    from tree_sitter_language_pack import get_language
    from neo.index.language_parser import _resolve_parser_name

    grammar = get_language(_resolve_parser_name(language))
    try:
        Query(grammar, EDGE_QUERIES[language][query_name])
    except Exception as exc:  # pragma: no cover - the assert carries the message
        pytest.fail(
            f"EDGE_QUERIES[{language!r}][{query_name!r}] does not compile "
            f"against the installed grammar ({exc}) — every edge it matches "
            f"is silently missing."
        )


def test_edge_queries_are_reached_through_the_parser(parser):
    """Guards the cache-key convention the compile test deliberately avoids.

    If `_extract_edges` and `_compile_cached` ever disagree on the key, the
    per-query compile test above still passes; this one does not.
    """
    edges = parser.extract_edges(
        Path("sample.py"), "import os\n\n\nclass A(B):\n    pass\n", "python"
    )

    kinds = {e.edge_type for e in edges}
    assert 'imports' in kinds and 'inherits' in kinds


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


def test_typescript_interface_is_extracted(parser):
    """A TypeScript interface must produce an `interface:` chunk.

    The pre-existing TypeScript tests only asserted on classes and functions,
    so a never-compiling `interfaces` query looked exactly like one that found
    nothing. This is the assertion that distinguishes them.
    """
    chunks = parser.parse_file(Path('test.ts'), SAMPLE_TYPESCRIPT, 'typescript')

    assert 'interface:User' in {c.chunk_id for c in chunks}


def test_tsx_interface_is_extracted(parser):
    """TSX aliases the TypeScript queries, so it needs its own assertion.

    `QUERIES['tsx'] = QUERIES['typescript']` binds the same dict object, which
    means one broken definition breaks two languages and one fix repairs both
    — but only a separate test proves the alias is still wired up.
    """
    chunks = parser.parse_file(Path('Button.tsx'), SAMPLE_TSX, 'tsx')

    chunk_ids = {c.chunk_id for c in chunks}
    assert 'interface:ButtonProps' in chunk_ids
    assert 'class:Button' in chunk_ids


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


def test_broken_chunk_query_warns_once_not_once_per_file(monkeypatch, caplog):
    """A query that cannot compile must warn once, not once per parsed file.

    Before failures were cached, `_get_query` recompiled on every call, so a
    single broken query emitted one warning per file: a run over 175
    TypeScript files produced 9,699 identical lines and buried the answer.
    """
    monkeypatch.setitem(
        QUERIES['python'], 'functions', '(this_node_type_does_not_exist) @x'
    )
    parser = TreeSitterParser()

    with caplog.at_level(logging.WARNING, logger='neo.index.language_parser'):
        for i in range(5):
            parser.parse_file(Path(f'f{i}.py'), SAMPLE_PYTHON, 'python')

    failures = [
        r for r in caplog.records
        if 'Failed to compile query python:functions' in r.getMessage()
    ]
    assert len(failures) == 1, (
        f"expected exactly 1 compile warning across 5 files, got {len(failures)}"
    )


def test_broken_edge_query_warns_once_not_once_per_file(monkeypatch, caplog):
    """The edge path recompiled per file with no cache of any kind."""
    monkeypatch.setitem(
        EDGE_QUERIES['python'], 'imports', '(this_node_type_does_not_exist) @x'
    )
    parser = TreeSitterParser()

    with caplog.at_level(logging.WARNING, logger='neo.index.language_parser'):
        for i in range(5):
            parser.extract_edges(Path(f'f{i}.py'), SAMPLE_PYTHON, 'python')

    failures = [
        r for r in caplog.records
        if 'Failed to compile query' in r.getMessage()
        and 'imports' in r.getMessage()
    ]
    assert len(failures) == 1


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
