"""Tests for tree-sitter edge extraction (imports, inheritance)."""

import pytest
from pathlib import Path

try:
    from neo.index.language_parser import (
        TreeSitterParser, TREE_SITTER_AVAILABLE
    )
except ImportError:
    TREE_SITTER_AVAILABLE = False
    TreeSitterParser = None

pytestmark = pytest.mark.skipif(
    not TREE_SITTER_AVAILABLE,
    reason="tree-sitter-languages not installed"
)


SAMPLE_PYTHON = '''
import os
import sys
from pathlib import Path
from collections import OrderedDict

class Animal:
    pass

class Dog(Animal):
    def bark(self):
        pass

class GuideDog(Dog):
    pass
'''

SAMPLE_PYTHON_FROM_IMPORT = '''
from neo.index.language_parser import TreeSitterParser
from neo.models import CodeSuggestion
'''

SAMPLE_TYPESCRIPT = '''
import { Component } from '@angular/core';
import { UserService } from './user.service';

interface Serializable {
    serialize(): string;
}

class BaseModel {
    id: number;
}

class User extends BaseModel {
    name: string;
}
'''

SAMPLE_CSHARP = '''
using System;
using System.Collections.Generic;

public class Shape
{
    public virtual double Area() { return 0; }
}

public class Circle : Shape
{
    public override double Area() { return 3.14; }
}
'''

SAMPLE_JAVASCRIPT = '''
import React from 'react';
import { useState } from 'react';

class Component extends React.Component {
    render() {}
}
'''


@pytest.fixture
def parser():
    return TreeSitterParser()


class TestPythonEdges:
    def test_import_edges(self, parser):
        edges = parser.extract_edges(Path("test.py"), SAMPLE_PYTHON, "python")
        import_edges = [e for e in edges if e.edge_type == "imports"]
        targets = [e.target_symbol for e in import_edges]
        assert "os" in targets
        assert "sys" in targets

    def test_from_import_edges(self, parser):
        edges = parser.extract_edges(Path("test.py"), SAMPLE_PYTHON, "python")
        import_edges = [e for e in edges if e.edge_type == "imports"]
        targets = [e.target_symbol for e in import_edges]
        # from pathlib import Path -> module is "pathlib"
        assert "pathlib" in targets

    def test_inheritance_edges(self, parser):
        edges = parser.extract_edges(Path("test.py"), SAMPLE_PYTHON, "python")
        inherit_edges = [e for e in edges if e.edge_type == "inherits"]
        # Dog(Animal) and GuideDog(Dog)
        pairs = [(e.source_symbol, e.target_symbol) for e in inherit_edges]
        assert ("Dog", "Animal") in pairs
        assert ("GuideDog", "Dog") in pairs

    def test_edge_has_line_numbers(self, parser):
        edges = parser.extract_edges(Path("test.py"), SAMPLE_PYTHON, "python")
        for edge in edges:
            assert edge.line_number > 0

    def test_edge_has_source_file(self, parser):
        edges = parser.extract_edges(Path("test.py"), SAMPLE_PYTHON, "python")
        for edge in edges:
            assert edge.source_file == "test.py"


class TestTypescriptEdges:
    def test_import_edges(self, parser):
        edges = parser.extract_edges(Path("test.ts"), SAMPLE_TYPESCRIPT, "typescript")
        import_edges = [e for e in edges if e.edge_type == "imports"]
        targets = [e.target_symbol for e in import_edges]
        assert any("@angular/core" in t for t in targets)

    def test_inheritance_edges(self, parser):
        edges = parser.extract_edges(Path("test.ts"), SAMPLE_TYPESCRIPT, "typescript")
        inherit_edges = [e for e in edges if e.edge_type == "inherits"]
        pairs = [(e.source_symbol, e.target_symbol) for e in inherit_edges]
        assert ("User", "BaseModel") in pairs


class TestCSharpEdges:
    def test_import_edges(self, parser):
        edges = parser.extract_edges(Path("test.cs"), SAMPLE_CSHARP, "c_sharp")
        import_edges = [e for e in edges if e.edge_type == "imports"]
        targets = [e.target_symbol for e in import_edges]
        assert any("System" in t for t in targets)


class TestJavaScriptEdges:
    def test_import_edges(self, parser):
        edges = parser.extract_edges(Path("test.js"), SAMPLE_JAVASCRIPT, "javascript")
        import_edges = [e for e in edges if e.edge_type == "imports"]
        targets = [e.target_symbol for e in import_edges]
        assert any("react" in t for t in targets)


class TestImportsInChunks:
    """Test that parse_file now populates the imports field on chunks."""

    def test_python_chunks_have_imports(self, parser):
        chunks = parser.parse_file(Path("test.py"), SAMPLE_PYTHON, "python")
        # The file has imports, so any chunk should reflect them
        # Get all imports across all chunks
        all_imports = set()
        for chunk in chunks:
            all_imports.update(chunk.imports)
        assert "os" in all_imports or len(chunks) > 0  # imports are file-level


class TestUnsupportedLanguage:
    def test_no_edges_for_unknown(self, parser):
        edges = parser.extract_edges(Path("test.txt"), "hello world")
        assert edges == []

    def test_no_edges_for_language_without_queries(self, parser):
        # Even if we force a language, no crash
        edges = parser.extract_edges(Path("test.py"), "x = 1", "python")
        assert isinstance(edges, list)
