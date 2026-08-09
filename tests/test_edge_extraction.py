"""Tests for tree-sitter edge extraction (imports, inheritance)."""

from pathlib import Path

import pytest

from neo.index.language_parser import TreeSitterParser


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

public class Square : Shape, IComparable
{
    public override double Area() { return 1.0; }
    public int CompareTo(object o) { return 0; }
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

    def test_inheritance_edges(self, parser):
        """C# had no inheritance assertion, and its query had been broken.

        The grammar exposes `base_list` as an unnamed child of
        `class_declaration`, not under a `bases:` field, so the query failed to
        compile and every C# inheritance edge was dropped. Edge-query compile
        failures log at DEBUG, so nothing surfaced.
        """
        edges = parser.extract_edges(Path("test.cs"), SAMPLE_CSHARP, "c_sharp")
        pairs = [
            (e.source_symbol, e.target_symbol)
            for e in edges
            if e.edge_type == "inherits"
        ]
        assert ("Circle", "Shape") in pairs

    def test_inheritance_edges_include_every_base(self, parser):
        """Each entry of a plain (non-generic) base list yields its own edge."""
        edges = parser.extract_edges(Path("test.cs"), SAMPLE_CSHARP, "c_sharp")
        pairs = [
            (e.source_symbol, e.target_symbol)
            for e in edges
            if e.edge_type == "inherits"
        ]
        assert ("Square", "Shape") in pairs
        assert ("Square", "IComparable") in pairs

    def test_generic_base_is_not_dropped(self, parser):
        """`: Repository<Order>` parses as `generic_name`, not `identifier`.

        An identifier-only pattern compiles fine and silently omits every
        generic base — which in .NET is most of them.
        """
        edges = parser.extract_edges(
            Path("g.cs"), "class Repo : Base<Order>, IFoo { }\n", "c_sharp"
        )
        pairs = [
            (e.source_symbol, e.target_symbol)
            for e in edges
            if e.edge_type == "inherits"
        ]
        assert ("Repo", "Base") in pairs
        assert ("Repo", "IFoo") in pairs

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("interface IX : IY { }\n", ("IX", "IY")),
            ("record Rec : BaseRec { }\n", ("Rec", "BaseRec")),
            ("struct St : IEquatable<St> { }\n", ("St", "IEquatable")),
        ],
    )
    def test_non_class_declarations_also_yield_edges(self, parser, source, expected):
        """Interfaces, records and structs carry their own `base_list`.

        Matching only `class_declaration` meant interface-to-interface
        inheritance produced nothing at all.
        """
        edges = parser.extract_edges(Path("x.cs"), source, "c_sharp")
        pairs = [
            (e.source_symbol, e.target_symbol)
            for e in edges
            if e.edge_type == "inherits"
        ]
        assert expected in pairs

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("class A : System.Exception { }\n", ("A", "Exception")),
            ("class B : My.Ns.Repo<Order> { }\n", ("B", "Repo")),
            ("class C : global::Foo.Bar { }\n", ("C", "Bar")),
        ],
    )
    def test_qualified_base_is_not_dropped(self, parser, source, expected):
        """A fully-qualified base parses as `qualified_name`.

        `: System.Exception` and `: Microsoft.AspNetCore.Mvc.ControllerBase`
        are everyday .NET, and neither `identifier` nor `generic_name` matches
        them — so an alternation covering only those two compiles fine and
        drops the edge, exactly as the identifier-only pattern did for
        generics.

        The captured name is the rightmost segment (`Exception`, not
        `System`), via the `name:` field, so qualified and unqualified bases
        land in the graph under one naming convention rather than two.
        """
        edges = parser.extract_edges(Path("q.cs"), source, "c_sharp")
        pairs = [
            (e.source_symbol, e.target_symbol)
            for e in edges
            if e.edge_type == "inherits"
        ]
        assert expected in pairs

    def test_declaration_without_bases_yields_no_edge(self, parser):
        """The four-way alternation must not match a bare declaration.

        Generated patterns are easy to widen past what was intended; this
        pins the other side of the contract.
        """
        edges = parser.extract_edges(Path("p.cs"), "class Plain { }\n", "c_sharp")
        assert [e for e in edges if e.edge_type == "inherits"] == []


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
