"""Every declared language query must match something on real code.

`test_every_chunk_query_compiles` and `test_every_edge_query_compiles` prove
compilation ONLY. That is the exact gap #158 lived in: a tree-sitter query
that fails to compile is indistinguishable from one that matches nothing --
`_get_query` returns None, `parse_file` moves on, and the index is silently
missing a whole language's worth of structure. Four queries were broken at
once for months, so TypeScript interfaces and ALL C# inheritance edges were
absent from every index since they shipped.

Compilation tests would not have caught any of them. These would.

Five languages -- java, go, c, ruby, php -- had no corpus anywhere in the
local checkout, so until this file they had never been run against real
source in any form, only compiled. The fixtures are deliberately small and
hand-written rather than sampled, so each one exercises the specific node
types its query names.

Two coverage gaps are pinned rather than fixed, because closing them is a
feature decision and a silent gap is worse than a documented one:

- **ruby has no edge queries at all.** It is the only language in `QUERIES`
  absent from `EDGE_QUERIES`, so ruby files contribute nodes and no edges.
- **php has an imports query but no inheritance query**, so `class X
  implements Y` yields nothing where the java/c_sharp/typescript equivalents
  yield an edge.
"""

import pathlib
import tempfile

import pytest

from neo.index.language_parser import EDGE_QUERIES, QUERIES, TreeSitterParser

# Each fixture exercises the node types its language's queries actually name.
# `php` uses `use App\Contracts\Greeter;` rather than a bare `namespace`,
# because the php imports query matches `namespace_use_declaration` -- a
# fixture that only declares a namespace yields zero and looks like a defect.
FIXTURES = {
    "java": ("A.java", """package x;
import java.util.List;
public interface Greeter { String greet(String n); }
public class Impl implements Greeter {
    public String greet(String n) { return "hi " + n; }
    private int count(List<String> xs) { return xs.size(); }
}
"""),
    "go": ("a.go", """package main

import "fmt"

type Greeter interface { Greet(n string) string }

type Impl struct{ N int }

func (i Impl) Greet(n string) string { return fmt.Sprintf("hi %s", n) }

func main() { fmt.Println(Impl{}.Greet("x")) }
"""),
    "c": ("a.c", """#include <stdio.h>

struct Point { int x; int y; };

int add(int a, int b) { return a + b; }

int main(void) { printf("%d", add(1, 2)); return 0; }
"""),
    "ruby": ("a.rb", """require 'set'

module Greetable
  def greet(n)
    "hi #{n}"
  end
end

class Impl
  include Greetable

  def initialize(n)
    @n = n
  end

  def count
    @n.size
  end
end
"""),
    "php": ("a.php", """<?php
namespace App;

use App\\Contracts\\Greeter;

class Impl implements Greeter {
    public function greet(string $n): string { return "hi $n"; }
    private function total(array $xs): int { return count($xs); }
}
"""),
}


@pytest.fixture(scope="module")
def parser():
    return TreeSitterParser()


def _parse(parser, lang):
    name, source = FIXTURES[lang]
    path = pathlib.Path(tempfile.mkdtemp()) / name
    path.write_text(source)
    return (parser.parse_file(path, source) or [],
            parser.extract_edges(path, source) or [])


class TestChunkQueriesMatchRealCode:
    @pytest.mark.parametrize("lang", sorted(FIXTURES))
    def test_language_produces_chunks(self, parser, lang):
        chunks, _ = _parse(parser, lang)
        assert chunks, (
            f"{lang} chunk queries compile but matched nothing -- the #158 "
            f"failure mode, which is invisible from the outside"
        )


class TestEdgeQueriesMatchRealCode:
    @pytest.mark.parametrize("lang", sorted(set(FIXTURES) & set(EDGE_QUERIES) - {"php"}))
    def test_language_produces_edges(self, parser, lang):
        _, edges = _parse(parser, lang)
        assert edges, f"{lang} edge queries compile but matched nothing"

    def test_php_extracts_imports(self, parser):
        """php declares only an imports query, so that is all it can yield."""
        _, edges = _parse(parser, "php")
        assert edges, "php imports query matched nothing on a `use` statement"


class TestKnownCoverageGaps:
    """Pinned so the gaps are visible, and so closing one updates a test
    rather than silently changing behaviour."""

    def test_ruby_declares_no_edge_queries(self):
        """Ruby is the only language with chunk queries and no edge queries,
        so ruby files contribute nodes to the graph and never edges. Delete
        this test when ruby edge queries are added -- its failure is the
        signal that the gap closed.
        """
        assert set(QUERIES) - set(EDGE_QUERIES) == {"ruby"}

    def test_php_has_imports_but_no_inheritance(self, parser):
        """`class Impl implements Greeter` yields an edge in java, c_sharp
        and typescript, and nothing in php."""
        assert set(EDGE_QUERIES["php"]) == {"imports"}

        _, edges = _parse(parser, "php")
        kinds = {getattr(e, "kind", getattr(e, "type", "")) for e in edges}
        assert not any("inherit" in str(k).lower() for k in kinds)

    def test_javascript_imports_are_esm_only(self, parser):
        """The js imports query matches `import_statement`. CommonJS
        `require()` is a call expression and yields nothing, so a codebase on
        `require` contributes no import edges at all.
        """
        esm = "import fs from 'fs';\nexport class A { m() { return 1; } }\n"
        cjs = "const fs = require('fs');\nclass A { m() { return 1; } }\n"
        directory = pathlib.Path(tempfile.mkdtemp())

        (directory / "esm.js").write_text(esm)
        (directory / "cjs.js").write_text(cjs)

        esm_edges = parser.extract_edges(directory / "esm.js", esm) or []
        cjs_edges = parser.extract_edges(directory / "cjs.js", cjs) or []

        assert esm_edges, "ESM import should yield an edge"
        assert not cjs_edges, (
            "CommonJS require yielding edges means the query widened -- "
            "good news, but update this test and the note in "
            "docs/tree-sitter-setup.md"
        )
