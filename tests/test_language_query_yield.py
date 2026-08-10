"""Every declared query must match something on real code -- PER QUERY.

`test_every_chunk_query_compiles` and `test_every_edge_query_compiles` prove
compilation ONLY. That is the exact gap #158 lived in: a tree-sitter query
that fails to compile is indistinguishable from one that matches nothing --
`_get_query` returns None, `parse_file` moves on, and a whole class of
structure is missing from every index built since. Four queries were broken at
once, so TypeScript interfaces and ALL C# inheritance edges were absent for
months.

**Per query, not per language.** The first version of this file asserted
`parse_file(...)` and `extract_edges(...)` were non-empty, which passes as
long as ANY sibling query in that language survives. Measured against that
version: all 18 "break every query in a language" mutations were caught, and
**0 of 28** "break one query" mutations were caught. #158 was TypeScript
`interfaces` breaking while `functions` and `classes` kept working, and C#
`inheritance` breaking while `imports` kept working -- so the per-language
form would not have caught either of the bugs it was written for.

The fixtures are hand-written rather than sampled so that each one exercises
every query its language declares. Writing them surfaced two queries that were
dead against the earlier fixtures and reported nothing: `java.inheritance`
(the fixture had `implements` but no `extends`) and `php.functions` (only
methods, no top-level function).

Five languages -- java, go, c, ruby, php -- have no corpus anywhere in the
local checkout, so until this file they had never been run against source in
any form, only compiled.
"""

import pathlib
import tempfile

import pytest

from neo.index.language_parser import EDGE_QUERIES, QUERIES, TreeSitterParser

# Each fixture must exercise EVERY query its language declares. When you add a
# query, extend the matching fixture in the same change -- the parametrized
# tests below will fail until you do, which is the point.
FIXTURES = {
    "java": """package x;

import java.util.List;
import java.util.Map;

public interface Greeter { String greet(String n); }

abstract class Base { abstract int seed(); }

public class Impl extends Base implements Greeter {
    public String greet(String n) { return "hi " + n; }
    int seed() { return 1; }
    private int count(List<String> xs) { return xs.size(); }
}
""",
    "go": """package main

import "fmt"

type Greeter interface { Greet(n string) string }

type Impl struct{ N int }

func (i Impl) Greet(n string) string { return fmt.Sprintf("hi %s", n) }

func main() { fmt.Println(Impl{}.Greet("x")) }
""",
    "c": """#include <stdio.h>
#include <stdlib.h>

struct Point { int x; int y; };

int add(int a, int b) { return a + b; }

int main(void) { printf("%d", add(1, 2)); return 0; }
""",
    "ruby": """require 'set'

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
""",
    "php": """<?php
namespace App;

use App\\Contracts\\Greeter;
use App\\Support\\Str;

function helper(int $x): int { return $x + 1; }

class Impl implements Greeter {
    public function greet(string $n): string { return "hi $n"; }
    private function total(array $xs): int { return count($xs); }
}
""",
}

_EXT = {"java": ".java", "go": ".go", "c": ".c", "ruby": ".rb", "php": ".php"}


@pytest.fixture(scope="module")
def parser():
    return TreeSitterParser()


def _captures(parser, language, table, query_name):
    """Capture count for ONE query against its language's fixture.

    Goes through `_compile_cached` + `_run_query` rather than `parse_file`,
    because the whole point is to attribute a zero to the specific query that
    produced it instead of letting a sibling mask it.
    """
    source = FIXTURES[language]
    compiled = parser._compile_cached(
        f"yieldtest:{language}:{query_name}", language, table[language][query_name]
    )
    if compiled is None:
        return None  # failed to compile -- reported distinctly below
    tree = parser._get_parser(language).parse(source.encode())
    return len(parser._run_query(compiled, tree.root_node))


def _cases(table):
    return [(lang, name) for lang in sorted(FIXTURES) if lang in table
            for name in sorted(table[lang])]


class TestEveryChunkQueryMatches:
    @pytest.mark.parametrize("language,query_name", _cases(QUERIES))
    def test_query_captures_something(self, parser, language, query_name):
        count = _captures(parser, language, QUERIES, query_name)
        assert count is not None, f"{language}.{query_name} failed to COMPILE"
        assert count > 0, (
            f"{language}.{query_name} compiles and matches nothing -- the "
            f"#158 failure mode, which is invisible from the outside. Either "
            f"the query is broken or the fixture does not exercise it; both "
            f"need fixing here."
        )


class TestEveryEdgeQueryMatches:
    @pytest.mark.parametrize("language,query_name", _cases(EDGE_QUERIES))
    def test_query_captures_something(self, parser, language, query_name):
        count = _captures(parser, language, EDGE_QUERIES, query_name)
        assert count is not None, f"{language}.{query_name} failed to COMPILE"
        assert count > 0, (
            f"{language}.{query_name} compiles and matches nothing"
        )


class TestEndToEndYield:
    """The per-query tests above can all pass while the wiring that turns
    captures into `CodeChunk`/`CodeEdge` objects drops them, so assert the
    public surface too."""

    @pytest.mark.parametrize("language", sorted(FIXTURES))
    def test_parse_file_produces_chunks(self, parser, language):
        source = FIXTURES[language]
        path = pathlib.Path(tempfile.mkdtemp()) / f"a{_EXT[language]}"
        path.write_text(source)
        assert parser.parse_file(path, source)

    @pytest.mark.parametrize("language", sorted(set(FIXTURES) & set(EDGE_QUERIES)))
    def test_extract_edges_produces_edges(self, parser, language):
        source = FIXTURES[language]
        path = pathlib.Path(tempfile.mkdtemp()) / f"a{_EXT[language]}"
        path.write_text(source)
        assert parser.extract_edges(path, source)


class TestKnownCoverageGaps:
    """Pinned so the gaps stay visible, and so closing one updates a test
    rather than silently changing behaviour."""

    def test_ruby_declares_no_edge_queries(self):
        """Ruby is the only language with chunk queries and no edge queries,
        so ruby files contribute nodes to the graph and never edges. Delete
        this test when ruby edge queries land -- its failure is the signal
        that the gap closed.
        """
        assert set(QUERIES) - set(EDGE_QUERIES) == {"ruby"}

    def test_php_has_imports_but_no_inheritance(self, parser):
        """`class Impl implements Greeter` yields an edge in java, c_sharp
        and typescript, and nothing in php.

        Asserted on `edge_type`, the field `CodeEdge` actually declares. An
        earlier version of this test read `getattr(e, "kind", getattr(e,
        "type", ""))` -- neither attribute exists, so the set was always
        `{""}` and the assertion could not fail. In a file whose subject is
        "a query that matches nothing is indistinguishable from one that
        failed to compile", that was an assertion indistinguishable from no
        assertion.
        """
        assert set(EDGE_QUERIES["php"]) == {"imports"}

        source = FIXTURES["php"]
        path = pathlib.Path(tempfile.mkdtemp()) / "a.php"
        path.write_text(source)
        edges = parser.extract_edges(path, source) or []

        assert edges, "fixture should still yield import edges"
        assert {e.edge_type for e in edges} == {"imports"}

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

        assert parser.extract_edges(directory / "esm.js", esm)
        assert not parser.extract_edges(directory / "cjs.js", cjs), (
            "CommonJS require yielding edges means the query widened -- good "
            "news, but update this test and docs/tree-sitter-setup.md"
        )
