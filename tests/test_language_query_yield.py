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

**Standing limit, narrower than it first looked.** No ruby or php source
exists anywhere in the local checkout, which is why these languages went
unvalidated. But the GRAMMARS are installed, so "wait for a real repository"
was the wrong conclusion and it cost real bugs: the first version of these
fixtures used a bare `class Impl < Base` and a bare `implements Greeter`, so
they agreed with queries that silently dropped `ActiveRecord::Base` and
`\\Foo\\Bar` -- every namespaced supertype in both ecosystems. A fixture
written by the query's author proves the query matches the fixture.

The fixtures are now framework-shaped rather than minimal: Rails
(`< ActiveRecord::Base`, `include ActiveSupport::Concern`) and PSR/Laravel
(`extends \\Foo\\BaseImpl implements \\ArrayAccess`, an 8.1 enum, a trait
`use`). Write the next one the same way -- copy a real declaration from a real
framework, do not invent a tidy one.
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
    # Framework-shaped, not hand-shaped. The first version of this fixture
    # used a bare `class Impl < Base` and a bare `include Greetable`, so it
    # agreed with a query that could not see `ActiveRecord::Base` -- a fixture
    # written by the query's author proves the query matches the fixture and
    # nothing else. `scope_resolution` is what real Ruby looks like.
    "ruby": """require 'set'
require_relative 'helper'

module Greetable
  def greet(n)
    "hi #{n}"
  end
end

class ApplicationRecord < ActiveRecord::Base
  include ActiveSupport::Concern
  include Greetable
  extend Forwardable

  def initialize(n)
    @n = n
  end

  def count
    @n.size
  end
end
""",
    # PSR/Laravel-shaped. The earlier fixture had no namespaced supertype, so
    # it agreed with a query that silently dropped every one of them.
    "php": """<?php
namespace App;

use App\\Contracts\\Greeter;
use App\\Support\\Str;

function helper(int $x): int { return $x + 1; }

interface Extended extends \\Psr\\Log\\LoggerInterface {}

trait Loggable { public function log(): void {} }

enum Status: string implements Greeter { case Active = 'active'; }

class Impl extends \\Foo\\BaseImpl implements \\ArrayAccess, Countable {
    use Loggable;

    public function greet(string $n): string { return "hi $n"; }
    private function total(array $xs): int { return count($xs); }
}
""",
}

_EXT = {"java": ".java", "go": ".go", "c": ".c", "ruby": ".rb", "php": ".php"}


@pytest.fixture(scope="module")
def parser():
    return TreeSitterParser()


def parser_edges(path, source):
    """Module-level edge extraction, so the closure tests below can call it
    without threading the fixture through."""
    return TreeSitterParser().extract_edges(path, source) or []


def _captures(parser, language, table, query_name):
    """Capture count for ONE query against its language's fixture.

    Resolves the query through the SAME accessor production uses, which is
    the whole point and was got wrong first time round. The original went
    through `_compile_cached("yieldtest:{lang}:{name}", ...)` -- a private
    namespace no production path touches -- so it verified the query TEXT
    while leaving the lookup untested. Two mutations that break real
    extraction left all 29 tests green: making `_get_query` return None for
    `java.classes` (production drops every `class:` chunk), and collapsing
    its cache key to `f"{language}"` so the first query per language wins and
    its siblings silently receive the wrong compiled object.

    That second one is one query yielding the wrong structure while its
    neighbours look fine -- the #158 shape this file exists for.

    The two tables resolve differently in production and must be followed
    separately: chunk queries go through `_get_query` (key `"{lang}:{name}"`),
    edge queries are compiled inline by `_extract_edges` under a third
    namespace (`"{lang}:edge:{name}"`) and never touch `_get_query` at all.
    """
    source = FIXTURES[language]
    if table is QUERIES:
        compiled = parser._get_query(language, query_name)
    else:
        compiled = parser._compile_cached(
            f"{language}:edge:{query_name}", language, table[language][query_name]
        )
    if compiled is None:
        return None  # failed to compile / not found -- reported distinctly below
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


# What `parse_file` must produce per language, as chunk_type counts. This is
# the check a capture COUNT cannot make: collapsing `_get_query`'s cache key
# to `f"{language}"` still returns a compiled query for every name, so every
# capture count stays non-zero while the wrong query answers -- java yields
# duplicated methods and zero classes. Only the shape of the output shows it.
EXPECTED_KINDS = {
    "c": {"function", "struct"},
    "go": {"function", "struct"},
    "java": {"class", "method"},
    "php": {"class", "function", "method"},
    "ruby": {"class", "method"},
}


class TestEndToEndYield:
    """The per-query tests above can all pass while the wiring that turns
    captures into `CodeChunk`/`CodeEdge` objects drops them, so assert the
    public surface too."""

    @pytest.mark.parametrize("language", sorted(EXPECTED_KINDS))
    def test_parse_file_produces_every_declared_kind(self, parser, language):
        """The kind-level guard, and the reason it exists.

        `_captures` proves each query name resolves to something that
        matches. It cannot prove the name resolved to the RIGHT query:
        mutating `_get_query`'s cache key to `f"{language}"` makes the first
        query per language win and hands its compiled object to every
        sibling. Every capture count stays positive; java then yields six
        duplicated `method` chunks and zero `class` chunks. Measured: that
        mutation left all 29 other tests in this file green.
        """
        source = FIXTURES[language]
        path = pathlib.Path(tempfile.mkdtemp()) / f"a{_EXT[language]}"
        path.write_text(source)

        kinds = {c.chunk_type for c in (parser.parse_file(path, source) or [])}
        assert kinds == EXPECTED_KINDS[language], (
            f"{language} produced {sorted(kinds)}, expected "
            f"{sorted(EXPECTED_KINDS[language])} -- a query resolved to the "
            f"wrong compiled object, or a declared kind stopped being emitted"
        )

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


class TestClosedCoverageGaps:
    """These two gaps were pinned as failing-when-fixed tests, and the pins
    have now fired. Kept as the inverse assertions so a regression that
    removes the queries fails here rather than going quiet -- which is the
    whole failure mode this file exists for."""

    def test_every_language_with_chunks_also_declares_edges(self):
        """Ruby used to be the sole exception, contributing nodes to the
        graph and never edges: no require/inherit/mixin relationship from any
        ruby file in any index."""
        assert set(QUERIES) - set(EDGE_QUERIES) == set()

    def test_php_extracts_inheritance_and_interfaces(self):
        """Namespaced supertypes are the point.

        `class Impl extends \\Foo\\BaseImpl implements \\ArrayAccess,
        Countable` yielded exactly ONE edge before this -- `Impl implements
        Countable` -- because the pattern required `name` as a direct child of
        `base_clause` and the real shape is `base_clause -> qualified_name ->
        name`. Silent PARTIAL extraction, which is worse than a total miss:
        the consumer sees a plausible answer with no signal that two thirds of
        it is absent. Exactly what the `_CS_BASE_TYPES` comment 150 lines
        above the query warns about.
        """
        source = FIXTURES["php"]
        path = pathlib.Path(tempfile.mkdtemp()) / "a.php"
        path.write_text(source)
        edges = parser_edges(path, source)

        inherits = {(e.source_symbol, e.target_symbol)
                    for e in edges if e.edge_type == "inherits"}
        implements = {(e.source_symbol, e.target_symbol)
                      for e in edges if e.edge_type == "implements"}

        assert ("Impl", "BaseImpl") in inherits          # namespaced extends
        assert ("Extended", "LoggerInterface") in inherits  # interface extends
        assert ("Impl", "ArrayAccess") in implements     # namespaced interface
        assert ("Impl", "Countable") in implements       # bare interface
        assert ("Status", "Greeter") in implements       # PHP 8.1 enum
        assert ("Impl", "Loggable") in implements        # trait use

    def test_ruby_extracts_requires_inheritance_and_mixins(self):
        """`class ApplicationRecord < ActiveRecord::Base` is the single most
        common inheritance statement in the Ruby ecosystem and produced no
        edge at all until the `scope_resolution` arm was added."""
        source = FIXTURES["ruby"]
        path = pathlib.Path(tempfile.mkdtemp()) / "a.rb"
        path.write_text(source)
        edges = parser_edges(path, source)

        by_type = {}
        for edge in edges:
            by_type.setdefault(edge.edge_type, set()).add(edge.target_symbol)

        assert by_type["imports"] == {"set", "helper"}
        assert ("ApplicationRecord", "Base") in {
            (e.source_symbol, e.target_symbol)
            for e in edges if e.edge_type == "inherits"
        }
        assert {"Concern", "Greetable", "Forwardable"} <= by_type["implements"]

    def test_ruby_import_predicate_is_enforced(self):
        """`require` is an ordinary method call, so the pattern matches any
        call taking a string. Only the `#match?` predicate separates an
        import from `puts 'hello'`.

        If a future tree-sitter release stopped applying predicates in
        `QueryCursor.captures`, the query would keep compiling and start
        emitting an import edge per string-argument call -- silent, and in
        the noisiest possible direction. This is the test that notices.
        """
        source = "require 'set'\nputs 'hello'\nlog_error 'boom'\n"
        path = pathlib.Path(tempfile.mkdtemp()) / "b.rb"
        path.write_text(source)

        targets = {e.target_symbol for e in parser_edges(path, source)}
        assert targets == {"set"}, (
            f"predicate not applied: {targets} -- every string-argument call "
            f"became an import"
        )


class TestJavascriptGap:
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
