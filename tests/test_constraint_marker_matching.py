"""#196, second pass: the language-aware check must still catch a real miss.

Making the markers language-aware fixed the permanent false alarm. Review of
that fix found the other half of the same defect had been introduced with it:
on the languages the fix *added*, a genuine missing handler now passed silently.

Two independent causes, pinned separately below.

1. The stripper followed the wrong grammar. `_strip_comments_and_strings` used
   Python's `tokenize`, which does not fail on C# — `//` lexes as floor-division
   — so line comments survived as code and `// TODO: use HashSet<int> here`
   satisfied the uniqueness marker for a file that deduplicates nothing. C#
   `///` and TS `/** */` doc comments name `HashSet`, `Distinct` and `OrderBy`
   constantly, so this was the common case, not the corner.

2. The matcher was a substring test over whitespace-stripped text, so `set(`
   matched `offset(`, `reset(` and `dataset(`, and `Set<` matched `Subset<`.
   UNIQUE_ELEMENTS was close to inert on real Python and TypeScript.

Both directions are asserted here: satisfied code stays quiet, unsatisfied code
still warns. A test that only pins the quiet half is what let this through.
"""

from neo.constraint_verification import (
    Constraint,
    ConstraintType,
    language_for_path,
    markers_for_language,
)
from neo.engine import NeoEngine
from neo.models import CodeSuggestion


class FakeLM:
    model = "fake"
    provider = "fake"

    def generate(self, messages, **kwargs):
        return ""

    def name(self):
        return "fake-lm"


def _engine() -> NeoEngine:
    return NeoEngine(lm_adapter=FakeLM(), enable_persistent_memory=False)


UNIQUE = Constraint(
    type=ConstraintType.UNIQUE_ELEMENTS,
    description="Elements must be unique",
    parameters={"variable": "result"},
)
SORTED = Constraint(
    type=ConstraintType.SORTED,
    description="Output must be sorted",
    parameters={"variable": "result"},
)


def _warns(file_path: str, code: str, constraint=UNIQUE) -> bool:
    """True when the check raises an actionable 'no obvious handler' warning."""
    suggestion = CodeSuggestion(
        file_path=file_path,
        unified_diff="",
        description="",
        confidence=0.9,
        code_block=code,
    )
    result = _engine()._check_constraints_static([suggestion], [constraint])
    if result is None:
        return False
    return any(d["severity"] == "warning" for d in result.diagnostics)


# --------------------------------------------- (1) comments, per language

def test_a_csharp_line_comment_naming_the_marker_does_not_satisfy_it():
    """The regression: Python's tokenizer left `//` comments in the code."""
    code = (
        "public List<int> F(List<int> xs)\n"
        "{\n"
        "    // TODO: use HashSet<int> here one day\n"
        "    return xs;\n"
        "}\n"
    )
    assert _warns("src/A.cs", code) is True


def test_a_csharp_doc_comment_naming_the_marker_does_not_satisfy_it():
    code = (
        "/// <summary>Callers should use .Distinct() on the result.</summary>\n"
        "public List<int> F(List<int> xs) { return xs; }\n"
    )
    assert _warns("src/A.cs", code) is True


def test_a_csharp_block_comment_naming_a_sort_marker_does_not_satisfy_it():
    code = (
        "/* we could OrderBy(x => x) here, but we do not */\n"
        "public List<int> F(List<int> xs) { return xs; }\n"
    )
    assert _warns("src/A.cs", code, SORTED) is True


def test_a_typescript_jsdoc_naming_the_marker_does_not_satisfy_it():
    code = (
        "/** Consider new Set(xs) if duplicates ever matter. */\n"
        "export function f(xs: string[]) { return xs; }\n"
    )
    assert _warns("src/a.ts", code) is True


def test_a_string_literal_naming_the_marker_does_not_satisfy_it():
    code = 'public string F() { return "call .Distinct() someday"; }'
    assert _warns("src/A.cs", code) is True


def test_a_python_comment_naming_the_marker_does_not_satisfy_it():
    """The Python half already worked; it is pinned so it stays working."""
    code = "# someday use set(xs)\ndef f(xs):\n    return xs\n"
    assert _warns("src/a.py", code) is True


def test_real_csharp_code_still_clears_the_constraint():
    """The other direction: stripping comments must not eat the code."""
    code = (
        "public HashSet<int> F(List<int> xs)\n"
        "{\n"
        "    // dedupe below\n"
        "    return new HashSet<int>(xs);\n"
        "}\n"
    )
    assert _warns("src/A.cs", code) is False


def test_a_url_inside_a_string_does_not_swallow_the_rest_of_the_file():
    """`//` inside a literal must be consumed as a string, not as a comment.

    If the scanner read `//x` in the URL as a line comment it would delete the
    rest of that line only — but a naive comment-first alternation would run to
    the newline from inside the string and could hide the real marker.
    """
    code = (
        'var url = "http://example.com/x";\n'
        "var seen = new HashSet<int>(xs);\n"
    )
    assert _warns("src/A.cs", code) is False


# ------------------------------------- (2) identifier-boundary matching

def test_offset_does_not_satisfy_the_unique_marker():
    """`set(` inside `offset(` — the collision that made UNIQUE_ELEMENTS inert."""
    assert _warns("src/a.py", "def f(rows, n):\n    return rows[offset(n):]\n") is True


def test_reset_and_dataset_do_not_satisfy_the_unique_marker():
    assert _warns("src/a.py", "def f(x):\n    x.reset(0)\n    return x\n") is True
    assert _warns("src/a.py", "def f(x):\n    return load_dataset(x)\n") is True


def test_subset_does_not_satisfy_the_typescript_unique_marker():
    code = "export function f(x: Subset<string>) { return x; }"
    assert _warns("src/a.ts", code) is True


def test_new_settings_does_not_satisfy_the_typescript_unique_marker():
    code = "export function f(xs: string[]) { const s = new Settings(); return xs; }"
    assert _warns("src/a.ts", code) is True


def test_a_real_set_call_still_clears_the_constraint():
    assert _warns("src/a.py", "def f(xs):\n    return list(set(xs))\n") is False


def test_frozenset_clears_the_constraint_on_its_own_entry():
    """It used to ride in as a substring of `set(`; now it is listed."""
    assert _warns("src/a.py", "def f(xs):\n    return frozenset(xs)\n") is False


def test_both_typescript_set_spellings_clear_the_constraint():
    assert _warns("src/a.ts", "const s = new Set(); return s;") is False
    assert _warns("src/a.ts", "const s: Set<string> = new Set<string>(); return s;") is False


def test_markers_are_prefixes_of_a_family_and_must_stay_unanchored_right():
    """The right edge is deliberately not anchored.

    `OrderBy` is written to cover `OrderByDescending`, `bisect` to cover
    `bisect_left`. Anchoring both edges rejects them, and a rejected marker is
    a warning fired at code that satisfies the constraint — #196 returning
    through the matcher.
    """
    assert _warns("src/A.cs", "return xs.OrderByDescending(x => x).ToList();", SORTED) is False
    assert _warns("src/a.py", "def f(a, x):\n    return bisect_left(a, x)\n", SORTED) is False
    assert _warns("src/a.py", "import heapq\ndef f(h, x):\n    heapq.heappush(h, x)\n", SORTED) is False


def test_the_token_rejoin_regression_stays_fixed():
    """Python's tokenizer rejoins as `sorted ( x )`; the marker is `sorted(`."""
    assert _warns("src/a.py", "def f(xs):\n    return sorted(xs)\n", SORTED) is False
    assert _warns("src/a.py", "def f(xs):\n    return max( 0 , xs )\n",
                  Constraint(
                      type=ConstraintType.NON_NEGATIVE,
                      description="Must be non-negative",
                      parameters={},
                  )) is False


def test_a_genuine_miss_still_warns_in_every_mapped_language():
    """The whole point of the check, asserted once per language table."""
    assert _warns("src/a.py", "def f(xs):\n    return xs\n") is True
    assert _warns("src/A.cs", "public List<int> F(List<int> xs) { return xs; }") is True
    assert _warns("src/a.ts", "export function f(xs: string[]) { return xs; }") is True


def test_every_marker_in_every_table_matches_itself():
    """A marker no code can contain is unsatisfiable by construction — #196.

    Each marker is checked against a minimal snippet that is the marker with an
    identifier boundary in front of it, which is the weakest thing a real usage
    can look like.
    """
    for language, table in [
        ("python", markers_for_language("python")),
        ("csharp", markers_for_language("csharp")),
        ("typescript", markers_for_language("typescript")),
    ]:
        for constraint_type, hints in table.items():
            for hint in hints:
                normalized = NeoEngine._normalize_for_marker_match(f"x = {hint}")
                assert NeoEngine._marker_present(hint, normalized), (
                    f"{language}/{constraint_type.value} marker {hint!r} "
                    "cannot match its own text"
                )


def test_unmapped_languages_are_left_untouched_by_the_stripper():
    """No marker table is consulted, so no grammar is assumed either."""
    source = "// go comment\nfunc F(xs []int) []int { return xs }"
    assert language_for_path("src/a.go") not in ("python", "csharp", "typescript")
    assert NeoEngine._strip_comments_and_strings(source, "go") == source
