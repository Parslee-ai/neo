"""#196: the constraint marker check is language-aware, or it says it did not check.

`CONSTRAINT_CODE_MARKERS` was a single Python-only table applied to every
target, so on a C#, TypeScript or markdown file the expectation it printed
("expected one of: set(, dict.fromkeys") could not be met by any code. The
caution fired permanently on non-Python repos, which is alarm fatigue on the
same channel that carries Neo's real warnings.

Two things are pinned here: a language with a marker table gets a real check in
its own vocabulary, and a language without one gets an honest "not checked"
note that no downstream counter treats as a problem.
"""

from neo.constraint_verification import (
    LANGUAGE_CONSTRAINT_MARKERS,
    UNKNOWN_LANGUAGE,
    Constraint,
    ConstraintType,
    language_for_path,
)
from neo.engine import NeoEngine
from neo.models import CodeSuggestion, StaticCheckResult


class FakeLM:
    model = "fake"
    provider = "fake"

    def generate(self, messages, **kwargs):
        return ""

    def name(self):
        return "fake-lm"


def _engine() -> NeoEngine:
    return NeoEngine(lm_adapter=FakeLM(), enable_persistent_memory=False)


def _suggestion(file_path: str, code: str) -> CodeSuggestion:
    return CodeSuggestion(
        file_path=file_path,
        unified_diff="",
        description="",
        confidence=0.9,
        code_block=code,
    )


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


def _check(file_path: str, code: str, constraints=(UNIQUE,)):
    return _engine()._check_constraints_static(
        [_suggestion(file_path, code)], list(constraints)
    )


def _warnings(result):
    if result is None:
        return []
    return [d for d in result.diagnostics if d["severity"] == "warning"]


def _notes(result):
    if result is None:
        return []
    return [d for d in result.diagnostics if d["severity"] == "info"]


# ------------------------------------------------------- the reported case


def test_a_markdown_target_never_reports_a_missing_python_handler():
    """The exact run from #196: a uniqueness constraint against `/decision.md`.

    Prose cannot contain `set(`, so the old check fired on every invocation.
    """
    result = _check("/decision.md", "# Decision\n\nWe will use the second option.")

    assert _warnings(result) == []
    notes = _notes(result)
    assert len(notes) == 1
    assert notes[0]["language"] == "markdown"
    assert "not checked" in notes[0]["message"]
    assert "set(" not in notes[0]["message"]


def test_a_csharp_hashset_satisfies_uniqueness_without_a_caution():
    result = _check(
        "src/Odi/Ledger.cs",
        "var unique = new HashSet<string>(names);\nreturn unique.ToList();",
    )
    assert result is None


def test_a_csharp_distinct_satisfies_uniqueness_without_a_caution():
    result = _check("src/Odi/Ledger.cs", "return names.Distinct().ToList();")
    assert result is None


def test_a_csharp_orderby_satisfies_sorting_without_a_caution():
    result = _check(
        "src/Odi/Ledger.cs", "return names.OrderBy(n => n).ToList();", (SORTED,)
    )
    assert result is None


# ------------------------------------- real misses still warn, per language


def test_a_csharp_file_with_no_handler_warns_in_csharp_vocabulary():
    result = _check("src/Odi/Ledger.cs", "return names.ToList();")

    warnings = _warnings(result)
    assert len(warnings) == 1
    message = warnings[0]["message"]
    assert "no obvious handler" in message
    assert "HashSet<" in message
    # The Python spelling must not survive into a C# expectation — naming a
    # marker the language does not have is what made the caution unclearable.
    assert "dict.fromkeys" not in message
    assert warnings[0]["language"] == "csharp"


def test_a_python_file_with_no_handler_still_warns():
    """The check the fix must not throw away while removing the false alarm."""
    result = _check("src/neo/ledger.py", "return list(names)")

    warnings = _warnings(result)
    assert len(warnings) == 1
    assert "dict.fromkeys" in warnings[0]["message"]
    assert warnings[0]["language"] == "python"


def test_a_python_set_satisfies_uniqueness():
    result = _check("src/neo/ledger.py", "return list(set(names))")
    assert result is None


def test_a_satisfied_python_constraint_survives_the_token_rejoin():
    """The Python half of the same permanent caution.

    `_strip_comments_and_strings` rebuilds the source by joining tokens with a
    space, so `sorted(x)` reached the matcher as `sorted ( x )` and the marker
    `sorted(` could not match. Code that plainly satisfied the constraint still
    warned — on the one language the table was written for.
    """
    assert _check("a.py", "result = sorted(values)", (SORTED,)) is None
    assert _check("a.py", "result = sorted( values )", (SORTED,)) is None
    assert _check("a.py", "values.sort( reverse = True )", (SORTED,)) is None


def test_a_marker_word_in_a_comment_still_warns():
    """Stripping comments is why the matcher is not run on raw source."""
    result = _check("a.py", "# remember to use sorted() here\nreturn values", (SORTED,))
    assert len(_warnings(result)) == 1


def test_a_typescript_set_satisfies_uniqueness():
    result = _check("src/app/ledger.ts", "return [...new Set(names)];")
    assert result is None


def test_a_typescript_file_with_no_handler_warns_in_typescript_vocabulary():
    result = _check("src/app/ledger.ts", "return names.slice();")

    warnings = _warnings(result)
    assert len(warnings) == 1
    assert "new Set" in warnings[0]["message"]
    assert "dict.fromkeys" not in warnings[0]["message"]


def test_javascript_shares_the_typescript_markers():
    assert _check("src/app/ledger.js", "return [...new Set(names)];") is None


# ------------------------------------------------------ honest degradation


def test_an_unmapped_extension_is_named_rather_than_guessed_at():
    """Rust has no marker table, and the note says so by name."""
    result = _check("src/ledger.rs", "let unique: HashSet<_> = names.iter().collect();")

    assert _warnings(result) == []
    assert _notes(result)[0]["language"] == "rust"


def test_a_path_with_no_extension_is_reported_as_unknown():
    result = _check("Makefile", "all:\n\techo hi")

    assert _warnings(result) == []
    assert _notes(result)[0]["language"] == UNKNOWN_LANGUAGE


def test_the_analysis_only_placeholder_paths_are_unknown_not_guessed():
    """`/` and `N/A` are the schema's "no code change" markers, not files."""
    for path in ("/", "N/A", ""):
        assert language_for_path(path) == UNKNOWN_LANGUAGE


def test_language_for_path_prefers_the_canonical_name():
    assert language_for_path("a/b/c.py") == "python"
    assert language_for_path("A/B/C.CS") == "csharp"
    assert language_for_path("/abs/path/app.tsx") == "typescript"


def test_the_summary_reports_unchecked_constraints_separately_from_failures():
    result = _engine()._check_constraints_static(
        [_suggestion("notes.md", "prose"), _suggestion("a.py", "return list(names)")],
        [UNIQUE],
    )
    assert "1 constraint(s) may not be handled" in result.summary
    assert "1 not checked (markdown)" in result.summary


# ------------------------------- informational notes are not problem reports


def test_a_check_that_only_said_not_checked_does_not_count_as_passed():
    """`passed` is what lets a run exit early on clean static analysis.

    A checker with no marker set for the target language evaluated nothing, so
    it has no cleanliness to vouch for.
    """
    result = _check("/decision.md", "prose")
    assert NeoEngine._static_check_status(result) == "skipped"


def test_a_real_warning_still_maps_to_warning_status():
    result = _check("src/neo/ledger.py", "return list(names)")
    assert NeoEngine._static_check_status(result) == "warning"


def test_an_error_still_maps_to_failed_and_an_empty_check_to_passed():
    failed = StaticCheckResult(
        tool_name="ruff", diagnostics=[{"severity": "error"}], summary=""
    )
    clean = StaticCheckResult(tool_name="ruff", diagnostics=[], summary="")
    assert NeoEngine._static_check_status(failed) == "failed"
    assert NeoEngine._static_check_status(clean) == "passed"


def test_a_diagnostic_with_no_severity_counts_as_actionable():
    """Unknown severities fail toward being surfaced, not toward silence."""
    assert NeoEngine._is_actionable_diagnostic({}) is True
    assert NeoEngine._is_actionable_diagnostic({"severity": "warning"}) is True
    assert NeoEngine._is_actionable_diagnostic({"severity": "info"}) is False


def test_unchecked_notes_raise_no_next_question():
    """"constraint_verifier found 1 issues" is the line #196 reported."""
    engine = _engine()
    result = _check("/decision.md", "prose")
    questions = engine._generate_questions([], [], [], [result])
    assert questions == []


def test_unchecked_notes_do_not_dock_confidence():
    engine = _engine()
    suggestion = _suggestion("src/app.ts", "return [...new Set(names)];")
    unchecked = _check("/decision.md", "prose")

    scored, _ = engine._calculate_confidence([], [], [suggestion], [unchecked])
    unpenalized, _ = engine._calculate_confidence([], [], [suggestion], [])
    assert scored == unpenalized


# ---------------------------------------------------------- table hygiene


def test_every_language_table_covers_the_same_constraint_types():
    """Otherwise a constraint added to one language degrades to "not checked"
    on the others — honest, but silently useless."""
    covered = {
        language: frozenset(markers)
        for language, markers in LANGUAGE_CONSTRAINT_MARKERS.items()
    }
    assert len(set(covered.values())) == 1, covered


def test_every_marker_is_a_non_empty_string():
    for language, markers in LANGUAGE_CONSTRAINT_MARKERS.items():
        for constraint_type, hints in markers.items():
            assert hints, f"{language}/{constraint_type} has no markers"
            for hint in hints:
                assert isinstance(hint, str) and hint.strip(), (language, hint)
