"""Tests for context_gatherer relevance scoring — multi-language entry points."""

import pytest

from neo.context_gatherer import score_candidate


# Common args reused across tests. score_candidate is pure.
_EMPTY = set()
_ENTRY = {"main", "app", "server", "index", "login", "auth", "__init__"}


def _score(path: str, size: int = 1000) -> float:
    """Helper: score `path` with a no-keyword, no-git baseline."""
    return score_candidate(path, size, _EMPTY, _EMPTY, _ENTRY,
                           demote_tests=False)


class TestMainImplBoost:
    def test_main_py_boosted(self):
        # main.py beats foo.py — entry point + main_impl bonuses both fire.
        assert _score("main.py") > _score("foo.py")

    def test_main_go_boosted(self):
        # The basename is lowercased, so the main_impl substring check
        # ('main') and the entry_point startswith both apply across
        # languages.
        assert _score("main.go") > _score("foo.go")

    def test_main_java_capitalcase_boosted(self):
        # Main.java → main.java after lowercasing — both boosts apply.
        assert _score("Main.java") > _score("Foo.java")

    def test_index_js_boosted(self):
        assert _score("index.js") > _score("widget.js")

    def test_app_ts_boosted(self):
        assert _score("app.ts") > _score("widget.ts")

    def test_server_rs_boosted(self):
        assert _score("server.rs") > _score("widget.rs")


class TestNoNeoBias:
    def test_neo_specific_filenames_no_longer_special(self):
        # Pre-refactor, files named like neo's internals (persistent.py,
        # structured_parser.py, etc.) got an unconditional +0.4. Now they
        # score the same as any other no-keyword Python file.
        assert _score("persistent.py") == _score("widget.py")
        assert _score("structured_parser.py") == _score("widget.py")

    def test_generic_core_boosted(self):
        # `core` is in the new generic list — boost stays for legitimate
        # main-implementation names.
        assert _score("core.py") > _score("widget.py")


class TestStemEqualityNotSubstring:
    # Substring matching would have flagged any of these as main-impl
    # via 'lib' or 'index' or 'app'. Stem-equality keeps them clean.

    def test_library_not_boosted(self):
        assert _score("library.py") == _score("widget.py")

    def test_accessibility_not_boosted(self):
        assert _score("accessibility.tsx") == _score("widget.tsx")

    def test_reindex_not_boosted(self):
        assert _score("reindex.py") == _score("widget.py")

    def test_application_loses_main_impl_boost(self):
        # Application.java used to get +0.4 from substring `'app' in
        # 'application'`. Stem-equality drops that. The entry_points
        # startswith check (separate mechanism) still fires for `app*`,
        # giving +0.2 — so it still beats Widget.java by less than it
        # used to.
        app_score = _score("Application.java")
        widget_score = _score("Widget.java")
        assert app_score > widget_score
        # And the gap is the entry_point bonus (0.2), not main_impl (0.4)
        # plus entry_point (0.2). Approximate because of depth penalty etc.
        assert (app_score - widget_score) == pytest.approx(0.2, abs=0.01)


class TestLargeFilePenalty:
    # Without some baseline score the max(0.0, …) clamp hides the penalty,
    # so each test gives the candidate a prompt-token match to lift it
    # above zero before the size hit lands.
    _TOKENS = {"widget"}

    def test_large_non_main_file_penalized_heavily(self):
        big = score_candidate(
            "widget.py", 50 * 1024, self._TOKENS, _EMPTY, _ENTRY,
            demote_tests=False)
        small = score_candidate(
            "widget.py", 1000, self._TOKENS, _EMPTY, _ENTRY,
            demote_tests=False)
        assert big < small

    def test_large_main_file_penalized_lightly(self):
        big = score_candidate(
            "main.py", 80 * 1024, self._TOKENS, _EMPTY, _ENTRY,
            demote_tests=False)
        small = score_candidate(
            "main.py", 1000, self._TOKENS, _EMPTY, _ENTRY,
            demote_tests=False)
        assert big < small
        # And it should beat a same-size non-main file (lighter penalty
        # leaves it higher).
        non_main_big = score_candidate(
            "widget.py", 80 * 1024, self._TOKENS, _EMPTY, _ENTRY,
            demote_tests=False,
        )
        assert big > non_main_big


class TestExplicitPathPinning:
    """A path the prompt names outright must reach the context bundle.

    Regression: asked to fix a named function in `src/neo/subcommands.py`, neo
    ranked that file 163rd of 296 candidates (score 0.838) while an unrelated
    file ranked 1st — filename-token overlap caps at 3 hits and an 86KB file
    takes a large size penalty. The file never entered the context, the model
    answered that the function body was not provided and emitted no patch, and a
    suggestion carrying no diff text can never be git-verified or learned from.
    """

    def test_named_path_outranks_an_organically_strong_file(self):
        from neo.context_gatherer import (
            EXPLICIT_PATH_BOOST, extract_explicit_paths, matches_explicit_path,
        )
        explicit = extract_explicit_paths(
            "Fix _classify_suggestion in src/neo/subcommands.py please")
        assert matches_explicit_path("src/neo/subcommands.py", explicit)
        # The boost must exceed every organic signal combined: filename overlap
        # caps at +1.8, the two re-rank boosts at +1.0 and +1.2.
        assert EXPLICIT_PATH_BOOST > 1.8 + 1.0 + 1.2

    def test_bare_filename_matches_but_a_neighbour_does_not(self):
        from neo.context_gatherer import extract_explicit_paths, matches_explicit_path
        explicit = extract_explicit_paths("look at subcommands.py")
        assert matches_explicit_path("src/neo/subcommands.py", explicit)
        # Matching the test file would re-bury the named file under its own
        # neighbours, which is the failure this exists to fix.
        assert not matches_explicit_path("tests/test_subcommands.py", explicit)

    def test_prose_and_versions_are_not_paths(self):
        from neo.context_gatherer import matches_explicit_path, extract_explicit_paths
        explicit = extract_explicit_paths(
            "Upgrade to 0.42.0, e.g. the parser, cf. the docs")
        # Junk tokens may be extracted, but nothing real can match them.
        for path in ("src/neo/engine.py", "README.md", "pyproject.toml"):
            assert not matches_explicit_path(path, explicit)

    def test_empty_prompt_pins_nothing(self):
        from neo.context_gatherer import extract_explicit_paths, matches_explicit_path
        assert extract_explicit_paths("") == set()
        assert not matches_explicit_path("src/neo/engine.py", set())


class TestSelectChunksRelevance:
    """`select_chunks` used to take the first five matching lines in FILE ORDER.

    Matching is a substring test and `extract_prompt_tokens` emits every 3+
    character word, so "in" matched `int`/`using`/`point` and virtually every
    line qualified. "First five matches" therefore meant "lines 1-5" for any
    prompt against any large file — the module docstring and imports — emitted
    as two overlapping near-duplicate windows that consumed both slots of the
    per-file cap.
    """

    @staticmethod
    def _big_file(target_line: str, target_at: int = 500, total: int = 900) -> str:
        lines = ["import os", "import sys", '"""Module docstring."""']
        lines += [f"    value_{i} = compute(i)  # padding using a point" for i in range(total)]
        lines[target_at] = target_line
        return "\n".join(lines)

    def test_window_centers_on_the_named_symbol_with_a_realistic_token_set(self):
        """Uses the REAL tokenizer rather than a hand-picked clean set. That
        distinction is the whole bug: with only discriminative tokens the old
        file-order code also found the symbol, so a clean set proves nothing.
        `extract_prompt_tokens` always emits the short words too."""
        from neo.context_gatherer import extract_prompt_tokens, select_chunks
        content = self._big_file("def _classify_suggestion(file_path, root):")
        tokens = extract_prompt_tokens(
            "Fix the bug in _classify_suggestion in src/neo/subcommands.py")
        assert "in" in tokens, "tokenizer no longer emits short noise words"
        chunks = select_chunks(content, tokens)
        assert chunks, "expected at least one window"
        top, start, _end = chunks[0]
        assert "def _classify_suggestion" in top
        assert start > 100, f"window started at {start} — that is the file header again"

    def test_short_noise_tokens_do_not_drag_windows_to_the_header(self):
        from neo.context_gatherer import select_chunks
        content = self._big_file("def target_function(arg):")
        # "in"/"is"/"py" are the tokens that matched nearly every line before.
        chunks = select_chunks(content, {"in", "is", "py", "target_function"})
        assert "def target_function" in chunks[0][0]

    def test_overlapping_windows_are_merged(self):
        from neo.context_gatherer import select_chunks
        # Must exceed max_chunk_bytes (12_000) or the whole file comes back as a
        # single chunk and the assertion below is vacuous.
        lines = [f"line {i} filler padding to grow the file" for i in range(1200)]
        lines[300] = "needle alpha here"
        lines[303] = "needle alpha again"
        content = "\n".join(lines)
        assert len(content) > 12_000, "fixture too small to exercise chunking"
        chunks = select_chunks(content, {"needle", "alpha"})
        ranges = [(s, e) for _c, s, e in chunks]
        assert len(ranges) == 1, f"centers 3 lines apart must merge, got {ranges}"
        # And two centers far apart must NOT merge.
        lines[900] = "needle alpha distant"
        far = select_chunks("\n".join(lines), {"needle", "alpha"})
        far_ranges = sorted((s, e) for _c, s, e in far)
        assert len(far_ranges) == 2
        for i, (s1, e1) in enumerate(far_ranges):
            for s2, e2 in far_ranges[i + 1:]:
                assert e1 < s2 or e2 < s1, f"overlapping windows {(s1,e1)} {(s2,e2)}"

    def test_small_file_returned_whole(self):
        from neo.context_gatherer import select_chunks
        content = "def f():\n    return 1\n"
        assert select_chunks(content, {"anything"}) == [(content, 1, 2)]

    def test_no_match_falls_back_to_the_header(self):
        from neo.context_gatherer import select_chunks
        content = "\n".join(f"line {i}" for i in range(2000))
        chunks = select_chunks(content, {"zzzznomatchzzzz"})
        assert len(chunks) == 1 and chunks[0][1] == 1


class TestExplicitPathSpellings:
    """Absolute paths are the highest-value input and were silently unmatched:
    `str.strip("./")` strips a character SET from both ends, so a leading "/"
    was eaten, and containment was tested in one direction only. Tracebacks,
    IDE copy-path and neo's own suggestion output all emit absolute paths."""

    @staticmethod
    def _explicit(text):
        from neo.context_gatherer import extract_explicit_paths
        return extract_explicit_paths(text)

    def test_absolute_path_matches_repo_relative_candidate(self):
        from neo.context_gatherer import matches_explicit_path
        explicit = self._explicit(
            "fix the bug in /Users/me/repo/src/neo/subcommands.py now")
        assert matches_explicit_path("src/neo/subcommands.py", explicit)

    def test_windows_spelling_normalizes(self):
        from neo.context_gatherer import matches_explicit_path
        explicit = self._explicit(r"look at src\neo\subcommands.py")
        assert matches_explicit_path("src/neo/subcommands.py", explicit)

    def test_neighbour_still_excluded_under_both_directions(self):
        from neo.context_gatherer import matches_explicit_path
        for text in ("see subcommands.py",
                     "see /Users/me/repo/src/neo/subcommands.py"):
            assert not matches_explicit_path("tests/test_subcommands.py",
                                             self._explicit(text))

    def test_empty_rel_path_never_matches(self):
        from neo.context_gatherer import matches_explicit_path
        assert not matches_explicit_path("", self._explicit("src/neo/x.py"))


class TestSelectChunksBudget:
    def test_chain_merge_cannot_blow_the_byte_cap(self):
        """Unbounded merging chained ~20 centers into one window measured at
        8.3x max_chunk_bytes. The caller admits a chunk all-or-nothing, so an
        oversized chunk that no longer fits the global budget is DROPPED and the
        named file contributes nothing — the original bug, new door."""
        from neo.context_gatherer import select_chunks
        lines = ["x" * 55 for _ in range(2000)]
        for i in range(0, 1600, 75):
            lines[i] = "needle_token_here alpha"
        chunks = select_chunks("\n".join(lines), {"needle_token_here", "alpha"})
        assert chunks
        for chunk, _s, _e in chunks:
            assert len(chunk) <= 12_000, f"chunk of {len(chunk)} exceeds the cap"

    def test_truncation_keeps_the_line_that_earned_the_window(self):
        from neo.context_gatherer import select_chunks
        lines = ["y" * 80 for _ in range(600)]
        lines[300] = "the_unique_target_symbol lives here"
        chunks = select_chunks("\n".join(lines), {"the_unique_target_symbol"})
        assert "the_unique_target_symbol" in chunks[0][0]

    def test_emitted_ranges_are_one_based_and_round_trip(self):
        from neo.context_gatherer import select_chunks
        lines = [f"line {i} padding text to grow the file" for i in range(1500)]
        lines[400] = "alpha_needle one"
        lines[1100] = "alpha_needle two"
        content = "\n".join(lines)
        for chunk, start, end in select_chunks(content, {"alpha_needle"}):
            assert "\n".join(lines[start - 1:end]) == chunk

    def test_header_does_not_win_when_a_better_region_exists(self):
        """Direct regression for the original defect: keyword-bearing module
        docstring must not beat the function body."""
        from neo.context_gatherer import extract_prompt_tokens, select_chunks
        lines = ['"""Handlers for CLI subcommands and classification."""']
        lines += [f"    filler_{i} = compute(i) using a point in time" for i in range(900)]
        lines[500] = "def _classify_suggestion(file_path, root):"
        chunks = select_chunks("\n".join(lines), extract_prompt_tokens(
            "fix _classify_suggestion in src/neo/subcommands.py"))
        assert "def _classify_suggestion" in chunks[0][0]
        assert chunks[0][1] > 100

    def test_single_match_at_each_boundary_clamps(self):
        from neo.context_gatherer import select_chunks
        for at in (0, 1499):
            lines = [f"line {i} padding text to grow the file" for i in range(1500)]
            lines[at] = "edge_needle here"
            chunks = select_chunks("\n".join(lines), {"edge_needle"})
            assert chunks and "edge_needle" in chunks[0][0]
            assert chunks[0][1] >= 1 and chunks[0][2] <= 1500

    def test_empty_token_set_falls_back_to_header(self):
        from neo.context_gatherer import select_chunks
        content = "\n".join(f"line {i} padding" for i in range(2000))
        chunks = select_chunks(content, set())
        assert len(chunks) == 1 and chunks[0][1] == 1
