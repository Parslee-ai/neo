"""Tests for context_gatherer relevance scoring — multi-language entry points."""

import pytest

from neo.context_gatherer import score_candidate


# Common args reused across tests. score_candidate is pure.
_EMPTY = set()
_ENTRY = {"main", "app", "server", "index", "login", "auth", "__init__"}


def _score(path: str, size: int = 1000) -> float:
    """Helper: score `path` with a no-keyword, no-git baseline."""
    return score_candidate(path, size, _EMPTY, _EMPTY, _ENTRY)


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
        big = score_candidate("widget.py", 50 * 1024, self._TOKENS, _EMPTY, _ENTRY)
        small = score_candidate("widget.py", 1000, self._TOKENS, _EMPTY, _ENTRY)
        assert big < small

    def test_large_main_file_penalized_lightly(self):
        big = score_candidate("main.py", 80 * 1024, self._TOKENS, _EMPTY, _ENTRY)
        small = score_candidate("main.py", 1000, self._TOKENS, _EMPTY, _ENTRY)
        assert big < small
        # And it should beat a same-size non-main file (lighter penalty
        # leaves it higher).
        non_main_big = score_candidate(
            "widget.py", 80 * 1024, self._TOKENS, _EMPTY, _ENTRY
        )
        assert big > non_main_big
