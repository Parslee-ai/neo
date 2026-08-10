"""BM25 over file content — the signal file selection never had.

Selection used to score the path string and the byte count and never open the
file. The dominant term was `score -= 0.01 * size_kb`, uncapped, which made a
file with one keyword hit unrankable above 60 KB — measured, the 162 KB
`store.py` scored 0.000 and ranked 200th of 284 for a prompt about the fact
store. These tests pin the replacement and the properties that make it work.
"""

import pathlib

import pytest

from neo.file_retrieval import (
    PATH_TOKEN_WEIGHT,
    build_index,
    code_tokens,
    normalize,
)


class TestCodeTokens:
    def test_emits_whole_identifier_and_parts(self):
        """Both, not either.

        Dropping the whole identifier loses exact matches; dropping the parts
        loses every prose query. `getUserById` has to be reachable from a
        prompt that says "user by id" AND from one that names it exactly.
        """
        tokens = code_tokens("getUserById")
        assert "getuserbyid" in tokens
        assert {"get", "user", "by", "id"} <= set(tokens)

    def test_splits_on_separators(self):
        assert set(code_tokens("src/neo/fact_store.py")) >= {
            "src", "neo", "fact", "store", "py"
        }

    def test_short_identifiers_survive(self):
        """`db`, `os`, `fs`, `ui`, `id` carry signal.

        A token-length floor was measured against this corpus and rejected for
        exactly this reason — it drops the short identifiers and keeps the long
        English stopwords. BM25's IDF demotes ubiquitous terms from the corpus
        instead of guessing from length.
        """
        for short in ("db", "os", "fs", "ui", "id"):
            assert short in code_tokens(f"the {short} layer")

    def test_case_insensitive(self):
        assert code_tokens("FactStore") == code_tokens("factstore") + ["fact", "store"]

    def test_empty_and_punctuation_only(self):
        assert code_tokens("") == []
        assert code_tokens("--- /// ...") == []

    def test_camel_split_is_not_applied_to_all_caps(self):
        """`HTTPServer` has no lower-to-upper boundary before `S`, so the
        regex leaves `HTTPS`-style runs alone rather than shattering acronyms
        into letters."""
        assert "httpserver" in code_tokens("HTTPServer")


class TestFileIndex:
    @pytest.fixture
    def repo(self, tmp_path):
        (tmp_path / "src").mkdir()
        # Large and relevant — the case the old scorer could not rank.
        (tmp_path / "src" / "big.py").write_text(
            "def supersede_fact(fact):\n    '''supersession threshold'''\n"
            + "    # padding\n" * 8000
        )
        # Small and irrelevant.
        (tmp_path / "src" / "tiny.py").write_text("def unrelated():\n    return 1\n")
        # Named for the thing but empty of it.
        (tmp_path / "src" / "supersession.py").write_text("x = 1\n")
        return tmp_path

    def _candidates(self, repo):
        return [(str(p), str(p.relative_to(repo)), p.stat().st_size)
                for p in sorted(repo.rglob("*.py"))]

    def test_a_large_relevant_file_outranks_a_small_irrelevant_one(self, repo):
        """The property the size penalty made impossible."""
        index = build_index(self._candidates(repo))
        scores = index.scores("supersede fact supersession threshold")

        assert scores.get("src/big.py", 0) > scores.get("src/tiny.py", 0)

    def test_content_beats_a_matching_filename_with_no_content(self, repo):
        """`supersession.py` is named for the query and contains nothing.
        `big.py` is named for nothing and contains the answer.

        The path is still indexed (weighted `PATH_TOKEN_WEIGHT`) so a filename
        remains real evidence — it is just no longer the ONLY evidence, which
        is precisely what the old scorer got wrong.
        """
        index = build_index(self._candidates(repo))
        scores = index.scores("supersession threshold on a fact")

        assert scores.get("src/big.py", 0) > scores.get("src/supersession.py", 0)

    def test_path_tokens_are_weighted(self, repo):
        """Pins that the path contributes at all, so the weight cannot be
        silently dropped to zero."""
        assert PATH_TOKEN_WEIGHT >= 1
        index = build_index(self._candidates(repo))
        assert index.scores("supersession") .get("src/supersession.py", 0) > 0

    def test_empty_candidate_list_returns_none(self):
        assert build_index([]) is None

    def test_query_with_no_matching_terms_scores_nothing(self, repo):
        index = build_index(self._candidates(repo))
        assert index.scores("zzzq nonexistent gibberish") == {}

    def test_unreadable_file_does_not_break_the_index(self, repo, tmp_path):
        """A binary or permission-denied file must not take the walk with it."""
        missing = [(str(tmp_path / "gone.py"), "gone.py", 10)]
        index = build_index(self._candidates(repo) + missing)
        assert index is not None
        assert index.scores("supersede fact")


class TestNormalize:
    def test_scales_to_the_ceiling(self):
        out = normalize({"a": 4.0, "b": 2.0, "c": 1.0}, ceiling=1.0)
        assert out["a"] == pytest.approx(1.0)
        assert out["b"] == pytest.approx(0.5)

    def test_empty_input(self):
        assert normalize({}) == {}

    def test_all_zero_scores(self):
        assert normalize({"a": 0.0}) == {}

    def test_relative_order_is_preserved(self):
        raw = {"a": 9.1, "b": 3.3, "c": 7.7}
        out = normalize(raw)
        assert sorted(out, key=out.get, reverse=True) == sorted(raw, key=raw.get, reverse=True)

    def test_normalization_is_why_the_bonuses_stay_calibrated(self):
        """BM25 is unbounded and its scale moves with corpus and query length.

        Without normalization a raw score could dwarf `EXPLICIT_PATH_BOOST`
        (10.0) on one query and be dwarfed by git recency (0.3) on the next.
        """
        small = normalize({"a": 0.4, "b": 0.1})
        large = normalize({"a": 4000.0, "b": 1000.0})
        assert small == large


class TestRealRepository:
    """The measured case, against this repository."""

    def test_store_py_ranks_for_a_fact_store_prompt(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        candidates = [
            (str(p), str(p.relative_to(root)), p.stat().st_size)
            for p in (root / "src").rglob("*.py")
        ]
        index = build_index(candidates)
        scores = index.scores("fix the fact store supersession threshold")
        ranked = sorted(scores, key=scores.get, reverse=True)

        position = (ranked.index("src/neo/memory/store.py") + 1
                    if "src/neo/memory/store.py" in ranked else None)
        assert position is not None and position <= 5, (
            f"store.py ranked {position}; it ranked 200th of 284 under the "
            f"old scorer, because it is 162 KB"
        )
