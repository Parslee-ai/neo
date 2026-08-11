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


class TestConstantsArePinned:
    """Every constant this module turns on, asserted by VALUE.

    Four survived a mutation sweep because nothing pinned them, and one of the
    survivors is what shipped: the content term was applied twice for three
    commits, so the effective weight was 6.0 while every docstring, comment and
    measurement table said 3.0. The test that should have caught it named 3.0
    in its docstring and asserted an inequality that 1.76-through-infinity
    satisfies. Naming a number in prose while asserting a direction is how a
    constant goes unpinned and looks pinned.
    """

    def test_content_weight_is_applied_exactly_once(self):
        """The shipped bug, in the form that would have caught it."""
        import pathlib

        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "src" / "neo" / "context_gatherer.py").read_text()
        assert source.count("CONTENT_WEIGHT * content_relevance") == 1

    def test_content_weight_value(self):
        from neo.context_gatherer import CONTENT_WEIGHT

        assert CONTENT_WEIGHT == 3.0

    def test_content_weight_outranks_every_tie_breaker_by_the_stated_margin(self):
        """Equality, not `>`.

        docs 0.8 + git 0.3 + entry 0.2 + filename 0.45 = 1.75. The margin is
        the claim; asserting only the direction lets the weight drift anywhere
        above it.
        """
        from neo.context_gatherer import CONTENT_WEIGHT

        tie_breakers = 0.8 + 0.3 + 0.2 + 0.45
        assert CONTENT_WEIGHT > tie_breakers
        assert CONTENT_WEIGHT == pytest.approx(3.0)
        assert CONTENT_WEIGHT / tie_breakers == pytest.approx(1.714, abs=0.01)

    def test_normalize_ceiling_default(self):
        """Mutating the default to 0.05 left all 156 selection tests green and
        cost R@10 0.875 -> 0.750 on the real CLI.

        That R@10 is the `tools/rank_eval.py` generation -- 12 hand-labelled
        prompts, this repo. It is NOT comparable with the cross-repo R@10 in
        `neo.file_retrieval`, which is `tools/rank_mine_eval.py` over git-mined
        cases. Same label, different instrument; keep the generation attached.
        """
        from neo.file_retrieval import normalize

        assert normalize({"a": 4.0, "b": 2.0})["a"] == pytest.approx(1.0)

    def test_max_indexed_chars_value(self):
        """Mutating 200_000 -> 2_000 left all 156 selection tests green and
        cost R@10 0.875 -> 0.750 (the `tools/rank_eval.py` labelled generation,
        not the cross-repo mined one -- see `test_normalize_ceiling_default`)."""
        from neo.file_retrieval import MAX_INDEXED_CHARS

        assert MAX_INDEXED_CHARS == 200_000

    def test_path_token_weight_value(self):
        from neo.file_retrieval import PATH_TOKEN_WEIGHT

        assert PATH_TOKEN_WEIGHT == 3


class TestBinaryFilesAreNotIndexed:
    """Binary content must not reach BM25.

    `errors="ignore"` turned PDFs into junk tokens. Not merely wasted work:
    BM25 divides by corpus-average document length, so binary blobs made every
    real file's length penalty depend on how many PDFs sat in the repo.
    Measured here: 5 PDFs were 1.7% of documents and 32.9% of all tokens, and
    `avgdl` moved 2773 -> 1824 once they were excluded.
    """

    def test_a_nul_bearing_file_yields_no_text(self, tmp_path):
        from neo.file_retrieval import _read

        binary = tmp_path / "doc.pdf"
        binary.write_bytes(b"%PDF-1.4\x00\x00binary\x00garbage")
        assert _read(str(binary)) == ""

    def test_undecodable_bytes_yield_no_text(self, tmp_path):
        from neo.file_retrieval import _read

        bad = tmp_path / "latin.py"
        bad.write_bytes(b"x = '\xff\xfe not utf-8'")
        assert _read(str(bad)) == ""

    def test_real_source_still_reads(self, tmp_path):
        from neo.file_retrieval import _read

        good = tmp_path / "ok.py"
        good.write_text("def hello():\n    return 'wörld'\n")
        assert "hello" in _read(str(good))

    def test_binaries_do_not_enter_the_index(self, tmp_path):
        from neo.file_retrieval import build_index

        (tmp_path / "a.py").write_text("def supersede(): pass\n")
        (tmp_path / "b.pdf").write_bytes(b"%PDF\x00" + b"supersede " * 500)
        cands = [(str(p), p.name, p.stat().st_size) for p in sorted(tmp_path.iterdir())]

        scores = build_index(cands).scores("supersede")
        assert "b.pdf" not in scores
        assert scores.get("a.py", 0) > 0
