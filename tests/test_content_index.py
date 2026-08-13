"""Tests for the persistent BM25 content index.

Three properties carry the change, and each of them is a way it could have
gone wrong silently:

1. **Parity.** Persisting the index must not re-rank anything. Every test in
   `TestParity` scores the same corpus through both the per-call `FileIndex`
   and the persisted `ContentIndex` and asserts they agree — a persistence
   change that quietly moved scores would present as a retrieval regression
   with no diff to blame it on.
2. **Freshness.** An index that answers from stale postings is worse than no
   index, because the answer looks right. `TestFreshness` edits, adds, deletes
   and touches files and asserts what comes back.
3. **Degradation.** Corrupt, unwritable, or written by a different tokenizer —
   each must produce a warning and a correct answer, never a crash and never a
   silently empty ranking.
"""

import os


import pytest

from neo import eligibility
from neo.file_retrieval import FileIndex, code_tokens
from neo.index import content_index as ci
from neo.index.content_index import ContentIndex


def _walk(root):
    return eligibility.walk_paths(str(root), max_file_bytes=512_000).paths


def _repo(tmp_path, files):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return tmp_path


CORPUS = {
    "src/store.py": (
        "def supersede(fact, other):\n"
        "    '''Retire a fact superseded by a newer one.'''\n"
        "    threshold = 0.85\n"
        "    return fact.similarity(other) >= threshold\n"
    ),
    "src/engine.py": (
        "class Engine:\n"
        "    def process(self, request):\n"
        "        return self.reason(request)\n"
    ),
    "src/parser/lexer.py": "def tokenize(text):\n    return text.split()\n",
    "README.md": "# Project\n\nA store of facts with supersession.\n",
    "tests/test_store.py": "def test_supersede():\n    assert True\n",
}


class TestParity:
    """The persisted index ranks exactly like the per-call one it replaces."""

    def _both(self, tmp_path, prompt):
        paths = _walk(tmp_path)
        memory = FileIndex([(p.path, p.rel_path, p.size) for p in paths])
        with ContentIndex(str(tmp_path)) as index:
            index.refresh(paths, quiet=True)
            return memory.scores(prompt), index.scores(prompt)

    @pytest.mark.parametrize(
        "prompt",
        [
            "supersession threshold",
            "how does the engine process a request",
            "tokenize",
            "supersede supersede supersede",  # repeated term: multiplicity
            "getUserById",  # camelCase split
            "src/parser/lexer.py",  # a path as the query
        ],
    )
    def test_scores_match_the_in_memory_index(self, tmp_path, prompt):
        _repo(tmp_path, CORPUS)
        expected, actual = self._both(tmp_path, prompt)
        assert set(expected) == set(actual)
        for path, score in expected.items():
            assert actual[path] == pytest.approx(score, rel=1e-9, abs=1e-12)

    def test_ranking_order_matches(self, tmp_path):
        _repo(tmp_path, CORPUS)
        expected, actual = self._both(tmp_path, "supersession threshold")
        rank = lambda scores: [  # noqa: E731 - a local, used twice
            p for p, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        assert rank(expected) == rank(actual)

    def test_query_term_multiplicity_is_preserved(self, tmp_path):
        """A term repeated in the prompt weighs more, as it did in the list.

        The obvious optimization -- dedupe the query terms before hitting the
        postings table -- is a silent ranking change, because the scorer this
        replaces iterated the token LIST and `code_tokens` emits a repeated
        word repeatedly.
        """
        _repo(tmp_path, CORPUS)
        assert code_tokens("supersede supersede").count("supersede") == 2
        once, _ = self._both(tmp_path, "supersede")
        twice, persisted_twice = self._both(tmp_path, "supersede supersede")
        assert twice["src/store.py"] > once["src/store.py"]
        assert persisted_twice["src/store.py"] == pytest.approx(twice["src/store.py"])

    def test_gibberish_scores_nothing(self, tmp_path):
        _repo(tmp_path, CORPUS)
        expected, actual = self._both(tmp_path, "zzqx wibble frobnicate")
        assert expected == {}
        assert actual == {}

    def test_a_binary_file_still_contributes_its_path(self, tmp_path):
        """Its bytes are not text; its name is still a real name in the repo."""
        _repo(tmp_path, CORPUS)
        (tmp_path / "src" / "supersede.bin").write_bytes(b"\x00\x01\x02binary")
        expected, actual = self._both(tmp_path, "supersede")
        assert "src/supersede.bin" in expected
        assert actual["src/supersede.bin"] == pytest.approx(
            expected["src/supersede.bin"]
        )


class TestFreshness:
    """Only what moved is re-read, and what moved is always reflected."""

    def _refresh(self, tmp_path, index):
        return index.refresh(_walk(tmp_path), quiet=True)

    def test_first_run_is_cold_and_says_so(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            report = self._refresh(tmp_path, index)
        assert report.mode == "cold"
        assert report.indexed == len(CORPUS)
        assert "cold build" in report.describe()

    def test_second_run_is_warm_and_re_reads_nothing(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            self._refresh(tmp_path, index)
        with ContentIndex(str(tmp_path)) as index:
            report = self._refresh(tmp_path, index)
        assert report.mode == "warm"
        assert report.indexed == 0
        assert "read warm" in report.describe()

    def test_an_edit_is_reflected_with_no_stale_hits(self, tmp_path):
        """The staleness test the plan asks for: edit, re-run, see the change.

        Both halves matter. The new content has to become findable, and the
        old content has to stop being findable -- an index that only ever adds
        postings passes the first check and answers with text that is no
        longer in the file.
        """
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            self._refresh(tmp_path, index)
            assert "src/engine.py" in index.scores("reason request")
            assert not index.scores("quantumleap")

        (tmp_path / "src" / "engine.py").write_text(
            "class Engine:\n    def quantumleap(self):\n        return 1\n"
        )
        with ContentIndex(str(tmp_path)) as index:
            report = self._refresh(tmp_path, index)
            assert report.mode == "incremental"
            assert report.changed == 1
            assert report.indexed == 1
            assert index.scores("quantumleap").get("src/engine.py", 0) > 0
            # The removed word must be gone from the postings, not merely
            # outranked.
            assert "src/engine.py" not in index.scores("reason request")

    def test_an_added_file_becomes_queryable(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            self._refresh(tmp_path, index)
        (tmp_path / "src" / "cache.py").write_text("def evict(entry):\n    pass\n")
        with ContentIndex(str(tmp_path)) as index:
            report = self._refresh(tmp_path, index)
            assert report.added == 1
            assert "src/cache.py" in index.scores("evict")

    def test_a_deleted_file_stops_answering(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            self._refresh(tmp_path, index)
        os.unlink(tmp_path / "src" / "parser" / "lexer.py")
        with ContentIndex(str(tmp_path)) as index:
            report = self._refresh(tmp_path, index)
            assert report.removed == 1
            assert "src/parser/lexer.py" not in index.scores("tokenize")

    def test_a_newly_excluded_file_stops_answering(self, tmp_path):
        """Eligibility is the walker's verdict, and the index obeys it.

        A file that becomes gitignored is still on disk and still hashes the
        same, so nothing about the FILE changed. It leaves the index because
        it left the walk -- which is the only reason this module never needs
        an exclusion rule of its own.
        """
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            self._refresh(tmp_path, index)
            assert "src/parser/lexer.py" in index.scores("tokenize")

        (tmp_path / ".gitignore").write_text("src/parser/\n")
        with ContentIndex(str(tmp_path)) as index:
            self._refresh(tmp_path, index)
            assert "src/parser/lexer.py" not in index.scores("tokenize")

    def test_a_touch_without_an_edit_re_tokenizes_nothing(self, tmp_path):
        """`touch` moves mtime; the hash says the content did not move.

        Reported separately from a real change, because a run that says it
        re-indexed 400 files when it tokenized none is lying about its work.
        """
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            self._refresh(tmp_path, index)
        target = tmp_path / "src" / "store.py"
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with ContentIndex(str(tmp_path)) as index:
            report = self._refresh(tmp_path, index)
        assert report.touched == 1
        assert report.changed == 0
        assert report.indexed == 0
        assert "touched (content identical)" in report.describe()

    def test_a_touch_is_stamped_so_it_is_not_re_hashed_forever(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            self._refresh(tmp_path, index)
        target = tmp_path / "src" / "store.py"
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        with ContentIndex(str(tmp_path)) as index:
            self._refresh(tmp_path, index)
        with ContentIndex(str(tmp_path)) as index:
            report = self._refresh(tmp_path, index)
        assert report.mode == "warm"


class TestCorpusScope:
    """Per-call flags narrow the answer; they never prune the stored corpus."""

    def test_a_candidate_subset_narrows_the_result(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            index.refresh(_walk(tmp_path), quiet=True)
            everything = index.scores("supersede")
            subset = index.scores("supersede", ["src/store.py"])
        assert len(everything) > 1
        assert set(subset) == {"src/store.py"}
        # The score itself is unchanged: corpus statistics are global, so
        # narrowing the answer cannot move a file's number.
        assert subset["src/store.py"] == pytest.approx(everything["src/store.py"])

    def test_an_unknown_candidate_path_is_simply_absent(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            index.refresh(_walk(tmp_path), quiet=True)
            assert index.scores("supersede", ["nope/missing.py"]) == {}


class TestDegradation:
    """Every failure is loud, correct, and survivable."""

    def test_a_tokenizer_bump_invalidates_cleanly(self, tmp_path, monkeypatch):
        """Every posting is wrong and no file hash can tell you so."""
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            index.refresh(_walk(tmp_path), quiet=True)

        monkeypatch.setattr(ci, "TOKENIZER_VERSION", ci.TOKENIZER_VERSION + 1)
        with ContentIndex(str(tmp_path)) as index:
            report = index.refresh(_walk(tmp_path), quiet=True)
            assert report.mode == "rebuilt"
            assert report.indexed == len(CORPUS)
            # And it still answers.
            assert index.scores("supersede")

    def test_a_schema_bump_invalidates_cleanly(self, tmp_path, monkeypatch):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            index.refresh(_walk(tmp_path), quiet=True)
        monkeypatch.setattr(ci, "SCHEMA_VERSION", ci.SCHEMA_VERSION + 1)
        with ContentIndex(str(tmp_path)) as index:
            assert index.refresh(_walk(tmp_path), quiet=True).mode == "rebuilt"

    def test_a_corrupt_store_is_rebuilt_not_raised(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            index.refresh(_walk(tmp_path), quiet=True)
        db = tmp_path / ".neo" / ci.INDEX_FILENAME
        db.write_bytes(b"this is not a database, it is a sentence" * 64)

        with ContentIndex(str(tmp_path)) as index:
            report = index.refresh(_walk(tmp_path), quiet=True)
            assert report.mode in ("cold", "rebuilt")
            assert index.scores("supersede")

    def test_a_truncated_store_is_rebuilt_not_raised(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            index.refresh(_walk(tmp_path), quiet=True)
        db = tmp_path / ".neo" / ci.INDEX_FILENAME
        raw = db.read_bytes()
        db.write_bytes(raw[: len(raw) // 3])
        with ContentIndex(str(tmp_path)) as index:
            report = index.refresh(_walk(tmp_path), quiet=True)
            assert report.mode in ("cold", "rebuilt")
            assert index.scores("supersede")

    def test_an_unwritable_store_still_scores_and_warns(self, tmp_path, monkeypatch):
        """A read-only checkout costs a warning, not a run.

        The alternative failure is worse than a crash: an empty ranking is
        indistinguishable, to the operator reading the output, from "nothing
        in your repository matches this prompt".
        """
        _repo(tmp_path, CORPUS)
        index = ContentIndex(str(tmp_path))
        monkeypatch.setattr(index, "_connect", lambda: None)
        report = index.refresh(_walk(tmp_path), quiet=True)
        assert report.mode == "memory"
        assert report.warning
        assert index.scores("supersede")
        assert "src/store.py" in index.scores("supersede")

    def test_the_memory_fallback_honours_a_candidate_subset(self, tmp_path):
        _repo(tmp_path, CORPUS)
        index = ContentIndex(str(tmp_path))
        index._connect = lambda: None
        index.refresh(_walk(tmp_path), quiet=True)
        assert set(index.scores("supersede", ["src/store.py"])) == {"src/store.py"}

    def test_an_empty_repository_scores_nothing_and_does_not_raise(self, tmp_path):
        with ContentIndex(str(tmp_path)) as index:
            report = index.refresh([], quiet=True)
            assert report.total_files == 0
            assert index.scores("anything") == {}


class TestReporting:
    """`--dry-run` has to be able to say which of the three happened."""

    def test_last_report_survives_the_index_being_closed(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            index.refresh(_walk(tmp_path), quiet=True)
        report = ci.last_report()
        assert report is not None
        assert report.mode == "cold"
        assert report.to_dict()["summary"] == report.describe()

    def test_the_report_serializes_for_json_consumers(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            report = index.refresh(_walk(tmp_path), quiet=True)
        payload = report.to_dict()
        assert payload["mode"] == "cold"
        assert payload["total_files"] == len(CORPUS)
        assert set(payload) >= {
            "mode",
            "total_files",
            "indexed",
            "added",
            "changed",
            "touched",
            "removed",
            "elapsed_ms",
            "summary",
        }

    def test_an_incremental_report_names_what_moved(self, tmp_path):
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            index.refresh(_walk(tmp_path), quiet=True)
        (tmp_path / "src" / "engine.py").write_text("def replaced():\n    pass\n")
        (tmp_path / "src" / "new.py").write_text("def added():\n    pass\n")
        os.unlink(tmp_path / "README.md")
        with ContentIndex(str(tmp_path)) as index:
            report = index.refresh(_walk(tmp_path), quiet=True)
        summary = report.describe()
        assert "1 added" in summary
        assert "1 changed" in summary
        assert "1 removed" in summary

    def test_a_cold_build_announces_itself_before_it_starts(self, tmp_path, capsys):
        """Silence for minutes is indistinguishable from a hang."""
        _repo(tmp_path, CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            index.refresh(_walk(tmp_path))
        err = capsys.readouterr().err
        assert "no usable index for this repository" in err
        assert "runs once" in err


class TestNoSecondWalker:
    """The index is handed eligibility; it never computes it."""

    def test_the_module_does_not_walk_the_filesystem(self):
        source = (
            __import__("pathlib")
            .Path(ci.__file__)
            .read_text()
        )
        for forbidden in ("os.walk", "rglob", "iglob", "glob.glob"):
            assert forbidden not in source, (
                f"{forbidden} in content_index.py: eligibility has exactly one "
                "implementation and it is neo.eligibility"
            )

    def test_it_accepts_the_walkers_own_records(self, tmp_path):
        _repo(tmp_path, CORPUS)
        paths = _walk(tmp_path)
        assert all(isinstance(p, eligibility.EligiblePath) for p in paths)
        with ContentIndex(str(tmp_path)) as index:
            assert index.refresh(paths, quiet=True).total_files == len(CORPUS)

    def test_it_also_accepts_bare_tuples(self, tmp_path):
        """Older callers hold `(abs, rel, size)`; they get real freshness."""
        _repo(tmp_path, CORPUS)
        tuples = [(p.path, p.rel_path, p.size) for p in _walk(tmp_path)]
        with ContentIndex(str(tmp_path)) as index:
            assert index.refresh(tuples, quiet=True).total_files == len(CORPUS)
        with ContentIndex(str(tmp_path)) as index:
            assert index.refresh(tuples, quiet=True).mode == "warm"


class TestWalkerCarriesMtime:
    """The stat the walk already does now yields the freshness stamp too."""

    def test_eligible_paths_carry_mtime(self, tmp_path):
        _repo(tmp_path, CORPUS)
        for entry in _walk(tmp_path):
            assert entry.mtime_ns > 0
            assert entry.mtime_ns == os.stat(entry.path).st_mtime_ns
