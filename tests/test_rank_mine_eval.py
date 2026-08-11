"""Tests for `tools/rank_mine_eval.py`.

This harness produces the ranking figures quoted in CLAUDE.md and on the PR,
so a silent defect in it does not cause a crash -- it publishes a wrong number
with a confident table around it. That is the exact failure the file-selection
work exists to stop, and it has already happened twice here: the first draft
scored every case zero because it read stdout while the CLI writes to stderr,
and the second scored subprocess failures as perfect-zero retrieval.

What is pinned below is therefore the arithmetic and the fail-closed paths,
not the mining (which needs a real repository).
"""
import importlib.util
import subprocess
from pathlib import Path

import pytest

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "rank_mine_eval.py"
_spec = importlib.util.spec_from_file_location("rank_mine_eval", _TOOL)
rme = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rme)


class TestScore:
    def test_recall_and_hit_rate_answer_different_questions(self):
        """The divergence that makes reporting only one of them misleading."""
        cases = [{"truth": ["a.py", "b.py", "c.py", "d.py"]}]
        ranked = [["z.py", "b.py", "y.py"]]

        result = rme.score(cases, ranked, [3])

        # One of four truth files found: quarter the recall, but the run DID
        # surface something relevant, so the hit rate is 1.0.
        assert result["recall"]["R@3"] == pytest.approx(0.25)
        assert result["hit_rate"]["H@3"] == pytest.approx(1.0)
        assert result["MRR"] == pytest.approx(0.5)   # first hit at rank 2

    def test_mrr_uses_the_first_relevant_rank_only(self):
        cases = [{"truth": ["a.py", "b.py"]}]
        ranked = [["a.py", "b.py"]]

        assert rme.score(cases, ranked, [10])["MRR"] == pytest.approx(1.0)

    def test_a_miss_scores_zero_rather_than_erroring(self):
        result = rme.score([{"truth": ["a.py"]}], [["x.py", "y.py"]], [1, 10])

        assert result["recall"]["R@10"] == 0.0
        assert result["hit_rate"]["H@10"] == 0.0
        assert result["MRR"] == 0.0

    def test_k_truncates_the_ranking(self):
        cases = [{"truth": ["a.py"]}]
        ranked = [["x.py", "y.py", "z.py", "a.py"]]

        result = rme.score(cases, ranked, [1, 3, 10])

        assert result["recall"]["R@1"] == 0.0
        assert result["recall"]["R@3"] == 0.0
        assert result["recall"]["R@10"] == pytest.approx(1.0)


class TestRankFilesFailsClosed:
    """None means the run FAILED; [] means it completed and chose nothing.

    Collapsing the two is what let a timeout average into a table as evidence
    about ranking quality.
    """

    def _stub(self, monkeypatch, *, returncode=0, stdout="", raises=None):
        def fake_run(*args, **kwargs):
            if raises:
                raise raises
            return subprocess.CompletedProcess(args, returncode, stdout=stdout)
        monkeypatch.setattr(rme.subprocess, "run", fake_run)

    def test_non_zero_exit_is_a_failure_not_an_empty_ranking(self, monkeypatch):
        self._stub(monkeypatch, returncode=1, stdout="=== DRY RUN ===\n")

        assert rme.rank_files("/repo", "/tree", "q", 10) is None

    def test_missing_marker_is_a_failure(self, monkeypatch):
        """Format drift must not read as 'nothing matched'."""
        self._stub(monkeypatch, returncode=0, stdout="some other output\n")

        assert rme.rank_files("/repo", "/tree", "q", 10) is None

    def test_timeout_is_a_failure(self, monkeypatch):
        self._stub(monkeypatch, raises=subprocess.TimeoutExpired("cmd", 1))

        assert rme.rank_files("/repo", "/tree", "q", 10) is None

    def test_parses_ranking_and_collapses_chunks_of_one_file(self, monkeypatch):
        self._stub(monkeypatch, stdout=(
            "=== DRY RUN: Context that would be sent ===\n\n"
            "  src/a.py (lines 1-140) - 6183 bytes (score: 4.23)\n"
            "  src/a.py (lines 141-329) - 8467 bytes (score: 4.23)\n"
            "  src/b.py - 3967 bytes (score: 3.72)\n"
        ))

        assert rme.rank_files("/repo", "/tree", "q", 10) == ["src/a.py", "src/b.py"]

    def test_git_recency_is_disabled_unless_asked_for(self, monkeypatch):
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, stdout="=== DRY RUN ===\n")
        monkeypatch.setattr(rme.subprocess, "run", fake_run)

        rme.rank_files("/repo", "/tree", "q", 10)
        assert "--no-git" in seen["cmd"]

        rme.rank_files("/repo", "/tree", "q", 10, use_git=True)
        assert "--no-git" not in seen["cmd"]


class TestSubjectFiltering:
    def test_conventional_prefix_is_stripped_before_the_skip_test(self):
        """`chore: bump version to 0.39.0` was mined as a real case."""
        subject = "chore: bump version to 0.39.0"
        stripped = rme._CONVENTIONAL_PREFIX_RE.sub("", subject)

        assert rme._SKIP_SUBJECT_RE.match(stripped)
        assert not rme._SKIP_SUBJECT_RE.match(subject)   # why the strip is needed

    def test_real_work_survives_the_filter(self):
        subject = "fix(gatherer): select files by content, not by filename"
        stripped = rme._CONVENTIONAL_PREFIX_RE.sub("", subject)

        assert not rme._SKIP_SUBJECT_RE.match(stripped)

    @pytest.mark.parametrize("path", [
        "tests/test_store.py", "src/__tests__/a.ts", "a/b_test.go",
        "src/foo.spec.ts", "spec/models/user.rb",
    ])
    def test_test_files_are_not_ground_truth(self, path):
        """They are demoted by design; counting them would score that as a bug."""
        assert rme._TEST_RE.search(path)

    @pytest.mark.parametrize("path", [
        "src/neo/context_gatherer.py", "src/latest/protest.py", "lib/attestation.cs",
    ])
    def test_source_files_are_not_mistaken_for_tests(self, path):
        assert not rme._TEST_RE.search(path)
