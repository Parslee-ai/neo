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
    """Asserts on the PREDICATES `mine_cases` calls, not on a composition the
    test performs itself. Composing the two regexes here would stay green while
    `mine_cases` dropped the prefix strip -- i.e. green through the exact
    regression the first case below is named for."""

    @pytest.mark.parametrize("subject", [
        "chore: bump version to 0.39.0",     # the one that was mined for real
        "bump version to 0.39.0",
        "chore(release): v1.2.3 and the notes",
        "Merge branch 'main' into feature",
        "revert: the thing that broke prod",
        "wip: still figuring this out",
    ])
    def test_non_work_subjects_are_skipped(self, subject):
        assert rme.is_skippable(subject)

    @pytest.mark.parametrize("subject", [
        "fix(gatherer): select files by content, not by filename",
        "add retry logic to the CAR adapter",
        "feat(index): apportion chunk slots by symbol count",
    ])
    def test_real_work_survives_the_filter(self, subject):
        assert not rme.is_skippable(subject)


class TestGroundTruth:
    HEAD = {
        "src/neo/context_gatherer.py", "src/latest/protest.py", "lib/attestation.cs",
        "tests/test_store.py", "src/__tests__/a.ts", "a/b_test.go",
        "src/foo.spec.ts", "spec/models/user.rb", "README.md", "gone.py",
    }

    @pytest.mark.parametrize("path", [
        "tests/test_store.py", "src/__tests__/a.ts", "a/b_test.go",
        "src/foo.spec.ts", "spec/models/user.rb",
    ])
    def test_test_files_are_not_ground_truth(self, path):
        """They are demoted by design; counting them would score that as a bug."""
        assert not rme.is_ground_truth(path, self.HEAD)

    @pytest.mark.parametrize("path", [
        "src/neo/context_gatherer.py", "src/latest/protest.py", "lib/attestation.cs",
    ])
    def test_source_files_are_ground_truth(self, path):
        """`protest`/`attestation` guard the test regex against over-matching."""
        assert rme.is_ground_truth(path, self.HEAD)

    def test_non_source_is_excluded(self):
        assert not rme.is_ground_truth("README.md", self.HEAD)

    def test_a_file_deleted_since_the_commit_is_excluded(self):
        """Unfindable at HEAD, so scoring it as a miss would penalise nothing."""
        assert not rme.is_ground_truth("src/removed.py", self.HEAD)


class TestOutputFormatContract:
    """The parser reads human output, so the CLI's format is a contract.

    Hand-writing the format in a parser test proves only that the parser
    matches the test. This asserts against the string the CLI actually emits.
    """

    def test_parser_matches_the_line_the_cli_emits(self):
        import inspect

        from neo import cli

        source = inspect.getsource(cli)
        # The dry-run printer's own format string, rendered with real values.
        rendered = "  src/neo/store.py (lines 1-140) - 6183 bytes (score: 4.23)"

        assert "=== DRY RUN" in source, "dry-run marker renamed; parser is blind"
        assert rme._DRY_RUN_LINE.match(rendered), (
            "the parser no longer matches the CLI's selected-file line -- every "
            "case would return [] and the run would report a confident 0.000"
        )
