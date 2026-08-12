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
import os
import subprocess
import sys
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

    The format string is EXTRACTED from `cli.py` and rendered here, rather than
    hand-written. A hand-written literal only proves the parser matches the
    test: change `- {gf.bytes} bytes` to `- {gf.bytes}b` in the CLI and a
    literal-based test stays green while every case parses to nothing.
    """

    def _cli_format_string(self):
        import inspect

        from neo import cli

        for line in inspect.getsource(cli).splitlines():
            if "bytes (score:" in line and "print(" in line:
                return line[line.index('f"'):].split('"')[1]
        pytest.fail("the dry-run selected-file print was not found in cli.py -- "
                    "the parser's contract has moved and this test is blind")

    def test_parser_matches_the_format_the_cli_actually_uses(self):
        fmt = self._cli_format_string()
        rendered = (fmt
                    .replace("{gf.rel_path}", "src/neo/store.py")
                    .replace("{lines_info}", " (lines 1-140)")
                    .replace("{gf.bytes}", "6183")
                    .replace("{gf.score:.2f}", "4.23"))

        assert rme._DRY_RUN_LINE.match(rendered), (
            f"parser no longer matches the CLI's line format {fmt!r} -- every "
            f"case would return [] and the run would report a confident 0.000"
        )

    def test_parser_matches_it_without_a_line_range_too(self):
        fmt = self._cli_format_string()
        rendered = (fmt
                    .replace("{gf.rel_path}", "src/neo/store.py")
                    .replace("{lines_info}", "")
                    .replace("{gf.bytes}", "6183")
                    .replace("{gf.score:.2f}", "4.23"))

        assert rme._DRY_RUN_LINE.match(rendered)

    def test_the_marker_the_parser_splits_on_still_exists(self):
        import inspect

        from neo import cli

        assert "=== DRY RUN" in inspect.getsource(cli), (
            "dry-run marker renamed; the parser fails closed on it, so every "
            "run would abort rather than mislead -- but fix the parser"
        )


class TestGuards:
    """The two guards added because each was a way to publish a wrong number.

    Verifying a guard by hand once is how a guard goes unpinned and looks
    pinned -- the same failure as naming a constant in a docstring while
    asserting only a direction.
    """

    def test_a_tree_without_src_neo_is_refused(self, tmp_path):
        with pytest.raises(SystemExit) as exc:
            rme.assert_tree_is_effective(str(tmp_path), str(tmp_path))

        assert "src/neo" in str(exc.value)

    def test_the_probe_runs_from_the_repo_the_measurement_runs_from(
            self, tmp_path, monkeypatch):
        """cwd lands at sys.path[0], AHEAD of PYTHONPATH.

        A probe run from anywhere else cannot see a repo-local `neo/` package
        shadowing the named tree, so the guard would pass while the run
        measured the shadow.
        """
        (tmp_path / "src" / "neo").mkdir(parents=True)
        repo = tmp_path / "somerepo"
        repo.mkdir()
        seen = {}

        def fake_run(*args, **kwargs):
            seen["cwd"] = kwargs.get("cwd")
            return subprocess.CompletedProcess(
                args, 0, stdout=str(tmp_path / "src" / "neo" / "__init__.py"))
        monkeypatch.setattr(rme.subprocess, "run", fake_run)

        rme.assert_tree_is_effective(str(tmp_path), str(repo))

        assert seen["cwd"] == str(repo)

    def test_a_tree_that_does_not_win_the_import_is_refused(self, tmp_path,
                                                            monkeypatch):
        """Structure alone is not proof: the editable .pth can still win."""
        (tmp_path / "src" / "neo").mkdir(parents=True)

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(
                args, 0, stdout="/somewhere/else/neo/__init__.py")
        monkeypatch.setattr(rme.subprocess, "run", fake_run)

        with pytest.raises(SystemExit) as exc:
            rme.assert_tree_is_effective(str(tmp_path), str(tmp_path))

        assert "did not take effect" in str(exc.value)

    def test_a_tree_that_wins_the_import_is_accepted(self, tmp_path, monkeypatch):
        (tmp_path / "src" / "neo").mkdir(parents=True)
        resolved = str(tmp_path / "src" / "neo" / "__init__.py")

        def fake_run(*args, **kwargs):
            return subprocess.CompletedProcess(args, 0, stdout=resolved)
        monkeypatch.setattr(rme.subprocess, "run", fake_run)

        rme.assert_tree_is_effective(str(tmp_path), str(tmp_path))   # no raise

    def test_relative_paths_are_absolutized_before_the_guard_sees_them(
            self, tmp_path, monkeypatch):
        """The headline fix of b45cc92, which was verified by hand only.

        `--repo x --tree .` resolved `.` against the harness's cwd while the
        measurement ran with `cwd=repo` -- guard green, wrong tree measured.
        """
        (tmp_path / "src" / "neo").mkdir(parents=True)
        (tmp_path / "repo").mkdir()
        monkeypatch.chdir(tmp_path)
        seen = {}

        def fake_guard(tree, repo):
            seen["tree"], seen["repo"] = tree, repo
            raise SystemExit("stop here -- only the paths are under test")
        monkeypatch.setattr(rme, "assert_tree_is_effective", fake_guard)
        monkeypatch.setattr(sys, "argv",
                            ["rank_mine_eval.py", "--repo", "repo", "--tree", "."])

        with pytest.raises(SystemExit):
            rme.main()

        assert os.path.isabs(seen["tree"]) and os.path.isabs(seen["repo"])
        assert seen["tree"] == str(tmp_path.resolve())


class TestNothingParsedIsAFailure:
    """Marker found, zero lines parsed anywhere = the format moved.

    Scored as a result it reads `R@10 0.000, failed_cases: 0` -- a confident
    claim that the ranker found nothing, when the parser found nothing.
    """

    def _run_main(self, monkeypatch, tmp_path, ranking):
        monkeypatch.setattr(rme, "assert_tree_is_effective", lambda t, r: None)
        monkeypatch.setattr(rme, "_git_head", lambda p: "deadbeefcafe")
        monkeypatch.setattr(rme, "mine_cases",
                            lambda *a, **k: [{"sha": "abc", "query": "q",
                                              "truth": ["a.py"]}])
        monkeypatch.setattr(rme, "rank_files", lambda *a, **k: ranking)
        monkeypatch.setattr(sys, "argv",
                            ["rank_mine_eval.py", "--repo", str(tmp_path),
                             "--tree", str(tmp_path)])
        return rme.main()

    def test_every_case_parsing_nothing_exits_nonzero(self, monkeypatch,
                                                      tmp_path, capsys):
        rc = self._run_main(monkeypatch, tmp_path, [])

        assert rc == 1
        assert "not one case yielded a parsable file line" in \
            capsys.readouterr().err.lower()

    def test_a_real_ranking_still_scores(self, monkeypatch, tmp_path):
        assert self._run_main(monkeypatch, tmp_path, ["a.py"]) == 0


class TestHeadStamp:
    def test_a_non_git_tree_stamps_instead_of_exploding(self, tmp_path):
        """`--tree` need not be a git checkout, and a prune'd worktree passes
        the effectiveness guard. Raising here discarded a finished 50-case run
        with a traceback pointing at a string format."""
        assert rme._git_head(str(tmp_path)) == "not-a-git-checkout"

    def test_a_real_checkout_stamps_a_short_sha(self):
        head = rme._git_head(str(Path(__file__).resolve().parent.parent))

        assert len(head) == 12   # the marker is longer, so this excludes it

    def test_a_non_git_dir_INSIDE_a_checkout_does_not_borrow_its_sha(self,
                                                                     tmp_path):
        """`rev-parse HEAD` searches ancestors, so this stamped the enclosing
        repo -- a confident wrong label on the field that says which ranker
        ran."""
        nested = Path(__file__).resolve().parent.parent / ".tmp_nongit_probe"
        nested.mkdir(exist_ok=True)
        try:
            assert rme._git_head(str(nested)) == "not-a-git-checkout"
        finally:
            nested.rmdir()
