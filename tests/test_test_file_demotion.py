"""Test files must not outrank the code they test.

A test file's name is a strict SUPERSET of its subject's filename tokens:
`test_car_adapter.py` matches {adapter, car} where `adapters.py` matches
{adapter}. On the keyword path it can therefore only ever score at least as
high as the code it tests -- and then the implementation takes a size penalty
on top, because implementations are large. Measured on a real prompt, "add
retry logic to the CAR adapter" put `src/neo/adapters.py` at rank 14, below
nine test and doc files, split into two chunks.

`TEST_PENALTY` already existed and already fixed this -- on the ProjectIndex
semantic-boost path only. The two scoring paths disagreed about test files and
only one of them was ever corrected, which is the same duplicated-rule shape
that let `EXCLUDED_DIR_NAMES` hide files from one subsystem while the other
was fixed. There is now one constant and one predicate, used by both.

Measured over 12 code prompts, mean of the first source file's rank and of how
many of the top 10 slots went to tests and docs:

    baseline           first-src 4.50   tests+docs 5.58
    with the penalty   first-src 1.75   tests+docs 4.41

Doc-seeking prompts improve too (2.16 -> 3.00 docs in the top 10): demoting
tests leaves room rather than competing with documentation.
"""

import os

import pytest

from neo.context_gatherer import (
    TEST_PENALTY,
    is_test_path,
    prompt_targets_tests,
    score_candidate,
)

_EMPTY: set = set()
_ENTRY = {"main", "app", "server", "index", "login", "auth", "__init__"}


def _score(path, tokens, *, demote):
    return score_candidate(path, 4000, tokens, _EMPTY, _ENTRY, demote_tests=demote)


class TestIsTestPath:
    @pytest.mark.parametrize("path", [
        "tests/test_engine.py",
        "test_engine.py",
        "src/pkg/tests/test_x.py",
        "tests/subdir/helper.py",
    ])
    def test_recognised(self, path):
        assert is_test_path(path.replace("/", os.sep) if os.sep != "/" else path)

    @pytest.mark.parametrize("path", [
        "src/neo/adapters.py",
        "src/neo/testing_utils.py",   # not a test file; a module about testing
        "docs/testing.md",
        "src/latest.py",
    ])
    def test_not_recognised(self, path):
        assert not is_test_path(path)


class TestPromptTargetsTests:
    """The escape hatch: when the prompt IS about testing, test files are the
    answer and must keep full score."""

    @pytest.mark.parametrize("prompt", [
        "why does test_fact_store fail",
        "add unit tests for the parser",
        "the testing harness is slow",
        "run the rspec specs",
        "check the spec file",
    ])
    def test_fires_on_real_test_prompts(self, prompt):
        assert prompt_targets_tests(prompt)

    @pytest.mark.parametrize("prompt", [
        "the A2UI inspector shows a stale fact count",
        "make the query more specific",
        "their respective owners",
        "the latest commit broke it",
        "one aspect of the design",
        "add retry logic to the CAR adapter",
    ])
    def test_does_not_fire_on_substring_lookalikes(self, prompt):
        """The bug this predicate had, and why it mattered more after hoisting.

        The original was `any(t in prompt.lower() for t in (..., "spec"))`, so
        "in-spec-tor" classified a prompt about the A2UI inspector as a prompt
        about testing, and every test file kept full score. That was survivable
        while the flag only softened a cosine boost; it stopped being
        survivable once the same flag gated the keyword path. Measured: that
        one prompt moved the 12-prompt mean first-source rank from 1.75 to
        2.83 on its own.

        Same defect class as `'a'` matching 49 of 85 filenames -- a substring
        test on a short token.
        """
        assert not prompt_targets_tests(prompt)


class TestScoringDemotion:
    _TOKENS = {"adapter", "car"}

    def test_a_test_file_no_longer_outranks_its_subject(self):
        """The measured failure, reduced to its mechanism.

        `test_car_adapter.py` matches both prompt tokens; `adapters.py`
        matches one. Without the penalty the test file wins on filename
        overlap alone, before any size penalty is applied to the
        implementation.
        """
        impl = _score("src/neo/adapters.py", self._TOKENS, demote=True)
        test = _score("tests/test_car_adapter.py", self._TOKENS, demote=True)

        assert test < impl, (
            f"test file scored {test:.2f} vs implementation {impl:.2f}"
        )

    def test_without_demotion_the_test_file_still_wins(self):
        """Pins the mechanism rather than the fix, so this file documents WHY
        the penalty exists and not merely that it is applied."""
        impl = _score("src/neo/adapters.py", self._TOKENS, demote=False)
        test = _score("tests/test_car_adapter.py", self._TOKENS, demote=False)

        assert test > impl

    def test_the_penalty_scales_the_bonuses_not_the_final_score(self):
        """Placement is deliberate: the multiplier lands after the bonuses and
        BEFORE the additive depth and size penalties.

        So it is not `final * TEST_PENALTY`. Scaling the final score would
        shrink the penalties too, making a deep or oversized test file
        cheaper than a shallow one -- the wrong direction. Scaling only the
        positive signal is also what the ProjectIndex path does, where the
        multiplier applies to a cosine similarity before it becomes a boost.

        Measured for `tests/test_car_adapter.py` at 4KB and depth 1:
        bonuses 1.20, depth penalty 0.05, so plain = 1.15 and demoted =
        1.20 * 0.4 - 0.05 = 0.43.
        """
        depth_penalty = 0.05  # one separator in "tests/test_car_adapter.py"
        plain = _score("tests/test_car_adapter.py", self._TOKENS, demote=False)
        demoted = _score("tests/test_car_adapter.py", self._TOKENS, demote=True)

        bonuses = plain + depth_penalty
        assert demoted == pytest.approx(bonuses * TEST_PENALTY - depth_penalty)
        assert demoted < plain

    def test_non_test_files_are_untouched(self):
        for path in ("src/neo/adapters.py", "docs/architecture.md", "README.md"):
            assert (_score(path, self._TOKENS, demote=True)
                    == _score(path, self._TOKENS, demote=False))

    def test_demotion_is_off_by_default(self):
        """The existing pure-scoring tests call `score_candidate` positionally
        and must keep their meaning."""
        assert (score_candidate("tests/test_x.py", 4000, self._TOKENS, _EMPTY, _ENTRY)
                == _score("tests/test_x.py", self._TOKENS, demote=False))


class TestBothPathsShareOneRule:
    def test_the_semantic_path_uses_the_module_level_predicate(self):
        """`gather_context`'s ProjectIndex boost had its own inline copy of
        both the constant and the is-a-test check. A second copy is how the
        two paths came to disagree in the first place, so the source is
        grepped rather than trusted.
        """
        import pathlib

        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "src" / "neo" / "context_gatherer.py").read_text()

        assert "TEST_PENALTY = 0.4" in source
        assert source.count("TEST_PENALTY = 0.4") == 1, (
            "TEST_PENALTY is defined more than once -- the two scoring paths "
            "are free to drift again"
        )
        assert 'rel.startswith("tests"' not in source, (
            "the semantic path has re-grown its own is-a-test check instead "
            "of calling is_test_path"
        )
