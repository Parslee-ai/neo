"""A checker that evaluated nothing may not be reported as a checker that passed.

`_static_check_status` returns `skipped` for a result carrying only
informational diagnostics, which is what stops an unevaluable constraint check
from vouching for an early exit. Review found four consumers that never asked
it, each re-deriving "the check said something / the check ran" from the raw
list, and each turning #196's info note back into either a false alarm or false
assurance one layer down:

- the episode ledger recomputed the status inline as `warning if diagnostics`,
  and `aggregate_verification_status` ranks `warning` above `skipped`, so an
  unchecked-language note became the run's whole `verification_verdict`;
- the static-checks phase summary said "All 1 checker(s) clean";
- VERIFY's verdict said "nothing pushed back";
- and the "No checkers on this machine. This code is unverified." caution was
  suppressed, because the list was not empty.

The last is the worst of the four: a true caution dropped, not a false one added.
"""

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


def _info_only() -> StaticCheckResult:
    """What the constraint checker returns for an unmapped language."""
    return StaticCheckResult(
        tool_name="constraint_verifier",
        diagnostics=[{
            "severity": "info",
            "message": (
                "Constraint 'Elements must be unique' was not checked: no go "
                "marker set for constraint type 'unique_elements'"
            ),
            "constraint_type": "unique_elements",
            "file_path": "src/a.go",
            "language": "go",
        }],
        summary="1 not checked (go)",
    )


def _real_warning() -> StaticCheckResult:
    return StaticCheckResult(
        tool_name="constraint_verifier",
        diagnostics=[{
            "severity": "warning",
            "message": "no obvious handler",
            "constraint_type": "unique_elements",
            "file_path": "src/a.py",
            "language": "python",
        }],
        summary="1 constraint(s) may not be handled",
    )


def _clean() -> StaticCheckResult:
    return StaticCheckResult(tool_name="ruff", diagnostics=[], summary="clean")


def test_an_info_only_check_reports_skipped():
    assert NeoEngine._static_check_status(_info_only()) == "skipped"
    assert NeoEngine._static_check_status(_real_warning()) == "warning"
    assert NeoEngine._static_check_status(_clean()) == "passed"


def test_checks_that_evaluated_drops_only_the_skipped_ones():
    checks = [_info_only(), _clean(), _real_warning()]
    evaluated = NeoEngine._checks_that_evaluated(checks)
    assert [c.tool_name for c in evaluated] == ["ruff", "constraint_verifier"]
    assert NeoEngine._checks_that_evaluated([_info_only()]) == []


def test_pyright_informational_severity_is_not_actionable():
    """`information` is pyright's spelling; `info` alone missed it."""
    assert NeoEngine._is_actionable_diagnostic({"severity": "information"}) is False
    assert NeoEngine._is_actionable_diagnostic({"severity": "info"}) is False
    assert NeoEngine._is_actionable_diagnostic({"severity": "warning"}) is True
    # An unrecognized severity still fails toward being surfaced.
    assert NeoEngine._is_actionable_diagnostic({"severity": "weird"}) is True
    assert NeoEngine._is_actionable_diagnostic({}) is True


def test_the_episode_ledger_records_skipped_not_warning(tmp_path):
    """The #196 false alarm arriving through the episode ledger."""
    from neo.memory.episodes import aggregate_verification_status

    from neo.memory.episodes import LearningEpisode

    engine = _engine()
    engine.dry_run = False
    engine.current_learning_episode = LearningEpisode(
        objective="dedupe the ids",
        repository_root=str(tmp_path),
    )
    metadata: dict = {}
    engine._complete_learning_episode(
        code_suggestions=[CodeSuggestion(
            file_path="src/a.go",
            unified_diff="",
            description="",
            confidence=0.9,
            code_block="func F(xs []int) []int { return xs }",
        )],
        static_checks=[_info_only()],
        reasoning_fact=None,
        simulation_facts=[],
        metadata=metadata,
    )
    statuses = [v.status for v in engine.current_learning_episode.verification]
    assert "warning" not in statuses, (
        "an info-only 'not checked for <language>' note was recorded as a "
        "warning; aggregate_verification_status ranks warning above skipped, "
        "so it becomes the run's whole verdict"
    )
    assert "skipped" in statuses
    assert metadata["verification_verdict"] != "warning"
    assert aggregate_verification_status(
        engine.current_learning_episode.verification
    ) != "warning"


def test_a_real_warning_still_reaches_the_episode_ledger(tmp_path):
    """The other direction: the fix must not mute genuine findings."""
    from neo.memory.episodes import LearningEpisode

    engine = _engine()
    engine.dry_run = False
    engine.current_learning_episode = LearningEpisode(
        objective="dedupe the ids",
        repository_root=str(tmp_path),
    )
    metadata: dict = {}
    engine._complete_learning_episode(
        code_suggestions=[CodeSuggestion(
            file_path="src/a.py",
            unified_diff="",
            description="",
            confidence=0.9,
            code_block="def f(xs):\n    return xs\n",
        )],
        static_checks=[_real_warning()],
        reasoning_fact=None,
        simulation_facts=[],
        metadata=metadata,
    )
    statuses = [v.status for v in engine.current_learning_episode.verification]
    assert "warning" in statuses
    assert metadata["verification_verdict"] == "warning"


class TestSkipReasonIsReportedHonestly:
    """Two different reasons to skip the checkers, two different sentences.

    Saying "out of time" when there was simply nothing to check reports a
    budget problem that did not happen and hides the real state: the model
    proposed no change, so there was nothing to verify. Observed on a live run
    that produced no diff and still announced "I ran out of time before the
    checkers."
    """

    def test_both_reasons_have_distinct_voice_lines(self):
        import yaml
        from pathlib import Path

        deck = yaml.safe_load(
            (
                Path(__file__).parent.parent
                / "src/neo/config/beats/neo_matrix.yaml"
            ).read_text()
        )
        lines = deck["orchestrator_voice"]["lines"]

        nothing = lines["phase_checks_nothing_to_check"]
        assert "time" not in nothing.lower(), (
            "the nothing-to-check message must not blame the clock"
        )
        # The out-of-time phrasing is GONE, not merely unused: with no time
        # gate on the checkers there is no state it could describe, and a
        # retired sentence sitting in the deck is one grep away from being
        # wired back up.
        assert "phase_checks_skipped" not in lines

    def test_the_user_facing_caution_does_not_blame_the_clock(self):
        """The FINAL caution line is a separate key from the phase summary, and
        it was the one users actually saw: "I ran out of time before the
        checkers." It kept saying that on runs where nothing was proposed to
        check — and now that the checkers have no time gate at all, no run can
        legitimately blame the clock."""
        import yaml
        from pathlib import Path

        deck = yaml.safe_load(
            (
                Path(__file__).parent.parent
                / "src/neo/config/beats/neo_matrix.yaml"
            ).read_text()
        )
        caution = deck["orchestrator_voice"]["lines"]["caution_unverified_skipped"]

        assert "time" not in caution.lower(), (
            f"caution still blames the clock: {caution!r}"
        )

    def test_static_checks_have_no_time_gate(self):
        """Bounded work does not need a budget guard. The old gate resolved to
        "run the checkers only if the whole run finished in 27 seconds", which
        for repo-scale work is never."""
        from pathlib import Path

        src = (Path(__file__).parent.parent / "src/neo/engine.py").read_text()
        assert "STATIC_CHECK_BUFFER" not in src
        assert "STATIC_CHECK_RUNAWAY_MULTIPLE" not in src

    def test_no_dead_ternary_on_the_skip_message(self):
        """The skip branch is reached only when have_changes is false, so a
        `if not have_changes` ternary inside it is a branch that cannot be
        taken. It was there, selecting a message that could never be emitted.
        """
        from pathlib import Path

        src = (
            Path(__file__).parent.parent / "src/neo/engine.py"
        ).read_text()

        assert "phase_checks_nothing_to_check" in src
        assert "phase_checks_skipped" not in src, (
            "retired voice key is referenced again"
        )


class TestConstraintExtractionFailureIsVisible:
    """"The prompt declared no constraints" and "the extractor broke" are
    different facts, and only one of them means there was nothing to verify.

    It was a DOUBLE swallow: extract_constraints catches Exception and returns
    [] itself, and the caller caught again at logger.debug — invisible at the
    default WARNING level. The empty list then met the `if constraints:` guard
    at the call site, no constraint check was appended, and the run presented
    itself as constraint-clean. Measured: a prompt yielding 2 constraints
    yields 0 when the extractor raises.
    """

    @staticmethod
    def _suggestion():
        from neo.models import CodeSuggestion
        return CodeSuggestion(
            file_path="x.py",
            unified_diff="--- a\n+++ b\n@@ -1 +1 @@\n-a\n+b\n",
            code_block="", description="", confidence=0.9,
        )

    def test_extractor_failure_produces_an_unavailable_check(self, capsys):
        from unittest.mock import patch

        import neo.constraint_verification as cv
        from neo.engine import NeoEngine

        class Boom:
            def extract_constraints(self, *a, **k):
                raise RuntimeError("extractor blew up")

        engine = NeoEngine.__new__(NeoEngine)
        with patch.object(cv, "ConstraintVerifier", Boom):
            constraints = NeoEngine._extract_prompt_constraints(
                engine, "the result must be non-negative"
            )
        assert constraints == []

        result = NeoEngine._check_constraints_static(
            engine, [self._suggestion()], constraints
        )
        assert result is not None, (
            "a failed extraction left no trace at all"
        )
        assert result.status == "unavailable"
        assert "NOT checked" in result.summary
        # And it must not count toward "N checker(s) clean".
        assert NeoEngine._checks_that_evaluated([result]) == []

    def test_a_prompt_with_no_constraints_stays_silent(self):
        """The other branch. No constraints declared genuinely means there was
        nothing to verify, and inventing a warning there would train operators
        to ignore the channel."""
        from neo.engine import NeoEngine

        engine = NeoEngine.__new__(NeoEngine)
        NeoEngine._extract_prompt_constraints(engine, "just do the thing")

        assert NeoEngine._check_constraints_static(
            engine, [self._suggestion()], []
        ) is None
