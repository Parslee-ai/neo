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
