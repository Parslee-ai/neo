"""Deterministic evidence-learning benchmark and acceptance-gate tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

from neo.memory.evaluation import (
    EvaluationMode,
    LearningLoopEvaluator,
    load_corpus,
    run_learning_evaluation,
)


def test_benchmark_passes_all_required_scenarios_and_safety_gates(tmp_path):
    report = run_learning_evaluation(workspace=tmp_path / "evaluation")

    assert report.accepted is True
    assert report.acceptance_failures == []
    evidence = next(
        mode for mode in report.modes if mode.mode == EvaluationMode.EVIDENCE.value
    )
    assert len(evidence.scenarios) == 12
    assert all(scenario.passed for scenario in evidence.scenarios)
    assert evidence.metrics.harmful_memory_rate == 0.0
    assert evidence.metrics.unsupported_promotion_rate == 0.0
    assert evidence.metrics.repeat_error_rate == 0.0
    assert evidence.metrics.project_leakage_rate == 0.0
    assert evidence.metrics.model_calls == 0
    assert evidence.metrics.token_usage == 0


def test_evidence_mode_improves_quality_without_legacy_harm(tmp_path):
    report = run_learning_evaluation(workspace=tmp_path / "comparison")
    modes = {mode.mode: mode for mode in report.modes}
    baseline = modes[EvaluationMode.DISABLED.value].metrics
    legacy = modes[EvaluationMode.LEGACY.value].metrics
    evidence = modes[EvaluationMode.EVIDENCE.value].metrics

    assert evidence.task_success_rate > baseline.task_success_rate
    assert evidence.retrieval_precision > baseline.retrieval_precision
    assert evidence.harmful_memory_rate < legacy.harmful_memory_rate
    assert evidence.unsupported_promotion_rate < legacy.unsupported_promotion_rate


def test_safety_threshold_violation_fails_acceptance(tmp_path):
    corpus = load_corpus()
    corpus["safety_thresholds"]["harmful_memory_rate_max"] = -0.01

    report = LearningLoopEvaluator(corpus, workspace=tmp_path / "failed-gate").run()

    assert report.accepted is False
    assert any("harmful_memory_rate" in failure for failure in report.acceptance_failures)


class TestLatencyIsNotACorrectnessGate:
    """A slow machine must not invalidate a correctness verdict (#183).

    `accepted` used to include a wall-clock budget, so a GitHub runner
    recording 592.44ms against a 500ms limit failed the benchmark with all
    twelve scenarios passing and every safety rate at zero — while the same
    commit ran at ~53ms locally. An 11x spread with no code difference means
    no threshold can be both sensitive enough to catch a regression and loose
    enough to survive a shared runner, so the verdicts are separated rather
    than the number retuned.

    The budget is set to something no machine can meet rather than mocking the
    clock: the point is the harness's response to a real overrun, and a
    patched timer would test the patch.
    """

    def _over_budget(self, tmp_path, name):
        corpus = load_corpus()
        corpus["performance_budget"]["latency_ms_max"] = 0.001
        return LearningLoopEvaluator(corpus, workspace=tmp_path / name).run()

    def test_overrun_does_not_fail_acceptance(self, tmp_path):
        report = self._over_budget(tmp_path, "slow")

        assert report.accepted is True
        assert report.acceptance_failures == []

    def test_overrun_is_still_reported(self, tmp_path):
        """Not gating is not the same as not measuring."""
        report = self._over_budget(tmp_path, "slow-reported")

        assert report.performance_within_budget is False
        assert any("latency_ms" in note for note in report.performance_notes)

    def test_no_latency_text_leaks_into_acceptance_failures(self, tmp_path):
        """The two lists must not blur back together.

        Cheap to assert and the thing a future edit would get wrong: appending
        the note to `failures` alongside `performance_notes` would keep both
        tests above green while restoring the defect.
        """
        report = self._over_budget(tmp_path, "slow-separate")

        assert not any("latency" in failure for failure in report.acceptance_failures)

    def _within_budget(self, tmp_path, name):
        """Run against a budget no machine can exceed.

        Mirrors `_over_budget` in the other direction and for the same reason.
        Asserting that the AMBIENT machine comes in under the real 500ms budget
        is the coin-flip this class exists to document: CI measured 534.80ms on
        a commit that runs at ~50ms locally, with all twelve scenarios passing
        and every safety rate at zero. That is the same 11x shared-runner
        spread that moved latency out of `accepted` in the first place —
        reappearing one level up, as a flaky test that fails a correctness PR
        for a reason that has nothing to do with correctness.
        """
        corpus = load_corpus()
        corpus["performance_budget"]["latency_ms_max"] = 1_000_000.0
        return LearningLoopEvaluator(corpus, workspace=tmp_path / name).run()

    def test_within_budget_run_reports_clean(self, tmp_path):
        report = self._within_budget(tmp_path, "fast")

        assert report.performance_within_budget is True
        assert report.performance_notes == []

    def test_latency_is_still_measured(self, tmp_path):
        """Removing the gate must not quietly remove the metric."""
        report = run_learning_evaluation(workspace=tmp_path / "measured")
        evidence = next(
            mode for mode in report.modes if mode.mode == EvaluationMode.EVIDENCE.value
        )

        assert evidence.metrics.latency_ms > 0

    def test_safety_thresholds_no_longer_carry_a_wall_clock_number(self):
        """Every remaining safety threshold is a deterministic rate.

        The category error was structural, not just a stray branch: latency
        sat in the same block as four rates that evaluate to exactly 0.0 on
        any machine. Keeping it out is what stops the gate being rebuilt by
        someone reading the corpus and assuming everything there is enforced.
        """
        corpus = load_corpus()

        assert "latency_ms_max" not in corpus["safety_thresholds"]
        assert corpus["performance_budget"]["latency_ms_max"] > 0

    def test_schema_1_corpus_still_loads_and_reads_its_budget(self, tmp_path):
        """`--corpus` lets a caller supply their own file; a moved key is a
        gratuitous reason to break it."""
        corpus = load_corpus()
        legacy = {
            **corpus,
            "schema_version": 1,
            "safety_thresholds": {
                **corpus["safety_thresholds"],
                "latency_ms_max": 0.001,
            },
        }
        legacy.pop("performance_budget")
        path = tmp_path / "legacy_corpus.json"
        path.write_text(json.dumps(legacy))

        loaded = load_corpus(path)
        report = LearningLoopEvaluator(loaded, workspace=tmp_path / "legacy").run()

        assert loaded["schema_version"] == 1
        assert report.accepted is True
        assert report.performance_within_budget is False


def test_benchmark_ranking_evidence_is_repeatable(tmp_path):
    first = run_learning_evaluation(workspace=tmp_path / "first")
    second = run_learning_evaluation(workspace=tmp_path / "second")

    def ranking(report):
        evidence = report.modes[-1]
        scenario = next(
            item for item in evidence.scenarios if item.id == "ranking_is_deterministic"
        )
        # Fact IDs are intentionally random, so compare the within-run causal
        # invariant rather than IDs across independent benchmark executions.
        return scenario.passed, scenario.evidence["first"] == scenario.evidence["second"]

    assert ranking(first) == (True, True)
    assert ranking(second) == (True, True)


def test_corpus_is_versioned_and_contains_repeated_task_families():
    corpus = load_corpus()

    assert corpus["schema_version"] == 2
    assert corpus["benchmark_id"] == "neo-evidence-learning-v1"
    assert len(corpus["task_families"]) >= 3
    assert any(
        family["training_outcomes"].count("accepted") >= 2
        for family in corpus["task_families"]
    )
    assert sum(
        family["expected_later_behavior"] == "retrieve"
        for family in corpus["task_families"]
    ) >= 2


def test_evaluation_cli_runs_without_provider_keys_or_user_memory(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    workspace = tmp_path / "retained-evaluation"
    env = {
        **os.environ,
        "HOME": str(fake_home),
        "PYTHONPATH": str(repository / "src"),
        "NEO_SKIP_UPDATE_CHECK": "1",
        "NEO_OBSERVER_AUTOSTART": "0",
    }
    for key in ("NEO_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        env.pop(key, None)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "neo",
            "memory",
            "evaluate-learning",
            "--json",
            "--workspace",
            str(workspace),
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["accepted"] is True
    assert payload["benchmark_id"] == "neo-evidence-learning-v1"
    assert all(mode["metrics"]["model_calls"] == 0 for mode in payload["modes"])
    assert payload["performance_within_budget"] is True
    assert workspace.joinpath("facts").exists()
    assert not fake_home.joinpath(".neo", "facts").exists()
    assert not fake_home.joinpath(".neo", "metrics.jsonl").exists()


def test_cli_exits_zero_when_only_the_latency_budget_is_missed(tmp_path):
    """The user-visible half of #183.

    A separated verdict is worth nothing if the command still exits 1 — CI
    reads the exit code, not the report. Driven through the real CLI with a
    corpus whose budget no machine can meet, so this covers the wiring between
    the report and `SystemExit` rather than the report alone.
    """
    repository = Path(__file__).resolve().parents[1]
    corpus = load_corpus()
    corpus["performance_budget"]["latency_ms_max"] = 0.001
    corpus_path = tmp_path / "impossible_budget.json"
    corpus_path.write_text(json.dumps(corpus))

    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    env = {
        **os.environ,
        "HOME": str(fake_home),
        "PYTHONPATH": str(repository / "src"),
        "NEO_SKIP_UPDATE_CHECK": "1",
        "NEO_OBSERVER_AUTOSTART": "0",
    }
    for key in ("NEO_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        env.pop(key, None)

    result = subprocess.run(
        [
            sys.executable, "-m", "neo", "memory", "evaluate-learning",
            "--corpus", str(corpus_path),
            "--workspace", str(tmp_path / "slow-cli"),
        ],
        capture_output=True, text=True, env=env, timeout=60,
    )

    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "PASS" in output
    # Reported, and worded so it cannot be read as a failed gate.
    assert "advisory" in output
    assert "latency_ms" in output
    assert "failure:" not in output
