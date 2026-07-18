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

    assert corpus["schema_version"] == 1
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
    assert workspace.joinpath("facts").exists()
    assert not fake_home.joinpath(".neo", "facts").exists()
    assert not fake_home.joinpath(".neo", "metrics.jsonl").exists()
