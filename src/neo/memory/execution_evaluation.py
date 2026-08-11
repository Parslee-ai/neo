"""Deterministic safety benchmark for proof-aware agent-loop decisions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from neo.execution_context import (
    assess_loop,
    assess_validation,
    execution_fields_from_dict,
    resolve_execution_context,
)
from neo.memory.episodes import LearningEpisode
from neo.models import NeoInput


@dataclass
class ExecutionScenarioResult:
    id: str
    passed: bool
    actual: dict[str, Any] = field(default_factory=dict)
    expected: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class ExecutionEvaluationReport:
    benchmark_id: str
    accepted: bool
    scenarios: list[ExecutionScenarioResult]
    latency_ms: float
    latency_budget_ms: float
    model_calls: int = 0
    token_usage: int = 0
    acceptance_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_corpus_path() -> Path:
    source = Path(__file__).resolve().parents[3] / "benchmarks" / "execution_loop_v1.json"
    installed = Path(__file__).resolve().parents[1] / "benchmarks" / "execution_loop_v1.json"
    return source if source.exists() else installed


def _evaluate_scenario(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("kind", "assessment"))
    payload = item.get("input", {})
    if kind == "legacy_episode":
        episode = LearningEpisode.from_dict(payload)
        return {
            "schema_version": episode.schema_version,
            "verification_count": len(episode.verification),
        }

    fields = execution_fields_from_dict(payload)
    context = resolve_execution_context(NeoInput(
        prompt=str(payload.get("prompt", "evaluate")),
        **fields,
    ))
    if kind == "identity":
        identity = context.execution_identity
        return {
            "task_id": identity.task_id,
            "parent_task_id": identity.parent_task_id,
            "repository_count": len(identity.repositories_touched),
        }
    if kind == "hypothesis":
        hypothesis = context.hypotheses[0] if context.hypotheses else None
        return {
            "hypothesis_status": hypothesis.status if hypothesis else "missing",
            "public_claim_safe": hypothesis.public_claim_safe if hypothesis else False,
        }

    goal, strategy = assess_loop(context)
    validation = assess_validation(context)
    return {
        "goal_status": goal.status,
        "strategy": strategy.decision,
        "required": validation.required,
        "passed": validation.passed,
        "failed": validation.failed,
        "pending": validation.pending,
        "unavailable": validation.unavailable,
        "waived": validation.waived,
        "stale": len(validation.stale_gate_ids),
    }


def run_execution_evaluation(
    corpus_path: Optional[Path] = None,
) -> ExecutionEvaluationReport:
    """Run every corpus case twice and require identical zero-model results."""
    path = corpus_path or _default_corpus_path()
    corpus = json.loads(path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    results: list[ExecutionScenarioResult] = []
    failures: list[str] = []
    for item in corpus.get("scenarios", []):
        scenario_id = str(item.get("id", "unnamed"))
        expected = dict(item.get("expected", {}))
        try:
            first = _evaluate_scenario(item)
            second = _evaluate_scenario(item)
            deterministic = first == second
            matched = all(first.get(key) == value for key, value in expected.items())
            passed = deterministic and matched
            error = "" if deterministic else "non-deterministic replay"
        except Exception as exc:  # benchmark reports failures rather than aborting
            first = {}
            passed = False
            error = f"{type(exc).__name__}: {exc}"
        if not passed:
            failures.append(scenario_id)
        results.append(ExecutionScenarioResult(
            id=scenario_id,
            passed=passed,
            actual=first,
            expected=expected,
            error=error,
        ))
    latency_ms = (time.perf_counter() - started) * 1000.0
    budget = float(corpus.get("latency_budget_ms", 500))
    if latency_ms > budget:
        failures.append(f"latency {latency_ms:.1f}ms exceeds {budget:.1f}ms")
    return ExecutionEvaluationReport(
        benchmark_id=str(corpus.get("benchmark_id", "execution_loop_v1")),
        accepted=not failures,
        scenarios=results,
        latency_ms=latency_ms,
        latency_budget_ms=budget,
        acceptance_failures=failures,
    )
