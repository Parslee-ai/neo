"""Versioned zero-model acceptance gate for proof-aware loop decisions."""

import json
import os
from pathlib import Path
import subprocess
import sys

from neo.memory.execution_evaluation import run_execution_evaluation


def test_execution_evaluation_accepts_versioned_corpus():
    report = run_execution_evaluation()

    assert report.accepted, report.acceptance_failures
    assert report.model_calls == 0
    assert report.token_usage == 0
    assert report.latency_ms <= report.latency_budget_ms
    assert len(report.scenarios) >= 10
    assert all(item.passed for item in report.scenarios)


def test_execution_evaluation_cli_json_is_machine_readable():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-m", "neo", "memory", "evaluate-execution", "--json"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["accepted"] is True
    assert payload["model_calls"] == 0
    assert payload["token_usage"] == 0
