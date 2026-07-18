"""Operating-mode authority and lifecycle tests."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from neo.engine import NeoEngine
from neo.models import (
    AppliedAction,
    NeoInput,
    ProposedChange,
    StaticCheckResult,
)
from neo.operating_mode import (
    AuthorityPolicy,
    ModeValidationError,
    OperatingMode,
)


class _ModeLM:
    provider = "test"
    model = "mode-model"

    def __init__(self):
        self.calls = 0

    def name(self):
        return "test/mode-model"

    def generate(self, messages, **kwargs):
        self.calls += 1
        return """<<<NEO:SCHEMA=v3:KIND=plan>>>
[{"id":"ps_1","description":"change it","rationale":"requested","dependencies":[],"schema_version":"3"}]
<<<END:plan>>>
<<<NEO:SCHEMA=v3:KIND=simulation>>>
[{"n":1,"input_data":"x","expected_output":"y","reasoning_steps":["**NO_MODIFY**"],"issues_found":[],"schema_version":"3"}]
<<<END:simulation>>>
<<<NEO:SCHEMA=v3:KIND=code>>>
[{"file_path":"src/example.py","unified_diff":"+value = 1","code_block":"value = 1","description":"set value","confidence":0.8,"tradeoffs":[],"schema_version":"3"}]
<<<END:code>>>"""


class _Executor:
    def __init__(self):
        self.calls = 0

    def execute(self, suggestions, policy):
        self.calls += 1
        return [AppliedAction(
            suggestion_id=suggestions[0].suggestion_id,
            file_path=suggestions[0].file_path,
            status="applied",
            summary="host applied authorized patch",
            repository_revision="revision-after",
        )]


def _engine(tmp_path, *, executor=None, memory=False):
    with patch("neo.memory.store.FactStore.initialize"):
        engine = NeoEngine(
            lm_adapter=_ModeLM(),
            enable_persistent_memory=memory,
            codebase_root=str(tmp_path),
            execution_adapter=executor,
        )
    engine._car_route_capability = lambda prompt: (False, 0, None)
    engine._run_static_checks = lambda suggestions, constraints=None: []
    return engine


def test_default_mode_preserves_backward_compatible_learning_behavior():
    request = NeoInput(prompt="help")

    assert request.operating_mode is OperatingMode.LEARN


def test_advise_records_episode_but_never_creates_learning_candidate(tmp_path):
    engine = _engine(tmp_path, memory=True)

    output = engine.process(NeoInput(
        prompt="advise me",
        operating_mode=OperatingMode.ADVISE,
    ))

    episode = engine.episode_store.load(output.metadata["learning_episode_id"])
    assert output.metadata["operating_mode"] == "advise"
    assert output.metadata["learning_enabled_for_request"] is False
    assert episode.operating_mode == "advise"
    assert episode.final_outcome == "advised_no_learning"
    assert episode.memory_candidates == []


def test_verify_mode_uses_caller_change_and_never_calls_lm(tmp_path):
    engine = _engine(tmp_path)
    engine._run_static_checks = lambda suggestions, constraints=None: [StaticCheckResult(
        tool_name="parser",
        diagnostics=[],
        summary="parse passed",
    )]

    output = engine.process(NeoInput(
        prompt="verify this change",
        operating_mode=OperatingMode.VERIFY,
        proposed_changes=[ProposedChange(
            file_path="src/example.py",
            code_block="value = 1",
        )],
    ))

    assert engine.lm.calls == 0
    assert output.metadata["verification_only"] is True
    assert output.metadata["lm_calls"] == 0
    assert output.code_suggestions[0].file_path == "src/example.py"
    episode = engine.episode_store.load(output.metadata["learning_episode_id"])
    assert episode.final_outcome == "verification_complete"
    assert episode.applied_actions[0]["status"] == "provided_for_verification"


def test_verify_mode_fails_closed_without_a_change_before_lm_call(tmp_path):
    engine = _engine(tmp_path)

    with pytest.raises(ModeValidationError, match="requires at least one"):
        engine.process(NeoInput(
            prompt="verify",
            operating_mode=OperatingMode.VERIFY,
        ))

    assert engine.lm.calls == 0


def test_agent_requires_explicit_policy_and_host_executor_before_lm_call(tmp_path):
    engine = _engine(tmp_path)

    with pytest.raises(ModeValidationError, match="authority policy"):
        engine.process(NeoInput(prompt="act", operating_mode=OperatingMode.AGENT))

    with pytest.raises(ModeValidationError, match="host-provided execution adapter"):
        engine.process(NeoInput(
            prompt="act",
            operating_mode=OperatingMode.AGENT,
            authority=AuthorityPolicy(
                workspace_root=str(tmp_path),
                allowed_write_paths=["src/**"],
            ),
        ))

    assert engine.lm.calls == 0


def test_agent_rejects_generated_path_outside_allowlist_without_execution(tmp_path):
    executor = _Executor()
    engine = _engine(tmp_path, executor=executor)

    with pytest.raises(ModeValidationError, match="unauthorized path"):
        engine.process(NeoInput(
            prompt="act",
            operating_mode=OperatingMode.AGENT,
            authority=AuthorityPolicy(
                workspace_root=str(tmp_path),
                allowed_write_paths=["tests/**"],
            ),
        ))

    assert executor.calls == 0


def test_agent_delegates_only_authorized_suggestions_and_records_action(tmp_path):
    executor = _Executor()
    engine = _engine(tmp_path, executor=executor)

    output = engine.process(NeoInput(
        prompt="act",
        operating_mode=OperatingMode.AGENT,
        authority=AuthorityPolicy(
            workspace_root=str(tmp_path),
            allowed_write_paths=["src/**"],
            allow_learning=False,
        ),
    ))

    assert executor.calls == 1
    assert output.metadata["repository_actions"] == 1
    assert output.metadata["learning_enabled_for_request"] is False
    episode = engine.episode_store.load(output.metadata["learning_episode_id"])
    assert episode.authority["allowed_write_paths"] == ["src/**"]
    assert episode.applied_actions[0]["status"] == "applied"
    assert episode.final_outcome == "agent_actions_pending_downstream_outcome"


def test_authority_rejects_workspace_escape(tmp_path):
    policy = AuthorityPolicy(
        workspace_root=str(tmp_path),
        allowed_write_paths=["src/**"],
    )

    assert policy.allows_path("src/example.py") is True
    assert policy.allows_path("../outside.py") is False
    assert policy.allows_path(str(tmp_path.parent / "outside.py")) is False


def test_verify_cli_needs_no_provider_and_agent_cli_fails_before_provider(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(repository / "src"),
        "NEO_SKIP_UPDATE_CHECK": "1",
        "NEO_OBSERVER_AUTOSTART": "0",
    }
    for key in ("NEO_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        env.pop(key, None)
    payload = {
        "prompt": "verify this",
        "operating_mode": "verify",
        "working_directory": str(tmp_path),
        "proposed_changes": [{
            "file_path": "src/example.py",
            "code_block": "value = 1",
        }],
        "goal": {
            "description": "All checks pass",
            "success_criteria": [{
                "type": "command", "command": "pytest", "expected_exit_code": 0,
            }],
        },
        "intent": {"type": "verify_attempt"},
        "outcome": {"status": "passed", "summary": "checks passed"},
        "role": "verifier",
    }

    verified = subprocess.run(
        [sys.executable, "-m", "neo", "--json"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert verified.returncode == 0, verified.stderr
    output = json.loads(verified.stdout)
    assert output["metadata"]["operating_mode"] == "verify"
    assert output["metadata"]["lm_calls"] == 0
    assert output["goal_assessment"]["status"] == "satisfied"
    assert output["strategy_assessment"]["decision"] == "stop_success"

    agent = subprocess.run(
        [sys.executable, "-m", "neo", "--json"],
        input=json.dumps({
            "prompt": "act",
            "operating_mode": "agent",
            "working_directory": str(tmp_path),
            "authority": {
                "workspace_root": str(tmp_path),
                "allowed_write_paths": ["src/**"],
            },
        }),
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert agent.returncode == 1
    error = json.loads(agent.stdout)
    assert error["error"] == "AgentExecutorUnavailable"
    assert "adapter" not in error["error"].lower()
