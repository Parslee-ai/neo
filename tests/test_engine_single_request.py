"""A NeoEngine handles one request at a time.

Almost everything a run needs lives on the instance — `context`,
`current_learning_episode`, `_phase_records`, `last_applied_actions`,
`resolved_execution_context`. Two overlapping `process()` calls would interleave
into each other and cross-attribute suggestions, facts and learning episodes
between unrelated requests.

That corruption is silent, which is the problem. `neo.car_host` caches engines
per working directory and reuses them across calls, relying on CAR's drain task
being single-threaded — an upstream implementation detail neo does not control.
So the guard fails loudly instead of trusting it.
"""

import json
import threading

import pytest

from neo.engine import EngineBusyError, NeoEngine
from neo.models import NeoInput, TaskType
from neo.schemas import SCHEMA_VERSION


def _block(kind, payload):
    return f"<<<NEO:SCHEMA=v3:KIND={kind}>>>\n{json.dumps(payload)}\n<<<END:{kind}>>>"


def _response():
    plan = [{"id": "ps_1", "description": "Add a guard", "rationale": "crash",
             "dependencies": [], "schema_version": SCHEMA_VERSION}]
    sim = [{"n": 1, "input_data": "x", "expected_output": "y",
            "reasoning_steps": ["z"], "issues_found": [],
            "schema_version": SCHEMA_VERSION}]
    code = [{"file_path": "src/parser.py", "unified_diff": "", "description": "d",
             "confidence": 0.9, "tradeoffs": [], "schema_version": SCHEMA_VERSION}]
    return "\n".join([_block("plan", plan), _block("simulation", sim),
                      _block("code", code)])


class _ReentrantLM:
    """Calls back into the engine mid-run, which is the cheapest faithful
    simulation of an overlapping request."""

    model = "fake"
    provider = "fake"

    def __init__(self, engine_ref, response):
        self.engine_ref = engine_ref
        self.response = response
        self.inner_error = None

    def generate(self, messages, **kwargs):
        if self.inner_error is None:
            try:
                self.engine_ref[0].process(NeoInput(prompt="overlapping request"))
                self.inner_error = False
            except EngineBusyError as exc:
                self.inner_error = exc
        return self.response

    def name(self):
        return "reentrant-lm"


class _SlowLM:
    """Holds the run open so a second thread can collide with it."""

    model = "fake"
    provider = "fake"

    def __init__(self, response, started, release):
        self.response = response
        self.started = started
        self.release = release

    def generate(self, messages, **kwargs):
        self.started.set()
        self.release.wait(timeout=10)
        return self.response

    def name(self):
        return "slow-lm"


def test_overlapping_process_raises_rather_than_interleaving():
    ref = []
    lm = _ReentrantLM(ref, _response())
    engine = NeoEngine(lm_adapter=lm, enable_persistent_memory=False)
    ref.append(engine)

    engine.process(NeoInput(prompt="first request", task_type=TaskType.BUGFIX))

    assert isinstance(lm.inner_error, EngineBusyError), (
        "an overlapping process() call was allowed through"
    )


def test_concurrent_threads_cannot_share_one_engine():
    started, release = threading.Event(), threading.Event()
    engine = NeoEngine(
        lm_adapter=_SlowLM(_response(), started, release),
        enable_persistent_memory=False,
    )

    result = {}

    def first():
        try:
            engine.process(NeoInput(prompt="first", task_type=TaskType.BUGFIX))
            result["first"] = "ok"
        except Exception as exc:  # pragma: no cover - diagnostic
            result["first"] = exc

    worker = threading.Thread(target=first)
    worker.start()
    try:
        assert started.wait(timeout=10), "the first run never reached the LM"
        with pytest.raises(EngineBusyError):
            engine.process(NeoInput(prompt="second", task_type=TaskType.BUGFIX))
    finally:
        release.set()
        worker.join(timeout=15)

    assert result.get("first") == "ok", "the guard broke the run it was protecting"


def test_the_lock_is_released_so_the_engine_stays_reusable():
    """Sequential reuse is the normal case — car_host caches engines per cwd."""
    engine = NeoEngine(
        lm_adapter=_SlowLM(_response(), threading.Event(), _preset_event()),
        enable_persistent_memory=False,
    )
    for _ in range(3):
        engine.process(NeoInput(prompt="again", task_type=TaskType.BUGFIX))


def test_the_lock_is_released_after_a_failed_run():
    """A raising run must not leave the engine permanently busy."""

    class Exploding:
        model = "fake"
        provider = "fake"

        def __init__(self):
            self.calls = 0

        def generate(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("model exploded")
            return _response()

        def name(self):
            return "exploding-lm"

    engine = NeoEngine(lm_adapter=Exploding(), enable_persistent_memory=False)
    with pytest.raises(Exception):
        engine.process(NeoInput(prompt="first", task_type=TaskType.BUGFIX))

    # Must not raise EngineBusyError.
    engine.process(NeoInput(prompt="second", task_type=TaskType.BUGFIX))


def test_busy_error_is_a_runtime_error():
    """Existing broad handlers must keep catching it."""
    assert issubclass(EngineBusyError, RuntimeError)


def test_a2a_host_reports_busy_as_retryable():
    """A peer that sees a generic ProcessingError assumes its own request was
    malformed and stops retrying."""
    source = (
        __import__("pathlib").Path("src/neo/car_host.py").read_text()
    )
    assert "EngineBusyError" in source
    assert '"error": "EngineBusy"' in source
    assert '"retryable": True' in source


def _preset_event():
    event = threading.Event()
    event.set()
    return event
