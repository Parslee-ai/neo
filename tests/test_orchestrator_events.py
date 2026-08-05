"""Tests for the host-facing layer: lifecycle events and OrchestratorMessage.

These cover the contract a host (Claude Code, an MCP server, an IDE) relies on:
which events fire and in what order, what the derived summary claims, and when
Neo stays silent rather than emitting unearned personality.
"""

import io
import json

import pytest

from neo.engine import NeoEngine
from neo.events import (
    JsonlSink,
    NeoEvent,
    NeoEventType,
    NullSink,
    RecordingSink,
    safe_emit,
)
from neo.models import NeoInput, ProposedChange, TaskType
from neo.operating_mode import OperatingMode
from neo.schemas import SCHEMA_VERSION


def _block(kind, payload):
    return f"<<<NEO:SCHEMA=v3:KIND={kind}>>>\n{json.dumps(payload)}\n<<<END:{kind}>>>"


def _response(*, issues=None, confidence=0.9):
    plan = [{
        "id": "ps_1",
        "description": "Add a guard clause before the deref",
        "rationale": "the parser crashes on empty input",
        "dependencies": [],
        "risk": "low",
        "schema_version": SCHEMA_VERSION,
    }]
    sim = [{
        "n": 1,
        "input_data": "empty string",
        "expected_output": "returns None",
        "reasoning_steps": ["parser sees an empty token"],
        "issues_found": list(issues or []),
        "schema_version": SCHEMA_VERSION,
    }]
    code = [{
        "file_path": "src/parser.py",
        "unified_diff": "--- a\n+++ b\n",
        "description": "guard the deref",
        "confidence": confidence,
        "tradeoffs": [],
        "schema_version": SCHEMA_VERSION,
    }]
    return "\n".join([
        _block("plan", plan), _block("simulation", sim), _block("code", code),
    ])


class FakeLM:
    """Returns one canned structured response regardless of the prompt."""

    model = "fake"
    provider = "fake"

    def __init__(self, response):
        self._response = response

    def generate(self, messages, **kwargs):
        return self._response

    def name(self):
        return "fake-lm"


def _run(response, prompt="fix the crash in the parser", task_type=TaskType.BUGFIX):
    sink = RecordingSink()
    engine = NeoEngine(
        lm_adapter=FakeLM(response),
        enable_persistent_memory=False,
        event_sink=sink,
    )
    output = engine.process(NeoInput(prompt=prompt, task_type=task_type))
    return output, sink


# ---------------------------------------------------------------- sinks


def test_null_sink_is_the_default():
    engine = NeoEngine(lm_adapter=FakeLM(""), enable_persistent_memory=False)
    assert isinstance(engine.event_sink, NullSink)


def test_jsonl_sink_writes_one_json_object_per_line():
    stream = io.StringIO()
    sink = JsonlSink(stream)
    sink.emit(NeoEvent(type=NeoEventType.STARTED, message="a"))
    sink.emit(NeoEvent(type=NeoEventType.COMPLETED, phase="finalize", message="b"))

    lines = stream.getvalue().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"type": "started", "message": "a"}
    assert json.loads(lines[1])["phase"] == "finalize"


def test_event_to_dict_drops_empty_fields():
    """Keeps lines short; a host must not have to skip past empty keys."""
    assert NeoEvent(type=NeoEventType.STARTED).to_dict() == {"type": "started"}


def test_a_failing_sink_cannot_break_the_run():
    """An observer must never take down the thing it observes."""

    class Exploding:
        def emit(self, event):
            raise RuntimeError("sink is broken")

    safe_emit(Exploding(), NeoEvent(type=NeoEventType.STARTED))  # must not raise


def test_engine_survives_a_failing_sink_end_to_end():
    class Exploding:
        def emit(self, event):
            raise RuntimeError("sink is broken")

    engine = NeoEngine(
        lm_adapter=FakeLM(_response()),
        enable_persistent_memory=False,
        event_sink=Exploding(),
    )
    output = engine.process(NeoInput(prompt="fix the parser", task_type=TaskType.BUGFIX))
    assert output.code_suggestions


# ---------------------------------------------------------------- event stream


def test_run_emits_started_and_completed():
    _, sink = _run(_response())
    types = [event.type for event in sink.events]
    assert types[0] is NeoEventType.STARTED
    assert types[-1] is NeoEventType.COMPLETED


def test_completed_carries_confidence_and_elapsed():
    output, sink = _run(_response())
    data = sink.of_type(NeoEventType.COMPLETED)[0].data
    assert data["confidence"] == pytest.approx(output.confidence, abs=0.01)
    assert data["elapsed_seconds"] >= 0


def test_phases_run_in_order_and_each_one_closes():
    _, sink = _run(_response())
    started = [e.phase for e in sink.of_type(NeoEventType.PHASE_STARTED)]
    completed = [e.phase for e in sink.of_type(NeoEventType.PHASE_COMPLETED)]
    assert started == ["context", "reasoning", "static_checks"]
    assert completed == started


def test_findings_are_emitted_before_their_phase_closes():
    """A host replaying the stream should learn what was found before it is
    told the phase finished."""
    _, sink = _run(_response(issues=["callers may not handle None"]))
    risk_index = next(
        i for i, e in enumerate(sink.events)
        if e.type is NeoEventType.RISK_FOUND
    )
    close_index = next(
        i for i, e in enumerate(sink.events)
        if e.type is NeoEventType.PHASE_COMPLETED and e.phase == "reasoning"
    )
    assert risk_index < close_index


def test_simulation_issues_surface_as_risk_events():
    _, sink = _run(_response(issues=["callers may not handle None"]))
    risks = sink.of_type(NeoEventType.RISK_FOUND)
    assert [r.message for r in risks] == ["callers may not handle None"]
    assert risks[0].data["source"] == "simulation"


def test_events_never_claim_a_phase_that_already_closed():
    """Facts are retrieved from several call sites, most of them while the
    reasoning phase is open. An event labeled with a closed phase would put a
    host's progress display out of order."""
    _, sink = _run(_response())
    open_phases: set[str] = set()
    for event in sink.events:
        if event.type is NeoEventType.PHASE_STARTED:
            open_phases.add(event.phase)
        elif event.type is NeoEventType.PHASE_COMPLETED:
            open_phases.discard(event.phase)
        elif event.phase:
            assert event.phase in open_phases or event.phase == "finalize", (
                f"{event.type.value} claimed closed phase {event.phase!r}"
            )


def test_leading_plan_step_surfaces_as_a_hypothesis():
    _, sink = _run(_response())
    formed = sink.of_type(NeoEventType.HYPOTHESIS_FORMED)
    assert formed[0].message == "Add a guard clause before the deref"


# ------------------------------------------------- branches with their own shape


def test_verify_mode_does_not_claim_neo_reasoned_or_proposed():
    """VERIFY makes no LM call and echoes the CALLER's changes back for
    checking. Saying "Neo reasoned" or "Neo proposes" there credits Neo with
    the caller's work — and the host is told to lead with this line."""
    sink = RecordingSink()
    engine = NeoEngine(
        lm_adapter=FakeLM(""), enable_persistent_memory=False, event_sink=sink,
    )
    output = engine.process(NeoInput(
        prompt="verify this change",
        operating_mode=OperatingMode.VERIFY,
        proposed_changes=[ProposedChange(
            file_path="src/example.py", code_block="value = 1",
        )],
    ))

    summary = output.orchestrator.summary
    assert "You gave me" in summary
    assert "I didn't write them" in summary
    assert "I read it" not in summary
    assert "proposes" not in summary


def test_verify_mode_does_not_tell_the_caller_to_verify_its_own_verification():
    """VERIFY reports confidence as a pass/fail verdict, not self-assessed
    certainty, so the low-confidence caution would be circular."""
    engine = NeoEngine(lm_adapter=FakeLM(""), enable_persistent_memory=False)
    output = engine.process(NeoInput(
        prompt="verify this change",
        operating_mode=OperatingMode.VERIFY,
        proposed_changes=[ProposedChange(
            file_path="src/example.py", code_block="value = 1",
        )],
    ))
    assert not any("Don't take it on trust" in c for c in output.orchestrator.cautions)


def test_skipped_static_checks_still_open_the_phase_they_close(monkeypatch):
    """A phase_completed with no phase_started leaves a host tracking a close
    for something it never saw open."""
    sink = RecordingSink()
    engine = NeoEngine(
        lm_adapter=FakeLM(_response()),
        enable_persistent_memory=False,
        event_sink=sink,
    )
    # Zero budget: elapsed always exceeds the static-check buffer.
    monkeypatch.setattr(engine, "_get_time_budget", lambda difficulty: 0.0)
    engine.process(NeoInput(prompt="fix the parser", task_type=TaskType.BUGFIX))

    started = [e.phase for e in sink.of_type(NeoEventType.PHASE_STARTED)]
    completed = [e.phase for e in sink.of_type(NeoEventType.PHASE_COMPLETED)]
    assert started == completed
    assert "static_checks" in started
    skip = [
        e for e in sink.of_type(NeoEventType.PHASE_COMPLETED)
        if e.phase == "static_checks"
    ][0]
    assert skip.data["status"] == "skipped"


def test_skipped_static_checks_caution_names_the_budget_not_missing_tools(monkeypatch):
    """Two different facts with two different remedies: install a linter, or
    give the run more time."""
    engine = NeoEngine(lm_adapter=FakeLM(_response()), enable_persistent_memory=False)
    monkeypatch.setattr(engine, "_get_time_budget", lambda difficulty: 0.0)
    output = engine.process(NeoInput(prompt="fix the parser", task_type=TaskType.BUGFIX))
    assert any("ran out of time" in c for c in output.orchestrator.cautions)
    assert not any("No checkers on this machine" in c for c in output.orchestrator.cautions)


def test_panel_fallback_closes_its_phase_and_opens_a_second_one(monkeypatch):
    """A failed panel closes `reasoning` as a fallback and the fast path opens
    a second `reasoning` record — the case `_end_phase`'s reverse search
    exists for."""
    from neo.reasoning_mode import ReasoningMode

    sink = RecordingSink()
    engine = NeoEngine(
        lm_adapter=FakeLM(_response()),
        enable_persistent_memory=False,
        event_sink=sink,
    )

    class _Decision:
        mode = ReasoningMode.MULTI_AGENT
        reason = "forced for test"

    monkeypatch.setattr(
        engine, "_decide_reasoning_mode",
        lambda context, difficulty, neo_input: (_Decision(), lambda *a, **k: ""),
    )
    # Panel returns nothing usable — the documented fallback path.
    monkeypatch.setattr(
        engine, "_deliberate", lambda context, route_fn: (None, None, None, None),
    )
    engine.process(NeoInput(prompt="fix the parser", task_type=TaskType.BUGFIX))

    reasoning_closes = [
        e for e in sink.of_type(NeoEventType.PHASE_COMPLETED) if e.phase == "reasoning"
    ]
    assert [e.data["status"] for e in reasoning_closes] == ["fallback", "complete"]
    assert len(
        [e for e in sink.of_type(NeoEventType.PHASE_STARTED) if e.phase == "reasoning"]
    ) == 2
    assert sink.of_type(NeoEventType.HYPOTHESIS_REJECTED)


def test_a_failed_run_still_emits_a_terminal_event():
    """STARTED then silence is worse than never emitting: a host cannot tell a
    crash from a hang."""
    class Exploding:
        model = "fake"
        provider = "fake"

        def generate(self, messages, **kwargs):
            raise RuntimeError("model exploded")

        def name(self):
            return "fake-lm"

    sink = RecordingSink()
    engine = NeoEngine(
        lm_adapter=Exploding(), enable_persistent_memory=False, event_sink=sink,
    )
    with pytest.raises(Exception):
        engine.process(NeoInput(prompt="fix the parser", task_type=TaskType.BUGFIX))

    failures = sink.of_type(NeoEventType.FAILED)
    assert failures and failures[0].data["error_type"]
    assert sink.events[-1].type is NeoEventType.FAILED


def test_a_failed_run_leaves_no_phase_stuck_running():
    class Exploding:
        model = "fake"
        provider = "fake"

        def generate(self, messages, **kwargs):
            raise RuntimeError("model exploded")

        def name(self):
            return "fake-lm"

    engine = NeoEngine(lm_adapter=Exploding(), enable_persistent_memory=False)
    with pytest.raises(Exception):
        engine.process(NeoInput(prompt="fix the parser", task_type=TaskType.BUGFIX))
    assert all(r["status"] != "running" for r in engine._phase_records)


# ---------------------------------------------------------------- orchestrator


def test_summary_names_what_ran_and_what_it_produced():
    output, _ = _run(_response())
    summary = output.orchestrator.summary
    assert "1 thing(s)" in summary
    assert "src/parser.py" in summary
    assert f"{output.confidence:.2f}" in summary


def test_phase_summary_mirrors_the_phase_events():
    output, sink = _run(_response())
    names = [record["name"] for record in output.orchestrator.phase_summary]
    assert names == [e.phase for e in sink.of_type(NeoEventType.PHASE_COMPLETED)]
    assert all(r["status"] != "running" for r in output.orchestrator.phase_summary)


def test_low_confidence_becomes_a_caution():
    output, _ = _run(_response(confidence=0.2))
    assert any("Don't take it on trust" in c for c in output.orchestrator.cautions)


def test_simulation_issues_become_a_caution():
    output, _ = _run(_response(issues=["callers may not handle None"]))
    assert any("problem(s) with my own approach" in c for c in output.orchestrator.cautions)


def test_narration_carries_only_closed_phases():
    output, _ = _run(_response())
    assert output.orchestrator.recommended_narration
    assert all(line for line in output.orchestrator.recommended_narration)


# ---------------------------------------------------------------- voice


class _Memory:
    """Just enough persistent-memory surface to drive the stage lookup."""

    def __init__(self, level):
        self._level = level

    def memory_level(self):
        return self._level


def _summary_at(level, *, mode="fast", confidence=0.88):
    from neo.models import CodeSuggestion, PlanStep

    engine = NeoEngine(lm_adapter=FakeLM(""), enable_persistent_memory=False)
    engine.persistent_memory = _Memory(level)
    engine.context = NeoInput(prompt="fix the parser crash")
    engine.last_reasoning_mode = mode
    message = engine._build_orchestrator_message(
        plan=[PlanStep(description="Add a guard", rationale="crash")],
        simulation_traces=[],
        code_suggestions=[CodeSuggestion(
            file_path="src/parser.py", unified_diff="", description="guard",
            confidence=confidence,
        )],
        static_checks=[],
        next_questions=[],
        confidence=confidence,
        early_exit=False,
    )
    return message


# 0.1 Sleeper, 0.3 Glitch, 0.5 Unplugged, 0.7 Training, 0.9 The One
_LEVELS = [0.1, 0.3, 0.5, 0.7, 0.9]


def test_voice_changes_with_memory_level():
    """Neo's certainty grows with what he remembers, so the same facts must not
    produce the same sentence at every stage."""
    summaries = [_summary_at(level).summary for level in _LEVELS]
    assert len(set(summaries)) == len(_LEVELS)


def test_early_stages_hedge_and_the_one_does_not():
    assert _summary_at(0.1).summary.endswith("Confidence 0.88.")
    assert ", maybe" in _summary_at(0.1).summary
    assert ", maybe" not in _summary_at(0.9).summary


def test_the_one_drops_the_opener_and_leads_with_the_fact():
    """Stage 5 is two words where stage 1 is a paragraph."""
    terse = _summary_at(0.9).summary
    assert terse.startswith("src/parser.py")
    assert "Confidence" not in terse  # the number alone
    assert len(terse) < len(_summary_at(0.1).summary)


def test_every_stage_still_states_the_facts():
    """Voice may vary; the numbers may not go missing."""
    for level in _LEVELS:
        summary = _summary_at(level).summary
        assert "0.88" in summary, level
        assert "src/parser.py" in summary, level


def test_the_panel_opener_differs_from_the_solo_one():
    for level in _LEVELS:
        solo = _summary_at(level, mode="fast").summary
        panel = _summary_at(level, mode="multi_agent").summary
        assert solo != panel, level


def test_cautions_do_not_vary_by_stage():
    """A host is told never to drop a caution, so a warning must read the same
    whether Neo is a Sleeper or The One. Voice is not licence to soften."""
    cautions = [tuple(_summary_at(level, confidence=0.4).cautions) for level in _LEVELS]
    assert len(set(cautions)) == 1
    assert any("Don't take it on trust" in c for c in cautions[0])


def test_engine_holds_no_prose_of_its_own():
    """All wording lives in the beat deck so the character can be retuned
    without editing code. A literal here is a personality change hidden in a
    diff nobody reads as one."""
    from pathlib import Path

    # Comments may quote the wording to explain a decision; only executable
    # lines are the concern, since those are what actually reach a user.
    source = "\n".join(
        line for line in Path("src/neo/engine.py").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    for phrase in ("I'd change", "Reading the code", "Don't take it on trust",
                   "Running the checkers", "I remember"):
        assert phrase not in source, phrase


def test_a_broken_voice_template_degrades_to_silence():
    """A formatting slip in a personality file must not take down a run."""
    engine = NeoEngine(lm_adapter=FakeLM(""), enable_persistent_memory=False)
    engine.beat_deck["orchestrator_voice"]["lines"]["started"] = "{nonexistent}"
    assert engine._voice("started") == ""
    assert engine._voice("no_such_key_at_all") == ""


def test_voice_falls_back_when_the_stage_is_missing():
    engine = NeoEngine(lm_adapter=FakeLM(""), enable_persistent_memory=False)
    engine.beat_deck["orchestrator_voice"]["stages"] = {
        3: {"opener_solo": "I read it.", "hedge": "", "confidence_lead": "Confidence"}
    }
    engine.persistent_memory = _Memory(0.95)  # stage 5, absent from the deck
    assert engine._voice_stage()["opener_solo"] == "I read it."


# ---------------------------------------------------------------- personality


def _engine_for_beats():
    return NeoEngine(lm_adapter=FakeLM(""), enable_persistent_memory=False)


def test_beat_requiring_a_finding_is_withheld_without_one():
    """A beat that claims insight must not fire on a run that found nothing."""
    engine = _engine_for_beats()
    engine.context = NeoInput(prompt="optimize the search algorithm")
    engine._findings = []
    assert engine._orchestrator_beat(confidence=0.9) == ""


def test_beat_requiring_a_finding_fires_once_there_is_one():
    engine = _engine_for_beats()
    engine.context = NeoInput(prompt="optimize the search algorithm")
    engine._findings = ["prior knowledge: hash lookups beat scans here"]
    assert engine._orchestrator_beat(confidence=0.9) == (
        "The obvious answer is probably the wrong one here."
    )


def test_internal_only_beats_never_reach_the_host():
    """Absent or non-orchestrator `surface` means terminal-only."""
    engine = _engine_for_beats()
    engine.context = NeoInput(prompt="refactor the module")
    engine._findings = ["something"]
    for beat in engine.beat_deck["beats"]:
        beat["surface"] = "internal"
    assert engine._orchestrator_beat(confidence=0.9) == ""


def test_no_matching_beat_means_silence_not_a_fallback_line():
    engine = _engine_for_beats()
    engine.context = NeoInput(prompt="tell me about this repository layout")
    engine._findings = []
    assert engine._orchestrator_beat(confidence=0.5) == ""


def test_surfaced_beat_is_also_emitted_as_an_event():
    output, sink = _run(_response(), prompt="fix the crash in the parser")
    beats = sink.of_type(NeoEventType.PERSONALITY_BEAT)
    if output.orchestrator.personality:
        assert beats and beats[0].message == output.orchestrator.personality
    else:
        assert not beats


def test_orchestrator_survives_the_cli_serialization_path():
    """The CLI emits `asdict(output.orchestrator)` into its JSON document; if
    that is not serializable the whole result is lost, not just the summary."""
    from dataclasses import asdict

    output, _ = _run(_response(issues=["callers may not handle None"]))
    payload = asdict(output.orchestrator)
    round_tripped = json.loads(json.dumps(payload))

    assert round_tripped["summary"] == output.orchestrator.summary
    assert round_tripped["cautions"] == output.orchestrator.cautions
    assert round_tripped["phase_summary"] == output.orchestrator.phase_summary


def test_orchestrator_is_populated_on_every_finalized_path():
    """Both the full pipeline and the early-exit path build the message; a host
    must never receive an empty envelope it has to fall back from."""
    output, _ = _run(_response())
    assert output.orchestrator.summary
    assert output.orchestrator.phase_summary


def test_one_run_speaks_with_one_voice():
    """`--json` carries both `notes` and `orchestrator.personality`. When each
    surface selected its own beat they could disagree — one character, two
    voices, same response."""
    engine = _engine_for_beats()
    engine.context = NeoInput(prompt="optimize the search algorithm")
    engine._findings = ["prior knowledge: something"]

    first = engine._run_beat(confidence=0.95)
    # A later caller with no confidence (the notes path) must not reselect.
    assert engine._run_beat() is first
    assert engine._run_beat(confidence=0.1) is first


def test_beat_selection_resets_between_runs():
    engine = NeoEngine(
        lm_adapter=FakeLM(_response()), enable_persistent_memory=False,
    )
    engine.process(NeoInput(prompt="fix the parser", task_type=TaskType.BUGFIX))
    first = engine._selected_beat
    engine.process(NeoInput(prompt="refactor the module", task_type=TaskType.REFACTOR))
    assert engine._beat_selected
    assert engine._selected_beat is not first


def test_no_memory_match_is_reachable():
    """`unfamiliar_codebase` was configured, surfaceable, and unreachable: no
    trigger it declares was ever produced, so its line was dead text."""
    engine = _engine_for_beats()
    beat = engine._select_beat({"prompt": "explain this module", "memory_checked": True})
    assert beat is not None
    assert beat["beat_id"] == "unfamiliar_codebase"


def test_memory_hit_beats_no_memory_match():
    engine = _engine_for_beats()
    beat = engine._select_beat({
        "prompt": "explain this module", "memory_checked": True, "memory_hit": True,
    })
    assert beat is not None
    assert beat["beat_id"] == "pattern_match"


def test_every_declared_beat_can_actually_fire():
    """A beat whose triggers `_select_beat` never emits is dead configuration
    that no wording test can catch."""
    engine = _engine_for_beats()
    reachable = {
        "error_trace_present", "bugfix", "refactor", "algorithm", "optimization",
        "high_confidence", "memory_hit", "familiar_pattern", "no_memory_match",
    }
    for beat in engine.beat_deck["beats"]:
        triggers = set(beat.get("trigger_contexts", []))
        assert triggers & reachable, f"{beat['beat_id']} can never fire"


def test_every_surfaceable_beat_declares_its_wording():
    """A beat marked orchestrator-surfaceable with no line would emit nothing
    and silently look like a non-matching beat."""
    engine = _engine_for_beats()
    for beat in engine.beat_deck["beats"]:
        if beat.get("surface") == "orchestrator":
            assert beat.get("orchestrator_line"), beat["beat_id"]
