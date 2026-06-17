"""Tests for Claude Code transcript parsing (Stage A) and ingestion (B/C)."""

import json

import pytest

from neo.memory.models import FactKind, FactScope
from neo.memory.transcript import (
    CarSource,
    CodexSource,
    Episode,
    TranscriptIngester,
    _parse_json,
    build_episodes,
    collect_episodes,
    resolve_transcript_dir,
)


class _StaticSource:
    """Test source yielding a fixed episode list."""

    name = "test"
    scope = FactScope.PROJECT

    def __init__(self, episodes):
        self._episodes = list(episodes)

    def collect_episodes(self):
        return list(self._episodes)


def _write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")


def _user(uuid, sid, text=None, blocks=None, **extra):
    content = text if text is not None else (blocks or [])
    return {"type": "user", "uuid": uuid, "sessionId": sid,
            "timestamp": "t", "message": {"role": "user", "content": content}, **extra}


def _assistant(uuid, sid, text=None, tools=None, **extra):
    content = []
    if text:
        content.append({"type": "text", "text": text})
    for name in (tools or []):
        content.append({"type": "tool_use", "name": name})
    return {"type": "assistant", "uuid": uuid, "sessionId": sid,
            "timestamp": "t", "message": {"role": "assistant", "content": content}, **extra}


def test_resolve_transcript_dir_path_encoding():
    d = resolve_transcript_dir("/Users/x/git/neo")
    assert d is not None and d.name == "-Users-x-git-neo"
    assert resolve_transcript_dir(None) is None


def test_basic_episode_capture(tmp_path):
    fp = tmp_path / "s.jsonl"
    _write(fp, [
        _user("u1", "s1", text="add retry logic to the client"),
        _assistant("a1", "s1", text="Sure, editing now", tools=["Edit", "Bash"]),
    ])
    eps = build_episodes(fp)
    assert len(eps) == 1
    ep = eps[0]
    assert ep.ask == "add retry logic to the client"
    assert ep.anchor_uuid == "u1"
    assert ep.last_uuid == "a1"          # advanced through the assistant record
    assert ep.tools == ["Edit", "Bash"]
    assert ep.is_substantive


def test_tool_result_is_not_human_text(tmp_path):
    """A user record whose content is a tool_result must not open an episode."""
    fp = tmp_path / "s.jsonl"
    _write(fp, [
        _user("u1", "s1", text="run the tests"),
        _assistant("a1", "s1", text="running", tools=["Bash"]),
        _user("u2", "s1", blocks=[{"type": "tool_result", "is_error": True,
                                   "content": "ImportError: no module"}]),
    ])
    eps = build_episodes(fp)
    assert len(eps) == 1                  # the tool_result did NOT start a new episode
    assert eps[0].ask == "run the tests"
    assert eps[0].errors == ["ImportError: no module"]
    assert eps[0].last_uuid == "u2"       # watermark advanced over the tool_result


def test_episode_boundary_on_new_human_message(tmp_path):
    fp = tmp_path / "s.jsonl"
    _write(fp, [
        _user("u1", "s1", text="first task"),
        _assistant("a1", "s1", text="done"),
        _user("u2", "s1", text="second task"),
        _assistant("a2", "s1", text="done too"),
    ])
    eps = build_episodes(fp)
    assert [e.ask for e in eps] == ["first task", "second task"]


def test_sidechain_skipped(tmp_path):
    fp = tmp_path / "s.jsonl"
    _write(fp, [
        _user("u1", "s1", text="real ask"),
        _user("u2", "s1", text="sidechain ask", isSidechain=True),
        _assistant("a1", "s1", text="ok"),
    ])
    eps = build_episodes(fp)
    assert len(eps) == 1
    assert eps[0].ask == "real ask"


def test_synthetic_command_and_control_strings_are_not_human(tmp_path):
    """CLI/control envelopes wear the user role but are not human asks."""
    fp = tmp_path / "s.jsonl"
    _write(fp, [
        _user("c1", "s1", text="<command-name>/clear</command-name>"),
        _user("c2", "s1", text="<local-command-stdout>output</local-command-stdout>"),
        _user("c3", "s1", text="<task-notification>done</task-notification>"),
        _user("c4", "s1", text="[Request interrupted by user for tool use]"),
        _user("c5", "s1", blocks=[{"type": "text",
                                   "text": "<local-command-caveat>Caveat</local-command-caveat>"}]),
        _assistant("a0", "s1", text="orphan assistant work"),
        _user("u1", "s1", text="a genuine human request"),
        _assistant("a1", "s1", text="done"),
    ])
    eps = build_episodes(fp)
    assert len(eps) == 1                         # only the genuine ask opened an episode
    assert eps[0].ask == "a genuine human request"


def test_sessions_are_partitioned(tmp_path):
    fp = tmp_path / "s.jsonl"
    _write(fp, [
        _user("u1", "s1", text="ask in session one"),
        _user("u2", "s2", text="ask in session two"),
        _assistant("a2", "s2", text="reply"),
        _assistant("a1", "s1", text="reply"),
    ])
    eps = build_episodes(fp)
    asks = sorted(e.ask for e in eps)
    assert asks == ["ask in session one", "ask in session two"]


def test_records_without_identity_are_dropped(tmp_path):
    fp = tmp_path / "s.jsonl"
    _write(fp, [
        {"type": "user", "message": {"role": "user", "content": "no uuid"}},
        {"type": "ai-title", "aiTitle": "x"},
        _user("u1", "s1", text="valid ask"),
        _assistant("a1", "s1", text="ok"),
    ])
    eps = build_episodes(fp)
    assert len(eps) == 1 and eps[0].ask == "valid ask"


def test_non_substantive_episode(tmp_path):
    fp = tmp_path / "s.jsonl"
    _write(fp, [_user("u1", "s1", text="just a question with no work")])
    eps = build_episodes(fp)
    assert len(eps) == 1 and not eps[0].is_substantive


def test_malformed_lines_skipped(tmp_path):
    fp = tmp_path / "s.jsonl"
    fp.write_text(
        "not json\n"
        + json.dumps(_user("u1", "s1", text="valid")) + "\n"
        + json.dumps(_assistant("a1", "s1", text="ok")) + "\n",
        encoding="utf-8",
    )
    eps = build_episodes(fp)
    assert len(eps) == 1 and eps[0].ask == "valid"


def test_collect_episodes_missing_dir():
    assert collect_episodes("/nonexistent/path/xyz") == []


# --------------------------------------------------------------------------
# Stage B/C: extraction + verify-at-admission
# --------------------------------------------------------------------------

class _StubAdapter:
    """LM boundary stub: returns a canned extract payload, then verify verdicts."""

    def __init__(self, lessons, keep=True):
        self._lessons = {"lessons": lessons}
        self._keep = keep
        self.calls = []

    def generate(self, messages, **kw):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        if '"lessons"' in prompt:  # extraction prompt
            return "preamble " + json.dumps(self._lessons) + " trailer"
        return json.dumps({"keep": self._keep, "reason": "t"})  # verify prompt


def _episode(ask="why did the test fail", asst="the venv was missing pytest-asyncio"):
    return Episode(session_id="s1", anchor_uuid="u1", last_uuid="u2",
                   timestamp="t", ask=ask, assistant_text=[asst], tools=["Bash"])


def test_parse_json_tolerant():
    assert _parse_json('{"keep": true}')["keep"] is True              # strict
    assert _parse_json('preamble {"keep": true} trailer')["keep"] is True  # sliced
    assert _parse_json('{"keep": true} note: see {x}')["keep"] is True     # brace in trailer
    assert _parse_json("no json here") is None
    assert _parse_json('[1, 2, 3]') is None                            # non-object
    assert _parse_json("") is None


def test_extract_lessons_parses_and_filters():
    ad = _StubAdapter([
        {"kind": "pattern", "subject": "verify env first", "body": "check the venv",
         "evidence_span": "venv was missing"},
        {"kind": "pattern", "subject": "no body", "body": ""},  # dropped
    ])
    ing = TranscriptIngester(store=None, lm_adapter=ad, codebase_root="/x")
    lessons = ing.extract_lessons(_episode())
    assert len(lessons) == 1 and lessons[0]["subject"] == "verify env first"


def test_verify_rejects_non_verbatim_evidence():
    ad = _StubAdapter([], keep=True)  # judge would keep, but evidence is bogus
    ing = TranscriptIngester(store=None, lm_adapter=ad, codebase_root="/x")
    lesson = {"subject": "x", "body": "y", "evidence_span": "this phrase is not in the episode"}
    assert ing.verify(lesson, _episode()) is False
    assert ad.calls == []  # short-circuits before calling the judge


def test_verify_rejects_when_judge_rejects():
    ad = _StubAdapter([], keep=False)
    ing = TranscriptIngester(store=None, lm_adapter=ad, codebase_root="/x")
    lesson = {"subject": "x", "body": "y", "evidence_span": "venv was missing pytest-asyncio"}
    assert ing.verify(lesson, _episode()) is False


def test_verify_accepts_with_verbatim_evidence_and_keep():
    ad = _StubAdapter([], keep=True)
    ing = TranscriptIngester(store=None, lm_adapter=ad, codebase_root="/x")
    lesson = {"subject": "x", "body": "y", "evidence_span": "venv was missing pytest-asyncio"}
    assert ing.verify(lesson, _episode()) is True


@pytest.fixture
def temp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEO_METRICS", "off")
    import neo.memory.store as store_mod
    monkeypatch.setattr(store_mod, "FACTS_DIR", tmp_path / "facts")
    return store_mod.FactStore(codebase_root="/tmp/proj_transcript_test", eager_init=False)


def test_ingest_episode_admits_capped_pattern(temp_store):
    ad = _StubAdapter([
        {"kind": "pattern", "subject": "verify env first",
         "body": "Stale virtualenvs cause spurious test failures; check the env before the code.",
         "domain": "testing", "confidence": 0.95,  # should be capped to 0.6
         "evidence_span": "venv was missing pytest-asyncio"},
    ], keep=True)
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, codebase_root="/x")
    n = ing.ingest_episode(_episode())
    assert n == 1
    facts = [f for f in temp_store._facts if f.is_valid]
    assert len(facts) == 1
    f = facts[0]
    assert f.kind == FactKind.PATTERN
    assert "transcript-derived" in f.tags
    assert f.metadata.confidence <= 0.6
    assert f.domain == "testing"          # domain lands on the first-class field, not tags
    assert "testing" not in f.tags


def test_admit_handles_non_numeric_confidence_and_bounds_body(temp_store):
    ad = _StubAdapter([], keep=True)
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, codebase_root="/x")
    lesson = {"kind": "pattern", "subject": "s", "body": "B" * 5000,
              "confidence": "high", "domain": "other",
              "evidence_span": "venv was missing pytest-asyncio"}
    fact = ing.admit(lesson, _episode())
    assert fact.metadata.confidence == 0.5          # non-numeric -> conservative default
    assert len(fact.body) <= 600                     # bounded
    assert fact.domain is None                       # "other" -> unset


def test_ingest_skips_nonsubstantive(temp_store):
    ad = _StubAdapter([{"kind": "pattern", "subject": "x", "body": "y"}])
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, codebase_root="/x")
    ep = Episode(session_id="s", anchor_uuid="u", last_uuid="u", timestamp="t", ask="just asking")
    assert ing.ingest_episode(ep) == 0
    assert ad.calls == []  # no LM calls for a non-substantive episode


# --------------------------------------------------------------------------
# Watermark / incremental ingest
# --------------------------------------------------------------------------

def _ep(uid, ask):
    return Episode(session_id="s", anchor_uuid=uid, last_uuid=uid, timestamp="t",
                   ask=ask, assistant_text=["the venv was missing pytest-asyncio"], tools=["Bash"])


_LESSON = {"kind": "pattern", "subject": "verify env", "body": "Check the venv before the code.",
           "domain": "testing", "evidence_span": "venv was missing pytest-asyncio"}


def test_ingest_is_idempotent(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    ad = _StubAdapter([_LESSON], keep=True)
    src = _StaticSource([_ep("e1", "ask one"), _ep("e2", "ask two")])
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, sources=[src])

    s1 = ing.ingest()
    assert s1["episodes_new"] == 2 and s1["episodes_processed"] == 2 and s1["facts_admitted"] == 2
    calls_after_first = len(ad.calls)

    s2 = ing.ingest()  # re-run: everything already consumed
    assert s2["episodes_new"] == 0 and s2["episodes_processed"] == 0
    assert len(ad.calls) == calls_after_first  # zero new LM calls on re-run


def test_ingest_budget_resumes(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    ad = _StubAdapter([_LESSON], keep=True)
    src = _StaticSource([_ep(f"e{i}", f"ask {i}") for i in range(5)])
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, sources=[src])

    s1 = ing.ingest(max_episodes=2)
    assert s1["episodes_new"] == 5 and s1["episodes_processed"] == 2  # budget honored

    s2 = ing.ingest(max_episodes=10)  # resumes the rest
    assert s2["episodes_new"] == 3 and s2["episodes_processed"] == 3


def test_watermark_persisted(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    ad = _StubAdapter([_LESSON], keep=True)
    src = _StaticSource([_ep("e1", "ask one")])
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, sources=[src])
    ing.ingest()
    assert ing._load_consumed(src) == {"e1"}


def test_ingest_respects_stop_and_deadline(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    ad = _StubAdapter([_LESSON], keep=True)
    src = _StaticSource([_ep(f"e{i}", f"ask {i}") for i in range(5)])
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, sources=[src])

    # should_stop fires immediately -> nothing dispatched
    s = ing.ingest(should_stop=lambda: True)
    assert s["episodes_processed"] == 0 and len(ad.calls) == 0

    # max_seconds=0 -> deadline already passed before the first episode
    s = ing.ingest(max_seconds=0)
    assert s["episodes_processed"] == 0 and len(ad.calls) == 0


class _GlobalSource(_StaticSource):
    name = "carlike"
    scope = FactScope.GLOBAL


def test_default_sources_include_claude_code(temp_store):
    ing = TranscriptIngester(store=temp_store, lm_adapter=_StubAdapter([]), codebase_root="/x")
    assert any(s.name == "claude-code" for s in ing.sources)


def test_multiple_sources_independent_watermarks_and_scope(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    ad = _StubAdapter([_LESSON], keep=True)
    s_proj = _StaticSource([_ep("p1", "project ask")])
    s_glob = _GlobalSource([_ep("g1", "global ask")])
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, sources=[s_proj, s_glob])

    stats = ing.ingest()
    assert stats["episodes_processed"] == 2
    # watermarks are namespaced per source — no collision
    assert ing._load_consumed(s_proj) == {"p1"}
    assert ing._load_consumed(s_glob) == {"g1"}
    # facts admitted at each source's scope
    scopes = {f.scope for f in temp_store._facts if f.is_valid}
    assert FactScope.PROJECT in scopes
    assert FactScope.GLOBAL in scopes


def test_shared_budget_across_sources(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    ad = _StubAdapter([_LESSON], keep=True)
    s1 = _StaticSource([_ep("a1", "ask"), _ep("a2", "ask")])
    s2 = _GlobalSource([_ep("b1", "ask")])
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, sources=[s1, s2])
    # budget of 1 is shared: only the first source's first episode is processed
    stats = ing.ingest(max_episodes=1)
    assert stats["episodes_processed"] == 1
    assert ing._load_consumed(s2) == set()  # second source never reached


# --------------------------------------------------------------------------
# CAR source adapter
# --------------------------------------------------------------------------

def _write_session(path, d):
    path.write_text(json.dumps(d), encoding="utf-8")


def test_car_source_parses_session(tmp_path):
    sdir = tmp_path / "car_sessions"
    sdir.mkdir()
    _write_session(sdir / "abc.json", {
        "id": "abc", "task": "do thing", "created_at": 123.0, "provider": "openai",
        "finished": True,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "fix the flaky test"},
            {"role": "assistant", "content": "I retried and found the race"},
        ],
    })
    src = CarSource(sessions_dir=sdir)
    assert src.name == "car" and src.scope == FactScope.GLOBAL
    eps = src.collect_episodes()
    assert len(eps) == 1
    ep = eps[0]
    assert ep.ask == "fix the flaky test"          # first user msg, not the task field
    assert ep.anchor_uuid == "abc"                 # session id = watermark anchor
    assert ep.assistant_text == ["I retried and found the race"]
    assert ep.is_substantive


def test_car_source_falls_back_to_task(tmp_path):
    sdir = tmp_path / "s"
    sdir.mkdir()
    _write_session(sdir / "x.json",
                   {"id": "x", "task": "the task", "finished": True,
                    "messages": [{"role": "assistant", "content": "did it"}]})
    eps = CarSource(sessions_dir=sdir).collect_episodes()
    assert len(eps) == 1 and eps[0].ask == "the task"


def test_car_source_skips_unfinished_sessions(tmp_path):
    sdir = tmp_path / "s"
    sdir.mkdir()
    _write_session(sdir / "live.json",
                   {"id": "live", "task": "in progress", "finished": False,
                    "messages": [{"role": "assistant", "content": "working"}]})
    # no `finished` key at all -> also skipped (treated as in-flight)
    _write_session(sdir / "nokey.json",
                   {"id": "nokey", "task": "t",
                    "messages": [{"role": "assistant", "content": "x"}]})
    assert CarSource(sessions_dir=sdir).collect_episodes() == []


def test_car_source_dedups_identical_asks(tmp_path):
    sdir = tmp_path / "s"
    sdir.mkdir()
    for i in range(4):
        _write_session(sdir / f"dup{i}.json",
                       {"id": f"dup{i}", "task": "What is 6 * 7?", "finished": True,
                        "messages": [{"role": "user", "content": "What is 6 * 7?"},
                                     {"role": "assistant", "content": "42"}]})
    eps = CarSource(sessions_dir=sdir).collect_episodes()
    assert len(eps) == 1  # the fan-out duplicates collapse to one episode


def test_car_source_skips_bad_and_missing(tmp_path):
    sdir = tmp_path / "s"
    sdir.mkdir()
    (sdir / "bad.json").write_text("not json", encoding="utf-8")
    _write_session(sdir / "noask.json", {"id": "e", "messages": []})  # no ask -> skipped
    assert CarSource(sessions_dir=sdir).collect_episodes() == []
    assert CarSource(sessions_dir=tmp_path / "nope").collect_episodes() == []


def test_default_sources_include_car(temp_store):
    ing = TranscriptIngester(store=temp_store, lm_adapter=_StubAdapter([]), codebase_root="/x")
    assert {s.name for s in ing.sources} >= {"claude-code", "car"}


# --------------------------------------------------------------------------
# Codex source adapter
# --------------------------------------------------------------------------

def _write_rollout(path, cwd, sid, records):
    lines = [{"timestamp": "t0", "type": "session_meta", "payload": {"id": sid, "cwd": cwd}}]
    lines += records
    path.write_text("\n".join(json.dumps(r) for r in lines), encoding="utf-8")


def _ev(pt, **payload):
    return {"timestamp": "t", "type": "event_msg", "payload": {"type": pt, **payload}}


def _ri(pt, **payload):
    return {"timestamp": "t", "type": "response_item", "payload": {"type": pt, **payload}}


def test_codex_source_parses_rollout(tmp_path):
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-1.jsonl", cwd="/work/proj", sid="sess1", records=[
        _ev("user_message", message="fix the failing build"),
        _ev("agent_message", message="tracing the build failure"),
        _ri("function_call", name="exec_command", arguments="{}"),
        _ev("exec_command_end", exit_code=1, stderr="compile error: missing symbol"),
    ])
    src = CodexSource(codebase_root="/work/proj", sessions_dir=sdir)
    assert src.name == "codex" and src.scope == FactScope.PROJECT
    eps = src.collect_episodes()
    assert len(eps) == 1
    ep = eps[0]
    assert ep.ask == "fix the failing build"
    assert ep.assistant_text == ["tracing the build failure"]
    assert ep.tools == ["exec_command"]
    assert ep.errors == ["compile error: missing symbol"]
    assert ep.anchor_uuid.startswith("sess1:")


def test_codex_source_filters_by_cwd(tmp_path):
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-other.jsonl", cwd="/some/other/repo", sid="s",
                   records=[_ev("user_message", message="not my project")])
    _write_rollout(sdir / "rollout-sub.jsonl", cwd="/work/proj/subdir", sid="s2",
                   records=[_ev("user_message", message="within the repo"),
                            _ev("agent_message", message="ok")])
    eps = CodexSource(codebase_root="/work/proj", sessions_dir=sdir).collect_episodes()
    assert len(eps) == 1 and eps[0].ask == "within the repo"  # only the in-repo cwd


def test_codex_source_multiple_user_messages(tmp_path):
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-m.jsonl", cwd="/work/proj", sid="s", records=[
        _ev("user_message", message="first ask"),
        _ev("agent_message", message="a1"),
        _ev("user_message", message="second ask"),
        _ev("agent_message", message="a2"),
    ])
    eps = CodexSource(codebase_root="/work/proj", sessions_dir=sdir).collect_episodes()
    assert [e.ask for e in eps] == ["first ask", "second ask"]
    assert eps[0].anchor_uuid != eps[1].anchor_uuid


def test_codex_source_skips_synthetic_user_message(tmp_path):
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-s.jsonl", cwd="/work/proj", sid="s", records=[
        _ev("user_message", message="<task-notification>x</task-notification>"),
        _ev("user_message", message="a real ask"),
        _ev("agent_message", message="ok"),
    ])
    eps = CodexSource(codebase_root="/work/proj", sessions_dir=sdir).collect_episodes()
    assert len(eps) == 1 and eps[0].ask == "a real ask"


def test_codex_source_no_root_or_missing_dir(tmp_path):
    assert CodexSource(codebase_root=None, sessions_dir=tmp_path).collect_episodes() == []
    assert CodexSource(codebase_root="/x", sessions_dir=tmp_path / "nope").collect_episodes() == []


def test_default_sources_include_codex(temp_store):
    ing = TranscriptIngester(store=temp_store, lm_adapter=_StubAdapter([]), codebase_root="/x")
    assert {s.name for s in ing.sources} >= {"claude-code", "codex", "car"}


def test_codex_source_captures_function_call_output_errors(tmp_path):
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-e.jsonl", cwd="/work/proj", sid="s", records=[
        _ev("user_message", message="run the migration"),
        _ri("function_call", name="exec_command"),
        _ri("function_call_output", output="Process exited with code 1\nsed: no such file"),
        _ri("function_call", name="exec_command"),
        _ri("function_call_output", output="Process exited with code 0\nOutput:\nok"),  # success: ignored
    ])
    eps = CodexSource(codebase_root="/work/proj", sessions_dir=sdir).collect_episodes()
    assert len(eps) == 1
    assert len(eps[0].errors) == 1 and "code 1" in eps[0].errors[0]  # only the failure


def test_codex_source_captures_timeout_and_patch_failure(tmp_path):
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-t.jsonl", cwd="/work/proj", sid="s", records=[
        _ev("user_message", message="apply the patch"),
        _ri("function_call_output", output="command timed out after 120015 milliseconds"),
        _ev("patch_apply_end", success=False, stderr="hunk failed to apply"),
    ])
    eps = CodexSource(codebase_root="/work/proj", sessions_dir=sdir).collect_episodes()
    assert len(eps) == 1
    joined = " ".join(eps[0].errors)
    assert "timed out" in joined and "hunk failed" in joined


def test_codex_source_skips_agent_history_wrapper(tmp_path):
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-w.jsonl", cwd="/work/proj", sid="s", records=[
        _ev("user_message", message="The following is the Codex agent history whose request action you are assessing..."),
        _ev("user_message", message="a genuine ask"),
        _ev("agent_message", message="ok"),
    ])
    eps = CodexSource(codebase_root="/work/proj", sessions_dir=sdir).collect_episodes()
    assert len(eps) == 1 and eps[0].ask == "a genuine ask"


# --------------------------------------------------------------------------
# Stage D: suggestion-outcome mining (durable ledger <-> transcript episodes)
# --------------------------------------------------------------------------

import time as _time
from dataclasses import dataclass as _dataclass

import numpy as _np

from neo.memory.outcomes import OUTCOME_CORRELATION_WINDOW_SECONDS


@_dataclass
class _Sugg:
    file_path: str = ""
    description: str = ""
    confidence: float = 0.8
    unified_diff: str = ""
    code_block: str = ""


@pytest.fixture
def mining_store(tmp_path, monkeypatch):
    """Real FactStore + OutcomeTracker, both rooted in tmp, with a deterministic
    embedder so correlation is controllable (vector keyed on 'validation')."""
    monkeypatch.setenv("NEO_METRICS", "off")
    import neo.memory.outcomes as out_mod
    import neo.memory.store as store_mod
    monkeypatch.setattr(store_mod, "FACTS_DIR", tmp_path / "facts")
    monkeypatch.setattr(out_mod, "SESSIONS_DIR", tmp_path / "sessions")
    st = store_mod.FactStore(codebase_root="/tmp/proj_mining_test", eager_init=False)

    def fake_embed(text):
        return _np.array([1.0, 0.0]) if "validation" in (text or "").lower() else _np.array([0.0, 1.0])

    monkeypatch.setattr(st, "_embed_text", fake_embed)
    return st


def _linked_review_fact(st, body="apply the validation fix"):
    fact = st.add_fact(subject="Validation review", body=body,
                       kind=FactKind.REVIEW, scope=FactScope.PROJECT, confidence=0.5)
    st._outcome_tracker.append_suggestion_ledger(
        [_Sugg(file_path="/REVIEW.md", description=body, confidence=0.8)],
        "review", {"/REVIEW.md": fact.id},
    )
    return fact


def test_mine_match_is_weak_reinforcement(mining_store):
    st = mining_store
    fact = _linked_review_fact(st)
    ts = st._outcome_tracker.load_suggestion_ledger()[0]["ts"]
    ep = Episode(session_id="s", anchor_uuid="a", last_uuid="b",
                 timestamp=str(ts + 60), ask="please apply the validation fix", errors=[])

    ing = TranscriptIngester(store=st, lm_adapter=None, sources=[])
    applied = ing.mine_suggestion_outcomes([ep])

    assert applied == 1
    # Weak UNVERIFIED delta: +1 success (enough to promote off probation) and
    # +0.1 confidence — never the strong +0.2 reserved for verified acceptance.
    assert fact.metadata.success_count == 1
    assert fact.metadata.confidence == pytest.approx(0.6)
    assert st._outcome_tracker.load_suggestion_ledger() == []  # entry consumed


def test_mine_ignores_episode_errors(mining_store):
    """Tool errors in the matched episode are the assistant's process noise, not
    a 'modify' signal about the suggestion — the reinforcement is unchanged."""
    st = mining_store
    fact = _linked_review_fact(st)
    ts = st._outcome_tracker.load_suggestion_ledger()[0]["ts"]
    ep = Episode(session_id="s", anchor_uuid="a", last_uuid="b",
                 timestamp=str(ts + 60), ask="please apply the validation fix",
                 errors=["TypeError: boom"])

    ing = TranscriptIngester(store=st, lm_adapter=None, sources=[])
    applied = ing.mine_suggestion_outcomes([ep])

    assert applied == 1
    assert fact.metadata.success_count == 1          # still reinforced, not demoted
    assert fact.metadata.confidence == pytest.approx(0.6)


def test_mine_no_match_keeps_entry_until_window_lapses(mining_store):
    st = mining_store
    fact = _linked_review_fact(st)
    ts = st._outcome_tracker.load_suggestion_ledger()[0]["ts"]
    # An unrelated episode within the window must not match.
    ep = Episode(session_id="s", anchor_uuid="a", last_uuid="b",
                 timestamp=str(ts + 60), ask="completely unrelated chatter", errors=[])

    ing = TranscriptIngester(store=st, lm_adapter=None, sources=[])
    assert ing.mine_suggestion_outcomes([ep]) == 0
    assert fact.metadata.success_count == 0
    assert len(st._outcome_tracker.load_suggestion_ledger()) == 1  # still pending


def test_mine_gives_up_after_window(mining_store):
    import json
    st = mining_store
    fact = _linked_review_fact(st)
    # Backdate the ledger entry past the correlation window: no episode will come.
    path = st._outcome_tracker._suggestion_ledger_path
    entry = st._outcome_tracker.load_suggestion_ledger()[0]
    entry["ts"] = _time.time() - OUTCOME_CORRELATION_WINDOW_SECONDS - 100
    path.write_text(json.dumps(entry) + "\n")

    ing = TranscriptIngester(store=st, lm_adapter=None, sources=[])
    assert ing.mine_suggestion_outcomes([]) == 0
    assert st._outcome_tracker.load_suggestion_ledger() == []  # expired, dropped
    assert fact.metadata.success_count == 0
