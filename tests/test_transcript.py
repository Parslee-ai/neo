"""Tests for Claude Code transcript parsing (Stage A) and ingestion (B/C)."""

import datetime
import json
import time
import time as _time
import types
from dataclasses import dataclass as _dataclass

import numpy as _np
import pytest

from neo.memory.models import FactKind, FactScope
from neo.memory.outcomes import OUTCOME_CORRELATION_WINDOW_SECONDS
from neo.memory.transcript import (
    CarSource,
    CodexSource,
    Episode,
    GitHubPRSource,
    TranscriptIngester,
    _fetch_merged_prs,
    _gh_graphql,
    _owner_repo_from_remote,
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


def test_codex_source_root_slash_yields_to_any_known_peer(tmp_path):
    """``"/"`` must not claim the machine's entire Codex history.

    Every path really is inside ``/``, so containment alone can't refuse it —
    the protection is that ``/`` loses to any more specific known root, and
    that observer discovery drops it as a container in the first place. This
    pins the half that lives here.
    """
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-a.jsonl", cwd="/work/proj", sid="s1",
                   records=[_ev("user_message", message="anywhere on disk")])
    peers = ["/", "/work/proj"]
    assert CodexSource("/", sessions_dir=sdir, peer_roots=peers).collect_episodes() == []
    won = CodexSource("/work/proj", sessions_dir=sdir, peer_roots=peers).collect_episodes()
    assert [e.ask for e in won] == ["anywhere on disk"]


def test_codex_source_nested_peer_root_wins(tmp_path):
    """A container root must not claim a nested project's sessions."""
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-nested.jsonl", cwd="/work/git/proj", sid="s1",
                   records=[_ev("user_message", message="belongs to proj")])
    _write_rollout(sdir / "rollout-loose.jsonl", cwd="/work/git", sid="s2",
                   records=[_ev("user_message", message="belongs to the container")])
    peers = ["/work/git", "/work/git/proj"]

    container = CodexSource(codebase_root="/work/git", sessions_dir=sdir,
                            peer_roots=peers).collect_episodes()
    assert [e.ask for e in container] == ["belongs to the container"]

    nested = CodexSource(codebase_root="/work/git/proj", sessions_dir=sdir,
                         peer_roots=peers).collect_episodes()
    assert [e.ask for e in nested] == ["belongs to proj"]


def test_codex_source_rejects_false_prefix_sibling(tmp_path):
    """``/work/github/x`` is NOT inside ``/work/git`` — component-aware, not string prefix.

    This is the same family as the ``"/".rstrip("/") == ""`` bug and is exactly
    what regresses when someone "simplifies" the containment test.
    """
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-sib.jsonl", cwd="/work/github/x", sid="s1",
                   records=[_ev("user_message", message="different project")])
    assert CodexSource(codebase_root="/work/git", sessions_dir=sdir).collect_episodes() == []


def test_codex_source_deepest_of_three_levels_wins(tmp_path):
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-deep.jsonl", cwd="/a/b/c", sid="s1",
                   records=[_ev("user_message", message="deepest")])
    peers = ["/a", "/a/b", "/a/b/c"]
    for root in ("/a", "/a/b"):
        assert CodexSource(root, sessions_dir=sdir, peer_roots=peers).collect_episodes() == []
    won = CodexSource("/a/b/c", sessions_dir=sdir, peer_roots=peers).collect_episodes()
    assert [e.ask for e in won] == ["deepest"]


def test_codex_source_unattributable_rollouts(tmp_path):
    """No session_meta, malformed JSON, and a session_meta with no cwd all mean 'skip'."""
    sdir = tmp_path / "codex"
    sdir.mkdir()
    (sdir / "rollout-bad.jsonl").write_text("{not json", encoding="utf-8")
    (sdir / "rollout-nometa.jsonl").write_text(
        json.dumps({"timestamp": "t", "type": "event_msg", "payload": {}}), encoding="utf-8")
    _write_rollout(sdir / "rollout-nocwd.jsonl", cwd="", sid="s",
                   records=[_ev("user_message", message="no cwd")])
    assert CodexSource(codebase_root="/work/proj", sessions_dir=sdir).collect_episodes() == []


def test_ingester_threads_peer_roots_to_codex_source():
    """The middle hops: TranscriptIngester must actually reach CodexSource."""
    ing = TranscriptIngester(store=object(), lm_adapter=None, codebase_root="/a",
                             peer_roots=["/a", "/a/b", "/other"])
    codex = next(s for s in ing.sources if isinstance(s, CodexSource))
    assert codex._nested_peers == ["/a/b"]  # only nested peers, "/other" excluded


def test_ingester_rejects_peer_roots_with_explicit_sources():
    """peer_roots would be silently ignored alongside explicit sources — refuse instead."""
    with pytest.raises(ValueError, match="peer_roots"):
        TranscriptIngester(store=object(), lm_adapter=None, codebase_root="/a",
                           sources=[CodexSource("/a")], peer_roots=["/a", "/a/b"])


def test_codex_source_without_peers_keeps_prefix_behavior(tmp_path):
    """Single-project callers (the request path) pass no peers and are unchanged."""
    sdir = tmp_path / "codex"
    sdir.mkdir()
    _write_rollout(sdir / "rollout-nested.jsonl", cwd="/work/git/proj", sid="s1",
                   records=[_ev("user_message", message="nested")])
    eps = CodexSource(codebase_root="/work/git", sessions_dir=sdir).collect_episodes()
    assert [e.ask for e in eps] == ["nested"]


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


def test_mine_match_is_observation_only(mining_store):
    st = mining_store
    fact = _linked_review_fact(st)
    ts = st._outcome_tracker.load_suggestion_ledger()[0]["ts"]
    ep = Episode(session_id="s", anchor_uuid="a", last_uuid="b",
                 timestamp=str(ts + 60), ask="please apply the validation fix", errors=[])

    ing = TranscriptIngester(store=st, lm_adapter=None, sources=[])
    applied = ing.mine_suggestion_outcomes([ep])

    assert applied == 1
    assert fact.metadata.success_count == 0
    assert fact.metadata.confidence == pytest.approx(0.5)
    assert fact.metadata.effectiveness_n == 0
    assert "probation" in fact.tags
    assert st._outcome_tracker.load_suggestion_ledger() == []  # entry consumed


def test_mine_episode_errors_still_do_not_create_learning_signal(mining_store):
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
    assert fact.metadata.success_count == 0
    assert fact.metadata.confidence == pytest.approx(0.5)
    assert fact.metadata.effectiveness_n == 0


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


# ===========================================================================
# GitHubPRSource — merged PRs + review threads as a transcript source.
# All network is faked; no test touches the real `gh` CLI or GitHub.
# ===========================================================================


def _pr(number=1, title="t", body="", updated="u1", merged="m1",
        reviews=None, comments=None, threads=None):
    """A GraphQL PR node shaped like _fetch_merged_prs returns."""
    return {
        "number": number, "title": title, "body": body,
        "updatedAt": updated, "mergedAt": merged,
        "author": {"login": "author"},
        "reviews": {"nodes": reviews or []},
        "comments": {"nodes": comments or []},
        "reviewThreads": {"nodes": threads or []},
    }


class TestOwnerRepoParsing:
    @pytest.mark.parametrize("url,expected", [
        ("git@github.com:Parslee-ai/neo.git", ("Parslee-ai", "neo")),
        ("https://github.com/Parslee-ai/neo", ("Parslee-ai", "neo")),
        ("https://x-token@github.com/Parslee-ai/neo.git", ("Parslee-ai", "neo")),
        ("ssh://git@github.com/Parslee-ai/neo.git", ("Parslee-ai", "neo")),
        ("git@gitlab.com:foo/bar.git", None),   # non-GitHub remote
        ("", None),                              # no remote
    ])
    def test_parses(self, monkeypatch, url, expected):
        monkeypatch.setattr("neo.memory.transcript._get_git_remote_url",
                            lambda root: url)
        assert _owner_repo_from_remote("/x") == expected


class TestPRToEpisode:
    def test_maps_reviews_comments_inline_and_errors(self):
        pr = _pr(
            number=42, title="Add cache", body="why", updated="2026-06-02",
            merged="2026-06-01",
            reviews=[
                {"body": "LGTM", "state": "APPROVED", "author": {"login": "bob"}},
                {"body": "key collides", "state": "CHANGES_REQUESTED",
                 "author": {"login": "carol"}},
            ],
            comments=[
                {"body": "bump", "author": {"login": "dependabot[bot]"}},  # bot
                {"body": "nice", "author": {"login": "dave"}},
            ],
            threads=[{"isResolved": True, "comments": {"nodes": [
                {"body": "use LRU", "path": "cache.py", "author": {"login": "carol"}},
            ]}}],
        )
        ep = GitHubPRSource._pr_to_episode(pr)
        joined = " ".join(ep.assistant_text)
        assert ep.ask.startswith("PR #42: Add cache")
        assert "review APPROVED by bob" in joined
        assert "review CHANGES_REQUESTED by carol" in joined
        assert "comment by dave" in joined
        assert "inline cache.py by carol" in joined
        assert "dependabot" not in joined            # bot filtered out
        assert ep.errors and "changes requested by carol" in ep.errors[0]
        assert ep.anchor_uuid == "pr-42"             # mine-once: keyed on number
        assert ep.timestamp == "2026-06-01"
        assert ep.is_substantive

    def test_no_discussion_returns_none(self):
        assert GitHubPRSource._pr_to_episode(_pr(number=7, title="typo")) is None

    def test_all_bot_discussion_returns_none(self):
        pr = _pr(comments=[{"body": "bump", "author": {"login": "renovate"}}])
        assert GitHubPRSource._pr_to_episode(pr) is None

    def test_missing_title_returns_none(self):
        assert GitHubPRSource._pr_to_episode(_pr(title="")) is None

    def test_handles_null_author_and_null_nodes(self):
        """GraphQL returns author:null for deleted accounts and may null whole
        connections — none of it should raise."""
        pr = {
            "number": 3, "title": "x", "body": "", "updatedAt": "u", "mergedAt": "m",
            "author": None,                       # deleted account
            "reviews": None,                      # whole connection null
            "comments": {"nodes": [None, {"body": "real point", "author": None}]},
            "reviewThreads": {"nodes": [None]},   # null node in list
        }
        ep = GitHubPRSource._pr_to_episode(pr)
        assert ep is not None
        assert any("real point" in t for t in ep.assistant_text)
        assert ep.anchor_uuid == "pr-3"


class TestGhFetchFailureModes:
    def _run(self, monkeypatch, *, rc=0, stdout="", raises=None):
        def fake_run(*a, **k):
            if raises:
                raise raises
            return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr="e")
        monkeypatch.setattr("neo.memory.transcript.subprocess.run", fake_run)

    def test_none_on_nonzero(self, monkeypatch):
        self._run(monkeypatch, rc=1)
        assert _gh_graphql("q") is None

    def test_none_on_exception(self, monkeypatch):
        self._run(monkeypatch, raises=OSError("no gh"))
        assert _gh_graphql("q") is None

    def test_none_on_timeout(self, monkeypatch):
        import subprocess
        self._run(monkeypatch, raises=subprocess.TimeoutExpired("gh", 20))
        assert _gh_graphql("q") is None

    def test_none_on_bad_json(self, monkeypatch):
        self._run(monkeypatch, rc=0, stdout="not json")
        assert _gh_graphql("q") is None

    def test_string_vars_use_lowercase_f_int_vars_use_uppercase_F(self, monkeypatch):
        """Regression: a repo named "2048" must go via -f (String!), not -F,
        which would coerce it to an Int and silently fail the query."""
        captured = {}

        def fake_run(cmd, **k):
            captured["cmd"] = cmd
            return types.SimpleNamespace(returncode=0, stdout='{"data":{}}', stderr="")
        monkeypatch.setattr("neo.memory.transcript.subprocess.run", fake_run)
        _gh_graphql("Q", str_vars={"owner": "2048", "repo": "2048"}, int_vars={"n": 25})
        cmd = captured["cmd"]
        assert cmd[cmd.index("owner=2048") - 1] == "-f"   # String! via -f
        assert cmd[cmd.index("repo=2048") - 1] == "-f"
        assert cmd[cmd.index("n=25") - 1] == "-F"         # Int! via -F

    def test_fetch_empty_when_graphql_none(self, monkeypatch):
        monkeypatch.setattr("neo.memory.transcript._gh_graphql", lambda *a, **k: None)
        assert _fetch_merged_prs("o", "r") == []

    def test_fetch_empty_on_malformed_payload(self, monkeypatch):
        monkeypatch.setattr("neo.memory.transcript._gh_graphql",
                            lambda *a, **k: {"repository": None})
        assert _fetch_merged_prs("o", "r") == []


class TestCollectEpisodesGuards:
    def test_empty_when_no_gh(self, monkeypatch):
        monkeypatch.setattr("neo.memory.transcript._gh_available", lambda: False)
        monkeypatch.setattr("neo.memory.transcript._owner_repo_from_remote",
                            lambda root: ("o", "r"))
        assert GitHubPRSource("/x").collect_episodes() == []

    def test_empty_when_non_github_remote(self, monkeypatch):
        monkeypatch.setattr("neo.memory.transcript._gh_available", lambda: True)
        monkeypatch.setattr("neo.memory.transcript._owner_repo_from_remote",
                            lambda root: None)
        assert GitHubPRSource("/x").collect_episodes() == []

    def test_maps_fetched_prs(self, tmp_path, monkeypatch):
        # SESSIONS_DIR → tmp so the throttle stamp never touches real state.
        monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "s")
        monkeypatch.setattr("neo.memory.transcript._gh_available", lambda: True)
        monkeypatch.setattr("neo.memory.transcript._owner_repo_from_remote",
                            lambda root: ("o", "r"))
        monkeypatch.setattr(
            "neo.memory.transcript._fetch_merged_prs",
            lambda o, r: [_pr(number=1, title="x",
                              comments=[{"body": "good", "author": {"login": "u"}}])])
        eps = GitHubPRSource("/x").collect_episodes()
        assert len(eps) == 1 and eps[0].session_id == "pr-1"

    def test_fetch_is_throttled_per_repo(self, tmp_path, monkeypatch):
        monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "s")
        monkeypatch.setattr("neo.memory.transcript._gh_available", lambda: True)
        monkeypatch.setattr("neo.memory.transcript._owner_repo_from_remote",
                            lambda root: ("o", "r"))
        calls = []
        monkeypatch.setattr(
            "neo.memory.transcript._fetch_merged_prs",
            lambda o, r: calls.append(1) or [
                _pr(number=1, comments=[{"body": "x", "author": {"login": "u"}}])])
        src = GitHubPRSource("/x")
        src.collect_episodes()          # no stamp → fetches
        src.collect_episodes()          # within interval → throttled, no fetch
        assert len(calls) == 1
        monkeypatch.setattr("neo.memory.transcript._GH_PR_FETCH_INTERVAL", 0)
        src.collect_episodes()          # interval elapsed → fetches again
        assert len(calls) == 2


def test_pr_facts_enter_as_review_on_probation(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("neo.memory.transcript._gh_available", lambda: True)
    monkeypatch.setattr("neo.memory.transcript._owner_repo_from_remote",
                        lambda root: ("o", "r"))
    pr = _pr(number=5, title="Add cache",
             reviews=[{"body": "key collides under load", "state": "CHANGES_REQUESTED",
                       "author": {"login": "carol"}}])
    monkeypatch.setattr("neo.memory.transcript._fetch_merged_prs", lambda o, r: [pr])
    ad = _StubAdapter([{"kind": "pattern", "subject": "cache key design",
                        "body": "Cache keys must include all inputs to avoid collisions.",
                        "evidence_span": "key collides under load"}], keep=True)
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, sources=[GitHubPRSource("/x")])
    stats = ing.ingest()
    assert stats["facts_admitted"] == 1
    facts = [f for f in temp_store._facts if f.is_valid]
    assert len(facts) == 1
    f = facts[0]
    assert f.kind == FactKind.REVIEW                  # trust-first, not PATTERN
    assert "imported:github-pr" in f.tags
    assert "transcript-derived" in f.tags
    assert "probation" in f.tags                      # enters on probation


def test_pr_ingest_is_mine_once(temp_store, tmp_path, monkeypatch):
    """A merged PR is mined exactly once (watermark keyed on PR number), even if
    its thread grows after merge — keeps the watermark bounded and avoids re-
    paying extraction. Throttle disabled so this exercises the watermark itself."""
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("neo.memory.transcript._GH_PR_FETCH_INTERVAL", 0)  # no throttle
    monkeypatch.setattr("neo.memory.transcript._gh_available", lambda: True)
    monkeypatch.setattr("neo.memory.transcript._owner_repo_from_remote",
                        lambda root: ("o", "r"))
    lesson = [{"kind": "pattern", "subject": "bounded cache",
               "body": "Use a bounded LRU to cap cache memory.",
               "evidence_span": "use a bounded LRU here"}]
    ad = _StubAdapter(lesson, keep=True)
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, sources=[GitHubPRSource("/x")])

    v1 = [_pr(number=9, title="x", updated="2026-01-01",
              comments=[{"body": "use a bounded LRU here", "author": {"login": "carol"}}])]
    monkeypatch.setattr("neo.memory.transcript._fetch_merged_prs", lambda o, r: v1)
    assert ing.ingest()["episodes_new"] == 1
    assert ing.ingest()["episodes_new"] == 0          # already mined

    # Thread grew (new updatedAt) — but mine-once keys on number, so still 0 new.
    v2 = [_pr(number=9, title="x", updated="2026-02-02",
              comments=[{"body": "use a bounded LRU here", "author": {"login": "carol"}},
                        {"body": "and add a metric", "author": {"login": "dave"}}])]
    monkeypatch.setattr("neo.memory.transcript._fetch_merged_prs", lambda o, r: v2)
    assert ing.ingest()["episodes_new"] == 0          # not re-mined


def test_source_without_fact_kind_defaults_to_pattern(temp_store, tmp_path, monkeypatch):
    """Backward-compat: a source that declares no fact_kind still yields PATTERN."""
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    src = _StaticSource([_episode()])  # no fact_kind / extra_tags attributes
    ad = _StubAdapter([{"kind": "pattern", "subject": "s",
                        "body": "Check the venv before blaming the code.",
                        "evidence_span": "venv was missing pytest-asyncio"}], keep=True)
    ing = TranscriptIngester(store=temp_store, lm_adapter=ad, sources=[src])
    ing.ingest()
    facts = [f for f in temp_store._facts if f.is_valid]
    assert facts and facts[0].kind == FactKind.PATTERN
    assert "imported:github-pr" not in facts[0].tags


class TestSkipUnchangedInputs:
    """Parsing dominated the observer's memory: one project measured 104 MB for
    ClaudeCodeSource and 224 MB for CodexSource, and the sweep does 25 projects
    a cycle. The watermark only gated *admission*, so an unchanged project still
    paid the full parse — the opposite of the documented "near-zero work".
    """

    def test_unchanged_file_is_skipped(self, tmp_path):
        from neo.memory.transcript import _unchanged_since
        import os
        f = tmp_path / "t.jsonl"
        f.write_text("{}")
        old = time.time() - 10_000
        os.utime(f, (old, old))
        assert _unchanged_since(f, time.time()) is True

    def test_modified_file_is_never_skipped(self, tmp_path):
        from neo.memory.transcript import _unchanged_since
        f = tmp_path / "t.jsonl"
        f.write_text("{}")
        assert _unchanged_since(f, time.time() - 10_000) is False

    def test_no_since_parses_everything(self, tmp_path):
        """First run has no watermark — must not skip anything."""
        from neo.memory.transcript import _unchanged_since
        f = tmp_path / "t.jsonl"
        f.write_text("{}")
        assert _unchanged_since(f, None) is False
        assert _unchanged_since(f, 0) is False

    def test_unreadable_file_is_parsed_not_skipped(self, tmp_path):
        """Skipping is an optimization; a wrong skip loses learning silently, so
        every error path must fall back to parsing."""
        from neo.memory.transcript import _unchanged_since
        assert _unchanged_since(tmp_path / "gone.jsonl", time.time()) is False

    def test_codex_source_skips_only_stale_rollouts(self, tmp_path):
        import os
        sdir = tmp_path / "codex"
        sdir.mkdir()
        _write_rollout(sdir / "rollout-old.jsonl", cwd="/work/proj", sid="s1",
                       records=[_ev("user_message", message="stale")])
        _write_rollout(sdir / "rollout-new.jsonl", cwd="/work/proj", sid="s2",
                       records=[_ev("user_message", message="fresh")])
        old = time.time() - 10_000
        os.utime(sdir / "rollout-old.jsonl", (old, old))
        eps = CodexSource(codebase_root="/work/proj",
                          sessions_dir=sdir).collect_episodes(since=time.time() - 5_000)
        assert [e.ask for e in eps] == ["fresh"]

    def test_source_without_since_still_works(self, temp_store, tmp_path, monkeypatch):
        """Sources are an open interface — one lacking `since` must not be
        called with it and silently fail as 'source errored'."""
        monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
        ad = _StubAdapter([_LESSON], keep=True)
        legacy = _StaticSource([_ep("x1", "ask")])
        assert "since" not in __import__("inspect").signature(
            legacy.collect_episodes).parameters
        stats = TranscriptIngester(store=temp_store, lm_adapter=ad,
                                   sources=[legacy]).ingest(max_episodes=1)
        assert stats["episodes_total"] == 1


# --------------------------------------------------------------------------
# An LM failure is not "no lessons", and a backlog is not "already mined".
# Both used to advance the watermark: 610 DNS failures on a live observer each
# consumed an episode with zero lessons, and 152 of 1,800 live episodes sat in
# transcript files the mtime gate had stopped reading.
# --------------------------------------------------------------------------

class _FailingAdapter:
    """LM boundary that raises ``exc`` on every call."""

    def __init__(self, exc):
        self._exc = exc
        self.calls = 0

    def generate(self, messages, **kw):
        self.calls += 1
        raise self._exc


def _connection_error():
    import httpx
    import openai
    return openai.APIConnectionError(
        message="[Errno 8] nodename nor servname provided, or not known",
        request=httpx.Request("POST", "https://api.openai.com/v1/responses"))


def _fresh_ingester(store, adapter, src):
    """A NEW ingester each time, so every assertion about the watermark reads
    what was persisted to disk, not state held by the instance that wrote it."""
    return TranscriptIngester(store=store, lm_adapter=adapter, sources=[src])


def test_transient_lm_failure_does_not_consume_the_episode(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    src = _StaticSource([_ep("e1", "ask one"), _ep("e2", "ask two"), _ep("e3", "ask three")])

    down = _FailingAdapter(_connection_error())
    stats = _fresh_ingester(temp_store, down, src).ingest()
    assert stats["lm_unavailable"] is True
    assert down.calls == 2, "one failure proves nothing, a second with no answer stops the pass"
    assert _fresh_ingester(temp_store, down, src)._load_consumed(src) == set()
    assert _fresh_ingester(temp_store, down, src)._load_failures(src) == {}

    up = _StubAdapter([_LESSON], keep=True)
    stats = _fresh_ingester(temp_store, up, src).ingest()
    assert stats["episodes_processed"] == 3 and stats["facts_admitted"] == 3
    assert _fresh_ingester(temp_store, up, src)._load_consumed(src) == {"e1", "e2", "e3"}


def test_a_real_openai_503_outage_stops_without_charging(temp_store, tmp_path, monkeypatch):
    """Through the adapter neo actually runs, not a hand-built exception."""
    import httpx
    from neo.adapters import OpenAIAdapter
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    adapter = OpenAIAdapter(model="gpt-5.6", api_key="example-key")
    adapter.client = adapter.client.copy(http_client=httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(503, json={"error": {"message": "overloaded"}}))))
    src = _StaticSource([_ep("e1", "ask one"), _ep("e2", "ask two")])
    stats = _fresh_ingester(temp_store, adapter, src).ingest()
    assert stats["lm_unavailable"] is True
    ing = _fresh_ingester(temp_store, None, src)
    assert ing._load_consumed(src) == set() and ing._load_watermark(src)["transient_failures"] == {}


def test_provider_wide_rejection_never_writes_off_the_backlog(temp_store, tmp_path, monkeypatch):
    """A rotated key (401) is non-transient by class, but it fails EVERY
    episode. Charging per episode would abandon the whole backlog after three
    passes — about fifteen minutes of observer time."""
    import httpx
    import openai
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    req = httpx.Request("POST", "https://x/v1/responses")
    revoked = openai.AuthenticationError(
        "invalid key", response=httpx.Response(401, request=req), body=None)
    src = _StaticSource([_ep(f"e{i}", f"ask {i}") for i in range(5)])
    for _ in range(6):
        stats = _fresh_ingester(temp_store, _FailingAdapter(revoked), src).ingest()
        assert stats["lm_unavailable"] is True
    ing = _fresh_ingester(temp_store, None, src)
    assert ing._load_consumed(src) == set()
    assert ing._load_failures(src) == {}


def test_transient_classification_is_about_reachability_only():
    import httpx
    import openai
    from neo.memory.transcript import _is_transient_lm_error
    req = httpx.Request("POST", "https://x/v1/responses")

    def status(code):
        return openai.APIStatusError("e", response=httpx.Response(code, request=req), body=None)

    for code in (408, 429, 500, 502, 503, 504):
        assert _is_transient_lm_error(status(code)), code
    # Deterministic for the request: the per-episode cap handles these.
    for code in (400, 401, 404, 409, 413, 501, 505):
        assert not _is_transient_lm_error(status(code)), code
    assert _is_transient_lm_error(httpx.ConnectError("dns", request=req))
    class _GenaiError(Exception):  # google-genai's APIError carries `code`
        code = 429
    try:
        try:
            raise _GenaiError("RESOURCE_EXHAUSTED")
        except _GenaiError as inner:  # exactly how GoogleAdapter re-raises
            raise ValueError(f"Rate limit exceeded: {inner}")
    except ValueError as google_style:
        assert _is_transient_lm_error(google_style)
    wrapped = RuntimeError("adapter wrapper")
    wrapped.__cause__ = ConnectionResetError(54, "Connection reset by peer")
    assert _is_transient_lm_error(wrapped), "the cause chain is walked"
    assert not _is_transient_lm_error(ValueError("No completed message in response"))


def test_persistent_failure_is_abandoned_after_the_cap(temp_store, tmp_path, monkeypatch):
    """A poison episode (a 400 for an oversized prompt, a reply cut off by
    max_output_tokens) fails identically forever. Once the provider is known
    to be answering, it is charged and given up after MAX_EPISODE_LM_FAILURES,
    and the episodes behind it still get mined."""
    from neo.memory.transcript import MAX_EPISODE_LM_FAILURES
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    src = _StaticSource([_ep("bad", "the poison ask"), _ep("good", "a healthy ask")])

    class _PoisonAdapter(_StubAdapter):
        def generate(self, messages, **kw):
            if "the poison ask" in messages[0]["content"]:
                raise ValueError("No completed message in response: {'status': 'incomplete'}")
            return super().generate(messages, **kw)

    # Pass 1: the healthy episode's answer is what makes the held failure chargeable.
    stats = _fresh_ingester(temp_store, _PoisonAdapter([_LESSON]), src).ingest()
    assert stats["lm_unavailable"] is False
    assert _fresh_ingester(temp_store, None, src)._load_consumed(src) == {"good"}
    assert _fresh_ingester(temp_store, None, src)._load_failures(src) == {"bad": 1}

    # Later passes: "bad" is alone; the caller's evidence (an earlier project
    # answered this cycle) is what charges it.
    for attempt in range(2, MAX_EPISODE_LM_FAILURES + 1):
        _fresh_ingester(temp_store, _PoisonAdapter([_LESSON]), src).ingest(provider_known_good=True)
        consumed = _fresh_ingester(temp_store, None, src)._load_consumed(src)
        if attempt < MAX_EPISODE_LM_FAILURES:
            assert _fresh_ingester(temp_store, None, src)._load_failures(src) == {"bad": attempt}
    assert "bad" in consumed
    assert _fresh_ingester(temp_store, None, src)._load_failures(src) == {}


def test_lone_failure_without_evidence_is_not_charged(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    src = _StaticSource([_ep("only", "the only ask")])
    stats = _fresh_ingester(temp_store, _FailingAdapter(ValueError("bad request")), src).ingest()
    assert stats["lm_unavailable"] is False, "one failure is not an outage"
    assert _fresh_ingester(temp_store, None, src)._load_failures(src) == {}


def test_failure_during_verify_admits_nothing(temp_store, tmp_path, monkeypatch):
    """Extraction succeeded, then the judge call failed. Admitting the lessons
    verified so far and retrying the episode later re-extracts them in
    different LM wording, which exact-signature dedup does not catch — so an
    episode's lessons are admitted all together or not at all."""
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    two = [_LESSON, {**_LESSON, "subject": "second lesson", "body": "Another one."}]

    class _SecondJudgeFails(_StubAdapter):
        """Extraction and the first verify succeed; the second verify raises."""

        def generate(self, messages, **kw):
            if '"lessons"' not in messages[0]["content"]:
                self.verifies = getattr(self, "verifies", 0) + 1
                if self.verifies == 2:
                    raise _connection_error()
            return super().generate(messages, **kw)

    src = _StaticSource([_ep("e1", "ask one")])
    _fresh_ingester(temp_store, _SecondJudgeFails(two, keep=True), src).ingest()
    assert [f for f in temp_store._facts if f.is_valid] == []
    assert _fresh_ingester(temp_store, None, src)._load_consumed(src) == set()


def test_backlog_in_an_old_transcript_file_drains(temp_store, tmp_path, monkeypatch):
    """The real mtime gate, end to end. A session file last written days ago
    holds more episodes than one cycle's budget. The watermark file is
    rewritten as the first episodes are consumed, and the old gate — "skip
    files older than the watermark's mtime minus an hour" — then skipped the
    very file still holding the rest. Measured live: every one of 152
    unconsumed episodes was in such a file."""
    import os
    from neo.memory.transcript import ClaudeCodeSource
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    projects = tmp_path / "projects"
    monkeypatch.setattr("neo.memory.transcript.CLAUDE_PROJECTS_DIR", projects)
    root = "/work/proj"
    tdir = projects / root.replace("/", "-")
    tdir.mkdir(parents=True)
    records = []
    for i in range(3):
        records += [_user(f"u{i}", "s1", text=f"please fix bug number {i}"),
                    _assistant(f"a{i}", "s1", text="the venv was missing pytest-asyncio",
                               tools=["Bash"])]
    fp = tdir / "old-session.jsonl"
    _write(fp, records)
    three_days_ago = time.time() - 3 * 86400
    os.utime(fp, (three_days_ago, three_days_ago))

    src = ClaudeCodeSource(root)
    for _ in range(3):
        _fresh_ingester(temp_store, _StubAdapter([_LESSON], keep=True), src).ingest(max_episodes=1)
    consumed = _fresh_ingester(temp_store, None, src)._load_consumed(src)
    assert consumed == {"u0", "u1", "u2"}


def test_drain_mark_advances_while_a_backlog_persists(temp_store, tmp_path, monkeypatch):
    """The mark is the earliest episode still unconsumed, not "set when a pass
    finished everything": a project producing more episodes per rotation than
    the budget would otherwise never re-arm the skip and pay a full parse on
    every visit — the RSS cost the gate exists to avoid."""
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    seen_since = []

    class _SinceSource(_StaticSource):
        def collect_episodes(self, since=None):
            seen_since.append(since)
            return list(self._episodes)

    base = 1_700_000_000
    eps = []
    for i in range(4):
        ep = _ep(f"e{i}", f"ask {i}")
        ep.timestamp = datetime.datetime.fromtimestamp(
            base + i * 60, datetime.timezone.utc).isoformat().replace("+00:00", "Z")
        eps.append(ep)
    src = _SinceSource(eps)
    skew = TranscriptIngester._COLLECT_SKEW_SECONDS

    _fresh_ingester(temp_store, _StubAdapter([_LESSON]), src).ingest(max_episodes=1)
    _fresh_ingester(temp_store, _StubAdapter([_LESSON]), src).ingest(max_episodes=1)
    _fresh_ingester(temp_store, _StubAdapter([_LESSON]), src).ingest()
    before = time.time()
    _fresh_ingester(temp_store, _StubAdapter([_LESSON]), src).ingest()
    assert seen_since[0] is None
    assert seen_since[1] == pytest.approx(base + 60 - skew), "e0 consumed; e1 is earliest pending"
    assert seen_since[2] == pytest.approx(base + 120 - skew), "the mark moved with the backlog"
    assert seen_since[3] <= before and seen_since[3] > base + 180, "drained: mark is the pass start"


def test_unreadable_timestamp_keeps_the_previous_mark(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    seen_since = []

    class _SinceSource(_StaticSource):
        def collect_episodes(self, since=None):
            seen_since.append(since)
            return list(self._episodes)

    src = _SinceSource([_ep("e1", "ask one"), _ep("e2", "ask two")])  # timestamp "t"
    _fresh_ingester(temp_store, _StubAdapter([_LESSON]), src).ingest(max_episodes=1)
    _fresh_ingester(temp_store, _StubAdapter([_LESSON]), src).ingest(max_episodes=1)
    assert seen_since == [None, None], "cannot place e2 in time, so the skip must not arm"


def test_clones_sharing_a_watermark_keep_separate_drain_marks(temp_store, tmp_path, monkeypatch):
    """Two clones of one remote share a project_id and so a watermark file.
    One draining must not arm the skip against the other's transcripts."""
    import os
    from neo.memory.transcript import ClaudeCodeSource
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    projects = tmp_path / "projects"
    monkeypatch.setattr("neo.memory.transcript.CLAUDE_PROJECTS_DIR", projects)
    old = time.time() - 3 * 86400
    for root, n in (("/work/fms", 1), ("/work/fms2", 3)):
        tdir = projects / root.replace("/", "-")
        tdir.mkdir(parents=True)
        records = []
        for i in range(n):
            records += [_user(f"{root}-u{i}", "s", text=f"fix bug {i} in {root}"),
                        _assistant(f"{root}-a{i}", "s", text="the venv was missing pytest-asyncio",
                                   tools=["Bash"])]
        fp = tdir / "session.jsonl"
        _write(fp, records)
        os.utime(fp, (old, old))

    fms2 = ClaudeCodeSource("/work/fms2")
    fms = ClaudeCodeSource("/work/fms")
    _fresh_ingester(temp_store, _StubAdapter([_LESSON]), fms2).ingest(max_episodes=1)
    _fresh_ingester(temp_store, _StubAdapter([_LESSON]), fms).ingest()  # drains fms
    for _ in range(2):
        _fresh_ingester(temp_store, _StubAdapter([_LESSON]), fms2).ingest(max_episodes=1)
    consumed = _fresh_ingester(temp_store, None, fms2)._load_consumed(fms2)
    assert {f"/work/fms2-u{i}" for i in range(3)} <= consumed


def test_legacy_watermark_without_drain_mark_parses_everything(temp_store, tmp_path, monkeypatch):
    """Watermarks written before the drain mark existed carry only `consumed`.
    Their mtime is exactly the untrustworthy signal, so the first pass reads
    everything — which is also what recovers episodes already stranded."""
    sdir = tmp_path / "sessions"
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", sdir)
    src = _StaticSource([])
    ing = _fresh_ingester(temp_store, None, src)
    path = ing._watermark_path(src)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"consumed": ["old"]}), encoding="utf-8")
    assert ing._collected_through(src) is None
    assert ing._load_consumed(src) == {"old"}


def test_repeated_transient_failure_is_eventually_abandoned(temp_store, tmp_path, monkeypatch):
    """An episode whose own prompt always times out, while the provider keeps
    answering everything else. Uncapped it costs a full timeout every visit."""
    from neo.memory.transcript import MAX_EPISODE_TRANSIENT_FAILURES
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    src = _StaticSource([_ep("slow", "an ask that always times out")])
    for attempt in range(1, MAX_EPISODE_TRANSIENT_FAILURES + 1):
        _fresh_ingester(temp_store, _FailingAdapter(TimeoutError("read")), src).ingest(
            provider_known_good=True)
        consumed = _fresh_ingester(temp_store, None, src)._load_consumed(src)
        assert ("slow" in consumed) == (attempt == MAX_EPISODE_TRANSIENT_FAILURES)
    assert _fresh_ingester(temp_store, None, src)._load_watermark(src)["transient_failures"] == {}


def test_transient_attempts_do_not_spend_the_non_transient_budget(temp_store, tmp_path, monkeypatch):
    from neo.memory.transcript import MAX_EPISODE_LM_FAILURES
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    src = _StaticSource([_ep("e1", "ask one")])
    for _ in range(MAX_EPISODE_LM_FAILURES + 1):
        _fresh_ingester(temp_store, _FailingAdapter(_connection_error()), src).ingest(
            provider_known_good=True)
    ing = _fresh_ingester(temp_store, None, src)
    assert ing._load_consumed(src) == set()
    assert ing._load_failures(src) == {}
    assert ing._load_watermark(src)["transient_failures"] == {"e1": MAX_EPISODE_LM_FAILURES + 1}


def test_unparseable_output_is_a_failure_not_no_lessons(temp_store, tmp_path, monkeypatch):
    """A reply cut off mid-JSON used to parse to None, read as "no lessons",
    and consume the episode on the first pass."""
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")

    class _Truncated:
        def generate(self, messages, **kw):
            return '{"lessons": [{"kind": "pattern", "subject": "cut off mid'

    src = _StaticSource([_ep("e1", "ask one")])
    stats = _fresh_ingester(temp_store, _Truncated(), src).ingest()
    assert stats["episodes_failed"] == 1 and stats["lm_unavailable"] is False
    assert _fresh_ingester(temp_store, None, src)._load_consumed(src) == set()
    assert _fresh_ingester(temp_store, None, src)._load_failures(src) == {"e1": 1}


def test_naive_timestamp_does_not_advance_the_drain_mark():
    """Read as local time, a naive UTC stamp lands the mark hours late on a
    machine west of UTC — past the skew margin — and skips unconsumed files."""
    from neo.memory.transcript import _drain_mark
    naive = _ep("e1", "ask")
    naive.timestamp = "2026-01-01T00:00:00"
    assert _drain_mark(time.time(), [naive]) is None
    aware = _ep("e2", "ask")
    aware.timestamp = "2026-01-01T00:00:00Z"
    assert _drain_mark(time.time(), [aware]) == datetime.datetime(
        2026, 1, 1, tzinfo=datetime.timezone.utc).timestamp()


def _revoked_key():
    import httpx
    import openai
    req = httpx.Request("POST", "https://x/v1/responses")
    return openai.AuthenticationError(
        "invalid key", response=httpx.Response(401, request=req), body=None)


class _DiesAfter(_StubAdapter):
    """Answers normally for ``ok_calls`` calls, then raises ``exc`` forever."""

    def __init__(self, ok_calls, exc):
        super().__init__([_LESSON], keep=True)
        self._ok_calls, self._exc, self.n = ok_calls, exc, 0

    def generate(self, messages, **kw):
        self.n += 1
        if self.n > self._ok_calls:
            raise self._exc
        return super().generate(messages, **kw)


def test_provider_dying_mid_pass_stops_without_charging(temp_store, tmp_path, monkeypatch):
    """An answer early in a pass is not evidence about calls made after the
    provider went away: a key revoked after episode 1 must not put a strike on
    every remaining episode in the budget."""
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    src = _StaticSource([_ep(f"e{i}", f"ask {i}") for i in range(6)])
    ad = _DiesAfter(ok_calls=2, exc=_revoked_key())  # e0: extract + verify
    stats = _fresh_ingester(temp_store, ad, src).ingest()
    assert stats["lm_unavailable"] is True
    assert ad.n == 4, "e1 and e2 fail, then the pass stops"
    ing = _fresh_ingester(temp_store, None, src)
    assert ing._load_consumed(src) == {"e0"}
    assert ing._load_failures(src) == {}


def test_two_poison_episodes_after_known_answers_are_charged(temp_store, tmp_path, monkeypatch):
    """With an answer earlier this cycle, two failures in a row are likelier two
    bad episodes than an outage starting at that instant. Left uncharged they
    would stop the sweep on every visit."""
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    src = _StaticSource([_ep("p1", "poison one"), _ep("p2", "poison two")])
    stats = _fresh_ingester(temp_store, _FailingAdapter(ValueError("context too long")), src).ingest(
        provider_known_good=True)
    assert stats["lm_unavailable"] is True, "the pass still stops"
    assert _fresh_ingester(temp_store, None, src)._load_failures(src) == {"p1": 1, "p2": 1}


def test_a_charge_that_cannot_be_written_stops_the_pass(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")

    class _PoisonFirst(_StubAdapter):
        def __init__(self):
            super().__init__([_LESSON], keep=True)
            self.asks = []

        def generate(self, messages, **kw):
            content = messages[0]["content"]
            if '"lessons"' in content:
                self.asks.append(content)
            if "the poison ask" in content:
                raise ValueError("bad request")
            return super().generate(messages, **kw)

    def unwritable(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("neo.memory.transcript.atomic_write_json", unwritable)
    ad = _PoisonFirst()
    src = _StaticSource([_ep("bad", "the poison ask"), _ep("good", "a healthy ask"),
                         _ep("third", "a third ask")])
    _fresh_ingester(temp_store, ad, src).ingest()
    assert not any("a third ask" in a for a in ad.asks), \
        "no episode may be mined after progress stopped being recordable"


def test_non_substantive_success_is_not_an_answer(temp_store, tmp_path, monkeypatch):
    monkeypatch.setattr("neo.memory.transcript.SESSIONS_DIR", tmp_path / "sessions")
    hollow = Episode(session_id="s", anchor_uuid="hollow", last_uuid="hollow",
                     timestamp="t", ask="just asking")  # no assistant text: no LM call
    src = _StaticSource([_ep("f1", "ask one"), hollow, _ep("f2", "ask two")])
    stats = _fresh_ingester(temp_store, _FailingAdapter(_revoked_key()), src).ingest()
    assert stats["lm_unavailable"] is True
    assert _fresh_ingester(temp_store, None, src)._load_failures(src) == {}
