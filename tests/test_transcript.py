"""Tests for Claude Code transcript parsing (Stage A) and ingestion (B/C)."""

import json

import pytest

from neo.memory.models import FactKind
from neo.memory.transcript import (
    Episode,
    TranscriptIngester,
    _parse_json,
    build_episodes,
    collect_episodes,
    resolve_transcript_dir,
)


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
