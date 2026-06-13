"""Tests for Claude Code transcript parsing (Stage A)."""

import json

from neo.memory.transcript import (
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
