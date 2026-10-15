"""`neo recall`: per-prompt memory for the Claude Code mod.

It runs on every user prompt, so the tests pin the three promises that make
that affordable: it never loads a model, never mutates the store, and only
returns lessons the model does not already have in context.
"""

import io
import json
from pathlib import Path

import numpy as np

from neo import recall as recall_mod
from neo.memory.models import Fact, FactKind, FactScope
from neo.memory.store import FactStore
from neo.recall import MIN_SIMILARITY, recall, run_recall

DIM = 8


def _unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _embed(table):
    """Maps known texts to fixed vectors; everything else to axis 7. Batched,
    like the resident embedder."""
    return lambda texts: [table.get(text, _unit(7)) for text in texts]


def _seed(root: Path, *facts: Fact) -> Path:
    store = FactStore(codebase_root=str(root), eager_init=False)
    store._facts.extend(facts)
    store.save()
    return store._project_path


def _fact(subject, *, axis=0, kind=FactKind.PATTERN, tags=(), body="Do the careful thing."):
    return Fact(subject=subject, body=body, kind=kind, scope=FactScope.PROJECT,
                tags=list(tags), embedding=_unit(axis))


def test_a_matching_learned_lesson_is_returned(tmp_path):
    _seed(tmp_path, _fact("Verify unrelated hook failures", tags=["transcript-derived"]))
    text = recall("hook fails", str(tmp_path), embed=_embed({"hook fails": _unit(0)}))["text"]
    assert text.startswith("Neo:")
    assert "- Verify unrelated hook failures [pattern, project]: Do the careful thing." in text


def test_what_the_model_already_has_is_never_returned(tmp_path):
    # Every one of these is a perfect match, so only provenance can exclude it.
    _seed(
        tmp_path,
        _fact("commit subject", tags=["history"]),
        _fact("memory file", tags=["claude-memory", "auto-ingested"]),
        _fact("textbook", tags=["seed"]),
        _fact("feed", tags=["community"]),
        _fact("outcome:modified x.py", tags=["outcome", "modified"]),
        _fact("rule file line", kind=FactKind.CONSTRAINT),
        _fact("Simulation: trace", kind=FactKind.EPISODE),
    )
    assert recall("q", str(tmp_path), embed=_embed({"q": _unit(0)}))["text"] == ""


def test_an_unrelated_prompt_gets_nothing(tmp_path):
    _seed(tmp_path, _fact("lesson", axis=0))
    near_miss = _unit(0) * (MIN_SIMILARITY - 0.05) + _unit(1)
    near_miss /= np.linalg.norm(near_miss)
    assert recall("q", str(tmp_path), embed=_embed({"q": near_miss}))["text"] == ""


def test_at_most_three_lessons_best_first(tmp_path):
    lessons = [_fact(f"lesson {i}", axis=0) for i in range(5)]
    _seed(tmp_path, _fact("best", axis=1), *lessons)
    query = 0.8 * _unit(1) + 0.6 * _unit(0)  # best 0.8, the rest 0.6
    text = recall("q", str(tmp_path), embed=_embed({"q": query / np.linalg.norm(query)}))["text"]
    rows = [line for line in text.splitlines() if line.startswith("- ")]
    assert len(rows) == 3
    assert rows[0].startswith("- best ")


def test_without_the_resident_embedder_it_never_loads_a_model(tmp_path, monkeypatch):
    from neo.memory import embed_service, store

    _seed(tmp_path, _fact("lesson"))
    monkeypatch.setattr(embed_service, "socket_path", lambda: tmp_path / "absent.sock")

    def _loaded(*a, **k):
        raise AssertionError("recall must never build the 1.3 GB model locally")

    monkeypatch.setattr(store, "_build_embedder", _loaded)
    assert recall("anything", str(tmp_path))["text"] == ""


def _tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): (p.stat().st_mtime_ns, p.read_bytes() if p.is_file() else b"")
        for p in sorted(root.rglob("*"))
    }


def test_nothing_under_the_neo_home_is_written(tmp_path):
    """Constructing a FactStore writes outside save() (mkdir, legacy-name
    migration, corrupt-file backup), so recall must not construct one; and
    it must not append a metrics line per prompt."""
    _seed(tmp_path, _fact("lesson", tags=["transcript-derived"]))
    neo_home = Path.home() / ".neo"
    (neo_home / "facts" / "facts_global.json").write_text("{not json")
    before = _tree(neo_home)
    recall("q", str(tmp_path), embed=_embed({"q": _unit(0)}))["text"]
    assert _tree(neo_home) == before


def test_legacy_suggestion_facts_are_not_injected(tmp_path):
    """A cold start would retire these; recall never runs one."""
    legacy = Fact(subject="feature: add caching", body="Reasoning: because",
                  kind=FactKind.DECISION, scope=FactScope.PROJECT,
                  tags=["feature"], embedding=_unit(0))
    _seed(tmp_path, legacy)
    assert recall("q", str(tmp_path), embed=_embed({"q": _unit(0)}))["text"] == ""


def test_a_fact_from_another_embedding_model_is_skipped_not_fatal(tmp_path):
    foreign = _fact("foreign", axis=0)
    foreign.embedding = np.ones(DIM * 2, dtype=np.float32)
    _seed(tmp_path, foreign, _fact("lesson", axis=0))
    text = recall("q", str(tmp_path), embed=_embed({"q": _unit(0)}))["text"]
    assert "- lesson " in text and "foreign" not in text


def test_low_confidence_facts_are_not_injected(tmp_path):
    weak = _fact("weak")
    weak.metadata.confidence = 0.2
    _seed(tmp_path, weak)
    assert recall("q", str(tmp_path), embed=_embed({"q": _unit(0)}))["text"] == ""


def test_the_command_never_fails_the_host(monkeypatch, capsys):
    def _boom(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(recall_mod, "recall", _boom)
    assert run_recall([], stdin=io.StringIO("prompt")) == 0
    assert run_recall(["--no-such-flag"], stdin=io.StringIO("prompt")) == 0
    assert capsys.readouterr().out == ""


def test_recall_stays_off_the_slow_path():
    """Like the hook: dispatched before anything `cli.main` builds."""
    source = Path(recall_mod.__file__).with_name("cli.py").read_text()
    body = source.split("def main():", 1)[1]
    dispatch = body.index('sys.argv[1] == "recall"')
    for expensive in ("FactStore(", "check_for_updates", "maybe_autostart_observer", "parse_args()"):
        assert expensive not in body[:dispatch], expensive


# ------------------------------------------------------------- Neo's voice


def _deck():
    import yaml

    return yaml.safe_load(
        (Path(recall_mod.__file__).parent / "config" / "beats" / "neo_matrix.yaml").read_text())


def _beat(beat_id):
    return next(b for b in _deck()["recall_beats"] if b["beat_id"] == beat_id)


def test_the_situation_picks_the_beat(tmp_path):
    """A recalled failure speaks as a failure, not as a generic memory."""
    _seed(tmp_path, _fact("Beware shell glob failures", kind=FactKind.FAILURE,
                          tags=["transcript-derived"]))
    result = recall("q", str(tmp_path), embed=_embed({"q": _unit(0)}))
    assert result["beat_id"] == "recall_failure_early"
    assert result["text"].splitlines()[0] == f"Neo: {_beat('recall_failure_early')['line']}"


def test_a_beat_spoken_recently_is_not_repeated(tmp_path):
    _seed(tmp_path, _fact("Beware shell glob failures", kind=FactKind.FAILURE,
                          tags=["transcript-derived"]))
    embed = _embed({"q": _unit(0)})
    first = recall("q", str(tmp_path), embed=embed)
    again = recall("q", str(tmp_path), embed=embed, recent_beats={first["beat_id"]})
    assert again["beat_id"] != first["beat_id"]
    # The lesson itself is still delivered; only the line changes or goes quiet.
    assert "Beware shell glob failures" in again["text"]


def test_with_nothing_distinctive_neo_stays_quiet(tmp_path):
    """Silence is the default: a middling match of an ordinary lesson."""
    _seed(tmp_path, _fact("lesson", tags=["transcript-derived"]))
    mid = 0.55 * _unit(0) + np.sqrt(1 - 0.55 ** 2) * _unit(1)
    result = recall("q", str(tmp_path), embed=_embed({"q": mid}))
    assert result["beat_id"] == ""
    assert result["text"].splitlines()[0] == "Neo:"


def test_the_beats_available_change_as_his_memory_grows(monkeypatch, tmp_path):
    """Same situation, different stage: the hedge gives way to certainty."""
    from neo.memory import store

    _seed(tmp_path, _fact("lesson", tags=["transcript-derived"]))
    embed = _embed({"q": _unit(0)})
    spoken = {}
    for stage in (1, 3, 5):
        monkeypatch.setattr(store, "memory_stage", lambda level, s=stage: s)
        spoken[stage] = recall("q", str(tmp_path), embed=embed)["beat_id"]
    assert spoken == {1: "recall_deja_vu", 3: "recall_remember", 5: "recall_know"}


def test_every_recall_beat_is_reachable_and_well_formed():
    triggers = {"failure_lesson", "proven_lesson", "elsewhere", "strong_match", "faint_match"}
    beats = _deck()["recall_beats"]
    assert len({b["beat_id"] for b in beats}) == len(beats)
    for beat in beats:
        assert beat["line"].strip()
        assert set(beat["stages"]) <= {1, 2, 3, 4, 5} and beat["stages"]
        assert set(beat["trigger_contexts"]) <= triggers


def test_recall_beats_never_enter_the_engine_deck():
    assert not {b["beat_id"] for b in _deck()["beats"]} & {
        b["beat_id"] for b in _deck()["recall_beats"]}


def test_recall_and_the_engine_share_one_stage_rule():
    from neo.engine import NeoEngine
    from neo.memory.store import memory_stage

    for level in (0.0, 0.19, 0.2, 0.45, 0.6, 0.79, 0.8, 0.99):
        assert NeoEngine._memory_level_to_stage(None, level) == memory_stage(level)


def test_the_json_form_reports_the_beat(tmp_path, monkeypatch, capsys):
    import json as _json

    monkeypatch.setattr(recall_mod, "recall",
                        lambda p, c, recent_beats=None, context="", **_: {"text": "Neo: x", "beat_id": "b1"})
    assert run_recall(["--json", "--recent-beats", "a,b"], stdin=io.StringIO("q")) == 0
    assert _json.loads(capsys.readouterr().out) == {"text": "Neo: x", "beat_id": "b1"}


def test_delivered_fact_ids_are_returned(tmp_path):
    lesson = _fact("Verify unrelated hook failures", tags=["transcript-derived"])
    _seed(tmp_path, lesson)
    result = recall("q", str(tmp_path), embed=_embed({"q": _unit(0)}))
    assert result["fact_ids"] == [lesson.id]
    assert recall("q", str(tmp_path), embed=_embed({"q": _unit(3)}))["fact_ids"] == []


def test_a_lesson_this_session_already_has_is_not_sent_again(tmp_path):
    """Re-sent on every matching prompt, one lesson reached a single session 43
    times. What the session was handed stays in its context (rule 3)."""
    lessons = [_fact(f"lesson {i}", axis=0, tags=["transcript-derived"]) for i in range(5)]
    _seed(tmp_path, *lessons)
    embed = _embed({"q": _unit(0)})

    first = recall("q", str(tmp_path), embed=embed)["fact_ids"]
    again = recall("q", str(tmp_path), embed=embed, seen=frozenset(first))
    partly = recall("q", str(tmp_path), embed=embed, seen=frozenset(first[:1]))["fact_ids"]

    assert len(first) == recall_mod.MAX_FACTS
    assert again == {"text": "", "beat_id": "", "fact_ids": []}
    # No backfill: a freed slot never goes to a weaker lesson, which would earn
    # deliveries (and probation lift) it never earned before.
    assert partly == first[1:]


def test_the_command_reads_what_the_session_has_from_the_environment(tmp_path, monkeypatch):
    """A variable, not a flag, for the same reason as the transcript."""
    got = {}
    monkeypatch.setenv("NEO_RECALL_SEEN", "a,,b")
    monkeypatch.setattr(recall_mod, "recall",
                        lambda p, c, recent_beats=None, context="", seen=frozenset():
                        got.update(seen=seen) or {"text": "", "beat_id": "", "fact_ids": []})
    assert run_recall(["--json"], stdin=io.StringIO("q")) == 0
    assert got["seen"] == frozenset({"a", "b"})


def test_a_matching_note_passage_is_offered_with_its_path(tmp_path):
    from neo.memory.store import DEFAULT_EMBED_MODEL
    from neo.notes import refresh_index

    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "observer.md").write_text("- The drain mark gates re-parsing, not the watermark file's mtime, which moves.\n")
    (tmp_path / "CLAUDE.md").write_text("Read `docs/observer.md` first.\n")
    refresh_index(tmp_path, lambda text: _unit(2), DEFAULT_EMBED_MODEL)

    result = recall("q", str(tmp_path), embed=_embed({"q": _unit(2)}))
    assert "- docs/observer.md [note]: - The drain mark gates re-parsing" in result["text"]
    assert result["fact_ids"] == []
    assert recall("q", str(tmp_path), embed=_embed({"q": _unit(5)}))["text"] == ""


def _transcript(path: Path, *records) -> str:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(path)


def _reply(text, *, sidechain=False):
    return {"type": "assistant", "isSidechain": sidechain,
            "message": {"content": [{"type": "text", "text": text}]}}


def test_a_short_follow_up_is_read_with_the_reply_it_answers(tmp_path):
    """"fix it" names nothing; the reply it answers does."""
    _seed(tmp_path, _fact("Sequence dependent pull request merges",
                          tags=["transcript-derived"]))
    reply = "Merge #303 first, then rebase #304 onto it."
    embed = _embed({f"{reply}\n\nfix it": _unit(0)})

    assert recall("fix it", str(tmp_path), embed=embed)["text"] == ""
    text = recall("fix it", str(tmp_path), embed=embed, context=reply)["text"]
    assert "Sequence dependent pull request merges" in text


def test_a_prompt_that_says_enough_ignores_the_reply(tmp_path):
    _seed(tmp_path, _fact("Sequence dependent pull request merges",
                          tags=["transcript-derived"]))
    prompt = " ".join(["word"] * (recall_mod.FOLLOW_UP_MAX_WORDS + 1))
    reply = "Merge #303 first."
    seen = []

    def embed(texts):
        seen.extend(texts)
        return [_unit(7) for _ in texts]

    recall(prompt, str(tmp_path), embed=embed, context=reply)
    assert seen == [prompt]


def test_only_the_tail_of_a_long_reply_is_used(tmp_path):
    seen = []

    def embed(texts):
        seen.extend(texts)
        return [_unit(7) for _ in texts]

    reply = "x" * 5000 + "Should I merge it?"
    recall("yes", str(tmp_path), embed=embed, context=reply)
    assert seen[1].endswith("Should I merge it?\n\nyes")
    assert len(seen[1]) == recall_mod.CONTEXT_TAIL_CHARS + len("\n\nyes")


def test_notes_are_matched_on_the_prompt_alone(tmp_path, monkeypatch):
    queries = []
    monkeypatch.setattr(recall_mod, "_note_hits",
                        lambda root, query: queries.append(query) or [])
    embed = _embed({"yes": _unit(1), "Merge it?\n\nyes": _unit(2)})
    recall("yes", str(tmp_path), embed=embed, context="Merge it?")
    assert np.array_equal(queries[0], _unit(1))


def test_last_reply_is_the_last_main_thread_assistant_text(tmp_path):
    from neo.recall import last_reply
    path = _transcript(
        tmp_path / "t.jsonl",
        _reply("an older reply"),
        _reply("the reply being answered"),
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "Bash"}]}},
        _reply("a subagent's words", sidechain=True),
        {"type": "user", "message": {"content": "fix it"}},
    )
    assert last_reply(path) == "the reply being answered"


def test_last_reply_never_fails(tmp_path):
    from neo.recall import last_reply
    assert last_reply("") == ""
    assert last_reply(str(tmp_path / "missing.jsonl")) == ""
    garbage = tmp_path / "g.jsonl"
    garbage.write_bytes(b"\xff\xfe not json\n{\"type\": \"assistant\"}\n")
    assert last_reply(str(garbage)) == ""


def test_the_command_reads_the_transcript_from_the_environment(tmp_path, monkeypatch):
    """A variable, not a flag: an older CLI exits 2 on an unknown flag, and the
    mod reads exit 2 as "no recall here" for the rest of the session."""
    path = _transcript(tmp_path / "t.jsonl", _reply("Merge it?"))
    seen = {}
    monkeypatch.setenv("NEO_RECALL_TRANSCRIPT", path)
    monkeypatch.setattr(recall_mod, "recall",
                        lambda p, c, recent_beats=None, context="", **_: seen.update(context=context)
                        or {"text": "", "beat_id": "", "fact_ids": []})
    assert run_recall(["--json"], stdin=io.StringIO("yes")) == 0
    assert seen["context"] == "Merge it?"


def test_context_never_displaces_a_lesson_the_prompt_matched_alone(tmp_path):
    """Scoring each lesson by its better match is not enough: three lessons the
    reply raised would outrank the prompt's own and push it out of the slots.
    Measured: 3 of 57 real follow-ups lost a lesson that way."""
    _seed(tmp_path,
          _fact("Verify service shutdown", axis=0, tags=["transcript-derived"]),
          *[_fact(f"Reply lesson {i}", axis=1, tags=["transcript-derived"])
            for i in range(3)])
    reply = "I'll stop the server when you're done."
    near = _unit(0) * 0.5 + _unit(7) * np.sqrt(0.75)  # cosine 0.5 to axis 0
    embed = _embed({"kill the server": near, f"{reply}\n\nkill the server": _unit(1)})

    text = recall("kill the server", str(tmp_path), embed=embed, context=reply)["text"]

    assert "Verify service shutdown" in text
    assert text.count("Reply lesson") == recall_mod.MAX_FACTS - 1


def test_a_twelve_word_prompt_is_still_a_follow_up(tmp_path):
    seen = []

    def embed(texts):
        seen.extend(texts)
        return [_unit(7) for _ in texts]

    prompt = " ".join(["word"] * recall_mod.FOLLOW_UP_MAX_WORDS)
    recall(prompt, str(tmp_path), embed=embed, context="Merge it?")
    assert len(seen) == 2


def test_a_short_answer_from_the_embedder_is_silence(tmp_path):
    _seed(tmp_path, _fact("lesson", tags=["transcript-derived"]))
    out = recall("yes", str(tmp_path), embed=lambda texts: [_unit(0)], context="Merge it?")
    assert out["text"] == ""


def test_claude_codes_own_error_lines_are_not_the_reply(tmp_path):
    """Recorded as assistant messages from model "<synthetic>"; a "continue"
    after one answers the reply before it, not the network error."""
    from neo.recall import last_reply
    error = {"type": "assistant", "message": {
        "model": "<synthetic>",
        "content": [{"type": "text", "text": "API Error: Can't reach the API server"}]}}
    path = _transcript(tmp_path / "t.jsonl", _reply("Should I merge it?"), error)
    assert last_reply(path) == "Should I merge it?"


def test_a_reply_with_a_line_separator_is_still_found(tmp_path):
    """Claude Code writes U+2028 unescaped; `str.splitlines` cut the record."""
    from neo.recall import last_reply
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in (
        _reply("an older reply"), _reply("first\u2028second"))) + "\n", encoding="utf-8")
    assert last_reply(str(path)) == "first\u2028second"


def test_a_reply_beyond_the_tail_window_is_not_read(tmp_path, monkeypatch):
    from neo.recall import last_reply
    monkeypatch.setattr(recall_mod, "TRANSCRIPT_TAIL_BYTES", 4096)
    filler = {"type": "user", "message": {"content": "x" * 8192}}
    path = _transcript(tmp_path / "t.jsonl", _reply("too far back"), filler)
    assert last_reply(path) == ""


def test_a_malformed_record_is_skipped_not_fatal(tmp_path):
    from neo.recall import last_reply
    path = _transcript(tmp_path / "t.jsonl", _reply("the reply"),
                       {"type": "assistant", "message": "not an object"},
                       {"type": "assistant", "message": {"content": [{"type": "text", "text": 7}]}})
    assert last_reply(path) == "the reply"


def test_a_long_prompt_never_reads_the_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv("NEO_RECALL_TRANSCRIPT", str(tmp_path / "t.jsonl"))
    monkeypatch.setattr(recall_mod, "last_reply",
                        lambda path: (_ for _ in ()).throw(AssertionError("read")))
    seen = {}
    monkeypatch.setattr(recall_mod, "recall",
                        lambda p, c, recent_beats=None, context="", **_: seen.update(context=context)
                        or {"text": "", "beat_id": "", "fact_ids": []})
    prompt = " ".join(["word"] * (recall_mod.FOLLOW_UP_MAX_WORDS + 1))
    assert run_recall(["--json"], stdin=io.StringIO(prompt)) == 0
    assert seen["context"] == ""
