"""`neo recall`: per-prompt memory for the Claude Code mod.

It runs on every user prompt, so the tests pin the three promises that make
that affordable: it never loads a model, never mutates the store, and only
returns lessons the model does not already have in context.
"""

import io
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
    """Maps known prompts to fixed vectors; everything else to axis 7."""
    return lambda text: table.get(text, _unit(7))


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
    text = recall("hook fails", str(tmp_path), embed=_embed({"hook fails": _unit(0)}))
    assert text.startswith("Neo memory:")
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
    assert recall("q", str(tmp_path), embed=_embed({"q": _unit(0)})) == ""


def test_an_unrelated_prompt_gets_nothing(tmp_path):
    _seed(tmp_path, _fact("lesson", axis=0))
    near_miss = _unit(0) * (MIN_SIMILARITY - 0.05) + _unit(1)
    near_miss /= np.linalg.norm(near_miss)
    assert recall("q", str(tmp_path), embed=_embed({"q": near_miss})) == ""


def test_at_most_three_lessons_best_first(tmp_path):
    lessons = [_fact(f"lesson {i}", axis=0) for i in range(5)]
    _seed(tmp_path, _fact("best", axis=1), *lessons)
    query = 0.8 * _unit(1) + 0.6 * _unit(0)  # best 0.8, the rest 0.6
    text = recall("q", str(tmp_path), embed=_embed({"q": query / np.linalg.norm(query)}))
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
    assert recall("anything", str(tmp_path)) == ""


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
    recall("q", str(tmp_path), embed=_embed({"q": _unit(0)}))
    assert _tree(neo_home) == before


def test_legacy_suggestion_facts_are_not_injected(tmp_path):
    """A cold start would retire these; recall never runs one."""
    legacy = Fact(subject="feature: add caching", body="Reasoning: because",
                  kind=FactKind.DECISION, scope=FactScope.PROJECT,
                  tags=["feature"], embedding=_unit(0))
    _seed(tmp_path, legacy)
    assert recall("q", str(tmp_path), embed=_embed({"q": _unit(0)})) == ""


def test_a_fact_from_another_embedding_model_is_skipped_not_fatal(tmp_path):
    foreign = _fact("foreign", axis=0)
    foreign.embedding = np.ones(DIM * 2, dtype=np.float32)
    _seed(tmp_path, foreign, _fact("lesson", axis=0))
    text = recall("q", str(tmp_path), embed=_embed({"q": _unit(0)}))
    assert "- lesson " in text and "foreign" not in text


def test_low_confidence_facts_are_not_injected(tmp_path):
    weak = _fact("weak")
    weak.metadata.confidence = 0.2
    _seed(tmp_path, weak)
    assert recall("q", str(tmp_path), embed=_embed({"q": _unit(0)})) == ""


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
