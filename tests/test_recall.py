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


class _Embedder:
    """Maps known prompts to fixed vectors; everything else to axis 7."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return iter([self.table.get(t, _unit(7)) for t in texts])


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
    text = recall("hook fails", str(tmp_path), embedder=_Embedder({"hook fails": _unit(0)}))
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
    assert recall("q", str(tmp_path), embedder=_Embedder({"q": _unit(0)})) == ""


def test_an_unrelated_prompt_gets_nothing(tmp_path):
    _seed(tmp_path, _fact("lesson", axis=0))
    near_miss = _unit(0) * (MIN_SIMILARITY - 0.05) + _unit(1)
    near_miss /= np.linalg.norm(near_miss)
    assert recall("q", str(tmp_path), embedder=_Embedder({"q": near_miss})) == ""


def test_at_most_three_lessons_best_first(tmp_path):
    lessons = [_fact(f"lesson {i}", axis=0) for i in range(5)]
    _seed(tmp_path, _fact("best", axis=1), *lessons)
    query = 0.8 * _unit(1) + 0.6 * _unit(0)  # best 0.8, the rest 0.6
    text = recall("q", str(tmp_path), embedder=_Embedder({"q": query / np.linalg.norm(query)}))
    rows = [line for line in text.splitlines() if line.startswith("- ")]
    assert len(rows) == 3
    assert rows[0].startswith("- best ")


def test_without_the_resident_embedder_it_never_loads_a_model(tmp_path, monkeypatch):
    from neo.memory import embed_service, store

    _seed(tmp_path, _fact("lesson"))
    monkeypatch.setattr(embed_service, "connect", lambda *a, **k: None)

    def _loaded(*a, **k):
        raise AssertionError("recall must never build the 1.3 GB model locally")

    monkeypatch.setattr(store, "_build_embedder", _loaded)
    assert recall("anything", str(tmp_path)) == ""


def test_the_fallback_it_hands_the_resident_embedder_loads_nothing(monkeypatch):
    from neo.memory import embed_service

    seen = {}
    monkeypatch.setattr(embed_service, "connect",
                        lambda model, local_factory: seen.setdefault("factory", local_factory))
    recall_mod._resident_embedder()
    assert seen["factory"]() is None


def test_the_store_is_not_modified(tmp_path):
    path = _seed(tmp_path, _fact("lesson", tags=["transcript-derived"]))
    before = path.read_bytes(), path.stat().st_mtime_ns
    recall("q", str(tmp_path), embedder=_Embedder({"q": _unit(0)}))
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


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
