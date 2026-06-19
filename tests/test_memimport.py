"""Tests for memory ingestion (neo.memory.memimport).

Uses real temp memory files (exercising the real parser) with a fake store and
a redirected watermark dir, so it stays model-free and deterministic.
"""

import types

import neo.memory.memimport as mi
from neo.memory.memimport import import_memory
from neo.memory.models import FactKind, FactScope, Provenance


class _FakeStore:
    project_id = "testpid"
    codebase_root = "/x"

    def __init__(self):
        self._facts = []
        self.added = []

    def add_fact(self, **kwargs):
        self.added.append(kwargs)
        f = types.SimpleNamespace(id=str(len(self._facts)))
        self._facts.append(f)
        return f


def _write(d, name, *, desc="a useful description", body="the body",
           mtype="project", frontmatter=True):
    p = d / f"{name}.md"
    if frontmatter:
        fm = f"---\nname: {name}\n"
        if desc:
            fm += f"description: {desc}\n"
        fm += f"metadata:\n  type: {mtype}\n---\n{body}\n"
        p.write_text(fm, encoding="utf-8")
    else:
        p.write_text(body, encoding="utf-8")
    return p


def _setup(tmp_path, monkeypatch):
    mem = tmp_path / "memory"
    mem.mkdir()
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setattr(mi, "resolve_memory_dir", lambda root: mem)
    monkeypatch.setattr("neo.memory.outcomes.SESSIONS_DIR", sessions)
    return mem, sessions


def test_import_maps_to_review_inferred_probation(tmp_path, monkeypatch):
    mem, _ = _setup(tmp_path, monkeypatch)
    _write(mem, "a", mtype="feedback")
    store = _FakeStore()
    stats = import_memory(store)
    assert stats.imported == 1
    kw = store.added[0]
    assert kw["kind"] == FactKind.REVIEW            # decaying kind, not CONSTRAINT
    assert kw["provenance"] == Provenance.INFERRED   # -> probation, lowest trust
    assert kw["scope"] == FactScope.PROJECT
    assert "imported" in kw["tags"]
    assert "imported:claude-memory" in kw["tags"]
    assert "memtype:feedback" in kw["tags"]


def test_import_skips_malformed(tmp_path, monkeypatch):
    mem, _ = _setup(tmp_path, monkeypatch)
    _write(mem, "good")
    _write(mem, "nodesc", desc="")          # missing description
    _write(mem, "nofm", frontmatter=False)  # missing frontmatter
    store = _FakeStore()
    stats = import_memory(store)
    assert stats.imported == 1
    assert stats.skipped_malformed == 2


def test_memory_md_index_is_not_imported(tmp_path, monkeypatch):
    mem, _ = _setup(tmp_path, monkeypatch)
    _write(mem, "a")
    (mem / "MEMORY.md").write_text("- [A](a.md) — x\n", encoding="utf-8")
    store = _FakeStore()
    stats = import_memory(store)
    assert stats.imported == 1  # MEMORY.md skipped


def test_dry_run_does_not_mutate(tmp_path, monkeypatch):
    mem, sessions = _setup(tmp_path, monkeypatch)
    _write(mem, "a")
    _write(mem, "b")
    store = _FakeStore()
    stats = import_memory(store, dry_run=True)
    assert stats.imported == 2
    assert store.added == []
    assert not list(sessions.glob("memimport_*"))


def test_watermark_makes_reruns_idempotent(tmp_path, monkeypatch):
    mem, _ = _setup(tmp_path, monkeypatch)
    _write(mem, "a")
    _write(mem, "b")

    first = import_memory(_FakeStore())
    assert first.imported == 2

    store2 = _FakeStore()
    second = import_memory(store2)
    assert second.imported == 0
    assert second.skipped_existing == 2
    assert store2.added == []


def test_edited_memory_reimports(tmp_path, monkeypatch):
    mem, _ = _setup(tmp_path, monkeypatch)
    _write(mem, "a", body="version one")
    import_memory(_FakeStore())

    _write(mem, "a", body="version two edited")  # content changed -> new hash
    store = _FakeStore()
    stats = import_memory(store)
    assert stats.imported == 1


def test_no_memory_dir_noop(monkeypatch):
    monkeypatch.setattr(mi, "resolve_memory_dir", lambda root: None)
    stats = import_memory(_FakeStore())
    assert stats.imported == 0
    assert "no memory directory" in stats.note
