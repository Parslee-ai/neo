"""Tests for the memory-file audit (neo.memory.memaudit). Model-free."""

import numpy as np

from neo.memory.memaudit import (
    MemoryEntry,
    audit_memories,
    parse_memory_file,
)

V_A = np.array([1.0, 0.0, 0.0], dtype=np.float32)
V_B = np.array([0.0, 1.0, 0.0], dtype=np.float32)
V_TABS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
# ~0.85 cosine with V_TABS: aligned (>=0.80) but below DUP (0.93) -> conflict candidate.
V_SPACES = np.array([0.0, np.sqrt(1 - 0.85**2), 0.85], dtype=np.float32)


class _FakeLM:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, messages, max_tokens=None, temperature=None):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _entry(name, *, emb=None, mtype="project", desc="desc", body="body", malformed=None):
    return MemoryEntry(
        path=f"/{name}.md", filename=f"{name}.md", name=name, description=desc,
        mtype=mtype, body=body, links=[], malformed=list(malformed or []), embedding=emb,
    )


# --------------------------------------------------------------------------
# parse_memory_file
# --------------------------------------------------------------------------


def test_parse_valid(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("---\nname: x\ndescription: a desc\nmetadata:\n  type: feedback\n---\nbody here\n")
    e = parse_memory_file(p)
    assert e.name == "x"
    assert e.mtype == "feedback"
    assert e.description == "a desc"
    assert e.body == "body here"
    assert e.malformed == []


def test_parse_missing_frontmatter(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("just a body, no frontmatter")
    e = parse_memory_file(p)
    assert "missing frontmatter" in e.malformed


def test_parse_invalid_type(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("---\nname: x\ndescription: d\nmetadata:\n  type: bogus\n---\nb")
    e = parse_memory_file(p)
    assert any("invalid type" in m for m in e.malformed)


def test_parse_missing_description(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("---\nname: x\nmetadata:\n  type: project\n---\nb")
    e = parse_memory_file(p)
    assert "missing description" in e.malformed


def test_parse_wikilinks(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("---\nname: x\ndescription: d\nmetadata:\n  type: project\n---\nsee [[other]] and [[third]]")
    e = parse_memory_file(p)
    assert e.links == ["other", "third"]


# --------------------------------------------------------------------------
# audit_memories
# --------------------------------------------------------------------------


def test_audit_reports_malformed():
    rep = audit_memories([_entry("bad", malformed=["missing description"])])
    assert ("bad.md", "missing description") in rep.malformed


def test_audit_detects_duplicates():
    rep = audit_memories([_entry("a", emb=V_A), _entry("b", emb=V_A)])
    assert len(rep.duplicates) == 1
    assert set(rep.duplicates[0].names) == {"a", "b"}


def test_audit_distinct_memories_no_duplicate():
    rep = audit_memories([_entry("a", emb=V_A), _entry("b", emb=V_B)])
    assert rep.duplicates == []


def test_audit_detects_conflict_with_lm():
    lm = _FakeLM('{"conflict": true, "explanation": "tabs vs spaces"}')
    rep = audit_memories([_entry("tabs", emb=V_TABS), _entry("spaces", emb=V_SPACES)], lm_adapter=lm)
    assert len(rep.conflicts) == 1
    assert rep.conflicts[0].explanation == "tabs vs spaces"
    assert rep.duplicates == []  # 0.85 < DUP threshold


def test_audit_no_lm_no_conflicts():
    rep = audit_memories([_entry("tabs", emb=V_TABS), _entry("spaces", emb=V_SPACES)], lm_adapter=None)
    assert rep.conflicts == []


def test_audit_conflict_lm_failure_graceful():
    rep = audit_memories(
        [_entry("tabs", emb=V_TABS), _entry("spaces", emb=V_SPACES)],
        lm_adapter=_FakeLM(RuntimeError("down")),
    )
    assert rep.conflicts == []


def test_audit_index_issues():
    rep = audit_memories(
        [_entry("a")], index_targets={"b.md"}, existing_filenames={"a.md"}
    )
    assert any("a.md is not listed" in s for s in rep.index_issues)
    assert any("missing file b.md" in s for s in rep.index_issues)


def test_audit_clean_when_no_issues():
    rep = audit_memories([_entry("a", emb=V_A)])
    assert rep.clean


def test_dangling_wikilinks_not_flagged():
    e = _entry("a", emb=V_A)
    e.links = ["does-not-exist"]  # intentional per memory spec
    rep = audit_memories([e])
    assert rep.clean


# --------------------------------------------------------------------------
# find_memory_audit (orchestration)
# --------------------------------------------------------------------------


def test_find_memory_audit_end_to_end(tmp_path, monkeypatch):
    import neo.memory.memaudit as ma

    fm = "---\nname: {n}\ndescription: use pytest\nmetadata:\n  type: project\n---\nuse pytest always\n"
    (tmp_path / "a.md").write_text(fm.format(n="a"))
    (tmp_path / "b.md").write_text(fm.format(n="b"))  # duplicate of a
    (tmp_path / "MEMORY.md").write_text("- [A](a.md) — x\n")  # b.md not listed

    monkeypatch.setattr(ma, "resolve_memory_dir", lambda root: tmp_path)

    class _Store:
        codebase_root = "/x"

        def _embed_text(self, text):
            return V_A  # everything identical -> duplicate

    rep = ma.find_memory_audit(_Store(), check_conflicts=False)
    assert rep.entry_count == 2
    assert len(rep.duplicates) == 1
    assert any("b.md is not listed" in s for s in rep.index_issues)


def test_find_memory_audit_no_dir(monkeypatch):
    import neo.memory.memaudit as ma

    monkeypatch.setattr(ma, "resolve_memory_dir", lambda root: None)

    class _Store:
        codebase_root = "/x"

    rep = ma.find_memory_audit(_Store())
    assert "no memory directory" in rep.note
    assert rep.clean
