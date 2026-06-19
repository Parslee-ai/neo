"""Tests for the rule-file sync diagnostic (neo.memory.rulesync).

Model-free: fixed embedding vectors and a fake LM, hand-built rule files.
"""

import math

import numpy as np

from neo.memory.rulesync import (
    RuleFile,
    analyze_rule_sync,
    discover_rule_files,
    find_rule_sync,
    parse_units,
)

V_PYTEST = np.array([1.0, 0.0, 0.0], dtype=np.float32)
V_RUFF = np.array([0.0, 1.0, 0.0], dtype=np.float32)
V_TABS = np.array([0.0, 0.0, 1.0], dtype=np.float32)
# ~0.9 cosine with V_TABS: aligned (>=0.78) but not identical (<0.97) -> conflict candidate.
V_SPACES = np.array([0.0, math.sqrt(1 - 0.81), 0.9], dtype=np.float32)


class _FakeLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, messages, max_tokens=None, temperature=None):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class _NoEmbedStore:
    codebase_root = "/tmp/x"

    def _embed_text(self, text):
        return None


class _EmbStore:
    codebase_root = "/tmp/x"

    def _embed_text(self, text):
        t = text.lower()
        if "pytest" in t:
            return V_PYTEST
        if "ruff" in t:
            return V_RUFF
        return V_TABS


# --------------------------------------------------------------------------
# parse_units
# --------------------------------------------------------------------------


def test_parse_units_folds_continuations_drops_prose_and_fences():
    text = (
        "# Heading\n"
        "- Use pytest, never unittest\n"
        "  under tests/ only\n"
        "- Run ruff before every commit\n"
        "\n"
        "This is top-level prose, not a rule.\n"
        "```\n"
        "code block contents\n"
        "```\n"
    )
    units = parse_units(text)
    assert any("Use pytest" in u and "under tests" in u for u in units)  # continuation folded
    assert any("Run ruff before every commit" in u for u in units)
    assert all("top-level prose" not in u for u in units)
    assert all("code block" not in u for u in units)


def test_parse_units_skips_trivial_fragments():
    assert parse_units("- ok\n- short") == []  # both < 8 chars


# --------------------------------------------------------------------------
# discover_rule_files
# --------------------------------------------------------------------------


def test_discover_rule_files_case_insensitive(tmp_path):
    (tmp_path / "AGENTS.md").write_text("- a rule here")
    (tmp_path / "claude.md").write_text("- another rule")  # lowercase
    (tmp_path / "README.md").write_text("nope")
    found = dict((t, p) for t, p in discover_rule_files(str(tmp_path)))
    assert sorted(found) == ["agents", "claude"]


def test_discover_rule_files_missing_dir():
    assert discover_rule_files("/no/such/dir/xyz") == []


# --------------------------------------------------------------------------
# analyze_rule_sync — gaps
# --------------------------------------------------------------------------


def test_analyze_detects_gap():
    agents = RuleFile("agents", "A", ["use pytest for tests"], [V_PYTEST])
    claude = RuleFile(
        "claude", "C",
        ["use pytest for tests", "run ruff before commit"],
        [V_PYTEST, V_RUFF],
    )
    rep = analyze_rule_sync([agents, claude])
    assert len(rep.gaps) == 1
    assert "ruff" in rep.gaps[0].rule
    assert rep.gaps[0].missing_from == ["agents"]
    assert "claude" in rep.gaps[0].present_in
    assert "AGENTS.md" in rep.gaps[0].suggestion


def test_analyze_no_gap_when_aligned():
    agents = RuleFile("agents", "A", ["use pytest for tests"], [V_PYTEST])
    claude = RuleFile("claude", "C", ["use pytest for tests"], [V_PYTEST])
    rep = analyze_rule_sync([agents, claude])
    assert rep.gaps == []
    assert rep.in_sync


def test_analyze_single_file_noop():
    rep = analyze_rule_sync([RuleFile("agents", "A", ["use pytest for tests"], [V_PYTEST])])
    assert rep.in_sync
    assert "single" in rep.note


# --------------------------------------------------------------------------
# analyze_rule_sync — conflicts (LM-judged)
# --------------------------------------------------------------------------


def test_analyze_detects_conflict_with_lm():
    agents = RuleFile("agents", "A", ["use tabs for indentation"], [V_TABS])
    claude = RuleFile("claude", "C", ["use spaces for indentation"], [V_SPACES])
    lm = _FakeLM('{"conflict": true, "explanation": "tabs vs spaces"}')
    rep = analyze_rule_sync([agents, claude], lm_adapter=lm)
    assert len(rep.conflicts) == 1
    assert rep.conflicts[0].explanation == "tabs vs spaces"
    assert rep.gaps == []  # aligned pair is not a gap
    assert "Reconcile" in rep.conflicts[0].suggestion


def test_analyze_lm_says_no_conflict():
    agents = RuleFile("agents", "A", ["use tabs for indentation"], [V_TABS])
    claude = RuleFile("claude", "C", ["use spaces for indentation"], [V_SPACES])
    rep = analyze_rule_sync([agents, claude], lm_adapter=_FakeLM('{"conflict": false}'))
    assert rep.conflicts == []


def test_analyze_no_lm_no_conflicts():
    agents = RuleFile("agents", "A", ["use tabs for indentation"], [V_TABS])
    claude = RuleFile("claude", "C", ["use spaces for indentation"], [V_SPACES])
    rep = analyze_rule_sync([agents, claude], lm_adapter=None)
    assert rep.conflicts == []


def test_analyze_conflict_lm_failure_graceful():
    agents = RuleFile("agents", "A", ["use tabs for indentation"], [V_TABS])
    claude = RuleFile("claude", "C", ["use spaces for indentation"], [V_SPACES])
    rep = analyze_rule_sync([agents, claude], lm_adapter=_FakeLM(RuntimeError("down")))
    assert rep.conflicts == []  # not fatal


# --------------------------------------------------------------------------
# find_rule_sync — orchestration
# --------------------------------------------------------------------------


def test_find_identical_files_in_sync(tmp_path):
    same = "- Use pytest for tests\n- Run ruff before commit\n"
    (tmp_path / "AGENTS.md").write_text(same)
    (tmp_path / "CLAUDE.md").write_text(same)
    rep = find_rule_sync(_NoEmbedStore(), root=str(tmp_path))
    assert rep.in_sync
    assert "identical" in rep.note


def test_find_single_file_in_sync(tmp_path):
    (tmp_path / "AGENTS.md").write_text("- Use pytest for tests\n")
    rep = find_rule_sync(_NoEmbedStore(), root=str(tmp_path))
    assert rep.in_sync
    assert "single" in rep.note


def test_find_rule_sync_gap_end_to_end(tmp_path):
    (tmp_path / "AGENTS.md").write_text("- Use pytest for tests\n")
    (tmp_path / "CLAUDE.md").write_text("- Use pytest for tests\n- Run ruff before commit\n")
    rep = find_rule_sync(_EmbStore(), root=str(tmp_path), check_conflicts=False)
    assert not rep.in_sync
    assert len(rep.gaps) == 1
    assert "ruff" in rep.gaps[0].rule
    assert rep.gaps[0].missing_from == ["agents"]
