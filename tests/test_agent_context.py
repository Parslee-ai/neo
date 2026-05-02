"""Tests for neo.agent_context — discovery of AI-tool instruction docs."""

from pathlib import Path

from neo.agent_context import (
    AgentDoc,
    PER_FILE_CAP_BYTES,
    TOTAL_CAP_BYTES,
    discover,
    format_for_prompt,
)


def _write(root: Path, rel: str, content: str = "stub guidance") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Root-level files
# ---------------------------------------------------------------------------

class TestRootFiles:
    def test_claude_md_at_root(self, tmp_path: Path):
        _write(tmp_path, "CLAUDE.md", "## Project rules\nUse type hints.")
        docs = discover(tmp_path)
        assert any(d.path == "CLAUDE.md" and d.source == "claude" for d in docs)

    def test_agents_md_at_root(self, tmp_path: Path):
        _write(tmp_path, "AGENTS.md", "Agents read this.")
        docs = discover(tmp_path)
        assert any(d.path == "AGENTS.md" and d.source == "agents" for d in docs)

    def test_legacy_cursorrules(self, tmp_path: Path):
        _write(tmp_path, ".cursorrules", "Use 2-space indent")
        docs = discover(tmp_path)
        assert any(d.source == "cursor" for d in docs)


# ---------------------------------------------------------------------------
# Dot-directory ecosystems
# ---------------------------------------------------------------------------

class TestDotDirectories:
    def test_claude_subdir_md(self, tmp_path: Path):
        _write(tmp_path, ".claude/CLAUDE.md", "in-dir version")
        _write(tmp_path, ".claude/agents/reviewer.md", "review style")
        docs = discover(tmp_path)
        sources = {d.source for d in docs}
        assert "claude" in sources
        paths = {d.path for d in docs}
        assert ".claude/CLAUDE.md" in paths
        assert ".claude/agents/reviewer.md" in paths

    def test_cursor_modern_rules_dir(self, tmp_path: Path):
        _write(tmp_path, ".cursor/rules/python.md", "follow PEP8")
        _write(tmp_path, ".cursor/rules/api.mdc", "REST conventions")
        docs = discover(tmp_path)
        cursor_paths = {d.path for d in docs if d.source == "cursor"}
        assert ".cursor/rules/python.md" in cursor_paths
        assert ".cursor/rules/api.mdc" in cursor_paths

    def test_copilot_instructions(self, tmp_path: Path):
        _write(tmp_path, ".github/copilot-instructions.md", "Copilot guidance.")
        docs = discover(tmp_path)
        assert any(d.source == "copilot" for d in docs)

    def test_specify_recursive(self, tmp_path: Path):
        # Spec Kit produces nested spec docs.
        _write(tmp_path, ".specify/auth/login.md", "Spec for login flow.")
        docs = discover(tmp_path)
        assert any(d.source == "specify" and "auth/login.md" in d.path for d in docs)


# ---------------------------------------------------------------------------
# Robustness / edge cases
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_none_root_returns_empty(self):
        assert discover(None) == []

    def test_missing_root_returns_empty(self, tmp_path: Path):
        assert discover(tmp_path / "does-not-exist") == []

    def test_empty_root_returns_empty(self, tmp_path: Path):
        assert discover(tmp_path) == []

    def test_string_root_accepted(self, tmp_path: Path):
        _write(tmp_path, "CLAUDE.md", "x")
        docs = discover(str(tmp_path))
        assert len(docs) == 1

    def test_per_file_cap_truncates(self, tmp_path: Path):
        big = "x" * (PER_FILE_CAP_BYTES * 2)
        _write(tmp_path, "CLAUDE.md", big)
        docs = discover(tmp_path)
        assert len(docs) == 1
        assert docs[0].content.endswith("[truncated]")
        # Original content + suffix; size cap honored on the source slice.
        assert len(docs[0].content) <= PER_FILE_CAP_BYTES + 32

    def test_total_cap_stops_early(self, tmp_path: Path):
        # Drop enough docs to exceed TOTAL_CAP_BYTES; discovery should stop.
        big = "y" * PER_FILE_CAP_BYTES
        _write(tmp_path, "CLAUDE.md", big)
        _write(tmp_path, "AGENTS.md", big)
        # Cursor rules: many files, each at the per-file cap.
        for i in range(20):
            _write(tmp_path, f".cursor/rules/r{i:02d}.md", big)

        docs = discover(tmp_path)
        total = sum(d.size() for d in docs)
        assert total <= TOTAL_CAP_BYTES
        # We always include at least the first couple before hitting the cap.
        assert len(docs) >= 2

    def test_no_duplicate_when_pattern_overlaps(self, tmp_path: Path):
        # `.cursor/rules/*.md` and `.cursor/rules/**/*.md` both match top-level
        # files; we should de-dupe.
        _write(tmp_path, ".cursor/rules/x.md", "a")
        docs = discover(tmp_path)
        cursor = [d for d in docs if d.source == "cursor"]
        assert len(cursor) == 1


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

class TestFormatForPrompt:
    def test_empty_returns_empty_string(self):
        assert format_for_prompt([]) == ""

    def test_includes_header_and_per_doc_paths(self):
        docs = [
            AgentDoc(path="CLAUDE.md", source="claude", content="rule A"),
            AgentDoc(path=".cursor/rules/x.md", source="cursor", content="rule B"),
        ]
        out = format_for_prompt(docs)
        assert "PROJECT-LOCAL AGENT CONTEXT" in out
        assert "CLAUDE.md" in out and "(source: claude)" in out
        assert ".cursor/rules/x.md" in out and "(source: cursor)" in out
        assert "rule A" in out and "rule B" in out


# ---------------------------------------------------------------------------
# End-to-end: realistic mixed project
# ---------------------------------------------------------------------------

def test_end_to_end_mixed_project(tmp_path: Path):
    _write(tmp_path, "CLAUDE.md", "Always run pytest before committing.")
    _write(tmp_path, "AGENTS.md", "All agents must respect the lint config.")
    _write(tmp_path, ".cursor/rules/python.md", "Use Black formatting.")
    _write(tmp_path, ".cursor/rules/api/rest.mdc", "Use kebab-case URLs.")
    _write(tmp_path, ".github/copilot-instructions.md", "Prefer named exports.")
    _write(tmp_path, ".github/AGENTS.md", "PR titles use conventional commits.")
    _write(tmp_path, ".specify/feature/auth.md", "Auth must use OAuth 2.")

    docs = discover(tmp_path)
    sources = {d.source for d in docs}
    # All five ecosystems represented in the discovery.
    assert {"claude", "agents", "cursor", "copilot", "specify"} <= sources

    rendered = format_for_prompt(docs)
    assert "Always run pytest" in rendered
    assert "Use Black formatting" in rendered
    assert "OAuth 2" in rendered
