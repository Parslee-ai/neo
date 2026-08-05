"""Tests for the version-propagation tool.

`pyproject.toml` holds the release version; `src/neo/__init__.py` and the two
plugin manifests restate it. `prepare-release` used to ask a human to edit all
four, which is how the package, the Claude manifest and the Codex manifest
reached 0.41.0 / 0.37.0 / 0.19.0. This tool makes three of the four derived.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "sync_version.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("sync_version", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A miniature repo with the same four files, so tests never write to the
    real tree."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "neo-reasoner"\nversion = "1.2.3"\n'
    )
    pkg = tmp_path / "src" / "neo"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text('"""doc."""\n\n__version__ = "0.0.1"\n')
    claude = tmp_path / ".claude-plugin"
    claude.mkdir()
    (claude / "plugin.json").write_text(
        '{\n  "name": "neo",\n  "version": "0.0.2",\n  "description": "x"\n}\n'
    )
    codex = tmp_path / "plugins" / "neo" / ".codex-plugin"
    codex.mkdir(parents=True)
    (codex / "plugin.json").write_text(
        '{\n  "name": "neo",\n  "version": "0.0.3",\n  "description": "y"\n}\n'
    )

    tool = _load_tool()
    monkeypatch.setattr(tool, "REPO", tmp_path)
    monkeypatch.setattr(tool, "PYPROJECT", tmp_path / "pyproject.toml")
    monkeypatch.setattr(tool, "DERIVED", [
        (tmp_path / "src" / "neo" / "__init__.py", tool.DERIVED[0][1]),
        (claude / "plugin.json", tool.DERIVED[1][1]),
        (codex / "plugin.json", tool.DERIVED[2][1]),
    ])
    return tool, tmp_path


def test_source_version_reads_pyproject(fake_repo):
    tool, _ = fake_repo
    assert tool.source_version() == "1.2.3"


def test_sync_rewrites_every_derived_file(fake_repo):
    tool, root = fake_repo
    stale = tool.sync("1.2.3", check_only=False)

    assert len(stale) == 3
    assert '__version__ = "1.2.3"' in (root / "src/neo/__init__.py").read_text()
    for manifest in (root / ".claude-plugin/plugin.json",
                     root / "plugins/neo/.codex-plugin/plugin.json"):
        assert json.loads(manifest.read_text())["version"] == "1.2.3"


def test_check_mode_reports_without_writing(fake_repo):
    tool, root = fake_repo
    before = (root / ".claude-plugin/plugin.json").read_text()

    stale = tool.sync("1.2.3", check_only=True)

    assert len(stale) == 3
    assert (root / ".claude-plugin/plugin.json").read_text() == before


def test_sync_is_idempotent_and_leaves_formatting_alone(fake_repo):
    """Surgical replacement, not re-serialization: a version bump must stay a
    one-line diff, or it becomes unreviewable."""
    tool, root = fake_repo
    manifest = root / ".claude-plugin/plugin.json"

    tool.sync("1.2.3", check_only=False)
    first = manifest.read_text()
    assert tool.sync("1.2.3", check_only=False) == []
    assert manifest.read_text() == first
    # Original hand-written layout survives.
    assert first.startswith('{\n  "name": "neo",\n  "version": "1.2.3",')


def test_a_missing_version_field_is_an_error_not_a_silent_skip(fake_repo):
    tool, root = fake_repo
    (root / ".claude-plugin/plugin.json").write_text('{\n  "name": "neo"\n}\n')

    with pytest.raises(SystemExit):
        tool.sync("1.2.3", check_only=True)


def test_only_the_first_version_field_is_rewritten(fake_repo):
    """A nested "version" added later must not be clobbered by a release bump."""
    tool, root = fake_repo
    manifest = root / ".claude-plugin/plugin.json"
    manifest.write_text(
        '{\n  "name": "neo",\n  "version": "0.0.2",\n'
        '  "engine": {\n  "version": "9.9.9"\n  }\n}\n'
    )

    tool.sync("1.2.3", check_only=False)

    data = json.loads(manifest.read_text())
    assert data["version"] == "1.2.3"
    assert data["engine"]["version"] == "9.9.9"


# ------------------------------------------------------- against the real tree


def test_the_real_repository_is_in_sync():
    """Same invariant test_host_adapter_parity asserts, reached through the
    tool a human actually runs."""
    tool = _load_tool()
    assert tool.sync(tool.source_version(), check_only=True) == []


def test_release_skill_no_longer_asks_for_hand_edits():
    """The process gap, not just its symptom: a documented four-file edit with
    no enforcement is a step that gets skipped."""
    skill = (REPO / ".agents" / "skills" / "source-command-prepare-release"
             / "SKILL.md").read_text()
    assert "make sync-version" in skill
    assert "Do **not** hand-edit" in skill
    # The old Step 4 listed each derived file as its own "Change ..." edit.
    assert 'plugin.json`: Change `"version"' not in skill
