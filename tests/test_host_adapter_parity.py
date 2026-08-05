"""Parity tests for Neo's host adapters.

Neo publishes one host-neutral contract (`neo.events` + `OrchestratorMessage`)
and thin per-host adapters that tell an orchestrator how to consume it:

    .claude-plugin/   -> Claude Code agent + slash commands
    plugins/neo/      -> Codex CLI plugin + skills

Adapters drift. The Codex skills sat on "parse the four structured sections"
for a full release after the Claude side moved to `--json`, which is exactly
the failure these tests exist to catch: a contract change landing in one
adapter and silently not the other.

Manifest versions drift the same way. `prepare-release` documents bumping both
plugin manifests, but a documented step with no enforcement is a step that gets
skipped — the two manifests and the package had reached three different
versions.
"""

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CLAUDE_PLUGIN = REPO / ".claude-plugin"
CODEX_PLUGIN = REPO / "plugins" / "neo"

# The six capabilities Neo claims on every integration surface.
CAPABILITIES = ["neo", "neo-review", "neo-debug", "neo-architect",
                "neo-optimize", "neo-pattern"]


def _package_version() -> str:
    for line in (REPO / "pyproject.toml").read_text().splitlines():
        if line.startswith("version = "):
            return line.split("=", 1)[1].strip().strip('"')
    raise AssertionError("no version in pyproject.toml")


def _codex_skill(name: str) -> str:
    return (CODEX_PLUGIN / "skills" / name / "SKILL.md").read_text()


def _claude_command(name: str) -> str:
    return (CLAUDE_PLUGIN / "commands" / f"{name}.md").read_text()


# ------------------------------------------------------------ both exist


def test_both_integration_surfaces_are_checked_in():
    """The README claims three surfaces. Two of them are directories here, and
    a claim in a README that no directory backs is a claim that will be found
    false by a user, not by us."""
    assert (CLAUDE_PLUGIN / "plugin.json").is_file()
    assert (CODEX_PLUGIN / ".codex-plugin" / "plugin.json").is_file()


@pytest.mark.parametrize("name", CAPABILITIES)
def test_every_capability_exists_on_both_surfaces(name):
    assert (CODEX_PLUGIN / "skills" / name / "SKILL.md").is_file(), f"codex: {name}"
    assert (CLAUDE_PLUGIN / "commands" / f"{name}.md").is_file(), f"claude: {name}"


def test_neither_surface_carries_extra_capabilities():
    """'The same six skills' has to stay six on both sides."""
    codex = {p.name for p in (CODEX_PLUGIN / "skills").iterdir() if p.is_dir()}
    claude = {p.stem for p in (CLAUDE_PLUGIN / "commands").glob("*.md")}
    assert codex == set(CAPABILITIES)
    assert claude == set(CAPABILITIES)


def test_local_marketplace_points_at_the_codex_plugin():
    """`codex plugin marketplace add ./` is a documented install path; a stale
    path here breaks it silently."""
    manifest = json.loads((REPO / ".agents" / "plugins" / "marketplace.json").read_text())
    paths = [p["source"]["path"] for p in manifest["plugins"]]
    assert "./plugins/neo" in paths
    for path in paths:
        assert (REPO / path).is_dir(), path


# ------------------------------------------------------------ versions


def test_plugin_manifests_match_the_package_version():
    version = _package_version()
    for manifest in (CLAUDE_PLUGIN / "plugin.json",
                     CODEX_PLUGIN / ".codex-plugin" / "plugin.json"):
        assert json.loads(manifest.read_text())["version"] == version, manifest


def test_dunder_version_matches_the_package_version():
    from neo import __version__

    assert __version__ == _package_version()


# ------------------------------------------------------------ the contract


@pytest.mark.parametrize("name", CAPABILITIES)
def test_codex_skills_invoke_neo_with_json(name):
    assert "neo --json" in _codex_skill(name), name


@pytest.mark.parametrize("name", CAPABILITIES)
def test_codex_skills_teach_the_orchestrator_envelope(name):
    body = _codex_skill(name)
    assert "orchestrator.summary" in body, name
    assert "orchestrator.cautions" in body, name


@pytest.mark.parametrize("name", CAPABILITIES)
def test_no_surface_still_instructs_parsing_the_text_output(name):
    """The specific regression: Codex was told to read `CONFIDENCE`, `PLAN`,
    `SIMULATIONS`, `CODE SUGGESTIONS` out of terminal-formatted prose."""
    for body in (_codex_skill(name), _claude_command(name)):
        assert "four structured sections" not in body, name


def test_claude_agent_and_codex_entry_skill_teach_the_same_contract():
    """Both entry points must name the same stream split, the same fields, and
    the same failure shape. Wording differs; the contract may not."""
    agent = (CLAUDE_PLUGIN / "agents" / "neo.md").read_text()
    skill = _codex_skill("neo")
    for token in ("--json", "stdout", "stderr", "orchestrator.summary",
                  "orchestrator.cautions", "personality", "context",
                  "reasoning", "static_checks"):
        assert token in agent, f"claude agent missing {token}"
        assert token in skill, f"codex skill missing {token}"


def test_both_entry_points_document_the_error_shape():
    """A host that reads `orchestrator` before checking `error` reports a
    successful run that never happened."""
    agent = (CLAUDE_PLUGIN / "agents" / "neo.md").read_text()
    skill = _codex_skill("neo")
    for body in (agent, skill):
        assert '"error"' in body


def test_both_entry_points_require_attribution():
    """Neo's conclusions and the host's own analysis must stay separable."""
    agent = (CLAUDE_PLUGIN / "agents" / "neo.md").read_text()
    skill = _codex_skill("neo")
    for body in (agent, skill):
        assert "Neo found" in body


def test_documented_phase_names_match_the_code():
    """Phase names are a compatibility surface; docs that drift from `events.py`
    send hosts looking for a phase that never fires."""
    from neo import events

    agent = (CLAUDE_PLUGIN / "agents" / "neo.md").read_text()
    skill = _codex_skill("neo")
    for phase in (events.PHASE_CONTEXT, events.PHASE_REASONING,
                  events.PHASE_STATIC_CHECKS):
        assert phase in agent, phase
        assert phase in skill, phase


def test_documented_event_types_exist_in_the_code():
    from neo.events import NeoEventType

    skill = _codex_skill("neo")
    documented = [t for t in ("memory_found", "hypothesis_formed",
                              "hypothesis_rejected", "risk_found",
                              "completed", "failed") if t in skill]
    assert documented, "the entry skill documents no event types"
    valid = {member.value for member in NeoEventType}
    for name in documented:
        assert name in valid, name
