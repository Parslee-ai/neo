"""Parity tests for Neo's host adapters.

Neo publishes one host-neutral contract (`neo.events` + `OrchestratorMessage`)
and thin per-host adapters that tell an orchestrator how to consume it:

    .claude-plugin/    -> Claude Code manifests (plugin.json, marketplace.json)
    agents/ commands/  -> Claude Code components, at the PLUGIN ROOT
    plugins/neo/       -> Codex CLI plugin + skills

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
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
# Claude Code discovers components at the PLUGIN ROOT and reads only manifests
# out of `.claude-plugin/`. This plugin's root is the repository itself
# (`marketplace.json` declares `"source": "./"`), so those are two different
# directories and must not hide behind one name — which is how the components
# sat somewhere Claude Code never looks.
CLAUDE_MANIFEST = REPO / ".claude-plugin"
CLAUDE_ROOT = REPO
CODEX_PLUGIN = REPO / "plugins" / "neo"

# What is allowed to live in the manifest directory.
CLAUDE_MANIFEST_FILES = {"plugin.json", "marketplace.json"}

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


def _claude_agent() -> str:
    return (CLAUDE_ROOT / "agents" / "neo.md").read_text()


def _claude_command(name: str) -> str:
    return (CLAUDE_ROOT / "commands" / f"{name}.md").read_text()


# ------------------------------------------------------------ both exist


def test_both_integration_surfaces_are_checked_in():
    """The README claims three surfaces. Two of them are directories here, and
    a claim in a README that no directory backs is a claim that will be found
    false by a user, not by us."""
    assert (CLAUDE_MANIFEST / "plugin.json").is_file()
    assert (CODEX_PLUGIN / ".codex-plugin" / "plugin.json").is_file()


def test_claude_components_live_at_the_plugin_root():
    """Claude Code reads components from the plugin root and manifests from
    `.claude-plugin/`. Nested inside `.claude-plugin/` they are simply not
    found — the plugin installs, `claude plugin validate` passes, and it loads
    nothing. `claude plugin details` is the check that proves otherwise."""
    assert (CLAUDE_ROOT / "agents" / "neo.md").is_file()
    assert (CLAUDE_ROOT / "commands").is_dir()


def test_the_manifest_directory_holds_no_components():
    """The complement of the test above, and the half that actually holds.

    Asserting components exist at the root proves only that they are *also*
    there. A copy left behind in `.claude-plugin/` keeps every other assertion
    in this file green while Claude Code loads neither — which is how the
    layout stayed wrong for as long as it did. CAR shipped the identical
    mistake in its own repo, so this is pinned rather than left to review.
    """
    stray = sorted(
        entry.name
        for entry in CLAUDE_MANIFEST.iterdir()
        if entry.name not in CLAUDE_MANIFEST_FILES and not entry.name.startswith(".")
    )
    assert not stray, (
        f".claude-plugin/ holds manifests only; found {stray}. Component "
        "directories (agents/, commands/, skills/, hooks/) belong at the "
        "plugin root, which is the repository root."
    )


@pytest.mark.parametrize("name", CAPABILITIES)
def test_every_capability_exists_on_both_surfaces(name):
    assert (CODEX_PLUGIN / "skills" / name / "SKILL.md").is_file(), f"codex: {name}"
    assert (CLAUDE_ROOT / "commands" / f"{name}.md").is_file(), f"claude: {name}"


def test_neither_surface_carries_extra_capabilities():
    """'The same six skills' has to stay six on both sides."""
    codex = {p.name for p in (CODEX_PLUGIN / "skills").iterdir() if p.is_dir()}
    claude = {p.stem for p in (CLAUDE_ROOT / "commands").glob("*.md")}
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
    for manifest in (CLAUDE_MANIFEST / "plugin.json",
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
def test_codex_skills_disable_implicit_directory_scan(name):
    """Codex curates explicit context; a second CLI scan can leak unrelated files."""
    body = _codex_skill(name)
    assert "neo --json --no-scan" in body, name
    assert "`--no-scan` is mandatory" in body, name


@pytest.mark.parametrize("name", CAPABILITIES)
def test_codex_skills_disclose_external_provider_data_scope(name):
    """Network approval must identify both the recipient and the payload class."""
    body = _codex_skill(name)
    for phrase in ("external-provider", "which Neo provider", "data categories",
                   "approval flow"):
        assert phrase in body, f"{name}: missing {phrase}"
    assert re.search(r"explicit\s+authorization", body), name


def test_codex_entry_skill_rejects_blanket_sensitive_context_consent():
    body = _codex_skill("neo")
    assert "not blanket consent" in body
    for secret_class in ("secrets", "credentials", "tokens", "cookies",
                         "session"):
        assert secret_class in body


@pytest.mark.parametrize("name", [name for name in CAPABILITIES if name != "neo-pattern"])
def test_codex_advise_skills_disable_implicit_memory(name):
    body = _codex_skill(name)
    assert "neo --json --no-scan --no-memory --mode advise" in body, name
    assert "stored facts" in body, name


def test_codex_learning_skill_discloses_memory_retrieval_and_persistence():
    body = _codex_skill("neo-pattern")
    assert "relevant stored facts" in body
    assert "episode candidate" in body


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
    agent = _claude_agent()
    skill = _codex_skill("neo")
    for token in ("--json", "stdout", "stderr", "orchestrator.summary",
                  "orchestrator.cautions", "personality", "context",
                  "reasoning", "static_checks"):
        assert token in agent, f"claude agent missing {token}"
        assert token in skill, f"codex skill missing {token}"


def test_both_entry_points_document_the_error_shape():
    """A host that reads `orchestrator` before checking `error` reports a
    successful run that never happened."""
    agent = _claude_agent()
    skill = _codex_skill("neo")
    for body in (agent, skill):
        assert '"error"' in body


def test_both_entry_points_teach_the_null_confidence_contract():
    """`confidence` became `Optional[float]` with a `confidence_basis` (#199).

    An LLM host handed `null` under a standing instruction to "report Neo's
    confidence as the number it is" has no guidance, and the failure mode is
    the exact one #199 fixed: treating a run with no number as less trustworthy
    than an empty patch that self-reported 0.96. Both surfaces must name the
    null case and both basis values, or one host relays a contract the other
    does not.
    """
    from neo.models import (
        CONFIDENCE_BASIS_ANALYSIS_ONLY,
        CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE,
    )

    agent = _claude_agent()
    skill = _codex_skill("neo")
    for body in (agent, skill):
        assert "confidence_basis" in body
        assert "null" in body
        assert CONFIDENCE_BASIS_ANALYSIS_ONLY in body
        assert CONFIDENCE_BASIS_NO_VERIFIABLE_CHANGE in body


def test_both_entry_points_require_attribution():
    """Neo's conclusions and the host's own analysis must stay separable."""
    agent = _claude_agent()
    skill = _codex_skill("neo")
    for body in (agent, skill):
        assert "Neo found" in body


def test_documented_phase_names_match_the_code():
    """Phase names are a compatibility surface; docs that drift from `events.py`
    send hosts looking for a phase that never fires."""
    from neo import events

    agent = _claude_agent()
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


# ------------------------------------------------------------ descriptions


def _description(body: str) -> str:
    """The `description:` a host reads when deciding whether to invoke.

    Parsed as YAML rather than pattern-matched, because that is how the host
    reads it — and an unquoted plain scalar containing ": " is a parse error,
    not a long description. One was written during this change and caught here.
    """
    return yaml.safe_load(body.split("---")[1])["description"]


@pytest.mark.parametrize("name", CAPABILITIES)
def test_both_surfaces_describe_when_to_invoke_and_when_not(name):
    """A description that names no trigger is a component that never fires.

    All six read as restatements of their own title — "Get architectural
    guidance from Neo on design decisions" tells a model what the command is
    and gives it no basis for choosing one. `claude plugin details` inventories
    every one of them as model-invocable, so the trigger is the whole interface.

    The *negative* half is what is pinned here rather than merely encouraged.
    Each invocation costs 5-30s and an LLM call, so a description that says
    when to reach for Neo and never when not to buys reach with spend. It is
    also the half most likely to be dropped when someone shortens a line.
    """
    for surface, body in (("codex", _codex_skill(name)), ("claude", _claude_command(name))):
        description = _description(body)
        assert "Skip" in description, (
            f"{surface}:{name} description names no condition NOT to invoke Neo"
        )


@pytest.mark.parametrize("name", CAPABILITIES)
def test_every_description_is_readable_by_the_host(name):
    """Both hosts parse this frontmatter as YAML. A description that raises is
    indistinguishable from a component that does not exist."""
    for surface, body in (("codex", _codex_skill(name)), ("claude", _claude_command(name))):
        description = _description(body)
        assert isinstance(description, str) and description.strip(), f"{surface}:{name}"
