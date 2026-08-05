"""Shared test fixtures for Neo tests.

Prevents all tests from touching ~/.neo/ by redirecting Path.home()
to a temporary directory. This stops tests from corrupting live
memory files (global_memory.json, local_*.json, facts/).

Patching `Path.home()` is NOT sufficient on its own. Modules across the
codebase capture their paths in constants evaluated at IMPORT time
(`SESSIONS_DIR = Path.home() / ".neo" / "sessions"`), and pytest imports every
module during collection — before any fixture runs. Those constants therefore
keep pointing at the real home no matter what the fixture does to `Path.home()`.

That was not hypothetical: a run of test_outcomes + test_fact_store +
test_transcript wrote `~/.neo/constraints/checksums.json` and
`~/.neo/sessions/watermark_testproj1234.json` into the developer's live state.

`HOME_PATH_CONSTANTS` below re-points each captured constant at the fake home.
`test_home_isolation.py` asserts the table stays complete, so a newly added
`Path.home()` constant fails the suite instead of silently leaking.
"""

import importlib
import site
import pytest
from pathlib import Path

# Captured before any fixture patches `Path.home()` or `$HOME`, so tests can
# still tell where the developer's real state lives.
REAL_HOME = Path.home()

# (module, attribute, path relative to home). Class attributes use
# "Class.ATTR" in the attribute slot.
#
# Deliberately absent: `store.FASTEMBED_CACHE_DIR`. That is a read-mostly
# ~400 MB model cache, not neo state — redirecting it to a throwaway home
# would re-download the model on every test run. `isolate_neo_home` pins it to
# the real cache on purpose.
HOME_PATH_CONSTANTS: list[tuple[str, str, str]] = [
    ("neo.memory.outcomes", "SESSIONS_DIR", ".neo/sessions"),
    # transcript does `from ...outcomes import SESSIONS_DIR`, which binds a
    # SECOND name at import time; patching only outcomes leaves this one live.
    ("neo.memory.transcript", "SESSIONS_DIR", ".neo/sessions"),
    ("neo.memory.transcript", "CLAUDE_PROJECTS_DIR", ".claude/projects"),
    ("neo.memory.transcript", "CAR_SESSIONS_DIR", ".car/sessions"),
    ("neo.memory.transcript", "CODEX_SESSIONS_DIR", ".codex/sessions"),
    ("neo.memory.store", "FACTS_DIR", ".neo/facts"),
    ("neo.memory.constraints", "CHECKSUM_DIR", ".neo/constraints"),
    ("neo.memory.constraints", "CHECKSUM_FILE", ".neo/constraints/checksums.json"),
    ("neo.memory.seed", "CHECKSUM_DIR", ".neo/constraints"),
    ("neo.memory.seed", "CHECKSUM_FILE", ".neo/constraints/checksums.json"),
    ("neo.memory.claude_memory", "CLAUDE_PROJECTS_DIR", ".claude/projects"),
    ("neo.memory.claude_memory", "CHECKSUM_DIR", ".neo/constraints"),
    ("neo.memory.claude_memory", "CHECKSUM_FILE", ".neo/constraints/checksums.json"),
    ("neo.memory.community", "CACHE_DIR", ".neo"),
    ("neo.memory.community", "CACHE_FILE", ".neo/community_facts_cache.json"),
    ("neo.memory.community", "CHECKSUM_DIR", ".neo/constraints"),
    ("neo.memory.community", "CHECKSUM_FILE", ".neo/constraints/checksums.json"),
    ("neo.prompt.evolution", "EvolutionTracker.EVOLUTION_FILE",
     ".neo/prompt_evolutions.json"),
    ("neo.prompt.knowledge_base", "PromptKnowledgeBase.STORAGE_FILE",
     ".neo/prompt_knowledge.json"),
    ("neo.prompt.change_detector", "ChangeDetector.WATERMARK_FILE",
     ".neo/prompt_watermarks.json"),
]


def _resolve_target(module_name: str, attribute: str):
    """Return (owner, attr_name) so class attributes patch on the class."""
    owner = importlib.import_module(module_name)
    *outer, attr_name = attribute.split(".")
    for part in outer:
        owner = getattr(owner, part)
    return owner, attr_name


@pytest.fixture(autouse=True)
def isolate_neo_home(tmp_path, monkeypatch):
    """Redirect Path.home() so no test touches ~/.neo/.

    Pin PYTHONUSERBASE to the real user-base before patching HOME so
    subprocess CLI invocations (e.g. `python -m neo`) can still resolve
    a user-site editable install — user-site is otherwise derived from
    $HOME and would point into the fake home.
    """
    monkeypatch.setenv("PYTHONUSERBASE", site.getuserbase())
    # Pin the fastembed model cache to the REAL user cache before patching HOME,
    # so in-process and subprocess embedders reuse the already-downloaded ~400 MB
    # model instead of re-fetching it into the throwaway fake home (which would
    # blow past subprocess CLI timeouts). Model cache is a shared read-mostly
    # asset, distinct from the ~/.neo state this fixture isolates.
    monkeypatch.setenv(
        "NEO_FASTEMBED_CACHE_DIR", str(Path.home() / ".cache" / "neo" / "fastembed")
    )
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))  # Also patch $HOME for expanduser()

    # Re-point every path constant captured at import time. Without this the
    # fixture's promise is false for any module already imported — which, at
    # collection time, is all of them.
    for module_name, attribute, relative in HOME_PATH_CONSTANTS:
        owner, attr_name = _resolve_target(module_name, attribute)
        monkeypatch.setattr(owner, attr_name, fake_home / relative)

    return fake_home


@pytest.fixture(autouse=True)
def clear_remote_url_cache():
    """Reset the memoized git-remote lookups between tests.

    ``scope._REMOTE_URL_CACHE`` is process-global and its ``codebase_root=None``
    key resolves to ``os.getcwd()`` — so without this, the first test to take
    that path poisons the answer for every later test in the process.
    """
    from neo.memory import scope

    scope.clear_remote_url_cache()
    yield
    scope.clear_remote_url_cache()
