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
    ("neo.hook", "HOOK_LEDGER", ".neo/sessions/host_events.jsonl"),
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
    # `os.path.expanduser`, not `Path.home()` — same escape, different spelling.
    # `_LOCK_PATH` is the observer's cross-process flock target, so a test
    # reaching `run_daemon` would contend with the developer's live daemon.
    ("neo.memory.observer", "_CAR_HINT_FLAG", ".neo/.car_observer_hint_shown"),
    ("neo.memory.observer", "_LOCK_PATH", ".neo/observer.lock"),
]

# Constants stored as `str`, not `Path`. Patching them with a Path would break
# callers doing string operations, so the fixture writes back the same type.
STR_PATH_CONSTANTS = {
    ("neo.memory.observer", "_CAR_HINT_FLAG"),
    ("neo.memory.observer", "_LOCK_PATH"),
}


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
        value = fake_home / relative
        if (module_name, attribute) in STR_PATH_CONSTANTS:
            value = str(value)
        monkeypatch.setattr(owner, attr_name, value)

    return fake_home


# Variables through which an ambient git process redirects every `git`
# invocation a child makes. `GIT_DIR` and `GIT_WORK_TREE` are the dangerous
# pair: they override the repository discovery that `cwd=` is supposed to
# drive, so `subprocess.run(["git", "init"], cwd=tmp_path)` silently operates
# on the INHERITED repository instead of the temporary one.
#
# Sourced from `git(1)`'s environment section; the object/alternate ones are
# included because they redirect writes even when GIT_DIR is correct.
_GIT_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_PREFIX",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_INTERNAL_SUPER_PREFIX",
)


@pytest.fixture(autouse=True)
def scrub_ambient_git_env(monkeypatch):
    """Detach the suite from any git process that spawned it.

    Tests and production code both shell out to `git` with `cwd=` pointing at a
    temporary repository. `cwd` does NOT override `GIT_DIR`: when it is set,
    git ignores repository discovery entirely and operates on the directory it
    names. So an inherited `GIT_DIR` silently redirects every one of those
    calls at the real repository.

    Which is not hypothetical, and the damage is not subtle. Committing from a
    LINKED WORKTREE exports `GIT_DIR=<repo>/.git/worktrees/<name>` and an
    absolute `GIT_INDEX_FILE` to hooks — a commit from the main worktree
    exports neither, which is why this hid for so long. `.githooks/pre-commit`
    runs the full suite, so a single `git commit` inside a worktree ran
    `tests/test_pending_session_retention.py`'s `git init -q .` against the
    real gitdir. That set `core.bare = true` in `.git/config`, which worktrees
    SHARE with the main repository, and the fixture's subsequent
    `git add -A` / `git commit` wrote its two-file temp tree onto the
    worktree's checked-out branch as a new commit. Reproduced from first
    principles, then again end to end: the main checkout stopped being a work
    tree and the worktree's history was replaced.

    Scrubbed for the whole suite rather than fixed in the one fixture that
    tripped it, because the exposure is every `subprocess` git call in `src/`
    as much as in `tests/` — `_get_changed_files_since`,
    `_get_working_tree_changes`, `_get_git_remote_url` and the acceptance
    detector all shell out, and under an inherited `GIT_DIR` they would answer
    about the wrong repository while looking perfectly healthy.

    `GIT_AUTHOR_*` / `GIT_COMMITTER_*` are deliberately left alone: they only
    supply identity, which a temp repo needs anyway, and removing them would
    break commits on machines with no `user.email` configured.
    """
    for name in _GIT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


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
