"""The suite must not inherit git's environment from a process that spawned it.

`cwd=` does not override `GIT_DIR`. When `GIT_DIR` is set, git skips repository
discovery entirely and operates on the directory it names, so every
`subprocess.run(["git", ...], cwd=some_tmp_repo)` in the suite — and every one
in `src/` — is silently redirected at whatever repository the parent was using.

This is not a theoretical exposure. `git commit` from a LINKED WORKTREE exports
`GIT_DIR=<repo>/.git/worktrees/<name>` plus an absolute `GIT_INDEX_FILE` to
hooks; a commit from the main worktree exports neither. `.githooks/pre-commit`
runs this suite, so one commit inside a worktree ran a fixture's `git init -q .`
against the real gitdir, set `core.bare = true` in the `.git/config` that
worktrees SHARE with the main repository, and committed a temp fixture's tree
onto the worktree's branch. The main checkout stopped being a work tree.

`conftest.scrub_ambient_git_env` removes those variables. These tests pin both
that it happens and that it is sufficient, because a scrub that silently stops
working leaves no trace until something is already destroyed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import _GIT_ENV_VARS


def _git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    )


@pytest.mark.parametrize("name", _GIT_ENV_VARS)
def test_redirecting_git_variables_are_absent(name):
    """The scrub covers every variable that can redirect a git invocation."""
    assert name not in os.environ


def test_git_dir_is_scrubbed_even_when_the_parent_exported_it(monkeypatch):
    """The fixture must win against a value present before the test starts.

    `monkeypatch.delenv` in an autouse fixture runs before the test body, so
    setting it here proves ordering, not the scrub. Instead assert the scrub's
    own contract directly: after the autouse fixture, nothing is set.
    """
    assert os.environ.get("GIT_DIR") is None
    assert os.environ.get("GIT_INDEX_FILE") is None


def test_git_init_targets_cwd_not_an_inherited_gitdir(tmp_path):
    """The exact operation that caused the damage, in miniature.

    A fixture calling `git init` with `cwd=` a temp path must create a
    repository THERE. Under an inherited `GIT_DIR` it instead reinitializes the
    inherited one and flips it bare, which is what happened to the real
    repository.
    """
    victim = tmp_path / "victim"
    victim.mkdir()
    _git(victim, "init", "-q", ".")
    _git(victim, "config", "user.email", "t@t.t")
    _git(victim, "config", "user.name", "T")
    (victim / "kept.txt").write_text("original\n")
    _git(victim, "add", "-A")
    _git(victim, "commit", "-qm", "victim-base")

    sandbox = tmp_path / "sandbox"
    (sandbox / "src").mkdir(parents=True)
    (sandbox / "src" / "foo.py").write_text("def f():\n    return 1\n")
    _git(sandbox, "init", "-q", ".")
    _git(sandbox, "config", "user.email", "t@t.t")
    _git(sandbox, "config", "user.name", "T")
    _git(sandbox, "add", "-A")
    _git(sandbox, "commit", "-qm", "sandbox-init")

    # The victim is untouched: still a work tree, still one commit, still its
    # own file. Each assertion is one of the three observed symptoms.
    bare = _git(victim, "config", "--get", "core.bare").stdout.strip()
    assert bare == "false"

    log = _git(victim, "log", "--oneline").stdout.strip().splitlines()
    assert len(log) == 1 and log[0].endswith("victim-base")

    assert (victim / "kept.txt").read_text() == "original\n"
    assert not (victim / "src" / "foo.py").exists()

    # And the sandbox really did get its own repository, so the test above is
    # not passing because nothing happened at all.
    sandbox_log = _git(sandbox, "log", "--oneline").stdout.strip()
    assert sandbox_log.endswith("sandbox-init")


@pytest.fixture(scope="session")
def _git_env_seen_by_a_session_fixture():
    return {name: os.environ.get(name) for name in _GIT_ENV_VARS}


def test_session_fixture_sees_no_git_env(_git_env_seen_by_a_session_fixture):
    """Pinned through the subprocess test below; trivially true on its own."""
    assert not any(_git_env_seen_by_a_session_fixture.values())


def test_scrub_runs_before_session_scoped_fixtures(tmp_path):
    """`git commit -a` exports `GIT_INDEX_FILE` to hooks even from the main
    checkout. A session-scoped fixture is built before any function-scoped
    one, so a function-scoped scrub let `fixture_repos` write into the
    commit's index. Re-run the probe above under exactly that environment.
    """
    env = {**os.environ, "GIT_INDEX_FILE": str(tmp_path / "next-index.lock")}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"{Path(__file__)}::test_session_fixture_sees_no_git_env"],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:]
