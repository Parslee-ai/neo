"""A one-word prompt that names a command must not start a reasoning run (#278).

`neo version` took "version" as a question, gathered repository context and
called the configured model. The guard runs right after argument parsing,
before the update check, the observer autostart or the engine.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from neo.cli import bare_command_hint


@pytest.mark.parametrize(
    "prompt,hint",
    [
        ("version", "neo --version"),
        ("Version", "neo --version"),
        ("--version", "neo --version"),
        ("help", "neo --help"),
        ("status", "neo workspace status"),
        ("index", "neo --index"),
        ("upgrade", "neo update"),
        ("version?", "neo --version"),
        ("Memory", "neo memory"),
        ("Lattice", "neo lattice"),
        ("hook", "neo hook"),
        # A removed command stays refused rather than becoming a paid prompt.
        ("contribute", "neo --help"),
    ],
)
def test_command_words_and_near_misses_are_refused(prompt, hint):
    assert bare_command_hint(prompt) == hint


@pytest.mark.parametrize(
    "prompt",
    [
        "what is the version scheme?",  # a real question about versions
        "version handling in setup.py",
        "refactor",  # one word, but not a command
        "explain",
        # Real words near a subcommand name: fuzzy matching refused these.
        "server",
        "service",
        "updates",
        "prompts",
        "memo",
        "cart",
        "hooks",
        "",
        None,
    ],
)
def test_questions_and_other_words_pass(prompt):
    assert bare_command_hint(prompt) is None


def test_neo_version_word_exits_without_inference(tmp_path):
    # Isolated even if the guard regresses: no update check, no self-upgrade,
    # no observer, a throwaway HOME, and src resolved from this file.
    env = {
        "HOME": str(tmp_path),
        "NEO_OBSERVER_AUTOSTART": "0",
        "NEO_SKIP_UPDATE_CHECK": "1",
        "NEO_AUTO_INSTALL_UPDATES": "0",
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
    }
    out = subprocess.run(
        [sys.executable, "-m", "neo", "version"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert out.returncode == 2, out.stderr
    assert "neo --version" in out.stderr


def test_the_guard_runs_before_any_side_effect():
    """The refusal must come before the update check and the observer
    autostart, or a mistyped command still reaches the network."""
    source = (Path(__file__).resolve().parents[1] / "src/neo/cli.py").read_text()
    body = source.split("def main():", 1)[1]
    guard = body.index("bare_command_hint(")
    for later in ("check_for_updates", "maybe_autostart_observer", "_configure_logging(args)"):
        assert later in body and body.index(later) > guard, later
