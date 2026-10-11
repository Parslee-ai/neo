"""A one-word prompt that names a command must not start a reasoning run (#278).

`neo version` took "version" as a question, gathered repository context and
called the configured model. The guard runs right after argument parsing,
before the update check, the observer autostart or the engine.
"""

import subprocess
import sys

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
        ("workspce", "neo workspace"),
        ("memroy", "neo memory"),
        ("lattic", "neo lattice"),
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
        "",
        None,
    ],
)
def test_questions_and_other_words_pass(prompt):
    assert bare_command_hint(prompt) is None


def test_neo_version_word_exits_without_inference(tmp_path):
    env = {
        "HOME": str(tmp_path),
        "NEO_OBSERVER_AUTOSTART": "0",
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": "src",
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
    assert "Adaptive limit" not in out.stderr  # context gathering never started
