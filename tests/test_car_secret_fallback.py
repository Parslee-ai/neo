"""Neo reads a provider key from CAR's secret store when nothing cheaper has it.

The two tools used disjoint names for the same key in the same keychain —
`car`/`OPENAI_API_KEY` against `neo-reasoner:openai:api_key` — so on a machine
where CAR held the credential, both truthfully reported "no key" and neo could
not reach a model. Neo already treats CAR as a first-class backend (`car-runtime`
extra, `CarAdapter` provider, `neo serve` on car-server, the observer under CAR's
supervisor); not consulting its store was an oversight, not a boundary.

Two properties matter beyond "it finds the key":

- **Ordering.** The lookup forks a subprocess. It must never run when an
  environment variable or neo's own keychain already answered, or every neo
  invocation pays a fork for nothing.
- **Silence on absence.** No CAR, no entry, a timeout, a broken install — all
  must return None rather than raise, because this sits on the path that builds
  every adapter.
"""

import subprocess
from unittest.mock import MagicMock

import pytest

from neo import config as cfg


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch):
    for name in ("NEO_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                 "GOOGLE_API_KEY", "AZURE_OPENAI_API_KEY", "NEO_CAR_SECRETS"):
        monkeypatch.delenv(name, raising=False)


def _car_returning(value, returncode=0):
    def _run(argv, **kwargs):
        return MagicMock(returncode=returncode, stdout=value)
    return _run


class TestLookup:
    def test_reads_the_key_car_holds(self, monkeypatch):
        monkeypatch.setattr(cfg.shutil, "which", lambda _: "/usr/local/bin/car")
        monkeypatch.setattr(cfg.subprocess, "run", _car_returning("sk-from-car\n"))
        assert cfg.load_api_key_from_car("openai") == "sk-from-car"

    def test_the_trailing_newline_is_stripped(self, monkeypatch):
        """`car secrets get` prints the value with a newline. A key stored with
        one attached would fail auth in a way that reads as a bad credential."""
        monkeypatch.setattr(cfg.shutil, "which", lambda _: "/usr/local/bin/car")
        monkeypatch.setattr(cfg.subprocess, "run", _car_returning("  sk-x  \n\n"))
        assert cfg.load_api_key_from_car("openai") == "sk-x"

    def test_asks_car_for_the_name_car_uses(self, monkeypatch):
        """CAR calls the Gemini key `GEMINI_API_KEY`; neo calls the provider
        `google`. Asking for neo's name alone would miss it."""
        seen = []

        def _run(argv, **kwargs):
            seen.append(argv[-1])
            return MagicMock(returncode=1, stdout="")

        monkeypatch.setattr(cfg.shutil, "which", lambda _: "/usr/local/bin/car")
        monkeypatch.setattr(cfg.subprocess, "run", _run)
        cfg.load_api_key_from_car("google")
        assert seen[0] == "GEMINI_API_KEY"
        assert "GOOGLE_API_KEY" in seen, "neo's own name must also be tried"

    def test_a_key_under_neos_name_is_still_found(self, monkeypatch):
        calls = {"n": 0}

        def _run(argv, **kwargs):
            calls["n"] += 1
            if argv[-1] == "GOOGLE_API_KEY":
                return MagicMock(returncode=0, stdout="sk-google\n")
            return MagicMock(returncode=1, stdout="")

        monkeypatch.setattr(cfg.shutil, "which", lambda _: "/usr/local/bin/car")
        monkeypatch.setattr(cfg.subprocess, "run", _run)
        assert cfg.load_api_key_from_car("google") == "sk-google"


class TestQuietOnAbsence:
    """This runs while building every adapter. None of it may raise."""

    def test_no_car_installed(self, monkeypatch):
        monkeypatch.setattr(cfg.shutil, "which", lambda _: None)
        assert cfg.load_api_key_from_car("openai") is None

    def test_no_entry(self, monkeypatch):
        monkeypatch.setattr(cfg.shutil, "which", lambda _: "/usr/local/bin/car")
        monkeypatch.setattr(cfg.subprocess, "run", _car_returning("", returncode=1))
        assert cfg.load_api_key_from_car("openai") is None

    def test_an_empty_value_is_not_a_key(self, monkeypatch):
        monkeypatch.setattr(cfg.shutil, "which", lambda _: "/usr/local/bin/car")
        monkeypatch.setattr(cfg.subprocess, "run", _car_returning("   \n"))
        assert cfg.load_api_key_from_car("openai") is None

    @pytest.mark.parametrize("boom", [
        OSError("exec failed"),
        subprocess.TimeoutExpired(cmd="car", timeout=5),
        subprocess.SubprocessError("broken"),
    ])
    def test_a_broken_car_install_does_not_raise(self, boom, monkeypatch):
        monkeypatch.setattr(cfg.shutil, "which", lambda _: "/usr/local/bin/car")

        def _run(*a, **k):
            raise boom

        monkeypatch.setattr(cfg.subprocess, "run", _run)
        assert cfg.load_api_key_from_car("openai") is None

    def test_an_unknown_provider_asks_car_nothing(self, monkeypatch):
        monkeypatch.setattr(cfg.shutil, "which", lambda _: "/usr/local/bin/car")
        monkeypatch.setattr(cfg.subprocess, "run",
                            lambda *a, **k: pytest.fail("must not fork"))
        assert cfg.load_api_key_from_car("some-local-thing") is None

    def test_opt_out(self, monkeypatch):
        monkeypatch.setenv("NEO_CAR_SECRETS", "0")
        monkeypatch.setattr(cfg.shutil, "which",
                            lambda _: pytest.fail("must not even look for car"))
        assert cfg.load_api_key_from_car("openai") is None


class TestItIsLastInTheChain:
    """The lookup forks a subprocess. Anything cheaper that can answer, must."""

    def test_an_env_var_wins_and_car_is_never_asked(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setattr(cfg, "load_api_key_from_car",
                            lambda p: pytest.fail("forked despite an env var"))
        monkeypatch.setattr(cfg, "load_api_key_from_keychain", lambda p: None)
        assert cfg.NeoConfig.load().api_key == "sk-from-env"

    def test_neos_own_keychain_wins_and_car_is_never_asked(self, monkeypatch):
        monkeypatch.setattr(cfg, "load_api_key_from_keychain", lambda p: "sk-from-neo")
        monkeypatch.setattr(cfg, "load_api_key_from_car",
                            lambda p: pytest.fail("forked despite neo's keychain"))
        assert cfg.NeoConfig.load().api_key == "sk-from-neo"

    def test_car_answers_only_when_both_have_missed(self, monkeypatch):
        monkeypatch.setattr(cfg, "load_api_key_from_keychain", lambda p: None)
        monkeypatch.setattr(cfg, "load_api_key_from_car", lambda p: "sk-from-car")
        assert cfg.NeoConfig.load().api_key == "sk-from-car"


def test_the_provider_env_map_is_single_sourced():
    """It was duplicated at both resolution sites and a third copy was about to
    be added here. A map that disagrees with itself sends one lookup to the
    right variable and another to nothing."""
    source = (cfg.__file__ and open(cfg.__file__).read()) or ""
    assert source.count('"anthropic": "ANTHROPIC_API_KEY"') == 1, (
        "provider->env-var mapping appears more than once in config.py"
    )
