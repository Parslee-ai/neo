"""`NeoConfig.save()` must persist every field a user can set.

`from_file` accepts every field the dataclass defines, but `save` used to write
a hand-maintained allow-list. Anything missing from that list could be written
into config.json, read back correctly, and then silently erased by the next
unrelated save. `inference_mode` and `reasoning_mode` were live instances of
that bug; six more fields were latent.

These tests pin the two properties that keep the class of bug closed:
every non-secret field round-trips, and `api_key` never reaches disk unless
plaintext storage is explicitly enabled.
"""

import dataclasses
import json

import pytest

from neo import config as config_module
from neo.config import NeoConfig


# One non-default value per field. Deliberately exhaustive: the assertion below
# fails if a field is added to NeoConfig without being covered here, which is
# the only way this test keeps working as the config grows.
NON_DEFAULT_VALUES = {
    "provider": "car",
    "model": None,
    "base_url": "http://localhost:1234",
    "inference_mode": "auto",
    "reasoning_mode": "deep",
    "default_temperature": 0.1,
    "default_max_tokens": 999,
    "reasoning_effort_cap": "high",
    "safe_read_patterns": ["*.rs"],
    "forbidden_paths": ["*.pem"],
    "exemplar_dir": "/tmp/exemplars",
    "enable_ruff": False,
    "enable_pyright": False,
    "enable_mypy": True,
    "enable_eslint": False,
    "auto_install_updates": False,
    "memory_backend": "legacy",
    "constraint_auto_scan": False,
    "log_level": "DEBUG",
}


def test_every_field_is_covered_by_this_test():
    """A new NeoConfig field must be added to NON_DEFAULT_VALUES."""
    declared = {f.name for f in dataclasses.fields(NeoConfig)}
    # api_key is the one secret; it is asserted separately below.
    assert declared - set(NON_DEFAULT_VALUES) == {"api_key"}


@pytest.mark.parametrize("field_name", sorted(NON_DEFAULT_VALUES))
def test_field_survives_a_save_load_round_trip(field_name, tmp_path):
    value = NON_DEFAULT_VALUES[field_name]
    path = tmp_path / "config.json"

    NeoConfig(**{field_name: value}).save(str(path))

    assert field_name in json.loads(path.read_text()), (
        f"{field_name} was not persisted; the next save would erase it"
    )
    assert getattr(NeoConfig.from_file(str(path)), field_name) == value


def test_all_fields_persist_together(tmp_path):
    path = tmp_path / "config.json"
    NeoConfig(**NON_DEFAULT_VALUES).save(str(path))

    loaded = NeoConfig.from_file(str(path))
    for name, value in NON_DEFAULT_VALUES.items():
        assert getattr(loaded, name) == value, name


def test_defaults_are_not_frozen_into_the_file(tmp_path):
    """Only user-set fields are written.

    Persisting defaults would pin today's values for every user who ever ran
    `--config set`, so a future change to a default would never reach them —
    the failure the auto_install_updates migration exists to undo.
    """
    path = tmp_path / "config.json"
    NeoConfig(log_level="DEBUG").save(str(path))

    saved = json.loads(path.read_text())
    assert sorted(saved) == ["api_key", "log_level"]


def test_all_default_config_writes_no_settings(tmp_path):
    path = tmp_path / "config.json"
    NeoConfig().save(str(path))

    # Only the explicit secret null; every setting is left to the defaults.
    assert json.loads(path.read_text()) == {"api_key": None}


def test_api_key_from_environment_is_not_written_to_disk(tmp_path, monkeypatch):
    """load() fills api_key from env/Keychain; save() must not spill it."""
    home = tmp_path / "home"
    (home / ".neo").mkdir(parents=True)
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-from-env")
    monkeypatch.delenv("NEO_API_KEY", raising=False)
    monkeypatch.delenv("NEO_ALLOW_PLAINTEXT_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "load_api_key_from_keychain", lambda provider: None)

    config = NeoConfig.load()
    assert config.api_key == "sk-secret-from-env"

    path = tmp_path / "config.json"
    config.save(str(path))

    saved = json.loads(path.read_text())
    assert saved["api_key"] is None
    assert "sk-secret-from-env" not in path.read_text()


def test_plaintext_api_key_written_only_when_explicitly_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("NEO_ALLOW_PLAINTEXT_API_KEY", "1")
    path = tmp_path / "config.json"

    NeoConfig(api_key="sk-plain").save(str(path))

    assert json.loads(path.read_text())["api_key"] == "sk-plain"


def test_auto_update_opt_out_survives_the_migration(tmp_path, monkeypatch):
    """False + the explicit marker must not be flipped back to True on load."""
    monkeypatch.delenv("NEO_AUTO_INSTALL_UPDATES", raising=False)
    path = tmp_path / "config.json"

    NeoConfig(auto_install_updates=False).save(str(path))

    saved = json.loads(path.read_text())
    assert saved["auto_install_updates"] is False
    assert saved["_auto_update_explicit"] is True
    assert NeoConfig.from_file(str(path)).auto_install_updates is False


def test_unknown_keys_in_the_file_are_ignored(tmp_path):
    """Fields removed from the dataclass must not break loading."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"provider": "anthropic", "retired_setting": 1}))

    assert NeoConfig.from_file(str(path)).provider == "anthropic"
