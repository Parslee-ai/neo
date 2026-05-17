import json
import subprocess
from types import SimpleNamespace

import pytest

from neo import config as config_module
from neo.config import NeoConfig, load_api_key_from_keychain, store_api_key_in_keychain
from neo.subcommands import handle_config


def test_load_uses_keychain_when_config_has_no_api_key(tmp_path, monkeypatch):
    home = tmp_path / "home"
    neo_dir = home / ".neo"
    neo_dir.mkdir(parents=True)
    (neo_dir / "config.json").write_text(json.dumps({
        "provider": "anthropic",
        "model": "claude-sonnet",
        "api_key": None,
    }))

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NEO_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "load_api_key_from_keychain", lambda provider: "keychain-key")

    loaded = NeoConfig.load()

    assert loaded.provider == "anthropic"
    assert loaded.api_key == "keychain-key"


def test_env_api_key_matches_selected_provider(monkeypatch):
    monkeypatch.setenv("NEO_PROVIDER", "anthropic")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("NEO_API_KEY", raising=False)

    loaded = NeoConfig.from_env()

    assert loaded.provider == "anthropic"
    assert loaded.api_key == "anthropic-key"


def test_neo_api_key_overrides_provider_specific_key(monkeypatch):
    monkeypatch.setenv("NEO_PROVIDER", "anthropic")
    monkeypatch.setenv("NEO_API_KEY", "generic-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    loaded = NeoConfig.from_env()

    assert loaded.api_key == "generic-key"


def test_load_corrupt_config_falls_back_to_defaults(tmp_path, monkeypatch):
    home = tmp_path / "home"
    neo_dir = home / ".neo"
    neo_dir.mkdir(parents=True)
    (neo_dir / "config.json").write_text("not json {{{")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NEO_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "load_api_key_from_keychain", lambda provider: None)

    loaded = NeoConfig.load()

    assert loaded.provider == "openai"
    assert loaded.model == "gpt-5.5"


def test_env_provider_can_reset_saved_provider_to_default(tmp_path, monkeypatch):
    home = tmp_path / "home"
    neo_dir = home / ".neo"
    neo_dir.mkdir(parents=True)
    (neo_dir / "config.json").write_text(json.dumps({
        "provider": "anthropic",
        "model": "claude-sonnet",
        "api_key": None,
    }))

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NEO_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    loaded = NeoConfig.load()

    assert loaded.provider == "openai"
    assert loaded.api_key == "openai-key"


def test_env_provider_switch_does_not_reuse_saved_api_key(tmp_path, monkeypatch):
    home = tmp_path / "home"
    neo_dir = home / ".neo"
    neo_dir.mkdir(parents=True)
    (neo_dir / "config.json").write_text(json.dumps({
        "provider": "anthropic",
        "model": "claude-sonnet",
        "api_key": "anthropic-plaintext-key",
    }))

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NEO_PROVIDER", "openai")
    monkeypatch.delenv("NEO_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "load_api_key_from_keychain", lambda provider: "openai-key")

    loaded = NeoConfig.load()

    assert loaded.provider == "openai"
    assert loaded.api_key == "openai-key"


def test_env_model_can_reset_saved_model_to_default(tmp_path, monkeypatch):
    home = tmp_path / "home"
    neo_dir = home / ".neo"
    neo_dir.mkdir(parents=True)
    (neo_dir / "config.json").write_text(json.dumps({
        "provider": "openai",
        "model": "gpt-4",
        "api_key": None,
    }))

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NEO_MODEL", "gpt-5.5")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(config_module, "load_api_key_from_keychain", lambda provider: None)

    loaded = NeoConfig.load()

    assert loaded.model == "gpt-5.5"


def test_env_auto_update_can_reset_saved_opt_out(tmp_path, monkeypatch):
    home = tmp_path / "home"
    neo_dir = home / ".neo"
    neo_dir.mkdir(parents=True)
    (neo_dir / "config.json").write_text(json.dumps({
        "auto_install_updates": False,
        "_auto_update_explicit": True,
    }))

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NEO_AUTO_INSTALL_UPDATES", "true")
    monkeypatch.setattr(config_module, "load_api_key_from_keychain", lambda provider: None)

    loaded = NeoConfig.load()

    assert loaded.auto_install_updates is True


def test_save_does_not_write_plaintext_api_key_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("NEO_ALLOW_PLAINTEXT_API_KEY", raising=False)
    path = tmp_path / "config.json"

    NeoConfig(provider="openai", api_key="secret-key").save(str(path))

    saved = json.loads(path.read_text())
    assert saved["api_key"] is None


def test_save_can_write_plaintext_api_key_when_explicitly_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv("NEO_ALLOW_PLAINTEXT_API_KEY", "1")
    path = tmp_path / "config.json"

    NeoConfig(provider="openai", api_key="secret-key").save(str(path))

    saved = json.loads(path.read_text())
    assert saved["api_key"] == "secret-key"
    assert list(tmp_path.glob("*.tmp")) == []


def test_keychain_load_invokes_security(monkeypatch):
    monkeypatch.setattr(config_module.platform, "system", lambda: "Darwin")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="stored-key\n", stderr="")

    monkeypatch.setattr(config_module.subprocess, "run", fake_run)

    assert load_api_key_from_keychain("openai") == "stored-key"
    assert calls == [[
        "security",
        "find-generic-password",
        "-s",
        "neo-reasoner:openai:api_key",
        "-a",
        "openai",
        "-w",
    ]]


def test_keychain_store_invokes_security(monkeypatch):
    monkeypatch.setattr(config_module.platform, "system", lambda: "Darwin")
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(config_module.subprocess, "run", fake_run)

    store_api_key_in_keychain("anthropic", "secret-key")

    assert calls == [[
        "security",
        "add-generic-password",
        "-U",
        "-s",
        "neo-reasoner:anthropic:api_key",
        "-a",
        "anthropic",
        "-w",
        "secret-key",
    ]]


def test_keychain_store_errors_off_macos(monkeypatch):
    monkeypatch.setattr(config_module.platform, "system", lambda: "Linux")

    with pytest.raises(RuntimeError, match="macOS Keychain"):
        store_api_key_in_keychain("openai", "secret-key")


def test_config_set_api_key_prompts_and_stores_in_keychain(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    neo_dir = home / ".neo"
    neo_dir.mkdir(parents=True)
    (neo_dir / "config.json").write_text(json.dumps({
        "provider": "anthropic",
        "model": "claude-sonnet",
        "api_key": None,
    }))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("getpass.getpass", lambda prompt: "prompted-key")
    stored = []
    monkeypatch.setattr(
        "neo.config.store_api_key_in_keychain",
        lambda provider, api_key: stored.append((provider, api_key)),
    )

    handle_config(SimpleNamespace(config="set", config_key="api_key", config_value=None))

    assert stored == [("anthropic", "prompted-key")]
    saved = json.loads((neo_dir / "config.json").read_text())
    assert saved["api_key"] is None
    assert "Stored api_key in Keychain" in capsys.readouterr().out


def test_config_set_api_key_can_persist_plaintext_when_explicitly_allowed(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / "home"
    neo_dir = home / ".neo"
    neo_dir.mkdir(parents=True)
    (neo_dir / "config.json").write_text(json.dumps({
        "provider": "openai",
        "model": "gpt-5.5",
        "api_key": None,
    }))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NEO_ALLOW_PLAINTEXT_API_KEY", "1")
    store_calls = []
    monkeypatch.setattr(
        "neo.config.store_api_key_in_keychain",
        lambda provider, api_key: store_calls.append((provider, api_key)),
    )

    handle_config(SimpleNamespace(config="set", config_key="api_key", config_value="plain-key"))

    assert store_calls == []
    saved = json.loads((neo_dir / "config.json").read_text())
    assert saved["api_key"] == "plain-key"
    assert "plaintext enabled" in capsys.readouterr().out


def test_config_set_log_level_normalizes_and_persists(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(config_module, "load_api_key_from_keychain", lambda provider: None)

    handle_config(SimpleNamespace(config="set", config_key="log_level", config_value="debug"))

    saved = json.loads((home / ".neo" / "config.json").read_text())
    assert saved["log_level"] == "DEBUG"
    assert "Set log_level = DEBUG" in capsys.readouterr().out


def test_config_set_log_level_rejects_invalid(monkeypatch, capsys):
    monkeypatch.setattr(config_module, "load_api_key_from_keychain", lambda provider: None)

    with pytest.raises(SystemExit) as exc:
        handle_config(SimpleNamespace(config="set", config_key="log_level", config_value="chatty"))

    assert exc.value.code == 1
    assert "Invalid log level" in capsys.readouterr().err
