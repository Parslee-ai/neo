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
