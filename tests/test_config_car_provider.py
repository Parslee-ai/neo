"""CLI reachability of the CAR inference surface.

`adapters.create_adapter` has always been able to build a CAR adapter, and
`resolve_adapter` has always honored `inference_mode`. Neither was reachable
from `neo --config set`: the provider allowlist omitted 'car' and
`EXPOSED_FIELDS` omitted `inference_mode`. Worse, `NeoConfig.save()` did not
serialize `inference_mode` at all, so a hand-edited value was erased by the
next save. These pin all three.
"""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neo.cli import parse_args
from neo.config import NeoConfig
from neo.subcommands import handle_config


def _set(key, value):
    return SimpleNamespace(config="set", config_key=key, config_value=value)


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / ".neo").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(h))
    monkeypatch.setattr("pathlib.Path.home", lambda: h)
    for var in ("NEO_API_KEY", "OPENAI_API_KEY", "NEO_PROVIDER", "NEO_INFERENCE_MODE"):
        monkeypatch.delenv(var, raising=False)
    return h


def _config_json(home):
    return json.loads((home / ".neo" / "config.json").read_text())


def test_car_is_a_settable_provider(home, capsys):
    handle_config(_set("provider", "car"))

    assert _config_json(home)["provider"] == "car"
    assert "Set provider = car" in capsys.readouterr().out


def test_invalid_provider_still_rejected(home):
    with pytest.raises(SystemExit) as exc:
        handle_config(_set("provider", "not-a-provider"))
    assert exc.value.code == 1


def test_inference_mode_is_settable_and_persists(home):
    handle_config(_set("inference_mode", "auto"))

    # Both on disk and after a round-trip through load(): `save()` used to drop
    # this field, so the value survived in memory but never on disk.
    assert _config_json(home)["inference_mode"] == "auto"
    assert NeoConfig.load().inference_mode == "auto"


def test_inference_mode_rejects_unknown_value(home):
    with pytest.raises(SystemExit) as exc:
        handle_config(_set("inference_mode", "turbo"))
    assert exc.value.code == 1


def test_save_does_not_erase_a_hand_edited_inference_mode(home):
    (home / ".neo" / "config.json").write_text(json.dumps({
        "provider": "openai",
        "inference_mode": "auto",
        "reasoning_mode": "deep",
    }))

    # Any unrelated set triggers a full rewrite of config.json.
    handle_config(_set("log_level", "INFO"))

    data = _config_json(home)
    assert data["inference_mode"] == "auto"
    assert data["reasoning_mode"] == "deep"


def test_empty_value_clears_model_so_the_router_can_choose(home, capsys):
    handle_config(_set("model", "gpt-5.6"))
    handle_config(_set("model", ""))

    assert _config_json(home)["model"] is None
    assert NeoConfig.load().model is None
    assert "Cleared model" in capsys.readouterr().out


def test_empty_value_still_rejected_for_non_nullable_fields(home):
    with pytest.raises(SystemExit) as exc:
        handle_config(_set("memory_backend", ""))
    assert exc.value.code == 1


def test_omitted_value_is_not_treated_as_a_clear(home):
    """`--config-key model` with no `--config-value` is a mistake, not a clear."""
    with pytest.raises(SystemExit) as exc:
        handle_config(_set("model", None))
    assert exc.value.code == 1


def test_car_provider_passes_no_api_key_kwargs(home):
    from neo.adapters import _adapter_kwargs_for_config

    handle_config(_set("provider", "car"))
    handle_config(_set("model", ""))

    config = NeoConfig.load()
    assert config.provider == "car"
    # CAR's router owns backend selection; api_key/base_url are meaningless.
    assert _adapter_kwargs_for_config(config) == {}


def test_max_files_defaults_differ_per_subsystem():
    """One flag feeds two caps: context gathering (30) and indexing (100)."""
    with patch.object(sys, "argv", ["neo", "--index"]):
        assert parse_args().max_files is None
    with patch.object(sys, "argv", ["neo", "a prompt"]):
        assert parse_args().max_files is None
    with patch.object(sys, "argv", ["neo", "--index", "--max-files", "500"]):
        assert parse_args().max_files == 500


def test_subcommand_parsers_have_no_max_files_attribute():
    """The --index branch must use getattr, not args.max_files."""
    with patch.object(sys, "argv", ["neo", "memory", "prune", "--dry-run"]):
        assert not hasattr(parse_args(), "max_files")
