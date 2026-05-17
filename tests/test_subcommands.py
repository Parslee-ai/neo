"""Tests for CLI subcommand handlers."""

from unittest.mock import patch


def test_show_version_does_not_eager_initialize_fact_store(capsys):
    """Version display should read stored facts without startup ingestion."""
    from neo.config import NeoConfig
    from neo.memory.models import Fact, FactMetadata, FactKind, FactScope
    from neo.subcommands import show_version

    calls = {}

    class FakeFactStore:
        def __init__(self, codebase_root=None, config=None, eager_init=True):
            calls["eager_init"] = eager_init
            self.entries = [
                Fact(
                    subject="Stored pattern",
                    body="Loaded from disk.",
                    kind=FactKind.PATTERN,
                    scope=FactScope.PROJECT,
                    metadata=FactMetadata(confidence=0.8),
                )
            ]

        def memory_level(self):
            return 0.1

        def find_contributable(self):
            return []

    with patch.object(NeoConfig, "load", return_value=NeoConfig()), \
         patch("neo.memory.store.FactStore", FakeFactStore), \
         patch("neo.car_discovery.discover_car", side_effect=RuntimeError("skip car")):
        show_version("/tmp/project")

    assert calls["eager_init"] is False
    assert "neo " in capsys.readouterr().out
