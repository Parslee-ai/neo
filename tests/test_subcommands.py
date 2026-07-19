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


def test_citation_stats_aggregates_per_signal(capsys):
    """citation-stats sums per-signal counts and ignores other event types."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from neo.subcommands import _handle_citation_stats

    metrics = Path.home() / ".neo" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {"ts": 1000, "event": "citation_survival", "retrieved": 10, "included": 8,
         "used": 2, "by_marker": 0, "by_self_report": 2, "by_overlap": 1,
         "by_overlap_only": 0, "model": "gpt-5.5"},
        {"ts": 2000, "event": "citation_survival", "retrieved": 5, "included": 5,
         "used": 1, "by_marker": 0, "by_self_report": 0, "by_overlap": 1,
         "by_overlap_only": 1, "model": "gpt-5.5"},
        {"ts": 3000, "event": "lm_call", "model": "other"},  # must be ignored
        '["citation_survival"]',  # valid JSON, non-object — must not crash
    ]
    metrics.write_text(
        "\n".join(e if isinstance(e, str) else json.dumps(e) for e in events) + "\n")

    _handle_citation_stats(SimpleNamespace(json=True, since=None))
    out = json.loads(capsys.readouterr().out)
    assert out["requests"] == 2
    assert out["included"] == 13
    assert out["used"] == 3
    assert out["by_self_report"] == 2
    assert out["by_overlap"] == 2
    assert out["by_overlap_only"] == 1  # the decision number
    assert out["by_marker"] == 0
    assert out["by_model"]["gpt-5.5"]["requests"] == 2


def test_citation_stats_since_filters_old_events(capsys):
    """--since excludes events older than the window."""
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from neo.subcommands import _handle_citation_stats

    metrics = Path.home() / ".neo" / "metrics.jsonl"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps({
        "ts": 1000, "event": "citation_survival", "retrieved": 3, "included": 3,
        "used": 1, "by_marker": 0, "by_self_report": 1, "by_overlap": 0, "model": "m",
    }) + "\n")

    _handle_citation_stats(SimpleNamespace(json=True, since="1d"))  # ts=1000 is ancient
    out = json.loads(capsys.readouterr().out)
    assert out["requests"] == 0
