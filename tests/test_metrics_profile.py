"""Tests for the NEO_PROFILE / NEO_METRICS gating in neo.memory.metrics."""

import json
from pathlib import Path

import pytest

from neo.memory import metrics


@pytest.fixture(autouse=True)
def _clear_metrics_env(monkeypatch):
    """Each test gets a clean env so leaked NEO_PROFILE/NEO_METRICS values
    don't bleed in from the shell."""
    monkeypatch.delenv("NEO_PROFILE", raising=False)
    monkeypatch.delenv("NEO_METRICS", raising=False)


@pytest.fixture
def isolated_home(monkeypatch, tmp_path):
    """Redirect ~/.neo to a temp dir so metrics writes don't pollute $HOME."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return tmp_path


class TestProfileResolution:
    def test_default_is_standard(self):
        assert metrics._profile() == metrics.PROFILE_STANDARD

    def test_explicit_minimal(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "minimal")
        assert metrics._profile() == metrics.PROFILE_MINIMAL

    def test_explicit_strict(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "strict")
        assert metrics._profile() == metrics.PROFILE_STRICT

    def test_explicit_off(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "off")
        assert metrics._profile() == metrics.PROFILE_OFF

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "MINIMAL")
        assert metrics._profile() == metrics.PROFILE_MINIMAL

    def test_whitespace_tolerated(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "  strict  ")
        assert metrics._profile() == metrics.PROFILE_STRICT

    def test_unknown_falls_back_to_standard(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "verbose")
        assert metrics._profile() == metrics.PROFILE_STANDARD


class TestLegacyKillSwitch:
    def test_NEO_METRICS_off_forces_off(self, monkeypatch):
        monkeypatch.setenv("NEO_METRICS", "off")
        monkeypatch.setenv("NEO_PROFILE", "strict")  # ignored
        assert metrics._profile() == metrics.PROFILE_OFF

    def test_NEO_METRICS_zero_forces_off(self, monkeypatch):
        monkeypatch.setenv("NEO_METRICS", "0")
        assert metrics._profile() == metrics.PROFILE_OFF

    def test_NEO_METRICS_false_forces_off(self, monkeypatch):
        monkeypatch.setenv("NEO_METRICS", "false")
        assert metrics._profile() == metrics.PROFILE_OFF

    def test_NEO_METRICS_on_does_not_override_profile(self, monkeypatch):
        """``NEO_METRICS=on`` is no-op; NEO_PROFILE still picks the level."""
        monkeypatch.setenv("NEO_METRICS", "on")
        monkeypatch.setenv("NEO_PROFILE", "minimal")
        assert metrics._profile() == metrics.PROFILE_MINIMAL


class TestShouldEmit:
    def test_off_blocks_all_events(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "off")
        for event in ("lm_call", "retrieve", "add_fact", "overseer_tick"):
            assert metrics._should_emit(event) is False

    def test_minimal_allows_only_lm_call(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "minimal")
        assert metrics._should_emit("lm_call") is True
        assert metrics._should_emit("retrieve") is False
        assert metrics._should_emit("add_fact") is False
        assert metrics._should_emit("overseer_tick") is False

    def test_standard_allows_everything(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "standard")
        for event in ("lm_call", "retrieve", "add_fact", "overseer_tick", "future_event"):
            assert metrics._should_emit(event) is True

    def test_strict_allows_everything(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "strict")
        for event in ("lm_call", "retrieve", "add_fact", "overseer_tick", "future_event"):
            assert metrics._should_emit(event) is True


class TestRecordHonorsProfile:
    """End-to-end: record() should write or skip based on the profile."""

    def test_default_writes_retrieve(self, isolated_home):
        metrics.record("retrieve", k=10)
        path = isolated_home / ".neo" / "metrics.jsonl"
        assert path.exists()
        line = json.loads(path.read_text().strip())
        assert line["event"] == "retrieve"

    def test_minimal_drops_retrieve(self, monkeypatch, isolated_home):
        monkeypatch.setenv("NEO_PROFILE", "minimal")
        metrics.record("retrieve", k=10)
        path = isolated_home / ".neo" / "metrics.jsonl"
        assert not path.exists()

    def test_minimal_keeps_lm_call(self, monkeypatch, isolated_home):
        monkeypatch.setenv("NEO_PROFILE", "minimal")
        metrics.record("lm_call", provider="openai", status="success")
        path = isolated_home / ".neo" / "metrics.jsonl"
        assert path.exists()
        line = json.loads(path.read_text().strip())
        assert line["event"] == "lm_call"

    def test_off_drops_everything(self, monkeypatch, isolated_home):
        monkeypatch.setenv("NEO_PROFILE", "off")
        metrics.record("lm_call", provider="openai", status="success")
        metrics.record("retrieve", k=1)
        path = isolated_home / ".neo" / "metrics.jsonl"
        assert not path.exists()

    def test_legacy_NEO_METRICS_off_drops_everything(self, monkeypatch, isolated_home):
        monkeypatch.setenv("NEO_METRICS", "off")
        metrics.record("lm_call", provider="openai", status="success")
        path = isolated_home / ".neo" / "metrics.jsonl"
        assert not path.exists()


class TestEnabledLegacy:
    def test_enabled_true_by_default(self):
        assert metrics._enabled() is True

    def test_enabled_false_when_off_profile(self, monkeypatch):
        monkeypatch.setenv("NEO_PROFILE", "off")
        assert metrics._enabled() is False

    def test_enabled_true_when_minimal(self, monkeypatch):
        """`minimal` still emits *something*, so _enabled is True."""
        monkeypatch.setenv("NEO_PROFILE", "minimal")
        assert metrics._enabled() is True
