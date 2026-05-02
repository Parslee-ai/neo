"""Tests for neo.reasoning_effort — memory→effort heuristic and helpers."""

import pytest

from neo.config import NeoConfig
from neo.reasoning_effort import (
    DEFAULT_EFFORT,
    EFFORT_LEVELS,
    MemorySignal,
    apply_cap,
    effort_from_memory,
    escalate,
    signal_from_facts,
    signal_from_legacy_entries,
    validate_effort,
)


class TestEffortFromMemory:
    def test_no_patterns_yields_high(self):
        # Cold start: nothing to lean on, spend tokens on thinking.
        assert effort_from_memory(MemorySignal()) == "high"

    def test_low_confidence_yields_high(self):
        # Memory exists but isn't trustworthy — same as cold start.
        signal = MemorySignal(pattern_count=4, avg_confidence=0.3)
        assert effort_from_memory(signal) == "high"

    def test_mid_confidence_yields_medium(self):
        signal = MemorySignal(pattern_count=2, avg_confidence=0.65)
        assert effort_from_memory(signal) == "medium"

    def test_high_confidence_few_patterns_yields_medium(self):
        # High confidence but only 1 supporting pattern → not enough evidence
        # to drop to low.
        signal = MemorySignal(pattern_count=1, avg_confidence=0.9)
        assert effort_from_memory(signal) == "medium"

    def test_high_confidence_many_patterns_yields_low(self):
        signal = MemorySignal(pattern_count=5, avg_confidence=0.85)
        assert effort_from_memory(signal) == "low"

    def test_boundary_confidence_080_with_3_patterns_yields_low(self):
        signal = MemorySignal(pattern_count=3, avg_confidence=0.8)
        assert effort_from_memory(signal) == "low"

    def test_boundary_confidence_just_below_threshold_yields_medium(self):
        signal = MemorySignal(pattern_count=3, avg_confidence=0.79)
        assert effort_from_memory(signal) == "medium"


class TestEscalate:
    def test_each_level_bumps_one(self):
        assert escalate("none") == "low"
        assert escalate("low") == "medium"
        assert escalate("medium") == "high"
        assert escalate("high") == "xhigh"

    def test_xhigh_caps_at_xhigh(self):
        assert escalate("xhigh") == "xhigh"

    def test_unknown_falls_back_to_default(self):
        assert escalate("turbo") == DEFAULT_EFFORT


class TestApplyCap:
    def test_no_cap_passes_through(self):
        assert apply_cap("high", None) == "high"

    def test_below_cap_passes_through(self):
        assert apply_cap("low", "high") == "low"

    def test_at_cap_passes_through(self):
        assert apply_cap("medium", "medium") == "medium"

    def test_above_cap_clamps(self):
        assert apply_cap("xhigh", "low") == "low"

    def test_unknown_value_passes_through(self):
        # Don't crash on bad input — let API surface the error.
        assert apply_cap("turbo", "high") == "turbo"


class TestValidateEffort:
    def test_none_returns_none(self):
        assert validate_effort(None) is None

    @pytest.mark.parametrize("level", EFFORT_LEVELS)
    def test_each_valid_level_passes(self, level):
        assert validate_effort(level) == level

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="Invalid reasoning effort"):
            validate_effort("minimal")  # gpt-5-codex used to support this; gpt-5.5 does not


class TestSignalFromFacts:
    def test_empty_yields_zero_signal(self):
        sig = signal_from_facts([])
        assert sig.pattern_count == 0
        assert sig.avg_confidence == 0.0

    def test_averages_metadata_confidence(self):
        # Stand-in objects matching FactStore's Fact shape (metadata.confidence).
        class _Meta:
            def __init__(self, c):
                self.confidence = c

        class _Fact:
            def __init__(self, c):
                self.metadata = _Meta(c)

        sig = signal_from_facts([_Fact(0.8), _Fact(0.6), _Fact(1.0)])
        assert sig.pattern_count == 3
        assert sig.avg_confidence == pytest.approx(0.8)

    def test_missing_metadata_treated_as_zero(self):
        class _Fact:
            metadata = None

        sig = signal_from_facts([_Fact(), _Fact()])
        assert sig.pattern_count == 2
        assert sig.avg_confidence == 0.0


class TestSignalFromLegacyEntries:
    def test_averages_confidence(self):
        class _Entry:
            def __init__(self, c):
                self.confidence = c

        sig = signal_from_legacy_entries([_Entry(0.9), _Entry(0.7)])
        assert sig.pattern_count == 2
        assert sig.avg_confidence == pytest.approx(0.8)


class TestNeoConfigValidation:
    def test_default_is_none(self):
        config = NeoConfig()
        assert config.reasoning_effort_cap is None

    def test_valid_cap_accepted(self):
        config = NeoConfig(reasoning_effort_cap="medium")
        assert config.reasoning_effort_cap == "medium"

    def test_invalid_cap_raises(self):
        with pytest.raises(ValueError, match="Invalid reasoning effort"):
            NeoConfig(reasoning_effort_cap="turbo")

    def test_env_var_loads_cap(self, monkeypatch):
        monkeypatch.setenv("NEO_REASONING_EFFORT", "high")
        config = NeoConfig.from_env()
        assert config.reasoning_effort_cap == "high"

    def test_env_var_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("NEO_REASONING_EFFORT", "bogus")
        with pytest.raises(ValueError, match="Invalid reasoning effort"):
            NeoConfig.from_env()
