"""Tests for the reasoning-mode gate (fast vs multi-agent routing)."""

from neo.reasoning_effort import MemorySignal
from neo.reasoning_mode import ReasoningMode, decide_mode


def _novel():
    # no patterns -> effort high/xhigh -> deliberation-worthy
    return MemorySignal(pattern_count=0, avg_confidence=0.0)


def _familiar():
    # >=3 patterns, high confidence -> effort low -> memory covers it
    return MemorySignal(pattern_count=5, avg_confidence=0.9)


def _low_conf():
    return MemorySignal(pattern_count=2, avg_confidence=0.3)


# --- auto mode ------------------------------------------------------------

def test_novel_with_diverse_pool_deliberates():
    d = decide_mode(_novel(), car_available=True, capable_model_count=3)
    assert d.mode is ReasoningMode.MULTI_AGENT
    assert "novel" in d.reason


def test_familiar_takes_fast_path_even_with_car():
    d = decide_mode(_familiar(), car_available=True, capable_model_count=5)
    assert d.mode is ReasoningMode.FAST
    assert "familiar" in d.reason


def test_low_confidence_is_deliberation_worthy():
    d = decide_mode(_low_conf(), car_available=True, capable_model_count=2)
    assert d.mode is ReasoningMode.MULTI_AGENT


# --- CAR / pool gating ----------------------------------------------------

def test_novel_without_car_falls_back_to_fast():
    d = decide_mode(_novel(), car_available=False, capable_model_count=0)
    assert d.mode is ReasoningMode.FAST
    assert "no diverse panel" in d.reason


def test_novel_with_only_one_model_falls_back():
    # CAR present but only one capable model -> single-model panel is worthless
    d = decide_mode(_novel(), car_available=True, capable_model_count=1)
    assert d.mode is ReasoningMode.FAST


def test_min_models_threshold_is_honored():
    d = decide_mode(_novel(), car_available=True, capable_model_count=2, min_models=3)
    assert d.mode is ReasoningMode.FAST
    d2 = decide_mode(_novel(), car_available=True, capable_model_count=3, min_models=3)
    assert d2.mode is ReasoningMode.MULTI_AGENT


# --- explicit overrides ---------------------------------------------------

def test_explicit_fast_always_fast():
    d = decide_mode(_novel(), car_available=True, capable_model_count=5, explicit="fast")
    assert d.mode is ReasoningMode.FAST
    assert "--fast" in d.reason


def test_explicit_deep_deliberates_when_possible():
    d = decide_mode(_familiar(), car_available=True, capable_model_count=3, explicit="deep")
    assert d.mode is ReasoningMode.MULTI_AGENT
    assert "--deep" in d.reason


def test_explicit_deep_degrades_without_pool_not_errors():
    d = decide_mode(_novel(), car_available=False, capable_model_count=0, explicit="deep")
    assert d.mode is ReasoningMode.FAST
    assert d.effort == "xhigh"  # degrades to a max-effort single pass
    assert "high-effort single pass" in d.reason


def test_decision_carries_effort():
    d = decide_mode(_familiar(), car_available=True, capable_model_count=5)
    assert d.effort in ("low", "medium", "high", "xhigh")
