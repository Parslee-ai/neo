"""Reproduction for #9003: subject-overlap alone must NOT feed the success_count
credit path (which grants permanent janitor immunity).

Overlap may keep the citation_survival metric alive (``used_in_reasoning``), but
it must not, by itself, mark a fact as *hard cited* — only hard signals (a
surviving ``[fact:<id>]`` marker or a structured ``Facts used: [...]`` self-report)
may flow into ``used_fact_ids`` and thus into ``success_count``.
"""

from __future__ import annotations

from neo.engine import NeoEngine
from neo.memory.episodes import LearningEpisode
from neo.memory.models import ContextResult, Fact
from neo.models import CodeSuggestion, PlanStep, SimulationTrace


class _CombinedLM:
    provider = "test-provider"
    model = "test-model"

    def name(self):
        return "test-provider/test-model"


def _used_fact_ids(engine):
    """Build the credit list exactly as engine.save_session does: only facts
    flagged ``hard_cited`` feed the success_count reinforcement path."""
    episode = engine.current_learning_episode
    return [
        evidence.fact_id
        for evidence in episode.retrieved_facts
        if evidence.hard_cited is True
    ]


def test_overlap_only_fact_is_not_hard_cited():
    """A fact credited purely by subject overlap must be marked used (metric)
    but NOT hard_cited, so it never enters the success_count credit path that
    grants permanent cleanup immunity."""
    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    engine._retrieved_fact_texts = {}
    facts = [
        Fact(id="overlap", subject="validate credentials before dispatch", body="x"),
    ]
    engine._capture_retrieval_context(
        ContextResult(valid_facts=facts, retrieval_scores={"overlap": 0.8}),
        included=True,
    )
    engine._capture_detectable_fact_use(
        [PlanStep(description="validate credentials before dispatch",
                  rationale="validate the credentials on dispatch")],
        [SimulationTrace("in", "out", [])],
        [CodeSuggestion("src/a.py", "", "add credential validation before dispatch", 0.8)],
    )
    evidence = engine.current_learning_episode.retrieved_facts[0]

    # Metric signal preserved.
    assert evidence.used_in_reasoning is True
    # But overlap alone must not confer hard-cited credit / immunity.
    assert evidence.hard_cited is not True
    assert "overlap" not in _used_fact_ids(engine)


def test_marker_credited_fact_is_hard_cited():
    """A fact cited by a surviving [fact:id] marker IS hard_cited and flows into
    the success_count credit path."""
    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    facts = [Fact(id="cited", subject="Convention", body="Use typed IDs")]
    engine._capture_retrieval_context(
        ContextResult(valid_facts=facts, retrieval_scores={"cited": 0.8}),
        included=True,
    )
    engine._capture_detectable_fact_use(
        [PlanStep(description="Apply", rationale="Use [fact:cited]")],
        [SimulationTrace("in", "out", [])],
        [CodeSuggestion("src/a.py", "", "Change", 0.8)],
    )
    evidence = engine.current_learning_episode.retrieved_facts[0]
    assert evidence.used_in_reasoning is True
    assert evidence.hard_cited is True
    assert "cited" in _used_fact_ids(engine)


def test_self_report_credited_fact_is_hard_cited():
    """A fact named in a 'Facts used: [...]' self-report is a hard signal."""
    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    facts = [Fact(id="reported", subject="Convention", body="Use typed IDs")]
    engine._capture_retrieval_context(
        ContextResult(valid_facts=facts, retrieval_scores={"reported": 0.8}),
        included=True,
    )
    engine._capture_detectable_fact_use(
        [PlanStep(description="Apply it", rationale="Applied. Facts used: [reported]")],
        [SimulationTrace("in", "out", [])],
        [CodeSuggestion("src/a.py", "", "Change", 0.8)],
    )
    evidence = engine.current_learning_episode.retrieved_facts[0]
    assert evidence.hard_cited is True
    assert "reported" in _used_fact_ids(engine)


def test_overlap_metric_tallies_unchanged():
    """citation_survival by_overlap / overlap_only tallies still fire for an
    overlap-only fact (the metric is preserved by the fix)."""
    engine = NeoEngine(lm_adapter=_CombinedLM(), enable_persistent_memory=False)
    engine.current_learning_episode = LearningEpisode()
    engine._retrieved_fact_texts = {}
    facts = [Fact(id="overlap", subject="validate credentials before dispatch", body="x")]
    engine._capture_retrieval_context(
        ContextResult(valid_facts=facts, retrieval_scores={"overlap": 0.8}),
        included=True,
    )
    engine._capture_detectable_fact_use(
        [PlanStep(description="validate credentials before dispatch",
                  rationale="validate the credentials on dispatch")],
        [SimulationTrace("in", "out", [])],
        [CodeSuggestion("src/a.py", "", "add credential validation before dispatch", 0.8)],
    )
    signals = engine._use_signal_counts
    assert signals["by_overlap"] == 1
    assert signals["overlap_only"] == 1
