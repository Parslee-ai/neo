"""Tests for neo.memory.context - context assembly and prompt formatting."""

import time

import numpy as np
import pytest

from neo.memory.context import ContextAssembler
from neo.memory.models import Fact, FactKind, FactMetadata, FactScope


@pytest.fixture
def assembler():
    return ContextAssembler()


def _make_fact(
    kind=FactKind.PATTERN,
    scope=FactScope.PROJECT,
    is_valid=True,
    confidence=0.8,
    subject="test",
    body="body",
    embedding=None,
    superseded_by=None,
    supersedes=None,
    fact_id=None,
):
    f = Fact(
        subject=subject,
        body=body,
        kind=kind,
        scope=scope,
        is_valid=is_valid,
        superseded_by=superseded_by,
        supersedes=supersedes,
        metadata=FactMetadata(confidence=confidence, last_accessed=time.time()),
        embedding=embedding,
    )
    if fact_id:
        f.id = fact_id
    return f


class TestContextAssemblerLayering:
    def test_constraints_separated(self, assembler):
        c = _make_fact(kind=FactKind.CONSTRAINT)
        p = _make_fact(kind=FactKind.PATTERN)
        result = assembler.assemble([c, p], "query")
        assert c in result.constraints
        assert p in result.valid_facts

    def test_constraints_sorted_by_scope(self, assembler):
        proj = _make_fact(kind=FactKind.CONSTRAINT, scope=FactScope.PROJECT, subject="proj")
        glob = _make_fact(kind=FactKind.CONSTRAINT, scope=FactScope.GLOBAL, subject="glob")
        org = _make_fact(kind=FactKind.CONSTRAINT, scope=FactScope.ORG, subject="org")
        result = assembler.assemble([proj, glob, org], "query")
        assert result.constraints[0].subject == "glob"
        assert result.constraints[1].subject == "org"
        assert result.constraints[2].subject == "proj"

    def test_known_unknowns_separated(self, assembler):
        ku = _make_fact(kind=FactKind.KNOWN_UNKNOWN)
        result = assembler.assemble([ku], "query")
        assert ku in result.known_unknowns
        assert ku not in result.valid_facts

    def test_session_facts_in_working_set(self, assembler):
        s = _make_fact(scope=FactScope.SESSION)
        result = assembler.assemble([s], "query")
        assert s in result.working_set
        assert s not in result.valid_facts

    def test_invalidated_facts_included(self, assembler):
        old = _make_fact(is_valid=False, superseded_by="new_id", subject="old")
        result = assembler.assemble([old], "query")
        assert old in result.invalidated_facts

    def test_invalidated_capped_at_3(self, assembler):
        invalids = [
            _make_fact(is_valid=False, superseded_by=f"new_{i}")
            for i in range(10)
        ]
        result = assembler.assemble(invalids, "query")
        assert len(result.invalidated_facts) == 3


class TestContextAssemblerScoring:
    def test_higher_confidence_ranked_higher(self, assembler):
        low = _make_fact(confidence=0.2, subject="low")
        high = _make_fact(confidence=0.9, subject="high")
        result = assembler.assemble([low, high], "query", k=2)
        assert result.valid_facts[0].subject == "high"

    def test_embedding_similarity_affects_ranking(self, assembler):
        query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        close = _make_fact(
            subject="close",
            embedding=np.array([0.9, 0.1, 0.0], dtype=np.float32),
        )
        far = _make_fact(
            subject="far",
            embedding=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        result = assembler.assemble([far, close], "query", query_embedding=query_emb, k=2)
        assert result.valid_facts[0].subject == "close"

    def test_k_limits_results(self, assembler):
        facts = [_make_fact(subject=f"fact_{i}") for i in range(10)]
        result = assembler.assemble(facts, "query", k=3)
        assert len(result.valid_facts) == 3


class TestFormatContextForPrompt:
    def test_constraints_section(self, assembler):
        ctx = assembler.assemble(
            [_make_fact(kind=FactKind.CONSTRAINT, subject="No mocks", body="Never use mocks")],
            "query",
        )
        formatted = assembler.format_context_for_prompt(ctx)
        assert "Project Constraints" in formatted
        assert "No mocks" in formatted

    def test_relevant_knowledge_section(self, assembler):
        ctx = assembler.assemble(
            [_make_fact(subject="BFS pattern", body="Use BFS for shortest path")],
            "query",
        )
        formatted = assembler.format_context_for_prompt(ctx)
        assert "Relevant Knowledge" in formatted
        assert "BFS pattern" in formatted

    def test_no_separate_invalidated_section(self, assembler):
        ctx = assembler.assemble(
            [_make_fact(is_valid=False, superseded_by="x", subject="Old approach")],
            "query",
        )
        formatted = assembler.format_context_for_prompt(ctx)
        assert "Recently Changed" not in formatted

    def test_inline_change_annotation(self, assembler):
        old = _make_fact(
            is_valid=False, superseded_by="new1", subject="DB config",
            body="Use PostgreSQL 14", fact_id="old1",
        )
        new = _make_fact(
            subject="DB config", body="Use PostgreSQL 16",
            supersedes="old1", fact_id="new1",
        )
        ctx = assembler.assemble([old, new], "query")
        formatted = assembler.format_context_for_prompt(ctx)
        assert "changed from: Use PostgreSQL 14" in formatted
        assert "Recently Changed" not in formatted

    def test_annotation_missing_old_fact_no_crash(self, assembler):
        new = _make_fact(
            subject="config", body="new value",
            supersedes="nonexistent_id",
        )
        ctx = assembler.assemble([new], "query")
        formatted = assembler.format_context_for_prompt(ctx)
        assert "changed from" not in formatted
        assert "new value" in formatted

    def test_empty_returns_empty(self, assembler):
        ctx = assembler.assemble([], "query")
        formatted = assembler.format_context_for_prompt(ctx)
        assert formatted == ""


class TestTokenBudgetEnforcement:
    def test_size_hint_approximation(self):
        fact = _make_fact(subject="hello", body="world of testing")
        assert fact.size_hint() == len("hello" + "world of testing") // 4

    def test_budget_limits_valid_facts(self, assembler):
        # Each fact: "fact_N" + "x"*40 = ~46 chars → size_hint ~11 tokens.
        # Budget of 30 should fit 2 facts (22 tokens), not 3 (33 tokens).
        facts = [
            _make_fact(subject=f"fact_{i}", body="x" * 40, confidence=0.9 - i * 0.01)
            for i in range(10)
        ]
        result = assembler.assemble(facts, "query", k=10, max_tokens=30)
        assert 1 <= len(result.valid_facts) <= 3
        assert len(result.valid_facts) < 10

    def test_at_least_one_fact_when_over_budget(self, assembler):
        big = _make_fact(subject="big", body="x" * 10000)
        result = assembler.assemble([big], "query", max_tokens=1)
        assert len(result.valid_facts) == 1

    def test_constraints_exempt_from_budget(self, assembler):
        constraint = _make_fact(
            kind=FactKind.CONSTRAINT, subject="rule", body="x" * 200,
        )
        fact = _make_fact(subject="info", body="y" * 40)
        result = assembler.assemble([constraint, fact], "query", max_tokens=20)
        assert constraint in result.constraints
        assert len(result.valid_facts) >= 1  # at least 1 always included

    def test_k_and_max_tokens_both_apply(self, assembler):
        facts = [
            _make_fact(subject=f"f{i}", body="short")
            for i in range(10)
        ]
        # k=3 is more restrictive than max_tokens=12000
        result = assembler.assemble(facts, "query", k=3, max_tokens=12000)
        assert len(result.valid_facts) == 3

    def test_default_max_tokens(self, assembler):
        facts = [_make_fact(subject=f"f{i}", body="content") for i in range(5)]
        result = assembler.assemble(facts, "query", k=5)
        assert len(result.valid_facts) == 5  # default 12000 easily fits 5 small facts

    def test_budget_shared_across_layers(self, assembler):
        # Each big fact: ~206 chars → size_hint ~51 tokens.
        # Budget of 120 fits ~2 valid facts (102 tokens), leaving ~18 for session.
        # Session fact is ~51 tokens → should NOT fit.
        big_facts = [
            _make_fact(subject=f"f{i}", body="x" * 200, confidence=0.9)
            for i in range(5)
        ]
        session = _make_fact(scope=FactScope.SESSION, subject="s", body="y" * 200)
        result = assembler.assemble(big_facts + [session], "query", k=5, max_tokens=120)
        assert len(result.valid_facts) >= 1
        assert len(result.valid_facts) <= 3  # budget should cap around 2
        assert len(result.working_set) == 0  # no budget left for session

    def test_negative_budget_does_not_cascade(self, assembler):
        # A single oversized valid fact exceeds budget via at_least_one guarantee.
        # Subsequent layers should get nothing (budget clamped to 0, no at_least_one).
        big = _make_fact(subject="huge", body="x" * 10000, confidence=0.9)
        session = _make_fact(scope=FactScope.SESSION, subject="s", body="y" * 100)
        ku = _make_fact(kind=FactKind.KNOWN_UNKNOWN, subject="q", body="z" * 100)
        result = assembler.assemble([big, session, ku], "query", max_tokens=50)
        assert len(result.valid_facts) == 1  # at_least_one kicks in
        assert len(result.working_set) == 0  # no budget cascade
        assert len(result.known_unknowns) == 0  # no budget cascade
