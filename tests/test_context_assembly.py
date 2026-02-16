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
):
    return Fact(
        subject=subject,
        body=body,
        kind=kind,
        scope=scope,
        is_valid=is_valid,
        superseded_by=superseded_by,
        metadata=FactMetadata(confidence=confidence, last_accessed=time.time()),
        embedding=embedding,
    )


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

    def test_invalidated_section(self, assembler):
        ctx = assembler.assemble(
            [_make_fact(is_valid=False, superseded_by="x", subject="Old approach")],
            "query",
        )
        formatted = assembler.format_context_for_prompt(ctx)
        assert "Recently Changed" in formatted
        assert "Old approach" in formatted

    def test_empty_returns_empty(self, assembler):
        ctx = assembler.assemble([], "query")
        formatted = assembler.format_context_for_prompt(ctx)
        assert formatted == ""
