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

    def test_invalidated_all_available_for_annotations(self, assembler):
        invalids = [
            _make_fact(is_valid=False, superseded_by=f"new_{i}")
            for i in range(10)
        ]
        result = assembler.assemble(invalids, "query")
        assert len(result.invalidated_facts) == 10


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

    def test_constraints_capped_to_reserve_budget(self, assembler):
        # Many large constraints should not consume entire budget.
        # With max_tokens=300, constraint cap = 200 (2/3).
        # Each constraint ~52 tokens. So ~3-4 fit under the cap.
        big_constraints = [
            _make_fact(kind=FactKind.CONSTRAINT, subject=f"rule_{i}", body="x" * 200)
            for i in range(20)
        ]
        fact = _make_fact(subject="useful", body="important info")
        result = assembler.assemble(big_constraints + [fact], "query", max_tokens=300)
        assert len(result.constraints) < 20  # constraints were capped
        assert len(result.valid_facts) >= 1  # facts still get budget

    def test_annotation_finds_old_fact_beyond_cap(self, assembler):
        # Old fact is the 5th invalidated (would have been missed with cap of 3).
        # The new fact should still get an annotation.
        invalids = [
            _make_fact(is_valid=False, superseded_by=f"n{i}", fact_id=f"old{i}", body=f"old_body_{i}")
            for i in range(5)
        ]
        new = _make_fact(subject="updated", body="new value", supersedes="old4", fact_id="n4")
        result = assembler.assemble(invalids + [new], "query")
        formatted = assembler.format_context_for_prompt(result)
        assert "changed from: old_body_4" in formatted

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


class TestRankingPolicyStaysSingleSourced:
    """`models.py` says the ranking formula "MUST stay in one place -- if the
    two paths diverge, outcome learning ranks inconsistently."

    That invariant was held by a comment and nothing else. A comment is what
    held `EXCLUDED_DIR_NAMES` in two places while the gatherer's copy was
    corrected and the index's copy kept hiding the same files. Checked at the
    time of writing: both paths do call `rank_score` and nothing recomputes
    the formula inline. These tests are here so that stays true.
    """

    def _facts(self, n=60):
        import random

        import numpy as np

        from neo.memory.models import Fact, FactKind

        random.seed(0)
        np.random.seed(0)
        out = []
        for i in range(n):
            fact = Fact(subject=f"s{i}", body="b",
                        kind=random.choice(list(FactKind)))
            fact.metadata.confidence = random.random()
            fact.metadata.success_count = random.randint(0, 9)
            fact.metadata.access_count = random.randint(0, 9)
            fact.embedding = list(np.random.rand(16))
            out.append(fact)
        return out

    def test_both_retrieval_paths_produce_the_same_ordering(self):
        import time
        from unittest.mock import patch

        import numpy as np

        from neo.math_utils import batched_cosine
        from neo.memory.context import ContextAssembler
        from neo.memory.models import rank_score

        np.random.seed(1)
        facts = self._facts()
        query = np.random.rand(16)
        now = time.time()

        # `now` is pinned rather than captured twice. Recall decay is a
        # function of elapsed time, so two `time.time()` calls make the
        # comparison depend on wall-clock drift between them -- green on a
        # quiet laptop, a coin-flip on a loaded CI runner, and the failure
        # would read as "my change broke ranking". That is the #183 shape,
        # and #183 is why latency is no longer a correctness gate here.
        with patch("neo.memory.context.time.time", return_value=now):
            assembled = ContextAssembler()._score_facts(facts, query)
        sims = batched_cosine([f.embedding for f in facts], query)
        direct = sorted(
            [(f, rank_score(f, s, now)) for f, s in zip(facts, sims)],
            key=lambda pair: pair[1], reverse=True,
        )

        assert [f.id for f, _ in assembled] == [f.id for f, _ in direct]
        # Identical inputs and one clock: exact equality, no tolerance.
        assert [s for _, s in assembled] == [s for _, s in direct]

    def test_no_module_recomputes_the_formula_inline(self):
        """The structural half. The differential above only compares the two
        paths that exist today; this catches a THIRD path being written with
        the formula pasted in, which is exactly how the duplication starts.
        """
        import pathlib

        import ast

        src = pathlib.Path(__file__).resolve().parents[1] / "src" / "neo"
        # By PATH, not by name: `src/neo/models.py` and
        # `src/neo/memory/models.py` both exist, so a name check exempted the
        # wrong file too and a formula pasted into the former was invisible.
        definition_site = src / "memory" / "models.py"

        offenders = []
        for path in src.rglob("*.py"):
            if path == definition_site:
                continue
            # Parsed, not grepped. A line filter cannot tell a call from
            # prose: `success_bonus(...)` inside a multi-line docstring, or a
            # trailing `# not success_bonus(x)`, both read as offenders, and
            # the `startswith` guard only ever skipped a docstring's FIRST
            # line. The AST sees calls and nothing else.
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:  # pragma: no cover - not our file to fix
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name in ("success_bonus", "provenance_bonus"):
                    offenders.append(f"{path.relative_to(src)}:{node.lineno}")

        assert not offenders, (
            "the ranking formula is being recomputed outside models.py: "
            f"{offenders} -- call rank_score() instead"
        )


class TestConstraintOverflowRanking:
    """When constraints overflow their cap, RELEVANCE decides which survive.

    Scope order is a stable sort, so within a scope the order was whatever the
    store yielded — effectively creation order. With a prefix cut that stopped
    at the first fact that did not fit, the injected set was "globals, plus the
    OLDEST project constraints until the budget fills". Measured on a real
    store: 2,445 valid constraints against an 8,000-token cap — ~1.4% injected,
    newest structurally unreachable however well they matched the query.
    """

    @staticmethod
    def _big(subject, embedding, scope=FactScope.PROJECT, body_len=400):
        return _make_fact(
            kind=FactKind.CONSTRAINT,
            scope=scope,
            subject=subject,
            body="x" * body_len,
            embedding=embedding,
        )

    def test_relevant_constraint_survives_overflow_despite_being_last(
        self, assembler
    ):
        """The match is placed LAST, where age-ordering would guarantee it is
        dropped. It must be injected anyway."""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        off_topic = np.array([0.0, 1.0], dtype=np.float32)

        facts = [self._big(f"filler-{i}", off_topic) for i in range(30)]
        facts.append(self._big("the-relevant-one", query_vec))

        result = assembler.assemble(
            facts, "query", query_embedding=query_vec, max_tokens=600,
        )

        subjects = [f.subject for f in result.constraints]
        assert subjects, "constraint layer came back empty"
        assert "the-relevant-one" in subjects, (
            f"relevance ignored on overflow; kept {subjects[:5]}"
        )

    def test_scope_still_outranks_relevance(self, assembler):
        """Globals are few and deliberately authoritative. Ranking applies
        WITHIN a scope tier, it does not let a project fact outrank a global."""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        off_topic = np.array([0.0, 1.0], dtype=np.float32)

        facts = [self._big(f"proj-{i}", query_vec) for i in range(30)]
        facts.append(self._big("glob", off_topic, scope=FactScope.GLOBAL))

        result = assembler.assemble(
            facts, "query", query_embedding=query_vec, max_tokens=600,
        )
        assert result.constraints[0].scope == FactScope.GLOBAL

    def test_no_embedding_preserves_previous_ordering(self, assembler):
        """Nothing to rank on means keep scope order — not a silent fallback
        to some other arrangement."""
        facts = [self._big(f"c-{i}", None) for i in range(30)]
        result = assembler.assemble(facts, "query", query_embedding=None, max_tokens=600)

        subjects = [f.subject for f in result.constraints]
        assert subjects == sorted(subjects, key=lambda s: int(s.split("-")[1]))

    def test_one_oversized_constraint_does_not_empty_the_layer(self, assembler):
        """A single verbose constraint used to truncate everything behind it:
        the accumulator stopped at the first fact that did not fit."""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        huge = self._big("huge-global", query_vec, scope=FactScope.GLOBAL, body_len=20000)
        small = [self._big(f"small-{i}", query_vec) for i in range(5)]

        result = assembler.assemble(
            [huge] + small, "query", query_embedding=query_vec, max_tokens=900,
        )

        subjects = [f.subject for f in result.constraints]
        assert any(s.startswith("small-") for s in subjects), (
            f"one oversized fact emptied the layer; kept {subjects}"
        )

    def test_under_cap_ordering_is_untouched(self, assembler):
        """Ranking must only engage on overflow. Below the cap, everything is
        injected and the established scope order is the contract."""
        query_vec = np.array([1.0, 0.0], dtype=np.float32)
        off_topic = np.array([0.0, 1.0], dtype=np.float32)
        a = _make_fact(kind=FactKind.CONSTRAINT, subject="a", body="short", embedding=off_topic)
        b = _make_fact(kind=FactKind.CONSTRAINT, subject="b", body="short", embedding=query_vec)

        result = assembler.assemble(
            [a, b], "query", query_embedding=query_vec, max_tokens=12000,
        )
        assert [f.subject for f in result.constraints] == ["a", "b"]
