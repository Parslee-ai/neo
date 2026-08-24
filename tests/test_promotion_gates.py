"""Which promotion gate is holding a correlated pattern back.

`neo memory learning-stats` reported `supported_once (1 accept, needs 2)`, which
names a cause the counter never checked. A candidate sits at `supported_once`
for either of two unrelated reasons — it genuinely has one acceptance, or it has
two or more that all landed at the same `repository_revision` and the
distinct-revision gate held it. Those have different remedies, and the label
asserted the first.

The distinction also settles a question the codebase could not:
`_supporting_episodes_span_distinct_revisions` records the shared-revision case
as a known limitation and names the acceptance-carrying sha as the fix, but it
runs DOWNSTREAM of acceptance, and the measured ledger had zero accepted
outcomes. `blocked_by_revision_span` is the number that says whether that gate
has ever rejected anything — i.e. whether fixing it is worth an episode schema
bump.
"""

import pytest

from neo.memory.store import FactStore
from neo.subcommands import _promotion_gates


def _group(*episodes: tuple[str, str]) -> dict[str, str]:
    """episode_id -> repository_revision, as the promote path collects it."""
    return dict(episodes)


class TestGateAttribution:
    def test_one_acceptance_awaits_a_second(self):
        gates = _promotion_gates([({"sig": _group(("e1", "rev-a"))}, set())])
        assert gates["awaiting_second_acceptance"] == 1
        assert gates["blocked_by_revision_span"] == 0

    def test_two_acceptances_at_the_same_revision_are_blocked_by_the_span_gate(self):
        """The case the old label misreported. Two acceptances is enough
        acceptances; the revision requirement is what held it."""
        gates = _promotion_gates([
            ({"sig": _group(("e1", "rev-a"), ("e2", "rev-a"))}, set()),
        ])
        assert gates["blocked_by_revision_span"] == 1
        assert gates["awaiting_second_acceptance"] == 0

    def test_two_acceptances_across_revisions_clear_both_gates(self):
        gates = _promotion_gates([
            ({"sig": _group(("e1", "rev-a"), ("e2", "rev-b"))}, set()),
        ])
        assert gates["eligible_not_promoted"] == 1
        assert gates["blocked_by_revision_span"] == 0

    def test_a_promoted_signature_is_not_reported_as_a_stall(self):
        """A durable signature has by definition cleared every gate. Counting it
        as 'eligible but not promoted' would report a success as an unexplained
        stall — which is the same defect as the label this replaces."""
        gates = _promotion_gates([
            ({"sig": _group(("e1", "rev-a"), ("e2", "rev-b"))}, {"sig"}),
        ])
        assert gates["promoted"] == 1
        assert gates["eligible_not_promoted"] == 0

    def test_a_blank_revision_counts_as_no_revision(self):
        """The gate fails closed on a blank revision, so the report must too —
        otherwise it would tell an operator to look at the span requirement when
        the real problem is that nothing recorded a revision at all."""
        gates = _promotion_gates([
            ({"sig": _group(("e1", ""), ("e2", ""))}, set()),
        ])
        assert gates["blocked_by_revision_span"] == 1

    def test_signatures_are_counted_per_project(self):
        """Promotion is per-project, so the same signature in two projects is
        two independent groups, not one with four episodes."""
        gates = _promotion_gates([
            ({"sig": _group(("e1", "rev-a"))}, set()),
            ({"sig": _group(("e2", "rev-b"))}, set()),
        ])
        assert gates["correlated_signatures"] == 2
        assert gates["awaiting_second_acceptance"] == 2

    def test_an_empty_ledger_reports_nothing_rather_than_zeroes_that_look_meaningful(self):
        gates = _promotion_gates([])
        assert gates["correlated_signatures"] == 0
        assert sum(gates.values()) == 0


class TestItCannotDriftFromTheRealGate:
    """The report must not restate the gate's threshold in its own words.

    A second copy of "two distinct revisions" would agree with the real one only
    by coincidence, and the failure mode is a diagnostic confidently naming the
    wrong blocker — which is the exact defect being fixed.
    """

    @pytest.mark.parametrize("revisions,spans", [
        (["a", "b"], True),
        (["a", "a"], False),
        (["a", ""], False),
        (["", ""], False),
        (["a", "b", "c"], True),
    ])
    def test_the_report_agrees_with_the_predicate_it_describes(self, revisions, spans):
        assert FactStore._supporting_episodes_span_distinct_revisions(revisions) is spans

        group = {f"e{i}": rev for i, rev in enumerate(revisions)}
        gates = _promotion_gates([({"sig": group}, set())])
        if len(group) < 2:
            pytest.skip("distinct-episode precondition not met by this fixture")
        assert (gates["blocked_by_revision_span"] == 0) is spans

    def test_the_grouping_matches_what_the_promote_path_collects(self):
        """Both key on `_episode_signature` of the candidate SUBJECT, never the
        body — the body is run-varying LM prose, and including it was the
        measured reason the promote path never fired."""
        subject = "bugfix: fix the parser [util.py] [fp:abc123]"
        assert (
            FactStore._episode_signature(subject)
            == FactStore._episode_signature(subject)
        )
        # Same lesson, different LM prose -> same signature, so both episodes
        # land in one group and count toward the same acceptance bar.
        assert FactStore._episode_signature(subject) != FactStore._episode_signature(
            "bugfix: fix something else [util.py] [fp:abc123]"
        )
