"""Every cut in the Execution Envelope must be marked.

`ResolvedExecutionContext.prompt_section()` reaches the model through seven
prompt builders in `engine.py`. Its list cuts were bare slices --
`constraints[:12]`, `success_criteria[:8]`, `attempts[-3:]` -- so a
thirteenth constraint vanished with nothing to show it had existed. Under a
prompt that says "satisfy these constraints", twelve of thirteen read as
thirteen: the model cannot ask about what it was not told exists, and until
`--dry-run` was fixed neither could the operator.

This module was missed by the #178 sweep that introduced `text_budget`, for
the reason that sweep wrote down about itself: it looked for slices inside the
known prompt builders, and this is a method on a dataclass reached indirectly
through `_retrieve_context`. A sweep that looks where the last bug was finds
the last bug.
"""

import pytest

from neo.execution_context import (
    _MAX_CONSTRAINTS,
    _MAX_RECENT_ATTEMPTS,
    _MAX_SUCCESS_CRITERIA,
    CallerRole,
    DerivedValue,
    ResolvedExecutionContext,
    SuccessCriterion,
    TrajectoryContext,
)
from neo.text_budget import shown_of


def _context(**overrides):
    base = dict(
        task="t",
        goal=DerivedValue("g", "explicit", 1.0),
        intent=DerivedValue("i", "explicit", 1.0),
        constraints=[],
        success_criteria=[],
        attempt=None,
        outcome=None,
        progress=None,
        trajectory=TrajectoryContext(),
        role=CallerRole.PLANNER,
        requested_output="next_action",
        current_state={},
    )
    base.update(overrides)
    return ResolvedExecutionContext(**base)


class TestConstraints:
    def test_an_elided_constraint_list_says_so(self):
        constraints = [f"c{i}" for i in range(_MAX_CONSTRAINTS + 5)]
        section = _context(constraints=constraints).prompt_section()

        assert f"[showing {_MAX_CONSTRAINTS} of {len(constraints)}]" in section
        # The survivors are still there...
        assert f"c{_MAX_CONSTRAINTS - 1}" in section
        # ...and the dropped one is genuinely absent, so the marker is the
        # only signal the reader gets. That is what makes it load-bearing.
        assert f"c{_MAX_CONSTRAINTS}" not in section

    def test_a_list_that_fits_gets_no_marker(self):
        """`shown_of` returns "" when nothing was dropped, so the annotation
        always means an omission. A marker that appeared on complete lists
        would train the reader to ignore it."""
        section = _context(constraints=["only", "two"]).prompt_section()

        assert "showing" not in section
        assert "Constraints: only; two" in section

    def test_exactly_at_the_cap_is_not_marked(self):
        """The off-by-one that would make the marker a liar in the other
        direction."""
        constraints = [f"c{i}" for i in range(_MAX_CONSTRAINTS)]
        section = _context(constraints=constraints).prompt_section()

        assert "showing" not in section
        assert f"c{_MAX_CONSTRAINTS - 1}" in section


class TestSuccessCriteria:
    def test_an_elided_criteria_list_says_so(self):
        criteria = [
            SuccessCriterion(type="cmd", description=f"crit{i}")
            for i in range(_MAX_SUCCESS_CRITERIA + 3)
        ]
        section = _context(success_criteria=criteria).prompt_section()

        assert f"[showing {_MAX_SUCCESS_CRITERIA} of {len(criteria)}]" in section
        assert f"crit{_MAX_SUCCESS_CRITERIA}" not in section


class TestRecentAttempts:
    def test_an_elided_attempt_history_says_so(self):
        """A TAIL slice, and marked for the same reason the head cuts are:
        "Recent attempts" names the intent but not the loss, so four attempts
        shown as three read as the complete history of a loop that has
        actually run four times -- which is exactly the signal a caller needs
        when deciding whether Neo is going in circles."""
        attempts = [{"summary": f"a{i}"} for i in range(_MAX_RECENT_ATTEMPTS + 4)]
        section = _context(
            trajectory=TrajectoryContext(iteration=7, attempts=attempts)
        ).prompt_section()

        # "last", not just a count. `[showing 3 of 7]` reads as a sample of
        # seven; `[showing last 3 of 7]` says the loop has run seven times and
        # you are seeing where it is now -- which is the signal a caller needs
        # to decide whether Neo is going in circles.
        assert f"[showing last {_MAX_RECENT_ATTEMPTS} of {len(attempts)}]" in section
        # Tail semantics: the most RECENT survive, not the first.
        assert f"a{len(attempts) - 1}" in section
        assert '"a0"' not in section


class TestRetrievalQueryIsDeliberatelyUnmarked:
    def test_no_marker_leaks_into_the_embedding_query(self):
        """`retrieval_query` is embedded and compared by cosine similarity;
        it is never read as instructions. A `[showing 8 of 13]` annotation
        there would be tokens in the query vector, moving every retrieval
        slightly toward facts about truncation.

        Pinned because the two methods sit next to each other and look
        identical, so the obvious "consistency" fix is wrong.
        """
        constraints = [f"c{i}" for i in range(40)]
        context = _context(constraints=constraints)

        assert "showing" not in context.retrieval_query()
        assert "showing" in context.prompt_section()


class TestNoBareSlicesRemain:
    @pytest.mark.parametrize("attr,cap", [
        ("constraints", _MAX_CONSTRAINTS),
        ("success_criteria", _MAX_SUCCESS_CRITERIA),
    ])
    def test_every_capped_list_in_the_section_is_annotated(self, attr, cap):
        """Generic guard: for each capped list, an over-long input must
        produce the annotation `shown_of` would generate. Adding a new capped
        list without a marker is the recurrence this file exists to catch."""
        items = (
            [f"x{i}" for i in range(cap + 2)]
            if attr == "constraints"
            else [SuccessCriterion(type="t", description=f"x{i}")
                  for i in range(cap + 2)]
        )
        section = _context(**{attr: items}).prompt_section()

        assert shown_of(items, cap) in section


class TestTailMarkerIsDistinctFromHead:
    def test_head_and_tail_markers_do_not_read_the_same(self):
        """The default must stay the head form: six of the seven `shown_of`
        call sites are head cuts, so a `last` that leaked into them would be
        actively false."""
        from neo.text_budget import shown_of

        items = list(range(20))
        assert shown_of(items, 3) == " [showing 3 of 20]"
        assert shown_of(items, 3, tail=True) == " [showing last 3 of 20]"

    def test_neither_form_fires_when_nothing_is_dropped(self):
        from neo.text_budget import shown_of

        assert shown_of([1, 2], 5) == ""
        assert shown_of([1, 2], 5, tail=True) == ""
