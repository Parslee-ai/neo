"""Tests for the shared prompt-budget helpers.

These exist because seven prompt-building sites each cut text with a bare
slice. A bare slice is indistinguishable from text that ended, so the model
reasons about absence from a fragment.

Three shapes, and picking the wrong one is a real defect rather than a style
choice: a head-cut traceback keeps the frame headers and discards the
exception line, which is the only part the reader wanted.
"""

import pytest

from neo.text_budget import (
    MARKER_PREFIX,
    apportion,
    elide_middle,
    shown_of,
    truncate_marked,
)


def was_truncated(rendered: str) -> bool:
    return MARKER_PREFIX in rendered


class TestFitsWithinBudget:
    def test_short_text_is_returned_unchanged(self):
        assert truncate_marked("hello", 100) == "hello"

    def test_text_exactly_at_budget_is_unchanged(self):
        """Boundary: the marker must mean a cut happened, always."""
        text = "x" * 100
        assert truncate_marked(text, 100) == text

    def test_no_marker_when_nothing_was_dropped(self):
        assert not was_truncated(truncate_marked("x" * 99, 100))


class TestMarksTheCut:
    def test_marker_is_appended(self):
        assert was_truncated(truncate_marked("x" * 200, 100))

    def test_marker_names_dropped_and_total(self):
        assert "100 of 200 characters not shown" in truncate_marked("x" * 200, 100)

    def test_content_is_cut_to_budget(self):
        out = truncate_marked("x" * 200, 100)
        payload = out.split(MARKER_PREFIX, 1)[0]
        assert len(payload) == 100

    def test_one_char_over_budget_is_marked(self):
        """Off-by-one: 101 chars against 100 is a cut, not a fit."""
        assert was_truncated(truncate_marked("x" * 101, 100))

    def test_marker_starts_on_its_own_line(self):
        """A marker glued to the last token could read as file content."""
        assert "\n... [truncated:" in truncate_marked("abc" * 100, 10)

    def test_budget_bounds_content_not_the_return_value(self):
        """Documented contract: the marker is appended ON TOP of the budget.

        A caller sizing a hard ceiling has to account for it; every current
        caller sizes a soft prompt section and does not.
        """
        out = truncate_marked("x" * 500, 100)
        assert len(out) > 100
        assert len(out.split(MARKER_PREFIX, 1)[0]) == 100


class TestEmptyAndInvalid:
    def test_none_is_treated_as_empty(self):
        assert truncate_marked(None, 100) == ""

    def test_empty_string_is_unchanged(self):
        assert truncate_marked("", 100) == ""

    @pytest.mark.parametrize("budget", [0, -1, -1000])
    def test_empty_content_never_claims_a_cut(self, budget):
        """The invariant the whole module rests on, at its weakest point.

        `budget >= len(content)` is `-1 >= 0` → False for empty content and a
        negative budget, which fell through to the marker branch and emitted
        "0 of 0 characters not shown" — a marker asserting a cut that did not
        happen, from the one function whose contract is that it never does.
        """
        assert truncate_marked("", budget) == ""
        assert truncate_marked(None, budget) == ""

    @pytest.mark.parametrize("budget", [0, -1, -1000])
    def test_non_positive_budget_on_real_content_raises(self, budget):
        """A caller bug, not a data condition — every call site passes a
        module constant, so this fires in development or never.

        Returning "" would hide the cut; returning a bare marker would emit
        ~50 characters from a budget of zero. Neither is a truncation.
        """
        with pytest.raises(ValueError, match="budget must be positive"):
            truncate_marked("x" * 50, budget)


class TestElideMiddle:
    def _traceback(self, frames=40):
        body = "".join(
            f'  File "mod{i}.py", line {i}, in f{i}\n    do_something()\n'
            for i in range(frames)
        )
        return body + "ValueError: database is locked\n"

    def test_short_text_is_unchanged(self):
        assert elide_middle("boom", 500) == "boom"

    def test_keeps_the_last_line(self):
        """The reason this helper exists. A head-cut loses the answer."""
        out = elide_middle(self._traceback(), 500)

        assert "ValueError: database is locked" in out

    def test_keeps_the_first_frame(self):
        out = elide_middle(self._traceback(), 500)

        assert 'File "mod0.py"' in out

    def test_marks_what_it_removed(self):
        out = elide_middle(self._traceback(), 500)

        assert "characters elided" in out

    def test_head_only_would_have_lost_the_answer(self):
        """Pins the distinction the two helpers exist to draw."""
        trace = self._traceback()

        assert "ValueError" not in truncate_marked(trace, 500)
        assert "ValueError" in elide_middle(trace, 500)


class TestShownOf:
    def test_elided_list_states_how_many_were_shown(self):
        assert shown_of(["a", "b", "c", "d", "e"], 3) == " [showing 3 of 5]"

    @pytest.mark.parametrize("items", [[], ["a"], ["a", "b"], ["a", "b", "c"]])
    def test_complete_list_is_not_annotated(self, items):
        """The annotation must always mean an omission."""
        assert shown_of(items, 3) == ""


class TestApportion:
    def test_short_sections_are_fully_funded(self):
        assert apportion({"a": 10, "b": 10}, 1000) == {"a": 10, "b": 10}

    def test_unused_capacity_goes_to_the_section_that_can_use_it(self):
        """The measured case: one 30-char section and one 33,144-char
        section against 6,000. A flat 2,000 cap spent 2,030 of it."""
        allocation = apportion({"envelope": 30, "memory": 33144}, 6000)

        assert allocation["envelope"] == 30
        assert allocation["memory"] == 5970
        assert sum(allocation.values()) == 6000

    def test_equal_demand_splits_evenly(self):
        allocation = apportion({"a": 9000, "b": 9000, "c": 9000}, 6000)

        assert allocation == {"a": 2000, "b": 2000, "c": 2000}

    def test_never_exceeds_the_budget(self):
        allocation = apportion({"a": 50_000, "b": 40_000, "c": 30_000}, 6000)

        assert sum(allocation.values()) <= 6000

    def test_empty_sections_are_dropped(self):
        assert apportion({"a": 0, "b": 100}, 6000) == {"b": 100}

    def test_no_sections(self):
        assert apportion({}, 6000) == {}

    @pytest.mark.parametrize("budget", [2, 1, 0, -1])
    def test_unseatable_budget_raises(self, budget):
        """Zeros would force the caller to drop sections silently.

        That is this module's own defect class reappearing as a guard clause,
        so the unseatable case is refused at the source instead.
        """
        with pytest.raises(ValueError, match="cannot seat"):
            apportion({"a": 100, "b": 100, "c": 100}, budget)

    def test_exactly_seatable_budget_is_allowed(self):
        """One character each is the boundary, and it is legal."""
        assert apportion({"a": 100, "b": 100, "c": 100}, 3) == {
            "a": 1, "b": 1, "c": 1,
        }

    def test_empty_sections_do_not_count_toward_seating(self):
        assert apportion({"a": 0, "b": 0, "c": 100}, 1) == {"c": 1}

    def test_order_does_not_change_the_outcome(self):
        """The nested-cut design was order-dependent; this must not be."""
        forward = apportion({"a": 30, "b": 33144, "c": 31000}, 6000)
        reverse = apportion({"c": 31000, "b": 33144, "a": 30}, 6000)

        assert forward == reverse
