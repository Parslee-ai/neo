"""Every path that cuts text into an LM prompt must mark the cut.

This file covers the class rather than one instance. Seventeen cuts across
eight prompt builders in six modules each sliced their input bare, so a
reader — the model — could not distinguish text that ended from text that
was severed.
`neo.agent_context` and `ContextAssembler` already did it correctly.

The tests drive the real prompt builders and assert on the text that would
actually be sent, so a path that reintroduces a bare slice fails here rather
than in production. Where a builder was unreachable without heavy
dependencies it was extracted into a function rather than tested by proxy —
a test that re-derives the helper call instead of exercising the builder
proves only that the helper works.
"""

import ast
from unittest.mock import patch

import pytest

from neo.text_budget import MARKER_PREFIX


def was_truncated(rendered: str) -> bool:
    """Did `rendered` come back carrying a head-cut marker?

    Lives here rather than in `neo.text_budget` because sniffing rendered
    text for a marker is only sound when the test controls the input. As
    production accounting it would false-positive on any content that quotes
    the marker — including this repository's own source, which neo indexes.
    Producers that need the fact already have it from the length comparison.
    """
    return MARKER_PREFIX in rendered


class TestDeliberationContext:
    """`engine._deliberate` — the multi-agent panel, taken for hard problems.

    Measured against a live fact store, `past_learnings` formatted to 33,144
    characters and reached the panel as 2,000: 94% dropped, ending mid-word,
    unannounced.
    """

    def _capture_context(self, context):
        """Run _deliberate and return the context string the panel received."""
        from neo.engine import NeoEngine

        engine = NeoEngine.__new__(NeoEngine)
        seen = {}

        class FakeReasoner:
            def __init__(self, *a, **kw):
                pass

            def deliberate(self, prompt, context=""):
                seen['prompt'] = prompt
                seen['context'] = context
                raise _PromptCaptured

        with patch.object(NeoEngine, "_build_car_role_factory", lambda *a, **kw: None), \
             patch("neo.multi_agent.MultiAgentReasoner", FakeReasoner):
            engine._deliberate(context, route_fn=None)
        assert 'context' in seen, "panel was never handed a context"
        return seen['context']

    def test_oversized_memory_is_marked(self):
        sent = self._capture_context({
            'prompt': 'why does this fail',
            'past_learnings': 'MEMORY. ' * 6000,      # ~48k chars
        })

        assert was_truncated(sent), "94% of retrieved memory dropped with no marker"

    def test_marker_names_the_dropped_amount(self):
        """A lone section gets the whole budget, so 10,000 - 6,000 = 4,000.

        Under the flat per-section cap this was 8,000 dropped: the section
        was held to 2,000 while 4,000 of the budget went unspent.
        """
        from neo.engine import _DELIBERATION_TOTAL_CHARS

        sent = self._capture_context({
            'prompt': 'p',
            'past_learnings': 'x' * 10_000,
        })

        dropped = 10_000 - _DELIBERATION_TOTAL_CHARS
        assert f"{dropped} of 10000 characters not shown" in sent

    def test_context_that_fits_is_not_marked(self):
        sent = self._capture_context({'prompt': 'p', 'past_learnings': 'short fact'})

        assert not was_truncated(sent)
        assert 'short fact' in sent

    def test_every_oversized_key_is_marked_independently(self):
        """Each key is cut on its own, so each needs its own marker."""
        sent = self._capture_context({
            'prompt': 'p',
            'execution_envelope_text': 'A' * 5000,
            'past_learnings': 'B' * 5000,
        })

        assert sent.count("... [truncated:") >= 2

    def test_no_section_loses_its_marker(self):
        """Every present section must arrive carrying its own honest marker.

        The nested cut this replaced delivered `verifiable_constraints` as
        1,890 of 31,000 characters with its marker sliced off, beneath an
        outer marker reporting 163 characters dropped — 29,110 characters
        gone behind a note asserting they were not. Deterministically the
        last section, because the order and the caps were both fixed.
        """
        sent = self._capture_context({
            'prompt': 'p',
            'execution_envelope_text': 'A' * 33000,
            'past_learnings': 'B' * 33144,
            'verifiable_constraints': 'C' * 31000,
        })

        assert sent.count("... [truncated:") == 3, (
            "one marker per cut section; a missing one means a cut ate a marker"
        )

    def test_every_marker_reports_its_own_sections_real_loss(self):
        """A marker's numbers must describe the source it belongs to."""
        import re

        sizes = {'execution_envelope_text': 33000,
                 'past_learnings': 33144,
                 'verifiable_constraints': 31000}
        sent = self._capture_context({'prompt': 'p', **{
            k: c * n for (k, n), c in zip(sizes.items(), "ABC")
        }})

        totals = {int(m) for m in re.findall(r"of (\d+) characters not shown", sent)}
        assert totals == set(sizes.values()), (
            f"markers report {totals}, sources are {set(sizes.values())} — "
            f"a marker describing the concatenated buffer cannot describe "
            f"source loss"
        )

    def test_the_whole_budget_is_used(self):
        """A flat per-section cap left 66% of the budget unspent on live data.

        Fails if the apportionment is replaced by a per-section constant.
        """
        from neo.engine import _DELIBERATION_TOTAL_CHARS

        sent = self._capture_context({
            'prompt': 'p',
            'execution_envelope_text': 'A' * 30,       # short: fully funded
            'past_learnings': 'B' * 33144,             # long: takes the rest
        })

        payload = "".join(c for c in sent if c in "AB")
        assert payload.count('A') == 30
        assert payload.count('B') > 5000, (
            f"long section got {payload.count('B')} chars of a "
            f"{_DELIBERATION_TOTAL_CHARS} budget — unused capacity was not "
            f"redistributed"
        )

    def test_separator_overhead_is_reserved(self):
        """The budget covers the `\n\n` joins, not just the sections.

        Without the reservation the rendered content exceeds the cap by two
        characters per join — small, but it is the kind of drift that makes a
        stated budget stop meaning anything.
        """
        from neo.engine import _DELIBERATION_TOTAL_CHARS

        sent = self._capture_context({
            'prompt': 'p',
            'execution_envelope_text': 'A' * 33000,
            'past_learnings': 'B' * 33144,
            'verifiable_constraints': 'C' * 31000,
        })

        separators = sent.count("\n\n")
        content = sum(sent.count(c) for c in "ABC")
        assert content + 2 * separators <= _DELIBERATION_TOTAL_CHARS

    def test_total_budget_is_respected(self):
        """Fails if the total cap stops binding."""
        from neo.engine import _DELIBERATION_TOTAL_CHARS

        sent = self._capture_context({
            'prompt': 'p',
            'execution_envelope_text': 'A' * 33000,
            'past_learnings': 'B' * 33144,
            'verifiable_constraints': 'C' * 31000,
        })

        content = sum(sent.count(c) for c in "ABC")
        assert content <= _DELIBERATION_TOTAL_CHARS


class TestMisconfiguredBudget:
    """What the two new ValueErrors do to their callers.

    Both are programmer errors — every call site passes a module constant, so
    neither can fire without someone editing a constant. The question the rule
    asks is what happens when one does.
    """

    def test_unseatable_budget_degrades_to_the_fast_path(self):
        """`_deliberate` already treats any panel failure as "fall back".

        Letting the ValueError propagate to that handler is deliberate: a
        misconfigured constant should be loud in the log and non-fatal to the
        user's run, not a traceback out of a reasoning call. Asserted rather
        than assumed, because "it degrades" and "it crashes" look identical
        until someone checks.
        """
        from neo.engine import NeoEngine

        engine = NeoEngine.__new__(NeoEngine)
        with patch.object(NeoEngine, "_build_car_role_factory", lambda *a, **kw: None), \
             patch("neo.engine._DELIBERATION_TOTAL_CHARS", 2):
            result = engine._deliberate({
                'prompt': 'p',
                'execution_envelope_text': 'A' * 100,
                'past_learnings': 'B' * 100,
                'verifiable_constraints': 'C' * 100,
            }, route_fn=None)

        assert result == (None, None, None, None)

    def test_the_failure_is_logged_not_swallowed(self, caplog):
        import logging

        from neo.engine import NeoEngine

        engine = NeoEngine.__new__(NeoEngine)
        with caplog.at_level(logging.WARNING, logger='neo.engine'), \
             patch.object(NeoEngine, "_build_car_role_factory", lambda *a, **kw: None), \
             patch("neo.engine._DELIBERATION_TOTAL_CHARS", 2):
            engine._deliberate({
                'prompt': 'p',
                'execution_envelope_text': 'A' * 100,
                'past_learnings': 'B' * 100,
                'verifiable_constraints': 'C' * 100,
            }, route_fn=None)

        assert any("cannot seat" in r.getMessage() for r in caplog.records)


class TestPatternExtractionPrompt:
    """`pattern_extraction` — a fence closing after a mid-function cut reads
    as a complete function, and the rule extracted here is durable."""

    def _prompt_for(self, **kw):
        import neo.pattern_extraction as pe

        adapter = _CapturingAdapter()
        args = dict(
            problem_description="describe", failed_code="bad", corrected_code="good",
            bug_category="off-by-one", root_cause="cause", adapter=adapter,
        )
        args.update(kw)
        try:
            pe.extract_pattern_from_correction(**args)
        except _PromptCaptured:
            pass
        assert adapter.called, "prompt builder never ran"
        return adapter.prompt

    def test_oversized_failed_code_is_marked(self):
        prompt = self._prompt_for(failed_code="def broken():\n    pass\n" * 400)

        assert was_truncated(prompt)

    def test_oversized_corrected_code_is_marked(self):
        prompt = self._prompt_for(corrected_code="def fixed():\n    return 1\n" * 400)

        assert was_truncated(prompt)

    def test_marker_lands_inside_the_fence(self):
        """The cut is inside ``` — that is precisely where it must be visible."""
        prompt = self._prompt_for(failed_code="X" * 5000)

        fenced = prompt.split("FAILED CODE:", 1)[1].split("```")[1]
        assert "... [truncated:" in fenced

    def test_small_code_is_not_marked(self):
        prompt = self._prompt_for(failed_code="def f(): pass")

        assert not was_truncated(prompt)
        assert "def f(): pass" in prompt


class TestConstraintExtractionPrompt:
    """`constraint_verification` — the caller treats the result as THE
    constraint set, so a constraint in a dropped tail is silently unchecked."""

    def _prompt_for(self, description):
        from neo.constraint_verification import ConstraintVerifier

        seen = {}

        class FakeAdapter:
            def generate(self, messages, **_):
                seen['prompt'] = messages[0]['content']
                return "none"

        ConstraintVerifier()._llm_extract_constraints(description, FakeAdapter())
        return seen.get('prompt', '')

    def test_oversized_description_is_marked(self):
        prompt = self._prompt_for("The array must be sorted. " * 200)

        assert was_truncated(prompt)

    def test_short_description_is_not_marked(self):
        prompt = self._prompt_for("The array must be sorted.")

        assert not was_truncated(prompt)
        assert "The array must be sorted." in prompt


class TestRepairPrompt:
    """`repair_loop` — malformed JSON is often malformed at the END, so a
    silent tail cut can remove the very defect being repaired."""

    def _prompt_for(self, **kw):
        from neo.repair_loop import create_repair_prompt

        args = dict(
            bad_response="{broken", error_code="X", error_message="m",
            kind="plan", original_prompt="do the thing",
        )
        args.update(kw)
        return create_repair_prompt(**args)

    def test_oversized_bad_response_is_marked(self):
        prompt = self._prompt_for(bad_response='{"k": "v"}, ' * 300)

        assert was_truncated(prompt)

    def test_marker_says_the_document_continued(self):
        """Without this the formatter may 'repair' a truncation into valid
        JSON that silently drops half the content."""
        prompt = self._prompt_for(bad_response="Z" * 4000)

        assert "3000 of 4000 characters not shown" in prompt

    def test_oversized_original_prompt_is_marked(self):
        prompt = self._prompt_for(original_prompt="P" * 900)

        assert was_truncated(prompt)

    def test_small_response_is_not_marked(self):
        prompt = self._prompt_for(bad_response='{"almost": "valid",}')

        assert not was_truncated(prompt)
        assert '{"almost": "valid",}' in prompt


class _PromptCaptured(Exception):
    """Sentinel so a capture helper never swallows a real error."""


class _CapturingAdapter:
    """Records the prompt it was handed, then stops the call."""

    def __init__(self):
        self.prompt = ""
        self.called = False

    def generate(self, messages, **_):
        self.prompt = messages[0]['content']
        self.called = True
        raise _PromptCaptured


class TestAlgorithmDesignPrompts:
    """`algorithm_design` — the design steers code generation, so a design
    derived from half a problem is confidently wrong about the other half."""

    def _design_prompt_for(self, description):
        from neo.algorithm_design import design_algorithm

        adapter = _CapturingAdapter()
        try:
            design_algorithm(description, adapter)
        except _PromptCaptured:
            pass
        assert adapter.called, "prompt builder never ran"
        return adapter.prompt

    def test_design_prompt_marks_the_problem_cut(self):
        prompt = self._design_prompt_for("Sort the array. " * 200)

        assert was_truncated(prompt)

    def test_design_prompt_leaves_short_problems_alone(self):
        prompt = self._design_prompt_for("Sort the array.")

        assert not was_truncated(prompt)
        assert "Sort the array." in prompt

    def test_codegen_prompt_marks_the_problem_cut(self):
        from neo.algorithm_design import (
            AlgorithmClass, AlgorithmDesign, generate_code_from_design,
        )

        design = AlgorithmDesign(
            algorithm_class=AlgorithmClass.SORTING,
            key_insight="sort first",
            steps=["a"], edge_cases=["empty"], data_structures=["list"],
            complexity="O(n log n)", example_trace="1,2 -> 1,2",
        )
        adapter = _CapturingAdapter()
        try:
            generate_code_from_design(
                "Sort the array. " * 200, design, adapter, language="python",
            )
        except _PromptCaptured:
            pass

        assert adapter.called, "prompt builder never ran"
        assert was_truncated(adapter.prompt)


class TestElidedDesignLists:
    """A list is truncated as silently as a string — and this prompt says
    'follow this exactly', so three bullets read as the complete set."""

    def _codegen_prompt(self, *, steps, edge_cases, data_structures):
        from neo.algorithm_design import (
            AlgorithmClass, AlgorithmDesign, generate_code_from_design,
        )

        design = AlgorithmDesign(
            algorithm_class=AlgorithmClass.SORTING, key_insight="sort first",
            steps=steps, edge_cases=edge_cases, data_structures=data_structures,
            complexity="O(n log n)", example_trace="1,2 -> 1,2",
        )
        adapter = _CapturingAdapter()
        try:
            generate_code_from_design("Sort it.", design, adapter, language="python")
        except _PromptCaptured:
            pass
        assert adapter.called, "prompt builder never ran"
        return adapter.prompt

    def test_elided_steps_are_annotated_in_the_prompt(self):
        prompt = self._codegen_prompt(
            steps=[f"step {i}" for i in range(9)],
            edge_cases=["only"], data_structures=["list"],
        )

        assert "[showing 5 of 9]" in prompt

    def test_elided_edge_cases_are_annotated_in_the_prompt(self):
        prompt = self._codegen_prompt(
            steps=["one"],
            edge_cases=[f"case {i}" for i in range(7)], data_structures=["list"],
        )

        assert "[showing 3 of 7]" in prompt

    def test_elided_data_structures_are_annotated_in_the_prompt(self):
        prompt = self._codegen_prompt(
            steps=["one"], edge_cases=["only"],
            data_structures=["heap", "array", "trie", "graph"],
        )

        assert "[showing 3 of 4]" in prompt

    def test_annotation_sits_on_the_header_not_inside_the_value(self):
        """Glued to the value list, `[showing 3 of 4]` reads as a 4th entry."""
        prompt = self._codegen_prompt(
            steps=["one"], edge_cases=["only"],
            data_structures=["heap", "array", "trie", "graph"],
        )

        line = next(ln for ln in prompt.splitlines() if ln.startswith("Data structures:"))
        assert line == "Data structures: [showing 3 of 4]"

    def test_complete_lists_are_not_annotated(self):
        """The annotation must always mean an omission."""
        prompt = self._codegen_prompt(
            steps=["one"], edge_cases=["only"], data_structures=["list"],
        )

        assert "[showing" not in prompt


class TestFailureAnalysisPrompt:
    """`persistent_reasoning` — the extracted root cause is stored as a
    learned pitfall and replayed, so a wrong one outlives the run."""

    def _trace(self, frames=40):
        body = "".join(
            f'  File "mod{i}.py", line {i}, in f{i}\n    do_something()\n'
            for i in range(frames)
        )
        return body + "ValueError: database is locked\n"

    def test_the_exception_line_survives(self):
        """A traceback's answer is its LAST line.

        A head-cut of a 40-frame traceback keeps forty `File "..."` headers
        and discards `ValueError: database is locked` — the one fact the
        analysis exists to find. This is why the trace is elided in the
        middle rather than truncated at the tail.
        """
        from neo.persistent_reasoning import build_failure_analysis_prompt

        prompt = build_failure_analysis_prompt(self._trace(), "some suggestion")

        assert "ValueError: database is locked" in prompt

    def test_the_first_frame_survives_too(self):
        from neo.persistent_reasoning import build_failure_analysis_prompt

        prompt = build_failure_analysis_prompt(self._trace(), "s")

        assert 'File "mod0.py"' in prompt

    def test_the_cut_is_marked(self):
        from neo.persistent_reasoning import build_failure_analysis_prompt

        prompt = build_failure_analysis_prompt(self._trace(), "s")

        assert "characters elided" in prompt

    def test_short_trace_is_untouched(self):
        from neo.persistent_reasoning import build_failure_analysis_prompt

        prompt = build_failure_analysis_prompt("ValueError: boom\n", "s")

        assert "elided" not in prompt
        assert "ValueError: boom" in prompt

    def test_long_suggestion_is_marked(self):
        from neo.persistent_reasoning import build_failure_analysis_prompt

        prompt = build_failure_analysis_prompt("boom", "S" * 900)

        assert was_truncated(prompt)


class TestExemplarFormatting:
    """`engine._format_exemplar` — the "Similar Past Tasks" section.

    The old form appended a literal `...` whether or not anything was cut, so
    a 40-character solution was presented as continuing and a 4,000-character
    one was presented as continuing by exactly as much. A marker that fires
    unconditionally carries no information; one that lies in both directions
    is worse than none.
    """

    def _exemplar(self, solution):
        from types import SimpleNamespace
        return SimpleNamespace(prompt="how do I sort", solution=solution)

    def test_long_solution_is_marked(self):
        from neo.engine import _format_exemplar

        assert was_truncated(_format_exemplar(self._exemplar("S" * 4000)))

    def test_short_solution_is_not_marked(self):
        """The old literal ellipsis claimed a cut here that never happened."""
        from neo.engine import _format_exemplar

        rendered = _format_exemplar(self._exemplar("sorted(xs)"))
        assert not was_truncated(rendered)
        assert not rendered.endswith("...")
        assert rendered.endswith("sorted(xs)")

    def test_marker_reports_the_real_loss(self):
        from neo.engine import _EXEMPLAR_SOLUTION_CHARS, _format_exemplar

        rendered = _format_exemplar(self._exemplar("S" * 4000))
        assert f"{4000 - _EXEMPLAR_SOLUTION_CHARS} of 4000" in rendered


@pytest.mark.parametrize("module_name,helpers", [
    ("neo.engine", {"truncate_marked", "apportion"}),
    ("neo.pattern_extraction", {"truncate_marked"}),
    ("neo.constraint_verification", {"truncate_marked"}),
    ("neo.repair_loop", {"truncate_marked"}),
    ("neo.algorithm_design", {"truncate_marked", "shown_of"}),
    ("neo.persistent_reasoning", {"elide_middle", "truncate_marked"}),
])
def test_prompt_builders_call_the_shared_helpers(module_name, helpers):
    """One implementation, not seven — asserted on the AST, not on a grep.

    A substring search cannot tell a call from a mention, and a bare slice
    written `[0:N]` instead of `[:N]` slips past a negative grep for the old
    form. Parsing and looking for actual Call nodes is the guard that was
    meant; the grep version passed against source where every call site was
    still a bare slice.
    """
    import importlib

    module = importlib.import_module(module_name)
    with open(module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    missing = helpers - called
    assert not missing, f"{module_name} does not call {sorted(missing)}"


def test_no_prompt_builder_slices_a_bare_string_into_a_fenced_or_labelled_block():
    """Catches the reintroduction a rename would hide.

    Walks every f-string in the fixed modules and fails on a slice
    expression used as an interpolation, regardless of whether it is spelled
    `[:N]` or `[0:N]`.
    """
    import importlib

    offenders = []
    for module_name in (
        "neo.engine", "neo.pattern_extraction", "neo.constraint_verification",
        "neo.repair_loop", "neo.algorithm_design", "neo.persistent_reasoning",
    ):
        module = importlib.import_module(module_name)
        with open(module.__file__, encoding="utf-8") as handle:
            tree = ast.parse(handle.read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            literal = " ".join(
                part.value for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            ).lower()
            if not any(cue in literal for cue in ("```", "trace:", "response:", "problem:", "code:")):
                continue
            for part in node.values:
                if (isinstance(part, ast.FormattedValue)
                        and isinstance(part.value, ast.Subscript)
                        and isinstance(part.value.slice, ast.Slice)
                        and part.value.slice.upper is not None):
                    offenders.append(f"{module_name}:{node.lineno}")
    assert not offenders, f"bare slice interpolated into a prompt: {offenders}"
