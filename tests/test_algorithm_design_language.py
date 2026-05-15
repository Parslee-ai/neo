"""Tests for algorithm_design language threading and code extraction."""

from neo.algorithm_design import (
    AlgorithmClass,
    AlgorithmDesign,
    _extract_code_block,
    generate_code_from_design,
)


class _CapturingAdapter:
    """Adapter stub that records prompts and replays a canned response."""

    def __init__(self, response: str = ""):
        self.prompts: list[str] = []
        self.response = response

    def generate(self, messages, **_kwargs):
        self.prompts.append(messages[0]["content"])
        return self.response


def _design() -> AlgorithmDesign:
    return AlgorithmDesign(
        algorithm_class=AlgorithmClass.GREEDY,
        key_insight="example",
        steps=["step 1", "step 2"],
        edge_cases=["empty input"],
        data_structures=["list"],
        complexity="O(n)",
        example_trace="x -> y",
    )


class TestLanguageInPrompt:
    def test_language_is_required(self):
        # language must be passed explicitly — no Python default to
        # silently mislabel prompts.
        import pytest
        a = _CapturingAdapter("```python\nprint(1)\n```")
        with pytest.raises(TypeError):
            generate_code_from_design("problem", _design(), a)

    def test_python_renders_in_prompt(self):
        a = _CapturingAdapter("```python\nprint(1)\n```")
        generate_code_from_design("problem", _design(), a, language="python")
        assert "in Python:" in a.prompts[0]
        assert "Generate Python code" in a.prompts[0]

    def test_javascript_renders_in_prompt(self):
        a = _CapturingAdapter("```javascript\nconsole.log(1)\n```")
        generate_code_from_design("problem", _design(), a, language="javascript")
        assert "in JavaScript:" in a.prompts[0]

    def test_csharp_uses_display_name(self):
        # Display name is the human-readable form ("C#"), distinct from
        # the fence tag (`csharp`).
        a = _CapturingAdapter("```csharp\nint x = 1;\n```")
        generate_code_from_design("problem", _design(), a, language="csharp")
        assert "in C#:" in a.prompts[0]

    def test_unknown_language_passes_through(self):
        # Defensive: anything not in the display map falls through as-is
        # rather than crashing.
        a = _CapturingAdapter("```\ncode\n```")
        generate_code_from_design("problem", _design(), a, language="brainfuck")
        assert "in brainfuck:" in a.prompts[0]


class TestExtractCodeBlock:
    def test_matching_tagged_fence(self):
        out = _extract_code_block("```python\nprint(1)\n```", "python")
        assert out == "print(1)"

    def test_bare_fence_with_no_tag(self):
        out = _extract_code_block("```\nprint(1)\n```", "python")
        assert out == "print(1)"

    def test_mismatched_tag_strips_leading_token(self):
        # Asked for `c_sharp`, LM emitted `csharp` — the tagged lookup
        # fails, the bare fallback fires and strips the leading
        # language-name line.
        out = _extract_code_block("```csharp\nint x = 1;\n```", "c_sharp")
        assert out == "int x = 1;"

    def test_no_fences_returns_raw(self):
        out = _extract_code_block("print(1)", "python")
        assert out == "print(1)"

    def test_first_real_code_line_not_stripped(self):
        # If the LM omits the language tag and starts with a code line
        # that happens to be a single token (rare but possible), the
        # heuristic could over-strip. This pins the limit: lines with
        # spaces or non-alnum punctuation aren't stripped.
        body = "```\nx = 1\nprint(x)\n```"
        assert _extract_code_block(body, "python") == "x = 1\nprint(x)"

    def test_picks_up_with_extra_text_before_fence(self):
        text = "Here is the code:\n```python\nx = 1\n```\nDone."
        assert _extract_code_block(text, "python") == "x = 1"

    def test_pass_keyword_first_line_not_stripped(self):
        # Pre-tightening, `pass` would have been stripped as a leading
        # "language tag" by the alnum heuristic. The known-tag set
        # protects against this — `pass` isn't in _KNOWN_FENCE_TAGS.
        out = _extract_code_block("```\npass\nprint(1)\n```", "python")
        assert out == "pass\nprint(1)"

    def test_single_identifier_first_line_not_stripped(self):
        # Same protection — `done`, `null`, `data`, etc. are not fence tags.
        out = _extract_code_block("```\ndone\n```", "python")
        assert out == "done"

    def test_pythonic_not_matched_as_python(self):
        # The tagged lookup requires a newline after the tag, so
        # ```pythonic doesn't collide with ```python.
        body = "```pythonic\nx = 1\n```"
        # Falls through to bare extraction. `pythonic` isn't a known
        # tag, so it stays as the first line.
        assert _extract_code_block(body, "python") == "pythonic\nx = 1"

    def test_multiple_fenced_blocks_takes_first(self):
        # Documented limit: when the LM emits explanation + code as
        # two fenced blocks, we take the first. Tagged-first lookup
        # protects the common case of "explanation in ```text, code
        # in ```python" — we still find the python block.
        text = (
            "Explanation:\n"
            "```text\n"
            "Some prose here.\n"
            "```\n"
            "Code:\n"
            "```python\n"
            "x = 1\n"
            "```\n"
        )
        # Looking for ```python\n — finds the second fence directly.
        assert _extract_code_block(text, "python") == "x = 1"
