"""Tests for pattern_extraction's language-aware fence tagging."""

import re

from neo.languages import _FENCE_TAGS
from neo.pattern_extraction import extract_pattern_from_correction


class _CapturingAdapter:
    """Adapter stub that records prompts instead of calling an LM.

    Returns a non-empty response so the (lenient) parser fills in
    something for each field — we don't care about the parse output,
    only the prompt we sent.
    """

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, messages, **_kwargs):
        self.prompts.append(messages[0]["content"])
        # Minimal response shape so parsing doesn't crash. Fields don't
        # have to be meaningful — pattern_extraction will produce
        # something we discard.
        return (
            "1. Signature keywords: foo, bar\n"
            "2. Common mistake: did the thing wrong\n"
            "3. Prevention rule: don't do the thing wrong\n"
            "4. Generalization: yes\n"
        )


def _extract(adapter, **overrides):
    """Invoke extract_pattern_from_correction with defaulted args."""
    kwargs = dict(
        problem_description="problem",
        failed_code="x = 1",
        corrected_code="x = 2",
        bug_category="cat",
        root_cause="cause",
        adapter=adapter,
    )
    kwargs.update(overrides)
    extract_pattern_from_correction(**kwargs)


class TestFenceTagging:
    def test_python_language_renders_python_fence(self):
        a = _CapturingAdapter()
        _extract(a, language="python")
        assert "```python" in a.prompts[0]

    def test_javascript_renders_javascript_fence(self):
        a = _CapturingAdapter()
        _extract(a, language="javascript")
        assert "```javascript" in a.prompts[0]

    def test_treesitter_c_sharp_normalized_to_csharp(self):
        # tree-sitter uses 'c_sharp'; GFM uses 'csharp'. The map
        # normalizes.
        a = _CapturingAdapter()
        _extract(a, language="c_sharp")
        assert "```csharp" in a.prompts[0]

    def test_no_language_means_no_fence_tag(self):
        # No language → bare ``` fences, no language tag of any kind.
        # Regex checks the invariant (no fence-start followed by alpha)
        # rather than a fragile substring match.
        a = _CapturingAdapter()
        _extract(a, language=None)
        assert not re.search(r"^```[a-z]", a.prompts[0], re.MULTILINE)

    def test_unknown_language_falls_back_to_bare_fence(self):
        # Defensive: an unrecognized language label shouldn't crash and
        # shouldn't fabricate a fence tag.
        a = _CapturingAdapter()
        _extract(a, language="nonexistent")
        assert not re.search(r"^```[a-z]", a.prompts[0], re.MULTILINE)

    def test_fence_map_keys_are_lowercased(self):
        # The lookup lowercases the input; map keys should already be
        # lowercase so the contract is clean.
        for key in _FENCE_TAGS:
            assert key == key.lower()
