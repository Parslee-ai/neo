"""Reproduction test for issue #9005.

The structural fingerprint (engine._suggestion_fingerprint) is meant to tighten
episode correlation for promotion: two accepted suggestions only merge toward a
durable memory when their prompt-prefix AND diff shape agree. But the underlying
_extract_code_skeleton ast.parse()s the raw snippet, which is empty for the two
MOST COMMON edit shapes:

  1. a method indented inside a class (IndentationError -> ""),
  2. a small/partial unified diff whose added lines are a mid-block fragment.

When the fingerprint is "", the correlation key degrades to subject (prompt+path)
only, so two unrelated fixes sharing a similar prompt and file can be promoted
together into one over-trusted memory. These tests assert the fingerprint IS
populated (and shape-sensitive) for those shapes, and therefore FAIL on the
buggy code.
"""

from types import SimpleNamespace

from neo.engine import NeoEngine


class FakeLM:
    def __init__(self):
        self.model = "fake"
        self.provider = "fake"

    def generate(self, messages, **kw):
        return ""

    def name(self):
        return "fake-lm"


def _engine():
    return NeoEngine(lm_adapter=FakeLM(), enable_persistent_memory=False)


def test_indented_class_method_has_non_empty_fingerprint():
    """An edit to a method inside a class is the dominant real-world suggestion
    shape. Its code_block is indented one level, so a bare ast.parse raises
    IndentationError and yields "". The fingerprint must be non-empty."""
    e = _engine()
    method_edit = SimpleNamespace(
        code_block=(
            "    def refresh(self, items):\n"
            "        result = []\n"
            "        seen = set()\n"
            "        for x in items:\n"
            "            if x not in seen:\n"
            "                seen.add(x)\n"
            "                result.append(x)\n"
            "        return result\n"
        ),
        unified_diff="",
    )
    fp = e._suggestion_fingerprint(method_edit)
    assert fp, "indented class method should yield a non-empty fingerprint"
    assert len(fp) == 12


def test_partial_unified_diff_has_non_empty_fingerprint():
    """A realistic small patch: the added lines are a mid-block fragment indented
    inside an existing function. ast.parse of the bare added lines fails, so the
    fingerprint is "". It must be populated."""
    e = _engine()
    partial_diff = SimpleNamespace(
        code_block="",
        unified_diff=(
            "--- a/src/neo/foo.py\n"
            "+++ b/src/neo/foo.py\n"
            "@@ -10,6 +10,9 @@ def handle(self, items):\n"
            "         seen = set()\n"
            "+        for x in items:\n"
            "+            if x not in seen:\n"
            "+                seen.add(x)\n"
            "         return seen\n"
        ),
    )
    fp = e._suggestion_fingerprint(partial_diff)
    assert fp, "partial unified diff should yield a non-empty fingerprint"
    assert len(fp) == 12


def test_two_different_method_edits_fingerprint_differently():
    """Structurally different method edits must not collapse to the same
    fingerprint (which would let unrelated fixes correlate)."""
    e = _engine()
    loop_method = SimpleNamespace(
        code_block=(
            "    def refresh(self, items):\n"
            "        result = []\n"
            "        for x in items:\n"
            "            result.append(x)\n"
            "        return result\n"
        ),
        unified_diff="",
    )
    comprehension_method = SimpleNamespace(
        code_block=(
            "    def refresh(self, items):\n"
            "        return list(dict.fromkeys(items))\n"
        ),
        unified_diff="",
    )
    fp_loop = e._suggestion_fingerprint(loop_method)
    fp_comp = e._suggestion_fingerprint(comprehension_method)
    assert fp_loop and fp_comp
    assert fp_loop != fp_comp


def test_non_python_code_still_degrades_to_empty():
    """Truly unparseable / non-Python code must still return "" (subject-only
    fallback preserved)."""
    e = _engine()
    prose = SimpleNamespace(
        code_block="this is not code at all -- just a sentence.",
        unified_diff="",
    )
    assert e._suggestion_fingerprint(prose) == ""
