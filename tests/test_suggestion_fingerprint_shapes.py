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


class TestFingerprintIsNameAgnostic:
    """The fingerprint must capture the SHAPE of a change, not the identifier.

    `_extract_code_skeleton` emits `def:<name>`, kept deliberately as readable
    metadata on the fact. Hashing it made the identical fix to `read_text` and
    `read_body` two different signatures, so a genuinely recurring lesson could
    never accumulate the two acceptances promotion requires. A live drill of
    real `neo` runs measured four git-verified acceptances and zero promotions;
    the next pair promoted after this.
    """

    def _fp(self, code):
        return _engine()._suggestion_fingerprint(
            SimpleNamespace(code_block=code, unified_diff="")
        )

    def _ctx_mgr(self, name):
        return (f"def {name}(path):\n"
                f"    with open(path) as f:\n"
                f"        data = f.read()\n"
                f"    return data\n")

    def test_same_shape_different_function_names_match(self):
        a, b = self._fp(self._ctx_mgr("read_text")), self._fp(self._ctx_mgr("read_body"))
        assert a and a == b

    def test_long_names_do_not_shift_the_truncation_window(self):
        """Names must be normalized BEFORE the 500-char cut, not after.

        `def:<name>` is the only unbounded-length token, so long identifiers are
        exactly what pushes a skeleton past the cap. Post-hoc stripping left two
        identical shapes truncated at different points and still hashing
        differently (measured: short names -> 316 chars, long names -> 500).
        """
        def module(prefix, n=40):
            return "\n\n".join(f"def {prefix}{i}(path):\n    return path"
                                for i in range(n))
        assert self._fp(module("f")) == self._fp(module("a_very_long_helper_name_"))

    def test_async_and_sync_variants_of_one_fix_correlate(self):
        """`AsyncFunctionDef` is not a `FunctionDef` subclass, so `async def`
        previously emitted no def token at all."""
        async_form = ("async def go(path):\n"
                      "    with open(path) as f:\n"
                      "        data = f.read()\n"
                      "    return data\n")
        assert self._fp(async_form) == self._fp(self._ctx_mgr("go"))

    def test_body_shape_still_discriminates(self):
        """Renamed from "different shape": the name is held constant here, so
        this pins body-sensitivity specifically."""
        other_body = ("def read_text(path):\n"
                      "    for line in open(path):\n"
                      "        print(line)\n")
        assert self._fp(self._ctx_mgr("read_text")) != self._fp(other_body)

    def test_method_names_are_still_shape(self):
        """`method:append` vs `method:pop` IS a structural difference — those
        come from a bounded whitelist, unlike a free function name."""
        a = self._fp("def f(xs):\n    out = []\n    out.append(1)\n    return out\n")
        b = self._fp("def f(xs):\n    out = []\n    out.pop()\n    return out\n")
        assert a != b

    def test_structurally_distinct_fixes_do_not_collide(self):
        """Guards #144's property: normalizing names must not make unrelated
        accepted fixes merge into one over-trusted PATTERN. Shapes that differ
        in control flow or data structures must keep distinct fingerprints even
        with identical names.
        """
        shapes = [
            "def f(xs):\n    return [x for x in xs]\n",
            "def f(xs):\n    return {x for x in xs}\n",
            "def f(xs):\n    d = {}\n    return d\n",
            "def f(xs):\n    while xs:\n        xs.pop()\n    return xs\n",
            "def f(xs):\n    for x in xs:\n        print(x)\n",
            "def f(xs):\n    if xs:\n        return xs\n    return None\n",
        ]
        fps = [self._fp(c) for c in shapes]
        assert all(fps), "every shape must fingerprint"
        assert len(set(fps)) == len(fps), f"collision among distinct shapes: {fps}"
