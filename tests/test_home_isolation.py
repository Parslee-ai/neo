"""The test suite must not touch the developer's real ~/.neo.

conftest redirects `Path.home()`, but that alone never worked: modules capture
their paths in constants evaluated at IMPORT time, and pytest imports every
module during collection, before any fixture runs. Those constants kept
pointing at the real home.

Measured before the fix — one run of test_outcomes + test_fact_store +
test_transcript wrote into live state:

    ~/.neo/constraints/checksums.json
    ~/.neo/sessions/watermark_testproj1234.json

These tests assert the isolation is real *and* that its lookup table stays
complete, so adding a new `Path.home()` constant fails here rather than
silently leaking into someone's memory store months later.

Deliberately NOT a filesystem watcher: the background observer daemon writes to
~/.neo on its own schedule, so watching for changes there would fail at random.
Asserting on the constants is deterministic.
"""

import ast
import importlib
from pathlib import Path

import pytest

# `tests` is a package and `--import-mode=importlib` is in effect, so the
# conftest is reachable by its dotted name but not as a bare `conftest`.
from tests.conftest import HOME_PATH_CONSTANTS, REAL_HOME, _resolve_target

SRC = Path(__file__).resolve().parent.parent / "src" / "neo"

# `FASTEMBED_CACHE_DIR` is a read-mostly ~400 MB model cache, not neo state.
# conftest pins it to the REAL cache on purpose; redirecting it would
# re-download the model every run.
EXEMPT = {("neo.memory.store", "FASTEMBED_CACHE_DIR")}


def test_every_redirected_constant_lands_in_the_fake_home(isolate_neo_home):
    """The fixture's actual promise, asserted rather than assumed."""
    fake_home = isolate_neo_home
    for module_name, attribute, _relative in HOME_PATH_CONSTANTS:
        owner, attr_name = _resolve_target(module_name, attribute)
        value = Path(getattr(owner, attr_name))
        assert fake_home in value.parents or value == fake_home, (
            f"{module_name}.{attribute} escaped isolation: {value}"
        )


def test_no_constant_still_points_at_the_real_home(isolate_neo_home):
    """A redirect that silently failed would leave the real path in place.

    Uses the home captured at conftest import: by the time this runs, both
    `Path.home()` and `$HOME` point at the fake one, so `expanduser()` would
    compare the fake home against itself and pass trivially.
    """
    for module_name, attribute, _relative in HOME_PATH_CONSTANTS:
        owner, attr_name = _resolve_target(module_name, attribute)
        value = Path(getattr(owner, attr_name))
        assert REAL_HOME not in value.parents, (
            f"{module_name}.{attribute} still points into the real home: {value}"
        )


def _module_name_for(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    return ".".join(rel.parts)


class _StripDeferred(ast.NodeTransformer):
    """Remove subtrees whose evaluation is deferred past import.

    A `default_factory=lambda: Path.home() / ".claude"` runs at INSTANTIATION,
    by which point the fixture's `Path.home()` patch is in effect — so it is
    already safe and must not be reported. Only `Path.home()` evaluated
    eagerly at import time escapes isolation.
    """

    def visit_Lambda(self, node):  # noqa: N802 - ast API casing
        return ast.Constant(value=None)


# The spellings of "the user's home". `Path.home()` was the only one the first
# version knew, which let `observer._CAR_HINT_FLAG = os.path.expanduser(...)`
# and `observer._LOCK_PATH` sit unprotected while this test reported success.
#
# Matched against `ast.unparse` output, which normalizes string quotes to
# SINGLE — so a double-quoted variant here would never fire. An earlier draft
# listed both and looked like it covered twice as much as it did; single-quoted
# forms match double-quoted source too.
_HOME_SPELLINGS = (
    "Path.home()",
    "expanduser(",
    "environ['HOME']",
    "environ.get('HOME'",
    "getenv('HOME'",
    # Windows: no $HOME, and `expanduser` consults USERPROFILE there.
    "environ['USERPROFILE']",
    "environ.get('USERPROFILE'",
    "getenv('USERPROFILE'",
)


def _evaluates_home_at_import(value: ast.AST) -> bool:
    stripped = ast.unparse(
        _StripDeferred().visit(ast.parse(ast.unparse(value), mode="eval"))
    )
    return any(spelling in stripped for spelling in _HOME_SPELLINGS)


def _home_constants_in_source() -> set[tuple[str, str]]:
    """Every module- or class-level constant assigned from `Path.home()`.

    Function-local `Path.home()` calls are fine: they resolve when called, so
    the fixture's `Path.home()` patch already covers them. Only assignments
    evaluated at import time can outlive it.
    """
    found: set[tuple[str, str]] = set()
    for py in SRC.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        module_name = _module_name_for(py)

        def scan(body):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    scan(node.body)
                    continue
                # Module-level conditionals still execute at import.
                if isinstance(node, ast.If):
                    scan(node.body)
                    scan(node.orelse)
                    continue
                if isinstance(node, ast.Try):
                    scan(node.body)
                    for handler in node.handlers:
                        scan(handler.body)
                    scan(node.orelse)
                    scan(node.finalbody)
                    continue
                # `with` / `for` / `match` bodies also execute at import.
                if isinstance(node, (ast.With, ast.AsyncWith)):
                    scan(node.body)
                    continue
                if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
                    scan(node.body)
                    scan(node.orelse)
                    continue
                if isinstance(node, ast.Match):
                    for case in node.cases:
                        scan(case.body)
                    continue
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                if not (node.value and _evaluates_home_at_import(node.value)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        found.add((module_name, target.id))

        scan(tree.body)
    return found


def test_the_lookup_table_covers_every_import_time_home_constant():
    """The durable half — but a partial one. Know its limits.

    Anyone adding `X = Path.home() / ".neo" / ...` (or an `expanduser`/`$HOME`
    spelling) at module or class level, including inside a module-level `if`
    or `try`, gets a failure here instead of a silent leak into real state.

    It CANNOT catch, and no AST grep of this shape could:

      - **Derived constants.** `CHECKSUM_FILE = CHECKSUM_DIR / "checksums.json"`
        unparses to a `Name`; nothing in it says "home". Those entries are in
        the table because a human put them there, and a new sibling would be
        invisible here.
      - **Re-imported bindings.** `from ...outcomes import SESSIONS_DIR` is an
        `ImportFrom`, never an `Assign` — and that is precisely the case the
        conftest comment records as having bitten us before.
      - **Aliased imports.** `from pathlib import Path as P` then `P.home()`.
      - **Non-`Name` targets.** `Foo.BAR = Path.home() / ...` at module level,
        or tuple unpacking.

    So treat a pass here as "no NEW constant of the common shape slipped in",
    not as proof of completeness. The two runtime assertions above are the
    ones that actually verify isolation for what is in the table.
    """
    declared = {(m, a.split(".")[-1]) for m, a, _ in HOME_PATH_CONSTANTS}
    in_source = {
        (m, a) for (m, a) in _home_constants_in_source()
        if (m, a) not in {(m2, a2.split(".")[-1]) for m2, a2 in EXEMPT}
    }

    missing = {
        (m, a) for (m, a) in in_source
        if (m, a) not in declared
    }
    assert not missing, (
        "These import-time Path.home() constants are not redirected by "
        "conftest.HOME_PATH_CONSTANTS, so tests will write to the real home:\n  "
        + "\n  ".join(f"{m}.{a}" for m, a in sorted(missing))
    )


def test_every_spelling_can_actually_match():
    """`ast.unparse` normalizes quotes to single, so a double-quoted spelling
    is inert. An earlier draft carried four such entries and looked like it
    covered twice the surface it did."""
    for spelling in _HOME_SPELLINGS:
        assert '"' not in spelling, (
            f"{spelling!r} can never match: ast.unparse emits single quotes"
        )


@pytest.mark.parametrize("source", [
    'X = Path.home() / ".neo" / "x"',
    'X = os.path.expanduser("~/.neo/x")',
    'X = os.environ["HOME"] + "/.neo"',
    'X = os.getenv("HOME")',
    'X = os.environ["USERPROFILE"]',
])
def test_the_scan_recognizes_each_home_spelling(source):
    """Double-quoted source must still be caught — the point of matching on
    unparse output rather than raw text."""
    node = ast.parse(source).body[0]
    assert _evaluates_home_at_import(node.value), source


@pytest.mark.parametrize("wrapper", [
    "if True:\n    {stmt}",
    "try:\n    {stmt}\nexcept Exception:\n    pass",
    "with open('f') as fh:\n    {stmt}",
    "for _ in range(1):\n    {stmt}",
])
def test_the_scan_recurses_into_module_level_blocks(wrapper, tmp_path):
    """These all execute at import, so a constant inside one escapes just the
    same as a top-level assignment."""
    src = wrapper.format(stmt='LEAKY = Path.home() / ".neo" / "x"')
    tree = ast.parse(src)

    found = []

    def scan(body):
        for node in body:
            if isinstance(node, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
                scan(node.body)
                for handler in getattr(node, "handlers", []):
                    scan(handler.body)
                scan(getattr(node, "orelse", []))
                scan(getattr(node, "finalbody", []))
                continue
            if isinstance(node, ast.Assign) and _evaluates_home_at_import(node.value):
                found.extend(t.id for t in node.targets if isinstance(t, ast.Name))

    scan(tree.body)
    assert found == ["LEAKY"], f"not caught inside: {wrapper}"


def test_the_lookup_table_has_no_stale_entries():
    """A renamed or deleted constant should not linger in the table pretending
    to protect something."""
    for module_name, attribute, _relative in HOME_PATH_CONSTANTS:
        owner, attr_name = _resolve_target(module_name, attribute)
        assert hasattr(owner, attr_name), (
            f"{module_name}.{attribute} no longer exists; drop it from the table"
        )


@pytest.mark.parametrize("module_name,attribute", sorted(EXEMPT))
def test_exempt_constants_still_exist(module_name, attribute):
    """If an exemption's target disappears, the exemption is misleading."""
    module = importlib.import_module(module_name)
    assert hasattr(module, attribute)


def test_fastembed_cache_is_deliberately_not_redirected(isolate_neo_home):
    """Pinned to the real cache so the model is not re-downloaded per run.
    Documented here so a future reader doesn't 'fix' it."""
    from neo.memory import store

    assert isolate_neo_home not in Path(store.FASTEMBED_CACHE_DIR).parents
