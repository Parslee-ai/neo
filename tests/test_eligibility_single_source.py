"""G4-inv, structural half: exactly one eligibility implementation.

The differential suite proves the walk is *right*. This one proves it is
*alone* — that a future change cannot quietly add a second answer to "which
paths exist and are eligible" beside it.

That is not a hypothetical failure mode; it is the one this whole goal
exists to close. The gatherer and the index each carried their own walk and
their own ignore list, the copies drifted for months, and the drift was
invisible from the outside: the index built an index of 83 stale-worktree
Python files for a repository of 4,272 C# files and exited 0 (#159), while
the gatherer served gitignored build output as context (#186). Each defect
was fixed on one side. Neither fix reached the other copy. A reviewer cannot
see a second list by reading a diff that does not contain it, so the check
has to be a test.

Three guards, each aimed at a different way a second implementation gets
introduced:

1. **Definition sites.** Every named piece of eligibility logic is defined
   exactly once, in `neo/eligibility.py`. Re-exporting or importing it
   elsewhere is fine and expected; defining it twice is not.
2. **Walks.** `os.walk` appears in exactly one module. A second walk is how
   both historical copies started — not as a copied ignore list, but as a
   loop that grew one.
3. **Ignore lists.** No module outside `eligibility.py` may hold a
   collection of directory names that is recognisably an exclusion list.
   This is the copy-paste route, and it is caught by content rather than by
   name because the name is the one thing a copy always changes.

The scan is AST-based, not grep-based, so a name inside a comment, a
docstring or this module's own prose cannot trip it — every previous
generation of "one place" comment in this codebase was a comment, and a
comment has never stopped anyone.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.invariants

SRC = Path(__file__).resolve().parent.parent / "src" / "neo"
ELIGIBILITY = SRC / "eligibility.py"

#: The functions that ARE the eligibility implementation. Each must have
#: exactly one `def` in the whole package, and it must be in `eligibility.py`.
CANONICAL_DEFINITIONS = (
    "load_ignore_patterns",   # what the ignore rules are
    "compile_glob",           # how one pattern is matched
    "should_ignore",          # whether a path is ignored
    "walk",                   # which paths exist
    "file_content_hash",      # whether two paths hold the same bytes
)

#: Names that, appearing together in one literal collection, mean somebody
#: has written an exclusion list. Chosen because each is a derived-artifact
#: directory with no other plausible reason to sit in a list of strings, and
#: because these exact names are what the two historical copies shared.
IGNORE_LIST_SENTINELS = frozenset({
    "node_modules", "__pycache__", ".venv", ".worktrees", "site-packages",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "bower_components",
    ".egg-info", "htmlcov", ".tox",
})

#: How many sentinels in one literal make it an exclusion list rather than a
#: coincidence. Two is reachable by accident (a test fixture naming a couple
#: of directories); three is not.
IGNORE_LIST_THRESHOLD = 3


def _modules():
    """Every module under `src/neo`, as (path, parsed tree)."""
    for path in sorted(SRC.rglob("*.py")):
        yield path, ast.parse(path.read_text(), filename=str(path))


def _relative(path: Path) -> str:
    return str(path.relative_to(SRC.parent.parent))


class TestOneDefinitionSite:
    @pytest.mark.parametrize("name", CANONICAL_DEFINITIONS)
    def test_defined_exactly_once_and_in_the_eligibility_module(self, name):
        sites = []
        for path, tree in _modules():
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == name:
                        sites.append(f"{_relative(path)}:{node.lineno}")

        assert len(sites) == 1, (
            f"`{name}` is eligibility logic and must have one definition; "
            f"found {len(sites)}: {sites}"
        )
        assert sites[0].startswith(_relative(ELIGIBILITY)), (
            f"`{name}` is defined in {sites[0]}, not in the eligibility module"
        )

    def test_the_default_pattern_list_is_defined_once(self):
        """The list itself, by name, in addition to the content guard below."""
        modules = sorted({
            _relative(path)
            for path, tree in _modules()
            for node in ast.walk(tree)
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if isinstance(target, ast.Name)
            and target.id == "DEFAULT_IGNORE_PATTERNS"
        })
        assert modules == [_relative(ELIGIBILITY)], (
            f"DEFAULT_IGNORE_PATTERNS is assigned in {modules}"
        )


#: Modules that each historically grew their own walk. A *recursive* traversal
#: in one of these is the regression this goal exists to prevent, so they are
#: held to a stricter rule than the rest of the package.
FORMER_WALKERS = (
    "context_gatherer.py",
    "index/project_index.py",
    "architecture_metrics.py",
)


class TestNoSecondWalk:
    def test_os_walk_is_called_in_one_module_only(self):
        """A second walk is how both historical copies began."""
        callers = set()
        for path, tree in _modules():
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "walk"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "os"
                ):
                    callers.add(_relative(path))

        assert callers == {_relative(ELIGIBILITY)}, (
            "os.walk must be called only by the shared eligibility walk; "
            f"also called by: {sorted(callers - {_relative(ELIGIBILITY)})}"
        )

    def test_no_module_reaches_for_another_directory_traversal_primitive(self):
        """`os.walk` is not the only spelling of "walk the filesystem".

        Guarding the attribute form alone leaves `from os import walk`,
        `os.scandir` and `os.listdir` as unwatched doors into the same
        mistake — and a guard that names one spelling of a defect reads as
        coverage while the other three walk past it. None of these appears
        anywhere in the package today, so this asserts a property that
        currently holds rather than grandfathering exceptions.
        """
        offenders = []
        for path, tree in _modules():
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.module == "os"
                    and any(
                        alias.name in ("walk", "scandir", "listdir")
                        for alias in node.names
                    )
                ):
                    offenders.append(f"{_relative(path)}:{node.lineno} from os import …")
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("scandir", "listdir")
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                ):
                    offenders.append(
                        f"{_relative(path)}:{node.lineno} os.{node.func.attr}"
                    )

        assert offenders == [], (
            "directory traversal belongs to `neo.eligibility.walk`:\n"
            + "\n".join(offenders)
        )

    @pytest.mark.parametrize("module", FORMER_WALKERS)
    def test_a_former_walker_does_not_recursively_glob(self, module):
        """`Path.rglob` / `glob("**/…")` is a walk wearing a shorter name.

        Scoped to the three modules that each grew their own traversal,
        rather than banned package-wide, because neo legitimately globs its
        OWN state directories (`~/.neo/facts`, session logs, transcript
        dirs) and a blanket rule would have to grandfather twenty call sites
        — at which point the next one slips in beside them. #159 was
        `ProjectIndex` reaching for `Path.glob`, so the mechanism is watched
        exactly where it has already fired.
        """
        tree = ast.parse((SRC / module).read_text())
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "rglob":
                offenders.append(f"{module}:{node.lineno} .rglob(")
            elif node.func.attr in ("glob", "iglob"):
                first = node.args[0] if node.args else None
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if "**" in first.value:
                        offenders.append(
                            f"{module}:{node.lineno} .glob({first.value!r})"
                        )

        assert offenders == [], (
            f"{module} is walking the repository again; call "
            "`neo.eligibility.walk` instead:\n" + "\n".join(offenders)
        )


class TestNoSecondIgnoreList:
    def test_no_module_holds_its_own_exclusion_list(self):
        """Caught by content, because a copy always renames the variable."""
        offenders = []
        for path, tree in _modules():
            if path == ELIGIBILITY:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                    continue
                literals = {
                    element.value
                    for element in node.elts
                    if isinstance(element, ast.Constant)
                    and isinstance(element.value, str)
                }
                hits = literals & IGNORE_LIST_SENTINELS
                if len(hits) >= IGNORE_LIST_THRESHOLD:
                    offenders.append(
                        f"{_relative(path)}:{node.lineno} -> {sorted(hits)}"
                    )

        assert offenders == [], (
            "a second exclusion list has appeared; call "
            "`neo.eligibility.load_ignore_patterns` instead of restating it:\n"
            + "\n".join(offenders)
        )

    def test_the_guard_actually_fires(self):
        """Proof the sentinel check is not vacuous.

        A structural guard that cannot fail is worse than no guard: it reads
        as coverage in CI while asserting nothing. This runs the same
        detection over a module that DOES restate the list.
        """
        tree = ast.parse(
            "SKIP = ['node_modules', '__pycache__', '.venv', 'src']\n"
        )
        found = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.List)
            and len(
                {
                    element.value
                    for element in node.elts
                    if isinstance(element, ast.Constant)
                }
                & IGNORE_LIST_SENTINELS
            )
            >= IGNORE_LIST_THRESHOLD
        ]
        assert len(found) == 1


class TestConsumersActuallyConsumeIt:
    """One definition proves nothing if a consumer stopped calling it.

    The definition-site guards above would all stay green if
    `project_index` quietly went back to `Path.glob` and simply never
    defined anything named like eligibility logic. These assert the wiring.
    """

    @pytest.mark.parametrize("module", [
        "context_gatherer.py",
        "index/project_index.py",
        "architecture_metrics.py",
    ])
    def test_consumer_imports_the_shared_module(self, module):
        tree = ast.parse((SRC / module).read_text())
        # Both spellings count: `from neo.eligibility import walk_paths` and
        # `from neo import eligibility`. Pinning one would be a style rule
        # wearing an invariant's clothes.
        imports_it = any(
            (isinstance(node, ast.ImportFrom) and node.module == "neo.eligibility")
            or (
                isinstance(node, ast.ImportFrom)
                and node.module == "neo"
                and any(alias.name == "eligibility" for alias in node.names)
            )
            or (
                isinstance(node, ast.Import)
                and any(alias.name == "neo.eligibility" for alias in node.names)
            )
            for node in ast.walk(tree)
        )
        assert imports_it, (
            f"{module} no longer imports the shared eligibility module"
        )

    def test_the_index_does_not_glob_the_repository_itself(self):
        """`self.repo_root.glob(...)` was the index's own walk."""
        source = (SRC / "index" / "project_index.py").read_text()
        assert "repo_root.glob(" not in source
        assert "repo_root.rglob(" not in source
