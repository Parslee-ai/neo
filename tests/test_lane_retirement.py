"""Lane retirement: the two-lane era leaves no remnant, in code or in prose.

Unified-store Goal 9. Goals 5-8 built the unified store; this file is the
guard that keeps it unified after the people who built it have moved on.

`tests/test_front_door.py` proves the pipeline BEHAVES as one path. That is a
runtime claim, and it is not the one that decays. What decays is the
surrounding text: a flag help string, a README row, a tip printed on a first
run. Every one of those was still describing a fork to a second gather
function four merged PRs after the fork was deleted, and none of them failed a
test, because prose is not executed.

The distinction matters more here than it usually does. `--semantic` no longer
selects an implementation, but a user who reads "use the semantic index for
file selection (requires `.neo/index.json`)" will still believe it does, will
still believe an unindexed repo gets a lesser selection, and will still run
`neo --index` as a prerequisite it has not been since Goal 7. A false mental
model produces the same wrong behaviour a real second lane would; it just does
it in the operator instead of in the process.

Four guards:

1. **Deleted lane functions are not named anywhere that ships.** Text scan, not
   AST — a comment or a docstring naming a function that no longer exists is
   precisely the failure being caught, so a scan that skips prose skips the bug.
2. **One gather entry point.** `gather_context` is defined once, and `cli.py`
   calls that name and no sibling of it.
3. **`--semantic` never routes.** The flag reaches the pipeline as data on
   `GatherConfig` and is never read by `cli.py` to choose what to call.
4. **The retired vocabulary is gone from the docs a user reads.** A named list
   of exact phrases, each of which was live in this repo and each of which
   asserts the retired model.

Scope of guards 1 and 4 is deliberately narrow and named: what ships to a user
or an agent. Two exemptions, both principled rather than convenient. **Dated
measurement records** under `docs/` (`goal<N>-*-measurements-<date>.md`) are
evidence of what was true on the day they were produced -- the Goal 8 record
compares an arm whose code DID contain `gather_context_semantic` against one
that did not, and editing that sentence to satisfy a grep would make the record
lie about the experiment. **`tests/`** is exempt from guard 1 for the same
reason in miniature: a test that asserts the absence of a symbol must be
allowed to spell it.

**Known limit of the scope, stated rather than papered over.**
`SHIPPED_DOC_ROOTS` is an ALLOW-list, so anything not enumerated is unguarded
by default. `CHANGELOG.md` is the live example: it holds a verbatim retired
claim at its `0.36.0` entry, and it is deliberately out of scope for the same
reason the measurement records are -- a changelog entry records what a release
actually printed, and rewriting it would falsify the release. But a
deny-list would be the stronger shape, because the failure mode of an
allow-list is silence about a file nobody remembered to add, and the failure
mode of a deny-list is a test that fails until someone justifies the
exemption. Inverting it is the obvious next improvement to this file.
"""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.invariants

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "neo"

#: Functions deleted when the second lane was removed (#214). Each was a
#: private or public member of the semantic gather path; none has a live
#: definition anywhere in the package. `_cut_to_bytes` is NOT here on purpose:
#: it was deleted in the same commit and then re-introduced with a different
#: signature for the unified budget path, and listing a name that is alive
#: again would make this guard fail for the opposite of the reason it exists.
DELETED_LANE_FUNCTIONS = (
    "gather_context_semantic",
    "mmr_pack_chunks",
    "log_context_metrics",
)

#: Exact strings that assert the retired mental model. **Every entry here must
#: be text that was actually live in this repository**, or the guard is
#: decorative — it fails nothing, and it advertises coverage it does not have.
#: A fourth entry ("run `neo --index` first to enable") was written from
#: imagination and removed on review: `git log --all -S` found it in exactly
#: one commit, the one that added the guard. Before adding a claim here, prove
#: it exists: `git log --all -S'<claim>' --oneline`.
#:
#: Three claims, two sites — the first two were the same README line
#: (`README.md:660` on `origin/main`), the third was the first-run tip in
#: `context_gatherer.py`. Matched case-insensitively as substrings, so a
#: rewording that keeps the claim keeps failing.
RETIRED_CLAIMS = (
    # `--semantic` as a mode you switch into, rather than a weight hint.
    "use the semantic index for file selection",
    # The catalog as a precondition of selection working at all.
    "requires `.neo/index.json`",
    "enable semantic file selection",
)

#: Where a claim reaches a human or an agent. Directories are walked for
#: `.md`; files are read directly.
SHIPPED_DOC_ROOTS = (
    ROOT / "README.md",
    ROOT / "QUICKSTART.md",
    ROOT / "INSTALL.md",
    ROOT / "AGENTS.md",
    ROOT / "docs",
    ROOT / ".claude-plugin",
    ROOT / "plugins",
)

#: A dated record of a measurement is a historical document. See the module
#: docstring for why it is exempt rather than edited.
_MEASUREMENT_RECORD = re.compile(r"^goal\d+-.*-measurements-\d{4}-\d{2}-\d{2}\.md$")


def _shipped_docs() -> list[Path]:
    """Every markdown file a user or a host agent can read, minus the dated
    measurement records."""
    out: list[Path] = []
    for target in SHIPPED_DOC_ROOTS:
        if target.is_file():
            out.append(target)
        elif target.is_dir():
            out.extend(
                p for p in sorted(target.rglob("*.md"))
                if not _MEASUREMENT_RECORD.match(p.name)
            )
    return out


def _shipped_sources() -> list[Path]:
    """Everything under `src/`, plus the plugin manifests. `tests/` is not
    here: see the module docstring."""
    return sorted(SRC.rglob("*.py")) + sorted(
        p for p in (ROOT / ".claude-plugin").rglob("*.json")
    ) + sorted(p for p in (ROOT / "plugins").rglob("*.json"))


class TestDeletedLaneFunctionsLeaveNoName:
    """Guard 1. A dead function's NAME outliving it is not cosmetic: it sends
    the next reader to grep for code that is not there, and it is the only
    trace a half-finished deletion leaves."""

    @pytest.mark.parametrize("symbol", DELETED_LANE_FUNCTIONS)
    def test_no_definition_survives(self, symbol):
        defined_in = []
        for path in sorted(SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == symbol:
                        defined_in.append(f"{path.relative_to(ROOT)}:{node.lineno}")

        assert not defined_in, (
            f"`{symbol}` was deleted with the second lane but is defined again "
            f"at {defined_in}. If the unified pipeline needs this behaviour it "
            f"belongs inside `gather_context`'s four stages, under a name that "
            f"does not claim to be a lane."
        )

    @pytest.mark.parametrize("symbol", DELETED_LANE_FUNCTIONS)
    def test_no_source_or_shipped_doc_mentions_it(self, symbol):
        mentions = []
        for path in _shipped_sources() + _shipped_docs():
            text = path.read_text(encoding="utf-8", errors="replace")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if symbol in line:
                    mentions.append(f"{path.relative_to(ROOT)}:{lineno}")

        assert not mentions, (
            f"`{symbol}` no longer exists, but is still named at {mentions}. "
            f"A comment or a doc that names a deleted function is a map to a "
            f"place that is not there."
        )


class TestThereIsOneGatherPath:
    """Guard 2. The fork this plan removed was two functions, not two
    branches, so the durable check is on the shape of the entry points."""

    def test_gather_context_is_defined_once(self):
        definitions = []
        for path in sorted(SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == "gather_context":
                        definitions.append(f"{path.relative_to(ROOT)}:{node.lineno}")

        assert len(definitions) == 1, (
            f"expected exactly one `gather_context`, found {definitions}"
        )
        assert definitions[0].startswith("src/neo/context_gatherer.py:"), (
            f"`gather_context` moved to {definitions[0]}; the front door lives "
            f"in context_gatherer.py"
        )

    def test_no_second_gather_entry_point_exists(self):
        """Any module-level `def gather_*` other than `gather_context` is a
        candidate second front door and has to be justified here first."""
        others = []
        for path in sorted(SRC.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("gather") and node.name != "gather_context":
                        others.append(f"{path.relative_to(ROOT)}:{node.name}")

        assert not others, (
            f"second gather entry point(s): {others}. The plan's standing "
            f"answer to 'which files does Neo see' is one pipeline; a sibling "
            f"of `gather_context` reopens the question."
        )

    def test_the_cli_calls_only_gather_context(self):
        tree = ast.parse((SRC / "cli.py").read_text(encoding="utf-8"))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id.startswith("gather")
        }

        assert called == {"gather_context"}, (
            f"cli.py calls {sorted(called)}; the fork it used to carry was "
            f"exactly this — one call site per retrieval strategy."
        )


class TestSemanticIsDataNotControlFlow:
    """Guard 3. `--semantic` must reach the pipeline as a value, never as a
    question `cli.py` answers on the pipeline's behalf."""

    def test_the_cli_never_branches_on_the_flag(self):
        tree = ast.parse((SRC / "cli.py").read_text(encoding="utf-8"))
        branches = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.If, ast.IfExp)):
                continue
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Attribute) and sub.attr == "semantic":
                    branches.append(getattr(node, "lineno", "?"))

        assert not branches, (
            f"cli.py branches on `.semantic` at line(s) {branches}. The flag is "
            f"a weight hint spent inside stage 4; a caller-side branch on it is "
            f"how the two-lane fork was spelled."
        )

    def test_the_flag_is_carried_on_the_config(self):
        from neo.context_gatherer import GatherConfig

        assert GatherConfig(root=".", prompt="x").semantic is False
        assert GatherConfig(root=".", prompt="x", semantic=True).semantic is True


class TestTheDocsDoNotTeachTheOldModel:
    """Guard 4. The claims below are false about this code and were all true
    of the code that preceded it, which is exactly why they survived the
    deletion that made them false."""

    @pytest.mark.parametrize("claim", RETIRED_CLAIMS)
    def test_nothing_shipped_makes_the_claim(self, claim):
        """Sources are scanned alongside the docs because two of these four
        claims were printed BY the program, not written about it: the
        first-run tip said "enable semantic file selection" and `--index`'s own
        help string read as a setup step. A doc-only scan would have passed on
        the exact text that taught the model."""
        needle = claim.lower()
        hits = []
        for path in _shipped_sources() + _shipped_docs():
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(),
                start=1,
            ):
                if needle in line.lower():
                    hits.append(f"{path.relative_to(ROOT)}:{lineno}")

        assert not hits, (
            f"{hits} still says {claim!r}. Since Goal 7 the walk and the keyword "
            f"index refresh themselves on every call and `--index` builds only "
            f"the stage-4 embedding catalog, so this describes a prerequisite "
            f"that does not exist."
        )

    def test_the_cli_help_calls_index_optional(self):
        """The flag's own help text is the doc most users actually read.

        Asserts the CONTRACT, not a stem. An earlier version tested
        `"ptional" in index_help`, which passes on "not optional", on
        "optionally required" and on any inflection of the word — it proves
        the seven letters are present, not that the sentence says what it
        needs to. It also selected its line with a bare `next()`, so a change
        of quoting style raised `StopIteration` instead of failing legibly.
        """
        help_text = (SRC / "cli.py").read_text(encoding="utf-8")
        candidates = [
            line for line in help_text.splitlines()
            if "'--index'" in line or '"--index"' in line
        ]

        assert candidates, "no `--index` argument declaration found in cli.py"
        index_help = candidates[0].lower()

        assert "optional cache-warmer" in index_help, (
            "`--index`'s help must name it a cache-warmer. It read 'Build "
            "semantic index for current directory', which is a setup step."
        )
        assert "works without it" in index_help, (
            "`--index`'s help must say selection works with no catalog. "
            "Calling it optional without saying what still works leaves the "
            "reader to assume the answer is 'less'."
        )
        assert "semantic index" not in index_help, (
            "the retired phrasing is back in `--index`'s help"
        )
