"""G4-inv: the shared walk, differentially compared against git itself.

`neo.eligibility` reimplements gitignore rather than shelling out to git, so
the only honest proof that it is right is to ask git the same question and
diff the answers. This module does that against a fixture corpus of real
git repositories built per-test, plus — when the suite runs inside one — the
neo checkout itself.

**The contract, stated exactly.** Over-exclusion is the failure this guards:
a file git tracks that the walk refuses to yield. It is the dangerous
direction, because a file that never reaches the walk never reaches the
prompt, and the model then answers a question about code it was not shown —
which is how #186 presented (a confident claim about a null check in a file
whose body was never delivered). On the synthetic corpora the assertion is
therefore absolute — **zero tracked files skipped** — and on the neo checkout
it takes the strong conditional form stated in the third bullet below.

Three exclusion classes are NOT the repo's gitignore and are stated here
rather than hidden, because a test that quietly folded them into "gitignore
says so" would be asserting something false:

- **neo's default patterns** (`DEFAULT_IGNORE_PATTERNS`) exclude derived
  artifacts that repos routinely leave untracked-but-not-ignored —
  `node_modules`, `__pycache__`, `.worktrees`. A repo that *tracks* a file
  under one of those names loses it, deliberately, and
  `test_defaults_exclude_a_tracked_artifact_directory_on_purpose` pins that
  asymmetry so it stays visible. The corpora below track no such file,
  because a repo that checks in `node_modules/` is not representative.
- **Policy knobs** — symlink rejection, the per-file size ceiling, extension
  filters — are `WalkPolicy` settings, not ignore rules. The differential
  runs with all of them off so the comparison is against the ignore layer and
  nothing else.
- **git's tracked-file override.** git applies ignore rules only to files it
  does not already track, so a file added before a rule was written stays
  tracked and `git check-ignore` calls it not-ignored. The walk evaluates the
  rules against the path and has no index to consult. On this checkout that
  is four `specs/*.md`. `test_tracked_but_ignored_files_are_a_known_divergence_from_git`
  pins it, and the neo-checkout assertion is therefore the strong form —
  *no tracked file is skipped unless an ignore rule independently says so* —
  which still catches every mis-parsed pattern and every over-broad default.

**Known accepted limit: nested `.gitignore` files are not read** — only the
repository root's. That produces UNDER-exclusion (the walk yields a file git
ignores), never over-exclusion, and it is the only source of under-exclusion
measured against `git check-ignore` over 7,534 on-disk paths in 33
repositories. `test_nested_gitignore_is_the_documented_under_exclusion`
pins the limit as a live fact rather than a comment, so that closing it later
is a visible change rather than a silent one.
"""

import os
import subprocess

import pytest

from neo.eligibility import (
    DEFAULT_IGNORE_PATTERNS,
    WalkPolicy,
    should_ignore,
    walk,
)

pytestmark = pytest.mark.invariants


def _git(root, *args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=check,
    )


def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


requires_git = pytest.mark.skipif(
    not _git_available(), reason="git is not installed"
)


def _init_repo(root):
    """A repo with identity set locally, so CI's empty global config works."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "eligibility@test.invalid")
    _git(root, "config", "user.name", "Eligibility Differential")
    return root


def _write(root, rel, content="x\n"):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def _tracked(root) -> set:
    """Every path git tracks, repo-relative and POSIX-separated.

    `-z` because a path may contain a newline, and `git ls-files` quotes such
    a path in its default output — parsing that back is a second
    implementation of an escape format, which is the class of thing this
    module exists to avoid.
    """
    out = _git(root, "ls-files", "-z").stdout
    return {entry for entry in out.split("\0") if entry}


def _walked(root, **policy_kwargs) -> set:
    """Every rel_path the shared walk yields, ignore layer only."""
    result = walk(str(root), WalkPolicy(**policy_kwargs))
    return {entry.rel_path for entry in result.paths}


def _check_ignore_indexed(root, paths) -> set:
    """The subset of `paths` git ignores *in practice*, index consulted.

    Without `--no-index`, git reports a tracked file as not-ignored no matter
    what the rules say, because ignore rules only govern what git will start
    tracking. That difference is the whole content of the tracked-but-ignored
    divergence recorded below.
    """
    if not paths:
        return set()
    proc = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=str(root),
        input="\n".join(sorted(paths)),
        capture_output=True,
        text=True,
    )
    assert proc.returncode in (0, 1), proc.stderr
    return {line for line in proc.stdout.splitlines() if line}


def _check_ignore(root, paths) -> set:
    """The subset of `paths` git reports as ignored.

    `--no-index` so the answer is the ignore rules alone; a tracked file is
    otherwise reported as not-ignored regardless of what the rules say, which
    would make the under-exclusion comparison vacuous.
    """
    if not paths:
        return set()
    proc = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=str(root),
        input="\n".join(sorted(paths)),
        capture_output=True,
        text=True,
    )
    # Exit 0 = some paths ignored, 1 = none, 128 = error.
    assert proc.returncode in (0, 1), proc.stderr
    return {line for line in proc.stdout.splitlines() if line}


# ---------------------------------------------------------------------------
# The fixture corpus
# ---------------------------------------------------------------------------

#: Each entry is (repo name, .gitignore body, {rel_path: should_git_track}).
#: The ignore bodies between them cover every pattern shape `should_ignore`
#: implements: root-anchored directory and file rules, unanchored directory
#: rules matching at depth, negation with last-match-wins, a `*` that must not
#: cross a separator, `**` spanning components, a character class, and a
#: suffix glob.
CORPORA = {
    # `artifacts` rather than `dist` on purpose: `dist` is in neo's DEFAULT
    # list, so `src/dist/keep.py` would be excluded by the defaults and this
    # corpus would be testing the default list instead of the anchoring rule
    # it is named for. That asymmetry has its own test below.
    "anchored": (
        """
        /artifacts/
        /secret.txt
        """,
        {
            "src/app.py": True,
            "src/artifacts/keep.py": True,   # `/artifacts/` is root-only
            "artifactory/x.py": True,        # prefix, not the component
            "artifacts/bundle.js": False,
            "artifacts/deep/nested/bundle.js": False,
            "secret.txt": False,
            "docs/secret.txt": True,         # anchored, so root-only
        },
    ),
    "unanchored-dir": (
        """
        codegen/
        *.log
        """,
        {
            "src/app.ts": True,
            "codegen/api.ts": False,
            "pkg/codegen/deep/api.ts": False,   # matches at any depth
            "codegen.ts": True,                 # a FILE of that name survives
            "logs/app.log": False,
            "src/app.log": False,
        },
    ),
    "negation": (
        """
        config/*
        !config/keep.yaml
        """,
        {
            "config/keep.yaml": True,
            "config/drop.yaml": False,
            "src/main.go": True,
        },
    ),
    "star-does-not-cross-a-separator": (
        """
        /*.png
        docs/[!_]*.md
        """,
        {
            "banner.png": False,
            "docs/img/diagram.png": True,   # `/*.png` is root-only, one component
            "docs/guide.md": False,
            "docs/_partial.md": True,       # the class excludes a leading `_`
            "src/lib.rs": True,
        },
    ),
    "globstar": (
        """
        **/generated/**
        build-tools/**/*.tmp
        """,
        {
            "src/generated/client.cs": False,
            "generated/client.cs": False,
            "src/Real.cs": True,
            "build-tools/a/b/scratch.tmp": False,
            "build-tools/a/b/keep.cs": True,
        },
    ),
}


def _build_corpus(root, ignore_body, files):
    _init_repo(root)
    (root / ".gitignore").write_text(
        "\n".join(line.strip() for line in ignore_body.strip().splitlines()) + "\n"
    )
    for rel, _ in files.items():
        _write(root, rel, f"// {rel}\n")
    # `git add -A` adds exactly what the ignore rules permit, so git — not
    # this test's expectations — decides what "tracked" means.
    _git(root, "add", "-A")
    return root


@requires_git
@pytest.mark.parametrize("name", sorted(CORPORA))
class TestDifferentialAgainstGit:
    def test_the_corpus_is_the_corpus_we_think_it_is(self, tmp_path, name):
        """git's own verdicts must match the table, or the diff below is moot.

        Without this, a `.gitignore` line that silently stopped matching
        would make the over-exclusion assertion pass by asking git a weaker
        question, and the suite would go green on a corpus that tests
        nothing.
        """
        body, files = CORPORA[name]
        root = _build_corpus(tmp_path / name, body, files)

        expected = {rel for rel, tracked in files.items() if tracked}
        assert _tracked(root) - {".gitignore"} == expected

    def test_zero_over_exclusion(self, tmp_path, name):
        """G4-inv. No file git tracks may be missing from the walk."""
        body, files = CORPORA[name]
        root = _build_corpus(tmp_path / name, body, files)

        tracked = _tracked(root)
        walked = _walked(root, skip_symlinks=False)
        over_excluded = tracked - walked

        assert over_excluded == set(), (
            f"{len(over_excluded)} tracked file(s) the walk refuses to yield: "
            f"{sorted(over_excluded)}"
        )

    def test_no_ignored_file_is_admitted(self, tmp_path, name):
        """The other direction, G1-inv: nothing git ignores may be selected.

        Under-exclusion is survivable where over-exclusion is not — a junk
        file costs a slot, a missing file costs the answer — but on a corpus
        with no nested `.gitignore` there is no excuse for any, so it is
        asserted exactly here and relaxed only in the nested-file test below.
        """
        body, files = CORPORA[name]
        root = _build_corpus(tmp_path / name, body, files)

        walked = _walked(root, skip_symlinks=False)
        assert _check_ignore(root, walked) == set()


@requires_git
class TestDocumentedLimits:
    """The exclusions that are NOT gitignore, stated as live facts."""

    def test_nested_gitignore_is_the_documented_under_exclusion(self, tmp_path):
        """Only the root `.gitignore` is read. This is the accepted limit.

        It fails in the safe direction: the walk yields a file git ignores,
        so the file costs a context slot rather than disappearing. Closing it
        means reading ignore files per directory during the walk; until then
        this test is what stops the limit from being rediscovered as a bug.
        """
        root = _init_repo(tmp_path / "nested")
        (root / ".gitignore").write_text("*.log\n")
        _write(root, "src/.gitignore", "vendored/\n")
        _write(root, "src/app.py")
        _write(root, "src/vendored/lib.py")
        _git(root, "add", "-A")

        walked = _walked(root, skip_symlinks=False)

        assert "src/vendored/lib.py" in walked, (
            "the limit is under-exclusion; if this now passes the walk must "
            "have learned nested ignore files — update the docs, not this test"
        )
        assert _check_ignore(root, walked) == {"src/vendored/lib.py"}, (
            "a nested .gitignore is the ONLY accepted source of "
            "under-exclusion; something else is leaking through"
        )

    def test_defaults_exclude_a_tracked_artifact_directory_on_purpose(
        self, tmp_path
    ):
        """neo's defaults are stricter than git, and that is deliberate.

        A repo that tracks `node_modules/` loses it. The defaults exist
        because the worst offenders are routinely
        untracked-but-not-ignored — a nested checkout or an agent worktree is
        a second COPY of a tree that competes with the originals for the same
        context slots — and no `.gitignore` in the repo says so. Pinned here
        so the asymmetry is a decision on the record rather than a surprise
        inside a green differential.
        """
        root = _init_repo(tmp_path / "vendored")
        _write(root, "app.js")
        _write(root, "node_modules/dep/index.js")
        _git(root, "add", "-A", "-f")

        assert "node_modules/dep/index.js" in _tracked(root)
        assert "node_modules/dep/index.js" not in _walked(root)
        assert "node_modules" in DEFAULT_IGNORE_PATTERNS

    def test_symlink_rejection_is_policy_not_an_ignore_rule(self, tmp_path):
        """A tracked symlink is skipped by POLICY, and it is switchable.

        Stated separately so the differential's "zero over-exclusion" claim
        is about the ignore layer and cannot be read as a claim about
        symlinks. Rejecting them is what keeps a link from pulling content
        from outside the repository into a prompt.
        """
        root = _init_repo(tmp_path / "links")
        _write(root, "real.py")
        os.symlink(root / "real.py", root / "link.py")
        _git(root, "add", "-A")

        assert "link.py" in _tracked(root)
        assert "link.py" not in _walked(root)
        assert "link.py" in _walked(root, skip_symlinks=False)


@requires_git
class TestAgainstThisRepository:
    """The corpus that was not written by the author of the walker.

    Synthetic fixtures test the shapes their author thought of. The neo
    checkout is a real repository with a real `.gitignore` and ~300 tracked
    files, and it is present in CI for free — so it costs one walk to find
    out whether the walk hides any of neo's own source from neo.
    """

    @staticmethod
    def _repo_root():
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if proc.returncode != 0:
            pytest.skip("not running inside a git work tree")
        return proc.stdout.strip()

    def test_no_tracked_file_is_skipped_without_an_ignore_rule_saying_so(self):
        """The strong form of G4-inv on a repository nobody wrote for it.

        A tracked file the walk skips is acceptable only when SOME rule
        independently accounts for it, and there are exactly two rule sets:
        the repo's own `.gitignore` (asked of git itself) and neo's shared
        defaults. Any OTHER skip is a defect in the matcher, and that is what
        this catches.

        The defaults are subtracted rather than folded into "git says so",
        because they are not git's rules and asserting otherwise would be
        false. They are also the reason this test must not simply diff
        against `git check-ignore`:
        `test_defaults_exclude_a_tracked_artifact_directory_on_purpose`
        deliberately blesses a tracked file under a default-only name, so a
        version of this assertion that knew only about `.gitignore` would
        contradict it — green today only because this checkout happens to
        track no `.vscode/settings.json`, `build/`, `dist/` or `obj/` file.
        The day one is added, the naive form fails with a message accusing
        the matcher of a defect it does not have, and the fix would look like
        weakening the guard. Attributing each skip to the rule set that
        caused it keeps the guard sharp AND keeps the two tests consistent.
        """
        root = self._repo_root()
        tracked = _tracked(root)
        if not tracked:
            pytest.skip("git work tree reports no tracked files")

        skipped = tracked - _walked(root, skip_symlinks=False)
        by_repo_rules = _check_ignore(root, skipped)
        by_neo_defaults = {
            path
            for path in skipped
            if should_ignore(path, list(DEFAULT_IGNORE_PATTERNS))
        }
        unexplained = skipped - by_repo_rules - by_neo_defaults

        assert unexplained == set(), (
            f"{len(unexplained)} of {len(tracked)} tracked files are invisible "
            f"to the walk and neither the repo's ignore rules nor neo's "
            f"defaults account for them: {sorted(unexplained)[:20]}"
        )

    def test_tracked_but_ignored_files_are_a_known_divergence_from_git(self):
        """git's ignore rules do not apply to files git already tracks.

        `.gitignore` governs what git will START tracking. A file added
        before a rule was written stays tracked forever, and `git
        check-ignore` without `--no-index` reports it as NOT ignored. The
        walk has no such override: it evaluates the rules against the path
        and skips it.

        Live on this checkout, that is the four `specs/*.md` files added
        before `.gitignore` gained `specs/`. Fixing it means asking git for
        the tracked set on every walk — a subprocess on the warm path, and a
        selection change rather than a unification — so it is recorded here
        rather than smuggled into a refactor. The named-path guarantee that
        would make it matter is Goal 8's.

        This test asserts the SHAPE, not the count: it is a statement that
        every divergence found is of this one kind, and it stays true if the
        repo gains or loses such files.

        Scoped to the skips the REPO's rules cause, for the same reason the
        test above subtracts them: a tracked file under a default-only name
        is skipped by neo, not by `.gitignore`, so `git check-ignore` is
        right to call it not-ignored and it is not an instance of this
        divergence. Folding it in here would make this test fail for the one
        behaviour the fixture suite explicitly blesses.
        """
        root = self._repo_root()
        tracked = _tracked(root)
        if not tracked:
            pytest.skip("git work tree reports no tracked files")

        skipped = tracked - _walked(root, skip_symlinks=False)
        # Every skipped-but-tracked path is accounted for by the repo's own
        # rules or by neo's defaults; the defaults are a separate contract.
        by_repo_rules = _check_ignore(root, skipped)
        by_neo_defaults = {
            path
            for path in skipped
            if should_ignore(path, list(DEFAULT_IGNORE_PATTERNS))
        }
        assert skipped == by_repo_rules | by_neo_defaults
        # ...and git, consulting its index, disagrees with the rules for
        # exactly the rule-caused paths — which is what makes them the
        # divergence this test is named for.
        if by_repo_rules:
            not_ignored_because_tracked = by_repo_rules - _check_ignore_indexed(
                root, by_repo_rules
            )
            assert not_ignored_because_tracked == by_repo_rules
