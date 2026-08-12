"""Tests for what the shared eligibility walk refuses to read.

Regression coverage for a prompt that came back holding the same file six
times. `should_ignore` did not understand a root-anchored `.gitignore`
pattern (`/path/`), which is the form people write for a top-level
directory, so three nested checkouts a repo deliberately ignored were walked
anyway — and the gatherer had no exclusion of its own for agent worktrees, so
each nested repo contributed its `.claude/worktrees/*` and `.codex/worktrees/*`
copies too. Twelve of sixteen selected files were duplicates of two.

The rules under test now live in `neo.eligibility` and are shared with the
project index and the architecture scan; `iter_paths` is the gatherer's thin
wrapper over that walk and is exercised here because it is the caller these
regressions were reported against.
"""

import os
import re

import pytest

from neo.context_gatherer import iter_paths
from neo.eligibility import (
    compile_glob,
    load_ignore_patterns,
    should_ignore,
    walk_paths,
)


class TestAnchoredPatterns:
    """A leading slash means "at the repo root", not "anywhere"."""

    @pytest.mark.parametrize("pattern", ["/path/", "/path"])
    def test_anchored_directory_is_ignored(self, pattern):
        """The live failure: `/path/` matched nothing, so `path/` was walked."""
        assert should_ignore("path", [pattern], is_dir=True)

    def test_files_under_an_anchored_directory_are_ignored(self):
        assert should_ignore("path/tests/Api.Tests/X.cs", ["/path/"])

    def test_anchor_does_not_match_at_depth(self):
        """That is the whole point of the leading slash.

        `/dist` means the top-level `dist`, not `src/dist`. Treating the
        anchor as decoration would over-exclude, which is the opposite
        failure and just as silent.
        """
        assert not should_ignore("src/dist", ["/dist"], is_dir=True)
        assert not should_ignore("nested/path/x.cs", ["/path/"])

    def test_unanchored_pattern_still_matches_at_depth(self):
        assert should_ignore("src/dist", ["dist"], is_dir=True)

    @pytest.mark.parametrize("pattern", ["/path/", "path/"])
    def test_subdirectory_of_an_ignored_directory_matches_directly(self, pattern):
        """A directory rule covers the directories beneath it, too.

        Neither current caller needs this — `iter_paths` prunes the parent
        before descending, and `project_index` walks ancestors — but a
        predicate that is only correct because its callers compensate is one
        refactor away from being wrong. That is the same reason the bare-name
        rule below tests every path component instead of the basename.
        """
        assert should_ignore("path/tests", [pattern], is_dir=True)
        assert should_ignore("path/tests/deep", [pattern], is_dir=True)

    def test_anchor_does_not_prefix_match(self):
        """`/path/` must not swallow `pathfinder`."""
        assert not should_ignore("pathfinder", ["/path/"], is_dir=True)
        assert not should_ignore("otherpath/x.cs", ["/path/"])

    def test_bare_slash_is_not_a_match_everything(self):
        """Degenerate pattern must not become match-everything."""
        assert not should_ignore("anything", ["/"])
        assert not should_ignore("a/b/c.py", ["/"])
        assert not should_ignore("a", ["//"], is_dir=True)

    def test_anchored_bare_name_covers_its_descendants(self):
        """`/dist` must reach `dist/foo.py`, and must not reach `distribution/`.

        This clause had zero coverage in either direction, so nothing would
        have noticed it being deleted OR being widened into a prefix match.
        """
        assert should_ignore("dist/foo.py", ["/dist"])
        assert should_ignore("dist/deep/foo.py", ["/dist"])
        assert not should_ignore("distribution/x.py", ["/dist"])
        assert not should_ignore("src/dist/foo.py", ["/dist"])

    def test_slash_bearing_pattern_is_anchored_and_covers_descendants(self):
        """An internal slash anchors to the root, per gitignore."""
        assert should_ignore("foo/bar/baz.py", ["foo/bar"])
        assert not should_ignore("x/foo/bar/baz.py", ["foo/bar"])


class TestComponentMatching:
    def test_bare_directory_name_matches_at_any_depth(self):
        """`node_modules` has to catch `a/node_modules/b.js`.

        It was tested against the basename alone — `b.js` — so it did not.
        The walk hid this by pruning the directory first, which means the
        predicate was only correct while its caller compensated.
        """
        assert should_ignore("a/node_modules/b.js", ["node_modules"])
        assert should_ignore("node_modules/b.js", ["node_modules"])

    def test_component_match_is_not_a_substring_match(self):
        """`build` must not swallow `rebuild`."""
        assert not should_ignore("src/rebuild/x.py", ["build"])

    def test_negation_re_includes_a_path(self):
        """`!` rescues, because gitignore is LAST-match-wins.

        Skipping negations made `!` unreachable by construction. A real repo
        pairs `.claude/*` with `!.claude/skills/`, and the skip hid seven
        git-tracked files that `git check-ignore` reports as NOT ignored.

        The pattern here excludes the CHILDREN of `dist`, not `dist` itself,
        which is what makes re-inclusion legal — see
        `test_cannot_re_include_below_an_excluded_directory` for the case
        where it is not.
        """
        assert not should_ignore("dist/keep.js", ["dist/*", "!dist/keep.js"])
        assert should_ignore("dist/other.js", ["dist/*", "!dist/keep.js"])

    def test_cannot_re_include_below_an_excluded_directory(self):
        """git: "It is not possible to re-include a file if a parent
        directory of that file is excluded."

        This predicate, asked about the file in isolation, says the negation
        wins — it has no ancestor state. The one caller supplies that state
        and so reproduces git: `eligibility.walk` prunes `dist` before
        descending, so nothing beneath it is ever asked about. (There used to
        be a second caller with its own ancestor handling; the unification is
        why there is now one.) The divergence is therefore not user-visible,
        but it is real, and an earlier version of this file asserted the
        divergence as if it were the correct answer. Pinned here as a known
        limit rather than defended as behaviour.
        """
        # What git says: ignored, because the parent directory is excluded.
        # What this predicate says, without ancestors:
        assert not should_ignore("dist/keep.js", ["dist", "!dist/keep.js"])
        # ...and what the CALLERS say, which is what actually matters:
        assert should_ignore("dist", ["dist", "!dist/keep.js"], is_dir=True)

    def test_order_decides_when_patterns_conflict(self):
        """Last match wins in BOTH directions, not just re-inclusion."""
        assert not should_ignore("a/x.py", ["a", "!a"])
        assert should_ignore("a/x.py", ["!a", "a"])

    def test_the_real_repo_pairing_is_honored(self):
        """`.claude/*` + `!.claude/skills/` — verified against git."""
        pats = [".claude/*", "!.claude/skills/"]
        assert not should_ignore(".claude/skills/deploy/run.py", pats)
        assert should_ignore(".claude/settings/local.json", pats)


class TestDefaultExclusions:
    """The gatherer had no exclusion for worktrees or agent scratch space."""

    @pytest.mark.parametrize(
        "name",
        ['.worktrees', 'obj', 'node_modules', '__pycache__', '.venv',
         '.next', '.pytest_cache', '.idea'],
    )
    def test_name_is_excluded_by_default(self, name, tmp_path):
        """Applies even when the repo's .gitignore says nothing.

        These are routinely untracked-but-not-ignored, and a worktree is a
        second copy of the tree — it does not add noise, it competes with the
        originals for the same slots.
        """
        patterns = load_ignore_patterns(str(tmp_path))
        assert should_ignore(name, patterns, is_dir=True), (
            f"{name!r} is not excluded by default"
        )


class TestAgentDirectoriesAreNotExcludedWholesale:
    """Exclude the copies, not the directory that happens to contain them.

    Two shortcuts were tried and both hid committed source, which is why the
    rule is written as a LAYOUT (`*.claude/worktrees`) rather than either
    bare component:

    - `.claude`/`.codex`/`.car` outright hid 60 git-tracked skill
      implementations across two local repos.
    - bare `worktrees` hid `src/worktrees/manager.ts` — a worktree MANAGER
      keeps its source in a directory named for what it manages.

    Same shape both times, and the same as excluding `bin/` and losing the
    Rust binary crates inside it: excluding a container for what sometimes
    sits inside it.
    """

    @pytest.mark.parametrize("path", [
        "src/worktrees/manager.ts",
        "packages/core/worktrees/index.ts",
    ])
    def test_a_first_party_worktrees_directory_is_kept(self, path, tmp_path):
        """Bare `worktrees` as a component would hide these.

        `forge/src/worktrees/manager.ts` is git-tracked TypeScript in a repo
        that manages worktrees for a living.
        """
        patterns = load_ignore_patterns(str(tmp_path))
        assert not should_ignore(path, patterns), (
            f"{path} is first-party source, not an agent worktree copy"
        )

    @pytest.mark.parametrize("path", [
        ".claude/skills/deploy-app/scripts/deploy_verify.py",
        ".claude/skills/create-car-agent/scripts/car_register.py",
        ".codex/skills/x/run.py",
        ".car/agents/thing.py",
    ])
    def test_tracked_source_under_an_agent_dir_is_kept(self, path, tmp_path):
        patterns = load_ignore_patterns(str(tmp_path))
        assert not should_ignore(path, patterns), (
            f"{path} is real committed source, not a worktree copy"
        )

    @pytest.mark.parametrize("path", [
        "m365dotnet/.claude/worktrees/local-run/tests/X.cs",
        "m365dotnet/.codex/worktrees/issue-123/docs/A.md",
        ".worktrees/PAR-1/adws/poll.py",
        ".claude/worktrees/a/src/real.py",
    ])
    def test_worktree_copies_are_still_excluded(self, path, tmp_path):
        """The `worktrees` component alone is what does the work."""
        patterns = load_ignore_patterns(str(tmp_path))
        assert should_ignore(path, patterns)


class TestMalformedPatterns:
    """A `.gitignore` line that will not compile must not take the walk with it.

    `should_ignore` replaced `fnmatch` with a real regex compile, and that
    swap introduced a failure mode `fnmatch` did not have: `fnmatch` degrades
    to a non-match on anything it cannot parse, while `re.compile` raises.
    Since this predicate runs on every directory and every file of every walk,
    one unparseable line would turn any neo invocation into a traceback.

    The expected values below are what `git check-ignore` returns for the same
    pattern -- these are not guesses about what is reasonable. `[]]` and
    `[a-]` in particular are well-formed POSIX classes that a naive
    scan-to-first-`]` mangles into unbalanced regex.
    """

    @pytest.mark.parametrize("pattern,matches_bracket,matches_path", [
        ('[]]', True, False),             # a class containing `]`
        ('[a-]', False, True),            # trailing `-` is a literal member
        ('[]', False, False),             # empty, unterminated
        ('[z-a]', False, False),          # inverted range
        ('[', False, False),              # bare
        ('[!', False, False),             # negation, unterminated
        ('a[b', False, False),            # unterminated mid-pattern
        ('*[', False, False),
        ('[[]', False, False),            # class containing `[`
        ('***', True, True),
        ('!', False, False),              # negation of nothing
        ('!!x', False, False),
        ('//', False, False),
        ('\\', False, False),
        ('**/', False, True),
    ])
    def test_matches_git_and_never_raises(self, pattern, matches_bracket, matches_path):
        assert should_ignore("]", [pattern]) is matches_bracket
        assert should_ignore("a/b.py", [pattern]) is matches_path

    def test_a_malformed_pattern_does_not_stop_the_walk(self, tmp_path):
        """The blast radius, not just the predicate: one bad line used to
        raise out of `iter_paths` and abort file gathering entirely."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src/real.py").write_text("def real(): pass\n")
        (tmp_path / ".gitignore").write_text("[z-a]\n*[\nbuild/\n")

        found = [rel for _, rel, _ in iter_paths(str(tmp_path), [], [], ["py"])]
        assert "src/real.py" in found

    def test_compiling_a_pattern_emits_no_warning(self, recwarn):
        """`[[]` compiles to a regex Python flags as a possible nested set.

        It is benign today, but the warning printed on every invocation for
        any repo whose `.gitignore` contains that line -- and Python has said
        the construct may become an error.
        """
        # Both caches, or this guard cannot fail. `re` keeps a module-level
        # compile cache that `lru_cache` knows nothing about, and the
        # parametrized test above already compiles `[[]` — so clearing only
        # neo's cache lets `re.compile` return the memoized object without
        # re-parsing, emitting no warning no matter how broken the escaping is.
        # Measured: with `re.purge()` removed, reverting the `[` escape leaves
        # this test PASSING.
        compile_glob.cache_clear()
        re.purge()
        for pattern in ('[[]', '[]]', '[a-]', '[!a-z]'):
            should_ignore("x", [pattern])
        assert [w for w in recwarn.list if issubclass(w.category, (FutureWarning, DeprecationWarning))] == []


class TestWalk:
    def _repo(self, tmp_path, gitignore=None):
        (tmp_path / "src").mkdir()
        (tmp_path / "src/real.py").write_text("def real(): pass\n")
        for wt in (".claude/worktrees/a", ".codex/worktrees/b"):
            d = tmp_path / wt / "src"
            d.mkdir(parents=True)
            (d / "real.py").write_text("def real(): pass\n")
        nested = tmp_path / "nested/src"
        nested.mkdir(parents=True)
        (nested / "other.py").write_text("def other(): pass\n")
        if gitignore is not None:
            (tmp_path / ".gitignore").write_text(gitignore)
        return tmp_path

    def test_agent_worktrees_are_not_walked(self, tmp_path):
        """The measured failure: one file, six times, one per worktree."""
        root = self._repo(tmp_path)
        found = [rel for _, rel, _ in iter_paths(str(root), [], [], ["py"])]

        assert "src/real.py" in found
        assert not [p for p in found if "worktrees" in p], found

    def test_anchored_gitignore_entry_is_honored_by_the_walk(self, tmp_path):
        root = self._repo(tmp_path, gitignore="/nested/\n")
        found = [rel for _, rel, _ in iter_paths(str(root), [], [], ["py"])]

        assert "src/real.py" in found
        assert not [p for p in found if p.startswith("nested/")], found

    def test_the_same_file_is_not_selected_once_per_worktree(self, tmp_path):
        """The user-visible symptom, asserted on identity rather than count."""
        root = self._repo(tmp_path)
        found = [rel for _, rel, _ in iter_paths(str(root), [], [], ["py"])]

        basenames = [os.path.basename(p) for p in found]
        assert basenames.count("real.py") == 1, found

    def test_ordinary_source_is_still_walked(self, tmp_path):
        """The exclusions must not be so broad that nothing is gathered."""
        root = self._repo(tmp_path)
        found = [rel for _, rel, _ in iter_paths(str(root), [], [], ["py"])]

        assert "src/real.py" in found
        assert "nested/src/other.py" in found


class TestExtensionFilterNarrows:
    """`exts` is a NARROWING argument and must never widen.

    Unification moved the extension check behind `normalize_exts`, and the
    first cut of that helper returned `None` — the sentinel for "no filter" —
    whenever the normalized set came out empty. That inverts the argument:
    pre-unification, `exts=[]` matched nothing (`ext not in []` is always
    true); after, it matched the entire repository. The same collapse dropped
    the empty string, which is a real selector because `""` is the extension
    of `Makefile`.

    Only `None` may mean "no filter". Every other input yields a set.
    """

    def _repo(self, tmp_path):
        (tmp_path / "a.py").write_text("x\n")
        (tmp_path / "b.md").write_text("x\n")
        (tmp_path / "Makefile").write_text("x\n")
        return tmp_path

    def _walk(self, root, exts):
        return sorted(
            entry.rel_path for entry in walk_paths(str(root), exts=exts).paths
        )

    def test_none_admits_every_extension(self, tmp_path):
        root = self._repo(tmp_path)
        assert self._walk(root, None) == ["Makefile", "a.py", "b.md"]

    def test_an_empty_list_admits_nothing(self, tmp_path):
        """The inversion, pinned. This is the regression, not a style point."""
        root = self._repo(tmp_path)
        assert self._walk(root, []) == []

    def test_the_empty_string_selects_extensionless_files(self, tmp_path):
        root = self._repo(tmp_path)
        assert self._walk(root, [""]) == ["Makefile"]

    def test_a_trailing_comma_does_not_silently_drop_its_selector(self, tmp_path):
        """`--exts py,` splits to `["py", ""]`; both members must survive."""
        root = self._repo(tmp_path)
        assert self._walk(root, ["py", ""]) == ["Makefile", "a.py"]

    def test_both_spellings_and_both_cases_are_accepted(self, tmp_path):
        """A disclosed widening: `.py`, `py` and `PY` now agree.

        Pre-unification `--exts .py` matched NOTHING (the leading dot was
        compared against a stripped extension) and the comparison was
        case-sensitive. Both are stated in the PR body rather than smuggled.
        """
        root = self._repo(tmp_path)
        (tmp_path / "c.PY").write_text("x\n")
        for spelling in ("py", ".py", "PY"):
            assert self._walk(root, [spelling]) == ["a.py", "c.PY"], spelling


class TestAgentDocsStillReachable:
    def test_agent_instructions_reach_the_prompt_independently(self, tmp_path):
        """`agent_context.discover` never consults `should_ignore`.

        Pinned because that independence was once used to justify excluding
        `.claude` wholesale. The justification was too narrow — `discover`
        handles MARKDOWN only, so skill *source* under those directories had
        no second route — but the independence itself is real and worth
        keeping true.
        """
        from neo.agent_context import discover

        (tmp_path / "CLAUDE.md").write_text("# project rules\n")
        (tmp_path / ".claude").mkdir()
        (tmp_path / ".claude/commands").mkdir()
        (tmp_path / ".claude/commands/ship.md").write_text("# ship\n")

        docs = {d.path for d in discover(str(tmp_path))}
        assert "CLAUDE.md" in docs
        assert any(p.startswith(".claude/") for p in docs), docs
