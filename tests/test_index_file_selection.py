"""Tests for which files `ProjectIndex.build_index` chooses to index.

Regression coverage for the failure where `neo --index` on a .NET repo
produced an index containing zero C#: file patterns were globbed in list
order and concatenated, then sliced to `max_files`, so whichever language
appeared first in `file_patterns` exhausted the budget. A repo of 4,272 C#
files indexed 83 Python files — 95 of them from a git worktree checked out
inside the repo — and exited 0 reporting success.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from neo.cli import _format_selection_report
from neo.index.project_index import CodeChunk, ProjectIndex


DEFAULT_PATTERNS = ["**/*.py", "**/*.cs", "**/*.ts", "**/*.tsx", "**/*.js"]


def _write(root: Path, rel: str, content: str = "") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    # Default to unique content so files are not treated as duplicates.
    path.write_text(content or f"// {rel}\n")
    return path


@pytest.fixture
def dotnet_repo(tmp_path):
    """A repo shaped like the one in the report: mostly C#, some TS, a
    little Python, plus a worktree full of Python that used to win.
    """
    for i in range(300):
        _write(tmp_path, f"src/Services/Service{i}.cs", f"public class S{i} {{}}\n")
    for i in range(20):
        _write(tmp_path, f"web/comp{i}.ts", f"export class C{i} {{}}\n")
    for i in range(5):
        _write(tmp_path, f"adws/tool{i}.py", f"def tool{i}(): return {i}\n")
    for i in range(200):
        _write(tmp_path, f".worktrees/PAR-1/adws/poll{i}.py", f"def poll{i}(): pass\n")
    return tmp_path


class TestLanguageAllocation:
    """`_allocate_slots` divides the budget by composition, with a floor."""

    def test_dominant_language_gets_most_slots(self):
        alloc = ProjectIndex._allocate_slots(
            {'c_sharp': 4272, 'typescript': 54, 'python': 18}, 100
        )
        assert alloc['c_sharp'] > 90
        assert sum(alloc.values()) == 100

    def test_no_present_language_is_eliminated(self):
        """The floor of one is what "cannot eliminate a language" means.

        Pure proportional allocation rounds 18 files out of 4,344 to zero
        slots. That is a smaller version of the bug being fixed.
        """
        alloc = ProjectIndex._allocate_slots(
            {'c_sharp': 4272, 'typescript': 54, 'python': 18}, 100
        )
        assert alloc['python'] >= 1
        assert alloc['typescript'] >= 1

    def test_allocation_never_exceeds_available_files(self):
        """A language cannot be given more slots than it has files."""
        alloc = ProjectIndex._allocate_slots({'python': 1, 'c_sharp': 5}, 100)
        assert alloc == {'python': 1, 'c_sharp': 5}

    @pytest.mark.parametrize(
        "counts,budget",
        [
            ({'python': 2, 'c_sharp': 500}, 100),
            ({'c_sharp': 4272, 'typescript': 54, 'python': 18}, 100),
            ({'a': 1, 'b': 1, 'c': 500}, 100),
            ({'go': 7, 'rust': 7}, 5),
            ({'java': 1000}, 1),
        ],
    )
    def test_no_slot_is_left_on_the_table(self, counts, budget):
        """Whenever enough files exist, the whole budget is allocated.

        Rounding down per language and capping at each language's file count
        both shed slots; the redistribution pass is what hands them back
        rather than silently indexing fewer files than the operator allowed.
        """
        alloc = ProjectIndex._allocate_slots(counts, budget)

        expected = min(budget, sum(counts.values()))
        assert sum(alloc.values()) == expected
        for lang, slots in alloc.items():
            assert slots <= counts[lang]

    def test_budget_smaller_than_language_count(self):
        """With fewer slots than languages, the biggest languages take them."""
        alloc = ProjectIndex._allocate_slots(
            {'a': 100, 'b': 50, 'c': 10, 'd': 5}, 2
        )
        assert alloc == {'a': 1, 'b': 1}

    def test_empty_and_zero_budget(self):
        assert ProjectIndex._allocate_slots({}, 100) == {}
        assert ProjectIndex._allocate_slots({'python': 5}, 0) == {}


class TestSelectionAcrossLanguages:
    def test_dominant_language_is_indexed(self, dotnet_repo):
        """The headline regression: C# must not be absent from a .NET repo.

        Pre-fix this selection was 100% Python because `**/*.py` is first in
        the pattern list and the worktree supplied enough files to exhaust
        the cap before `**/*.cs` was ever globbed.
        """
        index = ProjectIndex(str(dotnet_repo))
        selected, report = index._select_files(DEFAULT_PATTERNS, 100)

        suffixes = [p.suffix for p, _ in selected]
        assert suffixes.count('.cs') > 80, (
            f"C# is {suffixes.count('.cs')}/100 of the index of a C# repo"
        )
        assert report['per_language']['c_sharp']['selected'] > 80

    def test_no_language_present_is_dropped_entirely(self, dotnet_repo):
        index = ProjectIndex(str(dotnet_repo))
        selected, _ = index._select_files(DEFAULT_PATTERNS, 100)

        suffixes = {p.suffix for p, _ in selected}
        assert {'.cs', '.ts', '.py'} <= suffixes

    def test_pattern_order_does_not_change_composition(self, dotnet_repo):
        """Selection is a ranking, so reordering the patterns is a no-op.

        This is the property the old code lacked: its output was entirely
        determined by which pattern happened to be listed first.
        """
        index = ProjectIndex(str(dotnet_repo))
        forward, _ = index._select_files(DEFAULT_PATTERNS, 100)
        reverse, _ = index._select_files(list(reversed(DEFAULT_PATTERNS)), 100)

        assert sorted(str(p) for p, _ in forward) == sorted(
            str(p) for p, _ in reverse
        )


class TestExclusions:
    def test_worktree_files_are_excluded(self, dotnet_repo):
        """The subtree is pruned, and the report counts the subtree.

        This used to read `excluded == 200` — one per file under
        `.worktrees/PAR-1/`. That number was only available because the old
        selection globbed the whole repository first and filtered afterwards,
        which meant walking every worktree copy in order to count the files it
        was about to throw away. The shared walk prunes at `.worktrees` and
        never descends, so it does not know how many files are inside and does
        not claim to. One directory refused is what actually happened.
        """
        index = ProjectIndex(str(dotnet_repo))
        selected, report = index._select_files(DEFAULT_PATTERNS, 100)

        assert not any('.worktrees' in str(p) for p, _ in selected)
        assert report['excluded_dirs'] == 1
        assert report['excluded_files'] == 0
        assert report['excluded'] == 1

    @pytest.mark.parametrize(
        "excluded_dir",
        # Covered by the shared `load_gitignore_patterns` defaults: not
        # plausible source directory names, and routinely
        # untracked-but-not-gitignored.
        ['node_modules', '.git', 'obj', '__pycache__', '.venv',
         '.next', '.worktrees', 'Pods',
         # Excluded via the shared context_gatherer defaults, so the index
         # and prompt assembly agree on what counts as source.
         'dist', 'build', 'venv'],
    )
    def test_excluded_directories(self, tmp_path, excluded_dir):
        _write(tmp_path, "real.py", "def real(): pass\n")
        _write(tmp_path, f"{excluded_dir}/junk.py", "def junk(): pass\n")
        _write(tmp_path, f"nested/{excluded_dir}/deep.py", "def deep(): pass\n")

        index = ProjectIndex(str(tmp_path))
        selected, _ = index._select_files(["**/*.py"], 100)

        names = {p.name for p, _ in selected}
        assert names == {'real.py'}

    @pytest.mark.parametrize("ambiguous_dir", ['bin', 'out', 'target', 'vendor'])
    def test_ambiguous_directory_names_are_indexed_unless_gitignored(
        self, tmp_path, ambiguous_dir
    ):
        """These names are real source somewhere, so they are not blanket-excluded.

        `bin/` holds checked-in helper scripts, `target` and `out` are ordinary
        package names outside Cargo, `vendor` is source a user may ask about.
        A repo that generates into them gitignores them, and that is the layer
        that should decide — excluding a real source directory hides code
        permanently, while indexing generated code only spends slots.
        """
        _write(tmp_path, "real.py", "def real(): pass\n")
        _write(tmp_path, f"{ambiguous_dir}/script.py", "def script(): pass\n")

        index = ProjectIndex(str(tmp_path))
        selected, _ = index._select_files(["**/*.py"], 100)

        assert {p.name for p, _ in selected} == {'real.py', 'script.py'}

    @pytest.mark.parametrize("ambiguous_dir", ['bin', 'out', 'target', 'vendor'])
    def test_ambiguous_directory_is_excluded_when_gitignored(
        self, tmp_path, ambiguous_dir
    ):
        """The repo declaring it generated is what excludes it."""
        (tmp_path / ".gitignore").write_text(f"{ambiguous_dir}/\n")
        _write(tmp_path, "real.py", "def real(): pass\n")
        _write(tmp_path, f"{ambiguous_dir}/script.py", "def script(): pass\n")

        index = ProjectIndex(str(tmp_path))
        selected, _ = index._select_files(["**/*.py"], 100)

        assert {p.name for p, _ in selected} == {'real.py'}

    def test_gitignored_file_is_excluded(self, tmp_path):
        """The repo's own .gitignore is honored, as it is during prompt
        assembly — the index used to have no exclusions of any kind."""
        (tmp_path / ".gitignore").write_text("generated_client.py\n")
        _write(tmp_path, "real.py", "def real(): pass\n")
        _write(tmp_path, "generated_client.py", "def gen(): pass\n")

        index = ProjectIndex(str(tmp_path))
        selected, _ = index._select_files(["**/*.py"], 100)

        assert {p.name for p, _ in selected} == {'real.py'}

    def test_file_under_a_gitignored_directory_is_excluded(self, tmp_path):
        """The ancestor case: `should_ignore` alone tests only the path it
        is given, so a file inside an ignored directory does not match on
        its own name. The directory chain has to be walked."""
        (tmp_path / ".gitignore").write_text("codegen/\n")
        _write(tmp_path, "real.py", "def real(): pass\n")
        _write(tmp_path, "codegen/deep/nested/api.py", "def api(): pass\n")

        index = ProjectIndex(str(tmp_path))
        selected, _ = index._select_files(["**/*.py"], 100)

        assert {p.name for p, _ in selected} == {'real.py'}

    def test_excluded_count_is_not_inflated_by_overlapping_patterns(self, tmp_path):
        """A file matched by two patterns is one excluded path, not two."""
        _write(tmp_path, "node_modules/pkg/index.js", "module.exports = 1;\n")

        index = ProjectIndex(str(tmp_path))
        _, report = index._select_files(["**/*.js", "**/*.js"], 100)

        assert report['excluded'] == 1

    @pytest.mark.parametrize("path", [
        ".claude/skills/deploy-app/scripts/deploy_verify.py",
        ".codex/skills/x/run.py",
        ".car/agents/thing.py",
        "src/worktrees/manager.py",
    ])
    def test_tracked_agent_source_is_indexed(self, tmp_path, path):
        """The INDEX side of the same rule the gatherer got.

        Deleting `.claude`/`worktrees` from the old exclusion list only
        removed a negative assertion. Nothing said these are now indexed, so
        re-adding a name list tomorrow would break this silently and no test
        would notice. One live repo has 319 tracked files under `.claude/`.
        """
        _write(tmp_path, "real.py", "def real(): pass\n")
        _write(tmp_path, path, "def agent_helper(): return 1\n")

        index = ProjectIndex(str(tmp_path))
        selected, _ = index._select_files(["**/*.py"], 100)

        assert path in {str(p.relative_to(index.repo_root)) for p, _ in selected}

    @pytest.mark.parametrize("path", [
        ".claude/worktrees/a/src/copy.py",
        ".codex/worktrees/issue-1/src/copy.py",
        ".worktrees/PAR-1/copy.py",
    ])
    def test_worktree_copies_are_still_not_indexed(self, tmp_path, path):
        """The other half: the copies must stay out."""
        _write(tmp_path, "real.py", "def real(): pass\n")
        _write(tmp_path, path, "def copy_of_real(): pass\n")

        index = ProjectIndex(str(tmp_path))
        selected, _ = index._select_files(["**/*.py"], 100)

        assert path not in {str(p.relative_to(index.repo_root)) for p, _ in selected}

    def test_a_file_named_like_an_excluded_dir_is_kept(self, tmp_path):
        """Only directory components are matched, never the filename."""
        _write(tmp_path, "build.py", "def build(): pass\n")

        index = ProjectIndex(str(tmp_path))
        selected, _ = index._select_files(["**/*.py"], 100)

        assert {p.name for p, _ in selected} == {'build.py'}


class TestContentDeduplication:
    def test_duplicate_content_does_not_consume_a_slot(self, tmp_path):
        """A second copy of a file must not cost a slot the original needs.

        A worktree is a copy of the repository, so without this the same
        file competes with itself for the budget.
        """
        identical = "def shared():\n    return 1\n"
        _write(tmp_path, "a/dup.py", identical)
        _write(tmp_path, "b/dup.py", identical)
        _write(tmp_path, "c/unique.py", "def unique():\n    return 2\n")

        index = ProjectIndex(str(tmp_path))
        selected, report = index._select_files(["**/*.py"], 2)

        assert report['duplicates'] == 1
        assert len(selected) == 2
        # The slot freed by the duplicate went to the unique file, not to waste.
        assert 'unique.py' in {p.name for p, _ in selected}

    def test_unused_quota_refills_across_languages(self, tmp_path):
        """Dedup must not shrink the index below the cap the operator set.

        Backfilling only within a language leaves a hole: if every remaining
        Python candidate is a duplicate, Python's quota goes unspent even
        though unique C# files are sitting right there. The refill pass is
        what spends it.
        """
        identical = "def same():\n    return 1\n"
        for i in range(8):
            _write(tmp_path, f"py/dup{i}.py", identical)
        for i in range(8):
            _write(tmp_path, f"cs/File{i}.cs", f"public class C{i} {{ int a = {i}; }}\n")

        index = ProjectIndex(str(tmp_path))
        selected, report = index._select_files(["**/*.py", "**/*.cs"], 10)

        # 8 identical Python files contribute 1 unique file, so 9 unique
        # files exist against a cap of 10 — every one of them must land.
        # Quotas alone give python 5 and c_sharp 5, which would stop at 6.
        assert len(selected) == 9, (
            f"9 unique files exist under a cap of 10, got {len(selected)}"
        )
        assert report['duplicates'] == 7
        suffixes = [p.suffix for p, _ in selected]
        assert suffixes.count('.py') == 1
        assert suffixes.count('.cs') == 8

    def test_refill_does_not_re_examine_files(self, tmp_path):
        """Each candidate is hashed at most once across both passes."""
        for i in range(6):
            _write(tmp_path, f"cs/File{i}.cs", f"public class C{i} {{ int a = {i}; }}\n")
        _write(tmp_path, "py/only.py", "def only(): pass\n")

        index = ProjectIndex(str(tmp_path))
        hashed = []
        original = index._compute_file_hash
        index._compute_file_hash = lambda p: (hashed.append(p), original(p))[1]

        selected, _ = index._select_files(["**/*.py", "**/*.cs"], 100)

        assert len(hashed) == len(set(hashed)) == len(selected) == 7

    def test_selection_hash_is_the_one_staleness_checks_against(self, tmp_path):
        """The hash computed during selection is reused, not recomputed.

        `build_index` stopped calling `_compute_file_hash` in its own loop, so
        what matters is that the value carried over from selection is what
        `check_staleness` later compares against — otherwise an index would
        report itself stale the moment it was built.
        """
        _write(tmp_path, "a.py", "def a(): return 1\n")
        _write(tmp_path, "b.py", "def b(): return 2\n")

        index = ProjectIndex(str(tmp_path))
        index.build_index(file_patterns=["**/*.py"], max_files=10)

        assert set(index.snapshot.file_hashes) == {'a.py', 'b.py'}
        is_stale, ratio, changed = index.check_staleness()
        assert changed == []
        assert is_stale is False


class TestSecurityChecksStillApply:
    def test_symlinks_are_rejected(self, tmp_path):
        real = _write(tmp_path, "real.py", "def real(): pass\n")
        (tmp_path / "link.py").symlink_to(real)

        index = ProjectIndex(str(tmp_path))
        selected, _ = index._select_files(["**/*.py"], 100)

        assert {p.name for p, _ in selected} == {'real.py'}

    def test_broken_symlink_does_not_crash(self, tmp_path):
        _write(tmp_path, "real.py", "def real(): pass\n")
        (tmp_path / "dangling.py").symlink_to(tmp_path / "nothing_here.py")

        index = ProjectIndex(str(tmp_path))
        selected, _ = index._select_files(["**/*.py"], 100)

        assert {p.name for p, _ in selected} == {'real.py'}

    def test_symlink_is_rejected_without_stating_its_target(self, monkeypatch,
                                                            tmp_path):
        """The symlink check must run before anything that follows the link.

        `is_file()` and `resolve()` both dereference, so ordering either of
        them first would stat the target — which is precisely what rejecting
        symlinks is meant to avoid.
        """
        real = _write(tmp_path, "real.py", "def real(): pass\n")
        (tmp_path / "link.py").symlink_to(real)

        followed = []
        original_is_file = Path.is_file

        def tracking_is_file(self):
            if self.name == 'link.py':
                followed.append(self)
            return original_is_file(self)

        monkeypatch.setattr(Path, "is_file", tracking_is_file)

        index = ProjectIndex(str(tmp_path))
        index._select_files(["**/*.py"], 100)

        assert followed == [], "symlink was dereferenced before being rejected"

    def test_symlink_out_of_repo_is_rejected(self, tmp_path):
        outside = tmp_path.parent / "outside_secret.py"
        outside.write_text("SECRET = 1\n")
        repo = tmp_path / "repo"
        repo.mkdir()
        _write(repo, "real.py", "def real(): pass\n")
        (repo / "escape.py").symlink_to(outside)

        index = ProjectIndex(str(repo))
        selected, _ = index._select_files(["**/*.py"], 100)

        assert {p.name for p, _ in selected} == {'real.py'}


class TestSelectionReport:
    def test_reports_truncation(self, dotnet_repo):
        index = ProjectIndex(str(dotnet_repo))
        _, report = index._select_files(DEFAULT_PATTERNS, 100)

        assert report['truncated'] is True
        assert report['selected'] == 100
        assert report['eligible'] == 325  # 300 cs + 20 ts + 5 py, worktree excluded

    def test_dedup_alone_is_not_reported_as_a_cap_truncation(self, tmp_path):
        """Only the cap may set `truncated` — it names `--max-files` in the
        message, and raising a cap that was never reached does nothing.

        With 7 files, 5 of them identical, and a cap of 1000, the report used
        to read "Indexed 2 of 7 eligible files (capped at --max-files=1000)".
        """
        identical = "def same():\n    return 1\n"
        for i in range(5):
            _write(tmp_path, f"dup{i}.py", identical)
        _write(tmp_path, "a.py", "def a(): return 1\n")
        _write(tmp_path, "b.py", "def b(): return 2\n")

        index = ProjectIndex(str(tmp_path))
        _, report = index._select_files(["**/*.py"], 1000)

        assert report['truncated'] is False
        assert report['duplicates'] == 4
        assert _format_selection_report(report) != []  # duplicates still reported
        assert "--max-files" not in "\n".join(_format_selection_report(report))

    def test_cap_landing_exactly_on_the_unique_count_is_not_truncation(
        self, tmp_path
    ):
        """The only unindexed file is a duplicate, so the cap gained nothing.

        `selected >= max_files` called this truncated because the budget did
        fill — but raising `--max-files` would still index 2 files, which is
        the same wrong-knob advice the dedup case produced.
        """
        identical = "def same():\n    return 1\n"
        _write(tmp_path, "a.py", identical)
        _write(tmp_path, "b.py", identical)
        _write(tmp_path, "c.py", "def c(): return 2\n")

        index = ProjectIndex(str(tmp_path))
        _, report = index._select_files(["**/*.py"], 2)

        assert report['selected'] == 2
        assert report['duplicates'] == 1
        assert report['truncated'] is False
        assert "--max-files" not in "\n".join(_format_selection_report(report))

    def test_no_truncation_when_everything_fits(self, tmp_path):
        _write(tmp_path, "a.py", "def a(): pass\n")
        _write(tmp_path, "b.py", "def b(): pass\n")

        index = ProjectIndex(str(tmp_path))
        _, report = index._select_files(["**/*.py"], 100)

        assert report['truncated'] is False
        assert report['selected'] == report['eligible'] == 2

    def test_build_index_exposes_the_report(self, tmp_path):
        _write(tmp_path, "a.py", "def a(): return 1\n")

        index = ProjectIndex(str(tmp_path))
        index.build_index(file_patterns=["**/*.py"], max_files=10)

        assert index.selection_report is not None
        assert index.selection_report['selected'] == 1


class TestChunkCap:
    """`MAX_CHUNKS_PER_REPO` must not undo the file-level apportionment."""

    def _chunks(self, spec):
        """Build chunks as `[(file_path, count), ...]`, grouped by file."""
        out = []
        for file_path, count in spec:
            for i in range(count):
                out.append(CodeChunk(
                    file_path=file_path,
                    chunk_id=f"function:f{i}",
                    content="x",
                    chunk_type="function",
                    start_line=1,
                    end_line=1,
                ))
        return out

    def test_under_the_cap_is_returned_unchanged(self):
        chunks = self._chunks([("a.py", 3), ("b.cs", 4)])
        assert ProjectIndex._cap_chunks(chunks, 100) == chunks

    def test_cap_is_respected(self):
        chunks = self._chunks([("a.py", 50), ("b.cs", 50)])
        assert len(ProjectIndex._cap_chunks(chunks, 10)) == 10

    def test_a_chunk_heavy_language_cannot_evict_the_others(self):
        """The C1 regression: slicing kept 1000 C# chunks and zero of anything
        else, discarding the very allocation `_allocate_slots` produced.

        Chunks arrive grouped by file and files arrive grouped by language, so
        a plain `[:cap]` is language-ordered truncation by another name.
        """
        spec = [(f"src/S{i}.cs", 15) for i in range(60)]
        spec += [(f"web/c{i}.ts", 2) for i in range(7)]
        spec += [(f"adws/t{i}.py", 1) for i in range(2)]
        kept = ProjectIndex._cap_chunks(self._chunks(spec), 100)

        suffixes = {Path(c.file_path).suffix for c in kept}
        assert suffixes == {'.cs', '.ts', '.py'}

    def test_every_file_is_represented_when_slots_allow(self):
        """Round-robin means each file keeps a chunk before any keeps two."""
        spec = [(f"src/S{i}.cs", 15) for i in range(60)]
        spec += [(f"web/c{i}.ts", 2) for i in range(7)]
        chunks = self._chunks(spec)
        kept = ProjectIndex._cap_chunks(chunks, 100)

        assert len({c.file_path for c in kept}) == 67

    def test_build_index_reports_the_chunk_cap(self, tmp_path):
        """The cap was previously invisible: `--index` printed success either way."""
        for i in range(4):
            body = "\n".join(f"def f{i}_{j}(): return {j}" for j in range(10))
            _write(tmp_path, f"m{i}.py", body + "\n")

        index = ProjectIndex(str(tmp_path))
        with patch("neo.index.project_index.MAX_CHUNKS_PER_REPO", 10):
            index.build_index(file_patterns=["**/*.py"], max_files=10)

        report = index.selection_report
        assert report['chunks_capped'] is True
        assert report['chunks_kept'] == 10
        assert report['chunks_extracted'] == 40

    def test_files_left_without_chunks_are_reported(self, tmp_path):
        """Round-robin represents every file only while slots >= files.

        `MAX_CHUNKS_PER_REPO` is fixed and `--max-files` is not, so raising
        the file cap past the chunk cap leaves files with nothing in the
        index — and `truncated` is False there, because the file cap was not
        the binding constraint. Without this line the operator is told
        nothing was truncated while two thirds of their files are absent.
        """
        for i in range(6):
            _write(tmp_path, f"m{i}.py", f"def f{i}(): return {i}\n")

        index = ProjectIndex(str(tmp_path))
        with patch("neo.index.project_index.MAX_CHUNKS_PER_REPO", 2):
            index.build_index(file_patterns=["**/*.py"], max_files=100)

        report = index.selection_report
        assert report['truncated'] is False       # the file cap never bound
        assert report['selected'] == 6
        assert report['files_producing_chunks'] == 6   # every file had chunks
        assert report['files_with_chunks'] == 2        # the cap took four

        printed = "\n".join(_format_selection_report(report))
        assert "4 indexed files lost every chunk to the 2-chunk cap" in printed

    def test_no_shortfall_line_when_every_file_is_represented(self, tmp_path):
        for i in range(3):
            _write(tmp_path, f"m{i}.py", f"def f{i}(): return {i}\n")

        index = ProjectIndex(str(tmp_path))
        index.build_index(file_patterns=["**/*.py"], max_files=100)

        printed = "\n".join(_format_selection_report(index.selection_report))
        assert "lost every chunk" not in printed
        assert "yielded no chunks" not in printed

    def test_constructless_file_is_not_blamed_on_the_chunk_cap(self, tmp_path):
        """A file with nothing to extract is not a cap event.

        `_extract_chunks_from_file` returns [] for any file holding no
        function, class, interface or struct — an empty `__init__.py`, an
        enum-only `.cs`, a type-alias-only `.ts`. That is independent of every
        cap, and no cap setting changes it.

        The first version of this report subtracted post-cap
        `files_with_chunks` from `selected` and attributed the whole
        difference to `MAX_CHUNKS_PER_REPO`. Reproduced against the neo repo
        itself: 25 files selected, 559 of 559 chunks kept, `chunks_capped`
        False on the same report, and the console still printed "the
        1000-chunk cap is below the 25 files selected; lower --max-files".
        The culprit was an empty `tests/__init__.py`, and lowering
        --max-files would only have indexed less.
        """
        _write(tmp_path, "real.py", "def f(): return 1\n")
        _write(tmp_path, "__init__.py", "")
        _write(tmp_path, "constants.py", "TIMEOUT = 30\nRETRIES = 3\n")

        index = ProjectIndex(str(tmp_path))
        index.build_index(file_patterns=["**/*.py"], max_files=100)

        report = index.selection_report
        assert report['chunks_capped'] is False
        assert report['selected'] == 3
        assert report['files_producing_chunks'] == 1
        assert report['files_with_chunks'] == 1

        printed = "\n".join(_format_selection_report(report))
        assert "2 selected files yielded no chunks" in printed
        # The cap did not fire, so it must not be named and no remedy offered.
        assert "cap" not in printed
        assert "--max-files" not in printed

    def test_cap_starvation_and_barren_files_are_reported_separately(self, tmp_path):
        """Both causes at once must not be conflated into either one."""
        for i in range(4):
            _write(tmp_path, f"m{i}.py", f"def f{i}(): return {i}\n")
        _write(tmp_path, "empty.py", "")

        index = ProjectIndex(str(tmp_path))
        with patch("neo.index.project_index.MAX_CHUNKS_PER_REPO", 2):
            index.build_index(file_patterns=["**/*.py"], max_files=100)

        report = index.selection_report
        assert report['selected'] == 5
        assert report['files_producing_chunks'] == 4
        assert report['files_with_chunks'] == 2

        printed = "\n".join(_format_selection_report(report))
        assert "2 indexed files lost every chunk to the 2-chunk cap" in printed
        assert "1 selected file yielded no chunks" in printed

    def test_report_without_chunk_phase_makes_no_cap_accusation(self):
        """A selection-only report predates `files_producing_chunks`.

        Falling back to the post-cap count makes `starved` zero rather than
        equal to `selected`, so a missing key produces silence instead of a
        fabricated cap event.
        """
        printed = "\n".join(_format_selection_report({
            'eligible': 5, 'selected': 5, 'excluded': 0, 'duplicates': 0,
            'max_files': 100, 'truncated': False, 'per_language': {},
            'files_with_chunks': 5, 'max_chunks': 1000,
        }))
        assert printed == ""

    def test_no_chunk_cap_flag_when_everything_fits(self, tmp_path):
        _write(tmp_path, "a.py", "def a(): return 1\n")

        index = ProjectIndex(str(tmp_path))
        index.build_index(file_patterns=["**/*.py"], max_files=10)

        assert index.selection_report['chunks_capped'] is False


class TestReportFormatting:
    def test_truncation_is_stated_with_language_breakdown(self):
        lines = _format_selection_report({
            'eligible': 4344, 'selected': 100, 'excluded': 0, 'duplicates': 0,
            'max_files': 100, 'truncated': True,
            'per_language': {
                'c_sharp': {'selected': 96, 'eligible': 4272},
                'typescript': {'selected': 3, 'eligible': 54},
                'python': {'selected': 1, 'eligible': 18},
            },
        })
        joined = "\n".join(lines)
        assert "100 of 4344" in joined
        assert "--max-files=100" in joined
        # Largest language first — that is what an operator scans for.
        assert joined.index("c_sharp") < joined.index("typescript")
        assert "c_sharp 96/4272" in joined

    def test_silent_when_nothing_was_left_out(self):
        assert _format_selection_report({
            'eligible': 5, 'selected': 5, 'excluded': 0, 'duplicates': 0,
            'max_files': 100, 'truncated': False, 'per_language': {},
        }) == []

    def test_missing_report_is_tolerated(self):
        assert _format_selection_report(None) == []

    def test_exclusions_and_duplicates_are_reported(self):
        joined = "\n".join(_format_selection_report({
            'eligible': 5, 'selected': 5, 'excluded': 200, 'duplicates': 7,
            'max_files': 100, 'truncated': False, 'per_language': {},
        }))
        assert "200" in joined
        assert "7" in joined


class TestChunkAllocationIsProportional:
    """Chunk slots go where the code is, not one-per-file-per-round.

    Round-robin replaced a list slice for a good reason — `chunks[:cap]` on a
    language-ordered list kept 1000 C# chunks and dropped every other language.
    But taking every file's first chunk before any file's second gives every
    file the SAME COUNT regardless of how much is in it, which is a subtler
    version of the same bias. Measured on this repo before the change:

        src/neo/memory/store.py    82 symbols,  6 indexed,   7%
        src/neo/engine.py          95 symbols,  6 indexed,   6%
        src/neo/text_budget.py      4 symbols,  4 indexed, 100%

    A file cannot be retrieved for what was never indexed, so the semantic
    channel was blind to 93% of the two files most likely to be relevant.
    After: every file lands at roughly the same COVERAGE FRACTION instead.
    """

    @staticmethod
    def _chunks(spec):
        """`spec` maps file path -> number of chunks that file produces."""
        from neo.index.project_index import CodeChunk
        out = []
        for path, count in spec.items():
            for i in range(count):
                out.append(CodeChunk(
                    file_path=path, chunk_id=f"{path}:{i}", content=f"c{i}",
                    chunk_type="function", start_line=i, end_line=i,
                ))
        return out

    def test_a_big_file_gets_more_slots_than_a_small_one(self):
        from neo.index.project_index import ProjectIndex

        kept = ProjectIndex._cap_chunks(
            self._chunks({"big.py": 80, "small.py": 4}), cap=42
        )
        per = {}
        for c in kept:
            per[c.file_path] = per.get(c.file_path, 0) + 1

        assert per["big.py"] > per["small.py"], (
            "round-robin is back: every file got the same count regardless "
            "of how much is in it"
        )

    def test_every_file_keeps_at_least_one_slot(self):
        """The property round-robin was protecting. A file with zero chunks
        indexed cannot be retrieved at all, so proportional-without-a-floor
        would make small files invisible — trading one bias for another."""
        from neo.index.project_index import ProjectIndex

        spec = {"huge.py": 500, **{f"tiny{i}.py": 1 for i in range(20)}}
        kept = ProjectIndex._cap_chunks(self._chunks(spec), cap=60)
        represented = {c.file_path for c in kept}

        assert represented == set(spec), "a file was allocated zero chunks"

    def test_coverage_fraction_is_roughly_equal(self):
        """The intended shape: equal FRACTION, not equal COUNT."""
        from neo.index.project_index import ProjectIndex

        spec = {"a.py": 80, "b.py": 40, "c.py": 20}
        kept = ProjectIndex._cap_chunks(self._chunks(spec), cap=70)
        per = {}
        for c in kept:
            per[c.file_path] = per.get(c.file_path, 0) + 1

        fractions = [per[p] / spec[p] for p in spec]
        assert max(fractions) - min(fractions) < 0.25, f"uneven: {fractions}"

    def test_under_cap_returns_everything_untouched(self):
        from neo.index.project_index import ProjectIndex

        chunks = self._chunks({"a.py": 3, "b.py": 2})
        assert ProjectIndex._cap_chunks(chunks, cap=99) is chunks
