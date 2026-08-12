"""The guard-invariant battery — the free half of the release gate.

Runs on every PR (`pytest -m invariants`, and inside the ordinary suite). It
makes NO model call: everything here is answered by the gatherer, by the
prompt renderer, and by `git check-ignore`.

The invariants are G1-inv..G3-inv from `docs/unified-store-plan.md`, asserted
per language against the generated fixture repos in `tests/fixtures/`:

- **G1-inv** zero selected files that `git check-ignore` excludes; zero
  duplicate copies.
- **G2-inv** a prompt-named file is present, and an `--include` file is
  present whole (or explicitly marked when `--max-bytes` forces a cut).
- **G3-inv** no silent caps — every truncation is marked and the reported
  counts match what was actually sent.

Why per language rather than once: C# was absent from Neo's index for 8.5
months (#158/#159) precisely because every check that existed was run against
Python and passed. A single-language battery would have stayed green through
that entire window. The language axis is the point, not decoration.

The paid half — an actual LLM round trip per language — lives in
`tests/test_release_roundtrip.py` and runs only in the release flow. Both
halves build the same fixtures from the same module so that a red release
gate is diagnosable from a free CI run.
"""

import os
from pathlib import Path

import pytest

from neo.context_gatherer import GatherConfig, gather_context
from neo.engine import _CONTEXT_FILE_CHARS, _IMPORTANT_FILE_CHARS, NeoEngine
from neo.models import ContextFile
from tests.fixtures.language_repos import (
    LANGUAGES,
    FixtureRepo,
    build_fixture_repo,
    check_ignored,
)

pytestmark = pytest.mark.invariants


@pytest.fixture(scope="session")
def fixture_repos(tmp_path_factory) -> dict[str, FixtureRepo]:
    """Build each language fixture once for the whole session.

    Session-scoped because building them is three `git init`s and nothing
    here mutates them; the gathering that tests actually assert on happens
    per test, under conftest's per-test home isolation.
    """
    base = tmp_path_factory.mktemp("release_gate_fixtures")
    return {
        language: build_fixture_repo(language, base / language)
        for language in LANGUAGES
    }


@pytest.fixture
def repo(request, fixture_repos) -> FixtureRepo:
    return fixture_repos[request.param]


def _gather(fx: FixtureRepo, includes: list[str] | None = None):
    """Gather with the same config `cli.main` builds for a plain invocation.

    Deliberately not a reduced or tuned config: an invariant asserted under
    settings no user runs is an invariant about nothing.
    """
    return gather_context(
        GatherConfig(
            root=str(fx.root),
            prompt=fx.prompt,
            exts=None,
            includes=list(includes or []),
            excludes=[],
            max_files=30,
            use_git=True,
        )
    )


def _as_engine_files(gathered) -> list[ContextFile]:
    """The conversion `cli.main` performs before the engine sees them."""
    return [
        ContextFile(
            path=g.path,
            content=g.content,
            line_range=(g.start, g.end) if g.start else None,
            pinned=g.pinned,
        )
        for g in gathered
    ]


def _limit_for(path: str) -> int:
    from neo.engine import _IMPORTANT_FILE_PATTERNS

    lowered = path.lower()
    return (
        _IMPORTANT_FILE_CHARS
        if any(pat in lowered for pat in _IMPORTANT_FILE_PATTERNS)
        else _CONTEXT_FILE_CHARS
    )


_PARAM = pytest.mark.parametrize("repo", LANGUAGES, indirect=True)


@_PARAM
class TestLanguageReachesContext:
    """The 8.5-month failure, asserted as a positive per language."""

    def test_the_languages_own_source_is_selected(self, repo):
        gathered = _gather(repo)
        rels = [g.rel_path for g in gathered]

        of_language = [r for r in rels if r.endswith(f".{repo.ext}")]
        assert of_language, (
            f"no .{repo.ext} file reached context for {repo.language}; "
            f"selected {rels}"
        )

    def test_the_selected_source_carries_real_content(self, repo):
        """Selected-but-empty is the same absence with a better disguise."""
        gathered = _gather(repo)
        target = next(g for g in gathered if g.rel_path == repo.target_rel)

        assert (target.content or "").strip(), f"{repo.target_rel} arrived empty"
        assert repo.sentinel in (target.content or ""), (
            f"{repo.sentinel} missing from the selected copy of "
            f"{repo.target_rel}"
        )


@_PARAM
class TestNamedFileGuarantee:
    """G2-inv: a file the prompt names by path is present."""

    def test_the_prompt_named_file_is_selected(self, repo):
        gathered = _gather(repo)
        rels = [g.rel_path for g in gathered]

        assert repo.target_rel in rels, (
            f"prompt named {repo.target_rel} and it was not selected; "
            f"got {rels}"
        )

    def test_the_named_file_arrives_whole(self, repo):
        """Whole, not a window.

        `--include`/named-path semantics are "the named file, whole, or with
        an explicit marker" (standing ruling 1). At the gatherer layer that
        means no line range: the truncation that DOES happen to this file
        happens later, at the prompt renderer, and is marked there — which is
        what `TestTruncationIsAlwaysMarked` pins.
        """
        gathered = _gather(repo)
        target = next(g for g in gathered if g.rel_path == repo.target_rel)

        assert target.start is None and target.end is None, (
            f"{repo.target_rel} was windowed to lines "
            f"{target.start}-{target.end} despite being named"
        )
        on_disk = (repo.root / repo.target_rel).read_text(encoding="utf-8")
        assert (target.content or "") == on_disk


@_PARAM
class TestIncludedFileGuarantee:
    """G2-inv, `--include` half: 100% named-file presence, whole.

    The prompt-named half above is a scoring pin — the file wins the ranking
    and is then delivered like any winner, which is why the fixture target
    still arrives cut at the renderer. `--include` is a stronger claim
    (standing ruling 1): the operator asserted the input, so the file bypasses
    scoring, the file cap and the chunk cap, and the renderer's per-file
    character cap does not apply to it either. Only `--max-bytes` may cut it,
    and that cut is marked.

    Asserted per language for the same reason everything else here is: a
    guarantee verified on Python alone stayed green through the 8.5 months C#
    was missing entirely.
    """

    def test_the_included_file_is_present(self, repo):
        gathered = _gather(repo, includes=[repo.target_rel])

        assert repo.target_rel in [g.rel_path for g in gathered]

    def test_the_included_file_is_pinned_whole_and_uncut(self, repo):
        gathered = _gather(repo, includes=[repo.target_rel])
        entries = [g for g in gathered if g.rel_path == repo.target_rel]

        assert len(entries) == 1, f"{repo.target_rel} arrived in {len(entries)} pieces"
        target = entries[0]
        assert target.pinned is True
        assert target.truncated is False
        assert target.start is None and target.end is None
        on_disk = (repo.root / repo.target_rel).read_text(encoding="utf-8")
        assert (target.content or "") == on_disk

    def test_the_scan_still_runs_alongside_the_pin(self, repo):
        """Ruling 1 is a conjunction. `--include` used to narrow the walk, so
        naming one file meant nothing else could be selected."""
        gathered = _gather(repo, includes=[repo.target_rel])
        others = [g.rel_path for g in gathered if g.rel_path != repo.target_rel]

        assert others, f"{repo.language}: the scan contributed nothing"

    def test_the_renderer_sends_the_included_file_whole(self, repo):
        """The end of the pipeline, which is where the guarantee is spent.

        The same file WITHOUT `--include` is cut here — that is what
        `TestTruncationIsAlwaysMarked` pins — so this is the assertion that
        distinguishes an included file from a merely well-ranked one.
        """
        gathered = _gather(repo, includes=[repo.target_rel])
        files = _as_engine_files(gathered)
        sections, _banner, visible = NeoEngine._render_context_files(files)

        target = next(
            f for f in files if Path(f.path).name == Path(repo.target_rel).name
        )
        assert len(target.content or "") > _limit_for(target.path), (
            "fixture stopped exercising the renderer cap"
        )
        section = next(
            s for s in sections if Path(repo.target_rel).name in s.splitlines()[1]
        )
        assert "[truncated:" not in section
        shown = next(
            v for v in visible if Path(v.path).name == Path(repo.target_rel).name
        )
        assert shown.content == target.content


@_PARAM
class TestNothingGitIgnoredIsSelected:
    """G1-inv, differential half: git decides, not a second implementation."""

    def test_no_selected_path_is_excluded_by_git(self, repo):
        gathered = _gather(repo)
        rels = [g.rel_path for g in gathered]

        offenders = check_ignored(repo.root, rels)
        assert offenders == [], (
            f"{repo.language}: selected {len(offenders)} path(s) that "
            f"git check-ignore excludes: {offenders}"
        )

    def test_the_planted_junk_is_specifically_absent(self, repo):
        """A repo with nothing ignorable in it passes the test above trivially.

        These files are the same language as the target, share its filename
        stem and contain its sentinel, so they compete for the same slots.
        """
        gathered = _gather(repo)
        rels = set(g.rel_path for g in gathered)

        planted = sorted(rels & set(repo.ignored_rels))
        assert planted == [], f"{repo.language}: selected ignored junk {planted}"

    def test_the_planted_junk_is_ignored_by_git(self, repo):
        """Guards the guard: junk git does not ignore proves nothing above."""
        assert sorted(check_ignored(repo.root, list(repo.ignored_rels))) == sorted(
            repo.ignored_rels
        )


@_PARAM
class TestNoDuplicateSelections:
    """G1-inv, dedup half. Asserted on identity, not on a count."""

    def test_no_path_is_selected_twice(self, repo):
        gathered = _gather(repo)
        rels = [g.rel_path for g in gathered]

        assert len(rels) == len(set(rels)), f"duplicate paths in {rels}"

    def test_the_worktree_copy_does_not_compete_with_the_original(self, repo):
        """The measured failure was one file selected six times, once per
        agent worktree, at identical scores — 12 of 16 slots on 2 files."""
        gathered = _gather(repo)
        rels = [g.rel_path for g in gathered]

        assert repo.duplicate_rel not in rels
        basename = os.path.basename(repo.target_rel)
        copies = [r for r in rels if os.path.basename(r) == basename]
        assert copies == [repo.target_rel], f"{basename} selected as {copies}"

    def test_no_two_selected_files_have_identical_content(self, repo):
        """Path-level dedup misses a copy that landed under a different name."""
        gathered = _gather(repo)
        seen: dict[str, str] = {}
        for g in gathered:
            content = g.content or ""
            if not content.strip():
                continue
            assert content not in seen, (
                f"{g.rel_path} is a byte-identical copy of {seen[content]}"
            )
            seen[content] = g.rel_path


@_PARAM
class TestTruncationIsAlwaysMarked:
    """G3-inv: no silent caps, and reported counts match reality."""

    def test_every_cut_file_says_it_was_cut(self, repo):
        gathered = _gather(repo)
        files = _as_engine_files(gathered)
        sections, _banner, _visible = NeoEngine._render_context_files(files)

        for f, section in zip(files, sections):
            was_cut = len(f.content or "") > _limit_for(f.path)
            marked = "[truncated:" in section
            assert marked is was_cut, (
                f"{f.path}: cut={was_cut} but marker={marked}"
            )

    def test_the_target_is_actually_large_enough_to_be_cut(self, repo):
        """Guards the guard: with nothing oversized the test above is vacuous.

        The fixture keeps the target well past the cap for exactly this
        reason, and this assertion is what fails if that ever stops being
        true — rather than the marker test quietly passing on no evidence.
        """
        gathered = _gather(repo)
        target = next(g for g in gathered if g.rel_path == repo.target_rel)

        assert len(target.content or "") > _limit_for(target.path)

    def test_the_sentinel_survives_the_cut(self, repo):
        """The cut keeps the HEAD, and the fixture puts the sentinel there.

        Without this the round trip could fail for a reason that has nothing
        to do with the model: the symbol it is asked to name would have been
        truncated away before the prompt was built.
        """
        gathered = _gather(repo)
        _sections, _banner, visible = NeoEngine._render_context_files(
            _as_engine_files(gathered)
        )
        target_visible = next(
            v for v in visible if Path(v.path).name == Path(repo.target_rel).name
        )

        assert repo.sentinel in (target_visible.content or "")

    def test_the_banner_counts_what_was_sent_not_what_was_offered(self, repo):
        gathered = _gather(repo)
        files = _as_engine_files(gathered)
        _sections, banner, visible = NeoEngine._render_context_files(files)

        sent = sum(len(v.content or "") for v in visible)
        offered = sum(len(f.content or "") for f in files)

        assert f"{sent} of {offered} chars" in banner, banner
        assert sent < offered, (
            "fixture stopped exercising the truncating banner form"
        )

    def test_the_banner_counts_the_truncated_files(self, repo):
        """FILES, counted as distinct paths.

        This assertion used to re-implement a per-ENTRY count, so it codified
        the conflation it was named after rather than catching it: two windows
        of one cut file were reported as two files truncated, in a banner whose
        other clause had already said one file.
        """
        gathered = _gather(repo)
        files = _as_engine_files(gathered)
        _sections, banner, _visible = NeoEngine._render_context_files(files)

        cut = len({
            f.path for f in files if len(f.content or "") > _limit_for(f.path)
        })
        noun = "file" if cut == 1 else "files"
        assert f"{cut} {noun} truncated, marked inline" in banner, banner


class TestThePaidHalfStaysOptIn:
    """The round trip must not become a per-PR cost by accident.

    Deleting the env gate on `tests/test_release_roundtrip.py` turns every
    push into three model calls, and nothing else in the suite would notice —
    the tests would simply start passing. The bill is a slow signal; this is
    a fast one.
    """

    def test_the_round_trip_is_gated_on_an_environment_variable(self):
        import tests.test_release_roundtrip as roundtrip

        skipifs = [
            m for m in roundtrip.pytestmark if getattr(m, "name", "") == "skipif"
        ]
        assert skipifs, "the round trip module lost its opt-in gate"
        assert "NEO_RELEASE_ROUNDTRIP" in skipifs[0].kwargs.get("reason", "")

    def test_the_round_trip_carries_the_marker_the_release_job_selects(self):
        """`pytest -m roundtrip` is what publish.yml runs. An unmarked test
        in that module is a language silently dropped from the gate."""
        import tests.test_release_roundtrip as roundtrip

        names = [getattr(m, "name", "") for m in roundtrip.pytestmark]
        assert "roundtrip" in names

    def test_every_language_is_in_the_gate(self):
        """Both halves iterate `LANGUAGES`, so this pins the set itself.

        The plan names C#, TypeScript and Python (G5-inv). A language quietly
        dropped from the tuple would take its round trip with it and every
        remaining test would stay green.
        """
        assert set(LANGUAGES) == {"csharp", "typescript", "python"}
