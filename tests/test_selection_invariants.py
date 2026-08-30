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


def _limit_for(path: str, pinned: bool = False) -> int:
    """The renderer's per-file character cap for `path`.

    `pinned` is not a detail: `_render_context_files` applies no cap at all to
    a pinned file, because the gatherer already bounded it by `--max-bytes` and
    marked any cut it made. Both stages that pin — a path the prompt named and
    an `--include` pattern — land here, so a helper that only knew about the
    important/ordinary split reported a cut on files the renderer never cuts.
    """
    from neo.engine import _IMPORTANT_FILE_PATTERNS

    if pinned:
        return 10**9
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
        means no line range and content byte-identical to disk. It reaches the
        prompt uncut as well, which
        `TestTruncationIsAlwaysMarked.test_the_pinned_target_is_oversized_and_still_uncut`
        pins at the renderer.
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

    The two halves now make the SAME claim, which is the front door's stage 1
    and stage 2 arriving at one guarantee: the named file bypasses scoring, the
    file cap and the chunk cap, and the renderer's per-file character cap does
    not apply to it either. Only `--max-bytes` may cut it, and that cut is
    marked. It used to be weaker on the prompt-named side — a scoring pin,
    where the file won the ranking and was then delivered like any other
    winner, so the fixture target arrived cut at the renderer. "Ranked first"
    and "present whole" are different claims and only one of them is what a
    caller who typed a path asked for.

    Kept as a separate class rather than merged, because the two stages resolve
    a name differently: `--include` is an `fnmatch` glob with an exact-path
    rescue past the walker's size ceiling, and a prompt-named path is a
    boundary-anchored match against what the walk already found. Equal
    guarantees, different ways of failing to find the file.

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
            was_cut = len(f.content or "") > _limit_for(f.path, f.pinned)
            marked = "[truncated:" in section
            assert marked is was_cut, (
                f"{f.path}: cut={was_cut} but marker={marked}"
            )

    def test_something_unpinned_is_large_enough_to_be_cut(self, repo):
        """Guards the guard: with nothing oversized the test above is vacuous.

        The specimen is `bulk_rel`, not the target. Naming a path used to only
        BOOST it, so the target competed for delivery like anything else and
        the renderer cut it; the front door pins a named path, and a pin is
        delivered whole past the renderer's cap. Left pointing at the target,
        this assertion would fail — and the marker test beside it would go
        quietly vacuous, which is the outcome this exists to prevent.
        """
        gathered = _gather(repo)
        bulk = next(g for g in gathered if g.rel_path == repo.bulk_rel)

        assert len(bulk.content or "") > _limit_for(bulk.path)

    def test_the_pinned_target_is_oversized_and_still_uncut(self, repo):
        """G2-inv, at the end of the pipeline: a named file is not cut at all.

        The target is past the renderer's per-file cap by a wide margin, so
        "uncut" is a real result rather than a file that happened to fit.
        """
        gathered = _gather(repo)
        target = next(g for g in gathered if g.rel_path == repo.target_rel)
        assert target.pinned
        assert len(target.content or "") > _limit_for(target.path)

        files = _as_engine_files(gathered)
        sections, _banner, _visible = NeoEngine._render_context_files(files)
        target_section = next(
            s for s in sections if repo.target_rel in s or
            Path(repo.target_rel).name in s
        )

        assert "[truncated:" not in target_section
        assert repo.sentinel in target_section

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
            f.path for f in files
            if len(f.content or "") > _limit_for(f.path, f.pinned)
        })
        noun = "file" if cut == 1 else "files"
        assert f"{cut} {noun} truncated, marked inline" in banner, banner


class TestTheFileCapIsHonoured:
    """G3-inv, applied to `--max-files`: a cap that can be exceeded is not a cap.

    `calculate_adaptive_limit` picks a file budget from prompt specificity, and
    its three broad-prompt buckets used to be returned VERBATIM — so a vague
    prompt under `--max-files 5` was given 15 files, 3x the requested ceiling.
    Only the specific bucket honoured it, and that one does so by construction
    (it returns `default_max`), which is why nothing caught this.

    It also made the knob unmeasurable: sweeping `--max-files` below 25 moved
    nothing for any prompt that was not highly specific, so a delivery-cap
    sweep measured the adaptive floor rather than the cap.
    """

    @pytest.mark.parametrize("prompt", [
        "review this",                                   # very vague  -> 15
        "review this codebase",                          # somewhat    -> 20
        "refactor memory delete synthesis",              # moderate    -> 25
        "review ProjectIndex.retrieve() in src/neo/index/project_index.py",
    ])
    @pytest.mark.parametrize("ceiling", [1, 5, 10, 25, 30, 50])
    def test_never_exceeds_the_requested_ceiling(self, prompt, ceiling):
        from neo.context_gatherer import calculate_adaptive_limit
        assert calculate_adaptive_limit(prompt, ceiling) <= ceiling

    def test_the_default_budget_is_unchanged(self):
        """The fix must not silently re-tune the shipped default. At
        `--max-files 30` every bucket is already <= 30, so the mapping stands."""
        from neo.context_gatherer import calculate_adaptive_limit
        assert calculate_adaptive_limit("review this", 30) == 15
        assert calculate_adaptive_limit("review this codebase", 30) == 20
        assert calculate_adaptive_limit("refactor memory delete synthesis", 30) == 25

    def test_a_generous_ceiling_still_gets_the_specificity_budget(self):
        """The cap bounds the result; it does not replace the heuristic. A vague
        prompt with a huge ceiling must still get its small overview budget,
        not the ceiling."""
        from neo.context_gatherer import calculate_adaptive_limit
        assert calculate_adaptive_limit("review this", 500) == 15


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
