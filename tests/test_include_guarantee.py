"""Selection truthfulness: what `--include` promises, and what the count means.

Two defects, one theme — the operator's picture of what Neo saw was wrong in a
direction that read as reassuring:

- **#197** `Gathered N files` counted CHUNKS. A run that sent four windows of
  two files reported "4 files" to someone who had named four, so the line
  agreed with the request by arithmetic coincidence while half the named files
  were missing entirely.
- **#198** `--include` narrowed the candidate walk and then let the named file
  compete like any other: it could be evicted under `MIN_SCORE_THRESHOLD`,
  dropped by `--max-files`, or windowed by `MAX_CHUNKS_PER_FILE` — silently in
  every case, and indistinguishable from never having asked.

Standing ruling 1 of `docs/unified-store-plan.md` settles the second: the named
files are GUARANTEED, whole or with an explicit marker, AND the ordinary scan
still runs for anything else useful.

Several tests here assert a CONTROL run first — the same gather without
`--include` — and fail if the control already had what the include is supposed
to guarantee. A guarantee test that passes because the file would have been
selected anyway asserts nothing, and that is the failure mode this whole file
exists to prevent.
"""

import pytest

from neo import cli
from neo.context_gatherer import (
    MAX_CHUNKS_PER_FILE,
    ContextFile as GatheredFile,
    GatherConfig,
    _cut_to_bytes,
    gather_context,
    resolve_includes,
)
from neo.engine import _MAX_CONTEXT_FILES, NeoEngine
from neo.models import ContextFile
from neo.text_budget import MARKER_PREFIX

PROMPT = "Fix the connection pool timeout when the database is busy"

# Past `gather_context`'s 15,000-character chunking threshold, so an unpinned
# copy of this file is guaranteed to arrive as windows. That is what makes the
# "pinned files bypass the chunk cap" assertions non-vacuous.
_BULK = "\n".join(
    f"    # database pool connection timeout retry line {i}" for i in range(400)
)


@pytest.fixture
def repo(tmp_path):
    """A repo with three prompt-relevant files and one the scan will not want.

    `notes/zzz_placeholder.py` shares no token with the prompt, sits one
    directory deep (depth penalty) and carries the `archive` demotion, so
    `score_candidate` floors it at zero and it never becomes a candidate at
    all. It is the file whose presence can only come from `--include`.
    """
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "notes").mkdir()

    (tmp_path / "src" / "app" / "pool.py").write_text(
        "def connection_pool_timeout():\n"
        "    '''Database connection pool timeout handling.'''\n"
        f"{_BULK}\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "app" / "client.py").write_text(
        "def database_client():\n    '''Database client, uses the pool.'''\n"
        "    return connection_pool_timeout()\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "app" / "config.py").write_text(
        "DATABASE_POOL_TIMEOUT = 30  # connection pool timeout, seconds\n",
        encoding="utf-8",
    )
    (tmp_path / "notes" / "zzz_placeholder.py").write_text(
        '"""Archived scratch note, unrelated to anything asked here."""\n'
        "ALPHA = 1\n",
        encoding="utf-8",
    )
    return tmp_path


def _gather(repo, includes=None, **overrides):
    config = GatherConfig(
        root=str(repo),
        prompt=PROMPT,
        exts=None,
        includes=list(includes or []),
        excludes=[],
        max_files=30,
        max_bytes=300_000,
        use_git=False,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return gather_context(config)


def _rels(gathered):
    return [g.rel_path for g in gathered]


class TestTheCountSaysFilesAndChunks:
    """#197: the one-line gather report, which is what most runs show."""

    def test_two_chunks_of_one_file_report_one_file(self):
        gathered = [
            GatheredFile(path="/r/a.py", rel_path="a.py", bytes=10, start=1, end=5),
            GatheredFile(path="/r/a.py", rel_path="a.py", bytes=10, start=40, end=60),
        ]
        summary = cli._gathered_summary(gathered)

        assert "2 chunks" in summary
        assert "from 1 file" in summary
        assert "2 files" not in summary

    def test_the_reported_shape_of_the_run_in_the_issue(self):
        """Four windows over two files — the run that read as "4 files"."""
        gathered = [
            GatheredFile(path="/r/a.cs", rel_path="a.cs", bytes=1, start=164, end=360),
            GatheredFile(path="/r/a.cs", rel_path="a.cs", bytes=1, start=374, end=485),
            GatheredFile(path="/r/b.cs", rel_path="b.cs", bytes=1, start=796, end=850),
            GatheredFile(path="/r/b.cs", rel_path="b.cs", bytes=1, start=606, end=795),
        ]

        assert "4 chunks from 2 files" in cli._gathered_summary(gathered)

    def test_both_numbers_appear_even_when_they_agree(self):
        """One number on the runs where one number happens to be right is how
        a reader learns to expect one number."""
        gathered = [
            GatheredFile(path="/r/a.py", rel_path="a.py", bytes=7),
            GatheredFile(path="/r/b.py", rel_path="b.py", bytes=9),
        ]
        summary = cli._gathered_summary(gathered)

        assert "2 chunks" in summary and "from 2 files" in summary
        assert "16 bytes" in summary

    def test_singulars_are_singular(self):
        summary = cli._gathered_summary(
            [GatheredFile(path="/r/a.py", rel_path="a.py", bytes=1)]
        )

        assert "1 chunk from 1 file" in summary
        assert "chunks" not in summary and "files" not in summary

    def test_an_empty_gather_says_so_without_lying_about_units(self):
        assert "0 chunks from 0 files" in cli._gathered_summary([])


class TestIncludedFilesAreGuaranteed:
    """#198 / ruling 1: the named file arrives, whole, however it ranks."""

    def test_the_control_run_really_does_drop_it(self, repo):
        """Guards every test below: without this the guarantee is untested."""
        assert "notes/zzz_placeholder.py" not in _rels(_gather(repo))

    def test_an_unrankable_file_arrives_when_included(self, repo):
        gathered = _gather(repo, includes=["notes/zzz_placeholder.py"])
        target = next(
            g for g in gathered if g.rel_path == "notes/zzz_placeholder.py"
        )

        on_disk = (repo / "notes" / "zzz_placeholder.py").read_text(encoding="utf-8")
        assert target.content == on_disk
        assert target.pinned is True
        assert target.truncated is False
        assert target.start is None and target.end is None

    def test_the_scan_still_contributes_other_files(self, repo):
        """The other half of ruling 1. `--include` used to narrow the walk, so
        naming one file meant that file and nothing else could be selected."""
        rels = _rels(_gather(repo, includes=["notes/zzz_placeholder.py"]))

        others = [r for r in rels if r != "notes/zzz_placeholder.py"]
        assert others, f"the scan contributed nothing beside the pin: {rels}"
        assert "src/app/pool.py" in others

    def test_an_included_file_is_not_windowed(self, repo):
        """The cap that actually bit in #198: ~400 lines of any file, whatever
        its size, with nothing in the output saying so."""
        control = _gather(repo)
        control_pool = [g for g in control if g.rel_path == "src/app/pool.py"]
        assert control_pool and control_pool[0].start is not None, (
            "fixture stopped exercising the chunk cap"
        )
        assert len(control_pool) <= MAX_CHUNKS_PER_FILE

        gathered = _gather(repo, includes=["src/app/pool.py"])
        pinned = [g for g in gathered if g.rel_path == "src/app/pool.py"]

        assert len(pinned) == 1, "a pinned file was still split into windows"
        assert pinned[0].start is None and pinned[0].end is None
        assert pinned[0].content == (
            (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        )

    def test_a_pinned_file_is_not_also_selected_as_a_window(self, repo):
        """Delivering it whole AND ranking it would emit one file twice, which
        is a duplicate under G1-inv."""
        rels = _rels(_gather(repo, includes=["src/app/pool.py"]))

        assert rels.count("src/app/pool.py") == 1

    def test_the_file_cap_cannot_evict_a_pin(self, repo):
        gathered = _gather(repo, includes=["notes/zzz_placeholder.py"], max_files=1)

        assert "notes/zzz_placeholder.py" in _rels(gathered)

    def test_pins_come_first(self, repo):
        """Ahead of the scan, so the byte ceiling is spent on them first and
        the prompt renderer's file cap reaches them."""
        gathered = _gather(
            repo, includes=["notes/zzz_placeholder.py", "src/app/config.py"]
        )

        assert _rels(gathered)[:2] == [
            "notes/zzz_placeholder.py",
            "src/app/config.py",
        ]

    def test_a_glob_pins_every_match(self, repo):
        rels = _rels(_gather(repo, includes=["src/app/*.py"]))

        pinned = [g.rel_path for g in _gather(repo, includes=["src/app/*.py"])
                  if g.pinned]
        assert set(pinned) == {
            "src/app/pool.py", "src/app/client.py", "src/app/config.py"
        }
        assert len(rels) == len(set(rels))


class TestAForcedCutIsMarkedAndReported:
    """The one thing that may still cut a pin is the hard `--max-bytes`."""

    def test_the_cut_carries_a_marker(self, repo, capsys):
        gathered = _gather(repo, includes=["src/app/pool.py"], max_bytes=4_000)
        target = next(g for g in gathered if g.rel_path == "src/app/pool.py")

        assert target.truncated is True
        assert MARKER_PREFIX in (target.content or "")
        assert target.bytes <= 4_000

    def test_the_cut_is_reported_on_stderr(self, repo, capsys):
        _gather(repo, includes=["src/app/pool.py"], max_bytes=4_000)
        err = capsys.readouterr().err

        assert "--max-bytes" in err and "forced a cut" in err
        assert "src/app/pool.py" in err

    def test_every_pin_survives_a_ceiling_that_fits_none_of_them_whole(self, repo):
        """Fair shares, not first-come. Spending the ceiling greedily in order
        gives the last-named file zero bytes and no marker — the silent drop
        this issue is about, reintroduced inside the fix for it."""
        gathered = _gather(
            repo,
            includes=["src/app/pool.py", "notes/zzz_placeholder.py"],
            max_bytes=2_000,
        )
        rels = _rels(gathered)

        assert "src/app/pool.py" in rels
        assert "notes/zzz_placeholder.py" in rels

    def test_a_small_pin_stays_whole_when_a_large_one_overflows(self, repo):
        gathered = _gather(
            repo,
            includes=["src/app/pool.py", "notes/zzz_placeholder.py"],
            max_bytes=8_000,
        )
        small = next(
            g for g in gathered if g.rel_path == "notes/zzz_placeholder.py"
        )

        assert small.truncated is False
        assert small.content == (
            (repo / "notes" / "zzz_placeholder.py").read_text(encoding="utf-8")
        )

    def test_a_ceiling_below_the_pin_count_still_delivers_every_file(self, repo):
        """Nonsense config, but "never a silent drop" has no exceptions: each
        file arrives carrying a marker that says none of it fit."""
        gathered = _gather(
            repo,
            includes=["src/app/pool.py", "notes/zzz_placeholder.py"],
            max_bytes=1,
        )

        assert len(gathered) >= 2
        for g in gathered[:2]:
            assert g.truncated is True
            assert MARKER_PREFIX in (g.content or "")


class TestTheReportMatchesWhatHappened:
    """Every warning here replaced one built from a cheaper predicate that said
    something untrue. They are the whole point of the goal, so they get tests."""

    def test_a_cut_pin_is_not_described_as_whole(self, repo, capsys):
        _gather(repo, includes=["src/app/pool.py"], max_bytes=4_000)
        err = capsys.readouterr().err

        assert "pins 1 file(s) 0 whole, 1 cut and marked" in err
        assert "file(s) whole (" not in err

    def test_an_uncut_pin_is_described_as_whole(self, repo, capsys):
        _gather(repo, includes=["src/app/config.py"])
        err = capsys.readouterr().err

        assert "pins 1 file(s) whole" in err

    def test_a_ceiling_that_cannot_seat_the_markers_says_so(self, repo, capsys):
        """The ceiling bounds pinned CONTENT. Below one marker per named file
        the block runs over it, and the number stops meaning anything unless
        the overshoot is stated."""
        gathered = _gather(
            repo,
            includes=["src/app/*.py", "notes/*.py"],
            max_bytes=40,
        )
        err = capsys.readouterr().err

        assert sum(g.bytes for g in gathered) > 40
        assert "cannot seat" in err and "even as truncation markers" in err
        assert "Named files are never dropped" in err

    def test_no_overshoot_claim_when_the_ceiling_held(self, repo, capsys):
        """Guards the guard: the warning above must not be background noise."""
        _gather(repo, includes=["src/app/pool.py"], max_bytes=8_000)
        err = capsys.readouterr().err

        assert "cannot seat" not in err

    def test_a_byte_starved_scan_names_the_byte_cap_only(self, repo, capsys):
        pin = repo / "notes" / "zzz_placeholder.py"
        ceiling = len(pin.read_text(encoding="utf-8").encode("utf-8"))

        gathered = _gather(
            repo, includes=["notes/zzz_placeholder.py"], max_bytes=ceiling
        )
        err = capsys.readouterr().err

        assert len(gathered) == 1, "fixture stopped starving the scan"
        assert "byte budget" in err and f"--max-bytes={ceiling:,}" in err
        assert "file budget" not in err

    def test_a_file_starved_scan_names_the_file_cap_only(self, repo, capsys):
        gathered = _gather(
            repo, includes=["notes/zzz_placeholder.py"], max_files=1
        )
        err = capsys.readouterr().err

        assert len(gathered) == 1, "fixture stopped starving the scan"
        assert "file budget" in err and "--max-files=1" in err
        assert "byte budget" not in err

    def test_an_unstarved_scan_is_not_warned_about(self, repo, capsys):
        _gather(repo, includes=["notes/zzz_placeholder.py"])
        err = capsys.readouterr().err

        assert "contributed nothing" not in err

    def test_a_scan_with_nothing_to_add_blames_no_cap(self, tmp_path, capsys):
        """The failure this replaced: a 3-file repo with 135 bytes pinned
        against a 300,000-byte ceiling was told its budget was used up, which
        sent the operator to two settings that would have changed nothing."""
        (tmp_path / "only.py").write_text("ALPHA = 1\n", encoding="utf-8")
        gathered = _gather(tmp_path, includes=["only.py"])
        err = capsys.readouterr().err

        assert [g.rel_path for g in gathered] == ["only.py"]
        assert "found nothing to add" in err
        assert "--max-files" not in err and "--max-bytes" not in err


class TestTheLiteralPathRescue:
    """A named file over the walker's 512KB per-file limit is still named.

    That limit is a scan-cost guard, not a relevance judgement — the same class
    of rule as the score threshold and the chunk cap, which a pin already
    bypasses. Without the rescue the guarantee has an unstated cliff, and
    #198's own reported case was a 449KB file: 63KB under it.
    """

    @pytest.fixture
    def big_repo(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "huge.py").write_text(
            "# database pool connection timeout\n" + "x = 1\n" * 120_000,
            encoding="utf-8",
        )
        (tmp_path / "src" / "small.py").write_text(
            "def connection_pool_timeout(): pass\n", encoding="utf-8"
        )
        return tmp_path

    def test_the_walker_really_does_skip_it(self, big_repo):
        from neo.context_gatherer import iter_paths

        assert (big_repo / "src" / "huge.py").stat().st_size > 512_000
        rels = [rel for _abs, rel, _size in iter_paths(str(big_repo), [], [], None)]
        assert "src/huge.py" not in rels

    def test_a_literally_named_oversize_file_is_pinned_anyway(self, big_repo):
        gathered = _gather(big_repo, includes=["src/huge.py"], max_bytes=50_000)
        target = [g for g in gathered if g.rel_path == "src/huge.py"]

        assert target, "the named file was dropped by the walker's size limit"
        assert target[0].pinned is True
        assert target[0].truncated is True
        assert MARKER_PREFIX in (target[0].content or "")

    def test_the_rescue_does_not_admit_a_gitignored_file(self, tmp_path):
        """G1-inv outranks the rescue. Admitting an ignored file would trade a
        reported absence for an unreported presence."""
        (tmp_path / ".gitignore").write_text("secret.py\n", encoding="utf-8")
        (tmp_path / "secret.py").write_text("TOKEN = 1\n", encoding="utf-8")
        (tmp_path / "app.py").write_text("def pool(): pass\n", encoding="utf-8")

        rels = _rels(_gather(tmp_path, includes=["secret.py"]))
        assert "secret.py" not in rels

    def test_the_rescue_does_not_admit_a_file_in_an_ignored_directory(self, tmp_path):
        """`should_ignore` only tests the path it is handed; the walk gets
        ancestors for free by pruning, and the rescue does not."""
        (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "out.py").write_text("X = 1\n", encoding="utf-8")
        (tmp_path / "app.py").write_text("def pool(): pass\n", encoding="utf-8")

        rels = _rels(_gather(tmp_path, includes=["build/out.py"]))
        assert "build/out.py" not in rels

    def test_the_rescue_does_not_reach_outside_the_root(self, tmp_path):
        outside = tmp_path.parent / "outside_the_root.py"
        outside.write_text("X = 1\n", encoding="utf-8")
        (tmp_path / "app.py").write_text("def pool(): pass\n", encoding="utf-8")

        rels = _rels(_gather(tmp_path, includes=["../outside_the_root.py"]))
        assert not any("outside_the_root" in r for r in rels)

    def test_a_glob_gets_no_rescue_and_the_warning_says_so(self, big_repo, capsys):
        """Expanding a glob means walking, which is the cost the rescue avoids.
        The limit is real, so the diagnostic names it instead of listing four
        causes it never checked."""
        rels = _rels(_gather(big_repo, includes=["src/hu*.py"]))
        err = capsys.readouterr().err

        assert "src/huge.py" not in rels
        assert "matched no file" in err
        assert "512KB" in err


class TestCutToBytes:
    def test_content_that_fits_is_returned_untouched(self):
        text, cut = _cut_to_bytes("hello", 100)

        assert (text, cut) == ("hello", False)

    def test_the_whole_return_value_lands_inside_the_budget(self):
        text, cut = _cut_to_bytes("x" * 5_000, 500)

        assert cut is True
        assert len(text.encode("utf-8")) <= 500
        assert MARKER_PREFIX in text

    def test_a_budget_too_small_for_the_marker_yields_the_marker(self):
        """Not "" — an empty section is indistinguishable from an empty file."""
        text, cut = _cut_to_bytes("x" * 5_000, 0)

        assert cut is True
        assert MARKER_PREFIX in text
        assert "5000" in text

    def test_the_marker_counts_the_whole_file(self):
        text, _cut = _cut_to_bytes("y" * 900, 400)

        assert "of 900 characters not shown" in text

    def test_multibyte_content_is_bounded_in_bytes_not_characters(self):
        text, cut = _cut_to_bytes("é" * 4_000, 600)

        assert cut is True
        assert len(text.encode("utf-8")) <= 600


class TestResolveIncludes:
    def test_a_pattern_that_matches_nothing_is_reported(self, repo):
        matched, missed = resolve_includes(
            [("/r/src/app/pool.py", "src/app/pool.py", 10)], ["does/not/exist.py"]
        )

        assert matched == []
        assert missed == ["does/not/exist.py"]

    def test_the_miss_reaches_stderr(self, repo, capsys):
        _gather(repo, includes=["does/not/exist.py"])
        err = capsys.readouterr().err

        assert "--include" in err and "matched no file" in err

    def test_two_patterns_matching_one_file_pin_it_once(self):
        candidates = [("/r/a.py", "a.py", 1), ("/r/b.py", "b.py", 1)]
        matched, missed = resolve_includes(candidates, ["a.py", "*.py"])

        assert [rel for _abs, rel, _size in matched] == ["a.py", "b.py"]
        assert missed == []


class TestThePromptRendererHonoursThePin:
    """A guarantee the gatherer keeps and the renderer breaks is not one.

    3000 characters is roughly 40-75 lines, so routing a pinned file through
    the ordinary per-file cap would deliver LESS of it than the ~400-line
    windowing #198 complains about.
    """

    def test_a_pinned_file_is_not_cut_by_the_per_file_cap(self):
        body = "line\n" * 4_000
        sections, _banner, visible = NeoEngine._render_context_files(
            [ContextFile(path="big.py", content=body, pinned=True)]
        )

        assert MARKER_PREFIX not in sections[0]
        assert body in sections[0]
        assert visible[0].content == body

    def test_an_unpinned_file_of_the_same_size_still_is(self):
        """Guards the guard: the cap has to be live for the test above to mean
        anything."""
        body = "line\n" * 4_000
        sections, _banner, _visible = NeoEngine._render_context_files(
            [ContextFile(path="big.py", content=body)]
        )

        assert MARKER_PREFIX in sections[0]

    def test_the_file_cap_cannot_drop_a_pin(self):
        files = [
            ContextFile(path=f"scan{i}.py", content="x")
            for i in range(_MAX_CONTEXT_FILES + 4)
        ]
        files.append(ContextFile(path="pinned.py", content="y", pinned=True))
        sections, banner, _visible = NeoEngine._render_context_files(files)

        assert any("pinned.py" in s for s in sections)
        assert len(sections) == _MAX_CONTEXT_FILES
        assert f"{_MAX_CONTEXT_FILES} of {len(files)} files" in banner

    def test_the_incoming_order_survives(self):
        files = [
            ContextFile(path="a.py", content="a"),
            ContextFile(path="b.py", content="b", pinned=True),
            ContextFile(path="c.py", content="c"),
        ]
        sections, _banner, _visible = NeoEngine._render_context_files(files)

        assert [s.split(" ")[1] for s in sections] == ["a.py", "b.py", "c.py"]


class TestTheBannerTheModelReadsCountsFiles:
    """#197 on the surface that matters most.

    The gatherer hands the renderer one entry per window, so two chunks of one
    file arrived as two entries and the banner said "2 files" — to the model,
    inside the prompt, where no operator ever sees it to doubt it.
    """

    def test_two_chunks_of_one_file_are_not_two_files(self):
        files = [
            ContextFile(path="a.py", content="one", line_range=(1, 5)),
            ContextFile(path="a.py", content="two", line_range=(40, 60)),
        ]
        _sections, banner, _visible = NeoEngine._render_context_files(files)

        assert "2 chunks from 1 file" in banner
        assert "2 files" not in banner

    def test_unchunked_files_still_read_as_files(self):
        """The chunk count is stated only when it says something the file
        count does not."""
        files = [
            ContextFile(path="a.py", content="one"),
            ContextFile(path="b.py", content="two"),
        ]
        _sections, banner, _visible = NeoEngine._render_context_files(files)

        assert "2 files" in banner
        assert "chunk" not in banner

    def test_the_display_cap_is_reported_in_both_units(self):
        files = [
            ContextFile(path=f"f{i}.py", content="x", line_range=(1, 2))
            for i in range(_MAX_CONTEXT_FILES + 5)
        ]
        files.append(ContextFile(path="f0.py", content="y", line_range=(9, 10)))
        _sections, banner, _visible = NeoEngine._render_context_files(files)

        assert f"{_MAX_CONTEXT_FILES} of {len(files)} chunks" in banner
        assert "files" in banner
