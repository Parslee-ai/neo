"""The retrieval front door: one pipeline, four stages, whole-file delivery.

Unified-store Goal 8. `tests/test_include_guarantee.py` owns `--include`
(stage 2) and `tests/test_selection_invariants.py` owns the per-language
G-invariants; this file owns what Goal 8 added on top of them:

- **stage 1** — a path the prompt names is PINNED, not merely boosted
- **stage order** — stage 1 wins a tie with stage 2, and both precede the scan
- **delivery** — one entry per file, ever; the count of entries and the count
  of files are the same number by construction rather than by luck
- **budget** — `--max-bytes` is apportioned max-min fair, so one large file
  can no longer decide how many other files reach the model
- **stage 4** — `--semantic` is a hint that biases the catalog's weight and
  depth, not a fork to a second pipeline

Every assertion here is paired with a control wherever the new behaviour could
otherwise pass for the wrong reason: since delivery became whole-file, "the
file was not windowed" is the DEFAULT, so a test that only asserts it proves
nothing about the guarantee that produced it.
"""

import os

import pytest

from neo import context_gatherer
from neo.context_gatherer import (
    EXPLICIT_PATH_BOOST,
    SEMANTIC_HINT_DEPTH,
    SEMANTIC_HINT_WEIGHT,
    SEMANTIC_WEIGHT,
    GatherConfig,
    gather_context,
)
from neo.text_budget import MARKER_PREFIX

PROMPT_NAMING = (
    "Fix the connection pool timeout in src/app/pool.py when the database "
    "is busy"
)
PROMPT_PLAIN = "Fix the connection pool timeout when the database is busy"
# Three named paths against a one-file cap. Boosting cannot put three files
# through one slot; only a pin can, which is what makes the cap test bite.
PROMPT_NAMING_THREE = (
    "Fix the connection pool timeout across src/app/pool.py, "
    "src/app/client.py and src/app/config.py"
)

_BULK = "\n".join(
    f"    # database pool connection timeout retry line {i}" for i in range(400)
)


@pytest.fixture
def repo(tmp_path):
    """One large prompt-relevant file, two small ones, and one irrelevant.

    `notes/zzz_placeholder.py` shares no token with the prompt, sits a
    directory deep and carries the `archive` demotion, so it never becomes a
    candidate on its own — its presence can only come from a pin.
    """
    (tmp_path / "src" / "app").mkdir(parents=True)
    (tmp_path / "notes").mkdir()

    (tmp_path / "src" / "app" / "pool.py").write_text(
        "def connection_pool_timeout():\n"
        "    '''Database connection pool timeout handling.'''\n"
        f"{_BULK}\n",
        encoding="utf-8",
    )
    # See the matching note in tests/test_include_guarantee.py: a single big
    # file cannot separate pinned from unpinned at one ceiling, because any
    # ceiling that leaves the pin block room also leaves the scan's fair share
    # room. `PIN_BUDGET_SHARE` holds the pin block to half.
    (tmp_path / "src" / "app" / "worker.py").write_text(
        "def connection_pool_worker():\n"
        "    '''Database connection pool worker loop.'''\n"
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


def _gather(root, prompt=PROMPT_PLAIN, **overrides):
    config = GatherConfig(
        root=str(root),
        prompt=prompt,
        exts=None,
        includes=[],
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


class TestStageOneIsAPin:
    """A path the prompt spells out is delivered, not merely ranked first."""

    def test_a_named_path_is_marked_pinned(self, repo):
        gathered = _gather(repo, prompt=PROMPT_NAMING)
        named = next(g for g in gathered if g.rel_path == "src/app/pool.py")

        assert named.pinned is True

    def test_every_named_path_survives_a_file_cap_of_one(self, repo):
        """The claim `EXPLICIT_PATH_BOOST` could not make.

        The prompt names THREE files against a one-file cap, and that is the
        whole point: with one named file the boost alone still delivers it,
        so a single-file version of this test passes with stage 1 deleted and
        proves nothing. A verifier reverted the pin and watched exactly that
        happen. Boosting cannot put three files through one slot; pinning can,
        because a pin is not competing for the slot.
        """
        gathered = _gather(repo, prompt=PROMPT_NAMING_THREE, max_files=1)
        rels = _rels(gathered)

        for named in (
            "src/app/pool.py", "src/app/client.py", "src/app/config.py"
        ):
            assert named in rels, f"{named} was named and did not arrive: {rels}"

    def test_the_control_run_really_does_lose_them(self, repo):
        """Guards the guard: at `--max-files=1` exactly one file arrives when
        nothing names them, or the assertion above proves nothing."""
        gathered = _gather(repo, prompt=PROMPT_PLAIN, max_files=1)

        assert len(gathered) == 1
        assert all(not g.pinned for g in gathered)

    def test_a_named_path_arrives_whole_at_a_ceiling_that_would_cut_it(self, repo):
        """Same ceiling in both arms; only the naming differs."""
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        # TWICE the file: the pin block is held to half of `--max-bytes` while
        # anything else is eligible, so half seats the pin exactly while the
        # scan must split the same ceiling with `worker.py`.
        ceiling = 2 * len(source.encode("utf-8"))

        control = _gather(repo, prompt=PROMPT_PLAIN, max_bytes=ceiling)
        control_pool = next(
            g for g in control if g.rel_path == "src/app/pool.py"
        )
        assert control_pool.truncated is True, (
            "fixture stopped exercising the budget-forced excerpt"
        )

        named = _gather(repo, prompt=PROMPT_NAMING, max_bytes=ceiling)
        pool = next(g for g in named if g.rel_path == "src/app/pool.py")

        assert pool.truncated is False
        assert pool.start is None and pool.end is None
        assert pool.content == source

    def test_a_named_path_that_matches_nothing_warns(self, repo, capsys):
        _gather(repo, prompt="Fix the bug in src/app/nonexistent.py please")
        err = capsys.readouterr().err

        assert "prompt names a path but no scanned file matched" in err

    def test_the_boost_still_outranks_every_organic_signal(self):
        """The pin does not retire the boost — it covers the candidates the
        pin pool never held (a `--exts`-narrowed list, say)."""
        assert EXPLICIT_PATH_BOOST > 3.0 + 0.45 + 1.0 + 1.2


class TestStageOrder:
    """Stage 1 before stage 2, both before the scan."""

    def test_a_file_named_by_both_stages_is_delivered_once(self, repo):
        gathered = _gather(
            repo, prompt=PROMPT_NAMING, includes=["src/app/pool.py"]
        )

        assert _rels(gathered).count("src/app/pool.py") == 1

    def test_the_note_credits_the_stage_that_asked(self, repo, capsys):
        _gather(repo, prompt=PROMPT_NAMING)
        err = capsys.readouterr().err

        assert "1 named in the prompt" in err
        assert "from --include" not in err

    def test_both_stages_are_credited_separately(self, repo, capsys):
        _gather(
            repo,
            prompt=PROMPT_NAMING,
            includes=["notes/zzz_placeholder.py"],
        )
        err = capsys.readouterr().err

        assert "1 named in the prompt" in err
        assert "1 from --include" in err

    def test_pins_come_before_the_scan(self, repo):
        gathered = _gather(repo, prompt=PROMPT_NAMING)
        pinned_positions = [i for i, g in enumerate(gathered) if g.pinned]

        # Non-empty FIRST. `[] == list(range(0))` is True, so without this the
        # assertion below passes on a run with no pins at all — which is
        # exactly the state a reverted stage 1 leaves it in.
        assert pinned_positions, "nothing was pinned; the assertion is vacuous"
        assert len(gathered) > len(pinned_positions), "the scan added nothing"
        assert pinned_positions == list(range(len(pinned_positions)))


class TestOneEntryPerFile:
    """#197's divergence, closed at the source rather than reported."""

    def test_entries_and_files_are_the_same_number(self, repo):
        gathered = _gather(repo)

        assert len(gathered) == len({g.rel_path for g in gathered})

    def test_a_file_past_the_old_chunking_threshold_is_one_entry(self, repo):
        """15,000 characters used to split a file into up to two windows, each
        consuming a slot of the file cap for one file's worth of coverage."""
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        assert len(source) > 15_000, "fixture stopped clearing the old threshold"

        gathered = _gather(repo)

        assert _rels(gathered).count("src/app/pool.py") == 1

    def test_it_holds_when_the_budget_forces_an_excerpt(self, repo):
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        gathered = _gather(repo, max_bytes=len(source.encode("utf-8")))

        assert len(gathered) == len({g.rel_path for g in gathered})
        pool = next(g for g in gathered if g.rel_path == "src/app/pool.py")
        assert pool.start is not None and pool.end is not None


class TestTheExcerptTellsTheTruth:
    """A cut that understates itself is worse than no cut at all."""

    def test_the_marker_measures_against_the_whole_file(self, repo):
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        gathered = _gather(repo, max_bytes=len(source.encode("utf-8")))
        pool = next(g for g in gathered if g.rel_path == "src/app/pool.py")

        assert MARKER_PREFIX in (pool.content or "")
        assert f"of {len(source)} characters not shown" in (pool.content or "")

    def test_an_excerpt_is_flagged_truncated(self, repo):
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        gathered = _gather(repo, max_bytes=len(source.encode("utf-8")))
        pool = next(g for g in gathered if g.rel_path == "src/app/pool.py")

        assert pool.truncated is True

    def test_a_whole_file_carries_no_marker(self, repo):
        gathered = _gather(repo)
        client = next(g for g in gathered if g.rel_path == "src/app/client.py")

        assert client.truncated is False
        assert MARKER_PREFIX not in (client.content or "")


class TestTheBudgetIsApportionedNotSpent:
    """One large file no longer decides how many others reach the model."""

    def test_every_ranked_file_arrives_despite_one_large_one(self, repo):
        """The pool file is ~20 KB against three files of ~60-90 bytes. Filled
        greedily in rank order it takes the whole ceiling and the rest arrive
        or not depending on where they landed in the queue; apportioned, the
        small ones are funded in full and the pool file pays."""
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        gathered = _gather(repo, max_bytes=len(source.encode("utf-8")))
        by_rel = {g.rel_path: g for g in gathered}

        for rel in ("src/app/pool.py", "src/app/client.py", "src/app/config.py"):
            assert rel in by_rel, f"{rel} did not arrive: {sorted(by_rel)}"

        # PRESENT IS NOT ENOUGH. Under a greedy spend the small files still
        # "arrive" — as marker-only entries with a zero allowance — so an
        # `in rels` assertion passes with apportionment reverted. A verifier
        # swapped `apportion` for a greedy fill and watched this test stay
        # green. What apportionment buys is that the small files arrive
        # INTACT, so that is what is asserted.
        for rel in ("src/app/client.py", "src/app/config.py"):
            on_disk = (repo / rel).read_text(encoding="utf-8")
            assert (by_rel[rel].content or "") == on_disk, (
                f"{rel} arrived cut; the big file took its budget"
            )

    def test_the_small_files_are_not_the_ones_that_pay(self, repo):
        """Max-min fair means the LARGE files pay and the small ones do not.

        `pool.py` and `worker.py` are both ~21 KB and both legitimately pay at
        this ceiling; the assertion is about the three small files, which a
        greedy fill in rank order would have starved entirely.
        """
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        gathered = _gather(repo, max_bytes=len(source.encode("utf-8")))
        big = {"src/app/pool.py", "src/app/worker.py"}

        small = [g for g in gathered if g.rel_path not in big]
        assert small, "fixture stopped delivering any small file"
        for g in small:
            assert g.truncated is False, f"{g.rel_path} paid for the big files"
            on_disk = (repo / g.rel_path).read_text(encoding="utf-8")
            assert (g.content or "") == on_disk

    def test_the_ceiling_is_honoured(self, repo):
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        ceiling = len(source.encode("utf-8"))
        gathered = _gather(repo, max_bytes=ceiling)

        assert sum(g.bytes for g in gathered) <= ceiling


class TestThePinBlockCannotStarveTheScan:
    """Ruling 1 has two clauses, and funding pins to the last byte deletes one.

    Measured on the M2 battery before `PIN_BUDGET_SHARE` existed: the prompt
    naming `src/Parslee.M365.Api/Program.cs` (442,867 bytes) pinned it, spent
    299,959 of the 300,000-byte default ceiling, and delivered **one file**
    where main delivered 22. "The named files AND keep scanning" was satisfied
    by deleting the second half.
    """

    def test_the_scan_still_runs_when_a_named_file_would_eat_the_ceiling(
        self, repo
    ):
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        # Just over the named file's own size: without the reserve the pin
        # takes all but a few hundred bytes and `affordable` falls to zero.
        ceiling = len(source.encode("utf-8")) + 500

        gathered = _gather(repo, prompt=PROMPT_NAMING, max_bytes=ceiling)
        rels = _rels(gathered)

        assert "src/app/pool.py" in rels
        scan = [g for g in gathered if not g.pinned]
        assert scan, (
            "the pin block took the whole ceiling and the scan contributed "
            f"nothing: {rels}"
        )

    def test_the_reserve_is_what_makes_that_true(self, repo):
        """Guards the guard: prove the ceiling really is tight enough that an
        unreserved pin block would have consumed it.

        Without this the test above passes on any ceiling roomy enough for
        both, which is most of them.
        """
        from neo.context_gatherer import MIN_FILE_SHARE_BYTES, PIN_BUDGET_SHARE

        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        pin_bytes = len(source.encode("utf-8"))
        ceiling = pin_bytes + 500

        # What the scan would have been left with, unreserved.
        assert (ceiling - pin_bytes) // MIN_FILE_SHARE_BYTES == 0
        # And what it is left with under the reserve.
        assert (
            ceiling - int(ceiling * PIN_BUDGET_SHARE)
        ) // MIN_FILE_SHARE_BYTES >= 1

    def test_the_held_back_pin_is_cut_marked_and_announced(self, repo, capsys):
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        ceiling = len(source.encode("utf-8")) + 500

        gathered = _gather(repo, prompt=PROMPT_NAMING, max_bytes=ceiling)
        err = capsys.readouterr().err
        pool = next(g for g in gathered if g.rel_path == "src/app/pool.py")

        assert pool.truncated is True
        assert MARKER_PREFIX in (pool.content or "")
        assert "the pinned block is held to" in err.lower()

    def test_a_pin_that_fits_is_not_held_back(self, repo, capsys):
        """The reserve must not be background noise: at the default ceiling a
        pin of this size is nowhere near half, and nothing is said."""
        gathered = _gather(repo, prompt=PROMPT_NAMING)
        err = capsys.readouterr().err
        pool = next(g for g in gathered if g.rel_path == "src/app/pool.py")

        assert pool.truncated is False
        assert "held to" not in err

    def test_pins_take_the_whole_ceiling_when_nothing_else_is_eligible(
        self, tmp_path
    ):
        """The reserve funds a scan that can run. With no other candidate it
        would fund nothing, so it does not apply and the pin arrives whole."""
        (tmp_path / "only.py").write_text(
            "def connection_pool_timeout():\n" + "    # pool timeout\n" * 200,
            encoding="utf-8",
        )
        source = (tmp_path / "only.py").read_text(encoding="utf-8")

        gathered = _gather(
            tmp_path,
            prompt="Fix the timeout in only.py",
            max_bytes=len(source.encode("utf-8")),
        )

        assert len(gathered) == 1
        assert gathered[0].pinned is True
        assert gathered[0].truncated is False
        assert gathered[0].content == source


class TestEachCapIsChargedForWhatItRemoved:
    """The repo's own rule: never blame a cap for an absence it did not cause.

    The first version derived the file cap's verdict from the FULL candidate
    list and then let the byte cap cut further, so whenever both bound only the
    file cap was named. Reproduced live at `--max-files=30`: "pinned files
    filled the file budget (--max-files=30)" on a run with 29 of 30 slots free
    and the byte ceiling holding the count at one.
    """

    @staticmethod
    def _many(tmp_path, n=40):
        for i in range(n):
            (tmp_path / f"pool_{i:02d}.py").write_text(
                f"def connection_pool_timeout_{i}():\n"
                + f"    # database pool connection timeout retry {i}\n" * 4,
                encoding="utf-8",
            )
        return tmp_path

    def test_the_byte_cap_is_named_when_it_is_the_one_holding_the_count(
        self, tmp_path, capsys
    ):
        from neo.context_gatherer import MIN_FILE_SHARE_BYTES

        self._many(tmp_path)
        # Room for two files by bytes, and far more than two file slots.
        gathered = _gather(
            tmp_path, max_files=200, max_bytes=MIN_FILE_SHARE_BYTES * 2 + 100
        )
        err = capsys.readouterr().err

        assert len(gathered) == 2
        assert "--max-bytes" in err
        assert "file budget" not in err

    def test_raising_the_named_knob_actually_moves_the_number(self, tmp_path):
        """The property a mis-attributed warning breaks. If the run names
        `--max-bytes`, raising `--max-bytes` must change the outcome."""
        from neo.context_gatherer import MIN_FILE_SHARE_BYTES

        self._many(tmp_path)
        tight = _gather(
            tmp_path, max_files=200, max_bytes=MIN_FILE_SHARE_BYTES * 2 + 100
        )
        roomier = _gather(
            tmp_path, max_files=200, max_bytes=MIN_FILE_SHARE_BYTES * 20 + 100
        )

        assert len(roomier) > len(tight)

    def test_both_caps_are_reported_with_their_own_counts(
        self, tmp_path, capsys
    ):
        """Both can bind at once, and then both are named. Reporting one
        understates the loss and points at a knob that only half explains it."""
        from neo.context_gatherer import MIN_FILE_SHARE_BYTES

        self._many(tmp_path)
        gathered = _gather(
            tmp_path, max_files=10, max_bytes=MIN_FILE_SHARE_BYTES * 3 + 100
        )
        err = capsys.readouterr().err

        assert len(gathered) == 3
        assert "file budget" in err
        assert "--max-bytes" in err

    def test_only_the_file_cap_is_named_when_bytes_are_plentiful(
        self, tmp_path, capsys
    ):
        self._many(tmp_path)
        _gather(tmp_path, max_files=5, max_bytes=1_000_000)
        err = capsys.readouterr().err

        assert "file budget" in err
        assert "cannot fund them above" not in err


class _FakeChunk:
    def __init__(self, file_path, similarity):
        self.file_path = file_path
        self.similarity = similarity


class _FakeIndex:
    """A catalog that exists and answers, so stage 4 is reachable in a test.

    Records the `k` it was asked for, which is the only way to observe the
    depth half of the hint — weight shows up in the score, depth does not.
    """

    calls: list = []

    def __init__(self, root):
        self.root = root
        self.chunks = [object()]
        self.snapshot_path = type("P", (), {"exists": staticmethod(lambda: True)})()

    def retrieve(self, query, k=5):
        _FakeIndex.calls.append(k)
        # ABSOLUTE, as a real `CodeChunk.file_path` is — `_project_index_boost`
        # relpaths it against the root, and a fake that hands back a
        # repo-relative path exercises a normalization the live path never hits.
        return [_FakeChunk(os.path.join(self.root, "src/app/config.py"), 1.0)]


@pytest.fixture
def fake_catalog(monkeypatch):
    import neo.index.project_index as pi

    _FakeIndex.calls = []
    monkeypatch.setattr(pi, "ProjectIndex", _FakeIndex)
    return _FakeIndex


class TestSemanticIsAHintNotALane:
    """Stage 4 runs whenever the catalog exists; `--semantic` weighs it more."""

    def test_the_second_lane_is_gone(self):
        assert not hasattr(context_gatherer, "gather_context_semantic")

    def test_the_catalog_is_consulted_without_the_flag(self, repo, fake_catalog):
        boost = context_gatherer._project_index_boost(str(repo), PROMPT_PLAIN, k=10)

        assert boost == {"src/app/config.py": SEMANTIC_WEIGHT}

    def test_the_hint_raises_the_weight(self, repo, fake_catalog):
        boost = context_gatherer._project_index_boost(
            str(repo), PROMPT_PLAIN, k=10,
            weight=SEMANTIC_HINT_WEIGHT, hinted=True,
        )

        assert boost == {"src/app/config.py": SEMANTIC_HINT_WEIGHT}
        assert SEMANTIC_HINT_WEIGHT > SEMANTIC_WEIGHT

    def test_the_hint_raises_the_depth(self, repo, fake_catalog):
        _gather(repo, semantic=False)
        unhinted = fake_catalog.calls[-1]
        _gather(repo, semantic=True)
        hinted = fake_catalog.calls[-1]

        assert hinted == unhinted * SEMANTIC_HINT_DEPTH

    def test_the_flag_still_returns_files_with_no_catalog(self, repo):
        """No catalog is not a failure mode: `--semantic` degrades to the
        keyword ranking rather than to the empty result the old lane's
        fallback existed to avoid."""
        gathered = _gather(repo, semantic=True)

        assert "src/app/pool.py" in _rels(gathered)

    def test_a_missing_catalog_is_named_when_the_flag_asked_for_it(
        self, repo, capsys
    ):
        _gather(repo, semantic=True)
        err = capsys.readouterr().err

        assert "--semantic asked for the embedding catalog" in err

    def test_an_unflagged_run_gets_the_tip_not_the_warning(self, repo, capsys):
        _gather(repo, semantic=False)
        err = capsys.readouterr().err

        assert "--semantic asked for" not in err
        assert "neo --index" in err
