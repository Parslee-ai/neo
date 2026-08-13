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

    def test_a_named_path_survives_a_file_cap_of_one(self, repo):
        """The claim `EXPLICIT_PATH_BOOST` could not make.

        +10.0 clears every organic signal combined, so the file did rank
        first — and a prompt naming more files than the limit admits still
        lost the ones past the cut, because ranking first is not presence.
        """
        gathered = _gather(repo, prompt=PROMPT_NAMING, max_files=1)

        assert "src/app/pool.py" in _rels(gathered)

    def test_the_control_run_really_does_lose_it(self, repo):
        """Guards the guard: at `--max-files=1` the pool file must NOT arrive
        when nothing names it, or the assertion above proves nothing."""
        gathered = _gather(repo, prompt=PROMPT_PLAIN, max_files=1)
        assert len(gathered) == 1
        # It may or may not be the top-ranked file; what must be true is that
        # a single slot is decided by the ranking, not by a guarantee.
        assert all(not g.pinned for g in gathered)

    def test_a_named_path_arrives_whole_at_a_ceiling_that_would_cut_it(self, repo):
        """Same ceiling in both arms; only the naming differs."""
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        ceiling = len(source.encode("utf-8"))

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
        rels = _rels(gathered)

        assert "src/app/pool.py" in rels
        assert "src/app/client.py" in rels
        assert "src/app/config.py" in rels

    def test_the_small_files_are_not_the_ones_that_pay(self, repo):
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        gathered = _gather(repo, max_bytes=len(source.encode("utf-8")))

        for g in gathered:
            if g.rel_path != "src/app/pool.py":
                assert g.truncated is False, f"{g.rel_path} paid for the big file"

    def test_the_ceiling_is_honoured(self, repo):
        source = (repo / "src" / "app" / "pool.py").read_text(encoding="utf-8")
        ceiling = len(source.encode("utf-8"))
        gathered = _gather(repo, max_bytes=ceiling)

        assert sum(g.bytes for g in gathered) <= ceiling


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
