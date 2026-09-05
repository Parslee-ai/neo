"""Tests for context_gatherer relevance scoring — multi-language entry points."""

import pytest

from neo.context_gatherer import score_candidate


# Common args reused across tests. score_candidate is pure.
_EMPTY = set()
_ENTRY = {"main", "app", "server", "index", "login", "auth", "__init__"}


def _score(path: str, size: int = 1000) -> float:
    """Helper: score `path` with a no-keyword, no-git baseline."""
    return score_candidate(path, size, _EMPTY, _EMPTY, _ENTRY,
                           demote_tests=False)


class TestEntryPointBoost:
    """`entry_points` is the surviving name-based bonus; `main_impl_stems` is not.

    These files still outrank a plain one, but now for ONE reason (+0.2 for an
    entry-point basename) rather than two. The `main_impl_stems` whitelist that
    also gave them +0.4 is deleted: it existed to exempt seven hardcoded stems
    from a size penalty that was killing large central files, and with that
    penalty gone it has nothing to patch.
    """

    @pytest.mark.parametrize("special,plain", [
        ("main.py", "foo.py"),
        ("main.go", "foo.go"),
        ("Main.java", "Foo.java"),   # basename is lowercased first
        ("index.js", "widget.js"),
        ("app.ts", "widget.ts"),
        ("server.rs", "widget.rs"),
    ])
    def test_entry_point_basename_still_boosted(self, special, plain):
        assert _score(special) > _score(plain)

    @pytest.mark.parametrize("path", ["main.py", "index.js", "app.ts"])
    def test_the_gap_is_the_entry_point_bonus_alone(self, path):
        """Pins the SIZE of the gap, not just its direction — otherwise
        re-adding a second name-based bonus would pass unnoticed."""
        plain = "widget" + path[path.rindex("."):]
        assert _score(path) - _score(plain) == pytest.approx(0.2, abs=0.01)


class TestNoNeoBias:
    def test_neo_specific_filenames_no_longer_special(self):
        # Pre-refactor, files named like neo's internals (persistent.py,
        # structured_parser.py, etc.) got an unconditional +0.4. Now they
        # score the same as any other no-keyword Python file.
        assert _score("persistent.py") == _score("widget.py")
        assert _score("structured_parser.py") == _score("widget.py")

    def test_core_is_no_longer_special_either(self):
        """`core` was in the `main_impl_stems` whitelist and nowhere else — no
        entry-point prefix, no other mechanism. With the whitelist gone it
        scores like any other file, which is the intended change: a central
        file should be identified by its content, not by whether someone
        thought of its name."""
        assert _score("core.py") == _score("widget.py")


class TestStemEqualityNotSubstring:
    # Substring matching would have flagged any of these as main-impl
    # via 'lib' or 'index' or 'app'. Stem-equality keeps them clean.

    def test_library_not_boosted(self):
        assert _score("library.py") == _score("widget.py")

    def test_accessibility_not_boosted(self):
        assert _score("accessibility.tsx") == _score("widget.tsx")

    def test_reindex_not_boosted(self):
        assert _score("reindex.py") == _score("widget.py")

    def test_application_gets_only_the_entry_point_bonus(self):
        """`main_impl_stems` is gone; `entry_points` is not.

        The whitelist gave +0.4 to seven hardcoded stems AND exempted them
        from the size penalty. It existed only because that penalty was
        killing large central files — it rescued `engine.py` at -0.13 and left
        `store.py` at -1.62, a 12x disparity between two files of near-identical
        size decided by whether someone had thought of the name. With the
        penalty gone it has nothing to patch, and content BM25 identifies a
        central file by what is in it.
        """
        gap = _score("Application.java") - _score("Widget.java")
        assert gap == pytest.approx(0.2, abs=0.01)  # entry_point only

class TestSizeDoesNotAffectScore:
    """Size is not scored. It used to be the DOMINANT term, and backwards.

    `score -= 0.01 * size_kb` once over 10 KB, uncapped, against a realistic
    positive signal of +0.6 to +2.1 — so a file with one keyword hit became
    unrankable above 60 KB. Measured, `src/neo/memory/store.py` scored 0.000
    and ranked 200th of 284 for "fix the fact store supersession threshold",
    because it is 162 KB. Ground truth ran 31-177 KB against a corpus median
    of 10 KB: central files are large *because* they are central.

    BugLocator's rVSM (Zhou et al., ICSE 2012) ranks LARGER files HIGHER for
    exactly this task. So the sign was wrong, not the magnitude, and BM25's
    `b` handles the real concern with bounded, corpus-derived normalization.

    The tests replaced here asserted the penalty (`assert big < small`). They
    pinned the defect, which is part of why it survived a rewrite of
    everything around it.
    """

    _TOKENS = {"widget"}

    @pytest.mark.parametrize("path", ["widget.py", "main.py", "src/deep/widget.py"])
    def test_score_is_independent_of_size(self, path):
        tiny = score_candidate(path, 1_000, self._TOKENS, _EMPTY, _ENTRY,
                               demote_tests=False)
        huge = score_candidate(path, 500 * 1024, self._TOKENS, _EMPTY, _ENTRY,
                               demote_tests=False)
        assert tiny == huge, "size is back in the scoring function"

    def test_a_large_relevant_file_outranks_a_small_irrelevant_one(self):
        """The property the old scorer made impossible."""
        big_relevant = score_candidate(
            "src/store.py", 162 * 1024, self._TOKENS, _EMPTY, _ENTRY,
            demote_tests=False, content_relevance=1.0,
        )
        small_irrelevant = score_candidate(
            "src/tiny.py", 1_000, self._TOKENS, _EMPTY, _ENTRY,
            demote_tests=False, content_relevance=0.0,
        )
        assert big_relevant > small_irrelevant

    def test_content_relevance_outweighs_every_tie_breaker_combined(self):
        """Content decides the ranking, not the bonuses around it.

        docs 0.8 + git recency 0.3 + entry point 0.2 + filename 0.45 = 1.75,
        against 3.0 for a full content match. A file the prompt is *about*
        must beat a file that merely looks promising.
        """
        content_only = score_candidate(
            "src/obscurely_named.py", 5_000, set(), _EMPTY, _ENTRY,
            demote_tests=False, content_relevance=1.0,
        )
        every_bonus = score_candidate(
            "docs/main.py", 5_000, {"docs", "main"}, {"docs/main.py"}, _ENTRY,
            demote_tests=False, content_relevance=0.0,
        )
        assert content_only > every_bonus


class TestExplicitPathPinning:
    """A path the prompt names outright must reach the context bundle.

    Regression: asked to fix a named function in `src/neo/subcommands.py`, neo
    ranked that file 163rd of 296 candidates (score 0.838) while an unrelated
    file ranked 1st — filename-token overlap caps at 3 hits and an 86KB file
    takes a large size penalty. The file never entered the context, the model
    answered that the function body was not provided and emitted no patch, and a
    suggestion carrying no diff text can never be git-verified or learned from.
    """

    def test_named_path_outranks_an_organically_strong_file(self):
        from neo.context_gatherer import (
            EXPLICIT_PATH_BOOST, extract_explicit_paths, matches_explicit_path,
        )
        explicit = extract_explicit_paths(
            "Fix _classify_suggestion in src/neo/subcommands.py please")
        assert matches_explicit_path("src/neo/subcommands.py", explicit)
        # The boost must exceed every organic signal combined: filename overlap
        # caps at +1.8, the two re-rank boosts at +1.0 and +1.2.
        assert EXPLICIT_PATH_BOOST > 1.8 + 1.0 + 1.2

    def test_bare_filename_matches_but_a_neighbour_does_not(self):
        from neo.context_gatherer import extract_explicit_paths, matches_explicit_path
        explicit = extract_explicit_paths("look at subcommands.py")
        assert matches_explicit_path("src/neo/subcommands.py", explicit)
        # Matching the test file would re-bury the named file under its own
        # neighbours, which is the failure this exists to fix.
        assert not matches_explicit_path("tests/test_subcommands.py", explicit)

    def test_prose_and_versions_are_not_paths(self):
        from neo.context_gatherer import matches_explicit_path, extract_explicit_paths
        explicit = extract_explicit_paths(
            "Upgrade to 0.42.0, e.g. the parser, cf. the docs")
        # Junk tokens may be extracted, but nothing real can match them.
        for path in ("src/neo/engine.py", "README.md", "pyproject.toml"):
            assert not matches_explicit_path(path, explicit)

    def test_empty_prompt_pins_nothing(self):
        from neo.context_gatherer import extract_explicit_paths, matches_explicit_path
        assert extract_explicit_paths("") == set()
        assert not matches_explicit_path("src/neo/engine.py", set())


class TestSelectChunksRelevance:
    """`select_chunks` used to take the first five matching lines in FILE ORDER.

    Matching is a substring test and `extract_prompt_tokens` emits every 3+
    character word, so "in" matched `int`/`using`/`point` and virtually every
    line qualified. "First five matches" therefore meant "lines 1-5" for any
    prompt against any large file — the module docstring and imports — emitted
    as two overlapping near-duplicate windows that consumed both slots of the
    per-file cap.
    """

    @staticmethod
    def _big_file(target_line: str, target_at: int = 500, total: int = 900) -> str:
        lines = ["import os", "import sys", '"""Module docstring."""']
        lines += [f"    value_{i} = compute(i)  # padding using a point" for i in range(total)]
        lines[target_at] = target_line
        return "\n".join(lines)

    def test_window_centers_on_the_named_symbol_with_a_realistic_token_set(self):
        """Uses the REAL tokenizer rather than a hand-picked clean set. That
        distinction is the whole bug: with only discriminative tokens the old
        file-order code also found the symbol, so a clean set proves nothing.
        `extract_prompt_tokens` always emits the short words too."""
        from neo.context_gatherer import extract_prompt_tokens, select_chunks
        content = self._big_file("def _classify_suggestion(file_path, root):")
        tokens = extract_prompt_tokens(
            "Fix the bug in _classify_suggestion in src/neo/subcommands.py")
        assert "in" in tokens, "tokenizer no longer emits short noise words"
        chunks = select_chunks(content, tokens)
        assert chunks, "expected at least one window"
        top, start, _end = chunks[0]
        assert "def _classify_suggestion" in top
        assert start > 100, f"window started at {start} — that is the file header again"

    def test_short_noise_tokens_do_not_drag_windows_to_the_header(self):
        from neo.context_gatherer import select_chunks
        content = self._big_file("def target_function(arg):")
        # "in"/"is"/"py" are the tokens that matched nearly every line before.
        chunks = select_chunks(content, {"in", "is", "py", "target_function"})
        assert "def target_function" in chunks[0][0]

    def test_overlapping_windows_are_merged(self):
        from neo.context_gatherer import select_chunks
        # Must exceed max_chunk_bytes (12_000) or the whole file comes back as a
        # single chunk and the assertion below is vacuous.
        lines = [f"line {i} filler padding to grow the file" for i in range(1200)]
        lines[300] = "needle alpha here"
        lines[303] = "needle alpha again"
        content = "\n".join(lines)
        assert len(content) > 12_000, "fixture too small to exercise chunking"
        chunks = select_chunks(content, {"needle", "alpha"})
        ranges = [(s, e) for _c, s, e in chunks]
        assert len(ranges) == 1, f"centers 3 lines apart must merge, got {ranges}"
        # And two centers far apart must NOT merge.
        lines[900] = "needle alpha distant"
        far = select_chunks("\n".join(lines), {"needle", "alpha"})
        far_ranges = sorted((s, e) for _c, s, e in far)
        assert len(far_ranges) == 2
        for i, (s1, e1) in enumerate(far_ranges):
            for s2, e2 in far_ranges[i + 1:]:
                assert e1 < s2 or e2 < s1, f"overlapping windows {(s1,e1)} {(s2,e2)}"

    def test_small_file_returned_whole(self):
        from neo.context_gatherer import select_chunks
        content = "def f():\n    return 1\n"
        assert select_chunks(content, {"anything"}) == [(content, 1, 2)]

    def test_no_match_falls_back_to_the_header(self):
        from neo.context_gatherer import select_chunks
        content = "\n".join(f"line {i}" for i in range(2000))
        chunks = select_chunks(content, {"zzzznomatchzzzz"})
        assert len(chunks) == 1 and chunks[0][1] == 1


class TestExplicitPathSpellings:
    """Absolute paths are the highest-value input and were silently unmatched:
    `str.strip("./")` strips a character SET from both ends, so a leading "/"
    was eaten, and containment was tested in one direction only. Tracebacks,
    IDE copy-path and neo's own suggestion output all emit absolute paths."""

    @staticmethod
    def _explicit(text):
        from neo.context_gatherer import extract_explicit_paths
        return extract_explicit_paths(text)

    def test_absolute_path_matches_repo_relative_candidate(self):
        from neo.context_gatherer import matches_explicit_path
        explicit = self._explicit(
            "fix the bug in /Users/me/repo/src/neo/subcommands.py now")
        assert matches_explicit_path("src/neo/subcommands.py", explicit)

    def test_windows_spelling_normalizes(self):
        from neo.context_gatherer import matches_explicit_path
        explicit = self._explicit(r"look at src\neo\subcommands.py")
        assert matches_explicit_path("src/neo/subcommands.py", explicit)

    def test_neighbour_still_excluded_under_both_directions(self):
        from neo.context_gatherer import matches_explicit_path
        for text in ("see subcommands.py",
                     "see /Users/me/repo/src/neo/subcommands.py"):
            assert not matches_explicit_path("tests/test_subcommands.py",
                                             self._explicit(text))

    def test_empty_rel_path_never_matches(self):
        from neo.context_gatherer import matches_explicit_path
        assert not matches_explicit_path("", self._explicit("src/neo/x.py"))


class TestSelectChunksBudget:
    def test_chain_merge_cannot_blow_the_byte_cap(self):
        """Unbounded merging chained ~20 centers into one window measured at
        8.3x max_chunk_bytes. The caller admits a chunk all-or-nothing, so an
        oversized chunk that no longer fits the global budget is DROPPED and the
        named file contributes nothing — the original bug, new door."""
        from neo.context_gatherer import select_chunks
        lines = ["x" * 55 for _ in range(2000)]
        for i in range(0, 1600, 75):
            lines[i] = "needle_token_here alpha"
        chunks = select_chunks("\n".join(lines), {"needle_token_here", "alpha"})
        assert chunks
        for chunk, _s, _e in chunks:
            assert len(chunk) <= 12_000, f"chunk of {len(chunk)} exceeds the cap"

    def test_truncation_keeps_the_line_that_earned_the_window(self):
        from neo.context_gatherer import select_chunks
        lines = ["y" * 80 for _ in range(600)]
        lines[300] = "the_unique_target_symbol lives here"
        chunks = select_chunks("\n".join(lines), {"the_unique_target_symbol"})
        assert "the_unique_target_symbol" in chunks[0][0]

    def test_emitted_ranges_are_one_based_and_round_trip(self):
        from neo.context_gatherer import select_chunks
        lines = [f"line {i} padding text to grow the file" for i in range(1500)]
        lines[400] = "alpha_needle one"
        lines[1100] = "alpha_needle two"
        content = "\n".join(lines)
        for chunk, start, end in select_chunks(content, {"alpha_needle"}):
            assert "\n".join(lines[start - 1:end]) == chunk

    def test_header_does_not_win_when_a_better_region_exists(self):
        """Direct regression for the original defect: keyword-bearing module
        docstring must not beat the function body."""
        from neo.context_gatherer import extract_prompt_tokens, select_chunks
        lines = ['"""Handlers for CLI subcommands and classification."""']
        lines += [f"    filler_{i} = compute(i) using a point in time" for i in range(900)]
        lines[500] = "def _classify_suggestion(file_path, root):"
        chunks = select_chunks("\n".join(lines), extract_prompt_tokens(
            "fix _classify_suggestion in src/neo/subcommands.py"))
        assert "def _classify_suggestion" in chunks[0][0]
        assert chunks[0][1] > 100

    def test_single_match_at_each_boundary_clamps(self):
        from neo.context_gatherer import select_chunks
        for at in (0, 1499):
            lines = [f"line {i} padding text to grow the file" for i in range(1500)]
            lines[at] = "edge_needle here"
            chunks = select_chunks("\n".join(lines), {"edge_needle"})
            assert chunks and "edge_needle" in chunks[0][0]
            assert chunks[0][1] >= 1 and chunks[0][2] <= 1500

    def test_empty_token_set_falls_back_to_header(self):
        from neo.context_gatherer import select_chunks
        content = "\n".join(f"line {i} padding" for i in range(2000))
        chunks = select_chunks(content, set())
        assert len(chunks) == 1 and chunks[0][1] == 1


class TestRealPathPredicate:
    """The not-found WARNING fires on the no-match case, so every prose token
    `_PATH_LIKE` over-captures becomes a bogus "check spelling" for something
    the user never claimed was a path.

    Observed on a real run: "ASP.NET Core 10" produced
        Warning: prompt names a path but no scanned file matched (asp.net)
    """

    @pytest.mark.parametrize("token", [
        "asp.net",        # "ASP.NET Core 10"
        "e.g",            # prose
        "0.42.0",         # a version
        "vs.the",         # sloppy punctuation
        "etc.and",
    ])
    def test_prose_is_not_warnable(self, token):
        from neo.context_gatherer import _looks_like_a_real_path
        assert not _looks_like_a_real_path(token), (
            f"{token!r} would produce a bogus not-found warning"
        )

    @pytest.mark.parametrize("token", [
        "src/neo/context_gatherer.py",
        "context_gatherer.py",
        "README.md",
        "pyproject.toml",
        "web/src/App.tsx",
        "Program.cs",
        "some/dir/without-extension",   # a separator is proof enough
        "config.yaml",
        # Genuinely ambiguous, and deliberately resolved as "warn". `js` is a
        # real extension and a file named node.js is entirely plausible, so
        # "Node.js" cannot be told from a filename BY VALUE. The alternative is
        # a denylist of technology names, which never ends and would suppress
        # warnings for real files. A rare spurious warning about a token that
        # genuinely looks like a filename is the cheaper error.
        "node.js",
    ])
    def test_genuine_paths_are_warnable(self, token):
        from neo.context_gatherer import _looks_like_a_real_path
        assert _looks_like_a_real_path(token), (
            f"{token!r} is a real path reference and must still warn"
        )

    @pytest.mark.parametrize("token,expected", [
        ("schema.graphql", "schema.graphql"),
        ("notes.markdown", "notes.markdown"),
        ("Dockerfile.build", "dockerfile.build"),
    ])
    def test_long_extensions_are_not_truncated(self, token, expected):
        """At {0,5} the pattern matched a long extension but TRUNCATED it:
        "schema.graphql" came out as "schema.graphq". A truncated token
        matches no real file, so the named file lost EXPLICIT_PATH_BOOST, and
        it was not a known extension either, so the not-found warning stayed
        quiet — the file the user explicitly named quietly did not arrive."""
        from neo.context_gatherer import extract_explicit_paths
        assert expected in extract_explicit_paths(token)

    def test_predicate_is_case_insensitive(self):
        """The lowering was done by the caller and stated nowhere, so a direct
        call with "README.MD" returned False."""
        from neo.context_gatherer import _looks_like_a_real_path
        assert _looks_like_a_real_path("README.MD")
        assert _looks_like_a_real_path("Config.YAML")

    def test_extraction_itself_is_unchanged(self):
        """The predicate gates the WARNING only. The candidate set stays loose
        so EXPLICIT_PATH_BOOST keeps its reach — a token matching nothing costs
        nothing there, which is what the regex comment actually promises."""
        from neo.context_gatherer import extract_explicit_paths
        found = extract_explicit_paths("we use ASP.NET and src/neo/engine.py")
        assert "asp.net" in found
        assert "src/neo/engine.py" in found




class TestDocBonusRequiresBroadPromptOrContent:
    """+0.8 for any path containing "design"/"docs/"/"readme", with NO content
    signal, is four times MIN_SCORE_THRESHOLD and above the relative floor. It
    admitted documentation on filename alone — which is why a review prompt
    naming controllers and entities came back holding design documents while
    the implementation files it asked about were crowded out.

    Assertions are placed at the ADMISSION BOUNDARY, because that is where the
    bonus does its damage. Asserting merely that a matching source file
    outranks a content-free doc passes with the bonus ungated (1.25 vs 0.80)
    and proves nothing — an earlier version of this class did exactly that and
    survived mutation.
    """

    SPECIFIC = {"controller", "entity", "tenant", "isolation",
                "audit", "migration", "constraint"}

    @staticmethod
    def _score(path, prompt_tokens, content_relevance):
        return score_candidate(
            path, 1000, prompt_tokens, _EMPTY, _ENTRY,
            demote_tests=False, content_relevance=content_relevance,
        )

    @staticmethod
    def _floor(top_organic):
        from neo.context_gatherer import MIN_SCORE_THRESHOLD, RELATIVE_SCORE_FLOOR
        return max(MIN_SCORE_THRESHOLD, top_organic * RELATIVE_SCORE_FLOOR)

    def test_content_free_doc_falls_below_the_admission_floor(self):
        """The boundary. Ungated this scores 0.80 against a floor of ~0.48 and
        is admitted on its filename; gated it scores 0.0 and is not."""
        top_organic = self._score("api/src/TenantController.cs", self.SPECIFIC, 1.0)
        floor = self._floor(top_organic)
        doc = self._score("docs/plans/2026-design-notes.md", self.SPECIFIC, 0.0)

        assert doc < floor, (
            f"content-free design doc scores {doc:.2f} against floor "
            f"{floor:.2f} — admitted on its filename alone"
        )

    def test_broad_prompt_still_gets_the_full_doc_bonus(self):
        """The bonus's stated purpose. A prompt too short to name a target has
        documentation as its best available answer."""
        from neo.context_gatherer import BROAD_PROMPT_TOKENS

        broad = {"what", "is", "this"}
        assert len(broad) <= BROAD_PROMPT_TOKENS

        with_doc = self._score("README.md", broad, 0.0)
        without = self._score("srcfile.py", broad, 0.0)

        assert with_doc - without >= 0.79, (
            f"broad prompt lost the documentation boost "
            f"({with_doc:.2f} vs {without:.2f})"
        )

    def test_matching_doc_still_clears_the_floor(self):
        """Scaled, not dropped: a design doc that DOES match is still worth
        surfacing."""
        top_organic = self._score("api/src/TenantController.cs", self.SPECIFIC, 1.0)
        floor = self._floor(top_organic)
        matching = self._score("docs/tenant-isolation.md", self.SPECIFIC, 0.9)

        assert matching > floor, (
            f"a strongly matching doc ({matching:.2f}) was cut by the floor "
            f"({floor:.2f})"
        )


class TestRelativeFloorAdmissionBoundary:
    """Asserted through gather_context, on what it actually SELECTS.

    Two earlier attempts at this class were vacuous and both survived
    mutation. The first reimplemented `max(absolute, top * relative)` in the
    test file and asserted against the copy. The second asserted
    `len(selected) < 41` under `max_files=30` — true by construction. A third
    trap is recomputing the floor from the constants: the production bug would
    be in how gather_context USES them, and a test that derives the floor
    itself cannot see that.

    So: build a repo, run the real gatherer, count what comes back.
    Measured on this fixture — with the relative floor: 1 file. Without it: 26,
    of which 25 are content-free path-token matches.
    """

    PROMPT = (
        "how does the widget registry class register a widget and append "
        "it to the widget list"
    )

    @staticmethod
    def _build(tmp_path, n_helpers=25):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "widget_registry.py").write_text(
            "class WidgetRegistry:\n"
            "    def register_widget(self, widget):\n"
            "        self.widget_list.append(widget)\n" * 10
        )
        # Path-token matches with NO content signal: three prompt tokens in the
        # filename, nothing relevant inside. Exactly what cleared 0.2.
        for i in range(n_helpers):
            (tmp_path / "src" / f"widget_registry_list_helper_{i}.py").write_text(
                f"# note {i}\nX = {i}\n"
            )

    def _gather(self, tmp_path):
        from neo.context_gatherer import GatherConfig, gather_context

        # max_files well above the file count: the FLOOR must be the only
        # thing that can exclude anything, or the cap masks the result.
        return gather_context(GatherConfig(
            root=str(tmp_path), prompt=self.PROMPT, exts=None, includes=[],
            excludes=[], max_files=100, use_git=False,
        ))

    def test_path_only_matches_are_not_admitted(self, tmp_path):
        self._build(tmp_path)
        selected = [
            getattr(f, "rel_path", getattr(f, "path", ""))
            for f in self._gather(tmp_path)
        ]
        helpers = [p for p in selected if "helper" in p]

        assert any("widget_registry.py" in p for p in selected), (
            "the file that actually matches was dropped"
        )
        assert not helpers, (
            f"{len(helpers)} content-free path-token matches admitted: "
            f"{helpers[:3]}"
        )

    def test_the_matching_file_survives_the_floor(self, tmp_path):
        """The floor must not be so aggressive it cuts the real answer."""
        self._build(tmp_path)
        selected = [
            getattr(f, "rel_path", getattr(f, "path", ""))
            for f in self._gather(tmp_path)
        ]
        assert len(selected) >= 1
        assert any("widget_registry.py" in p for p in selected)


class TestGitFailureIsNotSilence:
    """An empty recent-files set must mean "git said nothing", never "git
    broke".

    Only the rev-parse probe passed check=True; the three calls that actually
    produce data ignored their exit status. A git exiting 128 — corrupt index,
    detected dubious ownership, a held lock — writes to stderr and nothing to
    stdout, so the parse loops saw no lines and the function returned an empty
    set, byte-identical to a clean repo.

    The signal is no longer only a tie-breaker either: recency feeds
    top_organic_score, so a silent zero changes WHICH files are admitted.
    """

    def _repo(self, tmp_path):
        import subprocess
        subprocess.run(["git", "init", "-q", "."], cwd=tmp_path, check=True)
        (tmp_path / "a.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "a.py"], cwd=tmp_path, check=True)
        return tmp_path

    def test_healthy_repo_reports_its_changes(self, tmp_path):
        from neo.context_gatherer import get_git_recent_files
        assert "a.py" in get_git_recent_files(str(self._repo(tmp_path)))

    def test_corrupt_index_is_reported_not_swallowed(self, tmp_path, capsys):
        from neo.context_gatherer import get_git_recent_files

        repo = self._repo(tmp_path)
        (repo / ".git" / "index").write_bytes(b"garbage")

        result = get_git_recent_files(str(repo))
        note = capsys.readouterr().err

        assert result == set()
        assert "git status" in note and "failed" in note, (
            f"a broken repo looked exactly like a clean one; notes were: {note!r}"
        )

    def test_every_git_query_is_bounded(self):
        """Four git invocations per gather, none of which was bounded. A held
        lock or a wedged credential helper could hang the run before a single
        file was scored."""
        import subprocess
        from unittest.mock import patch

        from neo import context_gatherer as cg

        seen = []

        def fake_run(cmd, **kwargs):
            seen.append(kwargs.get("timeout"))
            raise subprocess.TimeoutExpired(cmd, 10)

        with patch.object(cg.subprocess, "run", fake_run):
            cg.get_git_recent_files(".")

        assert seen, "no git query was attempted"
        assert all(t == cg.GIT_QUERY_TIMEOUT_SECONDS for t in seen), seen

    def test_timeout_is_reported(self, capsys):
        import subprocess
        from unittest.mock import patch

        from neo import context_gatherer as cg

        with patch.object(
            cg.subprocess, "run",
            lambda cmd, **kw: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(cmd, 10)
            ),
        ):
            assert cg.get_git_recent_files(".") == set()

        assert "timed out" in capsys.readouterr().err


class TestSystemicParserFailureIsReported:
    """A file this parser cannot read and the parser being absent are
    different events. The bare except treated them identically, so a broken
    tree-sitter removed the symbol signal from every file in the repo in
    silence."""

    def test_parser_init_failure_is_announced_once(self, capsys):
        from unittest.mock import patch

        from neo import context_gatherer as cg

        cache = {}
        with patch(
            "neo.index.language_parser.TreeSitterParser",
            side_effect=RuntimeError("no grammars"),
        ):
            for _ in range(5):
                assert cg._symbol_score("x.py", {"widget"}, cache) == 0.0

        note = capsys.readouterr().err
        assert "tree-sitter parser unavailable" in note
        assert note.count("tree-sitter parser unavailable") == 1, (
            "reported once per FILE instead of once per gather"
        )


class TestRetrievalBoostFailuresAreReported:
    """A feature the user never enabled may fail silently. One they DID enable
    and that is now contributing nothing must say so.

    The "no index" case is handled explicitly earlier in _project_index_boost and
    returns before the except, so what reaches the handler is a catalog that
    EXISTS and broke — a corrupt snapshot, faiss failing to import, retrieve()
    raising. Same distinction as an absent tree-sitter parser.
    """

    def test_semantic_rerank_failure_is_announced(self, capsys, tmp_path):
        from unittest.mock import patch

        from neo import context_gatherer as cg

        with patch(
            "neo.index.project_index.ProjectIndex",
            side_effect=RuntimeError("corrupt snapshot"),
        ):
            out = cg._project_index_boost(str(tmp_path), "widget registry", k=5)

        assert out == {}
        note = capsys.readouterr().err
        assert "semantic re-rank failed" in note, note

    def test_history_boost_failure_is_announced(self, capsys, tmp_path):
        from unittest.mock import patch

        from neo import context_gatherer as cg

        with patch(
            "neo.memory.store.FactStore",
            side_effect=RuntimeError("store corrupt"),
        ):
            out = cg._history_boost(str(tmp_path), "widget registry", k=5)

        assert out == {}
        note = capsys.readouterr().err
        assert "history boost failed" in note, note
