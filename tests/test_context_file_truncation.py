"""Tests for how the reasoning prompt renders REPOSITORY CONTEXT files.

Regression coverage for the failure where every context file was clipped to
its first 3000 characters with no marker, while the banner reported the
pre-truncation byte count. The model reasoned about the top of a file
believing it had the whole thing, and the operator was told it had been sent
several times more than it had.
"""

import re

from neo.engine import (
    _CONTEXT_FILE_CHARS,
    _IMPORTANT_FILE_CHARS,
    _MAX_CONTEXT_FILES,
    NeoEngine,
)
from neo.models import ContextFile


def _file(path: str, size: int) -> ContextFile:
    """A context file of exactly `size` characters."""
    return ContextFile(path=path, content="x" * size)


def _render(files):
    """Returns (sections, banner) — the third element is covered separately."""
    sections, banner, _visible = NeoEngine._render_context_files(files)
    return sections, banner


class TestTruncationIsMarked:
    def test_oversized_file_carries_a_marker(self):
        """The assertion the issue asks for: a clipped file says it was clipped.

        Without this the model cannot tell "the file ends here" from "the
        file was cut", so it answers questions about absence from a fragment.
        """
        sections, _ = _render([_file("src/Service.cs", _CONTEXT_FILE_CHARS + 5000)])

        assert "[truncated:" in sections[0]

    def test_marker_states_how_much_was_dropped(self):
        sections, _ = _render([_file("src/Service.cs", _CONTEXT_FILE_CHARS + 5000)])

        assert "5000 of 8000 characters not shown" in sections[0]

    def test_file_within_the_cap_is_not_marked(self):
        """A file that fits must not claim to have been cut."""
        sections, _ = _render([_file("src/Small.cs", 100)])

        assert "[truncated" not in sections[0]

    def test_content_is_still_capped(self):
        """The marker is added to the cut, not instead of it."""
        sections, _ = _render([_file("src/Service.cs", 50_000)])

        body = sections[0].split("---\n", 1)[1]
        payload = body.split("\n... [truncated:", 1)[0]
        assert len(payload) == _CONTEXT_FILE_CHARS

    def test_important_files_get_the_larger_cap(self):
        """README/CLAUDE.md/architecture keep their 8000-char allowance."""
        sections, _ = _render([_file("docs/README.md", 50_000)])

        payload = sections[0].split("---\n", 1)[1].split("\n... [truncated:", 1)[0]
        assert len(payload) == _IMPORTANT_FILE_CHARS

    def test_every_oversized_file_is_marked(self):
        files = [_file(f"src/F{i}.cs", _CONTEXT_FILE_CHARS + 100) for i in range(5)]
        sections, _ = _render(files)

        assert all("[truncated:" in s for s in sections)


class TestBannerReportsWhatWasSent:
    def test_banner_counts_post_truncation_size(self):
        """The banner's count must equal what was actually included.

        Previously it summed `len(f.content)` before the cut and labelled the
        result "bytes", so an operator reading "340000 bytes" concluded the
        model had seen 340 KB of source when it had seen a fraction of that.
        The unit is now stated as chars, which is what the caps and the sum
        actually measure.
        """
        files = [_file(f"src/F{i}.cs", 100_000) for i in range(3)]
        _, banner = _render(files)

        sent = 3 * _CONTEXT_FILE_CHARS
        assert f"{sent} of {3 * 100_000} chars" in banner

    def test_banner_count_matches_rendered_content(self):
        """Cross-check the banner against the sections actually produced."""
        files = [
            _file("src/Big.cs", 40_000),
            _file("src/Small.cs", 120),
            _file("docs/README.md", 30_000),
        ]
        sections, banner = _render(files)

        reported = int(re.search(r"\((?:.*?), (\d+) of", banner).group(1))
        rendered = 0
        for section in sections:
            body = section.split("---\n", 1)[1]
            rendered += len(body.split("\n... [truncated:", 1)[0])
        assert reported == rendered

    def test_banner_omits_the_of_clause_when_nothing_was_cut(self):
        _, banner = _render([_file("src/Small.cs", 120)])

        assert "120 chars" in banner
        assert " of " not in banner

    def test_banner_reports_dropped_files(self):
        """`files[:20]` silently discarded the rest; now it says so."""
        files = [_file(f"src/F{i}.cs", 10) for i in range(_MAX_CONTEXT_FILES + 4)]
        _, banner = _render(files)

        assert f"{_MAX_CONTEXT_FILES} of {_MAX_CONTEXT_FILES + 4} files" in banner

    def test_only_shown_files_are_rendered(self):
        files = [_file(f"src/F{i}.cs", 10) for i in range(_MAX_CONTEXT_FILES + 4)]
        sections, _ = _render(files)

        assert len(sections) == _MAX_CONTEXT_FILES

    def test_banner_counts_truncated_files(self):
        files = [
            _file("src/Big1.cs", 40_000),
            _file("src/Big2.cs", 40_000),
            _file("src/Small.cs", 10),
        ]
        _, banner = _render(files)

        assert "2 files truncated" in banner

    def test_single_truncated_file_is_singular(self):
        _, banner = _render([_file("src/Big.cs", 40_000)])

        assert "1 file truncated" in banner

    def test_dropped_files_still_count_toward_total(self):
        """The 21st file was never sent, and the total must say so."""
        files = [_file(f"src/F{i}.cs", 100) for i in range(_MAX_CONTEXT_FILES + 1)]
        _, banner = _render(files)

        sent = _MAX_CONTEXT_FILES * 100
        total = (_MAX_CONTEXT_FILES + 1) * 100
        assert f"{sent} of {total} chars" in banner


class TestEdgeCases:
    def test_empty_content_does_not_crash(self):
        sections, banner = _render([ContextFile(path="src/Empty.cs", content="")])

        assert "src/Empty.cs" in sections[0]
        assert "0 chars" in banner

    def test_none_content_is_treated_as_empty(self):
        sections, banner = _render([ContextFile(path="src/None.cs", content=None)])

        assert "src/None.cs" in sections[0]
        assert "0 chars" in banner

    def test_content_exactly_at_the_cap_is_not_marked(self):
        """Boundary: `>` not `>=`, so an exactly-fitting file is untouched."""
        sections, _ = _render([_file("src/Exact.cs", _CONTEXT_FILE_CHARS)])

        assert "[truncated" not in sections[0]


class TestVisibleContent:
    """The third return value: each file cut down to what the model saw."""

    def test_visible_content_is_truncated(self):
        _, _, visible = NeoEngine._render_context_files(
            [_file("src/Service.cs", 40_000)]
        )

        assert len(visible[0].content) == _CONTEXT_FILE_CHARS

    def test_visible_content_carries_no_marker(self):
        """The marker is prompt prose; downstream scanners must not see it."""
        _, _, visible = NeoEngine._render_context_files(
            [_file("src/Service.cs", 40_000)]
        )

        assert "[truncated" not in visible[0].content

    def test_untruncated_file_is_passed_through_unchanged(self):
        original = _file("src/Small.cs", 100)
        _, _, visible = NeoEngine._render_context_files([original])

        assert visible[0] is original

    def test_visible_excludes_files_past_the_cap(self):
        files = [_file(f"src/F{i}.cs", 10) for i in range(_MAX_CONTEXT_FILES + 4)]
        _, _, visible = NeoEngine._render_context_files(files)

        assert len(visible) == _MAX_CONTEXT_FILES


class TestPromptIntegration:
    def _engine(self):
        engine = NeoEngine.__new__(NeoEngine)
        engine.exemplar_index = None
        engine.fact_store = None
        engine.persistent_memory = None
        engine.context = None
        engine.codebase_root = "/nonexistent-for-this-test"
        return engine

    def test_smells_never_cite_a_line_the_model_cannot_see(self):
        """A finding about invisible code asserts what a bare cut merely invites.

        `scan_files` reads whatever content it is handed. Handed the
        originals, it emitted `src/Service.cs:401 [todo/warn] HACK: ...` for
        a file the model was shown to roughly line 215 — a line-numbered
        claim about text that was never sent.
        """
        head = "// padding line\n" * 400          # ~6400 chars, past the cap
        content = head + "        // HACK: this is broken\n"
        marker_file = ContextFile(path="src/Service.cs", content=content)

        prompt, _ = self._engine()._format_combined_prompt({
            'prompt': 'review this',
            'task_type': 'bugfix',
            'files': [marker_file],
        })

        assert "HACK: this is broken" not in prompt

    def test_smells_within_the_visible_window_are_still_reported(self):
        """The fix must scope the scan, not disable it."""
        content = "        // HACK: this is broken\n" + "// padding\n" * 400
        marker_file = ContextFile(path="src/Service.cs", content=content)

        prompt, _ = self._engine()._format_combined_prompt({
            'prompt': 'review this',
            'task_type': 'bugfix',
            'files': [marker_file],
        })

        assert "HACK: this is broken" in prompt

    def test_marker_reaches_the_assembled_prompt(self, monkeypatch):
        """End to end: the marker must survive into the text the model sees."""
        engine = NeoEngine.__new__(NeoEngine)
        engine.exemplar_index = None
        engine.fact_store = None
        engine.persistent_memory = None
        engine.context = None
        engine.codebase_root = "/nonexistent-for-this-test"

        prompt, _ = engine._format_combined_prompt({
            'prompt': 'why does this fail',
            'task_type': 'bugfix',
            'files': [_file("src/Service.cs", 40_000)],
        })

        assert "[truncated:" in prompt
        assert "REPOSITORY CONTEXT" in prompt
