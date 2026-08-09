"""Constraint ingestion: one rule file must produce one set of facts.

`CONSTRAINT_FILES` names several spellings of the same document on purpose,
because different tools write different ones. On a case-insensitive filesystem
those spellings are not alternatives, they are the same file listed twice — and
the ingester used to walk it once per spelling, keying both the checksum cache
and the supersession pass on the literal path string, so each visit was blind to
the last. Every extra spelling minted a complete duplicate set of CONSTRAINT
facts at confidence 1.0, which are exempt from recall decay and so never aged
out. See issue #138.
"""

import os

import pytest

from neo.memory.constraints import ConstraintIngester
from neo.memory.models import Fact, FactKind, FactMetadata, FactScope

RULES = """# Project

## Build

Run `make build`.

## Test

Run `make test`.
"""


def _ingester(root):
    return ConstraintIngester(
        codebase_root=str(root), org_id="org", project_id="proj"
    )


def _valid(facts):
    return [f for f in facts if f.is_valid]


def _case_insensitive(root) -> bool:
    """Does this filesystem treat AGENTS.md and agents.md as one file?

    Asked rather than assumed: the bug is native to macOS and Windows, and the
    fix must be exercised where it applies without failing the suite on the
    ext4 CI runners where it cannot reproduce.
    """
    probe = root / "CaseProbe.tmp"
    probe.write_text("x")
    try:
        return (root / "caseprobe.tmp").exists()
    finally:
        probe.unlink()


class TestAliasCollapsing:
    def test_one_file_two_spellings_ingests_once(self, tmp_path):
        """The reported defect, at its root.

        With `AGENTS.md` present and the filesystem case-insensitive, the
        `agents.md` entry resolves to the same file. Before the fix this
        produced two facts per section; every project on the reporting machine
        that had an `AGENTS.md` showed exactly this, and projects without one
        showed none.
        """
        if not _case_insensitive(tmp_path):
            pytest.skip("filesystem is case-sensitive; aliasing cannot occur")

        (tmp_path / "AGENTS.md").write_text(RULES)

        new_facts, _ = _ingester(tmp_path).ingest([])

        subjects = sorted(f.subject for f in new_facts)
        assert subjects == ["Build", "Test"]

    def test_symlinked_spelling_is_collapsed_on_any_filesystem(self, tmp_path):
        """Identity is `(st_dev, st_ino)`, so symlinks collapse too.

        This is the case-sensitive filesystem's version of the same fault, and
        it is why the fix asks the filesystem instead of normalizing strings:
        `realpath` would resolve this one but not the case-only alias, and
        `normcase` is a no-op outside Windows.
        """
        (tmp_path / "AGENTS.md").write_text(RULES)
        os.symlink(tmp_path / "AGENTS.md", tmp_path / "CLAUDE.md")

        new_facts, _ = _ingester(tmp_path).ingest([])

        assert sorted(f.subject for f in new_facts) == ["Build", "Test"]

    def test_genuinely_distinct_files_are_both_ingested(self, tmp_path):
        """The other side of the contract.

        A repo carrying a real `CLAUDE.md` AND a real `AGENTS.md` has two rule
        files, not one written twice. Collapsing those would lose rules, which
        is a worse failure than the duplication being fixed.
        """
        (tmp_path / "CLAUDE.md").write_text("## Build\n\nUse cmake.\n")
        (tmp_path / "AGENTS.md").write_text("## Deploy\n\nUse helm.\n")

        new_facts, _ = _ingester(tmp_path).ingest([])

        assert sorted(f.subject for f in new_facts) == ["Build", "Deploy"]

    def test_canonical_path_follows_constraint_files_order(self, tmp_path):
        """Which spelling wins must be stable, not filesystem-dependent."""
        (tmp_path / "AGENTS.md").write_text(RULES)
        os.symlink(tmp_path / "AGENTS.md", tmp_path / "CLAUDE.md")

        new_facts, _ = _ingester(tmp_path).ingest([])

        # CLAUDE.md precedes AGENTS.md in CONSTRAINT_FILES.
        assert {f.metadata.source_file for f in new_facts} == {
            str(tmp_path / "CLAUDE.md")
        }


class TestMigrationOfExistingDuplicates:
    """Duplicates already in a store must be retired, not merely stopped.

    CONSTRAINT facts bypass recall decay and are protected from eviction, so
    nothing else in the system will ever remove them. And the file they came
    from has not changed, so its checksum still matches — meaning the cleanup
    has to run ABOVE the unchanged-file short-circuit or it never runs at all.
    """

    def _stale(self, path, subject):
        return Fact(
            subject=subject,
            body="Old copy.",
            kind=FactKind.CONSTRAINT,
            scope=FactScope.PROJECT,
            project_id="proj",
            metadata=FactMetadata(source_file=str(path), confidence=1.0),
            tags=["constraint", "auto-ingested"],
        )

    def test_alias_duplicates_are_retired_when_file_is_unchanged(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text(RULES)
        os.symlink(tmp_path / "AGENTS.md", tmp_path / "CLAUDE.md")

        ingester = _ingester(tmp_path)
        # First pass populates the checksum cache, so the second pass takes the
        # unchanged-file path — which is exactly where the old code stopped.
        first_facts, _ = ingester.ingest([])

        store = list(first_facts) + [
            self._stale(tmp_path / "AGENTS.md", "Build"),
            self._stale(tmp_path / "AGENTS.md", "Test"),
        ]

        new_facts, superseded = ingester.ingest(store)

        assert new_facts == []                    # nothing re-minted
        assert len(superseded) == 2               # both alias copies retired
        assert all(not f.is_valid for f in superseded)
        assert {f.metadata.source_file for f in _valid(store)} == {
            str(tmp_path / "CLAUDE.md")
        }

    def test_canonical_facts_survive_the_migration(self, tmp_path):
        """Retiring aliases must not take the good copies with them."""
        (tmp_path / "AGENTS.md").write_text(RULES)
        os.symlink(tmp_path / "AGENTS.md", tmp_path / "CLAUDE.md")

        ingester = _ingester(tmp_path)
        canonical, _ = ingester.ingest([])
        store = list(canonical) + [self._stale(tmp_path / "AGENTS.md", "Build")]

        ingester.ingest(store)

        assert len(_valid(store)) == len(canonical)
        assert sorted(f.subject for f in _valid(store)) == ["Build", "Test"]

    def test_alias_checksum_rows_are_dropped(self, tmp_path):
        """A stale cache row must not outlive the spelling it belonged to."""
        (tmp_path / "AGENTS.md").write_text(RULES)
        os.symlink(tmp_path / "AGENTS.md", tmp_path / "CLAUDE.md")

        ingester = _ingester(tmp_path)
        ingester._checksums[str(tmp_path / "AGENTS.md")] = "stale"
        ingester.ingest([])

        assert str(tmp_path / "AGENTS.md") not in ingester._checksums
        assert str(tmp_path / "CLAUDE.md") in ingester._checksums


class TestUnchangedBehaviour:
    def test_unchanged_file_is_not_re_ingested(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(RULES)
        ingester = _ingester(tmp_path)

        first, _ = ingester.ingest([])
        second, superseded = ingester.ingest(list(first))

        assert first != []
        assert second == []
        assert superseded == []

    def test_changed_file_supersedes_and_re_mints(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(RULES)
        ingester = _ingester(tmp_path)
        first, _ = ingester.ingest([])

        (tmp_path / "CLAUDE.md").write_text("## Build\n\nUse bazel now.\n")
        store = list(first)
        second, superseded = ingester.ingest(store)

        assert len(superseded) == len(first)
        assert [f.subject for f in second] == ["Build"]
        assert "bazel" in second[0].body

    def test_missing_files_are_skipped(self, tmp_path):
        new_facts, superseded = _ingester(tmp_path).ingest([])
        assert new_facts == []
        assert superseded == []
