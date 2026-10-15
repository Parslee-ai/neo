"""Tests for neo.memory.outcomes - Outcome-based learning."""

import json
import time
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from neo.memory.outcomes import OutcomeTracker, OutcomeType, SessionRecord


@dataclass
class FakeSuggestion:
    """Minimal CodeSuggestion stand-in for tests."""
    file_path: str = ""
    description: str = ""
    confidence: float = 0.8
    unified_diff: str = ""
    code_block: str = ""


@pytest.fixture
def tmp_sessions_dir(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    return sessions


@pytest.fixture
def tracker(tmp_sessions_dir, tmp_path):
    with patch("neo.memory.outcomes.SESSIONS_DIR", tmp_sessions_dir):
        t = OutcomeTracker(codebase_root=str(tmp_path), project_id="testproj1234")
    return t


class TestSaveAndLoadSession:
    def test_round_trip(self, tracker):
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Add validation", confidence=0.9),
            FakeSuggestion(file_path="src/bar.py", description="Fix import", confidence=0.7),
        ]
        tracker.save_session(suggestions, "fix the bug")

        session = tracker._load_previous_session()
        assert session is not None
        assert session.project_id == "testproj1234"
        assert session.prompt == "fix the bug"
        assert len(session.suggestions) == 2
        assert session.suggestions[0]["file_path"] == "src/foo.py"
        assert session.suggestions[1]["confidence"] == 0.7
        assert session.timestamp > 0

    def test_skips_non_code_suggestions(self, tracker):
        suggestions = [
            FakeSuggestion(file_path="/", description="Review only"),
            FakeSuggestion(file_path="N/A", description="No file"),
            FakeSuggestion(file_path="src/real.py", description="Real change"),
        ]
        tracker.save_session(suggestions, "prompt")

        session = tracker._load_previous_session()
        assert len(session.suggestions) == 1
        assert session.suggestions[0]["file_path"] == "src/real.py"

    def test_no_project_id_skips_save(self, tmp_sessions_dir):
        with patch("neo.memory.outcomes.SESSIONS_DIR", tmp_sessions_dir):
            t = OutcomeTracker(codebase_root="/tmp", project_id="")
        t.save_session([FakeSuggestion(file_path="x.py")], "prompt")
        # Should not crash, and no file created
        assert not list(tmp_sessions_dir.iterdir())

    def test_corrupt_session_returns_none(self, tracker):
        # Write corrupt data
        with open(tracker._session_path, "w") as f:
            f.write("not valid json")
        assert tracker._load_previous_session() is None
        backups = list(tracker._session_path.parent.glob(f"{tracker._session_path.name}.corrupt-*"))
        assert len(backups) == 1
        assert backups[0].read_text() == "not valid json"

    def test_missing_session_returns_none(self, tracker):
        assert tracker._load_previous_session() is None


class TestSuggestionLedger:
    """The durable, append-only ledger that survives session-log clearing and
    feeds the async transcript outcome miner."""

    def test_save_session_appends_linked_suggestions(self, tracker):
        tracker.save_session(
            [FakeSuggestion(file_path="/REVIEW.md", description="review verdict", confidence=0.8)],
            "review my diff",
            suggestion_fact_ids={"/REVIEW.md": "factR"},
        )
        led = tracker.load_suggestion_ledger()
        assert len(led) == 1
        assert led[0]["fact_id"] == "factR"
        assert led[0]["file_path"] == "/REVIEW.md"
        assert led[0]["ts"] > 0

    def test_ledger_skips_unlinked_suggestions(self, tracker):
        # No suggestion_fact_ids → nothing to reinforce → nothing recorded.
        tracker.save_session([FakeSuggestion(file_path="src/x.py")], "prompt")
        assert tracker.load_suggestion_ledger() == []

    def test_compact_drops_expired_and_specified(self, tracker):
        import json
        now = time.time()
        path = tracker._suggestion_ledger_path
        rows = [
            {"id": "fresh", "ts": now, "fact_id": "f1", "file_path": "/A", "description": "a"},
            {"id": "old", "ts": now - 40 * 86400, "fact_id": "f2", "file_path": "/B", "description": "b"},
            {"id": "drop", "ts": now, "fact_id": "f3", "file_path": "/C", "description": "c"},
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        tracker.compact_suggestion_ledger(drop_ids={"drop"}, max_age_days=30)
        remaining = {e["id"] for e in tracker.load_suggestion_ledger()}
        assert remaining == {"fresh"}  # 'old' aged out, 'drop' explicitly removed


class TestSessionRecordWithFactIds:
    def test_save_and_load_with_fact_ids(self, tracker):
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Add validation", confidence=0.9),
        ]
        fact_ids = {"src/foo.py": "abc123def456"}
        tracker.save_session(suggestions, "fix bug", suggestion_fact_ids=fact_ids)

        session = tracker._load_previous_session()
        assert session is not None
        assert session.suggestion_fact_ids == {"src/foo.py": "abc123def456"}

    def test_backward_compat_no_fact_ids(self, tracker):
        """Old sessions without suggestion_fact_ids should load with empty dict."""
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Fix", confidence=0.8),
        ]
        # Save without fact_ids (old behavior)
        tracker.save_session(suggestions, "fix")

        session = tracker._load_previous_session()
        assert session is not None
        assert session.suggestion_fact_ids == {}

    def test_detect_outcomes_returns_fact_ids(self, tracker):
        """detect_outcomes should return suggestion_fact_ids from previous session."""
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Add validation", confidence=0.9),
        ]
        fact_ids = {"src/foo.py": "fact123"}
        tracker.save_session(suggestions, "fix", suggestion_fact_ids=fact_ids)

        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value="+code"):
            outcomes, returned_ids = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert returned_ids == {"src/foo.py": "fact123"}

    def test_suggested_code_persisted_in_session(self, tracker):
        suggestions = [
            FakeSuggestion(
                file_path="src/foo.py",
                description="Rewrite handler",
                confidence=0.9,
                code_block="def handler():\n    return 42\n",
            ),
        ]

        tracker.save_session(suggestions, "rewrite")

        session = tracker._load_previous_session()
        assert session is not None
        assert session.suggestions[0]["suggested_code"] == "def handler():\n    return 42\n"

    def test_episode_candidate_attribution_round_trips_to_outcome(self, tracker):
        suggestion = FakeSuggestion(
            file_path="src/foo.py",
            description="Add validation",
            confidence=0.8,
            unified_diff="+validate()",
        )
        suggestion.suggestion_id = "suggestion-1"
        tracker.save_session(
            [suggestion],
            "fix",
            learning_episode_id="episode-1",
            repository_revision="abc123",
            retrieved_fact_ids=["fact-a"],
            candidates_by_suggestion={
                "suggestion-1": {
                    "candidate_id": "candidate-1",
                    "subject": "pattern: validate",
                    "body": "validate before processing",
                    "kind": "pattern",
                }
            },
        )

        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value="+validate()"):
            outcomes, _ = tracker.detect_outcomes()

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.suggestion_id == "suggestion-1"
        assert outcome.learning_episode_id == "episode-1"
        assert outcome.repository_revision == "abc123"
        assert outcome.retrieved_fact_ids == ["fact-a"]
        assert outcome.candidate_id == "candidate-1"


class TestDetectOutcomesAccepted:
    def test_accepted_when_suggested_file_changed(self, tracker):
        # Save a session with suggestions (including a unified_diff for comparison)
        suggested_diff = "+    validate_input(data)\n-    process(data)"
        suggestions = [
            FakeSuggestion(
                file_path="src/foo.py", description="Add validation",
                confidence=0.9, unified_diff=suggested_diff,
            ),
        ]
        tracker.save_session(suggestions, "fix bug")

        # Mock git to return the same file as changed, with matching diff content
        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value=suggested_diff):
            outcomes, _ = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == OutcomeType.ACCEPTED
        assert outcomes[0].file_path == "src/foo.py"
        assert outcomes[0].suggestion_description == "Add validation"
        assert outcomes[0].suggestion_confidence == 0.9
        assert "validate_input" in outcomes[0].diff_summary


    def test_diff_content_captured_for_independent(self, tracker):
        suggestions = [FakeSuggestion(file_path="src/foo.py")]
        tracker.save_session(suggestions, "fix")

        diff_text = "+def new_helper():\n+    return 42"
        with patch.object(tracker, "_get_changed_files_since", return_value={"src/bar.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value=diff_text):
            outcomes, _ = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == OutcomeType.INDEPENDENT
        assert "new_helper" in outcomes[0].diff_summary

    def test_accepted_when_code_block_matches_added_lines(self, tracker):
        suggestions = [
            FakeSuggestion(
                file_path="src/foo.py",
                description="Rewrite handler",
                confidence=0.9,
                code_block="def handler():\n    return 42\n",
            ),
        ]
        tracker.save_session(suggestions, "fix bug")

        actual_diff = "@@ -1,0 +1,2 @@\n+def handler():\n+    return 42"
        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value=actual_diff):
            outcomes, _ = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == OutcomeType.ACCEPTED

    def test_modified_when_code_block_diverges_from_added_lines(self, tracker):
        suggestions = [
            FakeSuggestion(
                file_path="src/foo.py",
                description="Rewrite handler",
                confidence=0.9,
                code_block="def handler():\n    return 42\n",
            ),
        ]
        tracker.save_session(suggestions, "fix bug")

        actual_diff = "@@ -1,0 +1,2 @@\n+def handler():\n+    return 7"
        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value=actual_diff):
            outcomes, _ = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == OutcomeType.MODIFIED


class TestDetectOutcomesNoChanges:
    def test_no_outcomes_when_no_changes(self, tracker):
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Add validation"),
        ]
        tracker.save_session(suggestions, "fix bug")

        with patch.object(tracker, "_get_changed_files_since", return_value=set()):
            outcomes, _ = tracker.detect_outcomes()

        assert outcomes == []

    def test_no_outcomes_when_no_previous_session(self, tracker):
        outcomes, fact_ids = tracker.detect_outcomes()
        assert outcomes == []
        assert fact_ids == {}

    def test_processed_session_not_replayed_from_fallback_file(self, tracker):
        """Clearing the log also clears the legacy single-session fallback."""

        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Add validation"),
        ]
        tracker.save_session(suggestions, "fix bug")

        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value="+code"):
            outcomes, _ = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert not tracker._session_log_path.exists()
        assert not tracker._session_path.exists()

        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value="+code"):
            outcomes, fact_ids = tracker.detect_outcomes()

        assert outcomes == []
        assert fact_ids == {}


class TestDetectOutcomesIndependent:
    def test_independent_changes_detected(self, tracker):
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Fix foo"),
        ]
        tracker.save_session(suggestions, "fix bug")

        # User changed a different file
        with patch.object(tracker, "_get_changed_files_since", return_value={"src/bar.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value="+new code"):
            outcomes, _ = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == OutcomeType.INDEPENDENT
        assert outcomes[0].file_path == "src/bar.py"

    def test_non_code_files_ignored(self, tracker):
        suggestions = [FakeSuggestion(file_path="src/foo.py")]
        tracker.save_session(suggestions, "fix")

        with patch.object(
            tracker,
            "_get_changed_files_since",
            return_value={"README.md", "config.yaml", ".gitignore"},
        ):
            outcomes, _ = tracker.detect_outcomes()

        assert outcomes == []

    def test_mixed_accepted_and_independent(self, tracker):
        diff_text = "+change"
        suggestions = [
            FakeSuggestion(
                file_path="src/foo.py", description="Fix foo",
                unified_diff=diff_text,
            ),
        ]
        tracker.save_session(suggestions, "fix")

        with patch.object(
            tracker,
            "_get_changed_files_since",
            return_value={"src/foo.py", "src/bar.py"},
        ), patch.object(tracker, "_get_file_diff_since", return_value=diff_text):
            outcomes, _ = tracker.detect_outcomes()

        types = {o.outcome_type for o in outcomes}
        assert OutcomeType.ACCEPTED in types
        assert OutcomeType.INDEPENDENT in types


class TestReviewOnlyOutcomes:
    """Review/analysis suggestions are neo's most common workload but can't be
    tracked by git diff; they should still yield a weak UNVERIFIED signal."""

    @pytest.mark.parametrize("path", [
        "/REVIEW.md", "REVIEW.md", "/ARCHITECTURAL_REVIEW.md",
        "/review/commit-abc-architecture.md", "/architecture-review/foo.md",
        "NO_MODIFY", "/NO_MODIFY", "/docs/reviews/feedback.md",
    ])
    def test_review_path_classified_review_only(self, path):
        assert OutcomeTracker._is_review_only_path(path) is True

    @pytest.mark.parametrize("path", [
        "src/review_service.py", "src/foo.py", "review_notes.txt",
        "src/reviewer.py", "", "/", "N/A",
    ])
    def test_real_paths_not_review_only(self, path):
        assert OutcomeTracker._is_review_only_path(path) is False

    def test_review_suggestion_emits_unverified(self, tracker):
        """A persisted review-doc suggestion produces an UNVERIFIED outcome
        even when no files changed in git, preserving the linked fact id."""
        suggestions = [FakeSuggestion(
            file_path="/REVIEW.md", description="Linus review", confidence=0.87,
        )]
        tracker.save_session(
            suggestions, "review my diff",
            suggestion_fact_ids={"/REVIEW.md": "factR"},
        )
        with patch.object(tracker, "_get_changed_files_since", return_value=set()):
            outcomes, fact_ids = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == OutcomeType.UNVERIFIED
        assert outcomes[0].file_path == "/REVIEW.md"
        assert fact_ids == {"/REVIEW.md": "factR"}

    def test_git_match_wins_over_non_git_no_double_count(self, tracker):
        """A docs path that is BOTH git-changed and non-git-trackable must
        yield ONE outcome (the stronger git signal), not an ACCEPTED + a
        duplicate UNVERIFIED that double-bumps the same fact."""
        diff = "+new line"
        suggestions = [FakeSuggestion(
            file_path="docs/architecture.md", description="Update arch doc",
            confidence=0.8, unified_diff=diff,
        )]
        tracker.save_session(suggestions, "doc it")

        with patch.object(tracker, "_get_changed_files_since",
                          return_value={"docs/architecture.md"}), \
             patch.object(tracker, "_get_file_diff_since", return_value=diff):
            outcomes, _ = tracker.detect_outcomes()

        paths = [o.file_path for o in outcomes]
        assert paths.count("docs/architecture.md") == 1
        assert outcomes[0].outcome_type in (OutcomeType.ACCEPTED, OutcomeType.MODIFIED)
        assert all(o.outcome_type != OutcomeType.UNVERIFIED for o in outcomes)

    def test_same_review_path_across_sessions_counts_once(self, tracker):
        """The same review path queued in multiple unprocessed sessions
        collapses to a single weak-acceptance outcome."""
        for _ in range(3):
            tracker.save_session(
                [FakeSuggestion(file_path="/REVIEW.md", description="review", confidence=0.8)],
                "review",
                suggestion_fact_ids={"/REVIEW.md": "factR"},
            )
        with patch.object(tracker, "_get_changed_files_since", return_value=set()):
            outcomes, _ = tracker.detect_outcomes()

        review_outcomes = [o for o in outcomes if o.file_path == "/REVIEW.md"]
        assert len(review_outcomes) == 1
        assert review_outcomes[0].outcome_type == OutcomeType.UNVERIFIED


class TestGitNotAvailable:
    def test_graceful_when_git_missing(self, tracker):
        suggestions = [FakeSuggestion(file_path="src/foo.py")]
        tracker.save_session(suggestions, "fix")

        with patch("neo.memory.outcomes.subprocess.run", side_effect=FileNotFoundError):
            outcomes, _ = tracker.detect_outcomes()

        assert outcomes == []

    def test_graceful_when_not_git_repo(self, tracker):
        suggestions = [FakeSuggestion(file_path="src/foo.py")]
        tracker.save_session(suggestions, "fix")

        import subprocess
        with patch(
            "neo.memory.outcomes.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            outcomes, _ = tracker.detect_outcomes()

        assert outcomes == []


class TestDifferentProject:
    def test_ignores_session_from_different_project(self, tracker):
        # Manually write a session with a different project_id
        session = SessionRecord(
            timestamp=time.time() - 60,
            codebase_root="/other/project",
            project_id="different_project",
            prompt="old prompt",
            suggestions=[{"file_path": "x.py", "description": "old", "confidence": 0.5}],
        )
        with open(tracker._session_path, "w") as f:
            json.dump({
                "timestamp": session.timestamp,
                "codebase_root": session.codebase_root,
                "project_id": session.project_id,
                "prompt": session.prompt,
                "suggestions": session.suggestions,
            }, f)

        outcomes, _ = tracker.detect_outcomes()
        assert outcomes == []


class TestPathNormalization:
    def test_absolute_path_normalized(self, tracker):
        abs_path = f"{tracker.codebase_root}/src/foo.py"
        assert tracker._normalize_path(abs_path) == "src/foo.py"

    def test_relative_path_unchanged(self, tracker):
        assert tracker._normalize_path("src/foo.py") == "src/foo.py"

    def test_unrelated_absolute_path_unchanged(self, tracker):
        result = tracker._normalize_path("/completely/different/path.py")
        assert result == "/completely/different/path.py"


def _git_log_z(*commits):
    """`git log -z --name-only` output for ``(hash, time, subject, files)``
    in the `_get_commits_since` format."""
    return "".join(
        f"\x01{sha}\t{ts}\t{subject}" + ("\n" + "\0".join(files) if files else "") + "\0\0"
        for sha, ts, subject, files in commits
    )


class TestIngestGitHistory:
    def test_ingests_commits(self, tracker):
        git_log_output = _git_log_z(
            ("a" * 40, 1773000000, "feat: add user validation", ["src/auth.py", "src/models.py"]),
            ("b" * 40, 1772900000, "fix: handle null email in signup", ["src/auth.py"]),
        )
        diff_output = "+def validate_email(email):\n+    if not email:\n+        raise ValueError"

        def mock_run(cmd, **kwargs):
            result = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
            if "log" in cmd:
                result.stdout = git_log_output
            elif "show" in cmd:
                result.stdout = diff_output
            return result

        with patch("neo.memory.outcomes.subprocess.run", side_effect=mock_run):
            records = tracker.ingest_git_history(max_commits=10)

        assert len(records) == 2
        assert "user validation" in records[0]["subject"]
        assert "null email" in records[1]["subject"]
        assert "validate_email" in records[0]["body"]

    def test_skips_merge_commits(self, tracker):
        git_log_output = _git_log_z(
            ("a" * 40, 1773000000, "Merge pull request #42 from feature", ["src/foo.py"]),
            ("b" * 40, 1772900000, "fix: real bug fix", ["src/bar.py"]),
        )

        def mock_run(cmd, **kwargs):
            result = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
            if "log" in cmd:
                result.stdout = git_log_output
            elif "show" in cmd:
                result.stdout = "+fixed"
            return result

        with patch("neo.memory.outcomes.subprocess.run", side_effect=mock_run):
            records = tracker.ingest_git_history(max_commits=10)

        assert len(records) == 1
        assert "real bug fix" in records[0]["subject"]

    def test_skips_short_messages(self, tracker):
        git_log_output = _git_log_z(("a" * 40, 1773000000, "wip", ["src/foo.py"]))

        def mock_run(cmd, **kwargs):
            result = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
            if "log" in cmd:
                result.stdout = git_log_output
            elif "show" in cmd:
                result.stdout = "+stuff"
            return result

        with patch("neo.memory.outcomes.subprocess.run", side_effect=mock_run):
            records = tracker.ingest_git_history(max_commits=10)

        assert len(records) == 0

    def test_watermark_prevents_reingestion(self, tracker):
        git_log_output = _git_log_z(
            ("a" * 40, 1773000000, "feat: add something useful here", ["src/foo.py"]),
        )

        call_count = 0

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            result = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
            if "log" in cmd:
                # First call returns commits, second returns empty (watermark used)
                call_count += 1
                if call_count <= 1:
                    result.stdout = git_log_output
                else:
                    result.stdout = ""
            elif "show" in cmd:
                result.stdout = "+new code"
            return result

        with patch("neo.memory.outcomes.subprocess.run", side_effect=mock_run):
            records1 = tracker.ingest_git_history(max_commits=10)
            records2 = tracker.ingest_git_history(max_commits=10)

        assert len(records1) == 1
        assert len(records2) == 0

    def test_graceful_when_git_unavailable(self, tracker):
        with patch("neo.memory.outcomes.subprocess.run", side_effect=FileNotFoundError):
            records = tracker.ingest_git_history(max_commits=10)
        assert records == []

    def test_no_codebase_root(self, tmp_sessions_dir):
        with patch("neo.memory.outcomes.SESSIONS_DIR", tmp_sessions_dir):
            t = OutcomeTracker(codebase_root=None, project_id="test123")
        records = t.ingest_git_history()
        assert records == []


class TestIsMeaningfulCommit:
    def test_feature_commit(self):
        assert OutcomeTracker._is_meaningful_commit("feat: add user auth") is True

    def test_fix_commit(self):
        assert OutcomeTracker._is_meaningful_commit("fix: handle null pointer in parser") is True

    def test_merge_commit(self):
        assert OutcomeTracker._is_meaningful_commit("Merge pull request #42") is False

    def test_version_bump(self):
        assert OutcomeTracker._is_meaningful_commit("bump version to 1.2.3") is False

    def test_short_message(self):
        assert OutcomeTracker._is_meaningful_commit("wip") is False

    def test_release(self):
        assert OutcomeTracker._is_meaningful_commit("Release v2.0.0") is False


class TestIsCodeFile:
    def test_python_is_code(self):
        assert OutcomeTracker._is_code_file("src/foo.py") is True

    def test_typescript_is_code(self):
        assert OutcomeTracker._is_code_file("app/bar.tsx") is True

    def test_markdown_is_not_code(self):
        assert OutcomeTracker._is_code_file("README.md") is False

    def test_yaml_is_not_code(self):
        assert OutcomeTracker._is_code_file("config.yaml") is False

    def test_no_extension_is_not_code(self):
        assert OutcomeTracker._is_code_file("Makefile") is False


class TestComputeDiffOverlap:
    def test_identical_diffs(self):
        diff = "+line1\n+line2\n-line3"
        assert OutcomeTracker._compute_diff_overlap(diff, diff) == 1.0

    def test_completely_different_diffs(self):
        suggested = "+add_validation(x)\n-old_code()"
        actual = "+completely_different()\n-other_thing()"
        assert OutcomeTracker._compute_diff_overlap(suggested, actual) == 0.0

    def test_partial_overlap(self):
        suggested = "+line1\n+line2\n-line3"
        actual = "+line1\n+different\n-line3"
        overlap = OutcomeTracker._compute_diff_overlap(suggested, actual)
        # 2 of 4 unique lines match ("+line1" and "-line3")
        assert 0.4 <= overlap <= 0.6

    def test_both_empty(self):
        assert OutcomeTracker._compute_diff_overlap("", "") == 1.0

    def test_one_empty(self):
        assert OutcomeTracker._compute_diff_overlap("+line", "") == 0.0
        assert OutcomeTracker._compute_diff_overlap("", "+line") == 0.0

    def test_ignores_diff_headers(self):
        suggested = "--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,3 @@\n+real_line"
        actual = "--- a/foo.py\n+++ b/foo.py\n@@ -5,3 +5,3 @@\n+real_line"
        # Only "+real_line" counts as a change line
        assert OutcomeTracker._compute_diff_overlap(suggested, actual) == 1.0

    def test_no_change_lines_in_either(self):
        suggested = "--- a/foo.py\n+++ b/foo.py"
        actual = "--- a/bar.py\n+++ b/bar.py"
        # No change lines extracted from either -> both empty -> 1.0
        assert OutcomeTracker._compute_diff_overlap(suggested, actual) == 1.0


class TestModifiedOutcome:
    def test_modified_when_diff_diverges(self, tracker):
        """When suggested diff and actual diff have low overlap, outcome is 'modified'."""
        suggested_diff = "+validate_input(data)\n-process(data)"
        suggestions = [
            FakeSuggestion(
                file_path="src/foo.py",
                description="Add validation",
                confidence=0.9,
                unified_diff=suggested_diff,
            ),
        ]
        tracker.save_session(suggestions, "fix bug")

        actual_diff = "+completely_rewritten()\n-something_else()"
        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value=actual_diff):
            outcomes, _ = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == OutcomeType.MODIFIED
        assert outcomes[0].file_path == "src/foo.py"
        assert outcomes[0].suggestion_description == "Add validation"

    def test_accepted_when_diff_matches(self, tracker):
        """When suggested diff and actual diff overlap well, outcome is 'accepted'."""
        suggested_diff = "+validate_input(data)\n-process(data)"
        suggestions = [
            FakeSuggestion(
                file_path="src/foo.py",
                description="Add validation",
                confidence=0.9,
                unified_diff=suggested_diff,
            ),
        ]
        tracker.save_session(suggestions, "fix bug")

        # Actual diff matches the suggestion
        actual_diff = "+validate_input(data)\n-process(data)\n+extra_line()"
        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value=actual_diff):
            outcomes, _ = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == OutcomeType.ACCEPTED

    def test_unverified_when_no_suggested_diff(self, tracker):
        """When no suggested_diff was stored, classify as 'unverified' not 'accepted'."""
        suggestions = [
            FakeSuggestion(
                file_path="src/foo.py",
                description="Add validation",
                confidence=0.9,
                # No unified_diff
            ),
        ]
        tracker.save_session(suggestions, "fix bug")

        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value="+any_change"):
            outcomes, _ = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == OutcomeType.UNVERIFIED

    def test_suggested_diff_persisted_in_session(self, tracker):
        """Verify suggested_diff is saved and loaded in session records."""
        suggestions = [
            FakeSuggestion(
                file_path="src/foo.py",
                description="Fix",
                confidence=0.8,
                unified_diff="+new_code\n-old_code",
            ),
        ]
        tracker.save_session(suggestions, "fix")

        session = tracker._load_previous_session()
        assert session is not None
        assert session.suggestions[0]["suggested_diff"] == "+new_code\n-old_code"

    def test_independent_outcomes_rate_limited(self, tracker):
        """Independent outcomes are capped at MAX_INDEPENDENT_OUTCOMES."""
        from neo.memory.outcomes import MAX_INDEPENDENT_OUTCOMES

        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Fix", confidence=0.8),
        ]
        tracker.save_session(suggestions, "fix")

        # Simulate 20 independent files changing
        changed = {f"src/file{i}.py" for i in range(20)} | {"src/foo.py"}

        with patch.object(tracker, "_get_changed_files_since", return_value=changed), \
             patch.object(tracker, "_get_file_diff_since", return_value="+changes"):
            outcomes, _ = tracker.detect_outcomes()

        independent = [o for o in outcomes if o.outcome_type == OutcomeType.INDEPENDENT]
        assert len(independent) <= MAX_INDEPENDENT_OUTCOMES

    def test_independent_skipped_when_no_diff(self, tracker):
        """Independent changes with empty diffs are not recorded."""
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Fix", confidence=0.8),
        ]
        tracker.save_session(suggestions, "fix")

        changed = {"src/bar.py", "src/foo.py"}

        def fake_diff(path, ts, **_):
            if path == "src/bar.py":
                return ""  # No diff content
            return "+change"

        with patch.object(tracker, "_get_changed_files_since", return_value=changed), \
             patch.object(tracker, "_get_file_diff_since", side_effect=fake_diff):
            outcomes, _ = tracker.detect_outcomes()

        independent = [o for o in outcomes if o.outcome_type == OutcomeType.INDEPENDENT]
        assert len(independent) == 0  # bar.py skipped due to empty diff

    @staticmethod
    def _independent_outcomes(tracker, changed, sizes):
        """Run detection with a diff of ``sizes[path]`` characters per path,
        returning (kept paths, paths diffed). A size at or past the cap gets a
        capped summary, as `_get_file_diff_since` builds one."""
        from neo.memory.outcomes import DIFF_SUMMARY_CHARS, DIFF_TRUNCATION_MARKER

        diffed = []

        def fake_diff(path, ts, **_):
            diffed.append(path)
            size = sizes.get(path, 0)
            if size > DIFF_SUMMARY_CHARS:
                return "+" * DIFF_SUMMARY_CHARS + DIFF_TRUNCATION_MARKER
            return "+" * size

        tracker.save_session([FakeSuggestion(file_path="src/foo.py")], "fix")
        with patch.object(tracker, "_get_changed_files_since", return_value=set(changed)), \
             patch.object(tracker, "_get_file_diff_since", side_effect=fake_diff):
            outcomes, _ = tracker.detect_outcomes()
        kept = [o.file_path for o in outcomes if o.outcome_type == OutcomeType.INDEPENDENT]
        return kept, [p for p in diffed if p != "src/foo.py"]

    def test_independent_ranking_stops_once_the_kept_set_is_decided(self, tracker):
        """Diffing every changed file to keep five cost 5,795 `git` diffs (363 s)
        on one request in a busy repository. Once MAX_INDEPENDENT_OUTCOMES capped
        summaries are in hand in path order, no later path can displace one."""
        from neo.memory.outcomes import MAX_INDEPENDENT_OUTCOMES

        changed = [f"src/file{i:02d}.py" for i in range(40)]
        capped = {"src/file03.py", "src/file07.py", "src/file11.py",
                  "src/file19.py", "src/file23.py", "src/file31.py"}
        sizes = {p: (5000 if p in capped else 50) for p in changed}

        kept, diffed = self._independent_outcomes(tracker, changed, sizes)

        assert sorted(kept) == sorted(capped)[:MAX_INDEPENDENT_OUTCOMES]
        assert diffed == changed[:changed.index("src/file23.py") + 1]

    def test_independent_diffs_are_bounded_when_none_is_large(self, tracker):
        """The early stop needs large diffs; a codemod touching thousands of
        files lightly must still cost a bounded number of `git` diffs."""
        from neo.memory.outcomes import MAX_INDEPENDENT_DIFFS, MAX_INDEPENDENT_OUTCOMES

        changed = [f"src/file{i:04d}.py" for i in range(MAX_INDEPENDENT_DIFFS * 4)]
        kept, diffed = self._independent_outcomes(tracker, changed, {p: 40 for p in changed})

        assert diffed == changed[:MAX_INDEPENDENT_DIFFS]
        assert len(kept) == MAX_INDEPENDENT_OUTCOMES

    def test_a_real_oversized_diff_has_the_capped_length(self, tmp_path):
        """The early stop counts summaries of exactly DIFF_SUMMARY_FULL_LEN, so
        that length must be what `_get_file_diff_since` really produces."""
        import subprocess

        from neo.memory.outcomes import DIFF_SUMMARY_FULL_LEN

        def git(*args):
            subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

        git("init", "-q")
        git("config", "user.email", "t@t.t")
        git("config", "user.name", "T")
        (tmp_path / "big.py").write_text("x = 0\n")
        git("add", "-A")
        git("commit", "-qm", "init")
        (tmp_path / "big.py").write_text("".join(f"value_{i} = {i}\n" for i in range(2000)))
        tracker = OutcomeTracker(codebase_root=str(tmp_path), project_id="testproj1234")

        assert len(tracker._get_file_diff_since("big.py", time.time() - 60)) == DIFF_SUMMARY_FULL_LEN

    def test_independent_ranking_matches_diffing_everything(self, tmp_sessions_dir, tmp_path):
        """The early stop keeps exactly what ranking every diff would keep."""
        import random

        from neo.memory.outcomes import (
            DIFF_SUMMARY_CHARS, DIFF_TRUNCATION_MARKER, MAX_INDEPENDENT_OUTCOMES,
        )

        rng = random.Random(1234)
        for trial in range(60):
            with patch("neo.memory.outcomes.SESSIONS_DIR", tmp_sessions_dir / str(trial)):
                (tmp_sessions_dir / str(trial)).mkdir()
                tracker = OutcomeTracker(codebase_root=str(tmp_path), project_id="testproj1234")
                changed = [f"src/m{rng.randrange(1000):03d}.py" for _ in range(rng.randrange(1, 30))]
                sizes = {p: rng.choice([0, 10, 900, 1999, 2000, 2001, 9000]) for p in changed}

                kept, _ = self._independent_outcomes(tracker, changed, sizes)

            def summary_len(size):
                return DIFF_SUMMARY_CHARS + len(DIFF_TRUNCATION_MARKER) if size > DIFF_SUMMARY_CHARS else size

            reference = sorted(
                (p for p in set(changed) if sizes[p] > 0),
                key=lambda p: (-summary_len(sizes[p]), p),
            )[:MAX_INDEPENDENT_OUTCOMES]
            assert sorted(kept) == sorted(reference), (trial, sizes)


class TestCrossWorktreeAttribution:
    """A suggestion must be resolved against the root it was RECORDED under.

    Claude Code runs sessions in `.claude/worktrees/agent-*`. Those absolute
    paths never `relative_to()` the main checkout, so when the next neo run
    happened elsewhere in the same repo every suggestion silently failed to
    match and its learning was orphaned — 14 of 65 recorded sessions on a live
    install. `project_id` already spans worktrees and clones, so the session was
    found; only the path root was wrong.
    """

    def _session(self, recorded_root, file_path):
        return SessionRecord(
            timestamp=time.time() - 60,
            codebase_root=recorded_root,
            project_id="shared-pid",
            prompt="change the value",
            suggestions=[{
                "file_path": file_path,
                "description": "use a constant",
                "confidence": 0.8,
                "suggested_code": "x = 2\n",
                "suggested_diff": "",
            }],
        )

    def test_path_recorded_in_another_worktree_still_matches(self):
        tracker = OutcomeTracker(codebase_root="/repo/main")
        session = self._session("/repo/wt/agent-1", "/repo/wt/agent-1/src/a.py")
        outcomes = tracker._match_to_suggestions({"src/a.py"}, session)
        assert [o.file_path for o in outcomes] == ["src/a.py"]

    def test_same_root_behavior_unchanged(self):
        tracker = OutcomeTracker(codebase_root="/repo/main")
        session = self._session("/repo/main", "/repo/main/src/a.py")
        outcomes = tracker._match_to_suggestions({"src/a.py"}, session)
        assert [o.file_path for o in outcomes] == ["src/a.py"]

    def test_missing_recorded_root_falls_back_to_current(self):
        """Older session records predate the field; must not regress them."""
        tracker = OutcomeTracker(codebase_root="/repo/main")
        session = self._session("", "/repo/main/src/a.py")
        outcomes = tracker._match_to_suggestions({"src/a.py"}, session)
        assert [o.file_path for o in outcomes] == ["src/a.py"]

    def test_unrelated_repo_path_does_not_match(self):
        """Resolving against the recorded root must not become a wildcard."""
        tracker = OutcomeTracker(codebase_root="/repo/main")
        session = self._session("/other/project", "/other/project/src/a.py")
        assert tracker._match_to_suggestions({"src/b.py"}, session) == []


# The real case (neo, 2026-10-13): Neo's Stop review asked for the new
# `path.resolve()` to be guarded, Claude guarded it with its own comment, and
# the finding was recorded as MODIFIED, i.e. Neo contradicted.
_REVIEWED_NOTES = """\
def note_files(root, candidates, rules, found):
    for path in candidates:
        # A file in a named directory may itself be a symlink out.
        path = path.resolve()
        if (root in path.parents and path.suffix == ".md" and path.is_file()
                and path not in rules and path not in found):
            found.append(path)
    return found
"""
_NEO_FIX = """\
--- a/src/neo/notes.py
+++ b/src/neo/notes.py
@@ -2,6 +2,9 @@
     for path in candidates:
         # A file in a named directory may itself be a symlink out.
-        path = path.resolve()
+        try:
+            path = path.resolve()
+        except (OSError, RuntimeError):
+            continue
         if (root in path.parents and path.suffix == ".md" and path.is_file()
                 and path not in rules and path not in found):
             found.append(path)
"""
_CLAUDES_FIX = _REVIEWED_NOTES.replace(
    "        path = path.resolve()\n",
    "        try:\n"
    "            path = path.resolve()\n"
    "        except (OSError, RuntimeError):  # a symlink loop, say\n"
    "            continue\n",
)
# What the git matcher compared against: the file was new in its commit, so
# the diff is all additions and shares no change line with Neo's patch.
_COMMIT_DIFF = "\n".join("+" + line for line in _CLAUDES_FIX.splitlines())


class TestAddressedOutcome:
    def _review_then_edit(self, tracker, tmp_path, edited, *, reviewed=_REVIEWED_NOTES,
                          actual_diff=_COMMIT_DIFF):
        notes = tmp_path / "src" / "neo" / "notes.py"
        notes.parent.mkdir(parents=True, exist_ok=True)
        notes.write_text(reviewed)
        tracker.save_session(
            [FakeSuggestion(file_path=str(notes), description="Guard resolve()",
                            unified_diff=_NEO_FIX)],
            "review",
        )
        if edited is None:
            notes.unlink()
        else:
            notes.write_text(edited)
        with patch.object(tracker, "_get_changed_files_since",
                          return_value={"src/neo/notes.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value=actual_diff):
            outcomes, _ = tracker.detect_outcomes()
        assert len(outcomes) == 1
        return outcomes[0].outcome_type

    def test_flagged_code_rewritten_another_way_is_addressed(self, tracker, tmp_path):
        assert self._review_then_edit(tracker, tmp_path, _CLAUDES_FIX) == OutcomeType.ADDRESSED

    def test_change_elsewhere_in_the_file_stays_modified(self, tracker, tmp_path):
        elsewhere = _REVIEWED_NOTES.replace("    return found\n", "    return sorted(found)\n")
        assert self._review_then_edit(
            tracker, tmp_path, elsewhere, actual_diff="-    return found\n+    return sorted(found)"
        ) == OutcomeType.MODIFIED

    def test_hunk_the_file_never_held_cannot_read_as_addressed(self, tracker, tmp_path):
        """A patch written against another version of the file (or invented)
        targets code that was never there; its absence later proves nothing."""
        other_version = _REVIEWED_NOTES.replace("path = path.resolve()", "path = path.absolute()")
        assert self._review_then_edit(
            tracker, tmp_path, _CLAUDES_FIX, reviewed=other_version
        ) == OutcomeType.MODIFIED

    def test_deleted_file_stays_modified(self, tracker, tmp_path):
        assert self._review_then_edit(tracker, tmp_path, None) == OutcomeType.MODIFIED

    def test_session_records_the_flagged_region_not_the_fix(self, tracker, tmp_path):
        notes = tmp_path / "notes.py"
        notes.write_text(_REVIEWED_NOTES)
        tracker.save_session([FakeSuggestion(file_path="notes.py", unified_diff=_NEO_FIX)], "review")
        [region] = tracker._load_previous_session().suggestions[0]["flagged_regions"]
        assert region["span"] == [
            "# A file in a named directory may itself be a symlink out.",
            "path = path.resolve()",
            "if (root in path.parents and path.suffix == \".md\" and path.is_file()",
        ]
        assert region["outer"] == ["for path in candidates:", "found.append(path)"]

    def test_edit_to_outer_context_stays_modified(self, tracker, tmp_path):
        """A line Neo only quoted for context is not what it flagged."""
        nearby = _REVIEWED_NOTES.replace("found.append(path)", "found.append(path)  # keep order")
        assert self._review_then_edit(tracker, tmp_path, nearby) == OutcomeType.MODIFIED

    def test_function_deleted_stays_modified(self, tracker, tmp_path):
        assert self._review_then_edit(
            tracker, tmp_path, "def other():\n    return 1\n"
        ) == OutcomeType.MODIFIED

    def test_every_flagged_region_must_be_rewritten(self, tracker, tmp_path):
        reviewed = _REVIEWED_NOTES + "\n\ndef tail(x):\n    y = x\n    z = y\n    return z\n"
        two_hunks = _NEO_FIX + "@@ -12,3 +15,3 @@\n     y = x\n-    z = y\n+    z = y or 0\n     return z\n"
        notes = tmp_path / "src" / "neo" / "notes.py"
        notes.parent.mkdir(parents=True, exist_ok=True)
        notes.write_text(reviewed)
        tracker.save_session([FakeSuggestion(file_path=str(notes), unified_diff=two_hunks)], "r")
        assert len(tracker._load_previous_session().suggestions[0]["flagged_regions"]) == 2
        notes.write_text(_CLAUDES_FIX + "\n\ndef tail(x):\n    y = x\n    z = y\n    return z\n")
        with patch.object(tracker, "_get_changed_files_since", return_value={"src/neo/notes.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value=_COMMIT_DIFF):
            [outcome], _ = tracker.detect_outcomes()
        assert outcome.outcome_type == OutcomeType.MODIFIED

    def test_reads_the_checkout_the_suggestion_was_made_in(self, tmp_sessions_dir, tmp_path):
        """The reviewed code was uncommitted work in ITS checkout; the
        detecting run's checkout never held it, so reading there would call
        anything addressed."""
        review_root, other_root = tmp_path / "wt", tmp_path / "main"
        # The detecting checkout holds a rewrite; the reviewing one does not.
        for root, text in ((review_root, _REVIEWED_NOTES), (other_root, _CLAUDES_FIX)):
            (root / "src" / "neo").mkdir(parents=True)
            (root / "src" / "neo" / "notes.py").write_text(text)
        with patch("neo.memory.outcomes.SESSIONS_DIR", tmp_sessions_dir):
            reviewer = OutcomeTracker(codebase_root=str(review_root), project_id="shared1234")
            detector = OutcomeTracker(codebase_root=str(other_root), project_id="shared1234")
        reviewer.save_session([FakeSuggestion(file_path="src/neo/notes.py",
                                              unified_diff=_NEO_FIX)], "review")
        with patch.object(detector, "_get_changed_files_since", return_value={"src/neo/notes.py"}), \
             patch.object(detector, "_get_file_diff_since", return_value=_COMMIT_DIFF):
            [outcome], _ = detector.detect_outcomes()
        assert outcome.outcome_type == OutcomeType.MODIFIED

    def test_path_outside_the_checkout_is_not_read(self, tracker, tmp_path):
        outside = tmp_path.parent / f"{tmp_path.name}-outside" / "src" / "neo" / "notes.py"
        outside.parent.mkdir(parents=True)
        outside.write_text(_REVIEWED_NOTES)
        tracker.save_session([FakeSuggestion(file_path=str(outside), unified_diff=_NEO_FIX)], "r")
        assert tracker._load_previous_session().suggestions[0]["flagged_regions"] == []


class TestFlaggedRegions:
    def test_one_region_per_change_with_its_neighbours(self):
        from neo.memory.outcomes import flagged_regions
        diff = "@@ -1,4 +1,4 @@\n a = 1\n-b = 2\n+b = 3\n c = 4\n d = 5\n"
        assert flagged_regions(diff) == [
            {"span": ["a = 1", "b = 2", "c = 4"], "outer": ["d = 5"]},
        ]

    def test_pure_addition_targets_its_insertion_point(self):
        from neo.memory.outcomes import flagged_regions
        diff = "@@ -1,2 +1,3 @@\n a = 1\n+check()\n b = 2\n"
        assert flagged_regions(diff) == [{"span": ["a = 1", "b = 2"], "outer": []}]

    def test_removed_line_starting_with_dashes_is_not_a_header(self):
        from neo.memory.outcomes import flagged_regions
        diff = "--- a/q.sql\n+++ b/q.sql\n@@ -1,3 +1,2 @@\n SELECT 1;\n--- stale comment\n SELECT 2;\n"
        assert flagged_regions(diff, "q.sql") == [
            {"span": ["SELECT 1;", "-- stale comment", "SELECT 2;"], "outer": []},
        ]

    def test_no_newline_marker_does_not_end_the_hunk(self):
        from neo.memory.outcomes import flagged_regions
        diff = "@@ -1,3 +1,3 @@\n a = 1\n-b = 2\n\\ No newline at end of file\n+b = 3\n c = 4\n"
        assert flagged_regions(diff) == [{"span": ["a = 1", "b = 2", "c = 4"], "outer": []}]

    def test_hunks_for_another_file_are_skipped(self):
        from neo.memory.outcomes import flagged_regions
        diff = ("--- a/a.py\n+++ b/a.py\n@@ -1,3 +1,3 @@\n x = 1\n-y = 2\n+y = 3\n z = 4\n"
                "--- a/b.py\n+++ b/b.py\n@@ -1,2 +1,1 @@\n-import os\n keep = 1\n")
        assert flagged_regions(diff, "/repo/a.py") == [
            {"span": ["x = 1", "y = 2", "z = 4"], "outer": []},
        ]

    def test_region_must_be_consecutive(self):
        from neo.memory.outcomes import region_present
        assert region_present(["a", "b"], ["a", "b", "c"])
        assert not region_present(["a", "c"], ["a", "b", "c"])
        assert not region_present([], ["a"])
