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

    def test_missing_session_returns_none(self, tracker):
        assert tracker._load_previous_session() is None


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


class TestIngestGitHistory:
    def test_ingests_commits(self, tracker):
        git_log_output = (
            "abc123def456\t1773000000\tfeat: add user validation\n"
            "src/auth.py\n"
            "src/models.py\n"
            "\n"
            "def789abc012\t1772900000\tfix: handle null email in signup\n"
            "src/auth.py\n"
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
        git_log_output = (
            "abc123\t1773000000\tMerge pull request #42 from feature\n"
            "src/foo.py\n"
            "\n"
            "def456\t1772900000\tfix: real bug fix\n"
            "src/bar.py\n"
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
        git_log_output = "abc123\t1773000000\twip\nsrc/foo.py\n"

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
        git_log_output = "abc123\t1773000000\tfeat: add something useful here\nsrc/foo.py\n"

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

        def fake_diff(path, ts):
            if path == "src/bar.py":
                return ""  # No diff content
            return "+change"

        with patch.object(tracker, "_get_changed_files_since", return_value=changed), \
             patch.object(tracker, "_get_file_diff_since", side_effect=fake_diff):
            outcomes, _ = tracker.detect_outcomes()

        independent = [o for o in outcomes if o.outcome_type == OutcomeType.INDEPENDENT]
        assert len(independent) == 0  # bar.py skipped due to empty diff
