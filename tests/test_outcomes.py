"""Tests for neo.memory.outcomes - Outcome-based learning."""

import json
import time
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from neo.memory.outcomes import OutcomeTracker, SessionRecord


@dataclass
class FakeSuggestion:
    """Minimal CodeSuggestion stand-in for tests."""
    file_path: str = ""
    description: str = ""
    confidence: float = 0.8


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


class TestDetectOutcomesAccepted:
    def test_accepted_when_suggested_file_changed(self, tracker):
        # Save a session with suggestions
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Add validation", confidence=0.9),
        ]
        tracker.save_session(suggestions, "fix bug")

        # Mock git to return the same file as changed, with diff content
        with patch.object(tracker, "_get_changed_files_since", return_value={"src/foo.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value="+    validate_input(data)\n-    process(data)"):
            outcomes = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == "accepted"
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
            outcomes = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == "independent"
        assert "new_helper" in outcomes[0].diff_summary


class TestDetectOutcomesNoChanges:
    def test_no_outcomes_when_no_changes(self, tracker):
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Add validation"),
        ]
        tracker.save_session(suggestions, "fix bug")

        with patch.object(tracker, "_get_changed_files_since", return_value=set()):
            outcomes = tracker.detect_outcomes()

        assert outcomes == []

    def test_no_outcomes_when_no_previous_session(self, tracker):
        outcomes = tracker.detect_outcomes()
        assert outcomes == []


class TestDetectOutcomesIndependent:
    def test_independent_changes_detected(self, tracker):
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Fix foo"),
        ]
        tracker.save_session(suggestions, "fix bug")

        # User changed a different file
        with patch.object(tracker, "_get_changed_files_since", return_value={"src/bar.py"}), \
             patch.object(tracker, "_get_file_diff_since", return_value="+new code"):
            outcomes = tracker.detect_outcomes()

        assert len(outcomes) == 1
        assert outcomes[0].outcome_type == "independent"
        assert outcomes[0].file_path == "src/bar.py"

    def test_non_code_files_ignored(self, tracker):
        suggestions = [FakeSuggestion(file_path="src/foo.py")]
        tracker.save_session(suggestions, "fix")

        with patch.object(
            tracker,
            "_get_changed_files_since",
            return_value={"README.md", "config.yaml", ".gitignore"},
        ):
            outcomes = tracker.detect_outcomes()

        assert outcomes == []

    def test_mixed_accepted_and_independent(self, tracker):
        suggestions = [
            FakeSuggestion(file_path="src/foo.py", description="Fix foo"),
        ]
        tracker.save_session(suggestions, "fix")

        with patch.object(
            tracker,
            "_get_changed_files_since",
            return_value={"src/foo.py", "src/bar.py"},
        ), patch.object(tracker, "_get_file_diff_since", return_value="+change"):
            outcomes = tracker.detect_outcomes()

        types = {o.outcome_type for o in outcomes}
        assert "accepted" in types
        assert "independent" in types


class TestGitNotAvailable:
    def test_graceful_when_git_missing(self, tracker):
        suggestions = [FakeSuggestion(file_path="src/foo.py")]
        tracker.save_session(suggestions, "fix")

        with patch("neo.memory.outcomes.subprocess.run", side_effect=FileNotFoundError):
            outcomes = tracker.detect_outcomes()

        assert outcomes == []

    def test_graceful_when_not_git_repo(self, tracker):
        suggestions = [FakeSuggestion(file_path="src/foo.py")]
        tracker.save_session(suggestions, "fix")

        import subprocess
        with patch(
            "neo.memory.outcomes.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git"),
        ):
            outcomes = tracker.detect_outcomes()

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

        outcomes = tracker.detect_outcomes()
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
