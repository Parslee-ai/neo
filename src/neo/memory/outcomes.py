"""
Outcome-based learning for Neo.

Two learning modes:
1. Session-based: Tracks what the user did after neo's suggestions (between invocations)
2. History-based: Ingests git commit history to learn from past code evolution

Session data is persisted to ~/.neo/sessions/ so outcomes can be detected
across invocations. History watermarks are persisted to avoid re-ingesting.
"""

import datetime
import enum
import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SESSIONS_DIR = Path.home() / ".neo" / "sessions"

# Cap independent outcomes per session to avoid flooding facts with noise.
# This is the first layer of defense; store.py's MAX_INDEPENDENT_FACTS (50)
# caps the total across sessions. Together: 5/session * ~10 sessions before
# cap kicks in, then oldest are invalidated.
MAX_INDEPENDENT_OUTCOMES = 5


class OutcomeType(enum.Enum):
    """Classification of what the user did after a neo suggestion."""
    ACCEPTED = "accepted"       # User applied the suggestion (diff overlap > 0.3)
    MODIFIED = "modified"       # User changed the file differently (overlap <= 0.3)
    UNVERIFIED = "unverified"   # File changed but no diff to compare
    INDEPENDENT = "independent" # User changed a file neo didn't suggest


@dataclass
class SuggestionRecord:
    """Minimal record of a suggestion for outcome matching."""
    file_path: str
    description: str
    confidence: float = 0.0


@dataclass
class SessionRecord:
    """Persisted state from a neo invocation, used for outcome detection on next run."""
    timestamp: float = 0.0
    codebase_root: str = ""
    project_id: str = ""
    prompt: str = ""
    suggestions: list[dict] = field(default_factory=list)
    suggestion_fact_ids: dict[str, str] = field(default_factory=dict)


@dataclass
class Outcome:
    """A detected outcome from comparing suggestions to actual changes."""
    outcome_type: OutcomeType
    file_path: str
    diff_summary: str = ""  # actual git diff content (truncated)
    suggestion_description: str = ""  # empty for independent changes
    suggestion_confidence: float = 0.0


class OutcomeTracker:
    """Detects what the user actually did after neo's suggestions.

    On each invocation:
    1. Load previous session (if any) for this project
    2. Run git diff to see what changed since last session
    3. Match changes against previous suggestions
    4. Return outcomes for fact creation
    5. Save current session for next invocation
    """

    def __init__(self, codebase_root: Optional[str] = None, project_id: str = ""):
        self.codebase_root = codebase_root
        self.project_id = project_id
        self._session_path = self._get_session_path()

    def _get_session_path(self) -> Optional[Path]:
        if not self.project_id:
            return None
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        return SESSIONS_DIR / f"session_{self.project_id}.json"

    def save_session(
        self,
        suggestions: list,
        prompt: str,
        suggestion_fact_ids: Optional[dict[str, str]] = None,
    ) -> None:
        """Persist current session for outcome detection on next run.

        Args:
            suggestions: List of CodeSuggestion objects from current invocation.
            prompt: The user's prompt.
            suggestion_fact_ids: Mapping of file_path -> fact_id for linking outcomes.
        """
        if not self._session_path:
            logger.warning(
                "save_session: skipped — no project_id (codebase_root not set). "
                "Memory will NOT persist. Pass --cwd or working_directory in JSON input."
            )
            return

        records = []
        for s in suggestions:
            file_path = getattr(s, "file_path", "")
            if not file_path or file_path in ("/", "N/A"):
                continue
            records.append({
                "file_path": file_path,
                "description": getattr(s, "description", "")[:500],
                "confidence": getattr(s, "confidence", 0.0),
                "suggested_diff": getattr(s, "unified_diff", "")[:2000],
            })

        session = SessionRecord(
            timestamp=time.time(),
            codebase_root=self.codebase_root or "",
            project_id=self.project_id,
            prompt=prompt[:200],
            suggestions=records,
            suggestion_fact_ids=suggestion_fact_ids or {},
        )

        try:
            with open(self._session_path, "w") as f:
                json.dump(asdict(session), f, indent=2)
            logger.info(
                f"Saved session: {len(records)} suggestion(s), "
                f"{len(suggestion_fact_ids or {})} linked fact(s)"
            )
        except OSError as e:
            logger.warning(f"Failed to save session: {e}")

    def detect_outcomes(self) -> tuple[list[Outcome], dict[str, str]]:
        """Detect outcomes by comparing previous suggestions to actual git changes.

        Returns:
            Tuple of (outcomes list, suggestion_fact_ids mapping from previous session).
        """
        prev = self._load_previous_session()
        if not prev or not prev.timestamp:
            logger.debug("No previous session found for outcome detection")
            return [], {}

        # Only detect outcomes for the same project
        if prev.project_id != self.project_id:
            logger.debug(f"Previous session project mismatch: {prev.project_id} != {self.project_id}")
            return [], {}

        # Get files changed since last session
        changed_files = self._get_changed_files_since(prev.timestamp)
        if not changed_files:
            logger.debug("No files changed since last session")
            return [], {}

        suggested = [s.get("file_path", "") for s in prev.suggestions]
        logger.info(
            f"Outcome detection: {len(changed_files)} changed files, "
            f"{len(suggested)} suggestions, "
            f"{len(prev.suggestion_fact_ids)} linked fact(s)"
        )

        outcomes = self._match_to_suggestions(changed_files, prev)
        accepted = sum(1 for o in outcomes if o.outcome_type == OutcomeType.ACCEPTED)
        modified = sum(1 for o in outcomes if o.outcome_type == OutcomeType.MODIFIED)
        independent = sum(1 for o in outcomes if o.outcome_type == OutcomeType.INDEPENDENT)
        logger.info(f"Outcomes: {accepted} accepted, {modified} modified, {independent} independent")
        return outcomes, prev.suggestion_fact_ids

    def _load_previous_session(self) -> Optional[SessionRecord]:
        """Load the previous session record from disk."""
        if not self._session_path or not self._session_path.exists():
            return None
        try:
            with open(self._session_path) as f:
                data = json.load(f)
            return SessionRecord(**data)
        except (json.JSONDecodeError, OSError, TypeError) as e:
            logger.warning(f"Failed to load previous session: {e}")
            return None

    def _get_changed_files_since(self, since_timestamp: float) -> set[str]:
        """Get files that changed in git since a timestamp.

        Uses git log --since with ISO timestamp for reliable cross-platform behavior.
        """
        if not self.codebase_root:
            return set()

        try:
            # Convert timestamp to ISO format for git
            since_iso = datetime.datetime.fromtimestamp(
                since_timestamp, tz=datetime.timezone.utc
            ).isoformat()

            # Get committed changes since timestamp
            result = subprocess.run(
                ["git", "log", "--since", since_iso, "--name-only", "--pretty=format:"],
                cwd=self.codebase_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            committed = set()
            if result.returncode == 0:
                committed = {
                    line.strip()
                    for line in result.stdout.strip().split("\n")
                    if line.strip()
                }

            # Also get currently staged/unstaged changes
            result2 = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=self.codebase_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            working = set()
            if result2.returncode == 0:
                working = {
                    line.strip()
                    for line in result2.stdout.strip().split("\n")
                    if line.strip()
                }

            return committed | working

        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.debug(f"Git diff failed (non-fatal): {e}")
            return set()

    def _match_to_suggestions(
        self, changed_files: set[str], session: SessionRecord
    ) -> list[Outcome]:
        """Match changed files against previous suggestions to determine outcomes."""
        outcomes: list[Outcome] = []
        suggested_files: set[str] = set()

        for sugg in session.suggestions:
            file_path = sugg.get("file_path", "")
            if not file_path:
                continue

            # Normalize: suggestions may have absolute paths, git diff has relative
            normalized = self._normalize_path(file_path)
            suggested_files.add(normalized)

            if normalized in changed_files or file_path in changed_files:
                diff = self._get_file_diff_since(normalized, session.timestamp)
                suggested_diff = sugg.get("suggested_diff", "")

                # Determine if user applied our suggestion or did something different
                if suggested_diff and diff:
                    overlap = self._compute_diff_overlap(suggested_diff, diff)
                    outcome_type = OutcomeType.ACCEPTED if overlap > 0.3 else OutcomeType.MODIFIED
                else:
                    # Missing suggested_diff or actual diff — can't verify
                    outcome_type = OutcomeType.UNVERIFIED

                outcomes.append(Outcome(
                    outcome_type=outcome_type,
                    file_path=normalized,
                    diff_summary=diff,
                    suggestion_description=sugg.get("description", ""),
                    suggestion_confidence=sugg.get("confidence", 0.0),
                ))

        # Detect independent changes (user changed files neo didn't suggest).
        # Rate-limited: keep only the top MAX_INDEPENDENT_OUTCOMES by diff size
        # to avoid flooding facts with low-value noise in active repos.
        independent_candidates: list[Outcome] = []
        for changed in changed_files:
            normalized = self._normalize_path(changed)
            if normalized not in suggested_files and changed not in suggested_files:
                if self._is_code_file(changed):
                    diff = self._get_file_diff_since(changed, session.timestamp)
                    if not diff:
                        continue  # No diff content = no learning signal
                    independent_candidates.append(Outcome(
                        outcome_type=OutcomeType.INDEPENDENT,
                        file_path=changed,
                        diff_summary=diff,
                    ))

        # Keep only the most informative independent changes (deterministic: size desc, path asc)
        independent_candidates.sort(key=lambda o: (-len(o.diff_summary), o.file_path))
        outcomes.extend(independent_candidates[:MAX_INDEPENDENT_OUTCOMES])

        return outcomes

    def _get_file_diff_since(self, file_path: str, since_timestamp: float) -> str:
        """Get the actual diff content for a file since a timestamp.

        Returns a truncated diff summary (max 2000 chars) showing what changed.
        Tries committed diff first, falls back to working tree diff.
        """
        if not self.codebase_root:
            return ""

        MAX_DIFF_CHARS = 2000

        try:
            since_iso = datetime.datetime.fromtimestamp(
                since_timestamp, tz=datetime.timezone.utc
            ).isoformat()

            # Try committed changes first
            result = subprocess.run(
                ["git", "log", "--since", since_iso, "-p", "--", file_path],
                cwd=self.codebase_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            diff = ""
            if result.returncode == 0 and result.stdout.strip():
                diff = result.stdout.strip()

            # Also check working tree changes
            result2 = subprocess.run(
                ["git", "diff", "HEAD", "--", file_path],
                cwd=self.codebase_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result2.returncode == 0 and result2.stdout.strip():
                if diff:
                    diff += "\n" + result2.stdout.strip()
                else:
                    diff = result2.stdout.strip()

            if not diff:
                return ""

            # Extract meaningful parts: headers, hunks, and change lines
            summary_lines = [
                line for line in diff.split("\n")
                if line.startswith(("+++", "---", "@@"))
                or (line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
            ]
            summary = "\n".join(summary_lines)
            if len(summary) > MAX_DIFF_CHARS:
                summary = summary[:MAX_DIFF_CHARS] + "\n... (truncated)"

            return summary

        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.debug(f"File diff failed for {file_path} (non-fatal): {e}")
            return ""

    def _normalize_path(self, path: str) -> str:
        """Normalize a file path to relative form for comparison."""
        if not self.codebase_root:
            return path
        try:
            p = Path(path)
            root = Path(self.codebase_root)
            if p.is_absolute():
                return str(p.relative_to(root))
        except (ValueError, TypeError):
            pass
        return path

    @staticmethod
    def _compute_diff_overlap(suggested: str, actual: str) -> float:
        """Compute line-level overlap between suggested and actual diffs.

        Returns 0.0-1.0 where 1.0 means identical changes.
        Preserves the +/- prefix so additions and removals of the same
        content are not conflated.
        """
        def extract_change_lines(diff_text: str) -> set[str]:
            return {
                line.strip()
                for line in diff_text.split("\n")
                if line.strip()
                and line.strip()[0] in ("+", "-")
                and not line.strip().startswith(("+++", "---", "@@"))
            }

        suggested_lines = extract_change_lines(suggested)
        actual_lines = extract_change_lines(actual)

        if not suggested_lines and not actual_lines:
            return 1.0
        if not suggested_lines or not actual_lines:
            return 0.0

        intersection = suggested_lines & actual_lines
        union = suggested_lines | actual_lines
        return len(intersection) / len(union) if union else 0.0

    # ------------------------------------------------------------------ #
    # Git history ingestion
    # ------------------------------------------------------------------ #

    def ingest_git_history(self, max_commits: int = 50) -> list[dict]:
        """Learn from git commit history that hasn't been ingested yet.

        Reads commits since the last ingestion watermark (or last 50 commits
        on first run). For each commit, extracts the commit message, changed
        files, and diff summary. Returns structured records ready for fact
        creation.

        Args:
            max_commits: Maximum number of commits to ingest per run.

        Returns:
            List of dicts with keys: subject, body, commit_hash, timestamp.
        """
        if not self.codebase_root:
            return []

        watermark = self._load_watermark()
        commits = self._get_commits_since(watermark, max_commits)

        if not commits:
            return []

        records = []
        for commit in commits:
            if not self._is_meaningful_commit(commit["message"]):
                continue

            diff = self._get_commit_diff(commit["hash"])
            if not diff:
                continue

            # Build a learnable record from the commit
            subject = f"history:{commit['hash'][:8]} {commit['message'][:60]}"
            body_parts = [
                f"Commit: {commit['hash'][:12]}",
                f"Message: {commit['message']}",
                f"Files: {', '.join(commit['files'][:10])}",
            ]
            if diff:
                body_parts.append(f"Changes:\n{diff}")

            records.append({
                "subject": subject,
                "body": "\n".join(body_parts),
                "commit_hash": commit["hash"],
                "timestamp": commit["timestamp"],
            })

        # Update watermark to most recent commit
        if commits:
            self._save_watermark(commits[0]["hash"])

        logger.info(f"Ingested {len(records)} commits from git history")
        return records

    def _get_commits_since(
        self, since_hash: Optional[str], max_commits: int
    ) -> list[dict]:
        """Get commit metadata since a watermark hash.

        Returns commits in reverse chronological order (newest first).
        """
        if not self.codebase_root:
            return []

        try:
            # Build git log command
            cmd = [
                "git", "log",
                f"-{max_commits}",
                "--pretty=format:%H\t%at\t%s",  # hash, timestamp, subject
                "--name-only",
            ]
            if since_hash:
                cmd.append(f"{since_hash}..HEAD")

            result = subprocess.run(
                cmd,
                cwd=self.codebase_root,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []

            # Parse: each commit block is header line + file lines + blank line
            commits = []
            current = None
            for line in result.stdout.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                if "\t" in line and len(line.split("\t")) >= 3:
                    # This is a header line: hash\ttimestamp\tsubject
                    parts = line.split("\t", 2)
                    current = {
                        "hash": parts[0],
                        "timestamp": float(parts[1]),
                        "message": parts[2],
                        "files": [],
                    }
                    commits.append(current)
                elif current is not None:
                    # This is a file path
                    current["files"].append(line)

            return commits

        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.debug(f"Git log failed (non-fatal): {e}")
            return []

    def _get_commit_diff(self, commit_hash: str) -> str:
        """Get the diff for a specific commit, filtered to code files only."""
        if not self.codebase_root:
            return ""

        MAX_DIFF_CHARS = 2000

        try:
            result = subprocess.run(
                ["git", "show", "--stat", "--patch", "--format=", commit_hash],
                cwd=self.codebase_root,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return ""

            # Extract meaningful parts: headers, hunks, and change lines
            summary_lines = [
                line for line in result.stdout.split("\n")
                if line.startswith(("+++", "---", "@@"))
                or (line.startswith(("+", "-")) and not line.startswith(("+++", "---")))
            ]
            summary = "\n".join(summary_lines)
            if len(summary) > MAX_DIFF_CHARS:
                summary = summary[:MAX_DIFF_CHARS] + "\n... (truncated)"
            return summary

        except (subprocess.SubprocessError, FileNotFoundError, OSError) as e:
            logger.debug(f"Commit diff failed for {commit_hash} (non-fatal): {e}")
            return ""

    @staticmethod
    def _is_meaningful_commit(message: str) -> bool:
        """Filter out commits that aren't useful for learning.

        Skip merge commits, version bumps, and auto-generated commits.
        Keep: bug fixes, features, refactors, and anything with substance.
        """
        msg = message.lower().strip()

        # Skip noise
        skip_prefixes = (
            "merge ", "merge pull request", "merge branch",
            "bump version", "release v", "update changelog",
            "chore(deps)", "chore(release)",
            "initial commit",
        )
        if any(msg.startswith(p) for p in skip_prefixes):
            return False

        # Skip very short messages (likely not informative)
        if len(msg) < 10:
            return False

        return True

    def _load_watermark(self) -> Optional[str]:
        """Load the last-ingested commit hash for this project."""
        watermark_path = self._get_watermark_path()
        if not watermark_path or not watermark_path.exists():
            return None
        try:
            data = json.loads(watermark_path.read_text())
            return data.get("last_commit_hash")
        except (json.JSONDecodeError, OSError):
            return None

    def _save_watermark(self, commit_hash: str) -> None:
        """Save the last-ingested commit hash."""
        watermark_path = self._get_watermark_path()
        if not watermark_path:
            return
        try:
            watermark_path.write_text(json.dumps({
                "last_commit_hash": commit_hash,
                "updated_at": time.time(),
            }))
        except OSError as e:
            logger.debug(f"Failed to save watermark: {e}")

    def _get_watermark_path(self) -> Optional[Path]:
        if not self.project_id:
            return None
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        return SESSIONS_DIR / f"watermark_{self.project_id}.json"

    @staticmethod
    def _is_code_file(path: str) -> bool:
        """Check if a file looks like source code (not config, docs, etc.)."""
        code_extensions = {
            ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
            ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".swift", ".kt",
            ".scala", ".sql", ".sh", ".bash", ".zsh",
        }
        return Path(path).suffix.lower() in code_extensions
