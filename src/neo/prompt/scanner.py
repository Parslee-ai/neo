"""
Scanner module for reading Claude Code data sources.

Reads and parses:
- ~/.claude/history.jsonl - prompt history with timestamps and projects
- ~/.claude/projects/{path}/*.jsonl - session message histories
- CLAUDE.md files - global and project-specific instructions
- ~/.claude/commands/ - slash command definitions
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClaudeCodeSources:
    """Paths to Claude Code data sources."""

    claude_home: Path = field(default_factory=lambda: Path.home() / ".claude")

    @property
    def history_file(self) -> Path:
        """Path to the global history.jsonl file."""
        return self.claude_home / "history.jsonl"

    @property
    def projects_dir(self) -> Path:
        """Path to the projects directory containing session files."""
        return self.claude_home / "projects"

    @property
    def plans_dir(self) -> Path:
        """Path to the plans directory."""
        return self.claude_home / "plans"

    @property
    def global_claude_md(self) -> Path:
        """Path to the global CLAUDE.md file."""
        return self.claude_home / "CLAUDE.md"

    @property
    def commands_dir(self) -> Path:
        """Path to the global commands directory."""
        return self.claude_home / "commands"

    def find_project_claude_mds(self) -> list[Path]:
        """
        Find all project CLAUDE.md files.

        Searches common project locations and tracked project paths.

        Returns:
            List of paths to discovered CLAUDE.md files
        """
        found = []

        # Add global CLAUDE.md if it exists
        if self.global_claude_md.exists():
            found.append(self.global_claude_md)

        # Search common project directories
        search_patterns = [
            Path.home() / "git",
            Path.home() / "projects",
            Path.home() / "code",
            Path.home() / "src",
            Path.home() / "work",
        ]

        for base_dir in search_patterns:
            if base_dir.exists():
                # Search one level deep for CLAUDE.md
                for project_dir in base_dir.iterdir():
                    if project_dir.is_dir():
                        claude_md = project_dir / "CLAUDE.md"
                        if claude_md.exists():
                            found.append(claude_md)

        # Also check projects tracked by Claude Code
        if self.projects_dir.exists():
            for project_path_dir in self.projects_dir.iterdir():
                if project_path_dir.is_dir():
                    # Directory name is path with dashes (e.g., -Users-name-git-repo)
                    real_path = _decode_project_path(project_path_dir.name)
                    if real_path:
                        claude_md = Path(real_path) / "CLAUDE.md"
                        if claude_md.exists() and claude_md not in found:
                            found.append(claude_md)

        return found


@dataclass
class ScannedPrompt:
    """A prompt extracted from Claude Code history."""

    text: str
    timestamp: datetime
    project: str
    session_id: str
    source: Literal["history", "session", "claude_md", "command"]

    def __hash__(self) -> int:
        """Hash based on text and timestamp for deduplication."""
        return hash((self.text, self.timestamp.isoformat()))


@dataclass
class HistoryScanResult:
    """Result of scanning history.jsonl with position metadata."""

    prompts: list[ScannedPrompt]
    last_line_number: int  # Actual line number in the file (1-indexed)


@dataclass
class ScannedSession:
    """A complete conversation session."""

    session_id: str
    project: str
    messages: list[dict]
    start_time: datetime
    end_time: datetime
    tool_calls: int
    errors: list[str]
    outcome: Optional[str]  # Inferred outcome


@dataclass
class ScannedClaudeMd:
    """A CLAUDE.md file with metadata."""

    path: Path
    content: str
    last_modified: datetime
    hash: str  # SHA256 hash for change detection

    @classmethod
    def from_path(cls, path: Path) -> Optional["ScannedClaudeMd"]:
        """
        Create a ScannedClaudeMd from a file path.

        Args:
            path: Path to the CLAUDE.md file

        Returns:
            ScannedClaudeMd instance or None if file cannot be read
        """
        try:
            content = path.read_text(encoding="utf-8")
            stat = path.stat()
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            return cls(
                path=path,
                content=content,
                last_modified=datetime.fromtimestamp(stat.st_mtime),
                hash=content_hash,
            )
        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to read CLAUDE.md at {path}: {e}")
            return None


class Scanner:
    """
    Scans Claude Code data sources for prompts, sessions, and configurations.

    Provides methods to read and parse:
    - Prompt history from history.jsonl
    - Session transcripts from project .jsonl files
    - CLAUDE.md files (global and project-specific)
    - Slash command definitions
    """

    def __init__(self, sources: Optional[ClaudeCodeSources] = None):
        """
        Initialize scanner with data source paths.

        Args:
            sources: ClaudeCodeSources instance (uses default paths if None)
        """
        self.sources = sources or ClaudeCodeSources()

    def scan_history(
        self, since: Optional[datetime] = None, since_line: Optional[int] = None
    ) -> HistoryScanResult:
        """
        Scan history.jsonl for prompts.

        The history file contains one JSON object per line with format:
        {"display": "prompt text", "timestamp": epoch_ms, "project": "/path"}

        Args:
            since: Only return prompts after this datetime (None for all)
            since_line: Only return prompts after this line number (1-indexed).
                       If provided, starts reading from this line. More efficient
                       than timestamp filtering for incremental scans.

        Returns:
            HistoryScanResult containing prompts and the last line number processed
        """
        prompts: list[ScannedPrompt] = []
        history_file = self.sources.history_file
        last_line_number = 0

        if not history_file.exists():
            logger.debug(f"History file not found: {history_file}")
            return HistoryScanResult(prompts=prompts, last_line_number=0)

        try:
            with open(history_file, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    last_line_number = line_num

                    # Skip lines before since_line if provided
                    if since_line is not None and line_num <= since_line:
                        continue

                    line = line.strip()
                    if not line:
                        continue

                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Invalid JSON on line {line_num}: {e}")
                        continue

                    # Extract prompt data
                    display = entry.get("display", "")
                    timestamp_ms = entry.get("timestamp")
                    project = entry.get("project", "")

                    if not display or not timestamp_ms:
                        continue

                    # Validate timestamp range before conversion
                    # Reasonable range: year 2000 to 1 year from now
                    min_timestamp_ms = 946684800000  # 2000-01-01 00:00:00 UTC
                    max_timestamp_ms = (datetime.now().timestamp() + 365 * 24 * 60 * 60) * 1000

                    if timestamp_ms < min_timestamp_ms or timestamp_ms > max_timestamp_ms:
                        logger.warning(
                            f"Invalid timestamp {timestamp_ms} on line {line_num}: "
                            f"outside valid range (2000-01-01 to 1 year from now)"
                        )
                        continue

                    # Convert millisecond epoch to datetime
                    timestamp = datetime.fromtimestamp(timestamp_ms / 1000)

                    # Filter by since datetime (secondary filter if since_line not provided)
                    if since and timestamp < since:
                        continue

                    prompts.append(
                        ScannedPrompt(
                            text=display,
                            timestamp=timestamp,
                            project=project,
                            session_id="",  # History doesn't track session IDs
                            source="history",
                        )
                    )

        except (OSError, UnicodeDecodeError) as e:
            logger.error(f"Failed to read history file: {e}")

        # Sort by timestamp ascending
        prompts.sort(key=lambda p: p.timestamp)
        return HistoryScanResult(prompts=prompts, last_line_number=last_line_number)

    def scan_sessions(
        self, project: Optional[str] = None, since: Optional[datetime] = None
    ) -> list[ScannedSession]:
        """
        Scan project session files for full conversations.

        Session files are stored in ~/.claude/projects/{encoded_path}/*.jsonl
        Each line is a message with format:
        {"type": "user"|"assistant"|"system", "content": "...", "timestamp": "ISO8601", ...}

        Args:
            project: Filter to specific project path (None for all projects)
            since: Only return sessions modified after this datetime

        Returns:
            List of ScannedSession objects
        """
        sessions = []
        projects_dir = self.sources.projects_dir

        if not projects_dir.exists():
            logger.debug(f"Projects directory not found: {projects_dir}")
            return sessions

        # Find project directories to scan
        project_dirs = []
        if project:
            # Look for specific project
            encoded_name = _encode_project_path(project)
            project_path = projects_dir / encoded_name
            if project_path.exists():
                project_dirs.append((project_path, project))
        else:
            # Scan all project directories
            for path_dir in projects_dir.iterdir():
                if path_dir.is_dir():
                    real_path = _decode_project_path(path_dir.name)
                    if real_path:
                        project_dirs.append((path_dir, real_path))

        # Scan each project directory
        for project_dir, project_path in project_dirs:
            for session_file in project_dir.glob("*.jsonl"):
                # Check modification time filter
                if since:
                    mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                    if mtime < since:
                        continue

                session = self._parse_session_file(session_file, project_path)
                if session:
                    sessions.append(session)

        # Sort by start time
        sessions.sort(key=lambda s: s.start_time)
        return sessions

    def _parse_session_file(
        self, session_file: Path, project: str
    ) -> Optional[ScannedSession]:
        """
        Parse a single session .jsonl file.

        Args:
            session_file: Path to the session file
            project: Project path this session belongs to

        Returns:
            ScannedSession or None if parsing fails
        """
        messages = []
        errors = []
        tool_calls = 0
        timestamps = []

        try:
            with open(session_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    messages.append(msg)

                    # Extract timestamp
                    ts_str = msg.get("timestamp")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                            timestamps.append(ts)
                        except ValueError:
                            pass

                    # Count tool calls
                    if msg.get("type") == "tool" or msg.get("subtype") in (
                        "tool_use",
                        "tool_result",
                    ):
                        tool_calls += 1

                    # Collect errors
                    if msg.get("level") == "error" or "error" in msg.get(
                        "subtype", ""
                    ).lower():
                        content = msg.get("content", "")
                        if content:
                            errors.append(content[:200])  # Truncate long errors

        except (OSError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to parse session file {session_file}: {e}")
            return None

        if not messages:
            return None

        # Extract session ID from filename
        session_id = session_file.stem

        # Determine time range
        if timestamps:
            start_time = min(timestamps)
            end_time = max(timestamps)
        else:
            # Fall back to file modification time
            stat = session_file.stat()
            end_time = datetime.fromtimestamp(stat.st_mtime)
            start_time = end_time  # Unknown start

        # Infer outcome from final messages
        outcome = self._infer_outcome(messages)

        return ScannedSession(
            session_id=session_id,
            project=project,
            messages=messages,
            start_time=start_time,
            end_time=end_time,
            tool_calls=tool_calls,
            errors=errors,
            outcome=outcome,
        )

    def _infer_outcome(self, messages: list[dict]) -> Optional[str]:
        """
        Infer session outcome from messages.

        Looks for success/failure patterns in the last few messages.

        Args:
            messages: List of message dictionaries

        Returns:
            Outcome string or None if cannot be determined
        """
        if not messages:
            return None

        # Check last 5 messages for outcome signals
        last_messages = messages[-5:]
        combined_content = ""

        for msg in last_messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                combined_content += content.lower() + " "

        # Success patterns
        success_patterns = [
            "successfully",
            "completed",
            "done",
            "finished",
            "created",
            "implemented",
            "fixed",
        ]

        # Failure patterns
        failure_patterns = [
            "error",
            "failed",
            "cannot",
            "unable",
            "blocked",
        ]

        success_count = sum(1 for p in success_patterns if p in combined_content)
        failure_count = sum(1 for p in failure_patterns if p in combined_content)

        if success_count > failure_count and success_count > 0:
            return "success"
        elif failure_count > success_count and failure_count > 0:
            return "failure"

        return None

    def scan_claude_mds(self) -> list[ScannedClaudeMd]:
        """
        Find and scan all CLAUDE.md files.

        Searches for both the global CLAUDE.md and project-specific ones.

        Returns:
            List of ScannedClaudeMd objects
        """
        results = []
        paths = self.sources.find_project_claude_mds()

        for path in paths:
            scanned = ScannedClaudeMd.from_path(path)
            if scanned:
                results.append(scanned)

        return results

    def scan_commands(self) -> list[dict]:
        """
        Scan slash command definitions.

        Commands are stored in ~/.claude/commands/ as individual files
        (markdown or text format).

        Returns:
            List of command dictionaries with name, content, and path
        """
        commands = []
        commands_dir = self.sources.commands_dir

        if not commands_dir.exists():
            logger.debug(f"Commands directory not found: {commands_dir}")
            return commands

        # Scan command files
        for cmd_file in commands_dir.iterdir():
            if cmd_file.is_file() and cmd_file.suffix in (".md", ".txt", ""):
                try:
                    content = cmd_file.read_text(encoding="utf-8")
                    commands.append(
                        {
                            "name": cmd_file.stem,
                            "content": content,
                            "path": str(cmd_file),
                            "last_modified": datetime.fromtimestamp(
                                cmd_file.stat().st_mtime
                            ).isoformat(),
                        }
                    )
                except (OSError, UnicodeDecodeError) as e:
                    logger.warning(f"Failed to read command file {cmd_file}: {e}")

        # Also check project-specific commands directories
        projects_dir = self.sources.projects_dir
        if projects_dir.exists():
            for project_dir in projects_dir.iterdir():
                if project_dir.is_dir():
                    project_commands_dir = project_dir / "commands"
                    if project_commands_dir.exists():
                        real_path = _decode_project_path(project_dir.name)
                        for cmd_file in project_commands_dir.iterdir():
                            if cmd_file.is_file():
                                try:
                                    content = cmd_file.read_text(encoding="utf-8")
                                    commands.append(
                                        {
                                            "name": cmd_file.stem,
                                            "content": content,
                                            "path": str(cmd_file),
                                            "project": real_path,
                                            "last_modified": datetime.fromtimestamp(
                                                cmd_file.stat().st_mtime
                                            ).isoformat(),
                                        }
                                    )
                                except (OSError, UnicodeDecodeError) as e:
                                    logger.warning(
                                        f"Failed to read command file {cmd_file}: {e}"
                                    )

        return commands


def _encode_project_path(path: str) -> str:
    """
    Encode a project path to Claude Code directory name format.

    Replaces path separators with dashes.
    Example: /Users/name/git/repo -> -Users-name-git-repo

    Args:
        path: Filesystem path

    Returns:
        Encoded directory name
    """
    # Replace path separators with dashes
    return path.replace("/", "-").replace("\\", "-")


def _decode_project_path(encoded: str) -> Optional[str]:
    """
    Decode a Claude Code project directory name to a filesystem path.

    Example: -Users-name-git-repo -> /Users/name/git/repo

    Args:
        encoded: Encoded directory name

    Returns:
        Filesystem path or None if invalid
    """
    if not encoded.startswith("-"):
        return None

    # Replace leading dash with / and other dashes with /
    # This is a heuristic - Claude Code uses dashes for separators
    path = "/" + encoded[1:].replace("-", "/")

    # Validate path exists
    if Path(path).exists():
        return path

    # Try alternative interpretations for paths with dashes in names
    # This handles cases like my-project -> /Users/.../my-project
    return None
