"""
Change detection for the prompt enhancement system.

Tracks watermarks to detect changes since the last scan, avoiding reprocessing
of already-analyzed data.
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neo.prompt.scanner import (
        Scanner,
        ScannedPrompt,
        ScannedSession,
        ScannedClaudeMd,
    )

logger = logging.getLogger(__name__)


@dataclass
class Watermark:
    """Tracks last processed position for a data source."""

    source: str  # e.g., "history", "session:project_name", "claude_md:/path"
    timestamp: datetime
    position: Optional[int] = None  # Line number for files
    hash: Optional[str] = None  # Content hash for CLAUDE.md files

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "position": self.position,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Watermark":
        """Create from dictionary."""
        return cls(
            source=data["source"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            position=data.get("position"),
            hash=data.get("hash"),
        )


class ChangeDetector:
    """Detects changes since last scan using watermarks."""

    WATERMARK_FILE = Path.home() / ".neo" / "prompt_watermarks.json"

    def __init__(self):
        """Initialize the change detector and load existing watermarks."""
        self.watermarks: dict[str, Watermark] = {}
        self._load_watermarks()

    def _load_watermarks(self) -> None:
        """Load watermarks from the JSON file."""
        if not self.WATERMARK_FILE.exists():
            logger.debug(f"Watermark file not found: {self.WATERMARK_FILE}")
            return

        try:
            with open(self.WATERMARK_FILE) as f:
                data = json.load(f)

            # Validate version
            version = data.get("version", "1.0")
            if version != "1.0":
                logger.warning(f"Unknown watermark file version: {version}")

            # Load watermarks
            watermarks_data = data.get("watermarks", {})
            for source, wm_data in watermarks_data.items():
                # Add source to the data for reconstruction
                wm_data["source"] = source
                self.watermarks[source] = Watermark.from_dict(wm_data)

            logger.debug(f"Loaded {len(self.watermarks)} watermarks")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse watermarks file: {e}")
            raise
        except (PermissionError, IOError) as e:
            logger.error(f"Failed to read watermarks file: {e}")
            raise

    def _save_watermarks(self) -> None:
        """Save watermarks to the JSON file atomically."""
        # Ensure directory exists
        self.WATERMARK_FILE.parent.mkdir(parents=True, exist_ok=True)

        # Build the watermarks dictionary in the expected format
        watermarks_dict = {}
        for source, wm in self.watermarks.items():
            wm_dict = wm.to_dict()
            # Remove source from inner dict (it's the key)
            del wm_dict["source"]
            watermarks_dict[source] = wm_dict

        data = {
            "version": "1.0",
            "watermarks": watermarks_dict,
        }

        # Atomic write: write to temp file, then rename
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.WATERMARK_FILE.parent,
                prefix=".prompt_watermarks_",
                suffix=".tmp"
            )
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(data, f, indent=2)
                os.rename(temp_path, self.WATERMARK_FILE)
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
            logger.debug(f"Saved {len(self.watermarks)} watermarks")
        except (PermissionError, IOError) as e:
            logger.error(f"Failed to save watermarks file: {e}")
            raise

    def get_watermark(self, source: str) -> Optional[Watermark]:
        """
        Get the watermark for a specific data source.

        Args:
            source: The data source identifier (e.g., "history", "session:myproject")

        Returns:
            The watermark for the source, or None if not found.
        """
        return self.watermarks.get(source)

    def update_watermark(
        self,
        source: str,
        timestamp: datetime,
        position: Optional[int] = None,
        hash: Optional[str] = None,
    ) -> None:
        """
        Update or create a watermark for a data source.

        Args:
            source: The data source identifier.
            timestamp: The timestamp of the last processed item.
            position: Optional line/position number for file-based sources.
            hash: Optional content hash for CLAUDE.md files.
        """
        self.watermarks[source] = Watermark(
            source=source,
            timestamp=timestamp,
            position=position,
            hash=hash,
        )
        self._save_watermarks()

    def get_history_changes(self, scanner: "Scanner") -> list["ScannedPrompt"]:
        """
        Get prompts from history that have been added since the last watermark.

        Uses line-based position tracking for accurate incremental scanning.
        The watermark stores the actual line number in history.jsonl, allowing
        efficient resumption from the exact position in subsequent scans.

        Args:
            scanner: The Scanner instance to use for reading history.

        Returns:
            List of ScannedPrompt objects added since the last scan.
        """
        watermark = self.get_watermark("history")
        since_line = watermark.position if watermark else None

        result = scanner.scan_history(since_line=since_line)
        prompts = result.prompts

        # Update watermark with the actual line number from the file
        if result.last_line_number > 0:
            latest_timestamp = (
                max(p.timestamp for p in prompts)
                if prompts
                else (watermark.timestamp if watermark else datetime.now())
            )
            self.update_watermark(
                "history", latest_timestamp, position=result.last_line_number
            )

        return prompts

    def get_session_changes(
        self, scanner: "Scanner", project: str
    ) -> list["ScannedSession"]:
        """
        Get sessions for a project that have been added since the last watermark.

        Args:
            scanner: The Scanner instance to use for reading sessions.
            project: The project path or identifier.

        Returns:
            List of ScannedSession objects added since the last scan.
        """
        source = f"session:{project}"
        watermark = self.get_watermark(source)
        since = watermark.timestamp if watermark else None

        sessions = scanner.scan_sessions(project=project, since=since)

        # Update watermark if we found any sessions
        if sessions:
            latest_timestamp = max(s.end_time for s in sessions)
            self.update_watermark(source, latest_timestamp)

        return sessions

    def get_claude_md_changes(
        self, scanner: "Scanner"
    ) -> list[tuple["ScannedClaudeMd", "ScannedClaudeMd"]]:
        """
        Get CLAUDE.md files that have been modified since the last scan.

        Args:
            scanner: The Scanner instance to use for reading CLAUDE.md files.

        Returns:
            List of (old_content, new_content) tuples for modified files.
            For newly created files, old_content will have empty content.
        """
        current_files = scanner.scan_claude_mds()
        changes: list[tuple["ScannedClaudeMd", "ScannedClaudeMd"]] = []

        for current in current_files:
            source = f"claude_md:{current.path}"
            watermark = self.get_watermark(source)

            if watermark is None:
                # New file - create a placeholder for "old" version
                from neo.prompt.scanner import ScannedClaudeMd

                old_version = ScannedClaudeMd(
                    path=current.path,
                    content="",
                    last_modified=current.last_modified,
                    hash="",
                )
                changes.append((old_version, current))

                # Update watermark for this new file
                self.update_watermark(
                    source,
                    timestamp=current.last_modified,
                    hash=current.hash,
                )

            elif watermark.hash != current.hash:
                # File has changed - we need to reconstruct the old version
                # Since we only store the hash, not the content, we create a
                # placeholder with the old hash. The caller should use the
                # evolution tracker for full history.
                from neo.prompt.scanner import ScannedClaudeMd

                old_version = ScannedClaudeMd(
                    path=current.path,
                    content="",  # Content not available from watermark
                    last_modified=watermark.timestamp,
                    hash=watermark.hash or "",
                )
                changes.append((old_version, current))

                # Update watermark with new hash
                self.update_watermark(
                    source,
                    timestamp=current.last_modified,
                    hash=current.hash,
                )

        return changes

    def get_changes_since_last_scan(self, scanner: "Scanner") -> dict:
        """
        Get all changes since the last scan.

        This is a convenience method that aggregates changes from all sources.

        Args:
            scanner: The Scanner instance to use.

        Returns:
            Dictionary with keys:
                - new_prompts: List of new prompts from history
                - new_sessions: List of new sessions (all projects combined)
                - modified_claude_mds: List of (old, new) tuples for modified files
        """
        # Get history changes
        new_prompts = self.get_history_changes(scanner)

        # Get session changes from all projects
        # Use a global session watermark for simplicity
        session_watermark = self.get_watermark("sessions_global")
        since = session_watermark.timestamp if session_watermark else None

        new_sessions = scanner.scan_sessions(project=None, since=since)

        # Update session watermark if we found sessions
        if new_sessions:
            latest_timestamp = max(s.end_time for s in new_sessions)
            self.update_watermark("sessions_global", latest_timestamp)

        # Get CLAUDE.md changes
        modified_claude_mds = self.get_claude_md_changes(scanner)

        return {
            "new_prompts": new_prompts,
            "new_sessions": new_sessions,
            "modified_claude_mds": modified_claude_mds,
        }

    def clear_watermarks(self) -> None:
        """Clear all watermarks. Useful for forcing a full rescan."""
        self.watermarks.clear()
        if self.WATERMARK_FILE.exists():
            self.WATERMARK_FILE.unlink()
        logger.info("Cleared all watermarks")
