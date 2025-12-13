"""
Evolution tracking for CLAUDE.md files and commands.

Tracks changes to CLAUDE.md files and slash commands over time,
computing diffs and inferring reasons for changes.
"""

import difflib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from neo.prompt.scanner import ScannedClaudeMd

logger = logging.getLogger(__name__)


@dataclass
class ClaudeMdEvolution:
    """Tracks a change to a CLAUDE.md file."""

    path: Path
    timestamp: datetime
    previous_content: str
    new_content: str
    diff: str  # Unified diff
    change_type: Literal["created", "modified", "deleted"]
    inferred_reason: Optional[str]  # Why the change was made (from context)


@dataclass
class CommandEvolution:
    """Tracks a change to a slash command."""

    command_name: str
    path: Path
    timestamp: datetime
    previous_content: str
    new_content: str
    diff: str
    change_type: Literal["created", "modified", "deleted"]
    inferred_reason: Optional[str]


class EvolutionTracker:
    """Tracks evolution of CLAUDE.md files and commands."""

    EVOLUTION_FILE = Path.home() / ".neo" / "prompt_evolutions.json"

    def __init__(self):
        """Initialize the evolution tracker."""
        self._evolutions: list[ClaudeMdEvolution] = []
        self._command_evolutions: list[CommandEvolution] = []
        self._load_evolutions()

    def record_claude_md_change(
        self, old: "ScannedClaudeMd", new: "ScannedClaudeMd"
    ) -> ClaudeMdEvolution:
        """Record a change to a CLAUDE.md file.

        Args:
            old: Previous version of the CLAUDE.md file
            new: New version of the CLAUDE.md file

        Returns:
            ClaudeMdEvolution record of the change
        """
        diff = self._compute_diff(old.content, new.content)
        reason = self._infer_change_reason(old, new, diff)

        evolution = ClaudeMdEvolution(
            path=new.path,
            timestamp=datetime.now(),
            previous_content=old.content,
            new_content=new.content,
            diff=diff,
            change_type="modified",
            inferred_reason=reason,
        )

        self._evolutions.append(evolution)
        self._save_evolutions()
        return evolution

    def record_command_change(
        self,
        command_name: str,
        path: Path,
        old_content: str,
        new_content: str,
        change_type: Literal["created", "modified", "deleted"] = "modified",
    ) -> CommandEvolution:
        """Record a change to a slash command.

        Args:
            command_name: Name of the command (e.g., "build", "test")
            path: Path to the command file
            old_content: Previous content of the command
            new_content: New content of the command
            change_type: Type of change

        Returns:
            CommandEvolution record of the change
        """
        diff = self._compute_diff(old_content, new_content)

        evolution = CommandEvolution(
            command_name=command_name,
            path=path,
            timestamp=datetime.now(),
            previous_content=old_content,
            new_content=new_content,
            diff=diff,
            change_type=change_type,
            inferred_reason="Command updated",
        )

        self._command_evolutions.append(evolution)
        self._save_evolutions()
        return evolution

    def _compute_diff(self, old_content: str, new_content: str) -> str:
        """Compute unified diff between old and new content.

        Args:
            old_content: Previous content
            new_content: New content

        Returns:
            Unified diff as string
        """
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)

        diff_lines = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="old",
            tofile="new",
            lineterm="",
        )

        return "".join(diff_lines)

    def _infer_change_reason(
        self, old: "ScannedClaudeMd", new: "ScannedClaudeMd", diff: str
    ) -> str:
        """Infer why a CLAUDE.md was changed based on diff analysis.

        Args:
            old: Previous version of the CLAUDE.md file
            new: New version of the CLAUDE.md file
            diff: Unified diff between versions

        Returns:
            Inferred reason for the change
        """
        # Placeholder implementation - returns generic reason
        # Future: analyze recent sessions to infer specific reasons like:
        # - User repeatedly had to explain X -> Added rule about X
        # - Claude kept making mistake Y -> Added instruction to avoid Y
        # - User asked "can you do Z" multiple times -> Added capability Z

        added_lines = sum(
            1 for line in diff.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        )
        removed_lines = sum(
            1 for line in diff.split("\n")
            if line.startswith("-") and not line.startswith("---")
        )

        if added_lines > 0 and removed_lines == 0:
            return "Added new content to CLAUDE.md"
        elif removed_lines > 0 and added_lines == 0:
            return "Removed content from CLAUDE.md"
        elif added_lines > removed_lines:
            return "Expanded CLAUDE.md with new rules or instructions"
        elif removed_lines > added_lines:
            return "Simplified CLAUDE.md by removing rules"
        else:
            return "Modified CLAUDE.md content"

    def get_evolution_history(
        self, path: Optional[Path] = None
    ) -> list[ClaudeMdEvolution]:
        """Get evolution history, optionally filtered by path.

        Args:
            path: Optional path to filter by (None returns all)

        Returns:
            List of evolution records
        """
        if path is None:
            return list(self._evolutions)

        return [e for e in self._evolutions if e.path == path]

    def get_command_evolution_history(
        self, command_name: Optional[str] = None
    ) -> list[CommandEvolution]:
        """Get command evolution history, optionally filtered by name.

        Args:
            command_name: Optional command name to filter by

        Returns:
            List of command evolution records
        """
        if command_name is None:
            return list(self._command_evolutions)

        return [e for e in self._command_evolutions if e.command_name == command_name]

    def suggest_improvements(self, sessions: list) -> list[dict]:
        """Suggest CLAUDE.md improvements based on session patterns.

        Analyzes sessions for repeated clarifications, recurring errors,
        and other patterns that indicate CLAUDE.md could be improved.

        Args:
            sessions: List of ScannedSession objects to analyze

        Returns:
            List of improvement suggestions
        """
        suggestions = []

        # Pattern: User repeatedly clarifies same thing
        clarification_patterns = self._find_repeated_clarifications(sessions)
        for pattern in clarification_patterns:
            suggestions.append({
                "type": "add_rule",
                "target": pattern.get("project"),
                "suggestion": f"Add rule: {pattern.get('suggested_rule', 'N/A')}",
                "reason": (
                    f"User had to clarify '{pattern.get('topic', 'N/A')}' "
                    f"{pattern.get('count', 0)} times"
                ),
                "confidence": pattern.get("confidence", 0.5),
            })

        # Pattern: Same errors keep occurring
        error_patterns = self._find_recurring_errors(sessions)
        for pattern in error_patterns:
            suggestions.append({
                "type": "add_constraint",
                "target": pattern.get("project"),
                "suggestion": (
                    f"Add constraint: {pattern.get('suggested_constraint', 'N/A')}"
                ),
                "reason": (
                    f"Error '{pattern.get('error', 'N/A')}' occurred "
                    f"{pattern.get('count', 0)} times"
                ),
                "confidence": pattern.get("confidence", 0.5),
            })

        return suggestions

    def _find_repeated_clarifications(self, sessions: list) -> list[dict]:
        """Find patterns where users repeatedly clarify the same thing.

        Args:
            sessions: List of session objects to analyze

        Returns:
            List of clarification patterns found
        """
        # Placeholder implementation
        # Future: Analyze session messages to find repeated clarification patterns
        return []

    def _find_recurring_errors(self, sessions: list) -> list[dict]:
        """Find patterns where the same errors keep occurring.

        Args:
            sessions: List of session objects to analyze

        Returns:
            List of recurring error patterns found
        """
        # Placeholder implementation
        # Future: Analyze session errors to find recurring patterns
        return []

    def _save_evolutions(self) -> None:
        """Save evolutions to JSON file atomically."""
        # Ensure directory exists
        self.EVOLUTION_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "evolutions": [self._serialize_evolution(e) for e in self._evolutions],
            "command_evolutions": [
                self._serialize_command_evolution(e) for e in self._command_evolutions
            ],
        }

        # Atomic write: write to temp file, then rename
        try:
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.EVOLUTION_FILE.parent,
                prefix=".prompt_evolutions_",
                suffix=".tmp"
            )
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(data, f, indent=2)
                os.rename(temp_path, self.EVOLUTION_FILE)
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
            logger.debug(
                f"Saved {len(self._evolutions)} evolutions to {self.EVOLUTION_FILE}"
            )
        except (IOError, PermissionError) as e:
            logger.error(f"Failed to save evolutions to {self.EVOLUTION_FILE}: {e}")
            raise

    def _load_evolutions(self) -> None:
        """Load evolutions from JSON file."""
        if not self.EVOLUTION_FILE.exists():
            logger.debug(f"Evolution file not found: {self.EVOLUTION_FILE}")
            return

        try:
            with open(self.EVOLUTION_FILE) as f:
                data = json.load(f)

            self._evolutions = [
                self._deserialize_evolution(e) for e in data.get("evolutions", [])
            ]
            self._command_evolutions = [
                self._deserialize_command_evolution(e)
                for e in data.get("command_evolutions", [])
            ]
            logger.debug(
                f"Loaded {len(self._evolutions)} evolutions from {self.EVOLUTION_FILE}"
            )
        except (json.JSONDecodeError, IOError, PermissionError) as e:
            logger.error(f"Failed to load evolutions from {self.EVOLUTION_FILE}: {e}")
            raise

    def _serialize_evolution(self, evolution: ClaudeMdEvolution) -> dict:
        """Serialize a ClaudeMdEvolution to dict."""
        return {
            "path": str(evolution.path),
            "timestamp": evolution.timestamp.isoformat(),
            "previous_content": evolution.previous_content,
            "new_content": evolution.new_content,
            "diff": evolution.diff,
            "change_type": evolution.change_type,
            "inferred_reason": evolution.inferred_reason,
        }

    def _deserialize_evolution(self, data: dict) -> ClaudeMdEvolution:
        """Deserialize a dict to ClaudeMdEvolution."""
        return ClaudeMdEvolution(
            path=Path(data["path"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            previous_content=data["previous_content"],
            new_content=data["new_content"],
            diff=data["diff"],
            change_type=data["change_type"],
            inferred_reason=data.get("inferred_reason"),
        )

    def _serialize_command_evolution(self, evolution: CommandEvolution) -> dict:
        """Serialize a CommandEvolution to dict."""
        return {
            "command_name": evolution.command_name,
            "path": str(evolution.path),
            "timestamp": evolution.timestamp.isoformat(),
            "previous_content": evolution.previous_content,
            "new_content": evolution.new_content,
            "diff": evolution.diff,
            "change_type": evolution.change_type,
            "inferred_reason": evolution.inferred_reason,
        }

    def _deserialize_command_evolution(self, data: dict) -> CommandEvolution:
        """Deserialize a dict to CommandEvolution."""
        return CommandEvolution(
            command_name=data["command_name"],
            path=Path(data["path"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            previous_content=data["previous_content"],
            new_content=data["new_content"],
            diff=data["diff"],
            change_type=data["change_type"],
            inferred_reason=data.get("inferred_reason"),
        )
