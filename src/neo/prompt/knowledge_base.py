"""
Knowledge Base for Prompt Enhancement System.

Stores prompt patterns, effectiveness scores, evolutions, and suggestions.
Uses JSON file storage in ~/.neo directory.
"""

import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, TYPE_CHECKING

# Import canonical dataclasses from their home modules to avoid duplication
# NOTE: These are imported at runtime to avoid circular imports
if TYPE_CHECKING:
    from neo.prompt.analyzer import PromptPattern, PromptEffectivenessScore
    from neo.prompt.evolution import ClaudeMdEvolution

logger = logging.getLogger(__name__)


@dataclass
class PromptEntry:
    """An entry in the prompt knowledge base."""
    id: str
    entry_type: Literal["pattern", "score", "evolution", "suggestion"]
    data: dict
    embedding: Optional[list[float]]
    created_at: datetime
    updated_at: datetime
    project: Optional[str]  # None for global patterns


class PromptKnowledgeBase:
    """Separate knowledge base for prompt patterns and effectiveness data."""

    STORAGE_FILE = Path.home() / ".neo" / "prompt_knowledge.json"

    def __init__(self):
        """Initialize the prompt knowledge base."""
        self.entries: list[PromptEntry] = []
        self._load()

    def _load(self) -> None:
        """Load entries from JSON file."""
        try:
            if not self.STORAGE_FILE.exists():
                logger.debug(f"File not found: {self.STORAGE_FILE}")
                return

            with open(self.STORAGE_FILE) as f:
                data = json.load(f)

            raw_entries = data.get("entries", [])
            self.entries = []
            for entry_dict in raw_entries:
                entry = PromptEntry(
                    id=entry_dict["id"],
                    entry_type=entry_dict["entry_type"],
                    data=entry_dict["data"],
                    embedding=entry_dict.get("embedding"),
                    created_at=datetime.fromisoformat(entry_dict["created_at"]),
                    updated_at=datetime.fromisoformat(entry_dict["updated_at"]),
                    project=entry_dict.get("project"),
                )
                self.entries.append(entry)

            logger.debug(f"Loaded {len(self.entries)} entries from {self.STORAGE_FILE}")
        except FileNotFoundError:
            logger.debug(f"File not found: {self.STORAGE_FILE}")
        except (json.JSONDecodeError, PermissionError, IOError) as e:
            logger.error(f"Failed to load from {self.STORAGE_FILE}: {e}")
            raise

    def _save(self) -> None:
        """Save entries to JSON file atomically."""
        try:
            self.STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)

            entries_list = []
            for entry in self.entries:
                entry_dict = {
                    "id": entry.id,
                    "entry_type": entry.entry_type,
                    "data": entry.data,
                    "embedding": entry.embedding,
                    "created_at": entry.created_at.isoformat(),
                    "updated_at": entry.updated_at.isoformat(),
                    "project": entry.project,
                }
                entries_list.append(entry_dict)

            data = {
                "version": "1.0",
                "entries": entries_list,
            }

            # Atomic write: write to temp file, then rename
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.STORAGE_FILE.parent,
                prefix=".prompt_knowledge_",
                suffix=".tmp"
            )
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump(data, f, indent=2)
                os.rename(temp_path, self.STORAGE_FILE)
            except Exception:
                # Clean up temp file on failure
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

            logger.debug(f"Saved {len(self.entries)} entries to {self.STORAGE_FILE}")
        except Exception as e:
            logger.error(f"Error saving to file {self.STORAGE_FILE}: {e}")
            raise

    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID with the given prefix."""
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def add_pattern(self, pattern: "PromptPattern") -> None:
        """Add an effective prompt pattern.

        Args:
            pattern: A PromptPattern from neo.prompt.analyzer
        """
        now = datetime.now()
        entry = PromptEntry(
            id=self._generate_id("pattern"),
            entry_type="pattern",
            data={
                "pattern_id": pattern.pattern_id,
                "name": pattern.name,
                "description": pattern.description,
                "template": pattern.template,
                "examples": pattern.examples,
                "effectiveness_score": pattern.effectiveness_score,
                "use_cases": pattern.use_cases,
                "anti_patterns": pattern.anti_patterns,
            },
            embedding=None,
            created_at=now,
            updated_at=now,
            project=None,
        )
        self.entries.append(entry)
        self._save()

    def search_patterns(self, query: str, k: int = 5) -> list["PromptPattern"]:
        """
        Search for relevant patterns using simple keyword matching.

        Args:
            query: Search query string
            k: Maximum number of results to return

        Returns:
            List of matching PromptPattern objects
        """
        from neo.prompt.analyzer import PromptPattern

        query_lower = query.lower()
        query_terms = query_lower.split()

        pattern_entries = [e for e in self.entries if e.entry_type == "pattern"]

        # If no query, return all patterns sorted by effectiveness
        if not query_terms:
            pattern_entries.sort(
                key=lambda e: e.data.get("effectiveness_score", 0.0),
                reverse=True
            )
            scored_entries = [(0, e) for e in pattern_entries[:k]]
        else:
            scored_entries: list[tuple[int, PromptEntry]] = []
            for entry in pattern_entries:
                data = entry.data
                searchable_text = " ".join([
                    data.get("name", ""),
                    data.get("description", ""),
                    data.get("template", ""),
                    " ".join(data.get("examples", [])),
                    " ".join(data.get("use_cases", [])),
                ]).lower()

                score = sum(1 for term in query_terms if term in searchable_text)
                if score > 0:
                    scored_entries.append((score, entry))

            scored_entries.sort(key=lambda x: x[0], reverse=True)

        results = []
        for _, entry in scored_entries[:k]:
            data = entry.data
            pattern = PromptPattern(
                pattern_id=data.get("pattern_id", ""),
                name=data.get("name", ""),
                description=data.get("description", ""),
                template=data.get("template", ""),
                examples=data.get("examples", []),
                effectiveness_score=data.get("effectiveness_score", 0.0),
                use_cases=data.get("use_cases", []),
                anti_patterns=data.get("anti_patterns", []),
            )
            results.append(pattern)

        return results

    def update_effectiveness_score(self, score: "PromptEffectivenessScore") -> None:
        """
        Update effectiveness score, aggregating with existing data.

        If a score with the same prompt_hash exists, aggregates using weighted average.
        Otherwise, creates a new entry.

        Args:
            score: A PromptEffectivenessScore from neo.prompt.analyzer
        """
        # Convert EffectivenessSignal enums to string values for JSON storage
        def serialize_signals(signals: list) -> list[str]:
            return [s.value if hasattr(s, 'value') else str(s) for s in signals]

        existing_entry = None
        for entry in self.entries:
            if entry.entry_type == "score":
                if entry.data.get("prompt_hash") == score.prompt_hash:
                    existing_entry = entry
                    break

        if existing_entry:
            existing_data = existing_entry.data
            old_count = existing_data.get("sample_count", 1)
            new_count = old_count + 1

            old_score = existing_data.get("score", 0.0)
            new_score = (old_score * old_count + score.score) / new_count

            existing_data["score"] = new_score
            existing_data["sample_count"] = new_count
            existing_data["confidence"] = min(0.95, existing_data.get("confidence", 0.5) + 0.1)

            # Merge signals (convert new signals to strings)
            existing_signals = set(existing_data.get("signals", []))
            new_signals = serialize_signals(score.signals)
            existing_signals.update(new_signals)
            existing_data["signals"] = list(existing_signals)

            existing_data["iterations_to_complete"] = score.iterations_to_complete
            existing_data["tool_calls"] = score.tool_calls

            existing_entry.updated_at = datetime.now()
        else:
            now = datetime.now()
            entry = PromptEntry(
                id=self._generate_id("score"),
                entry_type="score",
                data={
                    "prompt_hash": score.prompt_hash,
                    "prompt_text": score.prompt_text,
                    "score": score.score,
                    "signals": serialize_signals(score.signals),
                    "iterations_to_complete": score.iterations_to_complete,
                    "tool_calls": score.tool_calls,
                    "sample_count": score.sample_count,
                    "confidence": score.confidence,
                },
                embedding=None,
                created_at=now,
                updated_at=now,
                project=None,
            )
            self.entries.append(entry)

        self._save()

    def add_evolution(self, evolution: "ClaudeMdEvolution") -> None:
        """Record a CLAUDE.md evolution.

        Args:
            evolution: A ClaudeMdEvolution from neo.prompt.evolution
        """
        now = datetime.now()
        entry = PromptEntry(
            id=self._generate_id("evolution"),
            entry_type="evolution",
            data={
                "path": str(evolution.path),
                "timestamp": evolution.timestamp.isoformat(),
                "previous_content": evolution.previous_content,
                "new_content": evolution.new_content,
                "diff": evolution.diff,
                "change_type": evolution.change_type,
                "inferred_reason": evolution.inferred_reason,
            },
            embedding=None,
            created_at=now,
            updated_at=now,
            project=str(Path(evolution.path).parent) if evolution.path else None,
        )
        self.entries.append(entry)
        self._save()

    def get_evolutions(self, path: Optional[Path] = None) -> list["ClaudeMdEvolution"]:
        """
        Get evolution history.

        Args:
            path: Optional path to filter evolutions by. If None, returns all evolutions.

        Returns:
            List of ClaudeMdEvolution objects, sorted by timestamp descending.
        """
        from neo.prompt.evolution import ClaudeMdEvolution

        evolution_entries = [e for e in self.entries if e.entry_type == "evolution"]

        if path:
            path_str = str(path)
            evolution_entries = [e for e in evolution_entries if e.data.get("path") == path_str]

        evolution_entries.sort(
            key=lambda e: datetime.fromisoformat(e.data.get("timestamp", "1970-01-01")),
            reverse=True
        )

        results = []
        for entry in evolution_entries:
            data = entry.data
            evolution = ClaudeMdEvolution(
                path=Path(data.get("path", "")),  # Convert string to Path
                timestamp=datetime.fromisoformat(data.get("timestamp", "1970-01-01")),
                previous_content=data.get("previous_content", ""),
                new_content=data.get("new_content", ""),
                diff=data.get("diff", ""),
                change_type=data.get("change_type", "modified"),
                inferred_reason=data.get("inferred_reason"),
            )
            results.append(evolution)

        return results

    def add_suggestion(self, suggestion: dict) -> None:
        """Add an improvement suggestion."""
        now = datetime.now()
        suggestion_data = dict(suggestion)
        suggestion_data.setdefault("status", "pending")

        entry = PromptEntry(
            id=self._generate_id("suggestion"),
            entry_type="suggestion",
            data=suggestion_data,
            embedding=None,
            created_at=now,
            updated_at=now,
            project=suggestion.get("target") or suggestion.get("project"),
        )
        self.entries.append(entry)
        self._save()

    def get_pending_suggestions(self, project: Optional[str] = None) -> list[dict]:
        """
        Get unresolved suggestions.

        Args:
            project: Optional project to filter by. If None, returns all pending suggestions.

        Returns:
            List of suggestion dictionaries with status "pending".
        """
        suggestion_entries = [
            e for e in self.entries
            if e.entry_type == "suggestion" and e.data.get("status") == "pending"
        ]

        if project:
            suggestion_entries = [e for e in suggestion_entries if e.project == project]

        return [e.data for e in suggestion_entries]

    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        pattern_count = len([e for e in self.entries if e.entry_type == "pattern"])
        score_count = len([e for e in self.entries if e.entry_type == "score"])
        evolution_count = len([e for e in self.entries if e.entry_type == "evolution"])
        pending_suggestions = len(self.get_pending_suggestions())
        projects = set(e.project for e in self.entries if e.project)

        return {
            "total_entries": len(self.entries),
            "patterns": pattern_count,
            "scores": score_count,
            "evolutions": evolution_count,
            "pending_suggestions": pending_suggestions,
            "projects_tracked": len(projects),
        }
