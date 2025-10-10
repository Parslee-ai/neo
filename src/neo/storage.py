"""
Storage backends for Neo's persistent memory.

Implementation:
- FileStorage: Local JSON files in ~/.neo directory
"""

import json
import logging
from pathlib import Path
from typing import List, Dict
from neo.storage_interface import StorageBackend

logger = logging.getLogger(__name__)


class FileStorage(StorageBackend):
    """Local file-based storage backend."""

    def __init__(self, base_path: str = None):
        """
        Initialize file storage.

        Args:
            base_path: Base directory for storage files (default: ~/.neo)
        """
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path.home() / ".neo"

        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, storage_key: str) -> Path:
        """Get file path for storage key."""
        if storage_key == "global":
            return self.base_path / "global_memory.json"
        else:
            # Local storage keys like "local_abc123"
            return self.base_path / f"{storage_key}.json"

    def load_entries(self, storage_key: str) -> List[Dict]:
        """Load entries from JSON file."""
        file_path = self._get_file_path(storage_key)

        try:
            with open(file_path) as f:
                data = json.load(f)
            return data.get("entries", [])
        except FileNotFoundError:
            # File doesn't exist yet - normal for first run
            logger.debug(f"File not found: {file_path}")
            return []
        except (json.JSONDecodeError, PermissionError, IOError) as e:
            # File exists but corrupted/unreadable - data loss risk
            logger.error(f"Failed to load from {file_path}: {e}")
            raise

    def save_entries(self, storage_key: str, entries: List[Dict]) -> None:
        """Save entries to JSON file."""
        file_path = self._get_file_path(storage_key)

        try:
            data = {
                "entries": entries,
                "version": "1.0"
            }
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(entries)} entries to {file_path}")
        except Exception as e:
            logger.error(f"Error saving to file {file_path}: {e}")
            raise

    def exists(self, storage_key: str) -> bool:
        """Check if file exists."""
        return self._get_file_path(storage_key).exists()
