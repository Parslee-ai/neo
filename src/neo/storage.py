"""
Storage backends for Neo's persistent memory.

Implementation:
- FileStorage: Local JSON files in ~/.neo directory
"""

import json
import logging
import os
import shutil
import tempfile
import time
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
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load from {file_path}: {e}")
            self._backup_corrupt_file(file_path)
            raise
        except (PermissionError, IOError) as e:
            logger.error(f"Failed to load from {file_path}: {e}")
            raise

    def save_entries(self, storage_key: str, entries: List[Dict]) -> None:
        """Save entries to JSON file atomically."""
        file_path = self._get_file_path(storage_key)

        try:
            data = {
                "entries": entries,
                "version": "1.0"
            }
            file_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=file_path.parent, suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_name, file_path)
            except BaseException:
                os.unlink(tmp_name)
                raise
            logger.debug(f"Saved {len(entries)} entries to {file_path}")
        except Exception as e:
            logger.error(f"Error saving to file {file_path}: {e}")
            raise

    def exists(self, storage_key: str) -> bool:
        """Check if file exists."""
        return self._get_file_path(storage_key).exists()

    @staticmethod
    def _backup_corrupt_file(path: Path) -> None:
        """Preserve a corrupt memory file for manual recovery."""
        backup = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
        try:
            shutil.copy2(path, backup)
            logger.warning(f"Backed up corrupt storage file to {backup}")
        except OSError as backup_error:
            logger.warning(f"Failed to back up corrupt storage file {path}: {backup_error}")
