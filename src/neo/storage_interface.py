"""
Storage abstraction interface for Neo's persistent memory.

Simple interface that allows pluggable storage backends:
- FileStorage: Local JSON files
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def load_entries(self, storage_key: str) -> List[Dict]:
        """
        Load entries from storage.

        Args:
            storage_key: Identifier for the storage location (e.g., "global", "local_abc123")

        Returns:
            List of entry dictionaries
        """
        pass

    @abstractmethod
    def save_entries(self, storage_key: str, entries: List[Dict]) -> None:
        """
        Save entries to storage.

        Args:
            storage_key: Identifier for the storage location
            entries: List of entry dictionaries to save
        """
        pass

    @abstractmethod
    def exists(self, storage_key: str) -> bool:
        """
        Check if storage location exists.

        Args:
            storage_key: Identifier for the storage location

        Returns:
            True if storage location exists
        """
        pass
