"""
Program Loader - The Operator uploads training packs into Neo's memory.

Matrix Metaphor:
- Operator: This module (loads programs)
- Training pack: HuggingFace dataset slice
- Neural jack: Embedding pipeline
- The Construct: ~/.neo memory + FAISS index

This is NOT model fine-tuning. This IS retrieval learning:
expanding local semantic memory with patterns from datasets.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProgramLoadResult:
    """Result of loading a program (training pack)."""
    loaded_count: int
    deduped_count: int
    failed_count: int
    index_rebuild_time: float
    total_patterns: int
    dataset_id: str
    split: str


class ProgramLoader:
    """
    The Operator - loads training packs into Neo's memory.

    Responsibilities:
    1. Pull dataset from HuggingFace
    2. Map rows to ReasoningEntry schema
    3. Deduplicate against existing memory
    4. Generate embeddings
    5. Update FAISS index
    6. Report results (Matrix-style)
    """

    def __init__(self, memory: 'PersistentReasoningMemory'):
        """
        Initialize program loader.

        Args:
            memory: PersistentReasoningMemory instance to load into
        """
        self.memory = memory

    def load_program(
        self,
        dataset_id: str,
        split: str = "train",
        column_mapping: Optional[dict] = None,
        limit: Optional[int] = 1000,
        dry_run: bool = False,
        quiet: bool = False
    ) -> ProgramLoadResult:
        """
        Load a program (dataset) into Neo's memory.

        Args:
            dataset_id: HuggingFace dataset identifier (e.g., "mbpp", "code_search_net")
            split: Dataset split (train/test/validation)
            column_mapping: Map dataset columns to pattern fields
                           e.g., {"text": "pattern", "code": "suggestion"}
            limit: Maximum number of samples to load
            dry_run: Preview without importing
            quiet: Suppress progress output

        Returns:
            ProgramLoadResult with counts and timing

        Raises:
            ValueError: Invalid dataset or column mapping
            ImportError: datasets library not installed
        """
        start_time = time.time()

        # Validate inputs
        if not dataset_id:
            raise ValueError("dataset_id is required")
        if split not in ["train", "test", "validation"]:
            raise ValueError(f"Invalid split: {split}")

        # Import datasets library
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )

        if not quiet:
            print(f"Loading program: {dataset_id} ({split})")

        # Load dataset
        try:
            dataset = load_dataset(dataset_id, split=split, streaming=False)
        except Exception as e:
            raise ValueError(f"Failed to load dataset '{dataset_id}': {e}")

        # Apply limit
        if limit and len(dataset) > limit:
            dataset = dataset.select(range(limit))

        if not quiet:
            print(f"Dataset loaded: {len(dataset)} samples")

        # Get column mapping
        mapping = column_mapping or self._get_default_mapping(dataset_id)

        # Validate columns exist
        available_columns = dataset.column_names if hasattr(dataset, 'column_names') else dataset.features.keys()
        for col in mapping.values():
            if col not in available_columns:
                raise ValueError(
                    f"Column '{col}' not found in dataset. "
                    f"Available: {list(available_columns)}"
                )

        if dry_run:
            print(f"\n[DRY RUN] Would import {len(dataset)} patterns")
            print(f"Column mapping: {mapping}")
            return ProgramLoadResult(
                loaded_count=0,
                deduped_count=0,
                failed_count=0,
                index_rebuild_time=0.0,
                total_patterns=len(self.memory.entries),
                dataset_id=dataset_id,
                split=split
            )

        # Import patterns
        loaded = 0
        deduped = 0
        failed = 0

        # Get existing pattern hashes for deduplication
        existing_hashes = self._get_existing_hashes()

        for idx, row in enumerate(dataset):
            try:
                # Map row to pattern
                pattern = self._map_row_to_pattern(row, mapping, dataset_id, split)

                # Check for duplicate
                pattern_hash = self._hash_pattern(pattern)
                if pattern_hash in existing_hashes:
                    deduped += 1
                    continue

                # Add to memory
                self.memory.add_reasoning(
                    pattern=pattern["pattern"],
                    context=pattern.get("context", "imported from dataset"),
                    reasoning=pattern.get("reasoning", ""),
                    suggestion=pattern["suggestion"],
                    confidence=0.3,  # Default for imported patterns
                    source_context={
                        "source": "hf",
                        "dataset": dataset_id,
                        "split": split,
                        "row_idx": idx,
                        "imported_at": time.time()
                    },
                    code_skeleton=pattern.get("code", ""),
                    common_pitfalls=[],
                    test_patterns=[]
                )

                existing_hashes.add(pattern_hash)
                loaded += 1

                if not quiet and (loaded % 100 == 0):
                    print(f"  Loaded: {loaded}, Deduped: {deduped}")

            except Exception as e:
                logger.warning(f"Failed to import row {idx}: {e}")
                failed += 1

        # Save memory
        self.memory.save()

        # Rebuild FAISS index (if available)
        index_time = 0.0
        if hasattr(self.memory, '_rebuild_faiss_index'):
            index_start = time.time()
            self.memory._rebuild_faiss_index()
            index_time = time.time() - index_start

        total_time = time.time() - start_time

        return ProgramLoadResult(
            loaded_count=loaded,
            deduped_count=deduped,
            failed_count=failed,
            index_rebuild_time=index_time,
            total_patterns=len(self.memory.entries),
            dataset_id=dataset_id,
            split=split
        )

    def _get_default_mapping(self, dataset_id: str) -> dict:
        """
        Get default column mapping for known datasets.

        Args:
            dataset_id: Dataset identifier

        Returns:
            Column mapping dict
        """
        # Common mappings for popular datasets
        mappings = {
            "mbpp": {
                "pattern": "text",
                "suggestion": "code",
                "context": "text"
            },
            "openai_humaneval": {
                "pattern": "prompt",
                "suggestion": "canonical_solution",
                "context": "entry_point"
            },
            "bigcode/humanevalpack": {
                "pattern": "prompt",
                "suggestion": "canonical_solution",
                "context": "entry_point"
            },
            "Muennighoff/natural-instructions": {
                "pattern": "definition",
                "suggestion": "targets",
                "context": "task_name"
            }
        }

        if dataset_id in mappings:
            return mappings[dataset_id]

        # Default fallback
        return {
            "pattern": "text",
            "suggestion": "code"
        }

    def _map_row_to_pattern(
        self,
        row: dict,
        mapping: dict,
        dataset_id: str,
        split: str
    ) -> dict:
        """
        Map dataset row to pattern dict.

        Args:
            row: Dataset row
            mapping: Column mapping
            dataset_id: Dataset identifier
            split: Dataset split

        Returns:
            Pattern dict ready for import
        """
        pattern = {}

        # Required fields
        pattern["pattern"] = str(row.get(mapping.get("pattern", "text"), ""))
        pattern["suggestion"] = str(row.get(mapping.get("suggestion", "code"), ""))

        # Optional fields
        if "context" in mapping:
            pattern["context"] = str(row.get(mapping["context"], ""))
        if "reasoning" in mapping:
            pattern["reasoning"] = str(row.get(mapping["reasoning"], ""))
        if "code" in mapping:
            pattern["code"] = str(row.get(mapping["code"], pattern["suggestion"]))
        if "algorithm_type" in mapping:
            pattern["algorithm_type"] = str(row.get(mapping["algorithm_type"], ""))

        # Validate required fields
        if not pattern["pattern"] or not pattern["suggestion"]:
            raise ValueError("Pattern and suggestion are required")

        return pattern

    def _get_existing_hashes(self) -> set:
        """Get hashes of existing patterns for deduplication."""
        hashes = set()
        for entry in self.memory.entries:
            pattern_hash = self._hash_pattern({
                "pattern": entry.pattern,
                "suggestion": entry.suggestion
            })
            hashes.add(pattern_hash)
        return hashes

    def _hash_pattern(self, pattern: dict) -> str:
        """
        Generate hash for pattern deduplication.

        Args:
            pattern: Pattern dict

        Returns:
            SHA256 hash (first 16 chars)
        """
        content = f"{pattern['pattern']}||{pattern['suggestion']}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def format_result(self, result: ProgramLoadResult, quote: str = None) -> str:
        """
        Format load result with Matrix-style output.

        Args:
            result: Load result
            quote: Optional Matrix quote (random if None)

        Returns:
            Formatted output string
        """
        if quote is None:
            quotes = [
                '"I know kung fu."',
                '"Show me."',
                '"There is no spoon."',
                '"I can only show you the door. You\'re the one that has to walk through it."',
                '"What is real? How do you define \'real\'?"'
            ]
            import random
            quote = random.choice(quotes)

        lines = [
            quote,
            "",
            f"Loaded: {result.loaded_count} patterns",
        ]

        if result.deduped_count > 0:
            lines.append(f"Deduped: {result.deduped_count} duplicates")
        if result.failed_count > 0:
            lines.append(f"Failed: {result.failed_count} errors")
        if result.index_rebuild_time > 0:
            lines.append(f"Index rebuilt: {result.index_rebuild_time:.1f}s")

        lines.append(f"Memory: {result.total_patterns} total patterns")

        return "\n".join(lines)
