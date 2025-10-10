"""
Logging layer for LM I/O capture.

Captures full LM interactions with metadata for observability and training.
"""

import json
import logging
import time
import hashlib
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LMInteraction:
    """Record of a single LM interaction."""

    # Request metadata
    request_id: str
    timestamp: float
    phase: str  # planning, simulation, codegen

    # LM parameters
    model: str
    provider: str
    temperature: float
    max_tokens: int
    stop_sequences: list[str]

    # Request/Response
    system_prompt: str
    user_prompt: str
    response: str

    # Metrics
    latency_ms: float
    response_tokens: Optional[int] = None
    truncated: bool = False

    # Parse results
    parse_success: bool = False
    parse_error_code: Optional[str] = None
    parse_error_message: Optional[str] = None
    raw_block: Optional[str] = None

    # Repair results (if attempted)
    repair_attempted: bool = False
    repair_success: bool = False
    repair_attempts: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def redacted_copy(self) -> dict:
        """Create redacted copy for storage (remove PII)."""
        data = self.to_dict()

        # Redact potentially sensitive content
        data['user_prompt'] = self._hash_content(self.user_prompt)
        data['response'] = self._hash_content(self.response) if self.response else None
        data['raw_block'] = self._hash_content(self.raw_block) if self.raw_block else None

        return data

    def _hash_content(self, content: str) -> str:
        """Hash content for privacy while maintaining linkability."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class LMLogger:
    """Logger for LM interactions with local storage."""

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        enable_full_logging: bool = False,
        enable_redacted_logging: bool = True
    ):
        """
        Initialize LM logger.

        Args:
            storage_dir: Directory for log storage (default: ~/.neo/lm_logs/)
            enable_full_logging: Store full interactions (including prompts/responses)
            enable_redacted_logging: Store redacted interactions (hashed content)
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path.home() / ".neo" / "lm_logs"

        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.enable_full_logging = enable_full_logging
        self.enable_redacted_logging = enable_redacted_logging

        # Current day's log file
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.full_log_path = self.storage_dir / f"full_{self.current_date}.jsonl"
        self.redacted_log_path = self.storage_dir / f"redacted_{self.current_date}.jsonl"

        logger.info(f"LM Logger initialized: storage_dir={self.storage_dir}")

    def log_interaction(self, interaction: LMInteraction):
        """
        Log an LM interaction.

        Args:
            interaction: LMInteraction object to log
        """
        # Rotate log files if date changed
        current_date = datetime.now().strftime("%Y-%m-%d")
        if current_date != self.current_date:
            self.current_date = current_date
            self.full_log_path = self.storage_dir / f"full_{self.current_date}.jsonl"
            self.redacted_log_path = self.storage_dir / f"redacted_{self.current_date}.jsonl"

        # Log full interaction
        if self.enable_full_logging:
            try:
                with open(self.full_log_path, 'a') as f:
                    f.write(json.dumps(interaction.to_dict()) + '\n')
                logger.debug(f"Logged full interaction: {interaction.request_id}")
            except Exception as e:
                logger.error(f"Failed to log full interaction: {e}")

        # Log redacted interaction
        if self.enable_redacted_logging:
            try:
                with open(self.redacted_log_path, 'a') as f:
                    f.write(json.dumps(interaction.redacted_copy()) + '\n')
                logger.debug(f"Logged redacted interaction: {interaction.request_id}")
            except Exception as e:
                logger.error(f"Failed to log redacted interaction: {e}")

    def log_parse_failure(
        self,
        interaction: LMInteraction,
        error_code: str,
        error_message: str,
        raw_block: Optional[str] = None
    ):
        """
        Log a parse failure with error details.

        Args:
            interaction: The LMInteraction that failed parsing
            error_code: Normalized error code
            error_message: Detailed error message
            raw_block: Raw extracted block (if any)
        """
        interaction.parse_success = False
        interaction.parse_error_code = error_code
        interaction.parse_error_message = error_message
        interaction.raw_block = raw_block

        self.log_interaction(interaction)

        # Also log to failures file for easy analysis
        failures_path = self.storage_dir / f"failures_{self.current_date}.jsonl"
        try:
            with open(failures_path, 'a') as f:
                f.write(json.dumps(interaction.redacted_copy()) + '\n')
            logger.info(f"Logged parse failure: {error_code}")
        except Exception as e:
            logger.error(f"Failed to log parse failure: {e}")

    def log_repair_attempt(
        self,
        interaction: LMInteraction,
        repair_success: bool,
        repair_attempts: int
    ):
        """
        Log a repair attempt.

        Args:
            interaction: The LMInteraction being repaired
            repair_success: Whether repair succeeded
            repair_attempts: Number of attempts made
        """
        interaction.repair_attempted = True
        interaction.repair_success = repair_success
        interaction.repair_attempts = repair_attempts

        self.log_interaction(interaction)

        # Also log to repairs file
        repairs_path = self.storage_dir / f"repairs_{self.current_date}.jsonl"
        try:
            with open(repairs_path, 'a') as f:
                f.write(json.dumps(interaction.redacted_copy()) + '\n')
            logger.info(f"Logged repair attempt: success={repair_success}, attempts={repair_attempts}")
        except Exception as e:
            logger.error(f"Failed to log repair attempt: {e}")

    def get_metrics(self, days: int = 7) -> dict:
        """
        Get aggregate metrics for the last N days.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with metrics
        """
        metrics = {
            "total_interactions": 0,
            "parse_failures": 0,
            "repair_attempts": 0,
            "repair_successes": 0,
            "error_codes": {},
            "by_phase": {},
            "avg_latency_ms": 0.0,
            "truncation_count": 0
        }

        # Read recent log files
        for i in range(days):
            date = datetime.fromtimestamp(time.time() - i * 86400).strftime("%Y-%m-%d")
            log_path = self.storage_dir / f"redacted_{date}.jsonl"

            if not log_path.exists():
                continue

            try:
                with open(log_path, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        metrics["total_interactions"] += 1

                        if not data.get("parse_success", False):
                            metrics["parse_failures"] += 1
                            error_code = data.get("parse_error_code", "unknown")
                            metrics["error_codes"][error_code] = metrics["error_codes"].get(error_code, 0) + 1

                        if data.get("repair_attempted", False):
                            metrics["repair_attempts"] += 1
                            if data.get("repair_success", False):
                                metrics["repair_successes"] += 1

                        phase = data.get("phase", "unknown")
                        if phase not in metrics["by_phase"]:
                            metrics["by_phase"][phase] = {"count": 0, "failures": 0}
                        metrics["by_phase"][phase]["count"] += 1
                        if not data.get("parse_success", False):
                            metrics["by_phase"][phase]["failures"] += 1

                        if data.get("truncated", False):
                            metrics["truncation_count"] += 1

                        metrics["avg_latency_ms"] += data.get("latency_ms", 0)

            except Exception as e:
                logger.warning(f"Failed to read log file {log_path}: {e}")

        if metrics["total_interactions"] > 0:
            metrics["avg_latency_ms"] /= metrics["total_interactions"]
            metrics["parse_failure_rate"] = metrics["parse_failures"] / metrics["total_interactions"]
            if metrics["repair_attempts"] > 0:
                metrics["repair_success_rate"] = metrics["repair_successes"] / metrics["repair_attempts"]

        return metrics


# Global logger instance
_global_logger: Optional[LMLogger] = None


def get_lm_logger() -> LMLogger:
    """Get or create the global LM logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = LMLogger()
    return _global_logger


def set_lm_logger(logger_instance: LMLogger):
    """Set a custom LM logger instance."""
    global _global_logger
    _global_logger = logger_instance