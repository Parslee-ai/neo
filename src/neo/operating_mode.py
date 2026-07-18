"""Typed operating modes and explicit authority boundaries for Neo."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePath
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from neo.models import AppliedAction, CodeSuggestion


class OperatingMode(str, Enum):
    """User-visible behavior contract for one Neo request."""

    ADVISE = "advise"
    PATCH = "patch"
    VERIFY = "verify"
    LEARN = "learn"
    AGENT = "agent"

    @property
    def allows_learning(self) -> bool:
        return self in {OperatingMode.LEARN, OperatingMode.AGENT}

    @property
    def may_execute(self) -> bool:
        return self is OperatingMode.AGENT


@dataclass
class AuthorityPolicy:
    """Explicit host-granted authority; absence always means no mutation."""

    workspace_root: str = ""
    allowed_write_paths: list[str] = field(default_factory=list)
    allowed_commands: list[str] = field(default_factory=list)
    allow_learning: bool = True

    def allows_path(self, path: str) -> bool:
        """Return whether a relative or absolute path is inside an allowed glob."""
        if not self.workspace_root or not self.allowed_write_paths or not path:
            return False
        root = Path(self.workspace_root).resolve()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            relative = candidate.resolve().relative_to(root)
        except (OSError, ValueError):
            return False
        relative_path = PurePath(relative.as_posix())
        return any(relative_path.match(pattern) for pattern in self.allowed_write_paths)

    def public_summary(self) -> dict:
        """Non-secret policy details safe to persist in a learning episode."""
        return {
            "workspace_root": self.workspace_root,
            "allowed_write_paths": list(self.allowed_write_paths),
            "allowed_command_hashes": [
                hashlib.sha256(command.encode("utf-8", errors="replace")).hexdigest()
                for command in self.allowed_commands
            ],
            "allow_learning": self.allow_learning,
        }


class ModeValidationError(ValueError):
    """Request mode and supplied authority are inconsistent or unsafe."""


class ExecutionAdapter(Protocol):
    """Host-provided executor; Neo itself never shells out generated commands."""

    def execute(
        self,
        suggestions: list["CodeSuggestion"],
        policy: AuthorityPolicy,
    ) -> list["AppliedAction"]:
        """Execute only the pre-authorized suggestions and return evidence."""


def validate_agent_authority(
    policy: AuthorityPolicy | None,
    *,
    has_executor: bool,
) -> None:
    """Fail closed before inference when agent execution cannot be authorized."""
    if policy is None:
        raise ModeValidationError("agent mode requires an explicit authority policy")
    if not policy.workspace_root:
        raise ModeValidationError("agent authority requires workspace_root")
    if not policy.allowed_write_paths:
        raise ModeValidationError(
            "agent authority must allow at least one workspace-relative write path"
        )
    if not has_executor:
        raise ModeValidationError(
            "agent mode requires a host-provided execution adapter; Neo has no built-in executor"
        )
