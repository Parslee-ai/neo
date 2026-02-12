"""
Org and project detection for scoped fact storage.

Parses git remotes to extract organization identity and generates
stable project IDs from codebase root paths.
"""

import hashlib
import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def detect_org_and_project(codebase_root: Optional[str] = None) -> tuple[str, str]:
    """Detect organization ID and project ID from git remote and codebase root.

    Args:
        codebase_root: Path to the codebase root directory.

    Returns:
        Tuple of (org_id, project_id). org_id defaults to "unknown",
        project_id defaults to "" if no codebase_root provided.
    """
    org_id = _detect_org(codebase_root)
    project_id = _compute_project_id(codebase_root)
    return org_id, project_id


def _detect_org(codebase_root: Optional[str] = None) -> str:
    """Extract org name from git remote URL.

    Supports:
        - GitHub:      github.com/{org}/repo
        - Azure DevOps: dev.azure.com/{org}/project/_git/repo
        - GitLab:      gitlab.com/{org}/repo
        - SSH variants: git@github.com:{org}/repo.git
    """
    remote_url = _get_git_remote_url(codebase_root)
    if not remote_url:
        return "unknown"

    return _parse_org_from_url(remote_url)


def _get_git_remote_url(codebase_root: Optional[str] = None) -> str:
    """Get git remote origin URL."""
    try:
        cmd = ["git", "remote", "get-url", "origin"]
        kwargs = {}
        if codebase_root:
            kwargs["cwd"] = codebase_root
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
            **kwargs,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        logger.debug("Failed to get git remote URL")
    return ""


def _parse_org_from_url(url: str) -> str:
    """Parse organization name from a git remote URL.

    Handles HTTPS and SSH formats for GitHub, Azure DevOps, and GitLab.
    """
    url = url.strip()

    # Azure DevOps SSH: git@ssh.dev.azure.com:v3/{org}/{project}/repo
    # Must be checked before generic SSH to avoid matching "v3" as org
    azure_ssh_match = re.match(r"git@ssh\.dev\.azure\.com:v3/([^/]+)/", url)
    if azure_ssh_match:
        return azure_ssh_match.group(1)

    # SSH format: git@github.com:org/repo.git
    ssh_match = re.match(r"git@([^:]+):([^/]+)/", url)
    if ssh_match:
        return ssh_match.group(2)

    # Azure DevOps HTTPS: https://dev.azure.com/{org}/{project}/_git/repo
    azure_match = re.match(r"https?://dev\.azure\.com/([^/]+)/", url)
    if azure_match:
        return azure_match.group(1)

    # Generic HTTPS: https://github.com/{org}/repo
    # Also handles gitlab.com, bitbucket.org, etc.
    https_match = re.match(r"https?://[^/]+/([^/]+)/", url)
    if https_match:
        return https_match.group(1)

    return "unknown"


def _compute_project_id(codebase_root: Optional[str] = None) -> str:
    """Compute stable project ID from codebase root path.

    Uses SHA256[:16] of the resolved absolute path, consistent with
    the existing PersistentReasoningMemory approach.
    """
    if not codebase_root:
        return ""
    resolved = str(Path(codebase_root).resolve())
    return hashlib.sha256(resolved.encode()).hexdigest()[:16]
