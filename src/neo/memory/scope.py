"""
Org and project detection for scoped fact storage.

Parses git remotes to extract organization identity and generates
stable project IDs from codebase root paths.
"""

import hashlib
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Resolving one project's identity asks git for the same remote 3x (org id,
# project id, and the transcript sources' repo lookup), each a fork+exec. Across
# the observer's 25-project sweep that measured ~100 forks per cycle, ~26k over
# a week, for an answer that cannot change mid-sweep. Cache it.
#
# Deliberately NOT unbounded-lifetime: a remote really can change under a daemon
# that runs for days, and a stale hit would silently write facts under the wrong
# project_id. The observer clears this once per cycle (see
# ``Observer._cycle_global``), so staleness is bounded by one cycle while the
# intra-sweep redundancy — the actual waste — is gone.
_REMOTE_URL_CACHE: dict[str, str] = {}


def clear_remote_url_cache() -> None:
    """Drop memoized git remotes. Called once per observer cycle."""
    _REMOTE_URL_CACHE.clear()


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
    """Get git remote origin URL. Memoized per root; see ``_REMOTE_URL_CACHE``."""
    try:
        key = os.path.abspath(codebase_root) if codebase_root else os.getcwd()
    except OSError:  # cwd deleted out from under us
        key = codebase_root or ""
    if key in _REMOTE_URL_CACHE:
        return _REMOTE_URL_CACHE[key]

    url = ""
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
            url = result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        logger.debug("Failed to get git remote URL")
    # A non-repo root is the common case (7 of 37 here), so "" is worth caching
    # — but ONLY when the answer is structural. A *transient* git failure (fork
    # limit, 5s timeout under load) also yields "", and memoizing that would
    # silently downgrade an established project to its path-hash project_id for
    # the rest of the cycle, writing its facts into a different file that
    # nothing reconciles back. Confirm the repo is genuinely absent first.
    if url or not _has_git_dir(key):
        _REMOTE_URL_CACHE[key] = url
    return url


def _has_git_dir(root: str) -> bool:
    """Is this a git working tree? ``.git`` is a dir in a clone, a file in a
    worktree/submodule — both count."""
    try:
        return Path(root, ".git").exists()
    except OSError:
        return False


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


def _normalize_remote_url(url: str) -> str:
    """Normalize a git remote URL so the same repo on different clones hashes
    to the same project ID.

    - Strips embedded credentials (e.g. `https://token@github.com/...`)
    - Strips scheme (`https://`, `git://`, `ssh://`)
    - Converts SSH `host:org/repo` form to `host/org/repo`
    - Strips `.git` suffix and trailing slashes
    - Lowercases network hosts (paths stay case-sensitive)

    Returns "" if the input is empty.
    """
    if not url:
        return ""

    url = url.strip()

    is_network = False
    if "://" in url and not url.startswith("file://"):
        is_network = True
    elif re.match(r"^[^@/:]+@[^:]+:", url):  # git@host:org/repo
        is_network = True

    url = re.sub(r"://[^@]+@", "://", url)
    url = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", url)
    url = re.sub(r"^([^@/:]+)@([^:/]+):", r"\2/", url)
    url = re.sub(r"\.git/?$", "", url)
    url = url.rstrip("/")

    if is_network:
        # Lowercase only the host portion, leave path alone (GitHub paths
        # are case-insensitive in practice but other forges aren't).
        parts = url.split("/", 1)
        parts[0] = parts[0].lower()
        url = "/".join(parts)

    return url


def _compute_legacy_project_id(codebase_root: Optional[str] = None) -> str:
    """Path-only project ID — the format used before git-remote hashing.

    Kept so the FactStore can find and migrate fact files written under the
    old scheme.
    """
    if not codebase_root:
        return ""
    resolved = str(Path(codebase_root).resolve())
    return hashlib.sha256(resolved.encode()).hexdigest()[:16]


def _compute_project_id(codebase_root: Optional[str] = None) -> str:
    """Compute stable project ID, preferring the git remote URL hash.

    Priority:
      1. SHA256[:16] of the normalized `git remote get-url origin`
         — portable across machines/clones/worktrees of the same repo
      2. SHA256[:16] of the resolved absolute path
         — fallback for repos without a remote, or non-git directories
      3. "" if no codebase_root provided
    """
    if not codebase_root:
        logger.warning(
            "No codebase_root provided — project_id will be empty. "
            "Session saves and outcome detection will be disabled."
        )
        return ""

    normalized_remote = _normalize_remote_url(_get_git_remote_url(codebase_root))
    if normalized_remote:
        return hashlib.sha256(normalized_remote.encode()).hexdigest()[:16]

    return _compute_legacy_project_id(codebase_root)
