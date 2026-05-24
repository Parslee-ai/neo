"""Tests for neo.memory.scope - org/project detection."""

from unittest.mock import patch

from neo.memory.scope import (
    _compute_legacy_project_id,
    _compute_project_id,
    _normalize_remote_url,
    _parse_org_from_url,
    detect_org_and_project,
)


class TestParseOrgFromUrl:
    def test_github_https(self):
        assert _parse_org_from_url("https://github.com/parslee-ai/neo.git") == "parslee-ai"

    def test_github_ssh(self):
        assert _parse_org_from_url("git@github.com:parslee-ai/neo.git") == "parslee-ai"

    def test_azure_devops_https(self):
        url = "https://dev.azure.com/mycompany/myproject/_git/myrepo"
        assert _parse_org_from_url(url) == "mycompany"

    def test_azure_devops_ssh(self):
        url = "git@ssh.dev.azure.com:v3/mycompany/myproject/myrepo"
        assert _parse_org_from_url(url) == "mycompany"

    def test_gitlab_https(self):
        assert _parse_org_from_url("https://gitlab.com/mygroup/myrepo.git") == "mygroup"

    def test_gitlab_ssh(self):
        assert _parse_org_from_url("git@gitlab.com:mygroup/myrepo.git") == "mygroup"

    def test_bitbucket_https(self):
        assert _parse_org_from_url("https://bitbucket.org/myteam/myrepo.git") == "myteam"

    def test_empty_url(self):
        assert _parse_org_from_url("") == "unknown"

    def test_malformed_url(self):
        assert _parse_org_from_url("not-a-url") == "unknown"

    def test_whitespace_stripped(self):
        assert _parse_org_from_url("  https://github.com/myorg/repo.git  ") == "myorg"


class TestNormalizeRemoteUrl:
    def test_strips_https_credentials(self):
        url = "https://ghp_xxxx@github.com/parslee-ai/neo.git"
        assert _normalize_remote_url(url) == "github.com/parslee-ai/neo"

    def test_strips_dot_git(self):
        assert _normalize_remote_url("https://github.com/o/r.git") == "github.com/o/r"

    def test_ssh_and_https_canonicalize_same(self):
        ssh = _normalize_remote_url("git@github.com:parslee-ai/neo.git")
        https = _normalize_remote_url("https://github.com/parslee-ai/neo.git")
        assert ssh == https == "github.com/parslee-ai/neo"

    def test_lowercases_network_host(self):
        assert _normalize_remote_url("HTTPS://GitHub.com/Org/Repo.git") == "github.com/Org/Repo"

    def test_empty_returns_empty(self):
        assert _normalize_remote_url("") == ""

    def test_trailing_slash_stripped(self):
        assert _normalize_remote_url("https://github.com/o/r/") == "github.com/o/r"


class TestComputeProjectId:
    """Tests use mocked git-remote lookups so they don't depend on the host's
    git config or current directory."""

    def test_returns_16_char_hex(self):
        with patch("neo.memory.scope._get_git_remote_url", return_value=""):
            pid = _compute_project_id("/some/path")
        assert len(pid) == 16
        assert all(c in "0123456789abcdef" for c in pid)

    def test_same_path_same_id_when_no_remote(self):
        with patch("neo.memory.scope._get_git_remote_url", return_value=""):
            assert _compute_project_id("/foo/bar") == _compute_project_id("/foo/bar")

    def test_different_paths_different_ids_when_no_remote(self):
        with patch("neo.memory.scope._get_git_remote_url", return_value=""):
            assert _compute_project_id("/foo/bar") != _compute_project_id("/baz/qux")

    def test_none_returns_empty(self):
        assert _compute_project_id(None) == ""

    def test_remote_url_preferred_over_path(self):
        """Different paths with the same remote URL must produce the same ID."""
        with patch("neo.memory.scope._get_git_remote_url",
                   return_value="git@github.com:parslee-ai/neo.git"):
            a = _compute_project_id("/Users/alice/git/neo")
            b = _compute_project_id("/tmp/clone/neo")
        assert a == b

    def test_ssh_and_https_remote_produce_same_id(self):
        with patch("neo.memory.scope._get_git_remote_url",
                   return_value="git@github.com:parslee-ai/neo.git"):
            ssh_id = _compute_project_id("/x")
        with patch("neo.memory.scope._get_git_remote_url",
                   return_value="https://github.com/parslee-ai/neo.git"):
            https_id = _compute_project_id("/y")
        assert ssh_id == https_id

    def test_falls_back_to_path_when_no_remote(self):
        with patch("neo.memory.scope._get_git_remote_url", return_value=""):
            pid = _compute_project_id("/foo/bar")
        assert pid == _compute_legacy_project_id("/foo/bar")

    def test_legacy_id_diverges_when_remote_exists(self):
        with patch("neo.memory.scope._get_git_remote_url",
                   return_value="git@github.com:parslee-ai/neo.git"):
            new_id = _compute_project_id("/some/path")
        legacy_id = _compute_legacy_project_id("/some/path")
        assert new_id != legacy_id


class TestDetectOrgAndProject:
    def test_no_codebase_root(self):
        org_id, project_id = detect_org_and_project(None)
        # org_id may be "unknown" or detected from cwd's git
        assert isinstance(org_id, str)
        assert project_id == ""

    def test_nonexistent_path(self):
        org_id, project_id = detect_org_and_project("/nonexistent/path/12345")
        assert org_id == "unknown"
        assert len(project_id) == 16  # Still computes hash
