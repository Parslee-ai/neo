"""Tests for neo.memory.scope - org/project detection."""

from neo.memory.scope import (
    _compute_project_id,
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


class TestComputeProjectId:
    def test_returns_16_char_hex(self):
        pid = _compute_project_id("/some/path")
        assert len(pid) == 16
        assert all(c in "0123456789abcdef" for c in pid)

    def test_same_path_same_id(self):
        assert _compute_project_id("/foo/bar") == _compute_project_id("/foo/bar")

    def test_different_paths_different_ids(self):
        assert _compute_project_id("/foo/bar") != _compute_project_id("/baz/qux")

    def test_none_returns_empty(self):
        assert _compute_project_id(None) == ""


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
