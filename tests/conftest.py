"""Shared test fixtures for Neo tests.

Prevents all tests from touching ~/.neo/ by redirecting Path.home()
to a temporary directory. This stops tests from corrupting live
memory files (global_memory.json, local_*.json, facts/).
"""

import site
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def isolate_neo_home(tmp_path, monkeypatch):
    """Redirect Path.home() so no test touches ~/.neo/.

    Pin PYTHONUSERBASE to the real user-base before patching HOME so
    subprocess CLI invocations (e.g. `python -m neo`) can still resolve
    a user-site editable install — user-site is otherwise derived from
    $HOME and would point into the fake home.
    """
    monkeypatch.setenv("PYTHONUSERBASE", site.getuserbase())
    # Pin the fastembed model cache to the REAL user cache before patching HOME,
    # so in-process and subprocess embedders reuse the already-downloaded ~400 MB
    # model instead of re-fetching it into the throwaway fake home (which would
    # blow past subprocess CLI timeouts). Model cache is a shared read-mostly
    # asset, distinct from the ~/.neo state this fixture isolates.
    monkeypatch.setenv(
        "NEO_FASTEMBED_CACHE_DIR", str(Path.home() / ".cache" / "neo" / "fastembed")
    )
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))  # Also patch $HOME for expanduser()
