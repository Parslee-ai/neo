"""Shared test fixtures for Neo tests.

Prevents all tests from touching ~/.neo/ by redirecting Path.home()
to a temporary directory. This stops tests from corrupting live
memory files (global_memory.json, local_*.json, facts/).
"""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def isolate_neo_home(tmp_path, monkeypatch):
    """Redirect Path.home() so no test touches ~/.neo/."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
