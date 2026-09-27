#!/usr/bin/env python3
"""Propagate the version in pyproject.toml to every file that restates it.

One release version is written down in five places:

    pyproject.toml                                <- the source of truth (the build reads it)
    src/neo/__init__.py                           <- __version__
    .claude-plugin/plugin.json                    <- Claude Code plugin manifest
    plugins/neo/.codex-plugin/plugin.json         <- Codex CLI plugin manifest
    plugins/cursor-neo/.cursor-plugin/plugin.json <- Cursor plugin manifest

`prepare-release` used to ask a human to edit the derived files by hand. That
is a documented step with no enforcement, and it got skipped: the package, the
Claude manifest and the Codex manifest reached 0.41.0 / 0.37.0 / 0.19.0 before
anyone noticed. `tests/test_host_adapter_parity.py` now catches the drift, but
catching it after the fact is worse than not creating it.

Usage:
    python tools/sync_version.py            # rewrite the derived files
    python tools/sync_version.py --check    # exit 1 if any is stale (CI/hook)

Edits are surgical string replacements, not re-serialization: rewriting a JSON
manifest through `json.dumps` would reformat the whole file and bury a one-line
version bump in an unreviewable diff.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PYPROJECT = REPO / "pyproject.toml"

# (path, regex with a single capturing group around the version literal)
DERIVED: list[tuple[Path, re.Pattern[str]]] = [
    (REPO / "src" / "neo" / "__init__.py",
     re.compile(r'^(__version__\s*=\s*")([^"]+)(")', re.MULTILINE)),
    (REPO / ".claude-plugin" / "plugin.json",
     re.compile(r'^(\s*"version"\s*:\s*")([^"]+)(")', re.MULTILINE)),
    (REPO / "plugins" / "neo" / ".codex-plugin" / "plugin.json",
     re.compile(r'^(\s*"version"\s*:\s*")([^"]+)(")', re.MULTILINE)),
    (REPO / "plugins" / "cursor-neo" / ".cursor-plugin" / "plugin.json",
     re.compile(r'^(\s*"version"\s*:\s*")([^"]+)(")', re.MULTILINE)),
]

_PYPROJECT_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def source_version() -> str:
    """Read the authoritative version out of pyproject.toml."""
    match = _PYPROJECT_VERSION.search(PYPROJECT.read_text())
    if not match:
        raise SystemExit(f"no `version = \"...\"` found in {PYPROJECT}")
    return match.group(1)


def sync(version: str, *, check_only: bool) -> list[str]:
    """Rewrite (or inspect) every derived file. Returns the stale ones."""
    stale: list[str] = []
    for path, pattern in DERIVED:
        text = path.read_text()
        match = pattern.search(text)
        if not match:
            raise SystemExit(f"no version field found in {path}")
        current = match.group(2)
        if current == version:
            continue
        rel = path.relative_to(REPO)
        stale.append(f"{rel}: {current} -> {version}")
        if not check_only:
            # count=1: only the first match, so a nested "version" key added
            # later cannot be clobbered silently.
            path.write_text(pattern.sub(rf"\g<1>{version}\g<3>", text, count=1))
    return stale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true",
        help="report drift and exit 1 without writing anything",
    )
    args = parser.parse_args()

    version = source_version()
    stale = sync(version, check_only=args.check)

    if not stale:
        print(f"version {version}: all files in sync")
        return 0

    if args.check:
        print(f"version {version}: {len(stale)} file(s) out of sync", file=sys.stderr)
        for line in stale:
            print(f"  {line}", file=sys.stderr)
        print("run: python tools/sync_version.py", file=sys.stderr)
        return 1

    for line in stale:
        print(f"  updated {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
