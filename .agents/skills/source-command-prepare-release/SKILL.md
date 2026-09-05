---
name: "source-command-prepare-release"
description: "Prepare a new release by updating versions, changelog, and building distributions"
---

# source-command-prepare-release

Use this skill when the user asks to run the migrated source command `prepare-release`.

## Command Template

# Prepare Release

Updates version numbers, CHANGELOG.md, and builds distributions for a new release.

## Usage

```bash
/prepare-release 0.7.7
```

## What This Does

1. **Validates** the new version is higher than current and follows semver format
2. **Reviews** commits since last release tag
3. **Updates** CHANGELOG.md with new version section and categorized changes
4. **Updates** the version in `pyproject.toml`, then runs `make sync-version`
   to propagate it to the four derived files (`src/neo/__init__.py`, Claude /
   Codex / Cursor plugin manifests)
5. **Syncs** the local editable install metadata so `importlib.metadata.version("neo-reasoner")` matches the new version
6. **Builds** wheel and sdist distributions
7. **Reports** what was done and next steps

## Workflow

### Step 1: Validate

- Read current version from `pyproject.toml`
- Verify new version is higher and matches format `X.Y.Z`
- Check that last git tag exists (e.g., `v0.7.6`)

### Step 2: Analyze Commits

Run `git log v{last_version}..HEAD --oneline` and categorize by type:
- **Fixed**: Commits with "fix:" prefix or fixing bugs
- **Added**: Commits with "feat:" prefix or new features
- **Changed**: Commits with "refactor:" or "perf:" or improvements
- **Documentation**: Commits with "docs:" prefix

### Step 3: Update CHANGELOG

Add new section at top of `CHANGELOG.md`:

```markdown
## [{new_version}] - {today's date}

### Fixed
- List of bug fixes from commits

### Added
- List of new features from commits

### Changed
- List of improvements from commits

### Documentation
- List of doc updates from commits
```

### Step 4: Update Versions

Edit **one** file, then propagate:

- `pyproject.toml`: change `version = "X.Y.Z"` — this is the source of truth.

```bash
make sync-version
```

That rewrites `src/neo/__init__.py`, `.claude-plugin/plugin.json`,
`plugins/neo/.codex-plugin/plugin.json`, and
`plugins/cursor-neo/.cursor-plugin/plugin.json` to match. Do **not** hand-edit
those four: this step used to be "update the version string in all files by
hand", and the manifests silently drifted to 0.37.0 and 0.19.0 while the
package was at 0.41.0. `tests/test_host_adapter_parity.py` fails if they are
out of sync, and `python tools/sync_version.py --check` reports drift without
writing.

### Step 5: Sync Local Editable Metadata

After version files are updated, refresh the current environment's editable
install metadata:

```bash
python -m pip install -e . --no-deps
python -c "import importlib.metadata as m; assert m.version('neo-reasoner') == '{new_version}', m.version('neo-reasoner')"
```

If the command runs in a project virtualenv, use that virtualenv's Python. This
prevents `neo --version` or local tests from reporting the previous release
after `pyproject.toml` was bumped.

### Step 6: Build Distributions

Run:
```bash
python -m build --wheel --outdir dist/
python -m build --sdist --outdir dist/
```

Verify both files were created in `dist/` directory.

### Step 7: Report Results

Show what was updated and provide next steps:
```bash
# Next steps:
git add CHANGELOG.md pyproject.toml src/neo/__init__.py \
  .claude-plugin/plugin.json \
  plugins/neo/.codex-plugin/plugin.json \
  plugins/cursor-neo/.cursor-plugin/plugin.json
git commit -m "chore: bump version to {new_version}"
git tag v{new_version}
git push origin main --tags
```

## Error Handling

- If version format is invalid, report and stop
- If new version isn't higher than current, report and stop
- If git tag doesn't exist, report and stop
- If build fails, show error and stop
- For any failure, explain what went wrong and how to fix it

## Notes

- This command does NOT commit, tag, or push - you review first
- Distributions are built to verify everything works
- Use `/ship-release` for the full release workflow including PR creation
