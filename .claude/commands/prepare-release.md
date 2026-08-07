---
name: prepare-release
pattern: /prepare-release
description: Prepare a new release by updating versions, changelog, and building distributions
parameters:
  - name: new_version
    description: Semantic version number (e.g., "0.7.7")
    required: true
---

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
4. **Updates** the version in `pyproject.toml` — the single source of truth — and
   runs `make sync-version` to propagate it to the three derived files
5. **Builds** wheel and sdist distributions
6. **Reports** what was done and next steps

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

**Edit ONE file, then propagate.** `pyproject.toml` is the source of truth:

```bash
# 1. Change `version = "X.Y.Z"` in pyproject.toml (the only hand edit)
make sync-version        # propagates to the three derived files
python tools/sync_version.py --check   # confirms all four in sync
```

`make sync-version` rewrites `src/neo/__init__.py`, `.claude-plugin/plugin.json`
and `plugins/neo/.codex-plugin/plugin.json`. **Never hand-edit those three** —
they are derived, and `tests/test_host_adapter_parity.py` fails if they drift
from `pyproject.toml`.

This step used to list the files for manual editing and named only three of the
four, omitting the Codex manifest. That is precisely how the package and the two
plugin manifests reached 0.41.0 / 0.37.0 / 0.19.0 — a documented step with no
enforcement gets skipped. Do not reintroduce a hand-edit list here.

### Step 5: Build Distributions

Run:
```bash
python -m build --wheel --outdir dist/
python -m build --sdist --outdir dist/
```

Verify both files were created in `dist/` directory.

### Step 6: Report Results

Show what was updated and provide next steps:
```bash
git add CHANGELOG.md pyproject.toml src/neo/__init__.py \
        .claude-plugin/plugin.json plugins/neo/.codex-plugin/plugin.json
git commit -m "chore: bump version to {new_version}"
```

**Do not tag or push to main from here.** `main` is protected and requires a
pull request plus one approving review, so `git push origin main --tags` — what
this step used to say — fails outright. Tagging before the release PR merges
also points the tag at a commit that may never reach `main`. Use
`/ship-release` for the branch → PR → merge → tag → GitHub Release sequence.

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
