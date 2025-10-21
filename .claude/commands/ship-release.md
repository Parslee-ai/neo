---
name: ship-release
pattern: /ship-release
description: Complete release workflow from version prep through PyPI publication
parameters:
  - name: version
    description: Semantic version number (e.g., "0.7.7")
    required: true
---

# Ship Release

Orchestrates the full release workflow: version prep, PR creation, tagging, and PyPI publication.

## Usage

```bash
/ship-release 0.7.7
```

## What This Does

Runs the complete release workflow for projects with protected main branches:

1. Invokes `/prepare-release` to update versions and changelog
2. Creates release branch (`release/v{version}`)
3. Commits the changes
4. Pushes branch and creates PR
5. **PAUSES for human PR review and merge**
6. After merge, creates git tag
7. Creates GitHub Release (triggers PyPI publish via Actions)
8. Monitors PyPI publication

## Workflow

### Phase 1: Prepare Release

Run `/prepare-release {version}` which:
- Updates CHANGELOG.md
- Updates version in pyproject.toml, src/neo/__init__.py, .claude-plugin/plugin.json
- Builds distributions

### Phase 2: Create Release Branch

```bash
git checkout -b release/v{version}
```

If branch already exists, check it out instead.

### Phase 3: Commit Changes

```bash
git add CHANGELOG.md pyproject.toml src/neo/__init__.py .claude-plugin/plugin.json
git commit -m "chore: bump version to {version}"
```

### Phase 4: Push and Create PR

```bash
git push origin release/v{version}
gh pr create --title "Release v{version}" --body "<changelog summary>"
```

**CHECKPOINT**: Command stops here. Report PR URL and next steps.

User must:
1. Review the PR
2. Verify changelog and version updates
3. Merge the PR

Then run: `/ship-release {version} --continue`

### Phase 5: Create Tag (after PR merge)

```bash
git checkout main
git pull origin main
git tag v{version}
git push origin v{version}
```

### Phase 6: Create GitHub Release

```bash
gh release create v{version} \
  --title "v{version}" \
  --notes "<changelog content>" \
  dist/neo_reasoner-{version}*
```

This triggers the GitHub Actions workflow (.github/workflows/publish.yml) which publishes to PyPI.

### Phase 7: Verify Publication

Check that:
- GitHub Actions workflow completed successfully
- Package appears on PyPI: https://pypi.org/project/neo-reasoner/

Report status and provide link to new release.

## Options

- `--continue`: Resume after PR is merged (skips phases 1-4)
- `--dry-run`: Show what would happen without making changes

## Error Handling

**If PR creation fails**: Check if PR already exists, provide URL if so

**If tag already exists**: Report conflict, suggest incrementing version

**If GitHub Actions fails**: Check workflow logs at github.com/{repo}/actions

**If PyPI publish fails**: Check Actions logs for authentication or build issues

## Notes

- Main branch must be protected (requires PR for merges)
- Requires `gh` CLI authenticated with GitHub
- Requires PyPI configured with Trusted Publishers in GitHub Actions
- Safe to re-run - checks existing state at each phase
- Use `/prepare-release` alone if you just want to prep without full release
