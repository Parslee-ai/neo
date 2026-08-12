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
- Updates the version in `pyproject.toml` (the only hand edit) and runs
  `make sync-version` to propagate it to `src/neo/__init__.py` and both plugin
  manifests
- Builds distributions

### Phase 2: Create Release Branch

```bash
git checkout -b release/v{version}
```

If branch already exists, check it out instead.

### Phase 3: Commit Changes

```bash
git add CHANGELOG.md pyproject.toml src/neo/__init__.py \
        .claude-plugin/plugin.json plugins/neo/.codex-plugin/plugin.json
git commit -m "chore: bump version to {version}"
```

Four version files, not three — the Codex plugin manifest is the one that gets
forgotten. `python tools/sync_version.py --check` should report all four in sync
before committing.

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

**Verify the release PR actually merged before tagging.** This phase only pulls
`main` and tags it, so running `--continue` early tags a tree with no version
bump — and Phase 6 then publishes anyway, leaving PyPI carrying `{version}` while
the tag points at a commit that still says the previous one. Nothing downstream
catches that.

```bash
gh pr view {pr-number} --json state -q .state   # must print MERGED
git checkout main
git pull origin main
grep '^version' pyproject.toml                  # must print {version}
git tag v{version}
git push origin v{version}
```

### Phase 6: Create GitHub Release

**The release gate runs here.** Creating the release triggers `publish.yml`,
which will not build a wheel until `language-roundtrip` is green: one real LLM
round trip each for C#, TypeScript and Python against generated fixture
repositories, asserting that the language's files reach the prompt and that
the model can name a symbol present in exactly one of them. A red language
stops the chain at `build` and nothing reaches PyPI.

The job **fails** rather than skips when `ANTHROPIC_API_KEY` is not configured
as a repository secret — an absent credential is a red gate, not a green one.
Configure it before the first gated release. Full details, including how to
dry-run the gate without cutting a release, are in
[`docs/release-gate.md`](../../docs/release-gate.md).

```bash
gh release create v{version} \
  --title "v{version}" \
  --notes "<changelog content>" \
  dist/neo_reasoner-{version}*
```

This triggers `.github/workflows/publish.yml`, which publishes to PyPI.

**The attached files are not what gets published.** The workflow's `build` job
checks out the tag, runs `python -m build` itself, and `publish-pypi` uploads
*that* artifact via Trusted Publishers. The Phase 1 `dist/` build is a local
verification step and the attachments are convenience copies for the release
page. Two consequences worth knowing: a stale `dist/` cannot corrupt what lands
on PyPI, but a missing one makes `gh release create` fail on the glob — likely
when `--continue` runs in a fresh clone or after `make clean`. Re-run
`python -m build` in that case.

The distinction is easy to miss because both builds normally produce
byte-identical artifacts, so a mismatch would not be visible on the release page.

### Phase 7: Verify Publication

Check that:
- `language-roundtrip` passed — all three languages green
- GitHub Actions workflow completed successfully
- Package appears on PyPI: https://pypi.org/project/neo-reasoner/

Report status and provide link to new release.

## Options

- `--continue`: Resume after the release PR is merged, skipping phases 1-4.
  Confirm the merge landed before tagging — see Phase 5.
- `--dry-run`: Report each phase and the commands it would run, changing nothing:
  no version edit, no CHANGELOG entry, no branch, no PR, no tag, no release.

## Dry-running the publish workflow

`publish-testpypi` exists for this and is gated to manual dispatch:

```bash
gh workflow run publish.yml
```

**Its publish step fails today.** No Trusted Publisher is configured for this
repo on TestPyPI, so the OIDC exchange returns `invalid-publisher` (last
attempted 2026-07-26). `ci-check`, `build` and the artifact round-trip still run,
which is most of what a workflow change needs smoke-tested — so a dispatch is
still worth doing, you just cannot read the upload as a signal. Configure at
<https://test.pypi.org/manage/account/publishing/> (owner `Parslee-ai`, repo
`neo`, workflow `publish.yml`) to make the publish half work. See also the NOTE
above `publish-testpypi` in the workflow, and CONTRIBUTING.md.

## Error Handling

**If PR creation fails**: Check if PR already exists, provide URL if so

**If tag already exists**: Report conflict, suggest incrementing version

**If GitHub Actions fails**: Check workflow logs at github.com/{repo}/actions

**If `language-roundtrip` fails**: reproduce it for free first —
`pytest -m invariants -v` runs the same fixtures with no model call and will
usually name the broken invariant. If the battery is green and only the round
trip is red, the model stopped seeing a file the gatherer still selects; run
`NEO_RELEASE_ROUNDTRIP=1 pytest -m roundtrip -v` locally. Do not work around
it by removing the job from `build`'s `needs`.

**If PyPI publish fails**: Check Actions logs for authentication or build issues

## Notes

- Main branch is protected: it requires a PR **and one approving review**.
  GitHub does not let an account approve its own PR, so a solo release needs
  either a second reviewer or `gh pr merge --admin` (works because
  `enforce_admins` is false). Plain `gh pr merge` fails with "the base branch
  policy prohibits the merge" — that is the protection working, not a bug.
- Fill the PR body from `.github/PULL_REQUEST_TEMPLATE.md`
- Requires `gh` CLI authenticated with GitHub
- Requires PyPI configured with Trusted Publishers in GitHub Actions
- Safe to re-run - checks existing state at each phase
- Use `/prepare-release` alone if you just want to prep without full release
