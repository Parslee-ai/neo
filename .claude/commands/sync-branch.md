---
name: sync-branch
pattern: /sync-branch
description: Sync current branch with base branch. Claude (you) resolves conflicts intelligently, not automated scripts.
parameters:
  - name: base_branch
    description: Base branch to sync with (e.g., main, master)
    required: true
argument-hint: <base_branch>
---

# Sync Branch with Base

Sync the current branch with the base branch before PR review. You (Claude) will intelligently resolve any merge conflicts.

## Workflow

### Step 1: Fetch Latest Changes

Fetch the latest changes from origin:

```bash
git fetch origin
```

### Step 2: Merge Base Branch

Attempt to merge the base branch:

```bash
git merge origin/$BASE_BRANCH --no-edit
```

### Step 3: Handle Conflicts (Claude's Responsibility)

If there are conflicts, YOU (Claude) must resolve them intelligently:

1. Check `git status` to see conflicted files
2. For each conflicted file, review the conflict markers
3. Use your understanding of the code to resolve conflicts intelligently
4. Stage resolved files with `git add`
5. Complete merge with `git commit --no-edit`

**If conflicts cannot be resolved** (e.g., too complex, unclear intent):
```bash
git merge --abort
```
Then return `synced: false` with a clear explanation.

### Step 4: Push Changes

If the merge was successful (with or without conflicts):

```bash
git push origin HEAD
```

## Final Output (STRICT JSON)

```json
{
  "synced": true|false,
  "had_conflicts": true|false,
  "summary": "Brief summary of sync result"
}
```

Examples:
- Success without conflicts: `{"synced": true, "had_conflicts": false, "summary": "Branch synced with main, no conflicts"}`
- Success with resolved conflicts: `{"synced": true, "had_conflicts": true, "summary": "Branch synced with main, resolved conflicts in 2 files"}`
- Aborted due to conflicts: `{"synced": false, "had_conflicts": true, "summary": "Aborted merge with main: conflicts in config.json require manual resolution"}`
- Failure: `{"synced": false, "had_conflicts": false, "summary": "Failed to fetch from origin"}`
