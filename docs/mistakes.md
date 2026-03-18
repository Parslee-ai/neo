# Neo Mistakes

Lessons learned from bugs, regressions, and implementation errors.

---

### Exempt Resources Can Starve Siblings (2026-03-18)

**Context**: Token budget enforcement with "exempt" categories
**Mistake**: Initially marked constraints as "exempt from budget" - this allowed large constraint sets to consume the entire context window, leaving nothing for facts.
**Fix**: Cap "exempt" categories at a fraction (2/3) to reserve minimum for non-exempt layers.
**Reference**: PR #75 commit a7e7d24

---

### Capping Display Lists Breaks Lookup Operations (2026-03-18)

**Context**: Limiting invalidated facts display while using them for annotation lookups
**Mistake**: Had `MAX_INVALIDATED_FACTS = 3` but annotations needed to lookup ANY invalidated fact by ID. The 4th+ invalidated facts couldn't be annotated.
**Fix**: Keep full lists for lookup operations even when display is limited. Separate "what to show" from "what to lookup".
**Reference**: PR #75 commit a7e7d24

---

### Check Conditions Before Side Effects (2026-03-18)

**Context**: Warning when truncation occurs
**Mistake**: Checked `if truncated > cap` AFTER calling `_accumulate_within_budget`. Since accumulation already caps the result, the condition rarely triggered.
**Fix**: Capture uncapped total BEFORE calling the capping function, then compare that to the cap.
**Reference**: PR #75 commit d1d82f6

---

### Docstrings Lying After Behavior Changes (2026-03-18)

**Context**: Documentation accuracy
**Mistake**: Docstring claimed "Constraints are always included (exempt from budget)" even after adding the 2/3 cap.
**Fix**: Update documentation in the SAME commit as behavior changes. Treat docstring updates as part of the feature, not cleanup.
**Reference**: PR #75 commit edcf68f
