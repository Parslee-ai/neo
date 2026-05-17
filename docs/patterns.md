# Neo Patterns

Reusable patterns discovered during development.

---

### Layered Budget Allocation (2026-03-18)

**Context**: Managing shared token/memory budgets across multiple content layers
**Learning**: Allocate budget in strict priority order with caps to prevent starvation. Use a greedy first-fit accumulator that stops when budget exhausted.
**Reference**: `_accumulate_within_budget` in `src/neo/memory/context.py`

---

### Inline Change Annotations vs Separate Sections (2026-03-18)

**Context**: Presenting change history to LLMs
**Learning**: Inline `(changed from: X)` annotations outperform separate "Recently Changed" sections. LLMs process co-located information better. Validated at 95.8% decision accuracy in statebench.
**Reference**: PR #75; see annotation handling in `src/neo/memory/context.py`

---

### Size Heuristics Without Tokenizer Dependencies (2026-03-18)

**Context**: Estimating token counts for budget enforcement
**Learning**: `len(text) // 4` is sufficient for budget comparisons. Exact counts aren't needed - monotonic estimates enable relative ordering. Avoids tokenizer dependencies and model-specific code.
**Reference**: `Fact.size_hint()` in `src/neo/memory/models.py`
