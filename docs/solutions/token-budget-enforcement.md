# Solution: Token Budget Enforcement and Inline Change Annotations

**Date**: 2026-03-18
**PR**: https://github.com/Parslee-ai/neo/pull/75
**Type**: feature

## Problem

Two issues with Neo's context assembly:

1. **Unbounded context growth** - Context grew without limit as facts accumulated. The `assemble()` method had no mechanism to cap output size, risking prompt overflow.

2. **Ineffective change presentation** - Superseded facts were shown in a separate "Recently Changed" section. LLMs process inline annotations better than separated sections (validated at 95.8% decision accuracy in statebench).

## Context

Applies when:
- Building context for LLM prompts from a fact store
- Managing memory systems with token/character budgets
- Presenting change history to language models

Ported from statebench's memgine engine.

## Solution

### Token Budget Enforcement

Added `max_tokens` parameter (default 12000) to `ContextAssembler.assemble()` with layered budget allocation:

```
Total Budget (max_tokens)
    |
    +-- Constraint Cap (2/3 of budget) --> constraints layer
    |
    +-- Remaining Budget (shared) --> valid_facts (guaranteed at-least-one)
                                  --> working_set (SESSION facts)
                                  --> known_unknowns
```

Key implementation: `_accumulate_within_budget()` static method that:
- Greedy first-fit: iterate facts in order, add until budget exhausted
- `at_least_one=True` guarantees first fact included even if over-budget
- Returns early on budget violation

### Inline Change Annotations

Replaced separate "Recently Changed" section with inline `(changed from: X)` annotations:
- Build lookup dict from invalidated facts: `{old_fact.id: old_fact}`
- For facts with `supersedes` field, append `(changed from: <old_body[:80]>)`
- Gracefully handles missing old facts

### Token Estimation

Simple heuristic in `Fact.size_hint()`:
```python
return len(self.subject + self.body) // 4
```
Not precise, just monotonic - sufficient for budget comparisons without tokenizer dependency.

## Key Files

- `src/neo/memory/context.py:29-226` - `ContextAssembler` with `assemble()` and `format_context_for_prompt()`
- `src/neo/memory/models.py:93-95` - `Fact.size_hint()` method
- `tests/test_context_assembly.py:171-263` - `TestTokenBudgetEnforcement` with 10+ edge case tests

## Implementation Notes

### Design Decisions

1. **2/3 constraint cap** - Without this, large CLAUDE.md could consume entire budget. 2/3 ensures constraints get majority priority while reserving 1/3 for facts.

2. **At-least-one only for valid_facts** - Session and known_unknowns are supplementary. If budget is exhausted, better to have one relevant fact than force-include unrelated session data.

3. **Full invalidated list retained** - Needed for annotation lookups even when facts wouldn't otherwise be displayed.

4. **View-layer only** - No storage mutations, no breaking changes to `ContextResult`.

### Patterns Used

- **Layered Budget Allocation** - Priority-ordered consumption of shared resource
- **Greedy First-Fit Accumulation** - Simple, predictable budget enforcement
- **Lookup Table for O(1) Access** - Dict for efficient annotation matching
- **Static Helper Methods** - `_accumulate_within_budget` and `_cosine_similarity` are testable without instance state

## Gotchas

1. **Constraints can starve other layers** - Initial implementation exempted constraints entirely. Fixed: cap at 2/3 of budget.

2. **Capping invalidated facts breaks annotations** - Original `MAX_INVALIDATED_FACTS = 3` caused annotation lookups to fail. Fixed: keep full list.

3. **Warning must trigger BEFORE truncation** - Checking after `_accumulate_within_budget` means truncation already happened. Fixed: capture uncapped total first.

4. **Docstrings must match behavior** - When adding the constraint cap, the docstring still claimed constraints were exempt. Always update docs with behavior changes.

5. **Token estimate is approximate** - `len // 4` can be off 2x either direction. This is acceptable for relative budget enforcement.

6. **No feedback to callers about truncation** - Constraint truncation only logs a warning. Callers don't know if rules were dropped.

## Related

- statebench memgine (source of these improvements)
- `docs/architecture/prompt-enhancement.md` - related prompt construction patterns
