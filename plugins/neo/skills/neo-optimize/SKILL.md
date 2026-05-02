---
name: neo-optimize
description: Ask Neo for optimization suggestions on a function, file, or hot path. Targets algorithmic improvements, redundant work, allocation/hot-loop issues — not micro-style.
---

# Neo Optimization Analysis

When the user invokes this skill (`$neo-optimize <target>`), do the following:

1. **Locate the target.** It may be a function name (`process_large_dataset`), a file, or a description ("the user-search query path"). Use Grep/Read to find the actual implementation.

2. **Capture the current implementation plus its callers** if you can do so cheaply. Neo can suggest better algorithms, but only if it sees how the code is used.

3. **Invoke Neo with an optimization-framed prompt.** Allow up to 5 minutes.

   ```bash
   neo <<'QUERY'
   Suggest optimizations for the following code. Focus on: algorithmic improvements (lower asymptotic complexity), redundant computation, allocation in hot loops, IO batching opportunities. Skip micro-style changes.

   <paste current implementation + relevant callers>
   QUERY
   ```

4. **Present Neo's suggestions ranked by expected impact.** Each CodeSuggestion includes `estimated_risk` and `blast_radius` — surface those alongside the recommendation.

5. **For high-risk changes, recommend benchmarking before applying.** Neo's confidence reflects pattern-match strength, not measured speedup.

## Notes

- Algorithmic suggestions tend to come back with high confidence when Neo has seen similar patterns before — that's the memory-driven reasoning effort kicking in.
- If Neo returns "I cannot find evidence" or low-confidence-only output, that's a signal the optimization isn't obvious and warrants human investigation rather than blind application.
