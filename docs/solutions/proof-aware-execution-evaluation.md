# Proof-aware execution evaluation

**Status:** implemented deterministic safety gate

`neo memory evaluate-execution [--json] [--corpus PATH]` runs the versioned
`benchmarks/execution_loop_v1.json` corpus without an LM, tokens, embeddings, or
user memory. Every scenario runs twice and must produce identical results.

The corpus pins the failure modes that motivated proof-aware execution:

- aggregate success cannot cover a missing configuration boundary;
- stale revisions and state fingerprints cannot satisfy current gates;
- skipped and unavailable evidence never alias to passed;
- waivers require gate policy and an explicit reason;
- duplicate IDs cannot collapse multiple obligations;
- unsupported causal confirmation is downgraded;
- a failed falsifier rejects a candidate;
- parent-task and cross-repository identity survive normalization;
- legacy criteria and version-3 episodes load conservatively.

The command exits nonzero when any scenario fails, deterministic replay differs,
or total local latency exceeds the corpus budget. Its report always includes
model-call and token counts, both of which must remain zero.
