# Evidence-learning evaluation

**Status:** deterministic milestone gate, corpus schema v1

Neo must not claim that persistent learning improved engineering decisions from
an anecdote or a self-graded generation. The command below runs a local,
repeatable comparison through the real FactStore retrieval, LearningEpisode,
candidate-promotion, regression, scoping, and explanation paths:

```bash
neo memory evaluate-learning
neo memory evaluate-learning --json
neo memory evaluate-learning --workspace /tmp/neo-learning-eval
```

The default workspace is temporary. `--workspace` retains the generated facts
and episodes for inspection. The runner disables normal metrics emission and
does not read or write the user's Neo memory.

## Comparison modes

The versioned corpus is
[`benchmarks/learning_loop_v1.json`](../../benchmarks/learning_loop_v1.json).
Every run compares:

1. `memory_disabled`: no persistence or retrieval;
2. `legacy_immediate_memory`: the pre-episode policy, modeled explicitly—each
   generation immediately becomes a fact and UNVERIFIED acts as weak success;
3. `evidence_driven`: the implemented candidate, attribution, contradiction,
   rollback, and cross-project promotion pipeline.

The legacy mode is an executable policy comparator, not a second production
memory implementation.

## Required deterministic scenarios

The evidence-driven mode must pass all twelve milestone scenarios:

1. retrieve a twice-verified project convention on a related later task;
2. retain but not promote an unverified generation;
3. reduce confidence after an attributed later failure;
4. preserve an unrelated structural fact during rollback;
5. retain explicit contradiction episode IDs and rollback reason;
6. prevent a project fact from appearing in an unrelated project;
7. prevent global promotion after one project, then allow it after four
   supporting episodes across two projects;
8. load an older record conservatively and preserve malformed input;
9. produce the same ranking twice from identical persisted state and query;
10. explain provenance and support without a model call.
11. repeat the verified-improvement sequence for a second, distinct repository
    task family;
12. reconstruct both complete local chains: supporting episodes, promotion
    mutation, later scored retrieval, detectable reasoning use, suggestion,
    verification, and accepted outcome.

## Metrics and acceptance

Each mode reports task-success rate, constraint adherence, retrieval precision,
harmful-memory rate, unsupported-promotion rate, repeat-error rate, project
leakage, wall-clock latency, model calls, and token usage. Task success and
retrieval precision are operational proxies over two repeated verified task
families: the expected convention must appear in later retrieval and drive a
persisted, cited follow-up decision.

The v1 safety gate requires zero harmful memory, unsupported promotion, repeat
error, and project leakage; evidence-mode latency must stay below 500 ms on the
small fixed corpus; model calls and token use must remain zero. Evidence-driven
learning must improve at least one primary quality metric over memory-disabled
and may not degrade task success, constraint adherence, or retrieval precision.
Any failed scenario or threshold makes the command exit nonzero.

The deterministic hashing embedder is benchmark-only. It removes network,
model-cache, and hardware variation while still exercising Neo's dense/BM25
fusion and ranking code. Production embedding behavior remains unchanged.
