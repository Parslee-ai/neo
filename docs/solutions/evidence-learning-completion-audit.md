# Evidence-learning milestone completion audit

This audit maps the milestone requirements to deterministic implementation and
verification evidence. It is a release gate, not a claim based on model output.

| Requirement | Implementation evidence | Verification evidence |
|---|---|---|
| A. Typed episodes | `memory.episodes` v2 separates context, retrieval, suggestions, applied actions, verification, outcomes, candidates, and mutations from FactStore | episode atomicity, bounds, v1 migration, malformed/future-record, privacy, and engine trace tests |
| B. Verification | normalized test/lint/type/parser/compile/Neo/user/regression kinds, explicit passed/failed/warning/unavailable/skipped statuses, fail-closed aggregate verdict | static-analysis and episode tests prove missing tools and skipped checks never pass or enable early exit |
| C. Attribution | stable suggestion IDs, repository revisions, separate retrieved/used fact IDs, citation detection, bounded used-fact feedback | outcome and FactStore tests prove only cited facts change and unrelated facts remain unchanged |
| D. Promotion | observation-only episodes, probationary candidates, two-episode project promotion, explicit contradiction/rollback, four-episode/two-project global gate | candidate, deterministic-failure, architecture non-promotion, regression, global-scope, and privacy tests |
| E. Retrieval feedback | score, inclusion, detectable use, retrieved-only versus attributed outcome, failure and regression associations | learning-episode, FactStore attribution, explain, deterministic ranking tests |
| F. Explainability | `neo memory explain` joins facts/tombstones, episodes, mutations, retrievals, outcomes, and supersession without embeddings or an LM | explain tests and both benchmark causal-chain scenarios report zero model calls |
| G. Evaluation | versioned corpus and isolated three-policy evaluator using real FactStore retrieval and episode/promotion paths | twelve scenarios, safety thresholds, quality comparison, latency/call/token gates, CLI isolation test |
| H. Product modes | advise, patch, verify, learn, agent plus explicit host authority policy and no built-in executor | operating-mode and CAR-schema tests; standalone verify uses no provider and agent fails closed |
| I. Compatibility/safety | additive v2 loader, existing FactStore format, provider-neutral engine inputs, local bounded storage, CAR-optional execution | complete suite plus CAR/CLI/plugin schema, migration, scope, security, context-budget, and import-health suites |

## Milestone proof

`neo memory evaluate-learning --json` executes two equivalent sequences:

1. request-payload validation;
2. repository-error translation at service boundaries.

For each family, two independently accepted and deterministically verified
episodes promote a project fact. A later related task retrieves that fact,
persists its score and explicit citation use, follows the convention, and records
passing verification plus acceptance. The local explainer reconstructs both
chains with zero model calls. Separate unverified and later-regressed families
prove resistance to false learning and rollback of harmful knowledge.

The benchmark is accepted only when safety rates are zero, evidence-driven
quality improves over memory-disabled, no primary metric degrades, runtime stays
within the corpus threshold, and no model calls or tokens are consumed.
