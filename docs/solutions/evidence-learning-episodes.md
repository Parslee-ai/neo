# Evidence Learning Episodes

**Status:** implemented evidence ledger and attributed promotion pipeline

Neo persists a versioned `LearningEpisode` for every engine request under:

```text
~/.neo/episodes/<project_id>/<episode_id>.json
```

These records are deliberately separate from `~/.neo/facts/`:

- episodes are task-level evidence;
- facts are generalized knowledge;
- creating an episode does not promote or increase the authority of a fact.

## Version 2 contents

Each record has stable episode, session, and task identifiers plus:

- objective and repository revision/dirty state;
- selected repository and project-instruction paths with content hashes;
- retrieved fact IDs, retrieval scores, final-context inclusion, detectable
  `[fact:<id>]` use, and attributed outcomes;
- reasoning mode and provider/model metadata;
- suggestion IDs, descriptions, paths, and diff/code hashes;
- normalized deterministic-check results;
- pending or terminal outcome state;
- fact mutations performed by the current legacy persistence path.

Raw repository context, generated code, and diffs are not duplicated into the
episode store. SHA-256 digests provide correlation while keeping the evidence
ledger from becoming a second sensitive-source corpus.

## Persistence contract

- Each record is written through a same-directory temporary file and atomic
  replace.
- Storage is bounded to the newest 500 records per project.
- Version 1 and partial pre-v1 records load with conservative defaults: missing verification
  remains absent, never passed.
- Legacy `static_check` and `downstream_outcome` evidence kinds migrate in
  memory to the normalized v2 vocabulary without rewriting source records.
- Malformed or unsupported future records are renamed with a `.corrupt-*`
  suffix and skipped instead of breaking Neo's request path.
- CAR is not required; the ledger is created by `NeoEngine` for every provider.

## Candidate promotion contract

Generated suggestions are stored as episode-local memory candidates, not as
facts. Candidate state remains inspectable and separate from durable memory:

- one attributed ACCEPTED outcome marks an ordinary pattern `supported_once`;
- two independently accepted episodes with the same canonical pattern may
  promote one project-scoped fact with both episode IDs as provenance;
- an earlier deterministic verification failure marks the candidate
  `rejected_by_verification` and blocks promotion even if the change was later
  accepted;
- MODIFIED records contradictory evidence;
- a correction or later regression demotes only the fact promoted from that
  candidate; two distinct source episodes contradicting the same fact roll it
  back with `invalidation_reason=repeated_attributed_contradiction`;
- UNVERIFIED records absence of evidence and never changes confidence,
  effectiveness, or success counts;
- architecture and decision candidates are never automatically promoted by
  the ordinary-pattern rule.
- global promotion requires at least four supporting episodes across at least
  two distinct project IDs; only deterministically generalized, path-stripped,
  secret-redacted text enters the global scope, with every source episode ID
  retained.

This is intentionally a narrow conflict policy. It represents candidate-to-fact
contradictions and regression-driven rollback without an LLM call, but does not
yet infer semantic conflicts between independently worded durable facts.

## Conservative retrieval credit

Relevant knowledge is rendered with stable `[fact:<id>]` citations. A retrieved
fact is marked `used_in_reasoning=true` only when that exact citation survives
into a plan rationale, simulation step, or suggestion description. Session and
outcome records persist `retrieved_fact_ids` and `used_fact_ids` separately.

An accepted downstream result increments success/effectiveness only for the
explicitly used subset. Modification or regression applies a bounded confidence
demotion and negative effectiveness update to that subset. Merely retrieved
facts receive neither credit nor blame; their episode association is labeled
`retrieved_only:<outcome>`. A deterministic failed check marks cited facts with
the `failure` association but does not itself promote or demote durable memory.

## Local explanation

`neo memory explain <fact-id-or-prefix> [--json]` deterministically joins a fact
or retained tombstone with its local episode evidence. The report includes:

- current confidence, success, effectiveness, validity, and provenance;
- supporting and contradicting episode verification;
- each recorded retrieval score, final-context inclusion, detectable reasoning
  use, and downstream association;
- memory mutations with before/after learning-state snapshots;
- rollback reasons and both directions of the supersession chain.

The command constructs `FactStore` with `eager_init=False`, does not initialize
embeddings, skips observer autostart, and performs zero model calls. Older
mutation records remain readable with empty state snapshots, and missing or
malformed episode records remain visible as missing evidence references.
