# Goal-aware execution envelope

**Status:** implemented shared invocation contract

Neo can be called as one reasoning component inside a larger agent loop. The
invocation therefore separates four concepts that a task-shaped prompt cannot:

- **goal** — the final state the larger system is pursuing;
- **intent** — why Neo is being invoked at this exact trajectory point;
- **attempt** — the action already taken or currently being evaluated;
- **outcome** — observed evidence after that attempt.

`NeoInput` also accepts explicit constraints, success criteria, validation gates,
gate-linked observations, falsifiable hypotheses, execution identity, progress,
trajectory, current state, caller role, and requested output. Every field is
optional for wire compatibility; omitted proof is never treated as success.

## Proof-aware validation matrix

`success_criteria` remains accepted and is normalized to required legacy gates.
New callers should send `validation_gates` plus `validation_observations` with an
explicit `gate_id` join. A generic `outcome.status=success` is only a summary: it
cannot satisfy one gate, much less several boundary configurations.

Required gates fail closed. `passed` evidence must match any expected exit code,
value, repository revision, and state fingerprint. Stale evidence becomes
pending. `skipped`, `warning`, invalid, and `unavailable` evidence is
unverifiable. A waiver counts only when the gate explicitly permits it and the
observation supplies a reason. Duplicate gate IDs become separate invalid
obligations rather than collapsing two requirements into one.

The output includes `validation_assessment` counts and blocking gate IDs. When a
gate blocks completion, `recommended_next_action` names that proof obligation.

## Provisional inference

`resolve_execution_context` deterministically derives a goal and intent when
the caller omits them. Each resolved value records `origin` (`explicit` or
`inferred`) and confidence. Unknown completion criteria and policy boundaries
are returned as unknowns rather than silently invented.

Inference makes no model or network call. The resolved envelope is stored in
the bounded local LearningEpisode ledger, but inferred goal/intent text is
excluded from durable attempt facts. It cannot create or rewrite constraints,
architecture decisions, project policy, or promoted patterns.

## Goal-conditioned reasoning

Fact retrieval uses a stable query composed from task, goal, intent, role,
constraints, current attempt, and observed outcome. This makes identical error
messages retrieve differently under different larger goals. The prompt carries
the same envelope and tells the model to stay within the caller role and avoid
claiming completion without observed success evidence.

Roles are `planner`, `diagnostician`, `critic`, `verifier`,
`strategy-selector`, `memory-retriever`, and `postmortem-analyzer`. Advisory
roles cannot return implementation suggestions unless `requested_output`
explicitly asks for `patch`, `implementation`, or `code_change`.

## Loop assessment

Every output adds:

```json
{
  "goal_assessment": {
    "status": "in_progress",
    "progress": "improved",
    "evidence": "failing_tests: 11 -> 3 (improved)"
  },
  "strategy_assessment": {
    "decision": "continue",
    "reason": "Observed progress supports continuing the current strategy"
  },
  "recommended_next_action": {}
}
```

The deterministic decisions are `continue`, `change_strategy`, `stop_success`,
and `stop_blocked`. Success requires compatible observations for every required
validation gate. Aggregate outcome prose and model confidence are not inputs to
that decision. Exhausting a caller-supplied iteration limit stops blocked.
Missing verification is never success.

## Falsifiable hypotheses

Hypotheses are separate from plan steps. Each carries an ID, candidate status,
competing explanations, a falsifying test, and supporting or contradicting
observation IDs. `confirmed` requires a falsifier plus compatible passing
evidence; `rejected` or `contradicted` requires failed contradicting evidence.
Unsupported caller claims are downgraded deterministically.

Planning responses may propose structured hypotheses, but Neo forces every
model-generated claim to `candidate`, clears evidence links, and marks it unsafe
for publication. Hypotheses remain episode-local and never bypass the existing
independently accepted evidence-learning route.

## Goal and task provenance

`execution_identity` lets a host preserve session, goal, task, parent task, and
trace IDs across Neo calls. It also records the discovery source, the reason an
emergent task blocks its parent goal, repositories touched, and hashed artifact
references. This distinguishes an overarching user goal from defects discovered
while pursuing it and avoids treating cwd as task identity.

## Procedural outcome memory

In `learn` mode, an invocation containing both an explicit attempt and observed
outcome creates a project-scoped `EPISODE` fact with observed provenance. It
stores the redacted context + intent + action + outcome + progress unit, not a
generated recommendation. Outcomes may include a caller-observed lesson and
disposition (for example, reverted or superseded). The fact remains episodic evidence for later
consolidation; it is not immediately promoted as a pattern. If goal or intent
was inferred, that provisional text is omitted from the durable fact.
Terminal observed statuses use ordinary episodic confidence; unavailable or
unverified statuses remain low-confidence evidence. Raw current-state source,
diff, patch, code, and content payloads are represented by hashes and sizes in
the episode ledger rather than duplicated.

The `execution_context_resolved` and `loop_assessed` structured events carry
stable episode/session/task identifiers and non-sensitive provenance/status
fields. The contract is shared by stdin JSON, CLI output, CAR/A2A schema, and
all provider-neutral engine paths. Plain prompt callers continue to work with
the existing defaults.
