# Transcript & journal ingestion — feeding synthesis from AI-tool artifacts

**Status:** proposal v3 (2026-06-13). Phase 0 spike complete (results below).
v1 → v2 corrected integration claims after @neo + @linus review; Phase 0 then
**retired the clustering approach** and v3 pivots the Design to direct ingestion
of verified lessons as PATTERN/FAILURE facts + cosine semantic dedup. This is the
architecture being implemented.

## Problem

neo's synthesis engine is starved, not broken. On a real project with 92 valid
REVIEW facts, `synthesize_reviews` produces **0 patterns** every cycle: the
reviews are too diverse to form the ≥3-member clusters at cosine ≥
`SYNTHESIS_SIMILARITY` (0.85) that synthesis requires (max pairwise similarity
0.90; only 5 pairs ≥ 0.85). Tuning cadence or lowering the threshold
manufactures junk. The engine has nothing dense to chew on.

Root cause is the **ingestion model**, which assumes git is the record of work.
It no longer is. neo's five sources are seeds, community feed, CLAUDE.md
constraints, curated Claude memory `*.md`, and git history (last 50 commits — the
crystallized result, reasoning stripped), plus REVIEWs from neo's own
suggestion→git-diff outcomes (only when neo itself is invoked).

Meanwhile ~847 MB of Claude Code transcripts across 191 projects (111 MB for one
repo) and 492 CAR journals sit ignored — every failed attempt, every user
correction, every recurring friction pattern, the reasoning behind each commit.
That signal recurs across sessions where polished git commits don't, so it is the
material synthesis exists to cluster. This is consistent with the project's own
*Beyond Conversation* and *Memgine* work: the behavioral stream, not the final
artifact, is the substrate for learning.

## Goal & constraint

Turn the AI-tool artifact firehose into high-quality, clusterable REVIEW/FAILURE
facts so synthesis fires on real, recurring signal.

**Quality is the constraint, not cost.** The design centers LM extraction and an
LM verification gate and does not minimize inference. But "quality" must be a
*measured number we regress against*, not a gate we trust — see Phase 0.

---

## Phase 0 — validate the clustering premise FIRST (gating, ~half a day)

This is existential and must run before any `TranscriptIngester` is written.
The whole proposal rests on one unproven claim: *transcript-extracted lessons
recur, therefore they cluster (≥3 members, complete-linkage, cosine ≥ 0.85).*
There is a paradox to resolve, raised in review:

> If lessons are similar enough to cluster at 0.85, they are near-duplicates that
> dedup should collapse (→ clusters of ~1 → synthesis still fires 0). If they are
> diverse enough to survive dedup, they don't cluster (→ synthesis still fires 0).

Phase 0 resolves this empirically against the 4 real transcript files already on
this machine (`~/.claude/projects/-Users-mliotta-git-neo/*.jsonl`, ~112 MB):

1. Hand-label correction / error-recovery / lesson spans in those 4 files →
   ground-truth set.
2. Run a throwaway Stage A + Stage B over them.
3. Run the **actual** pre-write dedup (`_exact_canonical_match`, which is exact
   canonical-signature, NOT cosine — see Correction D below) over the output.
4. Add a `transcript` synthesis group key and run `synthesize_reviews`'
   complete-linkage clustering (all-pairs ≥ 0.85, strict) over the deduped facts.
5. Plot the pairwise-similarity histogram and count ≥3-member complete-linkage
   clusters **after dedup**.

**Kill criterion:** if no ≥3-member clusters form after dedup, the one success
criterion ("synthesis produces PATTERNs, currently 0") is unmet and the design is
theater — stop here. **Quality contract:** Stage A must hit a high recall target
and Stage B + verify a high precision target against the labeled set, in the
spirit of the project's existing 95.8% decision-accuracy contract. No
`TranscriptIngester` ships until Phase 0 passes.

### Phase 0 RESULTS (2026-06-13) — ran against the 4 real transcript files

Faithful spike: schema-correct Stage A → gpt-5.5 extraction → neo's exact
`_canonical_signature` dedup → Jina Code v2 embeddings → complete-linkage @ 0.85.

- **Stage A:** 119 episodes (108 substantive), anchored on the 119 real human
  messages (of 1,953 `user` records — confirming the 94%-tool-output schema).
- **Stage B:** **269 high-quality, generalizable lessons** from 108 episodes,
  0 extraction errors. Eyeballed sample is genuinely good ("verify behavior, not
  installation", "don't overclaim test coverage", "gate merges on CI") — many are
  real lessons from actual sessions. **Extraction is the strong part.**
- **Dedup:** the spike's first measurement ran only `_exact_canonical_match`
  (exact-signature) → 269 → **269** (no-op on paraphrases). But `add_fact` runs a
  *second* dedup the spike skipped: `_find_supersession_candidate` (cosine > 0.85,
  same kind+scope). Re-measured through the **real full `add_fact` path** admitting
  all 269 as PATTERN/project: **269 → 256 valid** (13 superseded); the
  "filter routine events" cluster collapsed 6 → 4. So cosine dedup **already
  exists** via supersession — it does meaningful (not total) collapse, and the
  residual paraphrase redundancy is left to probation decay. **A parallel
  `DEDUP_SIMILARITY` path is rejected** (it would conflict with the 0.85
  supersession already in the path).
- **Clustering:** only 24 of 36,046 pairs ≥ 0.85 (0.067%). **2 clusters ≥3
  members** — but BOTH are the same paraphrase-duplicate lesson (the 6 "filter
  routine events" copies). Genuinely diverse lessons do **not** cluster.

**Verdict: the kill criterion mechanically passes (2 clusters) but qualitatively
FAILS.** The only clusters are dedup-failure artifacts of one repeated lesson,
not synthesis of diverse evidence. This empirically confirms the paradox: diverse
lessons don't cluster; the things that cluster are near-duplicates dedup should
have collapsed. **The "extract → cluster at 0.85 → PATTERN" mechanism is the wrong
tool.**

**Redirect (what the spike actually proved):**
1. **Drop clustering-into-PATTERNs as the goal.** It provably doesn't fire on
   diverse lessons and only produces redundant artifacts.
2. **The real win is direct ingestion of extracted lessons as retrievable facts.**
   269 high-quality lessons from 4 files is a massive enrichment of a store that
   currently has ~130 facts. neo's retrieval (hybrid dense+BM25, rank_score)
   surfaces them when relevant — no synthesis needed.
3. **Now-critical problems** (re-scoped from the original plan): (a) **semantic
   dedup** — exact-canonical is useless here; need cosine-based collapse so the
   "6 copies" become 1; (b) **verify-at-admission + caps** to keep the firehose
   from flooding; (c) if PATTERN-level compression is still wanted, use **LM
   thematic grouping/summarization**, not embedding-clustering at 0.85.

---

## Design (build only if Phase 0 passes)

### New source #6 — `TranscriptIngester`

**Schema-correct Stage A (structural, no LM, recall-oriented).** v1 modeled
transcripts as "user messages"; empirically ~92% of `user`-role records are
`tool_result` envelopes, not human prompts (11 real prompts of 146 user-role
records in one file). There are ~10 `type` values (`user`, `assistant`,
`attachment`, `system`, `permission-mode`, `file-history-snapshot`, `ai-title`,
`last-prompt`, `queue-operation`, `pr-link`, plus typeless `summary`/`leafUuid`
records), multiple interleaved `sessionId`s, and `isSidechain` threads. Stage A
must therefore:

- filter to `type ∈ {user, assistant}`;
- partition by `sessionId`; order by file position (verified: human anchors are
  monotonic in file order, only intra-episode tool/assistant records jitter
  sub-second, so `parentUuid` threading and timestamp sorting buy nothing here —
  documented invariant in the module);
- skip `isSidechain` records (the real meta field; `isMeta` does not occur);
- distinguish *human* text from tool output by content-block structure
  (`content[].type == 'text'`, not `tool_result`) AND reject synthetic CLI/control
  strings (leading `<…>` envelopes, interrupt marker) — the latter contaminated
  ~31% of opened episodes until filtered;
- open an episode on each genuine human message and fold the following assistant
  text, tool uses, and tool errors into it.

**Stage B (LM extraction, quality-first).** Per episode/span, the configured
model (provider stays `openai`/gpt-5.5 — honoring the no-Anthropic-for-our-install
directive) extracts already-generalized lessons
`{kind: PATTERN|FAILURE, subject, body, domain, confidence, evidence_span}`. The
prompt demands transfer-to-future-task lessons, an evidence citation, and a
self-rated confidence. Phase 0 confirmed this yields high-quality output (269
lessons / 4 files, 0 errors).

**Stage C — verify at admission, never at promotion (corrected from v1).** v1
stored raw extractions immediately and gated only later promotion — which floods
the project scope and evicts *real* facts to hold unverified noise. Fix: a single
adversarial LM verify pass runs **before `add_fact`**; rejected extractions are
dropped and never written. Two hard admission filters: (1) the `evidence_span`
must be present AND found **verbatim** in the source transcript (provable
evidence, not claimed); (2) the verifier must rate the lesson generalizable.
Mirrors the existing rule "facts without evidence are dropped, not stored at low
confidence." (A multi-judge panel is a future quality upgrade; v1 ships one strong
skeptical judge and measures precision by sampled inspection.)

### Admission as PATTERN/FAILURE — NO clustering (the Phase 0 pivot)

The original "extract → cluster REVIEWs at 0.85 → PATTERN" path is **dropped** —
Phase 0 proved diverse lessons don't cluster and the only clusters are
paraphrase-dup artifacts. Instead:

- Admit each verified lesson **directly as a `PATTERN`** (or `FAILURE` for
  anti-patterns), already generalized by the LM. These are terminal, retrievable
  facts — neo's hybrid retrieval (dense+BM25, `rank_score`) surfaces them on
  relevance. No synthesis step is involved.
- **Deliberately NOT kind `REVIEW`.** REVIEWs feed `synthesize_reviews`; admitting
  269 transcript REVIEWs would inflate the REVIEW pool and regenerate exactly the
  dup-cluster artifacts Phase 0 found. Keeping transcript lessons as PATTERN/
  FAILURE keeps them out of that path entirely.
- Tag `transcript-derived`, scope `project`, enter **probationary** so unhelpful
  lessons decay out via the existing lifecycle. PATTERN/FAILURE do not bypass
  decay (only CONSTRAINT/ARCHITECTURE/DECISION + seed/community/synthesized tags
  do), so probation/demotion/purge apply normally.

This removes all the net-new synthesis-path work v1 required (group keys, PATTERN
promotion rule, net-new FAILURE *synthesis*) — there is no clustering to wire.
`FAILURE` is just a `FactKind` we now *emit at admission*, not synthesize.

### Identity & where it runs (corrected from v1)

v1 called `_resolve_memory_dir().parent` "shared project resolution." It is not:
that resolver keys on a **path** hash (`codebase_root.replace('/','-')`), while
neo's fact `project_id` is `SHA256` of the normalized git remote (so clones/
worktrees share an ID). These are different identity schemes — fine for this
repo, but on a worktree/clone the path differs and source #6 would read the wrong
(or no) transcript dir while writing facts under the remote-keyed project.
Decision required and stated explicitly: resolve the transcript dir from the same
`codebase_root` the observer already holds, accept that transcripts are located
by path while facts are scoped by remote, and document the worktree behavior.

Also decide explicitly: source #6 is **observer-only**, not run inside
`FactStore.initialize()` (where sources #1–#5 live) — otherwise every neo
invocation pays the transcript-scan cost. The observer owns this.

### Observer loop — measure before splitting (revised per review)

Concern: `_cycle`'s synthesis already runs in a blocking `run_in_executor` to keep
the WS loop draining; adding LM extract+verify could lengthen the tick and move LM
calls (timeouts/rate limits) into the supervised path that does zero today. But
*pre-building* a second supervised lifecycle for an unmeasured cost is
over-engineering. Revised plan: run ingest **inside the existing `_cycle`, behind
the existing executor, with a hard per-cycle episode budget** (e.g. N episodes per
pass), and emit the tick duration. Split into a separate cycle ONLY if the tick is
measured to stall WS responsiveness. Start simple; measure; split if proven.

### Watermark — atomic-after-write, per-episode (corrected from v1)

The real risk is not file rewrite (transcripts are append-only) but the watermark
advancing before facts are durably written across `_cycle`'s swallowed
exceptions — and transcripts watermarked by uuid are **not** replayable like git.
Fix: advance the watermark only after an episode's facts are durably written, and
key it on `(sessionId, last_uuid)`. The data supports this cleanly: **100% of
`user`/`assistant` records carry `uuid` + `sessionId` + `timestamp`** (measured
over 5,235 records), and episodes are built *only* from those records — so every
episode has a well-defined anchor. The typeless records that lack a uuid
(`ai-title`, `last-prompt`, …) are exactly the types Stage A filters out, so they
never anchor an episode and cause no data loss. This is a **net-new** watermark
store (per-session, by last-consumed message uuid), not a reuse of the git or
synthesis watermark.

## Quality controls

- **Measured contract (Phase 0)** — recall on Stage A, precision on Stage B +
  verify, regressed against the labeled set. The non-negotiable line of defense.
- **Adversarial verify at admission** — drop on reject, never store unverified.
- **Evidence requirement** — every fact cites its source span or is dropped.
- **Dedup: reuse the existing supersession, do NOT fork it.** `add_fact` already
  runs `_find_supersession_candidate` (cosine > 0.85, same kind+scope) and was
  measured to collapse the 269 real lessons → 256 valid. No parallel
  `DEDUP_SIMILARITY` path. If we later want *bump-instead-of-supersede* semantics
  (keep one, bump access, avoid tombstones), extend the **one** existing
  `_supersede`/`add_fact` path — not a second cosine function with a second
  threshold.
- **Verbatim evidence check** (added per review) — at admission, reject any lesson
  whose `evidence_span` is not found **verbatim** in the source transcript. This
  converts "the LM says it has evidence" into "evidence provably exists" — the
  cheapest large precision gain against hallucination.
- **Probation + lifecycle + caps** — admitted facts ride the existing
  probation → demotion → purge path. Project cap is **`SCOPE_LIMITS[project]=500`**
  (200 is the *global* cap). Risk: fresh transcript PATTERNs have `success_count=0`
  / mid confidence, so they sit at the bottom of `_enforce_scope_limit`'s eviction
  order and could evict other unproven facts. The end-to-end task MUST project
  facts-per-project and confirm transcript lessons don't evict real REVIEWs; if a
  busy project approaches 500, add a transcript-derived sub-cap.

## Observability

Emit `transcript_ingest` events (`spans_found`, `facts_extracted`,
`facts_after_verify`, `facts_after_dedup`). Success = the store is enriched with
high-quality, deduped, retrievable PATTERN/FAILURE lessons that get surfaced and
earn ACCEPTED outcomes — not raw fact count. (Synthesis-clustering is explicitly
no longer the success metric; Phase 0 retired it.)

## Risks

- **Confident-hallucinated lessons** — dominant risk; mitigated by verify-at-
  admission + evidence requirement + probation decay. Precision measured against a
  sampled inspection in the end-to-end task.
- **Semantic-dedup mis-tuning** — too-low threshold merges distinct lessons;
  too-high lets paraphrase-dups through. Tunable, validated on real extractions.
- **Stage A schema drift** — Claude Code transcript format evolves; pin to a
  parser tested against real files and fail closed on unknown shapes.
- **Identity/scope leakage** — current-project-only; worktree behavior documented.

## Phased rollout

- **Phase 0:** clustering-premise spike (gating). ✅ DONE — retired the clustering
  approach, validated extraction, redirected to direct-ingest (see results above).
- **Phase 1 (this build):** this-repo transcripts; schema-correct Stage A; LM
  extract; verify-at-admission; **direct admission as PATTERN/FAILURE** (no
  clustering); cosine semantic dedup; per-message watermark; separate observer
  ingest cycle; metrics; end-to-end validation.
- **Phase 2:** CAR journal ingestion.
- **Phase 3:** org-scoped cross-project, explicit opt-in.

## Honest reuse vs net-new map

Genuinely reused: fact model/kinds (`memory.models`), probation/lifecycle/caps
(`memory.store`), embeddings + `batched_cosine`, retrieval (`rank_score`,
hybrid dense+BM25), observer cadence/supervision (CAR), LM adapter, redaction
(`lm_logger`).

**Net-new (do not call these reuse):** schema-correct transcript parser + episode
builder; LM extraction + adversarial verify-at-admission stage; cosine
semantic-dedup-on-admission (existing dedup is exact-canonical, a no-op here);
per-message watermark store (atomic-after-write); separate observer ingest cycle;
transcript-dir resolution from `codebase_root` (path-hash) decoupled from the
remote-hash fact `project_id`. NB: the synthesis-clustering path is **not** touched
— the pivot removed all net-new synthesis work.
