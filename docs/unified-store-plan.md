# Unified Store Plan — one walker, one index, one retrieval front door

**Date:** 2026-08-11
**Owner:** Keenan Keeling (dispatching via goalpool; Kai shepherds)
**Status:** **Complete — all ten goals landed 2026-08-13.** See [Completion ledger](#completion-ledger-2026-08-13).
**Origin:** 2026-08-11 forensic review of Neo file selection (issues #158/#159/#176/#186/#193–#199) + the 2026-08-10 Keenan/Matt sync.

## Intent

Unify Neo's file selection so there is exactly one answer to "which files does Neo see": land the open fix queue, then migrate both lanes onto a single maintained, self-freshening store — one walker, one content index, one retrieval front door — so that every Neo invocation, on any language repo, hands the model the right whole files in seconds instead of scanning the world on every call; files you name are guaranteed present; retrieval quality is eval-gated at every stage; and every stage ships as its own working release — no flag-day, Neo never breaks.

## Standing rulings (Keenan, 2026-08-11)

1. **`--include` semantics:** guarantee the named files (whole, or with an explicit marker) AND keep scanning for anything else useful. Never a silent drop.
2. **The store indexes, disk delivers:** the store holds index artifacts only (freshness stamps, keyword postings, vectors) — never file bodies. Delivery always reads whole files fresh from disk.
3. **Two query modes survive** (exact keyword + semantic). Unification targets the duplicated walking/exclusion/selection plumbing, not the search styles.
4. **No flag-day:** every goal leaves Neo releasable.
5. **Merge authority:** goals open PRs; auto-merge on green gates is authorized (Keenan, 2026-08-11). Jennifer is holding off on further Neo work while this plan executes.

## Definition of done — metrics

North stars (re-measure after every goal; published numbers, not vibes):

| # | Metric | Instrument | Baseline | Done |
|---|--------|-----------|----------|------|
| M1 | Retrieval quality: MRR + recall@10 on git-history eval sets | `tools/rank_eval.py` (#192) | neo repo 0.30 pre-#194; m365dotnet/aieweb unmeasured (Goal 1) | m365dotnet MRR ≥ 0.6; no flagship repo regresses goal-over-goal |
| M2 | Warm-call cost on m365dotnet: wall-clock + peak RSS, invocation → context assembled | `time` + `ru_maxrss` on `--dry-run` | unmeasured (Goal 1); projected ~27s/~1GB post-#194 | ≤ 5s / ≤ 500MB warm |
| M3 | Freshness cost: re-index work after touching k files | timed touch-and-rerun | n/a (full rebuild every call) | 10-file touch ≤ 5s; cold build bounded and reported |

Guard invariants (every goal's PR must show these hold; a break stops the climb):

- **G1-inv** Zero selected files that `git check-ignore` excludes; zero duplicate copies. (2026-08-10 reality: 14/16 ignored, 12/16 duplicates.)
- **G2-inv** 100% of prompt-named and `--include` files present whole or explicitly marked.
- **G3-inv** No silent caps — every truncation reports, and reported counts match reality.
- **G4-inv** Exactly one eligibility implementation; differential vs `git check-ignore`: 0 over-exclusion on the 33-repo/7,534-path corpus.
- **G5-inv** Per-language LLM round-trip tests (C#, TS, Python) green in the release gate.

## Goals

Already done inline (2026-08-11): #192 (eval harness), #186 (gitignore implementation), #191 (dependabot) merged. Deploy-safety proven: neo main has no deploy trigger; publish is release-gated.

| Goal | Scope | DoD | Depends on |
|------|-------|-----|-----------|
| **1. Trailhead** | Run/extend the #192 harness on m365dotnet, aieweb, neo at current HEAD; measure M2 actuals on m365dotnet | `docs/eval-baselines-2026-08.md` committed with MRR/recall@10 ×3 repos + M2 numbers + exact commands; no product code change | — (dispatched) |
| **2. Land the queue** | Rebase #194 onto main (post-#186), re-run its eval evidence, merge; rebase+merge #187; rebase+merge #190; cut release v0.45.0 | All three merged, evidence re-measured post-rebase and pasted; v0.45.0 published | 1 |
| **3. Selection truthfulness** | Fix #197 (chunk-count reporting) + #198 (`--include` per ruling 1, incl. skip-nothing scan behavior) with tests | Issues closed via merged PRs; G2-inv/G3-inv battery green | 2 |
| **4. Trust calibration** | Fix #196 (language-aware or degraded-gracefully constraint markers) + #199 (no-suggestions sentinel vs confidence) | Issues closed via merged PRs; a declining-but-correct run no longer outscored by an empty patch | 2 |
| **5. One walker** | Extract single eligibility module (gitignore-honoring, worktree-aware, deduping) consumed by BOTH gatherer and index builder; delete remaining duplicate lists | G4-inv structural + differential proof in CI; M1 unchanged or better | 2 |
| **6. Persistent content index** | Move #194's per-call BM25 into the on-disk store beside the semantic catalog, reusing its freshness machinery; closes #195 | M2 ≤ 5s/500MB warm on m365dotnet; M1 unchanged or better | 5 |
| **7. Auto-freshness** | Inline incremental update on every invocation (changed-files check); cold first build bounded + reported; `--index` becomes optional cache-warmer | M3 met; removing `--index` from a fresh clone workflow changes nothing but first-call latency | 6 |
| **8. One front door** | Single retrieval pipeline: named paths (pinned) → `--include` (ruling 1) → keyword → semantic re-rank; per-query budgets; whole-file delivery | M1 target met on all three flagships; G1-inv..G3-inv green | 3, 6 |
| **9. Lane retirement** | Collapse the `cli.py` gather fork; `--semantic` becomes a hint; delete dead lane code; cut release | One gather path in code; all M + G-inv green; release published | 7, 8 |
| **10. Release gate** | Per-language LLM round-trip integration tests (C#, TS, Python) + G-invariant battery wired into the release process | A release cannot publish with a red language round-trip; battery runs in CI | 1 (parallel track) |

## Worker notes (goalpool)

- neo is a **child repo** of parslee-knowledge — route goals to **Claude workers** (Forge fails the clean-worktree gate on child repos).
- Baseline/eval goals also touch m365dotnet and aieweb read-only; no writes outside neo.
- Every PR pastes before/after M1 numbers for the three flagship repos and the invariant battery output. A goal that can't show its numbers isn't done.
- Shepherd (Kai) watches each goal to terminal state; goalpool "done" claims are verified against the DoD, not trusted.

## Review & merge governance (added 2026-08-12, Keenan's option-3 ruling)

- Parslee-ai/neo is enrolled in the car-pr-review daemon (qa_enabled=false, delegation_enabled=true): every ready PR gets an adversarial review; an approve verdict auto-merges.
- Goals open their PRs as **DRAFT** and sustain on them. The shepherd verifies the metric evidence (M1/M2 tables, invariant battery) against this plan, then flips the PR ready — `gh pr ready` is the merge decision. The daemon's review is the independent gate after it.
- Docs/tools-only PRs may open ready (fast path). Runtime-touching goals (G2, G5–G8) MUST use the draft flow.
- In-flight exception: Goal 2 predates this ruling and opens its PR ready; its goal-side metric stop-rule plus daemon review cover it, and the shepherd converts it to draft if it appears before evidence is verified.

---

## Completion ledger (2026-08-13)

**Written at Goal 9, the plan's own closing goal.** 2026-08-11 → 2026-08-13: ten
goals, thirteen merged PRs plus one governance PR, six issues closed (#195, #196,
#197, #198, #199, #210). This section is the climb's ledger: what shipped, what the
numbers did, and what is deliberately left open.

### Per-goal PRs

| Goal | PR(s) | What landed |
|---|---|---|
| **1. Trailhead** | [#201](https://github.com/Parslee-ai/neo/pull/201) | `docs/eval-baselines-2026-08.md` — M1 + M2 baselines on neo / aieweb / m365dotnet, the numbers every later goal is measured against |
| **2. Land the queue** | [#194](https://github.com/Parslee-ai/neo/pull/194), [#187](https://github.com/Parslee-ai/neo/pull/187), [#190](https://github.com/Parslee-ai/neo/pull/190) → re-landed [#204](https://github.com/Parslee-ai/neo/pull/204), release [#205](https://github.com/Parslee-ai/neo/pull/205) (v0.45.0) | Content BM25 selection; `--dry-run` honesty; ruby/php edge queries |
| **3. Selection truthfulness** | [#207](https://github.com/Parslee-ai/neo/pull/207) | closes #197 (count says what it counts) and #198 (`--include` guarantees the file) |
| **4. Trust calibration** | [#206](https://github.com/Parslee-ai/neo/pull/206) | closes #196 (language-aware constraint markers) and #199 (confidence scale without a sentinel) |
| **5. One walker** | [#208](https://github.com/Parslee-ai/neo/pull/208) | `neo/eligibility.py` — one walk for gatherer, index and arch scan; G4-inv structural + differential proof |
| **6. Persistent content index** | [#209](https://github.com/Parslee-ai/neo/pull/209) | closes #195 — BM25 moves into the on-disk store |
| **7. Auto-freshness** | [#212](https://github.com/Parslee-ai/neo/pull/212) | closes #210 — the eligibility walk persists; `--index` becomes optional |
| **8. One front door** | [#214](https://github.com/Parslee-ai/neo/pull/214) | one pipeline, four stages; the `cli.py` gather fork and the second gather function deleted |
| **9. Lane retirement** | *this PR* | the two-lane vocabulary retired from code and docs; `tests/test_lane_retirement.py` guards it; release v0.46.0 |
| **10. Release gate** | [#202](https://github.com/Parslee-ai/neo/pull/202) | G5-inv — per-language LLM round trip gates the release, invariant battery gates every PR |

Governance PR [#203](https://github.com/Parslee-ai/neo/pull/203) added the draft-flow
ruling mid-climb. Goals 1, 3–8 and 10 used it.

### Final metrics vs the trailhead

Measured once at the final base (`80a8cfc`, `origin/main` post-#214 plus Goal 9's
commit). Full method, conditions and per-prompt tables:
[`docs/goal9-lane-retirement-measurements-2026-08-13.md`](goal9-lane-retirement-measurements-2026-08-13.md).

**M1 — retrieval quality.** The trailhead instrument (`5bbee46:tools/rank_eval.py`)
was retired by #194, which re-ran it on both arms before retiring it; that re-run is
the like-for-like trailhead comparison, on the same 209 / 221 / 173 mined cases:

| repo | MRR trailhead | MRR final | R@10 trailhead | R@10 final |
|---|---|---|---|---|
| neo | 0.136 | **0.714** | 0.212 | **0.751** |
| aieweb | 0.180 | **0.759** | 0.244 | **0.774** |
| **m365dotnet** (the gate) | **0.051** | **0.738** | 0.097 | **0.883** |

**M1's target — m365dotnet MRR ≥ 0.60 — is met: 0.051 → 0.738.** The trailhead doc
called it "a 11.8× gap, not a tuning gap"; it closed at 14.5×.

On `tools/rank_mine_eval.py`, the goal-over-goal instrument that replaced it (50
mined cases, git recency off, 0 failed), the final base reads **0.712 / 0.728 /
0.669** MRR and **0.708 / 0.778 / 0.771** R@10 — byte-identical to Goal 8's branch
arm in every cell. No flagship regressed at any goal.

**M2 — warm-call cost on m365dotnet**, canonical 6-prompt battery:

| | trailhead | final | target | verdict |
|---|---|---|---|---|
| battery median wall | 10.54 s | **8.41 s** | ≤ 5 s | **not met including the memory system; met excluding it** (3.04 s of a 6.47 s profiled call) |
| battery peak `ru_maxrss` | 1.43 GB | **1.46 GB** | ≤ 500 MB | **not met** — 1.39 GB of it is the FactStore (#211) |
| eligibility walk, warm | 4.6–6.9 s | **0.12 s** | — | the walk itself is 38–58× cheaper |
| file selection, warm, total | — | **0.37 s** | — | walk + content refresh + content scores + catalog boost |

**M3 — freshness cost** was met at Goal 7 and is unchanged here: 0.84 s for 10 files
edited and 0.75 s for 10 files added, against a ≤ 5 s target, with the cold build
bounded and reported.

**Guard invariants**, all green at the final base: G1-inv 0 gitignored and 0
duplicates of 125 selected files; G2-inv and G3-inv pinned by the invariant battery;
G4-inv one eligibility implementation, structural + differential; G5-inv per-language
round trip green in the release gate.

### The honest reading

Two of the three north stars closed. **M1 closed decisively** — the thing the plan
existed to fix. **M2 closed on the half the plan owns**: file selection is 0.37 s of
a warm call, the walk went from ~5 s to 0.12 s, and the unified store's own cost is
under the 5 s target with room. It did **not** close on the absolute number, because
the FactStore history boost and the engine's fact retrieval are 3.43 s and 1.39 GB
of every invocation and this plan never touched the memory system. That is issue
#211, scoped out on purpose at the trailhead and still open.

The plan's most durable output may not be a metric. Goals 5, 8 and 9 each ended with
a **structural test that fails if the property regresses** —
`test_eligibility_single_source.py`, `test_front_door.py`,
`test_lane_retirement.py`. Every defect this plan fixed had been invisible from the
outside and had exited 0.

### Open exits

| # | What | Why it is out of scope |
|---|---|---|
| [#211](https://github.com/Parslee-ai/neo/issues/211) | FactStore history boost loads 1.26 GB / ~3.4 s into every invocation | Named as out of scope in the plan from Goal 1. It is the whole remaining gap to M2's absolute targets, and it is a memory-system problem, not a retrieval one. **The natural successor plan.** |
| [#213](https://github.com/Parslee-ai/neo/issues/213) | `neo --index` builds a catalog of 100% test files on any `src/` + `tests/` repo | Found at Goal 8 while measuring stage 4. It makes the embedding catalog useless on a normal layout, which is why stage 4 is inert on all three flagships and why the concept-win the plan hoped for could not be measured. |
| *(no issue yet)* | **Pins can consume the entire byte budget.** A prompt naming a 442 KB file spends 299,959 of `--max-bytes`'s 300,000 default on the pin and leaves 41 bytes for the scan, delivering 1 file where pre-#214 main delivered 22. | Reads against standing ruling 1 — *guarantee the named files **and** keep scanning*. Introduced by #214, found by a fresh verifier in that goal's own committed evidence after merge. A fix (`PIN_BUDGET_SHARE`, capping the pin block at half the budget while the scan still has candidates) exists on a branch with mutation-verified tests; the Goal 8 author is opening a follow-up PR against post-#214 main. **Not a Goal 9 regression** — Goal 8's branch arm and this base produce identical file sets. |

### Rulings, kept

All five standing rulings survived the climb intact. Ruling 1 (`--include`
guarantees, and the scan keeps running) is the one that took the most work — #207
implemented it, #214's stage ordering made it structural, and the pin-budget exit
above is the one place it is still not fully honoured. Ruling 3 (*two query modes
survive*) is worth re-reading against the result: both modes do survive, but not as
lanes. Keyword and semantic are now stages 3 and 4 of one pipeline, and `--semantic`
is a weight-and-depth hint rather than a mode selector. The ruling said unification
targets the plumbing, not the search styles, and that is exactly what happened.
