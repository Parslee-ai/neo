# Unified Store Plan — one walker, one index, one retrieval front door

**Date:** 2026-08-11
**Owner:** Keenan Keeling (dispatching via goalpool; Kai shepherds)
**Status:** Active — Goal 1 dispatched 2026-08-11
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
