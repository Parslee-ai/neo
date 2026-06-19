# Conversation-mined issue diagnostic — `neo memory issues`

**Status:** v1 (2026-06-19). LM-free detectors over the existing multi-source
transcript episodes. Read-only view; no fact admission, no watermark mutation.

## Why

The memory ingester ([transcript-ingestion](transcript-ingestion.md)) already
parses Claude Code / Codex / CAR transcripts into `Episode`s and silently
admits PATTERN/FAILURE lessons that improve future retrieval. The friction
*signals* it computes (errors, assistant clarification) were never surfaced to
the human. This diagnostic flips the output mode: instead of quietly enriching
memory, it **reports recurring frictions as actionable issues** — the
memory-side instance of the continuous quality flywheel ("diagnose failures by
clustering root causes") and an automation of the manual advice *"add a rule
every time the agent does something it should not do again."*

## What it is

```
neo memory issues [--since 14d] [--min-cluster 3] [--json] [--cwd PATH]
```

A pull command. Pipeline (`neo/memory/issues.py`), all LM-free in v1:

1. **Collect** — episodes from every `TranscriptSource` within the `--since`
   window (reuses `ClaudeCodeSource` / `CodexSource` / `CarSource`). Filters
   `Episode.is_substantive`.
2. **Tag** (`tag_signals`) — per-episode friction derived purely from
   `Episode` fields: `has_tool_error` + normalized `error_class` (from
   `Episode.errors`), and `asst_asked_clarification` (clarification regex
   vocabulary lifted from `prompt.analyzer.CONFUSION_PATTERNS`, but applied to
   episode assistant text rather than the analyzer's Claude-only message dicts).
3. **Cluster** (`detect_issues`) — embed each ask (sharpened with its first
   error line when present) via the store's `_embed_text` (Jina — same vectors
   as fact retrieval, no extra model, no LM) and complete-linkage cluster at
   `CLUSTER_SIMILARITY = 0.85` (== `store.SYNTHESIS_SIMILARITY`), reusing
   `math_utils.cluster_by_similarity`.
4. **Gate** — a cluster becomes an `Issue` only if it has **≥ `min_cluster`
   members**, spans **≥ 2 distinct sessions**, has **≥ 2 frictional members**,
   and yields **≥ 1 verbatim evidence span** (Stage-C discipline: provable, not
   paraphrased). Pure-recurrence-without-friction clusters are dropped.
5. **Categorize + score** — precedence `missing-tool` > `absent-guardrail` >
   `vague-rule` (most structural signal wins; one cluster → one category,
   mapped to the harness failure taxonomy). Confidence =
   `0.4·size + 0.3·session-spread + 0.3·recency` (local exp decay, 14-day
   half-life, on episode epochs). Sorted descending.

## Read-only / no-consume contract

`find_issues` collects episodes within the window but **never constructs the
`TranscriptIngester`** and never reads or writes the ingester watermark
(`SESSIONS_DIR/transcript_watermark_*`). It is fully decoupled from fact
admission, idempotent, and safe to run repeatedly. Guarded by
`test_find_issues_is_read_only_never_invokes_ingester`.

## Design choices

- **Conservative by default.** A wrong issue wastes scarce human attention, so
  the gate is strict (≥3, ≥2 sessions, ≥2 frictional, verbatim evidence).
  Better to under-report in v1. On this repo's own 53 substantive episodes (12
  frictional), v1 reports zero — friction was real but scattered, never
  recurring on one topic across sessions. That is the correct answer, not a bug.
- **Episodes, not the analyzer's message model.** Episodes are the watermarked,
  multi-tool substrate; we lift the analyzer's regex vocabulary but operate on
  Episodes so Codex/CAR are covered for free.
- **Error relevance filter (learned from real data).** `Episode.errors` is
  dominated by signal that is *not* a project issue: Claude Code's own
  `<tool_use_error>` tool-protocol guards (read-before-edit, replace_all,
  file-modified, sleep-blocked) and bare command banners ("Exit code 2"). A
  probe across 8 local projects showed these produced 7 low-value "issues" on
  the busiest repo. `_episode_errors` now drops `<tool_use_error>` envelopes and
  banner-only errors, and `_substantive_error_line` skips a leading banner so a
  Codex `"Process exited with code 1\n<traceback>"` keeps the traceback as its
  class. Post-filter, that same repo surfaced one genuine recurring friction
  ("git merge repeatedly blocked by local changes," 7 sessions) and otherwise
  stayed quiet — the intended behavior.
- **Clustering reuse.** The complete-linkage loop was extracted from
  `store._cluster_by_similarity` into `math_utils.cluster_by_similarity(items,
  embed_fn, threshold)`; both synthesis and this diagnostic share one
  implementation.

## `--suggest-rules` (shipped v0.24.0)

`neo memory issues --suggest-rules` makes one bounded LM call per surviving
issue (highest-confidence first, capped at `_MAX_SUGGESTED_RULES`) to draft a
preventive AGENTS.md rule, populating `Issue.suggested_rule`. The LM-bearing
step is confined to `suggest_rules` in the CLI path — `find_issues` stays
deterministic and LM-free. Per-issue LM failure is graceful (leaves the rule
unset); the adapter is built via `resolve_adapter` only when the flag is set.

## Deferred (post-v1)

- Dedup against existing FAILURE facts (annotate "already in memory").
- Detectors 5–6: artifact drift (CLAUDE.md asserts X, transcripts show not-X)
  and contradiction (CLAUDE.md vs AGENTS.md vs Neo facts) — need an LM judge +
  `scanner.scan_claude_mds`.
