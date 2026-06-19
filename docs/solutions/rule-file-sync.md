# Rule-file sync diagnostic — `neo memory rules`

**Status:** v1 (2026-06-19). Static cross-file analysis of `AGENTS.md` /
`CLAUDE.md` / `GEMINI.md`. Read-only (flag + propose; never writes files).

## Why

A repo worked by more than one coding agent has more than one rule file —
`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`. Teams update one and forget the others,
so the agents drift apart: one is told "use pytest", another never hears it;
one says "tabs", another "spaces". Nothing in any single tool notices, because
each reads only its own file. This is the complement to `neo memory issues`:
issues *suggests* rules from observed friction; this keeps the rules you've
already written consistent across every agent's config — the paper's "treat
AGENTS.md/CLAUDE.md as versioned code" applied across tools.

## What it is

```
neo memory rules [--json] [--no-conflicts] [--cwd PATH]
```

A pull command (`neo/memory/rulesync.py`). Pipeline:

1. **Discover** rule files at the repo root (case-insensitive `AGENTS.md` /
   `CLAUDE.md` / `GEMINI.md`). If <2 exist, or all are byte-identical (e.g. a
   symlinked single source), report "in sync" and stop.
2. **Parse** each into rule *units* — bullet items with wrapped/sub-detail
   continuation lines folded in; headings, code fences, and top-level prose
   dropped (these files encode rules as bullets; treating prose as rules is
   noise — learned from running it on this repo, where naive line-splitting
   produced 12 fragment "gaps" vs. 1 real one after folding).
3. **Embed** each unit (Jina via `store._embed_text` — same vectors as the rest
   of memory, no new infra).
4. **Gaps**: a unit in one file whose best cosine against another file is below
   `ALIGN_THRESHOLD` (0.78) → "present in X, missing from Y", deduped, with a
   proposed `Add to <files>: <rule>` edit.
5. **Conflicts** (LM-judged, opt-out via `--no-conflicts`): a pair that aligns
   (≥0.78) but isn't near-identical (<`IDENTICAL_THRESHOLD` 0.97) is a candidate
   contradiction; a bounded, graceful LM judge (`resolve_adapter` + `_parse_json`,
   same pattern as `issues --suggest-rules`, capped at `_MAX_CONFLICT_CHECKS`)
   decides `{"conflict": bool, "explanation": str}`.

## Read-only / flag + propose

Never writes files. Gaps yield a proposed `Add to AGENTS.md: …` line; conflicts
yield a `Reconcile: X says …; Y says …` line. The developer applies the edit —
consistent with neo's advisory role. (An opt-in `--write` was considered and
deliberately deferred.)

## Output

```
[Neo] memory rules — files: AGENTS.md (32 rules), CLAUDE.md (33 rules)

  Gaps (1):
    • present in claude; missing from agents
      <the rule text>
      → Add to AGENTS.md: <the rule text>
```

`--json` emits `{files, in_sync, note, gaps[], conflicts[]}`.

## Deferred

- `--write` to apply proposed additions (with backup + diff).
- Nested rule files (sub-directory `AGENTS.md`), and other tools' conventions.
- Inspecting tools' *memory* artifacts (distinct from static rule files).
