# Memory-file audit — `neo memory audit`

**Status:** v1 (2026-06-19). Read-only hygiene inspection of an AI tool's
*accumulated* memory files. Phase 1 of cross-tool memory support.

## Why

Tools accumulate learned memory on disk, separate from their human-authored
rule files. Claude Code writes per-project `~/.claude/projects/<proj>/memory/`:
a `MEMORY.md` index plus individual fact files with YAML frontmatter (`name`,
`description`, `metadata.type ∈ {user, feedback, project, reference}`) and a
prose body. Over time this store accretes duplicates, contradictions, malformed
entries, and index drift — and nothing inspects it. `neo memory rules` compares
*config*; this audits *memory*.

## What it checks

`neo memory audit [--json] [--no-conflicts] [--cwd PATH]`
(`neo/memory/memaudit.py`). Read-only — never edits memory.

1. **malformed** — missing/unparseable frontmatter, missing `description`, or a
   `type` outside the valid set.
2. **near-duplicate** — bodies whose embeddings cluster at cosine ≥
   `DUP_THRESHOLD` (0.93), via the shared `math_utils.cluster_by_similarity`.
3. **conflict** — pairs that align (≥ `ALIGN_THRESHOLD` 0.80) but aren't
   duplicates, judged contradictory by a bounded, graceful LM judge
   (`resolve_adapter` + `_parse_json`; opt out with `--no-conflicts`).
4. **index** — a memory file absent from `MEMORY.md`, or `MEMORY.md` pointing at
   a file that doesn't exist.

**Dangling `[[links]]` are NOT flagged**: the memory spec says a link to a
not-yet-written memory is intentional ("marks something worth writing later").

## Discovery

`resolve_memory_dir` reuses `transcript.resolve_transcript_dir` (path→`~/.claude
/projects/<encoded>`) plus `/memory`. So the audit targets the same project's
Claude Code memory the transcript miner already knows how to locate.

## Roadmap

- **Phase 2 (deferred): ingest as a source.** A `MemorySource` adapter that
  imports peer-tool memory into neo's own store **on probation** with an
  `imported:<tool>` provenance, under existing hygiene (dedup, supersession,
  caps). Makes neo the cross-tool memory hub. Gated behind the trust concern
  that importing another tool's memory can import wrong/stale facts.
- Other tools' memory formats (Cursor, Copilot) as they stabilize.
