# Tree-sitter Multi-Language Support

## Overview

Neo uses [tree-sitter](https://tree-sitter.github.io/tree-sitter/) for multi-language parsing across the project index, empty-catch detection, god-file metrics, and edge extraction. Tree-sitter is a **required** core dependency — there is no Python-only fallback.

## Installation

Tree-sitter is installed automatically as part of `pip install neo-reasoner` (or `pip install -e .` for development). No extras flag is required:

```bash
pip install neo-reasoner
```

This pulls in:

- `tree-sitter` (core C library bindings)
- `tree-sitter-language-pack` (the maintained successor to the deprecated `tree-sitter-languages` package, with binary wheels for current Python versions)

> **Note on the 1.x boundary**: `tree-sitter-language-pack` 1.x changed `Parser.parse` and `Tree.root_node` incompatibly. Neo pins to the 0.x line until those changes are absorbed — see the `dependencies` block in `pyproject.toml`.

### Verify Installation

```bash
python -c "from tree_sitter_language_pack import get_parser; print('tree-sitter OK')"
```

## Usage

### Building a Multi-Language Index

```bash
# Index all supported languages (auto-detect)
neo --index

# Re-embed catalogued files whose contents changed (must be passed WITH --index)
neo --index --update

# Index from a specific directory
neo --index --cwd /path/to/project
```

The resulting catalog lives at `.neo/index.json` inside the target repo and is
**stage 4** of Neo's [Smart File Selection](../README.md#smart-file-selection) — a
re-rank and supplement over the keyword stage, not a prerequisite for it. The
eligibility walk and the keyword content index are separate artifacts in the same
`.neo/` directory (`walk_cache.json`, `content_index.sqlite3`) that maintain
themselves on every invocation; this command is an **optional cache-warmer** for the
embedding catalog alone. A repo with no catalog still selects files on every stage
below stage 4, and once a catalog exists it is read on every run with no flag.

### Supported Languages

The canonical map is `EXTENSION_TO_LANGUAGE` in `src/neo/languages.py`.

| Language   | Extensions                  |
|------------|-----------------------------|
| Python     | `.py`, `.pyi`               |
| C#         | `.cs`                       |
| TypeScript | `.ts`, `.tsx`               |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs` |
| Java       | `.java`                     |
| Go         | `.go`                       |
| Rust       | `.rs`                       |
| C / C++    | `.c`, `.cpp`, `.cc`, `.cxx`, `.h`, `.hpp`, `.hh` |
| Ruby       | `.rb`                       |
| PHP        | `.php`                      |
| Swift      | `.swift`                    |
| Kotlin     | `.kt`                       |

Per-subsystem coverage isn't uniform. For example, `code_smells` empty-catch detection registers a per-language detector in `_ERROR_SWALLOW_DETECTORS` (Ruby uses a custom `_is_empty_ruby_rescue` predicate because Ruby's grammar doesn't expose `catch_clause`); Go, Rust, and C have no try/catch construct, so they're omitted from that detector entirely. New languages need explicit registration there before empty-catch detection picks them up.

## Architecture

### How It Works

1. **Language Detection**: file extension → canonical tree-sitter language name (`src/neo/languages.py`).
2. **Parsing**: tree-sitter parses source code into an AST via `tree_sitter_language_pack.get_parser`.
3. **Chunk Extraction**: queries extract functions, classes, methods — see `src/neo/index/language_parser.py`.
4. **Embedding**: each chunk's `symbols + imports + first ~600 chars of body` is embedded via fastembed (Jina Code v2, 768 dims). Embedding the *signature surface* rather than the raw body is what defeats the "tests outrank source files" keyword-overlap bias — assertion strings inside tests no longer drown out the file's actual definitions.
5. **Indexing**: FAISS index enables fast cosine search. The only per-file bound here is the share `_cap_chunks` apportions under `MAX_CHUNKS_PER_REPO`; `MAX_CHUNKS_PER_FILE = 2` is a **gatherer** constant (`context_gatherer.py:26`) that bounds prompt windows, and this line claimed it for the index build for a long time.

### Code Organization

- `src/neo/languages.py` — pure-data extension/alias/fence/display maps
- `src/neo/index/language_parser.py` — tree-sitter parser wrappers + chunk extraction queries
- `src/neo/index/project_index.py` — indexing orchestration, FAISS persistence, freshness tracking
- `src/neo/cli.py` — CLI integration (`--index`, `--update`)
- `src/neo/context_gatherer.py` — consumes the index to boost per-file scores during prompt assembly

### Adding a New Language

1. Add the extension(s) to `EXTENSION_TO_LANGUAGE` in `src/neo/languages.py` using the canonical tree-sitter language name (e.g. `c_sharp`, not `csharp`). Add the fence tag to `_FENCE_TAGS` and display name to `_DISPLAY_NAMES`.
2. Confirm `tree-sitter-language-pack` ships the grammar: `python -c "from tree_sitter_language_pack import get_parser; get_parser('YOUR_LANG')"`.
3. Add chunk-extraction queries to `language_parser.py` (functions, classes, methods) and register them in `QUERIES` — `architecture_metrics.py` walks files only when `lang in QUERIES`.
4. If the language has try/catch-style error handling, register a detector in `code_smells._ERROR_SWALLOW_DETECTORS`; otherwise empty-catch detection is silently no-op for it.
5. Add a behavioural assertion, not just a compile check. `test_every_chunk_query_compiles` and `test_every_edge_query_compiles` will catch a query that does not compile, but a query that compiles and matches nothing looks identical to a file with nothing in it. Assert that a fixture containing the construct produces the chunk or edge you expect.

### Why queries break silently

A query that fails to compile is not an error anywhere the operator can see it. `_get_query` returns `None` and `parse_file` moves to the next query; `_extract_edges` logs at DEBUG. The construct is simply absent from the index.

Four queries were in this state at once: `typescript`/`tsx` `interfaces` (grammar renamed the interface body from `object_type` to `interface_body`), `c_sharp` `inheritance` (`bases:` was never a field name), and a `python` `module_doc` that also had no consumer. TypeScript interfaces and C# inheritance edges had been missing from every index since the queries shipped, with no signal but a warning line — one per query *per parsed file*, which is how a single run produced 9,699 of them. Compile results, failures included, are now cached per parser instance, so a broken query warns once.

Grammars move. `test_every_chunk_query_compiles` / `test_every_edge_query_compiles` are parametrized over every entry in `QUERIES` and `EDGE_QUERIES` so the next rename fails a test instead of quietly shrinking the index.

## Operational Notes

- **The catalog is optional and its refresh is manual — unlike everything else in `.neo/`.** The walk cache and the keyword content index update inline on every invocation; the embedding catalog does not. A stale catalog degrades quality, never correctness: stage 4 can only add to or re-order what stage 3 found, never remove from it.
- **`--update` is narrower than it sounds, in three ways.** It is read only *inside* the `--index` branch (`cli.py`), so it must be passed as `neo --index --update` — bare `neo --update` falls through to the reasoning path and refreshes nothing. `check_staleness` builds its candidate list from the snapshot's own `file_hashes`, so it re-embeds **changed** files and never sees **new** ones; a repo that grew needs a full build. And the refresh stops at `REFRESH_BUDGET_MS` (5 s) or `REFRESH_MAX_CHUNKS` (100), after which it still prints the catalog's total chunk count — so a partial refresh reads exactly like a complete one, which is the failure mode the rest of this document exists to stamp out. Tracked as [#217](https://github.com/Parslee-ai/neo/issues/217).
- The index build caps at **100 files** per run by default. Override it with `neo --index --max-files N`; the same flag caps *context gathering* at 30 by default when used without `--index`, so each subsystem keeps its own floor.
- **The cut is apportioned, not sliced.** `ProjectIndex._select_files` groups eligible files by language and gives each a share of `--max-files` proportional to how much of the repo it is, with a floor of one slot per language present (`_allocate_slots`). Before this, patterns were globbed in list order and concatenated, so whichever language globbed first ate the budget — a .NET repo of 4,272 C# files produced an index of 83 Python files and zero C#, and exited 0. Within a language, files are ordered shallowest-path-first; that is a weak heuristic, not centrality ranking.
- **`MAX_CHUNKS_PER_REPO` (1000) is applied the same way.** `_cap_chunks` apportions across files in proportion to what each holds, with a floor. A plain slice re-created the file-cap bug one layer down, since chunks arrive grouped by file and files grouped by language — but the first fix, round-robin (every file keeps a chunk before any file keeps a second), introduced a subtler version of the same bias: an equal share is not a fair share, and a 9 KB utility ended up fully represented while `store.py` contributed 6 of its 82 symbols. See `_cap_chunks`'s own docstring, which carries both measurements.
- **Eligibility lives in ONE place**: `neo/eligibility.py` — the walk, the ignore-pattern list, the gitignore matcher and the byte-identical dedup primitive — which the index, prompt assembly and the architecture scan all call. `tests/test_eligibility_single_source.py` fails if a second definition, a second `os.walk` or a second exclusion list appears anywhere under `src/`, and `tests/test_eligibility_differential.py` diffs the walk against `git check-ignore` on a fixture corpus and on this checkout. `load_ignore_patterns` carries the repo's own `.gitignore` plus defaults that apply when it says nothing. `should_ignore` implements gitignore rather than approximating it — root-anchored patterns (`/path/`), unanchored directory rules matching at any depth (`build/` → `src/build`), negation (`!`, last-match-wins), and `*` that does not cross a separator. Validated against `git check-ignore` over 7,534 on-disk paths in 33 repositories: zero over-exclusion, and the only under-exclusions come from **nested** `.gitignore` files, which are not read — only the repo root's. Two further exclusion classes are policy rather than gitignore and are named separately so a missing file can be attributed to the rule that dropped it: `WalkPolicy` knobs (symlink rejection, the gatherer's 512 KB ceiling, extension and per-language glob filters), and git's own tracked-file override — git applies ignore rules only to files it does not already track, so a file added before a rule was written stays tracked while the walk still skips it (four `specs/*.md` on this checkout). Ambiguous directory names (`bin`, `build`, `out`, `target`, `dist`, `vendor`) are deliberately not hardcoded, because each is real source somewhere; nor are `.claude`/`.codex`/`.car`, whose *worktree* subtrees are excluded by layout (`**/.claude/worktrees`) so committed skill source stays visible.
- Byte-identical files are indexed once. A duplicate does not consume a slot, and the freed slot is refilled — across languages if the original language has nothing unique left.
- **Anything left out is reported.** Each line is conditional — a build that indexed everything prints none of them, so their presence is the signal. `neo --index` adds: how many of the eligible files were indexed with a per-language breakdown (only when the file cap actually bound), a capped chunk total, how many selected files ended up with no chunks at all, and counts for skipped duplicates and excluded paths. "Built index" on its own no longer means the index represents the repository.
- `MAX_CHUNK_LENGTH` is set to **2000 characters** (defined in both `language_parser.py` and `project_index.py` — keep them in sync if you change one).
- Malformed code returns empty chunks by design (better than half-parsed garbage propagating into embeddings).

## References

- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- tree-sitter-language-pack: https://github.com/Goldziher/tree-sitter-language-pack
- Neo issues: https://github.com/Parslee-ai/neo/issues
