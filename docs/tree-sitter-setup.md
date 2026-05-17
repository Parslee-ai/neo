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

# Incrementally refresh after edits (re-embeds only changed files)
neo --update

# Index from a specific directory
neo --index --cwd /path/to/project
```

The resulting index lives at `.neo/index.json` inside the target repo and powers Neo's [Smart File Selection](../README.md#smart-file-selection).

### Supported Languages

The canonical map lives in `src/neo/languages.py:30-54`.

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
5. **Indexing**: FAISS index enables fast cosine search; per-file chunk cap of 2 prevents large files from eating the budget.

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

## Operational Notes

- Use `neo --update` instead of `--index` after the first build — it re-embeds only changed files.
- `neo --index --max-files N` caps the walk when you only care about the active subtree.
- Neo honors `.gitignore` by default; double-check large generated dirs aren't tracked.
- `MAX_CHUNK_LENGTH` is set to **2000 characters** (defined in both `language_parser.py` and `project_index.py` — keep them in sync if you change one).
- Malformed code returns empty chunks by design (better than half-parsed garbage propagating into embeddings).

## References

- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- tree-sitter-language-pack: https://github.com/Goldziher/tree-sitter-language-pack
- Neo issues: https://github.com/Parslee-ai/neo/issues
