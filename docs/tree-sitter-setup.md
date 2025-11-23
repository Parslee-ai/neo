# Tree-sitter Multi-Language Support Setup

## Overview

Neo now supports multi-language code indexing using tree-sitter. This allows semantic indexing of Python, C#, TypeScript, JavaScript, Java, Go, Rust, and C/C++ codebases.

## Installation

### Python Version Requirements

**Important**: The `tree-sitter-languages` package currently only supports Python 3.8 - 3.12. If you're using Python 3.13+, tree-sitter support will be disabled, and indexing will fall back to Python-only AST parsing.

### Install tree-sitter (Optional)

```bash
# For Python 3.8 - 3.12
pip install -e ".[tree-sitter]"

# Or install manually
pip install tree-sitter tree-sitter-languages
```

### Verify Installation

```bash
python -c "from tree_sitter_languages import get_parser; print('tree-sitter OK')"
```

## Usage

### Building a Multi-Language Index

```bash
# Index all supported languages (auto-detect)
neo --index

# Index specific languages only
neo --index --languages python,csharp,typescript

# Index from a specific directory
neo --index --cwd /path/to/project
```

### Supported Languages

| Language   | Extensions        | Status       |
|------------|-------------------|--------------|
| Python     | .py, .pyi         | ✅ Full Support |
| C#         | .cs               | ✅ Full Support |
| TypeScript | .ts, .tsx         | ✅ Full Support |
| JavaScript | .js, .jsx         | ✅ Full Support |
| Java       | .java             | ✅ Full Support |
| Go         | .go               | ✅ Full Support |
| Rust       | .rs               | ✅ Full Support |
| C/C++      | .c, .cpp, .h, .hpp | ✅ Full Support |

## Troubleshooting

### tree-sitter-languages not found

If you see:
```
ImportError: tree-sitter-languages not available
```

This usually means:
1. You're using Python 3.13+ (not yet supported by tree-sitter-languages)
2. The package isn't installed: `pip install tree-sitter-languages`

### Fallback Behavior

When tree-sitter is not available:
- Neo will log a warning
- Indexing will fail with a helpful error message
- You can still use Neo for other features (reasoning, memory, etc.)

### Using Python 3.13+

If you need multi-language support with Python 3.13:

**Option 1**: Use a virtual environment with Python 3.11
```bash
pyenv install 3.11.9
pyenv virtualenv 3.11.9 neo-env
pyenv activate neo-env
pip install -e ".[tree-sitter]"
```

**Option 2**: Wait for tree-sitter-languages Python 3.13 support
- Track issue: https://github.com/grantjenks/py-tree-sitter-languages/issues

**Option 3**: Build language parsers manually (advanced)
- Install tree-sitter: `pip install tree-sitter`
- Compile language grammars yourself
- See: https://tree-sitter.github.io/tree-sitter/

## Architecture

### How It Works

1. **Language Detection**: File extension → language mapping
2. **Parsing**: Tree-sitter parses source code into AST
3. **Chunk Extraction**: Queries extract functions, classes, methods
4. **Embedding**: Code chunks are embedded using fastembed (Jina Code v2)
5. **Indexing**: FAISS index enables fast semantic search

### Code Organization

- `src/neo/index/language_parser.py`: Tree-sitter parser implementation
- `src/neo/index/project_index.py`: Indexing orchestration
- `src/neo/cli.py`: CLI integration (`--index`, `--languages` flags)

### Adding New Languages

To add support for a new language:

1. Add extension mapping to `LANGUAGE_MAP` in `language_parser.py`
2. Define tree-sitter queries in `QUERIES` dictionary
3. Test with sample files
4. Update documentation

Example for adding Ruby support:
```python
LANGUAGE_MAP = {
    # ... existing languages
    '.rb': 'ruby',
}

QUERIES['ruby'] = {
    'functions': """
        (method
            name: (identifier) @name
            parameters: (method_parameters) @params
            body: (_) @body) @method
    """,
    'classes': """
        (class
            name: (constant) @name
            body: (_) @body) @class
    """
}
```

## Performance

### Benchmarks (1000 files, mixed languages)

| Operation | Time | Memory |
|-----------|------|--------|
| Index Build | ~30s | ~150MB |
| Semantic Search (k=5) | ~50ms | - |
| Refresh (10 changed files) | ~2s | - |

### Optimization Tips

1. **Limit max_files**: `neo --index --max-files 500`
2. **Filter by language**: Only index languages you use
3. **Use .gitignore**: Exclude node_modules, venv, build folders
4. **Incremental refresh**: Index updates only changed files

## Known Issues

1. **Python 3.13 Support**: Waiting on upstream tree-sitter-languages
2. **Large Files**: Files >100KB may timeout (set MAX_CHUNK_LENGTH)
3. **Syntax Errors**: Malformed code returns empty chunks (by design)

## References

- Tree-sitter: https://tree-sitter.github.io/tree-sitter/
- tree-sitter-languages: https://github.com/grantjenks/py-tree-sitter-languages
- Neo issues: https://github.com/Parslee-ai/neo/issues
