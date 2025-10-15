# Project: Neo - Semantic Reasoning Helper

## Quick Context
- **Purpose**: Read-only reasoning helper for CLI tools using MapCoder/CodeSim-style multi-agent reasoning with semantic memory
- **Tech Stack**: Python 3.9+, fastembed (Jina Code v2), FAISS, Anthropic/OpenAI/Google LMs
- **Installation**: `pip install -e ".[dev]"` for development

## Code Style
- Import convention: stdlib → third-party → local, specific imports
- Naming: PascalCase classes, snake_case functions, UPPER_SNAKE constants, _private methods
- Error handling: Try/except with specific exceptions, logger warnings, graceful fallbacks
- Testing: test_*.py pattern, pytest framework
- Type hints: Extensive with Optional, list[], dict[]
- Docstrings: Triple quotes, brief description first

## Project Rules
- Keep implementations simple first, enhance iteratively
- Test all changes before committing
- Use 3-5 minute timeout when executing `neo` commands
- Semantic memory: Local embeddings (Jina 768-dim) preferred over OpenAI (1536-dim)
- Memory stays under 2000 entries via auto-consolidation (triggers every 10 entries after 30+)
- Local storage uses JSON files in ~/.neo directory (efficient for <5000 entries)
- When creating a pull request, always use the PR template included in the repo.