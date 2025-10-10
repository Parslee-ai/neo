# Contributing to Neo

Thank you for your interest in contributing to Neo!

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Parslee-ai/neo.git
cd neo
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate  # Windows
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs Neo in editable mode with all development tools (pytest, black, ruff, mypy).

### 4. Configure API Keys

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_neo.py

# Run with coverage
pytest --cov=neo
```

### Code Formatting

```bash
# Format code with black
black src/ tests/
```

### Linting

```bash
# Lint with ruff
ruff check src/ tests/
```

### Type Checking

```bash
# Run mypy
mypy src/
```

## Project Structure

```
neo/
├── src/
│   └── neo/            # Main package
│       ├── cli.py              # CLI entry point
│       ├── persistent_reasoning.py  # Memory system
│       ├── adapters.py         # LM provider adapters
│       ├── config.py           # Configuration management
│       ├── context_gatherer.py # Context collection
│       ├── storage.py          # Local file storage
│       └── ...                 # Other modules
├── tests/
│   ├── test_neo.py             # Core tests
│   ├── test_integration.py     # Integration tests
│   └── ...                     # Other test files
├── .claude-plugin/             # Claude Code plugin
└── pyproject.toml              # Package configuration
```

## Contributing Guidelines

### Code Style

- **Formatting**: Use `black` with 100 character line length
- **Linting**: Follow `ruff` recommendations
- **Naming**: Use descriptive variable names
- **Comments**: Add docstrings for all public functions
- **Type hints**: Use type hints where helpful (but not required everywhere)

### Adding Features

1. **New LM Provider**
   - Add adapter in `adapters.py`
   - Inherit from `LMAdapter`
   - Implement `generate()` and `name()`
   - Update `create_adapter()` factory
   - Add to documentation

2. **New Static Analysis Tool**
   - Add function in `static_analysis.py`
   - Follow existing pattern (temp file, run tool, parse output)
   - Add tool detection logic
   - Update configuration options

3. **New Parser**
   - Add to `parsers.py`
   - Provide multiple fallback strategies
   - Handle edge cases gracefully
   - Test with various LM outputs

### Testing

- Add tests in `tests/` directory
- Test with multiple LM providers if applicable
- Include edge cases and error conditions
- Keep tests fast and focused
- Run full test suite before submitting PR

### Documentation

- Update `README.md` for user-facing changes
- Update `INSTALL.md` for setup changes
- Update `.claude-plugin/README.md` for plugin changes
- Add docstrings to new functions

## Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes**
4. **Format and lint**: `make format && make lint`
5. **Test**: `make test`
6. **Commit**: Use clear, descriptive commit messages
7. **Push**: `git push origin feature/your-feature`
8. **Create Pull Request**: Describe your changes clearly

### Commit Message Format

```
<type>: <short summary>

<optional longer description>

<optional footer>
```

Types:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/config changes

Examples:
```
feat: add support for Google Gemini models

fix: handle empty context_files in parser

docs: update INSTALL.md with Docker instructions
```

## Areas for Contribution

### High Priority

- [ ] Improve parsing robustness with more LM output formats
- [ ] Add more comprehensive tests
- [ ] Optimize exemplar search performance
- [ ] Better error messages and handling
- [ ] Performance benchmarking

### Medium Priority

- [ ] Support for more LM providers (e.g., Cohere, Together AI)
- [ ] Additional static analysis tools (e.g., pylint, clippy for Rust)
- [ ] Better diff application logic
- [ ] Streaming output support
- [ ] Caching for repeated queries

### Nice to Have

- [ ] Web UI for visualization
- [ ] VS Code extension
- [ ] GitHub Actions integration
- [ ] Metrics and observability
- [ ] Multi-language support for prompts

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Accept criticism gracefully
- Prioritize community benefit

### Unacceptable Behavior

- Harassment or discrimination
- Trolling or insulting comments
- Publishing private information
- Unprofessional conduct

## Questions?

- **General questions**: Open a GitHub Discussion
- **Bug reports**: Open a GitHub Issue
- **Feature requests**: Open a GitHub Issue with `[Feature Request]` prefix

## Recognition

Contributors will be acknowledged in:
- README.md
- Release notes
- Project documentation

Thank you for contributing to Neo!