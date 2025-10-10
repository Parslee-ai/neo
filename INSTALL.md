# Installation Guide

## Quick Install

### Recommended: Development Mode

```bash
# Clone repository
git clone https://github.com/Parslee-ai/neo.git
cd neo

# Install in development mode (includes all dependencies)
pip install -e .

# Verify installation
neo --version
```

This automatically installs:
- Core dependencies (numpy, scikit-learn, datasketch, faiss-cpu)
- Neo CLI command (`neo`)

### Alternative: Using pyproject.toml

```bash
# Install with specific extras
pip install -e .[anthropic]  # Include Anthropic SDK
pip install -e .[openai]     # Include OpenAI SDK
pip install -e .[google]     # Include Google SDK
pip install -e .[all]        # All LM providers
pip install -e .[dev]        # Development tools
```

## Requirements

- **Python**: 3.9 or higher
- **pip**: Latest version recommended

### Core Dependencies (Auto-Installed)

When you run `pip install -e .`, these are automatically installed:

```
numpy >= 1.24.0
scikit-learn >= 1.3.0
datasketch >= 1.6.0
faiss-cpu >= 1.7.0
```

### Optional: LM Provider SDKs

Choose your language model provider:

```bash
pip install openai                  # GPT models (recommended)
pip install anthropic               # Claude
pip install google-generativeai     # Gemini
pip install requests                # Ollama (usually already installed)
```

## Configuration

### 1. Set API Key

Neo requires an API key for your chosen LM provider:

```bash
# OpenAI (recommended)
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Google
export GOOGLE_API_KEY=...
```

Add to `~/.bashrc` or `~/.zshrc` for persistence.

## Verification

### Test Installation

```bash
# Check version and memory stats
neo --version

# Should output something like:
# "What is real? How do you define 'real'?"
#
# 120 patterns. 0.3 confidence.
```

### Test Core Functionality

```bash
# Ask Neo a simple question
neo "write a hello world function"

# With working directory context
neo --cwd /path/to/project "explain this code"
```

### Run Test Suite

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_neo.py

# Run with verbose output
pytest -v
```

## Platform-Specific Notes

### macOS

```bash
# Using Homebrew
brew install python@3.11

# Install Neo
cd neo
pip3 install -e .
```

### Linux

```bash
# Debian/Ubuntu
sudo apt-get update
sudo apt-get install python3 python3-pip

# Install Neo
cd neo
pip3 install -e .

# Fedora
sudo dnf install python3 python3-pip
cd neo
pip3 install -e .
```

### Windows

```bash
# Using WSL (recommended)
wsl --install
# Then follow Linux instructions

# Native Windows
cd neo
python -m pip install -e .
```

## Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv neo-env

# Activate
source neo-env/bin/activate  # Linux/macOS
# OR
neo-env\Scripts\activate     # Windows

# Install Neo
cd neo
pip install -e .

# Deactivate when done
deactivate
```

## Troubleshooting

### neo: command not found

The CLI wasn't registered properly:

```bash
# Reinstall in development mode
pip install -e .

# Or add to PATH manually
export PATH="${PATH}:$(pwd)"
```

### API Key Not Found

```bash
# Verify environment variable is set
echo $ANTHROPIC_API_KEY

# Or set it now
export ANTHROPIC_API_KEY=sk-ant-...
```

### sklearn/numpy Issues

```bash
# macOS with M1/M2 chip
pip install --upgrade numpy scikit-learn

# If still failing, try specific versions
pip install numpy==1.24.0 scikit-learn==1.3.0
```

### FAISS Installation Errors

```bash
# Use CPU version (included by default)
pip install faiss-cpu

# For GPU support (optional)
pip install faiss-gpu
```

### Permission Errors

```bash
# Use --user flag
pip install --user -e .

# Or fix ownership
sudo chown -R $USER /path/to/neo
```

## Optional Dependencies

### Development Tools

```bash
# Install development dependencies
pip install -e .[dev]
```

Includes:
- pytest (testing)
- black (formatting)
- ruff (linting)
- mypy (type checking)

### Running Benchmarks

Neo includes comprehensive test coverage for all features:

```bash
# Run full test suite
pytest

# Run specific test suites
pytest tests/test_failure_learning.py
pytest tests/test_strategy_evolution.py
```

## Migration from Old Installation

If you previously installed Neo using requirements.txt:

```bash
# Uninstall old version
pip uninstall neo-reasoner

# Remove old dependencies (optional)
pip freeze | grep -v "^-e" | xargs pip uninstall -y

# Install fresh using new method
pip install -e .
```

## Upgrading

```bash
# Pull latest changes
cd neo
git pull origin main

# Reinstall (picks up new dependencies)
pip install -e . --upgrade

# Verify version
neo --version
```

## Uninstallation

```bash
# Uninstall Neo
pip uninstall neo-reasoner

# Remove environment variables
# Edit ~/.bashrc or ~/.zshrc and remove:
# - ANTHROPIC_API_KEY
# - etc.

# Optional: Remove local memory cache
rm -rf ~/.neo
```

## Docker Installation (Advanced)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -e .

CMD ["neo", "--version"]
```

```bash
# Build image
docker build -t neo .

# Run Neo
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY neo "your prompt"
```

## Next Steps

After installation:

1. **Test**: Run `neo --version` to verify
2. **Configure**: Set API keys in environment
3. **Read Docs**: Check README.md for usage examples
4. **Run Tests**: Try `pytest tests/test_neo.py`
5. **Explore**: Use `neo "your question"` to test

## Getting Help

- **Installation Issues**: Check this guide's Troubleshooting section
- **API Key Problems**: Verify environment variables with `echo $ANTHROPIC_API_KEY`
- **Dependencies**: Run `pip list | grep neo` to see installed packages
- **GitHub Issues**: Report bugs at https://github.com/Parslee-ai/neo/issues

## Package Structure

After installation, you'll have:

```
neo/
├── src/
│   └── neo/            # Main package
│       ├── cli.py              # CLI entry point
│       ├── persistent_reasoning.py  # Memory system
│       ├── storage.py          # Local file storage
│       ├── adapters.py         # LM provider adapters
│       └── config/             # Configuration (including personality)
├── tests/              # Test suite
├── pyproject.toml      # Package configuration
└── README.md           # Documentation
```

## Verifying Installation

Check that all components are working:

```bash
# 1. CLI command exists
which neo

# 2. Python can import Neo package
python -c "from neo.cli import NeoEngine; print('✓ Neo package OK')"

# 3. FAISS working
python -c "import faiss; print('✓ FAISS OK')"

# 4. Memory system initializes
neo --version
```

All checks should pass without errors.
