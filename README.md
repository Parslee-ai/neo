# Neo

[![PyPI version](https://badge.fury.io/py/neo-reasoner.svg)](https://badge.fury.io/py/neo-reasoner)
[![Python Versions](https://img.shields.io/pypi/pyversions/neo-reasoner.svg)](https://pypi.org/project/neo-reasoner/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A self-improving code reasoning engine that learns from experience using persistent semantic memory. Neo uses multi-agent reasoning to analyze code, generate solutions, and continuously improve through feedback loops.

## Design Philosophy

**Persistent Learning**: Neo builds a semantic memory of successful and failed solutions, using vector embeddings to retrieve relevant patterns for new problems.

**Code-First Output**: Instead of generating diffs that need parsing, Neo outputs executable code blocks directly, eliminating extraction failures.

**Local File Storage**: Semantic memory stored in ~/.neo directory for privacy and offline access.

**Model-Agnostic**: Works with OpenAI, Anthropic, Google, local models, or Ollama via a simple adapter interface.

## How It Works

```
User Problem → Neo CLI → Semantic Retrieval → Reasoning → Code Generation
                           ↓
                    [Vector Search]
                    [Pattern Matching]
                    [Confidence Scoring]
                           ↓
                    Executable Code + Memory Update
```

Neo retrieves similar past solutions using Jina Code embeddings (768-dimensional vectors), applies learned patterns, generates solutions, and stores feedback for continuous improvement.

## Quick Start

```bash
# Install from PyPI (recommended)
pip install neo-reasoner

# Or install with specific LM provider
pip install neo-reasoner[openai]     # For GPT (recommended)
pip install neo-reasoner[anthropic]  # For Claude
pip install neo-reasoner[google]     # For Gemini
pip install neo-reasoner[all]        # All providers

# Set API key
export OPENAI_API_KEY=sk-...

# Test Neo
neo --version
```

**See [QUICKSTART.md](QUICKSTART.md) for 5-minute setup guide**

## Claude Code Plugin

Neo is available as a **Claude Code plugin** with specialized agents and slash commands for seamless integration:

```bash
# Install Neo as a Claude Code plugin
/plugin marketplace add Parslee-ai/neo
```

Once installed, you get:
- **Neo Agent**: Specialized subagent for semantic reasoning (`Use the Neo agent to...`)
- **Slash Commands**: `/neo`, `/neo-review`, `/neo-optimize`, `/neo-architect`, `/neo-debug`, `/neo-pattern`
- **Persistent Memory**: Neo learns from your codebase patterns over time
- **Multi-Agent Reasoning**: Solver, Critic, and Verifier agents collaborate on solutions

### Quick Examples

```bash
# Code review with semantic analysis
/neo-review src/api/handlers.py

# Get optimization suggestions
/neo-optimize process_large_dataset function

# Architectural guidance
/neo-architect Should I use microservices or monolith?

# Debug complex issues
/neo-debug Race condition in task processor
```

**See [.claude-plugin/README.md](.claude-plugin/README.md) for full plugin documentation**

## Installation

### From PyPI (Recommended)

```bash
# Install Neo
pip install neo-reasoner

# With specific LM provider
pip install neo-reasoner[openai]     # GPT (recommended)
pip install neo-reasoner[anthropic]  # Claude
pip install neo-reasoner[google]     # Gemini
pip install neo-reasoner[all]        # All providers

# Verify installation
neo --version
```

### From Source (Development)

```bash
# Clone repository
git clone https://github.com/Parslee-ai/neo.git
cd neo

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Verify installation
neo --version
```

### Dependencies

Core dependencies are automatically installed via `pyproject.toml`:
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- datasketch >= 1.6.0
- faiss-cpu >= 1.7.0

### Optional: LM Provider

Choose your language model provider:

```bash
pip install openai                  # GPT models (recommended)
pip install anthropic               # Claude
pip install google-generativeai     # Gemini
pip install requests                # Ollama
```

**See [INSTALL.md](INSTALL.md) for detailed installation instructions**

## Usage

### CLI Interface

```bash
# Ask Neo a question
neo "how do I fix the authentication bug?"

# With working directory context
neo --cwd /path/to/project "optimize this function"

# Check version and memory stats
neo --version
```

### Timeout Requirements

Neo makes blocking LLM API calls that typically take 30-120 seconds. When calling Neo from scripts or automation, use appropriate timeouts:

```bash
# From shell (10 minute timeout)
timeout 600 neo "your query"

# From Python subprocess
subprocess.run(["neo", query], timeout=600)
```

Insufficient timeouts will cause failures during LLM inference, not context gathering.

### Output Format

Neo outputs executable code blocks with confidence scores:

```python
def solution():
    # Neo's generated code
    pass
```

### Personality System

Neo responds with personality (Matrix-inspired quotes) when displaying version info:

```bash
$ neo --version
"What is real? How do you define 'real'?"

120 patterns. 0.3 confidence.
```

## Architecture

### Semantic Memory

Neo uses **Jina Code v2** embeddings (768 dimensions) optimized for code similarity:

1. **Pattern Storage**: Every solution attempt creates a reasoning pattern
2. **Vector Search**: Similar problems retrieve relevant patterns via FAISS
3. **Confidence Scoring**: Patterns track success/failure rates
4. **Local Persistence**: Patterns stored locally in JSON format

### Code Block Schema (Phase 1)

Neo generates executable code directly instead of diffs:

```python
@dataclass
class CodeSuggestion:
    file_path: str
    unified_diff: str           # Legacy: backward compatibility
    code_block: str = ""        # Primary: executable Python code
    description: str
    confidence: float
    tradeoffs: list[str]
```

This eliminates the 18% extraction failure rate from diff parsing.

### Storage Architecture

- **Local Files**: JSON storage in ~/.neo directory
- **FAISS Index**: Fast vector search for pattern retrieval
- **Auto-Consolidation**: Intelligent pattern merging to prevent fragmentation

## Performance

**Neo improves over time as it learns from experience.** Initial performance depends on available memory patterns. Performance grows as the semantic memory builds up successful and failed solution patterns.

## Configuration

### CLI Configuration Management

Neo provides a simple CLI for managing persistent configuration:

```bash
# List all configuration values
neo --config list

# Get a specific value
neo --config get --config-key provider

# Set a value
neo --config set --config-key provider --config-value anthropic
neo --config set --config-key model --config-value claude-3-5-sonnet-20241022
neo --config set --config-key api_key --config-value sk-ant-...

# Reset to defaults
neo --config reset
```

**Exposed Configuration Fields:**
- `provider` - LM provider (openai, anthropic, google, azure, ollama, local)
- `model` - Model name (e.g., gpt-4, claude-3-5-sonnet-20241022)
- `api_key` - API key for the chosen provider
- `base_url` - Base URL for local/Ollama endpoints

Configuration is stored in `~/.neo/config.json` and takes precedence over environment variables.

### Environment Variables

Alternatively, use environment variables for configuration:

```bash
# Required: LM Provider API Key
export ANTHROPIC_API_KEY=sk-ant-...
```

## LM Adapters

### OpenAI (Recommended)

```python
from neo.adapters import OpenAIAdapter
adapter = OpenAIAdapter(model="gpt-5-codex", api_key="sk-...")
```

Latest models: `gpt-5-codex` (recommended for coding), `gpt-5`, `gpt-5-mini`, `gpt-5-nano`

### Anthropic

```python
from neo.adapters import AnthropicAdapter
adapter = AnthropicAdapter(model="claude-sonnet-4-5-20250929")
```

Latest models: `claude-sonnet-4-5-20250929`, `claude-opus-4-1-20250805`, `claude-3-5-haiku-20241022`

### Google

```python
from neo.adapters import GoogleAdapter
adapter = GoogleAdapter(model="gemini-2.5-pro")
```

Latest models: `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`

### Ollama

```python
from neo.adapters import OllamaAdapter
adapter = OllamaAdapter(model="llama3.1")
```

## Extending Neo

### Add a New LM Provider

```python
from neo.cli import LMAdapter

class CustomAdapter(LMAdapter):
    def generate(self, messages, stop=None, max_tokens=4096, temperature=0.7):
        # Your implementation
        return response_text

    def name(self):
        return "custom/model-name"
```

## Key Features

- **Persistent Memory**: Learns from every solution attempt
- **Semantic Retrieval**: Vector search finds relevant patterns
- **Code-First Generation**: No diff parsing failures
- **Local Storage**: Privacy-first JSON storage in ~/.neo directory
- **Benchmarking**: LiveCodeBench integration for measurable progress
- **Model-Agnostic**: Works with any LM provider

## Architecture Decisions

See [docs/architecture-decisions.md](docs/architecture-decisions.md) for design rationale:
- Why code blocks instead of diffs
- Semantic embeddings vs keyword matching
- Local-first storage strategy
- Learning loop design

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_neo.py

# Run with coverage
pytest --cov=neo
```

## Research & References

### Academic Papers

Neo's design is informed by cutting-edge research in code reasoning and memory systems:

1. **ReasoningBank** (arXiv:2509.25140v1)
   *Systematic Failure Learning and Semantic Anchor Embedding*
   - Phase 2: Semantic anchor embedding (pattern+context only)
   - Phase 3: Failure root cause extraction and contrastive learning
   - Phase 4: Self-contrast consolidation (archetypal vs spurious patterns)
   - Phase 5: Strategy evolution tracking (procedural/adaptive/compositional)
   - Paper: https://arxiv.org/abs/2509.25140

2. **MapCoder** - Multi-agent reasoning framework
   Neo uses Solver-Critic-Verifier agent collaboration for code generation

3. **CodeSim** - Code similarity metrics
   Influenced Neo's semantic memory design and pattern matching approach

### Technologies

- **Jina Code v2 Embeddings** ([jinaai/jina-embeddings-v2-base-code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code))
  768-dimensional embeddings optimized for code similarity tasks

- **FAISS** ([facebookresearch/faiss](https://github.com/facebookresearch/faiss))
  Facebook AI Similarity Search - efficient vector similarity search and clustering

- **LiveCodeBench** - Competitive programming benchmark for measuring code generation quality

- **FastEmbed** ([qdrant/fastembed](https://github.com/qdrant/fastembed))
  Local embedding generation without external API dependencies

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
