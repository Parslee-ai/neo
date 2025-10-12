![Neo Banner: Imagery from The Matrix film series](https://ik.imagekit.io/xvpgfijuw/parslee/bannerFor__Neo--Github.webp)
***


# Neo

> A self-improving code reasoning engine that learns from experience using persistent semantic memory. Neo uses multi-agent reasoning to analyze code, generate solutions, and continuously improve through feedback loops.

- **Persistent Memory**: Learns from every solution attempt
- **Semantic Retrieval**: Vector search finds relevant patterns
- **Code-First Generation**: No diff parsing failures
- **Local Storage**: Privacy-first JSON storage in ~/.neo directory
- **Model-Agnostic**: Works with any LM provider
- **Available as a [Claude Code Plugin](#claude-code-plugin)**: Integrates seamlessly with Anthropic's Claude models and CLI.

![Claude Code Plugin Banner: Background is an illustration of a terminal or console.](https://ik.imagekit.io/xvpgfijuw/parslee/bannerFor__Claude-Code.webp)

[![PyPI version](https://img.shields.io/pypi/v/neo-reasoner.svg)](https://pypi.org/project/neo-reasoner/)
[![Python Versions](https://img.shields.io/pypi/pyversions/neo-reasoner.svg)](https://pypi.org/project/neo-reasoner/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Why Neo?  Why Care?  
If you've been Vibe Coding, then Vibe Planning, then Context Engineering, and on and on, you have likely hit walls where the models are both powerful and limited, brilliant and incompetent, wise and ignorant, humble yet overconfident. 

Worse, your speedy AI Code Assistant sometimes goes rogue and overwrites key code in a project, or writes redundant code even after just reading documentation and the source code, or violates your project's patterns and design philosophy....  _It can be infuriating._  Why doesn't the model remember?  Why doesn't it learn?  Why can't it keep the context of the code patterns and tech stack? ... -> This is what Neo is designed to solve.  

Neo is **_the missing context layer_** for AI Code Assistants.  It learns from every solution attempt, using vector embeddings to retrieve relevant patterns for new problems.  It then applies the learned patterns to generate solutions, and continuously improves through feedback loops.


# Table of Contents

- [Design Philosophy](#design-philosophy)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Claude Code Plugin](#claude-code-plugin)
  - [Quick Examples](#quick-examples)
- [Installation](#installation)
  - [From PyPI (Recommended)](#from-pypi-recommended)
  - [From Source (Development)](#from-source-development)
  - [Dependencies](#dependencies)
  - [Optional: LM Provider](#optional-lm-provider)
- [Usage](#usage)
  - [CLI Interface](#cli-interface)
  - [Timeout Requirements](#timeout-requirements)
  - [Output Format](#output-format)
  - [Personality System](#personality-system)
- [Architecture](#architecture)
  - [Semantic Memory](#semantic-memory)
  - [Code Block Schema (Phase 1)](#code-block-schema-phase-1)
  - [Storage Architecture](#storage-architecture)
- [Performance](#performance)
- [Configuration](#configuration)
  - [CLI Configuration Management](#cli-configuration-management)
  - [Environment Variables](#environment-variables)
- [LM Adapters](#lm-adapters)
  - [OpenAI (Recommended)](#openai-recommended)
  - [Anthropic](#anthropic)
  - [Google](#google)
  - [Ollama](#ollama)
- [Extending Neo](#extending-neo)
  - [Add a New LM Provider](#add-a-new-lm-provider)
- [Key Features](#key-features)
- [Development](#development)
  - [Running Tests](#running-tests)
- [Research & References](#research--references)
  - [Academic Papers](#academic-papers)
  - [Technologies](#technologies)
- [License](#license)
- [Contributing](#contributing)
- [Changelog](#changelog)


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

Neo retrieves similar past solutions using Jina Code embeddings (768-dimensional vectors),
applies learned patterns, generates solutions, and stores feedback for continuous improvement.

1. Jina's embeddings model (open source) is downloaded automatically when you first run Neo.
    This model runs locally on your machine to generate vector embeddings.

2. When you ask Neo for help:
    - Your query is embedded locally using the Jina model
    - Neo searches local memory for similar past solutions (using FAISS)
    - Retrieved patterns are combined with your original prompt
    - This combined context is sent to your chosen LLM API (OpenAI/Anthropic/Google)
    - The LLM generates a solution informed by both your query and past patterns
    - The result is stored back in local memory for future use

Local storage:
  ~/.neo/reasoning_patterns.json  ← Stores vectors + patterns
  ~/.neo/faiss_index.bin         ← FAISS index for fast search

Privacy:
  - Your code never leaves your machine during embedding/search
  - Only your prompt + retrieved patterns are sent to the LLM API
  - This is the same as using the LLM directly, but with added context from something akin to memory.
 
 ```
   Your Prompt
      ↓
  Local Jina Embedding (768-dim vector)
      ↓
  Local FAISS Search (finds similar past solutions)
      ↓
  Retrieve Pattern Text from ~/.neo/reasoning_patterns.json
      ↓
  Combine: Your Prompt + Retrieved Pattern Text
      ↓
  →→→ NETWORK CALL →→→ LLM API (OpenAI/Anthropic/etc.)
      ↓
  Solution Generated
      ↓
  Store in Local Memory for future use
 ```

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
# Add the marketplace
/plugin marketplace add Parslee-ai/claude-code-plugins

# Install Neo plugin
/plugin install neo
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
- fastembed >= 0.3.0
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

Neo responds with personality _(Matrix-inspired quotes)_ when displaying version info:

```bash
$ neo --version
"What is real? How do you define 'real'?"

120 patterns. 0.3 confidence.
```

### Load Program - Training Neo's Memory

**"The Operator uploads a program into Neo's head."**

Neo can bootstrap its memory by importing patterns from HuggingFace datasets. This is NOT model fine-tuning - it's retrieval learning that expands local semantic memory with reusable code patterns.

```bash
# Install datasets library
pip install datasets

# Load patterns from MBPP (recommended starter - 1000 Python problems)
neo --load-program mbpp --split train --limit 1000

# Load from OpenAI HumanEval (164 hand-written coding problems)
neo --load-program openai_humaneval --split test

# Load from BigCode HumanEvalPack (multi-language variants)
neo --load-program bigcode/humanevalpack --split test --limit 500

# Dry run to preview
neo --load-program mbpp --dry-run

# Custom column mapping
neo --load-program my_dataset \
    --columns '{"text":"pattern","code":"solution"}'
```

**Output (Matrix-style):**
```
"I know kung fu."

Loaded: 847 patterns
Deduped: 153 duplicates
Index rebuilt: 1.2s
Memory: 1247 total patterns
```

**How it works:**
1. **Acquire**: Pull dataset from HuggingFace
2. **Normalize**: Map rows to ReasoningEntry schema
3. **Dedupe**: Hash-based deduplication against existing memory
4. **Embed**: Generate local embeddings (Jina Code v2)
5. **Index**: Upsert into FAISS index
6. **Report**: Matrix quote + counts

**Key points:**
- NOT fine-tuning - just expanding retrieval memory
- Patterns start at 0.3 confidence (trainable via real-world usage)
- Automatic deduplication prevents memory bloat
- Uses local embeddings (no data leaves your machine)
- Stored in `~/.neo/` alongside learned patterns

**See [docs/LOAD_PROGRAM.md](docs/LOAD_PROGRAM.md) for detailed documentation**


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
- **Model-Agnostic**: Works with any LM provider

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

- **FastEmbed** ([qdrant/fastembed](https://github.com/qdrant/fastembed))
  Local embedding generation without external API dependencies

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

