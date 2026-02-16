![Neo Banner: Imagery from The Matrix film series](https://ik.imagekit.io/xvpgfijuw/parslee/bannerFor__Neo--Github.webp)
***


# Neo

> A self-improving code reasoning engine that learns from experience using persistent semantic memory. Neo uses multi-agent reasoning to analyze code, generate solutions, and continuously improve through feedback loops.

- **Fact-Based Memory**: Learns from every solution attempt using a scoped, supersession-based fact store
- **Semantic Retrieval**: Vector search finds relevant facts via Jina Code embeddings
- **Code-First Generation**: No diff parsing failures
- **Local Storage**: Privacy-first JSON storage in ~/.neo/facts/ directory
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
  - [Fact-Based Memory](#fact-based-memory)
  - [Output Schemas](#output-schemas)
  - [Storage Architecture](#storage-architecture)
- [Performance](#performance)
- [Configuration](#configuration)
  - [CLI Configuration Management](#cli-configuration-management)
  - [Environment Variables](#environment-variables)
- [LM Adapters](#lm-adapters)
  - [OpenAI (Default)](#openai-default)
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

**Fact-Based Learning**: Neo builds a semantic memory of facts — constraints, architectural decisions, patterns, review learnings, decisions, known unknowns, and failures — using vector embeddings for retrieval.

**Code-First Output**: Instead of generating diffs that need parsing, Neo outputs executable code blocks directly, eliminating extraction failures.

**Scoped Storage**: Facts are scoped to global, organization, or project level, stored locally in ~/.neo/facts/ for privacy and offline access.

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

Neo retrieves relevant facts using Jina Code embeddings (768-dimensional vectors),
applies learned patterns, generates solutions, and stores new facts for continuous improvement.

1. Jina's embeddings model (open source) is downloaded automatically when you first run Neo.
    This model runs locally on your machine to generate vector embeddings.


## The Construct

Neo includes **The Construct** - a curated library of architecture and design patterns with semantic search capabilities. Think of it as your personal reference library for common engineering patterns, indexed and searchable using the same embedding technology that powers Neo's reasoning memory.

### What is The Construct?

The Construct is a collection of vendor-agnostic design patterns covering:
- **Rate Limiting**: Token bucket, sliding window, distributed rate limiting
- **Caching**: Cache-aside, write-through, invalidation strategies
- **More domains**: Additional patterns contributed by the community

Each pattern follows a structured format inspired by the Gang of Four:
- **Intent**: What problem does this solve?
- **Forces**: Key constraints and tradeoffs
- **Solution**: Conceptual structure (no framework-specific code)
- **Consequences**: Benefits, risks, and observability signals
- **References**: Links to real-world implementations

### Using The Construct

```bash
# List all patterns
neo construct list

# Filter by domain
neo construct list --domain rate-limiting

# Show a specific pattern
neo construct show rate-limiting/token-bucket

# Semantic search across patterns
neo construct search "how to prevent api abuse"

# Build the search index
neo construct index
```

### Pattern Quality Standards

All patterns must:
- Include author attribution
- Be under 300 lines
- Remain vendor-agnostic (no AWS/GCP/Azure-specific solutions)
- Include concrete consequences and observability guidance

See `/construct/README.md` for contribution guidelines.

2. When you ask Neo for help:
    - Your query is embedded locally using the Jina model
    - Neo searches the fact store for relevant knowledge (using cosine similarity)
    - Retrieved facts are organized into layers: constraints, relevant knowledge, recent changes, known unknowns
    - This combined context is sent to your chosen LLM API (OpenAI/Anthropic/Google)
    - The LLM generates a solution informed by both your query and past facts
    - The result is stored back as a new fact in local memory for future use

Local storage:
  ~/.neo/facts/facts_global.json       ← Global-scoped facts
  ~/.neo/facts/facts_org_{id}.json     ← Organization-scoped facts
  ~/.neo/facts/facts_project_{id}.json ← Project-scoped facts

Privacy:
  - Your code never leaves your machine during embedding/search
  - Only your prompt + retrieved facts are sent to the LLM API
  - This is the same as using the LLM directly, but with added context from something akin to memory.

 ```
   Your Prompt
      ↓
  Local Jina Embedding (768-dim vector)
      ↓
  Cosine Similarity Search (finds relevant facts)
      ↓
  Retrieve Facts from ~/.neo/facts/
      ↓
  Assemble Context: Constraints → Knowledge → Recent Changes → Known Unknowns
      ↓
  →→→ NETWORK CALL →→→ LLM API (OpenAI/Anthropic/etc.)
      ↓
  Solution Generated
      ↓
  Store as New Fact in Local Memory
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

### Updating Neo

Neo supports both manual and fully automatic updates:

#### Manual Updates

```bash
# Option 1: Use neo's built-in update command (simplest)
neo update

# Option 2: Update with pip
pip install --upgrade neo-reasoner

# Option 3: Use pipx for isolated installation (recommended for end users)
pipx install neo-reasoner          # First-time install
pipx upgrade neo-reasoner           # Update to latest version
pipx upgrade-all                    # Update all pipx packages
```

#### Fully Automatic Updates

Enable automatic installation of updates in the background:

```bash
# Enable auto-install (persisted in ~/.neo/config.json)
neo --config set --config-key auto_install_updates --config-value true

# Or use environment variable
export NEO_AUTO_INSTALL_UPDATES=1
```

When enabled, Neo will:
- Check for updates once every 24 hours
- Automatically download and install new versions in the background
- Notify you when updates complete
- Log all auto-update activity to `~/.neo/auto_update.log`

**Example output when auto-install is enabled:**
```bash
$ neo "your query"

⚡ Auto-installing neo update: 0.9.0 → 0.10.0
   This happens in the background. Please wait...

✓ Auto-update completed: 0.10.0
   Restart neo to use the new version.

[Neo] Processing your query...
```

#### Update Notifications (Default)

By default, Neo checks for updates once every 24 hours and displays a notification when a new version is available. This check happens in the background and will not interrupt your workflow.

To disable update checks entirely:
```bash
export NEO_SKIP_UPDATE_CHECK=1
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
- jsonschema >= 4.0.0


### Optional: LM Provider

Choose your language model provider:

```bash
pip install openai                  # GPT models (recommended)
pip install anthropic               # Claude
pip install google-genai>=0.2.0     # Gemini (requires Python 3.10+)
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

neo 0.9.0
Provider: openai | Model: gpt-5.3-codex
Stage: Sleeper | Memory: 0.0%
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0 facts | 0.00 avg confidence
```

### Load Program - Training Neo's Memory

**"The Operator uploads a program into Neo's head."**

Neo can bootstrap its memory by importing facts from HuggingFace datasets. This is NOT model fine-tuning - it's retrieval learning that expands local semantic memory with reusable code knowledge.

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

Loaded: 847 facts
Deduped: 153 duplicates
Index rebuilt: 1.2s
Memory: 1247 total facts
```

**How it works:**
1. **Acquire**: Pull dataset from HuggingFace
2. **Normalize**: Map rows to fact schema
3. **Dedupe**: Hash-based deduplication against existing memory
4. **Embed**: Generate local embeddings (Jina Code v2)
5. **Store**: Add as facts to the fact store
6. **Report**: Matrix quote + counts

**Key points:**
- NOT fine-tuning - just expanding retrieval memory
- Facts start at 0.3 confidence (trainable via real-world usage)
- Automatic deduplication prevents memory bloat
- Uses local embeddings (no data leaves your machine)
- Stored in `~/.neo/facts/` alongside learned facts

**See [docs/LOAD_PROGRAM.md](docs/LOAD_PROGRAM.md) for detailed documentation**


## Architecture

### Fact-Based Memory

Neo uses a **scoped, supersession-based fact store** with **Jina Code v2** embeddings (768 dimensions) for semantic retrieval:

1. **Fact Storage**: Knowledge is stored as typed facts (constraints, architecture, patterns, review learnings, decisions, known unknowns, failures)
2. **Scoped Organization**: Facts are scoped to global, organization, or project level — automatically detected from git remotes
3. **Supersession**: When a new fact closely matches an existing one (cosine similarity > 0.85), the old fact is superseded rather than duplicated
4. **Confidence Scoring**: Facts carry confidence scores with automatic boosting on supersession
5. **Four-Layer Context**: Retrieved facts are organized into constraints, relevant knowledge, recent changes, and known unknowns (inspired by StateBench)
6. **Local Persistence**: Facts stored locally in JSON format in `~/.neo/facts/`

### Output Schemas

Neo generates structured outputs with executable code and planning artifacts:

**CodeSuggestion** - Executable code with actionable metadata:
```python
@dataclass
class CodeSuggestion:
    # Core fields
    file_path: str
    unified_diff: str           # Legacy: backward compatibility
    code_block: str = ""        # Primary: executable Python code
    description: str
    confidence: float
    tradeoffs: list[str]

    # Executable artifacts (v0.8.0+)
    patch_content: str = ""            # Full unified diff content
    apply_command: str = ""            # Shell command to apply (advisory)
    rollback_command: str = ""         # Shell command to undo (advisory)
    test_command: str = ""             # Shell command to verify (advisory)
    dependencies: list[str] = []       # Other suggestion IDs this depends on
    estimated_risk: str = ""           # "low", "medium", or "high"
    blast_radius: float = 0.0          # 0.0-100.0 percentage of codebase affected
```

**PlanStep** - Incremental planning with step-level metadata:
```python
@dataclass
class PlanStep:
    # Core fields
    description: str
    rationale: str
    dependencies: list[int] = []

    # Incremental planning (v0.8.0+)
    preconditions: list[str] = []      # Conditions before execution
    actions: list[str] = []            # Concrete actions to perform
    exit_criteria: list[str] = []      # Success verification criteria
    risk: str = "low"                  # "low", "medium", "high"
    retrieval_keys: list[str] = []     # Step-scoped memory retrieval
    failure_signatures: list[str] = [] # Known failure patterns
    verifier_checks: list[str] = []    # Validation checks (Solver-Critic-Verifier)
    expanded: bool = False             # Tracks seed → expansion
```

These schemas enable:
- **Actionable Output**: Commands and patches ready for execution
- **Incremental Planning**: Seed plans expand only when blocked (as-needed decomposition)
- **Step-Level Learning**: Failure signatures attach to specific steps for ReasoningBank
- **Multi-Agent Reasoning**: Verifier checks support MapCoder's Solver-Critic-Verifier pattern


### Storage Architecture

- **Scoped JSON Files**: Facts stored in `~/.neo/facts/` — separate files per scope (global, org, project)
- **Inline Embeddings**: Vector embeddings stored alongside facts in JSON (no separate FAISS index for memory)
- **Supersession**: Similar facts are superseded rather than merged — old facts remain but are marked invalid
- **Constraint Auto-Ingestion**: CLAUDE.md and similar files are automatically scanned and ingested as CONSTRAINT facts
- **Project Index** (separate system): Tree-sitter code indexing uses FAISS for per-repository semantic search in `.neo/`


## Performance

**Neo improves over time as it learns from experience.** Initial performance depends on available facts. Performance grows as the semantic memory builds up successful solutions, failure learnings, and architectural decisions.


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
neo --config set --config-key model --config-value claude-sonnet-4-5-20250929
neo --config set --config-key api_key --config-value sk-ant-...

# Reset to defaults
neo --config reset
```

**Exposed Configuration Fields:**
- `provider` - LM provider (openai, anthropic, google, azure, ollama, local)
- `model` - Model name (e.g., gpt-5.3-codex, claude-sonnet-4-5-20250929)
- `api_key` - API key for the chosen provider
- `base_url` - Base URL for local/Ollama endpoints
- `memory_backend` - Memory backend: "fact_store" (default) or "legacy"
- `auto_install_updates` - Automatically install updates in background (true/false)
- `constraint_auto_scan` - Auto-scan CLAUDE.md for constraints (true/false, default: true)

Configuration is stored in `~/.neo/config.json` and takes precedence over environment variables.

### Environment Variables

Alternatively, use environment variables for configuration:

```bash
# Required: LM Provider API Key
export ANTHROPIC_API_KEY=sk-ant-...
```

## LM Adapters

### OpenAI (Default)

```python
from neo.adapters import OpenAIAdapter
adapter = OpenAIAdapter(model="gpt-5.3-codex", api_key="sk-...")
```

Default model: `gpt-5.3-codex`. GPT-5/Codex models use the `/v1/responses` endpoint automatically.

### Anthropic

```python
from neo.adapters import AnthropicAdapter
adapter = AnthropicAdapter(model="claude-sonnet-4-5-20250929")
```

Default model: `claude-sonnet-4-5-20250929`

### Google

**Note: Requires Python 3.10+ and google-genai>=0.2.0**

```python
from neo.adapters import GoogleAdapter
adapter = GoogleAdapter(model="gemini-2.0-flash")
```

Default model: `gemini-2.0-flash`. Uses the `google-genai` SDK.

### Ollama

```python
from neo.adapters import OllamaAdapter
adapter = OllamaAdapter(model="llama2")
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

- **Fact-Based Memory**: Learns from every solution attempt using a scoped, supersession-based fact store
- **Semantic Retrieval**: Vector search finds relevant facts via Jina Code embeddings
- **Code-First Generation**: No diff parsing failures
- **Scoped Storage**: Privacy-first JSON storage in ~/.neo/facts/ with global, org, and project scopes
- **Model-Agnostic**: Works with any LM provider
- **The Construct**: Curated library of architecture patterns with semantic search
- **Project Indexing**: Tree-sitter based multi-language code indexing with FAISS
- **Prompt Enhancement**: Analyze and improve prompt effectiveness

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

Neo's architecture is grounded in peer-reviewed research on code generation, semantic memory, and multi-agent reasoning.

### Academic Papers

**Semantic Memory & Failure Learning:**

1. **ReasoningBank: Systematic Failure Learning and Semantic Anchor Embedding**
   *Chen et al., 2025* | [arXiv:2509.25140](https://arxiv.org/abs/2509.25140)
   - Phase 2: Semantic anchor embedding (pattern+context, not full reasoning)
   - Phase 3: Failure root cause extraction with contrastive learning
   - Phase 4: Self-contrast consolidation (archetypal vs spurious patterns)
   - Phase 5: Strategy evolution tracking (procedural/adaptive/compositional)
   - **Implementation**: Neo's persistent memory system with failure signatures

**Code Generation & Planning:**

2. **Planning with Large Language Models for Code Generation**
   *Liu et al., ICLR 2023* | [Paper](https://openreview.net/forum?id=Lr8cOOtYbfL)
   - Planning-guided test-driven decoding (PG-TD)
   - Step-level preconditions and exit criteria
   - **Implementation**: Neo's PlanStep schema with preconditions/exit_criteria fields

3. **Self-Planning Code Generation with Large Language Models**
   *Zhang et al., 2023* | [arXiv:2303.06689](https://arxiv.org/abs/2303.06689)
   - Two-phase plan-then-generate workflow
   - +7% improvement on HumanEval-X Pass@1
   - **Implementation**: Neo's planning phase before code generation

4. **AdaCoder: Adaptive Planning and Multi-Agent Framework for Function-Level Code Generation**
   *Huang et al., 2025* | [arXiv:2407.13433](https://arxiv.org/abs/2407.13433)
   - Task decomposition with planning, generation, and testing agents
   - Explicit risk assessment per step
   - **Implementation**: Neo's estimated_risk and verifier_checks fields

**Multi-Agent Reasoning:**

5. **MapCoder: Multi-Agent Code Generation for Competitive Programming**
   *Islam et al., 2024* | [arXiv:2405.11403](https://arxiv.org/abs/2405.11403)
   [GitHub](https://github.com/Md-Ashraful-Pramanik/MapCoder)
   - Solver-Critic-Verifier agent collaboration
   - Step-level verification and critique
   - **Implementation**: Neo's verifier_checks and multi-phase reasoning

**Retrieval & Similarity:**

6. **CodeSim: Effective Semantic Similarity Metrics for Code**
   *Xu et al., 2023* | [Paper](https://dl.acm.org/doi/10.1145/3611643.3616367)
   - Code-specific similarity metrics for retrieval
   - Step-scoped vs global retrieval tradeoffs
   - **Implementation**: Neo's retrieval_keys for per-step memory access

**Agent Architectures:**

7. **As-Needed Decomposition and Planning with Language Models**
   *Yao et al., NAACL 2024* | [arXiv:2311.05772](https://arxiv.org/abs/2311.05772)
   - Selective planning (seed → expand when blocked)
   - Avoids over-planning on simple tasks
   - **Implementation**: Neo's expanded flag and incremental planning design

8. **Large Language Model-Based Multi-Agents: A Survey of Progress and Challenges**
   *Wang et al., 2024* | [arXiv:2402.01680](https://arxiv.org/abs/2402.01680)
   - Task decomposition, plan selection, and reflection as standard components
   - Multi-agent coordination patterns
   - **Implementation**: Neo's architectural foundations

### Technologies & Libraries

**Embedding & Search:**

- **Jina Embeddings v2 (Code)**
  [HuggingFace](https://huggingface.co/jinaai/jina-embeddings-v2-base-code) | [GitHub](https://github.com/jina-ai/embeddings)
  - 768-dimensional embeddings optimized for code similarity
  - Local inference (no API calls)
  - **Used in**: Neo's semantic memory and pattern retrieval

- **FAISS (Facebook AI Similarity Search)**
  [GitHub](https://github.com/facebookresearch/faiss) | [Docs](https://faiss.ai/)
  - Efficient vector similarity search and clustering
  - Billion-scale index support
  - **Used in**: Neo's fast pattern matching (<13ms avg)

- **FastEmbed**
  [GitHub](https://github.com/qdrant/fastembed) | [Docs](https://qdrant.github.io/fastembed/)
  - Lightweight local embedding generation
  - ONNX Runtime backend
  - **Used in**: Neo's local embedding pipeline

**Datasets (for Load Program):**

- **MBPP (Mostly Basic Programming Problems)**
  [HuggingFace](https://huggingface.co/datasets/google-research-datasets/mbpp) | [Paper](https://arxiv.org/abs/2108.07732)
  - 1,000 crowd-sourced Python programming problems
  - **Used for**: Bootstrapping Neo's semantic memory

- **HumanEval**
  [HuggingFace](https://huggingface.co/datasets/openai/openai_humaneval) | [Paper](https://arxiv.org/abs/2107.03374)
  - 164 hand-written programming problems
  - **Used for**: Quality pattern seeding

### Citation

If you use Neo in academic research, please cite:

```bibtex
@software{neo2025,
  title={Neo: Self-Improving Code Reasoning Engine with Persistent Semantic Memory},
  author={Parslee AI},
  year={2025},
  url={https://github.com/Parslee-ai/neo},
  note={Built on ReasoningBank (Chen et al., 2025), MapCoder (Islam et al., 2024), and CodeSim (Xu et al., 2023)}
}
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

