![Neo Banner: Imagery from The Matrix film series](https://ik.imagekit.io/xvpgfijuw/parslee/bannerFor__Neo--Github.webp)
***


# Neo

> An evidence-learning code reasoning engine with explicit operating modes. Neo uses persistent semantic memory to improve from verified experience without silently gaining repository authority.

- **Fact-Based Memory**: Derives scoped, reversible knowledge from attributed and repeatedly verified outcomes—not from generation alone
- **Semantic Retrieval**: Vector search finds relevant facts via Jina Code embeddings
- **Code-First Generation**: No diff parsing failures
- **Local Storage**: Privacy-first JSON storage in ~/.neo/facts/ directory
- **Model-Agnostic**: Works with any LM provider
- **Three integration surfaces** — each on equal footing:
  - **[Run as an Agent (CAR / A2A)](#run-as-an-agent-car--a2a)** — host Neo as an Agent2Agent v1.0 endpoint other agents (or orchestrators) can call directly. Real inference path, not a CLI wrapper.
  - **[Claude Code Plugin](#claude-code-plugin)** — six slash commands + a specialized agent inside Anthropic's Claude Code CLI.
  - **[Codex Plugin](#codex-plugin)** — same six skills, packaged for OpenAI Codex CLI.

[![PyPI version](https://img.shields.io/pypi/v/neo-reasoner.svg)](https://pypi.org/project/neo-reasoner/)
[![Python Versions](https://img.shields.io/pypi/pyversions/neo-reasoner.svg)](https://pypi.org/project/neo-reasoner/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Why Neo?  Why Care?  
If you've been Vibe Coding, then Vibe Planning, then Context Engineering, and on and on, you have likely hit walls where the models are both powerful and limited, brilliant and incompetent, wise and ignorant, humble yet overconfident. 

Worse, your speedy AI Code Assistant sometimes goes rogue and overwrites key code in a project, or writes redundant code even after just reading documentation and the source code, or violates your project's patterns and design philosophy....  _It can be infuriating._  Why doesn't the model remember?  Why doesn't it learn?  Why can't it keep the context of the code patterns and tech stack? ... -> This is what Neo is designed to solve.  

Neo is **_the missing context layer_** for AI Code Assistants. It retrieves relevant verified patterns, records suggestions as evidence candidates, and promotes knowledge only after attributed support. Repository writes and execution are separate, explicit host-granted capabilities.


# Table of Contents

- [Design Philosophy](#design-philosophy)
- [How It Works](#how-it-works)
- [The Construct](#the-construct)
- [Quick Start](#quick-start)
- [Run as an Agent (CAR / A2A)](#run-as-an-agent-car--a2a)
- [Claude Code Plugin](#claude-code-plugin)
- [Codex Plugin](#codex-plugin)
- [Works Alongside Your AI Tools](#works-alongside-your-ai-tools)
- [Installation](#installation)
  - [From PyPI (Recommended)](#from-pypi-recommended)
  - [Updating Neo](#updating-neo)
  - [From Source (Development)](#from-source-development)
  - [Dependencies](#dependencies)
  - [Optional: Additional LM Providers](#optional-additional-lm-providers)
- [Usage](#usage)
  - [CLI Interface](#cli-interface)
  - [Command Reference](#command-reference)
  - [Operating Modes](#operating-modes)
  - [Goal-aware agent-loop envelope](#goal-aware-agent-loop-envelope)
  - [Memory Maintenance](#memory-maintenance)
  - [Background Observer](#background-observer)
  - [Memory Diagnostics](#memory-diagnostics)
  - [Timeout Requirements](#timeout-requirements)
  - [Output Format](#output-format)
  - [Personality System](#personality-system)
  - [Load Program — Training Neo's Memory](#load-program---training-neos-memory)
- [Architecture](#architecture)
  - [Fact-Based Memory](#fact-based-memory)
  - [Output Schemas](#output-schemas)
  - [Code Smell Detection in Context Assembly](#code-smell-detection-in-context-assembly)
  - [Smart File Selection](#smart-file-selection)
  - [Learning Feedback Loop](#learning-feedback-loop)
  - [Storage Architecture](#storage-architecture)
- [Performance](#performance)
  - [Memory-Driven Reasoning Effort](#memory-driven-reasoning-effort-gpt-5-models)
  - [Architectural Quality Feedback Loop](#architectural-quality-feedback-loop)
- [Configuration](#configuration)
  - [CLI Configuration Management](#cli-configuration-management)
  - [Secure API Key Storage](#secure-api-key-storage)
  - [Environment Variables](#environment-variables)
- [LM Adapters](#lm-adapters)
  - [OpenAI (Default)](#openai-default)
  - [Anthropic](#anthropic)
  - [Google](#google)
  - [Ollama](#ollama)
  - [CAR (Common Agent Runtime)](#car-common-agent-runtime)
- [Extending Neo](#extending-neo)
  - [Add a New LM Provider](#add-a-new-lm-provider)
- [Key Features](#key-features)
- [Development](#development)
  - [Running Tests](#running-tests)
- [Research & References](#research--references)
  - [Academic Papers](#academic-papers)
  - [Technologies & Libraries](#technologies--libraries)
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



## Quick Start

```bash
# Install from PyPI (recommended)
pip install neo-reasoner

# Or install with specific LM provider
pip install neo-reasoner[openai]     # For GPT (same provider as the default)
pip install neo-reasoner[anthropic]  # For Claude
pip install neo-reasoner[google]     # For Gemini
pip install neo-reasoner[all]        # All providers

# Set API key
export OPENAI_API_KEY=sk-...

# Test Neo
neo --version
```

**See [QUICKSTART.md](QUICKSTART.md) for 5-minute setup guide**


## Run as an Agent (CAR / A2A)

Neo integrates with Parslee's **Common Agent Runtime (CAR)** as a first-class peer of the CLI and the plugins. The integration runs **both directions**:

- **Inbound (host)** — `neo serve` exposes Neo as an Agent2Agent v1.0 endpoint. Other agents and orchestrators call Neo's `neo.process` tool over A2A directly. No CLI shell-out, no subprocess parsing.
- **Outbound (inference)** — set `provider="car"` to route Neo's *own* LLM calls through CAR's unified inference layer. CAR's adaptive router picks **local backends** (Candle + MLX for Qwen3, Gemma 4) or **remote providers** (OpenAI, Anthropic, Google) per call based on task complexity, context-window headroom, and per-model latency/cost. Rust-enforced policies, deterministic eventlog/replay, and semantic conversation compaction all come for free.

A single `CarRuntime` is shared per process — if `neo serve` is running and the same process makes outbound calls, both surfaces see the same state, policies, tool registry, and eventlog.

### Install the CAR extras

```bash
# CAR-backed serving and inference both require the car-runtime Python bindings
pip install "neo-reasoner[car]"
```

`car-runtime` ships as a sealed binary under a separate license (the rest of Neo stays Apache-2.0). Skip this extra if you only need the plugins and the CLI with direct provider SDKs.

### Inbound: host Neo as an A2A endpoint

```bash
# 1. Start the CAR daemon (default ws://127.0.0.1:9100)
python -m car_runtime.server
# or, if installed standalone:
car-server

# 2. In another terminal, host Neo as an A2A endpoint
neo serve
```

`neo serve` boots a `CarRuntime`, registers Neo as the `neo.process` tool with its full schema (`src/neo/car_tool_schema.py`), installs the Python `tools.execute` handler, and binds the A2A HTTP listener. It blocks until `SIGINT`/`SIGTERM`.

### Outbound: use CAR as Neo's inference layer

Outbound routing is controlled by the `inference_mode` config field, **not** by
`provider`:

| `inference_mode` | Behavior |
|------------------|----------|
| `static` (default) | Always use the configured `provider` / `model`. CAR is never called. |
| `auto`             | Use CAR's adaptive router when `car-runtime` is importable **and** the daemon is reachable; fall back to the static provider on absence or runtime failure. |

The default is `static` until a CAR release verifies the router's quality
behavior (see the known limitation below). Opt in persistently or per-shell:

```bash
# Persistently: prefer CAR's router, fall back to the static provider
neo --config set --config-key inference_mode --config-value auto

# Or per-invocation / per-shell
export NEO_INFERENCE_MODE=auto
```

To pin CAR as the **only** backend (no static fallback), set the provider
itself. Clear `model` so CAR's router chooses per call, or set it to pin one
backend:

```bash
neo --config set --config-key provider --config-value car

# Let the router pick per call (an empty value clears the field)
neo --config set --config-key model --config-value ""

# Or pin a specific backend — local or remote
neo --config set --config-key model --config-value qwen3-32b
```

Environment overrides work the same way: `NEO_PROVIDER=car`, `NEO_MODEL=...`.
Note that with `provider=car`, whatever `model` holds is passed straight to CAR
as a pin — leaving it at the default `gpt-5.6` pins that model rather than
letting the router choose, which is why clearing it matters.

The CAR daemon must be running (`car-server` / `python -m car_runtime.server`). From Python:

```python
from neo.adapters import create_adapter
adapter = create_adapter("car")                                    # router picks a code-capable model
adapter = create_adapter("car", model="Qwen3-4B")                  # pin a specific backend
adapter = create_adapter(
    "car",
    intent_hint={"task": "reasoning", "prefer_local": True},       # override the default
)
```

**Default intent**: `CarAdapter` sends `intent_json={"task": "code"}` on every call unless you supply your own `intent_hint`. Neo's workload is overwhelmingly code reasoning (review, optimization, debugging, generation), so the router gets to pick a code-capable model rather than the chat default. CAR's task enum is `chat | classify | reasoning | code`. The rest of `IntentHint` (`prefer_local`, `prefer_fast`, `require: ModelCapability[]`) is how you express *what else you need* without pinning a model ID.

> **Known limitation upstream:** CAR's `route_model` currently scores prompts as "simple" by heuristic length and picks the cheap chat-tier model (e.g. `gpt-4.1-mini`) even when models like `gpt-5.3-codex` and `o3` are registered and ranked as fallbacks. Tracked at [Parslee-ai/car-releases#52](https://github.com/Parslee-ai/car-releases/issues/52). The `task=code` default is Neo's local workaround — substantive prompts do escalate; trivial ones don't.

### Discover what's installed

```bash
# Detects native CLI, car-server, Python bindings, and the default daemon port
neo car status

# Also surfaced in --version output
neo --version
```

If the CLI/daemon are present but the Python bindings aren't, Neo reports that state cleanly. CAR install options live at [Parslee-ai/car-releases](https://github.com/Parslee-ai/car-releases).

### Why use the CAR surfaces

- **Real inference path both ways** — inbound, callers see Neo as a typed A2A tool; outbound, Neo gets local-first inference with automatic remote fallback through one provider-agnostic protocol
- **One runtime per host** — session state, tool registry, policies, and the eventlog stay consistent across A2A inbound and inference outbound in the same process
- **Local-first inference, free fallback** — Qwen3 / Gemma 4 on-device via Candle + MLX; remote OpenAI / Anthropic / Google when the router decides the task needs it
- **Policies enforced in Rust** — deny rules and capability requirements run before any side-effecting call
- **Memory is shared across all surfaces** — `~/.neo/facts/` and per-project indexes are the same whether you invoke via CLI, plugin, `neo serve`, or CAR inference


## Claude Code Plugin

Neo ships as a **Claude Code plugin** with a specialized agent and six slash commands. Anthropic's Claude Code CLI installs it from Parslee's plugin marketplace:

```bash
# Add the marketplace
/plugin marketplace add Parslee-ai/claude-code-plugins

# Install Neo
/plugin install neo
```

Once installed:

- **Slash commands**: `/neo`, `/neo-review`, `/neo-optimize`, `/neo-architect`, `/neo-debug`, `/neo-pattern`
- **Specialized agent**: invoke with `Use the Neo agent to ...` for delegated semantic reasoning
- **Shared memory**: same `~/.neo/facts/` store used by the CLI and the Codex plugin

Examples:

```bash
/neo-review src/api/handlers.py
/neo-optimize process_large_dataset function
/neo-architect Should I use microservices or monolith?
/neo-debug Race condition in task processor
```

The plugin wraps the local `neo` CLI, so the binary must be installed first (`pip install neo-reasoner[openai]` and `OPENAI_API_KEY` set, or your provider of choice).

Plugin sources live under [`.claude-plugin/`](.claude-plugin/) — `plugin.json` is the manifest, `agents/neo.md` defines the agent, and `commands/*.md` defines each slash command.


## Codex Plugin

Neo ships as a **Codex plugin** with the same six skills, packaged for [OpenAI Codex CLI](https://developers.openai.com/codex/plugins). Installing takes **two** steps — registering the marketplace does not install the plugin:

```bash
# 1. Register the marketplace (names it `neo-local`)
codex plugin marketplace add Parslee-ai/neo

# ...or, from a local checkout, point Codex at the in-tree marketplace
codex plugin marketplace add ./

# 2. Install the plugin from it
codex plugin add neo@neo-local
```

Verify with `codex plugin list` — you want `neo@neo-local  installed, enabled`.
Stopping after step 1 leaves a registered marketplace and **no installed
plugin**, which looks like success and provides no skills. The subcommand is
`add`, not `install`.

Once installed:

- **Skills**: `$neo`, `$neo-review`, `$neo-optimize`, `$neo-architect`, `$neo-debug`, `$neo-pattern`
- **Shared memory**: same `~/.neo/facts/` store used by the CLI and the Claude Code plugin
- **Explicit context boundary**: Codex selects the relevant excerpts and every
  skill invokes `neo --no-scan`, so Neo cannot silently add files or project
  instruction documents from the current directory
- **Privacy-safe advice**: advise skills also use `--no-memory` by default, so
  stored Neo facts cannot silently enter an external-provider prompt; memory is
  opt-in with its data category disclosed

Examples:

```bash
$neo-review src/api/handlers.py
$neo-optimize process_large_dataset function
$neo-architect Should I use microservices or monolith?
$neo-debug Race condition in task processor
```

The plugin wraps the local `neo` CLI, so the binary must be installed first
(`pip install neo-reasoner[openai]` and `OPENAI_API_KEY` set, or your provider
of choice). The plugin is intentionally a skills integration, not a Claude
agent or an MCP server: Codex invokes Neo inside its coding loop and then keeps
working. If Neo uses an external provider, Codex discloses the provider and the
files or data categories being sent in its approval request. Invoking a Neo
skill does not authorize unrelated production, private, or customer material.

Anything you deliberately teach Neo from Codex is immediately available in the
Claude Code plugin and the CAR endpoint, and vice versa — there is one fact
store per host. Codex's `--no-scan` and `--no-memory` boundaries change only
what is read and sent on that invocation; they do not alter or fork the shared
memory store. Deliberate pattern learning retains memory access and discloses
both retrieved-fact and persistence effects before provider approval.

Plugin sources live under [`plugins/neo/`](plugins/neo/) — see the manifest and skill definitions.

Both plugins consume the **same host-neutral contract**: the skills invoke
`neo --json` and read the `orchestrator` envelope described under
[Orchestrator output](#orchestrator-output). Neither hosts its own output
format, and `tests/test_host_adapter_parity.py` fails if one surface teaches a
contract the other does not. The difference is role, not protocol — under
Claude Code, Neo is usually a delegated subagent with a visible boundary;
under Codex, Neo is a step inside the same coding loop, which is why the Codex
skills insist on explicit attribution and on continuing the task rather than
treating Neo's answer as the deliverable.


## Works Alongside Your AI Tools

Neo automatically reads project-local agent instruction docs from a wide range
of ecosystems and folds them into its reasoning context — no configuration
needed. If you've already invested in writing a `CLAUDE.md`, an `AGENTS.md`,
`.cursor/rules/`, `.github/copilot-instructions.md`, or a Spec Kit project,
neo respects that work.

| Tool                  | Files / dirs neo discovers                                   |
|-----------------------|--------------------------------------------------------------|
| Claude / Claude Code  | `CLAUDE.md`, `.claude/CLAUDE.md`, `.claude/agents/*.md`, `.claude/commands/*.md` |
| Codex / AGENTS.md spec| `AGENTS.md`, `.github/AGENTS.md`, `.codex/**/*.md`           |
| Cursor                | `.cursorrules`, `.cursor/rules/**/*.md`, `.cursor/rules/**/*.mdc` |
| GitHub Copilot        | `.github/copilot-instructions.md`                            |
| Windsurf              | `.windsurfrules`                                             |
| Continue              | `.continue/**/*.md`                                          |
| Augment               | `.augment/**/*.md`                                           |
| Spec Kit              | `.specify/**/*.md`                                           |
| Aider                 | `.aider/*.md`                                                |
| Codeium               | `.codeium/*.md`                                              |

Discovered docs surface in neo's prompt under **PROJECT-LOCAL AGENT CONTEXT**,
included unconditionally — independent of relevance ranking — because their
value is global to the project. Per-file cap of 6KB and total cap of 32KB
keep prompt growth bounded.

**This means neo composes well with whichever AI coding workflow you already
use:**

- **Claude Code** users get the deepest integration via the [Claude Code Plugin](#claude-code-plugin), but neo runs standalone too.
- **Codex CLI** users get parity via the [Codex Plugin](#codex-plugin) — same six skills, packaged for Codex. Neo also automatically picks up `AGENTS.md` (the cross-tool standard Codex co-led) plus anything under `.codex/`.
- **Cursor / Windsurf / Aider / Continue / Augment** users — the rules dirs you've curated land in every neo session's context.
- **GitHub Copilot** users — `.github/copilot-instructions.md` is read on every invocation.
- **Spec Kit** projects — your specs are folded into neo's reasoning context, no manual paste.

Adding a new tool is a one-liner: extend the discovery rules in
`src/neo/agent_context.py`. The list is the load-bearing surface for keeping
this current as new agent ecosystems emerge.


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

Automatic update installation is enabled by default for pipx and virtualenv
installs. You can set it explicitly with:

```bash
# Enable auto-install (persisted in ~/.neo/config.json)
neo --config set --config-key auto_install_updates --config-value true

# Or use environment variable
export NEO_AUTO_INSTALL_UPDATES=1
```

When enabled, Neo will:
- Check for updates once every hour using a stale-while-revalidate cache
- Automatically download and install new versions in the background
- Notify you when updates complete
- Log all auto-update activity to `~/.neo/auto_update.log`

**Example output when auto-install is enabled:**
```bash
$ neo "your query"

⚡ Auto-installing neo update: 0.40.0 → 0.41.0
   This happens in the background. Please wait...

✓ Auto-update completed: 0.41.0
   Restart neo to use the new version.

[Neo] Processing your query...
```

#### Update Notifications (Default)

By default, Neo checks for updates once every hour and displays a notification
when a new version is available. This check happens in the background and will
not interrupt your workflow.

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
- pyyaml >= 6.0
- openai >= 1.0.0  *(default provider; base install is runnable with just `OPENAI_API_KEY`)*
- tree-sitter >= 0.23, < 0.26
- tree-sitter-language-pack >= 0.13.0, < 1.0


### Optional: Additional LM Providers

OpenAI is bundled in the base install. Add others as needed:

```bash
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

# Optional: build the embedding catalog (adds semantic re-ranking to file selection).
# Nothing requires this — file selection works on a fresh clone, and the walk and
# keyword index refresh themselves on every call.
neo --index

# Re-embed catalogued files whose contents changed. Must be passed WITH --index;
# new files are not picked up, so a repo that grew needs a full build.
neo --index --update

# Preview the assembled context without making an LLM call
neo --dry-run "your query"

# Check version and memory stats
neo --version

# Inspect detected local CAR runtime surfaces
neo car status
```

### Command Reference

Neo is one binary with a plain-text prompt plus a handful of subcommands. Run
`neo --help` for the flag list.

**Subcommands**

| Command | Purpose |
|---------|---------|
| `neo "<prompt>"` | Reason about a prompt in the current repo (the default path) |
| `neo memory <action>` | Memory maintenance, diagnostics, and the background observer — see [Memory Maintenance](#memory-maintenance) |
| `neo construct <action>` | The Construct pattern library: `list`, `show`, `search`, `index` |
| `neo car status` | Report detected CAR CLI / server / Python bindings / daemon |
| `neo serve` | Host Neo as an A2A endpoint (`--a2a-bind`, `--public-url`, `--agent-name`) |
| `neo prompt <action>` | Prompt-effectiveness tooling: `analyze`, `enhance`, `patterns`, `suggest`, `history`, `stats` |
| `neo contribute` | Export high-confidence patterns for community contribution |
| `neo update` | Update the installed `neo` package in place |

**Global flags**

| Flag | Purpose |
|------|---------|
| `--cwd PATH` | Working-directory override (which repo Neo reasons about) |
| `--mode {advise,patch,verify,learn,agent}` | Operating mode; default `learn` — see [Operating Modes](#operating-modes) |
| `--fast` / `--deep` | Force the single-call path / force multi-agent deliberation (default is `auto`, which gates on novelty + CAR availability) |
| `--dry-run` | Assemble and print the full context, then exit without an LLM call |
| `--json` | JSONL progress events on **stderr**, one final JSON object on **stdout** (also the JSON *input* path) — see [Orchestrator output](#orchestrator-output) |
| `--output-schema NAME_OR_PATH` | Constrain the shape of the final JSON response |
| `--index` / `--update` / `--languages CSV` | Optional cache-warmer: build, re-embed, and scope the per-project embedding catalog. Not a prerequisite for anything — see [Smart File Selection](#smart-file-selection) |
| `--semantic` | A hint, not a mode: reads the embedding catalog deeper and weighs it as heavily as the keyword index. The catalog is consulted on every run whether or not you pass this; with no catalog present, the flag says so |
| `--max-bytes N` | Hard cap on total context bytes (default 300000) |
| `--max-files N` | Cap on files: context gathering (default 30), or the index build when passed with `--index` (default 100). The index build apportions this budget across languages by repo composition and reports what the cap left out |
| `--include GLOB` / `--exclude GLOB` | Allow/block file patterns; both repeatable |
| `--exts CSV` | Restrict context to these file extensions |
| `--diff-since REV` | Prioritize files changed since a git rev or duration |
| `--no-git` / `--no-scan` | Skip git heuristics / skip the directory scan entirely |
| `--stdin-json` / `--stdin-text` | Force the stdin input mode instead of auto-detecting |
| `--quiet` | Suppress the `[Neo]` progress notices on stderr (implied by `--json`) |
| `--verbose` / `--debug` | INFO / DEBUG logging to stderr |
| `--allow-write-path GLOB` / `--allow-command CMD` | `agent`-mode authority grants (repeatable) |
| `--config {list,get,set,reset}` | Configuration management — see [Configuration](#configuration) |
| `--load-program DATASET_ID` | Import a HuggingFace dataset into memory — see [Load Program](#load-program---training-neos-memory) |
| `--regenerate-embeddings` | Rebuild legacy `ReasoningMemory` embeddings with the current model (automatic backup) |

### Operating Modes

```bash
# Read-only analysis; retrieves memory but does not learn
neo --mode advise "review the authorization flow"

# Produce applicable change artifacts without applying them or learning
neo --mode patch "add validation to the endpoint"

# Backward-compatible default: read-only repository reasoning plus evidence learning
neo --mode learn "fix the validation bug"

# VERIFY is JSON/A2A-only because it requires caller-provided change content
echo '{"prompt":"verify","operating_mode":"verify","proposed_changes":[{"file_path":"src/app.py","code_block":"value = 1"}]}' | neo --json
```

`agent` is not a synonym for unrestricted autonomy. It requires an explicit
workspace-rooted authority policy and a host-provided execution adapter. Neo has
no built-in shell or repository executor, never invokes generated command
strings, and the standalone CLI fails closed for `agent`. See the
[operating-mode contract](docs/solutions/operating-modes.md).

### Orchestrator output

`--json` writes two streams so a host can narrate a run instead of waiting on a
black box:

- **stdout** — exactly one JSON document. Safe to pipe: `neo --json … | jq`.
- **stderr** — JSONL lifecycle events, one object per line, flushed as they happen.

```bash
neo --json --mode advise "why is the parser crashing on empty input?"
```

```jsonc
// stderr, as the run progresses
{"type":"phase_completed","phase":"context","message":"Read 25 file(s) of context.","data":{"status":"complete"}}
{"type":"phase_started","phase":"reasoning","message":"Planning, simulating, and drafting changes."}
{"type":"memory_found","phase":"reasoning","message":"Recalled 20 relevant fact(s) from memory.","data":{"count":20}}
{"type":"risk_found","phase":"reasoning","message":"callers may not handle None","data":{"source":"simulation"}}
{"type":"completed","message":"…","data":{"confidence":0.9,"elapsed_seconds":25.98}}
```

Event types: `started`, `phase_started`, `phase_completed`, `memory_found`,
`hypothesis_formed`, `hypothesis_rejected`, `risk_found`, `personality_beat`,
`completed`, `failed`. Exactly one of `completed` or `failed` terminates every
run; on `failed`, stdout is an `{"error": …}` object with no `orchestrator` key.

Phase names are stable: `context`, `reasoning`, `static_checks`. `context`
covers file gathering only — fact retrieval runs during `reasoning`, which is
why `memory_found` above carries `phase: reasoning`.

`--json` implies `--quiet`, so the `[Neo]` progress lines are suppressed and
stderr is essentially pure JSONL. Logging warnings can still appear there, so
parse lines beginning with `{` and ignore the rest.

Neo's wording shifts with how much he remembers about the project — the same
run reads `Don't know this code. I'd change 1 thing(s) in src/parser.py, maybe.
Confidence 0.88.` at memory stage 1 and `src/parser.py. 1 change(s). 0.88.` at
stage 5. All of it is authored in
[`neo_matrix.yaml`](src/neo/config/beats/neo_matrix.yaml) under
`orchestrator_voice`, not in code.

The stdout document adds an `orchestrator` object stating what the run did, so
a host doesn't have to infer presentation from raw plans and traces:

```json
{
  "orchestrator": {
    "summary": "Neo reasoned over the request and proposes 1 change(s) in src/parser.py. Confidence 0.90.",
    "personality": "I've seen this shape before. Let me use what I remember.",
    "phase_summary": [{"name": "reasoning", "status": "complete", "message": "…"}],
    "cautions": ["Simulation surfaced 1 issue(s) with the proposed approach."],
    "recommended_narration": ["Read 25 file(s) of context.", "…"]
  }
}
```

`cautions` is the field to never drop — low confidence, failed checks, and open
questions live there. `personality` is present only when a beat matched and,
for beats that claim insight, only when the run actually found something; there
is no fallback line. Embedding hosts should read
[the presentation contract](.claude-plugin/agents/neo.md) and
[the design notes](docs/solutions/orchestrator-communication.md).

### Goal-aware agent-loop envelope

JSON, CAR, and A2A callers can distinguish the larger goal from the current
task, the reason Neo was invoked, the current attempt, and its observed outcome:

```json
{
  "prompt": "Tests still fail in auth/session_test.py",
  "goal": {
    "description": "All authentication tests pass",
    "success_criteria": [
      {"type": "command", "command": "pytest tests/auth", "expected_exit_code": 0}
    ]
  },
  "intent": {
    "type": "diagnose_failed_attempt",
    "description": "Explain why the last fix stalled"
  },
  "constraints": ["Do not weaken tests", "Do not change public APIs"],
  "attempt": {"summary": "Changed session expiry handling"},
  "outcome": {"status": "failed", "summary": "3 tests remain"},
  "progress": {"metric": "failing_tests", "before": 11, "after": 3},
  "trajectory": {"iteration": 4, "max_iterations": 10},
  "role": "diagnostician",
  "requested_output": "next_action"
}
```

The response includes `goal_assessment`, `strategy_assessment`, and
`recommended_next_action`. Missing goal and intent fields are inferred locally,
marked `origin=inferred` with bounded confidence, and retained only as
provisional episode context. They cannot become durable policy. Retrieval uses
the resolved goal, intent, constraints, attempt, and outcome—not just the task
string. See the [execution-envelope contract](docs/solutions/goal-aware-execution-envelope.md).


### Memory Maintenance

```bash
# Explain why a fact exists and how evidence changed it (no LM call)
neo memory explain <fact-id-or-prefix>

# Machine-readable causal history
neo memory explain <fact-id-or-prefix> --json

# Reproduce evidence-learning quality and safety claims without an LM
neo memory evaluate-learning
neo memory evaluate-learning --json

# Compact fact files by dropping old invalid tombstones (default: > 30 days since last access)
neo memory prune

# Across every local project Neo has touched
neo memory prune --all

# Preview without writing
neo memory prune --dry-run --max-invalid-age-days 14
```

Use `prune` when a `~/.neo/facts/facts_project_*.json` file grows much larger than its 500-valid-fact cap — that gap is tombstone bloat from supersession. Defaults are conservative; raising `--max-invalid-age-days` is safe, lowering it past ~7 may evict tombstones still referenced by recent supersession chains.

`memory explain` is read-only and works for current facts and retained tombstones. It joins
the fact with local learning episodes to show supporting and contradicting evidence,
retrieval scores and context inclusion, confidence/effectiveness mutations, rollback
reasons, and the supersession chain. It never initializes embeddings or calls an LM.

`memory evaluate-learning` runs the versioned repeated-task corpus against
memory-disabled, legacy immediate-memory, and evidence-driven policies. It reports
quality, harmful-memory, unsupported-promotion, repeat-error, isolation, latency,
model-call, and token metrics, and exits nonzero if any safety scenario or threshold
fails. See [the evaluation contract](docs/solutions/evidence-learning-evaluation.md).

**Is it actually learning?** Two read-only commands answer that without an LM
call — one for the retrieve side, one for the promote side:

```bash
# Retrieval side: were retrieved facts actually used, and which detector earned the credit?
neo memory citation-stats --since 7d

# Promotion side: episodes, outcomes, candidate statuses, promotions/rollbacks
neo memory learning-stats --since 7d
```

`citation-stats` summarizes the `citation_survival` metric from
`~/.neo/metrics.jsonl` (retrieved / included / used, split by
`by_marker` / `by_self_report` / `by_overlap`). `learning-stats` reads the
episode ledger in `~/.neo/episodes`. Both take `--json`. Note that
`learning-stats` covers only the **interactive, attributed** path: an `IDLE`
reading means suggestions aren't being accepted downstream, not that Neo has
stopped learning — the background observer mints facts with no episode
footprint.

```bash
# Re-run implicit-feedback processing over linked session outcomes
neo memory replay-feedback              # current project
neo memory replay-feedback --all        # every local project Neo has touched
neo memory replay-feedback --dry-run    # report what would change, mutate nothing
```

Use `replay-feedback` after a memory-loop fix, to re-apply confidence and
`success_count` updates from already-recorded ACCEPTED / MODIFIED / UNVERIFIED
outcomes. It only touches linked, non-independent outcomes.
`--include-legacy-fallback` also inspects legacy `session_*.json` files (which
may re-replay already-processed sessions); `--limit N` bounds `--all`.

### Background Observer

Transcript mining runs out-of-band in a single global background process
supervised by CAR, so it never sits on the request path. It sweeps every
discovered project each cycle, mining lessons from Claude Code, Codex, and CAR
transcripts plus merged GitHub PRs.

```bash
neo memory observer status   # CAR-reported state + orphaned-process check
neo memory observer start
neo memory observer stop
neo memory observer kick     # force a cycle now (maps to CAR agents_restart)
```

It **autostarts** whenever `car-server` is reachable — opt out with
`NEO_OBSERVER_AUTOSTART=0`. With no CAR present it prints a one-time hint and
stays silent. Requires the `[car]` extra and a running `car-server`
(car-runtime ≥ 0.18.0). Logs land in
`~/.car/logs/neo-observer.{stdout,stderr}.log`. Tunables:
`NEO_OBSERVER_INTERVAL_SECONDS` (default 300), `NEO_OBSERVER_COOLDOWN`
(default 60), `NEO_OBSERVER_RECYCLE_CYCLES` (default 48 — the daemon re-execs
itself to bound RSS drift; 0 disables). See
[the observer design note](docs/solutions/async-observer.md).

### Memory Diagnostics

Read-only, flag-and-propose commands that inspect your rules and your other
tools' memory rather than Neo's own facts:

```bash
# Recurring frictions mined from transcript history, as ranked evidence-cited issues
neo memory issues --since 14d --min-cluster 3
neo memory issues --suggest-rules          # adds a bounded LM call per issue

# Drift between AGENTS.md / CLAUDE.md / GEMINI.md (gaps + LM-judged conflicts)
neo memory rules
neo memory rules --no-conflicts            # skip the LM conflict pass

# Malformed entries, near-duplicates, conflicts, and index drift in Claude Code's memory/*.md
neo memory audit

# Ingest a peer tool's memory files into Neo as REVIEW facts on probation
neo memory import --dry-run
```

All four take `--json` (except `import`, which takes `--dry-run` and
`--confidence`). `issues` reuses the ingester's transcript episodes but never
admits facts or moves the ingest watermark, so it is idempotent and decoupled
from fact admission. See
[conversation-mined issues](docs/solutions/conversation-mined-issues.md),
[rule-file sync](docs/solutions/rule-file-sync.md), and
[memory audit](docs/solutions/memory-audit.md).


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

On a fresh install:

```bash
$ neo --version
"What is real? How do you define 'real'?"

neo 0.41.0
Provider: openai | Model: gpt-5.6
Storage: FactStore (path: /Users/you/.neo/facts)
CAR: not found
Stage: Sleeper | Memory: 0.0%
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0 patterns | 0.00 avg confidence
```

Once memory has built up, the quote, the stage (`Sleeper` → `Glitch` →
`Unplugged` → `Training` → `The One`), and the bar all move, and a line about
patterns approaching community contribution appears:

```bash
$ neo --version
"I can see it now. The code is showing me."

neo 0.41.0
Provider: openai | Model: gpt-5.6
Storage: FactStore (path: /Users/you/.neo/facts)
CAR: python binding car_runtime | daemon running
Stage: Unplugged | Memory: 49.7%
███████████████████░░░░░░░░░░░░░░░░░░░░░
521 patterns | 0.64 avg confidence

⚡ 6 pattern(s) approaching contribution (need 0.8 confidence + 3 successes)
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

1. **Typed Facts**: Eight kinds — CONSTRAINT, ARCHITECTURE, DECISION, PATTERN, REVIEW, FAILURE, KNOWN_UNKNOWN, and EPISODE (instance-specific events with `{when, where, why, with_whom}` context).
2. **Scoped Organization**: Facts are scoped to global, organization, or project level, with per-scope valid-fact caps (200 / 100 / 500 / 50). Org and project are auto-detected from git remotes.
3. **Supersession & Pre-Write Dedup**: New facts with cosine similarity ≥ 0.85 to an existing fact short-circuit (bump the existing fact's access count) or supersede it. The pre-write canonical-signature check uses entity abstraction + verb-synonym folding to catch near-duplicates before they hit the store.
4. **Confidence + Effectiveness Ranking**: `rank_score = recall_decay(sim)·confidence + success_bonus·effectiveness_f + provenance_bonus`. The Ebbinghaus recall-probability transform gives frequently-recalled facts slower decay; LessonL-style effectiveness (`c/n` over reuse outcomes) multiplies the success bonus. Curated facts (CONSTRAINT/ARCHITECTURE/DECISION and `seed`/`community`/`synthesized`-tagged facts) bypass decay. (Nothing mints the `synthesized` tag anymore — facts carrying it predate the removal of REVIEW→PATTERN synthesis and keep their immunity.)
5. **Hybrid Retrieval**: 0.7·dense (Jina) + 0.3·BM25. Half the result slots ranked by full `rank_score`, half by raw cosine — novel-but-relevant facts aren't crowded out by validated winners.
6. **Evidence-Gated Promotion**: a suggestion is recorded as a *candidate*, not a fact. It only becomes a durable PATTERN after **two independent, git-verified acceptances** whose task prompt and diff shape agree (`_episode_signature` for project scope, the path-agnostic `_global_signature` for cross-repo lessons), and only when its kind is promotable — `algorithm` / `bugfix` classify to `pattern`; `feature`, `refactor`, and `explanation` deliberately do not. Unverifiable suggestions (no path a diff could ever name) are downgraded to non-promotable `review` at mint. There is **no** similarity-clustered promotion path: the older REVIEW→PATTERN synthesis (`synthesize_reviews`, with its triple-trigger gate, Hebbian bump, and global decay) was removed after four months in which it minted 114 facts and not one PATTERN. See [evidence-learning episodes](docs/solutions/evidence-learning-episodes.md).
7. **Dual-Buffer Probation**: New non-curated facts enter with a `probation` tag and a 3-day stale window (vs 7/14 normal); promoted automatically on `access_count ≥ 2` or `success_count > 0` — quietly evicts noise while keeping real signal.
8. **Four-Layer Context**: Retrieved facts are organized into constraints, relevant knowledge, recent changes, and known unknowns. The four-layer state model is from *Beyond Conversation: A State-Based Context Architecture for Enterprise AI Agents* (Liotta, 2025) — see [`papers/state-based-context-architecture.pdf`](papers/state-based-context-architecture.pdf). The token-budget enforcement in `memory/context.py` is ported from the engine described in *Memgine: A Deterministic Memory Engine for Stateful AI Agents* (Liotta, 2026) — see [`papers/memgine-deterministic-memory-engine.pdf`](papers/memgine-deterministic-memory-engine.pdf). Both are evaluated by [StateBench](https://github.com/parslee-ai/statebench); the 95.8% decision-accuracy result on the v1.0 development split is what drove Neo's move from a separate "Recently Changed" section to inline `(changed from: X)` annotations.

### Output Schemas

Neo generates structured outputs with executable code and planning artifacts:

**CodeSuggestion** - Applicable code artifacts with advisory execution metadata:
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

    # Applicable/advisory artifacts (never executed without a host authority adapter)
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


### Code Smell Detection in Context Assembly

Neo scans the relevance-ranked file set during context assembly and surfaces
known issues to the model under **KNOWN ISSUES IN NEARBY CODE**. Detectors
are intentionally high-precision (false positives turn into prompt bloat
that hurts more than it helps):

- TODO / FIXME / HACK / XXX markers (any text file)
- Python stubs: `pass`-only / `...`-only / `raise NotImplementedError`
- Python bare `except:` and swallowed exceptions (`except ...: pass`)
- Hardcoded credentials matching well-known prefixed shapes (OpenAI `sk-`,
  AWS `AKIA`, GitHub `ghp_`, Slack `xox*-`)

Per-file cap of 8 + global cap of 20 findings keeps the prompt bounded.
Magic numbers and generic high-entropy secret detection are intentionally
out of scope — they'd add more noise than signal at this stage.


### Smart File Selection

**There is one retrieval path.** Every invocation — flagged or not, on a fresh clone or a warm one — goes through the same four-stage front door in `gather_context`. No flag routes around it, and there is nothing to build first.

1. **Paths the prompt named** — pinned. A path spelled out in the prompt carries `EXPLICIT_PATH_BOOST` (10.0), chosen to exceed every organic signal combined, so it cannot lose to a ranking.
2. **`--include` patterns** — pinned too, and the scan keeps running alongside them rather than narrowing to them. A named file arrives whole, or with an explicit truncation marker; never silently dropped.
3. **Keyword** — BM25 over file *content*, weighted `CONTENT_WEIGHT` (3.0), served from a persistent per-repo index that refreshes itself incrementally on every call.
4. **Semantic** — the embedding catalog, as a **re-rank and supplement** of stage 3 whenever it exists. It can raise a file or surface one stage 3 never scored; it can never remove a file stage 3 found, because the catalog is a snapshot and a stale snapshot must not suppress a live hit.

Stages 1 and 2 assert presence; 3 and 4 rank what is left. An assertion cannot lose to a score.

Two further signals ride on the ranking stages: **tree-sitter symbol overlap** (function/class names + imports from top candidates, up to +1.2 for substring matches against prompt tokens, length-3 floor) and the **EPISODE-history feedback loop** (each run stashes touched paths as `file:<rel>` tags on EPISODE facts; a later similar prompt gives those files up to +0.5). Test-file matches are demoted 0.4× unless the prompt is itself about tests. Delivery is one entry per file, read whole from disk, with `--max-bytes` apportioned max-min fair across the selection.

**What `--index` is, and is not.** Stages 1–3 need nothing built: the eligibility walk and the keyword index are cached in the repo's `.neo/` and brought up to date inline on every invocation, so removing `--index` from a fresh-clone workflow changes first-call latency and nothing else. `--index` builds the **embedding catalog** that stage 4 reads — tree-sitter chunks embedding `symbols + imports + first ~600 chars of body`, over FAISS, at `.neo/index.json`. Once it exists it is consulted on every run with no flag; `--semantic` only reads it deeper (3×) and raises its weight to `CONTENT_WEIGHT`. Without a catalog, stage 4 contributes nothing and a one-line tip fires.

The catalog build apportions its budget rather than truncating whatever globbed first: files are grouped by language and given a share of `--max-files` proportional to how much of the repo they are, with a floor of one slot per language, and the `MAX_CHUNKS_PER_REPO` cut is apportioned across files in proportion to what each holds, with a floor. (It was round-robin once — one chunk each before any file gets a second — and that was replaced because equal shares are not fair shares: a 9 KB utility came out fully represented while the modules the repo is built on did not.) Non-source paths (`.worktrees`, `node_modules`, `.claude`, virtualenvs, plus anything the repo's own `.gitignore` names) are excluded, and byte-identical duplicates are indexed once. When a cap bites, `neo --index` says so — a capped build no longer looks the same as a complete one. See [tree-sitter setup](docs/tree-sitter-setup.md#operational-notes).


### Learning Feedback Loop

After each Neo run, the next invocation diffs your repo against the suggestions it made and classifies the result. All confidence deltas are modulated by `±arch_mod` (∈ {−0.1, 0, +0.1}) from the architectural-quality snapshot — see [Architectural Quality Feedback Loop](#architectural-quality-feedback-loop) below.

| Outcome     | Trigger                                                                  | Effect                                                                              |
|-------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| ACCEPTED    | Code-block overlap ≥ 0.8 (modern path) or unified-diff overlap > 0.3 (legacy path) | linked fact conf +0.2 ± arch_mod, success_count +1, effectiveness "better"          |
| MODIFIED    | User changed the file differently                                        | linked fact conf −0.2 ± arch_mod (floored at 0.1) + new REVIEW at conf 0.4          |
| REGRESSION  | Later evidence identifies an accepted suggestion by episode + suggestion ID | derived fact conf −0.2; repeated independent contradictions roll it back          |
| UNVERIFIED  | File touched but suggestion had no diff to compare                       | evidence retained in the learning episode; no confidence or success mutation         |
| INDEPENDENT | File touched, never suggested by Neo                                     | new REVIEW at conf 0.2; capped 5/session, 50/project                                |


### Storage Architecture

- **Scoped JSON Files**: Facts stored in `~/.neo/facts/` — separate files per scope (global, org, project), with inline embeddings (no separate FAISS index for memory).
- **Bi-Temporal Supersession**: similar facts are soft-deleted by stamping `event_time_end` rather than dropped. Tombstones persist until `purge_dead_facts` runs on the next cold start.
- **Constraint Auto-Ingestion**: CLAUDE.md and similar files are automatically scanned and ingested as CONSTRAINT facts.
- **Sessions & Metrics**: `~/.neo/sessions/` holds session manifests + replay logs; `~/.neo/metrics.jsonl` logs every retrieve / add_fact / lm_call / overseer_tick (disable with `NEO_METRICS=off`).
- **Project Index** (separate system): Tree-sitter code indexing uses FAISS for per-repository semantic search in `.neo/`.


## Performance

**Neo improves over time as it learns from experience.** Initial performance depends on available facts. Performance grows as the semantic memory builds up successful solutions, failure learnings, and architectural decisions.

### Memory-Driven Reasoning Effort (gpt-5* models)

Neo monetizes its learning into inference cost. Each query's `reasoning.effort`
parameter is sized from the strength of the memory hit:

| Memory + difficulty                              | Effort  |
|--------------------------------------------------|---------|
| ≥3 patterns, avg confidence ≥ 0.8                | `low`   |
| Some patterns, avg confidence 0.5–0.8            | `medium` (API default) |
| No relevant patterns OR avg confidence < 0.5     | `high`  |
| No patterns AND difficulty == "hard"             | `xhigh` |

Familiar queries get cheap thinking; novel-and-hard queries get max thinking.
Cap with `NEO_REASONING_EFFORT={none,low,medium,high,xhigh}` for cost control.

> **Model note:** the effort vocabulary differs by model. gpt-5.6 (the default)
> accepts the full `none / low / medium / high / xhigh` range. Older
> `gpt-5-codex` only accepts `low / medium / high` — if you switch back to
> that model, set `NEO_REASONING_EFFORT=high` to cap the auto-selector.

### Architectural Quality Feedback Loop

When a session ends, neo snapshots three structural metrics — import cycles,
god files (LOC + function-count thresholds), and max nesting depth — and
diffs against the previous snapshot at the next outcome detection. A
regression weakens the accept/boost or strengthens the modify/penalty by
0.1; an improvement does the reverse. Confidence becomes a signal of
"helped the codebase," not just "got accepted."


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

**Exposed Configuration Fields** (the only keys `--config get/set` accepts):
- `provider` - LM provider: `openai` (default), `anthropic`, `google`, `azure`, `ollama`, `local`, `claude-code`, `car`
- `model` - Model name (default `gpt-5.6`; e.g. `claude-sonnet-4-5-20250929`). Pass an empty value (`--config-value ""`) to clear it and let the provider or CAR's router choose
- `api_key` - API key for the chosen provider
- `base_url` - Base URL for local/Ollama endpoints (also clearable with an empty value)
- `inference_mode` - `static` (default) or `auto` — see [Outbound: use CAR as Neo's inference layer](#outbound-use-car-as-neos-inference-layer)
- `memory_backend` - Memory backend: `fact_store` (default) or `legacy`
- `auto_install_updates` - Automatically install updates in background (true/false)
- `constraint_auto_scan` - Auto-scan CLAUDE.md for constraints (true/false, default: true)
- `log_level` - Logging level: DEBUG, INFO, WARNING, or ERROR
- `reasoning_effort_cap` - Optional cap for OpenAI gpt-5 reasoning effort

**Other fields in `~/.neo/config.json`** — real settings, but not reachable
through `--config set`; edit the file or use the environment variable:
- `reasoning_mode` - `auto` (default), `fast`, or `deep`. Per-run equivalents: `--fast` / `--deep`
- `default_temperature` / `default_max_tokens` - Env: `NEO_TEMPERATURE` / `NEO_MAX_TOKENS`
- `exemplar_dir` - Env: `NEO_EXEMPLAR_DIR`
- `enable_ruff` / `enable_pyright` / `enable_mypy` / `enable_eslint` - static-analysis toggles
- `safe_read_patterns` / `forbidden_paths` - file allowlist and blocklist

Configuration is stored in `~/.neo/config.json`. Environment variables override
stored config values for the current process.

### Secure API Key Storage

On macOS, Neo stores API keys in **Keychain** rather than `config.json`. Run:

```bash
# Securely prompt for and store an API key in Keychain
neo --config set --config-key api_key
```

`NeoConfig.load()` reads the Keychain entry automatically.

**Linux / Windows**: this command currently raises — Keychain support is macOS-only. Either set the provider env var directly (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) or export `NEO_ALLOW_PLAINTEXT_API_KEY=1` first so the key is persisted in `config.json`.


### Environment Variables

**Credentials**

```bash
# Provider-specific (read by NeoConfig.load() when set)
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=...

# Neo-generic override (takes precedence over provider-specific keys)
export NEO_PROVIDER=openai
export NEO_MODEL=gpt-5.6
export NEO_API_KEY=sk-...
export NEO_BASE_URL=http://localhost:11434       # for Ollama/local endpoints
```

**Behavior**

```bash
export NEO_INFERENCE_MODE=auto                    # prefer CAR's router, fall back to the static provider
export NEO_REASONING_EFFORT=high                  # cap auto-effort selection
export NEO_AUTO_INSTALL_UPDATES=1                 # auto-install background updates
export NEO_SKIP_UPDATE_CHECK=1                    # disable update checks entirely
export NEO_LOG_LEVEL=INFO                         # DEBUG/INFO/WARNING/ERROR
export NEO_TEMPERATURE=0.7                        # generation temperature
export NEO_MAX_TOKENS=4096                        # per-call max output tokens
export NEO_EXEMPLAR_DIR=/path/to/exemplars        # override the exemplar store location
export NEO_FASTEMBED_CACHE_DIR=/path/to/cache     # Jina model cache (default ~/.cache/neo/fastembed)
export NEO_STDIN_TIMEOUT_SECONDS=5                # wait for stdin to be readable (default 1.0)
export NEO_CAR_TIMEOUT_SECONDS=300                # per-call CAR watchdog deadline (default 240)
export NEO_ALLOW_PLAINTEXT_API_KEY=1              # permit storing api_key in config.json (see above)
```

**Background observer**

```bash
export NEO_OBSERVER_AUTOSTART=0                   # do not autostart the observer
export NEO_OBSERVER_INTERVAL_SECONDS=300          # sweep interval
export NEO_OBSERVER_COOLDOWN=60                   # per-process cooldown between cycles
export NEO_OBSERVER_RECYCLE_CYCLES=48             # re-exec after N cycles to bound RSS (0 disables)
```

### Install Sanity

If you have multiple local installs, make sure the `neo` command and your test
interpreter import the same package:

```bash
which neo
neo --version
python3 -c "import neo; print(neo.__file__)"
```

**Observability**

```bash
export NEO_PROFILE=standard                       # off | minimal | standard (default) | strict
export NEO_METRICS=off                            # hard kill-switch; overrides NEO_PROFILE
```

Neo writes structured per-operation events (retrieve / add_fact / lm_call / overseer_tick) to `~/.neo/metrics.jsonl` and per-session manifests + JSONL outcome logs to `~/.neo/sessions/`.

`NEO_PROFILE` selects which events are emitted: `off` emits nothing, `minimal`
emits only `lm_call`, `standard` emits everything, and `strict` is reserved for
future verbose events (identical to `standard` today). `NEO_METRICS=off` (or
`0`/`false`/`no`) is the legacy hard kill-switch and wins over `NEO_PROFILE`.

The log rotates to `metrics.jsonl.1` at 32 MB with one generation retained. The
readers (`memory citation-stats`, `memory learning-stats`) window by `--since`
and read only the active file — a `--since` older than the last rotation
silently sees less history.

## LM Adapters

### OpenAI (Default)

```python
from neo.adapters import OpenAIAdapter
adapter = OpenAIAdapter(model="gpt-5.6", api_key="sk-...")
```

Neo's configured default model is `gpt-5.6` (`NeoConfig.model`), which is what
the CLI uses. Constructing an adapter directly without a `model` falls back to
the adapter's own default of `gpt-4`, so pass the model explicitly when you
bypass `NeoConfig`. GPT-5/Codex models use the `/v1/responses` endpoint
automatically.

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

### CAR (Common Agent Runtime)

```python
from neo.adapters import CarAdapter
# Default: router-picked model with intent_json={"task": "code"} so the
# router selects a code-capable backend rather than the chat default.
adapter = CarAdapter()
# pin a specific backend if you need to:
adapter = CarAdapter(model="Qwen3-4B")
# override the default intent (CAR task enum: chat | classify | reasoning | code):
adapter = CarAdapter(intent_hint={"task": "reasoning", "prefer_local": True})
```

Requires the `[car]` extra (`pip install neo-reasoner[car]`) and a running `car-server`. See [Run as an Agent (CAR / A2A)](#run-as-an-agent-car--a2a) for the full setup.


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

- **Three integration surfaces on equal footing**:
  - **Run as an Agent (CAR / A2A)** — host Neo as an Agent2Agent v1.0 endpoint via `neo serve`; other agents call `neo.process` directly
  - **Claude Code Plugin** — six slash commands + a specialized agent inside Claude Code
  - **Codex Plugin** — the same six skills, packaged for OpenAI Codex CLI
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

The 0.18 memory architecture lands deterministic techniques from a focused reading of recent work on long-horizon agent memory and code generation. Citations below are anchored to the file where the technique is actually implemented — the full PDFs are checked into [`papers/`](papers/) for reproducibility.

### Academic Papers

**Memory architecture & lifecycle**

1. **SCM Sleep Memory: Sleep-Consolidation in Continual Memory**
   *Paper [2604.20943](https://arxiv.org/abs/2604.20943)*
   - 4-D ValueTagger composite (novelty, validation, task, repetition); adaptive forgetting threshold.
   - **Implementation**: `src/neo/memory/value_score.py`.
   - *Partially reverted*: the NREM Hebbian strengthening, global downscale and
     triple-trigger consolidation gate lived in `store.synthesize_reviews`, which
     was removed — in four months of production it minted 114 facts and not one
     PATTERN, while decaying the whole corpus on every run. See the CHANGELOG.

2. **Memory Systems Survey (1)**
   *Paper [2603.07670](https://arxiv.org/abs/2603.07670)*
   - Provenance taxonomy (`STRUCTURAL > OBSERVED > INFERRED`); dual-buffer / probation consolidation; Layer-1/2/3 observability split.
   - **Implementation**: `Provenance` in `src/neo/memory/models.py`, `store.py` (probation tag), `memory/metrics.py`.

3. **Memory Survey 2 — Zep / AriGraph bi-temporal pattern**
   *Paper [2512.13564](https://arxiv.org/abs/2512.13564) §5.2.2*
   - Bi-temporal stamps (`event_time` / `event_time_end` / `ingest_time`); supersession via soft-delete.
   - **Implementation**: the `event_time` / `event_time_end` / `ingest_time` fields on `FactMetadata` in `src/neo/memory/models.py`.

4. **Trajectory Memory — Canonical-signature dedup**
   *Paper [2603.10600](https://arxiv.org/abs/2603.10600) §7*
   - Entity abstraction + verb-synonym folding + context strip as a pre-write dedup signature.
   - **Implementation**: `src/neo/memory/generalize.py`.

5. **Memori — Hybrid dense+BM25 retrieval**
   *Paper [2603.19935](https://arxiv.org/abs/2603.19935) §3.3*
   - Sparse BM25 channel (k1=1.5, b=0.75) min-max-normalized and weighted with the dense channel at 0.7/0.3.
   - **Implementation**: `src/neo/memory/bm25.py` (sparse channel), `store._fuse_dense_sparse` (0.7/0.3 fusion).

6. **MemMachine — Query-shape routing & nucleus episode expansion**
   *Paper [2604.04853](https://arxiv.org/abs/2604.04853) §4.6, §5.3, §8.4.1*
   - DIRECT / CHAIN / SPLIT prompt classification with per-branch retrieval; episode-peer expansion at retrieval time; k=20–30 sweet spot.
   - **Implementation**: `src/neo/memory/query_routing.py`, `store.py` nucleus expansion.

7. **LessonL — Effectiveness multiplier on reuse outcomes**
   *Paper [2505.23946](https://arxiv.org/abs/2505.23946)*
   - Per-fact `c/n` effectiveness as a success-bonus multiplier; half-by-rank / half-by-cosine slot allocation (Algorithm 1).
   - **Implementation**: `EFFECTIVENESS_EPSILON` and `FactMetadata.effectiveness_f` in `src/neo/memory/models.py`; `store.retrieve_relevant`.

8. **Ebbinghaus Recall — Spaced-repetition decay for retrieval**
   *Hou et al., paper [2404.00573](https://arxiv.org/abs/2404.00573)*
   - Recall-probability transform `p_n(t) = (1 − exp(−r·exp(−t/g_n))) / (1 − e⁻¹)` applied to similarity scores for fluid facts.
   - **Implementation**: `math_utils.recall_probability`, `models.rank_score`.

9. **Episodic Memory — Five-property episodic context**
   *Paper [2502.06975](https://arxiv.org/abs/2502.06975) Table 1*
   - `{when, where, why, with_whom}` instance-specific event context.
   - **Implementation**: `EpisodeContext` in `src/neo/memory/models.py`.

10. **Multiple Memory Systems — Retrieval / context unit split**
    *Paper [2508.15294](https://arxiv.org/abs/2508.15294) §3*
    - Embed concise keywords (`retrieval_text`); inject full narrative (`context_text`) — same fact, two surfaces.
    - **Implementation**: the `retrieval_text` / `context_text` split on `Fact` in `src/neo/memory/models.py`.

**Engine & multi-agent reasoning**

11. **MapCoder — Solver–Critic–Verifier multi-agent collaboration**
    *Islam et al., paper [2405.11403](https://arxiv.org/abs/2405.11403)* | [GitHub](https://github.com/Md-Ashraful-Pramanik/MapCoder)
    - Per-step confidence, multi-plan iteration scaffolding.
    - **Implementation**: `PlanStep.confidence` in `src/neo/models.py`.

12. **CodeSim — MODIFY / NO_MODIFY decision token**
    *Hou et al., paper [2502.05664](https://arxiv.org/abs/2502.05664)*
    - Simulator emits an explicit "no modification needed" token; planner uses it as an override on the agreement-of-outputs heuristic. (Distinct from the 2023 ACM CodeSim paper of the same name.)
    - **Implementation**: `NeoEngine._simulation_consensus` / `_extract_plan_decision` in `src/neo/engine.py`.

13. **SICA — Asynchronous structured-output watchdog & cache-hit observability**
    *Paper [2504.15228](https://arxiv.org/abs/2504.15228) §A.2, Table 1*
    - Daemon-thread tick loop emitting `overseer_tick` events; loop detection via 5-identical-actions-in-a-row; LM-call cache-hit-rate tracking.
    - **Implementation**: `src/neo/overseer.py`; cache-hit-rate tracking in `src/neo/adapters.py`.

**In-house papers (Parslee)** — the foundational research behind Neo's context architecture

- **Beyond Conversation: A State-Based Context Architecture for Enterprise AI Agents**
  *Liotta, 2025* | [PDF](papers/state-based-context-architecture.pdf)
  - The theoretical foundations for the four-layer state model (constraints / valid facts / invalidated facts / known unknowns), supersession semantics, and the six classes of state failure (resurrection, hallucination, scope leak, stale reasoning, authority violation, temporal decay).
  - **Implementation**: `src/neo/memory/context.py`, `ContextResult` in `src/neo/memory/models.py`.

- **Memgine: A Deterministic Memory Engine for Stateful AI Agents**
  *Liotta, 2026* | [PDF](papers/memgine-deterministic-memory-engine.pdf)
  - The production engine implementing the full specification: query-relevance sorting, engine-level access control, adaptive inline repair, layered token-budget enforcement (2/3 constraint cap, greedy first-fit accumulation with `at_least_one`, `Fact.size_hint()` heuristic). Achieves 95.8% decision accuracy on the StateBench v1.0 development split with GPT-5.2 (97.3% with Opus 4.6).
  - **Implementation**: `_accumulate_within_budget` in `src/neo/memory/context.py`; `Fact.size_hint()` in `src/neo/memory/models.py`; design notes in `docs/solutions/token-budget-enforcement.md`.

- **StateBench** — [github.com/parslee-ai/statebench](https://github.com/parslee-ai/statebench) · [parslee-ai.github.io/statebench](https://parslee-ai.github.io/statebench/)
  - The conformance test suite that evaluates the two papers above. PyPI / HuggingFace Dataset / Space. Reference baselines (`state_based`, `rolling_summary`, `fact_extraction_with_supersession`, etc.) on the v1.0 test split set the bar Neo's port is measured against.

**Background reading (in [`papers/`](papers/) but not directly cited in code)**

The following papers shaped the design vocabulary but aren't wired into a specific implementation today: 2506.18902 (Jina v4 — Neo currently uses Jina v2), 2508.21290 (Jina Code Embeddings), 2509.17489 (MapCoder-Lite), 2511.20857 (Evo-Memory).

**Historical influences** (cited in legacy modules under deprecation): ReasoningBank ([2509.25140](https://arxiv.org/abs/2509.25140)) informed the original `src/neo/persistent_reasoning.py`; the 0.18 fact store supersedes it.

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
  note={Memory architecture draws on SCM Sleep Memory (2604.20943), MemMachine (2604.04853), LessonL (2505.23946), and the bi-temporal/Ebbinghaus/dual-buffer techniques cataloged in the README's Research \& References section}
}
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
