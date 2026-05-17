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
- [Codex Plugin](#codex-plugin)
- [Works Alongside Your AI Tools](#works-alongside-your-ai-tools)
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

Plugin sources live under [`.claude-plugin/`](.claude-plugin/) — `plugin.json` is the manifest, `agents/neo.md` defines the agent, and `commands/*.md` defines each slash command.


## Codex Plugin

Neo also ships as a **Codex plugin** with the same six skills the Claude Code plugin exposes — packaged for [OpenAI Codex CLI](https://developers.openai.com/codex/plugins) instead of slash commands.

```bash
# Add Neo's local marketplace (works in any clone of this repo)
codex plugin marketplace add Parslee-ai/neo

# Or, from a local checkout, point Codex at the in-tree marketplace:
codex plugin marketplace add ./
```

Then open Codex's plugin directory and install **Neo** from the `Neo (local)` marketplace. Once installed, you get six skills:

- `$neo` — semantic reasoning over the current codebase
- `$neo-review` — code review with semantic pattern matching
- `$neo-optimize` — performance/algorithmic optimization analysis
- `$neo-architect` — architectural guidance and design decisions
- `$neo-debug` — help debugging intermittent or hard-to-reproduce issues
- `$neo-pattern` — extract patterns from code or find pattern instances

Skills wrap the local `neo` CLI, so you still need the binary installed (`pip install neo-reasoner[openai]` and `OPENAI_API_KEY` set). Neo's persistent semantic memory in `~/.neo/` is shared across both plugins — anything you teach Neo from Claude Code is available from Codex too, and vice versa.

**See [plugins/neo/](plugins/neo/) for the manifest and skill sources**


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

⚡ Auto-installing neo update: 0.18.0 → 0.18.1
   This happens in the background. Please wait...

✓ Auto-update completed: 0.18.1
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

# Build the per-project semantic index (powers smart file selection)
neo --index

# Incrementally refresh the index after meaningful changes (re-embeds only changed files)
neo --update

# Preview the assembled context without making an LLM call
neo --dry-run "your query"

# Check version and memory stats
neo --version

# Inspect detected local CAR runtime surfaces
neo car status
```


### Memory Maintenance

```bash
# Compact fact files by dropping old invalid tombstones (default: > 30 days since last access)
neo memory prune

# Across every local project Neo has touched
neo memory prune --all

# Preview without writing
neo memory prune --dry-run --max-invalid-age-days 14
```

Use `prune` when a `~/.neo/facts/facts_project_*.json` file grows much larger than its 500-valid-fact cap — that gap is tombstone bloat from supersession. Defaults are conservative; raising `--max-invalid-age-days` is safe, lowering it past ~7 may evict tombstones still referenced by recent supersession chains.


### CAR Runtime Discovery

Neo detects local CAR installs across the native CLI, `car-server`, Python
bindings, and the default daemon port:

```bash
neo car status
neo --version
```

If the CLI/server are present but Python bindings are not, Neo reports that
state and keeps `neo serve` on the explicit `[car]` path. CAR install options
are documented at [Parslee-ai/car-releases](https://github.com/Parslee-ai/car-releases).


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

neo 0.18.1
Provider: openai | Model: gpt-5.5
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

1. **Typed Facts**: Eight kinds — CONSTRAINT, ARCHITECTURE, DECISION, PATTERN, REVIEW, FAILURE, KNOWN_UNKNOWN, and EPISODE (instance-specific events with `{when, where, why, with_whom}` context).
2. **Scoped Organization**: Facts are scoped to global, organization, or project level, with per-scope valid-fact caps (200 / 100 / 500 / 50). Org and project are auto-detected from git remotes.
3. **Supersession & Pre-Write Dedup**: New facts with cosine similarity ≥ 0.85 to an existing fact short-circuit (bump the existing fact's access count) or supersede it. The pre-write canonical-signature check uses entity abstraction + verb-synonym folding to catch near-duplicates before they hit the store.
4. **Confidence + Effectiveness Ranking**: `rank_score = recall_decay(sim)·confidence + success_bonus·effectiveness_f + provenance_bonus`. The Ebbinghaus recall-probability transform gives frequently-recalled facts slower decay; LessonL-style effectiveness (`c/n` over reuse outcomes) multiplies the success bonus. Curated facts (CONSTRAINT/ARCHITECTURE/DECISION and `seed`/`community`/`synthesized`-tagged facts) bypass decay.
5. **Hybrid Retrieval**: 0.7·dense (Jina) + 0.3·BM25. Half the result slots ranked by full `rank_score`, half by raw cosine — novel-but-relevant facts aren't crowded out by validated winners.
6. **Triple-Trigger Consolidation**: REVIEW facts cluster into PATTERN / FAILURE archetypes when ANY of count-delta ≥10, elapsed ≥1h, or confidence-decile entropy >0.9 fires. Clusters of ≥3 get an NREM-style Hebbian confidence bump; non-curated facts decay 3% globally after each pass.
7. **Dual-Buffer Probation**: New non-curated facts enter with a `probation` tag and a 3-day stale window (vs 7/14 normal); promoted automatically on `access_count ≥ 2` or `success_count > 0` — quietly evicts noise while keeping real signal.
8. **Four-Layer Context**: Retrieved facts are organized into constraints, relevant knowledge, recent changes, and known unknowns. The four-layer state model and the token-budget enforcement in `memory/context.py` are both adapted from Parslee's [**StateBench**](https://github.com/parslee-ai/statebench) (a conformance bench for stateful agents) and its **memgine** budget engine. StateBench's measured 95.8% decision-accuracy result is what drove the move from a separate "Recently Changed" section to inline `(changed from: X)` annotations.

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

The context gatherer picks files using three signals:

- **ProjectIndex semantic boost**: when `.neo/index.json` exists (run `neo --index` once per repo), per-project FAISS over tree-sitter chunks projects top-k chunk hits back to per-file boosts up to +1.0 cosine. Chunks embed `symbols + imports + first ~600 chars of body`, so prompt keywords match what a file *is*, not assertion strings inside tests. Test-file matches are demoted 0.4× unless the prompt mentions test/spec.
- **Tree-sitter symbol overlap**: the parser extracts function/class names + imports from top candidates and adds up to +1.2 for substring matches against prompt tokens (length-3 floor).
- **EPISODE-history feedback loop**: each Neo run stashes touched file paths as `file:<rel>` tags on EPISODE facts. On the next run, the gatherer queries for similar past prompts and gives those files up to +0.5 boost — past file selections measurably influence future ones.

A per-file chunk cap of 2 prevents large files from eating the budget; a one-time first-run hint fires if the index is missing.


### Learning Feedback Loop

After each Neo run, the next invocation diffs your repo against the suggestions it made and classifies the result. All confidence deltas are modulated by `±arch_mod` (∈ {−0.1, 0, +0.1}) from the architectural-quality snapshot — see [Architectural Quality Feedback Loop](#architectural-quality-feedback-loop) below.

| Outcome     | Trigger                                                                  | Effect                                                                              |
|-------------|--------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| ACCEPTED    | Code-block overlap ≥ 0.8 (modern path) or unified-diff overlap > 0.3 (legacy path) | linked fact conf +0.2 ± arch_mod, success_count +1, effectiveness "better"          |
| MODIFIED    | User changed the file differently                                        | linked fact conf −0.2 ± arch_mod (floored at 0.1) + new REVIEW at conf 0.4          |
| UNVERIFIED  | File touched but suggestion had no diff to compare                       | linked fact conf +0.1 ± arch_mod, success_count +1 (no REVIEW)                      |
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

> **Model note:** the effort vocabulary differs by model. gpt-5.5 (the default)
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

**Exposed Configuration Fields:**
- `provider` - LM provider (openai, anthropic, google, azure, ollama, local)
- `model` - Model name (e.g., gpt-5.5, claude-sonnet-4-5-20250929)
- `api_key` - API key for the chosen provider
- `base_url` - Base URL for local/Ollama endpoints
- `memory_backend` - Memory backend: "fact_store" (default) or "legacy"
- `auto_install_updates` - Automatically install updates in background (true/false)
- `constraint_auto_scan` - Auto-scan CLAUDE.md for constraints (true/false, default: true)
- `reasoning_effort_cap` - Optional cap for OpenAI gpt-5 reasoning effort

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

# Neo-generic (provider-specific keys take precedence)
export NEO_PROVIDER=openai
export NEO_MODEL=gpt-5.5
export NEO_API_KEY=sk-...
export NEO_BASE_URL=http://localhost:11434       # for Ollama/local endpoints
```

**Behavior**

```bash
export NEO_REASONING_EFFORT=high                  # cap auto-effort selection
export NEO_AUTO_INSTALL_UPDATES=1                 # auto-install background updates
export NEO_SKIP_UPDATE_CHECK=1                    # disable update checks entirely
export NEO_LOG_LEVEL=INFO                         # DEBUG/INFO/WARNING/ERROR
export NEO_TEMPERATURE=0.7                        # generation temperature
export NEO_MAX_TOKENS=4096                        # per-call max output tokens
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
export NEO_METRICS=off                            # disable ~/.neo/metrics.jsonl writes
```

Neo writes structured per-operation events (retrieve / add_fact / lm_call / overseer_tick) to `~/.neo/metrics.jsonl` and per-session manifests + JSONL outcome logs to `~/.neo/sessions/`.

## LM Adapters

### OpenAI (Default)

```python
from neo.adapters import OpenAIAdapter
adapter = OpenAIAdapter(model="gpt-5.5", api_key="sk-...")
```

Default model: `gpt-5.5`. GPT-5/Codex models use the `/v1/responses` endpoint automatically.

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

The 0.18 memory architecture lands deterministic techniques from a focused reading of recent work on long-horizon agent memory and code generation. Citations below are anchored to the file where the technique is actually implemented — the full PDFs are checked into [`papers/`](papers/) for reproducibility.

### Academic Papers

**Memory architecture & lifecycle**

1. **SCM Sleep Memory: Sleep-Consolidation in Continual Memory**
   *Paper [2604.20943](https://arxiv.org/abs/2604.20943)*
   - 4-D ValueTagger composite (novelty, validation, task, repetition); adaptive forgetting threshold; NREM Hebbian strengthening + global downscale; triple-trigger consolidation gate.
   - **Implementation**: `src/neo/memory/value_score.py`, `store.synthesize_reviews`.

2. **Memory Systems Survey (1)**
   *Paper [2603.07670](https://arxiv.org/abs/2603.07670)*
   - Provenance taxonomy (`STRUCTURAL > OBSERVED > INFERRED`); dual-buffer / probation consolidation; Layer-1/2/3 observability split.
   - **Implementation**: `src/neo/memory/models.py:42`, `store.py` (probation tag), `memory/metrics.py`.

3. **Memory Survey 2 — Zep / AriGraph bi-temporal pattern**
   *Paper [2512.13564](https://arxiv.org/abs/2512.13564) §5.2.2*
   - Bi-temporal stamps (`event_time` / `event_time_end` / `ingest_time`); supersession via soft-delete.
   - **Implementation**: `src/neo/memory/models.py:241`.

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
   - **Implementation**: `src/neo/memory/models.py:130, 233`; `store.retrieve_relevant`.

8. **Ebbinghaus Recall — Spaced-repetition decay for retrieval**
   *Hou et al., paper [2404.00573](https://arxiv.org/abs/2404.00573)*
   - Recall-probability transform `p_n(t) = (1 − exp(−r·exp(−t/g_n))) / (1 − e⁻¹)` applied to similarity scores for fluid facts.
   - **Implementation**: `src/neo/math_utils.py:40`, `models.rank_score`.

9. **Episodic Memory — Five-property episodic context**
   *Paper [2502.06975](https://arxiv.org/abs/2502.06975) Table 1*
   - `{when, where, why, with_whom}` instance-specific event context.
   - **Implementation**: `src/neo/memory/models.py:320` `EpisodeContext`.

10. **Multiple Memory Systems — Retrieval / context unit split**
    *Paper [2508.15294](https://arxiv.org/abs/2508.15294) §3*
    - Embed concise keywords (`retrieval_text`); inject full narrative (`context_text`) — same fact, two surfaces.
    - **Implementation**: `src/neo/memory/models.py:373`.

**Engine & multi-agent reasoning**

11. **MapCoder — Solver–Critic–Verifier multi-agent collaboration**
    *Islam et al., paper [2405.11403](https://arxiv.org/abs/2405.11403)* | [GitHub](https://github.com/Md-Ashraful-Pramanik/MapCoder)
    - Per-step confidence, multi-plan iteration scaffolding.
    - **Implementation**: `PlanStep.confidence` in `src/neo/models.py`.

12. **CodeSim — MODIFY / NO_MODIFY decision token**
    *Hou et al., paper [2502.05664](https://arxiv.org/abs/2502.05664)*
    - Simulator emits an explicit "no modification needed" token; planner uses it as an override on the agreement-of-outputs heuristic. (Distinct from the 2023 ACM CodeSim paper of the same name.)
    - **Implementation**: `src/neo/engine.py:427`.

13. **SICA — Asynchronous structured-output watchdog & cache-hit observability**
    *Paper [2504.15228](https://arxiv.org/abs/2504.15228) §A.2, Table 1*
    - Daemon-thread tick loop emitting `overseer_tick` events; loop detection via 5-identical-actions-in-a-row; LM-call cache-hit-rate tracking.
    - **Implementation**: `src/neo/overseer.py`, `src/neo/adapters.py:237`.

**In-house benchmarks & engines (Parslee)**

- **StateBench** — [github.com/parslee-ai/statebench](https://github.com/parslee-ai/statebench) · [parslee-ai.github.io/statebench](https://parslee-ai.github.io/statebench/) — a conformance test for stateful agents that measures state correctness over time. Neo's four-layer context model (constraints / valid facts / invalidated facts / known unknowns) and its layer-ordering heuristic are adapted from StateBench's winning approach. The 95.8% decision-accuracy result on inline change annotations is the validation behind Neo's `(changed from: X)` formatting.
  - **Implementation**: `src/neo/memory/context.py`, `src/neo/memory/models.py` `ContextResult`.
- **memgine** — the layered-budget engine inside StateBench. Neo's `ContextAssembler.assemble()` token-budget enforcement (2/3 constraint cap, greedy first-fit accumulation with `at_least_one`, `Fact.size_hint()` heuristic) is a direct port — see `docs/solutions/token-budget-enforcement.md` for the full attribution and design notes.
  - **Implementation**: `_accumulate_within_budget` in `src/neo/memory/context.py`; `Fact.size_hint()` in `src/neo/memory/models.py`.

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
