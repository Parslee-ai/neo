# Neo Quick Start

Get Neo running in 5 minutes.

## 1. Install Neo

```bash
pip install neo-reasoner[openai]
```

This installs Neo with OpenAI (GPT) support. Alternatively:
- `pip install neo-reasoner[anthropic]` for Claude
- `pip install neo-reasoner[google]` for Gemini
- `pip install neo-reasoner[all]` for all providers

## 2. Set API Key

```bash
export OPENAI_API_KEY=sk-your-key-here
```

Add to `~/.bashrc` or `~/.zshrc` for persistence.

## 3. Test Neo

```bash
neo --version
```

Expected output:
```
"What is real? How do you define 'real'?"

neo 0.18.1
Provider: openai | Model: gpt-5.5
Stage: Sleeper | Memory: 0.0%
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0 facts | 0.00 avg confidence
```

## 4. Try a Simple Query

```bash
neo "write a function to check if a number is prime"
```

Expected: Neo will analyze the request, plan the solution, and provide code suggestions with confidence scores.

## 5. Use from Python

```python
from neo.cli import NeoEngine, NeoInput, TaskType
from neo.adapters import create_adapter

# Create adapter (matches the OPENAI_API_KEY set in step 2)
adapter = create_adapter("openai")

# Create engine
engine = NeoEngine(lm_adapter=adapter)

# Process request
output = engine.process(NeoInput(
    prompt="Fix the division by zero bug",
    task_type=TaskType.BUGFIX,
    error_trace="ZeroDivisionError: division by zero",
))

# Review output
print(f"Confidence: {output.confidence:.0%}")
for step in output.plan:
    print(f"- {step.description}")
```

## 6. Use Neo from Your AI CLI (Optional)

Neo ships as a plugin for both major AI coding CLIs. Pick whichever you use — the same six commands are available in each, and the fact store is shared.

**Claude Code:**

```bash
/plugin marketplace add Parslee-ai/claude-code-plugins
/plugin install neo
```

Then: `/neo`, `/neo-review`, `/neo-optimize`, `/neo-architect`, `/neo-debug`, `/neo-pattern`.

**OpenAI Codex CLI:**

```bash
codex plugin marketplace add Parslee-ai/neo
# then install "Neo" from Codex's plugin directory
```

Then: `$neo`, `$neo-review`, `$neo-optimize`, `$neo-architect`, `$neo-debug`, `$neo-pattern`.

**Important:** Both plugins require the Neo CLI (step 1) installed and an API key (step 2) set.

## 7. Host Neo as an Agent (Optional)

For orchestrators and other agents that speak [Agent2Agent v1.0](https://github.com/Parslee-ai/car-releases), Neo can run as a CAR-backed A2A endpoint instead of a CLI:

```bash
pip install "neo-reasoner[car]"
python -m car_runtime.server &   # CAR daemon
neo serve                         # Neo as an A2A tool
```

This is the inference path other agents call directly — see [Run as an Agent (CAR / A2A)](README.md#run-as-an-agent-car--a2a) in the README for the full setup.

## Next Steps

- **Full Documentation**: See [README.md](README.md)
- **Installation Guide**: See [INSTALL.md](INSTALL.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Plugin Sources**: [`.claude-plugin/`](.claude-plugin/) (Claude Code) · [`plugins/neo/`](plugins/neo/) (Codex)

## Common Issues

### "Command not found: neo"
```bash
# Verify installation
pip show neo-reasoner

# If not installed, install it
pip install neo-reasoner
```

### "OPENAI_API_KEY not set" (or ANTHROPIC_API_KEY, GOOGLE_API_KEY)
```bash
export OPENAI_API_KEY=sk-...
# Add to ~/.bashrc or ~/.zshrc for persistence
```

### "No module named 'neo'"
```bash
# Reinstall package
pip install --upgrade neo-reasoner
```

## Alternative Providers

### Anthropic (Claude)
```bash
pip install neo-reasoner[anthropic]
export ANTHROPIC_API_KEY=sk-ant-...
neo --config set --config-key provider --config-value anthropic
neo --config set --config-key model --config-value claude-sonnet-4-5-20250929
```

### Google (Gemini)
```bash
pip install neo-reasoner[google]
export GOOGLE_API_KEY=...
neo --config set --config-key provider --config-value google
neo --config set --config-key model --config-value gemini-2.0-flash
```

### Local (Ollama)
```bash
pip install neo-reasoner
ollama serve
neo --config set --config-key provider --config-value ollama
neo --config set --config-key base_url --config-value http://localhost:11434
```

That's it! You're ready to use Neo.
