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
neo 0.7.0
Storage: FileStorage (path: /Users/you/.neo)
Stage: Sleeper | Memory: 0.0%
░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
0 patterns | 0.00 avg confidence
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

# Create adapter
adapter = create_adapter("anthropic")

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

## 6. Claude Code Plugin (Optional)

If you use Claude Code, install the Neo plugin for integrated reasoning:

```bash
/plugin marketplace add Parslee-ai/claude-code-plugins
/plugin install neo
```

Then use slash commands:
```bash
/neo How should I structure this feature?
/neo-review src/auth.py
/neo-optimize slow_function
```

**Important:** The plugin requires the Neo CLI (step 1) to be installed first.

## Next Steps

- **Full Documentation**: See [README.md](README.md)
- **Installation Guide**: See [INSTALL.md](INSTALL.md)
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Plugin Guide**: See [.claude-plugin/README.md](.claude-plugin/README.md)

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
neo --config set --config-key model --config-value gemini-2.5-pro
```

### Local (Ollama)
```bash
pip install neo-reasoner
ollama serve
neo --config set --config-key provider --config-value ollama
neo --config set --config-key base_url --config-value http://localhost:11434
```

That's it! You're ready to use Neo.
