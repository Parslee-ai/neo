# Neo Claude Code Plugin

Transform Claude Code into a semantic reasoning powerhouse with Neo's multi-agent collaboration system and persistent memory.

## 🧠 What is Neo?

Neo is a semantic reasoning helper that uses:
- **Multi-agent reasoning** (Solver, Critic, Verifier) for robust solutions
- **Persistent semantic memory** that learns from past problems and solutions
- **Confidence scoring** for all suggestions and recommendations
- **Pattern recognition** to identify reusable architectural patterns
- **Reinforcement learning** to improve over time

## 📦 Installation

### Method 1: Via Plugin Marketplace (Recommended)

```bash
/plugin marketplace add Parslee-ai/neo
```

### Method 2: Manual Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Parslee-ai/neo.git
   cd neo
   ```

2. **Install Neo globally:**
   ```bash
   pip install -e .
   ```

3. **Configure Neo:**
   Create `~/.neo/config.json`:
   ```json
   {
     "provider": "openai",
     "model": "gpt-5-codex",
     "api_key": "your-api-key-here"
   }
   ```

   Or set environment variables:
   ```bash
   export OPENAI_API_KEY="your-key"
   # or ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
   ```

4. **Verify installation:**
   ```bash
   timeout 600 neo "hello, what can you do?"
   ```

## 🚀 Quick Start

Once installed, Neo is available in Claude Code through:

### 1. Neo Subagent

Invoke the Neo agent directly:
```
Use the Neo agent to review this code for optimization opportunities.
```

Claude Code will automatically delegate to the Neo agent, which will:
- Gather relevant context from your codebase
- Query Neo with specific details
- Parse Neo's multi-agent reasoning
- Present actionable recommendations

### 2. Slash Commands

Use convenient shortcuts for common tasks:

#### `/neo` - General Reasoning
```
/neo How should I structure my new feature for user analytics?
```

#### `/neo-review` - Code Review
```
/neo-review src/api/handlers.py
```

#### `/neo-optimize` - Performance Optimization
```
/neo-optimize process_large_dataset function
```

#### `/neo-architect` - Architectural Guidance
```
/neo-architect Should I use microservices or monolith for this project?
```

#### `/neo-debug` - Debugging Help
```
/neo-debug TypeError in data processing pipeline
```

#### `/neo-pattern` - Pattern Extraction
```
/neo-pattern repository pattern implementation
```

## 💡 How It Works

### The Neo Agent Workflow

When you invoke Neo through Claude Code:

1. **Context Gathering**: The agent uses Read, Grep, Glob to collect relevant code
2. **Query Formulation**: Creates a detailed query with context for Neo
3. **Neo Execution**: Runs Neo with proper timeout (600s) and format
4. **Multi-Agent Reasoning**: Neo's Solver, Critic, and Verifier collaborate
5. **Memory Lookup**: Neo checks semantic memory for similar past problems
6. **Response Parsing**: Agent extracts insights from Neo's JSON output
7. **Presentation**: Translates findings into actionable recommendations

### Neo's Multi-Agent System

```
User Query → Context Gathering → Neo CLI
                                    ↓
                            Solver Agent (generates solutions)
                                    ↓
                            Critic Agent (finds flaws)
                                    ↓
                            Verifier Agent (validates correctness)
                                    ↓
                            Semantic Memory Lookup
                                    ↓
                            Confidence Scoring
                                    ↓
                            Structured JSON Output
```

### Semantic Memory

Neo maintains persistent memory that:
- **Learns from feedback**: Successful patterns → ⬆ confidence, failed → ⬇ confidence
- **Finds similar problems**: Uses 768-dim embeddings (Jina Code v2)
- **Consolidates patterns**: Merges similar entries into reusable archetypes
- **Tracks difficulty**: Knows when patterns work (easy/medium/hard problems)

### Personality System

Neo has a **Matrix-inspired personality** that evolves based on memory quality:

| Memory Level | Stage | Characteristics |
|--------------|-------|-----------------|
| 0.0-0.2 | **The Sleeper** | Curious, skeptical: "Whoa." "This can't be real." |
| 0.2-0.4 | **The Curious Hacker** | Growing trust: "Show me." "I need to understand." |
| 0.4-0.6 | **The Fighter** | Emerging confidence: "I can do this." "I know kung fu." |
| 0.6-0.8 | **The Believer** | Calm, cryptic: "Two paths. One fast. One safe." |
| 0.8-1.0 | **The One** | Zen-like authority: "You already know the answer." |

**Memory Level** = quality-weighted memory (confidence × success × recency × usage)

Check Neo's personality:
```bash
neo --version
# Example: "What is real?"
# 260 patterns. 0.21 level. Stage: The Curious Hacker
```

The personality affects communication style but NOT technical quality.

## 🎯 Use Cases

### 1. Architectural Decisions
```
/neo-architect We're building a real-time notification system.
Should we use WebSockets, Server-Sent Events, or polling?
Consider: scalability to 100k users, browser compatibility, infrastructure costs.
```

Neo will:
- Analyze tradeoffs between approaches
- Provide confidence scores for each option
- Reference similar systems from its memory
- Identify potential risks and mitigation strategies

### 2. Performance Optimization
```
/neo-optimize data_processing.py

The process_records function is slow with 10k+ records.
Current approach: sequential processing with API calls.
Need: <2s for 10k records.
```

Neo will:
- Identify algorithmic improvements (O(n²) → O(n log n))
- Suggest architectural changes (batch API calls, async processing)
- Provide confidence scores for each optimization
- Reference similar optimizations from memory

### 3. Code Review
```
/neo-review src/auth/login_handler.py

Focus on: security vulnerabilities, edge cases, error handling
```

Neo will:
- Check for common security anti-patterns
- Identify edge cases not handled
- Suggest improvements with confidence scores
- Reference similar code review findings from memory

### 4. Debugging Complex Issues
```
/neo-debug Intermittent race condition in concurrent task processor.
Happens ~5% of the time under high load (>100 concurrent tasks).
Error: "Task X processed twice" in logs.
```

Neo will:
- Identify likely root causes (based on semantic patterns)
- Suggest debugging strategies
- Provide fixes with confidence scores
- Reference similar concurrency bugs from memory

## 🔧 Advanced Usage

### Custom Queries with Context

For maximum effectiveness, provide Neo with rich context:

```bash
# First gather context in Claude Code
Read src/database/connection_pool.py
Grep -r "connection timeout" src/

# Then use Neo with detailed context
/neo We're seeing connection pool exhaustion under load.

Current implementation:
- Max 50 connections
- 30s timeout
- No retry logic

Observed behavior:
- Works fine up to 200 req/s
- Fails at 300+ req/s with "no available connections"

Environment:
- PostgreSQL 14
- Python 3.9
- asyncpg driver

Question: How should we fix this? Consider both quick wins and long-term architecture.
```

### Confidence Scores

Neo provides confidence levels:
- **0.9-1.0**: High confidence - well-known pattern, strong memory match
- **0.7-0.9**: Good confidence - reasonable approach, some uncertainty
- **0.5-0.7**: Medium confidence - plausible solution, needs verification
- **<0.5**: Low confidence - uncertain, multiple possible approaches

**Always verify low-confidence suggestions carefully.**

### Learning Over Time

Neo learns from outcomes:
- Mark successful suggestions as positive feedback
- Flag failed approaches for negative reinforcement
- Over time, Neo's recommendations become more accurate for your codebase patterns

## 📊 Technical Details

### Technologies
- **Python**: 3.9+
- **Embeddings**: Jina Code v2 (768-dim) via fastembed
- **Vector Search**: FAISS for similarity matching
- **Storage**: Local JSON files in ~/.neo directory
- **LLM Support**: OpenAI, Anthropic, Google, Ollama, local models

### Performance
- **Query time**: 5-30s for complex reasoning (depends on LLM)
- **Memory lookup**: <100ms for semantic search
- **Max timeout**: 600s (10 minutes) recommended
- **Memory size**: ~200 entries (auto-consolidates)

### Storage
- **Local files**: All memory stored in `~/.neo/` directory
- **Format**: JSON files with semantic embeddings
- **Auto-consolidation**: Keeps memory under 200 entries

## 🛡️ Best Practices

### DO ✅
- **Use timeouts**: Always `timeout 600 neo "..."`
- **Provide context**: Include relevant code, constraints, goals
- **Be specific**: Detailed queries get better responses
- **Check confidence**: Verify low-confidence suggestions
- **Iterate**: Follow up with clarifying questions

### DON'T ❌
- **No runtime use**: Neo is development-only, never use in production code
- **No blind trust**: Verify all suggestions before applying
- **No vague queries**: "Make this better" gets generic responses
- **No timeout omission**: Queries can hang without timeout
- **No API abuse**: Respect rate limits of your LLM provider

## 🐛 Troubleshooting

### "Command not found: neo"
- Run `pip install -e .` from the neo directory
- Check `which neo` shows correct path
- Add to PATH if needed

### "Timeout after 600s"
- Complex queries may need more time
- Try simplifying the query
- Check LLM API status
- Verify API keys are valid

### "Low confidence scores"
- Query may be too vague - add more context
- Neo may lack relevant memory - it will learn over time
- Problem may be genuinely complex - consider multiple approaches

### "Neo gives generic responses"
- Provide more specific context (code snippets, error messages)
- Include constraints and goals
- Specify what you've already tried

## 📚 Resources

- **Documentation**: [Neo README](https://github.com/Parslee-ai/neo/blob/main/README.md)
- **Installation Guide**: [INSTALL.md](https://github.com/Parslee-ai/neo/blob/main/INSTALL.md)
- **Contributing**: [CONTRIBUTING.md](https://github.com/Parslee-ai/neo/blob/main/CONTRIBUTING.md)
- **Issues**: [GitHub Issues](https://github.com/Parslee-ai/neo/issues)

## 🤝 Contributing

Neo is open source! Contributions welcome:
- Report bugs or request features via GitHub Issues
- Submit PRs for improvements
- Share your Neo patterns and archetypes
- Help improve documentation

## 📄 License

Apache License 2.0 - see [LICENSE](https://github.com/Parslee-ai/neo/blob/main/LICENSE)

## 🙏 Credits

Built by Parslee AI using:
- MapCoder/CodeSim-inspired multi-agent reasoning
- Jina AI embeddings (Jina Code v2)
- FastEmbed for local embedding generation
- FAISS for efficient vector search

---

**Ready to supercharge your Claude Code workflow?** Install Neo and start reasoning semantically! 🧠✨
