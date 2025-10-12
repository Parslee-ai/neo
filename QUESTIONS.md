## Open Questions:

1. Does this need to be installed in the working directory?  I have a monorepo setup, and the code I want to work directly in is nested three layers down.  
  - Seems so: I have environment variables working in the morepo root but when I navigate to cilantro-site they are no longer visible:
```bash
  `└─> neo "identify code in sections that should be in a component and suggest components"
[Neo] Adaptive limit: 30 files (based on prompt specificity)
[Neo] Gathered 9 files (59,109 bytes)
[Neo] Invoking LLM inference...
{
  "error": "Failed to initialize LM adapter: OpenAI API key required",
  "hint": "Set NEO_PROVIDER and NEO_MODEL in config.json or environment, or set provider-specific API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)"
}
[ :m4: mpstaton][ 20251012@14:39:36]
[ main ≡]
┣[ ~/code/lossless-monorepo/astro-knots/sites/cilantro-site/src/layouts]
                                                                   ┣[ 0.23s][ RAM: 26/48GB][ 82]
```

2. What is the syntax for the NEO_PROVIDER OR NEO_MODEL environment variables?  I see that the code references them, but I don't see any examples in the README.