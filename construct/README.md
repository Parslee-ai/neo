# The Construct - Pattern Library

Welcome to The Construct, Neo's curated library of architecture and design patterns. This collection provides vendor-agnostic solutions to common engineering problems, indexed with semantic search for easy discovery.

## Philosophy

The Construct follows these principles:

1. **Vendor-Agnostic**: Patterns describe concepts, not specific products
2. **Consequence-Aware**: Every pattern documents tradeoffs and failure modes
3. **Observability-First**: Include metrics and alerts for each pattern
4. **Community-Driven**: Anyone can contribute patterns with proper attribution

## Pattern Structure

Each pattern follows this template:

```markdown
# Pattern: <Name>
Author: <GitHub username or name>

## Intent
One sentence explaining the problem solved.

## Forces
Key constraints and tradeoffs (bulleted list).

## Solution Sketch
Conceptual structure or sequence—how components interact. No framework code.

## Consequences
Benefits, risks, failure modes, and observability signals.

## References
Optional links to real-world implementations, blog posts, or open specs.
```

## Quality Standards

All patterns must meet these requirements:

### Required Fields
- **Author**: Mandatory attribution (GitHub username or full name)
- **Intent**: Clear one-sentence problem statement
- **Forces**: Minimum 3 constraints/tradeoffs
- **Solution**: Conceptual explanation (no vendor-specific code)
- **Consequences**: Both benefits AND risks

### Size Constraints
- Maximum 300 lines total
- Intent: 1-2 sentences
- Forces: 3-10 bullet points
- Solution: 1-3 paragraphs or sequence diagram
- Consequences: Structured with benefits, risks, failure modes, observability

### Content Guidelines
- **DO**: Use generic terms (cache, queue, database, API)
- **DON'T**: Reference specific products in problem/solution descriptions (e.g., "use Redis for caching")
**DO**: Reference specific products in the **References** section as implementation examples
- **DO**: Include observability metrics (latency, error rate, capacity)
- **DON'T**: Include framework-specific code snippets
- **DO**: Link to real-world implementations in References section
- **DON'T**: Make patterns too abstract (include concrete failure modes)

## Contributing a Pattern

1. **Choose a Domain**: Pick an existing domain or propose a new one
   - Existing: `rate-limiting`, `caching`, `resilience`, `observability`
   - Propose new domains via issue/PR discussion

2. **Write the Pattern**: Use the template above
   - Fill in all required sections
   - Keep it under 300 lines
   - Focus on tradeoffs, not just happy paths

3. **Validate Locally**:
   ```bash
   # Run pattern validation
   neo construct index

   # Check output for errors
   # Fix any validation warnings
   ```

4. **Submit a PR**:
   - File: `/construct/<domain>/<pattern-name>.md`
   - Commit message: `add(construct): <domain>/<pattern-name> pattern`
   - Include rationale in PR description

5. **Review Process**:
   - Maintainer checks quality standards
   - Community feedback on clarity/accuracy
   - Merge once approved

## Pattern Domains

### rate-limiting/
Patterns for controlling request rates:
- `token-bucket.md` - Classic token bucket algorithm
- `sliding-window.md` - Sliding window rate limiter
- `distributed-rate-limiting.md` - Distributed rate limiting with shared state

### caching/
Patterns for caching strategies:
- `cache-aside.md` - Lazy loading cache pattern
- `write-through.md` - Write-through caching
- `cache-invalidation.md` - Cache invalidation strategies

## Using Patterns

### CLI Commands

```bash
# List all patterns
neo construct list

# Filter by domain
neo construct list --domain caching

# Show full pattern
neo construct show caching/cache-aside

# Semantic search
neo construct search "prevent thundering herd"

# Build search index (run after adding patterns)
neo construct index
```

### Search Tips

The semantic search understands intent, not just keywords:

**Good queries:**
- "How to handle rate limit bursts?"
- "Prevent stale data in distributed cache"
- "Scale rate limiting across servers"

**Less effective:**
- "rate limiting" (too broad, use `list --domain` instead)
- "Redis patterns" (too vendor-specific)

### Integration with Neo

When Neo encounters a problem related to a pattern, it can retrieve relevant patterns from The Construct to inform its reasoning. The same embedding model (Jina Code v2) indexes both patterns and Neo's reasoning memory.

## Pattern Examples

### Example: Token Bucket Pattern

```
Intent: Control API request rate while allowing burst traffic

Forces:
- Need smooth average throughput over time
- Must allow occasional bursts without rejection
- O(1) memory overhead preferred
- Thread-safe in concurrent environments

Solution: Maintain bucket with tokens that refill at constant rate...

Consequences:
Benefits: Simple, burst-tolerant, O(1) complexity
Risks: Clock skew, thundering herd after idle period
Observability: tokens_consumed, rejection_rate, bucket_utilization
```

## Advanced Topics

### Composing Patterns

Many real-world systems combine patterns:
- **Cache-aside + Invalidation**: Lazy caching with event-driven invalidation
- **Token Bucket + Distributed Rate Limiting**: Per-node buckets + global counter
- **Write-through + Cache Invalidation**: Strong consistency with fallback invalidation

Document pattern combinations in the References section.

### Anti-Patterns

The Construct also documents anti-patterns (what NOT to do):
- Mark with prefix: `antipattern-<name>.md`
- Explain why the approach fails
- Link to preferred alternatives

## Maintenance

### Updating Patterns

Patterns evolve as best practices change:
- Submit PR with changes
- Explain rationale (new failure mode discovered, better alternative exists)
- Preserve original author attribution
- Add "Updated by" line if significant changes

### Deprecating Patterns

If a pattern becomes obsolete:
- Move to `/construct/_deprecated/<domain>/`
- Add deprecation notice at top
- Link to replacement pattern

## Getting Help

- **Pattern unclear?** Open an issue with specific questions
- **Want to propose a domain?** Start a discussion in issues
- **Found an error?** Submit a PR with correction

## Credits

The Construct is inspired by:
- *Design Patterns* (Gang of Four, 1994)
- *Pattern-Oriented Software Architecture* (Buschmann et al.)
- *Release It!* (Michael Nygard)
- Martin Fowler's architecture patterns catalog

Built by the Neo community with contributions from engineers worldwide.
