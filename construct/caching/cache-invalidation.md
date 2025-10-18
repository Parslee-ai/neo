# Pattern: Cache Invalidation Strategies
Author: mliotta

## Intent
Ensure cached data stays consistent with source of truth by removing or updating stale entries when underlying data changes.

## Forces
- "There are only two hard things in Computer Science: cache invalidation and naming things" - Phil Karlton
- Over-invalidation wastes cache capacity and increases database load
- Under-invalidation serves stale data and violates business requirements
- Distributed caches require coordination to invalidate across all nodes
- Complex dependency graphs make invalidation brittle (changing user invalidates posts, comments, etc.)
- TTL-only invalidation trades freshness for simplicity

## Solution Sketch
Core strategies:

**1. TTL-Based (Time-To-Live)**
- Set expiration on cache.set(key, value, ttl=300)
- Pros: Simple, no coordination needed
- Cons: Stale data until expiration, unnecessary refreshes

**2. Explicit Invalidation**
- On write: cache.delete(key)
- Pros: Immediate consistency
- Cons: Requires tracking all affected keys

**3. Event-Driven Invalidation**
- Publish invalidation events to message bus (Kafka, SNS)
- Subscribers listen and invalidate local caches
- Pros: Decouples writers from cache topology
- Cons: Eventual consistency, complex setup

**4. Version-Based**
- Include version in cache key: `cache_key = f"user:{user_id}:v{version}"`
- Increment version on write (old keys auto-expire via TTL)
- Pros: No explicit invalidation needed
- Cons: Multiple versions can coexist temporarily

**5. Dependency Tracking**
- Track which cache keys depend on which entities
- On entity update, invalidate all dependent keys
- Pros: Handles complex dependencies
- Cons: High complexity, potential for cascading invalidations

## Consequences
**Benefits:**
- Controlled staleness (choose consistency vs. performance tradeoff)
- Reduced database load (serve from cache when valid)
- Flexibility (combine strategies for different data types)

**Risks:**
- Invalidation bugs cause hard-to-debug stale data issues
- Over-invalidation negates cache benefits (low hit rate)
- Distributed invalidation requires eventual consistency acceptance
- Cache key namespace pollution (old versions accumulate)

**Failure Modes:**
- Missed invalidation: Stale data served indefinitely (use TTL as backstop)
- Double invalidation: Thundering herd on popular keys (use locking)
- Cascading invalidation: One write invalidates entire cache (limit dependency depth)

**Observability:**
- Metrics: invalidation_rate, stale_read_count (via version mismatch), dependency_depth
- Log all invalidations with reason (write, TTL expiry, manual purge)
- Alert on invalidation storms (>1000/sec indicates bug or attack)

## References
- Facebook TAO invalidation: https://www.usenix.org/system/files/conference/atc13/atc13-bronson.pdf
- Varnish cache invalidation: https://varnish-cache.org/docs/trunk/users-guide/purging.html
- Martin Fowler on cache invalidation: https://martinfowler.com/bliki/TwoHardThings.html
