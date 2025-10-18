# Pattern: Distributed Rate Limiting
Author: mliotta

## Intent
Coordinate rate limits across multiple servers using shared state to prevent individual nodes from exceeding global quotas.

## Forces
- Multiple application instances must enforce a single global rate limit
- Network latency to shared state store impacts request latency
- Shared state becomes single point of failure (need fallback strategy)
- Strict consistency conflicts with availability (CAP theorem tradeoff)
- Local-only limits allow N * limit total throughput (N = node count)
- Synchronization overhead increases with node count and request volume

## Solution Sketch
Core components:
- **Shared State Store**: Distributed key-value store with atomic operations
- **Atomic Operations**: Atomic increment/decrement or transactional scripts for read-modify-write
- **Local Cache**: Optional layer to reduce remote calls (trades accuracy for latency)
- **Fallback Mode**: Degrade to local limits if shared store unavailable

Variants:
- **Central counter**: Single atomic counter key (simple, accurate, high latency)
- **Sharded counters**: Hash user_id to N counter keys, sum for limit check (scales reads)
- **Gossip protocol**: Nodes exchange local counts periodically (eventual consistency)
- **Sticky sessions**: Route user to same node, enforce locally (no coordination needed)

Sequence:
1. On request: compute rate limit key (e.g., "rl:user:12345:minute:20231017-1430")
2. Atomic increment with expiration: `count = ATOMIC_INCR(key, ttl=window_duration)`
3. If count <= limit, allow request
4. Else reject with 429, include global count in response

## Consequences
**Benefits:**
- Accurate global limits across all nodes
- Scales horizontally (add nodes without changing limits)
- Central visibility into traffic patterns

**Risks:**
- Latency penalty for every request (50-200ms for remote call)
- Shared store becomes critical dependency (outage blocks all traffic)
- Network partitions can cause split-brain limit enforcement
- Cost increases with request volume (CPU/memory/network)

**Failure Modes:**
- Store unavailable: Choose fail-open (allow all) or fail-closed (reject all)
- Thundering herd on key expiration (use jittered TTL)
- Transactional script deadlock under high concurrency (keep logic simple)

**Observability:**
- Metrics: store_call_latency, store_error_rate, fallback_mode_active
- Log local vs. remote limit decisions
- Alert on connection pool exhaustion

## References
- Figma distributed rate limiting: https://www.figma.com/blog/an-alternative-approach-to-rate-limiting/
- Redis rate limiting Lua scripts: https://redis.io/commands/incr#pattern-rate-limiter
- Envoy global rate limiting: https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/other_features/global_rate_limiting
