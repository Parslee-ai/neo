# Pattern: Sliding Window Rate Limiter
Author: mliotta

## Intent
Enforce precise per-window rate limits by tracking request timestamps and rejecting requests that exceed the limit within any rolling time window.

## Forces
- Need accurate per-window limits (e.g., 100 requests per minute, not averaged)
- Must prevent edge-case exploitation (e.g., 100 requests at 0:59, 100 at 1:00)
- Memory overhead proportional to request volume and window size
- Fixed windows create traffic spikes at boundary resets
- Distributed systems require coordination to prevent over-limit behavior
- Old timestamps must be pruned to prevent unbounded memory growth

## Solution Sketch
Core components:
- **Request Log**: Sorted list or deque of request timestamps (millisecond precision)
- **Window Calculation**: On each request, remove timestamps older than window duration
- **Limit Check**: Count remaining timestamps, reject if >= limit
- **Pruning**: Remove expired entries before checking (or use TTL-based storage)

Variants:
- **Fixed window**: Reset counter at fixed intervals (simpler, less accurate)
- **Sliding log**: Store exact timestamps (accurate, higher memory)
- **Sliding counter**: Approximate using two fixed windows weighted by elapsed time (lower memory)

Sequence:
1. On request arrival, prune timestamps older than (now - window_duration)
2. If len(timestamps) < limit, append current timestamp and allow
3. Else reject with 429 status, include oldest_timestamp + window_duration in Retry-After

## Consequences
**Benefits:**
- Precise enforcement prevents edge-case exploitation
- Sliding window smooths traffic compared to fixed resets
- Easy to audit (inspect timestamp log for debugging)

**Risks:**
- Memory grows with request volume (need bounded storage or pruning)
- O(n) pruning cost per request (mitigate with lazy cleanup)
- Distributed deployments need shared state (sorted timestamp storage with atomic operations)

**Failure Modes:**
- Clock drift between nodes causes inconsistent limits
- OOM if timestamps not pruned under sustained high traffic
- Race conditions in distributed setup without atomic operations

**Observability:**
- Metrics: window_size, current_count, rejection_rate
- Log rejected requests with timestamp delta to limit reset
- Alert on sustained limit violations (indicator of attack or misconfiguration)

## References
- Cloudflare rate limiting: https://blog.cloudflare.com/counting-things-a-lot-of-different-things/
- Redis ZSET-based implementation: https://redis.io/commands/zadd
- Kong sliding window: https://docs.konghq.com/hub/kong-inc/rate-limiting/
