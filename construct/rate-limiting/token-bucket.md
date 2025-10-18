# Pattern: Token Bucket
Author: mliotta

## Intent
Control API request rate by maintaining a bucket of tokens that refill at a constant rate, allowing burst traffic while enforcing average throughput limits.

## Forces
- Need to allow occasional bursts without rejecting legitimate traffic
- Must enforce average rate over time to protect backend services
- Simple implementation preferred over complex windowing algorithms
- Memory overhead should be O(1) per rate limit bucket
- Thread safety required in concurrent environments
- Token refill must be accurate without requiring background threads

## Solution Sketch
Core components:
- **Bucket State**: capacity (max tokens), current token count, last refill timestamp
- **Token Refill**: Calculate elapsed time since last refill, add tokens at fixed rate, cap at capacity
- **Consumption**: Check if tokens >= requested amount, decrement if available, reject if insufficient
- **Atomicity**: Use locks (threading.Lock) or atomic operations to prevent race conditions

Sequence:
1. On each request: calculate tokens to add based on elapsed time
2. Update token count (min of calculated + current, capacity)
3. If tokens >= request cost, consume and allow
4. Else reject request with retry-after header

## Consequences
**Benefits:**
- Simple O(1) time and space complexity
- Naturally handles burst traffic up to bucket capacity
- No scheduled background tasks needed
- Easy to reason about and debug

**Risks:**
- Clock skew can cause incorrect refill rates (use monotonic time)
- Large bursts can starve subsequent requests if capacity too small
- Does not enforce strict per-second limits (tokens accumulate during idle periods)

**Failure Modes:**
- Lock contention under high concurrency (consider lock-free implementations)
- Thundering herd after long idle period (bucket full = large burst allowed)

**Observability:**
- Log rejected requests with current token count
- Metrics: tokens_consumed, tokens_rejected, bucket_capacity_utilization
- Alert on sustained high rejection rates (>10% over 1 minute)

## References
- Generic Cell Rate Algorithm (GCRA): https://en.wikipedia.org/wiki/Generic_cell_rate_algorithm
- Nginx limit_req module: http://nginx.org/en/docs/http/ngx_http_limit_req_module.html
- Stripe rate limiting: https://stripe.com/blog/rate-limiters
