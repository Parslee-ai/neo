# Pattern: Cache-Aside (Lazy Loading)
Author: mliotta

## Intent
Application code explicitly manages cache by checking cache first, loading from database on miss, and storing result for subsequent requests.

## Forces
- Not all data is equally hot (80/20 rule - cache only frequently accessed items)
- Database load should be minimized without over-provisioning cache capacity
- Stale data tolerance varies by use case (user profiles vs. real-time prices)
- Cache warming at startup adds complexity and cold-start latency
- Application code must handle cache failures gracefully (cache optional, not required)
- Thundering herd on cache miss can overwhelm database

## Solution Sketch
Core components:
- **Cache Client**: Key-value store with TTL support (in-memory or remote)
- **Read Path**: Check cache → on miss, load from DB → store in cache → return
- **Write Path**: Update DB first, then invalidate or update cache entry
- **TTL Strategy**: Set expiration based on staleness tolerance (seconds to hours)

Sequence (read):
1. Generate cache key: `cache_key = f"user:{user_id}"`
2. Try cache.get(cache_key)
3. If hit: deserialize and return
4. If miss: query database, serialize result, cache.set(cache_key, data, ttl=300)
5. Return data

Sequence (write):
1. Update database: `db.update(user_id, new_data)`
2. Invalidate cache: `cache.delete(f"user:{user_id}")` (or update in place)

## Consequences
**Benefits:**
- Simple mental model (application controls all cache logic)
- Cache only hot data (efficient memory use)
- Resilient to cache failures (degrade to database-only mode)
- Easy to reason about consistency (cache reflects last write)

**Risks:**
- Cache miss storms during cold start or after eviction
- Code duplication (cache logic scattered across codebase)
- Inconsistent TTLs lead to confusing behavior
- Stale reads possible if write invalidation fails

**Failure Modes:**
- Thundering herd: N requests miss cache simultaneously, all query DB (use locking or probabilistic early refresh)
- Cache stampede: Evicted hot key causes cascading misses (use cache warming or longer TTL)
- Stale data served indefinitely if TTL too long and no invalidation

**Observability:**
- Metrics: cache_hit_rate, cache_miss_rate, db_fallback_count
- Log cache errors separately from business logic errors
- Alert on sustained low hit rates (<50% indicates cache ineffective)

## References
- AWS caching best practices: https://aws.amazon.com/caching/best-practices/
- Redis caching strategies: https://redis.io/docs/manual/patterns/
- Memcached basics: https://github.com/memcached/memcached/wiki/Overview
