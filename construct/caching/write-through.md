# Pattern: Write-Through Cache
Author: mliotta

## Intent
Synchronously update both cache and database on write operations, ensuring cache always reflects latest data without manual invalidation.

## Forces
- Read-heavy workloads benefit from always-warm cache
- Write latency increases due to dual updates (cache + database)
- Cache and database must stay consistent (atomicity required)
- Network failures can cause partial updates (cache updated, DB fails)
- Not suitable for write-heavy workloads (cache thrashing, high latency)
- Cold start avoided (cache populated on first write)

## Solution Sketch
Core components:
- **Write Path**: Update cache and database in single transaction or compensating action
- **Read Path**: Always read from cache (cache guaranteed to have latest data)
- **Consistency**: Use two-phase commit or write-ahead log to ensure atomicity
- **Rollback**: On database failure, invalidate cache entry to prevent stale reads

Sequence (write):
1. Validate input data
2. Begin transaction (if supported)
3. Update cache: `cache.set(key, value, ttl=None)` (no expiration for write-through)
4. Update database: `db.update(key, value)`
5. If DB update fails: `cache.delete(key)` and raise error
6. Commit transaction
7. Return success

Sequence (read):
1. result = cache.get(key)
2. If miss: load from DB, populate cache (fallback for edge cases)
3. Return result

## Consequences
**Benefits:**
- Cache always consistent with database (no stale reads)
- Read path simplified (no miss handling needed)
- Predictable read latency (always cache hit)

**Risks:**
- Higher write latency (blocking on cache update)
- Wasted cache space if writes are for cold data
- Cache becomes critical path (failure blocks writes)
- Complexity in handling partial failures

**Failure Modes:**
- Cache update succeeds, DB update fails: Stale data served until TTL (use cache.delete on DB error)
- Network partition between cache and DB: Inconsistent state (use distributed transactions)
- Cache full: Eviction causes cache misses, requires DB fallback (monitor cache capacity)

**Observability:**
- Metrics: write_latency (cache + DB), cache_write_failures, db_write_failures
- Log inconsistency events (cache update succeeded but DB failed)
- Alert on elevated write_latency (>p99 threshold)

## References
- Cache patterns comparison: https://docs.aws.amazon.com/whitepapers/latest/database-caching-strategies-using-redis/caching-patterns.html
- Write-through vs write-behind: https://hazelcast.com/blog/a-hitchhikers-guide-to-caching-patterns/
- DynamoDB Accelerator (DAX): https://aws.amazon.com/dynamodb/dax/
