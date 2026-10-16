# Distribution Mutability and Thread-Safety

This document describes the v2.x mutability model for distribution objects.

## Summary

Distribution instances are mutable: setters and `fit()` update parameters and invalidate caches. Read-only scalar and batch methods read parameters and cached values under `withCacheSnapshot` (a shared lock on the object's one `shared_mutex`). Values built from several locked calls, such as `getHazard` and `getCoefficientOfVariation`, go through `readConsistent`, which retries until no setter ran in between, so they see one parameter state. The per-parameter atomics behind `get*Atomic()` are read by no library path.

## Move semantics

All 27 distributions have `noexcept` move constructors and move assignment operators in v2.x. This ensures containers such as `std::vector<GaussianDistribution>` move rather than copy during reallocation.

The standard move contract applies: callers must not concurrently access an object while it is being moved from. Move assignment locks both objects.

## Copy semantics

Copy constructors acquire source-side locks where needed and copy parameter state. Caches may be invalidated or recomputed per distribution; correctness must not depend on cache state copying.

## Cache model

Distribution classes inherit cache infrastructure from `ThreadSafeCacheManager`.

Key rules:

- setters invalidate cache state
- `fit()` invalidates and recomputes as needed
- scalar getters take the shared lock
- setters validate and assign under the exclusive lock; delegating
  distributions update their delegate while holding their own lock
- batch methods should avoid hidden allocation and keep output caller-owned

## Batch APIs

v2.x uses span-based batch APIs:

```cpp
dist.getProbability(values, output, hint);
dist.getLogProbability(values, output, hint);
dist.getCumulativeProbability(values, output, hint);
```

Strategy suffix methods were removed. Use `detail::PerformanceHint` for advanced control.

One call, one parameter state: under `VECTORIZED`, `PARALLEL` and
`WORK_STEALING` a batch call takes one snapshot and evaluates every element
on it, so a concurrent setter cannot mix states within one result span. The
exception is `SCALAR` (forced, or AUTO's choice below the SIMD minimum),
which calls the locking scalar method per element and can mix states under a
concurrent setter.

## Analysis functions

Analysis workflows are free functions under `stats::analysis`. They do not mutate distribution instances unless explicitly documented.
