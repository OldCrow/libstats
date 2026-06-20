# Distribution Mutability and Thread-Safety

This document describes the v2.x mutability model for distribution objects.

## Summary

Distribution instances are mutable: setters and `fit()` update parameters and invalidate caches. Read-only scalar and batch methods may use cached values and atomic fast paths.

## Move semantics

All 16 distributions have `noexcept` move constructors and move assignment operators in v2.x. This ensures containers such as `std::vector<GaussianDistribution>` move rather than copy during reallocation.

The standard move contract applies: callers must not concurrently access an object while it is being moved from.

## Copy semantics

Copy constructors acquire source-side locks where needed and copy parameter state. Caches may be invalidated or recomputed per distribution; correctness must not depend on cache state copying.

## Cache model

Distribution classes inherit cache infrastructure from `ThreadSafeCacheManager`.

Key rules:

- setters invalidate cache state
- `fit()` invalidates and recomputes as needed
- scalar getters may use shared locks or atomics
- batch methods should avoid hidden allocation and keep output caller-owned

## Batch APIs

v2.x uses span-based batch APIs:

```cpp
dist.getProbability(values, output, hint);
dist.getLogProbability(values, output, hint);
dist.getCumulativeProbability(values, output, hint);
```

Strategy suffix methods were removed. Use `detail::PerformanceHint` for advanced control.

## Analysis functions

Analysis workflows are free functions under `stats::analysis`. They do not mutate distribution instances unless explicitly documented.
