# LibStats Diagnostics

This directory contains diagnostic tools for investigating performance issues and testing specific components of the libstats library.

## ✅ Investigation COMPLETED: Cache-Aware Performance Issue

### Background
During v0.8.x development, cache-aware batch operations showed severe performance regression:
- **Expected**: Cache-aware performance similar to parallel (~8-20x speedup)
- **Actual**: 100x performance degradation (102,369μs vs 1,797μs SIMD baseline)
- **Original Hypothesis**: `getOptimalGrainSize()` creating excessive thread counts ❌ **DISPROVEN**

### 🎯 Root Cause IDENTIFIED (2025-08-15)
**Issue**: Parallel cache write contention in `AdaptiveCache` synchronization mechanism  
**Evidence**: Identical 100x degradation with 0% cache hit rate using continuous distributions  
**Location**: Cache `put()` operations during concurrent writes  

📋 **See [`CACHE_PERFORMANCE_INVESTIGATION.md`](./CACHE_PERFORMANCE_INVESTIGATION.md) for complete findings**

## Diagnostic Tools

### `cache_grain_size_diagnostic.cpp`
**Status**: ✅ Completed - Grain sizes are reasonable  
**Purpose**: Test `AdaptiveCache::getOptimalGrainSize()` calculations  
**Result**: Grain sizes produce reasonable thread counts (~18 threads for 50K elements)  
**Conclusion**: Grain size calculation is NOT the performance bottleneck  

### `cache_aware_batch_diagnostic.cpp`
**Status**: ✅ Completed - Issue reproduced  
**Purpose**: Simulate v0.8.3 cache-aware batch processing with Poisson distribution  
**Result**: Reproduced 100x performance degradation (227ms for 5K elements)  
**Conclusion**: Confirmed the performance issue exists in batch operations  

### `cache_aware_continuous_diagnostic.cpp` 
**Status**: ✅ Completed - Root cause identified  
**Purpose**: Test parallel cache write performance with 0% cache hit rate  
**Result**: Identical performance degradation with continuous distributions (0% hits)  
**Conclusion**: **ROOT CAUSE = Parallel cache write contention**

## Building and Running

```bash
# Build all diagnostics
make all

# Run specific diagnostic
make run-cache

# Run all diagnostics
make run-all

# Clean built files
make clean
```

## Investigation Workflow

1. **Run Diagnostics**: Execute tools to identify specific issues
2. **Analyze Results**: Determine root cause of performance regression  
3. **Create Fixes**: Implement targeted fixes in `include/platform/adaptive_cache.h`
4. **Test Fixes**: Verify fixes resolve issues without breaking functionality
5. **Re-enable Cache-Aware**: Restore cache-aware batch operations in distributions

## Historical Context

The cache-aware implementations were present in v0.8.3 and earlier, located in:
- `*Distribution::*BatchCacheAware()` methods
- Called `cache_manager.getOptimalGrainSize(count, "operation_type")`
- Used adaptive grain sizing with `ParallelUtils::parallelFor(..., optimal_grain_size)`

These methods are currently disabled/fallback to avoid the performance regression.
