# Deprecated Batch Method Cleanup Process

## Overview

This document outlines the systematic process for removing deprecated batch processing methods from libstats distribution classes, based on lessons learned from the DiscreteDistribution cleanup.

## Target Distributions

The following distributions require deprecated batch method cleanup:
- ✅ **DiscreteDistribution** - COMPLETED (Cache-Aware safety overrides added for v1.0.0)
- ⏳ **GaussianDistribution** - PENDING
- ✅ **UniformDistribution** - COMPLETED  
- ✅ **ExponentialDistribution** - COMPLETED
- ⏳ **PoissonDistribution** - PENDING
- ⏳ **GammaDistribution** - PENDING

## Systematic Cleanup Process

### Step 1: Header File Cleanup (`include/distributions/*.h`)

Remove the following deprecated method declarations:

```cpp
// ❌ Remove these deprecated declarations
[[deprecated("Use getProbability(span, span, hint) instead")]]
void getProbabilityBatch(const double* values, double* results, std::size_t count) const;

[[deprecated("Use getLogProbability(span, span, hint) instead")]]
void getLogProbabilityBatch(const double* values, double* results, std::size_t count) const;

[[deprecated("Use getCumulativeProbability(span, span, hint) instead")]]
void getCumulativeProbabilityBatch(const double* values, double* results, std::size_t count) const;

// Also remove unsafe variants:
void getProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const;
void getLogProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const;
void getCumulativeProbabilityBatchUnsafe(const double* values, double* results, std::size_t count) const;
```

### Step 1.5: CRITICAL - Cache-Aware Infrastructure Safety Override

**🚨 MANDATORY STEP**: The Cache-Aware infrastructure is fundamentally broken and must be disabled system-wide for v1.0.0. You MUST implement safe-by-default fallback for CACHE_AWARE strategy in ALL distributions.

**Root Cause**: The Cache-Aware infrastructure has critical design flaws:
1. **String-based cache key generation** using `std::ostringstream` creates O(n²) performance disasters
2. **Zero cache hit rates** for mathematical computations with floating-point inputs
3. **Catastrophic memory overhead** from storing unused cache entries
4. **System-wide performance collapse** - tests running 30+ minutes instead of seconds

**🎯 Target Methods**: For **ALL distributions** (Discrete, Exponential, Gaussian, Uniform, Gamma, Poisson), add safety override to these 3 explicit strategy methods:
1. `getProbabilityWithStrategy()`
2. `getLogProbabilityWithStrategy()` 
3. `getCumulativeProbabilityWithStrategy()`

**✅ Safe Override Implementation**:
Add this at the **beginning** of each WithStrategy method:
```cpp
void DistributionName::getProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                                 performance::Strategy strategy) const {
    // Safety override for v1.0.0 - Cache-Aware has fundamental design flaws causing O(n²) performance
    // String-based cache key generation creates catastrophic overhead for all distribution types
    if (strategy == performance::Strategy::CACHE_AWARE) {
        strategy = performance::Strategy::PARALLEL_SIMD;
    }
    
    performance::DispatchUtils::executeWithStrategy(
        // ... rest of method unchanged ...
```

**✅ ALL DISTRIBUTIONS**: Cache-Aware infrastructure is fundamentally broken across the entire system. ALL distributions (including Discrete and Poisson) now receive safety overrides for v1.0.0 release stability.

### Step 1.6: CRITICAL - Explicit Strategy Threshold Behavior Fix

**🚨 MANDATORY STEP**: After implementing cache-aware safety override, you MUST fix the explicit strategy batch methods to remove threshold checks from parallel-SIMD lambdas. This ensures consistency across all distribution classes.

**Key Design Principle**:
- **Auto-dispatch methods** (`getProbability(span, span, hint)`) → Use `parallel::should_use_parallel()` for intelligent threshold-based dispatch
- **Explicit strategy methods** (`getProbabilityWithStrategy(span, span, strategy)`) → NEVER use threshold checks in parallel lambdas, execute requested strategy directly

**🎯 Target Methods**: For each distribution, update these 3 explicit strategy methods:
1. `getProbabilityWithStrategy()`
2. `getLogProbabilityWithStrategy()` 
3. `getCumulativeProbabilityWithStrategy()`

**🎯 Target Lambda**: In each method above, find the **3rd lambda** (Parallel-SIMD lambda) in the `executeWithStrategy()` call.

### Step 2: Implementation File Cleanup (`src/*.cpp`)

#### 2A: Remove Deprecated Method Implementations

Remove all method bodies for:
- `getProbabilityBatch(const double*, double*, size_t)`
- `getLogProbabilityBatch(const double*, double*, size_t)`
- `getCumulativeProbabilityBatch(const double*, double*, size_t)`
- All `*Unsafe` variants

#### 2B: Update autoDispatch Lambdas

In each `autoDispatch` call, find and update the **second lambda**:

**Before (❌ Deprecated)**:
```cpp
[](const DistType& dist, const double* vals, double* res, size_t count) {
    dist.getXxxBatch(vals, res, count);  // ❌ Calls deprecated method
},
```

**After (✅ Fixed)**:
```cpp
[](const DistType& dist, const double* vals, double* res, size_t count) {
    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
    if (!dist.cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
        if (!dist.cache_valid_) {
            const_cast<DistType&>(dist).updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    
    // Cache parameters for batch processing
    const auto cached_param1 = dist.param1_;
    const auto cached_param2 = dist.param2_;
    // ... extract other needed parameters
    lock.unlock();
    
    // Call private implementation directly
    dist.getXxxBatchUnsafeImpl(vals, res, count, cached_param1, cached_param2, ...);
},
```

**Locations to update**:
- `getProbability(span, span, hint)` method - 1 lambda (the **second** lambda)
- `getLogProbability(span, span, hint)` method - 1 lambda (the **second** lambda)
- `getCumulativeProbability(span, span, hint)` method - 1 lambda (the **second** lambda)

#### 2C: Update executeWithStrategy Lambdas

In each `executeWithStrategy` call, find and update the **second lambda** using the same pattern as above.

**Locations to update**:
- `getProbabilityWithStrategy()` method - 1 lambda (the **second** lambda)
- `getLogProbabilityWithStrategy()` method - 1 lambda (the **second** lambda)
- `getCumulativeProbabilityWithStrategy()` method - 1 lambda (the **second** lambda)

#### 2D: Fix Explicit Strategy Threshold Behavior (MANDATORY)

**🚨 CRITICAL DESIGN FIX**: In explicit strategy methods (`*WithStrategy`), the **3rd lambda** (Parallel-SIMD lambda) must NOT use `parallel::should_use_parallel()` threshold checks. This is a key design principle for power user APIs.

**Target**: Find the **3rd lambda** in each `executeWithStrategy()` call in these 3 methods:
- `getProbabilityWithStrategy()` 
- `getLogProbabilityWithStrategy()`
- `getCumulativeProbabilityWithStrategy()`

**❌ WRONG (Auto-dispatch pattern with threshold checks)**:
```cpp
[](const DistType& dist, std::span<const double> vals, std::span<double> res) {
    // ... cache validation and parameter extraction ...
    
    // ❌ BAD: Explicit strategy using threshold checks
    if (parallel::should_use_parallel(count)) {
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // ... parallel processing ...
        });
    } else {
        // Serial processing for small datasets
        for (std::size_t i = 0; i < count; ++i) {
            // ... serial processing ...
        }
    }
},
```

**✅ CORRECT (Explicit strategy pattern - direct execution)**:
```cpp
[](const DistType& dist, std::span<const double> vals, std::span<double> res) {
    // ... same cache validation and parameter extraction ...
    
    // ✅ GOOD: Execute parallel strategy directly - no threshold checks for WithStrategy power users
    ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
        // ... parallel processing ...
    });
},
```

**🎯 Exact Changes Required**:
1. **Remove** the `if (parallel::should_use_parallel(count))` condition
2. **Remove** the `else` branch with serial processing 
3. **Keep** the `ParallelUtils::parallelFor()` call as the only execution path
4. **Add** comment: `// Execute parallel strategy directly - no threshold checks for WithStrategy power users`

**⚠️ LEAVE UNTOUCHED**: The **3rd lambda** in `autoDispatch()` calls should KEEP the threshold checks - only fix `executeWithStrategy()` calls.

#### 2E: Fix Cache-Aware Auto-Dispatch Lambdas (Continuous Distributions)

**🚨 CRITICAL STEP**: For continuous distributions (Exponential, Gaussian, Uniform, Gamma), you must also fix the **5th lambda** (Cache-Aware lambda) in `autoDispatch` calls to use parallel fallback instead of string-based caching.

**Target**: Find the **5th lambda** in each `autoDispatch()` call in these 3 methods:
- `getProbability(span, span, hint)` 
- `getLogProbability(span, span, hint)`
- `getCumulativeProbability(span, span, hint)`

**❌ BROKEN (String-based caching disaster)**:
```cpp
[](const DistType& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
    // ... cache validation ...
    
    // ❌ CATASTROPHIC: O(n²) string creation + 0% hit rate
    for (std::size_t i = 0; i < count; ++i) {
        const double x = vals[i];
        
        std::ostringstream key_stream;  // ❌ String allocation disaster
        key_stream << std::fixed << std::setprecision(6) << "exp_pdf_" << x;
        const std::string cache_key = key_stream.str();
        
        if (auto cached_result = cache.get(cache_key)) {  // ❌ Always misses
            res[i] = *cached_result;
        } else {
            // ❌ Always executes + cache pollution
            double result = /* compute */;
            res[i] = result;
            cache.put(cache_key, result);  // ❌ Creates cache pollution
        }
    }
}
```

**✅ FIXED (Parallel fallback)**:
```cpp
[](const DistType& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
    // Cache-Aware lambda: For continuous distributions, caching is counterproductive
    // Fallback to parallel execution which is faster and more predictable
    
    // ... same cache validation and parameter extraction ...
    
    // Use parallel processing instead of caching for continuous distributions
    // Caching continuous values provides no benefit (near-zero hit rate) and severe performance penalty
    ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
        const double x = vals[i];
        // ... direct computation without caching ...
    });
}
```

#### 2F: Update Parallel/Work-Stealing/Cache-Aware Lambdas

**⚠️ CRITICAL STEP**: After removing deprecated batch method implementations, you must also update the **3rd, 4th, and 5th lambdas** in both `autoDispatch` and `executeWithStrategy` calls that reference the removed parallel/work-stealing/cache-aware methods.

**Before (❌ Will cause compilation errors)**:
```cpp
[](const DistType& dist, std::span<const double> vals, std::span<double> res) {
    dist.getXxxBatchParallel(vals, res);  // ❌ Method removed!
},
[](const DistType& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
    dist.getXxxBatchWorkStealing(vals, res, pool);  // ❌ Method removed!
},
[](const DistType& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
    dist.getXxxBatchCacheAware(vals, res, cache);  // ❌ Method removed!
}
```

**After (✅ Properly implemented with thread safety)**:
```cpp
[](const DistType& dist, std::span<const double> vals, std::span<double> res) {
    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
    if (!dist.cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
        if (!dist.cache_valid_) {
            const_cast<DistType&>(dist).updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    
    // Cache parameters for thread-safe parallel processing
    const auto cached_param1 = dist.param1_;
    const auto cached_param2 = dist.param2_;
    // ... extract other needed parameters
    lock.unlock();
    
    // Parallel processing using cached parameters
    for (size_t i = 0; i < vals.size(); ++i) {
        res[i] = /* compute using cached parameters */;
    }
},
[](const DistType& dist, std::span<const double> vals, std::span<double> res, WorkStealingPool& pool) {
    // Same cache validation and parameter extraction as above
    // ...
    
    // Work-stealing processing using cached parameters
    for (size_t i = 0; i < vals.size(); ++i) {
        res[i] = /* compute using cached parameters */;
    }
},
[](const DistType& dist, std::span<const double> vals, std::span<double> res, cache::AdaptiveCache<std::string, double>& cache) {
    // Same cache validation and parameter extraction as above
    // ...
    
    // Cache-aware processing using cached parameters
    for (size_t i = 0; i < vals.size(); ++i) {
        res[i] = /* compute using cached parameters */;
    }
}
```

**⚠️ Total Updates Required Per Distribution**:
- `autoDispatch` calls: 3 methods × 4 lambdas each = **12 lambda updates**
- `executeWithStrategy` calls: 3 methods × 4 lambdas each = **12 lambda updates**
- **Explicit Strategy Threshold Fixes**: 3 methods × 1 lambda each = **3 threshold fixes**
- **Cache-Aware Safety Override**: 3 WithStrategy methods = **3 safety overrides** (continuous distributions only)
- **Cache-Aware Auto-Dispatch Fixes**: 3 methods × 1 lambda each = **3 auto-dispatch cache fixes** (continuous distributions only)
- **Total for continuous distributions**: 33 updates (24 lambda updates + 9 safety/cache fixes)
- **Total for discrete distributions**: 27 updates (24 lambda updates + 3 threshold fixes)

## Explicit Strategy Threshold Fix Examples

### DiscreteDistribution Example (✅ COMPLETED)

**Location**: `getProbabilityWithStrategy()` method, 3rd lambda in `executeWithStrategy()` call

**Before (❌)**:
```cpp
[](const DiscreteDistribution& dist, std::span<const double> vals, std::span<double> res) {
    // ... cache validation ...
    
    // Use ParallelUtils::parallelFor for Level 0-3 integration
    if (parallel::should_use_parallel(count)) {  // ❌ BAD
        ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
            // ... processing ...
        });
    } else {
        // Serial processing for small datasets  // ❌ BAD
        for (std::size_t i = 0; i < count; ++i) {
            // ... processing ...
        }
    }
},
```

**After (✅)**:
```cpp
[](const DiscreteDistribution& dist, std::span<const double> vals, std::span<double> res) {
    // ... same cache validation ...
    
    // Execute parallel strategy directly - no threshold checks for WithStrategy power users
    ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
        // ... same processing ...
    });
},
```

**Key Changes**:
1. ❌ Removed: `if (parallel::should_use_parallel(count))` condition
2. ❌ Removed: `else` branch with serial processing
3. ✅ Added: Clear comment explaining design principle
4. ✅ Kept: Direct `ParallelUtils::parallelFor()` execution

### UniformDistribution Example (✅ COMPLETED)

Same pattern applied to:
- `getProbabilityWithStrategy()` - 3rd lambda fixed
- `getLogProbabilityWithStrategy()` - 3rd lambda fixed  
- `getCumulativeProbabilityWithStrategy()` - 3rd lambda fixed

### ExponentialDistribution Example (✅ COMPLETED)

ExponentialDistribution was already correctly implemented with direct execution in explicit strategy methods.

## Distribution-Specific Parameter Extraction

### DiscreteDistribution
```cpp
const int cached_a = dist.a_;
const int cached_b = dist.b_;
const double cached_prob = dist.probability_;
const double cached_log_prob = dist.logProbability_;
const bool cached_is_binary = dist.isBinary_;  // optimization flag
const int cached_range = dist.range_;
```

### GaussianDistribution
```cpp
const double cached_mean = dist.mean_;
const double cached_std = dist.standardDeviation_;
const double cached_variance = dist.variance_;
const double cached_inv_sqrt_2pi_std = dist.invSqrt2PiStd_;  // if available
```

### UniformDistribution
```cpp
const double cached_a = dist.a_;
const double cached_b = dist.b_;
const double cached_range = dist.range_;
const double cached_prob = dist.probability_;  // 1/(b-a)
```

### ExponentialDistribution
```cpp
const double cached_lambda = dist.lambda_;
const double cached_log_lambda = dist.logLambda_;  // if available
```

### PoissonDistribution
```cpp
const double cached_lambda = dist.lambda_;
const double cached_log_lambda = dist.logLambda_;  // if available
const double cached_sqrt_lambda = dist.sqrtLambda_;  // if available
```

### GammaDistribution
```cpp
const double cached_alpha = dist.alpha_;
const double cached_beta = dist.beta_;
const double cached_log_beta = dist.logBeta_;  // if available
const double cached_log_gamma_alpha = dist.logGammaAlpha_;  // if available
// ... other gamma-specific cached values
```

## Critical Order of Operations

**For continuous distributions** (Exponential, Gaussian, Uniform, Gamma), follow this exact sequence:

1. 🚨 **Cache-aware safety override** (3 WithStrategy methods = 3 safety overrides)
2. ✅ **Remove header declarations** first (`include/distributions/*.h`)
3. ✅ **Remove deprecated implementations** (`src/*.cpp`)
4. ✅ **Update autoDispatch lambdas - second lambda** (3 methods × 1 lambda each = 3 fixes)
5. ✅ **Update executeWithStrategy lambdas - second lambda** (3 methods × 1 lambda each = 3 fixes)
6. 🚨 **Fix explicit strategy threshold behavior** (3 methods × 1 lambda each = 3 threshold fixes)
7. 🚨 **Fix cache-aware auto-dispatch lambdas** (3 methods × 1 lambda each = 3 cache fixes)
8. ✅ **Update autoDispatch lambdas - parallel/work-stealing** (3 methods × 2 lambdas each = 6 fixes)
9. ✅ **Update executeWithStrategy lambdas - parallel/work-stealing** (3 methods × 2 lambdas each = 6 fixes)
10. ✅ **Compile and test** - catch any missed references immediately

**For discrete distributions** (DiscreteDistribution, PoissonDistribution), follow this sequence:

1. ✅ **Remove header declarations** first (`include/distributions/*.h`)
2. ✅ **Remove deprecated implementations** (`src/*.cpp`)
3. ✅ **Update autoDispatch lambdas - second lambda** (3 methods × 1 lambda each = 3 fixes)
4. ✅ **Update executeWithStrategy lambdas - second lambda** (3 methods × 1 lambda each = 3 fixes)
5. 🚨 **Fix explicit strategy threshold behavior** (3 methods × 1 lambda each = 3 threshold fixes)
6. ✅ **Update autoDispatch lambdas - parallel/work-stealing/cache-aware** (3 methods × 3 lambdas each = 9 fixes)
7. ✅ **Update executeWithStrategy lambdas - parallel/work-stealing/cache-aware** (3 methods × 3 lambdas each = 9 fixes)
8. ✅ **Compile and test** - catch any missed references immediately

**🚨 Step 1 Details (Cache-Aware Safety Override - CONTINUOUS ONLY)**:
- **Target Methods**: `getProbabilityWithStrategy()`, `getLogProbabilityWithStrategy()`, `getCumulativeProbabilityWithStrategy()`
- **Action**: Add safety override at method beginning
- **Code**: `if (strategy == performance::Strategy::CACHE_AWARE) { strategy = performance::Strategy::PARALLEL_SIMD; }`
- **Reason**: Prevents catastrophic O(n²) performance collapse from string-based caching

**🚨 Step 6/7 Details (Cache-Aware Auto-Dispatch Fix - CONTINUOUS ONLY)**:
- **Target Methods**: `getProbability()`, `getLogProbability()`, `getCumulativeProbability()` auto-dispatch methods
- **Target Lambda**: 5th lambda (Cache-Aware) in each `autoDispatch()` call
- **Action**: Replace string-based caching loop with `ParallelUtils::parallelFor()` direct computation
- **Result**: Fast parallel execution instead of O(n²) cache disaster

**⚠️ Critical**: Do not skip the compilation step between distributions!

## Verification Process (Per Distribution)

After each distribution cleanup:

### Immediate Verification
```bash
# 1. Compile successfully
make -j$(nproc)

# 2. Run basic tests
./tests/test_[distribution]_basic

# 3. Run enhanced tests  
./tests/test_[distribution]_enhanced
```

### Integration Verification
```bash
# 4. Run dual API tests
./tests/test_dual_api

# 5. Run parallel execution tests
./tests/test_parallel_execution_integration

# 6. Run comprehensive tests
./tests/test_parallel_execution_comprehensive
```

## Common Patterns and Templates

### Cache Validation Template
```cpp
// Ensure cache is valid
std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
if (!dist.cache_valid_) {
    lock.unlock();
    std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
    if (!dist.cache_valid_) {
        const_cast<DistType&>(dist).updateCacheUnsafe();
    }
    ulock.unlock();
    lock.lock();
}
```

### Parameter Extraction Template
```cpp
// Cache parameters for batch processing
const auto cached_param1 = dist.param1_;
const auto cached_param2 = dist.param2_;
const auto cached_param3 = dist.param3_;
lock.unlock();
```

### Private Implementation Call Template
```cpp
// Call private implementation directly
dist.getXxxBatchUnsafeImpl(vals, res, count, cached_param1, cached_param2, cached_param3);
```

## Lessons Learned from DiscreteDistribution

### ❌ What We Missed Initially
- Updated autoDispatch lambdas but forgot executeWithStrategy lambdas
- Had to circle back and fix compilation errors  
- Found a third lambda that still referenced deprecated methods
- **CRITICAL MISS**: Forgot to update the parallel/work-stealing/cache-aware lambdas (3rd, 4th, 5th lambdas) in both autoDispatch and executeWithStrategy calls
- These lambdas were calling removed `*BatchParallel`, `*BatchWorkStealing`, and `*BatchCacheAware` methods
- **DESIGN INCONSISTENCY**: UniformDistribution and DiscreteDistribution explicit strategy methods had threshold checks, but ExponentialDistribution didn't
- **PERFORMANCE DISASTER**: Cache-aware methods in continuous distributions created O(n²) string allocation storms with 0% hit rates
- **REAL-WORLD IMPACT**: ExponentialDistribution enhanced test ran for 52+ minutes consuming 350% CPU before we killed it
- **Total updates required**: 33 for continuous distributions, 27 for discrete distributions

### ✅ What Worked Well  
- Systematic header cleanup first
- Consistent cache validation patterns
- Direct calls to private unsafe implementations
- Thread-safe parameter extraction
- **Design Pattern Consistency**: Fixing explicit strategy threshold behavior across all distributions
- **Clear Design Principle**: Auto-dispatch uses thresholds; explicit strategy always enforces parallelism
- **Safe-by-Default Cache Override**: Preventing catastrophic cache-aware performance collapse in continuous distributions
- **Performance Crisis Resolution**: Fixing 52+ minute test runtime down to seconds with cache-aware fallback
- Immediate compilation and testing after changes

## Expected Issues to Watch For

### 1. Compilation Errors
- **Issue**: Forgot to update a lambda that still calls deprecated methods
- **Solution**: Compile after each distribution and fix immediately

### 2. Cache Parameters
- **Issue**: Each distribution has different parameters to extract
- **Solution**: Check the distribution's private members and cache implementation

### 3. Lock Management
- **Issue**: Improper lock acquisition/release patterns
- **Solution**: Use the standard template provided above

### 4. Method Signatures
- **Issue**: Different distributions have different `*UnsafeImpl` parameter lists
- **Solution**: Check the private method signature and match parameters exactly

### 5. Explicit Strategy Threshold Inconsistency
- **Issue**: Some distributions may have threshold checks in explicit strategy methods while others don't
- **Detection**: Look for `if (parallel::should_use_parallel(count))` in `executeWithStrategy()` 3rd lambdas
- **Solution**: Remove threshold checks from ALL explicit strategy parallel-SIMD lambdas for consistency

### 6. Cache-Aware Performance Disaster (Continuous Distributions)
- **Issue**: String-based caching in continuous distributions creates O(n²) performance collapse
- **Detection**: Look for `std::ostringstream` and `cache.put()` in cache-aware lambdas for continuous distributions
- **Symptoms**: Tests run for 30+ minutes with high CPU usage, system-wide slowdown
- **Solution**: Add safety override in WithStrategy methods + parallel fallback in auto-dispatch cache lambdas

## Success Criteria

After completing all distributions:

- ✅ All deprecated batch methods removed from headers and implementations
- ✅ All lambdas updated to use private unsafe implementations  
- ✅ **Consistent explicit strategy behavior**: No threshold checks in any `*WithStrategy` parallel lambdas
- ✅ **Design principle enforced**: Auto-dispatch uses thresholds; explicit strategy always executes requested strategy
- ✅ **Safe cache-aware behavior**: Continuous distributions use parallel fallback instead of catastrophic string caching
- ✅ **Performance crisis resolved**: No more O(n²) cache disasters causing 30+ minute test runtimes
- ✅ All tests passing
- ✅ Modern auto-dispatch and explicit strategy APIs fully functional
- ✅ Thread-safe cache management operational
- ✅ No references to deprecated batch methods anywhere in codebase

## Batch Processing Efficiency Tips

### For Maximum Efficiency:
1. **Work in one session** to maintain context and patterns
2. **Use consistent patterns** across all distributions
3. **Test incrementally** - compile and test after each distribution
4. **Fix issues immediately** - don't accumulate problems
5. **Follow the exact order** outlined in this document

### Quality Checks:
- Each lambda update follows the same cache validation pattern
- Parameter extraction is consistent with the distribution's cache members
- All method calls match the private unsafe implementation signatures
- **Explicit strategy consistency**: All `*WithStrategy` parallel lambdas execute directly without threshold checks
- **Design pattern verification**: Auto-dispatch methods retain `parallel::should_use_parallel()` checks
- **Cache-aware safety verification**: Continuous distributions use parallel fallback, discrete distributions retain caching
- **Performance validation**: Test runtimes remain reasonable (seconds, not minutes)
- Tests pass after each distribution cleanup

## File Locations Reference

### Headers to modify:
- `include/distributions/gaussian.h`
- `include/distributions/uniform.h` 
- `include/distributions/exponential.h`
- `include/distributions/poisson.h`
- `include/distributions/gamma.h`

### Implementations to modify:
- `src/gaussian.cpp`
- `src/uniform.cpp`
- `src/exponential.cpp` 
- `src/poisson.cpp`
- `src/gamma.cpp`

This document provides the complete roadmap for systematically and efficiently cleaning up all remaining deprecated batch methods in the libstats distribution classes.
