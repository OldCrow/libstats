# Phase 2: Aggressive Namespace Consolidation Plan

## Goal: 3-5 Namespaces TOTAL
1. `stats::` - Public API
2. `stats::detail::` - ALL implementation details
3. `stats::test::` - Testing utilities
4. `stats::arch::` - (optional) Architecture-specific code
5. `stats::simd::` - (optional) SIMD-specific code

## Current State Analysis

We currently have ~100+ nested namespaces including:
- `stats::constants::math::`
- `stats::constants::precision::`
- `stats::constants::probability::`
- `stats::constants::statistical::normal::`
- `stats::constants::benchmark::thresholds::`
- `stats::performance::`
- `stats::validation::`
- `stats::math::`
- Plus ~40+ architecture namespaces (sse, avx, neon, intel, amd, etc.)

## Consolidation Strategy: EVERYTHING Goes to stats::detail::

### Step 1: All Constants → stats::detail:: with Prefixing

```cpp
// BEFORE (deeply nested)
namespace stats::constants::math {
    constexpr double PI = 3.14159...;
}
namespace stats::constants::precision {
    constexpr double EPSILON = 1e-15;
}
namespace stats::constants::statistical::normal {
    constexpr double CRITICAL_VALUE_95 = 1.96;
}

// AFTER (flat in detail with prefixes)
namespace stats::detail {
    // Mathematical constants
    constexpr double MATH_PI = 3.14159...;
    constexpr double MATH_E = 2.71828...;

    // Precision constants
    constexpr double PRECISION_EPSILON = 1e-15;
    constexpr double PRECISION_TOLERANCE = 1e-8;

    // Statistical constants
    constexpr double STAT_NORMAL_CRITICAL_VALUE_95 = 1.96;
    constexpr double STAT_T_DIST_CRITICAL_VALUE_95 = 2.0;
}
```

### Step 2: Platform Constants → stats::detail:: OR stats::arch::

Option A: Everything in detail with prefixes:
```cpp
namespace stats::detail {
    constexpr size_t SSE_MIN_ELEMENTS = 2048;
    constexpr size_t AVX_MIN_ELEMENTS = 4096;
    constexpr size_t NEON_MIN_ELEMENTS = 1024;
}
```

Option B: Use arch namespace ONLY for conditionally compiled code:
```cpp
namespace stats::arch {
    #ifdef __SSE2__
    constexpr size_t MIN_ELEMENTS = 2048;
    #elif __AVX__
    constexpr size_t MIN_ELEMENTS = 4096;
    #elif __ARM_NEON
    constexpr size_t MIN_ELEMENTS = 1024;
    #endif
}
```

### Step 3: All Utilities → stats::detail::

```cpp
// BEFORE
namespace stats::math {
    double lgamma(double x);
}
namespace stats::validation {
    bool isValidProbability(double p);
}
namespace stats::performance {
    class PerformanceDispatcher;
}

// AFTER
namespace stats::detail {
    // Math utilities
    double lgamma(double x);

    // Validation utilities
    bool isValidProbability(double p);

    // Performance classes
    class PerformanceDispatcher;
}
```

## Implementation Plan

### Phase 2A: Constants Consolidation
1. Create temporary mapping of ALL constants with new names
2. Update all constant headers to use `stats::detail::`
3. Add prefixes to avoid collisions:
   - `MATH_*` for mathematical constants
   - `PREC_*` for precision/tolerance
   - `PROB_*` for probability bounds
   - `STAT_*` for statistical values
   - `BENCH_*` for benchmark thresholds
   - `ROBUST_*` for robust estimation
   - `THRESH_*` for algorithm thresholds

### Phase 2B: Platform/Architecture Code
Decision needed:
- **Option 1**: Everything in `stats::detail::` with prefixes (SSE_*, AVX_*, etc.)
- **Option 2**: Use `stats::arch::` for conditionally compiled architecture code
- **Option 3**: Use `stats::simd::` for SIMD operations, `stats::arch::` for CPU detection

### Phase 2C: Utilities and Classes
Move ALL remaining namespaces to `stats::detail::`:
- Performance framework
- Validation utilities
- Math utilities
- Safety checks

## Files to Modify

### Priority 1: Constants Headers (Most collisions)
```bash
include/core/mathematical_constants.h     # math:: → detail:: (MATH_*)
include/core/precision_constants.h        # precision:: → detail:: (PREC_*)
include/core/probability_constants.h      # probability:: → detail:: (PROB_*)
include/core/statistical_constants.h      # statistical::* → detail:: (STAT_*)
include/platform/platform_constants.h     # 49 namespaces! → detail:: or arch::
```

### Priority 2: Utility Headers
```bash
include/core/math_utils.h                # math:: → detail::
include/core/validation.h                # validation:: → detail::
include/core/safety.h                     # safety:: → detail::
include/core/performance_*.h             # performance:: → detail::
```

## Collision Resolution Examples

### Constants Collisions
```cpp
// Multiple "EPSILON" constants
PREC_MACHINE_EPSILON      // from precision::MACHINE_EPSILON
PROB_LOG_EPSILON          // from probability::LOG_PROBABILITY_EPSILON
MATH_EPSILON              // from math::EPSILON

// Multiple "THRESHOLD" constants
BENCH_CV_THRESHOLD        // from benchmark::CV_THRESHOLD
THRESH_PARALLEL_MIN       // from thresholds::MIN_PARALLEL
ROBUST_OUTLIER_THRESHOLD  // from robust::OUTLIER_THRESHOLD

// Multiple "DEFAULT" constants
BENCH_DEFAULT_ITERATIONS  // from benchmark::DEFAULT_ITERATIONS
THRESH_DEFAULT_GRAIN_SIZE // from thresholds::DEFAULT_GRAIN_SIZE
```

### Function Collisions
```cpp
// If multiple namespaces have "initialize()"
detail::initialize_performance()  // was performance::initialize()
detail::initialize_validation()   // was validation::initialize()
```

## Testing Strategy

```bash
# After each major change:
cd build && make -j4
ctest --output-on-failure

# Check for ambiguity errors
grep -r "ambiguous" build/ 2>&1
```

## Success Metrics
- [ ] Maximum 5 namespaces total (stats, detail, test, arch?, simd?)
- [ ] NO nested namespaces beyond one level
- [ ] All tests passing
- [ ] No ambiguity errors
- [ ] Clear, predictable naming with prefixes

## Decision Points

1. **Architecture code**: Should we use `stats::arch::` or put everything in `stats::detail::`?
2. **SIMD code**: Should we use `stats::simd::` or put everything in `stats::detail::`?
3. **Test utilities**: Do we need `stats::test::` or can test utilities stay local to test files?

## Next Steps

1. Start with mathematical_constants.h as pilot
2. Apply pattern to all constant headers
3. Update all references in source files
4. Run tests to verify
5. Continue with remaining namespaces
