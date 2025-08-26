# Phase 3B: SIMD Namespace Reorganization Plan

## Overview
Reorganize SIMD-related code from `stats::arch::simd::` to `stats::simd::` with clear separation of concerns.

## Current State
- SIMD detection utilities in `stats::arch::simd::` (simd.h)
- SIMD operations in `VectorOps` class within `stats::arch::simd::`
- Runtime dispatch logic in simd_dispatch.cpp

## Target Architecture
```cpp
stats::
├── simd::                    // All SIMD operations & dispatch
│   ├── ops::                // SIMD operations (VectorOps class)
│   ├── dispatch::           // Runtime SIMD dispatch
│   └── utils::              // SIMD utilities (alignment, detection)
└── arch::                   // Architecture tuning parameters only
    └── simd::               // Remove or keep only tuning constants
```

## Implementation Steps

### Step 1: Create New Namespace Structure
1. Create `stats::simd::utils::` namespace
2. Create `stats::simd::ops::` namespace
3. Create `stats::simd::dispatch::` namespace

### Step 2: Move Detection Utilities
**FROM:** `stats::arch::simd::` (in simd.h)
**TO:** `stats::simd::utils::`

Functions to move:
- `has_simd_support()`
- `double_vector_width()`
- `float_vector_width()`
- `optimal_alignment()`
- `feature_string()`
- `supports_vectorization()`
- Memory prefetching functions
- Alignment checking utilities

### Step 3: Move VectorOps Class
**FROM:** `stats::arch::simd::VectorOps`
**TO:** `stats::simd::ops::VectorOps`

This includes all the static methods for vector operations.

### Step 4: Create Dispatch Namespace
Move runtime dispatch logic to `stats::simd::dispatch::`

### Step 5: Update All References
Files that need updating (based on grep results):
1. src/simd_dispatch.cpp
2. src/simd_neon.cpp
3. src/simd_avx.cpp
4. src/simd_avx2.cpp
5. src/simd_avx512.cpp
6. src/simd_sse2.cpp
7. src/simd_fallback.cpp
8. src/simd_policy.cpp
9. tests/test_simd_integration.cpp
10. tools/simd_verification.cpp
11. tools/system_inspector.cpp
12. include/platform/simd_policy.h
13. include/core/performance_dispatcher.h

### Step 6: Update Forward Declarations
Update platform_constants_fwd.h to reflect new namespace structure.

## Namespace Alias Strategy
To minimize breaking changes, consider adding namespace aliases:
```cpp
namespace stats {
namespace arch {
namespace simd {
    // Temporary aliases for backward compatibility
    using namespace ::stats::simd::utils;
    using VectorOps = ::stats::simd::ops::VectorOps;
}
}
}
```

## Risk Assessment
- **High Risk**: Breaking existing code that uses `stats::arch::simd::`
- **Medium Risk**: Missing some references in less obvious places
- **Low Risk**: Performance impact (namespace changes don't affect runtime)

## Testing Strategy
1. Compile all files after namespace changes
2. Run SIMD verification tool
3. Run all SIMD integration tests
4. Verify performance benchmarks remain unchanged

## Rollback Plan
If issues arise:
1. Revert namespace changes
2. Use namespace aliases as intermediate step
3. Gradually migrate one component at a time

## Estimated Impact
- Files to modify: ~15-20
- Lines to change: ~200-300
- Compilation time: Should remain unchanged
- Runtime performance: No impact expected
