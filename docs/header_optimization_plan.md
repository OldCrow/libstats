# v0.11.0 Header Optimization Plan with IWYU Integration

**Created:** 2025-08-20
**Target Version:** v0.11.0
**Branch:** v0.11.0-header-optimizations
**Estimated Duration:** 1-2 weeks

---

## 🎯 Objectives

1. **Reduce compilation time by 20-35%** through systematic header cleanup
2. **Fix ~94 include issues** identified by Include What You Use (IWYU)
3. **Standardize header inclusion patterns** across the codebase
4. **Implement forward declarations** where appropriate
5. **Create reusable header modules** to reduce duplication

---

## 📊 IWYU Analysis Summary

Initial IWYU scan reveals:
- **94 include-related issues** across source files
- Most files have both "should add" and "should remove" recommendations
- Common pattern: over-inclusion of `constants.h` when only specific constants needed
- Platform-specific headers often include more than necessary

### Most Common Issues:
1. **Over-inclusion of monolithic headers**:
   - `core/constants.h` included when only specific constants needed
   - `platform/platform_constants.h` included for single values

2. **Missing granular includes**:
   - Missing `<cstdint>` for integer types
   - Missing `<cstddef>` for `size_t`
   - Missing specific math headers

3. **Unnecessary algorithm/container includes**:
   - `<algorithm>` included but not used
   - `<vector>` included when forward declaration sufficient

---

## 📝 Task Breakdown

### Phase 1: IWYU Cleanup (2-3 days)

#### Task 1.1: Create Granular Constant Headers
Split the monolithic `constants.h` into focused headers:

```bash
include/core/
├── constants.h                    # Main aggregator (kept for compatibility)
├── mathematical_constants.h       # Math constants (PI, E, etc.)
├── threshold_constants.h          # Algorithm thresholds
├── precision_constants.h          # Numerical precision limits
├── statistical_constants.h        # Statistical test constants
└── essential_constants.h          # Core integer constants
```

**Checklist:**
- [ ] Split constants.h into granular headers
- [ ] Update all source files to use specific constant headers
- [ ] Keep constants.h as aggregator for backward compatibility
- [ ] Measure compilation time improvement

#### Task 1.2: Fix IWYU-Identified Issues in Core Files

**Priority 1 - SIMD Files** (affects all builds):
- [ ] `src/simd_dispatch.cpp` - Remove `<algorithm>`, add specific headers
- [ ] `src/simd_fallback.cpp` - Use granular constant headers
- [ ] `src/simd_sse2.cpp` - Remove unused includes
- [ ] `src/simd_avx.cpp` - Add missing integer type headers
- [ ] `src/simd_avx2.cpp` - Clean up platform includes
- [ ] `src/simd_avx512.cpp` - Optimize header usage
- [ ] `src/simd_neon.cpp` - Fix ARM-specific includes
- [ ] `src/simd_policy.cpp` - Remove redundant includes

**Priority 2 - Distribution Files** (high impact on compilation):
- [ ] `src/gaussian.cpp` - Remove unused C++20 headers
- [ ] `src/exponential.cpp` - Use specific math headers
- [ ] `src/uniform.cpp` - Clean up container includes
- [ ] `src/poisson.cpp` - Fix algorithm includes
- [ ] `src/gamma.cpp` - Optimize math function includes
- [ ] `src/discrete.cpp` - Remove unused headers

**Priority 3 - Infrastructure Files**:
- [ ] `src/thread_pool.cpp` - Use forward declarations
- [ ] `src/work_stealing_pool.cpp` - Optimize threading includes
- [ ] `src/cpu_detection.cpp` - Clean up platform-specific includes
- [ ] `src/validation.cpp` - Use specific constant headers

#### Task 1.3: Run IWYU Fix Script
```bash
# Generate fixes
./scripts/run-iwyu.sh --src > iwyu_fixes.txt

# Review and apply selectively (not all suggestions are correct)
python3 /usr/local/bin/fix_includes.py < iwyu_fixes.txt --safe
```

---

### Phase 2: Forward Declaration Implementation (3-4 days)

#### Task 2.1: Create Forward Declaration Headers

Create lightweight forward declaration headers:

```cpp
// include/common/platform_constants_fwd.h
namespace libstats::platform {
    struct CacheInfo;
    struct AlignmentInfo;
    class PlatformConstants;
}

// include/common/parallel_execution_fwd.h
namespace libstats::parallel {
    class ThreadPool;
    class WorkStealingPool;
    template<typename T> class ParallelExecutor;
}
```

**Files to create:**
- [ ] `platform_constants_fwd.h`
- [ ] `parallel_execution_fwd.h`
- [ ] `distribution_base_fwd.h`
- [ ] `simd_policy_fwd.h`

#### Task 2.2: Update Source Files to Use Forward Declarations

Convert heavy includes to forward declarations where possible:

**Before:**
```cpp
#include "../include/platform/platform_constants.h"  // Heavy include
```

**After:**
```cpp
#include "../include/common/platform_constants_fwd.h"  // Light forward decl
// Only include full header in .cpp file
```

**Target files:**
- [ ] All SIMD implementation files
- [ ] Distribution source files
- [ ] Thread pool implementations
- [ ] Performance dispatcher

---

### Phase 3: Header Standardization (2-3 days)

#### Task 3.1: Create Common Include Headers

```cpp
// include/core/distribution_stdlib_common.h
#pragma once

// Standard library includes common to all distributions
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Threading includes (conditional)
#if defined(LIBSTATS_USE_PARALLEL)
#include <atomic>
#include <mutex>
#include <thread>
#endif
```

**Headers to create:**
- [ ] `distribution_stdlib_common.h` - Shared stdlib includes
- [ ] `simd_implementation_common.h` - Shared SIMD includes
- [ ] `test_common.h` - Shared test includes

#### Task 3.2: Remove Redundant Includes

**Patterns to fix:**
- [ ] Remove duplicate threading includes from all distributions
- [ ] Consolidate math function includes
- [ ] Standardize container includes
- [ ] Remove unused C++20 feature headers

---

## 📈 Success Metrics

### Compilation Time Targets:
- **Debug build**: 20-25% faster
- **Release build**: 25-35% faster
- **Incremental rebuild**: 40-50% faster

### Code Quality Metrics:
- [ ] Zero IWYU warnings for core files
- [ ] Reduced header dependencies (measurable via include graphs)
- [ ] No functionality regression (all tests pass)
- [ ] No performance regression (benchmarks stable)

---

## 🔧 Implementation Strategy

### Week 1:
1. **Day 1-2**: IWYU cleanup and constant header splitting
2. **Day 3-4**: Forward declaration implementation
3. **Day 5**: Testing and validation

### Week 2:
1. **Day 1-2**: Header standardization
2. **Day 3**: Performance testing and optimization
3. **Day 4-5**: Documentation and final cleanup

---

## 📊 Measurement Plan

### Before Optimization:
```bash
# Baseline compilation time
time cmake --build build --clean-first --parallel 8

# Header dependency count
find include -name "*.h" -exec grep -l "include" {} \; | wc -l

# IWYU issue count
./scripts/run-iwyu.sh --src 2>/dev/null | grep "should" | wc -l
```

### After Each Phase:
- Measure compilation time improvement
- Run full test suite
- Check performance benchmarks
- Update IWYU issue count

---

## 🚀 Next Steps

1. **Immediate**: Run comprehensive IWYU analysis
2. **Today**: Begin splitting constants.h
3. **This week**: Complete Phase 1 (IWYU cleanup)
4. **Next week**: Complete Phases 2-3

---

## 📝 Notes

### IWYU Caveats:
- Not all IWYU suggestions are correct (especially for templates)
- Some platform-specific includes must be preserved
- Forward declarations don't work for all template instantiations

### Risks:
- **Medium**: Breaking changes if includes are incorrectly removed
- **Low**: Performance regression (unlikely with header changes)
- **Low**: Platform-specific build failures

### Mitigation:
- Test on all platforms via CI after each change
- Keep backup of original includes
- Review IWYU suggestions manually before applying

---

**Status:** Ready to implement
**Branch:** v0.11.0-header-optimizations
**Target completion:** 1-2 weeks
