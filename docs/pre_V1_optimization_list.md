# Pre-v1.0.0 Optimization & Polish Task List

This document tracks all optimization, refactoring, and polish tasks that should be completed before the v1.0.0 release of libstats.

## Performance Optimizations

### 🔥 **Critical Performance Issues**
- [ ] **Auto-dispatch first-call initialization overhead** 
  - Issue: 17x performance penalty on first auto-dispatch call (34μs vs 2μs)
  - Root cause: Cold start initialization of global singletons (thread pools, SIMD detection, performance tracking)
  - Solution: Add explicit `libstats::initialize_performance_systems()` function
  - Status: Identified during discrete distribution refactoring

- [ ] **Architecture-dependent constant tuning**
  - [ ] MIN_ELEMENTS_FOR_PARALLEL thresholds for different systems
  - [ ] Adaptive grain sizes for various CPU architectures
  - [ ] SIMD crossover points for different operations
  - [ ] Thread pool sizing heuristics
  - [ ] Cache-aware batch processing thresholds

## Code Architecture & Organization

### 📚 **Statistical Methods Consolidation**
- [ ] **Extract common statistical methods to utility classes**
  - [ ] Kolmogorov-Smirnov (K-S) goodness-of-fit tests
  - [ ] Anderson-Darling (A-D) goodness-of-fit tests  
  - [ ] Chi-squared tests
  - [ ] Bootstrap confidence intervals
  - [ ] Cross-validation methods
  - [ ] Information criteria (AIC, BIC, AICc)
  - [ ] Consider base class template methods vs utility classes

### 🏗️ **Class Structure Standardization**
- [ ] **Review and standardize class layout across all distributions**
  - [ ] Public interface methods first
  - [ ] Private parameters and methods last
  - [ ] Constructors/Destructors first in public section
  - [ ] Consistent ordering: Core methods → Batch methods → Getters/Setters → Advanced methods
  - [ ] Consistent section commenting and organization

### 📖 **Documentation Review**
- [ ] **Comprehensive documentation audit**
  - [ ] Ensure all public methods have proper Doxygen comments
  - [ ] Standardize parameter descriptions and return value documentation
  - [ ] Add usage examples for complex methods
  - [ ] Review and enhance class-level documentation
  - [ ] Ensure thread-safety guarantees are documented

## Code Quality & Constants

### 🔢 **Magic Numbers Elimination**
- [ ] **Replace all magic numbers with named constants**
  - [ ] Review DiscreteDistribution for remaining magic numbers
  - [ ] Review GaussianDistribution implementation
  - [ ] Review ExponentialDistribution implementation  
  - [ ] Review UniformDistribution implementation
  - [ ] Review PoissonDistribution implementation
  - [ ] Review GammaDistribution implementation
  - [ ] Ensure all statistical constants are in `constants::math::` or `constants::probability::`

### 📦 **Header Management**
- [ ] **Convert remaining files to use Phase 2 PIMPL optimization headers**
  - **Files using `platform/platform_constants.h` (should use `platform/platform_constants_fwd.h` instead):**
    - [ ] `src/simd_fallback.cpp`
    - [ ] `include/platform/platform_common.h` 
    - [ ] `src/simd_avx2.cpp`
    - [ ] `include/core/distribution_base.h`
    - [ ] `src/adaptive_cache.cpp`
    - [ ] `tests/test_constants.cpp` (test file - may need full header)
    - [ ] `src/thread_pool.cpp`
    - [ ] `src/simd_sse2.cpp`
    - [ ] `src/simd_avx.cpp`
    - [ ] `src/simd_avx512.cpp`
    - [ ] `src/cpu_detection.cpp`
    - [ ] `src/work_stealing_pool.cpp`
    - [ ] `include/core/constants.h`
    - [ ] `src/simd_neon.cpp`
    - [ ] `src/simd_dispatch.cpp`
  - **Files using `platform/parallel_execution.h` (should use `platform/parallel_execution_fwd.h` instead):**
    - [ ] `tools/parallel_correctness_verification.cpp`
    - [ ] `src/distribution_base.cpp`
    - [ ] `src/exponential.cpp`
    - [ ] `src/uniform.cpp`
    - [ ] `src/discrete.cpp`
    - [ ] `src/poisson.cpp`
    - [ ] `src/gamma.cpp`
    - [ ] `include/distributions/distribution_platform_common.h`
    - [ ] `tests/test_parallel_compilation.cpp` (test file - may need full header)
    - [ ] `include/libstats.h` (main header - may need full header)
  - **Files that could benefit from STL consolidation headers:**
    - [ ] Replace standalone `#include <vector>` with `#include "common/libstats_vector_common.h"` (47+ files)
    - [ ] Replace standalone `#include <string>` with `#include "common/libstats_string_common.h"` (32+ files)
    - [ ] Replace standalone `#include <algorithm>` with `#include "common/libstats_algorithm_common.h"` (18+ files)
  - **Benefits:** Reduces compilation overhead by ~85% for platform constants, ~40% for parallel execution headers
  - **Estimated Impact:** 15-25% reduction in incremental build times, significant compile-time template instantiation reduction

- [ ] **Streamline header includes**
  - [ ] Remove unnecessary includes from headers
  - [ ] Forward declare when possible instead of including
  - [ ] **Document transitively included headers in .cpp files**
    - Add comments explaining what each include provides
    - Note which headers are included transitively via the class's .h file
  - [ ] Use `#include <xxx>` vs `#include "xxx"` consistently
  - [ ] Minimize header dependencies to reduce compile times

## Build System & Platform Support

### ⚙️ **CMake Build Optimization**
- [ ] **Fine-grained CMake build parameter tuning**
  - [ ] Optimize compiler flags for different OS (macOS/Windows/Linux)
  - [ ] Optimize for different compilers (GCC/Clang/MSVC)
  - [ ] Review and tune architecture-specific optimizations
  - [ ] Ensure proper debug vs release configurations

- [ ] **Improve CMake output and messaging**
  - [ ] Better progress indicators during build
  - [ ] Clear messages for optional components (Intel TBB, C++20 features)
  - [ ] Informative warnings when optional dependencies are missing
  - [ ] **Activate C++20 Thread Policies when available**

### 🖥️ **Platform Compatibility**
- [ ] **Review and minimize conditional compilation guards**
  - [ ] Audit all `#ifdef` blocks for necessity
  - [ ] Ensure minimal guarding that is truly required
  - [ ] Test macOS/Windows/Linux-specific library usage
  - [ ] Verify platform-specific optimizations are properly isolated
  - [ ] Review thread affinity and QoS implementations

## Performance Benchmarking

### 📊 **Systematic Performance Testing**
- [ ] **Cross-platform benchmarking suite**
  - [ ] Establish baseline performance metrics on different architectures
  - [ ] x86_64 (Intel/AMD) performance characterization
  - [ ] ARM64 (Apple Silicon/ARM Cortex) performance characterization
  - [ ] Different compiler optimization level impacts
  - [ ] Memory usage profiling across different workloads

- [ ] **Optimization Validation**
  - [ ] Before/after performance comparisons for each optimization
  - [ ] Regression testing to ensure optimizations don't break functionality
  - [ ] Memory leak detection and resource usage monitoring

## Future Optimization Ideas
*Items that occur during development will be added here*

- [ ] **SIMD Optimization Review**
  - Review discrete distribution SIMD implementations (currently limited due to integer/branching nature)
  - Investigate AVX-512 optimization opportunities where available

- [ ] **Memory Pool Optimization**
  - Consider custom allocators for frequently allocated temporary objects
  - Thread-local storage optimization for batch operations

---

## Status Legend
- 🔥 Critical (performance impacting)
- 📚 Architecture (code organization)
- 🏗️ Structure (standardization)  
- 📖 Documentation
- 🔢 Code Quality
- 📦 Dependencies
- ⚙️ Build System
- 🖥️ Platform Support
- 📊 Performance Testing

---

**Last Updated:** 2025-08-13
**Target Completion:** Before v1.0.0 release
**Priority Order:** Critical performance issues → Architecture → Code quality → Build system → Documentation
