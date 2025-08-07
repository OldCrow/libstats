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

**Last Updated:** 2024-08-06
**Target Completion:** Before v1.0.0 release
**Priority Order:** Critical performance issues → Architecture → Code quality → Build system → Documentation
