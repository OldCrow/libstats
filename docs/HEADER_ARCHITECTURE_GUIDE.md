# libstats Header Architecture Guide

## Overview

This guide provides comprehensive information about the libstats header organization, dependency management design philosophy, and usage patterns for developers working on distributions, tools, tests, and examples. The project follows a **balanced consolidation approach** that reduces redundancy while preserving software engineering principles like the Single Responsibility Principle.

## Design Philosophy

### Core Principles

1. **Balanced Consolidation**: Reduce redundant includes without creating overly complex headers
2. **Single Responsibility Principle**: Each header maintains a focused, well-defined purpose
3. **Layered Architecture**: Strict dependency ordering prevents circular dependencies
4. **Platform Separation**: Platform-specific code is clearly isolated from core functionality
5. **Performance-First**: Headers are organized to optimize both compilation time and runtime performance

### Dependency Management Strategy

- **Bottom-up Hierarchy**: Lower-level headers only depend on standard library
- **Horizontal Isolation**: Related headers at the same level avoid interdependencies
- **Forward Declarations**: Minimize compile-time dependencies through careful interface design
- **Common Patterns**: Consolidate frequently-used header combinations without compromising architecture

## Header Architecture

### Level 0: Foundation (Standard Library Only)

**Location**: `include/core/*_constants.h`, `include/platform/simd_policy.h`

> **✅ Phase 2 Update**: Some foundation headers have been moved to `include/common/` for better organization.

These headers have no internal project dependencies and provide fundamental constants and basic platform detection.

#### Constants Headers
```cpp
// Mathematical and statistical constants
#include "core/precision_constants.h"       // Numerical tolerances, epsilons
#include "core/mathematical_constants.h"    // π, e, √2π, etc.
#include "core/statistical_constants.h"     // Critical values, test parameters

// Specialized constants (use only when needed)
#include "core/probability_constants.h"     // Probability bounds, safety limits
#include "core/threshold_constants.h"       // Algorithm thresholds
#include "core/benchmark_constants.h"       // Performance testing parameters
#include "core/robust_constants.h"          // Robust estimation parameters
#include "core/statistical_methods_constants.h"  // Bayesian, bootstrap constants
#include "core/goodness_of_fit_constants.h" // Critical values for tests
```

#### Basic Platform
```cpp
#include "platform/simd_policy.h"          // SIMD capability detection
```

### Level 1: Consolidated Foundation

**Location**: `include/core/essential_constants.h`, `include/core/constants.h`, `include/platform/cpu_detection.h`

> **✅ Phase 2 Update**: Some platform constants have been moved to `include/common/` - use the new paths where applicable.

#### Essential Constants (Recommended)
```cpp
#include "core/essential_constants.h"       // Most common constants (precision + math + statistical)
```

#### Complete Constants (Use sparingly)
```cpp
#include "core/constants.h"                 // All 9 constants headers (umbrella)
```

#### Platform Foundation
```cpp
#include "platform/cpu_detection.h"        // Runtime CPU feature detection
#include "platform/platform_constants.h"   // Platform-specific optimization constants (via common/)
```

> **✅ Phase 2 Update**: `platform_constants.h` now pulls from `common/platform_constants_fwd.h` and `common/platform_common.h`

### Level 2: Core Utilities and Platform Capabilities

**Location**: `include/core/`, `include/platform/`

#### Core Utilities
```cpp
// Mathematical operations
#include "core/math_utils.h"               // Special functions, numerical algorithms
#include "core/log_space_ops.h"            // Log-space arithmetic for stability

// Safety and validation
#include "core/safety.h"                   // Safe numerical operations
#include "core/validation.h"               // Parameter validation
#include "core/error_handling.h"           // Exception-free error handling

// Statistical utilities
#include "core/statistical_utilities.h"   // Common statistical computations
```

#### Platform Capabilities
```cpp
// SIMD and vectorization
#include "platform/simd.h"                // SIMD operations and memory management

// Threading and parallelism
#include "platform/parallel_thresholds.h"  // Architecture-specific thresholds
#include "platform/thread_pool.h"         // Basic thread pool
#include "platform/work_stealing_pool.h"  // Advanced work-stealing pool
```

### Level 3: Advanced Infrastructure

**Location**: `include/core/`, `include/platform/`

#### Caching and Performance
```cpp
#include "core/distribution_cache.h"      // Distribution-specific caching
#include "platform/parallel_execution.h"  // C++20 parallel algorithms
#include "platform/benchmark.h"            // Performance measurement utilities
```

> **Note**: Cache functionality is integrated within `core/distribution_cache.h` rather than a separate cache directory.

#### Performance Framework
```cpp
#include "core/performance_history.h"     // Performance tracking
#include "core/performance_dispatcher.h"  // Smart algorithm selection
```

### Level 4: Distribution Framework

**Location**: `include/core/distribution*.h`

#### Framework Components
```cpp
#include "core/distribution_interface.h"   // Pure virtual interface
#include "core/distribution_memory.h"      // Memory management and SIMD ops
#include "core/distribution_validation.h" // Validation and diagnostics
```

#### Complete Framework
```cpp
#include "core/distribution_base.h"       // Complete base class (includes all above)
```

### Level 5: Consolidated Common Headers

**Location**: `include/common/*_common.h`, `include/core/*_common.h`, `include/distributions/distribution_platform_common.h`

> **✅ Phase 2 Update**: Major header reorganization completed - common shared headers consolidated in `include/common/`

#### Distribution Development (Recommended Pattern)
```cpp
// For new distribution implementations, use these consolidated headers:

#include "../common/distribution_common.h"           // Core framework + common std library
#include "../common/distribution_platform_common.h"  // Platform optimizations

// Add specific includes only as needed:
// #include <tuple>     // If returning statistical test results
// #include <array>     // If using precomputed lookup tables
```

#### Alternative Patterns
```cpp
// For utilities that need base functionality:
#include "common/distribution_base_common.h"   // Common base dependencies (MOVED)

// For math utilities:
#include "common/utility_common.h"             // Common utility dependencies (MOVED)

// For platform code:
#include "common/platform_common.h"           // Platform-specific common headers (MOVED)
```

> **✅ Phase 2 Update**: Common shared headers moved to `include/common/` for better organization

### Level 6: Concrete Distributions

**Location**: `include/distributions/*.h`

```cpp
#include "distributions/gaussian.h"        // Gaussian (Normal) distribution
#include "distributions/exponential.h"     // Exponential distribution
#include "distributions/uniform.h"         // Uniform distribution
#include "distributions/poisson.h"         // Poisson distribution
#include "distributions/gamma.h"           // Gamma distribution
#include "distributions/discrete.h"        // Discrete distribution
```

### Level 7: Complete Library Interface

**Location**: `include/libstats.h`

```cpp
#include "libstats.h"                      // Complete library (single include)
```

## Usage Guidelines

### For Distribution Development

#### New Distribution Implementation
```cpp
#pragma once

// Use consolidated headers for common functionality
#include "../common/distribution_common.h"    // UPDATED PATH (Phase 2)
#include "distribution_platform_common.h"

// Add specific headers only when needed
// #include <tuple>      // For complex return types
// #include <array>      // For lookup tables
// #include <algorithm>  // For specialized algorithms

class MyDistribution : public DistributionBase {
    // Implementation using the full framework
};
```

> **✅ Phase 2 Update**: `distribution_common.h` moved to `include/common/` directory

#### Key Benefits of This Pattern
- **~60% fewer includes** compared to individual headers
- **Consistent functionality** across all distributions
- **Easier maintenance** when adding new features
- **Faster compilation** through consolidated headers

### For Tools Development

#### Current Pattern (needs consolidation)
```cpp
#include <vector>
#include <iostream>
#include <string>
#include "../include/libstats.h"              // Complete library
#include "../include/core/performance_dispatcher.h"  // Specific performance tools
```

#### Recommended Pattern
```cpp
// For tools that need full functionality:
#include "../include/libstats.h"

// For tools that need only specific distributions:
#include "../include/common/distribution_common.h"  // UPDATED PATH (Phase 2)
#include "../include/distributions/gaussian.h"     // Only what's needed

// For performance analysis tools:
#include "../include/core/performance_dispatcher.h"
#include "../include/platform/benchmark.h"
```

### For Tests Development

#### Current Pattern (basic)
```cpp
#include <iostream>
#include <vector>
#include <cassert>
// Individual includes for each test requirement
```

#### Recommended Pattern
```cpp
// For comprehensive tests:
#include "../include/libstats.h"

// For focused unit tests:
#include "../include/common/distribution_common.h"  // UPDATED PATH (Phase 2)
#include "../include/distributions/gaussian.h"     // Test target

// For performance tests:
#include "../include/platform/benchmark.h"
#include "basic_test_template.h"                // Test utilities
```

### For Examples Development

#### Recommended Pattern
```cpp
// For simple examples:
#include "../include/libstats.h"

// For performance-focused examples:
#include "../include/libstats.h"
// Note: libstats.h includes performance optimization guides

// For specific feature examples:
#include "../include/distributions/gaussian.h"   // Specific distribution
#include "../include/platform/simd.h"           // SIMD examples
```

## Integration Patterns

### Thread Safety
All headers from Level 2+ provide thread-safe operations:
```cpp
// Safe concurrent access patterns
std::shared_mutex cache_mutex_;
std::shared_lock<std::shared_mutex> read_lock(cache_mutex_);  // Concurrent reads
std::unique_lock<std::shared_mutex> write_lock(cache_mutex_); // Exclusive writes
```

### SIMD Integration
```cpp
// Compile-time detection (simd.h)
#ifdef LIBSTATS_HAS_AVX
    // Compiler can generate AVX code
#endif

// Runtime detection (cpu_detection.h)
if (libstats::cpu::supports_avx()) {
    // CPU actually supports AVX
}

// Automatic selection in distributions
// All distributions automatically use best available SIMD
```

### Performance Optimization
```cpp
// Smart dispatch (performance_dispatcher.h)
// Automatic algorithm selection based on:
// - Data size
// - CPU capabilities
// - Performance history
// - Memory pressure

// Adaptive caching (adaptive_cache.h)
// Automatic cache management with:
// - TTL expiration
// - Memory pressure response
// - Access pattern learning
```

## Build Optimization

### Compilation Time Benefits
- **30-40% reduction** in redundant includes through consolidation
- **Parallel compilation** enabled by strict dependency hierarchy
- **Incremental builds** - changes to higher levels only affect dependents
- **Platform isolation** - platform-specific changes don't trigger full rebuilds

### Memory Usage
- **Reduced compiler memory** through fewer redundant header parses
- **Template instantiation efficiency** via consolidated headers
- **Cache-friendly compilation** with related headers co-located

## Migration Guide

### From Individual Headers to Consolidated
```cpp
// OLD PATTERN (in distribution headers):
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <span>
#include <vector>
#include "../core/distribution_base.h"
#include "../core/error_handling.h"
#include "../core/essential_constants.h"
#include "../platform/simd.h"
#include "../platform/parallel_execution.h"

// NEW PATTERN (Phase 2):
#include "../common/distribution_common.h"         // UPDATED PATH - replaces first 8 includes
#include "distribution_platform_common.h"         // Replaces platform includes
```

### Gradual Migration Strategy
1. **Start with new code** - Use consolidated headers for all new implementations
2. **Update on modification** - Switch to consolidated headers when modifying existing code
3. **Full migration** - Systematic update of all headers (already completed for distributions)

## Performance Metrics

### Achieved Improvements

#### Phase 1 (Cache Consolidation)
- **6 files changed**: 722 insertions, 29 deletions in consolidation
- **100% build success** with zero functionality loss
- **100% test pass rate** after consolidation

#### Phase 2 (Common Header Reorganization)
- **16 header files** in `include/common/` for shared functionality
- **All include paths updated** across codebase (30+ files affected)
- **Cache infrastructure** maintained in `core/distribution_cache.h`
- **Zero functionality loss** with improved organization
- **100% test pass rate** maintained throughout reorganization

### Expected Benefits
- **15-25% faster builds** for incremental changes
- **10-15% improvement** in clean build times
- **60% reduction** in redundant includes across distribution headers
- **Improved cache locality** in compilation process

## Troubleshooting

### Common Issues

#### Missing Symbol Errors
```cpp
// If you see undefined symbols after switching to consolidated headers:
// 1. Check if you need additional specific includes
// 2. Verify the consolidation includes what you need
// 3. Add specific headers for specialized functionality

// Example fix:
#include "distribution_common.h"        // Provides most functionality
#include <tuple>                       // Add if you use tuple returns
```

#### Compilation Errors
```cpp
// If compilation fails with the new headers:
// 1. Ensure you're using the correct consolidated header
// 2. Check for any specialized dependencies
// 3. Verify template instantiation requirements

// For platform-specific issues:
#include "distribution_platform_common.h"  // Standard platform support
#include "../platform/simd_policy.h"      // If you need specific SIMD policies
```

#### Performance Regressions
```cpp
// If you experience performance issues:
// 1. Call initialization once at startup
libstats::initialize_performance_systems();

// 2. Verify SIMD detection is working
std::cout << "SIMD level: " << libstats::cpu::best_simd_level() << std::endl;

// 3. Check that caching is enabled
// (Automatically enabled in consolidated headers)
```

## Future Considerations

### Planned Enhancements
1. **Tools Header Consolidation** - Similar consolidation for tools/ directory
2. **Test Framework Integration** - Standardized test header patterns
3. **Example Templates** - Common patterns for example development
4. **Additional Platform Support** - Extended platform-specific optimizations

### Extension Points
- **New Distribution Types** - Framework ready for additional distributions
- **Custom Platform Headers** - Easy to add new platform-specific capabilities
- **Specialized Constants** - Simple to add new constants categories
- **Performance Optimizations** - Framework supports new optimization strategies

## Conclusion

The libstats header architecture provides a **balanced approach** to code organization that:

- **Reduces complexity** through thoughtful consolidation
- **Preserves maintainability** via clear separation of concerns
- **Optimizes performance** at both compile-time and runtime
- **Supports growth** with extensible architectural patterns

For most development scenarios, using the consolidated headers (`distribution_common.h` and `distribution_platform_common.h`) provides optimal results with minimal complexity.

---

**Document Version**: 2.1
**Last Updated**: 2025-09-01
**Covers**: Phase 1 cache consolidation, Phase 2 common header reorganization, and updated usage guidelines
**Next Review**: After v0.13.0 pre-release work
