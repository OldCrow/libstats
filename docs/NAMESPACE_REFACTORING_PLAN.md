# Namespace Refactoring Plan for libstats v0.11.0

## Executive Summary

The libstats project currently has severe namespace fragmentation with approximately 160+ namespaces. This document outlines a comprehensive plan to consolidate these into 3-5 well-organized namespaces for better maintainability and user experience.

## Current State Analysis

### Namespace Inventory

Based on codebase analysis, we have identified the following namespace categories:

#### 1. Main Project Namespaces
- `libstats` - Current main namespace
- `libstats::*` - Various nested namespaces

#### 2. Constants Namespaces (Highly Fragmented)
- `libstats::constants`
- `libstats::constants::math`
- `libstats::constants::parallel`
- `libstats::constants::platform`
- `libstats::constants::precision`
- `libstats::constants::probability`
- `libstats::constants::simd`
- `libstats::constants::statistical`
- `libstats::constants::thresholds`

#### 3. Architecture/Platform Specific
- Platform detection: `cpu`, `intel`, `amd`, `arm`, `apple_silicon`
- SIMD variants: `sse`, `avx`, `avx2`, `avx512`, `neon`, `fallback`
- Memory: `memory`, `prefetch`, `bandwidth`, `alignment`, `cache`
- Threading: `parallel`, `execution_policy`

#### 4. Mathematical/Statistical
- `libstats::math`
- `libstats::validation`
- `libstats::safety`
- Statistical tests: `chi_square`, `kolmogorov_smirnov`, `anderson_darling`, `shapiro_wilk`
- Distributions: `distributions`, `poisson`, `normal`

#### 5. Performance/Optimization
- `libstats::performance`
- `libstats::performance::characteristics`
- `benchmark`, `tuning`, `optimization`, `adaptive`

#### 6. Testing/Tools
- `libstats::testing`
- `libstats::tools`
- `libstats::tools::display`
- `libstats::tools::strings`

#### 7. Implementation Details
- `detail` (inconsistently used)
- Various utility namespaces: `common`, `algorithm_utils`, `string_utils`, `vector_utils`

## Proposed New Structure

### Target Namespaces (3-5 total)

1. **`stats`** - Main public API namespace
   - All distributions (Gaussian, Exponential, Uniform, Poisson, Discrete, Gamma)
   - Public constants needed by users
   - Public utilities (Result, validation functions)
   - Main factory functions

2. **`stats::detail`** - Implementation details
   - Internal constants (most of current constants:: hierarchy)
   - Internal utilities
   - Memory management internals
   - Cache implementations
   - Performance dispatch internals

3. **`stats::test`** - Testing infrastructure
   - Test utilities
   - Test fixtures
   - Benchmark infrastructure

4. **`stats::simd`** (optional, for explicit SIMD control)
   - SIMD policy types
   - Explicit SIMD operations
   - SIMD detection utilities

5. **`stats::arch`** (optional, for architecture-specific code)
   - CPU feature detection
   - Platform-specific optimizations
   - Architecture-specific constants

## Detailed Mapping

### From `libstats` to `stats`

| Current | New | Notes |
|---------|-----|-------|
| `libstats::GaussianDistribution` | `stats::GaussianDistribution` | Main public classes |
| `libstats::ExponentialDistribution` | `stats::ExponentialDistribution` | |
| `libstats::UniformDistribution` | `stats::UniformDistribution` | |
| `libstats::PoissonDistribution` | `stats::PoissonDistribution` | |
| `libstats::DiscreteDistribution` | `stats::DiscreteDistribution` | |
| `libstats::GammaDistribution` | `stats::GammaDistribution` | |
| `libstats::Result<T>` | `stats::Result<T>` | Error handling |
| `libstats::initialize_performance_systems()` | `stats::initialize_performance_systems()` | |

### Constants Consolidation

| Current | New | Notes |
|---------|-----|-------|
| `libstats::constants::math::*` | `stats::detail::math_constants::*` | Internal use |
| `libstats::constants::precision::*` | `stats::detail::precision::*` | |
| `libstats::constants::probability::*` | `stats::detail::probability::*` | |
| `libstats::constants::statistical::*` | `stats::detail::statistical::*` | |
| `libstats::constants::simd::*` | `stats::detail::simd_constants::*` | |
| `libstats::constants::parallel::*` | `stats::detail::parallel_constants::*` | |
| `libstats::constants::thresholds::*` | `stats::detail::thresholds::*` | |

### Platform/Architecture Code

| Current | New | Notes |
|---------|-----|-------|
| `cpu::*`, `intel::*`, `amd::*`, etc. | `stats::arch::*` | If needed for public API |
| SIMD namespaces (`sse::*`, `avx::*`, etc.) | `stats::detail::simd_impl::*` | Internal implementation |
| `libstats::simd::*` | `stats::simd::*` | Public SIMD API if needed |

### Performance/Optimization

| Current | New | Notes |
|---------|-----|-------|
| `libstats::performance::*` | `stats::detail::performance::*` | Internal |
| `benchmark::*` | `stats::detail::benchmark::*` | |
| `tuning::*`, `adaptive::*` | `stats::detail::adaptive::*` | |

### Testing Infrastructure

| Current | New | Notes |
|---------|-----|-------|
| `libstats::testing::*` | `stats::test::*` | Test utilities |
| Test-specific utilities | `stats::test::utils::*` | |

## Collision Resolution Strategy

### Identified Collision Types

1. **Constant Name Collisions**
   - Multiple `BLOCK_SIZE`, `THRESHOLD`, `MIN_*`, `MAX_*` constants across namespaces
   - Example: `simd::DEFAULT_BLOCK_SIZE` vs `matrix::L1_BLOCK_SIZE` vs `parallel::DEFAULT_GRAIN_SIZE`

2. **Function Name Collisions**
   - Similar utility functions in different namespaces (e.g., `validate()`, `check()`, `initialize()`)
   - Performance-related functions with same names but different contexts

3. **Type Name Collisions**
   - Common type names like `Features`, `Options`, `Config` used in multiple contexts

### Resolution Strategies

#### Strategy 1: Contextual Prefixing
Add context-specific prefixes to avoid collisions:

```cpp
// Before (multiple namespaces)
namespace simd { constexpr size_t DEFAULT_BLOCK_SIZE = 8; }
namespace matrix { constexpr size_t L1_BLOCK_SIZE = 64; }

// After (consolidated in stats::detail)
namespace stats::detail {
    constexpr size_t SIMD_DEFAULT_BLOCK_SIZE = 8;
    constexpr size_t MATRIX_L1_BLOCK_SIZE = 64;
}
```

#### Strategy 2: Nested Namespace Preservation
For highly cohesive groups, preserve one level of nesting:

```cpp
namespace stats::detail {
    namespace simd_constants {
        constexpr size_t DEFAULT_BLOCK_SIZE = 8;
    }
    namespace matrix_constants {
        constexpr size_t L1_BLOCK_SIZE = 64;
    }
}
```

#### Strategy 3: Struct-based Grouping
Use structs as namespace-like containers:

```cpp
namespace stats::detail {
    struct SimdConstants {
        static constexpr size_t DEFAULT_BLOCK_SIZE = 8;
        static constexpr size_t MAX_BLOCK_SIZE = 64;
    };
    
    struct MatrixConstants {
        static constexpr size_t L1_BLOCK_SIZE = 64;
        static constexpr size_t L2_BLOCK_SIZE = 256;
    };
}
```

### Collision Resolution Rules

1. **Public API (stats namespace)**
   - No prefixing for user-facing APIs
   - Resolve by choosing the most general/appropriate name
   - Document any breaking changes

2. **Implementation Details (stats::detail)**
   - Use contextual prefixing for constants
   - Preserve one level of nesting for highly cohesive groups
   - Use struct-based grouping for related constants

3. **Architecture-specific (stats::arch)**
   - Keep platform prefixes (e.g., `avx_`, `neon_`, `sse_`)
   - Group by architecture using nested namespaces or structs

### Specific Collision Resolutions

| Collision | Current Locations | Resolution |
|-----------|------------------|------------|
| `DEFAULT_BLOCK_SIZE` | `simd::`, `constants_bridge::` | → `stats::detail::SIMD_DEFAULT_BLOCK_SIZE` |
| `MAX_BLOCK_SIZE` | `simd::`, `matrix::` | → `stats::detail::SIMD_MAX_BLOCK_SIZE`, `stats::detail::MATRIX_MAX_BLOCK_SIZE` |
| `MIN_SIZE` variants | Multiple locations | → Add context prefix: `SIMD_MIN_SIZE`, `PARALLEL_MIN_SIZE`, etc. |
| `Features` struct | `cpu::`, `platform::` | → `stats::arch::CpuFeatures`, `stats::arch::PlatformFeatures` |
| `initialize()` | Multiple namespaces | → Context-specific: `initialize_simd()`, `initialize_performance()` |

### Automated Detection

Before refactoring, run collision detection:

```bash
# Script to detect potential collisions
#!/bin/bash
# Find all constant declarations
grep -h "constexpr\|enum\|class\|struct" include/**/*.h | \
    sed 's/.*\(\w\+\)\s*=.*/\1/' | \
    sort | uniq -d > potential_collisions.txt

# Find all function declarations
grep -h "^[^/]*\w\+\s*\(" include/**/*.h | \
    sed 's/.*\s\(\w\+\)\s*(.*/\1/' | \
    sort | uniq -d >> potential_collisions.txt
```

### Collision Testing Process

1. **Pre-refactoring Analysis**
   ```bash
   # Create collision detection script
   scripts/detect_namespace_collisions.sh
   ```

2. **Compile-time Detection**
   - Rely on compiler errors to catch unresolved collisions
   - Use `-Wshadow` flag to detect shadowing issues

3. **Runtime Verification**
   - Ensure all tests pass with no ambiguity errors
   - Check that performance characteristics remain unchanged

### Priority Order for Collision Resolution

1. **Preserve Public API names** (highest priority)
   - User-facing names in `stats::` namespace take precedence
   - Internal conflicts must yield to public API needs

2. **Maintain semantic clarity**
   - Choose names that clearly indicate purpose
   - Avoid overly generic names in consolidated namespace

3. **Minimize code churn**
   - Where possible, use prefixing over renaming
   - Keep changes localized to implementation files

## Implementation Plan

### Phase 1: Preparation (Week 1)
1. **Create namespace aliases** for backward compatibility
2. **Document migration path** for users
3. **Set up automated refactoring scripts**
4. **Create comprehensive test suite** to verify no functional changes

### Phase 2: Core Refactoring (Week 2-3)
1. **Headers First**
   - Start with include/libstats.h (main public header)
   - Refactor distribution headers (gaussian.h, exponential.h, etc.)
   - Update common headers
   
2. **Implementation Files**
   - Update corresponding .cpp files
   - Maintain consistency between headers and implementation

3. **Platform-Specific Code**
   - Consolidate SIMD implementations
   - Unify architecture-specific code

### Phase 3: Testing and Tools (Week 4)
1. **Update test files** to use new namespaces
2. **Update tools and examples**
3. **Update build system** if needed
4. **Run comprehensive tests** on all platforms

### Phase 4: Documentation and Cleanup (Week 5)
1. **Update all documentation**
2. **Remove deprecated namespace aliases** (in next major version)
3. **Update WARP.md and other project documentation**
4. **Create migration guide for users**

## Migration Strategy

### For Library Users

```cpp
// Old code
#include "libstats.h"
using namespace libstats;

auto dist = GaussianDistribution::create(0.0, 1.0);

// New code (v0.11.0+)
#include "libstats.h"
using namespace stats;

auto dist = GaussianDistribution::create(0.0, 1.0);
```

### Backward Compatibility

For v0.11.0, we'll provide namespace aliases:

```cpp
// In libstats.h
namespace stats {
    // All new code here
}

// Backward compatibility
namespace libstats = stats;
```

### Internal Code Changes

```cpp
// Old internal code
namespace libstats {
namespace constants {
namespace simd {
    constexpr size_t BLOCK_SIZE = 8;
}}}

// New internal code
namespace stats {
namespace detail {
    namespace simd_constants {
        constexpr size_t BLOCK_SIZE = 8;
    }
}}
```

## Risk Mitigation

1. **Extensive Testing**: Run full test suite after each major change
2. **Gradual Migration**: Use namespace aliases for smooth transition
3. **Clear Documentation**: Provide migration guides and examples
4. **Version Control**: All changes in dedicated branch with ability to revert
5. **Platform Testing**: Test on macOS, Linux, Windows with various compilers

## Success Metrics

1. **Reduced namespace count**: From ~160 to 3-5 namespaces
2. **Cleaner API**: Simpler namespace hierarchy for users
3. **Better organization**: Clear separation of public API and implementation details
4. **Maintained performance**: No regression in benchmarks
5. **Backward compatibility**: Existing code continues to work with aliases

## Timeline

- **Week 1**: Preparation and planning
- **Week 2-3**: Core refactoring
- **Week 4**: Testing and tools update  
- **Week 5**: Documentation and final cleanup
- **Week 6**: Review, testing, and merge preparation

## Notes for Implementation

1. **Start Small**: Begin with a single distribution as a pilot
2. **Automate**: Use tools like `sed`, `awk`, or custom scripts for bulk changes
3. **Review Carefully**: Each file change should be reviewed for correctness
4. **Test Continuously**: Run tests after each significant change
5. **Document Changes**: Keep detailed log of what was changed and why

## Checklist for Each File

- [ ] Update namespace declarations
- [ ] Update using directives
- [ ] Update forward declarations
- [ ] Update friend declarations if any
- [ ] Update documentation comments
- [ ] Verify includes are still correct
- [ ] Run compilation test
- [ ] Run unit tests for that component
