# Namespace Architecture Proposal - libstats

## Current State Analysis

### Primary Namespace Structure
```
stats::
├── arch::         // Architecture-specific optimizations
│   ├── simd::     // SIMD feature detection and utilities
│   ├── parallel:: // Parallel processing with architecture-specific tuning
│   │   ├── sse::
│   │   ├── avx::
│   │   ├── avx2::
│   │   ├── avx512::
│   │   ├── neon::
│   │   └── fallback::
│   ├── cache::    // UNUSED - can be deleted
│   └── memory::   // Memory access patterns (4-level nesting)
└── simd::         // Currently minimal usage (closing braces only)
```

### Key Findings

1. **SIMD Organization Confusion**:
   - `stats::arch::simd::` contains SIMD feature detection utilities
   - `stats::simd::` namespace exists but appears mostly empty
   - Parallel architecture namespaces (sse, avx, etc.) contain tuning constants, not SIMD operations

2. **Deliberate Name Collisions**:
   - Architecture sub-namespaces (sse, avx, avx2, etc.) deliberately use same constant names
   - Conditional compilation selects appropriate namespace at build time
   - Flattening would break this design pattern

3. **Local detail Namespaces**:
   - Multiple files have local `namespace detail {}` blocks
   - These are file-local, not in stats:: hierarchy
   - Used for implementation details in: performance_dispatcher.cpp, validation.cpp, performance_history.cpp, safety.cpp, math_utils.cpp, system_capabilities.cpp

## Proposed Architecture

### Option A: Clear Separation of Concerns (RECOMMENDED)

```
stats::
├── arch::              // Machine characteristics & tuning parameters
│   ├── cpu::          // CPU-specific parameters
│   │   ├── intel::    // Intel-specific tuning
│   │   ├── amd::      // AMD-specific tuning
│   │   ├── arm::      // ARM-specific tuning
│   │   └── apple_silicon:: // Apple Silicon specific
│   ├── memory::       // Memory hierarchy (flattened to 3 levels)
│   │   ├── prefetch:: // All prefetch constants with prefixes
│   │   ├── access::   // All access patterns with prefixes
│   │   └── allocation:: // Allocation strategies
│   └── parallel::     // Parallel processing tuning (keep as-is)
│       ├── sse::      // SSE-specific parallel tuning
│       ├── avx::      // AVX-specific parallel tuning
│       ├── avx2::     // AVX2-specific parallel tuning
│       ├── avx512::   // AVX512-specific parallel tuning
│       ├── neon::     // NEON-specific parallel tuning
│       └── fallback:: // Generic fallback tuning
├── simd::             // SIMD operations & algorithms
│   ├── ops::          // SIMD operations (dot_product, vector_add, etc.)
│   ├── dispatch::     // Runtime SIMD dispatch
│   └── utils::        // SIMD utilities (alignment, detection)
├── tests::            // Test infrastructure and utilities
│   ├── constants::    // Test-specific thresholds and parameters
│   ├── fixtures::     // Reusable test fixtures and data generators
│   ├── validators::   // Test validation utilities
│   └── benchmarks::   // Benchmark-specific utilities
└── detail::           // Library-wide implementation details
```

### Option B: Unified Architecture Namespace

```
stats::
├── arch::              // All architecture-specific code
│   ├── simd::         // SIMD operations & dispatch
│   │   ├── ops::      // SIMD operations
│   │   ├── dispatch:: // Runtime dispatch
│   │   └── utils::    // Detection & utilities
│   ├── parallel::     // Parallel tuning (keep current structure)
│   │   ├── sse::
│   │   ├── avx::
│   │   ├── avx2::
│   │   ├── avx512::
│   │   ├── neon::
│   │   └── fallback::
│   ├── cpu::          // CPU characteristics
│   └── memory::       // Memory hierarchy (flattened)
└── detail::           // Library-wide implementation details
```

## Migration Strategy

### Phase 3A: Remove Unused Namespaces (IMMEDIATE)
1. Delete `stats::arch::cache::` and all sub-namespaces - completely unused
2. Update any documentation references

### Phase 3B: Reorganize SIMD (HIGH PRIORITY)
1. Move SIMD detection from `stats::arch::simd::` to `stats::simd::utils::`
2. Create `stats::simd::ops::` for SIMD operations
3. Create `stats::simd::dispatch::` for runtime dispatch
4. Update all references

### Phase 3C: Flatten Memory Namespaces (MEDIUM PRIORITY)
1. Flatten 4-level memory namespaces to 3-level maximum
2. Add prefixes to prevent naming conflicts:
   - `DISTANCE_*`, `STRATEGY_*`, `TIMING_*` for prefetch
   - `BANDWIDTH_*`, `LAYOUT_*`, `NUMA_*` for access
   - `GROWTH_*` for allocation
3. Update ~59 usage sites

### Phase 3D: Consolidate detail Namespaces (LOW PRIORITY)
1. Move file-local `detail` namespaces to `stats::detail::`
2. Add sub-namespaces if needed for organization
3. Or keep as file-local if truly implementation-specific

### Phase 3E: Create Test Infrastructure Namespace (MEDIUM PRIORITY)
1. Create `stats::tests::` namespace hierarchy
2. Move test-specific constants from main library to `stats::tests::constants::`
3. Create `stats::tests::fixtures::` for shared test data generators
4. Create `stats::tests::validators::` for common validation utilities
5. Create `stats::tests::benchmarks::` for benchmark-specific helpers

## Rationale

### Why Keep parallel:: Sub-namespaces
- **Deliberate name collisions**: Same constants with different values per architecture
- **Conditional compilation**: Build system selects appropriate namespace
- **Clean API**: Users reference constants without architecture suffixes
- **Working design**: Currently functional, no benefit to changing

### Why Separate simd:: from arch::
- **Clear separation**: SIMD operations vs. architecture tuning
- **Future extensibility**: Room for SIMD algorithms, patterns
- **Logical grouping**: SIMD dispatch/ops together, tuning parameters together

### Why Flatten Memory Namespaces
- **Too deep**: 4-level nesting is excessive
- **Low collision risk**: Prefixing prevents conflicts
- **Simpler navigation**: Easier to find and use constants
- **Consistent depth**: Match parallel:: namespace depth

### Why Add tests:: Namespace
- **Clear separation**: Test code separate from production code
- **Test-specific constants**: Tolerances, thresholds, iteration counts
- **Shared test utilities**: Fixtures, validators, data generators
- **No production dependencies**: Test namespace never used in production builds
- **Better organization**: All test infrastructure in one place

## Implementation Checklist

- [ ] Phase 3A: Remove unused cache namespaces
- [ ] Phase 3B: Reorganize SIMD namespaces
- [ ] Phase 3C: Flatten memory to 3 levels
- [ ] Phase 3D: Consolidate detail namespaces (optional)
- [ ] Phase 3E: Create tests:: namespace infrastructure
- [ ] Update all documentation
- [ ] Run full test suite
- [ ] Verify zero warnings

## Benefits

1. **Clearer Organization**: Separation of SIMD ops from architecture tuning
2. **Reduced Complexity**: Flatter namespace hierarchy
3. **Better Discoverability**: Logical grouping of related functionality
4. **Future-Proof**: Room for CPU-specific optimizations
5. **Maintained Compatibility**: Preserves working parallel:: design
6. **Test Isolation**: Clear separation between production and test code
7. **Test Reusability**: Shared test utilities in well-defined namespace

## Risks & Mitigations

1. **Risk**: Breaking conditional compilation
   - **Mitigation**: Keep parallel:: sub-namespaces unchanged

2. **Risk**: Naming conflicts when flattening
   - **Mitigation**: Use clear prefixes for flattened constants

3. **Risk**: Large refactoring effort
   - **Mitigation**: Phase approach, start with low-risk deletions

## Decision Required

**Recommend Option A with tests:: addition**: Clear separation between SIMD operations (`stats::simd::`) and architecture tuning (`stats::arch::`), plus dedicated test infrastructure (`stats::tests::`). This provides the clearest mental model, best extensibility, and proper test isolation.

## Example Test Namespace Usage

```cpp
// In test files
namespace stats::tests::constants {
    inline constexpr double DEFAULT_TOLERANCE = 1e-10;
    inline constexpr double SIMD_SPEEDUP_MIN_EXPECTED = 1.5;
    inline constexpr size_t DEFAULT_BENCHMARK_ITERATIONS = 1000;
    inline constexpr size_t LARGE_DATASET_SIZE = 1'000'000;
}

namespace stats::tests::fixtures {
    std::vector<double> generate_random_data(size_t n);
    std::vector<double> generate_aligned_data(size_t n);
}

namespace stats::tests::validators {
    bool approximately_equal(double a, double b, double tol);
    bool verify_simd_result(const std::vector<double>& expected,
                           const std::vector<double>& actual);
}
```
