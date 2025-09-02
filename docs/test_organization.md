# Test Organization Plan for libstats

## Tests for CTest Suite (Run with `make test` or `ctest`)

These tests should be included in the standard test suite, organized by dependency level:

### Level 0: Foundational Tests (Platform & Constants)
1. `test_constants` - All constants validation
2. `test_cpu_detection` - CPU feature detection
3. `test_simd_comprehensive` - SIMD operations correctness
4. `test_simd_policy` - SIMD policy decisions
5. `test_platform_optimizations` - Platform-specific optimizations

### Level 1: Core Infrastructure Tests
6. `test_numerical_safety` - Consolidated safe numerical operations including log-space arithmetic
7. `test_error_handling_comprehensive` - Unified error handling and API correctness
8. `test_math_comprehensive` - Consolidated math utilities and vectorized operations
9. `test_validation_enhanced` - Enhanced parameter validation

### Level 2: Statistical Infrastructure
10. `test_goodness_of_fit` - Goodness-of-fit tests

### Level 3: Parallel Infrastructure Tests
13. `test_thread_pool` - Thread pool implementation
14. `test_work_stealing_pool` - Work-stealing pool
15. `test_parallel_execution_integration` - Parallel execution integration
16. `test_benchmark_basic` - Basic benchmarking infrastructure

### Level 4: Performance Framework Tests
17. `test_performance_history` - Performance history tracking
18. `test_performance_dispatcher` - Performance auto-dispatch
19. `test_system_capabilities` - System capability detection
20. `test_performance_initialization` - Performance system init

### Level 5: Distribution Implementation Tests (Core Functionality)
21. `test_gaussian_basic` - Gaussian distribution basic
22. `test_exponential_basic` - Exponential distribution basic
23. `test_uniform_basic` - Uniform distribution basic
24. `test_poisson_basic` - Poisson distribution basic
25. `test_discrete_basic` - Discrete distribution basic
26. `test_gamma_basic` - Gamma distribution basic
27. `test_atomic_parameters` - Atomic parameter updates

### Level 6: Enhanced Distribution Tests (Advanced Features)
28. `test_gaussian_enhanced` - Gaussian advanced features
29. `test_exponential_enhanced` - Exponential advanced features
30. `test_uniform_enhanced` - Uniform advanced features
31. `test_poisson_enhanced` - Poisson advanced features
32. `test_discrete_enhanced` - Discrete advanced features
33. `test_gamma_enhanced` - Gamma advanced features

### Level 7: Integration Tests
34. `test_dynamic_linking` - Dynamic library linking

## On-Demand Tests (Compile but Don't Run in CTest)

These tests should be compiled and available but NOT included in the standard test suite:

### Stress Tests (Time-consuming, for manual verification)
- `test_copy_move_stress` - Intensive concurrent copy/move operations (5+ seconds)
- `test_parallel_execution_comprehensive` - Comprehensive parallel testing (time-consuming)
- `test_benchmark` - Full benchmark suite (vs basic)

### Diagnostic Tests (For debugging specific issues)
- `test_copy_move_fix` - Verifies specific bug fixes
- `test_parallel_compilation` - Tests compilation with different parallel backends

### Dynamic Library Variants (Redundant with static tests)
- `test_gaussian_basic_dynamic`
- `test_exponential_basic_dynamic`

## Recommended CTest Execution Order

```cmake
# Level 0: Foundational (no dependencies)
test_constants
test_cpu_detection
test_simd_comprehensive
test_simd_policy
test_platform_optimizations

# Level 1: Core utilities (depends on Level 0)
test_numerical_safety
test_error_handling_comprehensive
test_math_comprehensive
test_validation_enhanced

# Level 2: Statistical infrastructure (depends on Level 0-1)
test_goodness_of_fit

# Level 3: Parallel infrastructure (depends on Level 0-2)
test_thread_pool
test_work_stealing_pool
test_parallel_execution_integration
test_benchmark_basic

# Level 4: Performance framework (depends on Level 0-3)
test_performance_history
test_performance_dispatcher
test_system_capabilities
test_performance_initialization

# Level 5: Basic distribution tests (depends on Level 0-4)
test_gaussian_basic
test_exponential_basic
test_uniform_basic
test_poisson_basic
test_discrete_basic
test_gamma_basic
test_atomic_parameters

# Level 6: Enhanced distribution tests (depends on Level 0-5)
test_gaussian_enhanced
test_exponential_enhanced
test_uniform_enhanced
test_poisson_enhanced
test_discrete_enhanced
test_gamma_enhanced

# Level 7: Integration tests (depends on all)
test_dynamic_linking
```

## Rationale for On-Demand Tests

1. **Stress Tests**: Take significant time (5+ seconds each) and are primarily for verifying thread safety under extreme conditions
2. **Diagnostic Tests**: Target specific bug fixes or compilation scenarios, not needed for regular validation
3. **Dynamic Variants**: Test the same functionality as static tests, redundant for regular runs
4. **Comprehensive Tests**: Full versions of tests that have "basic" counterparts in the main suite

## Running On-Demand Tests

```bash
# Run specific on-demand test
./build/tests/test_copy_move_stress

# Run all stress tests
./build/tests/test_copy_move_stress && ./build/tests/test_parallel_execution_comprehensive

# Run diagnostic tests
./build/tests/test_copy_move_fix
./build/tests/test_parallel_compilation
```
