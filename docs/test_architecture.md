# libstats Test Architecture

**Authoritative Developer Reference for Testing Infrastructure**

## Overview

The libstats C++20 statistical computing library employs a sophisticated, dependency-aware testing architecture with dual standardized frameworks that ensure comprehensive validation from foundational components to high-level distribution implementations. This document serves as the definitive guide for developers working with the complete test infrastructure.

### Key Design Principles

1. **Dependency Hierarchy**: Tests are organized in levels (0-4) matching the library's dependency structure
2. **Mixed Test Frameworks**: Strategic use of both standalone tests and GTest for different purposes
3. **Performance Validation**: Built-in SIMD speedup verification and performance regression detection
4. **Maintainability**: Clear separation between basic functionality and advanced features
5. **Scalability**: Template pattern for extending to new distributions

## Test Framework Strategy

### Why Mixed Test Frameworks?

The libstats project strategically uses two testing approaches:

#### Standalone Tests (`main()` function)
- **Used for**: Service-level components, basic functionality validation, performance benchmarks
- **Benefits**:
  - Minimal dependencies (no GTest requirement)
  - Fast compilation and execution
  - Direct performance measurement capability
  - Simple debugging and profiling
- **Drawbacks**: Limited assertion capabilities, manual result reporting

#### GTest-Based Tests
- **Used for**: Complex statistical methods, comprehensive distribution testing, advanced features
- **Benefits**:
  - Rich assertion library with detailed failure reporting
  - Test fixtures for setup/teardown
  - Parameterized tests for multiple scenarios
  - Better integration with CI/CD systems
- **Drawbacks**: Additional dependency, slightly slower execution

### Current Framework Distribution
- **Standalone tests**: ~25 tests (service infrastructure, basic distribution functionality)
- **GTest tests**: ~25 tests (enhanced distribution tests with SIMD timing assertions; require GTest)
- **Correctness suite** (`-LE timing|benchmark`): 34 tests — always parallel-safe
- **Timing suite** (`-L timing`): 14 tests — run serially for accurate speedup measurement
- **Benchmark**: 1 test (`benchmark_simd_all`)
- GTest tests silently skipped when GTest is absent

## Test Organization by Dependency Levels

### Level 0: Foundational Tests (5 tests)
**Purpose**: Validate fundamental building blocks with no internal dependencies

- **`test_constants`** - Mathematical constants and precision tolerances from `constants.h`
- **`test_cpu_detection`** - Runtime CPU feature detection from `cpu_detection.h`
- **`test_simd_comprehensive`** - SIMD operations correctness and policy from `simd.h`
- **`test_simd_policy`** - SIMD policy decisions and level detection
- **`test_platform_optimizations`** - Platform-specific optimizations from `simd.h`

### Level 1: Core Infrastructure Tests (4 tests)
**Purpose**: Validate core mathematical and safety infrastructure

- **`test_numerical_safety`** - Consolidated safe numerical operations including log-space arithmetic
- **`test_error_handling_comprehensive`** - Unified error handling and dual API correctness
- **`test_math_comprehensive`** (**GTest**) - Consolidated math utilities and vectorized operations
- **`test_validation_enhanced`** - Enhanced statistical validation features

### Level 2: Core Framework Tests
**Purpose**: Distribution base framework (abstract base class)

- **Note**: No direct tests for `distribution_base.h` - validated through Level 4 distribution implementations

### Level 2: Statistical Infrastructure (1 test)
- **`test_goodness_of_fit`** - Statistical validation and goodness-of-fit testing

### Level 3: Parallel Infrastructure Tests (4 tests)
**Purpose**: Validate parallel computation and performance measurement

- **`test_thread_pool`** - Basic thread pool implementation
- **`test_work_stealing_pool`** - Work-stealing thread pool
- **`test_parallel_execution_integration`** - C++20 parallel execution policies
- **`test_benchmark_basic`** - Performance measurement utilities

### Level 4: Performance Framework Tests (4 tests)
- **`test_performance_history`** (**GTest**) - Performance history tracking
- **`test_performance_dispatcher`** (**GTest, timing**) - Smart auto-dispatch
- **`test_system_capabilities`** (**GTest**) - System capability detection
- **`test_performance_initialization`** (**GTest**) - Performance system initialization

### Level 5: Distribution Basic Tests (14 tests + atomic)
**Purpose**: Validate concrete statistical distribution implementations

- **`test_gaussian_basic`** - Fundamental operations (PDF, CDF, quantiles, fitting)
- **`test_exponential_basic`** - Fundamental operations
- **`test_uniform_basic`** - Fundamental operations
- **`test_poisson_basic`** - Fundamental operations (incl. exact methods)
- **`test_discrete_basic`** - Fundamental operations
- **`test_gamma_basic`** - Fundamental operations
- **`test_chi_squared_basic`** - Fundamental operations
- **`test_student_t_basic`** - Fundamental operations
- **`test_beta_basic`** - Fundamental operations
- **`test_lognormal_basic`** - Fundamental operations
- **`test_pareto_basic`** - Fundamental operations
- **`test_weibull_basic`** - Fundamental operations
- **`test_rayleigh_basic`** - Fundamental operations
- **`test_von_mises_basic`** - Fundamental operations (circular, Bessel cache)
- **`test_atomic_parameters`** - Lock-free parameter access across all distributions

### Level 6: Distribution Enhanced Tests (14 tests, GTest, timing label)
- **`test_gaussian_enhanced`** (**GTest, timing**) - Confidence intervals, bootstrap, KS/AD, SIMD batch
- **`test_exponential_enhanced`** (**GTest, timing**)
- **`test_uniform_enhanced`** (**GTest, timing**)
- **`test_poisson_enhanced`** (**GTest, timing**)
- **`test_discrete_enhanced`** (**GTest, timing**)
- **`test_gamma_enhanced`** (**GTest, timing**)
- **`test_chi_squared_enhanced`** (**GTest, timing**)
- **`test_student_t_enhanced`** (**GTest, timing**)
- **`test_beta_enhanced`** (**GTest, timing**)
- **`test_lognormal_enhanced`** (**GTest, timing**) - SIMD pipeline correctness and speedup
- **`test_pareto_enhanced`** (**GTest, timing**) - SIMD pipeline correctness and speedup
- **`test_weibull_enhanced`** (**GTest, timing**) - SIMD pipeline correctness and speedup
- **`test_rayleigh_enhanced`** (**GTest, timing**) - SIMD pipeline correctness and speedup
- **`test_von_mises_enhanced`** (**GTest, timing**) - PARALLEL correctness; Bessel cache behaviour

### Level 7: Integration Tests (3 tests)
**Purpose**: Dynamic library linking and cross-cutting functionality

- **`test_dynamic_linking`** - Dynamic library linking validation
- **`test_gaussian_basic_dynamic`** - Gaussian with dynamic library (Release CRT required; see AGENTS.md)
- **`test_exponential_basic_dynamic`** - Exponential with dynamic library

## Consolidated Testing Strategy

### Naming Conventions

- `test_*_basic.cpp` — standalone (no GTest required); covers the core 8-section test structure
- `test_*_enhanced.cpp` — GTest, labelled `timing`; covers SIMD correctness and speedup assertions
- Service-layer tests follow the same `*_basic` / `*_enhanced` convention where applicable

## Unified Test Infrastructure

### Test Infrastructure Namespace (`stats::tests::`)

#### Overview
All tests now use a unified namespace-organized infrastructure that supports both standalone tests and GTest-based enhanced tests, providing consistent coverage, output formatting, and user experience across all distributions.

#### Framework Components (stats::tests::fixtures)

**BasicTestFormatter Class Provides:**
- **Consistent Output Formatting**: Uniform headers, test start/success messages, completion messages
- **Property Display**: Standardized formatting for numerical properties and results
- **Sample Display**: Consistent formatting for random samples and batch results
- **Error Handling**: Uniform error message formatting
- **Summary Generation**: Standardized test summaries with checkmarks
- **Helper Functions**: Statistical validation, sample generation, utility methods

#### Universal Test Structure

Each basic test follows this standardized 7-10 test structure:

1. **Safe Factory Method**: Test distribution creation and initial properties
2. **PDF/PMF, CDF, and Quantile Functions**: Core probability functions
3. **Random Sampling**: Single and batch sampling validation
4. **Parameter Fitting**: MLE/Method of Moments parameter estimation
5. **Batch Operations**: SIMD-optimized batch probability calculations
6. **Large Batch SIMD Validation**: Performance and consistency verification
7. **Error Handling**: Invalid parameter validation
8. **Distribution-Specific Tests**: Unique features per distribution
9. **Special Cases**: Edge cases and corner scenarios

#### Distribution Coverage

**✅ Gaussian Distribution** — Box-Muller sampling, confidence intervals, bootstrap CIs, KS/AD tests
**✅ Exponential Distribution** — inverse transform, half-life utility, memoryless property
**✅ Uniform Distribution** — range estimation, support validation
**✅ Discrete Distribution** — integer sampling, outcome enumeration, binary special case
**✅ Poisson Distribution** — Knuth (small λ) and Atkinson (large λ) sampling, exact methods
**✅ Gamma Distribution** — Marsaglia-Tsang sampling, shape/rate parametrisation
**✅ Chi-squared Distribution** — delegation to Gamma(ν/2, 1/2); degrees-of-freedom interface
**✅ Student's t Distribution** — log-space SIMD LogPDF/PDF; incomplete beta CDF
**✅ Beta Distribution** — two-log SIMD pipeline; regularized incomplete beta CDF/quantile
**✅ Log-Normal Distribution** — 6-step SIMD LogPDF; closed-form MLE via log-space Gaussian transform
**✅ Pareto Distribution** — 3-step SIMD LogPDF (simplest pipeline); closed-form two-step MLE
**✅ Weibull Distribution** — 8-step SIMD LogPDF; Newton–Raphson profile score MLE; `isExponential()` flag
**✅ Rayleigh Distribution** — 5-step SIMD LogPDF (x² pipeline); closed-form MLE
**✅ Von Mises Distribution** — circular; Bessel-based normaliser cached; Mardia–Jupp + Newton–Raphson MLE

#### Consistent Output Format

All tests produce standardized output with:

```
Testing [Distribution]Distribution Implementation
=================================================

Test 1: Safe factory method
✅ Safe factory creation successful
[Property listings with consistent formatting]

Test 2: PDF, CDF, and quantile functions
[Function evaluations with standard precision]
✅ Test passed successfully

...

🎉 All [Distribution]Distribution tests completed successfully!

=== SUMMARY ===
✓ Safe factory creation and error handling
✓ All distribution properties (specific to distribution)
✓ PDF, Log PDF, CDF, and quantile functions
✓ Random sampling (algorithm-specific details)
✓ Parameter fitting (method-specific details)
✓ Batch operations with SIMD optimization
✓ Large batch SIMD validation
✓ [Distribution-specific features]
```

### Enhanced Test Framework (`tests/include/fixtures.h`)

#### Overview
The enhanced tests use Google Test for comprehensive statistical functionality testing, SIMD correctness verification, speedup assertions, and (for the original six distributions) advanced methods such as confidence intervals, bootstrap CIs, cross-validation, and KS/AD goodness-of-fit tests.

#### Framework Components

**StandardizedBenchmark Class:**
- **Performance Measurement**: Standardized timing for SIMD, parallel, work-stealing, and cache-aware operations
- **Speedup Analysis**: Automatic calculation and reporting of performance improvements
- **Hardware Detection**: Thread count and capability reporting
- **Results Formatting**: Professional tabular output with analysis

**StatisticalTestUtils Class:**
- **Sample Statistics**: Mean and variance calculation utilities
- **Batch Correctness**: Verification that batch operations match individual calls
- **Numerical Validation**: Tolerance-based equality checking
- **Template Support**: Generic utilities for any distribution type

**ThreadSafetyTester Class:**
- **Multi-threaded Sampling**: Concurrent sampling validation across threads
- **Race Condition Detection**: Ensures thread-safe operation
- **Resource Validation**: Verifies proper resource management

**EdgeCaseTester Class:**
- **Extreme Value Handling**: Tests with very large/small input values
- **Empty Batch Operations**: Validates graceful handling of edge cases
- **Boundary Conditions**: Tests at distribution support boundaries

#### Enhanced Test Structure

Each enhanced test includes:

1. **Test Fixtures**: Setup and teardown for consistent test environments
2. **Basic Enhanced Functionality**: Core distribution properties and operations
3. **Copy/Move Semantics**: Proper C++ object lifecycle management
4. **Batch Operations**: Comprehensive SIMD and parallel validation
5. **Performance Benchmarks**: Standardized performance measurement and reporting
6. **Parallel Batch Performance**: Work-stealing and cache-aware optimization tests
7. **Advanced Statistical Methods**: Cross-validation, bootstrap, confidence intervals
8. **Goodness-of-Fit Tests**: Statistical validation (KS, Anderson-Darling, etc.)
9. **Information Criteria**: AIC, BIC, AICc calculations
10. **Thread Safety**: Multi-threaded operation validation
11. **Edge Cases**: Extreme values and boundary condition testing
12. **Numerical Stability**: Precision and accuracy validation

#### Advanced Statistical Coverage

**Statistical Methods Tested:**
- **Cross-Validation**: K-fold and leave-one-out validation
- **Bootstrap Methods**: Parameter confidence interval estimation
- **Bayesian Estimation**: Prior/posterior analysis where applicable
- **Hypothesis Testing**: Statistical significance testing
- **Goodness-of-Fit**: Multiple statistical tests for distribution fitting
- **Information Criteria**: Model selection metrics

**Performance Analysis:**
- **SIMD Optimization**: Batch operation speedup measurement
- **Parallel Execution**: Multi-core performance scaling
- **Work-Stealing**: Advanced parallel algorithm performance
- **Cache-Aware**: Memory hierarchy optimization testing

### Performance Validation Results

All eleven distributions produce measurable SIMD speedup on aligned batches. Headline results
from `simd_verification` on Ivy Bridge AVX (see AGENTS.md for all four architectures):

- **Exponential LogPDF**: 20.8x
- **Uniform CDF**: 25.2x
- **Gamma PDF**: 9.7x
- **Pareto LogPDF**: among the highest (3-step pipeline, single vector_log + two scalars)
- **Log-Normal LogPDF**: strong (6-step pipeline, one vector_log + element-wise square)
- **Overall simd_verification speedup**: 4.10x on AVX, 3.49x on AVX2, 2.31x on NEON, 1.64x on AVX-512
- **Thread safety**: Verified across all implementations
- **Memory safety**: Comprehensive bounds checking and numerical stability

## Running Tests

### All Tests in Dependency Order
```bash
cd build
make run_tests        # correctness only (parallel-safe)
make run_tests_timing # timing/speedup assertions (run serially)
make run_all_tests    # everything
```

### By Dependency Level
```bash
# Level 0: Foundational tests
ctest -R "test_constants|test_cpu_detection|test_simd.*|test_platform.*|test_error_handling_comprehensive"

# Level 1: Core infrastructure tests
ctest -R "test_numerical_safety|test_math_comprehensive|test_goodness_of_fit|test_validation_enhanced"

# Level 3: Parallel infrastructure tests
ctest -R "test_thread_pool|test_work_stealing_pool|test_parallel_execution.*|test_benchmark_basic"

# Level 4: Distribution tests
ctest -R "test_gaussian.*|test_exponential.*" -E "dynamic"

# Cross-cutting: Dynamic linking tests
ctest -R ".*dynamic.*"
```

### Individual GTest Suites
```bash
# Mathematical functions (23 comprehensive test cases)
./tests/test_math_comprehensive

# Advanced Gaussian methods (19 comprehensive test cases)
./tests/test_gaussian_enhanced

# Advanced Exponential methods
./tests/test_exponential_enhanced
```

## Test Execution Statistics

- **Correctness suite** (`ctest -LE "timing|benchmark"`): 37 tests — always parallel-safe
- **Timing suite** (`ctest -j1 -L timing`): 17 GTest tests with SIMD speedup assertions
- **Benchmark**: 1 test (`benchmark_simd_all`)
- **Success Rate**: 100% on all registered targets
- **Execution Time**: ~40 seconds for the correctness suite

## Developer Guidelines

### Adding New Tests

#### For New Distributions
1. **Create basic test**: `test_[distribution]_basic.cpp`
   - Validate PDF, CDF, quantile functions
   - Test parameter estimation and sampling
   - Include basic SIMD speedup verification
   - Use standalone test framework for simplicity

2. **Create enhanced test**: `test_[distribution]_enhanced.cpp`
   - Use GTest framework for comprehensive coverage
   - Test advanced statistical methods
   - Include performance regression testing
   - Validate thread safety and edge cases

3. **Update CMakeLists.txt**:
   ```cmake
   # Add to Level 4 tests
   create_libstats_test(test_[distribution]_basic tests/test_[distribution]_basic.cpp)
   create_libstats_gtest(test_[distribution]_enhanced tests/test_[distribution]_enhanced.cpp)

   # Add to run_tests dependencies
   test_[distribution]_basic
   test_[distribution]_enhanced
   ```

#### For Service Components
- Determine appropriate dependency level (0-3)
- Use standalone tests for service infrastructure
- Focus on functionality validation and basic performance
- Maintain clear separation from distribution-specific logic

### Mathematical Function Testing

The `test_math_comprehensive.cpp` GTest suite provides comprehensive validation:
- **Error functions**: `erf`, `erfc`, `erf_inv` with high-precision verification
- **Gamma functions**: `gamma_p`, `gamma_q` with series vs continued fraction validation
- **Beta functions**: `beta_i` with SciPy-verified expected values
- **Distribution CDFs**: Normal, t, chi-squared, F-distribution with edge case handling
- **Special cases**: NaN handling, infinity handling, numerical precision limits

### Performance Testing Guidelines

1. **SIMD Validation**: All batch operations should demonstrate measurable speedup
2. **Thread Safety**: Use stress testing with multiple threads
3. **Memory Safety**: Validate bounds checking and numerical stability
4. **Regression Detection**: Compare against established baselines

## CMake Integration

### Test Creation Functions
- **`create_libstats_test()`**: For standalone tests with minimal dependencies
- **`create_libstats_gtest()`**: For GTest-based comprehensive testing
- **`create_libstats_test_dynamic()`**: For dynamic library linking validation

### Build Configuration
- Dependency-aware build order respecting Level 0-4 hierarchy
- Proper SIMD compilation flags for performance testing
- GTest integration with Homebrew detection on macOS
- Thread-safe execution with TBB support where available

## Adding New Distributions

Each new distribution requires:
1. `test_[name]_basic.cpp` — standalone test with sections matching the 24-section implementation template
2. `test_[name]_enhanced.cpp` — GTest suite including SIMD correctness and speedup assertions (label: `timing`)
3. Registration in `CMakeLists.txt` via `create_libstats_test()` / `create_libstats_gtest()`
4. Addition to `set_tests_properties(... PROPERTIES LABELS "timing")` for the enhanced test
5. Addition to the `run_all_tests` DEPENDS list

This architecture ensures libstats maintains production-quality standards while remaining maintainable and extensible.
