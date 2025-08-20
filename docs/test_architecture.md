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
- **Standalone tests**: 25 tests (service infrastructure, basic functionality)
- **GTest tests**: 2 tests (advanced distribution methods)
- **Total**: 27 tests

## Test Organization by Dependency Levels

### Level 0: Foundational Tests (9 tests)
**Purpose**: Validate fundamental building blocks with no internal dependencies

- **`test_cpp20_features`** - C++20 language features and standard library compatibility
- **`test_constants`** - Mathematical constants and precision tolerances from `constants.h`
- **`test_cpu_detection`** - Runtime CPU feature detection from `cpu_detection.h`
- **`test_adaptive_cache`** - Advanced adaptive cache management from `adaptive_cache.h`
- **`test_simd_integration_simple`** - Basic SIMD integration from `simd.h`
- **`test_simd_integration`** - Comprehensive SIMD integration from `simd.h`
- **`test_simd_operations`** - SIMD vector operations and performance validation from `simd.h`
- **`test_platform_optimizations`** - Platform-specific optimizations from `simd.h`
- **`test_safe_factory`** - ABI-safe error handling with Result<T> pattern from `error_handling.h`

### Level 1: Core Infrastructure Tests (5 tests)
**Purpose**: Validate core mathematical and safety infrastructure

- **`test_safety`** - Memory safety, bounds checking, numerical stability from `safety.h`
- **`test_math_utils`** (**GTest**) - Mathematical utilities and special functions from `math_utils.h`
- **`test_vectorized_math`** - SIMD-optimized mathematical operations from `math_utils.h`
- **`test_goodness_of_fit`** - Statistical validation and goodness-of-fit testing from `validation.h`
- **`test_validation_enhanced`** - Enhanced statistical validation features from `validation.h`

### Level 2: Core Framework Tests
**Purpose**: Distribution base framework (abstract base class)

- **Note**: No direct tests for `distribution_base.h` - validated through Level 4 distribution implementations

### Level 3: Parallel Infrastructure Tests (5 tests)
**Purpose**: Validate parallel computation and performance measurement

- **`test_thread_pool`** - Basic thread pool implementation from `thread_pool.h`
- **`test_work_stealing_pool`** - Work-stealing thread pool from `work_stealing_pool.h`
- **`test_parallel_execution_integration`** - C++20 parallel execution policies from `parallel_execution.h`
- **`test_parallel_execution_comprehensive`** - Comprehensive parallel execution testing from `parallel_execution.h`
- **`test_benchmark_basic`** - Performance measurement utilities from `benchmark.h`

### Level 4: Distribution Implementation Tests (10 tests)
**Purpose**: Validate concrete statistical distribution implementations

#### Gaussian Distribution
- **`test_gaussian_basic`** - Fundamental operations (PDF, CDF, quantiles, basic fitting)
- **`test_gaussian_enhanced`** (**GTest**) - Advanced statistical methods:
  - Confidence intervals and hypothesis testing
  - Cross-validation and bootstrap methods
  - Robust estimation and Bayesian approaches
  - Information criteria (AIC, BIC, DIC)
  - Goodness-of-fit tests (KS, AD)
  - Parallel batch operations and SIMD optimization

#### Exponential Distribution
- **`test_exponential_basic`** - Fundamental operations (PDF, CDF, quantiles, basic fitting)
- **`test_exponential_enhanced`** (**GTest**) - Advanced statistical methods (similar scope to Gaussian)

#### Uniform Distribution
- **`test_uniform_basic`** - Fundamental operations (PDF, CDF, quantiles, basic fitting)
- **`test_uniform_enhanced`** (**GTest**) - Advanced statistical methods including Anderson-Darling tests

#### Discrete Distribution
- **`test_discrete_basic`** - Fundamental operations (PMF, CDF, quantiles, basic fitting)
- **`test_discrete_enhanced`** (**GTest**) - Advanced statistical methods for discrete distributions

#### Poisson Distribution
- **`test_poisson_basic`** - Fundamental operations (PMF, CDF, quantiles, basic fitting, exact methods)
- **`test_poisson_enhanced`** (**GTest**) - Advanced statistical methods for count data

### Additional Tests: Cross-cutting Concerns (4 tests)
**Purpose**: Integration scenarios and cross-cutting functionality

- **`test_dynamic_linking`** - Dynamic library linking validation
- **`test_simd_integration_simple_dynamic`** - SIMD with dynamic linking
- **`test_gaussian_basic_dynamic`** - Gaussian distribution with dynamic linking
- **`test_exponential_basic_dynamic`** - Exponential distribution with dynamic linking

## Consolidated Testing Strategy

### Historical Consolidation (Completed)

The testing infrastructure was successfully consolidated from 30 to 27 tests:

#### What Was Removed
- **Debug files** (5 files): `debug_erf_inv_test.cpp`, `debug_beta_i_test.cpp`, `verify_beta_i.cpp`, etc.
- **Redundant Gaussian tests** (2 files): `test_advanced_methods.cpp`, `test_gaussian_phase3.cpp`

#### What Was Merged
- Advanced Gaussian functionality consolidated into `test_gaussian_enhanced.cpp` with comprehensive GTest coverage
- All Phase 2 & 3 statistical methods (19 GTest test cases)
- Mathematical function debugging consolidated into production-quality `test_math_utils.cpp`

#### What Was Standardized
- Consistent naming: `*_basic.cpp` for fundamental operations, `*_enhanced.cpp` for advanced features
- Clear separation of concerns between basic and advanced functionality
- Modern C++20 practices with proper GTest integration

## Standardized Test Frameworks

### Basic Test Framework (`basic_test_template.h`)

#### Overview
All basic distribution tests use a unified template system ensuring consistent coverage, output formatting, and user experience across all distributions.

#### Framework Components

**StandardizedBasicTest Class Provides:**
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

**✅ Gaussian Distribution**
- Properties: Mean, variance, standard deviation, skewness, kurtosis
- Sampling: Box-Muller algorithm validation
- Fitting: Maximum Likelihood Estimation

**✅ Exponential Distribution**
- Properties: Lambda (rate), mean, variance, skewness, kurtosis
- Sampling: Inverse transform method
- Fitting: Maximum Likelihood Estimation

**✅ Uniform Distribution**
- Properties: Lower/upper bounds, mean, variance, skewness, kurtosis
- Sampling: Direct transform method
- Fitting: Method of moments (range estimation)

**✅ Discrete Distribution**
- Properties: Lower/upper bounds, range, mean, variance, skewness, kurtosis
- Sampling: Single, batch, and integer sampling
- Features: Support validation, outcome enumeration, binary distribution special case

**✅ Poisson Distribution**
- Properties: Lambda, mean, variance, skewness, kurtosis, mode
- Sampling: Knuth's algorithm (small λ) and Atkinson algorithm (large λ)
- Features: Normal approximation capability, integer sampling, exact methods

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

### Enhanced Test Framework (`enhanced_test_template.h`)

#### Overview
The enhanced tests utilize Google Test for comprehensive testing of advanced statistical functionality, performance benchmarks, and thread safety validation.

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

- **Gaussian SIMD**: >1.0x speedup confirmed for batch operations
- **Exponential SIMD**: 6.7x speedup confirmed for batch operations
- **Uniform SIMD**: Excellent speedups for batch probability and cumulative calculations
- **Discrete SIMD**: Consistent performance across integer operations
- **Poisson SIMD**: Optimized performance for both small and large lambda values
- **Mathematical functions**: 99.34% accuracy vs SciPy (151/152 tests pass)
- **Thread safety**: Verified across all implementations
- **Memory safety**: Comprehensive bounds checking and numerical stability

## Running Tests

### All Tests in Dependency Order
```bash
cd build
make run_tests
# Expected: 100% tests passed, 0 tests failed out of 27
```

### By Dependency Level
```bash
# Level 0: Foundational tests
ctest -R "test_cpp20_features|test_constants|test_cpu_detection|test_adaptive_cache|test_simd.*|test_platform.*|test_safe_factory"

# Level 1: Core infrastructure tests
ctest -R "test_safety|test_math_utils|test_vectorized_math|test_goodness_of_fit|test_validation_enhanced"

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
./tests/test_math_utils

# Advanced Gaussian methods (19 comprehensive test cases)
./tests/test_gaussian_enhanced

# Advanced Exponential methods
./tests/test_exponential_enhanced
```

## Test Execution Statistics

- **Total Tests**: 27 (reduced from 30 after consolidation)
- **Foundational Tests (Level 0)**: 9 tests
- **Infrastructure Tests (Level 1)**: 5 tests (includes GTest math_utils)
- **Framework Tests (Level 2)**: 0 tests (covered by Level 4)
- **Parallel Tests (Level 3)**: 5 tests
- **Distribution Tests (Level 4)**: 4 tests (includes 2 GTest suites)
- **Cross-cutting Tests**: 4 tests
- **Success Rate**: 100% (27/27 tests pass)
- **Execution Time**: ~14 seconds for full suite

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

The `test_math_utils.cpp` GTest suite provides comprehensive validation:
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

## Future Development

### Planned Extensions
1. **New Distributions**: Poisson, Gamma, Beta distributions following the established pattern
2. **Test Categories**: CTest labels for filtering by dependency level or functionality
3. **Performance Baselines**: Automated performance regression detection
4. **Coverage Analysis**: Code coverage reports by dependency level
5. **CI/CD Integration**: Level-based pipeline stages for efficient failure detection

### Quality Assurance
- All new code must include both basic and enhanced test coverage
- Performance regressions must be validated against established baselines
- Mathematical accuracy must be verified against reference implementations (SciPy)
- Thread safety and memory safety must be comprehensively validated

This architecture ensures that libstats maintains production-quality standards while remaining maintainable and extensible for future development.
