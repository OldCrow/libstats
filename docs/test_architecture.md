# libstats Test Architecture

## Overview

The libstats test suite has been organized according to the header dependency hierarchy to ensure proper testing from the ground up. This organization ensures that foundational components are tested before dependent components, making regression identification and troubleshooting more efficient.

## Test Organization by Dependency Levels

### Level 0: Foundational Tests (No Internal Dependencies)
These test the most fundamental building blocks and should run first:

- **`test_cpp20_features`** - Tests C++20 features and mathematical constants from `constants.h`
- **`test_cpu_detection`** - Tests runtime CPU feature detection from `cpu_detection.h`
- **`test_simd_integration_simple`** - Tests basic SIMD integration from `simd.h`
- **`test_simd_integration`** - Tests comprehensive SIMD integration from `simd.h`
- **`test_simd_operations`** - Tests SIMD vector operations from `simd.h`
- **`test_safe_factory`** - Tests ABI-safe error handling from `error_handling.h`

### Level 1: Core Infrastructure Tests (Depends on Level 0)
These build on foundational components and should run second:

- **`test_safety`** - Tests memory safety and bounds checking from `safety.h`
- **`test_vectorized_math`** - Tests mathematical utilities and special functions from `math_utils.h`
- **`test_goodness_of_fit`** - Tests statistical validation from `validation.h`
- **`test_validation_enhanced`** - Tests enhanced validation features from `validation.h`

### Level 2: Core Framework Tests (Depends on Levels 0-1)
These test the distribution base framework:

- **Note**: No direct tests for `distribution_base.h` as it's an abstract base class tested through distribution implementations

### Level 3: Parallel Infrastructure Tests (Depends on Levels 0-2)
These test parallel computation infrastructure:

- **`test_work_stealing_pool`** - Tests work-stealing thread pool from `work_stealing_pool.h`
- **`test_benchmark_basic`** - Tests performance measurement utilities from `benchmark.h`

### Level 4: Distribution Implementation Tests (Depends on Levels 0-3)
These test concrete distribution implementations:

- **`test_gaussian_simple`** - Basic Gaussian distribution tests
- **`test_gaussian_enhanced`** - Enhanced Gaussian distribution tests
- **`test_exponential_simple`** - Basic exponential distribution tests
- **`test_exponential_enhanced`** - Enhanced exponential distribution tests
- **`test_uniform_simple`** - Basic uniform distribution tests
- **`test_uniform_enhanced`** - Enhanced uniform distribution tests

### Additional Tests: Cross-cutting Concerns
These test cross-cutting functionality and integration scenarios:

- **`test_dynamic_linking`** - Tests dynamic library linking
- **`test_copy_move_stress`** - Tests copy/move semantics under stress
- **`test_copy_move_fix`** - Tests copy/move semantics fixes
- **Dynamic test variants** - Tests simple tests with dynamic library linking

## Benefits of Hierarchical Test Organization

### 1. **Efficient Regression Detection**
- When a test fails, it's easier to identify the root cause
- Foundational failures surface immediately
- Cascading failures are clearly attributed to their source

### 2. **Logical Build Order**
- Tests are built in dependency order
- Foundational components are validated first
- Higher-level components can safely depend on tested foundations

### 3. **Maintainability**
- Clear separation of concerns
- Easy to add new tests at the appropriate level
- Consistent organization across the project

### 4. **Development Workflow**
- Developers can run tests level by level during development
- CI/CD pipelines can fail fast on foundational issues
- Parallel execution respects dependency constraints

## Running Tests

### All Tests in Dependency Order
```bash
cd build
make run_tests
```

### Individual Test Levels
```bash
# Level 0 tests
ctest -R "test_cpp20_features|test_cpu_detection|test_simd_.*|test_safe_factory"

# Level 1 tests  
ctest -R "test_safety|test_vectorized_math|test_goodness_of_fit|test_validation_enhanced"

# Level 3 tests
ctest -R "test_work_stealing_pool|test_benchmark_basic"

# Level 4 tests
ctest -R "test_.*_simple|test_.*_enhanced" -E "dynamic"
```

## Test Execution Statistics

- **Total Tests**: 25
- **Foundational Tests (Level 0)**: 6 tests
- **Infrastructure Tests (Level 1)**: 4 tests  
- **Framework Tests (Level 2)**: 0 tests (covered by Level 4)
- **Parallel Tests (Level 3)**: 2 tests
- **Distribution Tests (Level 4)**: 6 tests
- **Cross-cutting Tests**: 7 tests

## Future Enhancements

1. **Test Categorization**: Add CTest labels for filtering by level
2. **Parallel Execution**: Enable parallel test execution within levels
3. **Performance Baselines**: Add performance regression testing
4. **Coverage Analysis**: Generate coverage reports by dependency level
5. **Continuous Integration**: Implement level-based CI pipeline stages

## CMake Integration

The test hierarchy is implemented in CMakeLists.txt with:
- Clear level separation and documentation
- Proper dependency ordering in the `run_tests` target
- Conditional test inclusion based on file existence
- Support for both static and dynamic library testing

This organization ensures that the libstats library maintains high quality through systematic, dependency-aware testing.
