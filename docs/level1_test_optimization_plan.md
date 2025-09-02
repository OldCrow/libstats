# Level 1 Test Optimization Plan

## Current State Analysis

### Level 0 Tests (Foundational) - ✅ Complete
All Level 0 tests have comprehensive command-line options and excellent coverage:
- `test_constants` - 13+ command options
- `test_cpu_detection` - 9+ command options
- `test_simd_comprehensive` - 8+ command options
- `test_simd_policy` - 9+ command options
- `test_platform_optimizations` - CI-friendly JSON output

### Level 1 Tests (Core Infrastructure) - 🔧 Needs Optimization

#### Current Tests (6 tests, 39 files → 41 executables)
1. `test_safe_factory` - No CLI options
2. `test_dual_api` - No CLI options
3. `test_safety` - No CLI options
4. `test_math_utils` - GTest-based
5. `test_vectorized_math` - No CLI options
6. `test_validation_enhanced` - No CLI options

#### Missing Coverage
- `log_space_ops.h` - No dedicated test
- `distribution_validation.h` - Partial coverage only

## Proposed Consolidation Strategy

### 1. Error Handling & API Consolidation
**New Test:** `test_error_handling_comprehensive`
- Consolidates: `test_safe_factory` + `test_dual_api`
- Source files: Merge into single comprehensive test

**Command-line options:**
```
--all/-a              Test all error handling components (default)
--factory/-f          Test safe factory methods
--dual-api/-d         Test dual API (exception/Result)
--validation/-v       Test parameter validation
--distributions/-D    Test specific distribution(s)
--stress/-s           Concurrent access stress testing
--benchmarks/-b       Performance benchmarks
--help/-h             Show help
```

**Coverage:**
- Factory method validation
- Result<T> vs exception APIs
- Parameter validation consistency
- Thread-safe parameter updates
- Error message quality

### 2. Numerical Safety Consolidation
**New Test:** `test_numerical_safety`
- Consolidates: `test_safety` + new log-space ops tests
- Source files: Merge and expand

**Command-line options:**
```
--all/-a              Test all numerical safety (default)
--scalar/-s           Test scalar safety functions
--vector/-v           Test vectorized safety functions
--log-space/-l        Test log-space operations
--edge-cases/-e       Focus on edge cases
--benchmarks/-b       Performance benchmarks
--precision/-p        High-precision validation
--stress/-S           Stress test with extreme values
--help/-h             Show help
```

**Coverage:**
- safe_log, safe_exp, safe_sqrt
- clamp_probability, clamp_log_probability
- Log-space arithmetic (logSumExp, logMatrixVectorMultiply)
- Vectorized versions of all safety functions
- SIMD vs scalar performance comparison

### 3. Mathematical Utilities Consolidation
**New Test:** `test_math_comprehensive`
- Consolidates: `test_math_utils` + `test_vectorized_math`
- Source files: Merge, keep GTest support optional

**Command-line options:**
```
--all/-a              Test all math utilities (default)
--special/-s          Test special functions (erf, gamma, etc.)
--vectorized/-v       Test vectorized implementations
--accuracy/-A         High-precision accuracy tests
--performance/-p      Performance comparisons
--cross-validate/-c   Cross-validate with reference libs
--distribution/-d     Test distribution-specific math
--benchmarks/-b       Comprehensive benchmarks
--gtest/-g            Run GTest suite if available
--help/-h             Show help
```

**Coverage:**
- Error functions (erf, erfc, erf_inv)
- Gamma functions (lgamma, gamma_p, gamma_q)
- Beta functions (beta, lbeta, beta_i)
- Vectorized implementations
- SIMD optimizations
- Accuracy vs reference implementations

### 4. Validation Enhancement
**Keep as:** `test_validation_enhanced` (already well-structured)
**Add command-line options:**

```
--all/-a              Test all validation components (default)
--goodness-of-fit/-g  Test GoF tests (KS, AD, Chi-squared)
--bootstrap/-b        Test bootstrap methods
--diagnostics/-d      Test model diagnostics
--distribution/-D     Test distribution validation
--p-values/-p         Test p-value calculations
--stress/-s           Stress test with large datasets
--accuracy/-A         High-precision validation
--json/-j             Output results in JSON format
--help/-h             Show help
```

## Implementation Priority

### Phase 1: Add Missing Coverage (Week 1)
1. Create `test_log_space_ops.cpp` for log-space arithmetic
2. Add distribution_validation tests to `test_validation_enhanced`

### Phase 2: Add CLI Options to Existing Tests (Week 1-2)
1. Add command-line parsing to `test_validation_enhanced`
2. Add command-line parsing to `test_safety`
3. Convert `test_math_utils` to dual mode (GTest + CLI)

### Phase 3: Consolidation (Week 2-3)
1. Merge `test_safe_factory` + `test_dual_api` → `test_error_handling_comprehensive`
2. Merge `test_safety` + log-space → `test_numerical_safety`
3. Merge `test_math_utils` + `test_vectorized_math` → `test_math_comprehensive`

### Phase 4: Testing & Documentation (Week 3)
1. Update CMakeLists.txt for new test structure
2. Update test organization documentation
3. Validate all tests pass in both default and comprehensive modes
4. Performance benchmarking of consolidated tests

## Expected Outcomes

### Before Optimization
- 6 Level 1 tests
- 0 tests with CLI options
- Missing log-space coverage
- No performance benchmarking options

### After Optimization
- 4 consolidated Level 1 tests
- 4/4 tests with comprehensive CLI options
- Complete coverage of all Level 1 headers
- Integrated performance benchmarking
- CI-friendly JSON output options
- Consistent command-line interface across all tests

## Test Execution Modes

### Default CTest Mode
```bash
ctest                    # Runs all tests in default mode
```

### Comprehensive Testing
```bash
./test_error_handling_comprehensive --all --stress
./test_numerical_safety --all --benchmarks
./test_math_comprehensive --all --cross-validate
./test_validation_enhanced --all --accuracy
```

### CI Integration
```bash
./test_validation_enhanced --all --json > validation_results.json
./test_math_comprehensive --accuracy --json > math_accuracy.json
```

## Benefits

1. **Reduced Redundancy**: Fewer test files, better organization
2. **Better Coverage**: All Level 1 functionality fully tested
3. **Flexibility**: Command-line options for focused testing
4. **Performance**: Integrated benchmarking for optimization validation
5. **CI-Friendly**: JSON output for automated analysis
6. **Maintainability**: Consolidated tests easier to maintain
7. **Consistency**: Uniform CLI interface across test suite
