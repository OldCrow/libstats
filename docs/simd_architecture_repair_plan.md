# SIMD Architecture Repair Plan

## Executive Summary

The libstats project has a complete SIMD infrastructure but it's not being used effectively. The codebase misleadingly reports "SIMD" performance improvements that are actually from multi-threading, not vectorization. This document outlines a comprehensive plan to implement real SIMD vectorization.

## Current State Analysis

### What Exists
1. **SIMD Infrastructure** - Complete dispatch system with platform-specific implementations:
   - `simd_dispatch.cpp` - Runtime dispatch based on CPU capabilities
   - `simd_avx.cpp`, `simd_avx2.cpp`, `simd_avx512.cpp` - x86 implementations
   - `simd_sse2.cpp` - Legacy x86 support
   - `simd_neon.cpp` - ARM support
   - `simd_fallback.cpp` - Scalar fallback

2. **Basic Operations Implemented** - Only arithmetic operations have SIMD:
   - `vector_add`, `vector_subtract`, `vector_multiply`
   - `scalar_add`, `scalar_multiply`
   - `dot_product`

3. **Missing Critical Operations** - Transcendental functions use scalar fallback:
   - `vector_exp` - Falls back to scalar std::exp loop
   - `vector_log` - Falls back to scalar std::log loop
   - `vector_pow` - Falls back to scalar std::pow loop
   - `vector_erf` - Falls back to scalar std::erf loop

### Misleading Architecture

1. **Strategy Naming** - Strategies are misleadingly named:
   - `SIMD_BATCH` - Actually just scalar batch processing
   - `PARALLEL_SIMD` - Actually just multi-threading with ParallelUtils::parallelFor
   - `WORK_STEALING` - Work-stealing thread pool (legitimate)
   - `GPU_ACCELERATED` - Falls back to work-stealing (GPU not implemented)

2. **Performance Claims** - Tests report "SIMD speedup" but it's actually from:
   - Multi-threading across cores
   - Reduced function call overhead (batch vs individual)
   - Better cache locality
   - NOT from SIMD vectorization

## Implementation Plan

### Phase 1: Implement SIMD Transcendental Functions

#### 1.1 Exponential Function (vector_exp)

Implement fast SIMD exp using range reduction and polynomial approximation:

```cpp
// In simd_avx.cpp
void VectorOps::vector_exp_avx(const double* values, double* results, size_t size) {
    // Algorithm: exp(x) = 2^(x/ln(2)) = 2^n * 2^f where n is integer, f is fraction
    // 1. Range reduction: x = n*ln(2) + r where |r| < ln(2)/2
    // 2. Compute 2^n using bit manipulation
    // 3. Compute exp(r) using Padé approximant or minimax polynomial
    // 4. Combine: exp(x) = 2^n * exp(r)

    const __m256d ln2 = _mm256_set1_pd(0.693147180559945309417);
    const __m256d inv_ln2 = _mm256_set1_pd(1.4426950408889634074);

    for (size_t i = 0; i + 4 <= size; i += 4) {
        __m256d x = _mm256_loadu_pd(&values[i]);

        // Range reduction
        __m256d n = _mm256_round_pd(_mm256_mul_pd(x, inv_ln2), _MM_FROUND_TO_NEAREST_INT);
        __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(n, ln2));

        // Polynomial approximation for exp(r)
        // Use Remez polynomial or Padé approximant
        __m256d exp_r = polynomial_exp_approx(r);

        // Scale by 2^n
        __m256d result = scale_by_power_of_2(exp_r, n);

        _mm256_storeu_pd(&results[i], result);
    }

    // Handle remaining elements
    for (size_t i = (size / 4) * 4; i < size; ++i) {
        results[i] = std::exp(values[i]);
    }
}
```

Key techniques:
- Use `_mm256_mul_pd`, `_mm256_add_pd` for polynomial evaluation
- Use `_mm256_round_pd` for extracting integer part
- Use bit manipulation for fast 2^n computation
- Handle special cases (overflow, underflow) with masks

#### 1.2 Logarithm Function (vector_log)

Implement fast SIMD log using frexp and polynomial approximation:

```cpp
// In simd_avx.cpp
void VectorOps::vector_log_avx(const double* values, double* results, size_t size) {
    // Algorithm: log(x) = log(2^n * m) = n*log(2) + log(m) where m in [0.5, 1)
    // 1. Extract exponent and mantissa using bit manipulation
    // 2. Use polynomial approximation for log(1+f) where f = m-1
    // 3. Combine: log(x) = n*log(2) + log(1+f)

    const __m256d ln2 = _mm256_set1_pd(0.693147180559945309417);
    const __m256d one = _mm256_set1_pd(1.0);

    for (size_t i = 0; i + 4 <= size; i += 4) {
        __m256d x = _mm256_loadu_pd(&values[i]);

        // Extract exponent and mantissa
        __m256i exp_bits = extract_exponent(x);
        __m256d mantissa = extract_mantissa(x);

        // Convert exponent to double
        __m256d n = _mm256_cvtepi32_pd(exp_bits);

        // Compute log(mantissa) using polynomial
        __m256d f = _mm256_sub_pd(mantissa, one);
        __m256d log_mantissa = polynomial_log_approx(f);

        // Combine: n*ln(2) + log(mantissa)
        __m256d result = _mm256_add_pd(
            _mm256_mul_pd(n, ln2),
            log_mantissa
        );

        _mm256_storeu_pd(&results[i], result);
    }

    // Handle remaining elements
    for (size_t i = (size / 4) * 4; i < size; ++i) {
        results[i] = std::log(values[i]);
    }
}
```

#### 1.3 Power Function (vector_pow)

Implement SIMD pow using exp and log:

```cpp
// In simd_avx.cpp
void VectorOps::vector_pow_avx(const double* base, double exponent, double* results, size_t size) {
    // Algorithm: pow(x, y) = exp(y * log(x))
    // Special cases need careful handling:
    // - Negative base with integer exponent
    // - Base = 0, 1, or infinity
    // - Exponent = 0, 1, or infinity

    const __m256d exp_vec = _mm256_set1_pd(exponent);

    // Check if exponent is an integer for handling negative bases
    bool is_integer_exp = (exponent == std::floor(exponent));

    for (size_t i = 0; i + 4 <= size; i += 4) {
        __m256d x = _mm256_loadu_pd(&base[i]);

        // For general case: pow(x, y) = exp(y * log(x))
        __m256d log_x = vector_log_single_batch(x);  // Internal SIMD log
        __m256d y_log_x = _mm256_mul_pd(exp_vec, log_x);
        __m256d result = vector_exp_single_batch(y_log_x);  // Internal SIMD exp

        // Handle special cases with masks
        result = handle_pow_special_cases(x, exp_vec, result, is_integer_exp);

        _mm256_storeu_pd(&results[i], result);
    }

    // Handle remaining elements
    for (size_t i = (size / 4) * 4; i < size; ++i) {
        results[i] = std::pow(base[i], exponent);
    }
}
```

Special cases to handle:
- x^0 = 1 for all x (including NaN)
- 1^y = 1 for all y (including NaN)
- 0^y = 0 for y > 0, infinity for y < 0
- Negative base requires integer exponent check

#### 1.4 Error Function (vector_erf)

Implement using rational approximation:

```cpp
// In simd_avx.cpp
void VectorOps::vector_erf_avx(const double* values, double* results, size_t size) {
    // Use Abramowitz and Stegun approximation
    // erf(x) ≈ sign(x) * (1 - exp(-x^2) * P(|x|))
    // where P is a polynomial approximation

    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d neg_one = _mm256_set1_pd(-1.0);

    // Coefficients for rational approximation
    const __m256d a1 = _mm256_set1_pd(0.254829592);
    const __m256d a2 = _mm256_set1_pd(-0.284496736);
    const __m256d a3 = _mm256_set1_pd(1.421413741);
    const __m256d a4 = _mm256_set1_pd(-1.453152027);
    const __m256d a5 = _mm256_set1_pd(1.061405429);
    const __m256d p = _mm256_set1_pd(0.3275911);

    for (size_t i = 0; i + 4 <= size; i += 4) {
        __m256d x = _mm256_loadu_pd(&values[i]);

        // Save sign and work with absolute value
        __m256d sign = _mm256_and_pd(x, _mm256_set1_pd(-0.0));
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);

        // t = 1/(1 + p*|x|)
        __m256d t = _mm256_div_pd(one,
            _mm256_add_pd(one, _mm256_mul_pd(p, abs_x)));

        // Polynomial evaluation (Horner's method)
        __m256d poly = a5;
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), a4);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), a3);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), a2);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), a1);
        poly = _mm256_mul_pd(poly, t);

        // exp(-x^2)
        __m256d x_squared = _mm256_mul_pd(abs_x, abs_x);
        __m256d exp_neg_x2 = vector_exp_single_batch(_mm256_sub_pd(_mm256_setzero_pd(), x_squared));

        // erf(|x|) = 1 - poly * exp(-x^2)
        __m256d erf_abs = _mm256_sub_pd(one, _mm256_mul_pd(poly, exp_neg_x2));

        // Apply sign
        __m256d result = _mm256_or_pd(sign, erf_abs);

        _mm256_storeu_pd(&results[i], result);
    }

    // Handle remaining elements
    for (size_t i = (size / 4) * 4; i < size; ++i) {
        results[i] = std::erf(values[i]);
    }
}
```

### Phase 2: Fix Distribution Strategy Implementation

#### 2.1 Update Exponential Distribution

Replace scalar loops with VectorOps calls:

```cpp
// In exponential.cpp getProbabilityBatchUnsafeImpl
void ExponentialDistribution::getProbabilityBatchUnsafeImpl(...) {
    if (arch::simd::SIMDPolicy::shouldUseSIMD(count)) {
        // Use SIMD for exp computation
        alignas(32) double neg_lambda_x[count];
        VectorOps::scalar_multiply(values, neg_lambda, neg_lambda_x, count);
        VectorOps::vector_exp(neg_lambda_x, results, count);
        VectorOps::scalar_multiply(results, lambda, results, count);
    } else {
        // Scalar fallback for small arrays
        for (size_t i = 0; i < count; ++i) {
            results[i] = lambda * std::exp(neg_lambda * values[i]);
        }
    }
}
```

#### 2.2 Update Strategy Naming

Rename strategies to accurately reflect implementation:
- `SIMD_BATCH` → `VECTORIZED` (when actually using SIMD)
- `PARALLEL_SIMD` → `PARALLEL_VECTORIZED`
- Keep `WORK_STEALING` as is
- `GPU_ACCELERATED` → Remove or implement actual GPU

### Phase 3: Optimize Safety Functions

Update safety.cpp to use VectorOps when beneficial:

```cpp
void vector_safe_log(std::span<const double> input, std::span<double> output) {
    const size_t size = input.size();

    if (arch::simd::SIMDPolicy::shouldUseSIMD(size)) {
        // Use SIMD log
        VectorOps::vector_log(input.data(), output.data(), size);

        // Post-process for safety (vectorized comparison/blend)
        for (size_t i = 0; i < size; ++i) {
            if (input[i] <= 0.0 || std::isnan(input[i])) {
                output[i] = MIN_LOG_PROBABILITY;
            } else if (std::isinf(input[i])) {
                output[i] = std::numeric_limits<double>::max();
            }
        }
    } else {
        // Scalar fallback
        for (size_t i = 0; i < size; ++i) {
            output[i] = safe_log(input[i]);
        }
    }
}
```

### Phase 4: Performance Validation

#### 4.1 Create Accurate Benchmarks

Separate benchmarks for:
- Pure SIMD vectorization speedup (single-threaded)
- Multi-threading speedup (parallel execution)
- Combined speedup (parallel + SIMD)

#### 4.2 Update SIMD Verification Tool

- Test actual SIMD implementations
- Verify numerical accuracy of polynomial approximations
- Ensure special case handling (NaN, Inf, etc.)

## Implementation Priority

1. **High Priority** - Core transcendental functions
   - `vector_exp_avx` - Used by all distributions
   - `vector_log_avx` - Used by log-probability calculations
   - `vector_pow_avx` - Used by Gamma distribution and others
   - Update dispatch in `simd_dispatch.cpp`

2. **Medium Priority** - Distribution optimizations
   - Update Exponential distribution
   - Update Gaussian distribution (already partially done)
   - Update Gamma distribution (uses pow extensively)
   - Update other distributions

3. **Low Priority** - Additional functions
   - `vector_erf` - Only used by Gaussian CDF
   - GPU acceleration (if needed)
   - Additional special functions (lgamma, beta, etc.)

## Success Criteria

1. **Performance**: Real SIMD implementations show 2-4x speedup over scalar (AVX should give ~4x for doubles)
2. **Accuracy**: Polynomial approximations maintain accuracy within 1-2 ULPs
3. **Correctness**: All tests pass with SIMD implementations
4. **Clarity**: Strategy names accurately reflect implementation

## Technical References

1. **Fast Exponential**:
   - "A Fast, Compact Approximation of the Exponential Function" (Schraudolph, 1999)
   - Intel's SVML implementation details
   - SLEEF (SIMD Library for Evaluating Elementary Functions)

2. **Fast Logarithm**:
   - "Efficient Approximations for the Arctangent and Logarithm Functions" (Muller, 2016)
   - Sun's fdlibm implementation

3. **Fast Power**:
   - "Fast and Accurate Power Function" (Cawley, 2000)
   - AMD's AOCL implementation

4. **Error Function**:
   - "Rational Chebyshev Approximations for the Error Function" (Cody, 1969)
   - Boost.Math implementation

5. **SIMD Programming**:
   - Intel Intrinsics Guide
   - Agner Fog's optimization manuals
   - ARM NEON Intrinsics Reference

## Risks and Mitigations

1. **Risk**: Accuracy loss from polynomial approximations
   - **Mitigation**: Use high-order polynomials, extensive testing against reference implementations

2. **Risk**: Platform-specific bugs
   - **Mitigation**: Comprehensive testing on x86 (SSE2, AVX, AVX2, AVX-512) and ARM (NEON)

3. **Risk**: Backward compatibility
   - **Mitigation**: Keep scalar fallbacks, gradual rollout with feature flags

4. **Risk**: Special case handling complexity
   - **Mitigation**: Extensive edge case testing, use of SIMD masks for branch-free code

## Timeline Estimate

- Phase 1: 3-4 weeks (implement and test all four SIMD transcendentals)
  - Week 1: vector_exp and vector_log
  - Week 2: vector_pow and vector_erf
  - Week 3-4: Testing, optimization, special cases
- Phase 2: 1-2 weeks (update distributions)
- Phase 3: 1 week (optimize safety functions)
- Phase 4: 1 week (validation and benchmarking)

Total: 6-8 weeks for complete implementation

## Notes on Implementation

### Polynomial Coefficients

For production use, we should use:
- Remez exchange algorithm for minimax polynomial approximations
- Sollya or similar tools for coefficient generation
- Different polynomial orders for different accuracy requirements

### Testing Strategy

1. **Unit tests**: Compare against std:: functions for full input range
2. **Accuracy tests**: Verify ULP error bounds
3. **Performance tests**: Measure actual SIMD speedup
4. **Edge case tests**: NaN, Inf, denormals, negative zero
5. **Cross-platform tests**: x86 and ARM implementations
