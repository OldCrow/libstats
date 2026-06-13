# SIMD Optimization Reference

> **Note**: This document records SIMD optimization analysis and decisions made
> during development. Items marked ✅ are resolved; items marked ⏳ remain deferred
> as of v1.2.0. The implementation roadmap at the end of this document was the
> original planning timeline; the work described there is now complete or deferred
> indefinitely. See AGENTS.md for the current deferred-items list.


**Status as of v1.2.0**

This document records SIMD optimization analysis. It contains algorithm details
for the hand-rolled SIMD transcendental functions that would be needed to bring
AVX2/AVX-512 transcendental performance to full width.

Items marked ✅ below have been resolved; items marked ⏳ remain deferred.

---

## 1. ✅ erf Implementation — Resolved

The accuracy bug described here is fixed. The current implementation
uses Abramowitz & Stegun 7.1.26 with max error ~1.5×10⁻⁷, documented in both
`src/simd_avx.cpp` and `tests/test_math_comprehensive.cpp`. `simd_verification`
confirms 54/54 PASS including Gaussian CDF and Log-Normal CDF (which both use `vector_erf`).

For reference, the algorithm that was implemented:

### Algorithm: Abramowitz & Stegun 7.1.26
```
erf(x) = 1 - 1/(1 + p*|x|)^n * exp(-x²) * P(t) for x ≥ 0
where:
  p = 0.3275911
  t = 1/(1 + p*|x|)
  P(t) = t*(a₁ + t*(a₂ + t*(a₃ + t*(a₄ + t*a₅))))

Coefficients:
  a₁ = 0.254829592
  a₂ = -0.284496736
  a₃ = 1.421413741
  a₄ = -1.453152027
  a₅ = 1.061405429

Maximum error: 1.5×10⁻⁷
```

If higher accuracy is needed in future, `std::erf` in the scalar fallback
already delivers full double precision; the vectorized path is the tradeoff.

#### Reference Implementations to Study

1. **SLEEF** (`sleef/src/libm/sleefsimdsp.c:3478-3580`)
   - Look for `xerfff` and `xerf` functions
   - Uses different polynomial sets for ranges: |x| < 2.5 and |x| ≥ 2.5
   - Implements double-double arithmetic for higher precision

2. **Boost Math** (`boost/math/special_functions/erf.hpp`)
   - Rational approximation with configurable precision
   - Well-documented error bounds

3. **Cephes Math Library**
   - Classic implementation with proven stability
   - Available at: netlib.org/cephes/

#### Implementation Considerations for libstats

1. **Preserve SIMD Structure**:
   ```cpp
   // Key pattern to maintain
   __m256d sign = _mm256_and_pd(x, sign_mask);
   __m256d abs_x = _mm256_andnot_pd(sign_mask, x);
   // ... computation on abs_x ...
   result = _mm256_or_pd(result, sign);  // Restore sign
   ```

2. **Efficient Polynomial Evaluation**:
   - Use Horner's method for sequential evaluation
   - Consider Estrin's method for parallel evaluation (better ILP)

3. **Range Handling**:
   - Small |x| < 1e-8: Use linear approximation `erf(x) ≈ (2/√π)*x`
   - Large |x| > 6: Return ±1 directly
   - Medium range: Apply full polynomial

4. **Integration Points**:
   - Files to modify: `src/simd_avx.cpp`, `src/simd_avx2.cpp`, `src/simd_avx512.cpp`, `src/simd_sse2.cpp`, `src/simd_neon.cpp`
   - Test coverage: `tests/test_simd_comprehensive.cpp`, `tests/test_math_comprehensive.cpp`
   - Verification tool: `build/tools/simd_verification`

---

## 2. ⏳ Deferred: Power Function Optimization (3x Performance Gap)

### Current Implementation
- **Location**: `src/simd_avx.cpp:486-556`
- **Method**: Separate calls to `vector_log_avx` and `vector_exp_avx`
- **Issue**: Function call overhead, missed optimization opportunities

### Recommended Approach: Fused Computation

#### Algorithm: Range-Reduced Power Computation
```
pow(x, y) computation:
1. Special case handling (x=0, x=1, y=0, y=1, integer y)
2. For general case: x^y = 2^(y * log₂(x))
3. Range reduction: log₂(x) = n + log₂(m) where x = 2^n * m, m ∈ [1,2)
4. Polynomial approximation for log₂(m) and 2^frac
```

#### Reference Implementations

1. **EVE** (`eve/module/math/regular/pow.hpp`)
   - Template-based compile-time optimization
   - Aggressive inlining of log2/exp2 operations
   - Special handling for integer exponents

2. **SLEEF** (`sleef/src/libm/sleefsimddp.c`)
   - Look for `xpowf` and `xpow` functions
   - Implements fast path for common cases
   - Uses FMA instructions when available

3. **Intel SVML** (proprietary but documented)
   - Optimized for Intel architectures
   - Reference: Intel Intrinsics Guide

#### Implementation Strategy

1. **Inline Critical Functions**:
   ```cpp
   // Instead of:
   vector_log_avx(&base[i], temp_log, AVX_DOUBLE_WIDTH);
   // Use inline computation:
   __m256d log_base = inline_log2_avx(base_vec);
   ```

2. **Special Case Optimization**:
   ```cpp
   // Fast path for integer exponents
   if (is_integer_exponent) {
       return fast_integer_pow(base, int_exp);
   }
   ```

3. **Fused Operations**:
   - Combine log₂ and exp₂ range reduction
   - Share polynomial evaluation infrastructure
   - Use FMA instructions: `_mm256_fmadd_pd`

---

## 3. ⏳ Deferred: exp/log Function Optimization (2x Performance Gap)

### Exponential Function Optimization

#### Current Implementation
- **Location**: `src/simd_avx.cpp:234-350`
- **Method**: Range reduction with polynomial approximation

#### Recommended Improvements from EVE/SLEEF

1. **Table-Driven Approach** (SLEEF):
   - Precomputed table for 2^(j/64) where j = 0..63
   - Reduces polynomial degree needed
   - Better cache utilization

2. **Template Metaprogramming** (EVE):
   ```cpp
   template<typename T, std::size_t N>
   EVE_FORCEINLINE auto exp_impl(wide<T, N> x) {
       // Compile-time optimizations based on N
   }
   ```

3. **Polynomial Coefficients**:
   - Use Remez algorithm-optimized coefficients
   - Different sets for different accuracy requirements

### Logarithm Function Optimization

#### Current Implementation
- **Location**: `src/simd_avx.cpp:352-485`

#### Key Optimizations

1. **Faster Range Reduction**:
   ```cpp
   // Extract exponent using bit manipulation
   __m256i xi = _mm256_castpd_si256(x);
   __m256i exp_bits = _mm256_srli_epi64(xi, 52);
   ```

2. **Improved Polynomial**:
   - Use rational approximation for better accuracy/performance
   - Minimize division operations

---

## 4. ⏳ Deferred: Architectural Options

### Template-Based SIMD (EVE Approach)

> **Note**: Template-based SIMD (EVE-style) is on the explicit "not doing" list
> for the teaching library. This section is retained as reference only.

#### Benefits
- Compile-time optimization
- Better instruction scheduling
- Automatic algorithm selection

#### Implementation Pattern
```cpp
template<typename T, std::size_t Width>
struct simd_traits {
    static constexpr bool has_fma = /* detect FMA */;
    static constexpr std::size_t unroll_factor = /* compute */;
};

template<typename T, std::size_t Width>
auto optimized_exp(const T* input, T* output, std::size_t size) {
    if constexpr (simd_traits<T, Width>::has_fma) {
        // FMA-optimized path
    } else {
        // Standard path
    }
}
```

### Multiple Accuracy Modes (SLEEF Approach)

#### Implementation Structure
```cpp
namespace libstats::simd {
    namespace fast {  // ~1 ULP error
        void vector_exp(/*...*/);
    }
    namespace accurate {  // < 0.5 ULP error
        void vector_exp(/*...*/);
    }
    namespace exact {  // Correctly rounded
        void vector_exp(/*...*/);
    }
}
```

---

## 5. TESTING AND VALIDATION

### Accuracy Testing Framework

1. **Test Vectors**:
   - Edge cases: 0, ±∞, NaN, denormals
   - Special values: π, e, √2
   - Random sampling across domains
   - Worst-case inputs for each function

2. **Error Metrics**:
   ```cpp
   struct ErrorStats {
       double max_absolute_error;
       double max_relative_error;
       double rms_error;
       double max_ulp_error;
   };
   ```

3. **Reference Data Sources**:
   - MPFR library for high-precision reference
   - Wolfram Alpha for specific values
   - Table-based test vectors from literature

### Performance Benchmarking

1. **Micro-benchmarks** (as in current simd_library_benchmark.cpp):
   - Isolate individual operations
   - Test various input sizes
   - Measure throughput and latency

2. **Macro-benchmarks**:
   - Real distribution computations
   - End-to-end statistical operations
   - Cache effects with large datasets

---

## 6. IMPLEMENTATION CHECKLIST
