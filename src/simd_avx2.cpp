// AVX2-specific SIMD implementations
// This file is compiled ONLY with AVX2 flags to ensure safety

#include "libstats/common/simd_implementation_common.h"

#include <cmath>
#include <immintrin.h>  // AVX2 intrinsics

namespace stats {
namespace simd {
namespace ops {

// All AVX2 functions use double-precision (64-bit) values
// AVX2 processes 4 doubles per 256-bit register (same as AVX but with better integer support)

double VectorOps::dot_product_avx2(const double* a, const double* b, std::size_t size) noexcept {
    // Runtime safety check - bail out if AVX2 not supported
    if (!stats::arch::supports_avx2()) {
        return dot_product_fallback(a, b, size);
    }

    __m256d sum = _mm256_setzero_pd();
    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;

    // Process quartets of doubles with FMA if available
    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);

// Use FMA instruction if available for better performance and accuracy
#ifdef __FMA__
        sum = _mm256_fmadd_pd(va, vb, sum);
#else
        __m256d prod = _mm256_mul_pd(va, vb);
        sum = _mm256_add_pd(sum, prod);
#endif
    }

    // Extract horizontal sum
    double result[4];
    _mm256_storeu_pd(result, sum);
    double final_sum = result[0] + result[1] + result[2] + result[3];

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }

    return final_sum;
}

void VectorOps::vector_add_avx2(const double* a, const double* b, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_add_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_avx2(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_subtract_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_avx2(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_multiply_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_avx2(const double* a, double scalar, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }

    __m256d vscalar = _mm256_set1_pd(scalar);
    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vresult = _mm256_mul_pd(va, vscalar);
        _mm256_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_avx2(const double* a, double scalar, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return scalar_add_fallback(a, scalar, result, size);
    }

    __m256d vscalar = _mm256_set1_pd(scalar);
    constexpr std::size_t AVX2_DOUBLE_WIDTH = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / AVX2_DOUBLE_WIDTH) * AVX2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX2_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vresult = _mm256_add_pd(va, vscalar);
        _mm256_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

// AVX2 transcendental functions - reuse AVX implementations since AVX2 mainly adds integer ops
// We can call the AVX versions directly since AVX2 is a superset of AVX

void VectorOps::vector_exp_avx2(const double* values, double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_exp_fallback(values, results, size);
    }
    // AVX2 has same FP capabilities as AVX, delegate to AVX implementation
    return vector_exp_avx(values, results, size);
}

void VectorOps::vector_log_avx2(const double* values, double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_log_fallback(values, results, size);
    }
    // AVX2 has same FP capabilities as AVX, delegate to AVX implementation
    return vector_log_avx(values, results, size);
}

void VectorOps::vector_pow_avx2(const double* base, double exponent, double* results,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_pow_fallback(base, exponent, results, size);
    }
    // AVX2 has same FP capabilities as AVX, delegate to AVX implementation
    return vector_pow_avx(base, exponent, results, size);
}

void VectorOps::vector_pow_elementwise_avx2(const double* base, const double* exponent,
                                            double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        // Fallback to scalar implementation
        for (std::size_t i = 0; i < size; ++i) {
            results[i] = std::pow(base[i], exponent[i]);
        }
        return;
    }
    // AVX2 has same FP capabilities as AVX, delegate to AVX implementation
    return vector_pow_elementwise_avx(base, exponent, results, size);
}

void VectorOps::vector_erf_avx2(const double* values, double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_erf_fallback(values, results, size);
    }
    // AVX2 has same FP capabilities as AVX, delegate to AVX implementation
    return vector_erf_avx(values, results, size);
}

void VectorOps::vector_cos_avx2(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_cos_fallback(input, output, size);
    }

    // Native AVX2 implementation using FMA for the 7-term Horner polynomial.
    // Same two-step range reduction as vector_cos_avx; _mm256_fmadd_pd replaces
    // each mul+add pair in the Horner evaluation, reducing rounding error and
    // halving the instruction count for the polynomial step.
    // Max error ≈ 1×10⁻¹⁰ for |y| ≤ π/2. Scalar tail delegates to std::cos.
    constexpr std::size_t W = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    const __m256d inv_two_pi  = _mm256_set1_pd(1.0 / (2.0 * detail::PI));
    const __m256d two_pi      = _mm256_set1_pd(2.0 * detail::PI);
    const __m256d pi          = _mm256_set1_pd(detail::PI);
    const __m256d half_pi     = _mm256_set1_pd(detail::PI_OVER_2);
    const __m256d neg_pi      = _mm256_set1_pd(-detail::PI);
    const __m256d neg_half_pi = _mm256_set1_pd(-detail::PI_OVER_2);
    const __m256d one         = _mm256_set1_pd(1.0);
    const __m256d neg_one     = _mm256_set1_pd(-1.0);

    // Taylor coefficients: c_k = (-1)^k / (2k)!
    const __m256d c1 = _mm256_set1_pd(-0.5);                     // -1/2!
    const __m256d c2 = _mm256_set1_pd(4.166666666666667e-2);     //  1/4!
    const __m256d c3 = _mm256_set1_pd(-1.388888888888889e-3);    // -1/6!
    const __m256d c4 = _mm256_set1_pd(2.480158730158730e-5);     //  1/8!
    const __m256d c5 = _mm256_set1_pd(-2.755731922398589e-7);    // -1/10!
    const __m256d c6 = _mm256_set1_pd(2.087675698786810e-9);     //  1/12!
    const __m256d c7 = _mm256_set1_pd(-1.147074559772973e-11);   // -1/14!

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m256d x = _mm256_loadu_pd(&input[i]);

        // Step 1: reduce to [−π, π]
        __m256d q = _mm256_round_pd(_mm256_mul_pd(x, inv_two_pi),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d y = _mm256_sub_pd(x, _mm256_mul_pd(q, two_pi));

        // Step 2: reduce to [−π/2, π/2], tracking sign
        __m256d sign    = one;
        __m256d gt_hpi  = _mm256_cmp_pd(y, half_pi,     _CMP_GT_OQ);
        __m256d lt_nhpi = _mm256_cmp_pd(y, neg_half_pi, _CMP_LT_OQ);

        y    = _mm256_blendv_pd(y,    _mm256_sub_pd(pi,     y), gt_hpi);
        sign = _mm256_blendv_pd(sign, neg_one,                  gt_hpi);
        y    = _mm256_blendv_pd(y,    _mm256_sub_pd(neg_pi, y), lt_nhpi);
        sign = _mm256_blendv_pd(sign, neg_one,                  lt_nhpi);

        // Step 3: FMA Horner  cos(y) = 1 + y²·(c₁ + y²·(… + y²·c₇))
        __m256d y2   = _mm256_mul_pd(y, y);
        __m256d poly = c7;
        poly = _mm256_fmadd_pd(y2, poly, c6);
        poly = _mm256_fmadd_pd(y2, poly, c5);
        poly = _mm256_fmadd_pd(y2, poly, c4);
        poly = _mm256_fmadd_pd(y2, poly, c3);
        poly = _mm256_fmadd_pd(y2, poly, c2);
        poly = _mm256_fmadd_pd(y2, poly, c1);
        poly = _mm256_fmadd_pd(y2, poly, one);

        _mm256_storeu_pd(&output[i], _mm256_mul_pd(poly, sign));
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::cos(input[i]);
    }
}

}  // namespace ops
}  // namespace simd
}  // namespace stats
