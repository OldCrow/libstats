// SSE2-specific SIMD implementations
// This file is compiled ONLY with SSE2 flags to ensure safety

#include "libstats/common/cpu_detection_fwd.h"       // Use lightweight forward declarations
#include "libstats/common/platform_constants_fwd.h"  // Use lightweight forward declarations
#include "libstats/core/math_constants.h"
#include "libstats/core/statistical_constants.h"
#include "libstats/platform/simd.h"

#include <cmath>
#include <emmintrin.h>  // SSE2 intrinsics

namespace stats {
namespace simd {
namespace ops {

// All SSE2 functions use double-precision (64-bit) values
// SSE2 processes 2 doubles per 128-bit register

double VectorOps::dot_product_sse2(const double* a, const double* b, std::size_t size) noexcept {
    // Runtime safety check - bail out if SSE2 not supported
    if (!stats::arch::supports_sse2()) {
        return dot_product_fallback(a, b, size);
    }

    __m128d sum = _mm_setzero_pd();
    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    // Process pairs of doubles
    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d prod = _mm_mul_pd(va, vb);
        sum = _mm_add_pd(sum, prod);
    }

    // Extract horizontal sum
    double result[2];
    _mm_storeu_pd(result, sum);
    double final_sum = result[0] + result[1];

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }

    return final_sum;
}

void VectorOps::vector_add_sse2(const double* a, const double* b, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_add_fallback(a, b, result, size);
    }

    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vresult = _mm_add_pd(va, vb);
        _mm_storeu_pd(&result[i], vresult);
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_sse2(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_subtract_fallback(a, b, result, size);
    }

    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vresult = _mm_sub_pd(va, vb);
        _mm_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_sse2(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_multiply_fallback(a, b, result, size);
    }

    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vresult = _mm_mul_pd(va, vb);
        _mm_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_sse2(const double* a, double scalar, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }

    __m128d vscalar = _mm_set1_pd(scalar);
    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vresult = _mm_mul_pd(va, vscalar);
        _mm_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_sse2(const double* a, double scalar, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return scalar_add_fallback(a, scalar, result, size);
    }

    __m128d vscalar = _mm_set1_pd(scalar);
    constexpr std::size_t SSE2_DOUBLE_WIDTH = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / SSE2_DOUBLE_WIDTH) * SSE2_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += SSE2_DOUBLE_WIDTH) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vresult = _mm_add_pd(va, vscalar);
        _mm_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

// SSE2 doesn't have native exp/log/pow/erf instructions, so we use scalar fallback
// with SSE2 register management for better cache usage

void VectorOps::vector_exp_sse2(const double* values, double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_exp_fallback(values, results, size);
    }

    // SSE2 lacks exp intrinsics, use scalar math
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::exp(values[i]);
    }
}

void VectorOps::vector_log_sse2(const double* values, double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_log_fallback(values, results, size);
    }

    // SSE2 lacks log intrinsics, use scalar math with safety check
    for (std::size_t i = 0; i < size; ++i) {
        results[i] =
            values[i] > 0.0 ? std::log(values[i]) : -std::numeric_limits<double>::infinity();
    }
}

void VectorOps::vector_pow_sse2(const double* base, double exponent, double* results,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_pow_fallback(base, exponent, results, size);
    }

    // SSE2 lacks pow intrinsics, use scalar math
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::pow(base[i], exponent);
    }
}

void VectorOps::vector_pow_elementwise_sse2(const double* base, const double* exponent,
                                            double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        // Fallback to scalar implementation
        for (std::size_t i = 0; i < size; ++i) {
            results[i] = std::pow(base[i], exponent[i]);
        }
        return;
    }

    // SSE2 lacks pow intrinsics, use scalar math for element-wise power
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::pow(base[i], exponent[i]);
    }
}

void VectorOps::vector_erf_sse2(const double* values, double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_erf_fallback(values, results, size);
    }

    // SSE2 lacks erf intrinsics, use scalar math
    for (std::size_t i = 0; i < size; ++i) {
        results[i] = std::erf(values[i]);
    }
}

void VectorOps::vector_cos_sse2(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_cos_fallback(input, output, size);
    }

    // SSE2 lacks _mm_round_pd (requires SSE4.1). Range reduction uses the
    // magic-number trick: adding 2^52+2^51=6755399441055744 rounds a double
    // to the nearest integer (for |x| < 2^51, which holds for any angle value).

    constexpr std::size_t W = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    const __m128d inv_two_pi  = _mm_set1_pd(1.0 / (2.0 * detail::PI));
    const __m128d two_pi      = _mm_set1_pd(2.0 * detail::PI);
    const __m128d pi          = _mm_set1_pd(detail::PI);
    const __m128d half_pi     = _mm_set1_pd(detail::PI_OVER_2);
    const __m128d neg_pi      = _mm_set1_pd(-detail::PI);
    const __m128d neg_half_pi = _mm_set1_pd(-detail::PI_OVER_2);
    const __m128d one         = _mm_set1_pd(1.0);
    const __m128d neg_one     = _mm_set1_pd(-1.0);
    // 2^52 + 2^51 — adding then subtracting rounds to nearest integer
    const __m128d magic       = _mm_set1_pd(6755399441055744.0);

    const __m128d c1 = _mm_set1_pd(-0.5);
    const __m128d c2 = _mm_set1_pd(4.166666666666667e-2);
    const __m128d c3 = _mm_set1_pd(-1.388888888888889e-3);
    const __m128d c4 = _mm_set1_pd(2.480158730158730e-5);
    const __m128d c5 = _mm_set1_pd(-2.755731922398589e-7);
    const __m128d c6 = _mm_set1_pd(2.087675698786810e-9);
    const __m128d c7 = _mm_set1_pd(-1.147074559772973e-11);

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m128d x = _mm_loadu_pd(&input[i]);

        // Step 1: reduce to [-π, π] using magic-number rounding
        __m128d scaled = _mm_mul_pd(x, inv_two_pi);
        __m128d q      = _mm_sub_pd(_mm_add_pd(scaled, magic), magic);  // round-to-nearest
        __m128d y      = _mm_sub_pd(x, _mm_mul_pd(q, two_pi));

        // Step 2: reduce to [-π/2, π/2], tracking sign
        // SSE2 comparison returns all-ones (true) or all-zeros (false) per lane.
        __m128d sign    = one;
        __m128d gt_hpi  = _mm_cmpgt_pd(y, half_pi);
        __m128d lt_nhpi = _mm_cmplt_pd(y, neg_half_pi);

        // Blend: select new_y when mask is true, else keep y
        __m128d new_y_gt = _mm_sub_pd(pi,     y);
        __m128d new_y_lt = _mm_sub_pd(neg_pi, y);
        y    = _mm_or_pd(_mm_and_pd(gt_hpi,  new_y_gt), _mm_andnot_pd(gt_hpi,  y));
        sign = _mm_or_pd(_mm_and_pd(gt_hpi,  neg_one),  _mm_andnot_pd(gt_hpi,  sign));
        y    = _mm_or_pd(_mm_and_pd(lt_nhpi, new_y_lt), _mm_andnot_pd(lt_nhpi, y));
        sign = _mm_or_pd(_mm_and_pd(lt_nhpi, neg_one),  _mm_andnot_pd(lt_nhpi, sign));

        // Step 3: Horner evaluation
        __m128d y2   = _mm_mul_pd(y, y);
        __m128d poly = c7;
        poly = _mm_add_pd(c6, _mm_mul_pd(y2, poly));
        poly = _mm_add_pd(c5, _mm_mul_pd(y2, poly));
        poly = _mm_add_pd(c4, _mm_mul_pd(y2, poly));
        poly = _mm_add_pd(c3, _mm_mul_pd(y2, poly));
        poly = _mm_add_pd(c2, _mm_mul_pd(y2, poly));
        poly = _mm_add_pd(c1, _mm_mul_pd(y2, poly));
        poly = _mm_add_pd(one, _mm_mul_pd(y2, poly));

        _mm_storeu_pd(&output[i], _mm_mul_pd(poly, sign));
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::cos(input[i]);
    }
}

}  // namespace ops
}  // namespace simd
}  // namespace stats
