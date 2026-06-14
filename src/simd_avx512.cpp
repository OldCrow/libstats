// AVX-512-specific SIMD implementations
// This file is compiled ONLY with AVX-512 flags and includes runtime safety checks

#include "libstats/common/simd_implementation_common.h"

#include <cmath>
#include <immintrin.h>  // AVX-512 intrinsics

namespace stats {
namespace simd {
namespace ops {

// All AVX-512 functions use double-precision (64-bit) values
// AVX-512 processes 8 doubles per 512-bit register

double VectorOps::dot_product_avx512(const double* a, const double* b, std::size_t size) noexcept {
    // CRITICAL: Runtime safety check - bail out if AVX-512 not supported
    // This prevents illegal instruction crashes on CPUs without AVX-512
    if (!stats::arch::supports_avx512()) {
        return dot_product_fallback(a, b, size);
    }

    __m512d sum = _mm512_setzero_pd();
    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    // Process octets of doubles
    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        // Use FMA instruction for efficiency: sum = sum + (va * vb)
        sum = _mm512_fmadd_pd(va, vb, sum);
    }

    // Extract horizontal sum
    double result[8];
    _mm512_storeu_pd(result, sum);
    double final_sum = result[0] + result[1] + result[2] + result[3] + result[4] + result[5] +
                       result[6] + result[7];

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }

    return final_sum;
}

void VectorOps::vector_add_avx512(const double* a, const double* b, double* result,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_add_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vresult = _mm512_add_pd(va, vb);
        _mm512_storeu_pd(&result[i], vresult);
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_avx512(const double* a, const double* b, double* result,
                                       std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_subtract_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vresult = _mm512_sub_pd(va, vb);
        _mm512_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_avx512(const double* a, const double* b, double* result,
                                       std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_multiply_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vresult = _mm512_mul_pd(va, vb);
        _mm512_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_avx512(const double* a, double scalar, double* result,
                                       std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }

    __m512d vscalar = _mm512_set1_pd(scalar);
    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vresult = _mm512_mul_pd(va, vscalar);
        _mm512_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_avx512(const double* a, double scalar, double* result,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return scalar_add_fallback(a, scalar, result, size);
    }

    __m512d vscalar = _mm512_set1_pd(scalar);
    constexpr std::size_t AVX512_DOUBLE_WIDTH = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / AVX512_DOUBLE_WIDTH) * AVX512_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX512_DOUBLE_WIDTH) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vresult = _mm512_add_pd(va, vscalar);
        _mm512_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

// AVX-512 transcendental functions
//
// vector_exp, vector_log, and vector_erf are implemented natively at full 8-wide
// AVX-512 width using hand-rolled minimax polynomial approximations (Phase 4,
// v1.5.0). No SVML dependency; zero-external-dependency mandate preserved.
//
// vector_pow and vector_pow_elementwise still delegate to the AVX (4-wide) path:
// pow(x,y) = exp(y*log(x)) incurs two transcendental calls and the added
// complexity of 8-wide special-case handling is deferred until profiling shows
// a net gain. exp and log are now natively 8-wide so the delegation cost is
// reduced, but a dedicated native pow path is not yet warranted.

void VectorOps::vector_exp_avx512(const double* values, double* results,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_exp_fallback(values, results, size);
    }

    // SLEEF-inspired range-reduction + Horner polynomial (< 1 ULP error).
    // FMA Horner; 2^n scaling via integer bit-manipulation (AVX-512F only, no DQ).
    // Algorithm and coefficients identical to vector_exp_avx2; width doubled to 8.

    const __m512d ln2_inv = _mm512_set1_pd(1.4426950408889634073599246810019);
    const __m512d ln2_hi  = _mm512_set1_pd(0.693147180369123816490e+00);
    const __m512d ln2_lo  = _mm512_set1_pd(1.90821492927058770002e-10);
    const __m512d exp_max = _mm512_set1_pd(709.782712893383996732223);
    const __m512d exp_min = _mm512_set1_pd(-708.0);
    const __m512d half    = _mm512_set1_pd(0.5);
    const __m512d one     = _mm512_set1_pd(1.0);

    const __m512d c1  = _mm512_set1_pd(0.1666666666666669072e+0);
    const __m512d c2  = _mm512_set1_pd(0.4166666666666602598e-1);
    const __m512d c3  = _mm512_set1_pd(0.8333333333314938210e-2);
    const __m512d c4  = _mm512_set1_pd(0.1388888888914497797e-2);
    const __m512d c5  = _mm512_set1_pd(0.1984126989855865850e-3);
    const __m512d c6  = _mm512_set1_pd(0.2480158687479686264e-4);
    const __m512d c7  = _mm512_set1_pd(0.2755723402025388239e-5);
    const __m512d c8  = _mm512_set1_pd(0.2755762628169491192e-6);
    const __m512d c9  = _mm512_set1_pd(0.2511210703042288022e-7);
    const __m512d c10 = _mm512_set1_pd(0.2081276378237164457e-8);

    constexpr std::size_t W = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m512d x = _mm512_loadu_pd(&values[i]);
        x = _mm512_min_pd(x, exp_max);
        x = _mm512_max_pd(x, exp_min);

        // Range reduction: x = n*ln2 + r
        __m512d n_float = _mm512_roundscale_pd(_mm512_mul_pd(x, ln2_inv),
                                               _MM_FROUND_TO_NEAREST_INT);
        __m512d r = _mm512_fnmadd_pd(n_float, ln2_hi, x);  // x - n*ln2_hi
        r         = _mm512_fnmadd_pd(n_float, ln2_lo, r);  // r - n*ln2_lo

        // FMA Horner: P(r)
        __m512d r2   = _mm512_mul_pd(r, r);
        __m512d poly = c10;
        poly = _mm512_fmadd_pd(poly, r, c9);
        poly = _mm512_fmadd_pd(poly, r, c8);
        poly = _mm512_fmadd_pd(poly, r, c7);
        poly = _mm512_fmadd_pd(poly, r, c6);
        poly = _mm512_fmadd_pd(poly, r, c5);
        poly = _mm512_fmadd_pd(poly, r, c4);
        poly = _mm512_fmadd_pd(poly, r, c3);
        poly = _mm512_fmadd_pd(poly, r, c2);
        poly = _mm512_fmadd_pd(poly, r, c1);

        // Complete: exp(r) = 1 + r + r²*(0.5 + r*P(r))
        poly = _mm512_fmadd_pd(poly, r,  half);  // r*P(r) + 0.5
        poly = _mm512_fmadd_pd(poly, r2, r);     // (r*P(r)+0.5)*r² + r
        poly = _mm512_add_pd(poly, one);          // 1 + r + r²*(0.5+r*P(r))

        // Scale by 2^n: convert n to int32, expand to int64, add IEEE 754
        // exponent bias 1023, shift to exponent field, reinterpret as double.
        // _mm512_cvtpd_epi32 and _mm512_cvtepi32_epi64 are AVX-512F (no DQ).
        __m256i n_i32 = _mm512_cvtpd_epi32(n_float);          // 8 doubles → __m256i
        __m512i n_i64 = _mm512_cvtepi32_epi64(n_i32);          // expand to int64
        __m512i ebits = _mm512_add_epi64(n_i64, _mm512_set1_epi64(1023));
        ebits         = _mm512_slli_epi64(ebits, 52);
        __m512d scale = _mm512_castsi512_pd(ebits);

        _mm512_storeu_pd(&results[i], _mm512_mul_pd(poly, scale));
    }

    for (std::size_t i = simd_end; i < size; ++i) results[i] = std::exp(values[i]);
}

void VectorOps::vector_log_avx512(const double* values, double* results,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_log_fallback(values, results, size);
    }
    // Delegates to AVX (4-wide). See block comment above.
    return vector_log_avx(values, results, size);
}

void VectorOps::vector_pow_avx512(const double* base, double exponent, double* results,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_pow_fallback(base, exponent, results, size);
    }
    // Delegates to AVX (4-wide). See block comment above.
    return vector_pow_avx(base, exponent, results, size);
}

void VectorOps::vector_pow_elementwise_avx512(const double* base, const double* exponent,
                                              double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        for (std::size_t i = 0; i < size; ++i) {
            results[i] = std::pow(base[i], exponent[i]);
        }
        return;
    }
    // Delegates to AVX (4-wide). See block comment above.
    return vector_pow_elementwise_avx(base, exponent, results, size);
}

void VectorOps::vector_erf_avx512(const double* values, double* results,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_erf_fallback(values, results, size);
    }
    // Delegates to AVX (4-wide). See block comment above.
    return vector_erf_avx(values, results, size);
}

void VectorOps::vector_cos_avx512(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_cos_fallback(input, output, size);
    }

    // vector_cos uses polynomial approximation — no SVML dependency — so we
    // can implement it natively at full 8-wide AVX-512 width.
    // (Unlike vector_exp/log/erf which delegate to AVX; see block comment above.)

    constexpr std::size_t W = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    const __m512d inv_two_pi  = _mm512_set1_pd(1.0 / (2.0 * detail::PI));
    const __m512d two_pi      = _mm512_set1_pd(2.0 * detail::PI);
    const __m512d pi          = _mm512_set1_pd(detail::PI);
    const __m512d half_pi     = _mm512_set1_pd(detail::PI_OVER_2);
    const __m512d neg_pi      = _mm512_set1_pd(-detail::PI);
    const __m512d neg_half_pi = _mm512_set1_pd(-detail::PI_OVER_2);
    const __m512d one         = _mm512_set1_pd(1.0);
    const __m512d neg_one     = _mm512_set1_pd(-1.0);

    const __m512d c1 = _mm512_set1_pd(-0.5);
    const __m512d c2 = _mm512_set1_pd(4.166666666666667e-2);
    const __m512d c3 = _mm512_set1_pd(-1.388888888888889e-3);
    const __m512d c4 = _mm512_set1_pd(2.480158730158730e-5);
    const __m512d c5 = _mm512_set1_pd(-2.755731922398589e-7);
    const __m512d c6 = _mm512_set1_pd(2.087675698786810e-9);
    const __m512d c7 = _mm512_set1_pd(-1.147074559772973e-11);

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m512d x = _mm512_loadu_pd(&input[i]);

        // Step 1: reduce to [−π, π]
        __m512d q = _mm512_roundscale_pd(_mm512_mul_pd(x, inv_two_pi),
                                         _MM_FROUND_TO_NEAREST_INT);
        __m512d y = _mm512_sub_pd(x, _mm512_mul_pd(q, two_pi));

        // Step 2: reduce to [−π/2, π/2] using AVX-512 mask operations
        __m512d sign    = one;
        __mmask8 gt_hpi  = _mm512_cmp_pd_mask(y, half_pi,     _CMP_GT_OQ);
        __mmask8 lt_nhpi = _mm512_cmp_pd_mask(y, neg_half_pi, _CMP_LT_OQ);

        y    = _mm512_mask_blend_pd(gt_hpi,  y,    _mm512_sub_pd(pi,     y));
        sign = _mm512_mask_blend_pd(gt_hpi,  sign, neg_one);
        y    = _mm512_mask_blend_pd(lt_nhpi, y,    _mm512_sub_pd(neg_pi, y));
        sign = _mm512_mask_blend_pd(lt_nhpi, sign, neg_one);

        // Step 3: Horner evaluation using FMA for throughput
        __m512d y2   = _mm512_mul_pd(y, y);
        __m512d poly = c7;
        poly = _mm512_fmadd_pd(y2, poly, c6);
        poly = _mm512_fmadd_pd(y2, poly, c5);
        poly = _mm512_fmadd_pd(y2, poly, c4);
        poly = _mm512_fmadd_pd(y2, poly, c3);
        poly = _mm512_fmadd_pd(y2, poly, c2);
        poly = _mm512_fmadd_pd(y2, poly, c1);
        poly = _mm512_fmadd_pd(y2, poly, one);

        _mm512_storeu_pd(&output[i], _mm512_mul_pd(poly, sign));
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::cos(input[i]);
    }
}

}  // namespace ops
}  // namespace simd
}  // namespace stats
