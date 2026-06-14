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

// AVX2 transcendental functions
// vector_exp_avx2 and vector_log_avx2 are FMA-native (Phase 1b, v1.5.0).
// vector_pow_avx2, vector_pow_elementwise_avx2, and vector_erf_avx2 still delegate
// to AVX (unchanged in Phase 1; erf replaced in Phase 2, pow deferred).

void VectorOps::vector_exp_avx2(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_exp_fallback(input, output, size);
    }

    // FMA-accelerated port of vector_exp_avx (SLEEF-inspired, < 1 ULP error).
    // Horner steps use _mm256_fmadd_pd; range-reduction uses _mm256_fnmadd_pd.
    // Exponent bit-manipulation is identical to AVX (_mm256_cvtepi64_pd requires
    // AVX-512DQ which is absent on AVX2 hardware).

    const __m256d ln2_inv = _mm256_set1_pd(1.4426950408889634073599246810019);
    const __m256d ln2_hi  = _mm256_set1_pd(0.693147180369123816490e+00);
    const __m256d ln2_lo  = _mm256_set1_pd(1.90821492927058770002e-10);
    const __m256d exp_max = _mm256_set1_pd(709.782712893383996732223);
    const __m256d exp_min = _mm256_set1_pd(-708.0);
    const __m256d half    = _mm256_set1_pd(0.5);
    const __m256d one     = _mm256_set1_pd(1.0);

    const __m256d c1  = _mm256_set1_pd(0.1666666666666669072e+0);
    const __m256d c2  = _mm256_set1_pd(0.4166666666666602598e-1);
    const __m256d c3  = _mm256_set1_pd(0.8333333333314938210e-2);
    const __m256d c4  = _mm256_set1_pd(0.1388888888914497797e-2);
    const __m256d c5  = _mm256_set1_pd(0.1984126989855865850e-3);
    const __m256d c6  = _mm256_set1_pd(0.2480158687479686264e-4);
    const __m256d c7  = _mm256_set1_pd(0.2755723402025388239e-5);
    const __m256d c8  = _mm256_set1_pd(0.2755762628169491192e-6);
    const __m256d c9  = _mm256_set1_pd(0.2511210703042288022e-7);
    const __m256d c10 = _mm256_set1_pd(0.2081276378237164457e-8);

    constexpr std::size_t W = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m256d x = _mm256_loadu_pd(&input[i]);
        x = _mm256_min_pd(x, exp_max);
        x = _mm256_max_pd(x, exp_min);

        // Range reduction: x = n*ln2 + r
        __m256d n_float = _mm256_round_pd(_mm256_mul_pd(x, ln2_inv),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d r = _mm256_fnmadd_pd(n_float, ln2_hi, x);  // x - n*ln2_hi
        r         = _mm256_fnmadd_pd(n_float, ln2_lo, r);  // r - n*ln2_lo

        // FMA Horner: P(r)
        __m256d r2   = _mm256_mul_pd(r, r);
        __m256d poly = c10;
        poly = _mm256_fmadd_pd(poly, r, c9);
        poly = _mm256_fmadd_pd(poly, r, c8);
        poly = _mm256_fmadd_pd(poly, r, c7);
        poly = _mm256_fmadd_pd(poly, r, c6);
        poly = _mm256_fmadd_pd(poly, r, c5);
        poly = _mm256_fmadd_pd(poly, r, c4);
        poly = _mm256_fmadd_pd(poly, r, c3);
        poly = _mm256_fmadd_pd(poly, r, c2);
        poly = _mm256_fmadd_pd(poly, r, c1);

        // Complete: exp(r) = 1 + r + r^2*(0.5 + r*P(r))
        poly = _mm256_fmadd_pd(poly, r,  half);  // r*P(r) + 0.5
        poly = _mm256_fmadd_pd(poly, r2, r);     // (r*P(r)+0.5)*r^2 + r
        poly = _mm256_add_pd(poly, one);          // 1 + r + r^2*(0.5+r*P(r))

        // Scale by 2^n (same bit-manipulation as AVX)
        __m128i n_int  = _mm256_cvtpd_epi32(n_float);
        __m128i ebits  = _mm_add_epi32(n_int, _mm_set1_epi32(1023));
        __m128i elo    = _mm_slli_epi64(_mm_cvtepi32_epi64(ebits), 52);
        __m128i ehi    = _mm_slli_epi64(_mm_cvtepi32_epi64(_mm_shuffle_epi32(ebits, 0x0E)), 52);
        __m256d scale  = _mm256_set_m128d(_mm_castsi128_pd(ehi), _mm_castsi128_pd(elo));
        _mm256_storeu_pd(&output[i], _mm256_mul_pd(poly, scale));
    }

    for (std::size_t i = simd_end; i < size; ++i) output[i] = std::exp(input[i]);
}

void VectorOps::vector_log_avx2(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_log_fallback(input, output, size);
    }

    // FMA-accelerated port of vector_log_avx (SLEEF xlog_u1, < 1 ULP error).
    // Horner steps use _mm256_fmadd_pd; final ln2 reconstruction uses FMA.
    // Exponent extraction uses store-and-reload: _mm256_cvtepi64_pd requires
    // AVX-512DQ, absent on AVX2 hardware.

    const __m256d one     = _mm256_set1_pd(1.0);
    const __m256d ln2_hi  = _mm256_set1_pd(0.693147180559945286226764);
    const __m256d ln2_lo  = _mm256_set1_pd(2.319046813846299558417771e-17);
    const __m256d sqrt2   = _mm256_set1_pd(1.4142135623730950488016887242097);
    const __m256d half    = _mm256_set1_pd(0.5);
    const __m256d two     = _mm256_set1_pd(2.0);

    // SLEEF xlog_u1 coefficients (2*atanh series)
    const __m256d c1 = _mm256_set1_pd(0.6666666666667333541e+0);
    const __m256d c2 = _mm256_set1_pd(0.3999999999635251990e+0);
    const __m256d c3 = _mm256_set1_pd(0.2857142932794299317e+0);
    const __m256d c4 = _mm256_set1_pd(0.2222214519839380009e+0);
    const __m256d c5 = _mm256_set1_pd(0.1818605932937785996e+0);
    const __m256d c6 = _mm256_set1_pd(0.1525629051003428716e+0);
    const __m256d c7 = _mm256_set1_pd(0.1532076988502701353e+0);

    const __m256d zero    = _mm256_setzero_pd();
    const __m256d neg_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const __m256d pos_inf = _mm256_set1_pd( std::numeric_limits<double>::infinity());

    constexpr std::size_t W = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m256d x = _mm256_loadu_pd(&input[i]);

        __m256d is_zero     = _mm256_cmp_pd(x, zero,    _CMP_EQ_OQ);
        __m256d is_negative = _mm256_cmp_pd(x, zero,    _CMP_LT_OQ);
        __m256d is_inf      = _mm256_cmp_pd(x, pos_inf, _CMP_EQ_OQ);

        // Scale denormals by 2^54
        const __m256d min_normal = _mm256_set1_pd(2.2250738585072014e-308);
        const __m256d scale_up   = _mm256_set1_pd(18014398509481984.0);
        __m256d is_denormal = _mm256_cmp_pd(x, min_normal, _CMP_LT_OQ);
        __m256d scaled_x    = _mm256_blendv_pd(x, _mm256_mul_pd(x, scale_up), is_denormal);

        // Exponent extraction (two 128-bit halves)
        __m256i xi       = _mm256_castpd_si256(scaled_x);
        __m128i xi_lo    = _mm256_castsi256_si128(xi);
        __m128i xi_hi    = _mm256_extractf128_si256(xi, 1);
        __m128i exp_mask = _mm_set1_epi64x(0x7FF);
        __m128i ibias    = _mm_set1_epi64x(1023);
        __m128i exp_lo   = _mm_sub_epi64(_mm_and_si128(_mm_srli_epi64(xi_lo, 52), exp_mask), ibias);
        __m128i exp_hi   = _mm_sub_epi64(_mm_and_si128(_mm_srli_epi64(xi_hi, 52), exp_mask), ibias);

        // int64 -> double (store-and-reload)
        alignas(16) int64_t elo[2], ehi_arr[2];
        _mm_store_si128(reinterpret_cast<__m128i*>(elo),     exp_lo);
        _mm_store_si128(reinterpret_cast<__m128i*>(ehi_arr), exp_hi);
        __m128d elo_d = _mm_set_pd(static_cast<double>(elo[1]),     static_cast<double>(elo[0]));
        __m128d ehi_d = _mm_set_pd(static_cast<double>(ehi_arr[1]), static_cast<double>(ehi_arr[0]));
        __m256d e = _mm256_set_m128d(ehi_d, elo_d);
        e = _mm256_blendv_pd(e, _mm256_sub_pd(e, _mm256_set1_pd(54.0)), is_denormal);

        // Isolate mantissa in [1, 2)
        __m128i mant_mask = _mm_set1_epi64x(0x000FFFFFFFFFFFFF);
        __m128i exp_bias  = _mm_set1_epi64x(0x3FF0000000000000);
        __m128i m_lo  = _mm_or_si128(_mm_and_si128(xi_lo, mant_mask), exp_bias);
        __m128i m_hi  = _mm_or_si128(_mm_and_si128(xi_hi, mant_mask), exp_bias);
        __m256d m = _mm256_set_m128d(_mm_castsi128_pd(m_hi), _mm_castsi128_pd(m_lo));

        // Range adjustment: m -> [0.5, sqrt(2))
        __m256d needs_adj = _mm256_cmp_pd(m, sqrt2, _CMP_GT_OQ);
        m = _mm256_blendv_pd(m, _mm256_mul_pd(m, half), needs_adj);
        e = _mm256_blendv_pd(e, _mm256_add_pd(e, one),  needs_adj);

        // xr = (m-1)/(m+1)
        __m256d xr  = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
        __m256d xr2 = _mm256_mul_pd(xr, xr);

        // FMA Horner: t = c7 + xr2*(c6 + ... + xr2*c1)
        __m256d t = c7;
        t = _mm256_fmadd_pd(t, xr2, c6);
        t = _mm256_fmadd_pd(t, xr2, c5);
        t = _mm256_fmadd_pd(t, xr2, c4);
        t = _mm256_fmadd_pd(t, xr2, c3);
        t = _mm256_fmadd_pd(t, xr2, c2);
        t = _mm256_fmadd_pd(t, xr2, c1);

        // log(m) = 2*xr + xr^3*t  ->  fmadd(xr3, t, 2*xr)
        __m256d xr3    = _mm256_mul_pd(xr, xr2);
        __m256d two_xr = _mm256_mul_pd(xr, two);
        __m256d log_m  = _mm256_fmadd_pd(xr3, t, two_xr);

        // log(x) = log(m) + e*ln2  (high-low FMA decomposition)
        __m256d result = _mm256_fmadd_pd(e, ln2_hi, log_m);
        result         = _mm256_fmadd_pd(e, ln2_lo, result);

        result = _mm256_blendv_pd(result, neg_inf, is_zero);
        result = _mm256_blendv_pd(result, pos_inf, is_inf);
        result = _mm256_blendv_pd(result,
            _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()), is_negative);

        _mm256_storeu_pd(&output[i], result);
    }

    for (std::size_t i = simd_end; i < size; ++i) output[i] = std::log(input[i]);
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
