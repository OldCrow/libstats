#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
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
    const __m256d ln2_hi = _mm256_set1_pd(0.693147180369123816490e+00);
    const __m256d ln2_lo = _mm256_set1_pd(1.90821492927058770002e-10);
    const __m256d exp_max = _mm256_set1_pd(709.782712893383996732223);
    const __m256d exp_min = _mm256_set1_pd(-708.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d one = _mm256_set1_pd(1.0);

    const __m256d c1 = _mm256_set1_pd(0.1666666666666669072e+0);
    const __m256d c2 = _mm256_set1_pd(0.4166666666666602598e-1);
    const __m256d c3 = _mm256_set1_pd(0.8333333333314938210e-2);
    const __m256d c4 = _mm256_set1_pd(0.1388888888914497797e-2);
    const __m256d c5 = _mm256_set1_pd(0.1984126989855865850e-3);
    const __m256d c6 = _mm256_set1_pd(0.2480158687479686264e-4);
    const __m256d c7 = _mm256_set1_pd(0.2755723402025388239e-5);
    const __m256d c8 = _mm256_set1_pd(0.2755762628169491192e-6);
    const __m256d c9 = _mm256_set1_pd(0.2511210703042288022e-7);
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
        r = _mm256_fnmadd_pd(n_float, ln2_lo, r);          // r - n*ln2_lo

        // FMA Horner: P(r)
        __m256d r2 = _mm256_mul_pd(r, r);
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
        poly = _mm256_fmadd_pd(poly, r, half);  // r*P(r) + 0.5
        poly = _mm256_fmadd_pd(poly, r2, r);    // (r*P(r)+0.5)*r^2 + r
        poly = _mm256_add_pd(poly, one);        // 1 + r + r^2*(0.5+r*P(r))

        // Scale by 2^n (same bit-manipulation as AVX)
        __m128i n_int = _mm256_cvtpd_epi32(n_float);
        __m128i ebits = _mm_add_epi32(n_int, _mm_set1_epi32(1023));
        __m128i elo = _mm_slli_epi64(_mm_cvtepi32_epi64(ebits), 52);
        __m128i ehi = _mm_slli_epi64(_mm_cvtepi32_epi64(_mm_shuffle_epi32(ebits, 0x0E)), 52);
        __m256d scale = _mm256_set_m128d(_mm_castsi128_pd(ehi), _mm_castsi128_pd(elo));
        _mm256_storeu_pd(&output[i], _mm256_mul_pd(poly, scale));
    }

    for (std::size_t i = simd_end; i < size; ++i)
        output[i] = std::exp(input[i]);
}

void VectorOps::vector_log_avx2(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_log_fallback(input, output, size);
    }

    // FMA-accelerated port of vector_log_avx (SLEEF xlog_u1, < 1 ULP error).
    // Horner steps use _mm256_fmadd_pd; final ln2 reconstruction uses FMA.
    // Exponent extraction uses store-and-reload: _mm256_cvtepi64_pd requires
    // AVX-512DQ, absent on AVX2 hardware.

    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d ln2_hi = _mm256_set1_pd(0.693147180559945286226764);
    const __m256d ln2_lo = _mm256_set1_pd(2.319046813846299558417771e-17);
    const __m256d sqrt2 = _mm256_set1_pd(1.4142135623730950488016887242097);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d two = _mm256_set1_pd(2.0);

    // SLEEF xlog_u1 coefficients (2*atanh series)
    const __m256d c1 = _mm256_set1_pd(0.6666666666667333541e+0);
    const __m256d c2 = _mm256_set1_pd(0.3999999999635251990e+0);
    const __m256d c3 = _mm256_set1_pd(0.2857142932794299317e+0);
    const __m256d c4 = _mm256_set1_pd(0.2222214519839380009e+0);
    const __m256d c5 = _mm256_set1_pd(0.1818605932937785996e+0);
    const __m256d c6 = _mm256_set1_pd(0.1525629051003428716e+0);
    const __m256d c7 = _mm256_set1_pd(0.1532076988502701353e+0);

    const __m256d zero = _mm256_setzero_pd();
    const __m256d neg_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const __m256d pos_inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());

    constexpr std::size_t W = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m256d x = _mm256_loadu_pd(&input[i]);

        __m256d is_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
        __m256d is_negative = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
        __m256d is_inf = _mm256_cmp_pd(x, pos_inf, _CMP_EQ_OQ);

        // Scale denormals by 2^54
        const __m256d min_normal = _mm256_set1_pd(2.2250738585072014e-308);
        const __m256d scale_up = _mm256_set1_pd(18014398509481984.0);
        __m256d is_denormal = _mm256_cmp_pd(x, min_normal, _CMP_LT_OQ);
        __m256d scaled_x = _mm256_blendv_pd(x, _mm256_mul_pd(x, scale_up), is_denormal);

        // Exponent extraction (two 128-bit halves)
        __m256i xi = _mm256_castpd_si256(scaled_x);
        __m128i xi_lo = _mm256_castsi256_si128(xi);
        __m128i xi_hi = _mm256_extractf128_si256(xi, 1);
        __m128i exp_mask = _mm_set1_epi64x(0x7FF);
        __m128i ibias = _mm_set1_epi64x(1023);
        __m128i exp_lo = _mm_sub_epi64(_mm_and_si128(_mm_srli_epi64(xi_lo, 52), exp_mask), ibias);
        __m128i exp_hi = _mm_sub_epi64(_mm_and_si128(_mm_srli_epi64(xi_hi, 52), exp_mask), ibias);

        // int64 -> double (store-and-reload)
        alignas(16) int64_t elo[2], ehi_arr[2];
        _mm_store_si128(reinterpret_cast<__m128i*>(elo), exp_lo);
        _mm_store_si128(reinterpret_cast<__m128i*>(ehi_arr), exp_hi);
        __m128d elo_d = _mm_set_pd(static_cast<double>(elo[1]), static_cast<double>(elo[0]));
        __m128d ehi_d =
            _mm_set_pd(static_cast<double>(ehi_arr[1]), static_cast<double>(ehi_arr[0]));
        __m256d e = _mm256_set_m128d(ehi_d, elo_d);
        e = _mm256_blendv_pd(e, _mm256_sub_pd(e, _mm256_set1_pd(54.0)), is_denormal);

        // Isolate mantissa in [1, 2)
        __m128i mant_mask = _mm_set1_epi64x(0x000FFFFFFFFFFFFF);
        __m128i exp_bias = _mm_set1_epi64x(0x3FF0000000000000);
        __m128i m_lo = _mm_or_si128(_mm_and_si128(xi_lo, mant_mask), exp_bias);
        __m128i m_hi = _mm_or_si128(_mm_and_si128(xi_hi, mant_mask), exp_bias);
        __m256d m = _mm256_set_m128d(_mm_castsi128_pd(m_hi), _mm_castsi128_pd(m_lo));

        // Range adjustment: m -> [0.5, sqrt(2))
        __m256d needs_adj = _mm256_cmp_pd(m, sqrt2, _CMP_GT_OQ);
        m = _mm256_blendv_pd(m, _mm256_mul_pd(m, half), needs_adj);
        e = _mm256_blendv_pd(e, _mm256_add_pd(e, one), needs_adj);

        // xr = (m-1)/(m+1)
        __m256d xr = _mm256_div_pd(_mm256_sub_pd(m, one), _mm256_add_pd(m, one));
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
        __m256d xr3 = _mm256_mul_pd(xr, xr2);
        __m256d two_xr = _mm256_mul_pd(xr, two);
        __m256d log_m = _mm256_fmadd_pd(xr3, t, two_xr);

        // log(x) = log(m) + e*ln2  (high-low FMA decomposition)
        __m256d result = _mm256_fmadd_pd(e, ln2_hi, log_m);
        result = _mm256_fmadd_pd(e, ln2_lo, result);

        result = _mm256_blendv_pd(result, neg_inf, is_zero);
        result = _mm256_blendv_pd(result, pos_inf, is_inf);
        result = _mm256_blendv_pd(result, _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()),
                                  is_negative);

        _mm256_storeu_pd(&output[i], result);
    }

    for (std::size_t i = simd_end; i < size; ++i)
        output[i] = std::log(input[i]);
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
        return vector_pow_elementwise_fallback(base, exponent, results, size);
    }
    // AVX2 has same FP capabilities as AVX, delegate to AVX implementation
    return vector_pow_elementwise_avx(base, exponent, results, size);
}

void VectorOps::vector_erf_avx2(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx2()) {
        return vector_erf_fallback(input, output, size);
    }

    // FMA-native port of vector_erf_avx (musl rational polynomial, < 1 ULP error).
    // All Horner mul+add pairs use _mm256_fmadd_pd. Exp call uses vector_exp_avx2.

    // Region 1: rational P/Q in z = x^2
    const __m256d pp0 = _mm256_set1_pd(1.28379167095512558561e-01);
    const __m256d pp1 = _mm256_set1_pd(-3.25042107247001499370e-01);
    const __m256d pp2 = _mm256_set1_pd(-2.84817495755985104766e-02);
    const __m256d pp3 = _mm256_set1_pd(-5.77027029648944159157e-03);
    const __m256d pp4 = _mm256_set1_pd(-2.37630166566501626084e-05);
    const __m256d qq1 = _mm256_set1_pd(3.97917223959155352819e-01);
    const __m256d qq2 = _mm256_set1_pd(6.50222499887672944485e-02);
    const __m256d qq3 = _mm256_set1_pd(5.08130628187576562776e-03);
    const __m256d qq4 = _mm256_set1_pd(1.32494738004321644526e-04);
    const __m256d qq5 = _mm256_set1_pd(-3.96022827877536812320e-06);
    // Region 2: rational P/Q in s = |x|-1
    const __m256d erx = _mm256_set1_pd(8.45062911510467529297e-01);
    const __m256d pa0 = _mm256_set1_pd(-2.36211856075265944077e-03);
    const __m256d pa1 = _mm256_set1_pd(4.14856118683748331666e-01);
    const __m256d pa2 = _mm256_set1_pd(-3.72207876035701323847e-01);
    const __m256d pa3 = _mm256_set1_pd(3.18346619901161753674e-01);
    const __m256d pa4 = _mm256_set1_pd(-1.10894694282396677476e-01);
    const __m256d pa5 = _mm256_set1_pd(3.54783043256182359371e-02);
    const __m256d pa6 = _mm256_set1_pd(-2.16637559486879084300e-03);
    const __m256d qa1 = _mm256_set1_pd(1.06420880400844228286e-01);
    const __m256d qa2 = _mm256_set1_pd(5.40397917702171048937e-01);
    const __m256d qa3 = _mm256_set1_pd(7.18286544141962662868e-02);
    const __m256d qa4 = _mm256_set1_pd(1.26171219808761642112e-01);
    const __m256d qa5 = _mm256_set1_pd(1.36370839120290507362e-02);
    const __m256d qa6 = _mm256_set1_pd(1.19844998467991074170e-02);
    // Region 3: R/S in s = 1/x^2
    const __m256d ra0 = _mm256_set1_pd(-9.86494403484714822705e-03);
    const __m256d ra1 = _mm256_set1_pd(-6.93858572707181764372e-01);
    const __m256d ra2 = _mm256_set1_pd(-1.05586262253232909814e+01);
    const __m256d ra3 = _mm256_set1_pd(-6.23753324503260060396e+01);
    const __m256d ra4 = _mm256_set1_pd(-1.62396669462573470355e+02);
    const __m256d ra5 = _mm256_set1_pd(-1.84605092906711035994e+02);
    const __m256d ra6 = _mm256_set1_pd(-8.12874355063065934246e+01);
    const __m256d ra7 = _mm256_set1_pd(-9.81432934416914548592e+00);
    const __m256d sa1 = _mm256_set1_pd(1.96512716674392571292e+01);
    const __m256d sa2 = _mm256_set1_pd(1.37657754143519042600e+02);
    const __m256d sa3 = _mm256_set1_pd(4.34565877475229228821e+02);
    const __m256d sa4 = _mm256_set1_pd(6.45387271733267880336e+02);
    const __m256d sa5 = _mm256_set1_pd(4.29008140027567833386e+02);
    const __m256d sa6 = _mm256_set1_pd(1.08635005541779435134e+02);
    const __m256d sa7 = _mm256_set1_pd(6.57024977031928170135e+00);
    const __m256d sa8 = _mm256_set1_pd(-6.04244152148580987438e-02);
    // Region 4: R/S in s = 1/x^2
    const __m256d rb0 = _mm256_set1_pd(-9.86494292470009928597e-03);
    const __m256d rb1 = _mm256_set1_pd(-7.99283237680523006574e-01);
    const __m256d rb2 = _mm256_set1_pd(-1.77579549177547519889e+01);
    const __m256d rb3 = _mm256_set1_pd(-1.60636384855821916062e+02);
    const __m256d rb4 = _mm256_set1_pd(-6.37566443368389627722e+02);
    const __m256d rb5 = _mm256_set1_pd(-1.02509513161107724954e+03);
    const __m256d rb6 = _mm256_set1_pd(-4.83519191608651397019e+02);
    const __m256d sb1 = _mm256_set1_pd(3.03380607434824582924e+01);
    const __m256d sb2 = _mm256_set1_pd(3.25792512996573918826e+02);
    const __m256d sb3 = _mm256_set1_pd(1.53672958608443695994e+03);
    const __m256d sb4 = _mm256_set1_pd(3.19985821950859553908e+03);
    const __m256d sb5 = _mm256_set1_pd(2.55305040643316442583e+03);
    const __m256d sb6 = _mm256_set1_pd(4.74528541206955367215e+02);
    const __m256d sb7 = _mm256_set1_pd(-2.24409524465858183362e+01);

    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d t1 = _mm256_set1_pd(0.84375);
    const __m256d t2 = _mm256_set1_pd(1.25);
    const __m256d t3 = _mm256_set1_pd(2.857142857);
    const __m256d t5 = _mm256_set1_pd(6.0);
    const __m256d c0p5625 = _mm256_set1_pd(0.5625);

    constexpr std::size_t W = arch::simd::AVX2_DOUBLES;
    const std::size_t simd_end = (size / W) * W;
    alignas(32) double exp_buf[W];

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m256d x = _mm256_loadu_pd(&input[i]);
        __m256d sign = _mm256_and_pd(x, sign_mask);
        __m256d ax = _mm256_andnot_pd(sign_mask, x);

        __m256d m1 = _mm256_cmp_pd(ax, t1, _CMP_LT_OQ);
        __m256d m2 = _mm256_cmp_pd(ax, t2, _CMP_LT_OQ);
        __m256d m3 = _mm256_cmp_pd(ax, t3, _CMP_LT_OQ);

        // Region 1: FMA Horner for P/Q in z = x^2
        __m256d z = _mm256_mul_pd(ax, ax);
        __m256d P1 = pp4;
        P1 = _mm256_fmadd_pd(z, P1, pp3);
        P1 = _mm256_fmadd_pd(z, P1, pp2);
        P1 = _mm256_fmadd_pd(z, P1, pp1);
        P1 = _mm256_fmadd_pd(z, P1, pp0);
        __m256d Q1 = qq5;
        Q1 = _mm256_fmadd_pd(z, Q1, qq4);
        Q1 = _mm256_fmadd_pd(z, Q1, qq3);
        Q1 = _mm256_fmadd_pd(z, Q1, qq2);
        Q1 = _mm256_fmadd_pd(z, Q1, qq1);
        Q1 = _mm256_fmadd_pd(z, Q1, one);
        __m256d r1 = _mm256_fmadd_pd(ax, _mm256_div_pd(P1, Q1), ax);

        // Region 2: FMA Horner for P/Q in s = |x|-1
        __m256d s2 = _mm256_sub_pd(ax, one);
        __m256d P2 = pa6;
        P2 = _mm256_fmadd_pd(s2, P2, pa5);
        P2 = _mm256_fmadd_pd(s2, P2, pa4);
        P2 = _mm256_fmadd_pd(s2, P2, pa3);
        P2 = _mm256_fmadd_pd(s2, P2, pa2);
        P2 = _mm256_fmadd_pd(s2, P2, pa1);
        P2 = _mm256_fmadd_pd(s2, P2, pa0);
        __m256d Q2 = qa6;
        Q2 = _mm256_fmadd_pd(s2, Q2, qa5);
        Q2 = _mm256_fmadd_pd(s2, Q2, qa4);
        Q2 = _mm256_fmadd_pd(s2, Q2, qa3);
        Q2 = _mm256_fmadd_pd(s2, Q2, qa2);
        Q2 = _mm256_fmadd_pd(s2, Q2, qa1);
        Q2 = _mm256_fmadd_pd(s2, Q2, one);
        __m256d r2 = _mm256_add_pd(erx, _mm256_div_pd(P2, Q2));

        // Regions 3-4: erfc = exp(-x^2-0.5625+R/S)/|x|
        __m256d sax = _mm256_max_pd(ax, t2);
        __m256d inv_x2 = _mm256_div_pd(one, _mm256_mul_pd(sax, sax));

        __m256d R3 = ra7;
        R3 = _mm256_fmadd_pd(inv_x2, R3, ra6);
        R3 = _mm256_fmadd_pd(inv_x2, R3, ra5);
        R3 = _mm256_fmadd_pd(inv_x2, R3, ra4);
        R3 = _mm256_fmadd_pd(inv_x2, R3, ra3);
        R3 = _mm256_fmadd_pd(inv_x2, R3, ra2);
        R3 = _mm256_fmadd_pd(inv_x2, R3, ra1);
        R3 = _mm256_fmadd_pd(inv_x2, R3, ra0);
        __m256d S3 = sa8;
        S3 = _mm256_fmadd_pd(inv_x2, S3, sa7);
        S3 = _mm256_fmadd_pd(inv_x2, S3, sa6);
        S3 = _mm256_fmadd_pd(inv_x2, S3, sa5);
        S3 = _mm256_fmadd_pd(inv_x2, S3, sa4);
        S3 = _mm256_fmadd_pd(inv_x2, S3, sa3);
        S3 = _mm256_fmadd_pd(inv_x2, S3, sa2);
        S3 = _mm256_fmadd_pd(inv_x2, S3, sa1);
        S3 = _mm256_fmadd_pd(inv_x2, S3, one);

        __m256d R4 = rb6;
        R4 = _mm256_fmadd_pd(inv_x2, R4, rb5);
        R4 = _mm256_fmadd_pd(inv_x2, R4, rb4);
        R4 = _mm256_fmadd_pd(inv_x2, R4, rb3);
        R4 = _mm256_fmadd_pd(inv_x2, R4, rb2);
        R4 = _mm256_fmadd_pd(inv_x2, R4, rb1);
        R4 = _mm256_fmadd_pd(inv_x2, R4, rb0);
        __m256d S4 = sb7;
        S4 = _mm256_fmadd_pd(inv_x2, S4, sb6);
        S4 = _mm256_fmadd_pd(inv_x2, S4, sb5);
        S4 = _mm256_fmadd_pd(inv_x2, S4, sb4);
        S4 = _mm256_fmadd_pd(inv_x2, S4, sb3);
        S4 = _mm256_fmadd_pd(inv_x2, S4, sb2);
        S4 = _mm256_fmadd_pd(inv_x2, S4, sb1);
        S4 = _mm256_fmadd_pd(inv_x2, S4, one);

        __m256d RS = _mm256_div_pd(_mm256_blendv_pd(R4, R3, m3), _mm256_blendv_pd(S4, S3, m3));
        // exp_arg = -x^2 - 0.5625 + R/S
        __m256d exp_arg = _mm256_sub_pd(_mm256_sub_pd(RS, c0p5625), _mm256_mul_pd(sax, sax));
        exp_arg = _mm256_min_pd(exp_arg, _mm256_setzero_pd());
        _mm256_store_pd(exp_buf, exp_arg);
        vector_exp_avx2(exp_buf, exp_buf, W);
        __m256d r34 = _mm256_sub_pd(one, _mm256_div_pd(_mm256_load_pd(exp_buf), sax));

        // Blend regions
        __m256d result = one;
        result =
            _mm256_blendv_pd(result, r34, _mm256_andnot_pd(m2, _mm256_cmp_pd(ax, t5, _CMP_LT_OQ)));
        result = _mm256_blendv_pd(result, r2, _mm256_andnot_pd(m1, m2));
        result = _mm256_blendv_pd(result, r1, m1);
        __m256d nan_mask = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
        result = _mm256_blendv_pd(result, x, nan_mask);
        result = _mm256_or_pd(result, sign);
        _mm256_storeu_pd(&output[i], result);
    }

    for (std::size_t i = simd_end; i < size; ++i)
        output[i] = std::erf(input[i]);
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

    const __m256d inv_two_pi = _mm256_set1_pd(1.0 / (2.0 * detail::PI));
    const __m256d two_pi = _mm256_set1_pd(2.0 * detail::PI);
    const __m256d pi = _mm256_set1_pd(detail::PI);
    const __m256d half_pi = _mm256_set1_pd(detail::PI_OVER_2);
    const __m256d neg_pi = _mm256_set1_pd(-detail::PI);
    const __m256d neg_half_pi = _mm256_set1_pd(-detail::PI_OVER_2);
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d neg_one = _mm256_set1_pd(-1.0);

    // Taylor coefficients: c_k = (-1)^k / (2k)!
    const __m256d c1 = _mm256_set1_pd(-0.5);                    // -1/2!
    const __m256d c2 = _mm256_set1_pd(4.166666666666667e-2);    //  1/4!
    const __m256d c3 = _mm256_set1_pd(-1.388888888888889e-3);   // -1/6!
    const __m256d c4 = _mm256_set1_pd(2.480158730158730e-5);    //  1/8!
    const __m256d c5 = _mm256_set1_pd(-2.755731922398589e-7);   // -1/10!
    const __m256d c6 = _mm256_set1_pd(2.087675698786810e-9);    //  1/12!
    const __m256d c7 = _mm256_set1_pd(-1.147074559772973e-11);  // -1/14!

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m256d x = _mm256_loadu_pd(&input[i]);

        // Step 1: reduce to [−π, π]
        __m256d q = _mm256_round_pd(_mm256_mul_pd(x, inv_two_pi),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d y = _mm256_sub_pd(x, _mm256_mul_pd(q, two_pi));

        // Step 2: reduce to [−π/2, π/2], tracking sign
        __m256d sign = one;
        __m256d gt_hpi = _mm256_cmp_pd(y, half_pi, _CMP_GT_OQ);
        __m256d lt_nhpi = _mm256_cmp_pd(y, neg_half_pi, _CMP_LT_OQ);

        y = _mm256_blendv_pd(y, _mm256_sub_pd(pi, y), gt_hpi);
        sign = _mm256_blendv_pd(sign, neg_one, gt_hpi);
        y = _mm256_blendv_pd(y, _mm256_sub_pd(neg_pi, y), lt_nhpi);
        sign = _mm256_blendv_pd(sign, neg_one, lt_nhpi);

        // Step 3: FMA Horner  cos(y) = 1 + y²·(c₁ + y²·(… + y²·c₇))
        __m256d y2 = _mm256_mul_pd(y, y);
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
