// SSE2-specific SIMD implementations
// This file is compiled ONLY with SSE2 flags to ensure safety

#include "libstats/common/cpu_detection_fwd.h"       // Use lightweight forward declarations
#include "libstats/common/platform_constants_fwd.h"  // Use lightweight forward declarations
#include "libstats/common/simd_implementation_common.h"  // VectorOps class + cpu_detection_fwd + platform_constants_fwd
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

// SSE2 blend helper — must be defined before vector_log_sse2 and vector_erf_sse2 use it.
// Selects true_val where mask = all-ones, false_val elsewhere (no SSE4.1 blendv).
#define SSE2_BLEND(mask, true_val, false_val)                                                      \
    _mm_or_pd(_mm_and_pd((mask), (true_val)), _mm_andnot_pd((mask), (false_val)))

// SSE2 2-wide vector_exp and vector_log: same SLEEF Horner polynomial as AVX,
// ported to __m128d. No FMA on SSE2 — use mul+add. No _mm_round_pd (SSE4.1) —
// use magic-number rounding. No _mm_cvtepi32_epi64 (SSE4.1) — use unpacklo+slli.
// Unlocks vector_erf_sse2's inner exp call: erf regions 3-4 now run 2-wide.

void VectorOps::vector_exp_sse2(const double* values, double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_exp_fallback(values, results, size);
    }

    // Constants (identical coefficients to AVX/AVX2/AVX-512 paths)
    const __m128d ln2_inv = _mm_set1_pd(1.4426950408889634073599246810019);
    const __m128d ln2_hi = _mm_set1_pd(0.693147180369123816490e+00);
    const __m128d ln2_lo = _mm_set1_pd(1.90821492927058770002e-10);
    const __m128d exp_max = _mm_set1_pd(709.782712893383996732223);
    const __m128d exp_min = _mm_set1_pd(-708.0);
    const __m128d half = _mm_set1_pd(0.5);
    const __m128d one = _mm_set1_pd(1.0);
    // Magic-number rounding constant: 2^52 + 2^51 = 6755399441055744; adding it to x
    // rounds x (as double) to integer in the binade [2^51, 2^52], then subtracting it back
    // yields round(x) without SSE4.1 _mm_round_pd.
    const __m128d magic = _mm_set1_pd(6755399441055744.0);

    const __m128d c1 = _mm_set1_pd(0.1666666666666669072e+0);
    const __m128d c2 = _mm_set1_pd(0.4166666666666602598e-1);
    const __m128d c3 = _mm_set1_pd(0.8333333333314938210e-2);
    const __m128d c4 = _mm_set1_pd(0.1388888888914497797e-2);
    const __m128d c5 = _mm_set1_pd(0.1984126989855865850e-3);
    const __m128d c6 = _mm_set1_pd(0.2480158687479686264e-4);
    const __m128d c7 = _mm_set1_pd(0.2755723402025388239e-5);
    const __m128d c8 = _mm_set1_pd(0.2755762628169491192e-6);
    const __m128d c9 = _mm_set1_pd(0.2511210703042288022e-7);
    const __m128d c10 = _mm_set1_pd(0.2081276378237164457e-8);

    constexpr std::size_t W = arch::simd::SSE_DOUBLES;  // 2
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m128d x = _mm_loadu_pd(&values[i]);
        x = _mm_min_pd(x, exp_max);
        x = _mm_max_pd(x, exp_min);

        // Range reduction: n = round(x / ln2) via magic-number trick
        __m128d n_float = _mm_add_pd(_mm_mul_pd(x, ln2_inv), magic);
        n_float = _mm_sub_pd(n_float, magic);  // n_float = round(x/ln2)

        // r = x - n*ln2_hi - n*ln2_lo  (two-part ln2 for precision)
        __m128d r = _mm_sub_pd(x, _mm_mul_pd(n_float, ln2_hi));
        r = _mm_sub_pd(r, _mm_mul_pd(n_float, ln2_lo));

        // 10-term Horner polynomial P(r)
        __m128d r2 = _mm_mul_pd(r, r);
        __m128d poly = c10;
        poly = _mm_add_pd(_mm_mul_pd(poly, r), c9);
        poly = _mm_add_pd(_mm_mul_pd(poly, r), c8);
        poly = _mm_add_pd(_mm_mul_pd(poly, r), c7);
        poly = _mm_add_pd(_mm_mul_pd(poly, r), c6);
        poly = _mm_add_pd(_mm_mul_pd(poly, r), c5);
        poly = _mm_add_pd(_mm_mul_pd(poly, r), c4);
        poly = _mm_add_pd(_mm_mul_pd(poly, r), c3);
        poly = _mm_add_pd(_mm_mul_pd(poly, r), c2);
        poly = _mm_add_pd(_mm_mul_pd(poly, r), c1);
        // Complete: exp(r) = 1 + r + r^2*(0.5 + r*P(r))
        poly = _mm_add_pd(_mm_mul_pd(poly, r), half);
        poly = _mm_add_pd(_mm_mul_pd(poly, r2), r);
        poly = _mm_add_pd(poly, one);

        // Scale by 2^n: convert n to 32-bit int, zero-extend to 64-bit, shift left 52.
        // _mm_cvttpd_epi32 gives [n0, n1, 0, 0] as 4x32-bit;
        // _mm_unpacklo_epi32 with zero gives [n0, 0, n1, 0] as 32-bit = [n0, n1] as 64-bit;
        // add bias 1023 in 64-bit; shift left 52 to place in IEEE 754 exponent field.
        __m128i n_i32 = _mm_cvttpd_epi32(n_float);                       // [n0,n1,0,0] as 4x32-bit
        __m128i n_i64 = _mm_unpacklo_epi32(n_i32, _mm_setzero_si128());  // zero-extend to 64-bit
        __m128i ebits = _mm_add_epi64(n_i64, _mm_set1_epi64x(1023LL));
        ebits = _mm_slli_epi64(ebits, 52);
        __m128d scale = _mm_castsi128_pd(ebits);

        _mm_storeu_pd(&results[i], _mm_mul_pd(poly, scale));
    }

    for (std::size_t i = simd_end; i < size; ++i)
        results[i] = std::exp(values[i]);
}

void VectorOps::vector_log_sse2(const double* values, double* results, std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_log_fallback(values, results, size);
    }

    const __m128d one = _mm_set1_pd(1.0);
    const __m128d half = _mm_set1_pd(0.5);
    const __m128d two = _mm_set1_pd(2.0);
    const __m128d ln2_hi = _mm_set1_pd(0.693147180559945286226764);
    const __m128d ln2_lo = _mm_set1_pd(2.319046813846299558417771e-17);
    const __m128d sqrt2 = _mm_set1_pd(1.4142135623730950488016887242097);
    const __m128d neg_inf = _mm_set1_pd(-std::numeric_limits<double>::infinity());
    const __m128d pos_inf = _mm_set1_pd(std::numeric_limits<double>::infinity());
    const __m128d nan_val = _mm_set1_pd(std::numeric_limits<double>::quiet_NaN());
    const __m128d zero = _mm_setzero_pd();

    // SLEEF atanh series coefficients (identical to AVX/AVX2 paths)
    const __m128d c1 = _mm_set1_pd(0.6666666666667333541e+0);
    const __m128d c2 = _mm_set1_pd(0.3999999999635251990e+0);
    const __m128d c3 = _mm_set1_pd(0.2857142932794299317e+0);
    const __m128d c4 = _mm_set1_pd(0.2222214519839380009e+0);
    const __m128d c5 = _mm_set1_pd(0.1818605932937785996e+0);
    const __m128d c6 = _mm_set1_pd(0.1525629051003428716e+0);
    const __m128d c7 = _mm_set1_pd(0.1532076988502701353e+0);

    constexpr std::size_t W = arch::simd::SSE_DOUBLES;  // 2
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m128d x = _mm_loadu_pd(&values[i]);

        // Special-case detection
        __m128d is_zero = _mm_cmpeq_pd(x, zero);
        __m128d is_negative = _mm_cmplt_pd(x, zero);
        __m128d is_inf = _mm_cmpeq_pd(x, pos_inf);

        // Exponent extraction: cast to int, srli_epi64 by 52, mask 11-bit field, subtract 1023.
        // _mm_srli_epi64 shifts each 64-bit element right: exponent lands in bits [0..10].
        // _mm_shuffle_epi32 + _mm_cvtepi32_pd converts the two 32-bit exponent values to double
        // without SSE4.1 _mm_cvtepi64_pd: the exponent fits in 32 bits (0..2046).
        __m128i xi = _mm_castpd_si128(x);
        __m128i exp_i = _mm_srli_epi64(xi, 52);
        exp_i = _mm_and_si128(exp_i, _mm_set1_epi64x(0x7FFLL));
        // exp_i = [exp0:64, exp1:64]; lower 32 bits of each 64-bit lane hold the value.
        // Shuffle to [exp0_lo32, exp1_lo32, _, _] then cvtepi32_pd uses positions [0] and [1].
        __m128i exp_i32 = _mm_shuffle_epi32(exp_i, _MM_SHUFFLE(0, 0, 2, 0));
        __m128d e = _mm_cvtepi32_pd(exp_i32);
        e = _mm_sub_pd(e, _mm_set1_pd(1023.0));

        // Mantissa: clear exponent field, set exponent to 1023 (=> m in [1,2))
        __m128i mant_mask = _mm_set1_epi64x(0x000FFFFFFFFFFFFFLL);
        __m128i exp_bias = _mm_set1_epi64x(0x3FF0000000000000LL);
        __m128d m = _mm_castsi128_pd(_mm_or_si128(_mm_and_si128(xi, mant_mask), exp_bias));

        // Range adjust: if m > sqrt(2), halve m and increment e
        __m128d need_adj = _mm_cmpgt_pd(m, sqrt2);
        m = SSE2_BLEND(need_adj, _mm_mul_pd(m, half), m);
        e = SSE2_BLEND(need_adj, _mm_add_pd(e, one), e);

        // xr = (m-1)/(m+1); Horner: t = c7 + xr^2*(c6 + ... + xr^2*c1)
        __m128d xr = _mm_div_pd(_mm_sub_pd(m, one), _mm_add_pd(m, one));
        __m128d xr2 = _mm_mul_pd(xr, xr);
        __m128d t = c7;
        t = _mm_add_pd(_mm_mul_pd(t, xr2), c6);
        t = _mm_add_pd(_mm_mul_pd(t, xr2), c5);
        t = _mm_add_pd(_mm_mul_pd(t, xr2), c4);
        t = _mm_add_pd(_mm_mul_pd(t, xr2), c3);
        t = _mm_add_pd(_mm_mul_pd(t, xr2), c2);
        t = _mm_add_pd(_mm_mul_pd(t, xr2), c1);

        // log(m) = 2*xr + xr^3*t
        __m128d xr3 = _mm_mul_pd(xr, xr2);
        __m128d log_m = _mm_add_pd(_mm_mul_pd(xr, two), _mm_mul_pd(xr3, t));

        // log(x) = e*ln2_hi + log_m + e*ln2_lo  (same ordering as AVX2/NEON)
        __m128d result = _mm_add_pd(_mm_mul_pd(e, ln2_hi), log_m);
        result = _mm_add_pd(result, _mm_mul_pd(e, ln2_lo));

        // Special-case overrides
        result = SSE2_BLEND(is_zero, neg_inf, result);
        result = SSE2_BLEND(is_inf, pos_inf, result);
        result = SSE2_BLEND(is_negative, nan_val, result);

        _mm_storeu_pd(&results[i], result);
    }

    for (std::size_t i = simd_end; i < size; ++i)
        results[i] =
            values[i] > 0.0 ? std::log(values[i]) : -std::numeric_limits<double>::infinity();
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

void VectorOps::vector_erf_sse2(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_sse2()) {
        return vector_erf_fallback(input, output, size);
    }

    // __m128d port of the musl rational polynomial (vector_erf_avx algorithm).
    // SSE2 has no FMA and no _mm_blendv_pd (SSE4.1); blending uses and/andnot/or.
    // SSE2 comparisons use _mm_cmpgt_pd / _mm_cmplt_pd / _mm_cmpge_pd.
    // Exp call uses vector_exp_sse2 (scalar loop) for regions 3-4.
    // Error: < 1 ULP throughout (same musl coefficients as AVX/AVX2 versions).

    // Region 1
    const __m128d pp0 = _mm_set1_pd(1.28379167095512558561e-01);
    const __m128d pp1 = _mm_set1_pd(-3.25042107247001499370e-01);
    const __m128d pp2 = _mm_set1_pd(-2.84817495755985104766e-02);
    const __m128d pp3 = _mm_set1_pd(-5.77027029648944159157e-03);
    const __m128d pp4 = _mm_set1_pd(-2.37630166566501626084e-05);
    const __m128d qq1 = _mm_set1_pd(3.97917223959155352819e-01);
    const __m128d qq2 = _mm_set1_pd(6.50222499887672944485e-02);
    const __m128d qq3 = _mm_set1_pd(5.08130628187576562776e-03);
    const __m128d qq4 = _mm_set1_pd(1.32494738004321644526e-04);
    const __m128d qq5 = _mm_set1_pd(-3.96022827877536812320e-06);
    // Region 2
    const __m128d erx = _mm_set1_pd(8.45062911510467529297e-01);
    const __m128d pa0 = _mm_set1_pd(-2.36211856075265944077e-03);
    const __m128d pa1 = _mm_set1_pd(4.14856118683748331666e-01);
    const __m128d pa2 = _mm_set1_pd(-3.72207876035701323847e-01);
    const __m128d pa3 = _mm_set1_pd(3.18346619901161753674e-01);
    const __m128d pa4 = _mm_set1_pd(-1.10894694282396677476e-01);
    const __m128d pa5 = _mm_set1_pd(3.54783043256182359371e-02);
    const __m128d pa6 = _mm_set1_pd(-2.16637559486879084300e-03);
    const __m128d qa1 = _mm_set1_pd(1.06420880400844228286e-01);
    const __m128d qa2 = _mm_set1_pd(5.40397917702171048937e-01);
    const __m128d qa3 = _mm_set1_pd(7.18286544141962662868e-02);
    const __m128d qa4 = _mm_set1_pd(1.26171219808761642112e-01);
    const __m128d qa5 = _mm_set1_pd(1.36370839120290507362e-02);
    const __m128d qa6 = _mm_set1_pd(1.19844998467991074170e-02);
    // Region 3
    const __m128d ra0 = _mm_set1_pd(-9.86494403484714822705e-03);
    const __m128d ra1 = _mm_set1_pd(-6.93858572707181764372e-01);
    const __m128d ra2 = _mm_set1_pd(-1.05586262253232909814e+01);
    const __m128d ra3 = _mm_set1_pd(-6.23753324503260060396e+01);
    const __m128d ra4 = _mm_set1_pd(-1.62396669462573470355e+02);
    const __m128d ra5 = _mm_set1_pd(-1.84605092906711035994e+02);
    const __m128d ra6 = _mm_set1_pd(-8.12874355063065934246e+01);
    const __m128d ra7 = _mm_set1_pd(-9.81432934416914548592e+00);
    const __m128d sa1 = _mm_set1_pd(1.96512716674392571292e+01);
    const __m128d sa2 = _mm_set1_pd(1.37657754143519042600e+02);
    const __m128d sa3 = _mm_set1_pd(4.34565877475229228821e+02);
    const __m128d sa4 = _mm_set1_pd(6.45387271733267880336e+02);
    const __m128d sa5 = _mm_set1_pd(4.29008140027567833386e+02);
    const __m128d sa6 = _mm_set1_pd(1.08635005541779435134e+02);
    const __m128d sa7 = _mm_set1_pd(6.57024977031928170135e+00);
    const __m128d sa8 = _mm_set1_pd(-6.04244152148580987438e-02);
    // Region 4
    const __m128d rb0 = _mm_set1_pd(-9.86494292470009928597e-03);
    const __m128d rb1 = _mm_set1_pd(-7.99283237680523006574e-01);
    const __m128d rb2 = _mm_set1_pd(-1.77579549177547519889e+01);
    const __m128d rb3 = _mm_set1_pd(-1.60636384855821916062e+02);
    const __m128d rb4 = _mm_set1_pd(-6.37566443368389627722e+02);
    const __m128d rb5 = _mm_set1_pd(-1.02509513161107724954e+03);
    const __m128d rb6 = _mm_set1_pd(-4.83519191608651397019e+02);
    const __m128d sb1 = _mm_set1_pd(3.03380607434824582924e+01);
    const __m128d sb2 = _mm_set1_pd(3.25792512996573918826e+02);
    const __m128d sb3 = _mm_set1_pd(1.53672958608443695994e+03);
    const __m128d sb4 = _mm_set1_pd(3.19985821950859553908e+03);
    const __m128d sb5 = _mm_set1_pd(2.55305040643316442583e+03);
    const __m128d sb6 = _mm_set1_pd(4.74528541206955367215e+02);
    const __m128d sb7 = _mm_set1_pd(-2.24409524465858183362e+01);

    const __m128d one = _mm_set1_pd(1.0);
    const __m128d sign_mask = _mm_set1_pd(-0.0);
    const __m128d t1 = _mm_set1_pd(0.84375);
    const __m128d t2 = _mm_set1_pd(1.25);
    const __m128d t3 = _mm_set1_pd(2.857142857);
    const __m128d t5 = _mm_set1_pd(6.0);
    const __m128d c0p5625 = _mm_set1_pd(0.5625);

    constexpr std::size_t W = arch::simd::SSE_DOUBLES;
    const std::size_t simd_end = (size / W) * W;
    alignas(16) double exp_buf[W];

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m128d x = _mm_loadu_pd(&input[i]);
        __m128d sign = _mm_and_pd(x, sign_mask);
        __m128d ax = _mm_andnot_pd(sign_mask, x);

        // Region masks (SSE2 comparison returns all-ones mask)
        __m128d m1 = _mm_cmplt_pd(ax, t1);  // |x| < 0.84375
        __m128d m2 = _mm_cmplt_pd(ax, t2);  // |x| < 1.25
        __m128d m3 = _mm_cmplt_pd(ax, t3);  // |x| < 2.857

        // Region 1: Horner for P/Q in z = x^2
        __m128d z = _mm_mul_pd(ax, ax);
        __m128d P1 = pp4;
        P1 = _mm_add_pd(pp3, _mm_mul_pd(z, P1));
        P1 = _mm_add_pd(pp2, _mm_mul_pd(z, P1));
        P1 = _mm_add_pd(pp1, _mm_mul_pd(z, P1));
        P1 = _mm_add_pd(pp0, _mm_mul_pd(z, P1));
        __m128d Q1 = qq5;
        Q1 = _mm_add_pd(qq4, _mm_mul_pd(z, Q1));
        Q1 = _mm_add_pd(qq3, _mm_mul_pd(z, Q1));
        Q1 = _mm_add_pd(qq2, _mm_mul_pd(z, Q1));
        Q1 = _mm_add_pd(qq1, _mm_mul_pd(z, Q1));
        Q1 = _mm_add_pd(one, _mm_mul_pd(z, Q1));
        __m128d r1 = _mm_add_pd(ax, _mm_mul_pd(ax, _mm_div_pd(P1, Q1)));

        // Region 2: Horner for P/Q in s = |x|-1
        __m128d s2 = _mm_sub_pd(ax, one);
        __m128d P2 = pa6;
        P2 = _mm_add_pd(pa5, _mm_mul_pd(s2, P2));
        P2 = _mm_add_pd(pa4, _mm_mul_pd(s2, P2));
        P2 = _mm_add_pd(pa3, _mm_mul_pd(s2, P2));
        P2 = _mm_add_pd(pa2, _mm_mul_pd(s2, P2));
        P2 = _mm_add_pd(pa1, _mm_mul_pd(s2, P2));
        P2 = _mm_add_pd(pa0, _mm_mul_pd(s2, P2));
        __m128d Q2 = qa6;
        Q2 = _mm_add_pd(qa5, _mm_mul_pd(s2, Q2));
        Q2 = _mm_add_pd(qa4, _mm_mul_pd(s2, Q2));
        Q2 = _mm_add_pd(qa3, _mm_mul_pd(s2, Q2));
        Q2 = _mm_add_pd(qa2, _mm_mul_pd(s2, Q2));
        Q2 = _mm_add_pd(qa1, _mm_mul_pd(s2, Q2));
        Q2 = _mm_add_pd(one, _mm_mul_pd(s2, Q2));
        __m128d r2 = _mm_add_pd(erx, _mm_div_pd(P2, Q2));

        // Regions 3-4: erfc = exp(-x^2-0.5625+R/S)/|x|
        __m128d sax = _mm_max_pd(ax, t2);
        __m128d inv_x2 = _mm_div_pd(one, _mm_mul_pd(sax, sax));

        __m128d R3 = ra7;
        R3 = _mm_add_pd(ra6, _mm_mul_pd(inv_x2, R3));
        R3 = _mm_add_pd(ra5, _mm_mul_pd(inv_x2, R3));
        R3 = _mm_add_pd(ra4, _mm_mul_pd(inv_x2, R3));
        R3 = _mm_add_pd(ra3, _mm_mul_pd(inv_x2, R3));
        R3 = _mm_add_pd(ra2, _mm_mul_pd(inv_x2, R3));
        R3 = _mm_add_pd(ra1, _mm_mul_pd(inv_x2, R3));
        R3 = _mm_add_pd(ra0, _mm_mul_pd(inv_x2, R3));
        __m128d S3 = sa8;
        S3 = _mm_add_pd(sa7, _mm_mul_pd(inv_x2, S3));
        S3 = _mm_add_pd(sa6, _mm_mul_pd(inv_x2, S3));
        S3 = _mm_add_pd(sa5, _mm_mul_pd(inv_x2, S3));
        S3 = _mm_add_pd(sa4, _mm_mul_pd(inv_x2, S3));
        S3 = _mm_add_pd(sa3, _mm_mul_pd(inv_x2, S3));
        S3 = _mm_add_pd(sa2, _mm_mul_pd(inv_x2, S3));
        S3 = _mm_add_pd(sa1, _mm_mul_pd(inv_x2, S3));
        S3 = _mm_add_pd(one, _mm_mul_pd(inv_x2, S3));

        __m128d R4 = rb6;
        R4 = _mm_add_pd(rb5, _mm_mul_pd(inv_x2, R4));
        R4 = _mm_add_pd(rb4, _mm_mul_pd(inv_x2, R4));
        R4 = _mm_add_pd(rb3, _mm_mul_pd(inv_x2, R4));
        R4 = _mm_add_pd(rb2, _mm_mul_pd(inv_x2, R4));
        R4 = _mm_add_pd(rb1, _mm_mul_pd(inv_x2, R4));
        R4 = _mm_add_pd(rb0, _mm_mul_pd(inv_x2, R4));
        __m128d S4 = sb7;
        S4 = _mm_add_pd(sb6, _mm_mul_pd(inv_x2, S4));
        S4 = _mm_add_pd(sb5, _mm_mul_pd(inv_x2, S4));
        S4 = _mm_add_pd(sb4, _mm_mul_pd(inv_x2, S4));
        S4 = _mm_add_pd(sb3, _mm_mul_pd(inv_x2, S4));
        S4 = _mm_add_pd(sb2, _mm_mul_pd(inv_x2, S4));
        S4 = _mm_add_pd(sb1, _mm_mul_pd(inv_x2, S4));
        S4 = _mm_add_pd(one, _mm_mul_pd(inv_x2, S4));

        __m128d RS = _mm_div_pd(SSE2_BLEND(m3, R3, R4), SSE2_BLEND(m3, S3, S4));
        __m128d exp_arg = _mm_sub_pd(_mm_sub_pd(RS, c0p5625), _mm_mul_pd(sax, sax));
        exp_arg = _mm_min_pd(exp_arg, _mm_setzero_pd());
        _mm_store_pd(exp_buf, exp_arg);
        vector_exp_sse2(exp_buf, exp_buf, W);
        __m128d r34 = _mm_sub_pd(one, _mm_div_pd(_mm_load_pd(exp_buf), sax));

        // Blend regions (innermost wins)
        __m128d m34 = _mm_andnot_pd(m2, _mm_cmplt_pd(ax, t5));  // 1.25 <= |x| < 6
        __m128d result = one;                                   // R5 default
        result = SSE2_BLEND(m34, r34, result);
        result = SSE2_BLEND(_mm_andnot_pd(m1, m2), r2, result);
        result = SSE2_BLEND(m1, r1, result);
        // Restore sign (erf is odd), then propagate NaN.
        // Order matters: or_pd must come first so NaN lanes are overwritten with
        // the original x (which carries its own sign), not a sign-corrupted NaN.
        result = _mm_or_pd(result, sign);
        __m128d nan_mask = _mm_cmpunord_pd(x, x);
        result = SSE2_BLEND(nan_mask, x, result);
        _mm_storeu_pd(&output[i], result);
    }

#undef SSE2_BLEND

    for (std::size_t i = simd_end; i < size; ++i)
        output[i] = std::erf(input[i]);
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

    const __m128d inv_two_pi = _mm_set1_pd(1.0 / (2.0 * detail::PI));
    const __m128d two_pi = _mm_set1_pd(2.0 * detail::PI);
    const __m128d pi = _mm_set1_pd(detail::PI);
    const __m128d half_pi = _mm_set1_pd(detail::PI_OVER_2);
    const __m128d neg_pi = _mm_set1_pd(-detail::PI);
    const __m128d neg_half_pi = _mm_set1_pd(-detail::PI_OVER_2);
    const __m128d one = _mm_set1_pd(1.0);
    const __m128d neg_one = _mm_set1_pd(-1.0);
    // 2^52 + 2^51 — adding then subtracting rounds to nearest integer
    const __m128d magic = _mm_set1_pd(6755399441055744.0);

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
        __m128d q = _mm_sub_pd(_mm_add_pd(scaled, magic), magic);  // round-to-nearest
        __m128d y = _mm_sub_pd(x, _mm_mul_pd(q, two_pi));

        // Step 2: reduce to [-π/2, π/2], tracking sign
        // SSE2 comparison returns all-ones (true) or all-zeros (false) per lane.
        __m128d sign = one;
        __m128d gt_hpi = _mm_cmpgt_pd(y, half_pi);
        __m128d lt_nhpi = _mm_cmplt_pd(y, neg_half_pi);

        // Blend: select new_y when mask is true, else keep y
        __m128d new_y_gt = _mm_sub_pd(pi, y);
        __m128d new_y_lt = _mm_sub_pd(neg_pi, y);
        y = _mm_or_pd(_mm_and_pd(gt_hpi, new_y_gt), _mm_andnot_pd(gt_hpi, y));
        sign = _mm_or_pd(_mm_and_pd(gt_hpi, neg_one), _mm_andnot_pd(gt_hpi, sign));
        y = _mm_or_pd(_mm_and_pd(lt_nhpi, new_y_lt), _mm_andnot_pd(lt_nhpi, y));
        sign = _mm_or_pd(_mm_and_pd(lt_nhpi, neg_one), _mm_andnot_pd(lt_nhpi, sign));

        // Step 3: Horner evaluation
        __m128d y2 = _mm_mul_pd(y, y);
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
