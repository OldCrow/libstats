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

    // Extract horizontal sum with single-instruction horizontal reduction (AVX-512DQ).
    double final_sum = _mm512_reduce_add_pd(sum);

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
    const __m512d ln2_hi = _mm512_set1_pd(0.693147180369123816490e+00);
    const __m512d ln2_lo = _mm512_set1_pd(1.90821492927058770002e-10);
    const __m512d exp_max = _mm512_set1_pd(709.782712893383996732223);
    // exp_min sits below the true underflow-to-zero threshold (exp(x) rounds to 0
    // for x < -745.1332...), so clamped lanes still produce 0 via the two-step 2^n
    // scaling below. The old -708.0 clamp pinned every x < -708 to ~3.3e-308
    // instead of flushing through the subnormal range.
    const __m512d exp_min = _mm512_set1_pd(-746.0);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d one = _mm512_set1_pd(1.0);

    const __m512d c1 = _mm512_set1_pd(0.1666666666666669072e+0);
    const __m512d c2 = _mm512_set1_pd(0.4166666666666602598e-1);
    const __m512d c3 = _mm512_set1_pd(0.8333333333314938210e-2);
    const __m512d c4 = _mm512_set1_pd(0.1388888888914497797e-2);
    const __m512d c5 = _mm512_set1_pd(0.1984126989855865850e-3);
    const __m512d c6 = _mm512_set1_pd(0.2480158687479686264e-4);
    const __m512d c7 = _mm512_set1_pd(0.2755723402025388239e-5);
    const __m512d c8 = _mm512_set1_pd(0.2755762628169491192e-6);
    const __m512d c9 = _mm512_set1_pd(0.2511210703042288022e-7);
    const __m512d c10 = _mm512_set1_pd(0.2081276378237164457e-8);

    constexpr std::size_t W = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m512d x = _mm512_loadu_pd(&values[i]);
        x = _mm512_min_pd(x, exp_max);
        x = _mm512_max_pd(x, exp_min);

        // Range reduction: x = n*ln2 + r
        __m512d n_float =
            _mm512_roundscale_pd(_mm512_mul_pd(x, ln2_inv), _MM_FROUND_TO_NEAREST_INT);
        __m512d r = _mm512_fnmadd_pd(n_float, ln2_hi, x);  // x - n*ln2_hi
        r = _mm512_fnmadd_pd(n_float, ln2_lo, r);          // r - n*ln2_lo

        // FMA Horner: P(r)
        __m512d r2 = _mm512_mul_pd(r, r);
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
        poly = _mm512_fmadd_pd(poly, r, half);  // r*P(r) + 0.5
        poly = _mm512_fmadd_pd(poly, r2, r);    // (r*P(r)+0.5)*r² + r
        poly = _mm512_add_pd(poly, one);        // 1 + r + r²*(0.5+r*P(r))

        // Scale by 2^n in two steps: n = n1 + n2 with n1 = n>>1, so each factor
        // 2^n1, 2^n2 has biased exponent n_k + 1023 in [485, 1535] and stays a
        // normal double even when n reaches -1076 (x = -746). The second multiply
        // rounds once into the subnormal range, giving graceful underflow to 0.
        // _mm512_cvtpd_epi32 and _mm512_cvtepi32_epi64 are AVX-512F (no DQ).
        __m256i n_i32 = _mm512_cvtpd_epi32(n_float);   // 8 doubles → __m256i
        __m256i n1_i32 = _mm256_srai_epi32(n_i32, 1);  // floor(n/2), arithmetic shift
        __m256i n2_i32 = _mm256_sub_epi32(n_i32, n1_i32);
        const __m512i bias = _mm512_set1_epi64(1023);
        __m512i e1 = _mm512_slli_epi64(_mm512_add_epi64(_mm512_cvtepi32_epi64(n1_i32), bias), 52);
        __m512i e2 = _mm512_slli_epi64(_mm512_add_epi64(_mm512_cvtepi32_epi64(n2_i32), bias), 52);
        __m512d scale1 = _mm512_castsi512_pd(e1);
        __m512d scale2 = _mm512_castsi512_pd(e2);

        _mm512_storeu_pd(&results[i], _mm512_mul_pd(_mm512_mul_pd(poly, scale1), scale2));
    }

    for (std::size_t i = simd_end; i < size; ++i)
        results[i] = std::exp(values[i]);
}

void VectorOps::vector_log_avx512(const double* values, double* results,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_log_fallback(values, results, size);
    }

    // SLEEF xlog_u1-inspired: 2*atanh series, < 1 ULP error.
    // FMA Horner; exponent extraction via 512-bit integer ops (AVX-512F).
    // _mm512_cvtepi64_pd requires AVX-512DQ (enabled on Zen 4 by /arch:AVX512).
    // Algorithm and coefficients identical to vector_log_avx2; width doubled to 8.

    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d ln2_hi = _mm512_set1_pd(0.693147180559945286226764);
    const __m512d ln2_lo = _mm512_set1_pd(2.319046813846299558417771e-17);
    const __m512d sqrt2 = _mm512_set1_pd(1.4142135623730950488016887242097);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d two = _mm512_set1_pd(2.0);

    // SLEEF xlog_u1 coefficients (2*atanh series)
    const __m512d c1 = _mm512_set1_pd(0.6666666666667333541e+0);
    const __m512d c2 = _mm512_set1_pd(0.3999999999635251990e+0);
    const __m512d c3 = _mm512_set1_pd(0.2857142932794299317e+0);
    const __m512d c4 = _mm512_set1_pd(0.2222214519839380009e+0);
    const __m512d c5 = _mm512_set1_pd(0.1818605932937785996e+0);
    const __m512d c6 = _mm512_set1_pd(0.1525629051003428716e+0);
    const __m512d c7 = _mm512_set1_pd(0.1532076988502701353e+0);

    const __m512d zero = _mm512_setzero_pd();
    const __m512d neg_inf = _mm512_set1_pd(-std::numeric_limits<double>::infinity());
    const __m512d pos_inf = _mm512_set1_pd(std::numeric_limits<double>::infinity());
    const __m512d nan_val = _mm512_set1_pd(std::numeric_limits<double>::quiet_NaN());

    constexpr std::size_t W = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m512d x = _mm512_loadu_pd(&values[i]);

        __mmask8 is_zero = _mm512_cmp_pd_mask(x, zero, _CMP_EQ_OQ);
        __mmask8 is_negative = _mm512_cmp_pd_mask(x, zero, _CMP_LT_OQ);
        __mmask8 is_inf = _mm512_cmp_pd_mask(x, pos_inf, _CMP_EQ_OQ);

        // Scale denormals by 2^54 to bring into normal range
        const __m512d min_normal = _mm512_set1_pd(2.2250738585072014e-308);
        const __m512d scale_up = _mm512_set1_pd(18014398509481984.0);  // 2^54
        __mmask8 is_denormal = _mm512_cmp_pd_mask(x, min_normal, _CMP_LT_OQ);
        __m512d scaled_x = _mm512_mask_blend_pd(is_denormal, x, _mm512_mul_pd(x, scale_up));

        // Exponent extraction: cast to int, shift right 52 bits, mask 11-bit
        // exponent field, subtract bias 1023. All AVX-512F.
        __m512i xi = _mm512_castpd_si512(scaled_x);
        __m512i exp_i =
            _mm512_sub_epi64(_mm512_and_si512(_mm512_srli_epi64(xi, 52), _mm512_set1_epi64(0x7FF)),
                             _mm512_set1_epi64(1023));

        // int64 → double (AVX-512DQ, enabled by /arch:AVX512 on Zen 4)
        __m512d e = _mm512_cvtepi64_pd(exp_i);
        // Adjust exponent for denormals: subtract the 54 we scaled by
        e = _mm512_mask_blend_pd(is_denormal, e, _mm512_sub_pd(e, _mm512_set1_pd(54.0)));

        // Isolate mantissa in [1, 2) by masking out the exponent field and
        // replacing it with bias 1023 (= exponent for 1.0)
        __m512i mant_i =
            _mm512_or_si512(_mm512_and_si512(xi, _mm512_set1_epi64(0x000FFFFFFFFFFFFFLL)),
                            _mm512_set1_epi64(0x3FF0000000000000LL));
        __m512d m = _mm512_castsi512_pd(mant_i);

        // Range adjustment: m → [0.5, sqrt(2)), increment e where m > sqrt(2)
        __mmask8 needs_adj = _mm512_cmp_pd_mask(m, sqrt2, _CMP_GT_OQ);
        m = _mm512_mask_blend_pd(needs_adj, m, _mm512_mul_pd(m, half));
        e = _mm512_mask_blend_pd(needs_adj, e, _mm512_add_pd(e, one));

        // xr = (m-1)/(m+1); FMA Horner: t = c7 + xr²*(c6 + ... + xr²*c1)
        __m512d xr = _mm512_div_pd(_mm512_sub_pd(m, one), _mm512_add_pd(m, one));
        __m512d xr2 = _mm512_mul_pd(xr, xr);
        __m512d t = c7;
        t = _mm512_fmadd_pd(t, xr2, c6);
        t = _mm512_fmadd_pd(t, xr2, c5);
        t = _mm512_fmadd_pd(t, xr2, c4);
        t = _mm512_fmadd_pd(t, xr2, c3);
        t = _mm512_fmadd_pd(t, xr2, c2);
        t = _mm512_fmadd_pd(t, xr2, c1);

        // log(m) = 2*xr + xr³*t  (FMA form)
        __m512d xr3 = _mm512_mul_pd(xr, xr2);
        __m512d two_xr = _mm512_mul_pd(xr, two);
        __m512d log_m = _mm512_fmadd_pd(xr3, t, two_xr);

        // log(x) = log(m) + e*ln2  (high-low FMA decomposition)
        __m512d result = _mm512_fmadd_pd(e, ln2_hi, log_m);
        result = _mm512_fmadd_pd(e, ln2_lo, result);

        result = _mm512_mask_blend_pd(is_zero, result, neg_inf);
        result = _mm512_mask_blend_pd(is_inf, result, pos_inf);
        result = _mm512_mask_blend_pd(is_negative, result, nan_val);

        _mm512_storeu_pd(&results[i], result);
    }

    for (std::size_t i = simd_end; i < size; ++i)
        results[i] = std::log(values[i]);
}

void VectorOps::vector_pow_avx512(const double* base, double exponent, double* results,
                                  std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_pow_fallback(base, exponent, results, size);
    }
    // Native 8-wide FMA path: pow(x, e) = exp(e * log(x)).
    // Uses vector_log_avx512 and vector_exp_avx512, both native 8-wide with FMA.
    // Replaces the former 4-wide AVX delegation, doubling throughput on Zen 4.
    vector_log_avx512(base, results, size);                    // results = log(base)
    scalar_multiply_avx512(results, exponent, results, size);  // results = e * log(base)
    vector_exp_avx512(results, results, size);                 // results = exp(e * log(base))
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

    // Four-region rational polynomial (musl libc / Sun Microsystems derivation),
    // < 1 ULP error. FMA Horner; __mmask8 blends replace blendv_pd.
    // Sign extraction/restoration via _mm512_and/or_pd (AVX-512DQ, /arch:AVX512).
    // Exp call uses vector_exp_avx512 (native 8-wide).
    // Algorithm, coefficients, and region boundaries identical to vector_erf_avx2.
    //
    // Region 1: |x| < 0.84375  — rational P(x²)/Q(x²),  erf(x) = x + x·R
    // Region 2: 0.84375 ≤ |x| < 1.25  — rational around x=1
    // Region 3: 1.25 ≤ |x| < 2.857  — erfc via exp(-x²-0.5625+R/S)/|x|
    // Region 4: 2.857 ≤ |x| < 6     — same structure, different coefficients
    // Region 5: |x| ≥ 6             — erf ≈ ±1

    // ---- Region 1 coefficients (rational P/Q in z = x²) ----
    const __m512d pp0 = _mm512_set1_pd(1.28379167095512558561e-01);
    const __m512d pp1 = _mm512_set1_pd(-3.25042107247001499370e-01);
    const __m512d pp2 = _mm512_set1_pd(-2.84817495755985104766e-02);
    const __m512d pp3 = _mm512_set1_pd(-5.77027029648944159157e-03);
    const __m512d pp4 = _mm512_set1_pd(-2.37630166566501626084e-05);
    const __m512d qq1 = _mm512_set1_pd(3.97917223959155352819e-01);
    const __m512d qq2 = _mm512_set1_pd(6.50222499887672944485e-02);
    const __m512d qq3 = _mm512_set1_pd(5.08130628187576562776e-03);
    const __m512d qq4 = _mm512_set1_pd(1.32494738004321644526e-04);
    const __m512d qq5 = _mm512_set1_pd(-3.96022827877536812320e-06);
    // ---- Region 2 coefficients (rational P/Q in s = |x|-1) ----
    const __m512d erx = _mm512_set1_pd(8.45062911510467529297e-01);
    const __m512d pa0 = _mm512_set1_pd(-2.36211856075265944077e-03);
    const __m512d pa1 = _mm512_set1_pd(4.14856118683748331666e-01);
    const __m512d pa2 = _mm512_set1_pd(-3.72207876035701323847e-01);
    const __m512d pa3 = _mm512_set1_pd(3.18346619901161753674e-01);
    const __m512d pa4 = _mm512_set1_pd(-1.10894694282396677476e-01);
    const __m512d pa5 = _mm512_set1_pd(3.54783043256182359371e-02);
    const __m512d pa6 = _mm512_set1_pd(-2.16637559486879084300e-03);
    const __m512d qa1 = _mm512_set1_pd(1.06420880400844228286e-01);
    const __m512d qa2 = _mm512_set1_pd(5.40397917702171048937e-01);
    const __m512d qa3 = _mm512_set1_pd(7.18286544141962662868e-02);
    const __m512d qa4 = _mm512_set1_pd(1.26171219808761642112e-01);
    const __m512d qa5 = _mm512_set1_pd(1.36370839120290507362e-02);
    const __m512d qa6 = _mm512_set1_pd(1.19844998467991074170e-02);
    // ---- Region 3 coefficients (rational R/S in s = 1/x², 1.25 ≤ |x| < 2.857) ----
    const __m512d ra0 = _mm512_set1_pd(-9.86494403484714822705e-03);
    const __m512d ra1 = _mm512_set1_pd(-6.93858572707181764372e-01);
    const __m512d ra2 = _mm512_set1_pd(-1.05586262253232909814e+01);
    const __m512d ra3 = _mm512_set1_pd(-6.23753324503260060396e+01);
    const __m512d ra4 = _mm512_set1_pd(-1.62396669462573470355e+02);
    const __m512d ra5 = _mm512_set1_pd(-1.84605092906711035994e+02);
    const __m512d ra6 = _mm512_set1_pd(-8.12874355063065934246e+01);
    const __m512d ra7 = _mm512_set1_pd(-9.81432934416914548592e+00);
    const __m512d sa1 = _mm512_set1_pd(1.96512716674392571292e+01);
    const __m512d sa2 = _mm512_set1_pd(1.37657754143519042600e+02);
    const __m512d sa3 = _mm512_set1_pd(4.34565877475229228821e+02);
    const __m512d sa4 = _mm512_set1_pd(6.45387271733267880336e+02);
    const __m512d sa5 = _mm512_set1_pd(4.29008140027567833386e+02);
    const __m512d sa6 = _mm512_set1_pd(1.08635005541779435134e+02);
    const __m512d sa7 = _mm512_set1_pd(6.57024977031928170135e+00);
    const __m512d sa8 = _mm512_set1_pd(-6.04244152148580987438e-02);
    // ---- Region 4 coefficients (rational R/S in s = 1/x², 2.857 ≤ |x| < 6) ----
    const __m512d rb0 = _mm512_set1_pd(-9.86494292470009928597e-03);
    const __m512d rb1 = _mm512_set1_pd(-7.99283237680523006574e-01);
    const __m512d rb2 = _mm512_set1_pd(-1.77579549177547519889e+01);
    const __m512d rb3 = _mm512_set1_pd(-1.60636384855821916062e+02);
    const __m512d rb4 = _mm512_set1_pd(-6.37566443368389627722e+02);
    const __m512d rb5 = _mm512_set1_pd(-1.02509513161107724954e+03);
    const __m512d rb6 = _mm512_set1_pd(-4.83519191608651397019e+02);
    const __m512d sb1 = _mm512_set1_pd(3.03380607434824582924e+01);
    const __m512d sb2 = _mm512_set1_pd(3.25792512996573918826e+02);
    const __m512d sb3 = _mm512_set1_pd(1.53672958608443695994e+03);
    const __m512d sb4 = _mm512_set1_pd(3.19985821950859553908e+03);
    const __m512d sb5 = _mm512_set1_pd(2.55305040643316442583e+03);
    const __m512d sb6 = _mm512_set1_pd(4.74528541206955367215e+02);
    const __m512d sb7 = _mm512_set1_pd(-2.24409524465858183362e+01);

    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    const __m512d t1 = _mm512_set1_pd(0.84375);
    const __m512d t2 = _mm512_set1_pd(1.25);
    const __m512d t3 = _mm512_set1_pd(2.857142857);
    const __m512d t5 = _mm512_set1_pd(6.0);
    const __m512d c0p5625 = _mm512_set1_pd(0.5625);

    constexpr std::size_t W = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / W) * W;
    alignas(64) double exp_buf[W];

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m512d x = _mm512_loadu_pd(&values[i]);
        __m512d sign = _mm512_and_pd(x, sign_mask);   // extract sign bit; AVX-512DQ
        __m512d ax = _mm512_andnot_pd(sign_mask, x);  // |x|; AVX-512DQ

        // Region masks
        __mmask8 m1 = _mm512_cmp_pd_mask(ax, t1, _CMP_LT_OQ);  // |x| < 0.84375
        __mmask8 m2 = _mm512_cmp_pd_mask(ax, t2, _CMP_LT_OQ);  // |x| < 1.25
        __mmask8 m3 = _mm512_cmp_pd_mask(ax, t3, _CMP_LT_OQ);  // |x| < 2.857

        // ---- Region 1: FMA Horner for P/Q in z = x² ----
        __m512d z = _mm512_mul_pd(ax, ax);
        __m512d P1 = pp4;
        P1 = _mm512_fmadd_pd(z, P1, pp3);
        P1 = _mm512_fmadd_pd(z, P1, pp2);
        P1 = _mm512_fmadd_pd(z, P1, pp1);
        P1 = _mm512_fmadd_pd(z, P1, pp0);
        __m512d Q1 = qq5;
        Q1 = _mm512_fmadd_pd(z, Q1, qq4);
        Q1 = _mm512_fmadd_pd(z, Q1, qq3);
        Q1 = _mm512_fmadd_pd(z, Q1, qq2);
        Q1 = _mm512_fmadd_pd(z, Q1, qq1);
        Q1 = _mm512_fmadd_pd(z, Q1, one);
        __m512d r1 = _mm512_fmadd_pd(ax, _mm512_div_pd(P1, Q1), ax);

        // ---- Region 2: FMA Horner for P/Q in s = |x|-1 ----
        __m512d s2 = _mm512_sub_pd(ax, one);
        __m512d P2 = pa6;
        P2 = _mm512_fmadd_pd(s2, P2, pa5);
        P2 = _mm512_fmadd_pd(s2, P2, pa4);
        P2 = _mm512_fmadd_pd(s2, P2, pa3);
        P2 = _mm512_fmadd_pd(s2, P2, pa2);
        P2 = _mm512_fmadd_pd(s2, P2, pa1);
        P2 = _mm512_fmadd_pd(s2, P2, pa0);
        __m512d Q2 = qa6;
        Q2 = _mm512_fmadd_pd(s2, Q2, qa5);
        Q2 = _mm512_fmadd_pd(s2, Q2, qa4);
        Q2 = _mm512_fmadd_pd(s2, Q2, qa3);
        Q2 = _mm512_fmadd_pd(s2, Q2, qa2);
        Q2 = _mm512_fmadd_pd(s2, Q2, qa1);
        Q2 = _mm512_fmadd_pd(s2, Q2, one);
        __m512d r2 = _mm512_add_pd(erx, _mm512_div_pd(P2, Q2));

        // ---- Regions 3-4: erfc = exp(-x²-0.5625+R/S)/|x|, erf = 1-erfc ----
        // Clamp sax to ≥ t2: safe 1/x² for lanes that will be blended away.
        __m512d sax = _mm512_max_pd(ax, t2);
        __m512d inv_x2 = _mm512_div_pd(one, _mm512_mul_pd(sax, sax));

        __m512d R3 = ra7;
        R3 = _mm512_fmadd_pd(inv_x2, R3, ra6);
        R3 = _mm512_fmadd_pd(inv_x2, R3, ra5);
        R3 = _mm512_fmadd_pd(inv_x2, R3, ra4);
        R3 = _mm512_fmadd_pd(inv_x2, R3, ra3);
        R3 = _mm512_fmadd_pd(inv_x2, R3, ra2);
        R3 = _mm512_fmadd_pd(inv_x2, R3, ra1);
        R3 = _mm512_fmadd_pd(inv_x2, R3, ra0);
        __m512d S3 = sa8;
        S3 = _mm512_fmadd_pd(inv_x2, S3, sa7);
        S3 = _mm512_fmadd_pd(inv_x2, S3, sa6);
        S3 = _mm512_fmadd_pd(inv_x2, S3, sa5);
        S3 = _mm512_fmadd_pd(inv_x2, S3, sa4);
        S3 = _mm512_fmadd_pd(inv_x2, S3, sa3);
        S3 = _mm512_fmadd_pd(inv_x2, S3, sa2);
        S3 = _mm512_fmadd_pd(inv_x2, S3, sa1);
        S3 = _mm512_fmadd_pd(inv_x2, S3, one);

        __m512d R4 = rb6;
        R4 = _mm512_fmadd_pd(inv_x2, R4, rb5);
        R4 = _mm512_fmadd_pd(inv_x2, R4, rb4);
        R4 = _mm512_fmadd_pd(inv_x2, R4, rb3);
        R4 = _mm512_fmadd_pd(inv_x2, R4, rb2);
        R4 = _mm512_fmadd_pd(inv_x2, R4, rb1);
        R4 = _mm512_fmadd_pd(inv_x2, R4, rb0);
        __m512d S4 = sb7;
        S4 = _mm512_fmadd_pd(inv_x2, S4, sb6);
        S4 = _mm512_fmadd_pd(inv_x2, S4, sb5);
        S4 = _mm512_fmadd_pd(inv_x2, S4, sb4);
        S4 = _mm512_fmadd_pd(inv_x2, S4, sb3);
        S4 = _mm512_fmadd_pd(inv_x2, S4, sb2);
        S4 = _mm512_fmadd_pd(inv_x2, S4, sb1);
        S4 = _mm512_fmadd_pd(inv_x2, S4, one);

        // Blend R/S: R3/S3 where |x| < 2.857, R4/S4 otherwise
        __m512d RS =
            _mm512_div_pd(_mm512_mask_blend_pd(m3, R4, R3), _mm512_mask_blend_pd(m3, S4, S3));

        // exp_arg = -x²-0.5625+R/S; clamp to ≤ 0 (erfc ≤ 1 for |x| ≥ 1.25)
        __m512d exp_arg = _mm512_sub_pd(_mm512_sub_pd(RS, c0p5625), _mm512_mul_pd(sax, sax));
        exp_arg = _mm512_min_pd(exp_arg, _mm512_setzero_pd());
        _mm512_store_pd(exp_buf, exp_arg);
        vector_exp_avx512(exp_buf, exp_buf, W);
        __m512d r34 = _mm512_sub_pd(one, _mm512_div_pd(_mm512_load_pd(exp_buf), sax));

        // ---- Blend regions (innermost wins) ----
        // ~m2 & is_lt_t5: 1.25 ≤ |x| < 6  (R3+R4)
        // ~m1 & m2:        0.84375 ≤ |x| < 1.25  (R2)
        // m1:              |x| < 0.84375  (R1)
        __mmask8 is_lt_t5 = _mm512_cmp_pd_mask(ax, t5, _CMP_LT_OQ);
        __m512d result = one;                                        // R5: ±1
        result = _mm512_mask_blend_pd(is_lt_t5 & ~m2, result, r34);  // R3+R4
        result = _mm512_mask_blend_pd(m2 & ~m1, result, r2);         // R2
        result = _mm512_mask_blend_pd(m1, result, r1);               // R1

        // Restore sign (erf is odd), then propagate NaN.
        // Order matters: or_pd must come first so NaN lanes are overwritten with
        // the original x (which carries its own sign), not a sign-corrupted NaN.
        result = _mm512_or_pd(result, sign);  // AVX-512DQ
        __mmask8 nan_mask = _mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q);
        result = _mm512_mask_blend_pd(nan_mask, result, x);

        _mm512_storeu_pd(&results[i], result);
    }

    for (std::size_t i = simd_end; i < size; ++i)
        results[i] = std::erf(values[i]);
}

void VectorOps::vector_cos_avx512(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_cos_fallback(input, output, size);
    }

    // Native 8-wide AVX-512 implementation (polynomial approximation, no SVML).

    constexpr std::size_t W = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    const __m512d inv_two_pi = _mm512_set1_pd(1.0 / (2.0 * detail::PI));
    const __m512d two_pi = _mm512_set1_pd(2.0 * detail::PI);
    const __m512d pi = _mm512_set1_pd(detail::PI);
    const __m512d half_pi = _mm512_set1_pd(detail::PI_OVER_2);
    const __m512d neg_pi = _mm512_set1_pd(-detail::PI);
    const __m512d neg_half_pi = _mm512_set1_pd(-detail::PI_OVER_2);
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d neg_one = _mm512_set1_pd(-1.0);

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
        __m512d q = _mm512_roundscale_pd(_mm512_mul_pd(x, inv_two_pi), _MM_FROUND_TO_NEAREST_INT);
        __m512d y = _mm512_sub_pd(x, _mm512_mul_pd(q, two_pi));

        // Step 2: reduce to [−π/2, π/2] using AVX-512 mask operations
        __m512d sign = one;
        __mmask8 gt_hpi = _mm512_cmp_pd_mask(y, half_pi, _CMP_GT_OQ);
        __mmask8 lt_nhpi = _mm512_cmp_pd_mask(y, neg_half_pi, _CMP_LT_OQ);

        y = _mm512_mask_blend_pd(gt_hpi, y, _mm512_sub_pd(pi, y));
        sign = _mm512_mask_blend_pd(gt_hpi, sign, neg_one);
        y = _mm512_mask_blend_pd(lt_nhpi, y, _mm512_sub_pd(neg_pi, y));
        sign = _mm512_mask_blend_pd(lt_nhpi, sign, neg_one);

        // Step 3: Horner evaluation using FMA for throughput
        __m512d y2 = _mm512_mul_pd(y, y);
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
