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
    const __m512d pos_inf = _mm512_set1_pd(std::numeric_limits<double>::infinity());

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
        __m512d x_orig = _mm512_loadu_pd(&values[i]);
        __m512d x = _mm512_min_pd(x_orig, exp_max);
        x = _mm512_max_pd(x, exp_min);

        // Range reduction: x = n*ln2 + r
        // _MM_FROUND_NO_EXC (imm8[3], "suppress precision exception") matches the
        // AVX2/AVX _mm256_round_pd(..., _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
        // call below in intent: without it, VRNDSCALEPD raises #PE (masked by default,
        // but a real behavioral gap vs the other tiers if the caller unmasks FP
        // exceptions).
        __m512d n_float = _mm512_roundscale_pd(_mm512_mul_pd(x, ln2_inv),
                                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
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

        __m512d result = _mm512_mul_pd(_mm512_mul_pd(poly, scale1), scale2);
        // Match std::exp at the non-finite/overflow edges: x > exp_max (incl.
        // +inf) -> +inf, NaN -> NaN. Underflow/-inf already flush to +0 via
        // exp_min + two-step scaling. Keeps the SIMD body consistent with the
        // scalar remainder loop below (std::exp) and the NEON kernel.
        result = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x_orig, exp_max, _CMP_GT_OQ), result,
                                      pos_inf);
        result = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x_orig, x_orig, _CMP_UNORD_Q), result,
                                      x_orig);
        _mm512_storeu_pd(&results[i], result);
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
        result = _mm512_mask_blend_pd(_mm512_cmp_pd_mask(x, x, _CMP_UNORD_Q), result, x);

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

    // Clean-room quadrant-reduction cos/sin (issue #95), AVX-512 FMA form --
    // ported structurally from libhmm's cos_pd/sin_pd(__m512d) (issue #74
    // there), itself ported from this project's own vector_cos_neon. See
    // docs/NEON_TRIG_DERIVATION.md and docs/NEON_TRIG_DIVERGENCE_AUDIT.md for
    // the mathematics; scripts/gen_trig_cleanroom_table.py regenerates and
    // self-checks src/trig_cleanroom_data.inc against src/neon_trig_cleanroom_data.inc.
    // No third-party source. _mm512_cvtepi32_epi64 requires AVX-512F;
    // _mm512_xor_pd requires AVX-512DQ (this TU is compiled with -mavx512f
    // -mavx512dq / /arch:AVX512).
    #include "trig_cleanroom_data.inc"

// Reduction shared by cos_avx512/sin_avx512: n32 = round-to-nearest-even(x*2/pi)
// via the cvt round-trip (exact-product lemma holds for |n| <= 5,340,354,
// i.e. |x| <= kTrigDMax = 2^23); r/rlo carry the reduced argument
// compensated.
static inline void trig_reduce_8pd(__m512d x, __m512d& r, __m512d& rlo, __m512i& n64) noexcept {
    const __m256i n32 = _mm512_cvtpd_epi32(_mm512_mul_pd(x, _mm512_set1_pd(kTrigTwoOverPi)));
    const __m512d nf = _mm512_cvtepi32_pd(n32);  // exact
    n64 = _mm512_cvtepi32_epi64(n32);

    r = _mm512_fnmadd_pd(nf, _mm512_set1_pd(kTrigPio2[0]), x);  // exact (step 1)
    rlo = _mm512_setzero_pd();
    for (int k = 1; k < 4; ++k) {
        const __m512d pk = _mm512_set1_pd(kTrigPio2[k]);
        const __m512d rk = _mm512_fnmadd_pd(nf, pk, r);
        const __m512d e = _mm512_fnmadd_pd(nf, pk, _mm512_sub_pd(r, rk));
        rlo = _mm512_add_pd(rlo, e);
        r = rk;
    }
}

// Degree-6 minimax parity cores on u = r*r; cos's 1 - u/2 head is split into
// an exact (h, hl) pair (kTrigCosC[0] == -0.5 exactly, generator-asserted).
static inline void trig_cores_8pd(__m512d r, __m512d rlo, __m512d& s_core,
                                  __m512d& c_core) noexcept {
    const __m512d u = _mm512_mul_pd(r, r);

    __m512d ps = _mm512_set1_pd(kTrigSinC[6]);
    for (int i = 5; i >= 0; --i)
        ps = _mm512_fmadd_pd(ps, u, _mm512_set1_pd(kTrigSinC[i]));
    s_core = _mm512_add_pd(r, _mm512_fmadd_pd(_mm512_mul_pd(r, u), ps, rlo));

    __m512d pc = _mm512_set1_pd(kTrigCosC[6]);
    for (int i = 5; i >= 1; --i)
        pc = _mm512_fmadd_pd(pc, u, _mm512_set1_pd(kTrigCosC[i]));
    const __m512d one_c = _mm512_set1_pd(1.0);
    const __m512d half_c = _mm512_set1_pd(0.5);
    const __m512d h = _mm512_fnmadd_pd(u, half_c, one_c);                     // 1 - u/2, exact
    const __m512d hl = _mm512_fnmadd_pd(u, half_c, _mm512_sub_pd(one_c, h));  // (1-h) - u/2, exact
    __m512d mc = _mm512_fmadd_pd(_mm512_mul_pd(u, u), pc, hl);
    mc = _mm512_fnmadd_pd(r, rlo, mc);  // first-order effect of compensated reduction
    c_core = _mm512_add_pd(h, mc);
}

void VectorOps::vector_cos_avx512(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_cos_fallback(input, output, size);
    }

    // cos(x) for |x| <= kTrigDMax (2^23); scalar fixup beyond (and for inf;
    // NaN self-propagates through the polynomial path, see trig_reduce_8pd).
    // Quadrant table: q=0:+c 1:-s 2:-c 3:+s -> swap core on bit0, sign on
    // bit1 XOR bit0 (both taken from the low bits of n's two's-complement
    // form). Max ~1 ULP measured on the FMA tiers.
    constexpr std::size_t W = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / W) * W;
    const __m512d domain_bound = _mm512_set1_pd(kTrigDMax);
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    const __m512i one_i = _mm512_set1_epi64(1);

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m512d x = _mm512_loadu_pd(&input[i]);
        __m512d ax = _mm512_andnot_pd(sign_mask, x);

        __m512d r, rlo;
        __m512i n64;
        trig_reduce_8pd(x, r, rlo, n64);
        __m512d s_core, c_core;
        trig_cores_8pd(r, rlo, s_core, c_core);

        const __mmask8 swap = _mm512_test_epi64_mask(n64, one_i);
        const __m512i bit0 = _mm512_and_si512(n64, one_i);
        const __m512i bit1 = _mm512_and_si512(_mm512_srli_epi64(n64, 1), one_i);
        const __m512i sign_bit = _mm512_xor_si512(bit1, bit0);
        const __m512d cv = _mm512_mask_blend_pd(swap, c_core, s_core);
        const __m512d sign_v = _mm512_castsi512_pd(_mm512_slli_epi64(sign_bit, 63));
        _mm512_storeu_pd(&output[i], _mm512_xor_pd(cv, sign_v));

        // Out-of-domain / inf fixup decided from the pre-store REGISTER value
        // `x`, so this stays correct when output aliases input.
        const __mmask8 mask = _mm512_cmp_pd_mask(ax, domain_bound, _CMP_GT_OQ);
        if (mask) {
            alignas(64) double xbuf[8];
            _mm512_store_pd(xbuf, x);
            for (int lane = 0; lane < 8; ++lane)
                if (mask & (1 << lane))
                    output[i + static_cast<std::size_t>(lane)] = std::cos(xbuf[lane]);
        }
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::cos(input[i]);
    }
}

void VectorOps::vector_sin_avx512(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx512()) {
        return vector_sin_fallback(input, output, size);
    }

    // sin(x) for |x| <= kTrigDMax (2^23) -- see vector_cos_avx512's comment
    // for the domain contract. Quadrant table: q=0:+s 1:+c 2:-s 3:-c -> swap
    // core on bit0 (opposite selection order from cos), sign on bit1 alone.
    // Computed from the quadrant table directly, NOT cos(x - pi/2) (that
    // composition loses accuracy through the extra subtraction).
    constexpr std::size_t W = arch::simd::AVX512_DOUBLES;
    const std::size_t simd_end = (size / W) * W;
    const __m512d domain_bound = _mm512_set1_pd(kTrigDMax);
    const __m512d sign_mask = _mm512_set1_pd(-0.0);
    const __m512i one_i = _mm512_set1_epi64(1);

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m512d x = _mm512_loadu_pd(&input[i]);
        __m512d ax = _mm512_andnot_pd(sign_mask, x);

        __m512d r, rlo;
        __m512i n64;
        trig_reduce_8pd(x, r, rlo, n64);
        __m512d s_core, c_core;
        trig_cores_8pd(r, rlo, s_core, c_core);

        const __mmask8 swap = _mm512_test_epi64_mask(n64, one_i);
        const __m512i bit1 = _mm512_and_si512(_mm512_srli_epi64(n64, 1), one_i);
        const __m512d sv = _mm512_mask_blend_pd(swap, s_core, c_core);
        const __m512d sign_v = _mm512_castsi512_pd(_mm512_slli_epi64(bit1, 63));
        __m512d result = _mm512_xor_pd(sv, sign_v);
        // IEEE sign-of-zero: for x = -0 the core computes (-0) + (+0) = +0,
        // dropping the sign; sin(+/-0) must be +/-0 exactly. x == 0 matches
        // both zeros and no other double, so blend x itself back in.
        const __mmask8 zmask = _mm512_cmp_pd_mask(x, _mm512_setzero_pd(), _CMP_EQ_OQ);
        result = _mm512_mask_blend_pd(zmask, result, x);
        _mm512_storeu_pd(&output[i], result);

        const __mmask8 mask = _mm512_cmp_pd_mask(ax, domain_bound, _CMP_GT_OQ);
        if (mask) {
            alignas(64) double xbuf[8];
            _mm512_store_pd(xbuf, x);
            for (int lane = 0; lane < 8; ++lane)
                if (mask & (1 << lane))
                    output[i + static_cast<std::size_t>(lane)] = std::sin(xbuf[lane]);
        }
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::sin(input[i]);
    }
}

}  // namespace ops
}  // namespace simd
}  // namespace stats
