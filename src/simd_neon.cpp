// ARM NEON-specific SIMD implementations
// This file is compiled ONLY with NEON flags to ensure safety

#include "libstats/common/simd_implementation_common.h"

// Only include NEON intrinsics on ARM platforms
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)
    #include <arm_neon.h>
#endif

#include <cmath>

namespace stats {
namespace simd {
namespace ops {

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__) || defined(_M_ARM)

// All NEON functions use double-precision (64-bit) values
// NEON processes 2 doubles per 128-bit register

double VectorOps::dot_product_neon(const double* a, const double* b, std::size_t size) noexcept {
    // Runtime safety check - bail out if NEON not supported
    if (!stats::arch::supports_neon()) {
        return dot_product_fallback(a, b, size);
    }

    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    // Apple Silicon optimization: Use multiple accumulators to exploit
    // superscalar execution and out-of-order capabilities
    #if defined(LIBSTATS_APPLE_SILICON)
    if (size >= stats::arch::simd::OPT_APPLE_SILICON_AGGRESSIVE_THRESHOLD * 2) {
        float64x2_t sum1 = vdupq_n_f64(detail::ZERO_DOUBLE);
        float64x2_t sum2 = vdupq_n_f64(detail::ZERO_DOUBLE);

        const std::size_t unroll_end =
            (size / (stats::arch::simd::NEON_UNROLL * 2)) * (stats::arch::simd::NEON_UNROLL * 2);

        // Process 4 doubles per iteration (2 NEON registers)
        for (std::size_t i = 0; i < unroll_end; i += stats::arch::simd::NEON_UNROLL * 2) {
            // Load data
            float64x2_t va1 = vld1q_f64(&a[i]);
            float64x2_t vb1 = vld1q_f64(&b[i]);
            float64x2_t va2 = vld1q_f64(&a[i + 2]);
            float64x2_t vb2 = vld1q_f64(&b[i + 2]);

            // Multiply and accumulate with independent accumulators
            sum1 = vfmaq_f64(sum1, va1, vb1);
            sum2 = vfmaq_f64(sum2, va2, vb2);
        }

        // Combine accumulators
        float64x2_t sum = vaddq_f64(sum1, sum2);

        // Handle remaining SIMD-width elements
        for (std::size_t i = unroll_end; i < simd_end; i += NEON_DOUBLE_WIDTH) {
            float64x2_t va = vld1q_f64(&a[i]);
            float64x2_t vb = vld1q_f64(&b[i]);
            sum = vfmaq_f64(sum, va, vb);
        }

        // Extract horizontal sum
        double final_sum = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);

        // Handle remaining scalar elements
        for (std::size_t i = simd_end; i < size; ++i) {
            final_sum += a[i] * b[i];
        }

        return final_sum;
    }
    #endif

    // Standard NEON implementation for smaller sizes or non-Apple Silicon
    float64x2_t sum = vdupq_n_f64(detail::ZERO_DOUBLE);

    // Process pairs of doubles
    for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);

        // Multiply and accumulate: sum += va * vb
        sum = vfmaq_f64(sum, va, vb);
    }

    // Extract horizontal sum
    double final_sum = vgetq_lane_f64(sum, 0) + vgetq_lane_f64(sum, 1);

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }

    return final_sum;
}

void VectorOps::vector_add_neon(const double* a, const double* b, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_add_fallback(a, b, result, size);
    }

    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    // Apple Silicon optimization: Loop unrolling for better throughput
    #if defined(LIBSTATS_APPLE_SILICON)
    if (size >= stats::arch::simd::OPT_APPLE_SILICON_AGGRESSIVE_THRESHOLD * 2) {
        const std::size_t unroll_end =
            (size / (stats::arch::simd::NEON_UNROLL * 2)) * (stats::arch::simd::NEON_UNROLL * 2);

        // Process 4 doubles per iteration
        for (std::size_t i = 0; i < unroll_end; i += stats::arch::simd::NEON_UNROLL * 2) {
            // Load and process 2 NEON registers worth of data
            float64x2_t va1 = vld1q_f64(&a[i]);
            float64x2_t vb1 = vld1q_f64(&b[i]);
            float64x2_t va2 = vld1q_f64(&a[i + 2]);
            float64x2_t vb2 = vld1q_f64(&b[i + 2]);

            // Compute results
            float64x2_t vresult1 = vaddq_f64(va1, vb1);
            float64x2_t vresult2 = vaddq_f64(va2, vb2);

            // Store results
            vst1q_f64(&result[i], vresult1);
            vst1q_f64(&result[i + detail::TWO_INT], vresult2);
        }

        // Handle remaining SIMD-width elements
        for (std::size_t i = unroll_end; i < simd_end; i += NEON_DOUBLE_WIDTH) {
            float64x2_t va = vld1q_f64(&a[i]);
            float64x2_t vb = vld1q_f64(&b[i]);
            float64x2_t vresult = vaddq_f64(va, vb);
            vst1q_f64(&result[i], vresult);
        }
    } else
    #endif
    {
        // Standard NEON implementation
        for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
            float64x2_t va = vld1q_f64(&a[i]);
            float64x2_t vb = vld1q_f64(&b[i]);
            float64x2_t vresult = vaddq_f64(va, vb);
            vst1q_f64(&result[i], vresult);
        }
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_neon(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_subtract_fallback(a, b, result, size);
    }

    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vsubq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_neon(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_multiply_fallback(a, b, result, size);
    }

    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vresult = vmulq_f64(va, vb);
        vst1q_f64(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_neon(const double* a, double scalar, double* result,
                                     std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }

    float64x2_t vscalar = vdupq_n_f64(scalar);
    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vresult = vmulq_f64(va, vscalar);
        vst1q_f64(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_neon(const double* a, double scalar, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return scalar_add_fallback(a, scalar, result, size);
    }

    float64x2_t vscalar = vdupq_n_f64(scalar);
    constexpr std::size_t NEON_DOUBLE_WIDTH = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / NEON_DOUBLE_WIDTH) * NEON_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += NEON_DOUBLE_WIDTH) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vresult = vaddq_f64(va, vscalar);
        vst1q_f64(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

// Native SIMD transcendental implementations for NEON (aarch64).
// vector_exp_neon, vector_log_neon, vector_erf_neon: float64x2_t + vfmaq_f64 (v1.5.0 Phase 3).
// vector_cos_neon: native SIMD since v1.4.0 (unchanged here).
void VectorOps::vector_exp_neon(const double* a, double* result, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_exp_fallback(a, result, size);
    }

    // SLEEF-inspired FMA Horner exp(x) on float64x2_t, < 1 ULP error.
    // Ported from vector_exp_avx2 (simd_avx2.cpp). Range reduction: x = n·ln2 + r,
    // n = round(x/ln2). Reconstructs exp(x) = exp(r)·2^n via IEEE 754 exponent bit
    // manipulation. aarch64 vcvtq_s64_f64 converts directly to int64 without the
    // 32-bit round-trip required in the AVX/AVX2 implementation.

    const float64x2_t ln2_inv = vdupq_n_f64(1.4426950408889634073599246810019);
    const float64x2_t ln2_hi = vdupq_n_f64(0.693147180369123816490e+00);
    const float64x2_t ln2_lo = vdupq_n_f64(1.90821492927058770002e-10);
    const float64x2_t exp_max = vdupq_n_f64(709.782712893383996732223);
    const float64x2_t exp_min = vdupq_n_f64(-708.0);
    const float64x2_t half = vdupq_n_f64(0.5);
    const float64x2_t one = vdupq_n_f64(1.0);

    // SLEEF polynomial coefficients for exp(r), |r| < ln2/2, < 1 ULP
    const float64x2_t c1 = vdupq_n_f64(0.1666666666666669072e+0);
    const float64x2_t c2 = vdupq_n_f64(0.4166666666666602598e-1);
    const float64x2_t c3 = vdupq_n_f64(0.8333333333314938210e-2);
    const float64x2_t c4 = vdupq_n_f64(0.1388888888914497797e-2);
    const float64x2_t c5 = vdupq_n_f64(0.1984126989855865850e-3);
    const float64x2_t c6 = vdupq_n_f64(0.2480158687479686264e-4);
    const float64x2_t c7 = vdupq_n_f64(0.2755723402025388239e-5);
    const float64x2_t c8 = vdupq_n_f64(0.2755762628169491192e-6);
    const float64x2_t c9 = vdupq_n_f64(0.2511210703042288022e-7);
    const float64x2_t c10 = vdupq_n_f64(0.2081276378237164457e-8);

    constexpr std::size_t W = stats::arch::simd::NEON_DOUBLES;  // 2
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        float64x2_t x = vld1q_f64(&a[i]);
        x = vminq_f64(x, exp_max);
        x = vmaxq_f64(x, exp_min);

        // n = round(x/ln2); r = x - n·ln2 via two-part ln2 for precision
        float64x2_t n_float = vrndnq_f64(vmulq_f64(x, ln2_inv));
        float64x2_t r = vfmsq_f64(x, n_float, ln2_hi);  // x - n·ln2_hi
        r = vfmsq_f64(r, n_float, ln2_lo);              // r - n·ln2_lo

        // FMA Horner: P(r) — each step: poly = c_k + poly·r
        float64x2_t r2 = vmulq_f64(r, r);
        float64x2_t poly = c10;
        poly = vfmaq_f64(c9, poly, r);
        poly = vfmaq_f64(c8, poly, r);
        poly = vfmaq_f64(c7, poly, r);
        poly = vfmaq_f64(c6, poly, r);
        poly = vfmaq_f64(c5, poly, r);
        poly = vfmaq_f64(c4, poly, r);
        poly = vfmaq_f64(c3, poly, r);
        poly = vfmaq_f64(c2, poly, r);
        poly = vfmaq_f64(c1, poly, r);

        // Complete: exp(r) = 1 + r + r²·(0.5 + r·P(r))
        poly = vfmaq_f64(half, poly, r);  // 0.5 + r·P(r)
        poly = vfmaq_f64(r, poly, r2);    // r + r²·(0.5 + r·P(r))
        poly = vaddq_f64(poly, one);      // 1 + r + r²·(0.5 + r·P(r))

        // Scale by 2^n: biased exponent = n + 1023, placed at IEEE 754 bit 52.
        // aarch64 vcvtq_s64_f64 converts directly; no 32-bit round-trip needed.
        int64x2_t n_int = vcvtq_s64_f64(n_float);
        int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(n_int, vdupq_n_s64(1023)), 52);
        float64x2_t scale = vreinterpretq_f64_s64(exp_bits);
        vst1q_f64(&result[i], vmulq_f64(poly, scale));
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = std::exp(a[i]);
    }
}

void VectorOps::vector_log_neon(const double* a, double* result, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_log_fallback(a, result, size);
    }

    // SLEEF-inspired FMA Horner log(x) on float64x2_t, < 1 ULP error.
    // Ported from vector_log_avx2 (simd_avx2.cpp). Uses (m-1)/(m+1) reduction
    // so log(m) = 2*atanh(xr) via 7-term polynomial. aarch64 vcvtq_f64_s64
    // converts the int64 exponent to double directly; no store/reload needed.

    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t ln2_hi = vdupq_n_f64(0.693147180559945286226764);
    const float64x2_t ln2_lo = vdupq_n_f64(2.319046813846299558417771e-17);
    const float64x2_t sqrt2 = vdupq_n_f64(1.4142135623730950488016887242097);
    const float64x2_t half = vdupq_n_f64(0.5);
    const float64x2_t two = vdupq_n_f64(2.0);

    // SLEEF xlog_u1 coefficients (2·atanh series), < 1 ULP
    const float64x2_t c1 = vdupq_n_f64(0.6666666666667333541e+0);
    const float64x2_t c2 = vdupq_n_f64(0.3999999999635251990e+0);
    const float64x2_t c3 = vdupq_n_f64(0.2857142932794299317e+0);
    const float64x2_t c4 = vdupq_n_f64(0.2222214519839380009e+0);
    const float64x2_t c5 = vdupq_n_f64(0.1818605932937785996e+0);
    const float64x2_t c6 = vdupq_n_f64(0.1525629051003428716e+0);
    const float64x2_t c7 = vdupq_n_f64(0.1532076988502701353e+0);

    const float64x2_t zero = vdupq_n_f64(0.0);
    const float64x2_t neg_inf = vdupq_n_f64(-std::numeric_limits<double>::infinity());
    const float64x2_t pos_inf = vdupq_n_f64(std::numeric_limits<double>::infinity());
    const float64x2_t nan_val = vdupq_n_f64(std::numeric_limits<double>::quiet_NaN());

    constexpr std::size_t W = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        float64x2_t x = vld1q_f64(&a[i]);

        // Special-case detection
        uint64x2_t is_zero = vceqq_f64(x, zero);
        uint64x2_t is_negative = vcltq_f64(x, zero);
        uint64x2_t is_inf = vceqq_f64(x, pos_inf);

        // Scale denormals by 2^54 to bring into normal range
        const float64x2_t min_normal = vdupq_n_f64(2.2250738585072014e-308);
        const float64x2_t scale_up = vdupq_n_f64(18014398509481984.0);  // 2^54
        uint64x2_t is_denormal = vcltq_f64(x, min_normal);
        float64x2_t scaled_x = vbslq_f64(is_denormal, vmulq_f64(x, scale_up), x);

        // Exponent extraction: logical right-shift by 52, mask 11-bit field, subtract bias
        uint64x2_t xi = vreinterpretq_u64_f64(scaled_x);
        int64x2_t e_int =
            vsubq_s64(vreinterpretq_s64_u64(vandq_u64(vshrq_n_u64(xi, 52), vdupq_n_u64(0x7FFULL))),
                      vdupq_n_s64(1023));
        float64x2_t e = vcvtq_f64_s64(e_int);  // direct i64→f64, no store/reload
        e = vbslq_f64(is_denormal, vsubq_f64(e, vdupq_n_f64(54.0)), e);

        // Isolate mantissa in [1, 2) by clearing exponent field and forcing e=1023
        uint64x2_t m_bits = vorrq_u64(vandq_u64(xi, vdupq_n_u64(0x000FFFFFFFFFFFFFULL)),
                                      vdupq_n_u64(0x3FF0000000000000ULL));
        float64x2_t m = vreinterpretq_f64_u64(m_bits);

        // Range adjustment: if m > sqrt(2), halve m and increment e
        uint64x2_t needs_adj = vcgtq_f64(m, sqrt2);
        m = vbslq_f64(needs_adj, vmulq_f64(m, half), m);
        e = vbslq_f64(needs_adj, vaddq_f64(e, one), e);

        // xr = (m-1)/(m+1); FMA Horner: t = c7 + xr²·(c6 + xr²·(… + xr²·c1))
        float64x2_t xr = vdivq_f64(vsubq_f64(m, one), vaddq_f64(m, one));
        float64x2_t xr2 = vmulq_f64(xr, xr);
        float64x2_t t = c7;
        t = vfmaq_f64(c6, t, xr2);
        t = vfmaq_f64(c5, t, xr2);
        t = vfmaq_f64(c4, t, xr2);
        t = vfmaq_f64(c3, t, xr2);
        t = vfmaq_f64(c2, t, xr2);
        t = vfmaq_f64(c1, t, xr2);

        // log(m) = 2·xr + xr³·t
        float64x2_t xr3 = vmulq_f64(xr, xr2);
        float64x2_t two_xr = vmulq_f64(xr, two);
        float64x2_t log_m = vfmaq_f64(two_xr, xr3, t);

        // log(x) = log(m) + e·ln2 (high-low FMA decomposition)
        float64x2_t res = vfmaq_f64(log_m, e, ln2_hi);
        res = vfmaq_f64(res, e, ln2_lo);

        // Apply special cases
        res = vbslq_f64(is_zero, neg_inf, res);
        res = vbslq_f64(is_inf, pos_inf, res);
        res = vbslq_f64(is_negative, nan_val, res);

        vst1q_f64(&result[i], res);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = std::log(a[i]);
    }
}

void VectorOps::vector_pow_neon(const double* base, double exponent, double* result,
                                std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_pow_fallback(base, exponent, result, size);
    }
    // NEON doesn't have native power instructions, use scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = std::pow(base[i], exponent);
    }
}

void VectorOps::vector_pow_elementwise_neon(const double* base, const double* exponent,
                                            double* result, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        // Fallback to scalar implementation
        for (std::size_t i = 0; i < size; ++i) {
            result[i] = std::pow(base[i], exponent[i]);
        }
        return;
    }
    // NEON doesn't have native power instructions, use scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = std::pow(base[i], exponent[i]);
    }
}

void VectorOps::vector_erf_neon(const double* a, double* result, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_erf_fallback(a, result, size);
    }

    // Musl libc four-region rational polynomial erf on float64x2_t, < 1 ULP error.
    // Ported from vector_erf_avx2 (simd_avx2.cpp). FMA Horner via vfmaq_f64;
    // region blending via vbslq_f64; andnot via vbicq_u64; sign via vorrq_u64.
    // Regions (evaluated for every lane; blended by mask at the end):
    //   R1: |x| < 0.84375  — rational P(z)/Q(z), z = x²
    //   R2: 0.84375 ≤ |x| < 1.25  — rational P(s)/Q(s), s = |x|-1
    //   R3: 1.25 ≤ |x| < 2.857  — erfc via exp(-x²-0.5625+R/S)/|x|
    //   R4: 2.857 ≤ |x| < 6     — same structure, different coefficients
    //   R5: |x| ≥ 6             — erf ≈ ±1
    //
    // Regions 3-4 call vector_exp_neon on a 2-element buffer; the recursive call
    // is safe because NEON support has already been verified above.

    // ---- Region 1 coefficients (rational P/Q in z = x²) ----
    const float64x2_t pp0 = vdupq_n_f64(1.28379167095512558561e-01);
    const float64x2_t pp1 = vdupq_n_f64(-3.25042107247001499370e-01);
    const float64x2_t pp2 = vdupq_n_f64(-2.84817495755985104766e-02);
    const float64x2_t pp3 = vdupq_n_f64(-5.77027029648944159157e-03);
    const float64x2_t pp4 = vdupq_n_f64(-2.37630166566501626084e-05);
    const float64x2_t qq1 = vdupq_n_f64(3.97917223959155352819e-01);
    const float64x2_t qq2 = vdupq_n_f64(6.50222499887672944485e-02);
    const float64x2_t qq3 = vdupq_n_f64(5.08130628187576562776e-03);
    const float64x2_t qq4 = vdupq_n_f64(1.32494738004321644526e-04);
    const float64x2_t qq5 = vdupq_n_f64(-3.96022827877536812320e-06);
    // ---- Region 2 coefficients (rational P/Q in s = |x|-1) ----
    const float64x2_t erx = vdupq_n_f64(8.45062911510467529297e-01);
    const float64x2_t pa0 = vdupq_n_f64(-2.36211856075265944077e-03);
    const float64x2_t pa1 = vdupq_n_f64(4.14856118683748331666e-01);
    const float64x2_t pa2 = vdupq_n_f64(-3.72207876035701323847e-01);
    const float64x2_t pa3 = vdupq_n_f64(3.18346619901161753674e-01);
    const float64x2_t pa4 = vdupq_n_f64(-1.10894694282396677476e-01);
    const float64x2_t pa5 = vdupq_n_f64(3.54783043256182359371e-02);
    const float64x2_t pa6 = vdupq_n_f64(-2.16637559486879084300e-03);
    const float64x2_t qa1 = vdupq_n_f64(1.06420880400844228286e-01);
    const float64x2_t qa2 = vdupq_n_f64(5.40397917702171048937e-01);
    const float64x2_t qa3 = vdupq_n_f64(7.18286544141962662868e-02);
    const float64x2_t qa4 = vdupq_n_f64(1.26171219808761642112e-01);
    const float64x2_t qa5 = vdupq_n_f64(1.36370839120290507362e-02);
    const float64x2_t qa6 = vdupq_n_f64(1.19844998467991074170e-02);
    // ---- Region 3 coefficients (R/S in s = 1/x², 1.25 ≤ |x| < 2.857) ----
    const float64x2_t ra0 = vdupq_n_f64(-9.86494403484714822705e-03);
    const float64x2_t ra1 = vdupq_n_f64(-6.93858572707181764372e-01);
    const float64x2_t ra2 = vdupq_n_f64(-1.05586262253232909814e+01);
    const float64x2_t ra3 = vdupq_n_f64(-6.23753324503260060396e+01);
    const float64x2_t ra4 = vdupq_n_f64(-1.62396669462573470355e+02);
    const float64x2_t ra5 = vdupq_n_f64(-1.84605092906711035994e+02);
    const float64x2_t ra6 = vdupq_n_f64(-8.12874355063065934246e+01);
    const float64x2_t ra7 = vdupq_n_f64(-9.81432934416914548592e+00);
    const float64x2_t sa1 = vdupq_n_f64(1.96512716674392571292e+01);
    const float64x2_t sa2 = vdupq_n_f64(1.37657754143519042600e+02);
    const float64x2_t sa3 = vdupq_n_f64(4.34565877475229228821e+02);
    const float64x2_t sa4 = vdupq_n_f64(6.45387271733267880336e+02);
    const float64x2_t sa5 = vdupq_n_f64(4.29008140027567833386e+02);
    const float64x2_t sa6 = vdupq_n_f64(1.08635005541779435134e+02);
    const float64x2_t sa7 = vdupq_n_f64(6.57024977031928170135e+00);
    const float64x2_t sa8 = vdupq_n_f64(-6.04244152148580987438e-02);
    // ---- Region 4 coefficients (R/S in s = 1/x², 2.857 ≤ |x| < 6) ----
    const float64x2_t rb0 = vdupq_n_f64(-9.86494292470009928597e-03);
    const float64x2_t rb1 = vdupq_n_f64(-7.99283237680523006574e-01);
    const float64x2_t rb2 = vdupq_n_f64(-1.77579549177547519889e+01);
    const float64x2_t rb3 = vdupq_n_f64(-1.60636384855821916062e+02);
    const float64x2_t rb4 = vdupq_n_f64(-6.37566443368389627722e+02);
    const float64x2_t rb5 = vdupq_n_f64(-1.02509513161107724954e+03);
    const float64x2_t rb6 = vdupq_n_f64(-4.83519191608651397019e+02);
    const float64x2_t sb1 = vdupq_n_f64(3.03380607434824582924e+01);
    const float64x2_t sb2 = vdupq_n_f64(3.25792512996573918826e+02);
    const float64x2_t sb3 = vdupq_n_f64(1.53672958608443695994e+03);
    const float64x2_t sb4 = vdupq_n_f64(3.19985821950859553908e+03);
    const float64x2_t sb5 = vdupq_n_f64(2.55305040643316442583e+03);
    const float64x2_t sb6 = vdupq_n_f64(4.74528541206955367215e+02);
    const float64x2_t sb7 = vdupq_n_f64(-2.24409524465858183362e+01);

    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t t1 = vdupq_n_f64(0.84375);
    const float64x2_t t2 = vdupq_n_f64(1.25);
    const float64x2_t t3 = vdupq_n_f64(2.857142857);
    const float64x2_t t5 = vdupq_n_f64(6.0);
    const float64x2_t c0p5625 = vdupq_n_f64(0.5625);

    // Sign bit mask for sign extraction and restoration (erf is odd)
    const uint64x2_t sign_mask = vdupq_n_u64(0x8000000000000000ULL);
    // All-ones for XOR-based NaN detection: NaN != NaN, so vceqq_f64(x,x)=0 for NaN
    const uint64x2_t all_ones = vdupq_n_u64(0xFFFFFFFFFFFFFFFFULL);

    constexpr std::size_t W = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / W) * W;
    alignas(16) double exp_buf[W];

    for (std::size_t i = 0; i < simd_end; i += W) {
        float64x2_t x = vld1q_f64(&a[i]);
        uint64x2_t x_u = vreinterpretq_u64_f64(x);
        uint64x2_t sign = vandq_u64(x_u, sign_mask);                        // extract sign
        float64x2_t ax = vreinterpretq_f64_u64(vbicq_u64(x_u, sign_mask));  // |x|

        // Region masks
        uint64x2_t m1 = vcltq_f64(ax, t1);  // |x| < 0.84375
        uint64x2_t m2 = vcltq_f64(ax, t2);  // |x| < 1.25
        uint64x2_t m3 = vcltq_f64(ax, t3);  // |x| < 2.857

        // ---- Region 1: erf(x) ≈ x + x·P(z)/Q(z), z = x² ----
        float64x2_t z = vmulq_f64(ax, ax);
        float64x2_t P1 = pp4;
        P1 = vfmaq_f64(pp3, z, P1);
        P1 = vfmaq_f64(pp2, z, P1);
        P1 = vfmaq_f64(pp1, z, P1);
        P1 = vfmaq_f64(pp0, z, P1);
        float64x2_t Q1 = qq5;
        Q1 = vfmaq_f64(qq4, z, Q1);
        Q1 = vfmaq_f64(qq3, z, Q1);
        Q1 = vfmaq_f64(qq2, z, Q1);
        Q1 = vfmaq_f64(qq1, z, Q1);
        Q1 = vfmaq_f64(one, z, Q1);
        float64x2_t r1 = vfmaq_f64(ax, ax, vdivq_f64(P1, Q1));  // ax + ax·(P1/Q1)

        // ---- Region 2: erf(x) = erx + P(s)/Q(s), s = |x|-1 ----
        float64x2_t s2 = vsubq_f64(ax, one);
        float64x2_t P2 = pa6;
        P2 = vfmaq_f64(pa5, s2, P2);
        P2 = vfmaq_f64(pa4, s2, P2);
        P2 = vfmaq_f64(pa3, s2, P2);
        P2 = vfmaq_f64(pa2, s2, P2);
        P2 = vfmaq_f64(pa1, s2, P2);
        P2 = vfmaq_f64(pa0, s2, P2);
        float64x2_t Q2 = qa6;
        Q2 = vfmaq_f64(qa5, s2, Q2);
        Q2 = vfmaq_f64(qa4, s2, Q2);
        Q2 = vfmaq_f64(qa3, s2, Q2);
        Q2 = vfmaq_f64(qa2, s2, Q2);
        Q2 = vfmaq_f64(qa1, s2, Q2);
        Q2 = vfmaq_f64(one, s2, Q2);
        float64x2_t r2 = vaddq_f64(erx, vdivq_f64(P2, Q2));

        // ---- Regions 3-4: erfc = exp(-x²-0.5625+R/S)/|x|, erf = 1-erfc ----
        float64x2_t sax = vmaxq_f64(ax, t2);                       // clamp ≥ 1.25
        float64x2_t inv_x2 = vdivq_f64(one, vmulq_f64(sax, sax));  // 1/x²

        float64x2_t R3 = ra7;
        R3 = vfmaq_f64(ra6, inv_x2, R3);
        R3 = vfmaq_f64(ra5, inv_x2, R3);
        R3 = vfmaq_f64(ra4, inv_x2, R3);
        R3 = vfmaq_f64(ra3, inv_x2, R3);
        R3 = vfmaq_f64(ra2, inv_x2, R3);
        R3 = vfmaq_f64(ra1, inv_x2, R3);
        R3 = vfmaq_f64(ra0, inv_x2, R3);
        float64x2_t S3 = sa8;
        S3 = vfmaq_f64(sa7, inv_x2, S3);
        S3 = vfmaq_f64(sa6, inv_x2, S3);
        S3 = vfmaq_f64(sa5, inv_x2, S3);
        S3 = vfmaq_f64(sa4, inv_x2, S3);
        S3 = vfmaq_f64(sa3, inv_x2, S3);
        S3 = vfmaq_f64(sa2, inv_x2, S3);
        S3 = vfmaq_f64(sa1, inv_x2, S3);
        S3 = vfmaq_f64(one, inv_x2, S3);

        float64x2_t R4 = rb6;
        R4 = vfmaq_f64(rb5, inv_x2, R4);
        R4 = vfmaq_f64(rb4, inv_x2, R4);
        R4 = vfmaq_f64(rb3, inv_x2, R4);
        R4 = vfmaq_f64(rb2, inv_x2, R4);
        R4 = vfmaq_f64(rb1, inv_x2, R4);
        R4 = vfmaq_f64(rb0, inv_x2, R4);
        float64x2_t S4 = sb7;
        S4 = vfmaq_f64(sb6, inv_x2, S4);
        S4 = vfmaq_f64(sb5, inv_x2, S4);
        S4 = vfmaq_f64(sb4, inv_x2, S4);
        S4 = vfmaq_f64(sb3, inv_x2, S4);
        S4 = vfmaq_f64(sb2, inv_x2, S4);
        S4 = vfmaq_f64(sb1, inv_x2, S4);
        S4 = vfmaq_f64(one, inv_x2, S4);

        // Blend R/S: R3/S3 where |x| < 2.857, R4/S4 otherwise
        float64x2_t RS = vdivq_f64(vbslq_f64(m3, R3, R4), vbslq_f64(m3, S3, S4));

        // exp_arg = -x² - 0.5625 + R/S, clamped ≤ 0
        float64x2_t exp_arg = vsubq_f64(vsubq_f64(RS, c0p5625), vmulq_f64(sax, sax));
        exp_arg = vminq_f64(exp_arg, vdupq_n_f64(0.0));
        vst1q_f64(exp_buf, exp_arg);
        vector_exp_neon(exp_buf, exp_buf, W);
        float64x2_t r34 = vsubq_f64(one, vdivq_f64(vld1q_f64(exp_buf), sax));

        // ---- Blend regions (innermost wins) ----
        float64x2_t res = one;                                   // R5: |x| ≥ 6
        uint64x2_t r34_mask = vbicq_u64(vcltq_f64(ax, t5), m2);  // ~m2 & (ax<6)
        res = vbslq_f64(r34_mask, r34, res);
        res = vbslq_f64(vbicq_u64(m2, m1), r2, res);  // R2: m2 & ~m1
        res = vbslq_f64(m1, r1, res);                 // R1

        // Propagate NaN (vceqq_f64(x,x)=all-0 for NaN; XOR flips to all-1)
        uint64x2_t nan_mask = veorq_u64(vceqq_f64(x, x), all_ones);
        res = vbslq_f64(nan_mask, x, res);

        // Restore sign (erf is odd: sign of erf(x) == sign of x)
        vst1q_f64(&result[i], vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(res), sign)));
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = std::erf(a[i]);
    }
}

void VectorOps::vector_cos_neon(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_cos_fallback(input, output, size);
    }

    constexpr std::size_t W = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    const float64x2_t inv_two_pi = vdupq_n_f64(1.0 / (2.0 * detail::PI));
    const float64x2_t two_pi = vdupq_n_f64(2.0 * detail::PI);
    const float64x2_t pi = vdupq_n_f64(detail::PI);
    const float64x2_t half_pi = vdupq_n_f64(detail::PI_OVER_2);
    const float64x2_t neg_pi = vdupq_n_f64(-detail::PI);
    const float64x2_t neg_half_pi = vdupq_n_f64(-detail::PI_OVER_2);
    const float64x2_t one = vdupq_n_f64(1.0);
    const float64x2_t neg_one = vdupq_n_f64(-1.0);

    const float64x2_t c1 = vdupq_n_f64(-0.5);
    const float64x2_t c2 = vdupq_n_f64(4.166666666666667e-2);
    const float64x2_t c3 = vdupq_n_f64(-1.388888888888889e-3);
    const float64x2_t c4 = vdupq_n_f64(2.480158730158730e-5);
    const float64x2_t c5 = vdupq_n_f64(-2.755731922398589e-7);
    const float64x2_t c6 = vdupq_n_f64(2.087675698786810e-9);
    const float64x2_t c7 = vdupq_n_f64(-1.147074559772973e-11);

    for (std::size_t i = 0; i < simd_end; i += W) {
        float64x2_t x = vld1q_f64(&input[i]);

        float64x2_t q = vrndnq_f64(vmulq_f64(x, inv_two_pi));
        float64x2_t y = vsubq_f64(x, vmulq_f64(q, two_pi));

        float64x2_t sign = one;
        uint64x2_t gt_hpi = vcgtq_f64(y, half_pi);
        uint64x2_t lt_nhpi = vcltq_f64(y, neg_half_pi);

        y = vbslq_f64(gt_hpi, vsubq_f64(pi, y), y);
        sign = vbslq_f64(gt_hpi, neg_one, sign);
        y = vbslq_f64(lt_nhpi, vsubq_f64(neg_pi, y), y);
        sign = vbslq_f64(lt_nhpi, neg_one, sign);

        float64x2_t y2 = vmulq_f64(y, y);
        float64x2_t poly = c7;
        poly = vfmaq_f64(c6, y2, poly);
        poly = vfmaq_f64(c5, y2, poly);
        poly = vfmaq_f64(c4, y2, poly);
        poly = vfmaq_f64(c3, y2, poly);
        poly = vfmaq_f64(c2, y2, poly);
        poly = vfmaq_f64(c1, y2, poly);
        poly = vfmaq_f64(one, y2, poly);

        vst1q_f64(&output[i], vmulq_f64(poly, sign));
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::cos(input[i]);
    }
}

#else

// Fallback implementations for non-ARM platforms
// These will never be called, but we need them for linking

double VectorOps::dot_product_neon(const double* a, const double* b, std::size_t size) noexcept {
    return dot_product_fallback(a, b, size);
}

void VectorOps::vector_add_neon(const double* a, const double* b, double* result,
                                std::size_t size) noexcept {
    vector_add_fallback(a, b, result, size);
}

void VectorOps::vector_subtract_neon(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    vector_subtract_fallback(a, b, result, size);
}

void VectorOps::vector_multiply_neon(const double* a, const double* b, double* result,
                                     std::size_t size) noexcept {
    vector_multiply_fallback(a, b, result, size);
}

void VectorOps::scalar_multiply_neon(const double* a, double scalar, double* result,
                                     std::size_t size) noexcept {
    scalar_multiply_fallback(a, scalar, result, size);
}

void VectorOps::scalar_add_neon(const double* a, double scalar, double* result,
                                std::size_t size) noexcept {
    scalar_add_fallback(a, scalar, result, size);
}

void VectorOps::vector_exp_neon(const double* a, double* result, std::size_t size) noexcept {
    vector_exp_fallback(a, result, size);
}

void VectorOps::vector_log_neon(const double* a, double* result, std::size_t size) noexcept {
    vector_log_fallback(a, result, size);
}

void VectorOps::vector_pow_neon(const double* base, double exponent, double* result,
                                std::size_t size) noexcept {
    vector_pow_fallback(base, exponent, result, size);
}

void VectorOps::vector_pow_elementwise_neon(const double* base, const double* exponent,
                                            double* result, std::size_t size) noexcept {
    // Fallback to scalar implementation for non-ARM platforms
    for (std::size_t i = 0; i < size; ++i) {
        result[i] = std::pow(base[i], exponent[i]);
    }
}

void VectorOps::vector_erf_neon(const double* a, double* result, std::size_t size) noexcept {
    vector_erf_fallback(a, result, size);
}

void VectorOps::vector_cos_neon(const double* input, double* output, std::size_t size) noexcept {
    vector_cos_fallback(input, output, size);
}

#endif  // ARM platform check

}  // namespace ops
}  // namespace simd
}  // namespace stats
