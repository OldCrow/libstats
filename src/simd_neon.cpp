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
    // NEON doesn't have native error function instructions, use scalar implementation
    for (std::size_t i = 0; i < size; ++i) {
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
