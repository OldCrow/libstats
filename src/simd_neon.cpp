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
    // vector_exp_neon: float64x2_t + vfmaq_f64; N=128 tail-corrected table + order-5
    // polynomial, derived from ARM optimized-routines' MIT-licensed scalar exp
    // (math/exp.c + exp_data.c) -- see Issue #33 Q1 (docs/SIMD_BENCHMARK_RESULTS.md).
    // Replaces the v1.5.0 Phase 3 SLEEF polynomial (kept as the scalar fallback).
    // vector_log_neon, vector_erf_neon: float64x2_t + vfmaq_f64 (v1.5.0 Phase 3);
    // erf independently re-derived 2026-07-19 (Issue #67) -- see
    // docs/NEON_ERF_DERIVATION.md and docs/NEON_ERF_DIVERGENCE_AUDIT.md.
    // vector_cos_neon: native SIMD since v1.4.0 (unchanged here).

    // N=128 tail-corrected table, Array-of-Structs {tail_bits, sbits}, re-interleaved
    // from ARM optimized-routines' exp_data.c so one vld1q_u64 pulls both per lane
    // (the NEON software-gather pattern). See scripts/gen_neon_exp_table.py and
    // THIRD_PARTY_NOTICES.md.
    #include "neon_exp_data.inc"  // kExpNeonTable[128]

namespace {
// ARM AOR exp constants (math/exp_data.c, N=128 block); the algorithm is
// architecture-independent (shared with the AVX-512 Stage 3 experiment kernel).
constexpr double kExpNeonInvLn2N = 0x1.71547652b82fep0 * 128.0;  // N/ln2
constexpr double kExpNeonNegLn2hiN = -0x1.62e42fefa0000p-8;      // -ln2/N (hi)
constexpr double kExpNeonNegLn2loN = -0x1.cf79abc9e3b3ap-47;     // -ln2/N (lo)
constexpr double kExpNeonShift = 0x1.8p52;
constexpr double kExpNeonC2 = 0x1.ffffffffffdbdp-2;
constexpr double kExpNeonC3 = 0x1.555555555543cp-3;
constexpr double kExpNeonC4 = 0x1.55555cf172b91p-5;
constexpr double kExpNeonC5 = 0x1.1111167a4d017p-7;
constexpr double kExpNeonSpecialBound = 704.0;  // |x| beyond this: exact scalar fixup
}  // namespace

void VectorOps::vector_exp_neon(const double* a, double* result, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_exp_fallback(a, result, size);
    }

    const float64x2_t invln2N = vdupq_n_f64(kExpNeonInvLn2N);
    const float64x2_t shift = vdupq_n_f64(kExpNeonShift);
    const float64x2_t negln2hiN = vdupq_n_f64(kExpNeonNegLn2hiN);
    const float64x2_t negln2loN = vdupq_n_f64(kExpNeonNegLn2loN);
    const float64x2_t c2 = vdupq_n_f64(kExpNeonC2);
    const float64x2_t c3 = vdupq_n_f64(kExpNeonC3);
    const float64x2_t c4 = vdupq_n_f64(kExpNeonC4);
    const float64x2_t c5 = vdupq_n_f64(kExpNeonC5);
    const uint64x2_t idx_mask = vdupq_n_u64(127);
    const float64x2_t special_bound = vdupq_n_f64(kExpNeonSpecialBound);

    // Per-vector core: shift-trick range reduction + software gather + tail-
    // corrected order-5 polynomial. Inlined in Release, so the loop-invariant
    // const vectors above are hoisted and the unrolled + remainder paths share
    // one definition (no code duplication).
    const auto expCore = [&](float64x2_t x) -> float64x2_t {
        // n = round(x * N/ln2) via the shift trick; ki = bits(z), kd = n as double.
        float64x2_t z = vfmaq_f64(shift, x, invln2N);  // shift + x*invln2N (fused)
        uint64x2_t ki = vreinterpretq_u64_f64(z);
        float64x2_t kd = vsubq_f64(z, shift);

        // r = x - n*ln2/N (two-part reduction; the NegLn2*N constants are negative).
        float64x2_t r = vfmaq_f64(x, kd, negln2hiN);  // x + kd*negln2hiN
        r = vfmaq_f64(r, kd, negln2loN);              // r + kd*negln2loN

        // Table index and exponent injection derived at RUNTIME (no hardcoded bits).
        uint64x2_t idx = vandq_u64(ki, idx_mask);
        uint64x2_t top = vshlq_n_u64(ki, 45);  // 52 - EXP_TABLE_BITS(7)

        // Software gather: one 128-bit load per lane pulls the WHOLE {tail, sbits}
        // pair; vuzp deinterleaves. Costs a single load per lane because the pair
        // is 16 bytes = one vld1q (this is why the tail correction is affordable
        // on NEON where it was not on x86's hardware-gather AVX-512 experiment).
        uint64x2_t e0 = vld1q_u64(
            reinterpret_cast<const std::uint64_t*>(&kExpNeonTable[vgetq_lane_u64(idx, 0)]));
        uint64x2_t e1 = vld1q_u64(
            reinterpret_cast<const std::uint64_t*>(&kExpNeonTable[vgetq_lane_u64(idx, 1)]));
        uint64x2_t tail_bits = vuzp1q_u64(e0, e1);  // {tail0, tail1}
        uint64x2_t sbits = vuzp2q_u64(e0, e1);      // {sbits0, sbits1}

        float64x2_t tail = vreinterpretq_f64_u64(tail_bits);
        float64x2_t scale = vreinterpretq_f64_u64(vaddq_u64(sbits, top));  // 2^(n/N)

        // tmp = tail + r + r^2*(C2 + r*C3) + r^4*(C4 + r*C5) ~= (exp(r) - 1) + tail
        float64x2_t r2 = vmulq_f64(r, r);
        float64x2_t r4 = vmulq_f64(r2, r2);
        float64x2_t plo = vfmaq_f64(c2, r, c3);  // C2 + r*C3
        float64x2_t phi = vfmaq_f64(c4, r, c5);  // C4 + r*C5
        float64x2_t tmp = vaddq_f64(tail, r);
        tmp = vfmaq_f64(tmp, r2, plo);  // tmp + r2*plo
        tmp = vfmaq_f64(tmp, r4, phi);  // tmp + r4*phi

        // exp(x) = scale + scale*tmp = 2^(n/N) * (1 + (exp(r) - 1)).
        return vfmaq_f64(scale, scale, tmp);
    };

    // Scalar edge fixup for lanes with |x| >= 704 (also catches +/-inf). NaN is not
    // caught by the ordered compare, but propagates to NaN through the polynomial
    // path already, which is the correct result.
    //
    // IMPORTANT (aliasing): this takes the already-loaded register `xv` and a
    // precomputed per-lane `mask`, and never re-reads `a[]`. vector_exp is called
    // in-place in production (e.g. LogSpaceOps::logSumExpArrayFallback passes the
    // same buffer as both `a` and `result`), so by the time this runs, `a[base..]`
    // may already have been overwritten by the vst1q_f64 stores above. Deciding
    // the fixup from the original register value (not a[]) is what makes this
    // correct regardless of aliasing.
    const auto fixupSpecial = [&](std::size_t base, float64x2_t xv, uint64x2_t mask) {
        if (vgetq_lane_u64(mask, 0))
            result[base + 0] = std::exp(vgetq_lane_f64(xv, 0));
        if (vgetq_lane_u64(mask, 1))
            result[base + 1] = std::exp(vgetq_lane_f64(xv, 1));
    };

    const std::size_t unroll_end = (size / 4) * 4;  // 2 vectors (4 doubles) per iteration
    const std::size_t simd_end = (size / 2) * 2;

    std::size_t i = 0;
    for (; i < unroll_end; i += 4) {
        float64x2_t x0 = vld1q_f64(&a[i]);
        float64x2_t x1 = vld1q_f64(&a[i + 2]);
        vst1q_f64(&result[i], expCore(x0));
        vst1q_f64(&result[i + 2], expCore(x1));

        // Hoisted edge check computed from the registers (pre-store); the branch
        // is not taken for in-range batches.
        uint64x2_t mask0 = vcgeq_f64(vabsq_f64(x0), special_bound);
        uint64x2_t mask1 = vcgeq_f64(vabsq_f64(x1), special_bound);
        uint64x2_t any = vorrq_u64(mask0, mask1);
        if (vgetq_lane_u64(any, 0) | vgetq_lane_u64(any, 1)) {
            fixupSpecial(i, x0, mask0);
            fixupSpecial(i + 2, x1, mask1);
        }
    }
    for (; i < simd_end; i += 2) {
        float64x2_t x = vld1q_f64(&a[i]);
        vst1q_f64(&result[i], expCore(x));
        uint64x2_t mask = vcgeq_f64(vabsq_f64(x), special_bound);
        if (vgetq_lane_u64(mask, 0) | vgetq_lane_u64(mask, 1))
            fixupSpecial(i, x, mask);
    }
    // Scalar tail: disjoint from the SIMD range above, so a[i] is always still the
    // original input here regardless of aliasing.
    for (; i < size; ++i)
        result[i] = std::exp(a[i]);
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
    // Native 2-wide NEON path: pow(x, e) = exp(e * log(x)).
    // vector_log_neon and vector_exp_neon are both fully implemented (v1.5.0).
    vector_log_neon(base, result, size);                   // result = log(base)
    scalar_multiply_neon(result, exponent, result, size);  // result = e * log(base)
    vector_exp_neon(result, result, size);                 // result = exp(e * log(base))
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

    // Lookup table for vector_erf_neon: 769 entries of {erf(k/128), scale(k)}
    // where scale(k) = 2/sqrt(pi)*exp(-(k/128)^2), covering |x| in [0, 6-1/128].
    // Generated by scripts/gen_neon_erf_table.py — do not edit manually.
    #include "neon_erf_data.inc"
void VectorOps::vector_erf_neon(const double* a, double* result, std::size_t size) noexcept {
    if (!stats::arch::supports_neon()) {
        return vector_erf_fallback(a, result, size);
    }

    // ARM glibc erf_advsimd algorithm: table lookup + 5-term Taylor series, ~2.29 ULP.
    // Reference: sysdeps/aarch64/fpu/erf_advsimd.c (glibc, by ARM Ltd., LGPL-2.1+).
    //
    // erf(r+d) ≈ erf(r) + scale·d·poly(r,d)
    //   r     = round(|x|, 1/128) via shift trick (shift = 2^45, ULP = 1/128)
    //   d     = |x| - r  (small residual, |d| ≤ 1/256)
    //   scale = 2/sqrt(π)·exp(-r²)  [from kErfNeonTable]
    //   poly  = 1 - r·d + (2r²-1)/3·d² - r(2r²-3)/6·d³ + … (5 terms)
    //
    // Table: kErfNeonTable[769] in neon_erf_data.inc, 12,304 bytes.
    // Software gather: 2 sequential vld1q_f64 loads + vuzp1q/vuzp2q to deinterleave.
    // For |x| > 5.9921875 (k>767) or NaN, clamps to entry 768 (erf=1, scale≈0).

    // shift = 2^45: the binade [2^45, 2^46) has ULP = 1/128, so adding shift
    // rounds any |x| in [0,6] to the nearest 1/128 grid point.
    // shift_u = reinterpret(shift) — derived at runtime to avoid a wrong hardcoded constant.
    const float64x2_t shift = vdupq_n_f64(0x1p45);
    const uint64x2_t shift_u = vreinterpretq_u64_f64(shift);  // 0x42C0000000000000
    const uint64x2_t sign_mask = vdupq_n_u64(0x8000000000000000ULL);

    // Taylor polynomial coefficients (universal constants from ARM glibc)
    const float64x2_t third = vdupq_n_f64(0x1.5555555555556p-2);        // 1/3
    const float64x2_t two_over_15 = vdupq_n_f64(0x1.1111111111111p-3);  // 2/15
    const float64x2_t tenth = vdupq_n_f64(-0x1.999999999999ap-4);       // -1/10
    const float64x2_t two_over_5 = vdupq_n_f64(-0x1.999999999999ap-2);  // -2/5
    const float64x2_t two_over_9 = vdupq_n_f64(-0x1.c71c71c71c71cp-3);  // -2/9
    const float64x2_t two_over_45 = vdupq_n_f64(0x1.6c16c16c16c17p-5);  // 2/45
    const float64x2_t sixth = vdupq_n_f64(0x1.5555555555556p-3);        // 1/6

    constexpr std::size_t W = stats::arch::simd::NEON_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        float64x2_t x = vld1q_f64(&a[i]);
        uint64x2_t x_u = vreinterpretq_u64_f64(x);
        uint64x2_t sign = vandq_u64(x_u, sign_mask);
        float64x2_t ax = vreinterpretq_f64_u64(vbicq_u64(x_u, sign_mask));  // |x|

        // Round |x| to nearest 1/128 via the shift trick (only valid for |x| in [0,6])
        float64x2_t z = vaddq_f64(ax, shift);
        // idx = k = round(|x|*128). For |x|>5.9921875 or NaN, z falls outside
        // the expected binade and the uint64 subtraction yields a large value.
        uint64x2_t idx = vsubq_u64(vreinterpretq_u64_f64(z), shift_u);
        // Clamp OOB and NaN lanes (vminq_u64 absent on aarch64; use compare+select)
        uint64x2_t oob = vcgtq_u64(idx, vdupq_n_u64(768));  // true where idx > 768
        idx = vbslq_u64(oob, vdupq_n_u64(768), idx);

        // Software gather: load {erf_r, scale} struct for each lane; deinterleave
        float64x2_t e0 = vld1q_f64(reinterpret_cast<const double*>(&kErfNeonTable[idx[0]]));
        float64x2_t e1 = vld1q_f64(reinterpret_cast<const double*>(&kErfNeonTable[idx[1]]));
        float64x2_t erfr = vuzp1q_f64(e0, e1);   // {erf(r0), erf(r1)}
        float64x2_t scale = vuzp2q_f64(e0, e1);  // {scale(r0), scale(r1)}

        // r = z - shift (the 1/128-grid point); d = |x| - r (residual)
        float64x2_t r = vsubq_f64(z, shift);
        float64x2_t d = vsubq_f64(ax, r);
        float64x2_t d2 = vmulq_f64(d, d);
        float64x2_t r2 = vmulq_f64(r, r);

        // 5-term Taylor polynomial coefficients (functions of r)
        // p1..p5 appear in: y = p1 + d·p2 + d²·p3 + d³·p4 + d⁴·p5
        // then: erf(x) ≈ erf(r) + scale·(d - d²·y)
        float64x2_t p1 = r;
        float64x2_t p2 = vfmsq_f64(third, r2, vaddq_f64(third, third));          // (1-2r²)/3
        float64x2_t p3 = vmulq_f64(r, vfmaq_f64(vdupq_n_f64(-0.5), r2, third));  // r(r²/3-½)
        float64x2_t p4 = vfmaq_f64(two_over_5, r2, two_over_15);                 // -2/5+2r²/15
        p4 = vfmsq_f64(tenth, r2, p4);                                           // -⅓⁰+2r²/5-2r⁴/15
        float64x2_t p5 = vfmaq_f64(two_over_9, r2, two_over_45);                 // -2/9+2r²/45
        p5 = vmulq_f64(r, vfmaq_f64(sixth, r2, p5));                             // r(№+r²(…))

        // Horner-style assembly of y and final combination
        float64x2_t p34 = vfmaq_f64(p3, d, p4);
        float64x2_t p12 = vfmaq_f64(p1, d, p2);
        float64x2_t y = vfmaq_f64(p34, d2, p5);
        y = vfmaq_f64(p12, d2, y);                        // p1+d·p2+d²·(…)
        y = vfmaq_f64(erfr, scale, vfmsq_f64(d, d2, y));  // erf(r)+scale·(d-d²·y)

        // Restore sign (erf is odd)
        vst1q_f64(&result[i], vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(y), sign)));
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
