/**
 * @file neon_table_transcendental_probe.cpp
 * @brief Issue #33 Q1: NEON table-gather exp/log vs current SLEEF polynomials
 *
 * The x86 half of Issue #33 Q2 closed null on BOTH AVX2 (Kaby Lake) and AVX-512
 * (Zen 4). On AVX-512 the accurate (<1 ULP) table-exp needs a tail-corrected
 * table = a SECOND hardware gather per element, and that extra memory traffic
 * made it slower than the current SLEEF polynomial under realistic streaming
 * (see PLAN.md "Issue #33 Experiment" and the AVX-512 Stage 3 result).
 *
 * Q1 asks the untested aarch64 question. NEON has NO hardware gather; the
 * idiomatic software gather is a single 128-bit vld1q per lane plus a vuzp
 * deinterleave (exactly what vector_erf_neon does with kErfNeonTable). Crucially,
 * NEON's natural load width is 128 bits = 2 doubles, so an Array-of-Structs table
 * of adjacent double pairs (exp: {tail, sbits}; log: {invc, logc}) pulls BOTH
 * values an entry needs in ONE load. The "second gather" penalty that sank the
 * x86 table-exp therefore may not apply here -- that is the open empirical
 * question this probe settles, for exp and log independently.
 *
 * Both kernels are faithful 2-wide NEON ports of ARM optimized-routines' scalar
 * math (math/exp.c + exp_data.c; math/log.c + log_data.c; SPDX: MIT OR
 * Apache-2.0 WITH LLVM-exception), sharing the same N=128 tables
 * (src/neon_exp_data.inc, src/neon_log_data.inc) and polynomials. See
 * THIRD_PARTY_NOTICES.md.
 *
 * The two gates, applied per primitive (see PLAN.md):
 *   - Accuracy: hold the current <1 ULP floor vs a correctly-rounded mpmath
 *     reference, and be correct at the IEEE edges.
 *   - Performance: >=20% faster than the current NEON polynomial at the
 *     realistic streaming regime.
 *
 * The kernels live in this opt-in dev tool ONLY (LIBSTATS_BUILD_SIMD_DEV_TOOLS,
 * default OFF). Production vector_exp_neon/vector_log_neon (src/simd_neon.cpp)
 * are untouched and these symbols are never added to the dispatch table.
 */

#include "tool_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#if defined(LIBSTATS_HAS_NEON) && (defined(__aarch64__) || defined(_M_ARM64))
    #define LIBSTATS_NEON_PROBE_AVAILABLE 1
    #include <arm_neon.h>
#else
    #define LIBSTATS_NEON_PROBE_AVAILABLE 0
#endif

namespace {

#if LIBSTATS_NEON_PROBE_AVAILABLE

    // N=128 tail-corrected table, Array-of-Structs {tail_bits, sbits}. Same ARM
    // optimized-routines values as the AVX-512 experiment, re-interleaved so one
    // vld1q_u64 pulls both per lane. See scripts/gen_neon_exp_table.py.
    #include "neon_exp_data.inc"  // kExpNeonTable[128]

    // Correctly-rounded exp() reference vectors (input_bits, exp_bits), evaluated at
    // 200-bit precision with mpmath then rounded once to nearest double. This set is
    // architecture-neutral -- it is pure mathematics, not an AVX-512 artifact -- and is
    // also used by the production NEON exp regression test (tests/test_simd_neon_exp_accuracy.cpp).
    // Defines struct ExpUlpVector and kExpUlpVectors[]. See scripts/gen_exp_ulp_vectors.py.
    #include "exp_ulp_vectors.inc"

    // N=128 log table, Array-of-Structs {invc, logc} -- ARM's tab layout verbatim, so
    // one vld1q_f64 pulls both per lane. See scripts/gen_neon_log_table.py.
    #include "neon_log_data.inc"  // kLogNeonTable[128]

    // Correctly-rounded log() reference vectors. Defines struct LogUlpVector and
    // kLogUlpVectors[]. See scripts/gen_log_ulp_vectors.py.
    #include "log_ulp_vectors.inc"

// ARM AOR exp constants (math/exp_data.c, N=128 block) -- identical to the
// AVX-512 Stage 3 kernel; the algorithm is architecture-independent.
constexpr double kExpInvLn2N = 0x1.71547652b82fep0 * 128.0;  // N/ln2
constexpr double kExpNegLn2hiN = -0x1.62e42fefa0000p-8;      // -ln2/N (hi)
constexpr double kExpNegLn2loN = -0x1.cf79abc9e3b3ap-47;     // -ln2/N (lo)
constexpr double kExpShift = 0x1.8p52;
constexpr double kExpC2 = 0x1.ffffffffffdbdp-2;
constexpr double kExpC3 = 0x1.555555555543cp-3;
constexpr double kExpC4 = 0x1.55555cf172b91p-5;
constexpr double kExpC5 = 0x1.1111167a4d017p-7;
constexpr double kExpSpecialBound = 704.0;  // |x| beyond this: exact scalar fixup

// ARM AOR log constants (math/log_data.c + log.c, N=128, LOG_TABLE_BITS=7,
// LOG_POLY_ORDER=6). aarch64 has fast FMA, so r = fma(z, invc, -1) and tab2 is
// not needed. The near-1 band [1 - 2^-4, 1 + 0x1.09p-4] is routed to scalar log.
constexpr double kLogLn2hi = 0x1.62e42fefa3800p-1;
constexpr double kLogLn2lo = 0x1.ef35793c76730p-45;
constexpr double kLogA0 = -0x1.0000000000001p-1;
constexpr double kLogA1 = 0x1.555555551305bp-2;
constexpr double kLogA2 = -0x1.fffffffeb459p-3;
constexpr double kLogA3 = 0x1.999b324f10111p-3;
constexpr double kLogA4 = -0x1.55575e506c89fp-3;
constexpr std::uint64_t kLogOff = 0x3fe6000000000000ULL;

// Bit-cast helpers (used by the log kernel's near-1 band constants and by the
// ULP-measurement harnesses below).
inline double bitsToF64(std::uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

inline std::uint64_t f64ToBits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

// Experimental NEON table-gather exp. NOT wired into the dispatch table.
// 2x unrolled (4 doubles/iteration) to expose ILP to the M1's wide out-of-order
// backend, mirroring dot_product_neon/vector_add_neon; the rare |x| >= 704 edge
// fixup is hoisted to a single predicted-not-taken branch per iteration instead
// of one per lane.
void vectorExpNeonGather(const double* values, double* results, std::size_t size) {
    const float64x2_t invln2N = vdupq_n_f64(kExpInvLn2N);
    const float64x2_t shift = vdupq_n_f64(kExpShift);
    const float64x2_t negln2hiN = vdupq_n_f64(kExpNegLn2hiN);
    const float64x2_t negln2loN = vdupq_n_f64(kExpNegLn2loN);
    const float64x2_t c2 = vdupq_n_f64(kExpC2);
    const float64x2_t c3 = vdupq_n_f64(kExpC3);
    const float64x2_t c4 = vdupq_n_f64(kExpC4);
    const float64x2_t c5 = vdupq_n_f64(kExpC5);
    const uint64x2_t idx_mask = vdupq_n_u64(127);
    const float64x2_t special_bound = vdupq_n_f64(kExpSpecialBound);

    // Per-vector core: reduction + software gather + tail-corrected order-5 poly.
    // Inlined in Release, so the loop-invariant const vectors above are hoisted and
    // the unrolled + remainder paths share one definition (no code duplication).
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
        // pair; vuzp deinterleaves. This is the NEON analogue of the x86 two-gather,
        // but costs a single load per lane because the pair is 16 bytes = one vld1q.
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
    const auto fixupSpecial = [&](std::size_t base, std::size_t count) {
        for (std::size_t l = 0; l < count; ++l) {
            if (std::fabs(values[base + l]) >= kExpSpecialBound)
                results[base + l] = std::exp(values[base + l]);
        }
    };

    const std::size_t unroll_end = (size / 4) * 4;  // 2 vectors (4 doubles) per iteration
    const std::size_t simd_end = (size / 2) * 2;

    std::size_t i = 0;
    for (; i < unroll_end; i += 4) {
        float64x2_t x0 = vld1q_f64(&values[i]);
        float64x2_t x1 = vld1q_f64(&values[i + 2]);
        vst1q_f64(&results[i], expCore(x0));
        vst1q_f64(&results[i + 2], expCore(x1));

        // Hoisted edge check: OR both lane-masks of both vectors, branch once. The
        // SIMD compares are cheap; the branch is not taken for in-range batches.
        uint64x2_t any = vorrq_u64(vcgeq_f64(vabsq_f64(x0), special_bound),
                                   vcgeq_f64(vabsq_f64(x1), special_bound));
        if (vgetq_lane_u64(any, 0) | vgetq_lane_u64(any, 1))
            fixupSpecial(i, 4);
    }
    for (; i < simd_end; i += 2) {
        float64x2_t x = vld1q_f64(&values[i]);
        vst1q_f64(&results[i], expCore(x));
        fixupSpecial(i, 2);
    }
    for (; i < size; ++i)
        results[i] = std::exp(values[i]);
}

// Experimental NEON table-gather log. NOT wired into the dispatch table. Faithful
// 2-wide port of ARM optimized-routines scalar log (math/log.c, N=128,
// LOG_POLY_ORDER=6). Same 2x-unroll + hoisted-fallback structure as the exp
// kernel. The near-1.0 cancellation band and all non-normal-positive inputs
// (x<=0, subnormal, inf, NaN) route to scalar std::log -- a prototype
// simplification; a production kernel would port ARM's vectorized near-1 path
// (poly1) instead of falling back for that band.
void vectorLogNeonGather(const double* values, double* results, std::size_t size) {
    const uint64x2_t off = vdupq_n_u64(kLogOff);
    const uint64x2_t idx_mask = vdupq_n_u64(127);
    const uint64x2_t exp_field = vdupq_n_u64(0xfffULL << 52);
    const float64x2_t neg_one = vdupq_n_f64(-1.0);
    const float64x2_t ln2hi = vdupq_n_f64(kLogLn2hi);
    const float64x2_t ln2lo = vdupq_n_f64(kLogLn2lo);
    const float64x2_t a0 = vdupq_n_f64(kLogA0);
    const float64x2_t a1 = vdupq_n_f64(kLogA1);
    const float64x2_t a2 = vdupq_n_f64(kLogA2);
    const float64x2_t a3 = vdupq_n_f64(kLogA3);
    const float64x2_t a4 = vdupq_n_f64(kLogA4);
    const float64x2_t dbl_min = vdupq_n_f64(0x1p-1022);
    const float64x2_t inf = vdupq_n_f64(std::numeric_limits<double>::infinity());
    // near-1 band: [1 - 2^-4, 1 + 0x1.09p-4], detected as (ix - LO) < (HI - LO).
    const std::uint64_t lo_bits_s = f64ToBits(1.0 - 0x1p-4);
    const std::uint64_t band_w = f64ToBits(1.0 + 0x1.09p-4) - lo_bits_s;
    const uint64x2_t lo_bits = vdupq_n_u64(lo_bits_s);
    const uint64x2_t band = vdupq_n_u64(band_w);

    // Per-vector core: OFF-relative index + software gather + degree-5 poly.
    const auto logCore = [&](float64x2_t x) -> float64x2_t {
        uint64x2_t ix = vreinterpretq_u64_f64(x);
        uint64x2_t tmp = vsubq_u64(ix, off);
        uint64x2_t idx = vandq_u64(vshrq_n_u64(tmp, 45), idx_mask);  // (tmp >> 45) % 128
        int64x2_t k = vshrq_n_s64(vreinterpretq_s64_u64(tmp), 52);   // arithmetic shift
        uint64x2_t iz = vsubq_u64(ix, vandq_u64(tmp, exp_field));    // clear exponent -> z
        float64x2_t z = vreinterpretq_f64_u64(iz);

        // Software gather of {invc, logc} pairs; one vld1q_f64 per lane + vuzp.
        float64x2_t g0 =
            vld1q_f64(reinterpret_cast<const double*>(&kLogNeonTable[vgetq_lane_u64(idx, 0)]));
        float64x2_t g1 =
            vld1q_f64(reinterpret_cast<const double*>(&kLogNeonTable[vgetq_lane_u64(idx, 1)]));
        float64x2_t invc = vuzp1q_f64(g0, g1);  // {invc0, invc1}
        float64x2_t logc = vuzp2q_f64(g0, g1);  // {logc0, logc1}

        float64x2_t r = vfmaq_f64(neg_one, z, invc);  // z*invc - 1, |r| < 1/(2N)
        float64x2_t kd = vcvtq_f64_s64(k);
        float64x2_t w = vfmaq_f64(logc, kd, ln2hi);  // kd*ln2hi + logc
        float64x2_t hi = vaddq_f64(w, r);
        float64x2_t lo = vaddq_f64(vsubq_f64(w, hi), r);  // (w - hi) + r
        lo = vfmaq_f64(lo, kd, ln2lo);                    // + kd*ln2lo

        // y = lo + r2*A0 + r3*(A1 + r*A2 + r2*(A3 + r*A4)) + hi
        float64x2_t r2 = vmulq_f64(r, r);
        float64x2_t r3 = vmulq_f64(r, r2);
        float64x2_t mid = vfmaq_f64(a3, r, a4);          // A3 + r*A4
        mid = vfmaq_f64(vfmaq_f64(a1, r, a2), r2, mid);  // (A1 + r*A2) + r2*(A3 + r*A4)
        float64x2_t y = vfmaq_f64(lo, r2, a0);           // lo + r2*A0
        y = vfmaq_f64(y, r3, mid);                       // + r3*mid
        return vaddq_f64(y, hi);                         // + hi
    };

    // "ok" mask per lane: normal positive finite AND outside the near-1 band, i.e.
    // the table path is valid. Non-ok lanes are recomputed with scalar std::log.
    const auto okMask = [&](float64x2_t x) -> uint64x2_t {
        uint64x2_t normal = vandq_u64(vcgeq_f64(x, dbl_min), vcltq_f64(x, inf));
        uint64x2_t ix = vreinterpretq_u64_f64(x);
        uint64x2_t near1 = vcltq_u64(vsubq_u64(ix, lo_bits), band);
        return vbicq_u64(normal, near1);  // normal AND (NOT near1)
    };

    const auto fixupScalar = [&](std::size_t base, std::size_t count) {
        for (std::size_t l = 0; l < count; ++l) {
            const double x = values[base + l];
            const bool normal = (x >= 0x1p-1022) && (x < std::numeric_limits<double>::infinity());
            const bool near1 = (f64ToBits(x) - lo_bits_s) < band_w;
            if (!normal || near1)
                results[base + l] = std::log(x);
        }
    };

    const std::size_t unroll_end = (size / 4) * 4;
    const std::size_t simd_end = (size / 2) * 2;
    std::size_t i = 0;
    for (; i < unroll_end; i += 4) {
        float64x2_t x0 = vld1q_f64(&values[i]);
        float64x2_t x1 = vld1q_f64(&values[i + 2]);
        vst1q_f64(&results[i], logCore(x0));
        vst1q_f64(&results[i + 2], logCore(x1));
        // Hoisted fallback check: AND all four lanes' ok masks, branch once.
        uint64x2_t okboth = vandq_u64(okMask(x0), okMask(x1));
        if (~(vgetq_lane_u64(okboth, 0) & vgetq_lane_u64(okboth, 1)))
            fixupScalar(i, 4);
    }
    for (; i < simd_end; i += 2) {
        float64x2_t x = vld1q_f64(&values[i]);
        vst1q_f64(&results[i], logCore(x));
        fixupScalar(i, 2);
    }
    for (; i < size; ++i)
        results[i] = std::log(values[i]);
}

// ULP distance for nonnegative exp results; inf/NaN handled explicitly.
double expUlpError(double got, double ref) {
    if (std::isnan(ref))
        return std::isnan(got) ? 0.0 : 1e18;
    if (std::isinf(ref))
        return (got == ref) ? 0.0 : 1e18;
    if (!std::isfinite(got))
        return 1e18;
    const std::uint64_t g = f64ToBits(got), r = f64ToBits(ref);
    return static_cast<double>(g > r ? g - r : r - g);
}

// ULP distance for signed results (log spans negatives), via the IEEE total-order
// map (b<0 -> ~b; b>=0 -> b|2^63) so bit distance is monotone across zero.
// inf/NaN handled explicitly.
double logUlpError(double got, double ref) {
    if (std::isnan(ref))
        return std::isnan(got) ? 0.0 : 1e18;
    if (std::isinf(ref))
        return (got == ref) ? 0.0 : 1e18;
    if (!std::isfinite(got))
        return 1e18;
    auto mono = [](double d) -> std::uint64_t {
        const std::uint64_t b = f64ToBits(d);
        return (b & 0x8000000000000000ULL) ? ~b : (b | 0x8000000000000000ULL);
    };
    const std::uint64_t g = mono(got), r = mono(ref);
    return static_cast<double>(g > r ? g - r : r - g);
}

double nsPerElem(std::chrono::steady_clock::duration total, std::size_t op_count) {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(total).count()) /
           static_cast<double>(op_count);
}

template <typename Fn>
double benchNsPerElem(Fn&& fn, const double* in, double* out, std::size_t n, std::size_t iters) {
    fn(in, out, n);  // warmup + populate caches
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t it = 0; it < iters; ++it)
        fn(in, out, n);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    volatile double keep = out[n - 1];
    (void)keep;
    return nsPerElem(elapsed, n * iters);
}

void runExpExperimentNeon() {
    stats::detail::detail::subsectionHeader(
        "Issue #33 Q1: table-gather exp vs current polynomial exp (NEON)");

    auto current = [](const double* in, double* out, std::size_t n) {
        stats::arch::simd::VectorOps::vector_exp(in, out, n);
    };
    auto gather = [](const double* in, double* out, std::size_t n) {
        vectorExpNeonGather(in, out, n);
    };

    // ---- Accuracy gate: ULP vs correctly-rounded mpmath reference ----
    constexpr std::size_t NV = sizeof(kExpUlpVectors) / sizeof(kExpUlpVectors[0]);
    std::vector<double> xin(NV), yg(NV), yc(NV);
    for (std::size_t i = 0; i < NV; ++i)
        xin[i] = bitsToF64(kExpUlpVectors[i].x_bits);
    gather(xin.data(), yg.data(), NV);
    current(xin.data(), yc.data(), NV);

    // Head-to-head over the core range |x| <= 700, where both kernels run their
    // polynomial path. The table kernel routes |x| >= 704 to exact scalar exp, so
    // it is additionally correct at the edges the current kernel clamps (+/-708);
    // those edge points are therefore not a fair precision comparison and are
    // excluded from the current-kernel max.
    double g_core = 0, g_all = 0, g_sum = 0, c_core = 0, gworst = 0;
    std::size_t core_n = 0;
    for (std::size_t i = 0; i < NV; ++i) {
        const double ref = bitsToF64(kExpUlpVectors[i].exp_bits);
        const double ug = expUlpError(yg[i], ref);
        g_all = std::max(g_all, ug);
        if (std::abs(xin[i]) <= 700.0) {
            if (ug > g_core) {
                g_core = ug;
                gworst = xin[i];
            }
            c_core = std::max(c_core, expUlpError(yc[i], ref));
            g_sum += ug;
            ++core_n;
        }
    }
    std::cout << "Accuracy vs correctly-rounded reference (" << NV << " points):\n";
    std::cout << "  table-gather exp : core(|x|<=700) max " << g_core << " ULP, mean "
              << (g_sum / static_cast<double>(core_n)) << " ULP; full-range max " << g_all
              << " ULP (worst core x = " << gworst << ")\n";
    std::cout << "  current poly exp : core(|x|<=700) max " << c_core
              << " ULP (clamps beyond +/-708; edges not compared)\n";

    // Special IEEE inputs through the vectorized path. exp(-710) is a subnormal
    // ~4.7e-309 (not 0); use -750 for the true underflow-to-zero case.
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double sx[8] = {inf, -inf, nan, 0.0, 1.0, -1.0, 710.0, -750.0};
    double so[8];
    gather(sx, so, 8);
    const bool specials_ok = (so[0] == inf) && (so[1] == 0.0) && std::isnan(so[2]) &&
                             (so[3] == 1.0) && std::isinf(so[6]) && (so[7] == 0.0);
    std::cout << "  special inputs (+/-inf, NaN, overflow, underflow): "
              << (specials_ok ? "OK" : "MISMATCH") << "\n";
    const bool accuracy_ok = (g_core <= 1.0) && (g_core <= c_core) && specials_ok;
    std::cout << "  Accuracy gate (<=1 ULP, no regression vs current): "
              << (accuracy_ok ? "PASS" : "FAIL") << "\n\n";

    // ---- Performance gate: >=20% at a realistic streaming regime ----
    // Note: the M1 L1 data cache boundary is ~4096 elements (two-array footprint);
    // the "stream" regime is deliberately well past it. See
    // docs/SIMD_OPTIMIZATION_REFERENCE.md and the NEON threshold sweep in
    // docs/SIMD_BENCHMARK_RESULTS.md.
    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    auto measure = [&](const char* label, std::size_t n, std::size_t iters) {
        std::vector<double> in(n), out(n);
        for (auto& v : in)
            v = dist(rng);
        const double c = benchNsPerElem(current, in.data(), out.data(), n, iters);
        const double g = benchNsPerElem(gather, in.data(), out.data(), n, iters);
        const double speedup = c / g;
        std::cout << "  " << label << ": current " << c << " ns/elem, table-gather " << g
                  << " ns/elem, speedup " << speedup << "x (" << ((speedup - 1.0) * 100.0)
                  << "%)\n";
        return speedup;
    };

    stats::detail::detail::subsectionHeader("Throughput (ns per element, lower is better)");
    measure("hot    ( 2K elems, L1-resident)   ", 2'048, 40'000);
    const double stream_speedup = measure("stream (256K elems, well past L1)", 262'144, 600);

    std::cout << "\n  Performance gate (>=20% at stream): "
              << (stream_speedup >= 1.20 ? "PASS" : "FAIL") << "\n";
    std::cout << "  Overall verdict: table-gather exp is "
              << ((accuracy_ok && stream_speedup >= 1.20) ? "a WIN (both gates pass)"
                                                          : "NOT a clear win (see gates above)")
              << ".\n";
}

void runLogExperimentNeon() {
    stats::detail::detail::subsectionHeader(
        "Issue #33 Q1: table-gather log vs current polynomial log (NEON)");

    auto current = [](const double* in, double* out, std::size_t n) {
        stats::arch::simd::VectorOps::vector_log(in, out, n);
    };
    auto gather = [](const double* in, double* out, std::size_t n) {
        vectorLogNeonGather(in, out, n);
    };

    // ---- Accuracy gate: ULP vs correctly-rounded mpmath reference ----
    // The gather kernel routes the near-1 band and non-normal inputs to scalar
    // std::log; the reported max is dominated by the table path elsewhere.
    constexpr std::size_t NV = sizeof(kLogUlpVectors) / sizeof(kLogUlpVectors[0]);
    std::vector<double> xin(NV), yg(NV), yc(NV);
    for (std::size_t i = 0; i < NV; ++i)
        xin[i] = bitsToF64(kLogUlpVectors[i].x_bits);
    gather(xin.data(), yg.data(), NV);
    current(xin.data(), yc.data(), NV);

    double g_max = 0, g_sum = 0, c_max = 0, gworst = 0;
    for (std::size_t i = 0; i < NV; ++i) {
        const double ref = bitsToF64(kLogUlpVectors[i].log_bits);
        const double ug = logUlpError(yg[i], ref);
        if (ug > g_max) {
            g_max = ug;
            gworst = xin[i];
        }
        g_sum += ug;
        c_max = std::max(c_max, logUlpError(yc[i], ref));
    }
    std::cout << "Accuracy vs correctly-rounded reference (" << NV
              << " points, positive domain):\n";
    std::cout << "  table-gather log : max " << g_max << " ULP, mean "
              << (g_sum / static_cast<double>(NV)) << " ULP (worst x = " << gworst << ")\n";
    std::cout << "  current poly log : max " << c_max << " ULP\n";

    // Special/invalid IEEE inputs: log(0)=-inf, log(neg)=NaN, log(inf)=inf,
    // log(NaN)=NaN, log(1)=0.
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    double sx[8] = {0.0, -1.0, inf, nan, 1.0, 2.0, 0.5, 0x1p-1022};
    double so[8];
    gather(sx, so, 8);
    const bool specials_ok = (so[0] == -inf) && std::isnan(so[1]) && (so[2] == inf) &&
                             std::isnan(so[3]) && (so[4] == 0.0);
    std::cout << "  special inputs (0, negative, +inf, NaN, 1): "
              << (specials_ok ? "OK" : "MISMATCH") << "\n";
    const bool accuracy_ok = (g_max <= 1.0) && (g_max <= c_max) && specials_ok;
    std::cout << "  Accuracy gate (<=1 ULP, no regression vs current): "
              << (accuracy_ok ? "PASS" : "FAIL") << "\n\n";

    // ---- Performance gate: >=20% at a realistic streaming regime ----
    // Inputs are positive and span many binades (x = exp(U[-7,7]) ~ [9e-4, 1100]),
    // mirroring log-pdf workloads; few land in the near-1 scalar-fallback band.
    std::mt19937 rng(0x106);
    std::uniform_real_distribution<double> u(-7.0, 7.0);

    auto measure = [&](const char* label, std::size_t n, std::size_t iters) {
        std::vector<double> in(n), out(n);
        for (auto& v : in)
            v = std::exp(u(rng));
        const double c = benchNsPerElem(current, in.data(), out.data(), n, iters);
        const double g = benchNsPerElem(gather, in.data(), out.data(), n, iters);
        const double speedup = c / g;
        std::cout << "  " << label << ": current " << c << " ns/elem, table-gather " << g
                  << " ns/elem, speedup " << speedup << "x (" << ((speedup - 1.0) * 100.0)
                  << "%)\n";
        return speedup;
    };

    stats::detail::detail::subsectionHeader("Throughput (ns per element, lower is better)");
    measure("hot    ( 2K elems, L1-resident)   ", 2'048, 40'000);
    const double stream_speedup = measure("stream (256K elems, well past L1)", 262'144, 600);

    std::cout << "\n  Performance gate (>=20% at stream): "
              << (stream_speedup >= 1.20 ? "PASS" : "FAIL") << "\n";
    std::cout << "  Overall verdict: table-gather log is "
              << ((accuracy_ok && stream_speedup >= 1.20) ? "a WIN (both gates pass)"
                                                          : "NOT a clear win (see gates above)")
              << ".\n";
}

#endif  // LIBSTATS_NEON_PROBE_AVAILABLE

}  // namespace

int main() {
    return stats::detail::detail::runTool("neon_table_transcendental_probe", [] {
        stats::detail::detail::displayToolHeader(
            "NEON Table-Gather Transcendental Probe (Issue #33, Q1)",
            "Do ARM-glibc-style table+polynomial exp/log kernels beat the current "
            "SLEEF polynomials on NEON? See PLAN.md 'Issue #33 Experiment'.");
#if LIBSTATS_NEON_PROBE_AVAILABLE
        if (stats::arch::supports_neon()) {
            runExpExperimentNeon();
            runLogExperimentNeon();
        } else {
            std::cout << "NEON not available at runtime on this CPU -- skipping probe.\n";
        }
#else
        std::cout << "This probe is NEON-only (Issue #33 Q1) and is not applicable on this "
                     "build/architecture.\n";
#endif
    });
}
