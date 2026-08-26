// tests/test_log_special_gates.cpp
//
// Per-tier special-value gate for the vector_log_<tier> kernels (issue
// #105). The x86 tiers historically lacked the cmpunord NaN blend that
// vector_exp/vector_erf carry, so a NaN lane rode the exponent-extraction
// path and exited as 710.188... -- and through the public batch API,
// LogNormal returned cdf(NaN) = 1, pdf(NaN) = 0: finite, plausible values
// for garbage input.
//
// Structure mirrors tests/test_trig_ulp_gates.cpp (issue #95): each
// compiled-in tier's public static (VectorOps::vector_log_{sse2,avx,avx2,
// avx512,neon}) is called directly, bypassing runtime dispatch, so a CPU
// preferring a higher tier cannot hide a lower tier's regression; tiers the
// CPU lacks are skipped at runtime via stats::arch::supports_*(). The
// dispatched vector_log() entry point is gated too.
//
// Specials are placed FIRST in each input array: the sweep bug that hid
// #105 (tools/accuracy_sweep.cpp, NV2) appended specials after the finite
// grid, dropping all of them into the scalar libm tail on AVX-512 -- so the
// kernel body's special handling was never tested. The tail gates place one
// special as the final element of a (W+1)-length array so it provably lands
// in the scalar remainder loop of a width-W tier.
//
// The contract is std::log semantics on every lane: NaN -> NaN, log(-1) ->
// NaN, log(0) -> -inf, log(+inf) -> +inf, log(5e-324) finite (~ -744.44;
// the subnormal prescale path). Finite lanes are held to a small ULP budget
// against libm.

#include "libstats/distributions/lognormal.h"
#include "libstats/platform/cpu_detection.h"
#include "libstats/platform/simd.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kDenormMin = 5e-324;  // std::numeric_limits<double>::denorm_min()

// The five special inputs of the #105 gate, in the order they lead each
// input array (specials FIRST -- see file banner).
constexpr double kSpecials[] = {kNaN, -1.0, 0.0, kInf, kDenormMin};
constexpr std::size_t kSpecialsN = sizeof(kSpecials) / sizeof(kSpecials[0]);

// Finite-lane ULP budget vs std::log. The SLEEF-derived kernels claim < 1
// ULP; 4 leaves cross-platform libm headroom without ever admitting the
// 710.188 laundering failure (~1e13 ULP away).
constexpr double kFiniteUlpBudget = 4.0;

// Sign-aware ULP distance on the integer lattice (same metric as
// test_trig_ulp_gates.cpp's trigUlpError, minus the non-finite arms which
// the caller handles explicitly here).
double ulpDistance(double got, double ref) {
    const auto ordered = [](double v) -> std::int64_t {
        std::int64_t i;
        std::memcpy(&i, &v, sizeof i);
        return i < 0 ? static_cast<std::int64_t>(0x8000000000000000ULL) - i : i;
    };
    const std::int64_t g = ordered(got), r = ordered(ref);
    return static_cast<double>(g > r ? g - r : r - g);
}

// Asserts one output lane matches std::log semantics for its input.
void expectLogSemantics(const char* tier, const char* where, std::size_t lane, double x,
                        double got) {
    const double ref = std::log(x);
    if (std::isnan(ref)) {
        EXPECT_TRUE(std::isnan(got)) << tier << " " << where << " lane " << lane << ": log(" << x
                                     << ") must be NaN, got " << got;
    } else if (std::isinf(ref)) {
        EXPECT_EQ(got, ref) << tier << " " << where << " lane " << lane << ": log(" << x
                            << ") must be " << ref << ", got " << got;
    } else {
        EXPECT_TRUE(std::isfinite(got)) << tier << " " << where << " lane " << lane << ": log(" << x
                                        << ") must be finite, got " << got;
        if (std::isfinite(got)) {
            EXPECT_LE(ulpDistance(got, ref), kFiniteUlpBudget)
                << tier << " " << where << " lane " << lane << ": log(" << x << ") = " << got
                << " vs libm " << ref;
        }
    }
}

using LogFn = void (*)(const double*, double*, std::size_t) noexcept;

// Body gate: 16 lanes (a multiple of every tier's width, 2/4/8), the five
// specials leading, benign fill behind them -- every special is evaluated
// by the vector body on every tier. All 16 lanes are asserted, so the fix's
// blend must not disturb finite lanes either.
void runBodyGate(const char* tier, LogFn log_fn) {
    constexpr std::size_t n = 16;
    std::vector<double> in(n), out(n);
    for (std::size_t i = 0; i < kSpecialsN; ++i)
        in[i] = kSpecials[i];
    const double benign[] = {0.5, 1.0, 2.718281828459045, 10.0, 1e-10, 1e10, 3.5,
                             0.125, 7.0, 42.0, 0.9999999999999999};
    for (std::size_t i = kSpecialsN; i < n; ++i)
        in[i] = benign[i - kSpecialsN];

    log_fn(in.data(), out.data(), n);
    for (std::size_t i = 0; i < n; ++i)
        expectLogSemantics(tier, "body", i, in[i], out[i]);
}

// Tail gate: for each special, a (W+1)-length array with benign values in
// the single full vector and the special as the last element -- provably in
// the width-W tier's scalar remainder loop (std::log on every tier since
// the 2026-08-21 NV1 fix; this pins it).
void runTailGate(const char* tier, LogFn log_fn, std::size_t width) {
    for (std::size_t s = 0; s < kSpecialsN; ++s) {
        const std::size_t n = width + 1;
        std::vector<double> in(n, 1.5), out(n, 0.0);
        in[n - 1] = kSpecials[s];
        log_fn(in.data(), out.data(), n);
        for (std::size_t i = 0; i < n; ++i)
            expectLogSemantics(tier, "tail", i, in[i], out[i]);
    }
}

}  // namespace

using stats::arch::simd::VectorOps;

// =========================================================================
// Per-tier gates. Compile-guarded by LIBSTATS_HAS_<TIER> (wired via the
// libstats::simd interface target every gtest links) so a
// LIBSTATS_MAX_SIMD_TIER-capped build still compiles; runtime-guarded by
// stats::arch::supports_<tier>().
// =========================================================================

#ifdef LIBSTATS_HAS_SSE2
TEST(LogSpecialGates, Sse2) {
    if (!stats::arch::supports_sse2()) {
        GTEST_SKIP() << "SSE2 not supported on this CPU";
    }
    runBodyGate("sse2", VectorOps::vector_log_sse2);
    runTailGate("sse2", VectorOps::vector_log_sse2, 2);
}
#endif

#ifdef LIBSTATS_HAS_AVX
TEST(LogSpecialGates, Avx) {
    if (!stats::arch::supports_avx()) {
        GTEST_SKIP() << "AVX not supported on this CPU";
    }
    runBodyGate("avx", VectorOps::vector_log_avx);
    runTailGate("avx", VectorOps::vector_log_avx, 4);
}
#endif

#ifdef LIBSTATS_HAS_AVX2
TEST(LogSpecialGates, Avx2) {
    if (!stats::arch::supports_avx2()) {
        GTEST_SKIP() << "AVX2 not supported on this CPU";
    }
    runBodyGate("avx2", VectorOps::vector_log_avx2);
    runTailGate("avx2", VectorOps::vector_log_avx2, 4);
}
#endif

#ifdef LIBSTATS_HAS_AVX512
TEST(LogSpecialGates, Avx512) {
    if (!stats::arch::supports_avx512()) {
        GTEST_SKIP() << "AVX-512 not supported on this CPU";
    }
    runBodyGate("avx512", VectorOps::vector_log_avx512);
    runTailGate("avx512", VectorOps::vector_log_avx512, 8);
}
#endif

#ifdef LIBSTATS_HAS_NEON
TEST(LogSpecialGates, Neon) {
    if (!stats::arch::supports_neon()) {
        GTEST_SKIP() << "NEON not supported on this CPU";
    }
    runBodyGate("neon", VectorOps::vector_log_neon);
    runTailGate("neon", VectorOps::vector_log_neon, 2);
}
#endif

// Fallback kernel: always compiled, always runnable -- pins the reference
// behavior the SIMD tiers are being held to.
TEST(LogSpecialGates, Fallback) {
    runBodyGate("fallback", VectorOps::vector_log_fallback);
    runTailGate("fallback", VectorOps::vector_log_fallback, 8);
}

// Dispatched entry point: whichever tier the CPU selects. 21 lanes with the
// specials leading (vector body on every width) and repeated at the back
// (scalar tail on an 8-wide tier; still asserted wherever they land, since
// every lane is checked against std::log semantics).
TEST(LogSpecialGates, DispatchedEntryPoint) {
    constexpr std::size_t n = 21;
    std::vector<double> in(n, 1.5), out(n, 0.0);
    for (std::size_t i = 0; i < kSpecialsN; ++i) {
        in[i] = kSpecials[i];
        in[n - kSpecialsN + i] = kSpecials[i];
    }
    VectorOps::vector_log(in.data(), out.data(), n);
    for (std::size_t i = 0; i < n; ++i)
        expectLogSemantics("dispatched", "mixed", i, in[i], out[i]);
}

// =========================================================================
// Public-API consequence gate (#105's user-visible symptom): LogNormal
// batch pdf/logpdf/cdf of NaN must be NaN through the span overloads. NaN
// leads the array (vector body) and closes it (scalar tail on every tier
// width); a benign lane guards against the fix breaking finite values.
// =========================================================================

TEST(LogSpecialGates, LogNormalBatchNaNPropagation) {
    auto dist_result = stats::LogNormalDistribution::create(0.0, 1.0);
    ASSERT_TRUE(dist_result.isOk());
    auto& dist = *dist_result;

    constexpr std::size_t n = 69;  // 8*8+5: NaN lands in every tier's body and tail
    std::vector<double> xs(n, 1.5);
    xs[0] = kNaN;
    xs[n - 1] = kNaN;
    std::vector<double> out(n, 0.0);
    const stats::detail::PerformanceHint force_simd{
        stats::detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED, std::nullopt};

    dist.getProbability(std::span<const double>(xs), std::span<double>(out), force_simd);
    EXPECT_TRUE(std::isnan(out[0])) << "LogNormal batch pdf(NaN) [body] = " << out[0];
    EXPECT_TRUE(std::isnan(out[n - 1])) << "LogNormal batch pdf(NaN) [tail] = " << out[n - 1];
    EXPECT_FALSE(std::isnan(out[1])) << "LogNormal batch pdf(1.5) went NaN";

    dist.getLogProbability(std::span<const double>(xs), std::span<double>(out), force_simd);
    EXPECT_TRUE(std::isnan(out[0])) << "LogNormal batch logpdf(NaN) [body] = " << out[0];
    EXPECT_TRUE(std::isnan(out[n - 1])) << "LogNormal batch logpdf(NaN) [tail] = " << out[n - 1];
    EXPECT_FALSE(std::isnan(out[1])) << "LogNormal batch logpdf(1.5) went NaN";

    dist.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out), force_simd);
    EXPECT_TRUE(std::isnan(out[0])) << "LogNormal batch cdf(NaN) [body] = " << out[0];
    EXPECT_TRUE(std::isnan(out[n - 1])) << "LogNormal batch cdf(NaN) [tail] = " << out[n - 1];
    EXPECT_FALSE(std::isnan(out[1])) << "LogNormal batch cdf(1.5) went NaN";
}
