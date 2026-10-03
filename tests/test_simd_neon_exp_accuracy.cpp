/**
 * @file test_simd_neon_exp_accuracy.cpp
 * @brief Regression test: vector_exp_neon holds the <1 ULP accuracy floor.
 *
 * Issue #33 Q1: vector_exp_neon was replaced with a table+polynomial kernel
 * (see docs/SIMD_BENCHMARK_RESULTS.md and PLAN.md "Issue #33 Experiment").
 * This test guards that replacement against future regressions using the same
 * correctly-rounded mpmath reference set used to validate it originally.
 *
 * Architecture-neutral: on non-NEON builds (or if NEON is unavailable at
 * runtime) this test is skipped rather than failing, since vector_exp() will
 * dispatch to a different backend there.
 */

#include "libstats/platform/cpu_detection.h"
#include "libstats/platform/simd.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

// Correctly-rounded exp() reference vectors (input_bits, exp_bits), evaluated at
// 200-bit precision with mpmath then rounded once to nearest double.
// Architecture-neutral -- shared with the AVX-512 dev-tool experiment.
// Defines struct ExpUlpVector and kExpUlpVectors[]. See scripts/gen_exp_ulp_vectors.py.
#include "exp_ulp_vectors.inc"

double bitsToF64(std::uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

std::uint64_t f64ToBits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
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

}  // namespace

TEST(NeonExpAccuracy, HoldsSubUlpFloorVsCorrectlyRoundedReference) {
    if (!stats::arch::supports_neon()) {
        GTEST_SKIP() << "NEON not available at runtime on this machine";
    }

    constexpr std::size_t N = sizeof(kExpUlpVectors) / sizeof(kExpUlpVectors[0]);
    std::vector<double> xin(N), yout(N);
    for (std::size_t i = 0; i < N; ++i)
        xin[i] = bitsToF64(kExpUlpVectors[i].x_bits);

    // Public dispatch API: N is far above the SIMD threshold, so this exercises
    // vector_exp_neon on this machine.
    stats::arch::simd::VectorOps::vector_exp(xin.data(), yout.data(), N);

    double core_max = 0.0, core_sum = 0.0, core_worst_x = 0.0;
    std::size_t core_n = 0;
    for (std::size_t i = 0; i < N; ++i) {
        const double ref = bitsToF64(kExpUlpVectors[i].exp_bits);
        // Core range |x| <= 700: both the polynomial and edge-fixup paths are
        // exercised elsewhere; this range is where the table+poly result applies.
        if (std::abs(xin[i]) <= 700.0) {
            const double ulp = expUlpError(yout[i], ref);
            if (ulp > core_max) {
                core_max = ulp;
                core_worst_x = xin[i];
            }
            core_sum += ulp;
            ++core_n;
        }
    }
    ASSERT_GT(core_n, 0u);
    const double core_mean = core_sum / static_cast<double>(core_n);

    EXPECT_LE(core_max, 1.0) << "vector_exp_neon exceeded the <1 ULP floor: max=" << core_max
                             << " ULP at x=" << core_worst_x;
    // Sanity bound on the mean so a systematic (not just worst-case) regression
    // would also be caught.
    EXPECT_LT(core_mean, 0.01) << "vector_exp_neon mean ULP error grew unexpectedly: " << core_mean;
}

TEST(NeonExpAccuracy, HandlesIeeeSpecialValues) {
    if (!stats::arch::supports_neon()) {
        GTEST_SKIP() << "NEON not available at runtime on this machine";
    }

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // Large enough to force the vectorized path (well above the SIMD threshold);
    // pad with benign values so the buffer size clears dispatch policy.
    std::vector<double> in(64, 0.1);
    std::vector<double> out(64, 0.0);
    in[0] = inf;
    in[1] = -inf;
    in[2] = nan;
    in[3] = 0.0;
    in[4] = -0.0;
    in[5] = 1.0;
    in[6] = -1.0;
    in[7] = 710.0;   // beyond the table's special-bound -> overflow to +inf
    in[8] = -750.0;  // underflows to exactly 0.0

    stats::arch::simd::VectorOps::vector_exp(in.data(), out.data(), in.size());

    EXPECT_EQ(out[0], inf) << "exp(+inf) must be +inf";
    EXPECT_EQ(out[1], 0.0) << "exp(-inf) must be 0";
    EXPECT_TRUE(std::isnan(out[2])) << "exp(NaN) must be NaN";
    EXPECT_EQ(out[3], 1.0) << "exp(0) must be exactly 1";
    EXPECT_EQ(out[4], 1.0) << "exp(-0) must be exactly 1";
    EXPECT_TRUE(std::isinf(out[7]) && out[7] > 0) << "exp(710) must overflow to +inf";
    EXPECT_EQ(out[8], 0.0) << "exp(-750) must underflow to 0";
}
