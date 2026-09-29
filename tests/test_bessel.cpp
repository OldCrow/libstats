/**
 * @file test_bessel.cpp
 * @brief Guards for the Bessel helpers in bessel.h (corvus i0/i1/i0e/i1e since
 *        v2.5.0): values against mpmath, parity, and finiteness past the I₀
 *        overflow boundary.
 *
 * Unlabelled on purpose so it runs under `ctest -LE "timing|benchmark"`; the
 * pre-v2.5.0 version of this file (test_bessel_tier) guarded the #97 tier
 * plumbing, which no longer exists.
 */

#include "libstats/core/bessel.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>

namespace {

// Relative tolerance in units of 2^-52 (one double ULP at 1).
constexpr double kUlp = 2.220446049250313e-16;

void expectRel(double got, double ref, double ulps, const char* what) {
    EXPECT_NEAR(got, ref, ulps * kUlp * std::fabs(ref)) << what << " ref=" << ref;
}

}  // namespace

TEST(Bessel, ValuesAgainstMpmath) {
    // mpmath, dps 50.
    expectRel(stats::detail::bessel_i0(1.0), 1.2660658777520084, 4, "I0(1)");
    expectRel(stats::detail::bessel_i1(1.0), 0.565159103992485, 4, "I1(1)");

    // log I0: one form for every x, no asymptotic seam.
    expectRel(stats::detail::log_bessel_i0(1.0), 0.23591435850717865, 8, "logI0(1)");
    expectRel(stats::detail::log_bessel_i0(10.0), 7.942972083118695, 4, "logI0(10)");
    expectRel(stats::detail::log_bessel_i0(700.0), 695.8056999984434, 4, "logI0(700)");
    expectRel(stats::detail::log_bessel_i0(800.0), 795.738911950745, 4, "logI0(800)");
    expectRel(stats::detail::log_bessel_i0(1e5), 99993.32459998432, 4, "logI0(1e5)");

    // A(k) = I1/I0: i1e/i0e below the cut, 1 - series above it.
    expectRel(stats::detail::bessel_i1_over_i0(1.0), 0.4463899658965345, 8, "A(1)");
    expectRel(stats::detail::bessel_i1_over_i0(10.0), 0.9485998259548459, 8, "A(10)");
    expectRel(stats::detail::bessel_i1_over_i0(100.0), 0.9949873730051688, 8, "A(100)");
    expectRel(stats::detail::bessel_i1_over_i0(800.0), 0.9993748044428813, 8, "A(800)");

    // 1 - A(k): direct below the cut (1 - A cancels ~log2(2k) bits), series
    // above it (~130 ULP near the cut, sub-ULP away from it; see bessel.h).
    expectRel(stats::detail::bessel_i1_i0_complement(1.0), 0.5536100341034655, 8, "1-A(1)");
    expectRel(stats::detail::bessel_i1_i0_complement(10.0), 0.05140017404515404, 64, "1-A(10)");
    expectRel(stats::detail::bessel_i1_i0_complement(50.0), 0.010051032621502247, 512, "1-A(50)");
    expectRel(stats::detail::bessel_i1_i0_complement(100.0), 0.005012626994831235, 8, "1-A(100)");
    expectRel(stats::detail::bessel_i1_i0_complement(800.0), 0.0006251955571187059, 8, "1-A(800)");
}

TEST(Bessel, Parity) {
    // I0 is even, I1 is odd, log I0 is even — including past the I0 overflow.
    for (const double x : {0.5, 1.0, 3.0, 12.0, 800.0}) {
        EXPECT_DOUBLE_EQ(stats::detail::bessel_i0(-x), stats::detail::bessel_i0(x));
        EXPECT_DOUBLE_EQ(stats::detail::bessel_i1(-x), -stats::detail::bessel_i1(x));
        EXPECT_DOUBLE_EQ(stats::detail::log_bessel_i0(-x), stats::detail::log_bessel_i0(x));
    }
}

TEST(Bessel, LimitsAndOverflow) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    EXPECT_EQ(stats::detail::bessel_i0(0.0), 1.0);
    EXPECT_EQ(stats::detail::bessel_i1(0.0), 0.0);
    EXPECT_EQ(stats::detail::log_bessel_i0(0.0), 0.0);
    EXPECT_EQ(stats::detail::bessel_i0(800.0), inf);
    EXPECT_TRUE(std::isfinite(stats::detail::log_bessel_i0(800.0)));
    EXPECT_EQ(stats::detail::log_bessel_i0(inf), inf);
    // Ratio helpers at the ends of the domain (#93: no inf/inf NaN).
    EXPECT_EQ(stats::detail::bessel_i1_over_i0(0.0), 0.0);
    EXPECT_EQ(stats::detail::bessel_i1_i0_complement(0.0), 1.0);
    EXPECT_TRUE(std::isfinite(stats::detail::bessel_i1_over_i0(1e5)));
    EXPECT_EQ(stats::detail::bessel_i1_over_i0(inf), 1.0);
    EXPECT_EQ(stats::detail::bessel_i1_i0_complement(inf), 0.0);
}
