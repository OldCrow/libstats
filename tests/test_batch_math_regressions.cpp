/**
 * @file test_batch_math_regressions.cpp
 * @brief Regression tests for v2.0.0 mathematical correctness review findings.
 */

#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#include "include/tests.h"
#include "libstats/distributions/binomial.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/negative_binomial.h"
#include "libstats/distributions/poisson.h"
#include "libstats/distributions/von_mises.h"
#include "libstats/platform/simd.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>
#include <span>
#include <vector>

using namespace stats;

TEST(BatchMathRegressions, GammaBatchCDFLargeAlphaMatchesScalar) {
    auto gamma = GammaDistribution::create(200.0, 1.5).unwrap();
    std::vector<double> values = {100.0, 120.0, 140.0, 160.0};
    std::vector<double> batch(values.size());

    gamma.getCumulativeProbability(std::span<const double>(values), std::span<double>(batch));

    for (std::size_t i = 0; i < values.size(); ++i) {
        const double scalar = gamma.getCumulativeProbability(values[i]);
        EXPECT_TRUE(std::isfinite(batch[i])) << "batch CDF NaN/inf at index " << i;
        EXPECT_NEAR(batch[i], scalar, 1e-12) << "batch/scalar mismatch at index " << i;
    }
}

TEST(BatchMathRegressions, BinomialBatchProbabilityHandlesBoundaryPZero) {
    auto binom = BinomialDistribution::create(10, 0.0).unwrap();
    std::vector<double> values = {0.0, 1.0, 10.0};
    std::vector<double> prob(values.size());
    std::vector<double> log_prob(values.size());

    binom.getProbability(std::span<const double>(values), std::span<double>(prob));
    binom.getLogProbability(std::span<const double>(values), std::span<double>(log_prob));

    EXPECT_DOUBLE_EQ(prob[0], 1.0);
    EXPECT_DOUBLE_EQ(prob[1], 0.0);
    EXPECT_DOUBLE_EQ(prob[2], 0.0);
    EXPECT_DOUBLE_EQ(log_prob[0], 0.0);
    EXPECT_TRUE(std::isinf(log_prob[1]) && log_prob[1] < 0.0);
    EXPECT_TRUE(std::isinf(log_prob[2]) && log_prob[2] < 0.0);
    for (double x : prob)
        EXPECT_FALSE(std::isnan(x));
    for (double x : log_prob)
        EXPECT_FALSE(std::isnan(x));
}

TEST(BatchMathRegressions, BinomialBatchProbabilityHandlesBoundaryPOne) {
    auto binom = BinomialDistribution::create(10, 1.0).unwrap();
    std::vector<double> values = {0.0, 9.0, 10.0};
    std::vector<double> prob(values.size());
    std::vector<double> log_prob(values.size());

    binom.getProbability(std::span<const double>(values), std::span<double>(prob));
    binom.getLogProbability(std::span<const double>(values), std::span<double>(log_prob));

    EXPECT_DOUBLE_EQ(prob[0], 0.0);
    EXPECT_DOUBLE_EQ(prob[1], 0.0);
    EXPECT_DOUBLE_EQ(prob[2], 1.0);
    EXPECT_TRUE(std::isinf(log_prob[0]) && log_prob[0] < 0.0);
    EXPECT_TRUE(std::isinf(log_prob[1]) && log_prob[1] < 0.0);
    EXPECT_DOUBLE_EQ(log_prob[2], 0.0);
    for (double x : prob)
        EXPECT_FALSE(std::isnan(x));
    for (double x : log_prob)
        EXPECT_FALSE(std::isnan(x));
}

TEST(BatchMathRegressions, PoissonLargeLambdaQuantileFiniteAndOrdered) {
    auto poisson = PoissonDistribution::create(1'000'000.0).unwrap();
    const double q50 = poisson.getQuantile(0.50);
    const double q95 = poisson.getQuantile(0.95);

    EXPECT_TRUE(std::isfinite(q50));
    EXPECT_TRUE(std::isfinite(q95));
    EXPECT_GT(q50, 0.0);
    EXPECT_GT(q95, q50);
    EXPECT_GT(q50, 900'000.0);
    EXPECT_LT(q50, 1'100'000.0);
}

// TEST-8: fit() with degenerate data — regression guards for FIT-1 (Gamma NaN),
// FIT-4 (VonMises non-finite), FIT-5 (NegBin negative).

TEST(BatchMathRegressions, GammaFitRejectsNaN) {
    // FIT-1: NaN inputs silently passed `value <= 0` before this fix.
    GammaDistribution g;
    EXPECT_THROW(g.fit({1.0, std::numeric_limits<double>::quiet_NaN(), 2.0}),
                 std::invalid_argument);
    EXPECT_THROW(g.fit({1.0, std::numeric_limits<double>::infinity(), 2.0}), std::invalid_argument);
    EXPECT_THROW(g.fit({1.0, 0.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(g.fit({1.0, -1.0, 2.0}), std::invalid_argument);
}

TEST(BatchMathRegressions, VonMisesFitRejectsNonFinite) {
    // FIT-4: NaN/Inf inputs corrupted sin/cos accumulation silently.
    VonMisesDistribution vm;
    EXPECT_THROW(vm.fit({0.0, std::numeric_limits<double>::quiet_NaN(), 1.0}),
                 std::invalid_argument);
    EXPECT_THROW(vm.fit({0.0, std::numeric_limits<double>::infinity(), 1.0}),
                 std::invalid_argument);
}

TEST(BatchMathRegressions, NegativeBinomialFitRejectsNegative) {
    // FIT-5: negative inputs were silently discarded instead of throwing.
    NegativeBinomialDistribution nb;
    EXPECT_THROW(nb.fit({2.0, -1.0, 3.0}), std::invalid_argument);
    EXPECT_THROW(nb.fit({std::numeric_limits<double>::quiet_NaN(), 2.0}), std::invalid_argument);
}

TEST(BatchMathRegressions, VectorExpDeepUnderflowMatchesStdExp) {
    // SIMD exp kernels clamped input at -708.0, so every x < -708 returned
    // exp(-708) ~ 3.3e-308 — including x < -745.13 where the true result is 0 and
    // (-745.13, -708.4) where it is subnormal. The fix clamps at -746 with
    // two-step 2^n scaling; results must now track std::exp through the
    // subnormal range and flush to +0. The old single-step scaling also returned
    // inf for x >~ 709.44 (n = 1024 hit the inf exponent pattern); 709.7 guards
    // that region.
    const double points[] = {-708.5, -720.0, -745.0, -746.0, -800.0, -1e6,
                             -708.0, -700.0, -1.0,   0.5,    700.0,  709.7};
    // Repeat each value 8x so every SIMD width (2/4/8 doubles) hits the vector
    // kernel rather than the scalar remainder loop.
    std::vector<double> values;
    for (double p : points)
        values.insert(values.end(), 8, p);
    std::vector<double> out(values.size());

    arch::simd::VectorOps::vector_exp(values.data(), out.data(), values.size());

    for (std::size_t i = 0; i < values.size(); ++i) {
        const double ref = std::exp(values[i]);
        // Both results are >= +0, so the bit patterns are ordered and their
        // difference is the ULP distance — valid across the normal/subnormal
        // boundary, where a relative-error check would be meaningless.
        const auto ulp =
            std::abs(std::bit_cast<std::int64_t>(out[i]) - std::bit_cast<std::int64_t>(ref));
        EXPECT_LE(ulp, 4) << "x=" << values[i] << " simd=" << out[i] << " ref=" << ref;
        if (ref == 0.0) {
            EXPECT_FALSE(std::signbit(out[i])) << "x=" << values[i] << " returned -0";
        }
    }
}

TEST(BatchMathRegressions, VectorExpEdgeCasesMatchStdExp) {
    // Companion to VectorExpDeepUnderflowMatchesStdExp. The SIMD exp kernels
    // clamp inputs into [exp_min, exp_max] for the finite path, so non-finite
    // and overflow inputs need an explicit fixup to track std::exp. Guards the
    // 2026-07-19 AVX2/AVX audit findings: +inf/overflow -> +inf, NaN -> NaN,
    // -inf -> +0, and the SIMD body must agree with the scalar remainder loop
    // (also std::exp). Before the fix these lanes returned ~DBL_MAX (a finite
    // value), silently swallowing NaN and +inf.
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double points[] = {inf, -inf, nan, 710.0, 720.0, 1.0e6, -1.0, 2.5, -37.0, 0.0};
    // Repeat each value 8x so every SIMD width (2/4/8 doubles) hits the vector
    // kernel rather than the scalar remainder loop.
    std::vector<double> values;
    for (double p : points)
        values.insert(values.end(), 8, p);
    std::vector<double> out(values.size());

    arch::simd::VectorOps::vector_exp(values.data(), out.data(), values.size());

    for (std::size_t i = 0; i < values.size(); ++i) {
        const double x = values[i];
        const double ref = std::exp(x);
        if (std::isnan(ref)) {
            EXPECT_TRUE(std::isnan(out[i])) << "x=" << x << " simd=" << out[i];
        } else if (std::isinf(ref)) {
            EXPECT_TRUE(std::isinf(out[i]) && out[i] > 0.0) << "x=" << x << " simd=" << out[i];
        } else {
            // Finite: exp(x) >= +0, so bit patterns are ordered and their
            // difference is the ULP distance.
            const auto ulp =
                std::abs(std::bit_cast<std::int64_t>(out[i]) - std::bit_cast<std::int64_t>(ref));
            EXPECT_LE(ulp, 4) << "x=" << x << " simd=" << out[i] << " ref=" << ref;
            if (ref == 0.0)
                EXPECT_FALSE(std::signbit(out[i])) << "x=" << x << " returned -0";
        }
    }
}

TEST(BatchMathRegressions, VectorLogSubnormalMatchesStdLog) {
    // vector_log_avx/avx2/avx512 scale subnormal inputs by 2^54 before the
    // bit-level exponent/mantissa extraction (subnormals have no implicit
    // leading mantissa bit, so the trick is only valid for normalized
    // doubles). vector_log_sse2 was missing this scaling entirely, so
    // log(subnormal) was silently wrong by up to ~35 natural-log-units (e.g.
    // log(denorm_min()) returned -709.09 instead of -744.44). Found
    // 2026-07-19 via native SSE2 testing (LIBSTATS_MAX_SIMD_TIER=SSE2), which
    // surfaced this for the first time -- SSE2 had previously only run under
    // Rosetta 2. Guards the fix across whichever tier is dispatched.
    const double points[] = {
        std::numeric_limits<double>::denorm_min(),  // smallest subnormal, ~4.94e-324
        1.0e-310,                                   // subnormal
        1.0e-320,                                   // subnormal
        std::numeric_limits<double>::min(),         // smallest normal, ~2.22e-308 (boundary)
        1.0,
        1.0e100,
        std::numeric_limits<double>::max(),
    };
    // Repeat each value 8x so every SIMD width (2/4/8 doubles) hits the vector
    // kernel rather than the scalar remainder loop.
    std::vector<double> values;
    for (double p : points)
        values.insert(values.end(), 8, p);
    std::vector<double> out(values.size());

    arch::simd::VectorOps::vector_log(values.data(), out.data(), values.size());

    for (std::size_t i = 0; i < values.size(); ++i) {
        const double ref = std::log(values[i]);
        EXPECT_NEAR(out[i], ref, std::abs(ref) * 1e-9 + 1e-12)
            << "x=" << values[i] << " simd=" << out[i] << " ref=" << ref;
    }
}

TEST(BatchMathRegressions, GaussianStandardizedValueRebuildsCacheAfterReset) {
    auto gaussian = GaussianDistribution::create(10.0, 2.0).unwrap();
    EXPECT_NEAR(gaussian.getStandardizedValue(14.0), 2.0, 1e-15);

    gaussian.reset();
    // reset() invalidates cache; getStandardizedValue must rebuild before reading
    // invStandardDeviation_.
    EXPECT_NEAR(gaussian.getStandardizedValue(2.0), 2.0, 1e-15);
}
