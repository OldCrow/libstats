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

#include <cmath>
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

TEST(BatchMathRegressions, GaussianStandardizedValueRebuildsCacheAfterReset) {
    auto gaussian = GaussianDistribution::create(10.0, 2.0).unwrap();
    EXPECT_NEAR(gaussian.getStandardizedValue(14.0), 2.0, 1e-15);

    gaussian.reset();
    // reset() invalidates cache; getStandardizedValue must rebuild before reading
    // invStandardDeviation_.
    EXPECT_NEAR(gaussian.getStandardizedValue(2.0), 2.0, 1e-15);
}
