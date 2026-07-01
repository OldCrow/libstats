/**
 * @file test_poisson_analysis.cpp
 * @brief Tests for stats::analysis::poisson:: functions (v2.0.0).
 */

#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#include "include/tests.h"
#include "libstats/distributions/poisson.h"
#include "libstats/stats/analysis/poisson_analysis.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace stats;

// EXPECT_THROW on [[nodiscard]] functions is intentional: the function throws
// before returning, so discarding the return value is correct.
// cppcheck-suppress unusedResult
#ifdef _MSC_VER
#  pragma warning(disable: 4834)  // discarding return value of [[nodiscard]] function (intentional in EXPECT_THROW)
#else
#  pragma GCC diagnostic ignored "-Wunused-result"
#endif

static std::vector<double> poissonSample(std::size_t n, double lambda, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::poisson_distribution<int> d(lambda);
    std::vector<double> v(n);
    for (auto& x : v)
        x = static_cast<double>(d(rng));
    return v;
}

// ── Confidence interval for rate ────────────────────────────────────────────

TEST(PoissonAnalysis, ConfidenceIntervalRateOrdered) {
    auto data = poissonSample(200, 3.0);
    auto [lo, hi] = stats::analysis::poisson::confidenceIntervalRate(data, 0.95);
    EXPECT_LT(lo, hi);
    EXPECT_GT(lo, 0.0);
    EXPECT_TRUE(std::isfinite(lo));
    EXPECT_TRUE(std::isfinite(hi));
    // True rate 3.0 should lie inside with high probability
    EXPECT_LT(lo, 3.5);
    EXPECT_GT(hi, 2.5);
}

TEST(PoissonAnalysis, ConfidenceIntervalRateAllZerosLowerIsZero) {
    std::vector<double> zeros(50, 0.0);
    auto [lo, hi] = stats::analysis::poisson::confidenceIntervalRate(zeros, 0.95);
    EXPECT_DOUBLE_EQ(lo, 0.0);
    EXPECT_GT(hi, 0.0);
}

TEST(PoissonAnalysis, ConfidenceIntervalRateEmptyThrows) {
    EXPECT_THROW(stats::analysis::poisson::confidenceIntervalRate({}), std::invalid_argument);
}

// ── Overdispersion test ──────────────────────────────────────────────────────

TEST(PoissonAnalysis, OverdispersionTestPoissonData) {
    auto data = poissonSample(200, 5.0, 11);
    auto [d_index, p, is_over] = stats::analysis::poisson::overdispersionTest(data);
    EXPECT_GT(d_index, 0.0);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_FALSE(is_over);  // Poisson data should not be flagged as overdispersed
}

TEST(PoissonAnalysis, OverdispersionTestHighVarianceRejected) {
    // Mix of Poisson with very high variance (near-negative-binomial)
    std::vector<double> data(200, 0.0);
    for (std::size_t i = 0; i < 100; ++i)
        data[i] = 0.0;
    for (std::size_t i = 100; i < 200; ++i)
        data[i] = 20.0;  // high variance
    auto [d_index, p, is_over] = stats::analysis::poisson::overdispersionTest(data);
    EXPECT_TRUE(is_over);
}

TEST(PoissonAnalysis, OverdispersionTestTooFewThrows) {
    EXPECT_THROW(stats::analysis::poisson::overdispersionTest({1.0}), std::invalid_argument);
}

// ── Excess zeros test ────────────────────────────────────────────────────────

TEST(PoissonAnalysis, ExcessZerosTestPoissonData) {
    auto data = poissonSample(300, 4.0, 22);
    auto [z, p, excess] = stats::analysis::poisson::excessZerosTest(data);
    EXPECT_TRUE(std::isfinite(z));
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_FALSE(excess);
}

TEST(PoissonAnalysis, ExcessZerosTestDetectsInflation) {
    // 60% zeros in data where Poisson(2) predicts ~13.5%
    std::vector<double> data(200, 0.0);
    for (std::size_t i = 80; i < 200; ++i)
        data[i] = 2.0;
    auto [z, p, excess] = stats::analysis::poisson::excessZerosTest(data, 0.01);
    EXPECT_TRUE(excess);
}

// ── Rate stability test ──────────────────────────────────────────────────────

TEST(PoissonAnalysis, RateStabilityTestStableData) {
    auto data = poissonSample(100, 3.0, 33);
    auto [t, p, stable] = stats::analysis::poisson::rateStabilityTest(data);
    EXPECT_TRUE(std::isfinite(t));
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_TRUE(stable);
}

TEST(PoissonAnalysis, RateStabilityTestTooFewThrows) {
    EXPECT_THROW(stats::analysis::poisson::rateStabilityTest({1.0, 2.0}), std::invalid_argument);
}

// ── Chi-square GoF test ──────────────────────────────────────────────────────

TEST(PoissonAnalysis, ChiSquareGoodFit) {
    auto data = poissonSample(300, 3.0, 44);
    auto dist = PoissonDistribution::create(3.0).unwrap();
    auto [chi2, p, reject] = stats::analysis::poisson::chiSquareGoodnessOfFit(data, dist);
    EXPECT_GE(chi2, 0.0);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_FALSE(reject);
}

TEST(PoissonAnalysis, ChiSquareBadFit) {
    // Normal data, tested against Poisson(3) — should reject
    std::mt19937 rng(55);
    std::normal_distribution<double> nd(50.0, 5.0);
    std::vector<double> data(300);
    for (auto& x : data)
        x = std::max(0.0, nd(rng));
    auto dist = PoissonDistribution::create(3.0).unwrap();
    auto [chi2, p, reject] = stats::analysis::poisson::chiSquareGoodnessOfFit(data, dist);
    EXPECT_TRUE(reject);
}

TEST(PoissonAnalysis, ChiSquareTooFewThrows) {
    auto dist = PoissonDistribution::create(2.0).unwrap();
    EXPECT_THROW(stats::analysis::poisson::chiSquareGoodnessOfFit({1.0, 2.0}, dist),
                 std::invalid_argument);
}
