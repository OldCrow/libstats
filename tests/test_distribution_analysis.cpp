/**
 * @file test_distribution_analysis.cpp
 * @brief Tests for stats::analysis::exponential, gamma, and binomial namespaces (v2.0.0).
 */

#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#include "include/tests.h"
#include "libstats/stats/analysis/binomial_analysis.h"
#include "libstats/stats/analysis/exponential_analysis.h"
#include "libstats/stats/analysis/gamma_analysis.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <vector>

// EXPECT_THROW on [[nodiscard]] functions is intentional: the function throws
// before returning, so discarding the return value is correct.
// cppcheck-suppress unusedResult
#ifdef _MSC_VER
    #pragma warning(disable : 4834)  // discarding return value of [[nodiscard]] function
                                     // (intentional in EXPECT_THROW)
#else
    #pragma GCC diagnostic ignored "-Wunused-result"
#endif

// ── Exponential analysis ─────────────────────────────────────────────────────

static std::vector<double> expSample(std::size_t n, double lambda, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::exponential_distribution<double> d(lambda);
    std::vector<double> v(n);
    for (auto& x : v)
        x = d(rng);
    return v;
}

TEST(ExponentialAnalysis, ConfidenceIntervalRateOrdered) {
    auto data = expSample(200, 2.0);
    auto [lo, hi] = stats::analysis::exponential::confidenceIntervalRate(data, 0.95);
    EXPECT_LT(lo, hi);
    EXPECT_GT(lo, 0.0);
    EXPECT_TRUE(std::isfinite(lo));
    EXPECT_TRUE(std::isfinite(hi));
    EXPECT_LT(lo, 2.5);
    EXPECT_GT(hi, 1.5);
}

TEST(ExponentialAnalysis, ConfidenceIntervalRateEmptyThrows) {
    EXPECT_THROW(stats::analysis::exponential::confidenceIntervalRate({}), std::invalid_argument);
}

TEST(ExponentialAnalysis, CVTestAcceptsExponential) {
    auto data = expSample(200, 1.5, 11);
    auto [stat, p, reject] = stats::analysis::exponential::coefficientOfVariationTest(data);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_TRUE(std::isfinite(stat));
    EXPECT_FALSE(reject);
}

TEST(ExponentialAnalysis, CVTestRejectsNormal) {
    std::mt19937 rng(22);
    std::normal_distribution<double> nd(5.0, 0.5);  // CV ≈ 0.1, far from 1
    std::vector<double> data(200);
    for (auto& x : data)
        x = std::abs(nd(rng)) + 0.01;
    auto [stat, p, reject] = stats::analysis::exponential::coefficientOfVariationTest(data);
    EXPECT_TRUE(reject);
}

TEST(ExponentialAnalysis, CVTestTooFewThrows) {
    EXPECT_THROW(stats::analysis::exponential::coefficientOfVariationTest({1.0}),
                 std::invalid_argument);
}

// ── Gamma analysis ───────────────────────────────────────────────────────────

TEST(GammaAnalysis, NormalApproximationTestValidForLargeAlpha) {
    // Gamma(100, 2): shape=100 >> 30, so normal approximation is valid
    std::mt19937 rng(33);
    std::gamma_distribution<double> gd(100.0, 0.5);  // shape=100, scale=0.5 → rate=2
    std::vector<double> data(500);
    for (auto& x : data)
        x = gd(rng);

    auto [lo, hi, valid] = stats::analysis::gamma::normalApproximationTest(data);
    EXPECT_LT(lo, hi);  // CI is ordered
    EXPECT_TRUE(std::isfinite(lo));
    EXPECT_TRUE(std::isfinite(hi));
    EXPECT_GT(lo, 0.0);  // CI is in positive range (data mean > 0)
    EXPECT_TRUE(valid);  // alpha_hat >= 30 and mean inside the normal-approx CI
}

TEST(GammaAnalysis, NormalApproximationTestInvalidForSmallAlpha) {
    // Gamma(2, 1): shape=2 < 30, so normal approximation is NOT considered valid
    std::mt19937 rng(44);
    std::gamma_distribution<double> gd(2.0, 1.0);
    std::vector<double> data(200);
    for (auto& x : data)
        x = gd(rng);

    auto [lo, hi, valid] = stats::analysis::gamma::normalApproximationTest(data);
    EXPECT_TRUE(std::isfinite(lo));
    EXPECT_TRUE(std::isfinite(hi));
    EXPECT_FALSE(valid);  // alpha_hat ≈ 2, far below the threshold of 30
}

TEST(GammaAnalysis, NormalApproximationTestEmptyThrows) {
    EXPECT_THROW(stats::analysis::gamma::normalApproximationTest({}), std::invalid_argument);
}

// ── Binomial analysis ─────────────────────────────────────────────────────────

TEST(BinomialAnalysis, ClopperPearsonOrdered) {
    auto [lo, hi] = stats::analysis::binomial::clopperPearsonCI(30, 100, 0.95);
    EXPECT_LT(lo, hi);
    EXPECT_GT(lo, 0.0);
    EXPECT_LT(hi, 1.0);
    EXPECT_TRUE(std::isfinite(lo));
    EXPECT_TRUE(std::isfinite(hi));
    // True p=0.3; CI should contain 0.3
    EXPECT_LT(lo, 0.3);
    EXPECT_GT(hi, 0.3);
}

TEST(BinomialAnalysis, ClopperPearsonZeroSuccesses) {
    auto [lo, hi] = stats::analysis::binomial::clopperPearsonCI(0, 50, 0.95);
    EXPECT_DOUBLE_EQ(lo, 0.0);
    EXPECT_GT(hi, 0.0);
}

TEST(BinomialAnalysis, ClopperPearsonAllSuccesses) {
    auto [lo, hi] = stats::analysis::binomial::clopperPearsonCI(50, 50, 0.95);
    EXPECT_GT(lo, 0.0);
    EXPECT_DOUBLE_EQ(hi, 1.0);
}

TEST(BinomialAnalysis, ClopperPearsonBadInputThrows) {
    EXPECT_THROW(stats::analysis::binomial::clopperPearsonCI(-1, 10, 0.95), std::invalid_argument);
    EXPECT_THROW(stats::analysis::binomial::clopperPearsonCI(11, 10, 0.95), std::invalid_argument);
    EXPECT_THROW(stats::analysis::binomial::clopperPearsonCI(5, 0, 0.95), std::invalid_argument);
}

TEST(BinomialAnalysis, ProportionZTestAcceptsTrueProportion) {
    // k=50 out of n=200 → p̂=0.25; test against p₀=0.25
    auto [z, p, reject] = stats::analysis::binomial::proportionZTest(50, 200, 0.25);
    EXPECT_NEAR(z, 0.0, 0.1);
    EXPECT_FALSE(reject);
}

TEST(BinomialAnalysis, ProportionZTestRejectsWrong) {
    // k=95 out of n=100 → p̂=0.95; test against p₀=0.5 — should strongly reject
    auto [z, p, reject] = stats::analysis::binomial::proportionZTest(95, 100, 0.5);
    EXPECT_TRUE(reject);
    EXPECT_GT(std::abs(z), 5.0);
}

TEST(BinomialAnalysis, TwoProportionZTestSame) {
    // Both groups have identical proportions
    auto [z, p, reject] = stats::analysis::binomial::twoProportionZTest(50, 200, 50, 200);
    EXPECT_NEAR(z, 0.0, 1e-10);
    EXPECT_FALSE(reject);
}

TEST(BinomialAnalysis, TwoProportionZTestDifferent) {
    // Group 1: 90/100, Group 2: 10/100 — large difference
    auto [z, p, reject] = stats::analysis::binomial::twoProportionZTest(90, 100, 10, 100);
    EXPECT_TRUE(reject);
    EXPECT_GT(std::abs(z), 10.0);
}

TEST(BinomialAnalysis, TwoProportionZTestBadInputThrows) {
    EXPECT_THROW(stats::analysis::binomial::twoProportionZTest(5, 0, 5, 10), std::invalid_argument);
    EXPECT_THROW(stats::analysis::binomial::twoProportionZTest(5, 10, 11, 10),
                 std::invalid_argument);
}
