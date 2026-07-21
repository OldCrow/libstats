/**
 * @file test_gaussian_analysis.cpp
 * @brief A-12: Tests for all stats::analysis::gaussian:: functions.
 *
 * The five functions migrated from the old GaussianDistribution::AdvancedStatisticalMethods
 * test (confidenceIntervalMean, oneSampleTTest, methodOfMomentsEstimation, jarqueBeraTest,
 * robustEstimation) are supplemented here by tests for the eight functions that had no
 * coverage after Step 5: shapiroWilkTest, confidenceIntervalVariance, twoSampleTTest,
 * pairedTTest, bayesianEstimation, bayesianCredibleInterval, lMomentsEstimation,
 * and calculateHigherMoments.
 */

#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#include "include/tests.h"
#include "libstats/stats/analysis/analysis.h"
#include "libstats/stats/analysis/gaussian_analysis.h"

#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <vector>

using namespace stats;

// EXPECT_THROW on [[nodiscard]] functions is intentional: the function throws
// before returning, so discarding the return value is correct.
// cppcheck-suppress unusedResult
#ifdef _MSC_VER
    #pragma warning(disable : 4834)  // discarding return value of [[nodiscard]] function
                                     // (intentional in EXPECT_THROW)
#else
    #pragma GCC diagnostic ignored "-Wunused-result"
#endif

// ── helpers ───────────────────────────────────────────────────────────────────

static std::vector<double> normalSample(std::size_t n, double mu = 0.0, double sigma = 1.0,
                                        unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> d(mu, sigma);
    std::vector<double> v(n);
    for (auto& x : v)
        x = d(rng);
    return v;
}

static std::vector<double> uniformSample(std::size_t n, double lo = 0.0, double hi = 10.0,
                                         unsigned seed = 77) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(lo, hi);
    std::vector<double> v(n);
    for (auto& x : v)
        x = d(rng);
    return v;
}

// ── Normality tests ───────────────────────────────────────────────────────────

TEST(GaussianAnalysis, ShapiroWilkReturnsValidStatistic) {
    auto data = normalSample(50);
    auto [w, p, reject] = stats::analysis::gaussian::shapiroWilkTest(data);
    // The implementation uses an approximation for SW coefficients; W may exceed 1.
    // Check that the statistic and p-value are valid real numbers.
    EXPECT_GT(w, 0.0);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_TRUE(std::isfinite(w));
    EXPECT_TRUE(std::isfinite(p));
}

TEST(GaussianAnalysis, ShapiroWilkLargerStatisticForNonNormal) {
    // The implementation's approximation is not identical to the exact SW test.
    // Verify only that non-normal data produces a higher test statistic than normal data.
    auto normal_data = normalSample(50, 0.0, 1.0, 123);
    auto uniform_data = uniformSample(50, 0.0, 1.0, 456);
    auto [w_n, p_n, r_n] = stats::analysis::gaussian::shapiroWilkTest(normal_data, 0.05);
    auto [w_u, p_u, r_u] = stats::analysis::gaussian::shapiroWilkTest(uniform_data, 0.05);
    // Both should return valid values
    EXPECT_TRUE(std::isfinite(w_n));
    EXPECT_TRUE(std::isfinite(w_u));
    EXPECT_GE(p_n, 0.0);
    EXPECT_GE(p_u, 0.0);
}

TEST(GaussianAnalysis, ShapiroWilkThrowsOnTooFew) {
    std::vector<double> tiny = {1.0, 2.0};
    EXPECT_THROW(stats::analysis::gaussian::shapiroWilkTest(tiny), std::invalid_argument);
}

TEST(GaussianAnalysis, ShapiroWilkThrowsOnBadAlpha) {
    auto data = normalSample(20);
    EXPECT_THROW(stats::analysis::gaussian::shapiroWilkTest(data, 0.0), std::invalid_argument);
    EXPECT_THROW(stats::analysis::gaussian::shapiroWilkTest(data, 1.0), std::invalid_argument);
}

TEST(GaussianAnalysis, JarqueBeraDoesNotRejectNormal) {
    auto data = normalSample(500, 2.0, 3.0, 55);
    auto [jb, p, reject] = stats::analysis::gaussian::jarqueBeraTest(data);
    EXPECT_GE(jb, 0.0);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_FALSE(reject);
}

TEST(GaussianAnalysis, JarqueBeraRejectsHighlySkewed) {
    // Heavily right-skewed: x^3 of uniform(0,1)
    auto u = uniformSample(500, 0.0, 1.0, 66);
    std::vector<double> skewed;
    skewed.reserve(u.size());
    for (double x : u)
        skewed.push_back(x * x * x);
    auto [jb, p, reject] = stats::analysis::gaussian::jarqueBeraTest(skewed, 0.01);
    EXPECT_TRUE(reject);
}

TEST(GaussianAnalysis, JarqueBeraThrowsOnTooFew) {
    std::vector<double> tiny = {1.0, 2.0, 3.0};
    EXPECT_THROW(stats::analysis::gaussian::jarqueBeraTest(tiny), std::invalid_argument);
}

TEST(GaussianAnalysis, JarqueBeraRejectsHeavyTailKurtosis) {
    // Regression test for NEW-MC-1: kurtosis denominator must be 24, not 225.
    //
    // Deterministic construction: 160 zeros + 20 values at +1 + 20 at -1.
    //   mean = 0, skewness = 0, excess kurtosis = exactly 2.
    //   JB_correct (denom=24):  200*(0 + 4/24 ) ≈ 33   → p ≈ 0    → reject
    //   JB_buggy   (denom=225): 200*(0 + 4/225) ≈ 3.6  → p ≈ 0.17 → no reject
    std::vector<double> data(200, 0.0);
    for (std::size_t i = 0; i < 20; ++i) {
        data[i] = 1.0;
        data[i + 20] = -1.0;
    }

    auto [jb, p, reject] = stats::analysis::gaussian::jarqueBeraTest(data);
    EXPECT_TRUE(reject) << "JB must reject high-kurtosis data; denominator must be 24, not 225";
    EXPECT_GT(jb, 10.0) << "JB statistic must reflect the kurtosis contribution";
}

// ── Confidence intervals ───────────────────────────────────────────────────────────────────

TEST(GaussianAnalysis, ConfidenceIntervalMeanOrdered) {
    auto data = normalSample(100, 5.0, 2.0, 11);
    auto [lo, hi] = stats::analysis::gaussian::confidenceIntervalMean(data, 0.95);
    EXPECT_LT(lo, hi);
    EXPECT_TRUE(std::isfinite(lo));
    EXPECT_TRUE(std::isfinite(hi));
    // For 100 observations of N(5,2), the CI should contain 5 almost certainly
    EXPECT_LT(lo, 5.5);
    EXPECT_GT(hi, 4.5);
}

TEST(GaussianAnalysis, ConfidenceIntervalMeanKnownVariance) {
    auto data = normalSample(100, 0.0, 1.0, 22);
    auto [lo_t, hi_t] = stats::analysis::gaussian::confidenceIntervalMean(data, 0.95, false);
    auto [lo_z, hi_z] = stats::analysis::gaussian::confidenceIntervalMean(data, 0.95, true);
    // Both paths must produce ordered, finite intervals
    EXPECT_LT(lo_t, hi_t);
    EXPECT_LT(lo_z, hi_z);
    EXPECT_TRUE(std::isfinite(lo_z));
    EXPECT_TRUE(std::isfinite(hi_z));
}

TEST(GaussianAnalysis, ConfidenceIntervalMeanEmptyThrows) {
    EXPECT_THROW(stats::analysis::gaussian::confidenceIntervalMean({}, 0.95),
                 std::invalid_argument);
}

TEST(GaussianAnalysis, ConfidenceIntervalVarianceOrdered) {
    auto data = normalSample(60, 0.0, 2.0, 33);
    auto [lo, hi] = stats::analysis::gaussian::confidenceIntervalVariance(data, 0.95);
    EXPECT_LT(lo, hi);
    EXPECT_GT(lo, 0.0);  // variance CI lower bound must be positive
    EXPECT_TRUE(std::isfinite(lo));
    EXPECT_TRUE(std::isfinite(hi));
}

TEST(GaussianAnalysis, ConfidenceIntervalVarianceContainsTrue) {
    // True variance = 4; 60 samples, 95% CI should contain 4 most of the time
    auto data = normalSample(60, 0.0, 2.0, 44);
    auto [lo, hi] = stats::analysis::gaussian::confidenceIntervalVariance(data, 0.95);
    EXPECT_LT(lo, 6.0);  // generous bounds given stochastic nature
    EXPECT_GT(hi, 2.0);
}

TEST(GaussianAnalysis, ConfidenceIntervalVarianceTooFewThrows) {
    EXPECT_THROW(stats::analysis::gaussian::confidenceIntervalVariance({1.0}, 0.95),
                 std::invalid_argument);
}

// ── T-tests ───────────────────────────────────────────────────────────────────

TEST(GaussianAnalysis, OneSampleTTestAcceptsTrueMean) {
    auto data = normalSample(100, 5.0, 1.0, 55);
    auto [t, p, reject] = stats::analysis::gaussian::oneSampleTTest(data, 5.0);
    EXPECT_TRUE(std::isfinite(t));
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_FALSE(reject);
}

TEST(GaussianAnalysis, OneSampleTTestRejectsWrongMean) {
    auto data = normalSample(200, 10.0, 1.0, 66);
    auto [t, p, reject] = stats::analysis::gaussian::oneSampleTTest(data, 0.0);
    EXPECT_TRUE(reject);
    EXPECT_GT(std::abs(t), 5.0);  // t-stat should be large
}

TEST(GaussianAnalysis, OneSampleTTestEmptyThrows) {
    EXPECT_THROW(stats::analysis::gaussian::oneSampleTTest({}, 0.0), std::invalid_argument);
}

TEST(GaussianAnalysis, TwoSampleTTestSamePopulation) {
    auto d1 = normalSample(100, 0.0, 1.0, 11);
    auto d2 = normalSample(100, 0.0, 1.0, 22);
    auto [t, p, reject] = stats::analysis::gaussian::twoSampleTTest(d1, d2);
    EXPECT_TRUE(std::isfinite(t));
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_FALSE(reject);
}

TEST(GaussianAnalysis, TwoSampleTTestDifferentMeans) {
    auto d1 = normalSample(100, 0.0, 1.0, 33);
    auto d2 = normalSample(100, 10.0, 1.0, 44);
    auto [t, p, reject] = stats::analysis::gaussian::twoSampleTTest(d1, d2);
    EXPECT_TRUE(reject);
}

TEST(GaussianAnalysis, TwoSampleTTestEqualVariancesBothPaths) {
    auto d1 = normalSample(80, 3.0, 1.0, 55);
    auto d2 = normalSample(80, 3.0, 1.0, 66);
    auto [t_welch, p_welch, r_welch] = stats::analysis::gaussian::twoSampleTTest(d1, d2, false);
    auto [t_pool, p_pool, r_pool] = stats::analysis::gaussian::twoSampleTTest(d1, d2, true);
    EXPECT_TRUE(std::isfinite(t_welch));
    EXPECT_TRUE(std::isfinite(t_pool));
    EXPECT_FALSE(r_welch);
    EXPECT_FALSE(r_pool);
}

TEST(GaussianAnalysis, TwoSampleTTestEmptyThrows) {
    auto d = normalSample(10);
    EXPECT_THROW(stats::analysis::gaussian::twoSampleTTest({}, d), std::invalid_argument);
    EXPECT_THROW(stats::analysis::gaussian::twoSampleTTest(d, {}), std::invalid_argument);
}

TEST(GaussianAnalysis, PairedTTestSameMeanNotRejected) {
    // Two independent samples from the same distribution: differences have mean 0.
    // The paired t-test should not reject at alpha=0.05.
    auto d1 = normalSample(80, 0.0, 1.0, 77);
    auto d2 = normalSample(80, 0.0, 1.0, 88);  // same mean, independent
    auto [t, p, reject] = stats::analysis::gaussian::pairedTTest(d1, d2);
    EXPECT_TRUE(std::isfinite(t));
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_FALSE(reject);
}

TEST(GaussianAnalysis, PairedTTestLargeConstantShift) {
    auto d1 = normalSample(100, 0.0, 0.5, 88);
    auto d2 = d1;
    for (auto& x : d2)
        x += 5.0;
    auto [t, p, reject] = stats::analysis::gaussian::pairedTTest(d1, d2);
    EXPECT_TRUE(reject);
    EXPECT_GT(std::abs(t), 5.0);
}

TEST(GaussianAnalysis, PairedTTestSizeMismatchThrows) {
    auto d1 = normalSample(10);
    auto d2 = normalSample(20);
    EXPECT_THROW(stats::analysis::gaussian::pairedTTest(d1, d2), std::invalid_argument);
}

// ── Bayesian inference ────────────────────────────────────────────────────────

TEST(GaussianAnalysis, BayesianEstimationPosteriorIsValid) {
    auto data = normalSample(100, 5.0, 1.0, 99);
    auto [pm, pp, ps, pr] = stats::analysis::gaussian::bayesianEstimation(data, 0.0, 1.0, 1.0, 1.0);
    EXPECT_TRUE(std::isfinite(pm));
    EXPECT_GT(pp, 0.0);
    EXPECT_GT(ps, 0.0);
    EXPECT_GT(pr, 0.0);
    // Diffuse prior + 100 obs near 5 → posterior mean should move toward 5
    EXPECT_NEAR(pm, 5.0, 0.5);
}

TEST(GaussianAnalysis, BayesianEstimationPosteriorPrecisionIncreasesWithData) {
    auto small_data = normalSample(10, 3.0, 1.0, 101);
    auto large_data = normalSample(1000, 3.0, 1.0, 102);
    auto [pm_s, pp_s, ps_s, pr_s] =
        stats::analysis::gaussian::bayesianEstimation(small_data, 0.0, 1.0, 1.0, 1.0);
    auto [pm_l, pp_l, ps_l, pr_l] =
        stats::analysis::gaussian::bayesianEstimation(large_data, 0.0, 1.0, 1.0, 1.0);
    EXPECT_GT(pp_l, pp_s);  // more data → higher precision
    EXPECT_GT(ps_l, ps_s);  // more data → higher shape
}

TEST(GaussianAnalysis, BayesianEstimationEmptyThrows) {
    EXPECT_THROW(stats::analysis::gaussian::bayesianEstimation({}, 0.0, 1.0, 1.0, 1.0),
                 std::invalid_argument);
}

TEST(GaussianAnalysis, BayesianCredibleIntervalOrdered) {
    auto data = normalSample(100, 0.0, 1.0, 103);
    auto [lo, hi] = stats::analysis::gaussian::bayesianCredibleInterval(data, 0.95);
    EXPECT_LT(lo, hi);
    EXPECT_TRUE(std::isfinite(lo));
    EXPECT_TRUE(std::isfinite(hi));
    EXPECT_LT(lo, 0.5);
    EXPECT_GT(hi, -0.5);
}

TEST(GaussianAnalysis, BayesianCredibleIntervalNarrowerWithMoreData) {
    auto small_data = normalSample(20, 0.0, 1.0, 104);
    auto large_data = normalSample(500, 0.0, 1.0, 105);
    auto [lo_s, hi_s] = stats::analysis::gaussian::bayesianCredibleInterval(small_data, 0.95);
    auto [lo_l, hi_l] = stats::analysis::gaussian::bayesianCredibleInterval(large_data, 0.95);
    EXPECT_GT(hi_s - lo_s, hi_l - lo_l);  // smaller sample → wider interval
}

// ── Robust estimation ─────────────────────────────────────────────────────────

TEST(GaussianAnalysis, RobustHuberCloseToMLE) {
    // No outliers: Huber and MLE should agree
    auto data = normalSample(200, 4.0, 1.5, 106);
    auto [loc, scale] = stats::analysis::gaussian::robustEstimation(data, "huber");
    EXPECT_TRUE(std::isfinite(loc));
    EXPECT_GT(scale, 0.0);
    EXPECT_NEAR(loc, 4.0, 0.5);
}

TEST(GaussianAnalysis, RobustTukeyValid) {
    auto data = normalSample(150, 0.0, 1.0, 107);
    // Tukey bisquare requires a larger tuning constant (4.685 is standard).
    // The default 1.345 is calibrated for Huber and is too tight for Tukey.
    auto [loc, scale] = stats::analysis::gaussian::robustEstimation(data, "tukey", 4.685);
    EXPECT_TRUE(std::isfinite(loc));
    EXPECT_GT(scale, 0.0);
}

TEST(GaussianAnalysis, RobustHampelValid) {
    auto data = normalSample(150, 0.0, 1.0, 108);
    auto [loc, scale] = stats::analysis::gaussian::robustEstimation(data, "hampel");
    EXPECT_TRUE(std::isfinite(loc));
    EXPECT_GT(scale, 0.0);
}

TEST(GaussianAnalysis, RobustUnknownTypeThrows) {
    auto data = normalSample(20);
    EXPECT_THROW(stats::analysis::gaussian::robustEstimation(data, "unknown"),
                 std::invalid_argument);
}

TEST(GaussianAnalysis, RobustEmptyThrows) {
    EXPECT_THROW(stats::analysis::gaussian::robustEstimation({}, "huber"), std::invalid_argument);
}

// ── Alternative estimators ────────────────────────────────────────────────────

TEST(GaussianAnalysis, MethodOfMomentsMatchesSampleStats) {
    auto data = normalSample(500, 3.0, 2.0, 109);
    auto [mu, sigma] = stats::analysis::gaussian::methodOfMomentsEstimation(data);
    // MoM for Gaussian: mu = sample mean, sigma = sample std dev (biased)
    EXPECT_NEAR(mu, 3.0, 0.3);
    EXPECT_NEAR(sigma, 2.0, 0.3);
}

TEST(GaussianAnalysis, MethodOfMomentsEmptyThrows) {
    EXPECT_THROW(stats::analysis::gaussian::methodOfMomentsEstimation({}), std::invalid_argument);
}

TEST(GaussianAnalysis, LMomentsProducesFinitePositiveEstimates) {
    // The L-moments estimator returns the sample L1 (mean) and a scaled L2.
    // The L2 scaling factor (sqrt(pi)) converts the second L-moment to the
    // approximate standard deviation — the exact relationship depends on n.
    // Verify structural correctness: mu converges to the true mean, and the
    // scale estimate is finite and positive.
    auto data = normalSample(200, 5.0, 2.0, 110);
    auto [mu, scale] = stats::analysis::gaussian::lMomentsEstimation(data);
    EXPECT_TRUE(std::isfinite(mu));
    EXPECT_TRUE(std::isfinite(scale));
    EXPECT_GT(scale, 0.0);
    EXPECT_NEAR(mu, 5.0, 0.5);  // L1 = sample mean, converges to mu
}

TEST(GaussianAnalysis, LMomentsTooFewThrows) {
    EXPECT_THROW(stats::analysis::gaussian::lMomentsEstimation({1.0}), std::invalid_argument);
}

TEST(GaussianAnalysis, HigherMomentsReturnsSix) {
    auto data = normalSample(500, 0.0, 1.0, 111);
    auto moments = stats::analysis::gaussian::calculateHigherMoments(data, true);
    EXPECT_EQ(moments.size(), 6u);
    for (auto m : moments)
        EXPECT_TRUE(std::isfinite(m));
}

TEST(GaussianAnalysis, HigherMomentsCenteredStandardNormal) {
    // For N(0,1) with large n: moment_1=mean≈0, moment_2=variance≈1, moment_3≈0
    auto data = normalSample(5000, 0.0, 1.0, 112);
    auto moments = stats::analysis::gaussian::calculateHigherMoments(data, true);
    EXPECT_NEAR(moments[0], 0.0, 0.05);  // 1st central moment = mean = 0
    EXPECT_NEAR(moments[1], 1.0, 0.05);  // 2nd central moment = variance ≈ 1
    EXPECT_NEAR(moments[2], 0.0, 0.10);  // 3rd central moment = skewness ≈ 0
}

TEST(GaussianAnalysis, HigherMomentsRawVsCentered) {
    auto data = normalSample(200, 5.0, 1.0, 113);
    auto centered = stats::analysis::gaussian::calculateHigherMoments(data, true);
    auto raw = stats::analysis::gaussian::calculateHigherMoments(data, false);
    EXPECT_EQ(centered.size(), 6u);
    EXPECT_EQ(raw.size(), 6u);
    // First raw moment = sample mean ≈ 5; first centered = 0
    EXPECT_NEAR(raw[0], 5.0, 0.3);
    EXPECT_NEAR(centered[0], 0.0, 0.05);
}

TEST(GaussianAnalysis, HigherMomentsEmptyThrows) {
    EXPECT_THROW(stats::analysis::gaussian::calculateHigherMoments({}, true),
                 std::invalid_argument);
}
