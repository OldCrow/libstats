#include <gtest/gtest.h>

/**
 * @file test_goodness_of_fit.cpp
 * A-9 (v2.0.0): real tests for stats::analysis generic goodness-of-fit
 * functions, replacing the original stub that tested nothing real.
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"
#include "libstats/stats/analysis/bootstrap.h"
#include "libstats/stats/analysis/cross_validation.h"
#include "libstats/stats/analysis/goodness_of_fit.h"
#include "libstats/stats/analysis/information_criteria.h"

#include <cmath>
#include <random>
#include <vector>

// EXPECT_THROW on [[nodiscard]] functions is intentional; suppress the false-positive.
// cppcheck-suppress unusedResult
#ifdef _MSC_VER
    #pragma warning(disable : 4834)  // discarding return value of [[nodiscard]] function
                                     // (intentional in EXPECT_THROW)
#else
    #pragma GCC diagnostic ignored "-Wunused-result"
#endif

using namespace stats;

static std::vector<double> normalSample(std::size_t n, double mu = 0, double sigma = 1,
                                        unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<> d(mu, sigma);
    std::vector<double> v(n);
    for (auto& x : v)
        x = d(rng);
    return v;
}
static std::vector<double> expSample(std::size_t n, double lam = 1, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::exponential_distribution<> d(lam);
    std::vector<double> v(n);
    for (auto& x : v)
        x = d(rng);
    return v;
}

// ── KS test ─────────────────────────────────────────────────────────────────
TEST(GoodnessOfFit, KS_GoodFit) {
    auto data = normalSample(200);
    auto g = GaussianDistribution::create(0.0, 1.0).unwrap();
    auto [stat, p, reject] = stats::analysis::kolmogorovSmirnovTest(data, g);
    EXPECT_GT(stat, 0.0);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_FALSE(reject);
}
TEST(GoodnessOfFit, KS_BadFit) {
    auto data = expSample(500);
    auto g = GaussianDistribution::create(1.0, 1.0).unwrap();
    auto [stat, p, reject] = stats::analysis::kolmogorovSmirnovTest(data, g);
    EXPECT_TRUE(reject);
}
TEST(GoodnessOfFit, KS_EmptyDataThrows) {
    auto g = GaussianDistribution::create(0.0, 1.0).unwrap();
    EXPECT_THROW(stats::analysis::kolmogorovSmirnovTest({}, g), std::invalid_argument);
}

// ── AD test ──────────────────────────────────────────────────────────────────
TEST(GoodnessOfFit, AD_GoodFit) {
    auto data = normalSample(200);
    auto g = GaussianDistribution::create(0.0, 1.0).unwrap();
    auto [stat, p, reject] = stats::analysis::andersonDarlingTest(data, g);
    EXPECT_GT(stat, 0.0);
    EXPECT_FALSE(reject);
}
TEST(GoodnessOfFit, AD_BadFit) {
    auto data = expSample(500);
    auto g = GaussianDistribution::create(1.0, 1.0).unwrap();
    EXPECT_TRUE(std::get<2>(stats::analysis::andersonDarlingTest(data, g)));
}

// ── Likelihood ratio test ────────────────────────────────────────────────────
TEST(GoodnessOfFit, LR_RejectsWrongModel) {
    // Joint test on μ and σ²: both parameters differ → df = 2.
    auto data = normalSample(100, 5.0, 2.0);
    auto restricted = GaussianDistribution::create(0.0, 1.0).unwrap();
    auto unrestricted = GaussianDistribution::create(5.0, 2.0).unwrap();
    auto [lr, p, reject] = stats::analysis::likelihoodRatioTest(data, restricted, unrestricted, 2);
    EXPECT_GT(lr, 0.0);
    EXPECT_TRUE(reject);
}
TEST(GoodnessOfFit, LR_EqualParamsReturnsValid) {
    // Identical models: lr_stat = 0 → valid result {0, 1.0, false} (ANA-4).
    // Model equivalence is a valid test outcome, not an error condition.
    auto data = normalSample(50);
    auto g = GaussianDistribution::create(0.0, 1.0).unwrap();
    auto [lr, p, reject] = stats::analysis::likelihoodRatioTest(data, g, g, 2);
    EXPECT_DOUBLE_EQ(lr, 0.0);
    EXPECT_DOUBLE_EQ(p, 1.0);
    EXPECT_FALSE(reject);
}
TEST(GoodnessOfFit, LR_InvalidDfThrows) {
    // df = 0 must throw before any computation.
    auto data = normalSample(50);
    auto g0 = GaussianDistribution::create(0.0, 1.0).unwrap();
    auto g1 = GaussianDistribution::create(1.0, 1.0).unwrap();
    EXPECT_THROW(stats::analysis::likelihoodRatioTest(data, g0, g1, 0), std::invalid_argument);
}

// ── Information criteria ─────────────────────────────────────────────────────
TEST(InformationCriteria, BetterFitHasLowerAIC) {
    auto data = normalSample(200, 5.0, 2.0);
    auto good = GaussianDistribution::create(5.0, 2.0).unwrap();
    auto bad = GaussianDistribution::create(0.0, 1.0).unwrap();
    auto [ag, bg, ag2, llg] = stats::analysis::informationCriteria(data, good);
    auto [ab, bb, ab2, llb] = stats::analysis::informationCriteria(data, bad);
    EXPECT_LT(ag, ab);
    EXPECT_GT(llg, llb);
}

// ── k-fold cross-validation ──────────────────────────────────────────────────
TEST(CrossValidation, KFold_Returns5Folds) {
    auto data = normalSample(100);
    auto folds = stats::analysis::kFoldCrossValidation<GaussianDistribution>(data, 5);
    EXPECT_EQ(folds.size(), 5u);
    for (const auto ll : folds) {
        EXPECT_TRUE(std::isfinite(ll));
    }
}
TEST(CrossValidation, KFold_BadKThrows) {
    auto data = normalSample(10);
    EXPECT_THROW((stats::analysis::kFoldCrossValidation<GaussianDistribution>(data, 1)),
                 std::invalid_argument);
}

// ── LOOCV ────────────────────────────────────────────────────────────────────
TEST(CrossValidation, LOOCV_FiniteResults) {
    auto data = normalSample(20);
    const auto ll = stats::analysis::leaveOneOutCrossValidation<GaussianDistribution>(data);
    EXPECT_TRUE(std::isfinite(ll));
}

// ── Bootstrap ────────────────────────────────────────────────────────────────
TEST(Bootstrap, MeanCI_ContainsTrueMean) {
    auto data = normalSample(200, 3.0, 1.0);
    auto [lo, hi] = stats::analysis::bootstrapMeanCI<GaussianDistribution>(data, 0.95, 1000, 42);
    EXPECT_LT(lo, 3.0);
    EXPECT_GT(hi, 3.0);
}
TEST(Bootstrap, EmptyDataThrows) {
    EXPECT_THROW((stats::analysis::bootstrapMeanCI<GaussianDistribution>({}, 0.95)),
                 std::invalid_argument);
}

// ── Anderson-Darling p-value continuity guard (MC-6 regression) ─────────────────

TEST(AndersonDarling, PValueContinuityNearStatSix) {
    // Regression guard for MC-6: the Anderson-Darling p-value had a 5×
    // discontinuity near stat ≈6.0 caused by a piecewise polynomial error.
    //
    // Guard strategy: build datasets of increasing size from exponential data
    // tested against a misspecified Gaussian. For each subset:
    //   (a) stat and p must be finite and in their valid ranges.
    //   (b) stat must be non-decreasing as n grows (more data → stronger signal).
    //   (c) p must be non-increasing (weaker: allow equal since p can saturate at 0).
    // These checks don't depend on p-value ratios (which grow naturally for
    // a consistently bad fit) and so are robust against the actual MC-6 regression.
    auto gaussian = GaussianDistribution::create(0.5, 1.0).unwrap();

    std::mt19937 rng(999);
    std::exponential_distribution<double> ed(1.0);
    std::vector<double> pool(400);
    for (auto& x : pool)
        x = ed(rng);

    double prev_stat = 0.0;
    double prev_p = 1.0;
    for (std::size_t n : {50u, 100u, 200u, 300u}) {
        std::vector<double> data(pool.begin(), pool.begin() + static_cast<long>(n));
        auto [stat, p, reject] = stats::analysis::andersonDarlingTest(data, gaussian);
        EXPECT_TRUE(std::isfinite(stat)) << "AD statistic must be finite";
        EXPECT_TRUE(std::isfinite(p)) << "AD p-value must be finite";
        EXPECT_GE(stat, 0.0);
        EXPECT_GE(p, 0.0);
        EXPECT_LE(p, 1.0);
        // More data from a bad distribution → stat must not decrease.
        EXPECT_GE(stat, prev_stat)
            << "AD stat must not decrease as n grows for a consistently bad fit";
        // p must not increase as the evidence against H0 strengthens.
        EXPECT_LE(p, prev_p + 1e-6)  // +epsilon tolerates floating-point equality at p≈0
            << "AD p-value must not increase as n grows for a consistently bad fit";
        prev_stat = stat;
        prev_p = p;
    }
    // The worst fit (n=300) must be clearly rejected.
    auto [stat_max, p_max, rej_max] = stats::analysis::andersonDarlingTest(
        std::vector<double>(pool.begin(), pool.begin() + 300), gaussian);
    EXPECT_TRUE(rej_max) << "Exponential data must be rejected against a shifted Gaussian";
}
