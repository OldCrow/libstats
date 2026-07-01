#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/tests.h"
#include "include/enhanced_test_suite.h"
#include "libstats/distributions/geometric.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

// DistTraits specialisation for GeometricDistribution
template <>
struct stats::tests::DistTraits<stats::GeometricDistribution>
    : stats::tests::DistTraitsDefaults {
    static stats::GeometricDistribution make() {
        return stats::GeometricDistribution::create(0.5).unwrap();  // mean = 1, var = 2
    }
    static std::vector<double> domain() {
        return {0.0, 1.0, 2.0, 3.0, 4.0};  // non-negative integer inputs
    }
    static double batch_lo() { return 0.0; }
    static double batch_hi() { return 10.0; }
    static constexpr bool is_discrete = true;  // disables QuantileRoundTrip (floor property)
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::GeometricDistribution::create(0.0).isError(); },
            [] { return stats::GeometricDistribution::create(-0.5).isError(); },
            [] { return stats::GeometricDistribution::create(1.1).isError(); },
            [] { return stats::GeometricDistribution::create(
                             std::numeric_limits<double>::quiet_NaN()).isError(); },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Geometric, DistributionEnhancedTest,
                               ::testing::Types<stats::GeometricDistribution>);

// ─── Per-distribution fixture ───────────────────────────────────────────────

namespace stats {

class GeometricEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = GeometricDistribution::create(0.5);
        ASSERT_TRUE(r.isOk());
        g05_ = std::move(r.unwrap());  // Geometric(0.5): mean=1, var=2
    }
    GeometricDistribution g05_;
};

// ─── Known PMF values ────────────────────────────────────────────────────────

// PMF(k; p) = p*(1-p)^k  for Geometric(0.5): each PMF(k) = 0.5^(k+1)
TEST_F(GeometricEnhancedTest, KnownPMFValues) {
    EXPECT_NEAR(g05_.getProbability(0.0), 0.5,    1e-12) << "PMF(0) = p";
    EXPECT_NEAR(g05_.getProbability(1.0), 0.25,   1e-12) << "PMF(1) = p*(1-p)";
    EXPECT_NEAR(g05_.getProbability(2.0), 0.125,  1e-12) << "PMF(2) = p*(1-p)^2";
    EXPECT_NEAR(g05_.getProbability(3.0), 0.0625, 1e-12) << "PMF(3) = p*(1-p)^3";
    // Out-of-support
    EXPECT_EQ(g05_.getProbability(-1.0), 0.0) << "PMF(-1) = 0";
    EXPECT_EQ(g05_.getProbability(-0.5), 0.0) << "PMF(-0.5) = 0 (not integer)";
}

TEST_F(GeometricEnhancedTest, KnownLogPMFValues) {
    EXPECT_NEAR(g05_.getLogProbability(0.0), std::log(0.5),   1e-12);
    EXPECT_NEAR(g05_.getLogProbability(1.0), std::log(0.25),  1e-12);
    EXPECT_NEAR(g05_.getLogProbability(2.0), std::log(0.125), 1e-12);
    // LogPMF of -inf for out-of-support
    EXPECT_EQ(g05_.getLogProbability(-1.0), -std::numeric_limits<double>::infinity());
}

// ─── Known CDF values ────────────────────────────────────────────────────────

// CDF(k; p=0.5) = 1 - 0.5^(k+1)
TEST_F(GeometricEnhancedTest, KnownCDFValues) {
    EXPECT_NEAR(g05_.getCumulativeProbability(0.0), 0.5,    1e-12) << "CDF(0) = p";
    EXPECT_NEAR(g05_.getCumulativeProbability(1.0), 0.75,   1e-12) << "CDF(1) = 1-(1-p)^2";
    EXPECT_NEAR(g05_.getCumulativeProbability(2.0), 0.875,  1e-12) << "CDF(2)";
    EXPECT_NEAR(g05_.getCumulativeProbability(3.0), 0.9375, 1e-12) << "CDF(3)";
    EXPECT_EQ(g05_.getCumulativeProbability(-1.0), 0.0)      << "CDF(-1) = 0";
    EXPECT_NEAR(g05_.getCumulativeProbability(1e9), 1.0, 1e-9) << "CDF(+inf) ≈ 1";
}

// ─── Moment formulas ─────────────────────────────────────────────────────────

TEST_F(GeometricEnhancedTest, MomentFormulas) {
    // Geometric(p=0.5): mean=(1-p)/p=1, var=(1-p)/p^2=2
    EXPECT_NEAR(g05_.getMean(),     1.0,  1e-12);
    EXPECT_NEAR(g05_.getVariance(), 2.0,  1e-12);
    EXPECT_NEAR(g05_.getSkewness(), (2.0 - 0.5) / std::sqrt(0.5), 1e-12);
    EXPECT_NEAR(g05_.getKurtosis(), 6.0 + 0.5 * 0.5 / 0.5, 1e-12);

    // Verify p=0.3
    auto g03 = GeometricDistribution::create(0.3).unwrap();
    EXPECT_NEAR(g03.getMean(),     0.7 / 0.3,         1e-10) << "mean=(1-p)/p";
    EXPECT_NEAR(g03.getVariance(), 0.7 / (0.3 * 0.3), 1e-10) << "var=(1-p)/p^2";
}

// ─── Mode and Median ─────────────────────────────────────────────────────────

TEST_F(GeometricEnhancedTest, ModeIsAlwaysZero) {
    EXPECT_EQ(g05_.getMode(), 0.0);
    auto g09 = GeometricDistribution::create(0.9).unwrap();
    EXPECT_EQ(g09.getMode(), 0.0);  // still 0 even for high p
}

TEST_F(GeometricEnhancedTest, MedianFormula) {
    // Median = ceil(-ln2 / ln(1-p)) - 1
    // p=0.5: ceil(-ln2 / ln(0.5)) - 1 = ceil(1) - 1 = 0
    EXPECT_EQ(g05_.getMedian(), 0.0);

    // p=0.3: ceil(ln2 / (-ln(0.7))) - 1 = ceil(0.693/0.357) - 1 = ceil(1.942) - 1 = 1
    auto g03 = GeometricDistribution::create(0.3).unwrap();
    EXPECT_EQ(g03.getMedian(), 1.0);

    // p=1.0: degenerate, all mass at 0, median = 0
    auto g10 = GeometricDistribution::create(1.0).unwrap();
    EXPECT_EQ(g10.getMedian(), 0.0);
}

// ─── Entropy ─────────────────────────────────────────────────────────────────

TEST_F(GeometricEnhancedTest, EntropyFormula) {
    // H = [-(1-p)*ln(1-p) - p*ln(p)] / p
    const double p = 0.5;
    const double q = 0.5;
    const double expected = (-(q * std::log(q)) - p * std::log(p)) / p;
    EXPECT_NEAR(g05_.getEntropy(), expected, 1e-12);

    // p=1.0: degenerate, H=0
    auto g10 = GeometricDistribution::create(1.0).unwrap();
    EXPECT_EQ(g10.getEntropy(), 0.0);
}

// ─── Quantile floor property ─────────────────────────────────────────────────

// For discrete distributions: CDF(Q(p)) >= p
TEST_F(GeometricEnhancedTest, QuantileFloorProperty) {
    for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
        const double q = g05_.getQuantile(p);
        EXPECT_GE(q, 0.0) << "quantile must be non-negative";
        EXPECT_GE(g05_.getCumulativeProbability(q), p - 1e-12)
            << "CDF(Q(p)) >= p for p=" << p;
    }
}

// ─── Setter propagates to delegate ───────────────────────────────────────────

TEST_F(GeometricEnhancedTest, SetterPropagates) {
    auto g = GeometricDistribution::create(0.5).unwrap();
    EXPECT_NEAR(g.getMean(), 1.0, 1e-12);

    g.setP(0.25);
    EXPECT_NEAR(g.getP(), 0.25, 1e-12);
    EXPECT_NEAR(g.getMean(), 0.75 / 0.25, 1e-10);  // mean = (1-p)/p = 3

    // PMF should reflect new p immediately via delegate
    EXPECT_NEAR(g.getProbability(0.0), 0.25, 1e-12) << "PMF(0) = p after setP";

    g.setP(1.0);
    EXPECT_NEAR(g.getProbability(0.0), 1.0, 1e-12) << "Degenerate: all mass at 0";
    EXPECT_NEAR(g.getMean(), 0.0, 1e-12);
}

// ─── MLE accuracy ────────────────────────────────────────────────────────────

TEST_F(GeometricEnhancedTest, MLEFit) {
    std::mt19937 rng(42);
    auto source = GeometricDistribution::create(0.4).unwrap();  // true p = 0.4
    auto data   = source.sample(rng, 1000);

    auto fitted = GeometricDistribution::create(0.5).unwrap();
    fitted.fit(data);

    // MLE p_hat = 1/(1+x_bar). For Geometric(0.4), mean = (1-0.4)/0.4 = 1.5.
    // With n=1000, should be within 0.05 of 0.4.
    EXPECT_NEAR(fitted.getP(), 0.4, 0.05) << "MLE p should be close to 0.4";
}

// ─── LogPMF consistency: log(PMF(k)) == LogPMF(k) ───────────────────────────

TEST_F(GeometricEnhancedTest, LogPMFConsistency) {
    for (double k : {0.0, 1.0, 2.0, 3.0, 5.0, 10.0}) {
        const double pmf  = g05_.getProbability(k);
        const double lpmf = g05_.getLogProbability(k);
        EXPECT_NEAR(std::log(pmf), lpmf, 1e-10) << "at k=" << k;
    }
}

// ─── Batch vs scalar (VectorizedMatchesScalar) ───────────────────────────────

TEST_F(GeometricEnhancedTest, BatchMatchesScalar) {
    const size_t N = 100;
    vector<double> xs(N), pmf_b(N), lpmf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 20);  // integers 0..19

    g05_.getProbability(span<const double>(xs), span<double>(pmf_b));
    g05_.getLogProbability(span<const double>(xs), span<double>(lpmf_b));
    g05_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pmf_b[i],  g05_.getProbability(xs[i]),           1e-12) << "PMF i=" << i;
        EXPECT_NEAR(lpmf_b[i], g05_.getLogProbability(xs[i]),        1e-12) << "LogPMF i=" << i;
        EXPECT_NEAR(cdf_b[i],  g05_.getCumulativeProbability(xs[i]), 1e-12) << "CDF i=" << i;
    }
}

// ─── Speedup: PARALLEL should beat SCALAR for large batch ───────────────────

TEST_F(GeometricEnhancedTest, VectorizedSpeedup) {
    const size_t N = 50000;
    vector<double> xs(N), out_par(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 25);

    detail::PerformanceHint hint_par, hint_scl;
    hint_par.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    const auto t0 = std::chrono::high_resolution_clock::now();
    g05_.getLogProbability(span<const double>(xs), span<double>(out_par), hint_par);
    const auto t1 = std::chrono::high_resolution_clock::now();
    g05_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double par_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    const double scl_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    const double speedup = scl_us / std::max(par_us, 1.0);
    std::cout << "Geometric LogPMF PARALLEL speedup: " << speedup << "x "
              << "(PARALLEL " << par_us << "µs, SCALAR " << scl_us << "µs)\n";

    // Correctness check
    for (size_t i = 0; i < N; ++i)
        ASSERT_NEAR(out_par[i], out_scl[i], 1e-10) << "mismatch at i=" << i;
}

}  // namespace stats
