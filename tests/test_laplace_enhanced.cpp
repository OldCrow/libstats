#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/tests.h"
#include "include/enhanced_test_suite.h"
#include "libstats/distributions/laplace.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

// DistTraits specialisation for LaplaceDistribution
template <>
struct stats::tests::DistTraits<stats::LaplaceDistribution>
    : stats::tests::DistTraitsDefaults {
    static stats::LaplaceDistribution make() {
        return stats::LaplaceDistribution::create(0.0, 1.0).value;  // standard Laplace
    }
    static std::vector<double> domain() {
        return {-3.0, -1.0, 0.0, 1.0, 3.0};
    }
    static double batch_lo() { return -5.0; }
    static double batch_hi() { return  5.0; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::LaplaceDistribution::create(0.0, 0.0).isError(); },
            [] { return stats::LaplaceDistribution::create(0.0, -1.0).isError(); },
            [] { return stats::LaplaceDistribution::create(
                             std::numeric_limits<double>::infinity(), 1.0).isError(); },
            [] { return stats::LaplaceDistribution::create(
                             0.0, std::numeric_limits<double>::quiet_NaN()).isError(); },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Laplace, DistributionEnhancedTest,
                               ::testing::Types<stats::LaplaceDistribution>);

// ─── Per-distribution fixture ────────────────────────────────────────────────

namespace stats {

class LaplaceEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = LaplaceDistribution::create(0.0, 1.0);
        ASSERT_TRUE(r.isOk());
        sl_ = std::move(r.value);  // standard Laplace(0,1)
    }
    LaplaceDistribution sl_;
};

// ─── Known PDF / LogPDF values ────────────────────────────────────────────────

TEST_F(LaplaceEnhancedTest, PDFAtLocation) {
    // PDF(mu) = 1/(2b) — maximum at the location parameter
    EXPECT_NEAR(sl_.getProbability(0.0), 0.5, 1e-12) << "PDF(mu=0) = 1/(2*1) = 0.5";

    auto l = LaplaceDistribution::create(3.0, 2.0).value;
    EXPECT_NEAR(l.getProbability(3.0), 1.0/(2*2.0), 1e-12) << "PDF(mu=3, b=2) = 1/4";
}

TEST_F(LaplaceEnhancedTest, PDFFormula) {
    // PDF(x; mu=0, b=1) = 0.5 * exp(-|x|)
    for (double x : {-3.0, -1.5, 0.0, 1.5, 3.0}) {
        double expected = 0.5 * std::exp(-std::fabs(x));
        EXPECT_NEAR(sl_.getProbability(x), expected, 1e-12) << "at x=" << x;
    }
}

TEST_F(LaplaceEnhancedTest, LogPDFFormula) {
    // LogPDF(x; mu=0, b=1) = -log(2) - |x|
    for (double x : {-2.0, -0.5, 0.0, 0.5, 2.0}) {
        double expected = -std::log(2.0) - std::fabs(x);
        EXPECT_NEAR(sl_.getLogProbability(x), expected, 1e-12) << "at x=" << x;
    }
}

// ─── CDF ─────────────────────────────────────────────────────────────────────

TEST_F(LaplaceEnhancedTest, CDFAtLocation) {
    // CDF(mu) = 0.5 exactly for any Laplace distribution
    EXPECT_NEAR(sl_.getCumulativeProbability(0.0), 0.5, 1e-12) << "CDF(mu=0) = 0.5";

    auto l = LaplaceDistribution::create(-2.0, 3.0).value;
    EXPECT_NEAR(l.getCumulativeProbability(-2.0), 0.5, 1e-12) << "CDF(mu=-2) = 0.5";
}

TEST_F(LaplaceEnhancedTest, CDFFormula) {
    // For x > mu: CDF = 1 - 0.5*exp(-(x-mu)/b); for x <= mu: 0.5*exp((x-mu)/b)
    // Test x=1 (> mu=0): 1 - 0.5*exp(-1)
    EXPECT_NEAR(sl_.getCumulativeProbability(1.0), 1.0 - 0.5*std::exp(-1.0), 1e-12);
    // Test x=-1 (<= mu=0): 0.5*exp(-1)
    EXPECT_NEAR(sl_.getCumulativeProbability(-1.0), 0.5*std::exp(-1.0), 1e-12);
    // Symmetry: CDF(mu+d) + CDF(mu-d) = 1
    for (double d : {0.5, 1.0, 2.0, 5.0}) {
        EXPECT_NEAR(sl_.getCumulativeProbability(d) + sl_.getCumulativeProbability(-d),
                    1.0, 1e-12) << "Symmetry at d=" << d;
    }
}

// ─── Quantile ─────────────────────────────────────────────────────────────────

TEST_F(LaplaceEnhancedTest, QuantileRoundTrip) {
    for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
        const double q = sl_.getQuantile(p);
        EXPECT_NEAR(sl_.getCumulativeProbability(q), p, 1e-10) << "at p=" << p;
    }
}

TEST_F(LaplaceEnhancedTest, QuantileFormula) {
    // Q(p; mu=0, b=1): log(2p) for p < 0.5; -log(2*(1-p)) for p > 0.5
    EXPECT_NEAR(sl_.getQuantile(0.25), std::log(0.5),   1e-12);
    EXPECT_NEAR(sl_.getQuantile(0.75), -std::log(0.5),  1e-12);  // = log(2)
    EXPECT_EQ(sl_.getQuantile(0.5), 0.0);
}

// ─── Symmetry ─────────────────────────────────────────────────────────────────

TEST_F(LaplaceEnhancedTest, PDFSymmetry) {
    // Laplace is symmetric about mu: f(mu+d) == f(mu-d)
    auto l = LaplaceDistribution::create(2.5, 1.5).value;
    for (double d : {0.1, 0.5, 1.0, 3.0}) {
        EXPECT_NEAR(l.getProbability(2.5 + d), l.getProbability(2.5 - d), 1e-12)
            << "symmetry at d=" << d;
    }
}

// ─── Moment formulas ─────────────────────────────────────────────────────────

TEST_F(LaplaceEnhancedTest, MomentFormulas) {
    EXPECT_NEAR(sl_.getMean(),     0.0, 1e-12);
    EXPECT_NEAR(sl_.getVariance(), 2.0, 1e-12);  // 2*b^2 = 2*1 = 2
    EXPECT_NEAR(sl_.getSkewness(), 0.0, 1e-12);
    EXPECT_NEAR(sl_.getKurtosis(), 3.0, 1e-12);
    EXPECT_NEAR(sl_.getMedian(),   0.0, 1e-12);
    EXPECT_NEAR(sl_.getMode(),     0.0, 1e-12);
    EXPECT_NEAR(sl_.getEntropy(),  1.0 + std::log(2.0), 1e-12);

    auto l = LaplaceDistribution::create(5.0, 3.0).value;
    EXPECT_NEAR(l.getMean(),     5.0, 1e-12);
    EXPECT_NEAR(l.getVariance(), 2.0 * 9.0, 1e-12);  // 2*b^2 = 18
    EXPECT_NEAR(l.getMedian(),   5.0, 1e-12);
    EXPECT_NEAR(l.getEntropy(),  1.0 + std::log(6.0), 1e-12);  // 1+log(2*3)
}

// ─── Setter propagates ────────────────────────────────────────────────────────

TEST_F(LaplaceEnhancedTest, SetterPropagates) {
    auto l = LaplaceDistribution::create(0.0, 1.0).value;
    EXPECT_TRUE(l.isStandard());
    l.setMu(3.0);
    EXPECT_FALSE(l.isStandard());
    EXPECT_NEAR(l.getMean(),   3.0, 1e-12);
    EXPECT_NEAR(l.getMedian(), 3.0, 1e-12);
    EXPECT_NEAR(l.getProbability(3.0), 0.5, 1e-12);

    l.setParameters(0.0, 1.0);
    EXPECT_TRUE(l.isStandard());
}

// ─── MLE fit ─────────────────────────────────────────────────────────────────

TEST_F(LaplaceEnhancedTest, MLEFit) {
    std::mt19937 rng(42);
    auto source = LaplaceDistribution::create(2.0, 0.8).value;
    auto data   = source.sample(rng, 1000);

    auto fitted = LaplaceDistribution::create().value;
    fitted.fit(data);

    EXPECT_NEAR(fitted.getMu(), 2.0, 0.1) << "Fitted mu should be near 2.0";
    EXPECT_NEAR(fitted.getB(),  0.8, 0.1) << "Fitted b should be near 0.8";
}

// ─── Batch vs scalar ─────────────────────────────────────────────────────────

TEST_F(LaplaceEnhancedTest, VectorizedMatchesScalar) {
    const size_t N = 1024;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -5.0 + 10.0 * static_cast<double>(i) / static_cast<double>(N - 1);

    detail::PerformanceHint hint_vec, hint_scl;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    sl_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sl_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);

    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "LogPDF mismatch at i=" << i;

    sl_.getProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sl_.getProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "PDF mismatch at i=" << i;
}

// ─── VectorizedSpeedup (labelled timing) ─────────────────────────────────────

TEST_F(LaplaceEnhancedTest, VectorizedSpeedup) {
    const size_t N = 50000;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -5.0 + 10.0 * static_cast<double>(i + 1) / static_cast<double>(N);

    detail::PerformanceHint hint_vec, hint_scl;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    const auto t0 = std::chrono::high_resolution_clock::now();
    sl_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    const auto t1 = std::chrono::high_resolution_clock::now();
    sl_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double vec_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    const double scl_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    const double speedup = scl_us / std::max(vec_us, 1.0);
    std::cout << "Laplace LogPDF VECTORIZED speedup: " << speedup << "x "
              << "(VECTORIZED " << vec_us << "µs, SCALAR " << scl_us << "µs)\n";

    for (size_t i = 0; i < N; ++i)
        ASSERT_NEAR(out_vec[i], out_scl[i], 1e-10) << "mismatch at i=" << i;
    EXPECT_GT(speedup, 1.2) << "VECTORIZED should be faster than SCALAR for n=" << N;
}

}  // namespace stats
