#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/rayleigh.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

namespace stats {

class RayleighEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = stats::RayleighDistribution::create(1.0);
        ASSERT_TRUE(r.isOk());
        r1_ = std::move(r).unwrap();
    }
    RayleighDistribution r1_;  // standard Rayleigh (σ=1)
};

// PDF(x=σ) = exp(-0.5) for any σ
TEST_F(RayleighEnhancedTest, PDFAtSigma) {
    for (double sigma : {0.5, 1.0, 2.0, 5.0}) {
        auto d = RayleighDistribution::create(sigma).unwrap();
        EXPECT_NEAR(d.getProbability(sigma), std::exp(-0.5) / sigma, 1e-12)
            << "PDF(σ) != exp(-0.5)/σ for σ=" << sigma;
    }
}

// CDF(σ) = 1 - exp(-0.5) for any σ
TEST_F(RayleighEnhancedTest, CDFAtSigma) {
    for (double sigma : {0.5, 1.0, 2.0, 5.0}) {
        auto d = RayleighDistribution::create(sigma).unwrap();
        EXPECT_NEAR(d.getCumulativeProbability(sigma), 1.0 - std::exp(-0.5), 1e-12)
            << "CDF(σ) != 1-exp(-0.5) for σ=" << sigma;
    }
}

// Mean = σ·√(π/2), Variance = σ²·(4−π)/2
TEST_F(RayleighEnhancedTest, MomentFormulas) {
    EXPECT_NEAR(r1_.getMean(), std::sqrt(M_PI / 2.0), 1e-12);
    EXPECT_NEAR(r1_.getVariance(), (4.0 - M_PI) / 2.0, 1e-12);
    EXPECT_NEAR(r1_.getMode(), 1.0, 1e-14);
    EXPECT_NEAR(r1_.getMedian(), std::sqrt(2.0 * M_LN2), 1e-12);
}

// Skewness and kurtosis are constants
TEST_F(RayleighEnhancedTest, ConstantSkewnessKurtosis) {
    const double expected_skew = 2.0 * std::sqrt(M_PI) * (M_PI - 3.0) / std::pow(4.0 - M_PI, 1.5);
    const double four_minus_pi = 4.0 - M_PI;
    const double expected_kurt =
        -(6.0 * M_PI * M_PI - 24.0 * M_PI + 16.0) / (four_minus_pi * four_minus_pi);
    EXPECT_NEAR(r1_.getSkewness(), expected_skew, 1e-10);
    EXPECT_NEAR(r1_.getKurtosis(), expected_kurt, 1e-10);

    // Must be σ-independent
    auto r2 = RayleighDistribution::create(5.0).unwrap();
    EXPECT_NEAR(r2.getSkewness(), expected_skew, 1e-10);
    EXPECT_NEAR(r2.getKurtosis(), expected_kurt, 1e-10);
}

// log(PDF) == LogPDF
TEST_F(RayleighEnhancedTest, LogPDFConsistency) {
    for (double x : {0.5, 1.0, 2.0, 3.0, 5.0}) {
        const double pdf = r1_.getProbability(x);
        const double lpdf = r1_.getLogProbability(x);
        EXPECT_NEAR(std::log(pdf), lpdf, 1e-12) << "at x=" << x;
    }
}

// Out-of-support
TEST_F(RayleighEnhancedTest, OutOfSupport) {
    EXPECT_EQ(r1_.getProbability(0.0), 0.0);
    EXPECT_EQ(r1_.getProbability(-1.0), 0.0);
    EXPECT_EQ(r1_.getCumulativeProbability(0.0), 0.0);
    EXPECT_EQ(r1_.getLogProbability(0.0), -std::numeric_limits<double>::infinity());
}

// Quantile round-trip: CDF(Q(p)) = p
TEST_F(RayleighEnhancedTest, QuantileRoundTrip) {
    for (double p : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9}) {
        const double q = r1_.getQuantile(p);
        EXPECT_GE(q, 0.0);
        EXPECT_NEAR(r1_.getCumulativeProbability(q), p, 1e-10) << "at p=" << p;
    }
}

// Batch matches scalar
TEST_F(RayleighEnhancedTest, BatchMatchesScalar) {
    const size_t N = 200;
    vector<double> xs(N), pdf_b(N), lpdf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = 0.05 + 0.1 * static_cast<double>(i + 1);
    r1_.getProbability(span<const double>(xs), span<double>(pdf_b));
    r1_.getLogProbability(span<const double>(xs), span<double>(lpdf_b));
    r1_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pdf_b[i], r1_.getProbability(xs[i]), 1e-12) << "PDF i=" << i;
        EXPECT_NEAR(lpdf_b[i], r1_.getLogProbability(xs[i]), 1e-12) << "LogPDF i=" << i;
        EXPECT_NEAR(cdf_b[i], r1_.getCumulativeProbability(xs[i]), 1e-12) << "CDF i=" << i;
    }
}

// VECTORIZED matches SCALAR (exercises the 5-step pipeline)
TEST_F(RayleighEnhancedTest, VectorizedMatchesScalar) {
    const size_t N = 1024;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = 0.01 + 0.05 * static_cast<double>(i + 1);

    detail::PerformanceHint hint_vec, hint_scl;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;
    r1_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    r1_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);

    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "LogPDF mismatch at i=" << i;

    r1_.getCumulativeProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    r1_.getCumulativeProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "CDF mismatch at i=" << i;
}

// MLE from samples
TEST_F(RayleighEnhancedTest, MLEFit) {
    mt19937 rng(42);
    auto source = RayleighDistribution::create(2.5).unwrap();
    const auto data = source.sample(rng, 500);
    auto fitted = RayleighDistribution::create(1.0).unwrap();
    fitted.fit(data);
    EXPECT_NEAR(fitted.getSigma(), 2.5, 0.3) << "Fitted sigma should be near 2.5";
}

// Setter propagates
TEST_F(RayleighEnhancedTest, SetterPropagates) {
    auto d = RayleighDistribution::create(1.0).unwrap();
    EXPECT_NEAR(d.getMean(), std::sqrt(M_PI / 2.0), 1e-12);
    d.setSigma(2.0);
    EXPECT_NEAR(d.getMean(), 2.0 * std::sqrt(M_PI / 2.0), 1e-12);
    d.setParameters(1.0);
    EXPECT_NEAR(d.getMean(), std::sqrt(M_PI / 2.0), 1e-12);
}

// Invalid parameters rejected
TEST_F(RayleighEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(RayleighDistribution::create(-1.0).isError());
    EXPECT_TRUE(RayleighDistribution::create(0.0).isError());
    EXPECT_TRUE(RayleighDistribution::create(std::numeric_limits<double>::quiet_NaN()).isError());

    auto d = RayleighDistribution::create(1.0).unwrap();
    EXPECT_TRUE(d.trySetSigma(-1.0).isError());
    EXPECT_DOUBLE_EQ(d.getSigma(), 1.0);
}

// Speedup: VECTORIZED LogPDF (labelled timing)
TEST_F(RayleighEnhancedTest, VectorizedSpeedup) {
    const size_t N = 50000;
    vector<double> xs(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = 0.01 + 0.001 * static_cast<double>(i + 1);
    vector<double> out(N), scl(N);
    detail::PerformanceHint vec_hint, scl_hint;
    vec_hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    scl_hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    const auto t0 = std::chrono::high_resolution_clock::now();
    r1_.getLogProbability(span<const double>(xs), span<double>(out), vec_hint);
    const auto t1 = std::chrono::high_resolution_clock::now();
    r1_.getLogProbability(span<const double>(xs), span<double>(scl), scl_hint);
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double vec_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    const double scl_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    const double speedup = scl_us / std::max(vec_us, 1.0);
    std::cout << "Rayleigh LogPDF VECTORIZED speedup: " << speedup << "x "
              << "(VECTORIZED " << vec_us << "μs, SCALAR " << scl_us << "μs)\n";

    for (size_t i = 0; i < N; ++i)
        ASSERT_NEAR(out[i], scl[i], 1e-10) << "mismatch at i=" << i;
    EXPECT_GT(speedup, 1.5) << "VECTORIZED should be at least 1.5x faster";
}

}  // namespace stats

//==============================================================================
// DistTraits specialization for stats::RayleighDistribution
//==============================================================================
template <>
struct stats::tests::DistTraits<stats::RayleighDistribution> : stats::tests::DistTraitsDefaults {
    static stats::RayleighDistribution make() {
        return stats::RayleighDistribution::create(1.0).unwrap();
    }
    static std::vector<double> domain() { return {0.5, 1.0, 2.0, 3.0, 5.0}; }
    static double batch_lo() { return 0.1; }
    static double batch_hi() { return 10.0; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::RayleighDistribution::create(-1.0).isError(); },
            [] { return stats::RayleighDistribution::create(0.0).isError(); },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Rayleigh, DistributionEnhancedTest,
                               ::testing::Types<stats::RayleighDistribution>);
