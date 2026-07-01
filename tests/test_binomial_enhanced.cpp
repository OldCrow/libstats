#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/binomial.h"

#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

namespace stats {

class BinomialEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = stats::BinomialDistribution::create(10, 0.5);
        ASSERT_TRUE(r.isOk());
        b10_05_ = std::move(r).unwrap();
    }
    BinomialDistribution b10_05_;  // Binomial(n=10, p=0.5)
};

// PMF known values for Binomial(10, 0.5)
TEST_F(BinomialEnhancedTest, KnownPMFValues) {
    // C(10,k)/1024 for k=0..10
    const double pmf_expected[11] = {
        1.0 / 1024.0,    // k=0
        10.0 / 1024.0,   // k=1
        45.0 / 1024.0,   // k=2
        120.0 / 1024.0,  // k=3
        210.0 / 1024.0,  // k=4
        252.0 / 1024.0,  // k=5
        210.0 / 1024.0,  // k=6
        120.0 / 1024.0,  // k=7
        45.0 / 1024.0,   // k=8
        10.0 / 1024.0,   // k=9
        1.0 / 1024.0     // k=10
    };
    double total = 0.0;
    for (int k = 0; k <= 10; ++k) {
        EXPECT_NEAR(b10_05_.getProbability(static_cast<double>(k)), pmf_expected[k], 1e-10)
            << "PMF mismatch at k=" << k;
        total += b10_05_.getProbability(static_cast<double>(k));
    }
    // Probabilities sum to 1
    EXPECT_NEAR(total, 1.0, 1e-10) << "PMF does not sum to 1";
}

// LogPMF == log(PMF) for all k
TEST_F(BinomialEnhancedTest, LogPMFConsistency) {
    for (int k = 0; k <= 10; ++k) {
        const double pmf = b10_05_.getProbability(static_cast<double>(k));
        const double lpdf = b10_05_.getLogProbability(static_cast<double>(k));
        EXPECT_NEAR(std::log(pmf), lpdf, 1e-12) << "at k=" << k;
    }
}

// CDF is non-decreasing and reaches 1 at n
TEST_F(BinomialEnhancedTest, CDFMonotoneAndBoundary) {
    double prev_cdf = -1.0;
    for (int k = 0; k <= 10; ++k) {
        const double cdf = b10_05_.getCumulativeProbability(static_cast<double>(k));
        EXPECT_GT(cdf, prev_cdf) << "CDF not strictly increasing at k=" << k;
        EXPECT_GE(cdf, 0.0) << "CDF below 0 at k=" << k;
        EXPECT_LE(cdf, 1.0) << "CDF above 1 at k=" << k;
        prev_cdf = cdf;
    }
    EXPECT_DOUBLE_EQ(b10_05_.getCumulativeProbability(10.0), 1.0);
    EXPECT_DOUBLE_EQ(b10_05_.getCumulativeProbability(-1.0), 0.0);
}

// CDF(5) ≈ 0.623046875 for Binomial(10, 0.5)
TEST_F(BinomialEnhancedTest, KnownCDFValues) {
    EXPECT_NEAR(b10_05_.getCumulativeProbability(5.0), 0.623046875, 1e-6);
    EXPECT_NEAR(b10_05_.getCumulativeProbability(0.0), 1.0 / 1024.0, 1e-10);
    EXPECT_NEAR(b10_05_.getCumulativeProbability(9.0), 1023.0 / 1024.0, 1e-8);
}

// Moments for Binomial(10, 0.5) and Binomial(20, 0.3)
TEST_F(BinomialEnhancedTest, Moments) {
    EXPECT_NEAR(b10_05_.getMean(), 5.0, 1e-12);
    EXPECT_NEAR(b10_05_.getVariance(), 2.5, 1e-12);
    EXPECT_NEAR(b10_05_.getSkewness(), 0.0, 1e-12);

    auto b20_03 = BinomialDistribution::create(20, 0.3).unwrap();
    EXPECT_NEAR(b20_03.getMean(), 6.0, 1e-12);
    EXPECT_NEAR(b20_03.getVariance(), 4.2, 1e-12);
    // Skewness = (1-2p)/sqrt(npq) = 0.4/sqrt(4.2)
    const double expected_skew = 0.4 / std::sqrt(4.2);
    EXPECT_NEAR(b20_03.getSkewness(), expected_skew, 1e-10);
}

// PMF out of range returns 0; non-finite x returns 0
TEST_F(BinomialEnhancedTest, BoundaryPMF) {
    EXPECT_DOUBLE_EQ(b10_05_.getProbability(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(b10_05_.getProbability(11.0), 0.0);
    EXPECT_TRUE(std::isnan(
        b10_05_.getProbability(std::numeric_limits<double>::quiet_NaN())));  // NaN propagates
    EXPECT_DOUBLE_EQ(b10_05_.getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(b10_05_.getLogProbability(-1.0), -std::numeric_limits<double>::infinity());
}

// Quantile round-trip: quantile(CDF(k)) == k for all k in {0..10}
TEST_F(BinomialEnhancedTest, QuantileRoundTrip) {
    for (int k = 0; k <= 10; ++k) {
        const double cdf = b10_05_.getCumulativeProbability(static_cast<double>(k));
        const double q = b10_05_.getQuantile(cdf);
        EXPECT_NEAR(q, static_cast<double>(k), 0.5) << "at k=" << k;
    }
    EXPECT_DOUBLE_EQ(b10_05_.getQuantile(0.0), 0.0);
    EXPECT_DOUBLE_EQ(b10_05_.getQuantile(1.0), 10.0);
}

// Batch matches scalar (PDF, LogPDF, CDF)
TEST_F(BinomialEnhancedTest, BatchMatchesScalar) {
    const size_t N = 100;
    vector<double> xs(N), pdf_b(N), lpdf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 11);

    b10_05_.getProbability(span<const double>(xs), span<double>(pdf_b));
    b10_05_.getLogProbability(span<const double>(xs), span<double>(lpdf_b));
    b10_05_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pdf_b[i], b10_05_.getProbability(xs[i]), 1e-12) << "i=" << i;
        EXPECT_NEAR(lpdf_b[i], b10_05_.getLogProbability(xs[i]), 1e-12) << "i=" << i;
        EXPECT_NEAR(cdf_b[i], b10_05_.getCumulativeProbability(xs[i]), 1e-9) << "i=" << i;
    }
}

// VECTORIZED == SCALAR
// Note: VECTORIZED = cached scalar loop (no vector_lgamma); results are bit-exact.
TEST_F(BinomialEnhancedTest, VectorizedEqualsScalar) {
    const size_t N = 300;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 11);

    detail::PerformanceHint hint_vec, hint_scl;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;
    b10_05_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    b10_05_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);

    for (size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(out_vec[i], out_scl[i]) << "i=" << i;
}

// MLE fit recovers true p for Binomial(10, 0.7)
TEST_F(BinomialEnhancedTest, MLEFit) {
    mt19937 rng(42);
    auto source = BinomialDistribution::create(10, 0.7).unwrap();
    const auto data = source.sample(rng, 1000);
    auto fitted = BinomialDistribution::create(10, 0.5).unwrap();
    fitted.fit(data);
    EXPECT_NEAR(fitted.getP(), 0.7, 0.05) << "Fitted p should be near 0.7";
    EXPECT_GE(fitted.getN(), 1);
}

// Invalid parameters rejected
TEST_F(BinomialEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(BinomialDistribution::create(0, 0.5).isError());
    EXPECT_TRUE(BinomialDistribution::create(-1, 0.5).isError());
    EXPECT_TRUE(BinomialDistribution::create(10, -0.1).isError());
    EXPECT_TRUE(BinomialDistribution::create(10, 1.1).isError());
    EXPECT_TRUE(
        BinomialDistribution::create(10, std::numeric_limits<double>::quiet_NaN()).isError());

    auto d = BinomialDistribution::create(10, 0.5).unwrap();
    EXPECT_TRUE(d.trySetP(1.5).isError());
    EXPECT_TRUE(d.trySetN(0).isError());
    EXPECT_DOUBLE_EQ(d.getP(), 0.5);
    EXPECT_EQ(d.getN(), 10);
}

// Setter propagates: setP changes mean, variance, skewness
TEST_F(BinomialEnhancedTest, SetterPropagates) {
    auto d = BinomialDistribution::create(10, 0.5).unwrap();
    EXPECT_NEAR(d.getMean(), 5.0, 1e-12);
    d.setP(0.3);
    EXPECT_NEAR(d.getMean(), 3.0, 1e-12);
    EXPECT_NEAR(d.getVariance(), 10.0 * 0.3 * 0.7, 1e-12);
    d.setN(20);
    EXPECT_NEAR(d.getMean(), 20.0 * 0.3, 1e-12);
}

// p=0 and p=1 edge cases
TEST_F(BinomialEnhancedTest, ProbabilityEdgeCases) {
    auto b_p0 = BinomialDistribution::create(10, 0.0).unwrap();
    EXPECT_NEAR(b_p0.getProbability(0.0), 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(b_p0.getProbability(1.0), 0.0);
    EXPECT_DOUBLE_EQ(b_p0.getCumulativeProbability(-0.5), 0.0);
    EXPECT_DOUBLE_EQ(b_p0.getCumulativeProbability(0.0), 1.0);

    auto b_p1 = BinomialDistribution::create(10, 1.0).unwrap();
    EXPECT_NEAR(b_p1.getProbability(10.0), 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(b_p1.getProbability(9.0), 0.0);
}

// Parallel batch LogPDF == scalar (timing test)
TEST_F(BinomialEnhancedTest, ParallelBatchCorrectness) {
    const size_t N = 5000;
    vector<double> xs(N), out_par(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 11);

    detail::PerformanceHint hint_par, hint_scl;
    hint_par.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    const auto t0 = chrono::high_resolution_clock::now();
    b10_05_.getLogProbability(span<const double>(xs), span<double>(out_par), hint_par);
    const auto t1 = chrono::high_resolution_clock::now();
    b10_05_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    const auto t2 = chrono::high_resolution_clock::now();

    const double par_us =
        static_cast<double>(chrono::duration_cast<chrono::microseconds>(t1 - t0).count());
    const double scl_us =
        static_cast<double>(chrono::duration_cast<chrono::microseconds>(t2 - t1).count());

    cout << "Binomial LogPDF PARALLEL vs SCALAR: " << par_us << "μs vs " << scl_us << "μs (n=" << N
         << ")\n";
    cout << "Note: VECTORIZED = cached scalar loop (no vector_lgamma);\n"
         << "      PARALLEL provides true multi-core throughput.\n";

    for (size_t i = 0; i < N; ++i)
        ASSERT_NEAR(out_par[i], out_scl[i], 1e-12) << "parallel mismatch at i=" << i;
}

}  // namespace stats

//==============================================================================
// DistTraits specialization for stats::BinomialDistribution
//==============================================================================
template <>
struct stats::tests::DistTraits<stats::BinomialDistribution> : stats::tests::DistTraitsDefaults {
    static stats::BinomialDistribution make() {
        return stats::BinomialDistribution::create(10, 0.5).unwrap();
    }
    static std::vector<double> domain() { return {0.0, 2.0, 5.0, 8.0, 10.0}; }
    static double batch_lo() { return 0.0; }
    static double batch_hi() { return 10.0; }
    static constexpr bool is_discrete = true;
    static double cdf_tolerance() { return 1e-09; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::BinomialDistribution::create(0, 0.5).isError(); },
            [] { return stats::BinomialDistribution::create(-1, 0.5).isError(); },
            [] { return stats::BinomialDistribution::create(10, -0.1).isError(); },
            [] { return stats::BinomialDistribution::create(10, 1.1).isError(); },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Binomial, DistributionEnhancedTest,
                               ::testing::Types<stats::BinomialDistribution>);
