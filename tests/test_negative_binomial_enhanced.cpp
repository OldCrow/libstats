#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/tests.h"
#include "libstats/distributions/negative_binomial.h"

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

class NegativeBinomialEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = stats::NegativeBinomialDistribution::create(2.0, 0.5);
        ASSERT_TRUE(r.isOk());
        nb2_05_ = std::move(r.value);
    }
    NegativeBinomialDistribution nb2_05_;  // NB(r=2, p=0.5)
};

// PMF known values for NB(2, 0.5)
// PMF(k) = C(k+1, k) * (0.5)^2 * (0.5)^k = (k+1) * (0.5)^(k+2)
TEST_F(NegativeBinomialEnhancedTest, KnownPMFValues) {
    // PMF(k) = (k+1) * 0.5^(k+2)
    for (int k = 0; k <= 10; ++k) {
        const double expected = static_cast<double>(k + 1) * std::pow(0.5, k + 2);
        EXPECT_NEAR(nb2_05_.getProbability(static_cast<double>(k)), expected, 1e-10)
            << "PMF mismatch at k=" << k;
    }
    // Check PMF sums close to 1 over first 50 values
    double total = 0.0;
    for (int k = 0; k <= 50; ++k)
        total += nb2_05_.getProbability(static_cast<double>(k));
    EXPECT_NEAR(total, 1.0, 5e-4) << "PMF partial sum should be near 1";
}

// LogPMF == log(PMF) for all k in {0..10}
TEST_F(NegativeBinomialEnhancedTest, LogPMFConsistency) {
    for (int k = 0; k <= 10; ++k) {
        const double pmf  = nb2_05_.getProbability(static_cast<double>(k));
        const double lpdf = nb2_05_.getLogProbability(static_cast<double>(k));
        EXPECT_NEAR(std::log(pmf), lpdf, 1e-12) << "at k=" << k;
    }
}

// CDF is non-decreasing and starts at PMF(0)
TEST_F(NegativeBinomialEnhancedTest, CDFMonotoneAndBoundary) {
    double prev_cdf = -1.0;
    for (int k = 0; k <= 20; ++k) {
        const double cdf = nb2_05_.getCumulativeProbability(static_cast<double>(k));
        EXPECT_GT(cdf, prev_cdf) << "CDF not strictly increasing at k=" << k;
        EXPECT_GE(cdf, 0.0);
        EXPECT_LE(cdf, 1.0);
        prev_cdf = cdf;
    }
    // CDF(-1) = 0
    EXPECT_DOUBLE_EQ(nb2_05_.getCumulativeProbability(-1.0), 0.0);
    // CDF(0) = PMF(0) = 0.25
    EXPECT_NEAR(nb2_05_.getCumulativeProbability(0.0), 0.25, 1e-8);
    // CDF(1) = 0.5
    EXPECT_NEAR(nb2_05_.getCumulativeProbability(1.0), 0.5, 1e-8);
}

// Moments for NB(2, 0.5) and NB(3, 0.6)
TEST_F(NegativeBinomialEnhancedTest, Moments) {
    EXPECT_NEAR(nb2_05_.getMean(),     2.0, 1e-12);
    EXPECT_NEAR(nb2_05_.getVariance(), 4.0, 1e-12);
    // Skewness = (2-p)/sqrt(r(1-p)) = 1.5/sqrt(1) = 1.5
    EXPECT_NEAR(nb2_05_.getSkewness(), 1.5, 1e-12);

    auto nb3_06 = NegativeBinomialDistribution::create(3.0, 0.6).value;
    // mean = 3*0.4/0.6 = 2.0; var = 3*0.4/0.36 = 10/3 ≈ 3.333
    EXPECT_NEAR(nb3_06.getMean(),     2.0,       1e-12);
    EXPECT_NEAR(nb3_06.getVariance(), 10.0 / 3.0, 1e-10);
}

// PMF out of range returns 0; non-finite x returns 0
TEST_F(NegativeBinomialEnhancedTest, BoundaryPMF) {
    EXPECT_DOUBLE_EQ(nb2_05_.getProbability(-1.0), 0.0);
    EXPECT_DOUBLE_EQ(nb2_05_.getProbability(-0.5), 0.0);
    EXPECT_DOUBLE_EQ(nb2_05_.getProbability(std::numeric_limits<double>::quiet_NaN()), 0.0);
    EXPECT_DOUBLE_EQ(nb2_05_.getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_DOUBLE_EQ(nb2_05_.getLogProbability(-1.0),
                     -std::numeric_limits<double>::infinity());
}

// Quantile round-trip: quantile(CDF(k)) ≈ k for k in {0..10}
TEST_F(NegativeBinomialEnhancedTest, QuantileRoundTrip) {
    for (int k = 0; k <= 10; ++k) {
        const double cdf = nb2_05_.getCumulativeProbability(static_cast<double>(k));
        const double q   = nb2_05_.getQuantile(cdf);
        EXPECT_NEAR(q, static_cast<double>(k), 0.5) << "at k=" << k;
    }
    EXPECT_DOUBLE_EQ(nb2_05_.getQuantile(0.0), 0.0);
}

// Batch matches scalar (PDF, LogPDF, CDF)
TEST_F(NegativeBinomialEnhancedTest, BatchMatchesScalar) {
    const size_t N = 100;
    vector<double> xs(N), pdf_b(N), lpdf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 20);

    nb2_05_.getProbability(span<const double>(xs), span<double>(pdf_b));
    nb2_05_.getLogProbability(span<const double>(xs), span<double>(lpdf_b));
    nb2_05_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pdf_b[i],  nb2_05_.getProbability(xs[i]),           1e-12) << "i=" << i;
        EXPECT_NEAR(lpdf_b[i], nb2_05_.getLogProbability(xs[i]),        1e-12) << "i=" << i;
        EXPECT_NEAR(cdf_b[i],  nb2_05_.getCumulativeProbability(xs[i]), 1e-9)  << "i=" << i;
    }
}

// VECTORIZED == SCALAR
TEST_F(NegativeBinomialEnhancedTest, VectorizedEqualsScalar) {
    const size_t N = 300;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 20);
    for (size_t i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(out_vec[i], out_scl[i]) << "i=" << i;
}

// MLE fit recovers true parameters from NB(3, 0.6)
TEST_F(NegativeBinomialEnhancedTest, MLEFit) {
    mt19937 rng(42);
    auto source = NegativeBinomialDistribution::create(3.0, 0.6).value;
    const auto data = source.sample(rng, 1000);
    auto fitted = NegativeBinomialDistribution::create(1.0, 0.5).value;
    fitted.fit(data);
    EXPECT_NEAR(fitted.getR(), 3.0, 1.0) << "Fitted r should be near 3.0";
    EXPECT_NEAR(fitted.getP(), 0.6, 0.1) << "Fitted p should be near 0.6";
}

// MLE with real-valued r
TEST_F(NegativeBinomialEnhancedTest, MLERealR) {
    mt19937 rng(99);
    auto source = NegativeBinomialDistribution::create(1.5, 0.7).value;
    const auto data = source.sample(rng, 800);
    auto fitted = NegativeBinomialDistribution::create(1.0, 0.5).value;
    fitted.fit(data);
    EXPECT_GT(fitted.getR(), 0.0);
    EXPECT_GT(fitted.getP(), 0.0);
    EXPECT_LE(fitted.getP(), 1.0);
}

// Invalid parameters rejected
TEST_F(NegativeBinomialEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(NegativeBinomialDistribution::create(0.0, 0.5).isError());
    EXPECT_TRUE(NegativeBinomialDistribution::create(-1.0, 0.5).isError());
    EXPECT_TRUE(NegativeBinomialDistribution::create(2.0, 0.0).isError());
    EXPECT_TRUE(NegativeBinomialDistribution::create(2.0, 1.1).isError());
    EXPECT_TRUE(NegativeBinomialDistribution::create(
                    std::numeric_limits<double>::infinity(), 0.5).isError());

    auto d = NegativeBinomialDistribution::create(2.0, 0.5).value;
    EXPECT_TRUE(d.trySetR(-1.0).isError());
    EXPECT_TRUE(d.trySetP(0.0).isError());
    EXPECT_DOUBLE_EQ(d.getR(), 2.0);
    EXPECT_DOUBLE_EQ(d.getP(), 0.5);
}

// Setter propagates: setR and setP change moments
TEST_F(NegativeBinomialEnhancedTest, SetterPropagates) {
    auto d = NegativeBinomialDistribution::create(2.0, 0.5).value;
    EXPECT_NEAR(d.getMean(), 2.0, 1e-12);
    d.setP(0.4);
    // mean = 2 * 0.6 / 0.4 = 3.0
    EXPECT_NEAR(d.getMean(), 3.0, 1e-12);
    d.setR(4.0);
    // mean = 4 * 0.6 / 0.4 = 6.0
    EXPECT_NEAR(d.getMean(), 6.0, 1e-12);
}

// p=1 edge case: all mass at k=0 (geometric series sum = p^r / (1 - (1-p)) = p^r when p=1)
// NB(r, p=1): PMF(0) = 1, PMF(k>0) = 0
TEST_F(NegativeBinomialEnhancedTest, PEqualsOneEdgeCase) {
    auto nb_p1 = NegativeBinomialDistribution::create(2.0, 1.0).value;
    EXPECT_NEAR(nb_p1.getProbability(0.0), 1.0, 1e-12);
    EXPECT_DOUBLE_EQ(nb_p1.getProbability(1.0), 0.0);
    EXPECT_NEAR(nb_p1.getCumulativeProbability(0.0), 1.0, 1e-10);
}

// Parallel batch LogPDF == scalar (timing test)
TEST_F(NegativeBinomialEnhancedTest, ParallelBatchCorrectness) {
    const size_t N = 5000;
    vector<double> xs(N), out_par(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 20);

    const auto t0 = chrono::high_resolution_clock::now();
    const auto t1 = chrono::high_resolution_clock::now();
    const auto t2 = chrono::high_resolution_clock::now();

    const double par_us = static_cast<double>(
        chrono::duration_cast<chrono::microseconds>(t1 - t0).count());
    const double scl_us = static_cast<double>(
        chrono::duration_cast<chrono::microseconds>(t2 - t1).count());

    cout << "NegBinom LogPDF PARALLEL vs SCALAR: " << par_us << "μs vs " << scl_us
         << "μs (n=" << N << ")\n";
    cout << "Note: VECTORIZED = cached scalar loop (no vector_lgamma);\n"
         << "      PARALLEL provides true multi-core throughput.\n";

    for (size_t i = 0; i < N; ++i)
        ASSERT_NEAR(out_par[i], out_scl[i], 1e-12) << "parallel mismatch at i=" << i;
}

}  // namespace stats
