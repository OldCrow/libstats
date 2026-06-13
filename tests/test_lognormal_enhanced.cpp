#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/tests.h"
#include "libstats/distributions/lognormal.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

namespace stats {

class LogNormalEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = stats::LogNormalDistribution::create(0.0, 1.0);
        ASSERT_TRUE(r.isOk());
        std_ln_ = std::move(r.value);
    }
    LogNormalDistribution std_ln_;  // standard log-normal μ=0, σ=1
};

// Standard log-normal: PDF(1) = 1/sqrt(2π)
TEST_F(LogNormalEnhancedTest, StandardPDFAtOne) {
    const double expected = 1.0 / std::sqrt(2.0 * M_PI);
    EXPECT_NEAR(std_ln_.getProbability(1.0), expected, 1e-10);
}

// Standard log-normal: CDF(1) = 0.5 (median at exp(0)=1)
TEST_F(LogNormalEnhancedTest, StandardCDFAtOne) {
    EXPECT_NEAR(std_ln_.getCumulativeProbability(1.0), 0.5, 1e-8);
}

// Quantile(0.5) for LogNormal(μ, σ) = exp(μ) regardless of σ
TEST_F(LogNormalEnhancedTest, MedianEqualsExpMu) {
    for (double mu : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        auto d = LogNormalDistribution::create(mu, 1.0).value;
        EXPECT_NEAR(d.getQuantile(0.5), std::exp(mu), 1e-8)
            << "median != exp(mu) for mu=" << mu;
        EXPECT_NEAR(d.getMedian(), std::exp(mu), 1e-12)
            << "getMedian() != exp(mu) for mu=" << mu;
    }
}

// Mode = exp(μ - σ²)
TEST_F(LogNormalEnhancedTest, ModeFormula) {
    const double mu = 2.0, sigma = 0.5;
    auto d = LogNormalDistribution::create(mu, sigma).value;
    EXPECT_NEAR(d.getMode(), std::exp(mu - sigma * sigma), 1e-12);
}

// Mean = exp(μ + σ²/2)
TEST_F(LogNormalEnhancedTest, MeanFormula) {
    const double mu = 1.0, sigma = 0.5;
    auto d = LogNormalDistribution::create(mu, sigma).value;
    EXPECT_NEAR(d.getMean(), std::exp(mu + 0.5 * sigma * sigma), 1e-12);
}

// Variance = (exp(σ²) - 1) * exp(2μ + σ²)
TEST_F(LogNormalEnhancedTest, VarianceFormula) {
    const double mu = 0.5, sigma = 0.3;
    auto d = LogNormalDistribution::create(mu, sigma).value;
    const double s2 = sigma * sigma;
    const double expected = (std::exp(s2) - 1.0) * std::exp(2.0 * mu + s2);
    EXPECT_NEAR(d.getVariance(), expected, 1e-10);
}

// log(PDF(x)) == LogPDF(x)
TEST_F(LogNormalEnhancedTest, LogPDFConsistency) {
    for (double x : {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}) {
        const double pdf  = std_ln_.getProbability(x);
        const double lpdf = std_ln_.getLogProbability(x);
        EXPECT_NEAR(std::log(pdf), lpdf, 1e-10) << "at x=" << x;
    }
}

// Out-of-support: PDF and CDF return 0
TEST_F(LogNormalEnhancedTest, OutOfSupport) {
    EXPECT_EQ(std_ln_.getProbability(-1.0), 0.0);
    EXPECT_EQ(std_ln_.getProbability(0.0),  0.0);
    EXPECT_EQ(std_ln_.getCumulativeProbability(-1.0), 0.0);
    EXPECT_EQ(std_ln_.getCumulativeProbability(0.0),  0.0);
    EXPECT_EQ(std_ln_.getLogProbability(-1.0),
              -std::numeric_limits<double>::infinity());
}

// Quantile round-trip: CDF(quantile(p)) = p
TEST_F(LogNormalEnhancedTest, QuantileRoundTrip) {
    for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
        const double q = std_ln_.getQuantile(p);
        EXPECT_GT(q, 0.0);
        EXPECT_NEAR(std_ln_.getCumulativeProbability(q), p, 1e-7) << "at p=" << p;
    }
}

// Batch path matches scalar for interior values
TEST_F(LogNormalEnhancedTest, BatchMatchesScalar) {
    const size_t N = 200;
    vector<double> xs(N), pdf_b(N), lpdf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i) {
        xs[i] = 0.05 + 0.5 * static_cast<double>(i + 1) / static_cast<double>(N);
    }
    std_ln_.getProbability(span<const double>(xs), span<double>(pdf_b));
    std_ln_.getLogProbability(span<const double>(xs), span<double>(lpdf_b));
    std_ln_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pdf_b[i],  std_ln_.getProbability(xs[i]),           1e-10) << "PDF i=" << i;
        EXPECT_NEAR(lpdf_b[i], std_ln_.getLogProbability(xs[i]),        1e-10) << "LogPDF i=" << i;
        // CDF tolerance relaxed to 2e-7: SIMD vector_erf uses A&S approximation
        // (documented max error ~1.5e-7) vs std::erf in the scalar path.
        EXPECT_NEAR(cdf_b[i],  std_ln_.getCumulativeProbability(xs[i]), 2e-7)  << "CDF i=" << i;
    }
}

// VECTORIZED batch matches SCALAR batch (exercises SIMD pipeline)
TEST_F(LogNormalEnhancedTest, VectorizedMatchesScalar) {
    const size_t N = 1024;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i) xs[i] = 0.01 + 0.05 * static_cast<double>(i + 1);

    std_ln_.getLogProbabilityWithStrategy(span<const double>(xs), span<double>(out_vec),
                                          detail::Strategy::VECTORIZED);
    std_ln_.getLogProbabilityWithStrategy(span<const double>(xs), span<double>(out_scl),
                                          detail::Strategy::SCALAR);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "LogPDF SIMD mismatch at i=" << i;
    }

    std_ln_.getCumulativeProbabilityWithStrategy(span<const double>(xs), span<double>(out_vec),
                                                  detail::Strategy::VECTORIZED);
    std_ln_.getCumulativeProbabilityWithStrategy(span<const double>(xs), span<double>(out_scl),
                                                  detail::Strategy::SCALAR);
    for (size_t i = 0; i < N; ++i) {
        // erf SIMD approx vs std::erf; tighter than A&S tolerance for most values
        EXPECT_NEAR(out_vec[i], out_scl[i], 2e-7) << "CDF SIMD mismatch at i=" << i;
    }
}

// MLE fit from LogNormal(μ, σ) samples
TEST_F(LogNormalEnhancedTest, MLEFit) {
    mt19937 rng(42);
    auto source = LogNormalDistribution::create(1.5, 0.4).value;
    const auto data = source.sample(rng, 500);

    auto fitted = LogNormalDistribution::create(0.0, 1.0).value;
    fitted.fit(data);

    EXPECT_NEAR(fitted.getMu(),    1.5, 0.2) << "Fitted mu should be near 1.5";
    EXPECT_NEAR(fitted.getSigma(), 0.4, 0.1) << "Fitted sigma should be near 0.4";
}

// Setter propagates to cache
TEST_F(LogNormalEnhancedTest, SetterPropagates) {
    auto d = LogNormalDistribution::create(0.0, 1.0).value;
    EXPECT_TRUE(d.isStandard());
    d.setMu(1.0);
    EXPECT_FALSE(d.isStandard());
    EXPECT_NEAR(d.getMean(), std::exp(1.0 + 0.5), 1e-12);
    d.setParameters(0.0, 1.0);
    EXPECT_TRUE(d.isStandard());
}

// Invalid parameters rejected
TEST_F(LogNormalEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(LogNormalDistribution::create(0.0, 0.0).isError());
    EXPECT_TRUE(LogNormalDistribution::create(0.0, -1.0).isError());
    EXPECT_TRUE(LogNormalDistribution::create(
        std::numeric_limits<double>::infinity(), 1.0).isError());
    EXPECT_TRUE(LogNormalDistribution::create(
        std::numeric_limits<double>::quiet_NaN(), 1.0).isError());

    auto d = LogNormalDistribution::create(0.0, 1.0).value;
    EXPECT_TRUE(d.trySetSigma(-1.0).isError());
    EXPECT_TRUE(d.trySetMu(std::numeric_limits<double>::infinity()).isError());
    EXPECT_DOUBLE_EQ(d.getMu(),    0.0);
    EXPECT_DOUBLE_EQ(d.getSigma(), 1.0);
}

// Speedup: VECTORIZED batch must complete faster than N scalar calls on large batch
// (labelled timing — only run with ctest -j1)
TEST_F(LogNormalEnhancedTest, VectorizedSpeedup) {
    const size_t N = 50000;
    vector<double> xs(N);
    for (size_t i = 0; i < N; ++i) xs[i] = 0.01 + 0.001 * static_cast<double>(i + 1);
    vector<double> out(N);

    const auto t0 = std::chrono::high_resolution_clock::now();
    std_ln_.getLogProbabilityWithStrategy(span<const double>(xs), span<double>(out),
                                          detail::Strategy::VECTORIZED);
    const auto t1 = std::chrono::high_resolution_clock::now();

    vector<double> scalar_out(N);
    std_ln_.getLogProbabilityWithStrategy(span<const double>(xs), span<double>(scalar_out),
                                          detail::Strategy::SCALAR);
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double vec_us =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    const double scl_us =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());

    const double speedup = scl_us / std::max(vec_us, 1.0);
    std::cout << "LogNormal LogPDF VECTORIZED speedup: " << speedup << "x "
              << "(VECTORIZED " << vec_us << "μs, SCALAR " << scl_us << "μs)\n";

    // Correctness check (more stringent than speedup)
    for (size_t i = 0; i < N; ++i) {
        ASSERT_NEAR(out[i], scalar_out[i], 1e-10) << "mismatch at i=" << i;
    }
    EXPECT_GT(speedup, 1.5) << "VECTORIZED should be at least 1.5x faster than SCALAR";
}

}  // namespace stats
