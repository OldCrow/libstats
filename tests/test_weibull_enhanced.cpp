#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/tests.h"
#include "libstats/distributions/weibull.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

namespace stats {

class WeibullEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = stats::WeibullDistribution::create(2.0, 1.0);
        ASSERT_TRUE(r.isOk());
        w21_ = std::move(r.value);  // Weibull(k=2, λ=1)
    }
    WeibullDistribution w21_;
};

// Weibull(1,1) = Exponential(rate=1): PDF(1) = exp(-1)
TEST_F(WeibullEnhancedTest, ExponentialSpecialCase) {
    auto w11 = WeibullDistribution::create(1.0, 1.0).value;
    EXPECT_TRUE(w11.isExponential());
    EXPECT_NEAR(w11.getProbability(1.0), std::exp(-1.0), 1e-12);
    EXPECT_NEAR(w11.getCumulativeProbability(1.0), 1.0 - std::exp(-1.0), 1e-12);
    EXPECT_NEAR(w11.getMean(), 1.0, 1e-12);
    EXPECT_NEAR(w11.getVariance(), 1.0, 1e-12);
}

// Weibull(2,1): PDF(1) = 2*exp(-1), CDF(1) = 1-exp(-1)
TEST_F(WeibullEnhancedTest, KnownValues) {
    EXPECT_NEAR(w21_.getProbability(1.0), 2.0 * std::exp(-1.0), 1e-12);
    EXPECT_NEAR(w21_.getCumulativeProbability(1.0), 1.0 - std::exp(-1.0), 1e-12);
    EXPECT_FALSE(w21_.isExponential());
}

// CDF(λ) = 1 - 1/e for any (k, λ)
TEST_F(WeibullEnhancedTest, CDFAtScale) {
    for (double k : {0.5, 1.0, 2.0, 3.0, 5.0}) {
        for (double lam : {0.5, 1.0, 2.0}) {
            auto d = WeibullDistribution::create(k, lam).value;
            EXPECT_NEAR(d.getCumulativeProbability(lam), 1.0 - std::exp(-1.0), 1e-12)
                << "CDF(λ) != 1-1/e for k=" << k << " λ=" << lam;
        }
    }
}

// Mean = λ·Γ(1+1/k); Variance = λ²·(Γ(1+2/k) - Γ(1+1/k)²)
TEST_F(WeibullEnhancedTest, MomentFormulas) {
    const double g1 = std::exp(std::lgamma(1.5));  // Γ(1+1/2) = Γ(1.5) = √π/2
    const double g2 = std::exp(std::lgamma(2.0));  // Γ(1+2/2) = Γ(2) = 1
    EXPECT_NEAR(w21_.getMean(), g1, 1e-12);
    EXPECT_NEAR(w21_.getVariance(), g2 - g1 * g1, 1e-12);
}

// log(PDF) == LogPDF
TEST_F(WeibullEnhancedTest, LogPDFConsistency) {
    for (double x : {0.1, 0.5, 1.0, 2.0, 5.0}) {
        const double pdf  = w21_.getProbability(x);
        const double lpdf = w21_.getLogProbability(x);
        EXPECT_NEAR(std::log(pdf), lpdf, 1e-10) << "at x=" << x;
    }
}

// Out-of-support
TEST_F(WeibullEnhancedTest, OutOfSupport) {
    EXPECT_EQ(w21_.getProbability(-1.0), 0.0);
    EXPECT_EQ(w21_.getCumulativeProbability(-1.0), 0.0);
    EXPECT_EQ(w21_.getCumulativeProbability(0.0), 0.0);
    EXPECT_EQ(w21_.getLogProbability(-1.0), -std::numeric_limits<double>::infinity());
}

// Quantile round-trip
TEST_F(WeibullEnhancedTest, QuantileRoundTrip) {
    for (double p : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9}) {
        const double q = w21_.getQuantile(p);
        EXPECT_GE(q, 0.0);
        EXPECT_NEAR(w21_.getCumulativeProbability(q), p, 1e-10) << "at p=" << p;
    }
}

// Batch matches scalar
TEST_F(WeibullEnhancedTest, BatchMatchesScalar) {
    const size_t N = 200;
    vector<double> xs(N), pdf_b(N), lpdf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i) xs[i] = 0.05 + 0.05 * static_cast<double>(i + 1);
    w21_.getProbability(span<const double>(xs), span<double>(pdf_b));
    w21_.getLogProbability(span<const double>(xs), span<double>(lpdf_b));
    w21_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pdf_b[i],  w21_.getProbability(xs[i]),           1e-12) << "PDF i=" << i;
        EXPECT_NEAR(lpdf_b[i], w21_.getLogProbability(xs[i]),        1e-12) << "LogPDF i=" << i;
        EXPECT_NEAR(cdf_b[i],  w21_.getCumulativeProbability(xs[i]), 1e-12) << "CDF i=" << i;
    }
}

// VECTORIZED matches SCALAR
TEST_F(WeibullEnhancedTest, VectorizedMatchesScalar) {
    const size_t N = 1024;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i) xs[i] = 0.01 + 0.01 * static_cast<double>(i + 1);

    w21_.getLogProbabilityWithStrategy(span<const double>(xs), span<double>(out_vec),
                                       detail::Strategy::VECTORIZED);
    w21_.getLogProbabilityWithStrategy(span<const double>(xs), span<double>(out_scl),
                                       detail::Strategy::SCALAR);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "LogPDF SIMD mismatch at i=" << i;

    w21_.getCumulativeProbabilityWithStrategy(span<const double>(xs), span<double>(out_vec),
                                               detail::Strategy::VECTORIZED);
    w21_.getCumulativeProbabilityWithStrategy(span<const double>(xs), span<double>(out_scl),
                                               detail::Strategy::SCALAR);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "CDF SIMD mismatch at i=" << i;
}

// MLE from samples
TEST_F(WeibullEnhancedTest, MLEFit) {
    mt19937 rng(42);
    auto source = WeibullDistribution::create(2.5, 3.0).value;
    const auto data = source.sample(rng, 500);
    auto fitted = WeibullDistribution::create(1.0, 1.0).value;
    fitted.fit(data);
    EXPECT_NEAR(fitted.getShape(), 2.5, 0.5) << "Fitted shape should be near 2.5";
    EXPECT_NEAR(fitted.getScale(), 3.0, 1.0) << "Fitted scale should be near 3.0";
}

// Setter propagates
TEST_F(WeibullEnhancedTest, SetterPropagates) {
    auto d = WeibullDistribution::create(1.0, 1.0).value;
    EXPECT_TRUE(d.isExponential());
    d.setShape(2.0);
    EXPECT_FALSE(d.isExponential());
    d.setParameters(1.0, 1.0);
    EXPECT_TRUE(d.isExponential());
}

// Invalid parameters rejected
TEST_F(WeibullEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(WeibullDistribution::create(0.0, 1.0).isError());
    EXPECT_TRUE(WeibullDistribution::create(-1.0, 1.0).isError());
    EXPECT_TRUE(WeibullDistribution::create(1.0, 0.0).isError());
    EXPECT_TRUE(WeibullDistribution::create(
        std::numeric_limits<double>::quiet_NaN(), 1.0).isError());

    auto d = WeibullDistribution::create(2.0, 1.0).value;
    EXPECT_TRUE(d.trySetShape(-1.0).isError());
    EXPECT_TRUE(d.trySetScale(0.0).isError());
    EXPECT_DOUBLE_EQ(d.getShape(), 2.0);
    EXPECT_DOUBLE_EQ(d.getScale(), 1.0);
}

// Speedup: VECTORIZED LogPDF must be measurably faster (labelled timing)
TEST_F(WeibullEnhancedTest, VectorizedSpeedup) {
    const size_t N = 50000;
    vector<double> xs(N);
    for (size_t i = 0; i < N; ++i) xs[i] = 0.01 + 0.001 * static_cast<double>(i + 1);
    vector<double> out(N), scl(N);

    const auto t0 = std::chrono::high_resolution_clock::now();
    w21_.getLogProbabilityWithStrategy(span<const double>(xs), span<double>(out),
                                       detail::Strategy::VECTORIZED);
    const auto t1 = std::chrono::high_resolution_clock::now();
    w21_.getLogProbabilityWithStrategy(span<const double>(xs), span<double>(scl),
                                       detail::Strategy::SCALAR);
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double vec_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    const double scl_us = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    const double speedup = scl_us / std::max(vec_us, 1.0);
    std::cout << "Weibull LogPDF VECTORIZED speedup: " << speedup << "x "
              << "(VECTORIZED " << vec_us << "μs, SCALAR " << scl_us << "μs)\n";

    for (size_t i = 0; i < N; ++i)
        ASSERT_NEAR(out[i], scl[i], 1e-10) << "mismatch at i=" << i;
    EXPECT_GT(speedup, 1.5) << "VECTORIZED should be at least 1.5x faster";
}

}  // namespace stats
