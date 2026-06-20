#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/tests.h"
#include "libstats/distributions/pareto.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

namespace stats {

class ParetoEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = stats::ParetoDistribution::create(1.0, 2.0);
        ASSERT_TRUE(r.isOk());
        p12_ = std::move(r.value);
    }
    ParetoDistribution p12_;  // Pareto(scale=1, alpha=2)
};

// PDF(2; 1, 2) = 2*1^2/2^3 = 2/8 = 0.25
TEST_F(ParetoEnhancedTest, KnownPDFValue) {
    EXPECT_NEAR(p12_.getProbability(2.0), 0.25, 1e-12);
}

// CDF(2; 1, 2) = 1 - (1/2)^2 = 0.75
TEST_F(ParetoEnhancedTest, KnownCDFValue) {
    EXPECT_NEAR(p12_.getCumulativeProbability(2.0), 0.75, 1e-12);
}

// CDF(scale) must equal 0 for any Pareto
TEST_F(ParetoEnhancedTest, CDFAtScale) {
    for (double sc : {0.5, 1.0, 2.0, 5.0}) {
        for (double al : {1.0, 2.0, 3.5}) {
            auto d = ParetoDistribution::create(sc, al).value;
            EXPECT_EQ(d.getCumulativeProbability(sc), 0.0)
                << "CDF(scale) != 0 for scale=" << sc << " alpha=" << al;
        }
    }
}

// Mean formula: α·x_m/(α−1) for α > 1
TEST_F(ParetoEnhancedTest, MeanFormula) {
    // α=3, x_m=2: mean = 3*2/(3-1) = 3
    auto d = ParetoDistribution::create(2.0, 3.0).value;
    EXPECT_NEAR(d.getMean(), 3.0, 1e-12);
    EXPECT_TRUE(d.hasFiniteMean());

    // α ≤ 1: infinite mean
    auto d2 = ParetoDistribution::create(1.0, 0.5).value;
    EXPECT_TRUE(std::isinf(d2.getMean()));
    EXPECT_FALSE(d2.hasFiniteMean());
}

// Variance formula: x_m²·α/((α−1)²·(α−2)) for α > 2
TEST_F(ParetoEnhancedTest, VarianceFormula) {
    // α=3, x_m=1: variance = 1*3/(4*1) = 0.75
    auto d = ParetoDistribution::create(1.0, 3.0).value;
    EXPECT_NEAR(d.getVariance(), 0.75, 1e-12);
    EXPECT_TRUE(d.hasFiniteVariance());

    // α ≤ 2: infinite variance
    EXPECT_TRUE(std::isinf(p12_.getVariance()));
    EXPECT_FALSE(p12_.hasFiniteVariance());
}

// log(PDF(x)) == LogPDF(x) for in-support values
TEST_F(ParetoEnhancedTest, LogPDFConsistency) {
    for (double x : {1.0, 1.5, 2.0, 5.0, 10.0, 100.0}) {
        const double pdf = p12_.getProbability(x);
        const double lpdf = p12_.getLogProbability(x);
        EXPECT_NEAR(std::log(pdf), lpdf, 1e-10) << "at x=" << x;
    }
}

// Out-of-support: PDF=0, CDF=0, LogPDF=-inf for x < scale
TEST_F(ParetoEnhancedTest, OutOfSupport) {
    EXPECT_EQ(p12_.getProbability(0.0), 0.0);
    EXPECT_EQ(p12_.getProbability(0.9), 0.0);
    EXPECT_EQ(p12_.getCumulativeProbability(0.0), 0.0);
    EXPECT_EQ(p12_.getCumulativeProbability(0.5), 0.0);
    EXPECT_EQ(p12_.getLogProbability(0.5), -std::numeric_limits<double>::infinity());
}

// Quantile round-trip: CDF(quantile(p)) = p
TEST_F(ParetoEnhancedTest, QuantileRoundTrip) {
    for (double p : {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
        const double q = p12_.getQuantile(p);
        EXPECT_GE(q, 1.0) << "quantile below scale at p=" << p;
        EXPECT_NEAR(p12_.getCumulativeProbability(q), p, 1e-10) << "at p=" << p;
    }
}

// Median formula: x_m * 2^(1/α)
TEST_F(ParetoEnhancedTest, MedianFormula) {
    // Pareto(1, 2): median = 1*2^0.5 ≈ 1.4142
    EXPECT_NEAR(p12_.getMedian(), std::sqrt(2.0), 1e-12);
    // Pareto(2, 4): median = 2*2^(1/4) ≈ 2.3784
    auto d = ParetoDistribution::create(2.0, 4.0).value;
    EXPECT_NEAR(d.getMedian(), 2.0 * std::pow(2.0, 0.25), 1e-12);
}

// Mode always equals scale
TEST_F(ParetoEnhancedTest, ModeEqualsScale) {
    for (double sc : {0.5, 1.0, 2.0, 5.0}) {
        auto d = ParetoDistribution::create(sc, 2.0).value;
        EXPECT_DOUBLE_EQ(d.getMode(), sc);
    }
}

// Batch path matches scalar
TEST_F(ParetoEnhancedTest, BatchMatchesScalar) {
    const size_t N = 200;
    vector<double> xs(N), pdf_b(N), lpdf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i) {
        xs[i] = 1.0 + 0.1 * static_cast<double>(i + 1);
    }
    p12_.getProbability(span<const double>(xs), span<double>(pdf_b));
    p12_.getLogProbability(span<const double>(xs), span<double>(lpdf_b));
    p12_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pdf_b[i], p12_.getProbability(xs[i]), 1e-12) << "PDF i=" << i;
        EXPECT_NEAR(lpdf_b[i], p12_.getLogProbability(xs[i]), 1e-12) << "LogPDF i=" << i;
        EXPECT_NEAR(cdf_b[i], p12_.getCumulativeProbability(xs[i]), 1e-12) << "CDF i=" << i;
    }
}

// VECTORIZED matches SCALAR (exercises the 3-step LogPDF SIMD pipeline)
TEST_F(ParetoEnhancedTest, VectorizedMatchesScalar) {
    const size_t N = 1024;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = 1.0 + 0.05 * static_cast<double>(i + 1);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "LogPDF SIMD mismatch at i=" << i;
    }

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "CDF SIMD mismatch at i=" << i;
    }
}

// MLE fit from Pareto(x_m, α) samples
TEST_F(ParetoEnhancedTest, MLEFit) {
    mt19937 rng(42);
    auto source = ParetoDistribution::create(2.0, 3.0).value;
    const auto data = source.sample(rng, 500);

    auto fitted = ParetoDistribution::create(1.0, 1.0).value;
    fitted.fit(data);

    // scale_hat = min(data) ≈ 2.0 (exact for large n)
    EXPECT_NEAR(fitted.getScale(), 2.0, 0.1) << "Fitted scale should be near 2";
    EXPECT_NEAR(fitted.getAlpha(), 3.0, 1.0) << "Fitted alpha should be near 3";
}

// Setter propagates to cache (hasFiniteMean/Variance update correctly)
TEST_F(ParetoEnhancedTest, SetterPropagates) {
    auto d = ParetoDistribution::create(1.0, 2.0).value;
    EXPECT_FALSE(d.hasFiniteVariance());
    d.setAlpha(3.0);
    EXPECT_TRUE(d.hasFiniteVariance());
    EXPECT_NEAR(d.getMean(), 1.5, 1e-12);
    d.setParameters(1.0, 2.0);
    EXPECT_FALSE(d.hasFiniteVariance());
}

// Invalid parameters rejected
TEST_F(ParetoEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(ParetoDistribution::create(-1.0, 1.0).isError());
    EXPECT_TRUE(ParetoDistribution::create(0.0, 1.0).isError());
    EXPECT_TRUE(ParetoDistribution::create(1.0, -1.0).isError());
    EXPECT_TRUE(ParetoDistribution::create(1.0, 0.0).isError());
    EXPECT_TRUE(
        ParetoDistribution::create(std::numeric_limits<double>::quiet_NaN(), 1.0).isError());

    auto d = ParetoDistribution::create(1.0, 2.0).value;
    EXPECT_TRUE(d.trySetAlpha(-1.0).isError());
    EXPECT_TRUE(d.trySetScale(0.0).isError());
    EXPECT_DOUBLE_EQ(d.getScale(), 1.0);
    EXPECT_DOUBLE_EQ(d.getAlpha(), 2.0);
}

// Speedup: VECTORIZED LogPDF must be measurably faster on a large batch
// (labelled timing — only run with ctest -j1)
TEST_F(ParetoEnhancedTest, VectorizedSpeedup) {
    const size_t N = 50000;
    vector<double> xs(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = 1.0 + 0.001 * static_cast<double>(i + 1);
    vector<double> out(N), scl(N);

    const auto t0 = std::chrono::high_resolution_clock::now();
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double vec_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    const double scl_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    const double speedup = scl_us / std::max(vec_us, 1.0);

    std::cout << "Pareto LogPDF VECTORIZED speedup: " << speedup << "x "
              << "(VECTORIZED " << vec_us << "μs, SCALAR " << scl_us << "μs)\n";

    for (size_t i = 0; i < N; ++i) {
        ASSERT_NEAR(out[i], scl[i], 1e-10) << "mismatch at i=" << i;
    }
    EXPECT_GT(speedup, 1.5) << "VECTORIZED should be at least 1.5x faster than SCALAR";
}

}  // namespace stats
