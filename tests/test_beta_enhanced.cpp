#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/tests.h"
#include "libstats/distributions/beta.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

namespace stats {

class BetaEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = stats::BetaDistribution::create(2.0, 3.0);
        ASSERT_TRUE(r.isOk());
        dist23_ = std::move(r.value);  // mean=0.4, mode=0.25
    }
    BetaDistribution dist23_;
};

// Beta(1,1)=Uniform: PDF=1 everywhere in (0,1)
TEST_F(BetaEnhancedTest, UniformCase) {
    auto b11 = BetaDistribution::create(1.0, 1.0).value;
    EXPECT_TRUE(b11.isUniform());
    EXPECT_NEAR(b11.getProbability(0.1), 1.0, 1e-10);
    EXPECT_NEAR(b11.getProbability(0.5), 1.0, 1e-10);
    EXPECT_NEAR(b11.getProbability(0.9), 1.0, 1e-10);
    EXPECT_NEAR(b11.getMean(), 0.5, 1e-14);
    EXPECT_NEAR(b11.getVariance(), 1.0 / 12.0, 1e-12);
}

// Analytical PDF values for Beta(2,2): f(x)=6x(1-x)
TEST_F(BetaEnhancedTest, KnownValuesBeta22) {
    auto b22 = BetaDistribution::create(2.0, 2.0).value;
    EXPECT_TRUE(b22.isSymmetric());
    EXPECT_NEAR(b22.getProbability(0.5), 1.5, 1e-10);  // 6*0.5*0.5
    EXPECT_NEAR(b22.getProbability(0.25), 6.0 * 0.25 * 0.75, 1e-10);
    EXPECT_NEAR(b22.getCumulativeProbability(0.5), 0.5, 1e-8);  // symmetry
    EXPECT_NEAR(b22.getMean(), 0.5, 1e-14);
    EXPECT_NEAR(b22.getMode(), 0.5, 1e-14);  // (2-1)/(2+2-2)=0.5
}

// CDF symmetry: Beta(a,b) CDF(0.5) = 0.5 when a=b; CDF(-x) complement
TEST_F(BetaEnhancedTest, CDFSymmetry) {
    for (double a : {0.5, 1.0, 2.0, 3.0, 5.0}) {
        auto bd = BetaDistribution::create(a, a).value;
        EXPECT_NEAR(bd.getCumulativeProbability(0.5), 0.5, 1e-8) << "CDF(0.5) != 0.5 for a=b=" << a;
    }
    // CDF(x, a, b) + CDF(1-x, b, a) = 1
    auto b23 = BetaDistribution::create(2.0, 3.0).value;
    auto b32 = BetaDistribution::create(3.0, 2.0).value;
    for (double x : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        EXPECT_NEAR(b23.getCumulativeProbability(x) + b32.getCumulativeProbability(1.0 - x), 1.0,
                    1e-8)
            << "CDF reflection failed at x=" << x;
    }
}

// Moments: mean = a/(a+b), variance = ab/((a+b)^2(a+b+1))
TEST_F(BetaEnhancedTest, MomentProperties) {
    const double a = 2.0, b = 3.0;
    const double ab = a + b;
    EXPECT_NEAR(dist23_.getMean(), a / ab, 1e-14);
    EXPECT_NEAR(dist23_.getVariance(), a * b / (ab * ab * (ab + 1.0)), 1e-14);
    EXPECT_NEAR(dist23_.getMode(), (a - 1.0) / (ab - 2.0), 1e-14);  // (2-1)/(5-2) = 1/3
    EXPECT_EQ(dist23_.getNumParameters(), 2);
    EXPECT_NEAR(dist23_.getSupportLowerBound(), 0.0, 1e-14);
    EXPECT_NEAR(dist23_.getSupportUpperBound(), 1.0, 1e-14);
}

// log(PDF) == LogPDF for interior values
TEST_F(BetaEnhancedTest, LogPDFConsistency) {
    const vector<double> xs = {0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99};
    for (double x : xs) {
        const double pdf = dist23_.getProbability(x);
        EXPECT_NEAR(std::log(pdf), dist23_.getLogProbability(x), 1e-10) << "at x=" << x;
    }
}

// Batch path must match scalar for interior values
TEST_F(BetaEnhancedTest, BatchMatchesScalar) {
    const size_t N = 200;
    vector<double> xs(N), pdf_b(N), logpdf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i) {
        xs[i] = 0.005 + static_cast<double>(i) * 0.99 / static_cast<double>(N - 1);
    }
    dist23_.getProbability(span<const double>(xs), span<double>(pdf_b));
    dist23_.getLogProbability(span<const double>(xs), span<double>(logpdf_b));
    dist23_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pdf_b[i], dist23_.getProbability(xs[i]), 1e-10) << "PDF i=" << i;
        EXPECT_NEAR(logpdf_b[i], dist23_.getLogProbability(xs[i]), 1e-10) << "LogPDF i=" << i;
        EXPECT_NEAR(cdf_b[i], dist23_.getCumulativeProbability(xs[i]), 1e-8) << "CDF i=" << i;
    }
}

// Quantile round-trip: beta_i(inverse_beta_i(p)) = p
TEST_F(BetaEnhancedTest, QuantileRoundTrip) {
    for (double p : {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95}) {
        const double q = dist23_.getQuantile(p);
        EXPECT_GT(q, 0.0);
        EXPECT_LT(q, 1.0);
        EXPECT_NEAR(dist23_.getCumulativeProbability(q), p, 1e-6) << "at p=" << p;
    }
}

// Setter propagates to cache
TEST_F(BetaEnhancedTest, SetterPropagates) {
    auto b = BetaDistribution::create(2.0, 2.0).value;
    EXPECT_TRUE(b.isSymmetric());
    EXPECT_NEAR(b.getMean(), 0.5, 1e-14);
    b.setAlpha(3.0);
    EXPECT_NEAR(b.getMean(), 3.0 / 5.0, 1e-14);
    EXPECT_FALSE(b.isSymmetric());
    b.setParameters(1.0, 1.0);
    EXPECT_TRUE(b.isUniform());
}

// MLE fit from Beta(3,5) samples
TEST_F(BetaEnhancedTest, MLEFit) {
    mt19937 rng(42);
    auto source = BetaDistribution::create(3.0, 5.0).value;
    const auto data = source.sample(rng, 500);

    auto fitted = BetaDistribution::create(1.0, 1.0).value;
    fitted.fit(data);

    EXPECT_NEAR(fitted.getAlpha(), 3.0, 1.5) << "Fitted alpha should be near 3";
    EXPECT_NEAR(fitted.getBeta(), 5.0, 2.5) << "Fitted beta should be near 5";
}

// Invalid parameters rejected
TEST_F(BetaEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(BetaDistribution::create(0.0, 1.0).isError());
    EXPECT_TRUE(BetaDistribution::create(-1.0, 1.0).isError());
    EXPECT_TRUE(BetaDistribution::create(1.0, 0.0).isError());
    EXPECT_TRUE(BetaDistribution::create(1.0, -1.0).isError());
    EXPECT_TRUE(BetaDistribution::create(std::numeric_limits<double>::quiet_NaN(), 1.0).isError());

    auto b = BetaDistribution::create(2.0, 3.0).value;
    EXPECT_TRUE(b.trySetAlpha(-1.0).isError());
    EXPECT_TRUE(b.trySetBeta(-1.0).isError());
    EXPECT_DOUBLE_EQ(b.getAlpha(), 2.0);
    EXPECT_DOUBLE_EQ(b.getBeta(), 3.0);
}

// Out-of-support: PDF=0 outside [0,1]
TEST_F(BetaEnhancedTest, OutOfSupport) {
    EXPECT_EQ(dist23_.getProbability(-0.1), 0.0);
    EXPECT_EQ(dist23_.getProbability(1.1), 0.0);
    EXPECT_EQ(dist23_.getCumulativeProbability(0.0), 0.0);
    EXPECT_NEAR(dist23_.getCumulativeProbability(1.0), 1.0, 1e-14);
}

}  // namespace stats
