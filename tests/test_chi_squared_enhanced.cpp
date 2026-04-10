#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/tests.h"
#include "libstats/distributions/chi_squared.h"
#include "libstats/distributions/gamma.h"

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

namespace stats {

//==============================================================================
// TEST FIXTURE
//==============================================================================

class ChiSquaredEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // ChiSquared(k=4): mean=4, variance=8, mode=2
        auto result = stats::ChiSquaredDistribution::create(4.0);
        ASSERT_TRUE(result.isOk());
        dist4_ = std::move(result.value);
    }

    ChiSquaredDistribution dist4_;
};

//==============================================================================
// NUMERIC ACCURACY: chi-squared must match corresponding Gamma exactly
//==============================================================================

TEST_F(ChiSquaredEnhancedTest, DelegationMatchesGamma) {
    // ChiSquared(k) == Gamma(k/2, 0.5) by construction
    const double k = 4.0;
    auto gamma = GammaDistribution::create(k / 2.0, 0.5).value;

    const std::vector<double> xs = {0.01, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 50.0};
    for (double x : xs) {
        EXPECT_NEAR(dist4_.getProbability(x), gamma.getProbability(x), 1e-14)
            << "PDF mismatch at x=" << x;
        EXPECT_NEAR(dist4_.getLogProbability(x), gamma.getLogProbability(x), 1e-12)
            << "LogPDF mismatch at x=" << x;
        EXPECT_NEAR(dist4_.getCumulativeProbability(x), gamma.getCumulativeProbability(x), 1e-12)
            << "CDF mismatch at x=" << x;
    }
}

//==============================================================================
// KNOWN REFERENCE VALUES (k=2: analytically tractable)
//==============================================================================

TEST_F(ChiSquaredEnhancedTest, KnownValuesK2) {
    // ChiSquared(k=2) is Exp(1/2): PDF(x) = 0.5*exp(-x/2) for x > 0
    // CDF(x) = 1 - exp(-x/2)
    auto chi2 = ChiSquaredDistribution::create(2.0).value;

    EXPECT_NEAR(chi2.getMean(), 2.0, 1e-14);
    EXPECT_NEAR(chi2.getVariance(), 4.0, 1e-14);
    EXPECT_NEAR(chi2.getMode(), 0.0, 1e-14);  // max(k-2,0) = 0 for k=2

    const double x = 1.0;
    const double expected_pdf = 0.5 * std::exp(-0.5);
    EXPECT_NEAR(chi2.getProbability(x), expected_pdf, 1e-12);

    const double x2 = 2.0;
    const double expected_cdf = 1.0 - std::exp(-1.0);
    EXPECT_NEAR(chi2.getCumulativeProbability(x2), expected_cdf, 1e-10);

    // Quantile: CDF(q) = 1 - exp(-q/2) = p  =>  q = -2*ln(1-p)
    const double p = 0.95;
    const double expected_q = -2.0 * std::log(1.0 - p);
    EXPECT_NEAR(chi2.getQuantile(p), expected_q, 1e-6);
}

//==============================================================================
// MOMENT PROPERTIES
//==============================================================================

TEST_F(ChiSquaredEnhancedTest, MomentProperties) {
    const double k = 4.0;
    EXPECT_DOUBLE_EQ(dist4_.getMean(), k);
    EXPECT_DOUBLE_EQ(dist4_.getVariance(), 2.0 * k);
    EXPECT_NEAR(dist4_.getSkewness(), std::sqrt(8.0 / k), 1e-12);
    EXPECT_NEAR(dist4_.getKurtosis(), 12.0 / k, 1e-12);
    EXPECT_DOUBLE_EQ(dist4_.getMode(), std::max(k - 2.0, 0.0));
    EXPECT_EQ(dist4_.getNumParameters(), 1);
}

//==============================================================================
// SETTER INVALIDATES DELEGATION
//==============================================================================

TEST_F(ChiSquaredEnhancedTest, SetterPropagates) {
    // After setK, the internal gamma_ must reflect the new value immediately
    auto chi2 = ChiSquaredDistribution::create(2.0).value;
    EXPECT_NEAR(chi2.getMean(), 2.0, 1e-14);

    chi2.setK(6.0);
    EXPECT_NEAR(chi2.getMean(), 6.0, 1e-14);
    EXPECT_NEAR(chi2.getVariance(), 12.0, 1e-14);

    // PDF should correspond to Gamma(3, 0.5) now
    auto gamma6 = GammaDistribution::create(3.0, 0.5).value;
    EXPECT_NEAR(chi2.getProbability(2.0), gamma6.getProbability(2.0), 1e-12);
}

//==============================================================================
// BATCH OPERATION: scalar matches batch element-by-element
//==============================================================================

TEST_F(ChiSquaredEnhancedTest, BatchMatchesScalar) {
    const size_t N = 200;
    std::vector<double> xs(N);
    std::vector<double> pdf_batch(N), logpdf_batch(N), cdf_batch(N);
    for (size_t i = 0; i < N; ++i) {
        xs[i] = 0.1 + static_cast<double>(i) * 0.1;
    }

    dist4_.getProbability(span<const double>(xs), span<double>(pdf_batch));
    dist4_.getLogProbability(span<const double>(xs), span<double>(logpdf_batch));
    dist4_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_batch));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pdf_batch[i], dist4_.getProbability(xs[i]), 1e-12)
            << "PDF batch mismatch at i=" << i << ", x=" << xs[i];
        EXPECT_NEAR(logpdf_batch[i], dist4_.getLogProbability(xs[i]), 1e-10)
            << "LogPDF batch mismatch at i=" << i;
        EXPECT_NEAR(cdf_batch[i], dist4_.getCumulativeProbability(xs[i]), 1e-10)
            << "CDF batch mismatch at i=" << i;
    }
}

//==============================================================================
// FIT: MLE k_hat = sample_mean
//==============================================================================

TEST_F(ChiSquaredEnhancedTest, MLEFit) {
    std::mt19937 rng(42);

    // Generate data from ChiSquared(5) and fit
    auto source = ChiSquaredDistribution::create(5.0).value;
    const auto data = source.sample(rng, 500);

    auto fitted = ChiSquaredDistribution::create(1.0).value;
    fitted.fit(data);

    // With 500 samples, sample mean should be within 10% of 5
    EXPECT_NEAR(fitted.getK(), 5.0, 1.0) << "Fitted k should be close to true k=5";
    EXPECT_GT(fitted.getK(), 0.0);
}

//==============================================================================
// ERROR HANDLING
//==============================================================================

TEST_F(ChiSquaredEnhancedTest, InvalidParameters) {
    // Throwing constructor not tested directly: known ABI exception-unwinding
    // limitation with Homebrew LLVM libc++ on macOS Catalina; use create() instead.

    auto r0 = ChiSquaredDistribution::create(0.0);
    EXPECT_TRUE(r0.isError());

    auto r1 = ChiSquaredDistribution::create(-1.0);
    EXPECT_TRUE(r1.isError());

    auto r2 = ChiSquaredDistribution::create(-5.0);
    EXPECT_TRUE(r2.isError());

    auto r3 = ChiSquaredDistribution::create(std::numeric_limits<double>::quiet_NaN());
    EXPECT_TRUE(r3.isError());

    auto chi2 = ChiSquaredDistribution::create(3.0).value;
    auto vr = chi2.trySetK(-1.0);
    EXPECT_TRUE(vr.isError());
    EXPECT_DOUBLE_EQ(chi2.getK(), 3.0);  // unchanged
}

//==============================================================================
// SUPPORT BOUNDARIES
//==============================================================================

TEST_F(ChiSquaredEnhancedTest, SupportBoundaries) {
    EXPECT_EQ(dist4_.getProbability(-1.0), 0.0);
    EXPECT_EQ(dist4_.getLogProbability(-0.001),
              dist4_.getLogProbability(-1.0));  // both out-of-support
    EXPECT_EQ(dist4_.getCumulativeProbability(0.0), 0.0);
    EXPECT_NEAR(dist4_.getCumulativeProbability(1e6), 1.0, 1e-10);
}

}  // namespace stats
