#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/erlang.h"
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

class ErlangEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Erlang(k=4, lambda=2): mean=2, variance=1, mode=1.5
        auto result = stats::ErlangDistribution::create(4, 2.0);
        ASSERT_TRUE(result.isOk());
        dist4_ = std::move(result).unwrap();
    }

    ErlangDistribution dist4_;
};

//==============================================================================
// NUMERIC ACCURACY: Erlang must match corresponding Gamma exactly (finite x)
//==============================================================================

TEST_F(ErlangEnhancedTest, DelegationMatchesGamma) {
    const int k = 4;
    const double lambda = 2.0;
    auto gamma = GammaDistribution::create(static_cast<double>(k), lambda).unwrap();

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
// KNOWN REFERENCE VALUES (k=1: Erlang(1,lambda) = Exponential(lambda))
//==============================================================================

TEST_F(ErlangEnhancedTest, KnownValuesK1) {
    auto erl = ErlangDistribution::create(1, 2.0).unwrap();

    EXPECT_NEAR(erl.getMean(), 0.5, 1e-14);
    EXPECT_NEAR(erl.getVariance(), 0.25, 1e-14);
    EXPECT_NEAR(erl.getMode(), 0.0, 1e-14);  // (k-1)/lambda = 0 for k=1

    const double x = 1.0;
    const double expected_pdf = 2.0 * std::exp(-2.0);
    EXPECT_NEAR(erl.getProbability(x), expected_pdf, 1e-12);

    const double expected_cdf = 1.0 - std::exp(-2.0);
    EXPECT_NEAR(erl.getCumulativeProbability(x), expected_cdf, 1e-9);

    // Quantile: CDF(q) = 1 - exp(-lambda*q) = p  =>  q = -ln(1-p)/lambda
    const double p = 0.95;
    const double expected_q = -std::log(1.0 - p) / 2.0;
    EXPECT_NEAR(erl.getQuantile(p), expected_q, 1e-6);
}

//==============================================================================
// MOMENT PROPERTIES
//==============================================================================

TEST_F(ErlangEnhancedTest, MomentProperties) {
    const double k = 4.0;
    const double lambda = 2.0;
    EXPECT_DOUBLE_EQ(dist4_.getMean(), k / lambda);
    EXPECT_DOUBLE_EQ(dist4_.getVariance(), k / (lambda * lambda));
    EXPECT_NEAR(dist4_.getSkewness(), 2.0 / std::sqrt(k), 1e-12);
    EXPECT_NEAR(dist4_.getKurtosis(), 6.0 / k, 1e-12);
    EXPECT_DOUBLE_EQ(dist4_.getMode(), (k - 1.0) / lambda);
    EXPECT_EQ(dist4_.getNumParameters(), 2);
}

//==============================================================================
// SETTER INVALIDATES DELEGATION
//==============================================================================

TEST_F(ErlangEnhancedTest, SetterPropagates) {
    auto erl = ErlangDistribution::create(2, 1.0).unwrap();
    EXPECT_NEAR(erl.getMean(), 2.0, 1e-14);

    erl.setLambda(2.0);
    EXPECT_NEAR(erl.getMean(), 1.0, 1e-14);

    erl.setK(6);
    EXPECT_NEAR(erl.getMean(), 3.0, 1e-14);

    auto gamma6 = GammaDistribution::create(6.0, 2.0).unwrap();
    EXPECT_NEAR(erl.getProbability(2.0), gamma6.getProbability(2.0), 1e-12);
}

//==============================================================================
// BATCH OPERATION: scalar matches batch element-by-element (finite x)
//==============================================================================

TEST_F(ErlangEnhancedTest, BatchMatchesScalar) {
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
// #103: PDF/LogPDF/CDF at +-inf and NaN, scalar AND batch
//
// GammaDistribution::getProbability/getLogProbability have no top-of-function
// isfinite(x) guard; for alpha>=1 (always true for Erlang, alpha=k>=1) the
// log-space formula evaluates 0*inf / inf-inf (both NaN per IEEE 754) at
// x=+inf. ErlangDistribution guards this itself -- these tests pin down that
// the guard actually holds, scalar and batch. See erlang.h's class-level
// "±inf / NaN handling" note and erlang.cpp.
//==============================================================================

TEST_F(ErlangEnhancedTest, InfAndNaNContractScalar) {
    const double inf = std::numeric_limits<double>::infinity();
    const double ninf = -std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // k=4 (alpha=4 > 1): this is exactly the inf-inf case that motivated the guard.
    EXPECT_EQ(dist4_.getProbability(inf), 0.0);
    EXPECT_EQ(dist4_.getProbability(ninf), 0.0);
    EXPECT_TRUE(std::isnan(dist4_.getProbability(nan)));

    EXPECT_EQ(dist4_.getLogProbability(inf), ninf);
    EXPECT_EQ(dist4_.getLogProbability(ninf), ninf);
    EXPECT_TRUE(std::isnan(dist4_.getLogProbability(nan)));

    EXPECT_EQ(dist4_.getCumulativeProbability(ninf), 0.0);
    EXPECT_NEAR(dist4_.getCumulativeProbability(inf), 1.0, 1e-12);
    EXPECT_TRUE(std::isnan(dist4_.getCumulativeProbability(nan)));

    // k=1 (alpha=1): the 0*inf case -- also exercised, since it is a distinct
    // code path in Gamma's formula (alpha_minus_one == 0 exactly).
    auto erl_k1 = ErlangDistribution::create(1, 1.0).unwrap();
    EXPECT_EQ(erl_k1.getProbability(inf), 0.0);
    EXPECT_EQ(erl_k1.getLogProbability(inf), ninf);
}

TEST_F(ErlangEnhancedTest, InfAndNaNContractBatch) {
    const double inf = std::numeric_limits<double>::infinity();
    const double ninf = -std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> xs = {0.5, 2.0, inf, ninf, nan, 5.0};
    const size_t N = xs.size();
    std::vector<double> pdf_b(N), lpdf_b(N), cdf_b(N);

    dist4_.getProbability(std::span<const double>(xs), std::span<double>(pdf_b));
    dist4_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf_b));
    dist4_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        const bool pdf_scalar_nan = std::isnan(dist4_.getProbability(xs[i]));
        EXPECT_EQ(std::isnan(pdf_b[i]), pdf_scalar_nan) << "PDF batch/scalar NaN mismatch i=" << i;
        if (!pdf_scalar_nan)
            EXPECT_NEAR(pdf_b[i], dist4_.getProbability(xs[i]), 1e-12)
                << "PDF batch/scalar mismatch i=" << i;

        const bool lpdf_scalar_nan = std::isnan(dist4_.getLogProbability(xs[i]));
        EXPECT_EQ(std::isnan(lpdf_b[i]), lpdf_scalar_nan)
            << "LogPDF batch/scalar NaN mismatch i=" << i;
        if (!lpdf_scalar_nan)
            EXPECT_NEAR(lpdf_b[i], dist4_.getLogProbability(xs[i]), 1e-10)
                << "LogPDF batch/scalar mismatch i=" << i;

        const bool cdf_scalar_nan = std::isnan(dist4_.getCumulativeProbability(xs[i]));
        EXPECT_EQ(std::isnan(cdf_b[i]), cdf_scalar_nan) << "CDF batch/scalar NaN mismatch i=" << i;
        if (!cdf_scalar_nan)
            EXPECT_NEAR(cdf_b[i], dist4_.getCumulativeProbability(xs[i]), 1e-10)
                << "CDF batch/scalar mismatch i=" << i;
    }
    // Explicit finite-value pins at the non-finite indices (2=+inf, 3=-inf, 4=NaN)
    EXPECT_EQ(pdf_b[2], 0.0) << "PDF(+inf)";
    EXPECT_EQ(pdf_b[3], 0.0) << "PDF(-inf)";
    EXPECT_TRUE(std::isnan(pdf_b[4])) << "PDF(NaN)";
    EXPECT_EQ(lpdf_b[2], ninf) << "LogPDF(+inf)";
    EXPECT_EQ(lpdf_b[3], ninf) << "LogPDF(-inf)";
    EXPECT_TRUE(std::isnan(lpdf_b[4])) << "LogPDF(NaN)";
    EXPECT_NEAR(cdf_b[2], 1.0, 1e-12) << "CDF(+inf)";
    EXPECT_EQ(cdf_b[3], 0.0) << "CDF(-inf)";
    EXPECT_TRUE(std::isnan(cdf_b[4])) << "CDF(NaN)";
}

TEST_F(ErlangEnhancedTest, ExtremeTailQuantileFinite) {
    // #104 gate, inherited from Gamma by pure delegation: the v2.4.0 sweep's
    // erlang(10000, 1e-3) instance hits Gamma's extreme-tail quantile escape
    // to +inf (small-p asymptotic seed overflow for α ≳ 170; see the same
    // gate in test_gamma_enhanced.cpp). Reference: mpmath dps=60.
    auto e = ErlangDistribution::create(10000, 1e-3).unwrap();
    auto g = GammaDistribution::create(10000.0, 1e-3).unwrap();
    const double ps[] = {1e-300, 1e-200, 1e-15, 0.5, 1.0 - 1e-12};
    for (double p : ps) {
        const double qe = e.getQuantile(p);
        EXPECT_TRUE(std::isfinite(qe)) << "erlang quantile(" << p << ") not finite: " << qe;
        // Pure delegation: bit-identical to the underlying Gamma.
        EXPECT_EQ(qe, g.getQuantile(p)) << "delegation mismatch at p=" << p;
    }
    EXPECT_NEAR(e.getQuantile(1e-300) / 6737687.1915903291, 1.0, 1e-9);
}

//==============================================================================
// #104: quantile never NaN for p in (0,1)
//==============================================================================

TEST_F(ErlangEnhancedTest, QuantileNeverNaN) {
    for (double p : {0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999}) {
        const double q = dist4_.getQuantile(p);
        EXPECT_FALSE(std::isnan(q)) << "quantile must never be NaN for p=" << p;
        EXPECT_GE(q, 0.0);
        EXPECT_NEAR(dist4_.getCumulativeProbability(q), p, 1e-6) << "CDF(quantile(" << p << "))";
    }
}

//==============================================================================
// FIT: method of moments, k_hat/lambda_hat, and the #125-class int guard
//==============================================================================

TEST_F(ErlangEnhancedTest, MLEFit) {
    std::mt19937 rng(42);

    auto source = ErlangDistribution::create(5, 2.0).unwrap();  // mean=2.5
    const auto data = source.sample(rng, 2000);

    auto fitted = ErlangDistribution::create(1, 1.0).unwrap();
    fitted.fit(data);

    EXPECT_GE(fitted.getK(), 1);
    EXPECT_NEAR(fitted.getK(), 5, 3) << "Fitted k should be roughly close to true k=5";
    EXPECT_GT(fitted.getLambda(), 0.0);
    EXPECT_NEAR(fitted.getMean(), 2.5, 0.5) << "Fitted mean should be close to true mean 2.5";
}

TEST_F(ErlangEnhancedTest, MLEFitDegenerateDataFallsBackToKOne) {
    // Near-zero variance -> k_hat should fall back to 1, not diverge/UB.
    auto fitted = ErlangDistribution::create(9, 1.0).unwrap();
    std::vector<double> constant_data(50, 3.0);
    fitted.fit(constant_data);
    EXPECT_EQ(fitted.getK(), 1);
}

TEST_F(ErlangEnhancedTest, FitRejectsInvalidData) {
    auto fitted = ErlangDistribution::create(1, 1.0).unwrap();
    EXPECT_THROW(fitted.fit({}), std::invalid_argument);
    EXPECT_THROW(fitted.fit({1.0, -1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(fitted.fit({1.0, 0.0, 2.0}), std::invalid_argument);
}

//==============================================================================
// ERROR HANDLING
//==============================================================================

TEST_F(ErlangEnhancedTest, InvalidParameters) {
    auto r0 = ErlangDistribution::create(0, 1.0);
    EXPECT_TRUE(r0.isError());

    auto r1 = ErlangDistribution::create(-1, 1.0);
    EXPECT_TRUE(r1.isError());

    auto r2 = ErlangDistribution::create(1, -5.0);
    EXPECT_TRUE(r2.isError());

    auto r3 = ErlangDistribution::create(1, std::numeric_limits<double>::quiet_NaN());
    EXPECT_TRUE(r3.isError());

    auto erl = ErlangDistribution::create(3, 1.0).unwrap();
    auto vr = erl.trySetK(-1);
    EXPECT_TRUE(vr.isError());
    EXPECT_EQ(erl.getK(), 3);  // unchanged
}

//==============================================================================
// SUPPORT BOUNDARIES
//==============================================================================

TEST_F(ErlangEnhancedTest, SupportBoundaries) {
    EXPECT_EQ(dist4_.getProbability(-1.0), 0.0);
    EXPECT_EQ(dist4_.getCumulativeProbability(0.0), 0.0);
    EXPECT_NEAR(dist4_.getCumulativeProbability(1e6), 1.0, 1e-10);
}

}  // namespace stats

//==============================================================================
// DistTraits specialization for stats::ErlangDistribution
//==============================================================================
template <>
struct stats::tests::DistTraits<stats::ErlangDistribution> : stats::tests::DistTraitsDefaults {
    static stats::ErlangDistribution make() {
        return stats::ErlangDistribution::create(3, 1.0).unwrap();
    }
    static std::vector<double> domain() { return {0.5, 1.0, 2.0, 4.0, 8.0}; }
    static double batch_lo() { return 0.1; }
    static double batch_hi() { return 10.0; }
    static double pdf_tolerance() { return 1e-12; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::ErlangDistribution::create(0, 1.0).isError(); },
            [] { return stats::ErlangDistribution::create(-1, 1.0).isError(); },
            [] { return stats::ErlangDistribution::create(1, -1.0).isError(); },
            [] {
                return stats::ErlangDistribution::create(1, std::numeric_limits<double>::quiet_NaN())
                    .isError();
            },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Erlang, DistributionEnhancedTest,
                               ::testing::Types<stats::ErlangDistribution>);
