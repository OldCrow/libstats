#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/inverse_gamma.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

namespace stats {

//==============================================================================
// EXTERNAL REFERENCE VALUES
//
// Provenance: mpmath 1.4.1, mp.dps = 50, evaluated 2026-09-03, printed with
// mp.nstr(v, 17). Generating expressions:
//
//   logpdf = alpha*log(beta) - loggamma(alpha) - (alpha+1)*log(x) - beta/x
//   pdf    = exp(logpdf)
//   cdf    = gammainc(alpha, beta/x, inf, regularized=True)      # Q(alpha, beta/x)
//   sf     = 1 - cdf                                             # exact at 50 dps
//   q(p)   = 400-step bisection on log x against the 50-dps cdf, evaluated at
//            mpf(<the exact double>) rather than the decimal literal — see the
//            note on the quantile constants.
//
// mpmath's incomplete gamma is an independent arbitrary-precision routine, so
// agreement is evidence about this implementation rather than a restatement.
//==============================================================================

namespace ref {

// ---- InvGamma(alpha=3, scale beta=2) ----
constexpr double kPdf_0_25 = 0.34351373097217212;
constexpr double kPdf_0_5 = 1.1722008888789875;
constexpr double kPdf_1_0 = 0.54134113294645077;
constexpr double kPdf_2_0 = 0.091969860292860580;

constexpr double kLogPdf_0_25 = -1.0685281944005469;
constexpr double kLogPdf_0_5 = 0.15888308335967186;
constexpr double kLogPdf_1_0 = -0.61370563888010938;
constexpr double kLogPdf_2_0 = -2.3862943611198906;

constexpr double kCdf_0_25 = 0.013753967744002985;
constexpr double kCdf_0_5 = 0.23810330555354434;
constexpr double kCdf_1_0 = 0.67667641618306346;
constexpr double kCdf_2_0 = 0.91969860292860580;

// Lower tail — the regime the naive `1 - CDF_Gamma(1/x)` destroys completely.
constexpr double kCdf_0_05 = 3.5728659287002263e-15;
constexpr double kCdf_0_02 = 1.8976107553682284e-40;

// Upper tail — survival is the small quantity here.
constexpr double kSf_100 = 1.3134924482406743e-6;
constexpr double kSf_1000 = 1.3313349324448253e-9;

// Quantiles at the EXACT double value of p (see fisher_f.h / inverse_gamma.h on
// why the upper-tail p cannot carry its own complement).
constexpr double kQ_1e_12 = 0.058733055992991084;
constexpr double kQ_1e_6 = 0.10455237678297955;
constexpr double kQ_0_5 = 0.7479262863802243;
constexpr double kQ_1m1e_6 = 109.56332845305316;   // p = double(1 - 1e-6)
constexpr double kQ_1m1e_12 = 11006.00531543801;   // p = double(1 - 1e-12)

constexpr double kEntropy_3_2 = 0.695157020726;  // 40-dps quadrature cross-check

// ---- InvGamma(alpha=50, scale beta=3): large-shape case ----
constexpr double kBigPdf_0_06 = 46.937505270992355;
constexpr double kBigCdf_0_04 = 0.00090393204235400909;
constexpr double kBigCdf_0_06 = 0.48119168452795672;
constexpr double kBigCdf_0_02 = 7.4121008573228768e-22;
constexpr double kBigSf_0_3 = 1.8547268838697993e-19;
constexpr double kBigQ_1e_12 = 0.025661799859477699;
constexpr double kBigQ_0_5 = 0.060402200594578908;
constexpr double kBigQ_1m1e_12 = 0.1994405773659457;
constexpr double kBigEntropy = -3.33711710981;

// ---- InvGamma(alpha=0.5, scale beta=1): shape below 1 ----
constexpr double kHalfPdf_1_0 = 0.20755374871029735;
constexpr double kHalfCdf_1_0 = 0.15729920705028513;
constexpr double kHalfCdf_4_0 = 0.47950012218695346;

}  // namespace ref

// Tolerances gated to the measured accuracy law documented in inverse_gamma.h.
//
// PDF/LogPDF go through the delegate's log-space formula plus an exact
// Jacobian and reach full double accuracy (< 1e-13 relative).
//
// CDF and survival are bounded by detail::gamma_q / detail::gamma_p, whose
// series and continued fraction both stop on a relative residual of
// detail::DEFAULT_TOLERANCE = 1e-8. The binding case is beta/x ~ alpha, right
// at their series/continued-fraction switch: measured worst case over this
// grid is 1.04e-8 relative, at InvGamma(50,3) CDF(0.06) where beta/x = 50 and
// alpha = 50. Away from that switch the error drops by two or more orders of
// magnitude. 5e-8 is the law with headroom; a flat 1e-12 is not reachable
// through this math layer and asserting it would be asserting a fiction.
constexpr double kCdfRelTol = 5e-8;

// Quantiles divide the CDF's relative error by the local tail elasticity
// |d ln F / d ln x| (= alpha in the lower tail), so they come out tighter than
// the CDF that drives them.
constexpr double kQuantileRelTol = 1e-8;

// The delegate's vectorized log-density kernel differs from its own scalar
// path in the last ulp, so a batch that clears the dispatch threshold cannot
// be compared to the scalar result with EXPECT_EQ. Measured difference is
// exactly 1 ulp. (The CDF path has no such gap: it is this class's own scalar
// kernel on every tier, so it is compared exactly.)
constexpr double kBatchUlpRelTol = 1e-14;

static double relErr(double got, double expected) {
    if (expected == 0.0)
        return std::abs(got);
    return std::abs(got - expected) / std::abs(expected);
}

//==============================================================================
// TEST FIXTURE
//==============================================================================

class InverseGammaEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto result = stats::InverseGammaDistribution::create(3.0, 2.0);
        ASSERT_TRUE(result.isOk());
        ig_ = std::move(result).unwrap();
    }

    InverseGammaDistribution ig_;
};

//==============================================================================
// PARAMETERIZATION — beta is a SCALE here and the delegate's RATE
//
// This is the correction the #56 kickoff comment records: the delegate is
// GammaDistribution(alpha, beta) with beta passed unchanged, NOT
// Gamma(alpha, 1/beta). The identity below fails outright under the wrong
// convention, so this test is what pins the parameterization down.
//==============================================================================

TEST_F(InverseGammaEnhancedTest, DelegatesToGammaWithBetaAsIs) {
    const double alpha = 3.0, beta = 2.0;
    auto gam = GammaDistribution::create(alpha, beta).unwrap();  // (shape, RATE)

    for (double x : {0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 20.0}) {
        // Density transform: f_Y(x) = f_X(1/x) / x^2.
        EXPECT_NEAR(ig_.getLogProbability(x),
                    gam.getLogProbability(1.0 / x) - 2.0 * std::log(x), 1e-13)
            << "log density identity at x=" << x;
        // Distribution transform: F_Y(x) = 1 - F_X(1/x), checked in the region
        // where the subtraction is harmless (it is NOT how the CDF is computed).
        EXPECT_NEAR(ig_.getCumulativeProbability(x), 1.0 - gam.getCumulativeProbability(1.0 / x),
                    1e-10)
            << "CDF identity at x=" << x;
    }

    // The wrong parameterization (rate = 1/beta) must NOT reproduce the CDF —
    // a two-sided check, so this guard cannot pass on either convention.
    auto wrong = GammaDistribution::create(alpha, 1.0 / beta).unwrap();
    EXPECT_GT(std::abs(ig_.getCumulativeProbability(1.0) -
                       (1.0 - wrong.getCumulativeProbability(1.0))),
              1e-3);
}

//==============================================================================
// KNOWN VALUES — central region
//==============================================================================

TEST_F(InverseGammaEnhancedTest, KnownValuesCentral) {
    EXPECT_LT(relErr(ig_.getProbability(0.25), ref::kPdf_0_25), 1e-13);
    EXPECT_LT(relErr(ig_.getProbability(0.5), ref::kPdf_0_5), 1e-13);
    EXPECT_LT(relErr(ig_.getProbability(1.0), ref::kPdf_1_0), 1e-13);
    EXPECT_LT(relErr(ig_.getProbability(2.0), ref::kPdf_2_0), 1e-13);

    EXPECT_NEAR(ig_.getLogProbability(0.25), ref::kLogPdf_0_25, 1e-13);
    EXPECT_NEAR(ig_.getLogProbability(0.5), ref::kLogPdf_0_5, 1e-13);
    EXPECT_NEAR(ig_.getLogProbability(1.0), ref::kLogPdf_1_0, 1e-13);
    EXPECT_NEAR(ig_.getLogProbability(2.0), ref::kLogPdf_2_0, 1e-13);

    EXPECT_LT(relErr(ig_.getCumulativeProbability(0.25), ref::kCdf_0_25), kCdfRelTol);
    EXPECT_LT(relErr(ig_.getCumulativeProbability(0.5), ref::kCdf_0_5), kCdfRelTol);
    EXPECT_LT(relErr(ig_.getCumulativeProbability(1.0), ref::kCdf_1_0), kCdfRelTol);
    EXPECT_LT(relErr(ig_.getCumulativeProbability(2.0), ref::kCdf_2_0), kCdfRelTol);
}

//==============================================================================
// KNOWN VALUES — the lower tail, which is the whole point of using gamma_q
//
// CDF(x) = Q(alpha, beta/x). For small x, beta/x is large and the delegate's
// own CDF, P(alpha, beta/x), is 1 to every bit double has — so
// `1 - CDF_Gamma(1/x)` returns exactly 0 while the true value is 1.9e-40.
//==============================================================================

TEST_F(InverseGammaEnhancedTest, KnownValuesLowerTail) {
    EXPECT_LT(relErr(ig_.getCumulativeProbability(0.05), ref::kCdf_0_05), kCdfRelTol);
    EXPECT_LT(relErr(ig_.getCumulativeProbability(0.02), ref::kCdf_0_02), kCdfRelTol);

    // The contrast that motivates the formulation.
    auto gam = GammaDistribution::create(3.0, 2.0).unwrap();
    EXPECT_EQ(1.0 - gam.getCumulativeProbability(1.0 / 0.02), 0.0)
        << "the naive complement should be exactly zero here";
    EXPECT_GT(ig_.getCumulativeProbability(0.02), 0.0);
    EXPECT_LT(ig_.getCumulativeProbability(0.02), 1e-39);

    // Monotone and strictly positive all the way down until it genuinely
    // underflows — no clamp constant, no premature zero.
    double prev = ig_.getCumulativeProbability(0.05);
    for (double x : {0.04, 0.03, 0.02, 0.015, 0.01}) {
        const double c = ig_.getCumulativeProbability(x);
        EXPECT_LT(c, prev) << "CDF not strictly decreasing toward 0 at x=" << x;
        EXPECT_GE(c, 0.0);
        prev = c;
    }
}

TEST_F(InverseGammaEnhancedTest, KnownValuesUpperTailSurvival) {
    EXPECT_LT(relErr(ig_.getSurvivalProbability(100.0), ref::kSf_100), kCdfRelTol);
    EXPECT_LT(relErr(ig_.getSurvivalProbability(1000.0), ref::kSf_1000), kCdfRelTol);

    for (double x : {0.25, 0.5, 1.0, 2.0, 8.0}) {
        EXPECT_NEAR(ig_.getCumulativeProbability(x) + ig_.getSurvivalProbability(x), 1.0, 1e-12)
            << "CDF + SF != 1 at x=" << x;
    }
}

//==============================================================================
// KNOWN VALUES — quantiles on both tails (#104 accuracy)
//==============================================================================

TEST_F(InverseGammaEnhancedTest, KnownQuantilesBothTails) {
    EXPECT_LT(relErr(ig_.getQuantile(1e-12), ref::kQ_1e_12), kQuantileRelTol);
    EXPECT_LT(relErr(ig_.getQuantile(1e-6), ref::kQ_1e_6), kQuantileRelTol);
    EXPECT_LT(relErr(ig_.getQuantile(0.5), ref::kQ_0_5), kQuantileRelTol);
    EXPECT_LT(relErr(ig_.getQuantile(1.0 - 1e-6), ref::kQ_1m1e_6), kQuantileRelTol);
    EXPECT_LT(relErr(ig_.getQuantile(1.0 - 1e-12), ref::kQ_1m1e_12), kQuantileRelTol);
}

//==============================================================================
// LARGE PARAMETERS: InvGamma(50, 3)
//==============================================================================

TEST_F(InverseGammaEnhancedTest, LargeParameters) {
    auto big = InverseGammaDistribution::create(50.0, 3.0).unwrap();

    EXPECT_LT(relErr(big.getProbability(0.06), ref::kBigPdf_0_06), 1e-12);
    EXPECT_LT(relErr(big.getCumulativeProbability(0.04), ref::kBigCdf_0_04), kCdfRelTol);
    EXPECT_LT(relErr(big.getCumulativeProbability(0.06), ref::kBigCdf_0_06), kCdfRelTol);
    EXPECT_LT(relErr(big.getCumulativeProbability(0.02), ref::kBigCdf_0_02), kCdfRelTol);
    EXPECT_LT(relErr(big.getSurvivalProbability(0.3), ref::kBigSf_0_3), kCdfRelTol);

    EXPECT_LT(relErr(big.getQuantile(1e-12), ref::kBigQ_1e_12), kQuantileRelTol);
    EXPECT_LT(relErr(big.getQuantile(0.5), ref::kBigQ_0_5), kQuantileRelTol);
    EXPECT_LT(relErr(big.getQuantile(1.0 - 1e-12), ref::kBigQ_1m1e_12), kQuantileRelTol);

    EXPECT_NEAR(big.getEntropy(), ref::kBigEntropy, 1e-6);
}

//==============================================================================
// SHAPE BELOW 1: InvGamma(0.5, 1) — heavy tail, no finite mean
//==============================================================================

TEST_F(InverseGammaEnhancedTest, ShapeBelowOne) {
    auto h = InverseGammaDistribution::create(0.5, 1.0).unwrap();
    EXPECT_LT(relErr(h.getProbability(1.0), ref::kHalfPdf_1_0), 1e-13);
    EXPECT_LT(relErr(h.getCumulativeProbability(1.0), ref::kHalfCdf_1_0), kCdfRelTol);
    EXPECT_LT(relErr(h.getCumulativeProbability(4.0), ref::kHalfCdf_4_0), kCdfRelTol);

    EXPECT_TRUE(std::isnan(h.getMean()));
    EXPECT_TRUE(std::isnan(h.getVariance()));
    EXPECT_NEAR(h.getMode(), 1.0 / 1.5, 1e-14);

    // The alpha < 1 branch is exactly where the delegate's own log density
    // would return +inf at a zero argument; the transform layer must never let
    // that happen. x = +inf maps to 1/x = 0.
    EXPECT_EQ(h.getProbability(std::numeric_limits<double>::infinity()), 0.0);
    EXPECT_EQ(h.getLogProbability(std::numeric_limits<double>::infinity()),
              -std::numeric_limits<double>::infinity());
    // x = 0 maps to 1/x = +inf.
    EXPECT_EQ(h.getProbability(0.0), 0.0);
    EXPECT_EQ(h.getLogProbability(0.0), -std::numeric_limits<double>::infinity());
}

//==============================================================================
// MOMENTS — including the NaN regimes
//==============================================================================

TEST_F(InverseGammaEnhancedTest, MomentProperties) {
    // InvGamma(3,2): mean = 2/2 = 1, var = 4/(4*1) = 1, mode = 2/4 = 0.5
    EXPECT_NEAR(ig_.getMean(), 1.0, 1e-14);
    EXPECT_NEAR(ig_.getVariance(), 1.0, 1e-14);
    EXPECT_NEAR(ig_.getMode(), 0.5, 1e-14);
    EXPECT_EQ(ig_.getNumParameters(), 2);
    EXPECT_FALSE(ig_.isDiscrete());
    EXPECT_EQ(ig_.getDistributionName(), "InverseGamma");
    EXPECT_EQ(ig_.getSupportLowerBound(), 0.0);
    EXPECT_EQ(ig_.getSupportUpperBound(), std::numeric_limits<double>::infinity());

    // detail::digamma's asymptotic series is the binding constraint on entropy;
    // gated to the law rather than to full double precision.
    EXPECT_NEAR(ig_.getEntropy(), ref::kEntropy_3_2, 1e-6);

    // Undefined-moment regimes must be NaN.
    auto a1 = InverseGammaDistribution::create(1.0, 1.0).unwrap();
    EXPECT_TRUE(std::isnan(a1.getMean()));
    EXPECT_TRUE(std::isnan(a1.getVariance()));

    auto a2 = InverseGammaDistribution::create(2.0, 1.0).unwrap();
    EXPECT_FALSE(std::isnan(a2.getMean()));
    EXPECT_TRUE(std::isnan(a2.getVariance()));

    auto a3 = InverseGammaDistribution::create(3.0, 1.0).unwrap();
    EXPECT_FALSE(std::isnan(a3.getVariance()));
    EXPECT_TRUE(std::isnan(a3.getSkewness()));

    auto a4 = InverseGammaDistribution::create(4.0, 1.0).unwrap();
    EXPECT_FALSE(std::isnan(a4.getSkewness()));
    EXPECT_TRUE(std::isnan(a4.getKurtosis()));

    auto a6 = InverseGammaDistribution::create(6.0, 2.0).unwrap();
    EXPECT_FALSE(std::isnan(a6.getKurtosis()));
    EXPECT_NEAR(a6.getSkewness(), 4.0 * std::sqrt(4.0) / 3.0, 1e-13);
    EXPECT_NEAR(a6.getKurtosis(), (30.0 * 6.0 - 66.0) / (3.0 * 2.0), 1e-13);

    // Scale linearity: InvGamma(a, c*b) is InvGamma(a, b) scaled by c.
    auto s1 = InverseGammaDistribution::create(3.0, 2.0).unwrap();
    auto s2 = InverseGammaDistribution::create(3.0, 6.0).unwrap();
    EXPECT_NEAR(s2.getMean(), 3.0 * s1.getMean(), 1e-13);
    EXPECT_NEAR(s2.getCumulativeProbability(3.0), s1.getCumulativeProbability(1.0), 1e-12);
}

//==============================================================================
// SETTERS PROPAGATE THROUGH THE CACHE AND THE DELEGATE
//==============================================================================

TEST_F(InverseGammaEnhancedTest, SetterPropagates) {
    auto ig = InverseGammaDistribution::create(4.0, 1.0).unwrap();
    EXPECT_NEAR(ig.getMean(), 1.0 / 3.0, 1e-14);

    ig.setBeta(6.0);
    EXPECT_NEAR(ig.getMean(), 2.0, 1e-14);

    ig.setAlpha(3.0);
    auto fresh = InverseGammaDistribution::create(3.0, 6.0).unwrap();
    EXPECT_NEAR(ig.getProbability(2.0), fresh.getProbability(2.0), 1e-14);
    EXPECT_NEAR(ig.getCumulativeProbability(2.0), fresh.getCumulativeProbability(2.0), 1e-14);
    EXPECT_NEAR(ig.getEntropy(), fresh.getEntropy(), 1e-14);

    auto moved = std::move(fresh);
    EXPECT_NEAR(moved.getProbability(2.0), ig.getProbability(2.0), 1e-14);
}

//==============================================================================
// #103: ±inf and NaN, scalar — every edge the reciprocal creates
//==============================================================================

TEST_F(InverseGammaEnhancedTest, InfAndNaNContractScalar) {
    const double inf = std::numeric_limits<double>::infinity();
    const double ninf = -inf;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_EQ(ig_.getProbability(inf), 0.0);
    EXPECT_EQ(ig_.getProbability(ninf), 0.0);
    EXPECT_TRUE(std::isnan(ig_.getProbability(nan)));

    EXPECT_EQ(ig_.getLogProbability(inf), ninf);
    EXPECT_EQ(ig_.getLogProbability(ninf), ninf);
    EXPECT_TRUE(std::isnan(ig_.getLogProbability(nan)));

    EXPECT_EQ(ig_.getCumulativeProbability(ninf), 0.0);
    EXPECT_EQ(ig_.getCumulativeProbability(inf), 1.0);
    EXPECT_TRUE(std::isnan(ig_.getCumulativeProbability(nan)));

    EXPECT_EQ(ig_.getSurvivalProbability(inf), 0.0);
    EXPECT_EQ(ig_.getSurvivalProbability(ninf), 1.0);
    EXPECT_TRUE(std::isnan(ig_.getSurvivalProbability(nan)));

    // At and below the support edge.
    for (double x : {0.0, -0.0, -1.0, -1e-300, -1e300}) {
        EXPECT_EQ(ig_.getProbability(x), 0.0) << "pdf at/below support, x=" << x;
        EXPECT_EQ(ig_.getLogProbability(x), ninf) << "logpdf at/below support, x=" << x;
        EXPECT_EQ(ig_.getCumulativeProbability(x), 0.0) << "cdf at/below support, x=" << x;
    }

    // Positive x whose reciprocal overflows (x < ~5.6e-309): the delegate must
    // never see +inf, and the true density there is 0.
    for (double x : {1e-309, 1e-320, 5e-324}) {
        EXPECT_EQ(ig_.getProbability(x), 0.0) << "pdf at reciprocal-overflow x=" << x;
        EXPECT_EQ(ig_.getLogProbability(x), ninf) << "logpdf at reciprocal-overflow x=" << x;
        EXPECT_EQ(ig_.getCumulativeProbability(x), 0.0) << "cdf at reciprocal-overflow x=" << x;
        EXPECT_FALSE(std::isnan(ig_.getProbability(x)));
    }

    // Largest finite x: 1/x is subnormal but nonzero, so this exercises the
    // delegate at the far end of its own domain without any edge shortcut.
    const double huge = std::numeric_limits<double>::max();
    EXPECT_FALSE(std::isnan(ig_.getProbability(huge)));
    EXPECT_GE(ig_.getProbability(huge), 0.0);
    EXPECT_NEAR(ig_.getCumulativeProbability(huge), 1.0, 1e-12);
}

//==============================================================================
// #103: ±inf and NaN, batch — must equal the scalar path element for element
//==============================================================================

TEST_F(InverseGammaEnhancedTest, InfAndNaNContractBatch) {
    const double inf = std::numeric_limits<double>::infinity();
    const double ninf = -inf;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> xs = {0.5, 2.0, inf, ninf, nan, 5.0, 0.0, -1.0, 1e-320};
    const size_t N = xs.size();
    std::vector<double> pdf_b(N), lpdf_b(N), cdf_b(N);

    ig_.getProbability(std::span<const double>(xs), std::span<double>(pdf_b));
    ig_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf_b));
    ig_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        const double s_pdf = ig_.getProbability(xs[i]);
        const double s_lpdf = ig_.getLogProbability(xs[i]);
        const double s_cdf = ig_.getCumulativeProbability(xs[i]);

        EXPECT_EQ(std::isnan(pdf_b[i]), std::isnan(s_pdf)) << "PDF NaN mismatch i=" << i;
        if (!std::isnan(s_pdf)) {
            EXPECT_LT(relErr(pdf_b[i], s_pdf), kBatchUlpRelTol)
                << "PDF batch != scalar i=" << i;
        }

        EXPECT_EQ(std::isnan(lpdf_b[i]), std::isnan(s_lpdf)) << "LogPDF NaN mismatch i=" << i;
        if (!std::isnan(s_lpdf) && std::isfinite(s_lpdf)) {
            EXPECT_LT(relErr(lpdf_b[i], s_lpdf), kBatchUlpRelTol)
                << "LogPDF batch != scalar i=" << i;
        }
        else if (!std::isnan(s_lpdf))
            EXPECT_EQ(lpdf_b[i], s_lpdf) << "LogPDF batch != scalar (infinite) i=" << i;

        // The CDF is this class's own scalar kernel on every dispatch tier, so
        // batch and scalar must agree bit for bit.
        EXPECT_EQ(std::isnan(cdf_b[i]), std::isnan(s_cdf)) << "CDF NaN mismatch i=" << i;
        if (!std::isnan(s_cdf)) {
            EXPECT_EQ(cdf_b[i], s_cdf) << "CDF batch != scalar i=" << i;
        }
    }

    EXPECT_EQ(pdf_b[2], 0.0) << "PDF(+inf)";
    EXPECT_EQ(pdf_b[3], 0.0) << "PDF(-inf)";
    EXPECT_TRUE(std::isnan(pdf_b[4])) << "PDF(NaN)";
    EXPECT_EQ(pdf_b[6], 0.0) << "PDF(0)";
    EXPECT_EQ(pdf_b[7], 0.0) << "PDF(-1)";
    EXPECT_EQ(pdf_b[8], 0.0) << "PDF(reciprocal overflow)";
    EXPECT_EQ(lpdf_b[2], ninf) << "LogPDF(+inf)";
    EXPECT_EQ(lpdf_b[3], ninf) << "LogPDF(-inf)";
    EXPECT_TRUE(std::isnan(lpdf_b[4])) << "LogPDF(NaN)";
    EXPECT_EQ(lpdf_b[8], ninf) << "LogPDF(reciprocal overflow)";
    EXPECT_EQ(cdf_b[2], 1.0) << "CDF(+inf)";
    EXPECT_EQ(cdf_b[3], 0.0) << "CDF(-inf)";
    EXPECT_TRUE(std::isnan(cdf_b[4])) << "CDF(NaN)";
    EXPECT_EQ(cdf_b[6], 0.0) << "CDF(0)";
}

// Large enough to leave the scalar dispatch tier, so the parallel and
// work-stealing kernels get the same contract exercise.
TEST_F(InverseGammaEnhancedTest, InfAndNaNContractLargeBatch) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr size_t N = 40000;
    std::vector<double> xs(N), pdf_b(N), lpdf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = 0.05 + 0.001 * static_cast<double>(i % 5000);
    xs[10] = inf;
    xs[11] = nan;
    xs[12] = -inf;
    xs[13] = 0.0;
    xs[N - 1] = nan;

    ig_.getProbability(std::span<const double>(xs), std::span<double>(pdf_b));
    ig_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf_b));
    ig_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_b));

    EXPECT_EQ(pdf_b[10], 0.0);
    EXPECT_TRUE(std::isnan(pdf_b[11]));
    EXPECT_EQ(pdf_b[12], 0.0);
    EXPECT_EQ(pdf_b[13], 0.0);
    EXPECT_TRUE(std::isnan(pdf_b[N - 1]));
    EXPECT_EQ(cdf_b[10], 1.0);
    EXPECT_TRUE(std::isnan(cdf_b[11]));
    EXPECT_EQ(cdf_b[12], 0.0);
    EXPECT_EQ(lpdf_b[10], -std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(lpdf_b[11]));

    for (size_t i : {size_t{0}, size_t{1}, size_t{500}, size_t{4999}, size_t{20000},
                     size_t{39998}}) {
        EXPECT_LT(relErr(pdf_b[i], ig_.getProbability(xs[i])), kBatchUlpRelTol) << "i=" << i;
        EXPECT_LT(relErr(lpdf_b[i], ig_.getLogProbability(xs[i])), kBatchUlpRelTol)
            << "i=" << i;
        // CDF: own scalar kernel on every tier, so exact.
        EXPECT_EQ(cdf_b[i], ig_.getCumulativeProbability(xs[i])) << "i=" << i;
    }
}

//==============================================================================
// #104: quantile is never NaN on (0,1); ±inf only on true overflow
//==============================================================================

TEST_F(InverseGammaEnhancedTest, QuantileNeverNaN) {
    for (double p : {1e-300, 1e-100, 1e-12, 1e-6, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-9,
                     1.0 - 1e-12, 1.0 - 1e-15}) {
        const double q = ig_.getQuantile(p);
        EXPECT_FALSE(std::isnan(q)) << "quantile NaN at p=" << p;
        EXPECT_GE(q, 0.0) << "quantile negative at p=" << p;
        EXPECT_TRUE(std::isfinite(q)) << "quantile not finite at p=" << p;
    }

    EXPECT_EQ(ig_.getQuantile(0.0), 0.0);
    EXPECT_EQ(ig_.getQuantile(1.0), std::numeric_limits<double>::infinity());

    EXPECT_THROW((void)ig_.getQuantile(-0.1), std::invalid_argument);
    EXPECT_THROW((void)ig_.getQuantile(1.1), std::invalid_argument);
    EXPECT_THROW((void)ig_.getQuantile(std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);

    for (double p : {1e-12, 1e-8, 1e-4, 0.3, 0.5, 0.7, 0.9999}) {
        const double q = ig_.getQuantile(p);
        EXPECT_LT(relErr(ig_.getCumulativeProbability(q), p), 1e-8)
            << "CDF(quantile(" << p << ")) round trip";
    }
    // Upper tail: compare against the complement the solver actually received.
    // `1.0 - p` is exact by Sterbenz's lemma but is NOT the decimal literal —
    // double(1 - 1e-12) has complement 9.99978e-13. See inverse_gamma.h.
    for (double comp : {1e-12, 1e-8, 1e-4}) {
        const double p = 1.0 - comp;
        const double actual_comp = 1.0 - p;
        const double q = ig_.getQuantile(p);
        EXPECT_LT(relErr(ig_.getSurvivalProbability(q), actual_comp), 1e-8)
            << "SF(quantile(1-" << comp << ")) round trip";
    }
}

//==============================================================================
// (d) Batch size mismatch throws on all three overloads
//==============================================================================

TEST_F(InverseGammaEnhancedTest, BatchSizeMismatchThrows) {
    std::vector<double> in(8, 1.0);
    std::vector<double> out(7);
    EXPECT_THROW(ig_.getProbability(std::span<const double>(in), std::span<double>(out)),
                 std::invalid_argument);
    EXPECT_THROW(ig_.getLogProbability(std::span<const double>(in), std::span<double>(out)),
                 std::invalid_argument);
    EXPECT_THROW(ig_.getCumulativeProbability(std::span<const double>(in), std::span<double>(out)),
                 std::invalid_argument);

    std::vector<double> empty_in, empty_out;
    EXPECT_NO_THROW(
        ig_.getProbability(std::span<const double>(empty_in), std::span<double>(empty_out)));
}

//==============================================================================
// FIT — Gamma's MLE on the reciprocals
//==============================================================================

TEST_F(InverseGammaEnhancedTest, MLEFit) {
    std::mt19937 rng(4242);
    auto source = InverseGammaDistribution::create(5.0, 4.0).unwrap();
    const auto data = source.sample(rng, 5000);

    auto fitted = InverseGammaDistribution::create(1.0, 1.0).unwrap();
    fitted.fit(data);

    EXPECT_GT(fitted.getAlpha(), 0.0);
    EXPECT_GT(fitted.getBeta(), 0.0);
    EXPECT_NEAR(fitted.getAlpha(), 5.0, 1.0) << "fitted shape near the generating 5";
    EXPECT_NEAR(fitted.getBeta(), 4.0, 1.0) << "fitted scale near the generating 4";
    // Mean of InvGamma(5,4) is 1; the fit should reproduce it closely.
    EXPECT_NEAR(fitted.getMean(), 1.0, 0.15);
}

TEST_F(InverseGammaEnhancedTest, FitRejectsInvalidData) {
    auto fitted = InverseGammaDistribution::create(1.0, 1.0).unwrap();
    EXPECT_THROW(fitted.fit({}), std::invalid_argument);
    EXPECT_THROW(fitted.fit({1.0, -1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(fitted.fit({1.0, 0.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(fitted.fit({1.0, std::numeric_limits<double>::quiet_NaN()}),
                 std::invalid_argument);
    EXPECT_THROW(fitted.fit({1.0, std::numeric_limits<double>::infinity()}),
                 std::invalid_argument);
    // Reciprocal overflow is rejected explicitly rather than silently producing inf.
    EXPECT_THROW(fitted.fit({1.0, 1e-320}), std::invalid_argument);
}

//==============================================================================
// ERROR HANDLING
//==============================================================================

TEST_F(InverseGammaEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(InverseGammaDistribution::create(0.0, 1.0).isError());
    EXPECT_TRUE(InverseGammaDistribution::create(-1.0, 1.0).isError());
    EXPECT_TRUE(InverseGammaDistribution::create(1.0, 0.0).isError());
    EXPECT_TRUE(InverseGammaDistribution::create(1.0, -1.0).isError());
    EXPECT_TRUE(
        InverseGammaDistribution::create(std::numeric_limits<double>::quiet_NaN(), 1.0)
            .isError());
    EXPECT_TRUE(
        InverseGammaDistribution::create(1.0, std::numeric_limits<double>::infinity()).isError());

    auto ig = InverseGammaDistribution::create(3.0, 2.0).unwrap();
    auto vr = ig.trySetAlpha(-1.0);
    EXPECT_TRUE(vr.isError());
    EXPECT_EQ(ig.getAlpha(), 3.0);  // unchanged

    EXPECT_THROW(ig.setBeta(0.0), std::invalid_argument);
    EXPECT_EQ(ig.getBeta(), 2.0);  // unchanged
}

}  // namespace stats

//==============================================================================
// DistTraits specialization for stats::InverseGammaDistribution
//==============================================================================
template <>
struct stats::tests::DistTraits<stats::InverseGammaDistribution>
    : stats::tests::DistTraitsDefaults {
    static stats::InverseGammaDistribution make() {
        return stats::InverseGammaDistribution::create(3.0, 2.0).unwrap();
    }
    static std::vector<double> domain() { return {0.25, 0.5, 1.0, 2.0, 5.0}; }
    static double batch_lo() { return 0.1; }
    static double batch_hi() { return 6.0; }
    static double pdf_tolerance() { return 1e-13; }
    static double cdf_tolerance() { return 1e-13; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::InverseGammaDistribution::create(0.0, 1.0).isError(); },
            [] { return stats::InverseGammaDistribution::create(-1.0, 1.0).isError(); },
            [] { return stats::InverseGammaDistribution::create(1.0, 0.0).isError(); },
            [] {
                return stats::InverseGammaDistribution::create(
                           1.0, std::numeric_limits<double>::quiet_NaN())
                    .isError();
            },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(InverseGamma, DistributionEnhancedTest,
                               ::testing::Types<stats::InverseGammaDistribution>);

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
