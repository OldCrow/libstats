#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/beta.h"
#include "libstats/distributions/fisher_f.h"

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
// Provenance: mpmath 1.4.1, mp.dps = 50, evaluated 2026-09-03 and printed with
// mp.nstr(v, 17). The generating expressions were
//
//   a, b   = mpf(d1)/2, mpf(d2)/2
//   logpdf = a*log(d1) + b*log(d2) + (a-1)*log(x) - (a+b)*log(d1*x+d2)
//            - (loggamma(a) + loggamma(b) - loggamma(a+b))
//   pdf    = exp(logpdf)
//   cdf    = betainc(a, b, 0, d1*x/(d1*x+d2), regularized=True)
//   sf     = 1 - cdf                    (exact at 50 dps; the point of the test
//                                        is that double arithmetic must reach it
//                                        without ever forming 1 - cdf)
//   q(p)   = 400-step bisection on log x against the 50-dps cdf
//
// These are independent of libstats: mpmath computes the incomplete beta with
// its own arbitrary-precision routines, so agreement is evidence about this
// implementation, not a restatement of it.
//==============================================================================

namespace ref {

// ---- F(5, 10) ----
constexpr double kPdf_0_5 = 0.68760700277062333;
constexpr double kPdf_1_0 = 0.49547978348663871;
constexpr double kPdf_2_0 = 0.16200574218011492;
constexpr double kPdf_5_0 = 0.0096306379378100752;

constexpr double kLogPdf_0_5 = -0.37453782115848623;
constexpr double kLogPdf_1_0 = -0.70222872627322796;
constexpr double kLogPdf_2_0 = -1.8201234988216670;
constexpr double kLogPdf_5_0 = -4.6428058105261045;

constexpr double kCdf_0_5 = 0.22997511934989837;
constexpr double kCdf_1_0 = 0.53488057346219959;
constexpr double kCdf_2_0 = 0.83580504910026119;
constexpr double kCdf_5_0 = 0.98513119959188702;

// Lower tail (CDF itself is the small quantity).
constexpr double kCdf_1e_6 = 4.1473358908641244e-15;
constexpr double kCdf_1e_3 = 1.3079991120401546e-7;

// Upper tail (the survival function is the small quantity; 1 - CDF cannot
// represent either of these).
constexpr double kSf_1e3 = 3.7071680987086971e-13;
constexpr double kSf_1e6 = 3.7537030784668575e-28;

// Quantile references are evaluated at the EXACT double value of p that C++
// passes, not at the decimal literal. This matters only in the upper tail: the
// double nearest 1-1e-12 has complement 9.9997787827987850e-13, not 1e-12, and
// solving for that different p moves the F(5,10) answer by 4.4e-6 relative --
// which is the caller-side representation limit documented in fisher_f.h, not
// an implementation error. mpmath was fed mpf(<the double>), which is exact.
constexpr double kQ_1e_12 = 8.9721396668572137e-6;
constexpr double kQ_1e_6 = 0.0022591356950394809;
constexpr double kQ_0_5 = 0.93193316085104795;
constexpr double kQ_1m1e_6 = 49.356539766593084;    // p = double(1 - 1e-6)
constexpr double kQ_1m1e_12 = 819.54320819126184;   // p = double(1 - 1e-12)

// ---- F(100, 200): large-parameter case ----
constexpr double kBigPdf_1_0 = 2.2988201089185847;
constexpr double kBigCdf_1_0 = 0.50768538685954730;
constexpr double kBigCdf_0_5 = 7.3063613666031447e-5;
constexpr double kBigCdf_0_3 = 1.0857636466993138e-10;
constexpr double kBigSf_3_0 = 2.1328646959887624e-11;
constexpr double kBigQ_1e_12 = 0.25925977482453951;
constexpr double kBigQ_0_5 = 0.99666218777120147;
constexpr double kBigQ_1m1e_12 = 3.2260158143041223;  // p = double(1 - 1e-12)

// ---- F(1, 1): heaviest tail, a = b = 1/2 < 1 ----
constexpr double kTinyPdf_1_0 = 0.15915494309189534;
constexpr double kTinyCdf_4_0 = 0.70483276469913345;

}  // namespace ref

// Relative-error helper: the whole point of the tail tests is relative
// accuracy, which an absolute EXPECT_NEAR against 1e-28 would not measure.
// Tolerances gated to the measured accuracy law documented in fisher_f.h.
// PDF/LogPDF are cancellation-free and reach full double accuracy; CDF and
// survival are bounded by detail::beta_i's continued-fraction stopping rule
// (|delta-1| < detail::DEFAULT_TOLERANCE = 1e-8), whose measured worst case
// over this grid is 4.3e-10 relative. Quantiles divide the CDF error by the
// local tail elasticity and so come out tighter.
constexpr double kCdfRelTol = 5e-9;
constexpr double kQuantileRelTol = 1e-8;

static double relErr(double got, double expected) {
    if (expected == 0.0)
        return std::abs(got);
    return std::abs(got - expected) / std::abs(expected);
}

//==============================================================================
// TEST FIXTURE
//==============================================================================

class FisherFEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto result = stats::FDistribution::create(5.0, 10.0);
        ASSERT_TRUE(result.isOk());
        f_ = std::move(result).unwrap();
    }

    FDistribution f_;
};

//==============================================================================
// KNOWN VALUES — central region
//==============================================================================

TEST_F(FisherFEnhancedTest, KnownValuesCentral) {
    EXPECT_LT(relErr(f_.getProbability(0.5), ref::kPdf_0_5), 1e-13);
    EXPECT_LT(relErr(f_.getProbability(1.0), ref::kPdf_1_0), 1e-13);
    EXPECT_LT(relErr(f_.getProbability(2.0), ref::kPdf_2_0), 1e-13);
    EXPECT_LT(relErr(f_.getProbability(5.0), ref::kPdf_5_0), 1e-13);

    EXPECT_NEAR(f_.getLogProbability(0.5), ref::kLogPdf_0_5, 1e-13);
    EXPECT_NEAR(f_.getLogProbability(1.0), ref::kLogPdf_1_0, 1e-13);
    EXPECT_NEAR(f_.getLogProbability(2.0), ref::kLogPdf_2_0, 1e-13);
    EXPECT_NEAR(f_.getLogProbability(5.0), ref::kLogPdf_5_0, 1e-13);

    EXPECT_LT(relErr(f_.getCumulativeProbability(0.5), ref::kCdf_0_5), kCdfRelTol);
    EXPECT_LT(relErr(f_.getCumulativeProbability(1.0), ref::kCdf_1_0), kCdfRelTol);
    EXPECT_LT(relErr(f_.getCumulativeProbability(2.0), ref::kCdf_2_0), kCdfRelTol);
    EXPECT_LT(relErr(f_.getCumulativeProbability(5.0), ref::kCdf_5_0), kCdfRelTol);
}

//==============================================================================
// KNOWN VALUES — both tails to p ~ 1e-12 and beyond
//
// The lower tail exercises the direct I_y(a,b) branch; the upper tail is the
// one that the naive formulation destroys, because there y = d1x/(d1x+d2)
// rounds to 1 and `1 - y` keeps no significant bits of the complement. The
// implementation forms ybar = d2/(d1x+d2) directly instead — see fisher_f.h.
//==============================================================================

TEST_F(FisherFEnhancedTest, KnownValuesLowerTail) {
    EXPECT_LT(relErr(f_.getCumulativeProbability(1e-3), ref::kCdf_1e_3), kCdfRelTol);
    EXPECT_LT(relErr(f_.getCumulativeProbability(1e-6), ref::kCdf_1e_6), kCdfRelTol);
    // The CDF stays strictly positive far below where a subtraction-based
    // formulation would have flushed it to zero.
    EXPECT_GT(f_.getCumulativeProbability(1e-6), 0.0);
}

TEST_F(FisherFEnhancedTest, KnownValuesUpperTailSurvival) {
    EXPECT_LT(relErr(f_.getSurvivalProbability(1e3), ref::kSf_1e3), kCdfRelTol);
    EXPECT_LT(relErr(f_.getSurvivalProbability(1e6), ref::kSf_1e6), kCdfRelTol);

    // The contrast that motivates the design: 1 - CDF(1e6) is exactly 0 in
    // double, while the complement-native survival function is ~3.75e-28.
    EXPECT_EQ(1.0 - f_.getCumulativeProbability(1e6), 0.0);
    EXPECT_GT(f_.getSurvivalProbability(1e6), 0.0);

    // CDF and survival must still agree wherever double can represent both.
    for (double x : {0.25, 0.5, 1.0, 2.0, 4.0}) {
        EXPECT_NEAR(f_.getCumulativeProbability(x) + f_.getSurvivalProbability(x), 1.0, 1e-12)
            << "CDF + SF != 1 at x=" << x;
    }
}

//==============================================================================
// KNOWN VALUES — quantiles on both tails (#104 accuracy)
//==============================================================================

TEST_F(FisherFEnhancedTest, KnownQuantilesBothTails) {
    EXPECT_LT(relErr(f_.getQuantile(1e-12), ref::kQ_1e_12), kQuantileRelTol);
    EXPECT_LT(relErr(f_.getQuantile(1e-6), ref::kQ_1e_6), kQuantileRelTol);
    EXPECT_LT(relErr(f_.getQuantile(0.5), ref::kQ_0_5), kQuantileRelTol);
    EXPECT_LT(relErr(f_.getQuantile(1.0 - 1e-6), ref::kQ_1m1e_6), kQuantileRelTol);
    EXPECT_LT(relErr(f_.getQuantile(1.0 - 1e-12), ref::kQ_1m1e_12), kQuantileRelTol);
}

//==============================================================================
// LARGE PARAMETERS: F(100, 200)
//==============================================================================

TEST_F(FisherFEnhancedTest, LargeParameters) {
    auto big = FDistribution::create(100.0, 200.0).unwrap();

    EXPECT_LT(relErr(big.getProbability(1.0), ref::kBigPdf_1_0), 1e-13);
    EXPECT_LT(relErr(big.getCumulativeProbability(1.0), ref::kBigCdf_1_0), kCdfRelTol);
    EXPECT_LT(relErr(big.getCumulativeProbability(0.5), ref::kBigCdf_0_5), kCdfRelTol);
    EXPECT_LT(relErr(big.getCumulativeProbability(0.3), ref::kBigCdf_0_3), kCdfRelTol);
    EXPECT_LT(relErr(big.getSurvivalProbability(3.0), ref::kBigSf_3_0), kCdfRelTol);

    EXPECT_LT(relErr(big.getQuantile(1e-12), ref::kBigQ_1e_12), kQuantileRelTol);
    EXPECT_LT(relErr(big.getQuantile(0.5), ref::kBigQ_0_5), kQuantileRelTol);
    EXPECT_LT(relErr(big.getQuantile(1.0 - 1e-12), ref::kBigQ_1m1e_12), kQuantileRelTol);
}

//==============================================================================
// SMALL PARAMETERS: F(1,1), where a = b = 1/2 < 1 and the density diverges at 0
//==============================================================================

TEST_F(FisherFEnhancedTest, SmallParametersAndSupportEdge) {
    auto tiny = FDistribution::create(1.0, 1.0).unwrap();
    EXPECT_LT(relErr(tiny.getProbability(1.0), ref::kTinyPdf_1_0), 1e-13);
    EXPECT_LT(relErr(tiny.getCumulativeProbability(4.0), ref::kTinyCdf_4_0), kCdfRelTol);
    EXPECT_LT(relErr(tiny.getCumulativeProbability(1.0), 0.5), kCdfRelTol);

    // Support edge x = 0: the three regimes of the (a-1)ln(x) term.
    // a < 1 (d1 < 2): density diverges.
    EXPECT_EQ(tiny.getProbability(0.0), std::numeric_limits<double>::infinity());
    // a == 1 exactly (d1 == 2): the term is 0*(-inf), which must NOT become
    // NaN. For F(2, d2) the limit is exactly 1 for every d2.
    for (double d2 : {1.0, 5.0, 10.0, 200.0}) {
        auto f2 = FDistribution::create(2.0, d2).unwrap();
        EXPECT_NEAR(f2.getProbability(0.0), 1.0, 1e-13) << "F(2," << d2 << ") pdf(0)";
        EXPECT_FALSE(std::isnan(f2.getLogProbability(0.0)));
    }
    // a > 1 (d1 > 2): density vanishes.
    EXPECT_EQ(f_.getProbability(0.0), 0.0);
    EXPECT_EQ(f_.getLogProbability(0.0), -std::numeric_limits<double>::infinity());
}

//==============================================================================
// MOMENTS — including the NaN regimes
//==============================================================================

TEST_F(FisherFEnhancedTest, MomentProperties) {
    EXPECT_NEAR(f_.getMean(), 10.0 / 8.0, 1e-14);
    EXPECT_NEAR(f_.getVariance(), 1.3541666666666667, 1e-13);
    EXPECT_NEAR(f_.getMode(), (3.0 / 5.0) * (10.0 / 12.0), 1e-14);
    EXPECT_EQ(f_.getNumParameters(), 2);
    EXPECT_FALSE(f_.isDiscrete());
    EXPECT_EQ(f_.getDistributionName(), "FisherF");
    EXPECT_EQ(f_.getSupportLowerBound(), 0.0);
    EXPECT_EQ(f_.getSupportUpperBound(), std::numeric_limits<double>::infinity());

    // Entropy vs. numerical quadrature (mpmath, 40 dps): H(F(5,10)) = 1.1307598049091
    // detail::digamma's asymptotic series is the binding constraint here;
    // measured error 1.3e-8 absolute. Gated to the law, not to 1e-12.
    EXPECT_NEAR(f_.getEntropy(), 1.1307598049091, 1e-6);

    // Undefined-moment regimes must be NaN rather than a plausible-looking number.
    auto f_d2_2 = FDistribution::create(5.0, 2.0).unwrap();
    EXPECT_TRUE(std::isnan(f_d2_2.getMean()));
    EXPECT_TRUE(std::isnan(f_d2_2.getVariance()));

    auto f_d2_4 = FDistribution::create(5.0, 4.0).unwrap();
    EXPECT_FALSE(std::isnan(f_d2_4.getMean()));
    EXPECT_TRUE(std::isnan(f_d2_4.getVariance()));

    auto f_d2_6 = FDistribution::create(5.0, 6.0).unwrap();
    EXPECT_FALSE(std::isnan(f_d2_6.getVariance()));
    EXPECT_TRUE(std::isnan(f_d2_6.getSkewness()));

    auto f_d2_8 = FDistribution::create(5.0, 8.0).unwrap();
    EXPECT_FALSE(std::isnan(f_d2_8.getSkewness()));
    EXPECT_TRUE(std::isnan(f_d2_8.getKurtosis()));

    auto f_d2_12 = FDistribution::create(5.0, 12.0).unwrap();
    EXPECT_FALSE(std::isnan(f_d2_12.getKurtosis()));

    // Mode is 0 when d1 <= 2 (the density has no interior maximum).
    EXPECT_EQ(FDistribution::create(1.0, 10.0).unwrap().getMode(), 0.0);
    EXPECT_EQ(FDistribution::create(2.0, 10.0).unwrap().getMode(), 0.0);
}

//==============================================================================
// RELATIONSHIP TO THE BETA DELEGATE
//
// The probability functions deliberately do NOT route through the Beta
// delegate's public API, so this checks the mathematical identity rather than
// an implementation shortcut: CDF_F(x) == CDF_Beta(y) at y = d1x/(d1x+d2),
// in the region where forming y is harmless.
//==============================================================================

TEST_F(FisherFEnhancedTest, MatchesBetaIdentity) {
    auto beta = BetaDistribution::create(2.5, 5.0).unwrap();  // Beta(d1/2, d2/2)
    for (double x : {0.1, 0.5, 1.0, 2.0, 4.0}) {
        const double y = (5.0 * x) / (5.0 * x + 10.0);
        EXPECT_NEAR(f_.getCumulativeProbability(x), beta.getCumulativeProbability(y), 1e-11)
            << "CDF identity at x=" << x;
        // PDF identity carries the Jacobian dy/dx = d1 d2/(d1x+d2)^2.
        const double denom = 5.0 * x + 10.0;
        const double jac = (5.0 * 10.0) / (denom * denom);
        EXPECT_NEAR(f_.getProbability(x), beta.getProbability(y) * jac, 1e-11)
            << "PDF identity at x=" << x;
    }
}

//==============================================================================
// SETTERS PROPAGATE THROUGH THE CACHE AND THE DELEGATE
//==============================================================================

TEST_F(FisherFEnhancedTest, SetterPropagates) {
    auto f = FDistribution::create(3.0, 8.0).unwrap();
    EXPECT_NEAR(f.getMean(), 8.0 / 6.0, 1e-14);

    f.setD2(20.0);
    EXPECT_NEAR(f.getMean(), 20.0 / 18.0, 1e-14);

    f.setD1(5.0);
    auto fresh = FDistribution::create(5.0, 20.0).unwrap();
    EXPECT_NEAR(f.getProbability(1.5), fresh.getProbability(1.5), 1e-14);
    EXPECT_NEAR(f.getCumulativeProbability(1.5), fresh.getCumulativeProbability(1.5), 1e-14);
    EXPECT_NEAR(f.getEntropy(), fresh.getEntropy(), 1e-14);

    // A move must leave the moved-from object valid (d1 = d2 = 1) and the
    // moved-to object fully functional.
    auto moved = std::move(fresh);
    EXPECT_NEAR(moved.getProbability(1.5), f.getProbability(1.5), 1e-14);
}

//==============================================================================
// #103: ±inf and NaN, scalar
//==============================================================================

TEST_F(FisherFEnhancedTest, InfAndNaNContractScalar) {
    const double inf = std::numeric_limits<double>::infinity();
    const double ninf = -inf;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_EQ(f_.getProbability(inf), 0.0);
    EXPECT_EQ(f_.getProbability(ninf), 0.0);
    EXPECT_TRUE(std::isnan(f_.getProbability(nan)));

    EXPECT_EQ(f_.getLogProbability(inf), ninf);
    EXPECT_EQ(f_.getLogProbability(ninf), ninf);
    EXPECT_TRUE(std::isnan(f_.getLogProbability(nan)));

    EXPECT_EQ(f_.getCumulativeProbability(ninf), 0.0);
    EXPECT_EQ(f_.getCumulativeProbability(inf), 1.0);
    EXPECT_TRUE(std::isnan(f_.getCumulativeProbability(nan)));

    EXPECT_EQ(f_.getSurvivalProbability(inf), 0.0);
    EXPECT_EQ(f_.getSurvivalProbability(ninf), 1.0);
    EXPECT_TRUE(std::isnan(f_.getSurvivalProbability(nan)));

    // Below the support (and at it) — no clamp constant may escape.
    for (double x : {-1.0, -1e-300, -1e300}) {
        EXPECT_EQ(f_.getProbability(x), 0.0) << "pdf below support at x=" << x;
        EXPECT_EQ(f_.getLogProbability(x), ninf) << "logpdf below support at x=" << x;
        EXPECT_EQ(f_.getCumulativeProbability(x), 0.0) << "cdf below support at x=" << x;
    }

    // Repeat for the small-d1 branch, where the x==0 limit differs.
    auto tiny = FDistribution::create(1.0, 1.0).unwrap();
    EXPECT_EQ(tiny.getProbability(inf), 0.0);
    EXPECT_EQ(tiny.getLogProbability(inf), ninf);
    EXPECT_EQ(tiny.getCumulativeProbability(inf), 1.0);
    EXPECT_TRUE(std::isnan(tiny.getProbability(nan)));
}

//==============================================================================
// #103: ±inf and NaN, batch — must equal the scalar path element for element
//==============================================================================

TEST_F(FisherFEnhancedTest, InfAndNaNContractBatch) {
    const double inf = std::numeric_limits<double>::infinity();
    const double ninf = -inf;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> xs = {0.5, 2.0, inf, ninf, nan, 5.0, 0.0, -1.0};
    const size_t N = xs.size();
    std::vector<double> pdf_b(N), lpdf_b(N), cdf_b(N);

    f_.getProbability(std::span<const double>(xs), std::span<double>(pdf_b));
    f_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf_b));
    f_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        const double s_pdf = f_.getProbability(xs[i]);
        const double s_lpdf = f_.getLogProbability(xs[i]);
        const double s_cdf = f_.getCumulativeProbability(xs[i]);

        EXPECT_EQ(std::isnan(pdf_b[i]), std::isnan(s_pdf)) << "PDF NaN mismatch i=" << i;
        if (!std::isnan(s_pdf)) {
            EXPECT_EQ(pdf_b[i], s_pdf) << "PDF batch != scalar i=" << i;
        }

        EXPECT_EQ(std::isnan(lpdf_b[i]), std::isnan(s_lpdf)) << "LogPDF NaN mismatch i=" << i;
        if (!std::isnan(s_lpdf)) {
            EXPECT_EQ(lpdf_b[i], s_lpdf) << "LogPDF batch != scalar i=" << i;
        }

        EXPECT_EQ(std::isnan(cdf_b[i]), std::isnan(s_cdf)) << "CDF NaN mismatch i=" << i;
        if (!std::isnan(s_cdf)) {
            EXPECT_EQ(cdf_b[i], s_cdf) << "CDF batch != scalar i=" << i;
        }
    }

    // Explicit pins at the non-finite / edge slots.
    EXPECT_EQ(pdf_b[2], 0.0) << "PDF(+inf)";
    EXPECT_EQ(pdf_b[3], 0.0) << "PDF(-inf)";
    EXPECT_TRUE(std::isnan(pdf_b[4])) << "PDF(NaN)";
    EXPECT_EQ(pdf_b[6], 0.0) << "PDF(0) with d1=5>2";
    EXPECT_EQ(pdf_b[7], 0.0) << "PDF(-1)";
    EXPECT_EQ(lpdf_b[2], ninf) << "LogPDF(+inf)";
    EXPECT_EQ(lpdf_b[3], ninf) << "LogPDF(-inf)";
    EXPECT_TRUE(std::isnan(lpdf_b[4])) << "LogPDF(NaN)";
    EXPECT_EQ(cdf_b[2], 1.0) << "CDF(+inf)";
    EXPECT_EQ(cdf_b[3], 0.0) << "CDF(-inf)";
    EXPECT_TRUE(std::isnan(cdf_b[4])) << "CDF(NaN)";
    EXPECT_EQ(cdf_b[6], 0.0) << "CDF(0)";
}

// Large enough to leave the scalar dispatch tier, so the parallel/work-stealing
// kernels get the same contract exercise as the scalar one.
TEST_F(FisherFEnhancedTest, InfAndNaNContractLargeBatch) {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    constexpr size_t N = 40000;
    std::vector<double> xs(N), pdf_b(N), cdf_b(N), lpdf_b(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = 0.01 + 0.001 * static_cast<double>(i % 5000);
    xs[10] = inf;
    xs[11] = nan;
    xs[12] = -inf;
    xs[N - 1] = nan;

    f_.getProbability(std::span<const double>(xs), std::span<double>(pdf_b));
    f_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf_b));
    f_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_b));

    EXPECT_EQ(pdf_b[10], 0.0);
    EXPECT_TRUE(std::isnan(pdf_b[11]));
    EXPECT_EQ(pdf_b[12], 0.0);
    EXPECT_TRUE(std::isnan(pdf_b[N - 1]));
    EXPECT_EQ(cdf_b[10], 1.0);
    EXPECT_TRUE(std::isnan(cdf_b[11]));
    EXPECT_EQ(cdf_b[12], 0.0);
    EXPECT_EQ(lpdf_b[10], -std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(lpdf_b[11]));

    // Spot-check the finite slots against the scalar path.
    for (size_t i : {0u, 1u, 500u, 4999u, 20000u, 39998u}) {
        EXPECT_EQ(pdf_b[i], f_.getProbability(xs[i])) << "i=" << i;
        EXPECT_EQ(cdf_b[i], f_.getCumulativeProbability(xs[i])) << "i=" << i;
        EXPECT_EQ(lpdf_b[i], f_.getLogProbability(xs[i])) << "i=" << i;
    }
}

//==============================================================================
// #104: quantile is never NaN on (0,1); ±inf only on true overflow
//==============================================================================

TEST_F(FisherFEnhancedTest, QuantileNeverNaN) {
    for (double p : {1e-300, 1e-100, 1e-12, 1e-6, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-9,
                     1.0 - 1e-12, 1.0 - 1e-15}) {
        const double q = f_.getQuantile(p);
        EXPECT_FALSE(std::isnan(q)) << "quantile NaN at p=" << p;
        EXPECT_GE(q, 0.0) << "quantile negative at p=" << p;
        EXPECT_TRUE(std::isfinite(q)) << "quantile not finite at p=" << p;
    }

    // Endpoints are exact, not solved.
    EXPECT_EQ(f_.getQuantile(0.0), 0.0);
    EXPECT_EQ(f_.getQuantile(1.0), std::numeric_limits<double>::infinity());

    // Out-of-range p throws rather than returning a sentinel.
    EXPECT_THROW((void)f_.getQuantile(-0.1), std::invalid_argument);
    EXPECT_THROW((void)f_.getQuantile(1.1), std::invalid_argument);
    EXPECT_THROW((void)f_.getQuantile(std::numeric_limits<double>::quiet_NaN()),
                 std::invalid_argument);

    // Round trip across the whole range, including the tails that
    // detail::inverse_beta_i's absolute stopping rule cannot reach.
    for (double p : {1e-12, 1e-8, 1e-4, 0.3, 0.5, 0.7, 0.9999}) {
        const double q = f_.getQuantile(p);
        EXPECT_LT(relErr(f_.getCumulativeProbability(q), p), 1e-8)
            << "CDF(quantile(" << p << ")) round trip";
    }
    // Upper tail: assert against the complement the solver actually received.
    // `1.0 - p` is exact by Sterbenz's lemma, but it is NOT the decimal the
    // literal named -- double(1 - 1e-12) has complement 9.99978e-13. Comparing
    // to the literal would measure the caller's representation loss, not this
    // implementation. See fisher_f.h, "Achievable accuracy".
    for (double comp : {1e-12, 1e-8, 1e-4}) {
        const double p = 1.0 - comp;
        const double actual_comp = 1.0 - p;  // exact; generally != comp
        const double q = f_.getQuantile(p);
        EXPECT_LT(relErr(f_.getSurvivalProbability(q), actual_comp), 1e-8)
            << "SF(quantile(1-" << comp << ")) round trip";
    }
}

TEST_F(FisherFEnhancedTest, CDFSurvivesDenormalArguments) {
    // d1·x underflows to zero for x below ~DBL_TRUE_MIN/d1 even though the
    // beta argument y = x/(x + d2/d1) is representable, so the CDF collapsed
    // to 0 at points where its true value is ~0.0123 — and F(0.01,0.01)'s
    // own quantile(1e-300) output lands exactly there, so quantile→cdf did
    // not close. The x literals below are exact stored doubles; references
    // are mpmath dps=60 lead-term I_y(0.005, 0.005) at those doubles.
    auto f = FDistribution::create(0.01, 0.01).unwrap();
    EXPECT_LT(relErr(f.getCumulativeProbability(2.4209216646221081e-322), 0.012328426155182170),
              1e-3)
        << "CDF at a sub-(DBL_TRUE_MIN/d1) denormal";
    // One rounding step up: today this row survives only by the accident of
    // d1·x rounding to DBL_TRUE_MIN, with ~3.4e-3 relative error.
    EXPECT_LT(relErr(f.getCumulativeProbability(2.5e-322), 0.012330892415900902), 1e-3)
        << "CDF at a denormal where d1*x holds a single bit";
    // Accuracy is preserved where the current form is already fine.
    EXPECT_LT(relErr(f.getCumulativeProbability(1e-310), 0.014092489973099284), 1e-10)
        << "CDF at a many-bit denormal";
    // Complement-native survival must not saturate to 1 there.
    EXPECT_LT(relErr(f.getSurvivalProbability(2.4209216646221081e-322), 0.98767157384481783),
              1e-3)
        << "SF at a sub-(DBL_TRUE_MIN/d1) denormal";
    // quantile→cdf closes instead of collapsing to zero.
    const double q = f.getQuantile(1e-300);
    ASSERT_TRUE(std::isfinite(q));
    EXPECT_GT(f.getCumulativeProbability(q), 0.0) << "CDF(quantile(1e-300)) collapsed to 0";
}

//==============================================================================
// (d) Batch size mismatch throws on all three overloads
//==============================================================================

TEST_F(FisherFEnhancedTest, BatchSizeMismatchThrows) {
    std::vector<double> in(8, 1.0);
    std::vector<double> out(7);
    EXPECT_THROW(f_.getProbability(std::span<const double>(in), std::span<double>(out)),
                 std::invalid_argument);
    EXPECT_THROW(f_.getLogProbability(std::span<const double>(in), std::span<double>(out)),
                 std::invalid_argument);
    EXPECT_THROW(f_.getCumulativeProbability(std::span<const double>(in), std::span<double>(out)),
                 std::invalid_argument);

    // Empty spans are a no-op, not an error.
    std::vector<double> empty_in, empty_out;
    EXPECT_NO_THROW(
        f_.getProbability(std::span<const double>(empty_in), std::span<double>(empty_out)));
}

//==============================================================================
// FIT — documented best-effort method of moments
//==============================================================================

TEST_F(FisherFEnhancedTest, MLEFit) {
    std::mt19937 rng(12345);
    auto source = FDistribution::create(10.0, 40.0).unwrap();
    const auto data = source.sample(rng, 5000);

    auto fitted = FDistribution::create(1.0, 1.0).unwrap();
    fitted.fit(data);

    // The estimator is deliberately weak (see fisher_f.h); assert that it lands
    // in a sane region and produces a usable distribution, not that it recovers
    // the generating parameters closely.
    EXPECT_GT(fitted.getD1(), 0.0);
    EXPECT_GT(fitted.getD2(), 4.0);
    EXPECT_TRUE(std::isfinite(fitted.getMean()));
    EXPECT_GT(fitted.getMean(), 0.0);
    EXPECT_GT(fitted.getProbability(1.0), 0.0);
}

TEST_F(FisherFEnhancedTest, FitFallsBackWhenMomentsAreUninformative) {
    // Sample mean <= 1 leaves the mean equation with no valid d2 > 2 root; the
    // documented fallback keeps the fit total instead of throwing.
    auto fitted = FDistribution::create(3.0, 9.0).unwrap();
    // NB: not named `small` — <windef.h> defines that as a macro for `char`.
    std::vector<double> shrunk(100, 0.25);
    EXPECT_NO_THROW(fitted.fit(shrunk));
    EXPECT_GT(fitted.getD1(), 0.0);
    EXPECT_GT(fitted.getD2(), 0.0);
    EXPECT_TRUE(fitted.validateCurrentParameters().isOk());
}

TEST_F(FisherFEnhancedTest, FitRejectsInvalidData) {
    auto fitted = FDistribution::create(1.0, 1.0).unwrap();
    EXPECT_THROW(fitted.fit({}), std::invalid_argument);
    EXPECT_THROW(fitted.fit({1.0, -1.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(fitted.fit({1.0, 0.0, 2.0}), std::invalid_argument);
    EXPECT_THROW(fitted.fit({1.0, std::numeric_limits<double>::quiet_NaN()}),
                 std::invalid_argument);
}

//==============================================================================
// ERROR HANDLING
//==============================================================================

TEST_F(FisherFEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(FDistribution::create(0.0, 1.0).isError());
    EXPECT_TRUE(FDistribution::create(-1.0, 1.0).isError());
    EXPECT_TRUE(FDistribution::create(1.0, 0.0).isError());
    EXPECT_TRUE(FDistribution::create(1.0, -1.0).isError());
    EXPECT_TRUE(
        FDistribution::create(std::numeric_limits<double>::quiet_NaN(), 1.0).isError());
    EXPECT_TRUE(FDistribution::create(1.0, std::numeric_limits<double>::infinity()).isError());

    auto f = FDistribution::create(5.0, 10.0).unwrap();
    auto vr = f.trySetD1(-1.0);
    EXPECT_TRUE(vr.isError());
    EXPECT_EQ(f.getD1(), 5.0);  // unchanged

    EXPECT_THROW(f.setD2(0.0), std::invalid_argument);
    EXPECT_EQ(f.getD2(), 10.0);  // unchanged
}

}  // namespace stats

//==============================================================================
// DistTraits specialization for stats::FDistribution
//==============================================================================
template <>
struct stats::tests::DistTraits<stats::FDistribution> : stats::tests::DistTraitsDefaults {
    static stats::FDistribution make() {
        return stats::FDistribution::create(5.0, 10.0).unwrap();
    }
    static std::vector<double> domain() { return {0.2, 0.5, 1.0, 2.0, 5.0}; }
    static double batch_lo() { return 0.05; }
    static double batch_hi() { return 6.0; }
    static double pdf_tolerance() { return 1e-13; }
    static double cdf_tolerance() { return 1e-13; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::FDistribution::create(0.0, 1.0).isError(); },
            [] { return stats::FDistribution::create(-1.0, 1.0).isError(); },
            [] { return stats::FDistribution::create(1.0, 0.0).isError(); },
            [] {
                return stats::FDistribution::create(1.0,
                                                    std::numeric_limits<double>::quiet_NaN())
                    .isError();
            },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(FisherF, DistributionEnhancedTest,
                               ::testing::Types<stats::FDistribution>);

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
