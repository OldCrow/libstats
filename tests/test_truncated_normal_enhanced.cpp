// tests/test_truncated_normal_enhanced.cpp
//
// Enhanced tests for TruncatedNormalDistribution (#57): the regime-split
// normalization/CDF/quantile design under its worst conditions (same-tail
// truncation windows), the supported-window policy edges, degeneration to the
// plain Gaussian, and the Born-compliant edge contracts (#103 ±inf limits,
// NaN propagation, #104 quantile never-NaN, batch size-mismatch throws)
// asserted on BOTH the scalar and the forced-vectorized batch paths.
//
// Reference provenance: constants tagged "mpmath" below were computed with
// mpmath from the exact double inputs (scratchpad scripts, 2026-09-03),
// printed to 17 significant digits. Central-window references at dps=40;
// same-tail windows at dps=60 with Z and all CDF numerators evaluated in the
// SURVIVAL domain (erfc differences) so the references themselves never
// cancel:
//   Q(z) = erfc(z/sqrt2)/2, Z = Q(alpha) - Q(beta),
//   CDF(x) = (Q(alpha) - Q(xi))/Z, quantile by bisection on Q at 1e-55 tol.
//
// Budget notes (the #49 law): where a value is reconstructed from tail
// pieces, the achievable relative error is a LAW of the tail depth, not a
// constant — an ulp perturbation of the erfc argument w is amplified by
// d ln F/dw = -2w, so rel ~ |ln F|*2^-52. CDF round-trips near p -> 0 carry
// the amplification eps/p (the numerator Q(alpha)-Q(x) equals p*Z while its
// absolute error stays ~ulp(Q(alpha))). Per-case budgets below state which
// term dominates.

#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/truncated_normal.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

double relerr(double got, double ref) {
    return std::fabs(got - ref) / std::fabs(ref);
}
}  // namespace

namespace stats {

class TruncatedNormalEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = stats::TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0);
        ASSERT_TRUE(r.isOk());
        tn_ = std::move(r).unwrap();
    }
    TruncatedNormalDistribution tn_;  // TN(0, 1, -2, 2)
};

//==============================================================================
// Central-window accuracy vs mpmath (dps=40)
//==============================================================================

TEST_F(TruncatedNormalEnhancedTest, CentralWindowVsMpmath) {
    // TN(0,1,-2,2)
    EXPECT_LE(relerr(tn_.getNormalizationConstant(), 0.95449973610364159), 5e-15);
    EXPECT_NEAR(tn_.getMean(), 0.0, 1e-15);
    EXPECT_LE(relerr(tn_.getVariance(), 0.77374130354992325), 1e-13);
    EXPECT_NEAR(tn_.getSkewness(), 0.0, 1e-14);
    EXPECT_LE(relerr(tn_.getKurtosis(), -0.63446328287035049), 1e-12);
    EXPECT_LE(relerr(tn_.getEntropy(), 1.2592412726872442), 1e-13);

    const struct {
        double x, cdf, logpdf;
    } rows[] = {
        {-1.5, 0.046157235726983009, -1.9973706209122826},
        {-0.5, 0.29941067133702831, -0.99737062091228258},
        {0.0, 0.50000000000000000, -0.87237062091228258},
        {0.5, 0.70058932866297169, -0.99737062091228258},
        {1.5, 0.95384276427301699, -1.9973706209122826},
    };
    for (const auto& r : rows) {
        EXPECT_LE(relerr(tn_.getCumulativeProbability(r.x), r.cdf), 1e-13) << "cdf x=" << r.x;
        EXPECT_NEAR(tn_.getLogProbability(r.x), r.logpdf, 1e-13) << "logpdf x=" << r.x;
        EXPECT_LE(relerr(tn_.getProbability(r.x), std::exp(r.logpdf)), 1e-13)
            << "pdf x=" << r.x;
    }
    EXPECT_LE(relerr(tn_.getQuantile(0.05), -1.4722616410327654), 1e-12);
    EXPECT_NEAR(tn_.getQuantile(0.5), 0.0, 1e-15);
    EXPECT_LE(relerr(tn_.getQuantile(0.95), 1.4722616410327654), 1e-12);

    // TN(1.5, 2, -1, 4) — asymmetric window around a shifted mean
    auto t2 = TruncatedNormalDistribution::create(1.5, 2.0, -1.0, 4.0).unwrap();
    EXPECT_LE(relerr(t2.getNormalizationConstant(), 0.78870045266628948), 5e-15);
    EXPECT_LE(relerr(t2.getVariance(), 1.6841767394508728), 1e-13);
    EXPECT_LE(relerr(t2.getCumulativeProbability(0.5), 0.25724311983497285), 1e-13);
    EXPECT_NEAR(t2.getLogProbability(1.5), -1.3747170291260225, 1e-13);
    EXPECT_LE(relerr(t2.getQuantile(0.95), 3.6154993019596243), 1e-12);
    EXPECT_LE(relerr(t2.getEntropy(), 1.5852391215573816), 1e-13);
}

//==============================================================================
// THE core hazard: same-tail windows (mpmath dps=60, survival-domain refs)
//==============================================================================

TEST_F(TruncatedNormalEnhancedTest, SameTailWindow_5_6) {
    // TN(0,1,5,6): the naive Z = Phi(6) - Phi(5) subtracts 1-2.87e-7 from
    // 1-9.9e-10 — ~7 significant digits survive. The regime-split form keeps
    // full precision; these gates fail against the naive form by ~1e8x.
    auto t = TruncatedNormalDistribution::create(0.0, 1.0, 5.0, 6.0).unwrap();
    EXPECT_LE(relerr(t.getNormalizationConstant(), 2.8566498423415621e-7), 5e-15);
    EXPECT_LE(relerr(t.getMean(), 5.1831470904771735), 1e-13);
    // Variance loses ~3 digits to the (1 + eta - delta^2) cancellation at
    // this depth (terms ~27 vs result 0.029) — that is the moment formula's
    // conditioning, not Z's; documented in the header.
    EXPECT_LE(relerr(t.getVariance(), 0.029452430768483057), 1e-11);
    EXPECT_LE(relerr(t.getSkewness(), 1.5278985329995121), 1e-10);
    EXPECT_LE(relerr(t.getKurtosis(), 2.4918697912781989), 1e-9);
    EXPECT_LE(relerr(t.getEntropy(), -0.70227446717954450), 1e-11);

    const struct {
        double x, cdf, logpdf;
    } rows[] = {
        {5.1, 0.40895747680672739, 1.1445075633247806},
        {5.25, 0.73723409706272864, 0.36825756332478061},
        {5.5, 0.93697871347755665, -0.97549243667521939},
        {5.75, 0.98783335375113732, -2.3817424366752194},
        {5.9, 0.99709127732161727, -3.2554924366752194},
    };
    for (const auto& r : rows) {
        EXPECT_LE(relerr(t.getCumulativeProbability(r.x), r.cdf), 1e-13) << "cdf x=" << r.x;
        // LogPDF embeds log Z = -15.068... — exact log space, so a naive-Z
        // implementation shifts every value by its log-Z error.
        EXPECT_NEAR(t.getLogProbability(r.x), r.logpdf, 1e-12) << "logpdf x=" << r.x;
        EXPECT_LE(relerr(t.getProbability(r.x), std::exp(r.logpdf)), 1e-12)
            << "pdf x=" << r.x;
    }

    // Quantiles (bisection references at dps=60) and round trips.
    EXPECT_LE(relerr(t.getQuantile(0.01), 5.0019307356682967), 1e-13);
    EXPECT_LE(relerr(t.getQuantile(0.5), 5.1313717632839192), 1e-13);
    EXPECT_LE(relerr(t.getQuantile(0.99), 5.7751933896165602), 1e-13);
    // Round-trip budget: rel |CDF(Q(p)) - p| ~ eps/p (numerator
    // amplification, see file banner) plus the quantile's own law error.
    for (double p : {0.01, 0.1, 0.5, 0.9, 0.99}) {
        const double q = t.getQuantile(p);
        EXPECT_LE(relerr(t.getCumulativeProbability(q), p), 1e-16 / p + 1e-12)
            << "round trip p=" << p;
    }
}

TEST_F(TruncatedNormalEnhancedTest, SameTailWindow_8_9) {
    // TN(0,1,8,9): Z = 6.22e-16 — the naive Phi-difference retains ZERO
    // significant digits here (both Phi values round to within one ulp of 1).
    auto t = TruncatedNormalDistribution::create(0.0, 1.0, 8.0, 9.0).unwrap();
    // Z budget 2e-14: bounded by std::erfc's own relative accuracy at the
    // tail argument, NOT by the formulation (measured 5.7e-15 on MSVC at
    // z=8; a cancelling Phi-difference would miss by ~1e16x here).
    EXPECT_LE(relerr(t.getNormalizationConstant(), 6.2198319858658303e-16), 2e-14);
    EXPECT_LE(relerr(t.getMean(), 8.1211889929797971), 1e-13);
    EXPECT_LE(relerr(t.getVariance(), 0.014148542782748111), 1e-10);
    EXPECT_LE(relerr(t.getEntropy(), -1.1107504589929957), 1e-10);

    const struct {
        double x, cdf, logpdf;
    } rows[] = {
        {8.1, 0.55837540142012459, 1.2896800602324754},
        {8.5, 0.98494062861682903, -2.0303199397675246},
        {8.9, 0.99973250949981180, -5.5103199397675246},
    };
    for (const auto& r : rows) {
        EXPECT_LE(relerr(t.getCumulativeProbability(r.x), r.cdf), 1e-13) << "cdf x=" << r.x;
        EXPECT_NEAR(t.getLogProbability(r.x), r.logpdf, 1e-12) << "logpdf x=" << r.x;
    }
    EXPECT_LE(relerr(t.getQuantile(0.01), 8.0012371990595918), 1e-13);
    EXPECT_LE(relerr(t.getQuantile(0.5), 8.0848888990181664), 1e-13);
    EXPECT_LE(relerr(t.getQuantile(0.99), 8.5467036811274183), 1e-13);
    for (double p : {0.01, 0.5, 0.99}) {
        const double q = t.getQuantile(p);
        EXPECT_LE(relerr(t.getCumulativeProbability(q), p), 1e-16 / p + 1e-12)
            << "round trip p=" << p;
    }
}

TEST_F(TruncatedNormalEnhancedTest, OneSidedTail_10_Inf) {
    // TN(0,1,10,+inf): one-sided far-tail window, Z = Q(10) = 7.62e-24.
    auto t = TruncatedNormalDistribution::create(0.0, 1.0, 10.0, kInf).unwrap();
    // Z budget 2e-14: std::erfc's own relative accuracy at z=10 (measured
    // 9.1e-15 on MSVC) — the formulation adds nothing on a one-sided window.
    EXPECT_LE(relerr(t.getNormalizationConstant(), 7.6198530241605261e-24), 2e-14);
    EXPECT_LE(relerr(t.getMean(), 10.098093233962512), 1e-13);
    // Variance: moment-formula cancellation (~102 vs result 0.0094) on top
    // of erfc's tail error — measured 1.008e-10 on MSVC, budget 5e-10.
    EXPECT_LE(relerr(t.getVariance(), 0.0094453778256562612), 5e-10);
    EXPECT_EQ(t.getSupportUpperBound(), kInf);

    const struct {
        double x, cdf, logpdf;
    } rows[] = {
        {10.1, 0.63751145028564295, 1.3073466173077978},
        {10.5, 0.99433190337908775, -2.8126533826922022},
        {11.0, 0.99997492524372267, -8.1876533826922022},
        {12.0, 0.99999999976686137, -19.687653382692202},
        {15.0, 1.0, -60.187653382692202},
    };
    for (const auto& r : rows) {
        EXPECT_LE(relerr(t.getCumulativeProbability(r.x), r.cdf), 1e-13) << "cdf x=" << r.x;
        EXPECT_NEAR(t.getLogProbability(r.x), r.logpdf, 1e-11) << "logpdf x=" << r.x;
    }
    // Deep-in-window pdf underflowing territory stays exact in log space:
    EXPECT_LE(relerr(t.getProbability(15.0), 7.2582890146411067e-27), 1e-11);

    EXPECT_LE(relerr(t.getQuantile(1e-6), 10.000000099028646), 1e-12);
    EXPECT_LE(relerr(t.getQuantile(0.5), 10.068411836081429), 1e-13);
    EXPECT_LE(relerr(t.getQuantile(0.999999), 11.286852290251097), 1e-12);
    // Round trips including the eps/p amplification at p = 1e-6 (~4.4e-10)
    // and the cdf's quantization across one ulp of q, pdf(q)·ulp(q) —
    // ~1.8e-8 relative at p = 1e-6. The old budget without that term was
    // met only while the CDF and the quantile inverted the same erfc
    // expressions, so the quantization cancelled; the near-lower-bound
    // series CDF evaluates independently and sits ~1e-14 from the true
    // value at the stored q (mpmath dps=60 check, 2026-09-03).
    for (double p : {1e-6, 0.5, 0.999999}) {
        const double q = t.getQuantile(p);
        const double quant = t.getProbability(q) * (std::nextafter(q, kInf) - q);
        EXPECT_LE(relerr(t.getCumulativeProbability(q), p), quant / p + 1e-15 / p + 1e-11)
            << "round trip p=" << p;
    }
}

TEST_F(TruncatedNormalEnhancedTest, SupportedWindowEdge_37_38) {
    // The deepest accepted window class: Z = 5.73e-300 (subnormal-adjacent).
    auto r = TruncatedNormalDistribution::create(0.0, 1.0, 37.0, 38.0);
    ASSERT_TRUE(r.isOk()) << "(37,38) must be inside the supported window";
    auto t = std::move(r).unwrap();
    EXPECT_LE(relerr(t.getNormalizationConstant(), 5.7255712225245765e-300), 1e-13);
    EXPECT_LE(relerr(t.getMean(), 37.026987686126990), 1e-12);
    // Variance loses ~6 digits at this depth (terms ~1371 vs result 7.3e-4)
    // — moment-formula conditioning, documented.
    EXPECT_LE(relerr(t.getVariance(), 0.00072727809887746302), 1e-6);

    EXPECT_LE(relerr(t.getCumulativeProbability(37.1), 0.97546599436603788), 1e-12);
    // LogPDF embeds log Z = -689.03...; budget is absolute on the log scale
    // and dominated by |log Z|*ulp ~ 1.5e-13.
    EXPECT_NEAR(t.getLogProbability(37.1), -0.093352956314079091, 1e-12);
    EXPECT_NEAR(t.getLogProbability(37.9), -30.093352956314079, 1e-12);
    EXPECT_LE(relerr(t.getQuantile(0.25), 37.007768709688421), 1e-12);
    EXPECT_LE(relerr(t.getQuantile(0.5), 37.018715326832193), 1e-12);
    EXPECT_LE(relerr(t.getQuantile(0.75), 37.037421210395264), 1e-12);
}

//==============================================================================
// Supported-window policy (log Z underflow — the DECIDED regime)
//==============================================================================

TEST_F(TruncatedNormalEnhancedTest, WindowPolicyEdges) {
    // Accepted: deepest representable class (see SupportedWindowEdge_37_38).
    EXPECT_TRUE(TruncatedNormalDistribution::create(0.0, 1.0, 37.0, 38.0).isOk());
    // Rejected: erfc underflows for both bounds — Z would be exactly 0.
    EXPECT_TRUE(TruncatedNormalDistribution::create(0.0, 1.0, 40.0, 41.0).isError());
    // Rejected: window so narrow the tail-piece difference rounds to 0
    // (a < b holds, but the window mass is below the erfc values' ulp).
    EXPECT_TRUE(TruncatedNormalDistribution::create(0.0, 1.0, 5.0, 5.0 + 1e-18).isError());
    // ... but the same narrowness around the mode is representable.
    EXPECT_TRUE(TruncatedNormalDistribution::create(0.0, 1.0, -1e-18, 1e-18).isOk());

    // Setters enforce the same policy (window slides out of range).
    auto t = TruncatedNormalDistribution::create(0.0, 1.0, 37.0, 38.0).unwrap();
    EXPECT_TRUE(t.trySetMu(-10.0).isError());  // would put the window at 47..48 sigma
    EXPECT_TRUE(t.trySetSigma(0.1).isError()); // would put it at 370..380 sigma
    EXPECT_THROW(t.setMu(-10.0), std::invalid_argument);
    // Unchanged after the rejections:
    EXPECT_DOUBLE_EQ(t.getMu(), 0.0);
    EXPECT_DOUBLE_EQ(t.getSigma(), 1.0);
}

//==============================================================================
// Degeneration to the plain Gaussian (a = -inf, b = +inf) — ALLOWED
//==============================================================================

TEST_F(TruncatedNormalEnhancedTest, DegeneratesToGaussian) {
    auto t = TruncatedNormalDistribution::create(0.5, 2.0).unwrap();  // default ±inf bounds
    EXPECT_EQ(t.getNormalizationConstant(), 1.0) << "Z must be exactly 1";
    auto g = GaussianDistribution::create(0.5, 2.0).unwrap();
    for (double x : {-5.0, -1.0, 0.5, 2.0, 7.0}) {
        EXPECT_LE(relerr(t.getProbability(x), g.getProbability(x)), 1e-14) << "pdf x=" << x;
        EXPECT_NEAR(t.getLogProbability(x), g.getLogProbability(x), 1e-13) << "logpdf x=" << x;
        EXPECT_LE(relerr(t.getCumulativeProbability(x), g.getCumulativeProbability(x)), 1e-13)
            << "cdf x=" << x;
    }
    EXPECT_NEAR(t.getMean(), 0.5, 1e-15);
    EXPECT_NEAR(t.getVariance(), 4.0, 1e-14);
    EXPECT_NEAR(t.getSkewness(), 0.0, 1e-15);
    EXPECT_NEAR(t.getKurtosis(), 0.0, 1e-14);

    // Half-infinite windows collapse correctly too (mpmath dps=40 refs;
    // erfc(-inf) = 2 verified through the public surface, not assumed):
    auto h = TruncatedNormalDistribution::create(0.0, 1.0, -kInf, 0.0).unwrap();
    EXPECT_EQ(h.getNormalizationConstant(), 0.5);
    EXPECT_LE(relerr(h.getMean(), -0.79788456080286536), 1e-14);
    EXPECT_LE(relerr(h.getVariance(), 0.36338022763241866), 1e-13);
    EXPECT_LE(relerr(h.getCumulativeProbability(-1.0), 0.31731050786291410), 1e-14);
    EXPECT_LE(relerr(h.getQuantile(0.05), -1.9599639845400542), 1e-12);
    EXPECT_LE(relerr(h.getQuantile(0.95), -0.062706777943213784), 1e-12);
}

//==============================================================================
// Born-compliant contracts (#103/#104), scalar AND batch
//==============================================================================

TEST_F(TruncatedNormalEnhancedTest, InfinityAndBoundLimitsScalarAndBatch) {
    // Scalar (#103) on the finite window [-2, 2]:
    EXPECT_EQ(tn_.getProbability(kInf), 0.0);
    EXPECT_EQ(tn_.getProbability(-kInf), 0.0);
    EXPECT_EQ(tn_.getLogProbability(kInf), -kInf);
    EXPECT_EQ(tn_.getLogProbability(-kInf), -kInf);
    EXPECT_EQ(tn_.getCumulativeProbability(-kInf), 0.0);
    EXPECT_EQ(tn_.getCumulativeProbability(kInf), 1.0);
    // Exact 0/1 AT the finite truncation bounds:
    EXPECT_EQ(tn_.getCumulativeProbability(-2.0), 0.0);
    EXPECT_EQ(tn_.getCumulativeProbability(2.0), 1.0);
    // The bounds are IN the support: pdf positive, logpdf finite there.
    EXPECT_GT(tn_.getProbability(-2.0), 0.0);
    EXPECT_TRUE(std::isfinite(tn_.getLogProbability(2.0)));

    // And with infinite bounds, ±inf inputs still meet the contract:
    auto g = TruncatedNormalDistribution::create(0.0, 1.0).unwrap();
    EXPECT_EQ(g.getProbability(kInf), 0.0);
    EXPECT_EQ(g.getProbability(-kInf), 0.0);
    EXPECT_EQ(g.getLogProbability(kInf), -kInf);
    EXPECT_EQ(g.getLogProbability(-kInf), -kInf);
    EXPECT_EQ(g.getCumulativeProbability(-kInf), 0.0);
    EXPECT_EQ(g.getCumulativeProbability(kInf), 1.0);

    // Batch, FORCE_VECTORIZED (padded past the SIMD minimum):
    std::vector<double> xs = {-kInf, kInf, -2.0, 2.0, -1.5, -1.0, -0.5, 0.0,
                              0.25,  0.5,  0.75, 1.0, 1.25, 1.5,  1.75, 1.9};
    const size_t n = xs.size();
    std::vector<double> pdf_b(n), lpdf_b(n), cdf_b(n);
    detail::PerformanceHint vec_hint;
    vec_hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    tn_.getProbability(std::span<const double>(xs), std::span<double>(pdf_b), vec_hint);
    tn_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf_b), vec_hint);
    tn_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_b), vec_hint);

    EXPECT_EQ(pdf_b[0], 0.0);
    EXPECT_EQ(pdf_b[1], 0.0);
    EXPECT_EQ(lpdf_b[0], -kInf);
    EXPECT_EQ(lpdf_b[1], -kInf);
    EXPECT_EQ(cdf_b[0], 0.0);
    EXPECT_EQ(cdf_b[1], 1.0);
    EXPECT_EQ(cdf_b[2], 0.0);  // exact at x = a
    EXPECT_EQ(cdf_b[3], 1.0);  // exact at x = b

    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(pdf_b[i], tn_.getProbability(xs[i]), 1e-12) << "pdf lane " << i;
        EXPECT_NEAR(lpdf_b[i], tn_.getLogProbability(xs[i]), 1e-12) << "logpdf lane " << i;
        EXPECT_NEAR(cdf_b[i], tn_.getCumulativeProbability(xs[i]), 1e-12) << "cdf lane " << i;
    }
}

TEST_F(TruncatedNormalEnhancedTest, NaNPropagationScalarAndBatch) {
    EXPECT_TRUE(std::isnan(tn_.getProbability(kNaN)));
    EXPECT_TRUE(std::isnan(tn_.getLogProbability(kNaN)));
    EXPECT_TRUE(std::isnan(tn_.getCumulativeProbability(kNaN)));

    std::vector<double> xs = {kNaN, -1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75,
                              1.0,  1.1,  1.2,  1.3,  1.4, 1.5,  1.6, kNaN};
    const size_t n = xs.size();
    std::vector<double> out(n);
    detail::PerformanceHint vec_hint;
    vec_hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;

    tn_.getProbability(std::span<const double>(xs), std::span<double>(out), vec_hint);
    EXPECT_TRUE(std::isnan(out[0]) && std::isnan(out[n - 1])) << "batch PDF NaN lanes";
    tn_.getLogProbability(std::span<const double>(xs), std::span<double>(out), vec_hint);
    EXPECT_TRUE(std::isnan(out[0]) && std::isnan(out[n - 1])) << "batch LogPDF NaN lanes";
    tn_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out), vec_hint);
    EXPECT_TRUE(std::isnan(out[0]) && std::isnan(out[n - 1])) << "batch CDF NaN lanes";

    // Same for a same-tail window (whole-scalar batch branch):
    auto t = TruncatedNormalDistribution::create(0.0, 1.0, 5.0, 6.0).unwrap();
    std::vector<double> xs2(16, 5.5);
    xs2[3] = kNaN;
    t.getCumulativeProbability(std::span<const double>(xs2), std::span<double>(out), vec_hint);
    EXPECT_TRUE(std::isnan(out[3])) << "same-tail batch CDF NaN lane";
    EXPECT_FALSE(std::isnan(out[4]));
}

TEST_F(TruncatedNormalEnhancedTest, QuantileNeverNaNInOpenInterval) {
    // #104 across window regimes, including the deepest accepted one.
    const double ps[] = {1e-300, 1e-16, 1e-6, 0.25, 0.5, 0.75, 1.0 - 1e-6, 1.0 - 1e-14,
                         0.9999999999999999};
    const struct {
        double mu, sigma, a, b;
    } windows[] = {
        {0.0, 1.0, -2.0, 2.0},   {0.0, 1.0, 5.0, 6.0},   {0.0, 1.0, 8.0, 9.0},
        {0.0, 1.0, 10.0, kInf},  {0.0, 1.0, -kInf, 0.0}, {0.0, 1.0, -kInf, kInf},
        {0.0, 1.0, 37.0, 38.0},
    };
    for (const auto& w : windows) {
        auto t = TruncatedNormalDistribution::create(w.mu, w.sigma, w.a, w.b).unwrap();
        double prev = -kInf;
        for (double p : ps) {
            const double q = t.getQuantile(p);
            EXPECT_FALSE(std::isnan(q)) << "quantile(" << p << ") NaN in window [" << w.a << ","
                                        << w.b << "]";
            EXPECT_TRUE(std::isfinite(q))
                << "quantile(" << p << ") not finite in [" << w.a << "," << w.b << "]";
            EXPECT_GE(q, w.a);
            EXPECT_LE(q, w.b);
            EXPECT_GE(q, prev) << "quantile not monotone at p=" << p << " in [" << w.a << ","
                               << w.b << "]";
            prev = q;
        }
        EXPECT_EQ(t.getQuantile(0.0), w.a);
        EXPECT_EQ(t.getQuantile(1.0), w.b);
    }
    EXPECT_THROW((void)tn_.getQuantile(-0.1), std::invalid_argument);
    EXPECT_THROW((void)tn_.getQuantile(1.1), std::invalid_argument);
}

TEST_F(TruncatedNormalEnhancedTest, BatchSizeMismatchThrows) {
    std::vector<double> in(8), out(7);
    EXPECT_THROW(tn_.getProbability(std::span<const double>(in), std::span<double>(out)),
                 std::invalid_argument);
    EXPECT_THROW(tn_.getLogProbability(std::span<const double>(in), std::span<double>(out)),
                 std::invalid_argument);
    EXPECT_THROW(
        tn_.getCumulativeProbability(std::span<const double>(in), std::span<double>(out)),
        std::invalid_argument);
}

//==============================================================================
// Batch/SIMD consistency
//==============================================================================

TEST_F(TruncatedNormalEnhancedTest, VectorizedMatchesScalarStraddling) {
    // Straddling window: CDF batch runs the vector_erf chain with per-lane
    // fixups. Fixed-up lanes (xi <= 0, bound lanes) are bit-identical to
    // scalar; xi > 0 lanes may differ by the vector_erf ulp band (<~ 2e-15).
    const size_t N = 1024;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -2.5 + 5.0 * static_cast<double>(i) / static_cast<double>(N - 1);

    detail::PerformanceHint hint_vec, hint_scl;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    tn_.getCumulativeProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    tn_.getCumulativeProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_vec[i], out_scl[i], 2e-15) << "CDF mismatch at x=" << xs[i];
        if (xs[i] <= 0.0) {  // fixed-up or bound lanes: bit-identical
            EXPECT_EQ(out_vec[i], out_scl[i]) << "fixed-up lane not bit-identical x=" << xs[i];
        }
    }

    tn_.getProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    tn_.getProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-12) << "PDF mismatch at x=" << xs[i];

    tn_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    tn_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i) {
        if (std::isfinite(out_scl[i])) {
            EXPECT_NEAR(out_vec[i], out_scl[i], 1e-12) << "LogPDF mismatch at x=" << xs[i];
        } else {
            EXPECT_EQ(out_vec[i], out_scl[i]) << "LogPDF -inf lane mismatch at x=" << xs[i];
        }
    }
}

TEST_F(TruncatedNormalEnhancedTest, SameTailBatchIdenticalToScalar) {
    // Same-tail windows route every batch CDF lane through the scalar
    // regime-split kernel — bit-identical by construction.
    auto t = TruncatedNormalDistribution::create(0.0, 1.0, 5.0, 6.0).unwrap();
    const size_t N = 512;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = 4.9 + 1.2 * static_cast<double>(i) / static_cast<double>(N - 1);

    detail::PerformanceHint hint_vec, hint_scl;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;
    t.getCumulativeProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    t.getCumulativeProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(out_vec[i], out_scl[i]) << "same-tail CDF lane not bit-identical x=" << xs[i];
}

TEST_F(TruncatedNormalEnhancedTest, CDFNearLowerBoundRelativeAccuracy) {
    // Every difference-form CDF regime subtracts two independently rounded
    // normal-CDF pieces, so as x → a⁺ the true numerator vanishes while the
    // absolute rounding error of the pieces does not: relative error reaches
    // 0.32–0.45 at a+1ulp (v2.4.0 sweep finding). References: mpmath dps=60,
    // (Q(α)−Q(ξ))/(Q(α)−Q(β)) evaluated at the exact double of x.
    struct Case {
        double mu, sigma, a, b, x, ref;
    };
    const Case cases[] = {
        // Straddling window (−2,2): left-half Φ-difference regime.
        {0.0, 1.0, -2.0, 2.0, std::nextafter(-2.0, 0.0), 1.2559880716027472e-17},
        {0.0, 1.0, -2.0, 2.0, -2.0 + 1e-12, 5.6569702745044295e-14},
        {0.0, 1.0, -2.0, 2.0, -2.0 + 1e-9, 5.6564678849265993e-11},
        {0.0, 1.0, -2.0, 2.0, -2.0 + 1e-6, 5.6564730672568188e-8},
        // Same-tail right window (10,12) on N(0,2): α=5, β=6, Q-difference regime.
        {0.0, 2.0, 10.0, 12.0, std::nextafter(10.0, 11.0), 4.6224502897130269e-15},
        {0.0, 2.0, 10.0, 12.0, 10.0 + 1e-12, 2.6024395131051866e-12},
        {0.0, 2.0, 10.0, 12.0, 10.0 + 1e-9, 2.6022083873411935e-9},
        {0.0, 2.0, 10.0, 12.0, 10.0 + 1e-6, 2.6022049205811786e-6},
    };
    for (const auto& c : cases) {
        auto t = TruncatedNormalDistribution::create(c.mu, c.sigma, c.a, c.b).unwrap();
        const double got = t.getCumulativeProbability(c.x);
        EXPECT_NEAR(got / c.ref, 1.0, 1e-12)
            << "CDF(" << c.x << ") in [" << c.a << "," << c.b << "]: got " << got << ", ref "
            << c.ref;
    }

    // Both sides of the near-bound/difference-form handoff stay accurate
    // (these pass before and after the fix and pin the switch).
    {
        auto t = TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap();
        EXPECT_NEAR(t.getCumulativeProbability(-1.9) / 0.0062508428678860847, 1.0, 1e-13);
        EXPECT_NEAR(t.getCumulativeProbability(-1.8) / 0.013808476489002885, 1.0, 1e-13);
        auto u = TruncatedNormalDistribution::create(0.0, 2.0, 10.0, 12.0).unwrap();
        EXPECT_NEAR(u.getCumulativeProbability(10.05) / 0.12229466549084977, 1.0, 1e-13);
        EXPECT_NEAR(u.getCumulativeProbability(10.5) / 0.73723409706272864, 1.0, 1e-13);
    }

    // Near-bound batch lanes are bit-identical to scalar in both window
    // regimes (they are routed through the scalar regime-split kernel).
    {
        detail::PerformanceHint hint_vec;
        hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
        for (const auto& c : cases) {
            auto t = TruncatedNormalDistribution::create(c.mu, c.sigma, c.a, c.b).unwrap();
            vector<double> xs(64, c.x), out(64);
            xs[63] = 0.5 * (c.a + c.b);  // one mid-window lane alongside
            t.getCumulativeProbability(span<const double>(xs), span<double>(out), hint_vec);
            EXPECT_EQ(out[0], t.getCumulativeProbability(c.x))
                << "near-bound batch lane not bit-identical at x=" << c.x;
        }
    }
}

//==============================================================================
// Sampling and MLE
//==============================================================================

TEST_F(TruncatedNormalEnhancedTest, SamplingMatchesCDF) {
    // Deterministic KS-style gates: all samples in-window and the empirical
    // CDF within the 1% K-S band (1.63/sqrt(n) ~ 0.0163 at n = 10000).
    const struct {
        double mu, sigma, a, b;
    } windows[] = {
        {0.0, 1.0, -2.0, 2.0},
        {0.0, 1.0, 5.0, 6.0},     // two-sided far tail
        {0.0, 1.0, 10.0, kInf},   // one-sided far tail
    };
    for (const auto& w : windows) {
        auto t = TruncatedNormalDistribution::create(w.mu, w.sigma, w.a, w.b).unwrap();
        mt19937 rng(7);
        const size_t n = 10000;
        auto samples = t.sample(rng, n);
        std::sort(samples.begin(), samples.end());
        double ks = 0.0;
        for (size_t i = 0; i < n; ++i) {
            ASSERT_GE(samples[i], w.a) << "sample below a in [" << w.a << "," << w.b << "]";
            ASSERT_LE(samples[i], w.b) << "sample above b in [" << w.a << "," << w.b << "]";
            const double f = t.getCumulativeProbability(samples[i]);
            const double lo = static_cast<double>(i) / static_cast<double>(n);
            const double hi = static_cast<double>(i + 1) / static_cast<double>(n);
            ks = std::max(ks, std::max(std::fabs(f - lo), std::fabs(f - hi)));
        }
        EXPECT_LT(ks, 0.0163) << "KS statistic " << ks << " for window [" << w.a << "," << w.b
                              << "]";
    }
}

TEST_F(TruncatedNormalEnhancedTest, MLEFit) {
    // Bounds KNOWN (fixed); recover (mu, sigma) from samples.
    mt19937 rng(42);
    auto source = TruncatedNormalDistribution::create(0.5, 2.0, -1.0, 4.0).unwrap();
    const auto data = source.sample(rng, 2000);
    auto fitted = TruncatedNormalDistribution::create(0.0, 1.0, -1.0, 4.0).unwrap();
    fitted.fit(data);
    EXPECT_NEAR(fitted.getMu(), 0.5, 0.25) << "fitted mu";
    EXPECT_NEAR(fitted.getSigma(), 2.0, 0.3) << "fitted sigma";
    // Bounds unchanged by fit (the scope decision):
    EXPECT_DOUBLE_EQ(fitted.getLowerBound(), -1.0);
    EXPECT_DOUBLE_EQ(fitted.getUpperBound(), 4.0);

    // Data outside the window is rejected.
    EXPECT_THROW(fitted.fit({0.0, 1.0, 5.0}), std::invalid_argument);
    // Degenerate data is rejected.
    EXPECT_THROW(fitted.fit({1.0, 1.0, 1.0}), std::invalid_argument);
}

// Invalid parameters rejected
TEST_F(TruncatedNormalEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(TruncatedNormalDistribution::create(kNaN, 1.0, -2.0, 2.0).isError());
    EXPECT_TRUE(TruncatedNormalDistribution::create(0.0, 0.0, -2.0, 2.0).isError());
    EXPECT_TRUE(TruncatedNormalDistribution::create(0.0, -1.0, -2.0, 2.0).isError());
    EXPECT_TRUE(TruncatedNormalDistribution::create(0.0, 1.0, 2.0, -2.0).isError());
    EXPECT_TRUE(TruncatedNormalDistribution::create(0.0, 1.0, 2.0, 2.0).isError());
    EXPECT_TRUE(TruncatedNormalDistribution::create(0.0, 1.0, kNaN, 2.0).isError());
    EXPECT_TRUE(TruncatedNormalDistribution::create(0.0, 1.0, -2.0, kNaN).isError());
    EXPECT_TRUE(TruncatedNormalDistribution::create(kInf, 1.0, -2.0, 2.0).isError());

    auto d = TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap();
    EXPECT_TRUE(d.trySetSigma(-1.0).isError());
    EXPECT_TRUE(d.trySetUpperBound(-3.0).isError());  // would invert the window
    EXPECT_DOUBLE_EQ(d.getSigma(), 1.0);
    EXPECT_DOUBLE_EQ(d.getUpperBound(), 2.0);
}

}  // namespace stats

//==============================================================================
// DistTraits specialization for stats::TruncatedNormalDistribution
//
// The shared suite's make() constructs the 4-parameter fixture; the harness
// itself is parameter-count agnostic (everything flows through make()).
//==============================================================================
template <>
struct stats::tests::DistTraits<stats::TruncatedNormalDistribution>
    : stats::tests::DistTraitsDefaults {
    static stats::TruncatedNormalDistribution make() {
        return stats::TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap();
    }
    static std::vector<double> domain() { return {-1.5, -0.5, 0.0, 0.5, 1.5}; }
    static double batch_lo() { return -2.0; }
    static double batch_hi() { return 2.0; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] {
                return stats::TruncatedNormalDistribution::create(0.0, -1.0, -2.0, 2.0)
                    .isError();
            },
            [] {
                return stats::TruncatedNormalDistribution::create(0.0, 1.0, 2.0, 2.0).isError();
            },
            [] {
                return stats::TruncatedNormalDistribution::create(0.0, 1.0, 40.0, 41.0)
                    .isError();
            },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(TruncatedNormal, DistributionEnhancedTest,
                               ::testing::Types<stats::TruncatedNormalDistribution>);
