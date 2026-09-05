// tests/test_half_normal_enhanced.cpp
//
// Enhanced tests for HalfNormalDistribution (#57): mpmath-referenced accuracy,
// the shared typed suite, and the Born-compliant edge contracts (#103 ±inf
// limits, NaN propagation, #104 quantile never-NaN, batch size-mismatch
// throws) asserted on BOTH the scalar and the forced-vectorized batch paths.
//
// Reference provenance: every constant tagged "mpmath dps=40" below was
// computed with mpmath at 40 significant digits from the exact double inputs
// (scratchpad script, 2026-09-03):
//   CDF(x)   = erf(x/(sigma*sqrt(2)))
//   PDF(x)   = sqrt(2/pi)/sigma * exp(-x^2/(2 sigma^2))
//   quantile = sigma*sqrt(2)*erfinv(p)
// printed to 17 significant digits (exact double round-trip).

#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/half_normal.h"

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
}  // namespace

namespace stats {

class HalfNormalEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = stats::HalfNormalDistribution::create(1.0);
        ASSERT_TRUE(r.isOk());
        h1_ = std::move(r).unwrap();
    }
    HalfNormalDistribution h1_;  // standard Half-Normal (σ=1)
};

//==============================================================================
// Accuracy vs mpmath references (dps=40; see file banner for provenance)
//==============================================================================

TEST_F(HalfNormalEnhancedTest, CDFAccuracyVsMpmath) {
    // {x, CDF reference}. CDF = erf(x/√2) evaluated directly — the whole
    // support maps to erf arguments >= 0, so full relative precision is
    // expected everywhere (no #49 cancellation band exists here).
    const struct {
        double x, ref;
    } rows[] = {
        {0.1, 0.079655674554057963}, {0.5, 0.38292492254802621}, {1.0, 0.68268949213708590},
        {2.0, 0.95449973610364159},  {3.0, 0.99730020393673981}, {5.0, 0.99999942669685624},
        {8.0, 0.99999999999999876},
    };
    for (const auto& r : rows) {
        const double got = h1_.getCumulativeProbability(r.x);
        const double rel = std::fabs(got - r.ref) / r.ref;
        EXPECT_LE(rel, 5e-15) << "CDF(" << r.x << ") rel error " << rel;
    }
    // sigma=2.5 spot checks (mpmath dps=40)
    auto h25 = HalfNormalDistribution::create(2.5).unwrap();
    EXPECT_NEAR(h25.getCumulativeProbability(0.5), 0.15851941887820605, 1e-15);
    EXPECT_NEAR(h25.getCumulativeProbability(2.5), 0.68268949213708590, 1e-15);
}

TEST_F(HalfNormalEnhancedTest, PDFAccuracyVsMpmath) {
    // Relative budget 1e-13: exp() contributes rel ~ |x²/(2σ²)|·2⁻⁵²
    // (≈ 7e-15 at x=8), plus ~1 ulp each from the multiply chain.
    const struct {
        double x, ref;
    } rows[] = {
        {0.1, 0.79390509495402353},   {0.5, 0.70413065352859896},
        {1.0, 0.48394144903828670},   {2.0, 0.10798193302637610},
        {3.0, 0.0088636968238760144}, {5.0, 2.9734390294685954e-6},
        {8.0, 1.0104542167073785e-14},
    };
    for (const auto& r : rows) {
        const double got = h1_.getProbability(r.x);
        const double rel = std::fabs(got - r.ref) / r.ref;
        EXPECT_LE(rel, 1e-13) << "PDF(" << r.x << ") rel error " << rel;
        // LogPDF is exact log space: budget abs 1e-13 on the log scale.
        EXPECT_NEAR(h1_.getLogProbability(r.x), std::log(r.ref), 1e-13);
    }
    auto h25 = HalfNormalDistribution::create(2.5).unwrap();
    EXPECT_NEAR(h25.getProbability(0.5), 0.31283415518036470, 1e-15);
    EXPECT_NEAR(h25.getProbability(7.5), 0.0035454787295504057, 1e-16);
}

TEST_F(HalfNormalEnhancedTest, QuantileAccuracyVsMpmath) {
    // Central and moderate-tail quantiles (mpmath dps=40).
    EXPECT_NEAR(h1_.getQuantile(0.5), 0.67448975019608174, 1e-12);
    // Deep lower tail: erf_inv seed + Newton polish, full relative precision.
    // (The polish matters: bare detail::erf_inv stops at an absolute 1e-12
    // Halley tolerance and returned 1.4e-8 RELATIVE error here — this gate
    // failed against the unpolished implementation, 2026-09-03.)
    {
        const double got = h1_.getQuantile(1e-10);
        const double ref = 1.2533141373155003e-10;
        EXPECT_LE(std::fabs(got - ref) / ref, 1e-13) << "q(1e-10)=" << got;
    }
    // Upper tail p=0.999999: conditioning law limits any double
    // implementation to |δx| ≈ ulp(1)/pdf(x) ≈ 4.3e-11 here; budget 1e-9
    // gives ~20x headroom over the law while still catching a wrong branch.
    EXPECT_NEAR(h1_.getQuantile(0.999999), 4.8916384756985904, 1e-9);
}

TEST_F(HalfNormalEnhancedTest, MomentFormulasVsMpmath) {
    // mpmath dps=40: mean, var, skew, excess kurtosis, median, entropy for σ=1
    EXPECT_NEAR(h1_.getMean(), 0.79788456080286536, 1e-15);
    EXPECT_NEAR(h1_.getVariance(), 0.36338022763241866, 1e-15);
    EXPECT_NEAR(h1_.getSkewness(), 0.99527174643115604, 1e-14);
    EXPECT_NEAR(h1_.getKurtosis(), 0.86917730360597412, 1e-14);
    EXPECT_NEAR(h1_.getMedian(), 0.67448975019608174, 1e-12);
    EXPECT_NEAR(h1_.getEntropy(), 0.72579135264472743, 1e-14);
    EXPECT_EQ(h1_.getMode(), 0.0);

    auto h25 = HalfNormalDistribution::create(2.5).unwrap();
    EXPECT_NEAR(h25.getMean(), 1.9947114020071634, 1e-14);
    EXPECT_NEAR(h25.getVariance(), 2.2711264227026166, 1e-14);
    // Skewness/kurtosis are σ-invariant constants
    EXPECT_NEAR(h25.getSkewness(), h1_.getSkewness(), 1e-15);
    EXPECT_NEAR(h25.getKurtosis(), h1_.getKurtosis(), 1e-15);
}

//==============================================================================
// Born-compliant contracts (#103/#104), scalar AND batch
//==============================================================================

TEST_F(HalfNormalEnhancedTest, InfinityLimitsScalarAndBatch) {
    // (#103) scalar: pdf(±inf)=0, logpdf(±inf)=−inf, cdf(−inf)=0, cdf(+inf)=1
    EXPECT_EQ(h1_.getProbability(kInf), 0.0);
    EXPECT_EQ(h1_.getProbability(-kInf), 0.0);
    EXPECT_EQ(h1_.getLogProbability(kInf), -kInf);
    EXPECT_EQ(h1_.getLogProbability(-kInf), -kInf);
    EXPECT_EQ(h1_.getCumulativeProbability(-kInf), 0.0);
    EXPECT_EQ(h1_.getCumulativeProbability(kInf), 1.0);

    // batch (FORCE_VECTORIZED so the SIMD kernel + fixup actually runs):
    // pad with in-support values so the batch clears the SIMD minimum size.
    std::vector<double> xs = {-kInf, kInf, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
                              3.0,   3.5,  4.0, 4.5, 5.0, 5.5, 6.0, 6.5};
    const size_t n = xs.size();
    std::vector<double> pdf_b(n), lpdf_b(n), cdf_b(n);
    detail::PerformanceHint vec_hint;
    vec_hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    h1_.getProbability(std::span<const double>(xs), std::span<double>(pdf_b), vec_hint);
    h1_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf_b), vec_hint);
    h1_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_b), vec_hint);

    EXPECT_EQ(pdf_b[0], 0.0);    // pdf(-inf)
    EXPECT_EQ(pdf_b[1], 0.0);    // pdf(+inf)
    EXPECT_EQ(lpdf_b[0], -kInf); // logpdf(-inf)
    EXPECT_EQ(lpdf_b[1], -kInf); // logpdf(+inf)
    EXPECT_EQ(cdf_b[0], 0.0);    // cdf(-inf)
    EXPECT_EQ(cdf_b[1], 1.0);    // cdf(+inf)

    // batch ≡ scalar on every lane (exact for the specials; finite lanes
    // may differ by the documented vector-kernel ulp band)
    for (size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(pdf_b[i], h1_.getProbability(xs[i]), 1e-12) << "pdf lane " << i;
        EXPECT_NEAR(lpdf_b[i], h1_.getLogProbability(xs[i]), 1e-12) << "logpdf lane " << i;
        EXPECT_NEAR(cdf_b[i], h1_.getCumulativeProbability(xs[i]), 1e-12) << "cdf lane " << i;
    }
}

TEST_F(HalfNormalEnhancedTest, NaNPropagationScalarAndBatch) {
    // (b) NaN in → NaN out, scalar
    EXPECT_TRUE(std::isnan(h1_.getProbability(kNaN)));
    EXPECT_TRUE(std::isnan(h1_.getLogProbability(kNaN)));
    EXPECT_TRUE(std::isnan(h1_.getCumulativeProbability(kNaN)));

    // batch, forced vectorized
    std::vector<double> xs = {kNaN, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                              4.0,  4.5, 5.0, 5.5, 6.0, 6.5, 7.0, kNaN};
    const size_t n = xs.size();
    std::vector<double> out(n);
    detail::PerformanceHint vec_hint;
    vec_hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;

    h1_.getProbability(std::span<const double>(xs), std::span<double>(out), vec_hint);
    EXPECT_TRUE(std::isnan(out[0]) && std::isnan(out[n - 1])) << "batch PDF NaN lanes";
    h1_.getLogProbability(std::span<const double>(xs), std::span<double>(out), vec_hint);
    EXPECT_TRUE(std::isnan(out[0]) && std::isnan(out[n - 1])) << "batch LogPDF NaN lanes";
    h1_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out), vec_hint);
    EXPECT_TRUE(std::isnan(out[0]) && std::isnan(out[n - 1])) << "batch CDF NaN lanes";
}

TEST_F(HalfNormalEnhancedTest, QuantileNeverNaNInOpenInterval) {
    // (c) #104: finite, non-NaN, non-negative for every p in (0,1)
    const double ps[] = {1e-300, 1e-100, 1e-16, 1e-10, 0.01, 0.5,
                         0.99,   1.0 - 1e-10, 1.0 - 1e-14, 0.9999999999999999};
    for (double p : ps) {
        const double q = h1_.getQuantile(p);
        EXPECT_FALSE(std::isnan(q)) << "quantile(" << p << ") is NaN";
        EXPECT_TRUE(std::isfinite(q)) << "quantile(" << p << ") not finite";
        EXPECT_GE(q, 0.0) << "quantile(" << p << ") negative";
    }
    // Monotone across the grid
    double prev = -1.0;
    for (double p : ps) {
        const double q = h1_.getQuantile(p);
        EXPECT_GE(q, prev) << "quantile not monotone at p=" << p;
        prev = q;
    }
    EXPECT_EQ(h1_.getQuantile(0.0), 0.0);
    EXPECT_EQ(h1_.getQuantile(1.0), kInf);
    EXPECT_THROW((void)h1_.getQuantile(-0.1), std::invalid_argument);
    EXPECT_THROW((void)h1_.getQuantile(1.1), std::invalid_argument);
}

TEST_F(HalfNormalEnhancedTest, BatchSizeMismatchThrows) {
    // (d) all three overloads throw on size mismatch
    std::vector<double> in(8), out(7);
    EXPECT_THROW(
        h1_.getProbability(std::span<const double>(in), std::span<double>(out)),
        std::invalid_argument);
    EXPECT_THROW(
        h1_.getLogProbability(std::span<const double>(in), std::span<double>(out)),
        std::invalid_argument);
    EXPECT_THROW(
        h1_.getCumulativeProbability(std::span<const double>(in), std::span<double>(out)),
        std::invalid_argument);
}

//==============================================================================
// Batch/SIMD consistency and management
//==============================================================================

// VECTORIZED matches SCALAR strategy (exercises the SIMD pipelines
// deterministically; includes out-of-support and boundary lanes)
TEST_F(HalfNormalEnhancedTest, VectorizedMatchesScalar) {
    const size_t N = 1024;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -1.0 + 0.01 * static_cast<double>(i);  // spans [-1, 9.23]: fixup + bulk

    detail::PerformanceHint hint_vec, hint_scl;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    h1_.getProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    h1_.getProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-12) << "PDF mismatch at i=" << i;

    h1_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    h1_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i)
        EXPECT_EQ(std::isinf(out_vec[i]), std::isinf(out_scl[i])) << "LogPDF inf at i=" << i;
    for (size_t i = 0; i < N; ++i) {
        if (std::isfinite(out_scl[i])) {
            EXPECT_NEAR(out_vec[i], out_scl[i], 1e-12) << "LogPDF mismatch at i=" << i;
        }
    }

    h1_.getCumulativeProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    h1_.getCumulativeProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-15) << "CDF mismatch at i=" << i;
}

// MLE from samples
TEST_F(HalfNormalEnhancedTest, MLEFit) {
    mt19937 rng(42);
    auto source = HalfNormalDistribution::create(2.5).unwrap();
    const auto data = source.sample(rng, 500);
    auto fitted = HalfNormalDistribution::create(1.0).unwrap();
    fitted.fit(data);
    EXPECT_NEAR(fitted.getSigma(), 2.5, 0.3) << "Fitted sigma should be near 2.5";

    // Negative data rejected
    EXPECT_THROW(fitted.fit({1.0, -0.5, 2.0}), std::invalid_argument);
    // All-zero data rejected (σ̂ = 0 is not a valid scale)
    EXPECT_THROW(fitted.fit({0.0, 0.0, 0.0}), std::invalid_argument);
}

// Sampling agreement with the analytic CDF (KS-style sanity bound)
TEST_F(HalfNormalEnhancedTest, SamplingMatchesCDF) {
    mt19937 rng(7);
    const size_t n = 10000;
    auto samples = h1_.sample(rng, n);
    std::sort(samples.begin(), samples.end());
    double ks = 0.0;
    for (size_t i = 0; i < n; ++i) {
        EXPECT_GE(samples[i], 0.0);
        const double f = h1_.getCumulativeProbability(samples[i]);
        const double lo = static_cast<double>(i) / static_cast<double>(n);
        const double hi = static_cast<double>(i + 1) / static_cast<double>(n);
        ks = std::max(ks, std::max(std::fabs(f - lo), std::fabs(f - hi)));
    }
    // K-S 1% critical value ≈ 1.63/√n ≈ 0.0163 at n=10000
    EXPECT_LT(ks, 0.0163) << "KS statistic " << ks;
}

// Setter propagates
TEST_F(HalfNormalEnhancedTest, SetterPropagates) {
    auto d = HalfNormalDistribution::create(1.0).unwrap();
    EXPECT_NEAR(d.getMean(), std::sqrt(2.0 / M_PI), 1e-14);
    d.setSigma(2.0);
    EXPECT_NEAR(d.getMean(), 2.0 * std::sqrt(2.0 / M_PI), 1e-14);
    d.setParameters(1.0);
    EXPECT_NEAR(d.getMean(), std::sqrt(2.0 / M_PI), 1e-14);
}

// Invalid parameters rejected
TEST_F(HalfNormalEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(HalfNormalDistribution::create(-1.0).isError());
    EXPECT_TRUE(HalfNormalDistribution::create(0.0).isError());
    EXPECT_TRUE(HalfNormalDistribution::create(kNaN).isError());
    EXPECT_TRUE(HalfNormalDistribution::create(kInf).isError());

    auto d = HalfNormalDistribution::create(1.0).unwrap();
    EXPECT_TRUE(d.trySetSigma(-1.0).isError());
    EXPECT_DOUBLE_EQ(d.getSigma(), 1.0);
}

}  // namespace stats

//==============================================================================
// DistTraits specialization for stats::HalfNormalDistribution
//==============================================================================
template <>
struct stats::tests::DistTraits<stats::HalfNormalDistribution> : stats::tests::DistTraitsDefaults {
    static stats::HalfNormalDistribution make() {
        return stats::HalfNormalDistribution::create(1.0).unwrap();
    }
    static std::vector<double> domain() { return {0.1, 0.5, 1.0, 2.0, 3.0}; }
    static double batch_lo() { return 0.0; }
    static double batch_hi() { return 8.0; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::HalfNormalDistribution::create(-1.0).isError(); },
            [] { return stats::HalfNormalDistribution::create(0.0).isError(); },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(HalfNormal, DistributionEnhancedTest,
                               ::testing::Types<stats::HalfNormalDistribution>);
