#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

// Enhanced tests for LogisticDistribution.
//
// Every "expect" constant below was produced with mpmath at mp.dps = 60 and
// printed to 17 significant digits, using exactly the stable formulations under
// test:  pdf = e^(-|z|)/(s(1+e^(-|z|))^2),  logpdf = -|z| - 2*log1p(e^(-|z|)) - log s,
// cdf = 1/(1+e^(-z)),  quantile = mu + s*(log p - log1p(-p)).
// Tail probabilities were evaluated at the *exact double* value of the C++
// literal (e.g. the double nearest 1 - 1e-10, not the decimal 0.9999999999).

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/logistic.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

// DistTraits specialisation for LogisticDistribution
template <>
struct stats::tests::DistTraits<stats::LogisticDistribution> : stats::tests::DistTraitsDefaults {
    static stats::LogisticDistribution make() {
        return stats::LogisticDistribution::create(0.0, 1.0).unwrap();  // standard logistic
    }
    static std::vector<double> domain() { return {-3.0, -1.0, 0.0, 1.0, 3.0}; }
    static double batch_lo() { return -5.0; }
    static double batch_hi() { return 5.0; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::LogisticDistribution::create(0.0, 0.0).isError(); },
            [] { return stats::LogisticDistribution::create(0.0, -1.0).isError(); },
            [] {
                return stats::LogisticDistribution::create(std::numeric_limits<double>::infinity(),
                                                           1.0)
                    .isError();
            },
            [] {
                return stats::LogisticDistribution::create(
                           0.0, std::numeric_limits<double>::quiet_NaN())
                    .isError();
            },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Logistic, DistributionEnhancedTest,
                               ::testing::Types<stats::LogisticDistribution>);

// ─── Per-distribution fixture ────────────────────────────────────────────────

namespace stats {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

/// The four dispatch strategies that reach a distinct kernel implementation.
/// AUTO and FORCE_SCALAR share the scalar element path; FORCE_VECTORIZED hits
/// the SIMD pipeline; FORCE_PARALLEL hits the parallel fallback lambda.
std::vector<detail::PerformanceHint> allStrategies() {
    using PS = detail::PerformanceHint::PreferredStrategy;
    std::vector<detail::PerformanceHint> hints(4);
    hints[0].strategy = PS::AUTO;
    hints[1].strategy = PS::FORCE_SCALAR;
    hints[2].strategy = PS::FORCE_VECTORIZED;
    hints[3].strategy = PS::FORCE_PARALLEL;
    return hints;
}

const char* strategyName(const detail::PerformanceHint& h) {
    using PS = detail::PerformanceHint::PreferredStrategy;
    switch (h.strategy) {
        case PS::AUTO:
            return "AUTO";
        case PS::FORCE_SCALAR:
            return "FORCE_SCALAR";
        case PS::FORCE_VECTORIZED:
            return "FORCE_VECTORIZED";
        case PS::FORCE_PARALLEL:
            return "FORCE_PARALLEL";
        default:
            return "OTHER";
    }
}

/// Relative-tolerance comparison for reference values that span many decades.
void expectClose(double actual, double expected, double rel, const char* what) {
    const double tol = std::max(rel * std::fabs(expected), 1e-300);
    EXPECT_NEAR(actual, expected, tol) << what << " (expected " << expected << ")";
}

}  // namespace

class LogisticEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = LogisticDistribution::create(0.0, 1.0);
        ASSERT_TRUE(r.isOk());
        sl_ = std::move(r.unwrap());  // standard Logistic(0,1)
    }
    LogisticDistribution sl_;
};

// ─── Known values (mpmath references) ─────────────────────────────────────────

TEST_F(LogisticEnhancedTest, KnownPDFValues) {
    // Logistic(0,1)
    expectClose(sl_.getProbability(0.0), 0.25, 1e-15, "pdf(0)");
    expectClose(sl_.getProbability(-0.5), 0.23500371220159449, 1e-15, "pdf(-0.5)");
    expectClose(sl_.getProbability(0.5), 0.23500371220159449, 1e-15, "pdf(0.5)");
    expectClose(sl_.getProbability(-2.0), 0.10499358540350652, 1e-15, "pdf(-2)");
    expectClose(sl_.getProbability(8.0), 0.00033523767075647422, 1e-14, "pdf(8)");
    expectClose(sl_.getProbability(40.0), 4.2483542552915890e-18, 1e-14, "pdf(40)");

    // Logistic(2,3)
    auto l = LogisticDistribution::create(2.0, 3.0).unwrap();
    expectClose(l.getProbability(2.0), 0.083333333333333333, 1e-15, "pdf(2; 2,3)");
    expectClose(l.getProbability(-8.0), 0.011086241387242533, 1e-14, "pdf(-8; 2,3)");
    expectClose(l.getProbability(0.0), 0.074719129967076200, 1e-14, "pdf(0; 2,3)");
    expectClose(l.getProbability(40.0), 1.0515079676570560e-6, 1e-14, "pdf(40; 2,3)");
}

TEST_F(LogisticEnhancedTest, KnownLogPDFValues) {
    expectClose(sl_.getLogProbability(0.0), -1.3862943611198906, 1e-15, "logpdf(0)");
    expectClose(sl_.getLogProbability(-0.5), -1.4481539683602134, 1e-15, "logpdf(-0.5)");
    expectClose(sl_.getLogProbability(-2.0), -2.2538560220859450, 1e-15, "logpdf(-2)");
    expectClose(sl_.getLogProbability(8.0), -8.0006708127457915, 1e-15, "logpdf(8)");
    // At |z| = 40 the log1p term has underflowed entirely; logpdf == -|z| exactly.
    expectClose(sl_.getLogProbability(40.0), -40.0, 1e-15, "logpdf(40)");

    auto l = LogisticDistribution::create(2.0, 3.0).unwrap();
    expectClose(l.getLogProbability(2.0), -2.4849066497880003, 1e-15, "logpdf(2; 2,3)");
    expectClose(l.getLogProbability(-8.0), -4.5020504541603692, 1e-15, "logpdf(-8; 2,3)");
    expectClose(l.getLogProbability(40.0), -13.765285264412436, 1e-15, "logpdf(40; 2,3)");
}

TEST_F(LogisticEnhancedTest, KnownCDFValuesIncludingBothTails) {
    expectClose(sl_.getCumulativeProbability(0.0), 0.5, 1e-16, "cdf(0)");
    expectClose(sl_.getCumulativeProbability(-2.0), 0.11920292202211756, 1e-15, "cdf(-2)");
    expectClose(sl_.getCumulativeProbability(2.0), 0.88079707797788244, 1e-15, "cdf(2)");
    expectClose(sl_.getCumulativeProbability(-8.0), 0.00033535013046647810, 1e-14, "cdf(-8)");
    expectClose(sl_.getCumulativeProbability(8.0), 0.99966464986953352, 1e-15, "cdf(8)");

    // Deep lower tail — the whole point of the sign-branched form. The naive
    // 1/(1+e^(-z)) would evaluate e^(+100) here and return 0 by overflow.
    expectClose(sl_.getCumulativeProbability(-40.0), 4.2483542552915890e-18, 1e-14, "cdf(-40)");
    expectClose(sl_.getCumulativeProbability(-100.0), 3.7200759760208360e-44, 1e-14, "cdf(-100)");
    // Deep upper tail saturates to exactly 1 (1 - 3.7e-44 is not representable).
    EXPECT_EQ(sl_.getCumulativeProbability(100.0), 1.0);
    // Monotone all the way down, never negative, never > 1.
    double prev = 0.0;
    for (double x = -300.0; x <= 300.0; x += 7.5) {
        const double c = sl_.getCumulativeProbability(x);
        EXPECT_GE(c, prev) << "CDF not monotone at x=" << x;
        EXPECT_GE(c, 0.0);
        EXPECT_LE(c, 1.0);
        prev = c;
    }
}

TEST_F(LogisticEnhancedTest, KnownQuantileValuesIncludingBothTails) {
    expectClose(sl_.getQuantile(0.25), -1.0986122886681097, 1e-15, "q(0.25)");
    EXPECT_EQ(sl_.getQuantile(0.5), 0.0);
    expectClose(sl_.getQuantile(0.75), 1.0986122886681097, 1e-15, "q(0.75)");
    expectClose(sl_.getQuantile(0.01), -4.5951198501345899, 1e-15, "q(0.01)");
    expectClose(sl_.getQuantile(0.99), 4.5951198501345899, 1e-15, "q(0.99)");
    expectClose(sl_.getQuantile(1e-10), -23.025850929840457, 1e-15, "q(1e-10)");
    expectClose(sl_.getQuantile(1e-15), -34.538776394910684, 1e-15, "q(1e-15)");
    // Upper tail: these are the values that a naive log(p/(1-p)) loses entirely.
    expectClose(sl_.getQuantile(1.0 - 1e-10), 23.025850847100089, 1e-14, "q(1-1e-10)");
    expectClose(sl_.getQuantile(1.0 - 1e-15), 34.539575992340881, 1e-13, "q(1-1e-15)");

    auto l = LogisticDistribution::create(2.0, 3.0).unwrap();
    expectClose(l.getQuantile(1e-15), -101.61632918473205, 1e-15, "q(1e-15; 2,3)");
    expectClose(l.getQuantile(1.0 - 1e-15), 105.61872797702264, 1e-13, "q(1-1e-15; 2,3)");
}

TEST_F(LogisticEnhancedTest, QuantileRoundTripToTheTails) {
    for (double p : {1e-15, 1e-12, 1e-8, 0.01, 0.25, 0.5, 0.75, 0.99}) {
        const double q = sl_.getQuantile(p);
        EXPECT_NEAR(sl_.getCumulativeProbability(q), p, 1e-12 * p) << "lower/central p=" << p;
    }
    // Upper tail: compare through the survival function, since 1 - p is where
    // the representable resolution runs out, not the quantile itself.
    for (double p : {1.0 - 1e-8, 1.0 - 1e-12}) {
        const double q = sl_.getQuantile(p);
        const double sf = sl_.getCumulativeProbability(-q);  // symmetry: SF(q) = CDF(-q)
        EXPECT_NEAR(sf, 1.0 - p, 1e-9 * (1.0 - p)) << "upper p=" << p;
    }
}

// ─── Moments and utility ─────────────────────────────────────────────────────

TEST_F(LogisticEnhancedTest, MomentFormulas) {
    EXPECT_NEAR(sl_.getMean(), 0.0, 1e-15);
    expectClose(sl_.getVariance(), 3.2898681336964529, 1e-15, "var(0,1)");  // pi^2/3
    EXPECT_NEAR(sl_.getSkewness(), 0.0, 1e-15);
    EXPECT_NEAR(sl_.getKurtosis(), 1.2, 1e-15);
    EXPECT_NEAR(sl_.getMedian(), 0.0, 1e-15);
    EXPECT_NEAR(sl_.getMode(), 0.0, 1e-15);
    expectClose(sl_.getEntropy(), 2.0, 1e-15, "entropy(0,1)");

    auto l = LogisticDistribution::create(2.0, 3.0).unwrap();
    EXPECT_NEAR(l.getMean(), 2.0, 1e-15);
    expectClose(l.getVariance(), 29.608813203268076, 1e-15, "var(2,3)");
    EXPECT_NEAR(l.getMedian(), 2.0, 1e-15);
    expectClose(l.getEntropy(), 3.0986122886681097, 1e-15, "entropy(2,3)");  // log 3 + 2

    // Median must agree with the 0.5 quantile and with CDF = 1/2.
    EXPECT_EQ(l.getQuantile(0.5), l.getMedian());
    EXPECT_NEAR(l.getCumulativeProbability(l.getMedian()), 0.5, 1e-15);
}

TEST_F(LogisticEnhancedTest, PDFSymmetry) {
    auto l = LogisticDistribution::create(2.5, 1.5).unwrap();
    for (double d : {0.1, 0.5, 1.0, 3.0, 30.0}) {
        EXPECT_NEAR(l.getProbability(2.5 + d), l.getProbability(2.5 - d), 1e-18)
            << "PDF symmetry at d=" << d;
        EXPECT_NEAR(l.getCumulativeProbability(2.5 + d) + l.getCumulativeProbability(2.5 - d), 1.0,
                    1e-15)
            << "CDF symmetry at d=" << d;
    }
}

TEST_F(LogisticEnhancedTest, SetterPropagates) {
    auto l = LogisticDistribution::create(0.0, 1.0).unwrap();
    EXPECT_TRUE(l.isStandard());
    l.setMu(3.0);
    EXPECT_FALSE(l.isStandard());
    EXPECT_NEAR(l.getMean(), 3.0, 1e-15);
    EXPECT_NEAR(l.getMedian(), 3.0, 1e-15);
    EXPECT_NEAR(l.getProbability(3.0), 0.25, 1e-15);
    l.setParameters(0.0, 1.0);
    EXPECT_TRUE(l.isStandard());
}

// ─── Born-compliant contracts: #103 ±inf limits, scalar AND batch ────────────

TEST_F(LogisticEnhancedTest, Contract103_InfiniteLimits_ScalarAndBatch) {
    // Scalar side.
    EXPECT_EQ(sl_.getProbability(kInf), 0.0);
    EXPECT_EQ(sl_.getProbability(-kInf), 0.0);
    EXPECT_TRUE(std::isinf(sl_.getLogProbability(kInf)));
    EXPECT_LT(sl_.getLogProbability(kInf), 0.0) << "logpdf(+inf) must be -inf, not a clamp";
    EXPECT_TRUE(std::isinf(sl_.getLogProbability(-kInf)));
    EXPECT_LT(sl_.getLogProbability(-kInf), 0.0) << "logpdf(-inf) must be -inf, not a clamp";
    EXPECT_EQ(sl_.getCumulativeProbability(-kInf), 0.0);
    EXPECT_EQ(sl_.getCumulativeProbability(kInf), 1.0);

    // Batch side, through every dispatch strategy. Inputs are padded with
    // ordinary values so the SIMD body (not just a remainder lane) sees them.
    std::vector<double> xs;
    for (int i = 0; i < 16; ++i)
        xs.push_back(-3.0 + 0.25 * i);
    xs.insert(xs.begin() + 3, kInf);
    xs.insert(xs.begin() + 9, -kInf);
    xs.push_back(kInf);
    xs.push_back(-kInf);
    const std::size_t n = xs.size();

    for (const auto& hint : allStrategies()) {
        std::vector<double> pdf(n), lpdf(n), cdf(n);
        sl_.getProbability(std::span<const double>(xs), std::span<double>(pdf), hint);
        sl_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf), hint);
        sl_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf), hint);

        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isinf(xs[i]))
                continue;
            const bool pos = xs[i] > 0.0;
            EXPECT_EQ(pdf[i], 0.0) << strategyName(hint) << " pdf at i=" << i;
            EXPECT_TRUE(std::isinf(lpdf[i]) && lpdf[i] < 0.0)
                << strategyName(hint) << " logpdf must be exactly -inf at i=" << i
                << ", got " << lpdf[i];
            EXPECT_EQ(cdf[i], pos ? 1.0 : 0.0) << strategyName(hint) << " cdf at i=" << i;
            // Batch must equal scalar element-for-element.
            EXPECT_EQ(pdf[i], sl_.getProbability(xs[i]));
            EXPECT_EQ(lpdf[i], sl_.getLogProbability(xs[i]));
            EXPECT_EQ(cdf[i], sl_.getCumulativeProbability(xs[i]));
        }
    }
}

// ─── Born-compliant contracts: NaN in → NaN out, scalar AND batch ────────────

TEST_F(LogisticEnhancedTest, ContractNaNPropagation_ScalarAndBatch) {
    EXPECT_TRUE(std::isnan(sl_.getProbability(kNaN)));
    EXPECT_TRUE(std::isnan(sl_.getLogProbability(kNaN)));
    EXPECT_TRUE(std::isnan(sl_.getCumulativeProbability(kNaN)));

    std::vector<double> xs;
    for (int i = 0; i < 16; ++i)
        xs.push_back(-3.0 + 0.25 * i);
    xs.insert(xs.begin() + 2, kNaN);
    xs.insert(xs.begin() + 11, kNaN);
    xs.push_back(kNaN);
    const std::size_t n = xs.size();

    for (const auto& hint : allStrategies()) {
        std::vector<double> pdf(n), lpdf(n), cdf(n);
        sl_.getProbability(std::span<const double>(xs), std::span<double>(pdf), hint);
        sl_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf), hint);
        sl_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf), hint);

        for (std::size_t i = 0; i < n; ++i) {
            const bool want_nan = std::isnan(xs[i]);
            EXPECT_EQ(std::isnan(pdf[i]), want_nan) << strategyName(hint) << " pdf i=" << i;
            EXPECT_EQ(std::isnan(lpdf[i]), want_nan) << strategyName(hint) << " logpdf i=" << i;
            EXPECT_EQ(std::isnan(cdf[i]), want_nan) << strategyName(hint) << " cdf i=" << i;
        }
    }
}

// ─── Born-compliant contracts: #104 quantile ─────────────────────────────────

TEST_F(LogisticEnhancedTest, Contract104_QuantileNeverNaN) {
    auto l = LogisticDistribution::create(-7.5, 0.125).unwrap();
    for (double p : {1e-300, 1e-100, 1e-16, 1e-8, 0.1, 0.5, 0.9,
                     1.0 - 1e-8, 1.0 - 1e-15, std::nextafter(1.0, 0.0)}) {
        for (const auto* d : {&sl_, &l}) {
            const double q = d->getQuantile(p);
            EXPECT_FALSE(std::isnan(q)) << "quantile(" << p << ") is NaN";
            EXPECT_TRUE(std::isfinite(q))
                << "quantile(" << p << ") = " << q << " overflowed without cause";
        }
    }
    // The two closed endpoints are the only ±inf results.
    EXPECT_TRUE(std::isinf(sl_.getQuantile(0.0)) && sl_.getQuantile(0.0) < 0.0);
    EXPECT_TRUE(std::isinf(sl_.getQuantile(1.0)) && sl_.getQuantile(1.0) > 0.0);
    // Out of range and NaN are rejected, not silently answered.
    EXPECT_THROW((void)sl_.getQuantile(-0.1), std::invalid_argument);
    EXPECT_THROW((void)sl_.getQuantile(1.1), std::invalid_argument);
    EXPECT_THROW((void)sl_.getQuantile(kNaN), std::invalid_argument);
}

// ─── Born-compliant contracts: size mismatch throws ──────────────────────────

TEST_F(LogisticEnhancedTest, ContractBatchSizeMismatchThrows) {
    std::vector<double> in(10, 0.5), out(9);
    for (const auto& hint : allStrategies()) {
        EXPECT_THROW(sl_.getProbability(std::span<const double>(in), std::span<double>(out), hint),
                     std::invalid_argument)
            << strategyName(hint);
        EXPECT_THROW(
            sl_.getLogProbability(std::span<const double>(in), std::span<double>(out), hint),
            std::invalid_argument)
            << strategyName(hint);
        EXPECT_THROW(
            sl_.getCumulativeProbability(std::span<const double>(in), std::span<double>(out), hint),
            std::invalid_argument)
            << strategyName(hint);
    }
}

// ─── Batch vs scalar over the full pipeline ──────────────────────────────────

TEST_F(LogisticEnhancedTest, VectorizedMatchesScalar) {
    const size_t N = 1024;
    vector<double> xs(N), out_vec(N), out_scl(N), out_par(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -40.0 + 80.0 * static_cast<double>(i) / static_cast<double>(N - 1);

    detail::PerformanceHint hint_vec, hint_scl, hint_par;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;
    hint_par.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;

    // LogPDF: the vector path uses log(1 + e) where the scalar path uses
    // log1p(e). Since e = e^(-|z|) <= 1, the two differ by at most an ulp of 1
    // in the log term — an absolute deviation below 1e-15, far inside 1e-10.
    sl_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sl_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    sl_.getLogProbability(span<const double>(xs), span<double>(out_par), hint_par);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "LogPDF vec/scl mismatch at i=" << i;
        EXPECT_NEAR(out_par[i], out_scl[i], 1e-12) << "LogPDF par/scl mismatch at i=" << i;
    }

    sl_.getProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sl_.getProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    sl_.getProbability(span<const double>(xs), span<double>(out_par), hint_par);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "PDF vec/scl mismatch at i=" << i;
        EXPECT_NEAR(out_par[i], out_scl[i], 1e-12) << "PDF par/scl mismatch at i=" << i;
    }

    sl_.getCumulativeProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sl_.getCumulativeProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    sl_.getCumulativeProbability(span<const double>(xs), span<double>(out_par), hint_par);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "CDF vec/scl mismatch at i=" << i;
        EXPECT_NEAR(out_par[i], out_scl[i], 1e-12) << "CDF par/scl mismatch at i=" << i;
    }
}

// ─── MLE fit ─────────────────────────────────────────────────────────────────

TEST_F(LogisticEnhancedTest, MLEFit) {
    std::mt19937 rng(42);
    auto source = LogisticDistribution::create(2.0, 0.8).unwrap();
    auto data = source.sample(rng, 4000);

    auto fitted = LogisticDistribution::create().unwrap();
    fitted.fit(data);

    EXPECT_NEAR(fitted.getMu(), 2.0, 0.06) << "Fitted mu should be near 2.0";
    EXPECT_NEAR(fitted.getS(), 0.8, 0.05) << "Fitted s should be near 0.8";

    // The fitted scale must actually solve the score equation it claims to.
    const double s_hat = fitted.getS();
    const double mu_hat = fitted.getMu();
    double score = -static_cast<double>(data.size());
    for (double v : data) {
        const double z = (v - mu_hat) / s_hat;
        const double e = std::exp(-std::fabs(z));
        const double t = (z >= 0.0 ? 1.0 : -1.0) * (1.0 - e) / (1.0 + e);
        score += z * t;
    }
    EXPECT_NEAR(score, 0.0, 1e-6 * static_cast<double>(data.size()))
        << "Newton did not converge to the score root";
}

TEST_F(LogisticEnhancedTest, MLEFitRejectsDegenerateInput) {
    auto l = LogisticDistribution::create().unwrap();
    EXPECT_THROW(l.fit({}), std::invalid_argument);
    EXPECT_THROW(l.fit({3.0}), std::invalid_argument);
    EXPECT_THROW(l.fit({3.0, 3.0, 3.0}), std::invalid_argument);
    EXPECT_THROW(l.fit({1.0, kNaN, 2.0}), std::invalid_argument);
    EXPECT_THROW(l.fit({1.0, kInf}), std::invalid_argument);
}

}  // namespace stats

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
