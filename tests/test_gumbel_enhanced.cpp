#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

// Enhanced tests for GumbelDistribution (MAX-stable / gumbel_r only).
//
// Every "expect" constant below was produced with mpmath at mp.dps = 60 and
// printed to 17 significant digits, using exactly the formulations under test:
//   pdf = exp(-z - e^-z)/beta,  logpdf = -log(beta) - z - e^-z,
//   cdf = exp(-e^-z),           quantile = mu - beta*log(-log p).
// Tail probabilities were evaluated at the *exact double* value of the C++
// literal (e.g. the double nearest 1 - 1e-10, not the decimal 0.9999999999).

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/gumbel.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

// DistTraits specialisation for GumbelDistribution.
// batch_lo is held at -2 because the shared BatchMatchesScalar case compares on
// an *absolute* tolerance: below z ~ -3 the LogPDF is dominated by -e^(-z),
// whose SIMD evaluation carries a ~1 ULP *relative* error, so an absolute
// yardstick stops being meaningful there. The deep tail is covered instead by
// GumbelEnhancedTest.VectorizedMatchesScalar, which uses a relative tolerance.
template <>
struct stats::tests::DistTraits<stats::GumbelDistribution> : stats::tests::DistTraitsDefaults {
    static stats::GumbelDistribution make() {
        return stats::GumbelDistribution::create(0.0, 1.0).unwrap();  // standard Gumbel
    }
    static std::vector<double> domain() { return {-1.0, 0.0, 0.5, 1.0, 3.0}; }
    static double batch_lo() { return -2.0; }
    static double batch_hi() { return 8.0; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::GumbelDistribution::create(0.0, 0.0).isError(); },
            [] { return stats::GumbelDistribution::create(0.0, -1.0).isError(); },
            [] {
                return stats::GumbelDistribution::create(std::numeric_limits<double>::infinity(),
                                                         1.0)
                    .isError();
            },
            [] {
                return stats::GumbelDistribution::create(0.0,
                                                         std::numeric_limits<double>::quiet_NaN())
                    .isError();
            },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Gumbel, DistributionEnhancedTest,
                               ::testing::Types<stats::GumbelDistribution>);

// ─── Per-distribution fixture ────────────────────────────────────────────────

namespace stats {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();

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

void expectClose(double actual, double expected, double rel, const char* what) {
    const double tol = std::max(rel * std::fabs(expected), 1e-300);
    EXPECT_NEAR(actual, expected, tol) << what << " (expected " << expected << ")";
}

}  // namespace

class GumbelEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = GumbelDistribution::create(0.0, 1.0);
        ASSERT_TRUE(r.isOk());
        sg_ = std::move(r.unwrap());  // standard Gumbel(0,1)
    }
    GumbelDistribution sg_;
};

// ─── Known values (mpmath references) ─────────────────────────────────────────

TEST_F(GumbelEnhancedTest, KnownPDFValues) {
    expectClose(sg_.getProbability(0.0), 0.36787944117144232, 1e-15, "pdf(0)");
    expectClose(sg_.getProbability(-1.0), 0.17937407873401718, 1e-15, "pdf(-1)");
    expectClose(sg_.getProbability(0.5), 0.33070429889041807, 1e-15, "pdf(0.5)");
    expectClose(sg_.getProbability(1.0), 0.25464638004358250, 1e-15, "pdf(1)");
    expectClose(sg_.getProbability(3.0), 0.047369009677907915, 1e-15, "pdf(3)");
    expectClose(sg_.getProbability(10.0), 4.5397868655649820e-5, 1e-14, "pdf(10)");
    expectClose(sg_.getProbability(40.0), 4.2483542552915890e-18, 1e-14, "pdf(40)");
    // Lower tail — exp(logpdf) with logpdf = -17.0855...
    expectClose(sg_.getProbability(-3.0), 3.8005425040443577e-8, 1e-13, "pdf(-3)");

    auto g = GumbelDistribution::create(1.5, 2.5).unwrap();
    expectClose(g.getProbability(-3.0), 0.0057077053782269849, 1e-14, "pdf(-3; 1.5,2.5)");
    expectClose(g.getProbability(0.0), 0.11784211814155177, 1e-14, "pdf(0; 1.5,2.5)");
    expectClose(g.getProbability(10.0), 0.012911149974467011, 1e-14, "pdf(10; 1.5,2.5)");
}

TEST_F(GumbelEnhancedTest, KnownLogPDFValues) {
    EXPECT_EQ(sg_.getLogProbability(0.0), -1.0) << "logpdf(0) = -log(1) - 0 - 1 = -1 exactly";
    expectClose(sg_.getLogProbability(-1.0), -1.7182818284590452, 1e-15, "logpdf(-1)");
    expectClose(sg_.getLogProbability(1.0), -1.3678794411714423, 1e-15, "logpdf(1)");
    expectClose(sg_.getLogProbability(3.0), -3.0497870683678639, 1e-15, "logpdf(3)");
    expectClose(sg_.getLogProbability(10.0), -10.000045399929762, 1e-15, "logpdf(10)");
    expectClose(sg_.getLogProbability(40.0), -40.0, 1e-15, "logpdf(40)");
    // Deep lower tail: still an ordinary finite number, dominated by -e^(-z).
    expectClose(sg_.getLogProbability(-3.0), -17.085536923187668, 1e-15, "logpdf(-3)");
    expectClose(sg_.getLogProbability(-10.0), -22016.465794806717, 1e-15, "logpdf(-10)");
    // ...and only becomes -inf once e^(-z) itself overflows (z < -709.78).
    EXPECT_TRUE(std::isinf(sg_.getLogProbability(-800.0)));
    EXPECT_LT(sg_.getLogProbability(-800.0), 0.0);
    EXPECT_FALSE(std::isnan(sg_.getLogProbability(-800.0)))
        << "inf - inf must not leak out of the log-space form";

    auto g = GumbelDistribution::create(1.5, 2.5).unwrap();
    expectClose(g.getLogProbability(-3.0), -5.1659381962871011, 1e-15, "logpdf(-3; 1.5,2.5)");
    expectClose(g.getLogProbability(0.0), -2.1384095322646640, 1e-15, "logpdf(0; 1.5,2.5)");
    expectClose(g.getLogProbability(10.0), -4.3496640018344811, 1e-15, "logpdf(10; 1.5,2.5)");
}

TEST_F(GumbelEnhancedTest, KnownCDFValuesAndDoubleExpEndpoints) {
    expectClose(sg_.getCumulativeProbability(0.0), 0.36787944117144232, 1e-15, "cdf(0)");
    expectClose(sg_.getCumulativeProbability(-1.0), 0.065988035845312537, 1e-15, "cdf(-1)");
    expectClose(sg_.getCumulativeProbability(0.5), 0.54523921189260506, 1e-15, "cdf(0.5)");
    expectClose(sg_.getCumulativeProbability(1.0), 0.69220062755534635, 1e-15, "cdf(1)");
    expectClose(sg_.getCumulativeProbability(3.0), 0.95143199290045341, 1e-15, "cdf(3)");
    expectClose(sg_.getCumulativeProbability(10.0), 0.99995460110079873, 1e-15, "cdf(10)");
    expectClose(sg_.getCumulativeProbability(-3.0), 1.8921786948382926e-9, 1e-14, "cdf(-3)");

    // The double-exponential chain must reach the EXACT endpoints, not near ones.
    EXPECT_EQ(sg_.getCumulativeProbability(-10.0), 0.0) << "true value ~1e-9566";
    EXPECT_EQ(sg_.getCumulativeProbability(-100.0), 0.0);
    EXPECT_EQ(sg_.getCumulativeProbability(-1e300), 0.0);
    EXPECT_EQ(sg_.getCumulativeProbability(40.0), 1.0);
    EXPECT_EQ(sg_.getCumulativeProbability(1e300), 1.0);

    // Monotone, bounded, no NaN anywhere across a wide sweep.
    double prev = 0.0;
    for (double x = -50.0; x <= 50.0; x += 0.625) {
        const double c = sg_.getCumulativeProbability(x);
        EXPECT_FALSE(std::isnan(c)) << "NaN CDF at x=" << x;
        EXPECT_GE(c, prev) << "CDF not monotone at x=" << x;
        EXPECT_GE(c, 0.0);
        EXPECT_LE(c, 1.0);
        prev = c;
    }

    auto g = GumbelDistribution::create(1.5, 2.5).unwrap();
    expectClose(g.getCumulativeProbability(-3.0), 0.0023586933832932267, 1e-14,
                "cdf(-3; 1.5,2.5)");
    expectClose(g.getCumulativeProbability(0.0), 0.16168281414512645, 1e-15, "cdf(0; 1.5,2.5)");
    expectClose(g.getCumulativeProbability(10.0), 0.96717747390469231, 1e-15, "cdf(10; 1.5,2.5)");
}

TEST_F(GumbelEnhancedTest, KnownQuantileValuesIncludingBothTails) {
    expectClose(sg_.getQuantile(0.25), -0.32663425997828098, 1e-15, "q(0.25)");
    expectClose(sg_.getQuantile(0.5), 0.36651292058166433, 1e-15, "q(0.5)");
    expectClose(sg_.getQuantile(0.75), 1.2458993237072382, 1e-15, "q(0.75)");
    expectClose(sg_.getQuantile(0.01), -1.5271796258079011, 1e-15, "q(0.01)");
    expectClose(sg_.getQuantile(0.99), 4.6001492267765800, 1e-15, "q(0.99)");
    // Lower tail: the outer logarithm compresses it, so q diverges only like
    // log(-log p) — q(1e-15) is barely past q(1e-10).
    expectClose(sg_.getQuantile(1e-10), -3.1366175382420015, 1e-15, "q(1e-10)");
    expectClose(sg_.getQuantile(1e-15), -3.5420826463501659, 1e-15, "q(1e-15)");
    // Upper tail: this is where -log p = -log1p(p-1) earns its keep.
    expectClose(sg_.getQuantile(1.0 - 1e-10), 23.025850847150089, 1e-14, "q(1-1e-10)");
    expectClose(sg_.getQuantile(1.0 - 1e-15), 34.539575992340882, 1e-13, "q(1-1e-15)");

    auto g = GumbelDistribution::create(1.5, 2.5).unwrap();
    expectClose(g.getQuantile(1e-15), -7.3552066158754147, 1e-15, "q(1e-15; 1.5,2.5)");
    expectClose(g.getQuantile(0.5), 2.4162823014541608, 1e-15, "q(0.5; 1.5,2.5)");
    expectClose(g.getQuantile(1.0 - 1e-15), 87.848939980852204, 1e-13, "q(1-1e-15; 1.5,2.5)");
}

TEST_F(GumbelEnhancedTest, QuantileRoundTripToTheTails) {
    for (double p : {1e-15, 1e-12, 1e-8, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0 - 1e-8, 1.0 - 1e-12}) {
        const double q = sg_.getQuantile(p);
        EXPECT_NEAR(sg_.getCumulativeProbability(q), p, 1e-13 * std::max(p, 1e-13))
            << "round trip at p=" << p;
    }
}

// ─── Moments and utility ─────────────────────────────────────────────────────

TEST_F(GumbelEnhancedTest, MomentFormulas) {
    expectClose(sg_.getMean(), 0.57721566490153286, 1e-15, "mean(0,1)");    // gamma
    expectClose(sg_.getVariance(), 1.6449340668482264, 1e-15, "var(0,1)");  // pi^2/6
    expectClose(sg_.getSkewness(), 1.1395470994046487, 1e-15, "skewness");  // 12*sqrt6*zeta3/pi^3
    EXPECT_NEAR(sg_.getKurtosis(), 2.4, 1e-15);                             // 12/5
    expectClose(sg_.getMedian(), 0.36651292058166433, 1e-15, "median(0,1)");
    EXPECT_EQ(sg_.getMode(), 0.0);
    expectClose(sg_.getEntropy(), 1.5772156649015329, 1e-15, "entropy(0,1)");  // gamma + 1

    auto g = GumbelDistribution::create(1.5, 2.5).unwrap();
    expectClose(g.getMean(), 2.9430391622538322, 1e-15, "mean(1.5,2.5)");
    expectClose(g.getVariance(), 10.280837917801415, 1e-15, "var(1.5,2.5)");
    expectClose(g.getMedian(), 2.4162823014541608, 1e-15, "median(1.5,2.5)");
    expectClose(g.getEntropy(), 2.4935063967756879, 1e-15, "entropy(1.5,2.5)");
    // Skewness and excess kurtosis are parameter-free for every Gumbel.
    EXPECT_EQ(g.getSkewness(), sg_.getSkewness());
    EXPECT_EQ(g.getKurtosis(), sg_.getKurtosis());

    // Median must be the 0.5 quantile, and CDF(median) = 1/2.
    EXPECT_NEAR(g.getMedian(), g.getQuantile(0.5), 1e-14);
    EXPECT_NEAR(g.getCumulativeProbability(g.getMedian()), 0.5, 1e-15);
    // Mode is where the density peaks.
    EXPECT_GT(sg_.getProbability(0.0), sg_.getProbability(0.05));
    EXPECT_GT(sg_.getProbability(0.0), sg_.getProbability(-0.05));
}

TEST_F(GumbelEnhancedTest, RightSkewedNotSymmetric) {
    // Distinguishes gumbel_r from gumbel_l: the mass sits to the RIGHT of the
    // mode, so the mean exceeds the median which exceeds the mode.
    EXPECT_GT(sg_.getMean(), sg_.getMedian());
    EXPECT_GT(sg_.getMedian(), sg_.getMode());
    EXPECT_GT(sg_.getSkewness(), 1.0);
    // CDF(mode) = 1/e < 1/2 — the max-stable orientation, not the min-stable one
    // (for gumbel_l this would be 1 - 1/e).
    expectClose(sg_.getCumulativeProbability(0.0), 0.36787944117144232, 1e-15, "cdf(mode)");
}

TEST_F(GumbelEnhancedTest, SetterPropagates) {
    auto g = GumbelDistribution::create(0.0, 1.0).unwrap();
    EXPECT_TRUE(g.isStandard());
    g.setMu(3.0);
    EXPECT_FALSE(g.isStandard());
    expectClose(g.getMean(), 3.0 + 0.57721566490153286, 1e-15, "mean after setMu");
    expectClose(g.getProbability(3.0), 0.36787944117144232, 1e-15, "pdf(mode) after setMu");
    g.setParameters(0.0, 1.0);
    EXPECT_TRUE(g.isStandard());
}

// ─── Born-compliant contracts: #103 ±inf limits, scalar AND batch ────────────

TEST_F(GumbelEnhancedTest, Contract103_InfiniteLimits_ScalarAndBatch) {
    EXPECT_EQ(sg_.getProbability(kInf), 0.0);
    EXPECT_EQ(sg_.getProbability(-kInf), 0.0);
    EXPECT_TRUE(std::isinf(sg_.getLogProbability(kInf)));
    EXPECT_LT(sg_.getLogProbability(kInf), 0.0) << "logpdf(+inf) must be -inf, not a clamp";
    EXPECT_TRUE(std::isinf(sg_.getLogProbability(-kInf)));
    EXPECT_LT(sg_.getLogProbability(-kInf), 0.0) << "logpdf(-inf) must be -inf, not a clamp";
    EXPECT_EQ(sg_.getCumulativeProbability(-kInf), 0.0);
    EXPECT_EQ(sg_.getCumulativeProbability(kInf), 1.0);

    std::vector<double> xs;
    for (int i = 0; i < 16; ++i)
        xs.push_back(-2.0 + 0.5 * i);
    xs.insert(xs.begin() + 3, kInf);
    xs.insert(xs.begin() + 9, -kInf);
    xs.push_back(kInf);
    xs.push_back(-kInf);
    const std::size_t n = xs.size();

    for (const auto& hint : allStrategies()) {
        std::vector<double> pdf(n), lpdf(n), cdf(n);
        sg_.getProbability(std::span<const double>(xs), std::span<double>(pdf), hint);
        sg_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf), hint);
        sg_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf), hint);

        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isinf(xs[i]))
                continue;
            const bool pos = xs[i] > 0.0;
            EXPECT_EQ(pdf[i], 0.0) << strategyName(hint) << " pdf at i=" << i;
            EXPECT_TRUE(std::isinf(lpdf[i]) && lpdf[i] < 0.0)
                << strategyName(hint) << " logpdf must be exactly -inf at i=" << i << ", got "
                << lpdf[i];
            EXPECT_EQ(cdf[i], pos ? 1.0 : 0.0) << strategyName(hint) << " cdf at i=" << i;
            EXPECT_EQ(pdf[i], sg_.getProbability(xs[i]));
            EXPECT_EQ(lpdf[i], sg_.getLogProbability(xs[i]));
            EXPECT_EQ(cdf[i], sg_.getCumulativeProbability(xs[i]));
        }
    }
}

// The double-exponential chain must also saturate exactly in the BATCH path —
// this is the empirical check that vector_exp's ±inf/overflow handling behaves
// like std::exp on this machine's SIMD tier, rather than assuming it.
TEST_F(GumbelEnhancedTest, Contract103_BatchDoubleExpSaturatesExactly) {
    std::vector<double> xs;
    for (int i = 0; i < 24; ++i)
        xs.push_back(-40.0 + 4.0 * i);  // -40 .. 52, straddling both saturations
    xs.push_back(-1e300);
    xs.push_back(1e300);
    const std::size_t n = xs.size();

    for (const auto& hint : allStrategies()) {
        std::vector<double> cdf(n), pdf(n), lpdf(n);
        sg_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf), hint);
        sg_.getProbability(std::span<const double>(xs), std::span<double>(pdf), hint);
        sg_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf), hint);

        for (std::size_t i = 0; i < n; ++i) {
            const double x = xs[i];
            EXPECT_FALSE(std::isnan(cdf[i])) << strategyName(hint) << " NaN CDF at x=" << x;
            EXPECT_FALSE(std::isnan(pdf[i])) << strategyName(hint) << " NaN PDF at x=" << x;
            EXPECT_FALSE(std::isnan(lpdf[i])) << strategyName(hint) << " NaN LogPDF at x=" << x;
            EXPECT_GE(cdf[i], 0.0);
            EXPECT_LE(cdf[i], 1.0);
            EXPECT_GE(pdf[i], 0.0);
            if (x <= -20.0) {
                EXPECT_EQ(cdf[i], 0.0) << strategyName(hint) << " CDF must be exactly 0 at x=" << x;
                EXPECT_EQ(pdf[i], 0.0) << strategyName(hint) << " PDF must be exactly 0 at x=" << x;
            }
            if (x >= 40.0) {
                EXPECT_EQ(cdf[i], 1.0) << strategyName(hint) << " CDF must be exactly 1 at x=" << x;
            }
            // Batch must equal scalar exactly at the saturated points; in
            // between, the SIMD and libm exponentials may differ by an ulp.
            const double ref = sg_.getCumulativeProbability(x);
            if (x <= -20.0 || x >= 40.0) {
                EXPECT_EQ(cdf[i], ref) << strategyName(hint) << " x=" << x;
            } else {
                EXPECT_NEAR(cdf[i], ref, std::max(1e-12 * std::fabs(ref), 1e-300))
                    << strategyName(hint) << " x=" << x;
            }
        }
    }
}

// ─── Born-compliant contracts: NaN in → NaN out, scalar AND batch ────────────

TEST_F(GumbelEnhancedTest, ContractNaNPropagation_ScalarAndBatch) {
    EXPECT_TRUE(std::isnan(sg_.getProbability(kNaN)));
    EXPECT_TRUE(std::isnan(sg_.getLogProbability(kNaN)));
    EXPECT_TRUE(std::isnan(sg_.getCumulativeProbability(kNaN)));

    std::vector<double> xs;
    for (int i = 0; i < 16; ++i)
        xs.push_back(-2.0 + 0.5 * i);
    xs.insert(xs.begin() + 2, kNaN);
    xs.insert(xs.begin() + 11, kNaN);
    xs.push_back(kNaN);
    const std::size_t n = xs.size();

    for (const auto& hint : allStrategies()) {
        std::vector<double> pdf(n), lpdf(n), cdf(n);
        sg_.getProbability(std::span<const double>(xs), std::span<double>(pdf), hint);
        sg_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf), hint);
        sg_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf), hint);

        for (std::size_t i = 0; i < n; ++i) {
            const bool want_nan = std::isnan(xs[i]);
            EXPECT_EQ(std::isnan(pdf[i]), want_nan) << strategyName(hint) << " pdf i=" << i;
            EXPECT_EQ(std::isnan(lpdf[i]), want_nan) << strategyName(hint) << " logpdf i=" << i;
            EXPECT_EQ(std::isnan(cdf[i]), want_nan) << strategyName(hint) << " cdf i=" << i;
        }
    }
}

// ─── Born-compliant contracts: #104 quantile ─────────────────────────────────

TEST_F(GumbelEnhancedTest, Contract104_QuantileNeverNaN) {
    auto g = GumbelDistribution::create(-7.5, 0.125).unwrap();
    for (double p : {1e-300, 1e-100, 1e-16, 1e-8, 0.1, 0.5, 0.9, 1.0 - 1e-8, 1.0 - 1e-15,
                     std::nextafter(1.0, 0.0)}) {
        for (const auto* d : {&sg_, &g}) {
            const double q = d->getQuantile(p);
            EXPECT_FALSE(std::isnan(q)) << "quantile(" << p << ") is NaN";
            EXPECT_TRUE(std::isfinite(q))
                << "quantile(" << p << ") = " << q << " overflowed without cause";
        }
    }
    EXPECT_TRUE(std::isinf(sg_.getQuantile(0.0)) && sg_.getQuantile(0.0) < 0.0);
    EXPECT_TRUE(std::isinf(sg_.getQuantile(1.0)) && sg_.getQuantile(1.0) > 0.0);
    EXPECT_THROW((void)sg_.getQuantile(-0.1), std::invalid_argument);
    EXPECT_THROW((void)sg_.getQuantile(1.1), std::invalid_argument);
    EXPECT_THROW((void)sg_.getQuantile(kNaN), std::invalid_argument);
    // Strictly increasing across the whole open interval.
    double prev = -kInf;
    for (double p = 1e-6; p < 1.0; p += 0.0125) {
        const double q = sg_.getQuantile(p);
        EXPECT_GT(q, prev) << "quantile not strictly increasing at p=" << p;
        prev = q;
    }
}

// ─── Born-compliant contracts: size mismatch throws ──────────────────────────

TEST_F(GumbelEnhancedTest, ContractBatchSizeMismatchThrows) {
    std::vector<double> in(10, 0.5), out(9);
    for (const auto& hint : allStrategies()) {
        EXPECT_THROW(sg_.getProbability(std::span<const double>(in), std::span<double>(out), hint),
                     std::invalid_argument)
            << strategyName(hint);
        EXPECT_THROW(
            sg_.getLogProbability(std::span<const double>(in), std::span<double>(out), hint),
            std::invalid_argument)
            << strategyName(hint);
        EXPECT_THROW(
            sg_.getCumulativeProbability(std::span<const double>(in), std::span<double>(out), hint),
            std::invalid_argument)
            << strategyName(hint);
    }
}

// ─── Batch vs scalar over the full pipeline ──────────────────────────────────

TEST_F(GumbelEnhancedTest, VectorizedMatchesScalar) {
    const size_t N = 1024;
    vector<double> xs(N), out_vec(N), out_scl(N), out_par(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -8.0 + 28.0 * static_cast<double>(i) / static_cast<double>(N - 1);

    detail::PerformanceHint hint_vec, hint_scl, hint_par;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;
    hint_par.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;

    // LogPDF is -log(beta) - z - e^(-z). Below z ~ -3 the e^(-z) term dominates
    // and is large (e^8 ~ 2981 at the left edge here), so SIMD-vs-scalar must be
    // compared RELATIVELY: a ~1 ULP difference in vector_exp is a ~1e-13
    // absolute difference in the result at that magnitude, which is correct
    // behaviour rather than a defect.
    const auto mixedTol = [](double ref) { return std::max(1e-10, 1e-12 * std::fabs(ref)); };

    sg_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sg_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    sg_.getLogProbability(span<const double>(xs), span<double>(out_par), hint_par);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_vec[i], out_scl[i], mixedTol(out_scl[i]))
            << "LogPDF vec/scl mismatch at i=" << i << " x=" << xs[i];
        EXPECT_NEAR(out_par[i], out_scl[i], 1e-12 * std::max(1.0, std::fabs(out_scl[i])))
            << "LogPDF par/scl mismatch at i=" << i;
    }

    // PDF is exp(LogPDF), so the absolute deviation is bounded by the PDF value
    // itself times the LogPDF deviation — comfortably inside 1e-10 everywhere.
    sg_.getProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sg_.getProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    sg_.getProbability(span<const double>(xs), span<double>(out_par), hint_par);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "PDF vec/scl mismatch at i=" << i;
        EXPECT_NEAR(out_par[i], out_scl[i], 1e-12) << "PDF par/scl mismatch at i=" << i;
    }

    sg_.getCumulativeProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sg_.getCumulativeProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    sg_.getCumulativeProbability(span<const double>(xs), span<double>(out_par), hint_par);
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "CDF vec/scl mismatch at i=" << i;
        EXPECT_NEAR(out_par[i], out_scl[i], 1e-12) << "CDF par/scl mismatch at i=" << i;
    }
}

// ─── Fit (method of moments) ─────────────────────────────────────────────────

TEST_F(GumbelEnhancedTest, MomentFit) {
    std::mt19937 rng(42);
    auto source = GumbelDistribution::create(2.0, 0.8).unwrap();
    auto data = source.sample(rng, 4000);

    auto fitted = GumbelDistribution::create().unwrap();
    fitted.fit(data);

    EXPECT_NEAR(fitted.getMu(), 2.0, 0.06) << "Fitted mu should be near 2.0";
    EXPECT_NEAR(fitted.getBeta(), 0.8, 0.05) << "Fitted beta should be near 0.8";

    // The estimator is defined by the moment identities — check it reproduces
    // them exactly on the sample it was given (this is what makes it MoM and
    // not MLE).
    const std::size_t n = data.size();
    double mean = 0.0;
    for (double v : data)
        mean += v;
    mean /= static_cast<double>(n);
    double ss = 0.0;
    for (double v : data)
        ss += (v - mean) * (v - mean);
    const double sd = std::sqrt(ss / static_cast<double>(n - 1));

    EXPECT_NEAR(fitted.getBeta(), sd * std::sqrt(6.0) / 3.14159265358979323846, 1e-12);
    EXPECT_NEAR(fitted.getMu(), mean - 0.5772156649015329 * fitted.getBeta(), 1e-12);
}

TEST_F(GumbelEnhancedTest, MomentFitRejectsDegenerateInput) {
    auto g = GumbelDistribution::create().unwrap();
    EXPECT_THROW(g.fit({}), std::invalid_argument);
    EXPECT_THROW(g.fit({3.0}), std::invalid_argument);
    EXPECT_THROW(g.fit({3.0, 3.0, 3.0}), std::invalid_argument);
    EXPECT_THROW(g.fit({1.0, kNaN, 2.0}), std::invalid_argument);
    EXPECT_THROW(g.fit({1.0, kInf}), std::invalid_argument);
}

}  // namespace stats

#ifdef _MSC_VER
    #pragma warning(pop)
#endif
