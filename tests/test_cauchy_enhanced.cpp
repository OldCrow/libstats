#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/cauchy.h"

#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

// DistTraits specialisation for CauchyDistribution
template <>
struct stats::tests::DistTraits<stats::CauchyDistribution> : stats::tests::DistTraitsDefaults {
    static stats::CauchyDistribution make() {
        return stats::CauchyDistribution::create(0.0, 1.0).unwrap();  // standard Cauchy
    }
    static std::vector<double> domain() { return {-5.0, -1.0, 0.0, 1.0, 5.0}; }
    static double batch_lo() { return -10.0; }
    static double batch_hi() { return 10.0; }
    // pdf_tolerance: inherited 1e-10 is fine; StudentT delegation is numerically exact
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::CauchyDistribution::create(0.0, 0.0).isError(); },
            [] { return stats::CauchyDistribution::create(0.0, -1.0).isError(); },
            [] {
                return stats::CauchyDistribution::create(std::numeric_limits<double>::infinity(),
                                                         1.0)
                    .isError();
            },
            [] {
                return stats::CauchyDistribution::create(0.0,
                                                         std::numeric_limits<double>::quiet_NaN())
                    .isError();
            },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Cauchy, DistributionEnhancedTest,
                               ::testing::Types<stats::CauchyDistribution>);

// ─── Per-distribution fixture ────────────────────────────────────────────────

namespace stats {

class CauchyEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = CauchyDistribution::create(0.0, 1.0);
        ASSERT_TRUE(r.isOk());
        sc_ = std::move(r).unwrap();  // standard Cauchy(0,1)
    }
    CauchyDistribution sc_;
};

// ─── NaN moments ─────────────────────────────────────────────────────────────

TEST_F(CauchyEnhancedTest, MomentsAreNaN) {
    // All four conventional moments are undefined for the Cauchy distribution.
    EXPECT_TRUE(std::isnan(sc_.getMean())) << "getMean() should return NaN";
    EXPECT_TRUE(std::isnan(sc_.getVariance())) << "getVariance() should return NaN";
    EXPECT_TRUE(std::isnan(sc_.getSkewness())) << "getSkewness() should return NaN";
    EXPECT_TRUE(std::isnan(sc_.getKurtosis())) << "getKurtosis() should return NaN";
}

// ─── Known PDF values ─────────────────────────────────────────────────────────

TEST_F(CauchyEnhancedTest, PDFAtLocation) {
    // PDF(x0; x0, gamma) = 1/(pi*gamma) — maximum at the location parameter.
    EXPECT_NEAR(sc_.getProbability(0.0), 1.0 / detail::PI, 1e-12)
        << "PDF(x0=0; x0=0, gamma=1) = 1/pi";

    auto c = CauchyDistribution::create(3.0, 2.0).unwrap();
    EXPECT_NEAR(c.getProbability(3.0), 1.0 / (detail::PI * 2.0), 1e-12)
        << "PDF(x0=3; x0=3, gamma=2) = 1/(2*pi)";
}

TEST_F(CauchyEnhancedTest, PDFFormula) {
    // PDF(x; 0, 1) = 1/(pi*(1+x²))
    for (double x : {-3.0, -1.0, 0.0, 1.0, 3.0}) {
        double expected = 1.0 / (detail::PI * (1.0 + x * x));
        EXPECT_NEAR(sc_.getProbability(x), expected, 1e-12) << "PDF formula at x=" << x;
    }
}

TEST_F(CauchyEnhancedTest, LogPDFFormula) {
    // LogPDF(x; 0, 1) = -log(pi) - log(1+x²)
    for (double x : {-2.0, -0.5, 0.0, 0.5, 2.0}) {
        double expected = -std::log(detail::PI) - std::log(1.0 + x * x);
        EXPECT_NEAR(sc_.getLogProbability(x), expected, 1e-12) << "LogPDF formula at x=" << x;
    }
}

// ─── CDF ─────────────────────────────────────────────────────────────────────

TEST_F(CauchyEnhancedTest, CDFAtLocation) {
    // CDF(x0) = 0.5 for any Cauchy distribution (symmetry).
    EXPECT_NEAR(sc_.getCumulativeProbability(0.0), 0.5, 1e-12) << "CDF(x0=0) = 0.5";

    auto c = CauchyDistribution::create(-2.0, 3.0).unwrap();
    EXPECT_NEAR(c.getCumulativeProbability(-2.0), 0.5, 1e-12) << "CDF(x0=-2) = 0.5";
}

TEST_F(CauchyEnhancedTest, CDFFormula) {
    // CDF(x; 0, 1) = 0.5 + atan(x)/pi
    // Tolerance matches StudentT's CDFSymmetry / BatchMatchesScalar (1e-8): Cauchy
    // delegates to StudentT(nu=1)'s incomplete-beta CDF, which achieves ~1e-9 absolute
    // accuracy vs the analytical atan formula.  Using atan directly would require
    // abandoning the delegation pattern for this one scalar path.
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        double expected = 0.5 + std::atan(x) / detail::PI;
        EXPECT_NEAR(sc_.getCumulativeProbability(x), expected, 1e-8) << "CDF formula at x=" << x;
    }
    // Symmetry: CDF(x0+d) + CDF(x0-d) = 1
    for (double d : {0.5, 1.0, 2.0, 5.0}) {
        EXPECT_NEAR(sc_.getCumulativeProbability(d) + sc_.getCumulativeProbability(-d), 1.0, 1e-8)
            << "Symmetry at d=" << d;
    }
}

// ─── Quantile ─────────────────────────────────────────────────────────────────

TEST_F(CauchyEnhancedTest, QuantileRoundTrip) {
    for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
        const double q = sc_.getQuantile(p);
        EXPECT_NEAR(sc_.getCumulativeProbability(q), p, 1e-8) << "Quantile round-trip at p=" << p;
    }
}

TEST_F(CauchyEnhancedTest, QuantileFormula) {
    // Q(0.5; 0, 1) = tan(pi*0) = 0
    EXPECT_NEAR(sc_.getQuantile(0.5), 0.0, 1e-12);
    // Q(0.75; 0, 1) = tan(pi*0.25) = 1
    EXPECT_NEAR(sc_.getQuantile(0.75), 1.0, 1e-12);
    // Q(0.25; 0, 1) = tan(-pi*0.25) = -1
    EXPECT_NEAR(sc_.getQuantile(0.25), -1.0, 1e-12);
}

// ─── Symmetry ─────────────────────────────────────────────────────────────────

TEST_F(CauchyEnhancedTest, PDFSymmetry) {
    // Cauchy is symmetric about x0: f(x0+d) == f(x0-d)
    auto c = CauchyDistribution::create(2.5, 1.5).unwrap();
    for (double d : {0.1, 0.5, 1.0, 3.0, 10.0}) {
        EXPECT_NEAR(c.getProbability(2.5 + d), c.getProbability(2.5 - d), 1e-12)
            << "symmetry at d=" << d;
    }
}

// ─── Median, Mode, Entropy ────────────────────────────────────────────────────

TEST_F(CauchyEnhancedTest, MedianAndMode) {
    EXPECT_NEAR(sc_.getMedian(), 0.0, 1e-12);
    EXPECT_NEAR(sc_.getMode(), 0.0, 1e-12);

    auto c = CauchyDistribution::create(4.0, 2.0).unwrap();
    EXPECT_NEAR(c.getMedian(), 4.0, 1e-12);
    EXPECT_NEAR(c.getMode(), 4.0, 1e-12);
}

TEST_F(CauchyEnhancedTest, Entropy) {
    // H(Cauchy(0,1)) = log(4*pi)
    EXPECT_NEAR(sc_.getEntropy(), std::log(detail::FOUR_PI), 1e-12);

    // H(Cauchy(x0, gamma)) = log(4*pi*gamma) — independent of x0
    auto c1 = CauchyDistribution::create(0.0, 2.0).unwrap();
    auto c2 = CauchyDistribution::create(5.0, 2.0).unwrap();
    EXPECT_NEAR(c1.getEntropy(), std::log(detail::FOUR_PI * 2.0), 1e-12);
    EXPECT_NEAR(c1.getEntropy(), c2.getEntropy(), 1e-12) << "entropy depends only on gamma";
}

// ─── Setter propagates ────────────────────────────────────────────────────────

TEST_F(CauchyEnhancedTest, SetterPropagates) {
    auto c = CauchyDistribution::create(0.0, 1.0).unwrap();
    EXPECT_TRUE(c.isStandard());
    c.setX0(3.0);
    EXPECT_FALSE(c.isStandard());
    EXPECT_NEAR(c.getMedian(), 3.0, 1e-12);
    EXPECT_NEAR(c.getProbability(3.0), 1.0 / detail::PI, 1e-12);

    c.setParameters(0.0, 1.0);
    EXPECT_TRUE(c.isStandard());
}

// ─── MLE fit ─────────────────────────────────────────────────────────────────

TEST_F(CauchyEnhancedTest, MLEFitSingleObservation) {
    auto c = CauchyDistribution::create().unwrap();
    // Single observation: x0_hat should equal the observation; gamma falls to default.
    EXPECT_NO_THROW(c.fit({5.0}));
    EXPECT_NEAR(c.getX0(), 5.0, 1e-6);
    // gamma should be positive and finite.
    EXPECT_GT(c.getGamma(), 0.0);
    EXPECT_TRUE(std::isfinite(c.getGamma()));
}

TEST_F(CauchyEnhancedTest, MLEFit) {
    std::mt19937 rng(42);
    auto source = CauchyDistribution::create(2.0, 1.5).unwrap();
    auto data = source.sample(rng, 2000);

    auto fitted = CauchyDistribution::create().unwrap();
    fitted.fit(data);

    // Cauchy MLE converges reliably but with wider confidence intervals than
    // Gaussian MLE due to the heavy tails. Use generous tolerances.
    EXPECT_NEAR(fitted.getX0(), 2.0, 0.3) << "Fitted x0 should be near 2.0";
    EXPECT_NEAR(fitted.getGamma(), 1.5, 0.4) << "Fitted gamma should be near 1.5";
}

// ─── Batch vs scalar ─────────────────────────────────────────────────────────

TEST_F(CauchyEnhancedTest, VectorizedMatchesScalar) {
    const size_t N = 1024;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -10.0 + 20.0 * static_cast<double>(i) / static_cast<double>(N - 1);

    detail::PerformanceHint hint_vec, hint_scl;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    sc_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sc_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "LogPDF mismatch at i=" << i;

    sc_.getProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sc_.getProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "PDF mismatch at i=" << i;

    sc_.getCumulativeProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    sc_.getCumulativeProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "CDF mismatch at i=" << i;
}

// ─── VectorizedSpeedup (labelled timing) ─────────────────────────────────────

TEST_F(CauchyEnhancedTest, VectorizedSpeedup) {
    const size_t N = 50000;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -10.0 + 20.0 * static_cast<double>(i + 1) / static_cast<double>(N);

    detail::PerformanceHint hint_vec, hint_scl;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    const auto t0 = std::chrono::high_resolution_clock::now();
    sc_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    const auto t1 = std::chrono::high_resolution_clock::now();
    sc_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double vec_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    const double scl_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    const double speedup = scl_us / std::max(vec_us, 1.0);
    std::cout << "Cauchy LogPDF VECTORIZED speedup: " << speedup << "x "
              << "(VECTORIZED " << vec_us << "µs, SCALAR " << scl_us << "µs)\n";

    for (size_t i = 0; i < N; ++i)
        ASSERT_NEAR(out_vec[i], out_scl[i], 1e-10) << "mismatch at i=" << i;
    EXPECT_GT(speedup, 1.2) << "VECTORIZED should be faster than SCALAR for n=" << N;
}

}  // namespace stats
