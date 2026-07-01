#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/tests.h"
#include "libstats/distributions/von_mises.h"

#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>
#include "include/enhanced_test_suite.h"

using namespace std;
using namespace stats;

namespace stats {

class VonMisesEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = stats::VonMisesDistribution::create(0.0, 1.0);
        ASSERT_TRUE(r.isOk());
        vm01_ = std::move(r.value);
    }
    VonMisesDistribution vm01_;  // VM(μ=0, κ=1)
};

// kappa=0 → uniform: PDF = 1/(2π) everywhere
TEST_F(VonMisesEnhancedTest, UniformCase) {
    auto u = VonMisesDistribution::create(0.0, 0.0).value;
    EXPECT_TRUE(u.isUniform());
    const double inv2pi = 1.0 / (2.0 * M_PI);
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        EXPECT_NEAR(u.getProbability(x), inv2pi, 1e-8) << "PDF not uniform at x=" << x;
    }
    EXPECT_NEAR(u.getVariance(), 1.0, 1e-10);
    EXPECT_NEAR(u.getEntropy(), std::log(2.0 * M_PI), 1e-10);
}

// PDF is maximised at mu, minimised at mu+pi
TEST_F(VonMisesEnhancedTest, ModeAtMu) {
    for (double mu : {-1.5, 0.0, 1.0}) {
        auto d = VonMisesDistribution::create(mu, 3.0).value;
        EXPECT_NEAR(d.getMean(), mu, 1e-14) << "getMean() != mu for mu=" << mu;
        EXPECT_GT(d.getProbability(mu),
                  d.getProbability(mu + M_PI > M_PI ? mu + M_PI - 2.0 * M_PI : mu + M_PI))
            << "PDF(mu) should exceed PDF(mu+pi) for mu=" << mu;
    }
}

// Circular variance in [0,1]; increases as kappa decreases
TEST_F(VonMisesEnhancedTest, CircularVarianceMonotone) {
    const auto d0 = VonMisesDistribution::create(0.0, 0.0).value;
    const auto d1 = VonMisesDistribution::create(0.0, 1.0).value;
    const auto d5 = VonMisesDistribution::create(0.0, 5.0).value;
    const auto d20 = VonMisesDistribution::create(0.0, 20.0).value;
    EXPECT_NEAR(d0.getVariance(), 1.0, 1e-10);
    EXPECT_GT(d0.getVariance(), d1.getVariance());
    EXPECT_GT(d1.getVariance(), d5.getVariance());
    EXPECT_GT(d5.getVariance(), d20.getVariance());
    EXPECT_GE(d20.getVariance(), 0.0);
}

// log(PDF) == LogPDF
TEST_F(VonMisesEnhancedTest, LogPDFConsistency) {
    for (double x : {-2.0, -1.0, 0.0, 1.0, 2.0}) {
        const double pdf = vm01_.getProbability(x);
        const double lpdf = vm01_.getLogProbability(x);
        EXPECT_NEAR(std::log(pdf), lpdf, 1e-12) << "at x=" << x;
    }
}

// Mu wrapping: stored value always in (-pi, pi]
TEST_F(VonMisesEnhancedTest, AngleWrapping) {
    for (double mu : {-10.0, -4.0, 4.0, 7.0, 100.0}) {
        auto d = VonMisesDistribution::create(mu, 1.0).value;
        EXPECT_GT(d.getMu(), -M_PI) << "Mu below -pi for input=" << mu;
        EXPECT_LE(d.getMu(), M_PI) << "Mu above +pi for input=" << mu;
    }
}

// Batch matches scalar (LogPDF)
// Tolerance is 1e-10: the batch path routes through vector_cos (AVX-512) whose
// precision vs std::cos is ~6e-11 (see simd_verification VectorCos max_diff).
TEST_F(VonMisesEnhancedTest, BatchMatchesScalarLogPDF) {
    const size_t N = 200;
    vector<double> xs(N), out_b(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -M_PI + 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(N - 1);
    vm01_.getLogProbability(span<const double>(xs), span<double>(out_b));
    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(out_b[i], vm01_.getLogProbability(xs[i]), 1e-10) << "i=" << i;
    }
}

// VECTORIZED vs SCALAR: VECTORIZED routes through vector_cos (AVX-512) while
// SCALAR calls std::cos per element. They agree to within vector_cos precision
// (~6e-11); bit-exact equality is not expected.
TEST_F(VonMisesEnhancedTest, VectorizedEqualsScalar) {
    const size_t N = 500;
    vector<double> xs(N), out_vec(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -M_PI + 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(N);

    detail::PerformanceHint hint_vec, hint_scl;
    hint_vec.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;
    vm01_.getLogProbability(span<const double>(xs), span<double>(out_vec), hint_vec);
    vm01_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);

    for (size_t i = 0; i < N; ++i)
        EXPECT_NEAR(out_vec[i], out_scl[i], 1e-10) << "i=" << i;
}

// Quantile round-trip: CDF(quantile(p)) = p
TEST_F(VonMisesEnhancedTest, QuantileRoundTrip) {
    for (double p : {0.1, 0.25, 0.5, 0.75, 0.9}) {
        const double q = vm01_.getQuantile(p);
        EXPECT_GT(q, -M_PI);
        EXPECT_LE(q, M_PI);
        EXPECT_NEAR(vm01_.getCumulativeProbability(q), p, 1e-4) << "at p=" << p;
    }
}

// MLE recovers true parameters from samples
TEST_F(VonMisesEnhancedTest, MLEFit) {
    mt19937 rng(42);
    auto source = VonMisesDistribution::create(1.2, 3.0).value;
    const auto data = source.sample(rng, 500);
    auto fitted = VonMisesDistribution::create(0.0, 1.0).value;
    fitted.fit(data);
    EXPECT_NEAR(fitted.getMu(), 1.2, 0.3) << "Fitted mu should be near 1.2";
    EXPECT_NEAR(fitted.getKappa(), 3.0, 1.0) << "Fitted kappa should be near 3.0";
}

// Setter propagates to cache
TEST_F(VonMisesEnhancedTest, SetterPropagates) {
    auto d = VonMisesDistribution::create(0.0, 0.0).value;
    EXPECT_TRUE(d.isUniform());
    d.setKappa(2.0);
    EXPECT_FALSE(d.isUniform());
    EXPECT_LT(d.getVariance(), 1.0);
    d.setParameters(0.0, 0.0);
    EXPECT_TRUE(d.isUniform());
}

// Invalid parameters rejected
TEST_F(VonMisesEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(
        VonMisesDistribution::create(std::numeric_limits<double>::infinity(), 1.0).isError());
    EXPECT_TRUE(
        VonMisesDistribution::create(std::numeric_limits<double>::quiet_NaN(), 1.0).isError());
    EXPECT_TRUE(VonMisesDistribution::create(0.0, -1.0).isError());
    EXPECT_FALSE(VonMisesDistribution::create(0.0, 0.0).isError());  // kappa=0 is valid

    auto d = VonMisesDistribution::create(0.0, 1.0).value;
    EXPECT_TRUE(d.trySetKappa(-0.1).isError());
    EXPECT_DOUBLE_EQ(d.getKappa(), 1.0);
}

// Parallel batch LogPDF must produce same results as scalar (labelled timing)
TEST_F(VonMisesEnhancedTest, ParallelBatchCorrectness) {
    const size_t N = 5000;
    vector<double> xs(N), out_par(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = -M_PI + 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(N);

    detail::PerformanceHint hint_par, hint_scl;
    hint_par.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    const auto t0 = std::chrono::high_resolution_clock::now();
    vm01_.getLogProbability(span<const double>(xs), span<double>(out_par), hint_par);
    const auto t1 = std::chrono::high_resolution_clock::now();
    vm01_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double par_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    const double scl_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());

    std::cout << "Von Mises LogPDF PARALLEL vs SCALAR: " << par_us << "μs vs " << scl_us
              << "μs (n=" << N << ")\n";
    std::cout << "Note: PARALLEL routes through the batch dispatch; "
              << "SCALAR forces the scalar loop.\n";

    for (size_t i = 0; i < N; ++i)
        ASSERT_NEAR(out_par[i], out_scl[i], 1e-10) << "parallel mismatch at i=" << i;
}

// CDF accuracy at high kappa: normal approximation path
TEST_F(VonMisesEnhancedTest, HighKappaAccuracy) {
    // For large kappa, VM(mu, kappa) ~ N(mu, 1/kappa).
    // CDF(mu) = 0.5 by symmetry; CDF(mu + z/sqrt(kappa)) ≈ Phi(z).
    constexpr double mu = 0.5;
    for (double kappa : {51.0, 100.0, 500.0}) {
        auto d = VonMisesDistribution::create(mu, kappa).value;
        // P(X <= mu) = 0.5
        EXPECT_NEAR(d.getCumulativeProbability(mu), 0.5, 1e-4)
            << "kappa=" << kappa;
        // P(X <= mu + 1/sqrt(kappa)) ≈ Phi(1) ≈ 0.8413
        const double x1 = mu + 1.0 / std::sqrt(kappa);
        EXPECT_NEAR(d.getCumulativeProbability(x1), 0.8413, 1e-3)
            << "kappa=" << kappa;
        // CDF is monotone on [-pi, pi] relative to mu
        EXPECT_GT(d.getCumulativeProbability(x1), d.getCumulativeProbability(mu))
            << "kappa=" << kappa;
    }
}

}  // namespace stats

//==============================================================================
// DistTraits specialization for stats::VonMisesDistribution
//==============================================================================
template<>
struct stats::tests::DistTraits<stats::VonMisesDistribution> : stats::tests::DistTraitsDefaults {
    static stats::VonMisesDistribution make() { return stats::VonMisesDistribution::create(0.0, 1.0).value; }
    static std::vector<double> domain() { return {-1.5, -0.5, 0.0, 0.5, 1.5}; }
    static double batch_lo() { return -3.14159265358979; }
    static double batch_hi() { return 3.14159265358979; }
    static double pdf_tolerance() { return 1e-10; }
    static double cdf_tolerance() { return 1e-08; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::VonMisesDistribution::create(std::numeric_limits<double>::infinity(), 1.0).isError(); },
            [] { return stats::VonMisesDistribution::create(0.0, -1.0).isError(); },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(VonMises, DistributionEnhancedTest,
                               ::testing::Types<stats::VonMisesDistribution>);
