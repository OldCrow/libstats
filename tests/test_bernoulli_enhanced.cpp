#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/bernoulli.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

// DistTraits specialisation for BernoulliDistribution
template <>
struct stats::tests::DistTraits<stats::BernoulliDistribution> : stats::tests::DistTraitsDefaults {
    static stats::BernoulliDistribution make() {
        return stats::BernoulliDistribution::create(0.5).unwrap();  // mean = 0.5, var = 0.25
    }
    static std::vector<double> domain() {
        return {0.0, 1.0};  // only two support points
    }
    static double batch_lo() { return 0.0; }
    static double batch_hi() { return 1.0; }
    static constexpr bool is_discrete = true;  // disables QuantileRoundTrip
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::BernoulliDistribution::create(-0.5).isError(); },
            [] { return stats::BernoulliDistribution::create(1.1).isError(); },
            [] {
                return stats::BernoulliDistribution::create(
                           std::numeric_limits<double>::quiet_NaN())
                    .isError();
            },
            [] {
                return stats::BernoulliDistribution::create(
                           std::numeric_limits<double>::infinity())
                    .isError();
            },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(Bernoulli, DistributionEnhancedTest,
                               ::testing::Types<stats::BernoulliDistribution>);

// ─── Per-distribution fixture ───────────────────────────────────────────────

namespace stats {

class BernoulliEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto r = BernoulliDistribution::create(0.5);
        ASSERT_TRUE(r.isOk());
        b05_ = std::move(r.unwrap());  // Bernoulli(0.5): mean=0.5, var=0.25
    }
    BernoulliDistribution b05_;
};

// ─── Known PMF values ────────────────────────────────────────────────────────

TEST_F(BernoulliEnhancedTest, KnownPMFValues) {
    EXPECT_NEAR(b05_.getProbability(0.0), 0.5, 1e-12);
    EXPECT_NEAR(b05_.getProbability(1.0), 0.5, 1e-12);
    // Out-of-support
    EXPECT_EQ(b05_.getProbability(2.0), 0.0);
    EXPECT_EQ(b05_.getProbability(-1.0), 0.0);
    // Non-integer x rounds to nearest integer (matches Binomial's own
    // documented convention); 1.5 rounds to 2, which is out of {0,1}.
    EXPECT_EQ(b05_.getProbability(1.5), 0.0) << "rounds to 2, out of support";

    auto b02 = BernoulliDistribution::create(0.2).unwrap();
    EXPECT_NEAR(b02.getProbability(1.0), 0.2, 1e-12);
    EXPECT_NEAR(b02.getProbability(0.0), 0.8, 1e-12);
}

TEST_F(BernoulliEnhancedTest, KnownLogPMFValues) {
    EXPECT_NEAR(b05_.getLogProbability(0.0), std::log(0.5), 1e-12);
    EXPECT_NEAR(b05_.getLogProbability(1.0), std::log(0.5), 1e-12);
    EXPECT_EQ(b05_.getLogProbability(-1.0), -std::numeric_limits<double>::infinity());
}

// ─── Known CDF values ────────────────────────────────────────────────────────

TEST_F(BernoulliEnhancedTest, KnownCDFValues) {
    EXPECT_NEAR(b05_.getCumulativeProbability(0.0), 0.5, 1e-12);
    EXPECT_NEAR(b05_.getCumulativeProbability(1.0), 1.0, 1e-12);
    EXPECT_EQ(b05_.getCumulativeProbability(-1.0), 0.0);
    EXPECT_NEAR(b05_.getCumulativeProbability(0.5), 0.5, 1e-12) << "floor property: F(0.5)=F(0)";
    EXPECT_NEAR(b05_.getCumulativeProbability(5.0), 1.0, 1e-12);
}

// ─── Moment formulas ─────────────────────────────────────────────────────────

TEST_F(BernoulliEnhancedTest, MomentFormulas) {
    EXPECT_NEAR(b05_.getMean(), 0.5, 1e-12);
    EXPECT_NEAR(b05_.getVariance(), 0.25, 1e-12);
    EXPECT_NEAR(b05_.getSkewness(), 0.0, 1e-12) << "symmetric at p=0.5";

    auto b02 = BernoulliDistribution::create(0.2).unwrap();
    EXPECT_NEAR(b02.getMean(), 0.2, 1e-12);
    EXPECT_NEAR(b02.getVariance(), 0.16, 1e-12);
    EXPECT_NEAR(b02.getSkewness(), (1.0 - 2.0 * 0.2) / std::sqrt(0.2 * 0.8), 1e-10);
}

// ─── Mode and Median ─────────────────────────────────────────────────────────

TEST_F(BernoulliEnhancedTest, ModeAndMedian) {
    EXPECT_EQ(b05_.getMode(), 0.0) << "tie at p=0.5: 0 by convention";
    EXPECT_EQ(b05_.getMedian(), 0.5) << "tie at p=0.5: 0.5 by convention";

    auto b02 = BernoulliDistribution::create(0.2).unwrap();
    EXPECT_EQ(b02.getMode(), 0.0);
    EXPECT_EQ(b02.getMedian(), 0.0);

    auto b08 = BernoulliDistribution::create(0.8).unwrap();
    EXPECT_EQ(b08.getMode(), 1.0);
    EXPECT_EQ(b08.getMedian(), 1.0);
}

// ─── Entropy ─────────────────────────────────────────────────────────────────

TEST_F(BernoulliEnhancedTest, EntropyFormula) {
    const double expected = -0.5 * std::log(0.5) - 0.5 * std::log(0.5);
    EXPECT_NEAR(b05_.getEntropy(), expected, 1e-12);

    // Degenerate distributions have zero entropy
    auto b0 = BernoulliDistribution::create(0.0).unwrap();
    EXPECT_NEAR(b0.getEntropy(), 0.0, 1e-9);
    auto b1 = BernoulliDistribution::create(1.0).unwrap();
    EXPECT_NEAR(b1.getEntropy(), 0.0, 1e-9);
}

// ─── #104: right-continuous discrete quantile, never NaN on (0,1) ───────────

TEST_F(BernoulliEnhancedTest, QuantileContract) {
    // Q(q) = 0 for q <= 1-p, else 1; never NaN for q in (0,1)
    auto b03 = BernoulliDistribution::create(0.3).unwrap();  // 1-p = 0.7
    for (double q : {0.01, 0.1, 0.5, 0.69, 0.7, 0.71, 0.9, 0.99}) {
        const double qv = b03.getQuantile(q);
        EXPECT_FALSE(std::isnan(qv)) << "quantile must never be NaN for q=" << q;
        if (q <= 0.7)
            EXPECT_EQ(qv, 0.0) << "q=" << q << " <= 1-p=0.7";
        else
            EXPECT_EQ(qv, 1.0) << "q=" << q << " > 1-p=0.7";
    }
    EXPECT_EQ(b03.getQuantile(0.0), 0.0);
    EXPECT_EQ(b03.getQuantile(1.0), 1.0);
}

// ─── #103: PDF/LogPDF/CDF at +-inf and NaN, scalar AND batch ────────────────

TEST_F(BernoulliEnhancedTest, InfAndNaNContractScalar) {
    const double inf = std::numeric_limits<double>::infinity();
    const double ninf = -std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_EQ(b05_.getProbability(inf), 0.0);
    EXPECT_EQ(b05_.getProbability(ninf), 0.0);
    EXPECT_TRUE(std::isnan(b05_.getProbability(nan)));

    EXPECT_EQ(b05_.getLogProbability(inf), ninf);
    EXPECT_EQ(b05_.getLogProbability(ninf), ninf);
    EXPECT_TRUE(std::isnan(b05_.getLogProbability(nan)));

    EXPECT_EQ(b05_.getCumulativeProbability(ninf), 0.0);
    EXPECT_EQ(b05_.getCumulativeProbability(inf), 1.0);
    EXPECT_TRUE(std::isnan(b05_.getCumulativeProbability(nan)));
}

TEST_F(BernoulliEnhancedTest, InfAndNaNContractBatch) {
    const double inf = std::numeric_limits<double>::infinity();
    const double ninf = -std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> xs = {0.0, 1.0, inf, ninf, nan};
    const size_t N = xs.size();
    std::vector<double> pdf_b(N), lpdf_b(N), cdf_b(N);

    b05_.getProbability(std::span<const double>(xs), std::span<double>(pdf_b));
    b05_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf_b));
    b05_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(std::isnan(pdf_b[i]), std::isnan(b05_.getProbability(xs[i])))
            << "PDF batch/scalar NaN mismatch at i=" << i;
        if (!std::isnan(pdf_b[i])) {
            EXPECT_EQ(pdf_b[i], b05_.getProbability(xs[i])) << "PDF batch/scalar mismatch i=" << i;
        }

        EXPECT_EQ(std::isnan(lpdf_b[i]), std::isnan(b05_.getLogProbability(xs[i])))
            << "LogPDF batch/scalar NaN mismatch at i=" << i;
        if (!std::isnan(lpdf_b[i])) {
            EXPECT_EQ(lpdf_b[i], b05_.getLogProbability(xs[i]))
                << "LogPDF batch/scalar mismatch i=" << i;
        }

        EXPECT_EQ(std::isnan(cdf_b[i]), std::isnan(b05_.getCumulativeProbability(xs[i])))
            << "CDF batch/scalar NaN mismatch at i=" << i;
        if (!std::isnan(cdf_b[i])) {
            EXPECT_EQ(cdf_b[i], b05_.getCumulativeProbability(xs[i]))
                << "CDF batch/scalar mismatch i=" << i;
        }
    }
    // Explicit finite-value pins
    EXPECT_EQ(pdf_b[2], 0.0) << "PDF(+inf)";
    EXPECT_EQ(pdf_b[3], 0.0) << "PDF(-inf)";
    EXPECT_EQ(lpdf_b[2], ninf) << "LogPDF(+inf)";
    EXPECT_EQ(lpdf_b[3], ninf) << "LogPDF(-inf)";
    EXPECT_EQ(cdf_b[2], 1.0) << "CDF(+inf)";
    EXPECT_EQ(cdf_b[3], 0.0) << "CDF(-inf)";
}

// ─── Setter propagates to delegate ───────────────────────────────────────────

TEST_F(BernoulliEnhancedTest, SetterPropagates) {
    auto b = BernoulliDistribution::create(0.5).unwrap();
    EXPECT_NEAR(b.getMean(), 0.5, 1e-12);

    b.setP(0.25);
    EXPECT_NEAR(b.getP(), 0.25, 1e-12);
    EXPECT_NEAR(b.getMean(), 0.25, 1e-12);
    EXPECT_NEAR(b.getProbability(1.0), 0.25, 1e-12) << "PMF(1) = p after setP";

    b.setP(1.0);
    EXPECT_NEAR(b.getProbability(1.0), 1.0, 1e-12) << "Degenerate: all mass at 1";
    EXPECT_NEAR(b.getMean(), 1.0, 1e-12);
}

// ─── MLE accuracy ────────────────────────────────────────────────────────────

TEST_F(BernoulliEnhancedTest, MLEFit) {
    std::mt19937 rng(42);
    auto source = BernoulliDistribution::create(0.4).unwrap();
    auto data = source.sample(rng, 2000);

    auto fitted = BernoulliDistribution::create(0.5).unwrap();
    fitted.fit(data);

    EXPECT_NEAR(fitted.getP(), 0.4, 0.05) << "MLE p should be close to 0.4";
}

TEST_F(BernoulliEnhancedTest, MLEFitRejectsNonBinaryData) {
    auto fitted = BernoulliDistribution::create(0.5).unwrap();
    std::vector<double> bad_data = {0.0, 1.0, 0.5};
    EXPECT_THROW(fitted.fit(bad_data), std::invalid_argument);
}

// ─── LogPMF consistency: log(PMF(k)) == LogPMF(k) ───────────────────────────

TEST_F(BernoulliEnhancedTest, LogPMFConsistency) {
    for (double k : {0.0, 1.0}) {
        const double pmf = b05_.getProbability(k);
        const double lpmf = b05_.getLogProbability(k);
        EXPECT_NEAR(std::log(pmf), lpmf, 1e-10) << "at k=" << k;
    }
}

// ─── Batch vs scalar (VectorizedMatchesScalar) ───────────────────────────────

TEST_F(BernoulliEnhancedTest, BatchMatchesScalar) {
    const size_t N = 100;
    vector<double> xs(N), pmf_b(N), lpmf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 2);  // alternating 0/1

    b05_.getProbability(span<const double>(xs), span<double>(pmf_b));
    b05_.getLogProbability(span<const double>(xs), span<double>(lpmf_b));
    b05_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pmf_b[i], b05_.getProbability(xs[i]), 1e-12) << "PMF i=" << i;
        EXPECT_NEAR(lpmf_b[i], b05_.getLogProbability(xs[i]), 1e-12) << "LogPMF i=" << i;
        EXPECT_NEAR(cdf_b[i], b05_.getCumulativeProbability(xs[i]), 1e-12) << "CDF i=" << i;
    }
}

// ─── Speedup: PARALLEL should beat SCALAR for large batch ───────────────────

TEST_F(BernoulliEnhancedTest, VectorizedSpeedup) {
    const size_t N = 50000;
    vector<double> xs(N), out_par(N), out_scl(N);
    for (size_t i = 0; i < N; ++i)
        xs[i] = static_cast<double>(i % 2);

    detail::PerformanceHint hint_par, hint_scl;
    hint_par.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;
    hint_scl.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;

    const auto t0 = std::chrono::high_resolution_clock::now();
    b05_.getLogProbability(span<const double>(xs), span<double>(out_par), hint_par);
    const auto t1 = std::chrono::high_resolution_clock::now();
    b05_.getLogProbability(span<const double>(xs), span<double>(out_scl), hint_scl);
    const auto t2 = std::chrono::high_resolution_clock::now();

    const double par_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    const double scl_us =
        static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count());
    const double speedup = scl_us / std::max(par_us, 1.0);
    std::cout << "Bernoulli LogPMF PARALLEL speedup: " << speedup << "x "
              << "(PARALLEL " << par_us << "us, SCALAR " << scl_us << "us)\n";

    for (size_t i = 0; i < N; ++i)
        ASSERT_NEAR(out_par[i], out_scl[i], 1e-10) << "mismatch at i=" << i;
}

}  // namespace stats
