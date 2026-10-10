#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/enhanced_test_suite.h"
#include "include/tests.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/student_t.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>
#include <vector>

using namespace std;
using namespace stats;

namespace stats {

class StudentTEnhancedTest : public ::testing::Test {
   protected:
    void SetUp() override {
        auto result = stats::StudentTDistribution::create(5.0);
        ASSERT_TRUE(result.isOk());
        dist5_ = std::move(result).unwrap();
    }
    StudentTDistribution dist5_;
};

// Two-tailed alpha=0.05 critical values (t_{0.975}) from standard tables
TEST_F(StudentTEnhancedTest, TTableValues) {
    struct TestCase {
        double nu;
        double expected;
    };
    const TestCase cases[] = {
        {1.0, 12.706}, {2.0, 4.303}, {5.0, 2.571}, {10.0, 2.228}, {30.0, 2.042}, {120.0, 1.980},
    };
    for (const auto& tc : cases) {
        auto t = StudentTDistribution::create(tc.nu).unwrap();
        double q = t.getQuantile(0.975);
        EXPECT_NEAR(q, tc.expected, 0.002)
            << "t_{0.975}(nu=" << tc.nu << ") expected " << tc.expected << " got " << q;
    }
}

// CDF(0) = 0.5 and anti-symmetry CDF(-x) = 1 - CDF(x) for all nu
TEST_F(StudentTEnhancedTest, CDFSymmetry) {
    for (double nu : {0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0}) {
        auto t = StudentTDistribution::create(nu).unwrap();
        EXPECT_NEAR(t.getCumulativeProbability(0.0), 0.5, 1e-8) << "CDF(0) != 0.5 for nu=" << nu;
        const double x = 1.5;
        EXPECT_NEAR(t.getCumulativeProbability(-x), 1.0 - t.getCumulativeProbability(x), 1e-8)
            << "CDF anti-symmetry failed for nu=" << nu;
    }
}

// nu=1 is the Cauchy distribution: PDF(0) = 1/pi
TEST_F(StudentTEnhancedTest, CauchyCase) {
    auto cauchy = StudentTDistribution::create(1.0).unwrap();
    EXPECT_TRUE(cauchy.isCauchy());
    EXPECT_NEAR(cauchy.getProbability(0.0), 1.0 / M_PI, 1e-10);
    EXPECT_TRUE(std::isnan(cauchy.getMean()));
    EXPECT_TRUE(std::isnan(cauchy.getVariance()));
}

TEST_F(StudentTEnhancedTest, MomentProperties) {
    const double nu = 5.0;
    EXPECT_DOUBLE_EQ(dist5_.getMean(), 0.0);
    EXPECT_NEAR(dist5_.getVariance(), nu / (nu - 2.0), 1e-12);  // 5/3
    EXPECT_DOUBLE_EQ(dist5_.getSkewness(), 0.0);
    EXPECT_NEAR(dist5_.getKurtosis(), 6.0 / (nu - 4.0), 1e-12);  // 6
    EXPECT_DOUBLE_EQ(dist5_.getMode(), 0.0);
    EXPECT_DOUBLE_EQ(dist5_.getMedian(), 0.0);

    // nu=2: variance = +inf
    auto t2 = StudentTDistribution::create(2.0).unwrap();
    EXPECT_TRUE(std::isinf(t2.getVariance()));
    EXPECT_GT(t2.getVariance(), 0.0);

    // nu=4: kurtosis is undefined (nu <= 4)
    auto t4 = StudentTDistribution::create(4.0).unwrap();
    EXPECT_TRUE(std::isnan(t4.getKurtosis()));
}

// log(PDF(x)) must equal LogPDF(x) everywhere
TEST_F(StudentTEnhancedTest, LogPDFConsistency) {
    const vector<double> xs = {-5.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0};
    for (double x : xs) {
        const double pdf = dist5_.getProbability(x);
        const double logpdf = dist5_.getLogProbability(x);
        EXPECT_NEAR(std::log(pdf), logpdf, 1e-10) << "at x=" << x;
    }
}

// Batch path must match scalar element-by-element
TEST_F(StudentTEnhancedTest, BatchMatchesScalar) {
    const size_t N = 300;
    vector<double> xs(N), pdf_b(N), logpdf_b(N), cdf_b(N);
    for (size_t i = 0; i < N; ++i) {
        xs[i] = -6.0 + static_cast<double>(i) * 12.0 / static_cast<double>(N - 1);
    }
    dist5_.getProbability(span<const double>(xs), span<double>(pdf_b));
    dist5_.getLogProbability(span<const double>(xs), span<double>(logpdf_b));
    dist5_.getCumulativeProbability(span<const double>(xs), span<double>(cdf_b));

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(pdf_b[i], dist5_.getProbability(xs[i]), 1e-10) << "PDF i=" << i;
        EXPECT_NEAR(logpdf_b[i], dist5_.getLogProbability(xs[i]), 1e-10) << "LogPDF i=" << i;
        EXPECT_NEAR(cdf_b[i], dist5_.getCumulativeProbability(xs[i]), 1e-8) << "CDF i=" << i;
    }
}

// setNu propagates to the cache immediately
TEST_F(StudentTEnhancedTest, SetterPropagates) {
    auto t = StudentTDistribution::create(5.0).unwrap();
    EXPECT_NEAR(t.getVariance(), 5.0 / 3.0, 1e-12);
    t.setNu(10.0);
    EXPECT_NEAR(t.getVariance(), 10.0 / 8.0, 1e-12);
    EXPECT_FALSE(t.isCauchy());
    t.setNu(1.0);
    EXPECT_TRUE(t.isCauchy());
}

// MLE on t(5) samples should recover nu in a reasonable range.
// Use 2000 samples so the sample excess kurtosis is stable enough for the
// Newton-Raphson optimizer to converge, even when the stdlib's
// std::normal_distribution / std::gamma_distribution produce a different
// sequence from the same mt19937 seed (algorithm is implementation-defined).
TEST_F(StudentTEnhancedTest, MLEFit) {
    mt19937 rng(123);
    auto source = StudentTDistribution::create(5.0).unwrap();
    const auto data = source.sample(rng, 2000);

    auto fitted = StudentTDistribution::create(1.0).unwrap();
    fitted.fit(data);

    EXPECT_GT(fitted.getNu(), 2.0);
    EXPECT_LT(fitted.getNu(), 15.0);
}

// F1: the Newton step in ν was capped upward but not downward, so from the ν = 5 start on data
// with ν ≤ 2 the first step landed on the 0.1 floor and the fit returned 0.1 for ν = 0.5, 1,
// 1.5, 2. The MLE at these ν exists and is close to the truth at n = 20000; the bound is
// generous (the replicate SE at ν = 1 is ~0.02) but two-sided, and 0.1 fails it at every row.
TEST_F(StudentTEnhancedTest, MLEFitRecoversHeavyTails) {
    for (double nu : {0.5, 1.0, 1.5, 2.0, 3.0, 30.0}) {
        mt19937 rng(7);
        auto source = StudentTDistribution::create(nu).unwrap();
        const auto data = source.sample(rng, 20000);
        auto fitted = StudentTDistribution::create(1.0).unwrap();
        fitted.fit(data);
        EXPECT_NEAR(fitted.getNu(), nu, 0.15 * nu) << "fit at nu = " << nu;
    }
}

// The ν fit solved its score equation on [0.1, 1000] and clamped, so data from t(ν ≫ 1000) or a
// Gaussian returned 1000 or less, never more. In θ = 1/ν the MLE is asymptotically normal about
// the truth with variance 1/(3.5 n) at θ = 0 (Fisher information of the t family at the Gaussian
// limit), so n = 1e5 resolves θ only to σ ≈ 1.7e-3: ν = 1e4 and 1e5 are indistinguishable from a
// Gaussian at this n. The honest checks are therefore on the θ scale:
//   (a) every fit lies within 4.5σ of the true θ (two-sided; 1e8 counts as θ ≈ 0);
//   (b) the estimator reaches past the old cap: P(θ̂ < 1e-3) ≥ Φ(0.53) ≈ 0.70 per replicate
//       (ν = 1e4 the lowest), so pooled over the 24 replicates at least 6 must exceed 1000
//       (P(fewer) = 7.9e-7 under the model; the capped fit gives 0). Per case, 3 of 8 would
//       false-alarm at 1.1e-2, and libstdc++'s draws hit it on CI;
//   (c) data no heavier-tailed than a Gaussian returns the documented limit ν = 1e8 exactly;
//   (d) ordinary ν is recovered as before.
TEST_F(StudentTEnhancedTest, MLEFitBeyondOldCapAndGaussianLimit) {
    constexpr double kNuLimit = 1e8;  // StudentTDistribution::fit's documented Gaussian limit
    constexpr size_t n = 100000;
    const double sigma_theta = 1.0 / std::sqrt(3.5 * static_cast<double>(n));

    int above_old_cap = 0;
    for (double nu : {1e4, 1e5, std::numeric_limits<double>::infinity()}) {
        for (unsigned rep = 0; rep < 8; ++rep) {
            mt19937 rng(1000 + rep);
            vector<double> data(n);
            if (std::isfinite(nu)) {
                data = StudentTDistribution::create(nu).unwrap().sample(rng, n);
            } else {
                normal_distribution<double> g(0.0, 1.0);
                for (auto& x : data)
                    x = g(rng);
            }
            auto fitted = StudentTDistribution::create(1.0).unwrap();
            fitted.fit(data);
            const double nu_hat = fitted.getNu();
            EXPECT_LE(nu_hat, kNuLimit);
            EXPECT_NEAR(1.0 / nu_hat, 1.0 / nu, 4.5 * sigma_theta)
                << "nu = " << nu << ", rep " << rep << ", nu_hat = " << nu_hat;
            if (nu_hat > 1000.0)
                ++above_old_cap;
        }
    }
    EXPECT_GE(above_old_cap, 6) << "of 24 replicates at nu = 1e4, 1e5 and a Gaussian";

    // (c) Lighter than Gaussian: ±1, a uniform grid, and a Gaussian quantile grid (whose truncated
    // tails make its fourth moment fall short of 3). The score is positive at every ν, so the
    // likelihood still rises at the bound.
    vector<double> pm1(1000);
    for (size_t i = 0; i < pm1.size(); ++i)
        pm1[i] = (i % 2 == 0) ? 1.0 : -1.0;
    vector<double> uniform_grid(1001);
    for (size_t i = 0; i < uniform_grid.size(); ++i)
        uniform_grid[i] = -1.0 + 2.0 * static_cast<double>(i) / 1000.0;
    auto stdnorm = GaussianDistribution::create(0.0, 1.0).unwrap();
    vector<double> gauss_grid(20000);
    for (size_t i = 0; i < gauss_grid.size(); ++i)
        gauss_grid[i] = stdnorm.getQuantile((static_cast<double>(i) + 0.5) / 20000.0);
    for (const auto* data : {&pm1, &uniform_grid, &gauss_grid}) {
        auto fitted = StudentTDistribution::create(1.0).unwrap();
        fitted.fit(*data);
        EXPECT_EQ(fitted.getNu(), kNuLimit);
    }

    // (d) Ordinary ν: unchanged recovery (bands as in MLEFitRecoversHeavyTails, wider at ν = 100
    // where the replicate SE at n = 1e5 is ~13).
    for (double nu : {5.0, 30.0, 100.0}) {
        mt19937 rng(7);
        const auto data = StudentTDistribution::create(nu).unwrap().sample(rng, n);
        auto fitted = StudentTDistribution::create(1.0).unwrap();
        fitted.fit(data);
        EXPECT_NEAR(fitted.getNu(), nu, (nu < 50.0 ? 0.15 : 0.5) * nu) << "fit at nu = " << nu;
    }
}

TEST_F(StudentTEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(StudentTDistribution::create(0.0).isError());
    EXPECT_TRUE(StudentTDistribution::create(-1.0).isError());
    EXPECT_TRUE(StudentTDistribution::create(std::numeric_limits<double>::quiet_NaN()).isError());

    auto t = StudentTDistribution::create(3.0).unwrap();
    EXPECT_TRUE(t.trySetNu(-1.0).isError());
    EXPECT_DOUBLE_EQ(t.getNu(), 3.0);  // unchanged
}

}  // namespace stats

//==============================================================================
// DistTraits specialization for stats::StudentTDistribution
//==============================================================================
template <>
struct stats::tests::DistTraits<stats::StudentTDistribution> : stats::tests::DistTraitsDefaults {
    static stats::StudentTDistribution make() {
        return stats::StudentTDistribution::create(3.0).unwrap();
    }
    static std::vector<double> domain() { return {-3.0, -1.0, 0.0, 1.0, 3.0}; }
    static double batch_lo() { return -5.0; }
    static double batch_hi() { return 5.0; }
    static std::vector<std::function<bool()>> invalid_creators() {
        return {
            [] { return stats::StudentTDistribution::create(0.0).isError(); },
            [] { return stats::StudentTDistribution::create(-1.0).isError(); },
            [] {
                return stats::StudentTDistribution::create(std::numeric_limits<double>::quiet_NaN())
                    .isError();
            },
        };
    }
};

INSTANTIATE_TYPED_TEST_SUITE_P(StudentT, DistributionEnhancedTest,
                               ::testing::Types<stats::StudentTDistribution>);
