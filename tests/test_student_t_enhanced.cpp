#define LIBSTATS_ENABLE_GTEST_INTEGRATION
#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include "include/tests.h"
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
        dist5_ = std::move(result.value);
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
        auto t = StudentTDistribution::create(tc.nu).value;
        double q = t.getQuantile(0.975);
        EXPECT_NEAR(q, tc.expected, 0.002)
            << "t_{0.975}(nu=" << tc.nu << ") expected " << tc.expected << " got " << q;
    }
}

// CDF(0) = 0.5 and anti-symmetry CDF(-x) = 1 - CDF(x) for all nu
TEST_F(StudentTEnhancedTest, CDFSymmetry) {
    for (double nu : {0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0}) {
        auto t = StudentTDistribution::create(nu).value;
        EXPECT_NEAR(t.getCumulativeProbability(0.0), 0.5, 1e-8) << "CDF(0) != 0.5 for nu=" << nu;
        const double x = 1.5;
        EXPECT_NEAR(t.getCumulativeProbability(-x), 1.0 - t.getCumulativeProbability(x), 1e-8)
            << "CDF anti-symmetry failed for nu=" << nu;
    }
}

// nu=1 is the Cauchy distribution: PDF(0) = 1/pi
TEST_F(StudentTEnhancedTest, CauchyCase) {
    auto cauchy = StudentTDistribution::create(1.0).value;
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
    auto t2 = StudentTDistribution::create(2.0).value;
    EXPECT_TRUE(std::isinf(t2.getVariance()));
    EXPECT_GT(t2.getVariance(), 0.0);

    // nu=4: kurtosis is undefined (nu <= 4)
    auto t4 = StudentTDistribution::create(4.0).value;
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
    auto t = StudentTDistribution::create(5.0).value;
    EXPECT_NEAR(t.getVariance(), 5.0 / 3.0, 1e-12);
    t.setNu(10.0);
    EXPECT_NEAR(t.getVariance(), 10.0 / 8.0, 1e-12);
    EXPECT_FALSE(t.isCauchy());
    t.setNu(1.0);
    EXPECT_TRUE(t.isCauchy());
}

// MLE on t(5) samples should recover nu in a reasonable range
TEST_F(StudentTEnhancedTest, MLEFit) {
    mt19937 rng(123);
    auto source = StudentTDistribution::create(5.0).value;
    const auto data = source.sample(rng, 500);

    auto fitted = StudentTDistribution::create(1.0).value;
    fitted.fit(data);

    EXPECT_GT(fitted.getNu(), 2.0);
    EXPECT_LT(fitted.getNu(), 15.0);
}

TEST_F(StudentTEnhancedTest, InvalidParameters) {
    EXPECT_TRUE(StudentTDistribution::create(0.0).isError());
    EXPECT_TRUE(StudentTDistribution::create(-1.0).isError());
    EXPECT_TRUE(StudentTDistribution::create(std::numeric_limits<double>::quiet_NaN()).isError());

    auto t = StudentTDistribution::create(3.0).value;
    EXPECT_TRUE(t.trySetNu(-1.0).isError());
    EXPECT_DOUBLE_EQ(t.getNu(), 3.0);  // unchanged
}

}  // namespace stats
