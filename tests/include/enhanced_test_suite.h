#pragma once

/**
 * @file tests/include/enhanced_test_suite.h
 * @brief Shared typed test suite for distribution enhanced tests.
 *
 * Provides four TYPED_TEST_P patterns enforced across every distribution:
 *   1. LogPDFConsistency   — log(PDF(x)) == LogPDF(x) for domain() values
 *   2. BatchMatchesScalar  — batch span API matches scalar call element-wise
 *   3. QuantileRoundTrip   — CDF(quantile(p)) ≈ p  (continuous only)
 *   4. InvalidParameters   — create() returns isError() for each bad scenario
 *
 * MLEFit stays per-distribution because parameter names and tolerances vary.
 *
 * ## Adding a new distribution
 *
 * 1. In the distribution's enhanced test file, include this header:
 *    @code
 *    #include "include/enhanced_test_suite.h"
 *    @endcode
 *
 * 2. Specialise DistTraits<T> before INSTANTIATE_TYPED_TEST_SUITE_P:
 *    @code
 *    template<> struct stats::tests::DistTraits<MyDistribution> {
 *        static MyDistribution make() { return MyDistribution::create(p1, p2).unwrap(); }
 *        static std::vector<double> domain() { return {x1, x2, x3, x4, x5}; }
 *        static double batch_lo() { return lo; }
 *        static double batch_hi() { return hi; }
 *        static std::vector<std::function<bool()>> invalid_creators() {
 *            return { [] { return MyDistribution::create(-1.0).isError(); } };
 *        }
 *        // Override these when needed:
 *        // static constexpr bool is_discrete = true;  // skips QuantileRoundTrip
 *        // static double pdf_tolerance() { return 1e-10; }
 *        // static double cdf_tolerance() { return 1e-8; }
 *        // static double quantile_tolerance() { return 1e-6; }
 *    };
 *    @endcode
 *
 * 3. Register and instantiate the shared suite:
 *    @code
 *    INSTANTIATE_TYPED_TEST_SUITE_P(MyDist, stats::tests::DistributionEnhancedTest,
 *                                   ::testing::Types<MyDistribution>);
 *    @endcode
 */

#define LIBSTATS_ENABLE_GTEST_INTEGRATION

#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <vector>

namespace stats {
namespace tests {

//==============================================================================
// DistTraitsDefaults — inherited by every DistTraits<T> specialisation
//==============================================================================

/**
 * @brief Default values for optional DistTraits members.
 *
 * Specialisations should inherit from this to get default tolerances and
 * quantile probabilities without repeating them. Explicitly override any
 * member in the specialisation to change the value for one distribution.
 */
struct DistTraitsDefaults {
    /// true for PMF-based discrete distributions; gates QuantileRoundTrip.
    static constexpr bool is_discrete = false;

    /// Tolerance for PDF and LogPDF batch-vs-scalar comparisons.
    static double pdf_tolerance() { return 1e-10; }

    /// Tolerance for CDF batch-vs-scalar comparisons.
    static double cdf_tolerance() { return 1e-8; }

    /// Tolerance for CDF(quantile(p)) ≈ p in QuantileRoundTrip.
    static double quantile_tolerance() { return 1e-6; }

    /// Probability values used in QuantileRoundTrip.
    static std::vector<double> quantile_probs() {
        return {0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95};
    }
};

//==============================================================================
// DistTraits — specialize for each distribution
//==============================================================================

/**
 * @brief Traits template for the shared distribution enhanced test suite.
 *
 * Each distribution must provide a full specialisation that inherits from
 * DistTraitsDefaults. Required members (make, domain, batch_lo, batch_hi,
 * invalid_creators) have no defaults — omitting them causes a link error.
 * Optional members (tolerances, is_discrete, quantile_probs) are inherited
 * from DistTraitsDefaults and may be overridden.
 *
 * Minimal specialisation pattern:
 * @code
 * template<> struct stats::tests::DistTraits<MyDist>
 *     : stats::tests::DistTraitsDefaults {
 *     static MyDist make() { return MyDist::create(...).unwrap(); }
 *     static std::vector<double> domain()         { return {...}; }
 *     static double batch_lo()                    { return lo; }
 *     static double batch_hi()                    { return hi; }
 *     static std::vector<std::function<bool()>> invalid_creators() {
 *         return { [] { return MyDist::create(bad).isError(); } };
 *     }
 * };
 * @endcode
 */
template <typename Dist>
struct DistTraits : DistTraitsDefaults {
    // ---- Required (must specialise; no default implementation) ---------------

    /// Construct and return the canonical fixture instance.
    static Dist make();

    /// Representative x values used for LogPDFConsistency and QuantileRoundTrip.
    /// All values must be in the distribution's support interior.
    static std::vector<double> domain();

    /// Lower bound of the linspace used in BatchMatchesScalar (N=200 points).
    static double batch_lo();

    /// Upper bound of the linspace used in BatchMatchesScalar.
    static double batch_hi();

    /**
     * @brief Factory lambdas that each return true when a bad parameter is
     *        correctly rejected.
     *
     * Each lambda should call create() or trySet*() and return .isError().
     */
    static std::vector<std::function<bool()>> invalid_creators();
};

}  // namespace tests
}  // namespace stats

//==============================================================================
// Shared typed fixture — at global scope so GTest macros work without
// a qualified suite name in INSTANTIATE_TYPED_TEST_SUITE_P.
//==============================================================================

template <typename Dist>
class DistributionEnhancedTest : public ::testing::Test {
  protected:
    void SetUp() override { dist_ = stats::tests::DistTraits<Dist>::make(); }
    Dist dist_;
};

TYPED_TEST_SUITE_P(DistributionEnhancedTest);

//------------------------------------------------------------------------------
// 1. LogPDFConsistency
//------------------------------------------------------------------------------
TYPED_TEST_P(DistributionEnhancedTest, LogPDFConsistency) {
    using Traits = stats::tests::DistTraits<TypeParam>;
    for (double x : Traits::domain()) {
        const double pdf = this->dist_.getProbability(x);
        if (pdf <= 0.0) continue;  // skip out-of-support and zero-density points
        EXPECT_NEAR(std::log(pdf), this->dist_.getLogProbability(x),
                    Traits::pdf_tolerance())
            << "log(PDF) != LogPDF at x=" << x;
    }
}

//------------------------------------------------------------------------------
// 2. BatchMatchesScalar
//------------------------------------------------------------------------------
TYPED_TEST_P(DistributionEnhancedTest, BatchMatchesScalar) {
    using Traits = stats::tests::DistTraits<TypeParam>;
    constexpr std::size_t N = 200;
    std::vector<double> xs(N), pdf_b(N), lpdf_b(N), cdf_b(N);
    const double lo = Traits::batch_lo();
    const double hi = Traits::batch_hi();
    for (std::size_t i = 0; i < N; ++i)
        xs[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(N - 1);

    this->dist_.getProbability(std::span<const double>(xs), std::span<double>(pdf_b));
    this->dist_.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf_b));
    this->dist_.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_b));

    for (std::size_t i = 0; i < N; ++i) {
        const double x = xs[i];
        EXPECT_NEAR(pdf_b[i],   this->dist_.getProbability(x),           Traits::pdf_tolerance())
            << "PDF batch mismatch at i=" << i << " x=" << x;
        EXPECT_NEAR(lpdf_b[i],  this->dist_.getLogProbability(x),        Traits::pdf_tolerance())
            << "LogPDF batch mismatch at i=" << i << " x=" << x;
        EXPECT_NEAR(cdf_b[i],   this->dist_.getCumulativeProbability(x), Traits::cdf_tolerance())
            << "CDF batch mismatch at i=" << i << " x=" << x;
    }
}

//------------------------------------------------------------------------------
// 3. QuantileRoundTrip  (continuous distributions only)
//------------------------------------------------------------------------------
TYPED_TEST_P(DistributionEnhancedTest, QuantileRoundTrip) {
    using Traits = stats::tests::DistTraits<TypeParam>;
    if constexpr (Traits::is_discrete) {
        GTEST_SKIP() << "QuantileRoundTrip is undefined for discrete distributions";
    }
    for (double p : Traits::quantile_probs()) {
        const double q = this->dist_.getQuantile(p);
        EXPECT_NEAR(this->dist_.getCumulativeProbability(q), p,
                    Traits::quantile_tolerance())
            << "CDF(quantile(" << p << ")) != " << p;
    }
}

//------------------------------------------------------------------------------
// 4. InvalidParameters
//------------------------------------------------------------------------------
TYPED_TEST_P(DistributionEnhancedTest, InvalidParameters) {
    using Traits = stats::tests::DistTraits<TypeParam>;
    const auto creators = Traits::invalid_creators();
    ASSERT_FALSE(creators.empty()) << "DistTraits must provide at least one invalid_creator";
    for (std::size_t i = 0; i < creators.size(); ++i) {
        EXPECT_TRUE(creators[i]())
            << "invalid_creators()[" << i << "] did not produce an error";
    }
}

//------------------------------------------------------------------------------
// Register all four patterns
//------------------------------------------------------------------------------
REGISTER_TYPED_TEST_SUITE_P(DistributionEnhancedTest,
    LogPDFConsistency,
    BatchMatchesScalar,
    QuantileRoundTrip,
    InvalidParameters);
