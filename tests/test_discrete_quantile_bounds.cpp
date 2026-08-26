// tests/test_discrete_quantile_bounds.cpp
//
// Regression gate for issue #116: NegativeBinomialDistribution::getQuantile
// derived its bisection upper bound from mean + 10*sigma + 100, computed in
// double and cast to int without a guard. For small p or large r that bound
// passes INT_MAX; on x86 the cast yields INT_MIN, the search range collapses
// to a single point and every quantile comes back 0 --
// Geometric(1e-9).getQuantile(0.5) returned 0 where the answer is 6.93e8.
// Geometric is a delegation wrapper over NegativeBinomial(r=1), so both are
// covered by the one fix; both are exercised here anyway.
//
// This file is a separate binary rather than a case appended to
// test_negative_binomial_enhanced / test_geometric_enhanced because BOTH of
// those carry the "timing" label, and the correctness suite is
// `ctest -LE "timing|benchmark"` -- a guard added there would compile and
// never run, which is the second of the two #97 failure modes recorded in
// AGENTS.md. Same reasoning as test_bessel_tier, test_trig_ulp_gates and the
// three CDF accuracy gates; do not add this target to the timing label block
// in tests/CMakeLists.txt.
//
// What bounds the tolerances here is detail::beta_i, not the search. Its
// log-beta prefix forms lgamma(a+b) - lgamma(a) - lgamma(b) with b up to
// ~1e10, where each term reaches ~1e10-1e11 and one ulp is 2e-6 or worse, so
// the CDF carries ~1e-6 of absolute error at these counts. Divided by a PMF
// of 1e-11..1e-6 that is thousands of counts of quantile uncertainty. That is
// a separate, unfiled accuracy limitation of beta_i at large b; the defect
// gated here misses by 100%, so the gap between the two is enormous.

#include "libstats/core/math_utils.h"
#include "libstats/distributions/geometric.h"
#include "libstats/distributions/negative_binomial.h"

#include <cmath>
#include <gtest/gtest.h>

namespace {

// INT_MAX as a double: the threshold the shipped bound could not cross.
constexpr double kIntMax = 2147483647.0;

}  // namespace

// -------------------------------------------------------------------------
// Geometric(1e-9): mean ~1e9, so the bound is ~1.1e10.
//
// Checked against the geometric's closed form, which is independent of
// everything in the library: CDF(k) = 1 - (1-p)^(k+1), so the smallest k with
// CDF(k) >= q is ceil(log1p(-q)/log1p(-p)) - 1.
// -------------------------------------------------------------------------

TEST(DiscreteQuantileBounds, GeometricQuantileBeyondIntMax) {
    constexpr double p = 1e-9;
    auto g = stats::GeometricDistribution::create(p).unwrap();
    const auto cdf_at = [&](double k) { return stats::detail::beta_i(p, 1.0, k + 1.0); };

    for (double q : {0.5, 0.99}) {
        const double k = g.getQuantile(q);
        const double closed_form = std::ceil(std::log1p(-q) / std::log1p(-p)) - 1.0;
        EXPECT_NEAR(k, closed_form, 1e-5 * closed_form)
            << "Geometric(1e-9).getQuantile(" << q << ") = " << k << ", closed form "
            << closed_form;
        // The discrete quantile's defining property, evaluated through
        // detail::beta_i -- the same I_p(r, k+1) the public
        // getCumulativeProbability computes, but without the
        // static_cast<int>(std::floor(x)) it applies to its argument first.
        // A too-small search bound fails the first of these (the search
        // returns max_k, whose CDF is below q), so neither is tautological
        // against the bisection.
        EXPECT_GE(cdf_at(k), q - 1e-12) << "CDF(Q(" << q << ")) >= q";
        EXPECT_LT(cdf_at(k - 1.0), q) << "Q(" << q << ") is not the smallest such k";
    }

    // Round trip through the PUBLIC CDF, pinned at the one q whose quantile
    // stays under INT_MAX: getCumulativeProbability narrows its own argument
    // with static_cast<int>(std::floor(x)) and is separately broken past that
    // point. That narrowing is a different, unfiled defect and is deliberately
    // not exercised here.
    const double k50 = g.getQuantile(0.5);
    EXPECT_LT(k50, kIntMax) << "test premise changed: the q=0.5 quantile no longer fits an int";
    EXPECT_GT(k50, 0.0);
    EXPECT_NEAR(g.getCumulativeProbability(k50), 0.5, 1e-5);
}

// -------------------------------------------------------------------------
// NegativeBinomial(r=1e10, p=0.5): r is real-valued, mean = r(1-p)/p = 1e10,
// so the bound passes INT_MAX on the mean alone.
// -------------------------------------------------------------------------

TEST(DiscreteQuantileBounds, NegativeBinomialQuantileBeyondIntMax) {
    constexpr double r = 1e10, p = 0.5;
    auto nb = stats::NegativeBinomialDistribution::create(r, p).unwrap();
    const auto cdf_at = [&](double k) { return stats::detail::beta_i(p, r, k + 1.0); };

    for (double q : {0.5, 0.99}) {
        const double k = nb.getQuantile(q);
        ASSERT_GT(k, kIntMax) << "quantile collapsed to " << k << " at q=" << q;
        EXPECT_GE(cdf_at(k), q - 1e-12) << "CDF(Q(" << q << ")) >= q";
        EXPECT_LT(cdf_at(k - 1.0), q) << "Q(" << q << ") is not the smallest such k";
    }

    // Independent placement check: at r = 1e10 the shape is essentially normal
    // with mean r(1-p)/p = 1e10 and sigma = sqrt(r(1-p)/p^2) = 141421.36.
    // Measured 0.14 sigma and 2.3264 sigma; the 5 sigma window is slack, the
    // point being that the quantile lands where the normal limit says, not
    // merely that it is non-zero.
    constexpr double mean = 1e10, sigma = 141421.35623730952;
    EXPECT_NEAR(nb.getQuantile(0.5), mean, 5.0 * sigma);
    EXPECT_NEAR(nb.getQuantile(0.99), mean + 2.3263478740408408 * sigma, 5.0 * sigma);
    EXPECT_GT(nb.getQuantile(0.99), nb.getQuantile(0.5));
}

// -------------------------------------------------------------------------
// The widened bound must not disturb the small-parameter path, which still
// takes the linear scan.
// -------------------------------------------------------------------------

TEST(DiscreteQuantileBounds, SmallParametersUnchanged) {
    auto nb = stats::NegativeBinomialDistribution::create(2.0, 0.5).unwrap();
    EXPECT_DOUBLE_EQ(nb.getQuantile(0.0), 0.0);
    for (int k = 0; k <= 10; ++k) {
        const double cdf = nb.getCumulativeProbability(static_cast<double>(k));
        EXPECT_NEAR(nb.getQuantile(cdf), static_cast<double>(k), 0.5) << "at k=" << k;
    }

    auto g = stats::GeometricDistribution::create(0.5).unwrap();
    for (double q : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
        const double k = g.getQuantile(q);
        EXPECT_GE(k, 0.0);
        EXPECT_GE(g.getCumulativeProbability(k), q - 1e-12) << "CDF(Q(q)) >= q for q=" << q;
    }
}
