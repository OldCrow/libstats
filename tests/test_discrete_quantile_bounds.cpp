// tests/test_discrete_quantile_bounds.cpp
//
// Regression gate for issues #116 and #125 (second section). #116:
// NegativeBinomialDistribution::getQuantile derived its bisection upper bound from mean + 10*sigma
// + 100, computed in double and cast to int without a guard. For small p or large r that bound
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

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <span>

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

    // Round trip through the PUBLIC CDF at q = 0.5, whose quantile stays under
    // INT_MAX, so this held even while getCumulativeProbability narrowed its
    // argument to int. The past-INT_MAX round trip is the #125 section below.
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

// =========================================================================
// Issue #125: the public pmf/logpmf/cdf narrowed their count argument with
// static_cast<int>, which is UB past INT_MAX and ISA-dependent in practice:
// x86 wraps to INT_MIN (cdf -> 0, logpmf -> -inf), AArch64 saturates to
// INT_MAX (cdf -> 1, logpmf -> the constant value at INT_MAX). So after #116
// getQuantile could return a count its own CDF mapped to 0. sample() had the
// same class: std::poisson_distribution<int> with a rate past INT_MAX.
//
// Every assertion here is two-sided against a reference that is independent
// of the library (the geometric closed forms, or mpmath at 50 digits), so it
// fails on BOTH ISAs' wrong answers rather than encoding either one.
// Tolerances are set by the formulas' own floors, not by the defect: beta_i
// carries ~1e-6 absolute at b ~ 1e10 (see the header comment), the cached
// log(1-p) is ~1e-7 relative at p = 1e-9, and lgamma(k+r) - lgamma(k+1)
// cancels to ~1e-4 absolute at k ~ 1e10. The defect misses by 100%.
// =========================================================================

namespace {

// Geometric closed forms: log pmf(k) = log p + k log1p(-p);
// cdf(k) = -expm1((k+1) log1p(-p)).
double geometricLogPmf(double p, double k) {
    return std::log(p) + k * std::log1p(-p);
}
double geometricCdf(double p, double k) {
    return -std::expm1((k + 1.0) * std::log1p(-p));
}

}  // namespace

TEST(DiscreteCountNarrowing, GeometricPublicRoundTripBeyondIntMax) {
    constexpr double p = 1e-9;
    auto g = stats::GeometricDistribution::create(p).unwrap();

    const double k99 = g.getQuantile(0.99);
    ASSERT_GT(k99, kIntMax) << "test premise changed: the q=0.99 quantile fits an int";
    EXPECT_NEAR(g.getCumulativeProbability(k99), 0.99, 1e-5)
        << "the public CDF maps the library's own quantile " << k99 << " elsewhere";
}

TEST(DiscreteCountNarrowing, GeometricScalarMatchesClosedFormBeyondIntMax) {
    constexpr double p = 1e-9;
    auto g = stats::GeometricDistribution::create(p).unwrap();

    for (double k : {4.6e9, 1.6e10}) {
        const double ref_log = geometricLogPmf(p, k);
        EXPECT_NEAR(g.getLogProbability(k), ref_log, 1e-6 * std::fabs(ref_log)) << "k=" << k;
        EXPECT_NEAR(g.getProbability(k), std::exp(ref_log), 1e-5 * std::exp(ref_log)) << "k=" << k;
        EXPECT_NEAR(g.getCumulativeProbability(k), geometricCdf(p, k), 1e-5) << "k=" << k;
        // CDF floors a non-integer argument; pmf/logpmf round it.
        EXPECT_EQ(g.getCumulativeProbability(k + 0.75), g.getCumulativeProbability(k));
        EXPECT_EQ(g.getLogProbability(k + 0.25), g.getLogProbability(k));
    }
}

TEST(DiscreteCountNarrowing, NegativeBinomialScalarBeyondIntMax) {
    constexpr double r = 1e10, p = 0.5;
    auto nb = stats::NegativeBinomialDistribution::create(r, p).unwrap();

    // mpmath, 50 digits: loggamma(k+r) - loggamma(k+1) - loggamma(r)
    //                    + r log p + k log(1-p).
    EXPECT_NEAR(nb.getLogProbability(1e10), -12.778437588467374, 1e-3);
    EXPECT_NEAR(nb.getLogProbability(1.00003e10), -15.028426338776742, 1e-3);
    EXPECT_NEAR(nb.getProbability(1e10), std::exp(-12.778437588467374), 1e-8);

    for (double q : {0.5, 0.99}) {
        const double k = nb.getQuantile(q);
        ASSERT_GT(k, kIntMax);
        EXPECT_NEAR(nb.getCumulativeProbability(k), q, 1e-5) << "q=" << q;
    }
}

TEST(DiscreteCountNarrowing, BatchAgreesWithScalarBeyondIntMax) {
    auto nb = stats::NegativeBinomialDistribution::create(1e10, 0.5).unwrap();
    auto g = stats::GeometricDistribution::create(1e-9).unwrap();

    const std::array<double, 6> xs = {0.0, 3.0, 2147483647.0, 2147483648.0, 4.6e9, 1.6e10};
    std::array<double, 6> out{};

    const auto expect_same = [&](const char* what, auto scalar) {
        for (std::size_t i = 0; i < xs.size(); ++i) {
            const double s = scalar(xs[i]);
            EXPECT_TRUE(out[i] == s)
                << what << " batch " << out[i] << " vs scalar " << s << " at x=" << xs[i];
            EXPECT_TRUE(std::isfinite(s)) << what << " scalar not finite at x=" << xs[i];
        }
    };

    nb.getLogProbability(std::span<const double>(xs), std::span<double>(out));
    expect_same("nb logpmf", [&](double x) { return nb.getLogProbability(x); });
    nb.getProbability(std::span<const double>(xs), std::span<double>(out));
    expect_same("nb pmf", [&](double x) { return nb.getProbability(x); });
    nb.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out));
    expect_same("nb cdf", [&](double x) { return nb.getCumulativeProbability(x); });

    g.getLogProbability(std::span<const double>(xs), std::span<double>(out));
    expect_same("geometric logpmf", [&](double x) { return g.getLogProbability(x); });
    // The batch path must also clear the closed form, not merely agree with a
    // scalar path that could be wrong the same way.
    EXPECT_NEAR(out[5], geometricLogPmf(1e-9, 1.6e10), 1e-6 * 37.0);
}

TEST(DiscreteCountNarrowing, LogPmfIsContinuousAcrossIntMax) {
    // AArch64's saturating cast made logpmf constant past INT_MAX; x86's wrap
    // made it -inf. Either breaks the strict decrease of the geometric pmf.
    auto g = stats::GeometricDistribution::create(1e-9).unwrap();
    const double below = g.getLogProbability(2147483647.0);
    const double above = g.getLogProbability(2147483648.0);
    const double far = g.getLogProbability(4294967296.0);
    EXPECT_TRUE(std::isfinite(above));
    EXPECT_LT(above, below);
    EXPECT_NEAR(far - below, 2147483649.0 * std::log1p(-1e-9), 1e-6);
}

TEST(DiscreteCountNarrowing, SampleBeyondIntMax) {
    // Geometric(1e-9): mean (1-p)/p ~ 1e9, P(X > INT_MAX) = exp(-2.147) ~ 0.117.
    // The gamma-Poisson mixture draws rates past INT_MAX on ~12% of calls.
    auto g = stats::GeometricDistribution::create(1e-9).unwrap();
    std::mt19937 rng(125);

    constexpr int kDraws = 4000;
    double sum = 0.0;
    int beyond = 0;
    for (int i = 0; i < kDraws; ++i) {
        const double x = g.sample(rng);
        ASSERT_TRUE(std::isfinite(x)) << "draw " << i;
        ASSERT_GE(x, 0.0) << "draw " << i;
        ASSERT_EQ(x, std::floor(x)) << "draw " << i << " is not a count";
        sum += x;
        beyond += (x > kIntMax) ? 1 : 0;
    }
    // Standard error of the mean is 1e9/sqrt(4000) = 1.6%; 10% is ~6 sigma.
    EXPECT_NEAR(sum / kDraws, 1e9, 1e8);
    // Expected 0.117 * 4000 = 467, sigma ~ 20.
    EXPECT_NEAR(beyond, 467, 120);
}
