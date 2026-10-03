// tests/test_gamma_quantile_accuracy.cpp
//
// Gamma quantiles against mpmath (issue #160): GammaDistribution::getQuantile,
// the Erlang and ChiSquared quantiles that delegate to it, and
// detail::inverse_chi_squared_cdf and detail::gamma_inverse_cdf. Before v2.4.2
// the distribution solved P(α, βx) = p by Newton on the linear CDF from a
// Wilson-Hilferty seed and was 5e139 relative off at p = 1e-300; the two
// detail inverses bisected to an absolute 1e-8 in p.
//
// Two-sided per the regression-guard rule: each row must be finite and within
// the achievable-accuracy law (lawBudget below), or exactly 0 where the true
// quantile is below half the smallest subnormal.
//
// References: Gamma(α, 1) quantiles by bisection in log x on the small-side
// residual (P below the median, Q above it), mpmath at dps 60, evaluated at the
// double nearest each literal p.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>

using namespace stats;

namespace {

struct Row {
    double alpha;
    double p;
    double x;  // Gamma(alpha, 1) quantile, mpmath; 0 where it underflows a double
};

constexpr double kUpper = 1 - 1e-15;
constexpr double kLawFactor = 8.0;  // the worst row uses 0.32 of it (Zen 4, MSVC)

constexpr Row kRows[] = {
    {0.01, 1e-300, 0.0},  // 5.7e-30001
    {0.01, 1e-100, 0.0},  // 5.7e-10001
    {0.01, 1e-15, 0.0},   // 5.7e-1501
    {0.01, 0.3, 2.9174171917458686172e-53},
    {0.01, 0.7, 1.8309524563808334366e-16},
    {0.01, kUpper, 2.6654705498139439301e+1},
    // The upper tail at small shape, where Q = 1 − P cancelled: 500x the budget at α = 1e-4.
    {1e-4, 0.99995, 5.5325010273648020406e-1},
    {1e-3, 0.9995, 5.5350736918717457462e-1},
    {1e-3, 0.9999, 1.501028147278442429},
    {0.01, 0.995, 5.5606758767600336574e-1},
    {2.5, 1e-300, 1.6167038902915641898e-120},
    {2.5, 1e-100, 1.6167038902915641865e-40},
    {2.5, 1e-15, 1.6167046370724630017e-6},
    {2.5, 0.3, 1.499954066379953105},
    {2.5, 0.7, 3.032214992077452192},
    {2.5, kUpper, 3.9818831701263427637e+1},
    {1e4, 1e-300, 6.7376871915903292373e+3},
    {1e4, 1e-100, 8.0204643833717386077e+3},
    {1e4, 1e-15, 9.2264285779217920269e+3},
    {1e4, 0.3, 9.9473192620195836341e+3},
    {1e4, 0.7, 1.0052197405331651789e+4},
    {1e4, kUpper, 1.0814955460936724432e+4},
};

constexpr double kEps = 0x1p-52;

// The achievable-accuracy law (tests/test_lognormal_cdf_accuracy.cpp, law_budget): a tail
// probability q is reachable only to a relative |ln q|·2^-52, and the quantile turns that into
// a relative error in x of the same times κ = d(ln x)/d(ln q) = q/(x·pdf(x)) — 1/α in the deep
// lower tail, a hundredfold at α = 0.01. κ is evaluated at the reference x.
double lawBudget(double alpha, double p, double x) {
    if (x == 0.0)
        return 0.0;  // an underflow row: expectQuantile wants exactly 0
    const double q = std::min(p, 1.0 - p);
    const auto unit = GammaDistribution::create(alpha, 1.0).unwrap();
    const double kappa = q / (x * unit.getProbability(x));
    return kLawFactor * kEps * (1.0 + std::max(1.0, std::fabs(std::log(q))) * kappa);
}

void expectQuantile(const std::string& what, double got, double want, double budget) {
    if (want == 0.0) {
        EXPECT_EQ(got, 0.0) << what << ": the true quantile underflows";
        return;
    }
    ASSERT_TRUE(std::isfinite(got)) << what << " = " << got;
    EXPECT_LE(std::fabs(got - want) / want, budget) << what << " = " << got << ", want " << want;
}

std::string label(const char* name, double a, double p) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s(%g).quantile(%.17g)", name, a, p);
    return buf;
}

}  // namespace

TEST(GammaQuantileAccuracy, UnitRate) {
    for (const Row& r : kRows) {
        const auto d = GammaDistribution::create(r.alpha, 1.0).unwrap();
        expectQuantile(label("Gamma", r.alpha, r.p), d.getQuantile(r.p), r.x,
                       lawBudget(r.alpha, r.p, r.x));
    }
}

TEST(GammaQuantileAccuracy, RateScalesTheAnswer) {
    // Rate 4 is a power of two, so the reference scales exactly.
    for (const Row& r : kRows) {
        const auto d = GammaDistribution::create(r.alpha, 4.0).unwrap();
        expectQuantile(label("Gamma rate 4", r.alpha, r.p), d.getQuantile(r.p), r.x / 4.0,
                       lawBudget(r.alpha, r.p, r.x));
    }
}

TEST(GammaQuantileAccuracy, DetailGammaInverseCdf) {
    for (const Row& r : kRows)
        expectQuantile(label("gamma_inverse_cdf", r.alpha, r.p),
                       detail::gamma_inverse_cdf(r.p, r.alpha, 1.0), r.x,
                       lawBudget(r.alpha, r.p, r.x));
}

TEST(GammaQuantileAccuracy, ChiSquaredAndErlangDelegate) {
    // Chi-squared(k) = Gamma(k/2, scale 2); Erlang(k, rate λ) = Gamma(k, rate λ).
    struct Delegated {
        double k;
        double p;
        double x;  // Gamma(k/2 or k, 1) quantile
    };
    constexpr Delegated kChi[] = {
        {5.0, 1e-300, 1.6167038902915641898e-120},
        {1.0, 1e-15, 7.8539816339744843168e-31},
        {100.0, kUpper, 1.283187707744315709e+2},
    };
    for (const auto& c : kChi) {
        const auto d = ChiSquaredDistribution::create(c.k).unwrap();
        expectQuantile(label("ChiSquared", c.k, c.p), d.getQuantile(c.p), 2.0 * c.x,
                       lawBudget(0.5 * c.k, c.p, c.x));
        expectQuantile(label("inverse_chi_squared_cdf", c.k, c.p),
                       detail::inverse_chi_squared_cdf(c.p, c.k), 2.0 * c.x,
                       lawBudget(0.5 * c.k, c.p, c.x));
    }
    const auto erlang = ErlangDistribution::create(3, 2.0).unwrap();
    expectQuantile("Erlang(3, 2).quantile(1e-100)", erlang.getQuantile(1e-100),
                   8.4343266530174924847e-34 / 2.0,
                   lawBudget(3.0, 1e-100, 8.4343266530174924847e-34));
}

TEST(GammaQuantileAccuracy, Edges) {
    const auto d = GammaDistribution::create(2.5, 1.0).unwrap();
    EXPECT_EQ(d.getQuantile(0.0), 0.0);
    EXPECT_EQ(d.getQuantile(1.0), std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(detail::gamma_p_inv(2.5, std::nan(""))));
    EXPECT_TRUE(std::isnan(detail::gamma_p_inv(-1.0, 0.5)));
}
