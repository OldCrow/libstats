// tests/test_student_t_tails.cpp
//
// Student's t against mpmath (issue #159), one group per defect:
//   (a) the quantile returned −inf below p ~ 1e-17 (a linear-CDF Newton seeded from the normal
//       quantile) and the normal quantile itself for ν > 1000;
//   (b) the CDF replaced ν ≥ 1000 by the normal CDF, 1e-3 relative off in the tail;
//   (c) the CDF returned 0 past |t| ~ 1e154, where ν/(ν + t²) underflows;
//   (d) pdf and logpdf collapsed to 0 / −inf there, where t² overflows — scalar, and every batch
//       strategy (the SIMD pipeline forms x² too).
//
// Two-sided per the regression-guard rule: every row must be finite and within its budget. The
// budgets follow the achievable-accuracy law (tests/test_lognormal_cdf_accuracy.cpp, law_budget):
// a probability m is reachable only to a relative |ln m|·2^-52, which the quantile scales by
// κ = d(ln|t|)/d(ln m). κ comes from a closed form here, not from the library under test.
//
// References: mpmath at dps 60 — the CDF from the regularized incomplete beta, logpdf in closed
// form, the quantile by bisection in log|t| on the smaller of the tail and the central mass —
// evaluated at the double nearest each literal.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace stats;

namespace {

constexpr double kEps = 0x1p-52;
constexpr double kLawFactor = 8.0;
constexpr double kUpper = 1 - 1e-15;
constexpr std::size_t kN = 69;  // 8*8+5: SIMD body and scalar tail on every tier width

// log(|t|·pdf(t)), overflow-safe, independent of the library.
double logTPdf(double nu, double t) {
    const double u = t * t / nu;
    const double log1p_u =
        std::isfinite(u) ? std::log1p(u) : 2.0 * std::log(std::fabs(t)) - std::log(nu);
    return std::log(std::fabs(t)) + std::lgamma((nu + 1) / 2) - std::lgamma(nu / 2) -
           0.5 * std::log(nu * detail::PI) - (nu + 1) / 2 * log1p_u;
}

std::string caseLabel(const char* what, double nu, double arg) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s(nu=%g, %.17g)", what, nu, arg);
    return buf;
}

void expectRel(const std::string& what, double got, double want, double budget) {
    ASSERT_TRUE(std::isfinite(got)) << what << " = " << got << ", want " << want;
    EXPECT_LE(std::fabs(got - want) / std::fabs(want), budget)
        << what << " = " << got << ", want " << want;
}

using Strategy = detail::PerformanceHint::PreferredStrategy;
constexpr Strategy kStrategies[] = {Strategy::FORCE_SCALAR, Strategy::FORCE_VECTORIZED,
                                    Strategy::FORCE_PARALLEL};

// The batch path at every forced strategy, first and last lane.
template <typename Call>
void expectBatch(const std::string& what, double x, double want, double budget, Call call) {
    for (Strategy s : kStrategies) {
        std::vector<double> xs(kN, x), out(kN);
        call(std::span<const double>(xs), std::span<double>(out),
             detail::PerformanceHint{s, std::nullopt});
        for (std::size_t i : {std::size_t{0}, kN - 1})
            expectRel(what + " batch strategy " + std::to_string(static_cast<int>(s)) + " [" +
                          std::to_string(i) + "]",
                      out[i], want, budget);
    }
}

}  // namespace

// (a) Quantile.
TEST(StudentTTails, QuantileDeepTailAndLargeNu) {
    struct Row {
        double nu, p, t;
    };
    constexpr Row kRows[] = {
        {1.001, 1e-300, -1.5985654663241894774e+299},
        {1.001, 1e-100, -2.5323902134988576464e+99},
        {1.001, 1e-15, -3.0792553979926801568e+14},
        {1.001, 0.3, -7.2630652757620176791e-1},
        {1.001, 0.4999999999, -3.1409866883209447914e-10},
        {1.001, 0.7, 7.2630652757620150161e-1},
        {1.001, kUpper, 3.0817160856569633347e+14},
        {7.0, 1e-300, -1.4457941326481904259e+43},
        {7.0, 1e-100, -3.8786258404953985174e+14},
        {7.0, 1e-15, -2.7912799927928001304e+2},
        {7.0, 0.3, -5.4910965794728504901e-1},
        {7.0, 0.4999999999, -2.5974604905604773753e-10},
        {7.0, 0.7, 5.4910965794728487833e-1},
        {7.0, kUpper, 2.7915988793683710137e+2},
        {1e6, 1e-300, -3.7059820872774391305e+1},
        {1e6, 1e-100, -2.127586598549061903e+1},
        {1e6, 1e-15, -7.941472518403508474},
        {1e6, 0.3, -5.2440067986020892095e-1},
        {1e6, 0.4999999999, -2.5066291086875527419e-10},
        {1e6, 0.7, 5.2440067986020876129e-1},
        {1e6, kUpper, 7.9415716843636456048},
    };
    for (const Row& r : kRows) {
        // The solver works on the smaller mass: the tail q, or 1 − 2q at the centre.
        const double q = std::min(r.p, 1.0 - r.p);
        const bool central = q > 0.25;
        const double mass = central ? 1.0 - 2.0 * q : q;
        const double kappa = mass / ((central ? 2.0 : 1.0) * std::exp(logTPdf(r.nu, r.t)));
        const double budget =
            kLawFactor * kEps * (1.0 + std::max(1.0, std::fabs(std::log(mass))) * kappa);
        const auto d = StudentTDistribution::create(r.nu).unwrap();
        expectRel(caseLabel("quantile", r.nu, r.p), d.getQuantile(r.p), r.t, budget);
        expectRel(caseLabel("inverse_t_cdf", r.nu, r.p), detail::inverse_t_cdf(r.p, r.nu), r.t,
                  budget);
    }
}

// (b) and (c) CDF: large ν, which took the normal shortcut, and |t| past 1e154.
TEST(StudentTTails, CdfLargeNuAndBeyondOverflow) {
    struct Row {
        double nu, t, F;
    };
    constexpr Row kRows[] = {
        {1.001, -1e200, 2.0087882395461520971e-201},  // (c)
        {1.001, -1e300, 1.5956372162536324523e-301},  // (c)
        {0.3, -1e200, 3.495007233838577007e-61},      // (c)
        {2000.0, -6.0, 1.1681211371480029443e-9},     // (b)
        {1e4, -8.0, 6.9106043645326910272e-16},       // (b)
        {1e6, -30.0, 6.0100471168317189413e-198},     // (b)
        {1e6, 0.5, 6.9146240626381430611e-1},         // (b), the centre at large ν
        {1e6, -0.5, 3.0853759373618569389e-1},        // (b)
        {7.0, -2.5, 2.0496109292876448445e-2},        // moderate ν, tail
        {7.0, 0.001, 5.0038499137750057813e-1},       // moderate ν, centre
    };
    for (const Row& r : kRows) {
        const double budget = kLawFactor * kEps * std::max(1.0, std::fabs(std::log(r.F)));
        const auto d = StudentTDistribution::create(r.nu).unwrap();
        expectRel(caseLabel("cdf", r.nu, r.t), d.getCumulativeProbability(r.t), r.F, budget);
        expectRel(caseLabel("t_cdf", r.nu, r.t), detail::t_cdf(r.t, r.nu), r.F, budget);
        expectBatch(
            caseLabel("cdf", r.nu, r.t), r.t, r.F, budget,
            [&](auto in, auto out, auto hint) { d.getCumulativeProbability(in, out, hint); });
    }
}

// B2: from ν ≈ 4e17 both x = ν/(ν + t²) and the BGRAT bound (a + 1)/(a + 2.5) round to 1, so the
// comparison on x sent every |t| to the central continued fraction at y ≈ 0: cdf(−6.36) was
// −2e-8 at ν = 1e20, Q(1e-10) = −10.5 with F(Q) = 0.5 at ν = 1e18, and 3.5 s per quantile at
// ν = 1e300. The branch is now decided on u = t²/ν. References: mpmath dps=40 quadrature of the
// density below t (the lgamma difference by its asymptotic series from ν = 1e20); at ν = 1e18
// the tail still differs from the normal limit by 1.5e-12 at t = −37, so the rows are two-sided.
TEST(StudentTTails, CdfAndQuantileBeyondTheBranchCollapse) {
    struct Row {
        double nu, t, F;
    };
    constexpr Row kRows[] = {
        {4e17, -1.5, 6.6807201268858066399e-2},    {4e17, -6.36, 1.0087687466392944292e-10},
        {4e17, -12.0, 1.7764821120777023396e-33},  {4e17, -37.0, 5.7255712225312932684e-300},
        {1e18, -1.5, 6.6807201268858066162e-2},    {1e18, -6.36, 1.00876874663929378e-10},
        {1e18, -12.0, 1.7764821120776883345e-33},  {1e18, -37.0, 5.725571222527263401e-300},
        {1e20, -1.5, 6.6807201268858066006e-2},    {1e20, -6.36, 1.0087687466392933515e-10},
        {1e20, -12.0, 1.7764821120776790911e-33},  {1e20, -37.0, 5.7255712225246036885e-300},
        {1e100, -6.36, 1.0087687466392933472e-10}, {1e100, -37.0, 5.7255712225245768227e-300},
        {1e300, -1.5, 6.6807201268858066004e-2},   {1e300, -6.36, 1.0087687466392933472e-10},
        {1e300, -12.0, 1.7764821120776789977e-33}, {1e300, -37.0, 5.7255712225245768227e-300},
    };
    for (const Row& r : kRows) {
        const double budget = kLawFactor * kEps * std::max(1.0, std::fabs(std::log(r.F)));
        const auto d = StudentTDistribution::create(r.nu).unwrap();
        expectRel(caseLabel("cdf", r.nu, r.t), d.getCumulativeProbability(r.t), r.F, budget);
        expectRel(caseLabel("cdf", r.nu, -r.t), d.getCumulativeProbability(-r.t), 1 - r.F, kEps);
        // The quantile closes on the CDF, at the quantile's own conditioning.
        const double q = d.getQuantile(r.F);
        expectRel(caseLabel("quantile", r.nu, r.F), q, r.t, 1e-13);
        expectRel(caseLabel("cdf(quantile)", r.nu, r.F), d.getCumulativeProbability(q), r.F,
                  budget);
    }
    // Monotone in p across the collapse, and matching the ν = 3e17 side to the law.
    for (double nu : {4e17, 1e18, 1e20, 1e300}) {
        const auto d = StudentTDistribution::create(nu).unwrap();
        double prev = -std::numeric_limits<double>::infinity();
        for (double p = 1e-300; p < 0.5; p *= 10) {
            const double q = d.getQuantile(p);
            ASSERT_TRUE(std::isfinite(q)) << caseLabel("quantile", nu, p);
            EXPECT_GT(q, prev) << caseLabel("quantile", nu, p);
            prev = q;
        }
    }
}

// (d) pdf and logpdf past |t| ~ 1e154.
TEST(StudentTTails, PdfLogPdfBeyondOverflow) {
    struct Row {
        double nu, t, logpdf;
    };
    constexpr Row kRows[] = {
        {1.001, 1e200, -9.2263809111590240307e+2},
        {7.0, -1e200, -3.6773070423448232169e+3},
        {1e6, 1e200, -4.5360971784802910275e+8},
        {0.5, 1e200, -6.9260592120954517366e+2},  // pdf 1.6035048770711145745e-301
        {7.0, 3e154, -2.8387446264323592837e+3},
    };
    for (const Row& r : kRows) {
        const auto d = StudentTDistribution::create(r.nu).unwrap();
        const double budget = kLawFactor * kEps;
        expectRel(caseLabel("logpdf", r.nu, r.t), d.getLogProbability(r.t), r.logpdf, budget);
        expectBatch(caseLabel("logpdf", r.nu, r.t), r.t, r.logpdf, budget,
                    [&](auto in, auto out, auto hint) { d.getLogProbability(in, out, hint); });
    }
    // The one row whose density is a normal double: relative error |logpdf|·ε from the exp.
    const auto half = StudentTDistribution::create(0.5).unwrap();
    const double pdf = 1.6035048770711145745e-301;
    const double budget = kLawFactor * kEps * 692.6;
    expectRel("pdf(nu=0.5, 1e200)", half.getProbability(1e200), pdf, budget);
    expectBatch("pdf(nu=0.5, 1e200)", 1e200, pdf, budget,
                [&](auto in, auto out, auto hint) { half.getProbability(in, out, hint); });
}

TEST(StudentTTails, Edges) {
    const auto d = StudentTDistribution::create(7.0).unwrap();
    EXPECT_EQ(d.getQuantile(0.0), -std::numeric_limits<double>::infinity());
    EXPECT_EQ(d.getQuantile(1.0), std::numeric_limits<double>::infinity());
    EXPECT_EQ(d.getQuantile(0.5), 0.0);
    EXPECT_EQ(d.getCumulativeProbability(0.0), 0.5);
    EXPECT_TRUE(std::isnan(detail::t_cdf(std::nan(""), 7.0)));
    // ν = 0.01 at p = 1e-300: |t| = (1/q)^(1/ν) is past the double range.
    EXPECT_EQ(detail::inverse_t_cdf(1e-300, 0.01), -std::numeric_limits<double>::infinity());
}
