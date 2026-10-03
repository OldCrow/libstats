// tests/test_von_mises_tails.cpp
//
// Von Mises against mpmath, one group per defect found by the oracle
// (tools/accuracy_vs_mpmath.py, refs vonmises_cdf / vonmises_quantile):
//   (a) the CDF had only absolute accuracy (~1e-16) in both tails — the Bessel series sums terms
//       of order 1 to a result of order 1e-40, and returned 5.55e-17 or exactly 0 there; the
//       batch path shared the defect;
//   (b) the quantile at p = 1e-300 returned +π: the left end of the support mapped to the right;
//   (c) the quantile interpolated linearly on a 2049-point grid, 1e-4 relative off at p = 1e-6
//       and 4e-3 at p = 1 − 1e-15 (κ = 100).
//
// Two-sided per the regression-guard rule: every row must be finite and within its budget. The
// budgets follow the achievable-accuracy law (tests/test_lognormal_cdf_accuracy.cpp, law_budget):
// a probability m is reachable only to a relative |ln m|·2^-52. The CDF at an exact double x
// carries no further conditioning. The quantile scales the law by m/f(x) (the absolute change in
// x per unit relative change in m) and adds the representation of t + μ; the budget is absolute
// in x so that rows at x = 0 need no special case. f comes from mpmath, not from the library.
//
// References: mpmath at dps 50. The left-tail mass G(d) = ∫_0^d exp(−κ cos φ) dφ / (2π I₀(κ)) of
// (−π, −π + d] by adaptive quadrature, cross-checked against the Bessel series at dps 120 where
// that converges; F(x) = G(x + π) for x ≤ 0 and 1 − G(π − x) for x > 0 (μ = 0), evaluated at the
// double nearest each literal with π exact. Quantiles solve G(d) = min(p, 1 − p) for d by
// bisection in ln d then Newton, map to t = −π + d (or π − d), add μ and wrap into (−π, π].

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
constexpr double kUlpBelowOne = 0x1p-53;
constexpr std::size_t kN = 69;  // 8*8+5: SIMD body and scalar tail on every tier width

std::string label(const char* what, double kappa, double mu, double arg) {
    char buf[112];
    std::snprintf(buf, sizeof buf, "%s(kappa=%g, mu=%g, %.17g)", what, kappa, mu, arg);
    return buf;
}

void expectRel(const std::string& what, double got, double want, double budget) {
    ASSERT_TRUE(std::isfinite(got)) << what << " = " << got << ", want " << want;
    EXPECT_LE(std::fabs(got - want) / std::fabs(want), budget)
        << what << " = " << got << ", want " << want;
}

void expectAbs(const std::string& what, double got, double want, double budget) {
    ASSERT_TRUE(std::isfinite(got)) << what << " = " << got << ", want " << want;
    EXPECT_LE(std::fabs(got - want), budget) << what << " = " << got << ", want " << want;
}

using Strategy = detail::PerformanceHint::PreferredStrategy;
constexpr Strategy kStrategies[] = {Strategy::FORCE_SCALAR, Strategy::FORCE_VECTORIZED,
                                    Strategy::FORCE_PARALLEL};

// The batch CDF at every forced strategy, first and last lane.
template <typename Check>
void expectBatchCdf(const VonMisesDistribution& d, const std::string& what, double x, Check check) {
    for (Strategy s : kStrategies) {
        std::vector<double> xs(kN, x), out(kN);
        d.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out),
                                   detail::PerformanceHint{s, std::nullopt});
        for (std::size_t i : {std::size_t{0}, kN - 1})
            check(what + " batch strategy " + std::to_string(static_cast<int>(s)) + " [" +
                      std::to_string(i) + "]",
                  out[i]);
    }
}

}  // namespace

// (a) CDF, μ = 0. Rows with x ≤ 0 are relative to the law. Rows with x > 0 have F near 1: where
// the survival 1 − F is below 1e-3 they must sit within one ulp (2^-53) of the double nearest F,
// otherwise within the law.
TEST(VonMisesTails, CdfBothTails) {
    struct Row {
        double kappa, x, F;
    };
    constexpr Row kRows[] = {
        {1.0, -3.14159265358977, 1.073593742650702143e-15},
        {1.0, -3.1, 1.9240271501907508299e-3},
        {1.0, -2.5, 3.1790483694380358034e-2},
        {1.0, -1.0, 2.0564469256531652001e-1},
        {1.0, 0.3, 6.0100267520273135961e-1},
        {1.0, 2.0, 9.3424095588998316528e-1},
        {1.0, 2.5, 9.6820951630561964197e-1},
        {1.0, 3.1, 9.9807597284980924917e-1},
        {1.0, 3.14159265358977, 9.9999999999999892641e-1},
        {100.0, -3.14159265358977, 1.2800846798758089482e-100},
        {100.0, -2.5245, 1.0010654713582236404e-80},
        {100.0, -1.928, 1.0367500668489575239e-60},
        {100.0, -1.4595, 9.9250730750309780134e-41},
        {100.0, -0.9646, 9.91628848287436653e-21},
        {100.0, -0.3, 1.4180426759421573286e-3},
        {100.0, -0.05, 3.0877697580536987597e-1},
        {100.0, 0.02, 5.7916020718398579291e-1},
        {100.0, 0.1, 8.4093954261548011743e-1},
        {100.0, 0.3, 9.9858195732405784267e-1},
        {100.0, 0.9646, 9.9999999999999999999e-1},
        {100.0, 1.4595, 1.0},
        {100.0, 2.5245, 1.0},
        {1000.0, -1.2, 1.6096538689590076392e-279},
        {1000.0, -0.8, 3.3568602533084160337e-134},
        {1000.0, -0.3, 1.6925030145593740605e-21},
        {1000.0, -0.1, 7.8731889942622985559e-4},
        {1000.0, -0.02, 2.6357390643730920476e-1},
        {1000.0, 0.01, 6.2406967567963059479e-1},
        {1000.0, 0.03, 8.2856991118598107917e-1},
        {1000.0, 0.1, 9.9921268110057377014e-1},
        {1000.0, 0.3, 1.0},
        {1000.0, 0.8, 1.0},
    };
    for (const Row& r : kRows) {
        const auto d = VonMisesDistribution::create(0.0, r.kappa).unwrap();
        const std::string what = label("cdf", r.kappa, 0.0, r.x);
        if (r.x > 0.0 && 1.0 - r.F < 1e-3) {
            expectAbs(what, d.getCumulativeProbability(r.x), r.F, kUlpBelowOne);
            expectBatchCdf(d, what, r.x, [&](const std::string& w, double got) {
                expectAbs(w, got, r.F, kUlpBelowOne);
            });
        } else {
            const double budget = kLawFactor * kEps * std::max(1.0, std::fabs(std::log(r.F)));
            expectRel(what, d.getCumulativeProbability(r.x), r.F, budget);
            expectBatchCdf(d, what, r.x, [&](const std::string& w, double got) {
                expectRel(w, got, r.F, budget);
            });
        }
    }
}

// (b) and (c) Quantile. x is the double nearest the true quantile, wrapped into (−π, π].
// Rows marked "left end" have a true quantile −π + d with d below an ulp of π: the nearest double
// is −PI itself, which the library's (−π, π] convention reads as +π (wrapAngle(−PI) == PI), so
// the expected value is the smallest double above −PI, nextafter(−PI, 0). The budget's
// representation term (|x| + |μ|)·8ε covers that one-ulp shift.
TEST(VonMisesTails, QuantileTailsAndAccuracy) {
    struct Row {
        double kappa, mu, p, x, f;  // f: density at x
    };
    constexpr Row kRows[] = {
        {1e-06, 0.0, 1e-300, -3.1415926535897927, 1.5915478393699203262e-1},  // left end
        {1e-06, 2.5, 1e-300, -0.6415926535897932, 1.5915478393699203262e-1},
        {1e-06, 0.0, 1e-100, -3.1415926535897927, 1.5915478393699203262e-1},  // left end
        {1e-06, 2.5, 1e-100, -0.6415926535897932, 1.5915478393699203262e-1},
        {1e-06, 0.0, 1e-15, -3.141592653589787, 1.5915478393699203262e-1},
        {1e-06, 2.5, 1e-15, -0.641592653589787, 1.5915478393699203262e-1},
        {1e-06, 0.0, 1e-06, -3.141586370398203, 1.5915478393699203576e-1},
        {1e-06, 2.5, 1e-06, -0.6415863703982029, 1.5915478393699203576e-1},
        {1e-06, 0.0, 0.3, -1.2566361103796215, 1.5915499227358925715e-1},
        {1e-06, 2.5, 0.3, 1.2433638896203785, 1.5915499227358925715e-1},
        {1e-06, 0.0, 0.5, 0.0, 1.5915510224687821639e-1},
        {1e-06, 2.5, 0.5, 2.5, 1.5915510224687821639e-1},
        {1e-06, 0.0, 0.7, 1.256636110379621, 1.5915499227358925715e-1},
        {1e-06, 2.5, 0.7, -2.5265491967999654, 1.5915499227358925715e-1},
        {1e-06, 0.0, 0.999999, 3.1415863703982025, 1.5915478393699203576e-1},
        {1e-06, 2.5, 0.999999, -0.6415989367813838, 1.5915478393699203576e-1},
        {1e-06, 0.0, 0.999999999999999, 3.141592653589787, 1.5915478393699203262e-1},
        {1e-06, 2.5, 0.999999999999999, -0.6415926535897996, 1.5915478393699203262e-1},
        {1.0, 0.0, 1e-300, -3.1415926535897927, 4.6245485762777705692e-2},  // left end
        {1.0, 2.5, 1e-300, -0.6415926535897932, 4.6245485762777705692e-2},
        {1.0, 0.0, 1e-100, -3.1415926535897927, 4.6245485762777705692e-2},  // left end
        {1.0, 2.5, 1e-100, -0.6415926535897932, 4.6245485762777705692e-2},
        {1.0, 0.0, 1e-15, -3.141592653589772, 4.6245485762777705692e-2},
        {1.0, 2.5, 1e-15, -0.6415926535897716, 4.6245485762777705692e-2},
        {1.0, 0.0, 1e-06, -3.141571029857586, 4.6245485773589571795e-2},
        {1.0, 2.5, 1e-06, -0.6415710298575861, 4.6245485773589571795e-2},
        {1.0, 0.0, 0.3, -0.6226057931250757, 2.8324875141599173038e-1},
        {1.0, 2.5, 0.3, 1.8773942068749243, 2.8324875141599173038e-1},
        {1.0, 0.0, 0.5, 0.0, 3.4171048862346315949e-1},
        {1.0, 2.5, 0.5, 2.5, 3.4171048862346315949e-1},
        {1.0, 0.0, 0.7, 0.6226057931250755, 2.8324875141599176275e-1},
        {1.0, 2.5, 0.7, 3.1226057931250755, 2.8324875141599176275e-1},
        {1.0, 0.0, 0.999999, 3.1415710298575856, 4.6245485773589571796e-2},
        {1.0, 2.5, 0.999999, -0.6416142773220009, 4.6245485773589571796e-2},
        {1.0, 0.0, 0.999999999999999, 3.141592653589772, 4.6245485762777705692e-2},
        {1.0, 2.5, 0.999999999999999, -0.6415926535898149, 4.6245485762777705692e-2},
        {100.0, 0.0, 1e-300, -3.1415926535897927, 5.5140166607329903014e-87},  // left end
        {100.0, 2.5, 1e-300, -0.6415926535897932, 5.5140166607329903014e-87},
        {100.0, 0.0, 1e-100, -3.141592653589775, 5.5140166607329903014e-87},
        {100.0, 2.5, 1e-100, -0.6415926535897751, 5.5140166607329903014e-87},
        {100.0, 0.0, 1e-15, -0.8178232623889045, 7.3867790453407433955e-14},
        {100.0, 2.5, 1e-15, 1.6821767376110954, 7.3867790453407433955e-14},
        {100.0, 0.0, 1e-06, -0.48056876295598894, 4.7995845031173577191e-5},
        {100.0, 2.5, 1e-06, 2.019431237044011, 4.7995845031173577191e-5},
        {100.0, 0.0, 0.3, -0.05251203006676839, 3.4713593420780291464},
        {100.0, 2.5, 0.3, 2.4474879699332317, 3.4713593420780291464},
        {100.0, 0.0, 0.5, 0.0, 3.9844139747464927075},
        {100.0, 2.5, 0.5, 2.5, 3.9844139747464927075},
        {100.0, 0.0, 0.7, 0.05251203006676838, 3.4713593420780294378},
        {100.0, 2.5, 0.7, 2.5525120300667683, 3.4713593420780294378},
        {100.0, 0.0, 0.999999, 0.4805687629553898, 4.7995845032502906468e-5},
        {100.0, 2.5, 0.999999, 2.98056876295539, 4.7995845032502906468e-5},
        {100.0, 0.0, 0.999999999999999, 0.81783408704634, 7.3809470124311988698e-14},
        {100.0, 2.5, 0.999999999999999, -2.9653512201332464, 7.3809470124311988698e-14},
    };
    for (const Row& r : kRows) {
        const double m = std::min(r.p, 1.0 - r.p);  // the smaller side; 1 − p is exact here
        const double budget =
            kLawFactor * kEps *
            (std::fabs(r.x) + std::fabs(r.mu) + std::max(1.0, std::fabs(std::log(m))) * m / r.f);
        const auto d = VonMisesDistribution::create(r.mu, r.kappa).unwrap();
        expectAbs(label("quantile", r.kappa, r.mu, r.p), d.getQuantile(r.p), r.x, budget);
    }
}

TEST(VonMisesTails, Edges) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double kappa : {1.0, 100.0}) {
        const auto d = VonMisesDistribution::create(0.0, kappa).unwrap();
        EXPECT_TRUE(std::isnan(d.getCumulativeProbability(nan)));
        EXPECT_TRUE(std::isnan(d.getQuantile(nan)));
        expectBatchCdf(d, "cdf(nan)", nan, [](const std::string& w, double got) {
            EXPECT_TRUE(std::isnan(got)) << w << " = " << got;
        });
        // The left end of the support stays at the left end; the right end is +π.
        const double q0 = d.getQuantile(0.0);
        EXPECT_GT(q0, -detail::PI);
        EXPECT_LT(q0, -detail::PI + 1e-15);
        EXPECT_EQ(d.getQuantile(1.0), detail::PI);
        EXPECT_EQ(d.getQuantile(0.5), 0.0);
    }
}
