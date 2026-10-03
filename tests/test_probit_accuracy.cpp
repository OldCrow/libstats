// tests/test_probit_accuracy.cpp
//
// The probit — detail::inverse_normal_cdf and the Gaussian and LogNormal
// quantiles built on it — against mpmath references (issue #158). Before
// v2.4.2 all three formed √2·erf_inv(2p − 1): 2p − 1 rounds to −1 for
// p < 2^-54, so the quantile was −inf at p = 1e-300 (LogNormal: exactly 0), and
// it lost precision progressively above that (0.44 relative at p = 1e-15).
//
// Two-sided per the regression-guard rule: every row must be finite AND within
// its relative budget, so neither the old −inf nor a finite-but-wrong value
// passes. The central rows guard the Newton polish on the erf residual that
// replaced erf_inv's absolute 1e-12 stop near the median.
//
// References: tools/accuracy_vs_mpmath.py `_std_normal_quantile` at mp.dps = 50,
// evaluated at the double nearest each literal p.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <gtest/gtest.h>

using namespace stats;

namespace {

struct Row {
    double p;
    double z;  // Φ⁻¹(p), mpmath
};

constexpr Row kRows[] = {
    {1e-300, -37.047096299361199237},
    {1e-100, -21.273453560965324294},
    {1e-20, -9.2623400897984075796},
    {1e-15, -7.9413453261709967713},
    {1e-5, -4.2648907939228246102},
    {0.074, -1.4466320671589785099},  // either side of the 0.075 tail cut
    {0.076, -1.4325027208258116188},
    {0.3, -0.52440051270804081597},
    {0.5 + 0x1p-30, 2.3344794983332981399e-9},  // central: relative accuracy near 0
    {0.4999999999, -2.5066284820303539022e-10},
    {0.9, 1.2815515655446005935},
    {0.925, 1.4395314709384562291},  // either side of the 0.925 tail cut
    {0.926, 1.446632067158978807},
    {0.999999999999, 7.0344869100478352057},
};

// The probit's conditioning is |ln p|·2⁻⁵² relative near the tails (#49 law); 8e-15 covers it
// to p = 1e-300 with margin.
constexpr double kProbitBudget = 8e-15;

double relErr(double got, double want) {
    return std::fabs(got - want) / std::fabs(want);
}

}  // namespace

TEST(ProbitAccuracy, InverseNormalCdf) {
    for (const auto& r : kRows) {
        const double z = detail::inverse_normal_cdf(r.p);
        ASSERT_TRUE(std::isfinite(z)) << "inverse_normal_cdf(" << r.p << ") = " << z;
        EXPECT_LE(relErr(z, r.z), kProbitBudget)
            << "inverse_normal_cdf(" << r.p << ") = " << z << ", want " << r.z;
    }
}

TEST(ProbitAccuracy, GaussianQuantile) {
    const auto standard = GaussianDistribution::create(0.0, 1.0).unwrap();
    const auto shifted = GaussianDistribution::create(3.0, 2.0).unwrap();
    for (const auto& r : kRows) {
        const double q = standard.getQuantile(r.p);
        ASSERT_TRUE(std::isfinite(q)) << "Gaussian(0, 1) quantile(" << r.p << ") = " << q;
        EXPECT_LE(relErr(q, r.z), kProbitBudget)
            << "Gaussian(0, 1) quantile(" << r.p << ") = " << q << ", want " << r.z;

        // μ + σz: one more rounding, relative to the result, plus cancellation where μ + σz → 0
        // (none on this grid: |σz| < 3 only for the rows with z > −1.5).
        const double want = 3.0 + 2.0 * r.z;
        const double qs = shifted.getQuantile(r.p);
        ASSERT_TRUE(std::isfinite(qs)) << "Gaussian(3, 2) quantile(" << r.p << ") = " << qs;
        EXPECT_LE(std::fabs(qs - want), kProbitBudget * (std::fabs(2.0 * r.z) + 3.0))
            << "Gaussian(3, 2) quantile(" << r.p << ") = " << qs << ", want " << want;
    }
}

TEST(ProbitAccuracy, LogNormalQuantile) {
    // exp(μ + σz): the relative error of exp is |σ·Δz|, so the budget scales with |z|. Exactly 0
    // is the pre-v2.4.2 failure at p = 1e-300, where the true quantile is exp(−37.05) ≈ 8.2e-17.
    const auto d = LogNormalDistribution::create(0.0, 1.0).unwrap();
    for (const auto& r : kRows) {
        const double q = d.getQuantile(r.p);
        const double want = std::exp(r.z);
        ASSERT_TRUE(std::isfinite(q) && q > 0.0)
            << "LogNormal(0, 1) quantile(" << r.p << ") = " << q << ", want " << want;
        EXPECT_LE(relErr(q, want), kProbitBudget * (1.0 + std::fabs(r.z)))
            << "LogNormal(0, 1) quantile(" << r.p << ") = " << q << ", want " << want;
    }
}
