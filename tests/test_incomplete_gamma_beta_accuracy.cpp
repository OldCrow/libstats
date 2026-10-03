// tests/test_incomplete_gamma_beta_accuracy.cpp
//
// CDF accuracy of the distributions built on detail::gamma_p / gamma_q /
// beta_i, at large shape and against mpmath (issue #166). Before v2.4.2 the
// series and continued fractions stopped at DEFAULT_TOLERANCE = 1e-8 and at
// fixed caps of 100 (beta) and 1000 (gamma) iterations, and the gamma
// prefactor exp(−x + a·log x − lgamma(a)) cancelled terms of size a·log x:
// chi-squared k = 1e5 at its median was 8e-6 relative off.
//
// Two-sided: every row must be finite and within budget, scalar and batch.
// References: mpmath at dps 50, evaluated at the double nearest each literal.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace stats;

namespace {

constexpr std::size_t kN = 69;  // 8*8+5: SIMD body and scalar tail on every tier width

template <typename Dist>
void expectCdf(const std::string& name, const Dist& d, double x, double want, double budget) {
    const double got = d.getCumulativeProbability(x);
    ASSERT_TRUE(std::isfinite(got)) << name << " cdf(" << x << ") = " << got;
    EXPECT_LE(std::fabs(got - want) / want, budget)
        << name << " scalar cdf(" << x << ") = " << got << ", want " << want;

    std::vector<double> xs(kN, x), out(kN);
    const detail::PerformanceHint simd{detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED,
                                       std::nullopt};
    d.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out), simd);
    for (std::size_t i : {std::size_t{0}, kN - 1})
        EXPECT_LE(std::fabs(out[i] - want) / want, budget)
            << name << " batch cdf(" << x << ")[" << i << "] = " << out[i] << ", want " << want;
}

}  // namespace

// The #166 gate row: P(5e4, x/2) at the median, 8.3e-6 relative off before v2.4.2.
TEST(IncompleteGammaBetaAccuracy, ChiSquaredLargeK) {
    const auto d = ChiSquaredDistribution::create(1e5).unwrap();
    expectCdf("ChiSquared(1e5)", d, 99999.33, 0.49999702574336407182, 1e-12);
}

TEST(IncompleteGammaBetaAccuracy, GammaLargeShape) {
    const auto d = GammaDistribution::create(1e4, 1e-3).unwrap();
    expectCdf("Gamma(1e4, 1e-3)", d, 9.95e6, 0.30941788486118332678, 1e-12);
}

TEST(IncompleteGammaBetaAccuracy, InverseGammaLargeShape) {
    // CDF = Q(alpha, beta/x): the continued-fraction side.
    const auto d = InverseGammaDistribution::create(1e4, 1e-3).unwrap();
    expectCdf("InverseGamma(1e4, 1e-3)", d, 1e-7, 0.49867019166004216388, 1e-12);
}

TEST(IncompleteGammaBetaAccuracy, BetaLargeShape) {
    const auto d = BetaDistribution::create(1e4, 1e4).unwrap();
    expectCdf("Beta(1e4, 1e4)", d, 0.498166, 0.30197493918611573649, 1e-12);
}

TEST(IncompleteGammaBetaAccuracy, FisherFLargeDf) {
    const auto d = FDistribution::create(1e4, 1e4).unwrap();
    expectCdf("F(1e4, 1e4)", d, 0.99749, 0.45000266703024268636, 1e-12);
}

// Small shapes, where the 1e-8 stop cost 1e-9 to 1e-8.
TEST(IncompleteGammaBetaAccuracy, SmallShapes) {
    const auto g = GammaDistribution::create(2.0, 1.0).unwrap();
    expectCdf("Gamma(2, 1)", g, 1.5, 0.44217459962892542767, 1e-14);
    const auto t = StudentTDistribution::create(5.0).unwrap();
    expectCdf("StudentT(5)", t, 1.3, 0.87484968291466138803, 1e-14);
}
