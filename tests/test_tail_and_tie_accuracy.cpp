// tests/test_tail_and_tie_accuracy.cpp
//
// Defects found by the v2.4.2 full accuracy sweep, against mpmath:
//   - Cauchy quantile: tan(π(p − ½)) lost a small p in p − ½ (−1.6e16 at p = 1e-300 for −3.2e299).
//   - Discrete-uniform quantile: ceil(p·n) of the rounded product (p = 0.1, n = 10 gave 0, not 1).
//   - NegativeBinomial(5, ½) at p = ½: I_½(5, 5) came out an ulp below ½, so 5, not 4.
//   - Poisson scalar pmf: a normal approximation within 3σ of λ > 1000 (4e-3 off at λ = 1e5).
//   - log(1 − p) where log1p(−p) keeps p: Exponential, Rayleigh, Weibull and Pareto quantiles,
//     the Geometric pmf and the Binomial/Bernoulli logpdf; and Pareto's CDF 1 − (s/x)^α near the
//     scale, scalar and batch.
//
// Two-sided per the regression-guard rule. References: mpmath at dps 50, at the double nearest
// each literal.

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

constexpr double kEps = 0x1p-52;
constexpr std::size_t kN = 69;  // 8*8+5: SIMD body and scalar tail on every tier width

void expectRel(const std::string& what, double got, double want, double budget) {
    ASSERT_TRUE(std::isfinite(got)) << what << " = " << got << ", want " << want;
    EXPECT_LE(std::fabs(got - want) / std::fabs(want), budget)
        << what << " = " << got << ", want " << want;
}

}  // namespace

TEST(TailAndTie, CauchyQuantileTails) {
    const auto unit = CauchyDistribution::create(0.0, 1.0).unwrap();
    expectRel("q(1e-300)", unit.getQuantile(1e-300), -3.1830988618379066356e+299, 8 * kEps);
    expectRel("q(1 - 1e-15)", unit.getQuantile(1 - 1e-15), 3.1856450773459214697e+14, 8 * kEps);
    expectRel("q(0.3)", unit.getQuantile(0.3), -7.2654252800536093919e-1, 8 * kEps);
    const auto shifted = CauchyDistribution::create(5.0, 2.0).unwrap();
    expectRel("q(1e-20), x0 = 5, gamma = 2", shifted.getQuantile(1e-20), -6.3661977236758137794e+19,
              8 * kEps);
}

TEST(TailAndTie, DiscreteQuantileAtExactTies) {
    // Uniform on {0..9}: the smallest k with (k + 1)/10 >= p, for p the double nearest each
    // literal. 0.1 and 0.9 lie just above k/10 and 0.3 just below.
    const auto d = DiscreteDistribution::create(0, 9).unwrap();
    EXPECT_EQ(d.getQuantile(0.1), 1.0);
    EXPECT_EQ(d.getQuantile(0.9), 9.0);
    EXPECT_EQ(d.getQuantile(0.3), 2.0);
    EXPECT_EQ(d.getQuantile(0.25), 2.0);
    EXPECT_EQ(d.getQuantile(1.0), 9.0);
}

TEST(TailAndTie, NegativeBinomialMedianTie) {
    // CDF(4) = I_½(5, 5) = ½ exactly, so the median is 4.
    const auto d = NegativeBinomialDistribution::create(5.0, 0.5).unwrap();
    EXPECT_EQ(d.getCumulativeProbability(4.0), 0.5);
    EXPECT_EQ(d.getQuantile(0.5), 4.0);
}

TEST(TailAndTie, PoissonScalarPmfAtLargeLambda) {
    // The log-form pmf is good to ~1e-10 here (#172); the approximation it replaced was 4e-3 off.
    const auto d = PoissonDistribution::create(1e5).unwrap();
    expectRel("pmf(99368)", d.getProbability(99368.0), 1.7104691400673470723e-4, 1e-9);
    expectRel("pmf(100632)", d.getProbability(100632.0), 1.7140555741152663145e-4, 1e-9);
}

TEST(TailAndTie, Log1pQuantiles) {
    expectRel("exponential q(1e-15)",
              ExponentialDistribution::create(1.0).unwrap().getQuantile(1e-15),
              1.0000000000000005777e-15, 8 * kEps);
    expectRel("rayleigh q(1e-15)", RayleighDistribution::create(1.0).unwrap().getQuantile(1e-15),
              4.4721359549995806846e-8, 8 * kEps);
    expectRel("weibull(2, 1) q(1e-15)",
              WeibullDistribution::create(2.0, 1.0).unwrap().getQuantile(1e-15),
              3.1622776601683802454e-8, 8 * kEps);
    expectRel("weibull(0.5, 1) q(1e-10)",
              WeibullDistribution::create(0.5, 1.0).unwrap().getQuantile(1e-10),
              1.0000000001000000729e-20, 8 * kEps);
    // (1 − p)^(−1e6) at p = 1e-15: 1 + 1e-9, which the rounding of 1 − p moved by 8e-12.
    expectRel("pareto(1, 1e-6) q(1e-15)",
              ParetoDistribution::create(1.0, 1e-6).unwrap().getQuantile(1e-15),
              1.0000000010000000005, 8 * kEps);
}

TEST(TailAndTie, Log1pDensities) {
    expectRel("geometric(1e-6) pmf(6.5e8)",
              GeometricDistribution::create(1e-6).unwrap().getProbability(6.5e8),
              5.1102908331064815684e-289, 1e-12);  // |log pmf| = 664: the law allows ~1.5e-13
    expectRel("bernoulli(1e-10) logpdf(0)",
              BernoulliDistribution::create(1e-10).unwrap().getLogProbability(0.0),
              -1.0000000000500000364e-10, 8 * kEps);
    expectRel("binomial(10, 1e-10) logpdf(0)",
              BinomialDistribution::create(10, 1e-10).unwrap().getLogProbability(0.0),
              -1.0000000000500000364e-9, 8 * kEps);
}

TEST(TailAndTie, ParetoCdfNearScale) {
    struct Row {
        double alpha, scale, x, F;
    };
    constexpr Row kRows[] = {
        {1.0, 1.0, 1.0 + 1e-6, 9.9999899991873352559e-7},
        {3.0, 2.0, 2.000002, 2.9999939997632010584e-6},
    };
    using Strategy = detail::PerformanceHint::PreferredStrategy;
    for (const Row& r : kRows) {
        const auto d = ParetoDistribution::create(r.scale, r.alpha).unwrap();
        const std::string at = "pareto cdf(" + std::to_string(r.x) + ")";
        expectRel(at, d.getCumulativeProbability(r.x), r.F, 8 * kEps);
        for (Strategy s :
             {Strategy::FORCE_SCALAR, Strategy::FORCE_VECTORIZED, Strategy::FORCE_PARALLEL}) {
            std::vector<double> xs(kN, r.x), out(kN);
            d.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out),
                                       detail::PerformanceHint{s, std::nullopt});
            for (std::size_t i : {std::size_t{0}, kN - 1})
                expectRel(at + " batch strategy " + std::to_string(static_cast<int>(s)), out[i],
                          r.F, 8 * kEps);
        }
    }
}
