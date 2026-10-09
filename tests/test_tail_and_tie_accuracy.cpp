// tests/test_tail_and_tie_accuracy.cpp
//
// Defects found by the v2.4.2 full accuracy sweep, against mpmath:
//   - Cauchy quantile: tan(π(p − ½)) lost a small p in p − ½ (−1.6e16 at p = 1e-300 for −3.2e299).
//   - Discrete-uniform quantile: ceil(p·n) of the rounded product (p = 0.1, n = 10 gave 0, not 1).
//   - NegativeBinomial(5, ½) at p = ½: I_½(5, 5) came out an ulp below ½, so 5, not 4.
//   - Poisson scalar pmf: a normal approximation within 3σ of λ > 1000 (4e-3 off at λ = 1e5).
//   - log(1 − p) where log1p(−p) keeps p: Exponential, Rayleigh, Weibull and Pareto quantiles,
//     the Geometric pmf and the Binomial/Bernoulli logpdf; Pareto's CDF 1 − (s/x)^α near the
//     scale and Weibull's 1 − exp(−(x/λ)^k) in the lower tail, scalar and batch.
//
// Two-sided per the regression-guard rule. References: mpmath at dps 50, at the double nearest
// each literal.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

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
constexpr std::size_t kN = 69;  // 8*8+5: SIMD body and scalar tail on every tier width

void expectRel(const std::string& what, double got, double want, double budget) {
    ASSERT_TRUE(std::isfinite(got)) << what << " = " << got << ", want " << want;
    EXPECT_LE(std::fabs(got - want) / std::fabs(want), budget)
        << what << " = " << got << ", want " << want;
}

// The CDF at x, scalar and under every batch strategy.
template <typename Dist>
void expectCdfEverywhere(const std::string& at, const Dist& d, double x, double F, double budget) {
    using Strategy = detail::PerformanceHint::PreferredStrategy;
    expectRel(at, d.getCumulativeProbability(x), F, budget);
    for (Strategy s :
         {Strategy::FORCE_SCALAR, Strategy::FORCE_VECTORIZED, Strategy::FORCE_PARALLEL}) {
        std::vector<double> xs(kN, x), out(kN);
        d.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out),
                                   detail::PerformanceHint{s, std::nullopt});
        for (std::size_t i : {std::size_t{0}, kN - 1})
            expectRel(at + " batch strategy " + std::to_string(static_cast<int>(s)), out[i], F,
                      budget);
    }
}

std::string cdfLabel(const char* name, double a, double b, double x) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s(%g, %g) cdf(%.17g)", name, a, b, x);
    return buf;
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
    // Uniform on {0..9}: the smallest k with F(k) >= p, F(k) = fl((k + 1)/10) as
    // getCumulativeProbability returns it (DH D7). The doubles 0.1, 0.3 and 0.9 are exactly
    // F(0), F(2) and F(8), so each is its own quantile's CDF value. (Until v2.5.0 the quantile
    // compared against the exact (k + 1)/10 and returned 1 and 9 for 0.1 and 0.9, so
    // Q(F(k)) != k at those ties.)
    const auto d = DiscreteDistribution::create(0, 9).unwrap();
    EXPECT_EQ(d.getQuantile(0.1), 0.0);
    EXPECT_EQ(d.getQuantile(0.9), 8.0);
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

TEST(TailAndTie, WeibullCdfLowerTail) {
    // 1 − exp(−(x/λ)^k) was exactly 0 below (x/λ)^k = ε/2 and 8e-4 off at 1e-15; −expm1 now.
    // (x/λ)^k is exp(k·(log x − log λ)), which scales log's rounding by k·|log(x/λ)| — ~|ln F|
    // here — so the budget is the accuracy law, 8ε·|ln F|, not Pareto's flat 8ε.
    struct Row {
        double shape, scale, x, F;
    };
    constexpr Row kRows[] = {
        {1.5, 1.0, 1e-10, 9.9999999999999955465e-16},
        {1.5, 1.0, 4.6415888336127777e-14, 9.9999999999999959921e-21},
        {0.5, 2.0, 1e-20, 7.0710678116154750501e-11},
        {3.0, 0.5, 1e-4, 7.9999999999680011501e-12},
    };
    for (const Row& r : kRows)
        expectCdfEverywhere(cdfLabel("weibull", r.shape, r.scale, r.x),
                            WeibullDistribution::create(r.shape, r.scale).unwrap(), r.x, r.F,
                            8 * kEps * std::fabs(std::log(r.F)));
}

TEST(TailAndTie, ExponentialAndRayleighCdfLowerTail) {
    // 1 − exp(−λx) and 1 − exp(−x²/2σ²) were exactly 0 below an exponent of ε/2 (Exponential(2)
    // at 1e-280, Rayleigh(1.5) at 1e-140); −expm1 now, including the λ = 1 path.
    struct Row {
        double param, x, F;
    };
    constexpr Row kExponential[] = {
        {2.0, 1e-280, 1.9999999999999999147e-280},
        {2.0, 1e-17, 2.0000000000000001231e-17},
        {1.0, 1e-20, 9.9999999999999994515e-21},
        {0.25, 4e-9, 9.9999999950000006245e-10},
    };
    for (const Row& r : kExponential)
        expectCdfEverywhere(cdfLabel("exponential", r.param, 0, r.x),
                            ExponentialDistribution::create(r.param).unwrap(), r.x, r.F, 8 * kEps);
    constexpr Row kRayleigh[] = {
        {1.5, 1e-140, 2.2222222222222221478e-281},
        {1.5, 1e-9, 2.2222222222222224988e-19},
        {0.01, 3e-10, 4.4999999999999987403e-16},
    };
    for (const Row& r : kRayleigh)
        expectCdfEverywhere(cdfLabel("rayleigh", r.param, 0, r.x),
                            RayleighDistribution::create(r.param).unwrap(), r.x, r.F, 8 * kEps);
}

TEST(TailAndTie, SpecialParametersExactly) {
    // λ = 1 and α, β = 1 short-cuts took any parameter within DEFAULT_TOLERANCE = 1e-8 of 1:
    // Exponential(1 + 5e-9) used the λ = 1 formulas (9.5e-8 off in the pdf at x = 20), and Beta
    // with α or β just below 1 returned the finite α = 1 boundary value, not +inf.
    using Strategy = detail::PerformanceHint::PreferredStrategy;
    const auto e = ExponentialDistribution::create(1.000000005).unwrap();
    struct Row {
        const char* what;
        double x, want, budget;
        bool log;
    };
    const Row kRows[] = {
        {"pdf", 20.0, 2.0611534266289741615e-9, 8 * kEps * 20, false},
        {"logpdf", 20.0, -20.000000094999999435, 8 * kEps, true},
        {"pdf", 1e-3, 0.99900050482338245796, 8 * kEps, false},
        {"logpdf", 1e-3, -0.00099999500500004287778, 8 * kEps, true},
    };
    for (const Row& r : kRows) {
        const std::string at =
            std::string("exponential(1 + 5e-9) ") + r.what + "(" + std::to_string(r.x) + ")";
        expectRel(at, r.log ? e.getLogProbability(r.x) : e.getProbability(r.x), r.want, r.budget);
        for (Strategy s :
             {Strategy::FORCE_SCALAR, Strategy::FORCE_VECTORIZED, Strategy::FORCE_PARALLEL}) {
            std::vector<double> xs(kN, r.x), out(kN);
            const detail::PerformanceHint hint{s, std::nullopt};
            if (r.log)
                e.getLogProbability(std::span<const double>(xs), std::span<double>(out), hint);
            else
                e.getProbability(std::span<const double>(xs), std::span<double>(out), hint);
            for (std::size_t i : {std::size_t{0}, kN - 1})
                expectRel(at + " batch strategy " + std::to_string(static_cast<int>(s)), out[i],
                          r.want, r.budget);
        }
    }
    expectCdfEverywhere("exponential(1 + 5e-9) cdf(1e-3)", e, 1e-3, 0.00099950017162001082154,
                        8 * kEps);

    constexpr double kInf = std::numeric_limits<double>::infinity();
    const auto left = BetaDistribution::create(1 - 1e-10, 2.0).unwrap();
    EXPECT_EQ(left.getProbability(0.0), kInf);
    EXPECT_EQ(left.getLogProbability(0.0), kInf);
    const auto right = BetaDistribution::create(2.0, 1 - 1e-10).unwrap();
    EXPECT_EQ(right.getProbability(1.0), kInf);
    EXPECT_EQ(right.getLogProbability(1.0), kInf);
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
