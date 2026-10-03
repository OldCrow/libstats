// tests/test_support_boundary_gates.cpp
//
// pdf/logpdf at the edges of the support, scalar and batch, under every
// forced strategy. #103 requires the batch paths to agree with the scalar
// path; the cases here are the ones where they did not, or where both
// returned something other than the limit.
//
//   #161  Gamma, ChiSquared and Weibull at x = 0: the value is the limit by
//         shape — +inf for shape < 1, the finite density for shape = 1,
//         0 / −∞ for shape > 1. The batch logpdf returned the −4605
//         MIN_LOG_PROBABILITY clamp (Gamma) or 0 / −∞ for every shape
//         (Weibull) while the scalar path returned the limit.
//   #164  Rayleigh and Weibull at x = +inf: the log density is a sum of the
//         form log(x) − x^k, which evaluates inf − inf = NaN. The limit is
//         pdf 0, logpdf −∞.
//   #165  Poisson logpdf out of the support returned the −4605 clamp where
//         Binomial, NegativeBinomial and Discrete return −∞.
//
// The special value leads and closes a 69-element span (8*8+5), so it is
// evaluated by the SIMD body and by the scalar tail on every tier width; the
// benign fill is asserted unchanged so a fix cannot pass by overwriting it.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace stats;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t kN = 69;

using Strategy = detail::PerformanceHint::PreferredStrategy;

bool same(double got, double want) {
    if (std::isnan(want))
        return std::isnan(got);
    return got == want;
}

const char* strategyName(Strategy s) {
    switch (s) {
        case Strategy::FORCE_SCALAR:
            return "FORCE_SCALAR";
        case Strategy::FORCE_VECTORIZED:
            return "FORCE_VECTORIZED";
        case Strategy::FORCE_PARALLEL:
            return "FORCE_PARALLEL";
        default:
            return "other";
    }
}

// Requires pdf(x) == pdf_want and logpdf(x) == logpdf_want on the scalar path and on every batch
// strategy.
template <typename Dist>
void expectDensityAt(const std::string& name, const Dist& d, double x, double benign,
                     double pdf_want, double logpdf_want) {
    EXPECT_TRUE(same(d.getProbability(x), pdf_want))
        << name << " scalar pdf(" << x << ") = " << d.getProbability(x) << ", want " << pdf_want;
    EXPECT_TRUE(same(d.getLogProbability(x), logpdf_want))
        << name << " scalar logpdf(" << x << ") = " << d.getLogProbability(x) << ", want "
        << logpdf_want;

    std::vector<double> xs(kN, benign);
    xs.front() = x;
    xs.back() = x;
    std::vector<double> out(kN);
    const double benign_pdf = d.getProbability(benign);
    const double benign_logpdf = d.getLogProbability(benign);

    for (Strategy s :
         {Strategy::FORCE_SCALAR, Strategy::FORCE_VECTORIZED, Strategy::FORCE_PARALLEL}) {
        const detail::PerformanceHint hint{s, std::nullopt};
        const std::string where = name + " batch " + strategyName(s);

        d.getProbability(std::span<const double>(xs), std::span<double>(out), hint);
        EXPECT_TRUE(same(out.front(), pdf_want))
            << where << " pdf(" << x << ") [body] = " << out.front();
        EXPECT_TRUE(same(out.back(), pdf_want))
            << where << " pdf(" << x << ") [tail] = " << out.back();
        EXPECT_NEAR(out[1], benign_pdf, 1e-12 * std::abs(benign_pdf)) << where << " pdf(benign)";

        d.getLogProbability(std::span<const double>(xs), std::span<double>(out), hint);
        EXPECT_TRUE(same(out.front(), logpdf_want))
            << where << " logpdf(" << x << ") [body] = " << out.front();
        EXPECT_TRUE(same(out.back(), logpdf_want))
            << where << " logpdf(" << x << ") [tail] = " << out.back();
        EXPECT_NEAR(out[1], benign_logpdf, 1e-12 * std::abs(benign_logpdf))
            << where << " logpdf(benign)";
    }
}

}  // namespace

// #161 ------------------------------------------------------------------------------------------
// x = 0 and x = −0 for shape < 1, = 1 and > 1. The shape = 1 values are the expressions each
// library path evaluates (log β for Gamma; c = log(1) − 1·log(λ) for Weibull), so the comparison
// is exact.

TEST(SupportBoundary, GammaAtZero) {
    const double beta = 2.0;
    for (double zero : {0.0, -0.0}) {
        const auto below_one = GammaDistribution::create(0.01, beta).unwrap();
        expectDensityAt("Gamma(0.01, 2)", below_one, zero, 1.5, kInf, kInf);
        const auto one = GammaDistribution::create(1.0, beta).unwrap();
        expectDensityAt("Gamma(1, 2)", one, zero, 1.5, beta, std::log(beta));
        const auto above_one = GammaDistribution::create(2.5, beta).unwrap();
        expectDensityAt("Gamma(2.5, 2)", above_one, zero, 1.5, 0.0, -kInf);
    }
}

TEST(SupportBoundary, ChiSquaredAtZero) {
    // ChiSquared(k) = Gamma(k/2, 1/2): shape 0.01, 1 and 2.5.
    for (double zero : {0.0, -0.0}) {
        const auto below_one = ChiSquaredDistribution::create(0.02).unwrap();
        expectDensityAt("ChiSquared(0.02)", below_one, zero, 1.5, kInf, kInf);
        const auto two = ChiSquaredDistribution::create(2.0).unwrap();
        expectDensityAt("ChiSquared(2)", two, zero, 1.5, 0.5, std::log(0.5));
        const auto five = ChiSquaredDistribution::create(5.0).unwrap();
        expectDensityAt("ChiSquared(5)", five, zero, 1.5, 0.0, -kInf);
    }
}

TEST(SupportBoundary, WeibullAtZero) {
    const double lambda = 2.0;
    const double c = std::log(1.0) - 1.0 * std::log(lambda);
    for (double zero : {0.0, -0.0}) {
        const auto below_one = WeibullDistribution::create(0.01, lambda).unwrap();
        expectDensityAt("Weibull(0.01, 2)", below_one, zero, 0.8, kInf, kInf);
        const auto one = WeibullDistribution::create(1.0, lambda).unwrap();
        expectDensityAt("Weibull(1, 2)", one, zero, 0.8, std::exp(c), c);
        const auto above_one = WeibullDistribution::create(2.5, lambda).unwrap();
        expectDensityAt("Weibull(2.5, 2)", above_one, zero, 0.8, 0.0, -kInf);
    }
    // The shape = 1 case is exact, not tolerant: k = 1 + 1e-9 is a k > 1 density, 0 at x = 0.
    const auto near_one = WeibullDistribution::create(1.0 + 1e-9, lambda).unwrap();
    expectDensityAt("Weibull(1 + 1e-9, 2)", near_one, 0.0, 0.8, 0.0, -kInf);
}

// #165 ------------------------------------------------------------------------------------------
// Out of the support: pdf 0, logpdf −∞, the same for Poisson as for Binomial. (Counts past INT_MAX
// are #167's gate.)

TEST(SupportBoundary, PoissonOutOfSupport) {
    const auto d = PoissonDistribution::create(3.0).unwrap();
    for (double x : {kInf, -kInf, -1.0, -2.5})
        expectDensityAt("Poisson(3)", d, x, 2.0, 0.0, -kInf);
    expectDensityAt("Poisson(3)", d, kNaN, 2.0, kNaN, kNaN);
}

TEST(SupportBoundary, BinomialOutOfSupport) {
    const auto d = BinomialDistribution::create(10, 0.3).unwrap();
    for (double x : {kInf, -kInf, -1.0, -2.5, 11.0})
        expectDensityAt("Binomial(10, 0.3)", d, x, 2.0, 0.0, -kInf);
    expectDensityAt("Binomial(10, 0.3)", d, kNaN, 2.0, kNaN, kNaN);
}

// #167 ------------------------------------------------------------------------------------------
// Counts far outside int range: the double → int cast used to run before the range check (UB;
// on x86 it yields INT_MIN, so these rows pass there either way — the fail-first is a UBSan run,
// float-cast-overflow). Every one is out of the support: pdf 0, logpdf −∞.

TEST(SupportBoundary, CountsBeyondIntRange) {
    const auto poisson = PoissonDistribution::create(3.0).unwrap();
    const auto binomial = BinomialDistribution::create(10, 0.3).unwrap();
    const auto discrete = DiscreteDistribution::create(0, 9).unwrap();
    for (double x : {1e10, 1e20, 1e300, -1e10, -1e300}) {
        expectDensityAt("Poisson(3)", poisson, x, 2.0, 0.0, -kInf);
        expectDensityAt("Binomial(10, 0.3)", binomial, x, 2.0, 0.0, -kInf);
        expectDensityAt("Discrete(0, 9)", discrete, x, 2.0, 0.0, -kInf);
    }
}

// #164 ------------------------------------------------------------------------------------------

TEST(SupportBoundary, RayleighInfinities) {
    const auto d = RayleighDistribution::create(1.0).unwrap();
    expectDensityAt("Rayleigh(1)", d, kInf, 1.5, 0.0, -kInf);
    expectDensityAt("Rayleigh(1)", d, -kInf, 1.5, 0.0, -kInf);
    expectDensityAt("Rayleigh(1)", d, kNaN, 1.5, kNaN, kNaN);
}

TEST(SupportBoundary, WeibullInfinities) {
    // k < 1, k = 1 and k > 1: the (k − 1)·log(x) term is −inf, 0·inf and +inf respectively.
    for (double k : {0.5, 1.0, 1.5}) {
        const auto d = WeibullDistribution::create(k, 1.0).unwrap();
        const std::string name = "Weibull(" + std::to_string(k) + ", 1)";
        expectDensityAt(name, d, kInf, 0.8, 0.0, -kInf);
        expectDensityAt(name, d, -kInf, 0.8, 0.0, -kInf);
        expectDensityAt(name, d, kNaN, 0.8, kNaN, kNaN);
    }
}
