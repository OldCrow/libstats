// tests/test_gamma_density_accuracy.cpp
//
// Gamma-family densities at large shape against mpmath. Before v2.4.2 the log density was
// α·log β − lgamma(α) + (α − 1)·log x − βx, whose terms of size α·log α cancel to a result of order
// one: 2e-11 relative at α = 1e4 and 7e-11 for ChiSquared at k = 1e5, in the scalar path and in
// the SIMD pipeline alike. From α = 20 the density is now log P(α, βx) − log x, with the
// incomplete-gamma prefactor P in Stirling form (#166) and log1p(t) − t by its series near t = 0.
// ChiSquared, Erlang and InverseGamma reach it through their Gamma delegate. Small shapes keep the
// direct form; the α = 2.5 row guards that.
//
// Two-sided per the regression-guard rule: finite and within budget, scalar and every batch
// strategy. References: mpmath at dps 50, at the double nearest each literal.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>
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

// logpdf to 16ε relative; the pdf to 16ε times |logpdf|, the exp's amplification.
template <typename Dist>
void expectDensity(const std::string& what, const Dist& d, double x, double logpdf) {
    const double log_budget = 16 * kEps;
    const double pdf_budget = 16 * kEps * std::max(1.0, std::fabs(logpdf));
    expectRel(what + " logpdf", d.getLogProbability(x), logpdf, log_budget);
    expectRel(what + " pdf", d.getProbability(x), std::exp(logpdf), pdf_budget);
    using Strategy = detail::PerformanceHint::PreferredStrategy;
    for (Strategy s :
         {Strategy::FORCE_SCALAR, Strategy::FORCE_VECTORIZED, Strategy::FORCE_PARALLEL}) {
        std::vector<double> xs(kN, x), out(kN);
        const detail::PerformanceHint hint{s, std::nullopt};
        const std::string tag = " batch strategy " + std::to_string(static_cast<int>(s));
        d.getLogProbability(std::span<const double>(xs), std::span<double>(out), hint);
        for (std::size_t i : {std::size_t{0}, kN - 1})
            expectRel(what + " logpdf" + tag, out[i], logpdf, log_budget);
        d.getProbability(std::span<const double>(xs), std::span<double>(out), hint);
        for (std::size_t i : {std::size_t{0}, kN - 1})
            expectRel(what + " pdf" + tag, out[i], std::exp(logpdf), pdf_budget);
    }
}

}  // namespace

TEST(GammaDensityAccuracy, Gamma) {
    struct Row {
        double alpha, beta, x, logpdf;
    };
    constexpr Row kRows[] = {
        {1e4, 1.0, 1e4, -5.5241170525260946654},
        {1e4, 1.0, 9800.0, -7.5309875204030592974},
        {1e4, 1.0, 10300.0, -9.9656534393236117419},
        {50.0, 2.0, 25.0, -2.1834694998057838272},
        {25.0, 0.5, 40.0, -3.5803020133764806571},
        {2.5, 1.0, 1.3, -1.1911364737716825747},  // below 20: the direct form, unchanged
        {1e6, 1e3, 1e3, -9.1893861653800607511e-1},
    };
    for (const Row& r : kRows) {
        const auto d = GammaDistribution::create(r.alpha, r.beta).unwrap();
        expectDensity("gamma(" + std::to_string(r.alpha) + ", " + std::to_string(r.beta) + ") at " +
                          std::to_string(r.x),
                      d, r.x, r.logpdf);
    }
}

TEST(GammaDensityAccuracy, DelegatesChiSquaredErlangInverseGamma) {
    const auto chi = ChiSquaredDistribution::create(1e5).unwrap();
    expectDensity("chi_squared(1e5) at 1e5", chi, 1e5, -7.021976522636426251);
    expectDensity("chi_squared(1e5) at 1.01e5", chi, 1.01e5, -9.5153841950854519231);
    const auto erlang = ErlangDistribution::create(10000, 1e-3).unwrap();
    expectDensity("erlang(1e4, 1e-3) at 1e7", erlang, 1e7, -1.2431872331508231717e+1);
    const auto inv = InverseGammaDistribution::create(1e4, 1e4).unwrap();
    expectDensity("inverse_gamma(1e4, 1e4) at 1", inv, 1.0, 3.6862233194500880707);
    expectDensity("inverse_gamma(1e4, 1e4) at 1.02", inv, 1.02, 1.7185791029057942731);
}
