// tests/test_log_beta_and_pvalues.cpp
//
// Two cancellation defects found during the v2.4.2 work.
//
// (1) detail::lbeta formed lgamma(a) + lgamma(b) − lgamma(a + b) directly, cancelling terms of size
//     max(a, b)·log(a + b) when one argument is large and the result is not; so did the
//     Student-t normaliser (lgamma((ν+1)/2) − lgamma(ν/2), 2e-10 relative at ν = 1e6) and the Beta
//     batch prefix. The Student-t SIMD pipeline also took vector_log(1 + x²/ν), whose ~ε absolute
//     error the density multiplies by (ν + 1)/2.
// (2) Two-sided and upper-tail p-values were formed as 1 − CDF, which is 0 below ~1e-16.
//
// Two-sided per the regression-guard rule: each value must be finite and within budget. The
// p-value references are closed forms that do not use the library.
// References: mpmath at dps 50, at the double nearest each literal.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"
#include "libstats/stats/analysis/binomial_analysis.h"
#include "libstats/stats/analysis/gaussian_analysis.h"
#include "libstats/stats/analysis/poisson_analysis.h"

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

using Strategy = detail::PerformanceHint::PreferredStrategy;

template <typename Call>
void expectBatch(const std::string& what, double x, double want, double budget, Call call) {
    for (Strategy s :
         {Strategy::FORCE_SCALAR, Strategy::FORCE_VECTORIZED, Strategy::FORCE_PARALLEL}) {
        std::vector<double> xs(kN, x), out(kN);
        call(std::span<const double>(xs), std::span<double>(out),
             detail::PerformanceHint{s, std::nullopt});
        for (std::size_t i : {std::size_t{0}, kN - 1})
            expectRel(what + " batch strategy " + std::to_string(static_cast<int>(s)), out[i], want,
                      budget);
    }
}

}  // namespace

TEST(LogBeta, LargeAndSmallArguments) {
    struct Row {
        double a, b, lbeta;
    };
    constexpr Row kRows[] = {
        {5e5, 0.5, -5.9888164957774643103},     {1e6, 2.0, -2.7631022115928048209e+1},
        {3e5, 25.0, -2.6050471441562502243e+2}, {25.0, 1e4, -1.7550375543124824553e+2},
        {1e5, 1e5, -1.3863392706134806235e+5},  {30.0, 0.5, -1.1240672740766992267},
        {3.0, 4.0, -4.0943445622221006848},     {0.5, 0.5, 1.1447298858494001741},
    };
    for (const Row& r : kRows) {
        expectRel("lbeta(" + std::to_string(r.a) + ", " + std::to_string(r.b) + ")",
                  detail::lbeta(r.a, r.b), r.lbeta, 8 * kEps);
        expectRel("lbeta swapped", detail::lbeta(r.b, r.a), r.lbeta, 8 * kEps);
    }
}

TEST(LogBeta, StudentTNormaliserAtLargeNu) {
    struct Row {
        double nu, t, logpdf;
    };
    constexpr Row kRows[] = {
        {1e6, 0.0, -9.1893878320467274174e-1},
        {1e6, 1.0, -1.4189390332045894084},
        {1e6, -3.0, -5.4189230333059220431},
        {1e4, 2.0, -2.9187635998499714461},
    };
    for (const Row& r : kRows) {
        const auto d = StudentTDistribution::create(r.nu).unwrap();
        const std::string at = "(nu=" + std::to_string(r.nu) + ", " + std::to_string(r.t) + ")";
        const double log_budget = 16 * kEps;
        const double pdf_budget = 16 * kEps * std::max(1.0, std::fabs(r.logpdf));
        expectRel("logpdf" + at, d.getLogProbability(r.t), r.logpdf, log_budget);
        expectRel("pdf" + at, d.getProbability(r.t), std::exp(r.logpdf), pdf_budget);
        expectBatch("logpdf" + at, r.t, r.logpdf, log_budget,
                    [&](auto in, auto out, auto hint) { d.getLogProbability(in, out, hint); });
        expectBatch("pdf" + at, r.t, std::exp(r.logpdf), pdf_budget,
                    [&](auto in, auto out, auto hint) { d.getProbability(in, out, hint); });
    }
}

TEST(LogBeta, BetaAndFisherFNormalisers) {
    struct BetaRow {
        double a, b, x, logpdf;
    };
    constexpr BetaRow kBeta[] = {
        {2.0, 1e5, 1e-5, 1.0512940464936895503e+1},
        {0.5, 3e5, 2e-6, 1.1694586605931166751e+1},
        {1e4, 3.0, 0.9997, 7.714567698744313749},
    };
    for (const BetaRow& r : kBeta) {
        const auto d = BetaDistribution::create(r.a, r.b).unwrap();
        const std::string at = "beta(" + std::to_string(r.a) + ", " + std::to_string(r.b) + ")";
        expectRel(at + " logpdf", d.getLogProbability(r.x), r.logpdf, 32 * kEps);
        expectBatch(at + " logpdf", r.x, r.logpdf, 32 * kEps,
                    [&](auto in, auto out, auto hint) { d.getLogProbability(in, out, hint); });
    }
    struct FRow {
        double d1, d2, x, logpdf;
    };
    constexpr FRow kF[] = {
        {3.0, 1e6, 1.0, -7.7102160020075820769e-1},
        {1e6, 4.0, 1.0, -6.1370763887677605583e-1},
    };
    for (const FRow& r : kF) {
        const auto d = FDistribution::create(r.d1, r.d2).unwrap();
        const std::string at = "F(" + std::to_string(r.d1) + ", " + std::to_string(r.d2) + ")";
        expectRel(at + " logpdf", d.getLogProbability(r.x), r.logpdf, 32 * kEps);
        expectBatch(at + " logpdf", r.x, r.logpdf, 32 * kEps,
                    [&](auto in, auto out, auto hint) { d.getLogProbability(in, out, hint); });
    }
}

TEST(PValues, SmallTailsAreNotRoundedToZero) {
    // One-sample t-test, df = 2: P(T < −|t|) = 1/(r·(r + |t|)) with r = √(t² + 2), no cancellation.
    {
        const double d = 0x1p-20;
        const auto [t, p, reject] =
            analysis::gaussian::oneSampleTTest({1.0, 1.0 + d, 1.0 - d}, 0.0);
        const double r = std::sqrt(t * t + 2.0);
        expectRel("oneSampleTTest p", p, 2.0 / (r * (r + std::fabs(t))), 64 * kEps);
        EXPECT_TRUE(reject);
    }
    // Jarque-Bera on ±1 alternating: S = 0, K = −2, JB = n/6; χ²(2) upper tail is exp(−JB/2).
    {
        std::vector<double> data(600);
        for (std::size_t i = 0; i < data.size(); ++i)
            data[i] = (i % 2 == 0) ? 1.0 : -1.0;
        const auto [jb, p, reject] = analysis::gaussian::jarqueBeraTest(data);
        ASSERT_NEAR(jb, 100.0, 1e-9);
        expectRel("jarqueBeraTest p", p, std::exp(-0.5 * jb), 64 * kEps * 50.0);
    }
    // Proportion z-test: 2·Φ(−|z|) = erfc(|z|/√2).
    {
        const auto [z, p, reject] = analysis::binomial::proportionZTest(900, 1000, 0.5);
        const double want = std::erfc(std::fabs(z) / std::sqrt(2.0));
        expectRel("proportionZTest p", p, want, 64 * kEps * std::fabs(std::log(want)));
    }
    // Poisson dispersion, n = 3: χ²(2) upper tail of D is exp(−D/2).
    {
        const auto [ratio, p, reject] = analysis::poisson::overdispersionTest({0.0, 0.0, 300.0});
        const double dispersion = 2.0 * ratio;  // (n − 1)·S²/x̄
        expectRel("overdispersionTest p", p, std::exp(-0.5 * dispersion),
                  64 * kEps * std::max(1.0, 0.5 * dispersion));
    }
}
