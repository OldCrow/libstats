// Student-t and Beta batch densities at large ν and large shape, where the SIMD pipelines take
// VectorOps::vector_log1p: log(1 + x²/ν) and log(1 − x) stay relative-accurate, so the factor
// (ν + 1)/2 or β − 1 multiplies a few ulp of the term rather than 1 − x's or 1 + x²/ν's absolute
// rounding (5e-11 in the log at ν = 1e6, 1e-11 at β = 1e5). Before vector_log1p these batches
// took the scalar loop past ν = 31 and |β − 1| = 16; every strategy is checked against mpmath
// (dps 50).

#include "libstats/distributions/beta.h"
#include "libstats/distributions/student_t.h"

#include <cmath>
#include <cstddef>
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

TEST(Log1pDensityBatches, StudentTLargeNu) {
    struct Row {
        double nu, t, logpdf;
    };
    constexpr Row kRows[] = {
        {1e6, 0.5, -1.0439388925796597209},
        {1e6, 3.0, -5.4189230333059220431},
        {1e4, 1.0, -1.4189885323713394059},
        {100.0, 2.0, -2.9020845057837100197},
    };
    for (const Row& r : kRows) {
        const auto d = StudentTDistribution::create(r.nu).unwrap();
        expectDensity("student_t(" + std::to_string(r.nu) + ") at " + std::to_string(r.t), d, r.t,
                      r.logpdf);
    }
}

TEST(Log1pDensityBatches, BetaLargeShape) {
    struct Row {
        double alpha, beta, x, logpdf;
    };
    // α small, so the direct form's lgamma and log terms do not cancel: these isolate the
    // (β − 1)·log(1 − x) term. (Both shapes large is the direct form's own cancellation, a
    // separate limit of the scalar path as much as the batch.)
    constexpr Row kRows[] = {
        {2.0, 1e5, 1e-5, 10.512940464936895503},
        {3.0, 1e4, 2e-4, 7.9037875208711273805},
        {1.5, 60.0, 0.02, 3.1205433401126828667},
        {0.5, 1e3, 1e-3, 5.3357655028126952989},
    };
    for (const Row& r : kRows) {
        const auto d = BetaDistribution::create(r.alpha, r.beta).unwrap();
        expectDensity("beta(" + std::to_string(r.alpha) + ", " + std::to_string(r.beta) + ") at " +
                          std::to_string(r.x),
                      d, r.x, r.logpdf);
    }
}
