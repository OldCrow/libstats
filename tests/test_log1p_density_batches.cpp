// Student-t and Beta batch densities at large ν and large shape, against mpmath (dps 50), every
// strategy. Beta past |β − 1| = 16 takes VectorOps::vector_log1p(−x) in its SIMD pipeline, which
// keeps log(1 − x) relative-accurate (formed as vector_log of 1 − x, β − 1 multiplied 1 − x's
// rounding: 1e-11 at β = 1e5). Student-t past ν = 31 takes the scalar log1p loop: a vector_log1p
// pipeline was accurate there too but slower than that loop (Kaby Lake, 2026-10-04).

#include "libstats/distributions/beta.h"
#include "libstats/distributions/student_t.h"

#include <cmath>
#include <cstddef>
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

// logpdf to 16ε relative; the pdf to 16ε times |logpdf|, the exp's amplification.
template <typename Dist>
void expectDensity(const std::string& what, const Dist& d, double x, double logpdf) {
    const double log_budget = 16 * kEps;
    const double pdf_budget = 16 * kEps * std::max(1.0, std::fabs(logpdf));
    expectRel(what + " logpdf", d.getLogProbability(x), logpdf, log_budget);
    // The pdf only where it is a normal double; past that the logpdf carries the check.
    const bool check_pdf = std::exp(logpdf) >= std::numeric_limits<double>::min();
    if (check_pdf)
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
        if (!check_pdf)
            continue;
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

// Both shapes from 20: the direct form lnc + (α−1)·log x + (β−1)·log(1−x) cancels terms of size
// (α+β)·log 2 (2.2e-14 at Beta(50, 60), 1.8e-11 in the sweep); the Stirling-form prefactor less
// log x and log(1 − x) does not. Scalar and every batch strategy, to the same 16ε budget.
TEST(Log1pDensityBatches, BetaBothShapesLargeStirling) {
    struct Row {
        double alpha, beta, x, logpdf;
    };
    constexpr Row kRows[] = {
        {50.0, 60.0, 0.4, 1.4857656893234647146},
        {25.0, 25.0, 0.45, 1.484012422758838296},
        {1e4, 1e4, 0.5, 4.7259399236233417987},
        {1e3, 2e3, 0.33, 3.7659966897085566503},
        {1e3, 2e3, 0.2, -142.01781938431997945},  // |u| ≥ ½: the std::log1p(t) − t lanes
        {20.0, 1e5, 2e-4, 9.0920541051045891369},
        // x ≪ x₀ and x near 1: log(1 + u) as log(x/x₀) and log(1 + v) as log1p(−x) − log(1 − x₀),
        // not from the rounded u, v ≈ −1, which lose most of x (or 1 − x).
        {1e3, 2e3, 1e-15, -32592.363004301324928},
        {50.0, 60.0, 1e-10, -1051.7439722196318707},
        {25.0, 25.0, 0.999999999999, -628.14875255934435756},
    };
    for (const Row& r : kRows) {
        const auto d = BetaDistribution::create(r.alpha, r.beta).unwrap();
        expectDensity("beta(" + std::to_string(r.alpha) + ", " + std::to_string(r.beta) + ") at " +
                          std::to_string(r.x),
                      d, r.x, r.logpdf);
    }
}
