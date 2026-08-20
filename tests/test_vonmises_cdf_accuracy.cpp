// tests/test_vonmises_cdf_accuracy.cpp
//
// Absolute-error accuracy gate for VonMisesDistribution::getCumulativeProbability
// (issue #51). Ground truth comes from tests/vonmises_cdf_vectors.inc: F(x;
// mu, kappa) evaluated by DIRECT QUADRATURE at mpmath dps=40, deliberately
// independent of the Bessel-series/trapezoidal method the implementation
// uses (scripts/gen_vonmises_cdf_vectors.py) -- a budget miss here is a
// kernel bug for the orchestrator to fix, not a test-tuning problem; do not
// loosen a budget below without a matching implementation fix.
//
// Structure mirrors tests/test_trig_ulp_gates.cpp (issue #95): checked-in
// generated reference vectors, a self-checking generator, per-bucket
// max/mean one-liners printed for the orchestrator to record measured
// values from, and gates that are demonstrated to fail against the
// pre-fix implementation before being trusted (see PLAN.md / the landing
// commit for this file's fail-first numbers).
//
// Budgets PINNED from measurement on Zen 4, 2026-08-20, against the fixed
// #51 implementation: scalar max 2.2e-16 at EVERY kappa bucket including
// 1000 (std::sin per term, smallest-term-first summation); batch max
// 8.9e-16 (worst at kappa=500/1000, vector_sin per term); specials exact.
// The pinned values below carry >=4.5x headroom over those measurements to
// absorb cross-platform libm/tier variation. Do not loosen without a
// matching kernel fix; a budget miss here is a kernel bug, not a
// test-tuning problem.

#include "libstats/distributions/von_mises.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

namespace {

// {x_bits, mu_bits, kappa_bits, F_bits}; F evaluated by direct quadrature at
// 40-digit precision (see file banner). Defines struct VmCdfVector,
// kVmCdfVectors[] (main gate, bucketed by kappa_bits) and kVmCdfSpecials[]
// (NaN/+-inf, gated separately). See scripts/gen_vonmises_cdf_vectors.py.
#include "vonmises_cdf_vectors.inc"

double bitsToF64(std::uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

// -------------------------------------------------------------------------
// Budgets. PINNED from measurement -- see file banner. Flat across kappa:
// the measured kappa-dependence (3.3e-16 -> 8.9e-16 over kappa 0.5 -> 1000,
// batch) is mild enough that one bound with headroom covers the range.
// -------------------------------------------------------------------------
constexpr double kBudgetKappaLe100 = 2e-15;   // scalar+batch, kappa <= 100
constexpr double kBudgetKappaLe1000 = 4e-15;  // scalar+batch, kappa in (100, 1000]
constexpr double kBudgetSpecials = 0.0;       // NaN/+-inf: exact per contract
// Batch (VECTORIZED/PARALLEL/AUTO strategies inside the span overload) is
// allowed to differ from the scalar path -- different summation/step-count
// strategy -- but only up to this much.
constexpr double kBudgetBatchVsScalar = 4e-15;

double budgetForKappa(double kappa) {
    return kappa <= 100.0 ? kBudgetKappaLe100 : kBudgetKappaLe1000;
}

}  // namespace

// -------------------------------------------------------------------------
// Self-test for the reference-vector loader itself: the generator's own
// domain/self-check invariants, re-verified here so a stale or
// hand-edited .inc can't silently violate what the gates below assume.
// -------------------------------------------------------------------------

TEST(VonMisesCdfGates, ReferenceVectorsWellFormed) {
    constexpr std::size_t n = sizeof(kVmCdfVectors) / sizeof(kVmCdfVectors[0]);
    ASSERT_GT(n, 0u);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = bitsToF64(kVmCdfVectors[i].x_bits);
        const double mu = bitsToF64(kVmCdfVectors[i].mu_bits);
        const double kappa = bitsToF64(kVmCdfVectors[i].kappa_bits);
        const double F = bitsToF64(kVmCdfVectors[i].F_bits);
        ASSERT_TRUE(std::isfinite(x)) << "row " << i << ": x not finite";
        ASSERT_TRUE(std::isfinite(mu)) << "row " << i << ": mu not finite";
        ASSERT_GE(kappa, 0.0) << "row " << i << ": kappa negative";
        ASSERT_GE(F, 0.0) << "row " << i << ": F below 0";
        ASSERT_LE(F, 1.0) << "row " << i << ": F above 1";
    }
}

// -------------------------------------------------------------------------
// Main gate: per-kappa-bucket scalar and batch evaluation, absolute error
// vs. the quadrature oracle.
// -------------------------------------------------------------------------

namespace {

struct BucketResult {
    double scalar_max = 0.0, scalar_worst_x = 0.0;
    double batch_max = 0.0, batch_worst_x = 0.0;
    double batch_vs_scalar_max = 0.0;
    std::size_t n = 0;
};

// Runs one kappa bucket (contiguous run of kVmCdfVectors sharing the same
// kappa_bits, per the generator's emission order) through both the scalar
// getCumulativeProbability(x) and the batch span overload (one batch call
// per bucket, PerformanceHint left at its default AUTO so the dispatcher
// picks its normal strategy), gates each against the per-kappa absolute
// budget, and cross-checks batch against scalar. Prints machine-readable
// one-liners for the orchestrator to record.
BucketResult run_bucket(double kappa, double mu, const VmCdfVector* rows, std::size_t n) {
    auto dist_result = stats::VonMisesDistribution::create(mu, kappa);
    // kappa=0 is accepted by VonMisesDistribution::create (kappa >= 0 is the
    // only validation) -- see include/libstats/distributions/von_mises.h's
    // validateParameters. No skip path is needed for this distribution.
    if (dist_result.isError()) {
        ADD_FAILURE() << "VonMisesDistribution::create(mu=" << mu << ", kappa=" << kappa
                      << ") rejected -- expected kappa=0 (and every other kappa in the "
                         "coverage sweep) to be accepted; error: "
                      << dist_result.message();
        return {};
    }
    const stats::VonMisesDistribution dist = std::move(dist_result).unwrap();

    std::vector<double> xs(n), refs(n), scalar_out(n), batch_out(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = bitsToF64(rows[i].x_bits);
        refs[i] = bitsToF64(rows[i].F_bits);
    }

    for (std::size_t i = 0; i < n; ++i)
        scalar_out[i] = dist.getCumulativeProbability(xs[i]);

    dist.getCumulativeProbability(std::span<const double>(xs), std::span<double>(batch_out));

    BucketResult r;
    r.n = n;
    for (std::size_t i = 0; i < n; ++i) {
        const double se = std::fabs(scalar_out[i] - refs[i]);
        if (se > r.scalar_max) {
            r.scalar_max = se;
            r.scalar_worst_x = xs[i];
        }
        const double be = std::fabs(batch_out[i] - refs[i]);
        if (be > r.batch_max) {
            r.batch_max = be;
            r.batch_worst_x = xs[i];
        }
        const double bs = std::fabs(batch_out[i] - scalar_out[i]);
        r.batch_vs_scalar_max = std::max(r.batch_vs_scalar_max, bs);
    }

    std::cout << std::setprecision(17) << "vmcdf scalar kappa=" << kappa
              << " max_abs=" << r.scalar_max << " worst_x=" << r.scalar_worst_x << "\n";
    std::cout << std::setprecision(17) << "vmcdf batch  kappa=" << kappa
              << " max_abs=" << r.batch_max << " worst_x=" << r.batch_worst_x << "\n";
    std::cout << std::setprecision(17) << "vmcdf batch_vs_scalar kappa=" << kappa
              << " max_abs=" << r.batch_vs_scalar_max << "\n";
    return r;
}

// Groups kVmCdfVectors into contiguous per-kappa runs (generator emission
// order) without assuming a fixed bucket size, so this test stays correct
// even if the generator's per-kappa row count changes.
std::vector<std::pair<std::size_t, std::size_t>> find_kappa_buckets(const VmCdfVector* rows,
                                                                     std::size_t n) {
    std::vector<std::pair<std::size_t, std::size_t>> buckets;  // (start, count)
    std::size_t start = 0;
    for (std::size_t i = 1; i <= n; ++i) {
        if (i == n || rows[i].kappa_bits != rows[start].kappa_bits ||
            rows[i].mu_bits != rows[start].mu_bits) {
            buckets.emplace_back(start, i - start);
            start = i;
        }
    }
    return buckets;
}

}  // namespace

TEST(VonMisesCdfGates, MainSweepPerKappaBucket) {
    constexpr std::size_t n = sizeof(kVmCdfVectors) / sizeof(kVmCdfVectors[0]);
    const auto buckets = find_kappa_buckets(kVmCdfVectors, n);
    ASSERT_GT(buckets.size(), 0u);

    for (const auto& [start, count] : buckets) {
        const double kappa = bitsToF64(kVmCdfVectors[start].kappa_bits);
        const double mu = bitsToF64(kVmCdfVectors[start].mu_bits);
        const BucketResult r = run_bucket(kappa, mu, &kVmCdfVectors[start], count);

        const double budget = budgetForKappa(kappa);
        EXPECT_LE(r.scalar_max, budget) << "scalar kappa=" << kappa << " mu=" << mu
                                        << " over budget (worst x=" << r.scalar_worst_x << ")";
        EXPECT_LE(r.batch_max, budget) << "batch kappa=" << kappa << " mu=" << mu
                                       << " over budget (worst x=" << r.batch_worst_x << ")";
        EXPECT_LE(r.batch_vs_scalar_max, kBudgetBatchVsScalar)
            << "batch vs scalar mismatch kappa=" << kappa << " mu=" << mu;
    }
}

// -------------------------------------------------------------------------
// Specials gate: NaN -> NaN, +inf -> 1, -inf -> 0, exactly (both scalar and
// batch paths), per the documented contract in
// VonMisesDistribution::getCumulativeProbability.
// -------------------------------------------------------------------------

TEST(VonMisesCdfGates, Specials) {
    constexpr std::size_t n = sizeof(kVmCdfSpecials) / sizeof(kVmCdfSpecials[0]);
    ASSERT_EQ(n, 3u) << "specials set shape changed -- update this test's expectations";

    const double mu = bitsToF64(kVmCdfSpecials[0].mu_bits);
    const double kappa = bitsToF64(kVmCdfSpecials[0].kappa_bits);
    auto dist_result = stats::VonMisesDistribution::create(mu, kappa);
    ASSERT_TRUE(dist_result.isOk()) << dist_result.message();
    const stats::VonMisesDistribution dist = std::move(dist_result).unwrap();

    std::vector<double> xs(n), refs(n), scalar_out(n), batch_out(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = bitsToF64(kVmCdfSpecials[i].x_bits);
        refs[i] = bitsToF64(kVmCdfSpecials[i].F_bits);
        scalar_out[i] = dist.getCumulativeProbability(xs[i]);
    }
    dist.getCumulativeProbability(std::span<const double>(xs), std::span<double>(batch_out));

    double scalar_max = 0.0, batch_max = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::isnan(refs[i])) {
            EXPECT_TRUE(std::isnan(scalar_out[i]))
                << "scalar: F(x=" << xs[i] << ") must be NaN, got " << scalar_out[i];
            EXPECT_TRUE(std::isnan(batch_out[i]))
                << "batch: F(x=" << xs[i] << ") must be NaN, got " << batch_out[i];
        } else {
            const double se = std::fabs(scalar_out[i] - refs[i]);
            const double be = std::fabs(batch_out[i] - refs[i]);
            scalar_max = std::max(scalar_max, se);
            batch_max = std::max(batch_max, be);
            EXPECT_EQ(scalar_out[i], refs[i])
                << "scalar: F(x=" << xs[i] << ") must equal " << refs[i] << " exactly, got "
                << scalar_out[i];
            EXPECT_EQ(batch_out[i], refs[i])
                << "batch: F(x=" << xs[i] << ") must equal " << refs[i] << " exactly, got "
                << batch_out[i];
        }
    }
    std::cout << std::setprecision(17) << "vmcdf scalar specials max_abs=" << scalar_max << "\n";
    std::cout << std::setprecision(17) << "vmcdf batch  specials max_abs=" << batch_max << "\n";
    EXPECT_LE(scalar_max, kBudgetSpecials);
    EXPECT_LE(batch_max, kBudgetSpecials);
}
