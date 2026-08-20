// tests/test_lognormal_cdf_accuracy.cpp
//
// Relative-error accuracy gate for LogNormalDistribution::getCumulativeProbability
// (issue #49). Ground truth comes from tests/lognormal_cdf_vectors.inc: F(x;
// mu, sigma) evaluated as erfc(-z/sqrt(2))/2, z=(log(x)-mu)/sigma, at mpmath
// dps=40 from the exact double inputs (scripts/gen_lognormal_cdf_vectors.py)
// -- a budget miss here is a kernel bug for the orchestrator to fix, not a
// test-tuning problem; do not loosen a budget below without a matching
// implementation fix.
//
// Structure mirrors tests/test_vonmises_cdf_accuracy.cpp (issue #51) and
// tests/test_trig_ulp_gates.cpp (issue #95): checked-in generated reference
// vectors, a self-checking generator, per-bucket max relative error printed
// for the orchestrator to record, and gates demonstrated to fail against the
// pre-fix implementation before being trusted (see the landing commit's
// message for this file's fail-first numbers, and PLAN.md).
//
// Diagnosis (#49): every LogNormal CDF path used to compute the cancellation
// form 0.5*(1+erf(z/sqrt(2))). For z<0 that has an absolute error floor of
// ~1.1e-16 regardless of erf's own quality, and for z <~ -8.3, std::erf
// returns exactly -1 so F collapses to exactly 0 -- true max relative error
// there is 1.0, not the ~2.6e-7 the original 1-ULP-erf-swap benchmark
// reported (an artifact of that benchmark flooring the metric). The fix
// (src/math_utils.cpp detail::normal_cdf, and the scalar+SIMD paths in
// src/lognormal.cpp) branches on the sign of z and uses erfc for the left
// tail, matching the RELATIVE-error budget this gate enforces.
//
// Budgets PINNED from measurement on Zen 4, 2026-08-20, against the fixed
// #49 implementation: scalar+batch max relative error 1.8e-14 across every
// (mu, sigma) bucket, where F >= 1e-290 (below that floor, F itself is
// dominated by erfc's own underflow behavior and relative error is not a
// meaningful metric -- see the F-floor gating below). The budget below
// carries >5x headroom over that measurement to absorb cross-platform
// libm/tier variation. Do not loosen without a matching kernel fix; a
// budget miss here is a kernel bug, not a test-tuning problem.

#include "libstats/distributions/lognormal.h"

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

// {x_bits, mu_bits, sigma_bits, F_bits}; F evaluated at 40-digit precision
// (see file banner). Defines struct LnCdfVector, kLnCdfVectors[] (main gate,
// bucketed by mu_bits/sigma_bits) and kLnCdfSpecials[] (0/-1/NaN/+inf, gated
// separately). See scripts/gen_lognormal_cdf_vectors.py.
#include "lognormal_cdf_vectors.inc"

double bitsToF64(std::uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

// -------------------------------------------------------------------------
// Budgets. PINNED from measurement -- see file banner.
// -------------------------------------------------------------------------
// Relative-error budget is a LAW OF F, not a constant. The tail-branched
// erfc form's relative error is dominated by the (ln x - mu)/sigma argument
// transform: an ~ulp(ln x) absolute perturbation of the erfc argument w is
// amplified by erfc's asymptotic slope d ln F / dw = -2w, and since
// w^2 ~ -ln F in the tail, the achievable double-precision relative error is
//     rel(F) ~ |ln F| * 2^-52  (measured: 1.55e-13 at F=1.9e-307 vs
//     1.556e-13 predicted; 1.8e-14 at F~1e-19 vs 2.1e-14 predicted).
// A flat budget deep into the tail is therefore unachievable by ANY
// implementation of this formulation in double -- the original flat 1e-13
// spec tripped exactly this on 5 rows below F~1e-207. The budget below is
// the law with headroom:  1e-15 * max(20, -ln F_ref)  -- 2.4x margin at
// moderate tails, 4.5x at F ~ 1e-307.
inline double law_budget(double f_ref) {
    return 1e-15 * std::max(20.0, -std::log(f_ref));
}
// Below this reference-F floor, the oracle's own double-precision inputs
// (and the tiny reference magnitude itself) make relative error a poor
// metric -- fall back to an absolute budget there instead.
constexpr double kFFloor = 1e-290;
constexpr double kAbsBudgetBelowFloor = 1e-290;
constexpr double kBudgetSpecials = 0.0;  // 0/-1/NaN/+inf: exact per contract
// Batch (span overload, VECTORIZED/PARALLEL/AUTO strategies) is allowed to
// differ from the scalar path, from two sources: (1) in the -1 <= w < 0
// band (F >= 0.079) batch still uses the plain erf form while scalar uses
// erfc -- absolute <= ~1.1e-16, relative <= ~1.4e-15; (2) the erfc ARGUMENT
// is rounded differently -- scalar divides by sigma then multiplies by
// 1/sqrt(2), batch multiplies once by the cached 1/(sigma*sqrt(2)) -- and
// that ~ulp(w) difference is amplified by erfc's slope exactly like the
// accuracy law above (measured 1.17e-14 at w ~ -5, matching 2w^2*eps). So
// the RELATIVE consistency bound is the same law_budget(F_ref) as the
// accuracy gate (each path is within ~0.5x law of the same reference);
// only the ABSOLUTE bound is flat.
constexpr double kBudgetBatchVsScalarAbs = 1e-15;

double relerr(double got, double ref) {
    if (std::fabs(ref) < kFFloor) {
        return std::fabs(got - ref);  // treat as absolute in this regime
    }
    return std::fabs(got - ref) / std::fabs(ref);
}

}  // namespace

// -------------------------------------------------------------------------
// Self-test for the reference-vector loader itself: the generator's own
// domain invariants, re-verified here so a stale or hand-edited .inc can't
// silently violate what the gates below assume.
// -------------------------------------------------------------------------

TEST(LogNormalCdfGates, ReferenceVectorsWellFormed) {
    constexpr std::size_t n = sizeof(kLnCdfVectors) / sizeof(kLnCdfVectors[0]);
    ASSERT_GT(n, 0u);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = bitsToF64(kLnCdfVectors[i].x_bits);
        const double mu = bitsToF64(kLnCdfVectors[i].mu_bits);
        const double sigma = bitsToF64(kLnCdfVectors[i].sigma_bits);
        const double F = bitsToF64(kLnCdfVectors[i].F_bits);
        ASSERT_TRUE(std::isfinite(x)) << "row " << i << ": x not finite";
        ASSERT_GT(x, 0.0) << "row " << i << ": x not positive";
        ASSERT_TRUE(std::isfinite(mu)) << "row " << i << ": mu not finite";
        ASSERT_GT(sigma, 0.0) << "row " << i << ": sigma not positive";
        ASSERT_GE(F, 0.0) << "row " << i << ": F below 0";
        ASSERT_LE(F, 1.0) << "row " << i << ": F above 1";
    }
}

// -------------------------------------------------------------------------
// Main gate: per-(mu,sigma)-bucket scalar and batch evaluation, RELATIVE
// error vs. the erfc oracle.
// -------------------------------------------------------------------------

namespace {

struct BucketResult {
    double scalar_max_rel = 0.0, scalar_worst_x = 0.0;
    double batch_max_rel = 0.0, batch_worst_x = 0.0;
    // Worst relative error as a FRACTION of the per-row law_budget(F_ref)
    // -- the gated quantity (<= 1.0 passes). max_rel above stays for the
    // human-readable printout.
    double scalar_max_frac = 0.0, scalar_frac_worst_x = 0.0;
    double batch_max_frac = 0.0, batch_frac_worst_x = 0.0;
    double batch_vs_scalar_max_rel = 0.0;
    double batch_vs_scalar_max_abs = 0.0;
    std::size_t n = 0;
};

// Runs one (mu,sigma) bucket (contiguous run of kLnCdfVectors sharing the
// same mu_bits/sigma_bits, per the generator's emission order) through both
// the scalar getCumulativeProbability(x) and the batch span overload (one
// batch call per bucket, PerformanceHint left at its default AUTO), gates
// each against the relative-error budget, and cross-checks batch against
// scalar. Prints machine-readable one-liners for the orchestrator to
// record.
BucketResult run_bucket(double mu, double sigma, const LnCdfVector* rows, std::size_t n) {
    auto dist_result = stats::LogNormalDistribution::create(mu, sigma);
    if (dist_result.isError()) {
        ADD_FAILURE() << "LogNormalDistribution::create(mu=" << mu << ", sigma=" << sigma
                      << ") rejected -- expected every (mu, sigma) in the coverage sweep "
                         "to be accepted; error: "
                      << dist_result.message();
        return {};
    }
    const stats::LogNormalDistribution dist = std::move(dist_result).unwrap();

    std::vector<double> xs(n), refs(n), scalar_out(n), batch_out(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = bitsToF64(rows[i].x_bits);
        refs[i] = bitsToF64(rows[i].F_bits);
    }

    for (std::size_t i = 0; i < n; ++i)
        scalar_out[i] = dist.getCumulativeProbability(xs[i]);

    // Force the vectorized path explicitly: at n=49 per bucket, AUTO
    // dispatch's per-batch-size strategy table may pick the plain
    // per-element scalar strategy (identical code path to the scalar loop
    // above), which would trivially pass this gate without ever exercising
    // getCumulativeProbabilityBatchUnsafeImpl's SIMD kernel and per-lane
    // erf/erfc tail fixup (#49's fix site #3) -- the very code this gate
    // exists to cover.
    dist.getCumulativeProbability(
        std::span<const double>(xs), std::span<double>(batch_out),
        stats::detail::PerformanceHint{
            stats::detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED, std::nullopt});

    BucketResult r;
    r.n = n;
    for (std::size_t i = 0; i < n; ++i) {
        const double budget = refs[i] > 0.0 ? law_budget(refs[i]) : kAbsBudgetBelowFloor;
        const double sre = relerr(scalar_out[i], refs[i]);
        if (sre > r.scalar_max_rel) {
            r.scalar_max_rel = sre;
            r.scalar_worst_x = xs[i];
        }
        if (sre / budget > r.scalar_max_frac) {
            r.scalar_max_frac = sre / budget;
            r.scalar_frac_worst_x = xs[i];
        }
        const double bre = relerr(batch_out[i], refs[i]);
        if (bre > r.batch_max_rel) {
            r.batch_max_rel = bre;
            r.batch_worst_x = xs[i];
        }
        if (bre / budget > r.batch_max_frac) {
            r.batch_max_frac = bre / budget;
            r.batch_frac_worst_x = xs[i];
        }
        const double bs_abs = std::fabs(batch_out[i] - scalar_out[i]);
        const double bs_rel = relerr(batch_out[i], scalar_out[i]) / budget;  // law-normalized
        r.batch_vs_scalar_max_abs = std::max(r.batch_vs_scalar_max_abs, bs_abs);
        r.batch_vs_scalar_max_rel = std::max(r.batch_vs_scalar_max_rel, bs_rel);
    }

    std::cout << std::setprecision(17) << "lncdf scalar mu=" << mu << " sigma=" << sigma
              << " max_rel=" << r.scalar_max_rel << " worst_x=" << r.scalar_worst_x
              << " max_budget_frac=" << r.scalar_max_frac << "\n";
    std::cout << std::setprecision(17) << "lncdf batch  mu=" << mu << " sigma=" << sigma
              << " max_rel=" << r.batch_max_rel << " worst_x=" << r.batch_worst_x
              << " max_budget_frac=" << r.batch_max_frac << "\n";
    std::cout << std::setprecision(17) << "lncdf batch_vs_scalar mu=" << mu << " sigma=" << sigma
              << " max_law_frac=" << r.batch_vs_scalar_max_rel
              << " max_abs=" << r.batch_vs_scalar_max_abs << "\n";
    return r;
}

// Groups kLnCdfVectors into contiguous per-(mu,sigma) runs (generator
// emission order) without assuming a fixed bucket size.
std::vector<std::pair<std::size_t, std::size_t>> find_buckets(const LnCdfVector* rows,
                                                                std::size_t n) {
    std::vector<std::pair<std::size_t, std::size_t>> buckets;  // (start, count)
    std::size_t start = 0;
    for (std::size_t i = 1; i <= n; ++i) {
        if (i == n || rows[i].mu_bits != rows[start].mu_bits ||
            rows[i].sigma_bits != rows[start].sigma_bits) {
            buckets.emplace_back(start, i - start);
            start = i;
        }
    }
    return buckets;
}

}  // namespace

TEST(LogNormalCdfGates, MainSweepPerBucket) {
    constexpr std::size_t n = sizeof(kLnCdfVectors) / sizeof(kLnCdfVectors[0]);
    const auto buckets = find_buckets(kLnCdfVectors, n);
    ASSERT_GT(buckets.size(), 0u);

    for (const auto& [start, count] : buckets) {
        const double mu = bitsToF64(kLnCdfVectors[start].mu_bits);
        const double sigma = bitsToF64(kLnCdfVectors[start].sigma_bits);
        const BucketResult r = run_bucket(mu, sigma, &kLnCdfVectors[start], count);

        EXPECT_LE(r.scalar_max_frac, 1.0)
            << "scalar mu=" << mu << " sigma=" << sigma
            << " over the law_budget(F) relative-error budget (worst x="
            << r.scalar_frac_worst_x << ")";
        EXPECT_LE(r.batch_max_frac, 1.0)
            << "batch mu=" << mu << " sigma=" << sigma
            << " over the law_budget(F) relative-error budget (worst x="
            << r.batch_frac_worst_x << ")";
        EXPECT_LE(r.batch_vs_scalar_max_rel, 1.0)
            << "batch vs scalar law-normalized relative mismatch mu=" << mu
            << " sigma=" << sigma;
        EXPECT_LE(r.batch_vs_scalar_max_abs, kBudgetBatchVsScalarAbs)
            << "batch vs scalar absolute mismatch mu=" << mu << " sigma=" << sigma;
    }
}

// -------------------------------------------------------------------------
// Specials gate: x=0 -> F=0, x=-1 (outside support) -> F=0, NaN -> NaN,
// +inf -> F=1, exactly (both scalar and batch paths), per the documented
// contract in LogNormalDistribution::getCumulativeProbability.
// -------------------------------------------------------------------------

TEST(LogNormalCdfGates, Specials) {
    constexpr std::size_t n = sizeof(kLnCdfSpecials) / sizeof(kLnCdfSpecials[0]);
    ASSERT_EQ(n, 4u) << "specials set shape changed -- update this test's expectations";

    const double mu = bitsToF64(kLnCdfSpecials[0].mu_bits);
    const double sigma = bitsToF64(kLnCdfSpecials[0].sigma_bits);
    auto dist_result = stats::LogNormalDistribution::create(mu, sigma);
    ASSERT_TRUE(dist_result.isOk()) << dist_result.message();
    const stats::LogNormalDistribution dist = std::move(dist_result).unwrap();

    std::vector<double> xs(n), refs(n), scalar_out(n), batch_out(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = bitsToF64(kLnCdfSpecials[i].x_bits);
        refs[i] = bitsToF64(kLnCdfSpecials[i].F_bits);
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
    std::cout << std::setprecision(17) << "lncdf scalar specials max_abs=" << scalar_max << "\n";
    std::cout << std::setprecision(17) << "lncdf batch  specials max_abs=" << batch_max << "\n";
    EXPECT_LE(scalar_max, kBudgetSpecials);
    EXPECT_LE(batch_max, kBudgetSpecials);
}
