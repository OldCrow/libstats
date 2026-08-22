// tests/test_gaussian_cdf_accuracy.cpp
//
// Relative-error accuracy gate for GaussianDistribution::getCumulativeProbability
// (the Gaussian instance of issue #49's bug pattern). Ground truth comes from
// tests/gaussian_cdf_vectors.inc: F(x; mean, sigma) evaluated as
// erfc(-z/sqrt(2))/2, z=(x-mean)/sigma, at mpmath dps=40 from the exact
// double inputs (scripts/gen_gaussian_cdf_vectors.py) -- a budget miss here
// is a kernel bug for the orchestrator to fix, not a test-tuning problem; do
// not loosen a budget below without a matching implementation fix.
//
// Structure mirrors tests/test_lognormal_cdf_accuracy.cpp (issue #49):
// checked-in generated reference vectors, a self-checking generator,
// per-bucket max relative error printed for the orchestrator to record, and
// gates demonstrated to fail against the pre-fix implementation before being
// trusted (see the landing commit's message for this file's fail-first
// numbers, and PLAN.md).
//
// Diagnosis: GaussianDistribution never routed through detail::normal_cdf
// (tail-branched for #49 in src/math_utils.cpp) -- every one of its own CDF
// paths (scalar, the three parallel lambdas, and the SIMD batch impl)
// independently computed the cancellation form 0.5*(1+erf(z/sqrt(2))). For
// z<0 that has an absolute error floor of ~1.1e-16 regardless of erf's own
// quality, and for z <~ -8.3, std::erf returns exactly -1 so F collapses to
// exactly 0 -- true max relative error there is 1.0. The fix (all CDF sites
// in src/gaussian.cpp) branches on the sign of z and uses erfc for the left
// tail, matching the RELATIVE-error budget this gate enforces.
//
// Budgets PINNED from measurement on Zen 4 (MSVC Release, AVX-512),
// 2026-08-20, against the fixed implementation: scalar and batch max
// relative error 1.55e-13 across every (mean, sigma) bucket where
// F >= 1e-290, worst case in the deep tail near z ~ -35.5 (F ~ 3e-276),
// exactly where the |ln F|*2^-52 law predicts it; worst law-budget
// fraction 0.287, i.e. >3.4x headroom. Below the F floor, F itself is
// dominated by erfc's own underflow behavior and relative error is not a
// meaningful metric -- see the F-floor gating below. Do not loosen without
// a matching kernel fix; a budget miss here is a kernel bug, not a
// test-tuning problem.

#include "libstats/distributions/gaussian.h"

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

// {x_bits, mean_bits, sigma_bits, F_bits}; F evaluated at 40-digit precision
// (see file banner). Defines struct GaussCdfVector, kGaussCdfVectors[] (main
// gate, bucketed by mean_bits/sigma_bits; the (0,1) bucket covers the
// dedicated isStandardNormal_ code path) and kGaussCdfSpecials[]
// (-inf/NaN/+inf, gated separately). See scripts/gen_gaussian_cdf_vectors.py.
#include "gaussian_cdf_vectors.inc"

double bitsToF64(std::uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

// -------------------------------------------------------------------------
// Budgets. PINNED from measurement -- see file banner.
// -------------------------------------------------------------------------
// Relative-error budget is a LAW OF F, not a constant -- same law as the
// lognormal gate (#49), because the mechanism is identical: an ~ulp
// absolute perturbation of the erfc argument w (here from the (x-mean)/sigma
// subtract/divide, or x*INV_SQRT_2 for the standard-normal path) is
// amplified by erfc's asymptotic slope d ln F / dw = -2w, and since
// w^2 ~ -ln F in the tail, the achievable double-precision relative error is
//     rel(F) ~ |ln F| * 2^-52.
// A flat budget deep into the tail is therefore unachievable by ANY
// implementation of this formulation in double (see the lognormal gate's
// banner for the measured confirmation of the law). The budget below is the
// law with headroom:  1e-15 * max(20, -ln F_ref).
inline double law_budget(double f_ref) {
    return 1e-15 * std::max(20.0, -std::log(f_ref));
}
// Below this reference-F floor, the oracle's own double-precision inputs
// (and the tiny reference magnitude itself) make relative error a poor
// metric -- fall back to an absolute budget there instead.
constexpr double kFFloor = 1e-290;
constexpr double kAbsBudgetBelowFloor = 1e-290;
constexpr double kBudgetSpecials = 0.0;  // -inf/NaN/+inf: exact per contract
// Batch (span overload, FORCE_VECTORIZED) is allowed to differ from the
// scalar path only in the -1 <= w < 0 band (F in [0.079, 0.5)), where batch
// still uses the plain erf form while scalar uses erfc -- absolute
// <= ~1.1e-16, relative <= ~1.4e-15. Everywhere else the two paths are
// bit-identical by construction: the batch tail fixup (w < -1) recomputes w
// with the scalar path's exact expression, and for w >= 0 both compute the
// same plain form (vector_erf falls back to std::erf per element on tiers
// without a vectorized erf kernel; on tiers with one, this bound absorbs
// its documented ulp-level difference). Measured on Zen 4 AVX-512,
// 2026-08-20: max abs 1.11e-16 (exactly the predicted in-band ulp(1)/2),
// max law-normalized rel 9.0e-3.
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

TEST(GaussianCdfGates, ReferenceVectorsWellFormed) {
    constexpr std::size_t n = sizeof(kGaussCdfVectors) / sizeof(kGaussCdfVectors[0]);
    ASSERT_GT(n, 0u);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = bitsToF64(kGaussCdfVectors[i].x_bits);
        const double mean = bitsToF64(kGaussCdfVectors[i].mean_bits);
        const double sigma = bitsToF64(kGaussCdfVectors[i].sigma_bits);
        const double F = bitsToF64(kGaussCdfVectors[i].F_bits);
        ASSERT_TRUE(std::isfinite(x)) << "row " << i << ": x not finite";
        ASSERT_TRUE(std::isfinite(mean)) << "row " << i << ": mean not finite";
        ASSERT_GT(sigma, 0.0) << "row " << i << ": sigma not positive";
        ASSERT_GE(F, 0.0) << "row " << i << ": F below 0";
        ASSERT_LE(F, 1.0) << "row " << i << ": F above 1";
    }
}

// -------------------------------------------------------------------------
// Main gate: per-(mean,sigma)-bucket scalar and batch evaluation, RELATIVE
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

// Runs one (mean,sigma) bucket (contiguous run of kGaussCdfVectors sharing
// the same mean_bits/sigma_bits, per the generator's emission order) through
// both the scalar getCumulativeProbability(x) and the batch span overload
// (one batch call per bucket), gates each against the relative-error budget,
// and cross-checks batch against scalar. Prints machine-readable one-liners
// for the orchestrator to record.
BucketResult run_bucket(double mean, double sigma, const GaussCdfVector* rows, std::size_t n) {
    auto dist_result = stats::GaussianDistribution::create(mean, sigma);
    if (dist_result.isError()) {
        ADD_FAILURE() << "GaussianDistribution::create(mean=" << mean << ", sigma=" << sigma
                      << ") rejected -- expected every (mean, sigma) in the coverage sweep "
                         "to be accepted; error: "
                      << dist_result.message();
        return {};
    }
    const stats::GaussianDistribution dist = std::move(dist_result).unwrap();

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
    // erf/erfc tail fixup -- the very code this gate exists to cover.
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

    std::cout << std::setprecision(17) << "gausscdf scalar mean=" << mean << " sigma=" << sigma
              << " max_rel=" << r.scalar_max_rel << " worst_x=" << r.scalar_worst_x
              << " max_budget_frac=" << r.scalar_max_frac << "\n";
    std::cout << std::setprecision(17) << "gausscdf batch  mean=" << mean << " sigma=" << sigma
              << " max_rel=" << r.batch_max_rel << " worst_x=" << r.batch_worst_x
              << " max_budget_frac=" << r.batch_max_frac << "\n";
    std::cout << std::setprecision(17) << "gausscdf batch_vs_scalar mean=" << mean
              << " sigma=" << sigma << " max_law_frac=" << r.batch_vs_scalar_max_rel
              << " max_abs=" << r.batch_vs_scalar_max_abs << "\n";
    return r;
}

// Groups kGaussCdfVectors into contiguous per-(mean,sigma) runs (generator
// emission order) without assuming a fixed bucket size.
std::vector<std::pair<std::size_t, std::size_t>> find_buckets(const GaussCdfVector* rows,
                                                              std::size_t n) {
    std::vector<std::pair<std::size_t, std::size_t>> buckets;  // (start, count)
    std::size_t start = 0;
    for (std::size_t i = 1; i <= n; ++i) {
        if (i == n || rows[i].mean_bits != rows[start].mean_bits ||
            rows[i].sigma_bits != rows[start].sigma_bits) {
            buckets.emplace_back(start, i - start);
            start = i;
        }
    }
    return buckets;
}

}  // namespace

TEST(GaussianCdfGates, MainSweepPerBucket) {
    constexpr std::size_t n = sizeof(kGaussCdfVectors) / sizeof(kGaussCdfVectors[0]);
    const auto buckets = find_buckets(kGaussCdfVectors, n);
    ASSERT_GT(buckets.size(), 0u);

    for (const auto& [start, count] : buckets) {
        const double mean = bitsToF64(kGaussCdfVectors[start].mean_bits);
        const double sigma = bitsToF64(kGaussCdfVectors[start].sigma_bits);
        const BucketResult r = run_bucket(mean, sigma, &kGaussCdfVectors[start], count);

        EXPECT_LE(r.scalar_max_frac, 1.0)
            << "scalar mean=" << mean << " sigma=" << sigma
            << " over the law_budget(F) relative-error budget (worst x=" << r.scalar_frac_worst_x
            << ")";
        EXPECT_LE(r.batch_max_frac, 1.0)
            << "batch mean=" << mean << " sigma=" << sigma
            << " over the law_budget(F) relative-error budget (worst x=" << r.batch_frac_worst_x
            << ")";
        EXPECT_LE(r.batch_vs_scalar_max_rel, 1.0)
            << "batch vs scalar law-normalized relative mismatch mean=" << mean
            << " sigma=" << sigma;
        EXPECT_LE(r.batch_vs_scalar_max_abs, kBudgetBatchVsScalarAbs)
            << "batch vs scalar absolute mismatch mean=" << mean << " sigma=" << sigma;
    }
}

// -------------------------------------------------------------------------
// Specials gate: -inf -> F=0, +inf -> F=1, NaN -> NaN, exactly (both scalar
// and batch paths). The Gaussian support is the whole real line, so unlike
// the lognormal gate there is no outside-support finite x.
// -------------------------------------------------------------------------

TEST(GaussianCdfGates, Specials) {
    constexpr std::size_t n = sizeof(kGaussCdfSpecials) / sizeof(kGaussCdfSpecials[0]);
    ASSERT_EQ(n, 3u) << "specials set shape changed -- update this test's expectations";

    const double mean = bitsToF64(kGaussCdfSpecials[0].mean_bits);
    const double sigma = bitsToF64(kGaussCdfSpecials[0].sigma_bits);
    auto dist_result = stats::GaussianDistribution::create(mean, sigma);
    ASSERT_TRUE(dist_result.isOk()) << dist_result.message();
    const stats::GaussianDistribution dist = std::move(dist_result).unwrap();

    std::vector<double> xs(n), refs(n), scalar_out(n), batch_out(n);
    for (std::size_t i = 0; i < n; ++i) {
        xs[i] = bitsToF64(kGaussCdfSpecials[i].x_bits);
        refs[i] = bitsToF64(kGaussCdfSpecials[i].F_bits);
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
            EXPECT_EQ(scalar_out[i], refs[i]) << "scalar: F(x=" << xs[i] << ") must equal "
                                              << refs[i] << " exactly, got " << scalar_out[i];
            EXPECT_EQ(batch_out[i], refs[i]) << "batch: F(x=" << xs[i] << ") must equal "
                                             << refs[i] << " exactly, got " << batch_out[i];
        }
    }
    std::cout << std::setprecision(17) << "gausscdf scalar specials max_abs=" << scalar_max << "\n";
    std::cout << std::setprecision(17) << "gausscdf batch  specials max_abs=" << batch_max << "\n";
    EXPECT_LE(scalar_max, kBudgetSpecials);
    EXPECT_LE(batch_max, kBudgetSpecials);
}

// ---------------------------------------------------------------------------
// Review 2026-08-21 (N7): the standard-normal fast path must be selected only
// for exactly (0, 1). A tolerant predicate (|μ| ≤ 1e-8) returned cdf(0) = 0.5
// for Gaussian(5e-9, 1) and pdf(0) = φ(0) for σ = 1 + 5e-9 — constant in the
// parameter over the snap band and discontinuous at its edge. Reference
// values: mpmath, 40 digits. Both assertions fail against the tolerant
// predicate.
// ---------------------------------------------------------------------------
TEST(GaussianCdfGates, NoStandardNormalSnapNearZeroMean) {
    auto g = stats::GaussianDistribution::create(5e-9, 1.0);
    ASSERT_TRUE(g.isOk());
    EXPECT_NEAR(g->getCumulativeProbability(0.0), 0.4999999980052886, 1e-16);

    auto h = stats::GaussianDistribution::create(0.0, 1.0 + 5e-9);
    ASSERT_TRUE(h.isOk());
    EXPECT_NEAR(h->getProbability(0.0), 0.39894227840672129, 1e-16);

    // And exactly (0, 1) still takes the fast path to the same answers.
    auto z = stats::GaussianDistribution::create(0.0, 1.0);
    ASSERT_TRUE(z.isOk());
    EXPECT_DOUBLE_EQ(z->getCumulativeProbability(0.0), 0.5);
}
