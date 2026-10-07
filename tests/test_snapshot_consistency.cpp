// tests/test_snapshot_consistency.cpp
//
// A scalar method racing a setter must return the value of one parameter state or the other,
// never a mix (v2.4.2 review, H-1). Before the fix:
//   - Binomial pmf and log-pmf tested p_ == 0 unlocked, then took their snapshot; a setP(0) in
//     between sent p = 0 to detail::binomial_log_pmf, whose Stirling path returned NaN (#172
//     turned the old −inf into NaN). The CDF range-checked k against an unlocked n_ and handed
//     beta_i a negative shape once setN shrank n.
//   - Poisson's pmf took λ and e^{−λ} from the members after the snapshot, so a setLambda in
//     between paired the new λ with the stale e^{−λ}.
//   - Gamma's quantile read α and β unlocked, one at a time.
//
// A race shows only by chance. In each test a writer flips the parameters between two states
// while the reader makes kReads calls and compares every result bit for bit with the two states'
// own values. On the unfixed code every check failed in each of five runs, with 9 to 1514 of
// the 50000 reads matching neither state (Zen 4, MSVC).

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <atomic>
#include <cmath>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <thread>
#include <vector>

using namespace stats;

namespace {

constexpr long kReads = 50000;

template <typename Dist, typename Flip, typename Read>
void expectOneOfTwoStates(const char* what, Dist& d, Flip flip, Read read, double want_a,
                          double want_b) {
    std::atomic<bool> stop{false};
    std::thread writer([&] {
        for (long i = 0; !stop.load(); ++i)
            flip(d, i % 2 == 0);
    });
    long mixed = 0;
    double first_mixed = 0.0;
    for (long i = 0; i < kReads; ++i) {
        const double v = read(d);
        if (!(v == want_a || v == want_b) && mixed++ == 0)
            first_mixed = v;
    }
    stop.store(true);
    writer.join();
    EXPECT_EQ(mixed, 0) << what << ": " << mixed << " of " << kReads
                        << " reads matched neither state, first " << first_mixed << " (want "
                        << want_a << " or " << want_b << ")";
}

// A batch takes one parameter snapshot, so every element of one call comes from the same state
// (#175). The PARALLEL and WORK_STEALING lambdas of Binomial, NegativeBinomial and von Mises,
// and the VECTORIZED CDF of the first two, called the scalar method per element, each with its
// own snapshot: under a concurrent setter one call returned a mix of both states, and every
// element took the shared lock.
constexpr std::size_t kBatchN = 50000;
constexpr int kBatches = 60;

// state_a and state_b are fixed instances of the two states the writer flips between; the wanted
// values come from the same batch path on them, so only consistency is under test.
template <typename Dist, typename Flip, typename Batch>
void expectBatchFromOneState(const char* what, Dist& d, Flip flip, Batch batch, double x,
                             const Dist& state_a, const Dist& state_b) {
    using Strategy = detail::PerformanceHint::PreferredStrategy;
    // MAXIMIZE_THROUGHPUT reaches the WORK_STEALING lambda where the platform prefers it (macOS);
    // there is no FORCE_ hint for it.
    for (Strategy s :
         {Strategy::FORCE_VECTORIZED, Strategy::FORCE_PARALLEL, Strategy::MAXIMIZE_THROUGHPUT}) {
        const detail::PerformanceHint hint{s, std::nullopt};
        const std::vector<double> xs(kBatchN, x);
        std::vector<double> out(kBatchN);
        batch(state_a, std::span<const double>(xs), std::span<double>(out), hint);
        const double want_a = out[0];
        batch(state_b, std::span<const double>(xs), std::span<double>(out), hint);
        const double want_b = out[0];
        ASSERT_NE(want_a, want_b) << what << ": the two states must differ at x = " << x;
        std::atomic<bool> stop{false};
        std::thread writer([&] {
            for (long i = 0; !stop.load(); ++i)
                flip(d, i % 2 == 0);
        });
        int mixed_batches = 0, foreign = 0;
        for (int b = 0; b < kBatches; ++b) {
            batch(d, std::span<const double>(xs), std::span<double>(out), hint);
            std::size_t a = 0, bb = 0;
            for (const double v : out) {
                a += (v == want_a);
                bb += (v == want_b);
            }
            foreign += static_cast<int>(kBatchN - a - bb);
            mixed_batches += (a != kBatchN && bb != kBatchN);
        }
        stop.store(true);
        writer.join();
        EXPECT_EQ(mixed_batches, 0)
            << what << " strategy " << static_cast<int>(s) << ": " << mixed_batches << " of "
            << kBatches << " batches mixed the two states (" << foreign
            << " elements matched neither)";
    }
}

}  // namespace

TEST(SnapshotConsistency, BinomialPmfUnderSetP) {
    const auto at_half = BinomialDistribution::create(100, 0.5).unwrap();
    const auto at_zero = BinomialDistribution::create(100, 0.0).unwrap();
    auto d = BinomialDistribution::create(100, 0.5).unwrap();
    const auto flip = [](BinomialDistribution& b, bool zero) { b.setP(zero ? 0.0 : 0.5); };
    expectOneOfTwoStates(
        "Binomial(100, ·) logpmf(50)", d, flip,
        [](const BinomialDistribution& b) { return b.getLogProbability(50.0); },
        at_half.getLogProbability(50.0), at_zero.getLogProbability(50.0));
    expectOneOfTwoStates(
        "Binomial(100, ·) pmf(50)", d, flip,
        [](const BinomialDistribution& b) { return b.getProbability(50.0); },
        at_half.getProbability(50.0), at_zero.getProbability(50.0));
}

TEST(SnapshotConsistency, BinomialCdfUnderSetN) {
    const auto n100 = BinomialDistribution::create(100, 0.5).unwrap();
    const auto n50 = BinomialDistribution::create(50, 0.5).unwrap();
    auto d = BinomialDistribution::create(100, 0.5).unwrap();
    expectOneOfTwoStates(
        "Binomial(·, 0.5) cdf(70)", d,
        [](BinomialDistribution& b, bool shrink) { b.setN(shrink ? 50 : 100); },
        [](const BinomialDistribution& b) { return b.getCumulativeProbability(70.0); },
        n100.getCumulativeProbability(70.0), n50.getCumulativeProbability(70.0));
}

TEST(SnapshotConsistency, PoissonPmfUnderSetLambda) {
    const auto two = PoissonDistribution::create(2.0).unwrap();
    const auto three = PoissonDistribution::create(3.0).unwrap();
    auto d = PoissonDistribution::create(2.0).unwrap();
    expectOneOfTwoStates(
        "Poisson(·) pmf(4)", d,
        [](PoissonDistribution& p, bool second) { p.setLambda(second ? 3.0 : 2.0); },
        [](const PoissonDistribution& p) { return p.getProbability(4.0); }, two.getProbability(4.0),
        three.getProbability(4.0));
}

TEST(SnapshotConsistency, GammaQuantileUnderSetParameters) {
    const auto first = GammaDistribution::create(2.0, 1.0).unwrap();
    const auto second = GammaDistribution::create(5.0, 3.0).unwrap();
    auto d = GammaDistribution::create(2.0, 1.0).unwrap();
    expectOneOfTwoStates(
        "Gamma(·, ·) quantile(0.3)", d,
        [](GammaDistribution& g, bool other) {
            if (other)
                g.setParameters(5.0, 3.0);
            else
                g.setParameters(2.0, 1.0);
        },
        [](const GammaDistribution& g) { return g.getQuantile(0.3); }, first.getQuantile(0.3),
        second.getQuantile(0.3));
}

// #175: one batch call, one parameter state, under every batch strategy.
using Hint = detail::PerformanceHint;

TEST(SnapshotConsistency, BinomialBatchesUnderSetP) {
    const auto a = BinomialDistribution::create(100, 0.3).unwrap();
    const auto b = BinomialDistribution::create(100, 0.6).unwrap();
    auto d = BinomialDistribution::create(100, 0.3).unwrap();
    const auto flip = [](BinomialDistribution& x, bool second) { x.setP(second ? 0.6 : 0.3); };
    expectBatchFromOneState(
        "Binomial(100, ·) pmf(40)", d, flip,
        [](const BinomialDistribution& x, std::span<const double> v, std::span<double> r,
           const Hint& h) { x.getProbability(v, r, h); },
        40.0, a, b);
    expectBatchFromOneState(
        "Binomial(100, ·) cdf(45)", d, flip,
        [](const BinomialDistribution& x, std::span<const double> v, std::span<double> r,
           const Hint& h) { x.getCumulativeProbability(v, r, h); },
        45.0, a, b);
}

TEST(SnapshotConsistency, NegativeBinomialBatchesUnderSetP) {
    const auto a = NegativeBinomialDistribution::create(5.0, 0.3).unwrap();
    const auto b = NegativeBinomialDistribution::create(5.0, 0.6).unwrap();
    auto d = NegativeBinomialDistribution::create(5.0, 0.3).unwrap();
    const auto flip = [](NegativeBinomialDistribution& x, bool second) {
        x.setP(second ? 0.6 : 0.3);
    };
    expectBatchFromOneState(
        "NegativeBinomial(5, ·) logpmf(8)", d, flip,
        [](const NegativeBinomialDistribution& x, std::span<const double> v, std::span<double> r,
           const Hint& h) { x.getLogProbability(v, r, h); },
        8.0, a, b);
    expectBatchFromOneState(
        "NegativeBinomial(5, ·) cdf(8)", d, flip,
        [](const NegativeBinomialDistribution& x, std::span<const double> v, std::span<double> r,
           const Hint& h) { x.getCumulativeProbability(v, r, h); },
        8.0, a, b);
}

TEST(SnapshotConsistency, VonMisesCdfBatchUnderSetKappa) {
    const auto a = VonMisesDistribution::create(0.0, 2.0).unwrap();
    const auto b = VonMisesDistribution::create(0.0, 5.0).unwrap();
    auto d = VonMisesDistribution::create(0.0, 2.0).unwrap();
    expectBatchFromOneState(
        "VonMises(0, ·) cdf(0.7)", d,
        [](VonMisesDistribution& x, bool second) { x.setKappa(second ? 5.0 : 2.0); },
        [](const VonMisesDistribution& x, std::span<const double> v, std::span<double> r,
           const Hint& h) { x.getCumulativeProbability(v, r, h); },
        0.7, a, b);
}

// Beta's batch kernels took the boundary values (x <= 0, x >= 1) from the scalar methods, one
// lock and one read of the live shapes per element, not from the batch's snapshot.
TEST(SnapshotConsistency, BetaBoundaryBatchUnderSetAlpha) {
    const auto a = BetaDistribution::create(0.5, 2.0).unwrap();  // pdf(0) = +inf
    const auto b = BetaDistribution::create(2.0, 2.0).unwrap();  // pdf(0) = 0
    auto d = BetaDistribution::create(0.5, 2.0).unwrap();
    expectBatchFromOneState(
        "Beta(·, 2) pdf(0)", d,
        [](BetaDistribution& x, bool second) { x.setAlpha(second ? 2.0 : 0.5); },
        [](const BetaDistribution& x, std::span<const double> v, std::span<double> r,
           const Hint& h) { x.getProbability(v, r, h); },
        0.0, a, b);
}
