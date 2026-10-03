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
#include <thread>

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
