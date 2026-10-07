// tests/test_parallel_batch_gates.cpp
//
// Forced PARALLEL must beat forced VECTORIZED at n = 1M for the operations whose parallel path
// did not run in parallel:
//   - #175: the PARALLEL and WORK_STEALING lambdas of Binomial, NegativeBinomial and von Mises
//     called the scalar method per element, and each call took withCacheSnapshot's shared_lock
//     on the object's one mutex. Bernoulli and Geometric delegate to the first two. At 2M
//     elements PARALLEL ran 1.2-12x slower than single-threaded VECTORIZED on all three
//     machines (2026-10-06 strategy_profile captures).
//   - #176: the Student-t CDF's PARALLEL lambda was a serial loop (exactly 1.00x).
// Each case costs 10-300 ns per element, so with four or more hardware threads a working
// parallel path wins by far more than the 0.75 asked here. Timing-labelled: run on a quiet
// machine.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <gtest/gtest.h>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace stats;

namespace {

constexpr std::size_t kN = 1000000;
constexpr int kReps = 5;
constexpr double kMaxRatio = 0.75;

using Strategy = detail::PerformanceHint::PreferredStrategy;
using BatchFn =
    std::function<void(std::span<const double>, std::span<double>, const detail::PerformanceHint&)>;

// Minimum wall time of kReps calls, after one warm-up call.
double minSeconds(const BatchFn& fn, const std::vector<double>& xs, Strategy s) {
    std::vector<double> out(xs.size());
    const detail::PerformanceHint hint{s, std::nullopt};
    fn(std::span<const double>(xs), std::span<double>(out), hint);
    double best = 1e300;
    for (int r = 0; r < kReps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        fn(std::span<const double>(xs), std::span<double>(out), hint);
        best = std::min(
            best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    return best;
}

std::vector<double> counts(int max) {
    std::vector<double> xs(kN);
    for (std::size_t i = 0; i < kN; ++i)
        xs[i] = static_cast<double>(i % static_cast<std::size_t>(max + 1));
    return xs;
}

std::vector<double> reals(double lo, double hi) {
    std::vector<double> xs(kN);
    for (std::size_t i = 0; i < kN; ++i)
        xs[i] = lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(kN - 1);
    return xs;
}

void expectParallelWins(const std::string& what, const BatchFn& fn, const std::vector<double>& xs) {
    const double vec = minSeconds(fn, xs, Strategy::FORCE_VECTORIZED);
    const double par = minSeconds(fn, xs, Strategy::FORCE_PARALLEL);
    std::cout << "  " << what << ": PARALLEL/VECTORIZED " << par / vec << " (" << par * 1e3
              << " / " << vec * 1e3 << " ms)\n";
    EXPECT_LT(par, kMaxRatio * vec) << what << ": PARALLEL " << par * 1e3 << " ms, VECTORIZED "
                                    << vec * 1e3 << " ms (ratio " << par / vec << ")";
}

}  // namespace

TEST(ParallelBatchGates, ParallelBeatsVectorizedAtOneMillion) {
    if (std::thread::hardware_concurrency() < 4)
        GTEST_SKIP() << "needs at least 4 hardware threads";

    const auto bern = BernoulliDistribution::create(0.3).unwrap();
    const auto binom = BinomialDistribution::create(100, 0.3).unwrap();
    const auto geom = GeometricDistribution::create(0.3).unwrap();
    const auto negbin = NegativeBinomialDistribution::create(5.0, 0.3).unwrap();
    const auto vm = VonMisesDistribution::create(0.0, 2.0).unwrap();
    const auto t = StudentTDistribution::create(5.0).unwrap();

    const auto c1 = counts(1), c100 = counts(100), c30 = counts(30);
    const auto angles = reals(-3.0, 3.0), ts = reals(-6.0, 6.0);

    // #175
    expectParallelWins(
        "Bernoulli(0.3) pmf", [&](auto v, auto r, const auto& h) { bern.getProbability(v, r, h); },
        c1);
    expectParallelWins(
        "Binomial(100, 0.3) pmf",
        [&](auto v, auto r, const auto& h) { binom.getProbability(v, r, h); }, c100);
    expectParallelWins(
        "Binomial(100, 0.3) cdf",
        [&](auto v, auto r, const auto& h) { binom.getCumulativeProbability(v, r, h); }, c100);
    expectParallelWins(
        "Geometric(0.3) pmf", [&](auto v, auto r, const auto& h) { geom.getProbability(v, r, h); },
        c30);
    expectParallelWins(
        "NegativeBinomial(5, 0.3) logpmf",
        [&](auto v, auto r, const auto& h) { negbin.getLogProbability(v, r, h); }, c30);
    expectParallelWins(
        "NegativeBinomial(5, 0.3) cdf",
        [&](auto v, auto r, const auto& h) { negbin.getCumulativeProbability(v, r, h); }, c30);
    expectParallelWins(
        "VonMises(0, 2) cdf",
        [&](auto v, auto r, const auto& h) { vm.getCumulativeProbability(v, r, h); }, angles);
    // #176
    expectParallelWins(
        "StudentT(5) cdf",
        [&](auto v, auto r, const auto& h) { t.getCumulativeProbability(v, r, h); }, ts);
}
