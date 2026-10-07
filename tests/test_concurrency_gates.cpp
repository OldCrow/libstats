// tests/test_concurrency_gates.cpp
//
// Gates for the 2026-10-06 audit of locks and parallel paths (after #175, #176). Each was shown
// to fail on the code before its fix, except where noted:
//   - Poisson::getMedian took the shared lock and called getQuantile, which takes it again: a
//     recursive shared lock, undefined, and a deadlock with SRWLOCK once a writer queues between
//     the two acquisitions.
//   - Uniform's and Discrete's single-bound setters validated against a copy of the other bound
//     taken under a lock they then released (Discrete took none), so two concurrent setters could
//     leave a >= b.
//   - Bernoulli and Geometric updated their delegate after releasing their own lock, so two
//     concurrent setters could leave the delegate, which the pmf reads, on the other value.
//   - ParallelUtils::parallelTransform called func on a whole chunk from every index.
//   - Below ParallelUtils' fork threshold, a PARALLEL batch ran most distributions' per-element
//     scalar loop instead of the batch path, so its results were the scalar method's.
//   - d == d locked the same shared_mutex twice in 14 distributions. Undefined, but no platform
//     here fails observably, so that test pins the result and cannot fail first.
// Two fixes have no deterministic gate here: Uniform's copy-assignment, which took no locks (a
// torn copy needs the writer between two adjacent loads), and the work-stealing pool's
// construction, which started threads before their data was written. ThreadSanitizer on the Macs
// or Linux shows both.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <thread>
#include <vector>

using namespace stats;

namespace {

// Runs body on its own thread; a body still running after `seconds` is reported and the process
// exits, since a deadlocked thread can be neither joined nor detached safely.
template <typename Body>
void runWithWatchdog(const char* what, int seconds, Body body) {
    std::atomic<bool> done{false};
    std::thread t([&] {
        body();
        done.store(true);
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    while (!done.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (!done.load()) {
        std::fprintf(stderr, "[  FAILED  ] %s: no progress in %d s (deadlock)\n", what, seconds);
        std::fflush(stderr);
        std::_Exit(1);
    }
    t.join();
}

}  // namespace

TEST(ConcurrencyGates, PoissonMedianUnderSetLambda) {
    auto d = PoissonDistribution::create(3.0).unwrap();
    std::atomic<bool> stop{false};
    std::thread writer([&] {
        for (long i = 0; !stop.load(); ++i)
            d.setLambda(i % 2 == 0 ? 4.0 : 3.0);
    });
    runWithWatchdog("Poisson getMedian under setLambda", 60, [&] {
        for (int i = 0; i < 200000; ++i) {
            const double m = d.getMedian();
            ASSERT_TRUE(m == 3.0 || m == 4.0) << "median " << m;
        }
    });
    stop.store(true);
    writer.join();
}

namespace {

// Rounds of two calls started together: setup() between rounds, then first() and second() on two
// threads released at the same instant, then check(). Returns the rounds that check() rejected. A
// one-shot race needs both calls inside the same few hundred nanoseconds, which a shared start
// makes common and two free-running loops make rare: the loop form of these gates passed on the
// unfixed code.
template <typename Setup, typename First, typename Second, typename Check>
int racedRounds(int rounds, Setup setup, First first, Second second, Check check) {
    std::atomic<int> go{0}, done{0};
    std::atomic<bool> quit{false};
    const auto worker = [&](auto& call) {
        for (int r = 1;; ++r) {
            while (go.load(std::memory_order_acquire) < r)
                if (quit.load(std::memory_order_acquire))
                    return;
            call();
            done.fetch_add(1, std::memory_order_acq_rel);
        }
    };
    std::thread t1([&] { worker(first); });
    std::thread t2([&] { worker(second); });
    int rejected = 0;
    for (int r = 1; r <= rounds; ++r) {
        setup();
        done.store(0, std::memory_order_release);
        go.store(r, std::memory_order_release);
        while (done.load(std::memory_order_acquire) < 2) {
        }
        rejected += !check();
    }
    quit.store(true, std::memory_order_release);
    t1.join();
    t2.join();
    return rejected;
}

constexpr int kRounds = 20000;

}  // namespace

// From (0, 3), one thread sets a = 2 and the other b = 1. Under one lock per setter the result is
// (2, 3) or (0, 1), the other call rejected; validating against a stale copy leaves (2, 1).
TEST(ConcurrencyGates, UniformBoundSettersKeepAOrderedBelowB) {
    auto d = UniformDistribution::create(0.0, 3.0).unwrap();
    const int bad = racedRounds(
        kRounds, [&] { d.setBounds(0.0, 3.0); }, [&] { (void)d.trySetLowerBound(2.0); },
        [&] { (void)d.trySetUpperBound(1.0); },
        [&] { return !d.validateCurrentParameters().isError(); });
    EXPECT_EQ(bad, 0) << bad << " of " << kRounds << " rounds left a >= b";
}

TEST(ConcurrencyGates, DiscreteBoundSettersKeepAOrderedBelowB) {
    auto d = DiscreteDistribution::create(0, 3).unwrap();
    const auto quietly = [](auto call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
        }
    };
    const int bad = racedRounds(
        kRounds, [&] { d.setBounds(0, 3); }, [&] { quietly([&] { d.setLowerBound(2); }); },
        [&] { quietly([&] { d.setUpperBound(1); }); },
        [&] { return !d.validateCurrentParameters().isError(); });
    EXPECT_EQ(bad, 0) << bad << " of " << kRounds << " rounds left a > b";
}

namespace {

// Two threads set p to their own value at once; afterwards the pmf, which reads the delegate,
// must answer for the p that getP() reports.
template <typename Dist>
int disagreeingRounds(double p1, double p2, double x) {
    const auto at1 = Dist::create(p1).unwrap();
    const auto at2 = Dist::create(p2).unwrap();
    auto d = Dist::create(0.5).unwrap();
    return racedRounds(
        kRounds, [&] { d.setP(0.5); }, [&] { d.setP(p1); }, [&] { d.setP(p2); },
        [&] {
            const double p = d.getP();
            const double want = (p == p1 ? at1 : at2).getProbability(x);
            return d.getProbability(x) == want;
        });
}

}  // namespace

TEST(ConcurrencyGates, DelegateFollowsConcurrentSetters) {
    EXPECT_EQ(disagreeingRounds<BernoulliDistribution>(0.2, 0.7, 1.0), 0)
        << "Bernoulli: rounds whose pmf answered for the other p";
    EXPECT_EQ(disagreeingRounds<GeometricDistribution>(0.2, 0.7, 3.0), 0)
        << "Geometric: rounds whose pmf answered for the other p";
}

TEST(ConcurrencyGates, ParallelTransformCallsEachElementOnce) {
    const std::size_t n = 200000;
    std::vector<double> in(n, 1.0), out(n, 0.0);
    std::atomic<std::size_t> processed{0};
    ParallelUtils::parallelTransform(in.data(), out.data(), n,
                                     [&](const double* src, double* dst, std::size_t len) {
                                         for (std::size_t i = 0; i < len; ++i)
                                             dst[i] = src[i] + 1.0;
                                         processed.fetch_add(len);
                                     });
    EXPECT_EQ(processed.load(), n);
    for (std::size_t i = 0; i < n; ++i)
        ASSERT_EQ(out[i], 2.0) << "at " << i;
}

TEST(ConcurrencyGates, SelfComparisonIsTrue) {
    const auto g = GaussianDistribution::create(0.0, 1.0).unwrap();
    const auto b = BinomialDistribution::create(10, 0.3).unwrap();
    const auto u = UniformDistribution::create(0.0, 1.0).unwrap();
    const auto p = PoissonDistribution::create(2.0).unwrap();
    const auto v = VonMisesDistribution::create(0.0, 2.0).unwrap();
    EXPECT_TRUE(g == g);
    EXPECT_TRUE(b == b);
    EXPECT_TRUE(u == u);
    EXPECT_TRUE(p == p);
    EXPECT_TRUE(v == v);
}

TEST(ConcurrencyGates, ParallelBelowForkThresholdTakesTheBatchPath) {
    // One element under the fork threshold, PARALLEL forks nothing; it must give the batch
    // path's results, not the scalar loop's.
    const std::size_t n = arch::get_min_elements_for_parallel() - 1;
    using Strategy = detail::PerformanceHint::PreferredStrategy;
    const detail::PerformanceHint vec{Strategy::FORCE_VECTORIZED, std::nullopt};
    const detail::PerformanceHint par{Strategy::FORCE_PARALLEL, std::nullopt};
    std::vector<double> xs(n);
    for (std::size_t i = 0; i < n; ++i)
        xs[i] = -4.0 + 8.0 * static_cast<double>(i) / static_cast<double>(n);
    std::vector<double> a(n), b(n);
    const auto expectSame = [&](const char* what) {
        EXPECT_EQ(0, std::memcmp(a.data(), b.data(), n * sizeof(double)))
            << what << ": PARALLEL below the fork threshold differs from VECTORIZED";
    };
    const auto g = GaussianDistribution::create(0.0, 1.0).unwrap();
    g.getProbability(std::span<const double>(xs), std::span<double>(a), vec);
    g.getProbability(std::span<const double>(xs), std::span<double>(b), par);
    expectSame("Gaussian pdf");
    g.getCumulativeProbability(std::span<const double>(xs), std::span<double>(a), vec);
    g.getCumulativeProbability(std::span<const double>(xs), std::span<double>(b), par);
    expectSame("Gaussian cdf");
    const auto e = ExponentialDistribution::create(1.5).unwrap();
    e.getProbability(std::span<const double>(xs), std::span<double>(a), vec);
    e.getProbability(std::span<const double>(xs), std::span<double>(b), par);
    expectSame("Exponential pdf");
}
