// gamma_probe: why does table_ab_bench read Gamma pdf/logpdf under AUTO at 8192 at 2x the forced
// VECTORIZED time in three of four passes? The dispatcher is a pure table lookup, so AUTO at 8192
// (< the 25k PDF/LogPDF rows) must resolve to VECTORIZED. This probe (a) prints what selectStrategy
// returns per size, (b) times AUTO, FORCE_VECTORIZED and FORCE_PARALLEL at sizes around the fork
// threshold under three measurement orders, repeated, with the bench's timing helper, and (c) times
// AUTO immediately after a 500k-element PARALLEL call (thread pool just used) versus after a 300 ms
// pause.
#include "libstats/core/distribution_type.h"
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/performance_dispatcher.h"
#include "libstats/distributions/gamma.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <optional>
#include <random>
#include <span>
#include <thread>
#include <vector>

using clk = std::chrono::steady_clock;
using Strategy = stats::detail::PerformanceHint::PreferredStrategy;
volatile double sink = 0.0;
constexpr std::size_t kMinBlock = 200'000;

template <class F>
double nsPer(std::size_t n, F&& f) {
    const std::size_t reps = std::max<std::size_t>(1, (kMinBlock + n - 1) / n);
    double best = 1e300;
    for (int r = 0; r < 5; ++r) {
        const auto t0 = clk::now();
        for (std::size_t i = 0; i < reps; ++i)
            f();
        const auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / (reps * n));
    }
    return best;
}

const char* name(stats::detail::Strategy s) {
    switch (s) {
        case stats::detail::Strategy::SCALAR: return "SCALAR";
        case stats::detail::Strategy::VECTORIZED: return "VECTORIZED";
        case stats::detail::Strategy::PARALLEL: return "PARALLEL";
        case stats::detail::Strategy::WORK_STEALING: return "WORK_STEALING";
    }
    return "?";
}

int main() {
    const auto d = stats::GammaDistribution::create(2.0, 1.0).unwrap();
    std::vector<double> x(500'000), y(500'000);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(0.05, 8.0);
    for (auto& v : x)
        v = u(rng);

    std::printf("== selectStrategy (GAMMA)\n");
    stats::detail::PerformanceDispatcher disp;
    const auto& sys = stats::detail::SystemCapabilities::current();
    for (std::size_t n : {4096u, 6144u, 8192u, 10000u, 12288u, 16384u, 25000u, 50000u})
        std::printf("  n=%6zu pdf=%s logpdf=%s cdf=%s\n", n,
                    name(disp.selectStrategy(n, stats::detail::DistributionType::GAMMA, stats::detail::OperationType::PDF, sys)),
                    name(disp.selectStrategy(n, stats::detail::DistributionType::GAMMA, stats::detail::OperationType::LOG_PDF, sys)),
                    name(disp.selectStrategy(n, stats::detail::DistributionType::GAMMA, stats::detail::OperationType::CDF, sys)));

    struct Forced { const char* tag; Strategy s; };
    const Forced A{"auto", Strategy::AUTO}, V{"vect", Strategy::FORCE_VECTORIZED}, P{"para", Strategy::FORCE_PARALLEL};
    const std::vector<std::pair<const char*, std::vector<Forced>>> orders = {
        {"AVP", {A, V, P}}, {"VAP", {V, A, P}}, {"PVA", {P, V, A}}};
    std::printf("== timings: order,rep,n,strategy,ns_per_element (pdf)\n");
    for (int rep = 1; rep <= 3; ++rep)
        for (const auto& [otag, order] : orders)
            for (std::size_t n : {4096u, 6144u, 8192u, 10000u, 12288u, 16384u, 25000u}) {
                const std::span<const double> xs(x.data(), n);
                const std::span<double> ys(y.data(), n);
                for (const Forced& f : order) {
                    const stats::detail::PerformanceHint hint{f.s, std::nullopt};
                    const double ns = nsPer(n, [&] { d.getProbability(xs, ys, hint); sink = sink + y[n / 2]; });
                    std::printf("%s,%d,%zu,%s,%.3f\n", otag, rep, n, f.tag, ns);
                }
            }

    std::printf("== AUTO at 8192 right after a 500k PARALLEL call vs after a 300 ms pause (pdf), 6 trials each\n");
    const std::span<const double> xs8(x.data(), 8192);
    const std::span<double> ys8(y.data(), 8192);
    const std::span<const double> xsBig(x.data(), 500'000);
    const std::span<double> ysBig(y.data(), 500'000);
    const stats::detail::PerformanceHint hA{Strategy::AUTO, std::nullopt}, hP{Strategy::FORCE_PARALLEL, std::nullopt},
        hV{Strategy::FORCE_VECTORIZED, std::nullopt};
    for (int t = 0; t < 6; ++t) {
        d.getProbability(xsBig, ysBig, hP);
        const double afterP = nsPer(8192, [&] { d.getProbability(xs8, ys8, hA); sink = sink + y[1]; });
        d.getProbability(xsBig, ysBig, hP);
        const double afterPV = nsPer(8192, [&] { d.getProbability(xs8, ys8, hV); sink = sink + y[1]; });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const double afterPause = nsPer(8192, [&] { d.getProbability(xs8, ys8, hA); sink = sink + y[1]; });
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const double afterPauseV = nsPer(8192, [&] { d.getProbability(xs8, ys8, hV); sink = sink + y[1]; });
        std::printf("  trial %d: auto after PARALLEL %.3f  vect after PARALLEL %.3f  auto after pause %.3f  vect after pause %.3f\n",
                    t + 1, afterP, afterPV, afterPause, afterPauseV);
    }
    return 0;
}
