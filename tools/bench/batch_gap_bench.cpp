// batch_gap_bench: does PARALLEL still beat VECTORIZED when batches do not arrive back to back?
//
// The dispatch thresholds come from a profiler that calls each batch back to back, so the pool's
// workers never sleep between sections. An application usually does something else between
// batches. On the Ivy Bridge machine under Linux schedutil (2026-11-22) that changes the answer by
// more than an order of magnitude: once the workers have slept, they wake on cores at the lowest
// clock, and forced-PARALLEL Poisson PDF at 10000 elements costs 47-59 ns/element against 36 for
// VECTORIZED, where back to back it costs 17-22. The crossover moves from below 4096 to between
// 100000 and 200000. This program measures that on any machine.
//
// Each cell is one (distribution, size, gap, strategy): a batch call, then `gap` of serial work on
// the calling thread (a clock-reading loop; --sleep makes the caller sleep instead), repeated.
// Only the time inside the batch call is counted. A cell runs its pattern for 0.4 s unmeasured,
// then for at least 1 s and 20 calls; the CSV reports the median call in ns/element.
//
// Pool state matters as much as the gap, and it cannot be undone within a process, so it is a
// command-line choice and each state is its own run:
//   (no option)  only the thread pool exists, created by the first PARALLEL batch
//   --init       stats::initialize_performance_systems() first, as the README advises; on v2.5.x
//                and the v3.0.0 branch this also starts the work-stealing pool, whose idle workers
//                poll and so keep the cores awake
//   --auto       adds an AUTO column (what a caller gets without a hint); where AUTO selects work
//                stealing (hyper-threaded macOS) this starts that pool part-way through the run,
//                so read the forced columns from a run without it
//
// Run all three on a quiet machine, about 3 to 5 minutes each:
//   batch_gap_bench > gap_plain.csv; batch_gap_bench --init > gap_init.csv
//   batch_gap_bench --init --auto > gap_init_auto.csv
// The lines starting with '#' at the end give, per distribution and gap, the smallest measured
// size from which PARALLEL wins at every larger size.
//
// Build: as distributions_bench (tools/bench/README.md). Standard library and the public headers
// only, so it compiles against v2.5.x and against the v3.0.0 branch.
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/poisson.h"
#include "libstats/libstats.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;
using Strategy = stats::detail::PerformanceHint::PreferredStrategy;

constexpr double kWarmSeconds = 0.4;
constexpr double kTimedSeconds = 1.0;
constexpr std::size_t kMinCalls = 20;

double secondsSince(clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
}

/// Serial work (or sleep) on the calling thread for `gap_us` microseconds.
void gap(double gap_us, bool sleep) {
    if (gap_us <= 0.0)
        return;
    if (sleep) {
        std::this_thread::sleep_for(std::chrono::duration<double, std::micro>(gap_us));
        return;
    }
    const auto t0 = clk::now();
    while (std::chrono::duration<double, std::micro>(clk::now() - t0).count() < gap_us) {
    }
}

/// Median time inside `call`, in nanoseconds, with `gap_us` between calls.
template <class F>
double timeCell(F&& call, double gap_us, bool sleep) {
    for (const auto t0 = clk::now(); secondsSince(t0) < kWarmSeconds;) {
        call();
        gap(gap_us, sleep);
    }
    std::vector<double> ns;
    for (const auto t0 = clk::now(); secondsSince(t0) < kTimedSeconds || ns.size() < kMinCalls;) {
        const auto a = clk::now();
        call();
        ns.push_back(std::chrono::duration<double, std::nano>(clk::now() - a).count());
        gap(gap_us, sleep);
    }
    std::nth_element(ns.begin(), ns.begin() + static_cast<std::ptrdiff_t>(ns.size() / 2), ns.end());
    return ns[ns.size() / 2];
}

struct StrategyColumn {
    const char* name;
    Strategy strategy;
};

}  // namespace

int main(int argc, char** argv) {
    bool init = false, with_auto = false, sleep = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--init") == 0) {
            init = true;
        } else if (std::strcmp(argv[i], "--auto") == 0) {
            with_auto = true;
        } else if (std::strcmp(argv[i], "--sleep") == 0) {
            sleep = true;
        } else {
            std::fprintf(stderr, "usage: %s [--init] [--auto] [--sleep]\n", argv[0]);
            return 2;
        }
    }
    if (init)
        stats::initialize_performance_systems();

    const std::vector<std::size_t> sizes = {4096, 10000, 25000, 50000, 100000, 200000};
    const std::vector<double> gaps_us = {0.0, 50.0, 500.0, 2000.0, 20000.0};
    std::vector<StrategyColumn> strategies = {{"VECTORIZED", Strategy::FORCE_VECTORIZED},
                                              {"PARALLEL", Strategy::FORCE_PARALLEL}};
    if (with_auto)
        strategies.push_back({"AUTO", Strategy::AUTO});

    const auto poisson = stats::PoissonDistribution::create(3.5).unwrap();
    const auto gaussian = stats::GaussianDistribution::create(0.0, 1.0).unwrap();

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> counts(0.0, 12.0), reals(-5.0, 5.0);
    const std::size_t max_n = sizes.back();
    std::vector<double> x_counts(max_n), x_reals(max_n), y(max_n);
    for (auto& v : x_counts)
        v = std::round(counts(rng));
    for (auto& v : x_reals)
        v = reals(rng);

    const char* pool = init ? "init" : "plain";
    const char* gap_kind = sleep ? "sleep" : "busy";
    std::printf("# batch_gap_bench pool=%s gap=%s logical_cpus=%u\n", pool, gap_kind,
                std::thread::hardware_concurrency());
    std::printf(
        "Pool,GapKind,Distribution,Operation,BatchSize,Gap_us,Strategy,MedianNsPerElement\n");

    // results[distribution][gap][size][strategy] = ns/element
    std::map<std::string, std::map<double, std::map<std::size_t, std::map<std::string, double>>>>
        results;

    for (const char* dist : {"Poisson", "Gaussian"}) {
        const bool is_poisson = std::strcmp(dist, "Poisson") == 0;
        for (const double gap_us : gaps_us) {
            for (const std::size_t n : sizes) {
                const std::span<const double> xs(is_poisson ? x_counts.data() : x_reals.data(), n);
                const std::span<double> ys(y.data(), n);
                for (const auto& s : strategies) {
                    const stats::detail::PerformanceHint hint{s.strategy, std::nullopt};
                    const double ns = timeCell(
                        [&] {
                            if (is_poisson)
                                poisson.getProbability(xs, ys, hint);
                            else
                                gaussian.getProbability(xs, ys, hint);
                        },
                        gap_us, sleep);
                    const double per_element = ns / static_cast<double>(n);
                    results[dist][gap_us][n][s.name] = per_element;
                    std::printf("%s,%s,%s,PDF,%zu,%.0f,%s,%.3f\n", pool, gap_kind, dist, n, gap_us,
                                s.name, per_element);
                    std::fflush(stdout);
                }
            }
        }
    }

    std::printf("# PARALLEL/VECTORIZED by size, and the sustained crossover\n");
    for (const auto& [dist, by_gap] : results) {
        for (const auto& [gap_us, by_size] : by_gap) {
            std::optional<std::size_t> crossover;
            std::string ratios;
            for (auto it = by_size.rbegin(); it != by_size.rend(); ++it) {
                if (it->second.at("PARALLEL") < it->second.at("VECTORIZED"))
                    crossover = it->first;
                else
                    break;
            }
            for (const auto& [n, by_strategy] : by_size) {
                char buf[32];
                std::snprintf(buf, sizeof buf, " %.2f",
                              by_strategy.at("PARALLEL") / by_strategy.at("VECTORIZED"));
                ratios += buf;
            }
            std::printf("# %-8s gap %6.0f us: P/V%s  crossover %s\n", dist.c_str(), gap_us,
                        ratios.c_str(),
                        crossover ? std::to_string(*crossover).c_str() : "none measured");
        }
    }
    return 0;
}
