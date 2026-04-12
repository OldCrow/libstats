/**
 * @file gaussian_strategy_profile.cpp
 * @brief Profile Gaussian PDF and CDF with each execution strategy at various batch sizes
 *
 * Investigates a performance anomaly where Gaussian PDF at 100k elements is slower
 * than SciPy on AVX-512 machines, while CDF at the same size is faster, and both
 * win at 1M. This tool forces each strategy (SCALAR, VECTORIZED, PARALLEL,
 * WORK_STEALING) and compares against AUTO dispatch to identify the bottleneck.
 */

#include "tool_utils.h"

#include "libstats/core/dispatch_utils.h"
#include "libstats/core/performance_dispatcher.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <string>
#include <vector>

using namespace stats;
using namespace stats::detail;
using namespace std::chrono;

namespace {
constexpr int WARMUP = 3;
constexpr int REPEATS = 7;

/// Median of a vector of durations (modifies input).
double median_ms(std::vector<double>& times) {
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

/// Benchmark a callable, return median wall-clock milliseconds.
template <typename Fn>
double bench(Fn&& fn) {
    for (int i = 0; i < WARMUP; ++i) fn();
    std::vector<double> times;
    times.reserve(REPEATS);
    for (int i = 0; i < REPEATS; ++i) {
        auto t0 = high_resolution_clock::now();
        fn();
        auto t1 = high_resolution_clock::now();
        times.push_back(duration<double, std::milli>(t1 - t0).count());
    }
    return median_ms(times);
}

struct StrategyInfo {
    Strategy strategy;
    const char* name;
};

constexpr StrategyInfo STRATEGIES[] = {
    {Strategy::SCALAR, "SCALAR"},
    {Strategy::VECTORIZED, "VECTORIZED"},
    {Strategy::PARALLEL, "PARALLEL"},
    {Strategy::WORK_STEALING, "WORK_STEAL"},
};

}  // namespace

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n"
              << "║  Gaussian Strategy Profile — AVX-512 Investigation              ║\n"
              << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    // Print system info
    const auto& sys = SystemCapabilities::current();
    std::cout << "System: " << sys.logical_cores() << " logical cores, "
              << sys.physical_cores() << " physical cores\n";
    std::cout << "SIMD:   SSE2=" << sys.has_sse2() << " AVX=" << sys.has_avx()
              << " AVX2=" << sys.has_avx2() << " AVX-512=" << sys.has_avx512()
              << " NEON=" << sys.has_neon() << "\n";
    std::cout << "Cache:  L1=" << sys.l1_cache_size() / 1024 << "KB"
              << " L2=" << sys.l2_cache_size() / 1024 << "KB"
              << " L3=" << sys.l3_cache_size() / (1024 * 1024) << "MB\n\n";

    GaussianDistribution gauss(0.0, 1.0);

    std::vector<size_t> sizes = {1000, 10000, 50000, 100000, 250000, 500000, 1000000};

    // ── PDF profiling ────────────────────────────────────────────────────────
    std::cout << "── Gaussian PDF ──\n\n";
    std::cout << std::right
              << std::setw(10) << "N" << "  "
              << std::setw(12) << "AUTO" << "  "
              << std::setw(12) << "SCALAR" << "  "
              << std::setw(12) << "VECTORIZED" << "  "
              << std::setw(12) << "PARALLEL" << "  "
              << std::setw(12) << "WORK_STEAL" << "  "
              << std::setw(12) << "Best" << "\n";
    std::cout << std::string(96, '-') << "\n";

    for (auto n : sizes) {
        std::vector<double> input(n);
        std::vector<double> output(n);
        // Fill with linearly spaced values
        for (size_t i = 0; i < n; ++i)
            input[i] = -4.0 + 8.0 * static_cast<double>(i) / static_cast<double>(n - 1);

        std::span<const double> in_span(input);
        std::span<double> out_span(output);

        // AUTO dispatch
        double t_auto = bench([&] { gauss.getProbability(in_span, out_span); });

        // Each explicit strategy
        double t_strat[4];
        for (int s = 0; s < 4; ++s) {
            t_strat[s] = bench([&, strat = STRATEGIES[s].strategy] {
                gauss.getProbabilityWithStrategy(in_span, out_span, strat);
            });
        }

        // Find best
        int best_idx = 0;
        for (int s = 1; s < 4; ++s)
            if (t_strat[s] < t_strat[best_idx]) best_idx = s;

        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << n << "  "
                  << std::setw(11) << t_auto << "  "
                  << std::setw(11) << t_strat[0] << "  "
                  << std::setw(11) << t_strat[1] << "  "
                  << std::setw(11) << t_strat[2] << "  "
                  << std::setw(11) << t_strat[3] << "  "
                  << std::setw(11) << STRATEGIES[best_idx].name << "\n";
    }

    // ── CDF profiling ────────────────────────────────────────────────────────
    std::cout << "\n── Gaussian CDF ──\n\n";
    std::cout << std::right
              << std::setw(10) << "N" << "  "
              << std::setw(12) << "AUTO" << "  "
              << std::setw(12) << "SCALAR" << "  "
              << std::setw(12) << "VECTORIZED" << "  "
              << std::setw(12) << "PARALLEL" << "  "
              << std::setw(12) << "WORK_STEAL" << "  "
              << std::setw(12) << "Best" << "\n";
    std::cout << std::string(96, '-') << "\n";

    for (auto n : sizes) {
        std::vector<double> input(n);
        std::vector<double> output(n);
        for (size_t i = 0; i < n; ++i)
            input[i] = -4.0 + 8.0 * static_cast<double>(i) / static_cast<double>(n - 1);

        std::span<const double> in_span(input);
        std::span<double> out_span(output);

        double t_auto = bench([&] { gauss.getCumulativeProbability(in_span, out_span); });

        double t_strat[4];
        for (int s = 0; s < 4; ++s) {
            t_strat[s] = bench([&, strat = STRATEGIES[s].strategy] {
                gauss.getCumulativeProbabilityWithStrategy(in_span, out_span, strat);
            });
        }

        int best_idx = 0;
        for (int s = 1; s < 4; ++s)
            if (t_strat[s] < t_strat[best_idx]) best_idx = s;

        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << n << "  "
                  << std::setw(11) << t_auto << "  "
                  << std::setw(11) << t_strat[0] << "  "
                  << std::setw(11) << t_strat[1] << "  "
                  << std::setw(11) << t_strat[2] << "  "
                  << std::setw(11) << t_strat[3] << "  "
                  << std::setw(11) << STRATEGIES[best_idx].name << "\n";
    }

    // ── AUTO dispatch strategy report ────────────────────────────────────────
    std::cout << "\n── AUTO dispatch decisions ──\n\n";
    PerformanceDispatcher dispatcher;
    std::cout << std::setw(10) << "N" << "  "
              << std::setw(20) << "PDF Strategy" << "  "
              << std::setw(20) << "CDF Strategy" << "\n";
    std::cout << std::string(54, '-') << "\n";

    for (auto n : sizes) {
        auto pdf_strat = dispatcher.selectOptimalStrategy(
            n, DistributionType::GAUSSIAN, ComputationComplexity::MODERATE, sys);
        auto cdf_strat = dispatcher.selectOptimalStrategy(
            n, DistributionType::GAUSSIAN, ComputationComplexity::COMPLEX, sys);

        std::cout << std::setw(10) << n << "  "
                  << std::setw(20) << stats::detail::detail::strategyToString(pdf_strat) << "  "
                  << std::setw(20) << stats::detail::detail::strategyToString(cdf_strat) << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}
