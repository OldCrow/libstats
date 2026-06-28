/**
 * @file performance_dispatch_demo.cpp
 * @brief Demonstration of the auto-dispatch and strategy hint API
 *
 * Shows how libstats selects execution strategies (SCALAR, VECTORIZED,
 * PARALLEL, WORK_STEALING) based on batch size and hardware, and how
 * callers can influence that choice with PerformanceHint.
 *
 * Features demonstrated:
 * - Auto-dispatch: no hint, minimal_latency, maximum_throughput
 * - Forced strategies: FORCE_SCALAR, FORCE_VECTORIZED, FORCE_PARALLEL
 * - Hardware capability detection via SystemCapabilities
 * - Per-distribution threshold differences (dispatch thresholds vary
 *   by distribution type — Gaussian PDF != Exponential PDF != Binomial CDF)
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/libstats.h"

// Standard library includes
#include <chrono>    // for timing operations
#include <iomanip>   // for std::setw, std::setprecision, std::fixed, std::left
#include <iostream>  // for std::cout
#include <random>    // for std::mt19937, std::uniform_real_distribution
#include <span>      // for std::span
#include <string>    // for std::string, std::to_string
#include <tuple>     // for std::tuple
#include <vector>    // for std::vector

std::string strategyToString(stats::detail::Strategy strategy) {
    switch (strategy) {
        case stats::detail::Strategy::SCALAR:
            return "SCALAR";
        case stats::detail::Strategy::VECTORIZED:
            return "VECTORIZED";
        case stats::detail::Strategy::PARALLEL:
            return "PARALLEL";
        case stats::detail::Strategy::WORK_STEALING:
            return "WORK_STEALING";
        default:
            return "UNKNOWN";
    }
}

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void demonstrate_smart_dispatch() {
    print_separator("Smart Auto-Dispatch with Performance Hints");

    std::cout
        << "\nTesting smart auto-dispatch with performance hints on Gaussian N(0,1) distribution.\n"
        << "This demonstrates how different hints affect execution strategy selection:\n"
        << "  - No Hint: System chooses automatically based on data size\n"
        << "  - Min Latency: Prioritizes fastest single-element processing\n"
        << "  - Max Throughput: Prioritizes highest overall batch throughput\n"
        << "\nInput data: Random values from Uniform(-2.0, 2.0) distribution\n"
        << "Operation: Computing PDF values for each input\n"
        << std::endl;

    // Create distributions
    auto normal = stats::GaussianDistribution::create(0.0, 1.0).value;
    auto exponential = stats::ExponentialDistribution::create(2.0).value;

    // Create test data of various sizes
    std::vector<size_t> data_sizes = {100, 1000, 10000, 100000};
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    std::cout << std::left << std::setw(12) << "Data Size" << std::setw(18) << "Normal (No Hint)"
              << std::setw(21) << "Normal (Min Latency)" << std::setw(23)
              << "Normal (Max Throughput)" << std::endl;
    std::cout << std::string(74, '-') << std::endl;

    // Store timing data for potential note after table
    std::vector<std::tuple<size_t, long, long, long>> timing_results;

    for (auto size : data_sizes) {
        // Generate test data
        std::vector<double> input_data(size);
        std::vector<double> output_data(size);

        for (size_t i = 0; i < size; ++i) {
            input_data[i] = dist(rng);
        }

        // Test different performance hints
        auto start = std::chrono::high_resolution_clock::now();
        normal.getProbability(std::span<const double>(input_data), std::span<double>(output_data));
        auto time_no_hint = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::high_resolution_clock::now() - start)
                                .count();

        start = std::chrono::high_resolution_clock::now();
        auto hint_latency = stats::detail::PerformanceHint::minimal_latency();
        normal.getProbability(std::span<const double>(input_data), std::span<double>(output_data),
                              hint_latency);
        auto time_accuracy = std::chrono::duration_cast<std::chrono::microseconds>(
                                 std::chrono::high_resolution_clock::now() - start)
                                 .count();

        start = std::chrono::high_resolution_clock::now();
        auto hint_throughput = stats::detail::PerformanceHint::maximum_throughput();
        normal.getProbability(std::span<const double>(input_data), std::span<double>(output_data),
                              hint_throughput);
        auto time_speed = std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::high_resolution_clock::now() - start)
                              .count();

        // Store timing data
        timing_results.emplace_back(size, static_cast<long>(time_no_hint),
                                    static_cast<long>(time_accuracy),
                                    static_cast<long>(time_speed));

        std::cout << std::left << std::setw(12) << size << std::setw(18)
                  << (std::to_string(time_no_hint) + " μs") << std::setw(21)
                  << (std::to_string(time_accuracy) + " μs") << std::setw(23)
                  << (std::to_string(time_speed) + " μs") << std::endl;
    }

    // Dynamic observation: scan all sizes and report the actual winner pattern.
    std::size_t max_wins = 0, min_wins = 0;
    std::size_t best_ratio_size = 0;
    double best_ratio = 0.0;
    for (const auto& [size, no_hint, min_latency, max_throughput] : timing_results) {
        if (min_latency <= 0 || max_throughput <= 0)
            continue;
        if (max_throughput < min_latency) {
            ++max_wins;
            double ratio = static_cast<double>(min_latency) / static_cast<double>(max_throughput);
            if (ratio > best_ratio) {
                best_ratio = ratio;
                best_ratio_size = size;
            }
        } else {
            ++min_wins;
        }
    }
    if (max_wins > 0 && min_wins == 0) {
        std::cout << "\nOBSERVATION: Max Throughput (vectorized batch) outperforms Min Latency"
                  << " at all " << max_wins << " tested sizes"
                  << " (largest advantage: " << std::fixed << std::setprecision(1) << best_ratio
                  << "x at " << best_ratio_size << " elements).\n"
                  << "The vectorized batch path pays off even at small sizes on this hardware."
                  << std::endl;
    } else if (max_wins > 0) {
        std::cout << "\nOBSERVATION: Max Throughput outperforms Min Latency at " << max_wins
                  << " of " << (max_wins + min_wins) << " tested sizes"
                  << " (best advantage: " << std::fixed << std::setprecision(1) << best_ratio
                  << "x at " << best_ratio_size << " elements)." << std::endl;
    }
}

void demonstrate_forced_strategies() {
    print_separator("Forced Strategy Hints");

    std::cout
        << "\nBeyond minimal_latency/maximum_throughput, each strategy can be forced\n"
        << "explicitly. Use forced hints to benchmark individual strategies or to\n"
        << "lock a strategy for a specific workload that you have profiled.\n\n"
        << "Caution: forced strategies bypass the dispatch threshold logic and can\n"
        << "be slower than auto-dispatch for the given batch size.\n\n";

    auto normal = stats::GaussianDistribution::create(0.0, 1.0).value;
    constexpr size_t N = 50000;
    std::vector<double> xs(N), out_scalar(N), out_vectorized(N), out_parallel(N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> gen(-3.0, 3.0);
    for (auto& v : xs) v = gen(rng);

    stats::detail::PerformanceHint hint_scalar, hint_vec, hint_par;
    hint_scalar.strategy   = stats::detail::PerformanceHint::PreferredStrategy::FORCE_SCALAR;
    hint_vec.strategy      = stats::detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
    hint_par.strategy      = stats::detail::PerformanceHint::PreferredStrategy::FORCE_PARALLEL;

    auto time_us = [&](auto& hint, auto& output) {
        auto t0 = std::chrono::high_resolution_clock::now();
        normal.getLogProbability(std::span<const double>(xs), std::span<double>(output), hint);
        return std::chrono::duration_cast<std::chrono::microseconds>(
                   std::chrono::high_resolution_clock::now() - t0).count();
    };

    long t_scl = time_us(hint_scalar,    out_scalar);
    long t_vec = time_us(hint_vec,       out_vectorized);
    long t_par = time_us(hint_par,       out_parallel);

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Gaussian LogPDF on " << N << " elements:\n";
    std::cout << "  FORCE_SCALAR:     " << t_scl << " us\n";
    std::cout << "  FORCE_VECTORIZED: " << t_vec << " us"
              << "  (" << static_cast<double>(t_scl) / std::max(t_vec, 1L) << "x vs scalar)\n";
    std::cout << "  FORCE_PARALLEL:   " << t_par << " us"
              << "  (" << static_cast<double>(t_scl) / std::max(t_par, 1L) << "x vs scalar)\n";

    // Verify that all three strategies produce identical results
    bool all_match = true;
    for (size_t i = 0; i < N; ++i) {
        if (std::abs(out_scalar[i] - out_vectorized[i]) > 1e-10 ||
            std::abs(out_scalar[i] - out_parallel[i])   > 1e-10) {
            all_match = false;
            break;
        }
    }
    std::cout << "  Correctness: all three strategies match " << (all_match ? "✓" : "✗") << "\n";
}

void demonstrate_performance_dispatcher() {
    print_separator("Dispatch Thresholds Vary by Distribution");

    std::cout << "\nThe dispatcher chooses SCALAR/VECTORIZED/PARALLEL based on batch size,\n"
              << "but the crossover thresholds are per-distribution and per-operation.\n"
              << "A batch size that triggers VECTORIZED for Gaussian PDF may still use\n"
              << "SCALAR for Binomial CDF. This is why dispatch_thresholds.h has one\n"
              << "row per distribution per architecture.\n\n";

    // Get system capabilities
    const auto& capabilities = stats::detail::SystemCapabilities::current();

    std::cout << "Hardware capabilities:\n";
    std::cout << "  Logical Cores: " << capabilities.logical_cores() << std::endl;
    std::cout << "  Physical Cores: " << capabilities.physical_cores() << std::endl;
    std::cout << "  L1 Cache: " << capabilities.l1_cache_size() << " bytes" << std::endl;
    std::cout << "  L2 Cache: " << capabilities.l2_cache_size() << " bytes" << std::endl;
    std::cout << "  L3 Cache: " << capabilities.l3_cache_size() << " bytes" << std::endl;
    std::cout << "  SIMD Support: ";
    if (capabilities.has_avx512())
        std::cout << "AVX-512 ";
    else if (capabilities.has_avx2())
        std::cout << "AVX2 ";
    else if (capabilities.has_avx())
        std::cout << "AVX ";
    else if (capabilities.has_sse2())
        std::cout << "SSE2 ";
    else if (capabilities.has_neon())
        std::cout << "NEON ";
    else
        std::cout << "None";
    std::cout << std::endl;
    stats::detail::PerformanceDispatcher dispatcher;

    std::cout << "\nStrategy selection at batch sizes {50, 500, 5000, 50000, 500000}:\n";
    std::cout << std::left << std::setw(15) << "Batch Size"
              << std::setw(22) << "Gaussian PDF"
              << std::setw(22) << "Exponential PDF"
              << std::setw(22) << "Binomial CDF" << "\n";
    std::cout << std::string(80, '-') << "\n";

    std::vector<size_t> problem_sizes = {50, 500, 5000, 50000, 500000};
    for (auto size : problem_sizes) {
        auto sg = dispatcher.selectStrategy(size, stats::detail::DistributionType::GAUSSIAN,
                                            stats::detail::OperationType::PDF, capabilities);
        auto se = dispatcher.selectStrategy(size, stats::detail::DistributionType::EXPONENTIAL,
                                            stats::detail::OperationType::PDF, capabilities);
        auto sb = dispatcher.selectStrategy(size, stats::detail::DistributionType::BINOMIAL,
                                            stats::detail::OperationType::CDF, capabilities);
        std::cout << std::setw(15) << size
                  << std::setw(22) << strategyToString(sg)
                  << std::setw(22) << strategyToString(se)
                  << std::setw(22) << strategyToString(sb) << "\n";
    }
    std::cout << "\nThresholds are in include/core/dispatch_thresholds.h and are tuned\n"
              << "per architecture using the strategy_profile tool.\n";
}

int main() {
    std::cout << "=== libstats Dispatch System Demo ===" << std::endl;
    std::cout << "Demonstrating intelligent auto-dispatch with performance hints\n" << std::endl;

    try {
        demonstrate_smart_dispatch();
        demonstrate_forced_strategies();
        demonstrate_performance_dispatcher();

        print_separator("Summary");
        std::cout << "\u2705 Auto-dispatch (no hint, minimal_latency, maximum_throughput)\n";
        std::cout << "\u2705 Forced strategies (FORCE_SCALAR, FORCE_VECTORIZED, FORCE_PARALLEL)\n";
        std::cout << "\u2705 Per-distribution dispatch threshold differences shown\n";
        std::cout << "\nSee also: logpdf_and_likelihood_demo for actual distribution batch calls.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error during demonstration: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
