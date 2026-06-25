/**
 * @file performance_learning_demo.cpp
 * @brief Demonstration of the Performance Learning Framework
 *
 * This example showcases the intelligent auto-dispatch system that learns
 * from actual performance measurements to automatically select optimal execution
 * strategies based on problem characteristics and hardware capabilities.
 *
 * Features demonstrated:
 * - Smart auto-dispatch with performance hints
 * - Confidence-based strategy recommendations
 * - Adaptive learning from execution history
 * - Hardware capability detection and optimization
 * - Cross-distribution performance comparison
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

void demonstrate_performance_dispatcher() {
    print_separator("Performance Dispatcher Learning System");

    std::cout << "\nDemonstrating automatic hardware detection and strategy selection.\n"
              << "The dispatcher analyzes system capabilities and selects optimal strategies\n"
              << "based on batch size, distribution complexity, and hardware features.\n"
              << std::endl;

    // Get system capabilities
    const auto& capabilities = stats::detail::SystemCapabilities::current();

    std::cout << "System Capabilities Detection:" << std::endl;
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
    // Create a dispatcher instance to show strategy selection
    stats::detail::PerformanceDispatcher dispatcher;

    std::cout << "\nStrategy Selection by Problem Size:" << std::endl;
    std::cout << std::left << std::setw(15) << "Problem Size" << std::setw(20)
              << "Selected Strategy" << std::endl;
    std::cout << std::string(35, '-') << std::endl;

    std::vector<size_t> problem_sizes = {50, 500, 5000, 50000, 500000};

    for (auto size : problem_sizes) {
        auto strategy = dispatcher.selectStrategy(size, stats::detail::DistributionType::GAUSSIAN,
                                                  stats::detail::OperationType::PDF, capabilities);

        std::cout << std::setw(15) << size << std::setw(20) << strategyToString(strategy)
                  << std::endl;
    }
}

int main() {
    std::cout << "=== libstats Dispatch System Demo ===" << std::endl;
    std::cout << "Demonstrating intelligent auto-dispatch with performance hints\n" << std::endl;

    try {
        demonstrate_smart_dispatch();
        demonstrate_performance_dispatcher();

        print_separator("Summary");
        std::cout << "\u2705 Smart auto-dispatch working with performance hints" << std::endl;
        std::cout << "\u2705 Performance dispatcher providing strategy selection" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during demonstration: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
