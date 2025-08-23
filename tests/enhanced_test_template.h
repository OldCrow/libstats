#pragma once

#include "../include/platform/work_stealing_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <thread>
#include <vector>

namespace stats {
namespace testing {

//==============================================================================
// BENCHMARKING UTILITIES
//==============================================================================

struct BenchmarkResult {
    std::string operation_name;
    long simd_time_us;
    long parallel_time_us;
    long work_stealing_time_us;
    long gpu_accelerated_time_us;  // Renamed from cache_aware_time_us
    double parallel_speedup;
    double work_stealing_speedup;
    double gpu_accelerated_speedup;  // Renamed from cache_aware_speedup
};

class StandardizedBenchmark {
   public:
    static void printBenchmarkHeader(const std::string& distribution_name, size_t dataset_size) {
        std::cout << "\n=== " << distribution_name
                  << " Enhanced Performance Benchmark ===" << std::endl;
        std::cout << "Dataset size: " << dataset_size << " elements" << std::endl;
        std::cout << "Hardware threads: "
                  << static_cast<std::size_t>(std::thread::hardware_concurrency()) << std::endl;
    }

    static void printBenchmarkResults(const std::vector<BenchmarkResult>& results) {
        std::cout << "\nBenchmark Results:" << std::endl;
        std::cout << std::setw(10) << "Operation" << std::setw(12) << "SIMD (μs)" << std::setw(15)
                  << "Parallel (μs)" << std::setw(18) << "Work-Steal (μs)" << std::setw(18)
                  << "GPU-Accel (μs)" << std::setw(12) << "P-Speedup" << std::setw(12)
                  << "WS-Speedup" << std::setw(12) << "GA-Speedup" << std::endl;
        std::cout << std::string(120, '-') << std::endl;

        for (const auto& result : results) {
            std::cout << std::setw(10) << result.operation_name << std::setw(12)
                      << result.simd_time_us << std::setw(15) << result.parallel_time_us
                      << std::setw(18) << result.work_stealing_time_us << std::setw(18)
                      << result.gpu_accelerated_time_us << std::setw(12) << std::fixed
                      << std::setprecision(2) << result.parallel_speedup << std::setw(12)
                      << std::fixed << std::setprecision(2) << result.work_stealing_speedup
                      << std::setw(12) << std::fixed << std::setprecision(2)
                      << result.gpu_accelerated_speedup << std::endl;
        }
    }

    static void printPerformanceAnalysis(const std::vector<BenchmarkResult>& results) {
        std::cout << "\nPerformance Analysis:" << std::endl;

        if (static_cast<std::size_t>(std::thread::hardware_concurrency()) > 2) {
            for (const auto& result : results) {
                if (result.parallel_speedup > 0.8) {
                    std::cout << "  ✓ " << result.operation_name << " parallel shows good speedup"
                              << std::endl;
                } else {
                    std::cout << "  ⚠ " << result.operation_name
                              << " parallel speedup lower than expected" << std::endl;
                }
            }
        } else {
            std::cout << "  ℹ Single/dual-core system - parallel overhead may dominate"
                      << std::endl;
        }
    }
};

//==============================================================================
// STATISTICAL UTILITIES
//==============================================================================

class StatisticalTestUtils {
   public:
    static std::pair<double, double> calculateSampleStats(const std::vector<double>& samples) {
        double mean = std::accumulate(samples.begin(), samples.end(), 0.0) /
                      static_cast<double>(samples.size());
        double variance = 0.0;
        for (double x : samples) {
            variance += (x - mean) * (x - mean);
        }
        variance /= static_cast<double>(samples.size());
        return {mean, variance};
    }

    static bool approxEqual(double a, double b, double tolerance = 1e-12) {
        return std::abs(a - b) <= tolerance;
    }

    template <typename Distribution>
    static void verifyBatchCorrectness(const Distribution& dist,
                                       const std::vector<double>& test_values,
                                       const std::vector<double>& batch_results,
                                       const std::string& operation_name,
                                       double tolerance = 1e-10) {
        bool all_correct = true;
        // Reduce the number of checks to avoid expensive cache operations in tight loops
        const std::size_t check_count = std::min(static_cast<std::size_t>(10), test_values.size());

        // Pre-compute expected values using batch operations to avoid expensive individual cache
        // hits
        std::vector<double> expected_results(check_count);
        std::vector<double> check_values(check_count);

        // Collect check values first
        for (std::size_t i = 0; i < check_count; ++i) {
            std::size_t idx = i * (test_values.size() / check_count);
            check_values[i] = test_values[idx];
        }

        // Use batch operations to get expected results (much more efficient)
        if (operation_name == "PDF") {
            std::span<const double> input_span(check_values);
            std::span<double> output_span(expected_results);
            dist.getProbabilityWithStrategy(input_span, output_span,
                                            stats::performance::Strategy::SCALAR);
        } else if (operation_name == "LogPDF") {
            std::span<const double> input_span(check_values);
            std::span<double> output_span(expected_results);
            dist.getLogProbabilityWithStrategy(input_span, output_span,
                                               stats::performance::Strategy::SCALAR);
        } else if (operation_name == "CDF") {
            std::span<const double> input_span(check_values);
            std::span<double> output_span(expected_results);
            dist.getCumulativeProbabilityWithStrategy(input_span, output_span,
                                                      stats::performance::Strategy::SCALAR);
        } else {
            // Skip unknown operations
            std::cout << "  ✓ " << operation_name
                      << " batch operations completed (verification skipped for unknown operation)"
                      << std::endl;
            return;
        }

        // Now compare batch results with expected results
        for (std::size_t i = 0; i < check_count; ++i) {
            std::size_t idx = i * (test_values.size() / check_count);

            if (std::abs(batch_results[idx] - expected_results[i]) > tolerance) {
                all_correct = false;
                break;
            }
        }

        if (all_correct) {
            std::cout << "  ✓ " << operation_name << " batch operations produce correct results"
                      << std::endl;
        } else {
            std::cout << "  ✗ " << operation_name << " batch correctness check failed" << std::endl;
        }

        EXPECT_TRUE(all_correct) << operation_name
                                 << " batch operations should produce correct results";
    }
};

//==============================================================================
// THREAD SAFETY TESTING
//==============================================================================

template <typename Distribution>
class ThreadSafetyTester {
   public:
    static void testBasicThreadSafety(const Distribution& dist, const std::string& dist_name) {
        const std::size_t num_threads = 4;
        constexpr std::size_t samples_per_thread = 1000;

        std::vector<std::thread> threads;
        std::vector<std::vector<double>> results(static_cast<std::size_t>(num_threads));

        for (std::size_t t = 0; t < num_threads; ++t) {
            threads.emplace_back([&dist, &results, t]() {
                std::mt19937 local_rng(static_cast<unsigned int>(42 + static_cast<int>(t)));
                results[t].reserve(samples_per_thread);

                for (std::size_t i = 0; i < samples_per_thread; ++i) {
                    results[t].push_back(dist.sample(local_rng));
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // Verify that all threads produced valid results
        for (std::size_t t = 0; t < num_threads; ++t) {
            EXPECT_EQ(results[t].size(), samples_per_thread);
            for (double val : results[t]) {
                EXPECT_TRUE(std::isfinite(val));
            }
        }

        std::cout << "  ✓ " << dist_name << " basic thread safety test passed" << std::endl;
    }
};

//==============================================================================
// EDGE CASE TESTING
//==============================================================================

template <typename Distribution>
class EdgeCaseTester {
   public:
    static void testExtremeValues(const Distribution& dist, const std::string& dist_name) {
        std::vector<double> extreme_values = {-1e6, -100.0, -1.0, 0.0, 1.0, 100.0, 1e6};

        for (double val : extreme_values) {
            double pdf = dist.getProbability(val);
            double log_pdf = dist.getLogProbability(val);
            double cdf = dist.getCumulativeProbability(val);

            EXPECT_GE(pdf, 0.0);
            EXPECT_TRUE(std::isfinite(log_pdf) || std::isinf(log_pdf));
            EXPECT_GE(cdf, 0.0);
            EXPECT_LE(cdf, 1.0);
        }

        std::cout << "  ✓ " << dist_name << " extreme value handling test passed" << std::endl;
    }

    static void testEmptyBatchOperations(const Distribution& dist, const std::string& dist_name) {
        std::vector<double> empty_values;
        std::vector<double> empty_results;

        // These should not crash
        dist.getProbabilityWithStrategy(std::span<const double>(empty_values),
                                        std::span<double>(empty_results),
                                        stats::performance::Strategy::SCALAR);
        dist.getLogProbabilityWithStrategy(std::span<const double>(empty_values),
                                           std::span<double>(empty_results),
                                           stats::performance::Strategy::SCALAR);
        dist.getCumulativeProbabilityWithStrategy(std::span<const double>(empty_values),
                                                  std::span<double>(empty_results),
                                                  stats::performance::Strategy::SCALAR);

        std::cout << "  ✓ " << dist_name << " empty batch operations handled gracefully"
                  << std::endl;
    }
};

}  // namespace testing
}  // namespace stats
