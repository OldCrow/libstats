#pragma once

/**
 * @file tests/fixtures.h
 * @brief Reusable test fixtures and data generators for distribution testing
 *
 * This header provides standardized test fixtures, data generators, benchmark utilities,
 * and test infrastructure that is shared across multiple test files. This replaces the
 * functionality previously in basic_test_template.h and enhanced_test_template.h.
 *
 * Phase 3E: Test Infrastructure Namespace
 * Part of the stats::tests:: namespace hierarchy reorganization
 */

#include "../../include/platform/work_stealing_pool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

// Conditionally include gtest only if requested
#ifdef LIBSTATS_ENABLE_GTEST_INTEGRATION
    #include <gtest/gtest.h>
#else
// Provide fallback macros for non-gtest mode that support streaming
namespace libstats_test_impl {
struct NullStream {
    template <typename T>
    NullStream& operator<<(const T&) {
        return *this;
    }
};
inline void expect_true(bool condition, const char* condition_str) {
    if (!condition) {
        std::cout << "  ✗ Assertion failed: " << condition_str << std::endl;
    }
}
inline void expect_eq_impl(const char* a_str, const char* b_str) {
    std::cout << "  ✗ Expected " << a_str << " == " << b_str << std::endl;
}
template <typename A, typename B>
inline void expect_eq(const A& a, const B& b, const char* a_str, const char* b_str) {
    if (!(a == b)) {
        std::cout << "  ✗ Expected " << a_str << " == " << b_str << ", got " << a << " != " << b
                  << std::endl;
    }
}
template <typename A, typename B>
inline void expect_ge(const A& a, const B& b, const char* a_str, const char* b_str) {
    if (a < b) {
        std::cout << "  ✗ Expected " << a_str << " >= " << b_str << ", got " << a << " < " << b
                  << std::endl;
    }
}
template <typename A, typename B>
inline void expect_le(const A& a, const B& b, const char* a_str, const char* b_str) {
    if (a > b) {
        std::cout << "  ✗ Expected " << a_str << " <= " << b_str << ", got " << a << " > " << b
                  << std::endl;
    }
}
}  // namespace libstats_test_impl
    #define EXPECT_TRUE(condition)                                                                 \
        (libstats_test_impl::expect_true((condition), #condition), libstats_test_impl::NullStream())
    #define EXPECT_EQ(a, b)                                                                        \
        (libstats_test_impl::expect_eq((a), (b), #a, #b), libstats_test_impl::NullStream())
    #define EXPECT_GE(a, b)                                                                        \
        (libstats_test_impl::expect_ge((a), (b), #a, #b), libstats_test_impl::NullStream())
    #define EXPECT_LE(a, b)                                                                        \
        (libstats_test_impl::expect_le((a), (b), #a, #b), libstats_test_impl::NullStream())
#endif

namespace stats {
namespace tests {
namespace fixtures {

//==============================================================================
// Basic Test Utilities (formerly BasicTestUtilities)
//==============================================================================

/**
 * @brief Standardized basic test framework for distribution implementations
 */
class BasicTestFormatter {
   public:
    static void printTestHeader(const std::string& distributionName) {
        std::cout << "Testing " << distributionName << "Distribution Implementation" << std::endl;
        std::cout << std::string(40 + distributionName.length(), '=') << std::endl << std::endl;
    }

    static void printTestStart(int testNumber, const std::string& testName) {
        std::cout << "Test " << testNumber << ": " << testName << std::endl;
    }

    static void printTestSuccess(const std::string& message = "") {
        std::cout << "✅ " << (message.empty() ? "Test passed successfully" : message) << std::endl;
    }

    static void printTestError(const std::string& message) {
        std::cout << "❌ " << message << std::endl;
    }

    static void printProperty(const std::string& name, double value, int precision = 6) {
        std::cout << name << ": " << std::fixed << std::setprecision(precision) << value
                  << std::endl;
    }

    static void printPropertyInt(const std::string& name, int value) {
        std::cout << name << ": " << value << std::endl;
    }

    static void printSamples(const std::vector<double>& samples,
                             const std::string& prefix = "Samples", int precision = 3) {
        std::cout << prefix << ": ";
        for (double sample : samples) {
            std::cout << std::fixed << std::setprecision(precision) << sample << " ";
        }
        std::cout << std::endl;
    }

    static void printIntegerSamples(const std::vector<int>& samples,
                                    const std::string& prefix = "Integer samples") {
        std::cout << prefix << ": ";
        for (int sample : samples) {
            std::cout << sample << " ";
        }
        std::cout << std::endl;
    }

    static void printBatchResults(const std::vector<double>& results, const std::string& prefix,
                                  int precision = 4) {
        std::cout << prefix << ": ";
        for (double result : results) {
            std::cout << std::fixed << std::setprecision(precision) << result << " ";
        }
        std::cout << std::endl;
    }

    static void printLargeBatchValidation(double firstValue, double lastValue,
                                          const std::string& testType) {
        std::cout << "Large batch " << testType << " (first): " << std::fixed
                  << std::setprecision(6) << firstValue << std::endl;
        std::cout << "All values equal: " << (firstValue == lastValue ? "YES" : "NO") << std::endl;
    }

    static void printCompletionMessage(const std::string& distributionName) {
        std::cout << "\n🎉 All " << distributionName << "Distribution tests completed successfully!"
                  << std::endl;
    }

    static void printSummaryHeader() { std::cout << "\n=== SUMMARY ===" << std::endl; }

    static void printSummaryItem(const std::string& item) {
        std::cout << "✓ " << item << std::endl;
    }

    static void printNewline() { std::cout << std::endl; }
};

//==============================================================================
// Standard Test Data Generators
//==============================================================================

/**
 * @brief Standard test data generators for different distribution types
 */
class TestDataGenerators {
   public:
    // Helper function to validate samples are within expected range
    static bool validateSamplesInRange(const std::vector<double>& samples, double minVal,
                                       double maxVal) {
        return std::all_of(samples.begin(), samples.end(),
                           [minVal, maxVal](double x) { return x >= minVal && x <= maxVal; });
    }

    // Helper function to compute sample statistics
    static double computeSampleMean(const std::vector<double>& samples) {
        if (samples.empty())
            return 0.0;
        double sum = 0.0;
        for (double sample : samples) {
            sum += sample;
        }
        return sum / static_cast<double>(samples.size());
    }

    static double computeSampleVariance(const std::vector<double>& samples) {
        if (samples.size() < 2)
            return 0.0;
        double mean = computeSampleMean(samples);
        double sumSquaredDiffs = 0.0;
        for (double sample : samples) {
            double diff = sample - mean;
            sumSquaredDiffs += diff * diff;
        }
        return sumSquaredDiffs / static_cast<double>(samples.size() - 1);
    }

    // Helper function to check if two values are approximately equal
    static bool approxEqual(double a, double b, double tolerance = 1e-10) {
        return std::abs(a - b) < tolerance;
    }

    // Standard test data generators
    static std::vector<double> generateUniformTestData() {
        return {0.1, 0.3, 0.7, 0.2, 0.9, 0.4, 0.8, 0.6, 0.15, 0.85};
    }

    static std::vector<double> generateGaussianTestData() {
        return {0.5, 1.2, 0.8, -0.3, 0.9, -0.5, 1.1, 0.2, -0.8, 1.5};
    }

    static std::vector<double> generateExponentialTestData() {
        return {0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.8, 0.7, 1.1};
    }

    static std::vector<double> generateDiscreteTestData() { return {1, 2, 3, 4, 5, 6, 1, 2, 3, 4}; }

    static std::vector<double> generatePoissonTestData() {
        return {2, 1, 4, 3, 2, 5, 1, 3, 2, 4, 3, 2, 1, 4, 3};
    }

    static std::vector<double> generateGammaTestData() {
        return {0.8, 1.5, 2.1, 0.9, 1.2, 2.8, 1.1, 1.8, 0.7, 2.3};
    }
};

//==============================================================================
// Enhanced Test Benchmarking (formerly from enhanced_test_template.h)
//==============================================================================

/**
 * @brief Structure for holding benchmark results from performance tests
 */
struct BenchmarkResult {
    std::string operation_name;
    long simd_time_us;
    long parallel_time_us;
    long work_stealing_time_us;
    long gpu_accelerated_time_us;
    double parallel_speedup;
    double work_stealing_speedup;
    double gpu_accelerated_speedup;
};

/**
 * @brief Standardized benchmark utilities for enhanced performance testing
 */
class BenchmarkFormatter {
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
// Statistical Test Utilities
//==============================================================================

/**
 * @brief Statistical test utilities for enhanced distribution testing
 */
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
                                            stats::detail::Strategy::SCALAR);
        } else if (operation_name == "LogPDF") {
            std::span<const double> input_span(check_values);
            std::span<double> output_span(expected_results);
            dist.getLogProbabilityWithStrategy(input_span, output_span,
                                               stats::detail::Strategy::SCALAR);
        } else if (operation_name == "CDF") {
            std::span<const double> input_span(check_values);
            std::span<double> output_span(expected_results);
            dist.getCumulativeProbabilityWithStrategy(input_span, output_span,
                                                      stats::detail::Strategy::SCALAR);
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
// Thread Safety Testing
//==============================================================================

/**
 * @brief Thread safety testing utilities for distribution implementations
 */
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

        // Verify all threads produced valid results
        for (std::size_t t = 0; t < num_threads; ++t) {
            EXPECT_EQ(results[t].size(), samples_per_thread)
                << "Thread " << t << " should have produced " << samples_per_thread << " samples";
        }

        std::cout << "  ✓ " << dist_name << " thread safety test passed" << std::endl;
    }
};

//==============================================================================
// Edge Case Testing
//==============================================================================

/**
 * @brief Edge case testing utilities for distribution implementations
 */
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
                                        stats::detail::Strategy::SCALAR);
        dist.getLogProbabilityWithStrategy(std::span<const double>(empty_values),
                                           std::span<double>(empty_results),
                                           stats::detail::Strategy::SCALAR);
        dist.getCumulativeProbabilityWithStrategy(std::span<const double>(empty_values),
                                                  std::span<double>(empty_results),
                                                  stats::detail::Strategy::SCALAR);

        std::cout << "  ✓ " << dist_name << " empty batch operations handled gracefully"
                  << std::endl;
    }
};

// Backward compatibility aliases for easier migration
namespace legacy {
// Import old namespace for backward compatibility
using BasicTestFormatter = fixtures::BasicTestFormatter;
using TestDataGenerators = fixtures::TestDataGenerators;
using BenchmarkFormatter = fixtures::BenchmarkFormatter;
using StatisticalTestUtils = fixtures::StatisticalTestUtils;
template <typename Distribution>
using EdgeCaseTester = fixtures::EdgeCaseTester<Distribution>;

// Legacy class names
using StandardizedBasicTest = BasicTestFormatter;
using StandardizedBenchmark = BenchmarkFormatter;
}  // namespace legacy

}  // namespace fixtures
}  // namespace tests
}  // namespace stats
