#pragma once

/**
 * @file tests/benchmarks.h
 * @brief Benchmark-specific utilities and helpers for performance testing in test infrastructure
 *
 * This header provides specialized benchmarking utilities for the test infrastructure,
 * including performance measurement helpers, comparative benchmarks, and test-specific
 * benchmark configurations. These complement the main platform/benchmark.h infrastructure.
 *
 * Phase 3E: Test Infrastructure Namespace
 * Part of the stats::tests:: namespace hierarchy reorganization
 */

#include "../platform/benchmark.h"
#include "constants.h"

#include <chrono>
#include <functional>
#include <string>
#include <vector>

namespace stats {
namespace tests {
namespace benchmarks {

//==============================================================================
// Test-Specific Benchmark Configurations
//==============================================================================

/**
 * @brief Get benchmark configuration optimized for test environments
 */
struct TestBenchmarkConfig {
    std::size_t iterations;
    std::size_t warmup_runs;
    bool enable_warmup;
    std::chrono::milliseconds max_test_duration;

    static TestBenchmarkConfig getQuickConfig() {
        return {.iterations = constants::SMALL_BENCHMARK_ITERATIONS,
                .warmup_runs = 10,
                .enable_warmup = true,
                .max_test_duration = std::chrono::milliseconds(1000)};
    }

    static TestBenchmarkConfig getThoroughConfig() {
        return {.iterations = constants::LARGE_BENCHMARK_ITERATIONS,
                .warmup_runs = constants::BENCHMARK_WARMUP_ITERATIONS,
                .enable_warmup = true,
                .max_test_duration = std::chrono::milliseconds(10000)};
    }

    static TestBenchmarkConfig getDefaultConfig() {
        return {.iterations = constants::DEFAULT_BENCHMARK_ITERATIONS,
                .warmup_runs = constants::BENCHMARK_WARMUP_ITERATIONS,
                .enable_warmup = true,
                .max_test_duration = std::chrono::milliseconds(5000)};
    }
};

//==============================================================================
// Performance Comparison Utilities
//==============================================================================

/**
 * @brief Helper for comparing performance between different implementations
 */
class PerformanceComparator {
   public:
    struct ComparisonResult {
        std::string baseline_name;
        std::string comparison_name;
        double speedup_factor;
        double confidence_interval_lower;
        double confidence_interval_upper;
        bool is_significant_improvement;
        bool is_significant_regression;

        std::string toString() const;
    };

    /**
     * @brief Compare two benchmark results for statistical significance
     * @param baseline Baseline benchmark result
     * @param comparison Comparison benchmark result
     * @param significance_level Alpha level for statistical test (default 0.05)
     * @return Comparison result with statistical analysis
     */
    static ComparisonResult compare(const BenchmarkResult& baseline,
                                    const BenchmarkResult& comparison,
                                    double significance_level = 0.05);

    /**
     * @brief Compare multiple implementations against a baseline
     * @param baseline_name Name of baseline implementation
     * @param baseline_func Baseline function to benchmark
     * @param comparisons Vector of (name, function) pairs to compare
     * @param config Benchmark configuration to use
     * @return Vector of comparison results
     */
    static std::vector<ComparisonResult> compareMultiple(
        const std::string& baseline_name, std::function<void()> baseline_func,
        const std::vector<std::pair<std::string, std::function<void()>>>& comparisons,
        const TestBenchmarkConfig& config = TestBenchmarkConfig::getDefaultConfig());
};

//==============================================================================
// Distribution Performance Testing
//==============================================================================

/**
 * @brief Specialized benchmarking for distribution performance testing
 */
class DistributionBenchmarker {
   public:
    /**
     * @brief Benchmark different strategies for a distribution operation
     * @tparam Distribution Distribution type to benchmark
     * @param dist Distribution instance
     * @param test_data Input data for benchmarking
     * @param operation_name Name of the operation (PDF, CDF, etc.)
     * @param config Benchmark configuration
     * @return Map of strategy name to benchmark result
     */
    template <typename Distribution>
    static std::map<std::string, BenchmarkResult> benchmarkStrategies(
        const Distribution& dist, const std::vector<double>& test_data,
        const std::string& operation_name,
        const TestBenchmarkConfig& config = TestBenchmarkConfig::getDefaultConfig());

    /**
     * @brief Benchmark scaling behavior across different batch sizes
     * @tparam Distribution Distribution type to benchmark
     * @param dist Distribution instance
     * @param base_data Base test data to scale up
     * @param batch_sizes Vector of batch sizes to test
     * @param operation_name Name of the operation
     * @return Map of batch size to benchmark results
     */
    template <typename Distribution>
    static std::map<std::size_t, std::map<std::string, BenchmarkResult>> benchmarkScaling(
        const Distribution& dist, const std::vector<double>& base_data,
        const std::vector<std::size_t>& batch_sizes, const std::string& operation_name);
};

//==============================================================================
// Memory and Cache Benchmarking
//==============================================================================

/**
 * @brief Utilities for memory-aware benchmarking
 */
class MemoryBenchmarker {
   public:
    /**
     * @brief Benchmark with different memory access patterns
     * @param operation Function to benchmark
     * @param data_sizes Vector of data sizes to test
     * @param access_patterns Vector of access pattern names and functions
     * @return Benchmark results for each combination
     */
    static std::map<std::size_t, std::map<std::string, BenchmarkResult>> benchmarkMemoryPatterns(
        std::function<void(std::size_t)> operation, const std::vector<std::size_t>& data_sizes,
        const std::vector<std::pair<std::string, std::function<void(std::size_t)>>>&
            access_patterns);

    /**
     * @brief Estimate cache performance impact
     * @param cold_cache_func Function that clears/bypasses cache
     * @param warm_cache_func Function with warm cache
     * @param iterations Number of iterations to average
     * @return Cache speedup factor (warm/cold)
     */
    static double estimateCacheSpeedup(std::function<void()> cold_cache_func,
                                       std::function<void()> warm_cache_func,
                                       std::size_t iterations = 100);
};

//==============================================================================
// Regression Testing Utilities
//==============================================================================

/**
 * @brief Utilities for performance regression testing
 */
class RegressionBenchmarker {
   public:
    struct RegressionResult {
        std::string test_name;
        double current_performance;
        double baseline_performance;
        double regression_factor;
        bool is_regression;
        bool is_improvement;
        double threshold_used;

        std::string toString() const;
    };

    /**
     * @brief Check for performance regressions against historical baseline
     * @param test_name Name of the test
     * @param current_result Current benchmark result
     * @param baseline_result Historical baseline result
     * @param regression_threshold Threshold for considering something a regression (default 10%)
     * @return Regression analysis result
     */
    static RegressionResult checkRegression(const std::string& test_name,
                                            const BenchmarkResult& current_result,
                                            const BenchmarkResult& baseline_result,
                                            double regression_threshold = 0.10);

    /**
     * @brief Save benchmark results for future regression testing
     * @param results Vector of benchmark results to save
     * @param baseline_file File path to save baseline results
     */
    static void saveBaseline(const std::vector<BenchmarkResult>& results,
                             const std::string& baseline_file);

    /**
     * @brief Load baseline results for regression comparison
     * @param baseline_file File path to load baseline from
     * @return Vector of baseline benchmark results
     */
    static std::vector<BenchmarkResult> loadBaseline(const std::string& baseline_file);
};

//==============================================================================
// Quick Benchmark Utilities
//==============================================================================

/**
 * @brief Quick and simple benchmarking functions for common test scenarios
 */
class QuickBenchmark {
   public:
    /**
     * @brief Quick timing of a single operation
     * @param operation Function to time
     * @param iterations Number of iterations (0 = auto-detect)
     * @return Average time per iteration in seconds
     */
    static double timeOperation(std::function<void()> operation, std::size_t iterations = 0);

    /**
     * @brief Quick comparison of two operations
     * @param baseline_op Baseline operation
     * @param comparison_op Operation to compare
     * @param iterations Number of iterations for each
     * @return Speedup factor (baseline_time / comparison_time)
     */
    static double compareOperations(std::function<void()> baseline_op,
                                    std::function<void()> comparison_op,
                                    std::size_t iterations = 100);

    /**
     * @brief Quick throughput measurement
     * @param operation Operation to measure
     * @param operations_per_call Number of operations performed per function call
     * @param duration_seconds How long to run the benchmark
     * @return Operations per second
     */
    static double measureThroughput(std::function<void()> operation,
                                    std::size_t operations_per_call, double duration_seconds = 1.0);
};

//==============================================================================
// Benchmark Result Analysis
//==============================================================================

/**
 * @brief Utilities for analyzing and reporting benchmark results
 */
class BenchmarkAnalyzer {
   public:
    /**
     * @brief Generate summary report of benchmark results
     * @param results Vector of benchmark results to analyze
     * @return Formatted summary report string
     */
    static std::string generateSummaryReport(const std::vector<BenchmarkResult>& results);

    /**
     * @brief Find outliers in benchmark results
     * @param results Vector of benchmark results
     * @param outlier_threshold Z-score threshold for outlier detection (default 2.0)
     * @return Vector of result indices that are outliers
     */
    static std::vector<std::size_t> findOutliers(const std::vector<BenchmarkResult>& results,
                                                 double outlier_threshold = 2.0);

    /**
     * @brief Calculate coefficient of variation for benchmark stability
     * @param result Benchmark result to analyze
     * @return Coefficient of variation (stddev / mean)
     */
    static double calculateStability(const BenchmarkResult& result);

    /**
     * @brief Export results to CSV format for external analysis
     * @param results Vector of benchmark results
     * @param filename Output CSV filename
     * @param include_raw_data Whether to include raw timing data
     */
    static void exportToCSV(const std::vector<BenchmarkResult>& results,
                            const std::string& filename, bool include_raw_data = false);
};

}  // namespace benchmarks
}  // namespace tests
}  // namespace stats
