#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <span>
#include <iostream>
#include <iomanip>
#include "../include/platform/work_stealing_pool.h"
#include "../include/platform/adaptive_cache.h"

namespace libstats {
namespace testing {

//==============================================================================
// BENCHMARKING UTILITIES
//==============================================================================

struct BenchmarkResult {
    std::string operation_name;
    long simd_time_us;
    long parallel_time_us;
    long work_stealing_time_us;
    long cache_aware_time_us;
    double parallel_speedup;
    double work_stealing_speedup;
    double cache_aware_speedup;
};

class StandardizedBenchmark {
public:
    static void printBenchmarkHeader(const std::string& distribution_name, size_t dataset_size) {
        std::cout << "\n=== " << distribution_name << " Enhanced Performance Benchmark ===" << std::endl;
        std::cout << "Dataset size: " << dataset_size << " elements" << std::endl;
        std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << std::endl;
    }
    
    static void printBenchmarkResults(const std::vector<BenchmarkResult>& results) {
        std::cout << "\nBenchmark Results:" << std::endl;
        std::cout << std::setw(10) << "Operation" << std::setw(12) << "SIMD (μs)" 
                  << std::setw(15) << "Parallel (μs)" << std::setw(18) << "Work-Steal (μs)"
                  << std::setw(18) << "Cache-Aware (μs)" << std::setw(12) << "P-Speedup"
                  << std::setw(12) << "WS-Speedup" << std::setw(12) << "CA-Speedup" << std::endl;
        std::cout << std::string(120, '-') << std::endl;
        
        for (const auto& result : results) {
            std::cout << std::setw(10) << result.operation_name 
                      << std::setw(12) << result.simd_time_us
                      << std::setw(15) << result.parallel_time_us
                      << std::setw(18) << result.work_stealing_time_us
                      << std::setw(18) << result.cache_aware_time_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.parallel_speedup
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.work_stealing_speedup
                      << std::setw(12) << std::fixed << std::setprecision(2) << result.cache_aware_speedup
                      << std::endl;
        }
    }
    
    static void printPerformanceAnalysis(const std::vector<BenchmarkResult>& results) {
        std::cout << "\nPerformance Analysis:" << std::endl;
        
        if (std::thread::hardware_concurrency() > 2) {
            for (const auto& result : results) {
                if (result.parallel_speedup > 0.8) {
                    std::cout << "  ✓ " << result.operation_name << " parallel shows good speedup" << std::endl;
                } else {
                    std::cout << "  ⚠ " << result.operation_name << " parallel speedup lower than expected" << std::endl;
                }
            }
        } else {
            std::cout << "  ℹ Single/dual-core system - parallel overhead may dominate" << std::endl;
        }
    }
};

//==============================================================================
// STATISTICAL UTILITIES
//==============================================================================

class StatisticalTestUtils {
public:
    static std::pair<double, double> calculateSampleStats(const std::vector<double>& samples) {
        double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        double variance = 0.0;
        for (double x : samples) {
            variance += (x - mean) * (x - mean);
        }
        variance /= samples.size();
        return {mean, variance};
    }
    
    static bool approxEqual(double a, double b, double tolerance = 1e-12) {
        return std::abs(a - b) <= tolerance;
    }
    
    template<typename Distribution>
    static void verifyBatchCorrectness(const Distribution& dist, 
                                     const std::vector<double>& test_values,
                                     const std::vector<double>& batch_results,
                                     const std::string& operation_name,
                                     double tolerance = 1e-10) {
        bool all_correct = true;
        const size_t check_count = std::min(size_t(100), test_values.size());
        
        for (size_t i = 0; i < check_count; ++i) {
            size_t idx = i * (test_values.size() / check_count);
            double expected;
            
            if (operation_name == "PDF") {
                expected = dist.getProbability(test_values[idx]);
            } else if (operation_name == "LogPDF") {
                expected = dist.getLogProbability(test_values[idx]);
            } else if (operation_name == "CDF") {
                expected = dist.getCumulativeProbability(test_values[idx]);
            } else {
                continue; // Skip unknown operations
            }
            
            if (std::abs(batch_results[idx] - expected) > tolerance) {
                all_correct = false;
                break;
            }
        }
        
        if (all_correct) {
            std::cout << "  ✓ " << operation_name << " batch operations produce correct results" << std::endl;
        } else {
            std::cout << "  ✗ " << operation_name << " batch correctness check failed" << std::endl;
        }
        
        EXPECT_TRUE(all_correct) << operation_name << " batch operations should produce correct results";
    }
};

//==============================================================================
// THREAD SAFETY TESTING
//==============================================================================

template<typename Distribution>
class ThreadSafetyTester {
public:
    static void testBasicThreadSafety(const Distribution& dist, const std::string& dist_name) {
        const int num_threads = 4;
        const int samples_per_thread = 1000;
        
        std::vector<std::thread> threads;
        std::vector<std::vector<double>> results(num_threads);
        
        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&dist, &results, t, samples_per_thread]() {
                std::mt19937 local_rng(42 + t);
                results[t].reserve(samples_per_thread);
                
                for (int i = 0; i < samples_per_thread; ++i) {
                    results[t].push_back(dist.sample(local_rng));
                }
            });
        }
        
        for (auto& thread : threads) {
            thread.join();
        }
        
        // Verify that all threads produced valid results
        for (int t = 0; t < num_threads; ++t) {
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

template<typename Distribution>
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
        dist.getProbabilityBatch(empty_values.data(), empty_results.data(), 0);
        dist.getLogProbabilityBatch(empty_values.data(), empty_results.data(), 0);
        dist.getCumulativeProbabilityBatch(empty_values.data(), empty_results.data(), 0);
        
        std::cout << "  ✓ " << dist_name << " empty batch operations handled gracefully" << std::endl;
    }
};

} // namespace testing
} // namespace libstats
