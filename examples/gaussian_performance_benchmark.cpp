/**
 * @file gaussian_performance_benchmark.cpp
 * @brief Comprehensive performance benchmark for enhanced Gaussian distribution
 * 
 * This benchmark demonstrates and measures performance of all enhanced features
 * in the Gaussian distribution implementation, including:
 * - Basic probability functions (PDF, CDF, quantiles)
 * - SIMD-optimized batch operations
 * - Parallel processing with thread pools
 * - Work-stealing dynamic load balancing
 * - Cache-aware processing
 * - Parameter fitting with parallel algorithms
 * - Box-Muller sampling optimization
 * 
 * @author libstats Development Team
 * @version 1.0.0
 */

#include "../include/gaussian.h"
#include "../include/benchmark.h"
#include "../include/adaptive_cache.h"
#include "../include/work_stealing_pool.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace libstats;

int main() {
    std::cout << "=== GAUSSIAN DISTRIBUTION COMPREHENSIVE PERFORMANCE BENCHMARK ===" << std::endl;
    std::cout << "Testing all enhanced features with performance measurements\n" << std::endl;
    
    // Create Gaussian distributions for testing
    GaussianDistribution stdNormal(0.0, 1.0);  // Standard normal
    GaussianDistribution customGaussian(10.0, 2.5);  // Custom distribution
    
    // Benchmark setup
    Benchmark bench(true, 10, 3);  // Warmup enabled, 10 iterations, 3 warmup runs
    
    //==========================================================================
    // 1. BASIC OPERATIONS BENCHMARK
    //==========================================================================
    std::cout << "Setting up basic operations benchmarks..." << std::endl;
    
    // Single value operations
    bench.addTest("PDF Single Value", [&]() {
        volatile double result = stdNormal.getProbability(1.0);
        (void)result;  // Prevent optimization
    }, 0, 1000000.0);  // 1M operations for throughput measurement
    
    bench.addTest("CDF Single Value", [&]() {
        volatile double result = stdNormal.getCumulativeProbability(0.5);
        (void)result;
    }, 0, 1000000.0);
    
    bench.addTest("Log PDF Single Value", [&]() {
        volatile double result = stdNormal.getLogProbability(1.0);
        (void)result;
    }, 0, 1000000.0);
    
    bench.addTest("Quantile Single Value", [&]() {
        volatile double result = stdNormal.getQuantile(0.75);
        (void)result;
    }, 0, 1000000.0);
    
    //==========================================================================
    // 2. BATCH OPERATIONS BENCHMARK 
    //==========================================================================
    std::cout << "Setting up batch operations benchmarks..." << std::endl;
    
    // Create test data
    const std::vector<size_t> test_sizes = {100, 1000, 10000, 100000};
    
    for (size_t size : test_sizes) {
        std::vector<double> test_values(size);
        std::vector<double> results(size);
        
        // Fill test data with random values
        std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist(-3.0, 3.0);
        for (size_t i = 0; i < size; ++i) {
            test_values[i] = dist(rng);
        }
        
        // Scalar batch operations
        bench.addTest("Batch PDF Scalar " + std::to_string(size), [&, test_values, results, size]() mutable {
            stdNormal.getProbabilityBatch(test_values.data(), results.data(), size);
        }, 0, static_cast<double>(size));
        
        // Parallel batch operations
        bench.addTest("Batch PDF Parallel " + std::to_string(size), [&, test_values, results]() mutable {
            std::span<const double> values_span(test_values);
            std::span<double> results_span(results);
            stdNormal.getProbabilityBatchParallel(values_span, results_span);
        }, 0, static_cast<double>(size));
    }
    
    //==========================================================================
    // 3. ADVANCED FEATURES BENCHMARK
    //==========================================================================
    std::cout << "Setting up advanced features benchmarks..." << std::endl;
    
    // Cache-aware operations
    cache::AdaptiveCache<std::string, double> cache_manager;
    const std::vector<double> large_test(10000, 1.0);
    std::vector<double> large_results(10000);
    
    bench.addTest("Cache-Aware Batch PDF", [&]() mutable {
        std::span<const double> values_span(large_test);
        std::span<double> results_span(large_results);
        stdNormal.getProbabilityBatchCacheAware(values_span, results_span, cache_manager);
    }, 0, static_cast<double>(large_test.size()));
    
    // Work-stealing parallel operations
    WorkStealingPool work_pool(std::thread::hardware_concurrency());
    
    bench.addTest("Work-Stealing Batch PDF", [&]() mutable {
        std::span<const double> values_span(large_test);
        std::span<double> results_span(large_results);
        stdNormal.getProbabilityBatchWorkStealing(values_span, results_span, work_pool);
    }, 0, static_cast<double>(large_test.size()));
    
    //==========================================================================
    // 4. SAMPLING BENCHMARK
    //==========================================================================
    std::cout << "Setting up sampling benchmarks..." << std::endl;
    
    std::mt19937 sample_rng(42);
    
    bench.addTest("Box-Muller Sampling", [&]() mutable {
        volatile double result = stdNormal.sample(sample_rng);
        (void)result;
    }, 0, 1000000.0);
    
    //==========================================================================
    // 5. PARAMETER FITTING BENCHMARK  
    //==========================================================================
    std::cout << "Setting up fitting benchmarks..." << std::endl;
    
    // Generate realistic test data for fitting
    std::vector<double> fit_data_small(1000);
    std::vector<double> fit_data_large(100000);
    
    std::normal_distribution<double> normal_gen(5.0, 1.5);
    for (size_t i = 0; i < fit_data_small.size(); ++i) {
        fit_data_small[i] = normal_gen(sample_rng);
    }
    for (size_t i = 0; i < fit_data_large.size(); ++i) {
        fit_data_large[i] = normal_gen(sample_rng);
    }
    
    bench.addTest("Parameter Fitting Small Dataset", [&]() {
        GaussianDistribution temp_dist;
        temp_dist.fit(fit_data_small);
    }, 0, static_cast<double>(fit_data_small.size()));
    
    bench.addTest("Parameter Fitting Large Dataset", [&]() {
        GaussianDistribution temp_dist;
        temp_dist.fit(fit_data_large);
    }, 0, static_cast<double>(fit_data_large.size()));
    
    //==========================================================================
    // RUN BENCHMARKS
    //==========================================================================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "RUNNING GAUSSIAN DISTRIBUTION BENCHMARKS" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    auto results = bench.runAll();
    bench.printResults();
    
    //==========================================================================
    // ANALYSIS AND SUMMARY
    //==========================================================================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "PERFORMANCE ANALYSIS SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    // Find key performance metrics
    double single_pdf_ops_per_sec = 0.0;
    double batch_1k_ops_per_sec = 0.0;
    double batch_100k_ops_per_sec = 0.0;
    double parallel_100k_ops_per_sec = 0.0;
    
    for (const auto& result : results) {
        if (result.name == "PDF Single Value") {
            single_pdf_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Batch PDF Scalar 1000") {
            batch_1k_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Batch PDF Scalar 100000") {
            batch_100k_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Batch PDF Parallel 100000") {
            parallel_100k_ops_per_sec = result.stats.throughput;
        }
    }
    
    std::cout << "\nKEY PERFORMANCE METRICS:" << std::endl;
    std::cout << "├─ Single PDF Operations:      " << std::scientific << single_pdf_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "├─ Batch PDF (1K elements):    " << std::scientific << batch_1k_ops_per_sec << " elements/sec" << std::endl;
    std::cout << "├─ Batch PDF (100K elements):  " << std::scientific << batch_100k_ops_per_sec << " elements/sec" << std::endl;
    std::cout << "└─ Parallel PDF (100K elements): " << std::scientific << parallel_100k_ops_per_sec << " elements/sec" << std::endl;
    
    if (parallel_100k_ops_per_sec > 0 && batch_100k_ops_per_sec > 0) {
        double speedup = parallel_100k_ops_per_sec / batch_100k_ops_per_sec;
        std::cout << "\nPARALLEL SPEEDUP: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    std::cout << "\n🚀 GAUSSIAN DISTRIBUTION BENCHMARK COMPLETE! 🚀" << std::endl;
    std::cout << "All enhanced features successfully benchmarked." << std::endl;
    
    return 0;
}
