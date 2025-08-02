/**
 * @file exponential_performance_benchmark.cpp
 * @brief Comprehensive performance benchmark for enhanced Exponential distribution
 * 
 * This benchmark demonstrates and measures performance of all enhanced features
 * in the Exponential distribution implementation, including:
 * - Basic probability functions (PDF, CDF, quantiles, log PDF)
 * - SIMD-optimized batch operations
 * - Parallel processing with thread pools
 * - Work-stealing dynamic load balancing
 * - Cache-aware processing
 * - Parameter fitting with parallel algorithms
 * - Inverse transform sampling optimization
 * - Advanced statistical methods (confidence intervals, Bayesian estimation, etc.)
 * 
 * @author libstats Development Team
 * @version 1.0.0
 */

#include "../include/distributions/exponential.h"
#include "../include/platform/benchmark.h"
#include "../include/platform/adaptive_cache.h"
#include "../include/platform/work_stealing_pool.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

using namespace libstats;

int main() {
    std::cout << "\n🔬 === EXPONENTIAL DISTRIBUTION COMPREHENSIVE PERFORMANCE BENCHMARK ===\n";
    std::cout << "\n📊 The Exponential distribution models waiting times and lifetimes,\n";
    std::cout << "   crucial for queuing theory, reliability analysis, and survival studies.\n";
    std::cout << "\n🎯 This benchmark evaluates:\n";
    std::cout << "   • Basic probability functions (PDF, CDF, quantiles)\n";
    std::cout << "   • SIMD-vectorized batch operations\n";
    std::cout << "   • Parallel processing with dynamic load balancing\n";
    std::cout << "   • Inverse transform sampling (highly efficient for exponentials)\n";
    std::cout << "   • Parameter estimation and statistical inference\n";
    std::cout << "\n⚡ Key computational advantages for Exponential distributions:\n";
    std::cout << "   • Simple closed-form PDF: f(x) = λ * exp(-λ*x)\n";
    std::cout << "   • Closed-form CDF: F(x) = 1 - exp(-λ*x)\n";
    std::cout << "   • Trivial inverse CDF for sampling: -ln(1-u)/λ\n";
    std::cout << "   • Memoryless property simplifies many calculations\n\n";
    std::cout << "Testing all enhanced features with performance measurements\n" << std::endl;
    
    // Create Exponential distributions for testing
    ExponentialDistribution unitExponential(1.0);  // Unit exponential (λ = 1)
    ExponentialDistribution customExponential(2.5);  // Custom distribution (λ = 2.5)
    
    // Benchmark setup
    Benchmark bench(true, 10, 3);  // Warmup enabled, 10 iterations, 3 warmup runs
    
    //==========================================================================
    // 1. BASIC OPERATIONS BENCHMARK
    //==========================================================================
    std::cout << "\n📋 Phase 1: Setting up basic operations benchmarks..." << std::endl;
    std::cout << "   Testing individual probability computations with closed-form solutions.\n" << std::endl;
    
    // Single value operations
    bench.addTest("PDF Single Value", [&]() {
        volatile double result = unitExponential.getProbability(1.0);
        (void)result;  // Prevent optimization
    }, 0, 1000000.0);  // 1M operations for throughput measurement
    
    bench.addTest("CDF Single Value", [&]() {
        volatile double result = unitExponential.getCumulativeProbability(0.5);
        (void)result;
    }, 0, 1000000.0);
    
    bench.addTest("Log PDF Single Value", [&]() {
        volatile double result = unitExponential.getLogProbability(1.0);
        (void)result;
    }, 0, 1000000.0);
    
    bench.addTest("Quantile Single Value", [&]() {
        volatile double result = unitExponential.getQuantile(0.75);
        (void)result;
    }, 0, 1000000.0);
    
    //==========================================================================
    // 2. BATCH OPERATIONS BENCHMARK 
    //==========================================================================
    std::cout << "\n⚡ Phase 2: Setting up batch operations benchmarks..." << std::endl;
    std::cout << "   Testing SIMD-vectorized exp() operations on exponential data arrays.\n" << std::endl;
    
    // Create test data - exponential distribution domain is [0, ∞)
    const std::vector<size_t> test_sizes = {100, 1000, 10000, 100000};
    
    for (size_t size : test_sizes) {
        std::vector<double> test_values(size);
        std::vector<double> results(size);
        
        // Fill test data with positive random values suitable for exponential domain
        std::mt19937 rng(42);
        std::exponential_distribution<double> exp_dist(0.5);  // Generate exponential-like test data
        for (size_t i = 0; i < size; ++i) {
            test_values[i] = exp_dist(rng);
        }
        
        // Scalar batch operations
        bench.addTest("Batch PDF Scalar " + std::to_string(size), [&, test_values, results, size]() mutable {
            unitExponential.getProbabilityBatch(test_values.data(), results.data(), size);
        }, 0, static_cast<double>(size));
        
        bench.addTest("Batch Log PDF Scalar " + std::to_string(size), [&, test_values, results, size]() mutable {
            unitExponential.getLogProbabilityBatch(test_values.data(), results.data(), size);
        }, 0, static_cast<double>(size));
        
        bench.addTest("Batch CDF Scalar " + std::to_string(size), [&, test_values, results, size]() mutable {
            unitExponential.getCumulativeProbabilityBatch(test_values.data(), results.data(), size);
        }, 0, static_cast<double>(size));
        
        // Parallel batch operations
        bench.addTest("Batch PDF Parallel " + std::to_string(size), [&, test_values, results]() mutable {
            std::span<const double> values_span(test_values);
            std::span<double> results_span(results);
            unitExponential.getProbabilityBatchParallel(values_span, results_span);
        }, 0, static_cast<double>(size));
        
        bench.addTest("Batch Log PDF Parallel " + std::to_string(size), [&, test_values, results]() mutable {
            std::span<const double> values_span(test_values);
            std::span<double> results_span(results);
            unitExponential.getLogProbabilityBatchParallel(values_span, results_span);
        }, 0, static_cast<double>(size));
        
        bench.addTest("Batch CDF Parallel " + std::to_string(size), [&, test_values, results]() mutable {
            std::span<const double> values_span(test_values);
            std::span<double> results_span(results);
            unitExponential.getCumulativeProbabilityBatchParallel(values_span, results_span);
        }, 0, static_cast<double>(size));
    }
    
    //==========================================================================
    // 3. ADVANCED FEATURES BENCHMARK
    //==========================================================================
    std::cout << "\n💻 Phase 3: Setting up advanced features benchmarks..." << std::endl;
    std::cout << "   Testing cache-aware processing and work-stealing parallelism.\n" << std::endl;
    
    // Cache-aware operations
    cache::AdaptiveCache<std::string, double> cache_manager;
    const std::vector<double> large_test(10000);
    std::vector<double> large_results(10000);
    
    // Fill with exponential-like test data
    std::mt19937 cache_rng(42);
    std::exponential_distribution<double> cache_exp_dist(1.0);
    std::vector<double> large_test_mutable = large_test;
    for (size_t i = 0; i < large_test_mutable.size(); ++i) {
        large_test_mutable[i] = cache_exp_dist(cache_rng);
    }
    
    bench.addTest("Cache-Aware Batch PDF", [&, large_test_mutable]() mutable {
        std::span<const double> values_span(large_test_mutable);
        std::span<double> results_span(large_results);
        unitExponential.getProbabilityBatchCacheAware(values_span, results_span, cache_manager);
    }, 0, static_cast<double>(large_test.size()));
    
    // Work-stealing parallel operations
    WorkStealingPool work_pool(std::thread::hardware_concurrency());
    
    bench.addTest("Work-Stealing Batch PDF", [&, large_test_mutable]() mutable {
        std::span<const double> values_span(large_test_mutable);
        std::span<double> results_span(large_results);
        unitExponential.getProbabilityBatchWorkStealing(values_span, results_span, work_pool);
    }, 0, static_cast<double>(large_test.size()));
    
    //==========================================================================
    // 4. SAMPLING BENCHMARK
    //==========================================================================
    std::cout << "\n🎲 Phase 4: Setting up sampling benchmarks..." << std::endl;
    std::cout << "   Testing highly efficient inverse transform sampling for exponentials.\n" << std::endl;
    
    std::mt19937 sample_rng(42);
    
    bench.addTest("Inverse Transform Sampling Unit Exp", [&]() mutable {
        volatile double result = unitExponential.sample(sample_rng);
        (void)result;
    }, 0, 1000000.0);
    
    bench.addTest("Inverse Transform Sampling Custom Exp", [&]() mutable {
        volatile double result = customExponential.sample(sample_rng);
        (void)result;
    }, 0, 1000000.0);
    
    //==========================================================================
    // 5. PARAMETER FITTING BENCHMARK  
    //==========================================================================
    std::cout << "\n🔧 Phase 5: Setting up fitting benchmarks..." << std::endl;
    std::cout << "   Testing maximum likelihood estimation for exponential rate parameter.\n" << std::endl;
    
    // Generate realistic test data for fitting using true exponential distribution
    std::vector<double> fit_data_small(1000);
    std::vector<double> fit_data_large(100000);
    
    std::exponential_distribution<double> true_exp_gen(1.5);  // Rate = 1.5
    for (size_t i = 0; i < fit_data_small.size(); ++i) {
        fit_data_small[i] = true_exp_gen(sample_rng);
    }
    for (size_t i = 0; i < fit_data_large.size(); ++i) {
        fit_data_large[i] = true_exp_gen(sample_rng);
    }
    
    bench.addTest("Parameter Fitting Small Dataset", [&]() {
        ExponentialDistribution temp_dist(1.0);
        temp_dist.fit(fit_data_small);
    }, 0, static_cast<double>(fit_data_small.size()));
    
    bench.addTest("Parameter Fitting Large Dataset", [&]() {
        ExponentialDistribution temp_dist(1.0);
        temp_dist.fit(fit_data_large);
    }, 0, static_cast<double>(fit_data_large.size()));
    
    //==========================================================================
    // 6. ADVANCED STATISTICAL METHODS BENCHMARK
    //==========================================================================
    std::cout << "\n📊 Phase 6: Setting up advanced statistical methods benchmarks..." << std::endl;
    std::cout << "   Testing confidence intervals, hypothesis tests, and Bayesian methods.\n" << std::endl;
    
    // Use medium-sized dataset for statistical methods
    std::vector<double> stats_data(5000);
    for (size_t i = 0; i < stats_data.size(); ++i) {
        stats_data[i] = true_exp_gen(sample_rng);
    }
    
    bench.addTest("Confidence Interval Rate", [&]() {
        volatile auto result = ExponentialDistribution::confidenceIntervalRate(stats_data, 0.95);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
    bench.addTest("Likelihood Ratio Test", [&]() {
        volatile auto result = ExponentialDistribution::likelihoodRatioTest(stats_data, 1.5, 0.05);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
    bench.addTest("Bayesian Estimation", [&]() {
        volatile auto result = ExponentialDistribution::bayesianEstimation(stats_data, 1.0, 1.0);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
    bench.addTest("Method of Moments Estimation", [&]() {
        volatile double result = ExponentialDistribution::methodOfMomentsEstimation(stats_data);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
    bench.addTest("Robust Estimation (Winsorized)", [&]() {
        volatile double result = ExponentialDistribution::robustEstimation(stats_data, "winsorized", 0.1);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
    //==========================================================================
    // RUN BENCHMARKS
    //==========================================================================
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "RUNNING EXPONENTIAL DISTRIBUTION BENCHMARKS" << std::endl;
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
    double single_cdf_ops_per_sec = 0.0;
    double single_log_pdf_ops_per_sec = 0.0;
    double single_quantile_ops_per_sec = 0.0;
    double batch_pdf_1k_scalar = 0.0;
    double batch_pdf_1k_parallel = 0.0;
    double batch_log_pdf_1k_scalar = 0.0;
    double batch_log_pdf_1k_parallel = 0.0;
    double batch_cdf_1k_scalar = 0.0;
    double batch_cdf_1k_parallel = 0.0;
    double batch_pdf_100k_scalar = 0.0;
    double batch_pdf_100k_parallel = 0.0;
    double batch_log_pdf_100k_scalar = 0.0;
    double batch_log_pdf_100k_parallel = 0.0;
    double batch_cdf_100k_scalar = 0.0;
    double batch_cdf_100k_parallel = 0.0;
    double inverse_transform_unit = 0.0;
    double inverse_transform_custom = 0.0;
    double cache_aware_ops_per_sec = 0.0;
    double work_stealing_ops_per_sec = 0.0;
    double fitting_small_ops_per_sec = 0.0;
    double fitting_large_ops_per_sec = 0.0;
    double confidence_interval_ops_per_sec = 0.0;
    double likelihood_ratio_ops_per_sec = 0.0;
    double bayesian_estimation_ops_per_sec = 0.0;
    double moments_estimation_ops_per_sec = 0.0;
    double robust_estimation_ops_per_sec = 0.0;
    
    for (const auto& result : results) {
        if (result.name == "PDF Single Value") {
            single_pdf_ops_per_sec = result.stats.throughput;
        } else if (result.name == "CDF Single Value") {
            single_cdf_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Log PDF Single Value") {
            single_log_pdf_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Quantile Single Value") {
            single_quantile_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Batch PDF Scalar 1000") {
            batch_pdf_1k_scalar = result.stats.throughput;
        } else if (result.name == "Batch PDF Parallel 1000") {
            batch_pdf_1k_parallel = result.stats.throughput;
        } else if (result.name == "Batch Log PDF Scalar 1000") {
            batch_log_pdf_1k_scalar = result.stats.throughput;
        } else if (result.name == "Batch Log PDF Parallel 1000") {
            batch_log_pdf_1k_parallel = result.stats.throughput;
        } else if (result.name == "Batch CDF Scalar 1000") {
            batch_cdf_1k_scalar = result.stats.throughput;
        } else if (result.name == "Batch CDF Parallel 1000") {
            batch_cdf_1k_parallel = result.stats.throughput;
        } else if (result.name == "Batch PDF Scalar 100000") {
            batch_pdf_100k_scalar = result.stats.throughput;
        } else if (result.name == "Batch PDF Parallel 100000") {
            batch_pdf_100k_parallel = result.stats.throughput;
        } else if (result.name == "Batch Log PDF Scalar 100000") {
            batch_log_pdf_100k_scalar = result.stats.throughput;
        } else if (result.name == "Batch Log PDF Parallel 100000") {
            batch_log_pdf_100k_parallel = result.stats.throughput;
        } else if (result.name == "Batch CDF Scalar 100000") {
            batch_cdf_100k_scalar = result.stats.throughput;
        } else if (result.name == "Batch CDF Parallel 100000") {
            batch_cdf_100k_parallel = result.stats.throughput;
        } else if (result.name == "Inverse Transform Sampling Unit Exp") {
            inverse_transform_unit = result.stats.throughput;
        } else if (result.name == "Inverse Transform Sampling Custom Exp") {
            inverse_transform_custom = result.stats.throughput;
        } else if (result.name == "Cache-Aware Batch PDF") {
            cache_aware_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Work-Stealing Batch PDF") {
            work_stealing_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Parameter Fitting Small Dataset") {
            fitting_small_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Parameter Fitting Large Dataset") {
            fitting_large_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Confidence Interval Rate") {
            confidence_interval_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Likelihood Ratio Test") {
            likelihood_ratio_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Bayesian Estimation") {
            bayesian_estimation_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Method of Moments Estimation") {
            moments_estimation_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Robust Estimation (Winsorized)") {
            robust_estimation_ops_per_sec = result.stats.throughput;
        }
    }
    
    std::cout << "\nSINGLE OPERATION PERFORMANCE:" << std::endl;
    std::cout << "├─ PDF Operations:      " << std::scientific << single_pdf_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "├─ CDF Operations:      " << std::scientific << single_cdf_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "├─ Log PDF Operations:  " << std::scientific << single_log_pdf_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "└─ Quantile Operations: " << std::scientific << single_quantile_ops_per_sec << " ops/sec" << std::endl;
    
    std::cout << "\nBATCH OPERATION PERFORMANCE (1K Elements):" << std::endl;
    std::cout << "├─ Scalar PDF:          " << std::scientific << batch_pdf_1k_scalar << " elements/sec" << std::endl;
    std::cout << "├─ Parallel PDF:        " << std::scientific << batch_pdf_1k_parallel << " elements/sec" << std::endl;
    std::cout << "├─ Scalar Log PDF:      " << std::scientific << batch_log_pdf_1k_scalar << " elements/sec" << std::endl;
    std::cout << "├─ Parallel Log PDF:    " << std::scientific << batch_log_pdf_1k_parallel << " elements/sec" << std::endl;
    std::cout << "├─ Scalar CDF:          " << std::scientific << batch_cdf_1k_scalar << " elements/sec" << std::endl;
    std::cout << "└─ Parallel CDF:        " << std::scientific << batch_cdf_1k_parallel << " elements/sec" << std::endl;
    
    std::cout << "\nBATCH OPERATION PERFORMANCE (100K Elements):" << std::endl;
    std::cout << "├─ Scalar PDF:          " << std::scientific << batch_pdf_100k_scalar << " elements/sec" << std::endl;
    std::cout << "├─ Parallel PDF:        " << std::scientific << batch_pdf_100k_parallel << " elements/sec" << std::endl;
    std::cout << "├─ Scalar Log PDF:      " << std::scientific << batch_log_pdf_100k_scalar << " elements/sec" << std::endl;
    std::cout << "├─ Parallel Log PDF:    " << std::scientific << batch_log_pdf_100k_parallel << " elements/sec" << std::endl;
    std::cout << "├─ Scalar CDF:          " << std::scientific << batch_cdf_100k_scalar << " elements/sec" << std::endl;
    std::cout << "├─ Parallel CDF:        " << std::scientific << batch_cdf_100k_parallel << " elements/sec" << std::endl;
    std::cout << "├─ Cache-Aware PDF:     " << std::scientific << cache_aware_ops_per_sec << " elements/sec" << std::endl;
    std::cout << "└─ Work-Stealing PDF:   " << std::scientific << work_stealing_ops_per_sec << " elements/sec" << std::endl;
    
    std::cout << "\nSAMPLING AND FITTING PERFORMANCE:" << std::endl;
    std::cout << "├─ Inverse Transform (Unit):     " << std::scientific << inverse_transform_unit << " samples/sec" << std::endl;
    std::cout << "├─ Inverse Transform (Custom):   " << std::scientific << inverse_transform_custom << " samples/sec" << std::endl;
    std::cout << "├─ Parameter Fitting (Small):    " << std::scientific << fitting_small_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "└─ Parameter Fitting (Large):    " << std::scientific << fitting_large_ops_per_sec << " datapoints/sec" << std::endl;
    
    std::cout << "\nADVANCED STATISTICAL METHODS:" << std::endl;
    std::cout << "├─ Confidence Intervals:         " << std::scientific << confidence_interval_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "├─ Likelihood Ratio Tests:       " << std::scientific << likelihood_ratio_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "├─ Bayesian Estimation:          " << std::scientific << bayesian_estimation_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "├─ Method of Moments:            " << std::scientific << moments_estimation_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "└─ Robust Estimation:            " << std::scientific << robust_estimation_ops_per_sec << " datapoints/sec" << std::endl;
    
    if (batch_pdf_100k_parallel > 0 && batch_pdf_100k_scalar > 0) {
        double speedup = batch_pdf_100k_parallel / batch_pdf_100k_scalar;
        std::cout << "\nPARALLEL SPEEDUP: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    // Exponential-specific performance insights
    std::cout << "\nEXPONENTIAL DISTRIBUTION INSIGHTS:" << std::endl;
    std::cout << "├─ Unit rate optimization: Fast path for λ=1" << std::endl;
    std::cout << "├─ SIMD vectorization: Optimized exp() operations" << std::endl;
    std::cout << "├─ Inverse transform sampling: Single log() + division" << std::endl;
    std::cout << "├─ Advanced statistics: Gamma conjugate priors" << std::endl;
    std::cout << "└─ Numerical stability: Handles extreme λ values" << std::endl;
    
    std::cout << "\n🚀 EXPONENTIAL DISTRIBUTION BENCHMARK COMPLETE! 🚀" << std::endl;
    std::cout << "All enhanced features successfully benchmarked." << std::endl;
    
    return 0;
}
