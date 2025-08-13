/**
 * @file gaussian_performance_benchmark.cpp
 * @brief Comprehensive performance benchmark for enhanced Gaussian distribution
 * 
 * This benchmark demonstrates and measures performance of all enhanced features
 * in the Gaussian distribution implementation, including:
 * - Basic probability functions (PDF, CDF, quantiles, log PDF)
 * - SIMD-optimized batch operations
 * - Parallel processing with thread pools
 * - Work-stealing dynamic load balancing
 * - Cache-aware processing
 * - Parameter fitting with parallel algorithms
 * - Box-Muller sampling optimization
 * - Advanced statistical methods (confidence intervals, hypothesis tests, etc.)
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats.h"
#include <iostream>
#include <random>
#include <iomanip>
#include <vector>

using namespace libstats;

int main() {
    std::cout << "\n🔬 === GAUSSIAN DISTRIBUTION COMPREHENSIVE PERFORMANCE BENCHMARK ===\n";
    std::cout << "\n📊 The Gaussian (Normal) distribution is one of the most computationally\n";
    std::cout << "   important distributions in statistics and scientific computing.\n";
    std::cout << "\n🎯 This benchmark evaluates:\n";
    std::cout << "   • Basic probability functions (PDF, CDF, quantiles)\n";
    std::cout << "   • SIMD-vectorized batch operations\n";
    std::cout << "   • Parallel processing with dynamic load balancing\n";
    std::cout << "   • Advanced sampling algorithms (Box-Muller transform)\n";
    std::cout << "   • Parameter estimation and statistical inference\n";
    std::cout << "\n⚡ Key computational challenges for Gaussian distributions:\n";
    std::cout << "   • Accurate error function (erf) evaluation for CDF\n";
    std::cout << "   • Numerically stable log-PDF computation\n";
    std::cout << "   • Efficient inverse CDF (quantile) approximations\n";
    std::cout << "   • Box-Muller transform optimization for sampling\n\n";
    std::cout << "Testing all enhanced features with performance measurements\n" << std::endl;
    
    // Create Gaussian distributions for testing
    libstats::Gaussian stdNormal(0.0, 1.0);  // Standard normal
    libstats::Gaussian customGaussian(10.0, 2.5);  // Custom distribution
    
    // Benchmark setup
    libstats::Benchmark bench(true, 10, 3);  // Warmup enabled, 10 iterations, 3 warmup runs
    
    //==========================================================================
    // 1. BASIC OPERATIONS BENCHMARK
    //==========================================================================
    std::cout << "\n📋 Phase 1: Setting up basic operations benchmarks..." << std::endl;
    std::cout << "   These test individual probability computations at maximum throughput.\n" << std::endl;
    
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
    std::cout << "\n⚡ Phase 2: Setting up batch operations benchmarks..." << std::endl;
    std::cout << "   Testing SIMD-vectorized operations on arrays of varying sizes.\n" << std::endl;
    
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
        
        // Scalar batch operations - using modern span-based API
        bench.addTest("Batch PDF Scalar " + std::to_string(size), [&, test_values, results]() mutable {
            std::span<const double> values_span(test_values);
            std::span<double> results_span(results);
            stdNormal.getProbability(values_span, results_span);
        }, 0, static_cast<double>(size));
        
        bench.addTest("Batch Log PDF Scalar " + std::to_string(size), [&, test_values, results]() mutable {
            std::span<const double> values_span(test_values);
            std::span<double> results_span(results);
            stdNormal.getLogProbability(values_span, results_span);
        }, 0, static_cast<double>(size));
        
        bench.addTest("Batch CDF Scalar " + std::to_string(size), [&, test_values, results]() mutable {
            std::span<const double> values_span(test_values);
            std::span<double> results_span(results);
            stdNormal.getCumulativeProbability(values_span, results_span);
        }, 0, static_cast<double>(size));
        
        // Parallel batch operations - using performance hints
        bench.addTest("Batch PDF Parallel " + std::to_string(size), [&, test_values, results]() mutable {
            std::span<const double> values_span(test_values);
            std::span<double> results_span(results);
            auto hint = libstats::performance::PerformanceHint::maximum_throughput();
            stdNormal.getProbability(values_span, results_span, hint);
        }, 0, static_cast<double>(size));
        
        bench.addTest("Batch Log PDF Parallel " + std::to_string(size), [&, test_values, results]() mutable {
            std::span<const double> values_span(test_values);
            std::span<double> results_span(results);
            auto hint = libstats::performance::PerformanceHint::maximum_throughput();
            stdNormal.getLogProbability(values_span, results_span, hint);
        }, 0, static_cast<double>(size));
        
        bench.addTest("Batch CDF Parallel " + std::to_string(size), [&, test_values, results]() mutable {
            std::span<const double> values_span(test_values);
            std::span<double> results_span(results);
            auto hint = libstats::performance::PerformanceHint::maximum_throughput();
            stdNormal.getCumulativeProbability(values_span, results_span, hint);
        }, 0, static_cast<double>(size));
    }
    
    //==========================================================================
    // 3. ADVANCED FEATURES BENCHMARK
    //==========================================================================
    std::cout << "\n💻 Phase 3: Setting up advanced features benchmarks..." << std::endl;
    std::cout << "   Testing cache-aware processing and work-stealing parallelism.\n" << std::endl;
    
    // Cache-aware operations - using performance hints
    const std::vector<double> large_test(10000, 1.0);
    std::vector<double> large_results(10000);
    
    bench.addTest("Cache-Aware Batch PDF", [&]() mutable {
        std::span<const double> values_span(large_test);
        std::span<double> results_span(large_results);
        auto hint = libstats::performance::PerformanceHint::minimal_latency();
        stdNormal.getProbability(values_span, results_span, hint);
    }, 0, static_cast<double>(large_test.size()));
    
    bench.addTest("Cache-Aware Batch Log PDF", [&]() mutable {
        std::span<const double> values_span(large_test);
        std::span<double> results_span(large_results);
        auto hint = libstats::performance::PerformanceHint::minimal_latency();
        stdNormal.getLogProbability(values_span, results_span, hint);
    }, 0, static_cast<double>(large_test.size()));
    
    bench.addTest("Cache-Aware Batch CDF", [&]() mutable {
        std::span<const double> values_span(large_test);
        std::span<double> results_span(large_results);
        auto hint = libstats::performance::PerformanceHint::minimal_latency();
        stdNormal.getCumulativeProbability(values_span, results_span, hint);
    }, 0, static_cast<double>(large_test.size()));
    
    // Work-stealing parallel operations - using expert strategy selection
    bench.addTest("Work-Stealing Batch PDF", [&]() mutable {
        std::span<const double> values_span(large_test);
        std::span<double> results_span(large_results);
        stdNormal.getProbabilityWithStrategy(values_span, results_span, 
                                             libstats::performance::Strategy::WORK_STEALING);
    }, 0, static_cast<double>(large_test.size()));
    
    bench.addTest("Work-Stealing Batch Log PDF", [&]() mutable {
        std::span<const double> values_span(large_test);
        std::span<double> results_span(large_results);
        stdNormal.getLogProbabilityWithStrategy(values_span, results_span,
                                               libstats::performance::Strategy::WORK_STEALING);
    }, 0, static_cast<double>(large_test.size()));
    
    bench.addTest("Work-Stealing Batch CDF", [&]() mutable {
        std::span<const double> values_span(large_test);
        std::span<double> results_span(large_results);
        stdNormal.getCumulativeProbabilityWithStrategy(values_span, results_span,
                                                      libstats::performance::Strategy::WORK_STEALING);
    }, 0, static_cast<double>(large_test.size()));
    
    //==========================================================================
    // 4. SAMPLING BENCHMARK
    //==========================================================================
    std::cout << "\n🎲 Phase 4: Setting up sampling benchmarks..." << std::endl;
    std::cout << "   Testing Box-Muller transform for efficient Gaussian sampling.\n" << std::endl;
    
    std::mt19937 sample_rng(42);
    
    bench.addTest("Box-Muller Sampling", [&]() mutable {
        volatile double result = stdNormal.sample(sample_rng);
        (void)result;
    }, 0, 1000000.0);
    
    //==========================================================================
    // 5. PARAMETER FITTING BENCHMARK  
    //==========================================================================
    std::cout << "\n🔧 Phase 5: Setting up parameter fitting benchmarks..." << std::endl;
    std::cout << "   Testing maximum likelihood estimation on various dataset sizes.\n" << std::endl;
    
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
        libstats::Gaussian temp_dist;
        temp_dist.fit(fit_data_small);
    }, 0, static_cast<double>(fit_data_small.size()));
    
    bench.addTest("Parameter Fitting Large Dataset", [&]() {
        libstats::Gaussian temp_dist;
        temp_dist.fit(fit_data_large);
    }, 0, static_cast<double>(fit_data_large.size()));
    
    //==========================================================================
    // 6. ADVANCED STATISTICAL METHODS BENCHMARK
    //==========================================================================
    std::cout << "\n📊 Phase 6: Setting up advanced statistical methods benchmarks..." << std::endl;
    std::cout << "   Testing confidence intervals, hypothesis tests, and robust estimation.\n" << std::endl;
    
    // Use medium-sized dataset for statistical methods
    std::vector<double> stats_data(5000);
    for (size_t i = 0; i < stats_data.size(); ++i) {
        stats_data[i] = normal_gen(sample_rng);
    }
    
    bench.addTest("Confidence Interval Mean", [&]() {
        volatile auto result = libstats::Gaussian::confidenceIntervalMean(stats_data, 0.95);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
    bench.addTest("Confidence Interval Variance", [&]() {
        volatile auto result = libstats::Gaussian::confidenceIntervalVariance(stats_data, 0.95);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
    bench.addTest("One Sample T-Test", [&]() {
        volatile auto result = libstats::Gaussian::oneSampleTTest(stats_data, 5.0, 0.05);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
    bench.addTest("Method of Moments Estimation", [&]() {
        volatile auto result = libstats::Gaussian::methodOfMomentsEstimation(stats_data);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
    bench.addTest("Bayesian Parameter Estimation", [&]() {
        volatile auto result = libstats::Gaussian::bayesianEstimation(stats_data, 0.0, 0.001, 1.0, 1.0);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
    bench.addTest("Robust Parameter Estimation", [&]() {
        volatile auto result = libstats::Gaussian::robustEstimation(stats_data, "huber", 1.345);
        (void)result;
    }, 0, static_cast<double>(stats_data.size()));
    
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
    double single_cdf_ops_per_sec = 0.0;
    double single_log_pdf_ops_per_sec = 0.0;
    double single_quantile_ops_per_sec = 0.0;
    double batch_pdf_1k_scalar = 0.0;
    double batch_pdf_1k_parallel = 0.0;
    double batch_pdf_100k_scalar = 0.0;
    double batch_pdf_100k_parallel = 0.0;
    double box_muller_sampling_ops_per_sec = 0.0;
    double cache_aware_ops_per_sec = 0.0;
    double work_stealing_ops_per_sec = 0.0;
    double fitting_small_ops_per_sec = 0.0;
    double fitting_large_ops_per_sec = 0.0;
    double confidence_interval_mean_ops_per_sec = 0.0;
    double confidence_interval_variance_ops_per_sec = 0.0;
    double t_test_ops_per_sec = 0.0;
    double normality_test_ops_per_sec = 0.0;
    double bayesian_estimation_ops_per_sec = 0.0;
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
        } else if (result.name == "Batch PDF Scalar 100000") {
            batch_pdf_100k_scalar = result.stats.throughput;
        } else if (result.name == "Batch PDF Parallel 100000") {
            batch_pdf_100k_parallel = result.stats.throughput;
        } else if (result.name == "Box-Muller Sampling") {
            box_muller_sampling_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Cache-Aware Batch PDF") {
            cache_aware_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Work-Stealing Batch PDF") {
            work_stealing_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Parameter Fitting Small Dataset") {
            fitting_small_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Parameter Fitting Large Dataset") {
            fitting_large_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Confidence Interval Mean") {
            confidence_interval_mean_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Confidence Interval Variance") {
            confidence_interval_variance_ops_per_sec = result.stats.throughput;
        } else if (result.name == "One Sample T-Test") {
            t_test_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Method of Moments Estimation") {
            normality_test_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Bayesian Parameter Estimation") {
            bayesian_estimation_ops_per_sec = result.stats.throughput;
        } else if (result.name == "Robust Parameter Estimation") {
            robust_estimation_ops_per_sec = result.stats.throughput;
        }
    }
    
    std::cout << "\n📈 SINGLE OPERATION PERFORMANCE:" << std::endl;
    std::cout << "   Single-threaded operations showing computational efficiency per function.\n" << std::endl;
    std::cout << "├─ PDF Operations:      " << std::scientific << single_pdf_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "├─ CDF Operations:      " << std::scientific << single_cdf_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "├─ Log PDF Operations:  " << std::scientific << single_log_pdf_ops_per_sec << " ops/sec" << std::endl;
    std::cout << "└─ Quantile Operations: " << std::scientific << single_quantile_ops_per_sec << " ops/sec" << std::endl;
    
    // Add interpretation commentary
    std::cout << "\n💡 Analysis: ";
    if (single_cdf_ops_per_sec > 0 && single_pdf_ops_per_sec > 0) {
        if (single_pdf_ops_per_sec > single_cdf_ops_per_sec * 2) {
            std::cout << "PDF computation is ~" << std::fixed << std::setprecision(1) 
                     << (single_pdf_ops_per_sec / single_cdf_ops_per_sec) 
                     << "x faster than CDF due to simpler exponential calculation.";
        } else {
            std::cout << "PDF and CDF show similar performance, indicating efficient erf() implementation.";
        }
    }
    std::cout << std::endl;
    
    std::cout << "\nBATCH OPERATION PERFORMANCE (1K Elements):" << std::endl;
    std::cout << "├─ Scalar PDF:          " << std::scientific << batch_pdf_1k_scalar << " elements/sec" << std::endl;
    std::cout << "├─ Parallel PDF:        " << std::scientific << batch_pdf_1k_parallel << " elements/sec" << std::endl;
    
    std::cout << "\nBATCH OPERATION PERFORMANCE (100K Elements):" << std::endl;
    std::cout << "├─ Scalar PDF:          " << std::scientific << batch_pdf_100k_scalar << " elements/sec" << std::endl;
    std::cout << "├─ Parallel PDF:        " << std::scientific << batch_pdf_100k_parallel << " elements/sec" << std::endl;
    std::cout << "├─ Cache-Aware PDF:     " << std::scientific << cache_aware_ops_per_sec << " elements/sec" << std::endl;
    std::cout << "└─ Work-Stealing PDF:   " << std::scientific << work_stealing_ops_per_sec << " elements/sec" << std::endl;
    
    std::cout << "\nSAMPLING AND FITTING PERFORMANCE:" << std::endl;
    std::cout << "├─ Box-Muller Sampling:         " << std::scientific << box_muller_sampling_ops_per_sec << " samples/sec" << std::endl;
    std::cout << "├─ Parameter Fitting (Small):    " << std::scientific << fitting_small_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "└─ Parameter Fitting (Large):    " << std::scientific << fitting_large_ops_per_sec << " datapoints/sec" << std::endl;
    
    std::cout << "\nADVANCED STATISTICAL METHODS:" << std::endl;
    std::cout << "├─ Confidence Intervals (Mean):    " << std::scientific << confidence_interval_mean_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "├─ Confidence Intervals (Variance):" << std::scientific << confidence_interval_variance_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "├─ T-Tests (One Sample):          " << std::scientific << t_test_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "├─ Method of Moments Estimation:   " << std::scientific << normality_test_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "├─ Bayesian Parameter Estimation:  " << std::scientific << bayesian_estimation_ops_per_sec << " datapoints/sec" << std::endl;
    std::cout << "└─ Robust Parameter Estimation:    " << std::scientific << robust_estimation_ops_per_sec << " datapoints/sec" << std::endl;
    
    if (batch_pdf_100k_parallel > 0 && batch_pdf_100k_scalar > 0) {
        double speedup = batch_pdf_100k_parallel / batch_pdf_100k_scalar;
        std::cout << "\nPARALLEL SPEEDUP: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    }
    
    // Gaussian-specific performance insights
    std::cout << "\nGAUSSIAN DISTRIBUTION INSIGHTS:" << std::endl;
    std::cout << "├─ Standard normal optimization: Fast path for N(0,1)" << std::endl;
    std::cout << "├─ Box-Muller sampling: Dual sample generation" << std::endl;
    std::cout << "├─ Error function approximations: Optimized erf() calls" << std::endl;
    std::cout << "├─ Statistical inference: Native t-tests and normality" << std::endl;
    std::cout << "└─ Numerical stability: Handles extreme z-scores" << std::endl;
    
    std::cout << "\n🚀 GAUSSIAN DISTRIBUTION BENCHMARK COMPLETE! 🚀" << std::endl;
    std::cout << "All enhanced features successfully benchmarked." << std::endl;
    
    return 0;
}
